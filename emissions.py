"""
emissions.py — Pass 3 continuous emission generation.

Generates per-sample probabilities P(S1|t), P(S2|t), P(Noise|t) using:
  1. Pass 2 label_scores at raw peak positions as an initial seed.
  2. Gaussian-gated bumps from the Pass 3 state timeline (S1/S2 event centers).
  3. Spectral template sweep over systole windows using the S1 and S2 mean
     FFT profiles stored in analysis_data["pass3_spectral_context"].

Result stored as analysis_data["pass3_emissions"], shape (n_samples, 3):
  column 0: P(S1|t)
  column 1: P(S2|t)
  column 2: P(Noise|t)
"""

import logging
import os
from typing import Dict, Optional

import numpy as np


def generate_pass3_emissions(
    analysis_data: Dict,
    audio_envelope: np.ndarray,
    sample_rate: int,
    params: Dict,
    wav_file_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Build continuous emission probabilities from Pass 3 outputs.

    Returns the (n_samples, 3) float32 array and stores it in
    analysis_data["pass3_emissions"].  Returns None on early-exit.
    """
    state_labels = analysis_data.get("pass3_state_labels")
    state_boundaries = analysis_data.get("pass3_state_boundaries") or []
    if state_labels is None or len(state_labels) == 0:
        logging.debug("generate_pass3_emissions: no pass3_state_labels; skipping.")
        return None

    n_samples = int(len(state_labels))
    _eps = 1e-10

    # ── Params ────────────────────────────────────────────────────────────────
    gate_width_ms   = float(params.get("pass3_emission_gate_width_ms",  80.0))
    noise_floor     = float(params.get("pass3_emission_noise_floor",    0.05))
    tau             = float(params.get("pass3_emission_spectral_tau",   5.0))
    gate_width_samp = max(1, int(round(gate_width_ms * sample_rate / 1000.0)))

    # ── 1. Initialise arrays ──────────────────────────────────────────────────
    E_S1    = np.zeros(n_samples, dtype=np.float64)
    E_S2    = np.zeros(n_samples, dtype=np.float64)
    E_Noise = np.full(n_samples, noise_floor, dtype=np.float64)

    # ── 2. Seed from Pass 2 label_scores at existing peak positions ───────────
    pc = analysis_data.get("peak_classifications") or {}
    for peak_idx, entry in pc.items():
        if not isinstance(entry, dict):
            continue
        ls = entry.get("label_scores")
        if not isinstance(ls, dict):
            continue
        idx = int(peak_idx)
        if 0 <= idx < n_samples:
            E_S1[idx]  += float(ls.get("S1",    0.0))
            E_S2[idx]  += float(ls.get("S2",    0.0))
            E_Noise[idx] += float(ls.get("noise", 0.0))

    # ── 3. Gaussian bumps centered on S1/S2 events from state timeline ────────
    sigma       = gate_width_samp / 2.0
    half_kernel = min(int(3 * sigma), n_samples // 2)
    x           = np.arange(-half_kernel, half_kernel + 1, dtype=np.float64)
    gauss       = np.exp(-0.5 * (x / max(sigma, 1.0)) ** 2)

    def _add_gauss(arr: np.ndarray, center: int) -> None:
        lo = max(0, center - half_kernel)
        hi = min(n_samples, center + half_kernel + 1)
        klo = lo - (center - half_kernel)
        khi = klo + (hi - lo)
        if hi > lo and khi > klo:
            arr[lo:hi] += gauss[klo:khi]

    for bs, be, bst, _ in state_boundaries:
        center = (bs + be) // 2
        if bst == "S1":
            _add_gauss(E_S1, center)
        elif bst == "S2":
            _add_gauss(E_S2, center)

    # ── 4. Spectral sweep over systole windows ────────────────────────────────
    spectral_ctx = analysis_data.get("pass3_spectral_context")
    if (
        spectral_ctx is not None
        and wav_file_path is not None
        and os.path.isfile(wav_file_path)
        and spectral_ctx.get("mu_s1_db") is not None
        and spectral_ctx.get("mu_s2_db") is not None
    ):
        try:
            _run_spectral_sweep(
                E_S1, E_S2, n_samples,
                spectral_ctx, state_boundaries,
                wav_file_path, sample_rate, params, tau, _eps,
            )
        except Exception as exc:
            logging.warning("generate_pass3_emissions: spectral sweep failed: %s", exc)

    # ── 5. Normalize to probabilities ─────────────────────────────────────────
    total = E_S1 + E_S2 + E_Noise + _eps
    emissions = np.stack([E_S1 / total, E_S2 / total, E_Noise / total], axis=1).astype(np.float32)

    analysis_data["pass3_emissions"] = emissions
    logging.info(
        "Pass 3 emissions: shape %s  mean(P_S1=%.3f  P_S2=%.3f  P_Noise=%.3f).",
        emissions.shape,
        float(emissions[:, 0].mean()),
        float(emissions[:, 1].mean()),
        float(emissions[:, 2].mean()),
    )
    return emissions


# ─────────────────────────────────────────────────────────────────────────────
# Internal: spectral sweep over systole windows
# ─────────────────────────────────────────────────────────────────────────────

def _run_spectral_sweep(
    E_S1: np.ndarray,
    E_S2: np.ndarray,
    n_samples: int,
    spectral_ctx: Dict,
    state_boundaries,
    wav_file_path: str,
    sample_rate: int,
    params: Dict,
    tau: float,
    eps: float,
) -> None:
    """
    For each systole window in the state timeline, slide short-time FFT frames
    over the bandpass audio and score each frame against both mu_s1_db and mu_s2_db.

    Softmax([score_S1, score_S2] / tau) gives a per-frame spectral probability
    which is added into E_S1 / E_S2 gated to the systole region.
    """
    import librosa
    from audio_preprocessing import apply_bandpass_only

    mu_s1_db    = np.asarray(spectral_ctx["mu_s1_db"],  dtype=np.float64)
    mu_s2_db    = np.asarray(spectral_ctx["mu_s2_db"],  dtype=np.float64)
    freqs       = np.asarray(spectral_ctx["freqs"],      dtype=np.float64)
    n_fft       = int(spectral_ctx["n_fft"])
    half_samp   = int(spectral_ctx["half_samples"])
    target_sr   = int(spectral_ctx.get("full_sr", sample_rate))

    low_hz  = float(params.get("fft_separation_low_hz",  10.0))
    high_hz = float(params.get("fft_separation_high_hz", 15000.0))
    mask    = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask) or len(mu_s1_db) != len(freqs):
        return

    stride_ms    = float(params.get("pass3_insert_spectrum_stride_ms", 5.0))
    stride_full  = max(1, int(round(stride_ms * target_sr / 1000.0)))

    audio_raw, _ = librosa.load(wav_file_path, sr=target_sr, mono=True)
    bandpass     = apply_bandpass_only(audio_raw, target_sr, params).astype(np.float64)
    n_audio      = len(bandpass)

    hanning_win = np.hanning(2 * half_samp)
    padded_buf  = np.zeros(n_fft, dtype=np.float64)

    for bs, be, bst, bm in state_boundaries:
        if bst != "systole":
            continue
        s2_center = bm.get("s2")
        if s2_center is None:
            continue
        # Convert envelope-sample coordinates → full-SR coordinates.
        t_center     = float(s2_center) / float(sample_rate)
        center_full  = int(round(t_center * target_sr))
        span_sec     = max(0.05, (be - bs) / float(sample_rate) / 2.0 + 0.05)
        span_full    = int(round(span_sec * target_sr))
        lo_full      = max(half_samp, center_full - span_full)
        hi_full      = min(n_audio - half_samp, center_full + span_full)
        if hi_full <= lo_full:
            continue

        frame_p_s1:  list = []
        frame_p_s2:  list = []
        frame_env_ix: list = []

        for c in range(lo_full, hi_full + 1, stride_full):
            seg = bandpass[c - half_samp: c + half_samp]
            if len(seg) < 2 * half_samp:
                continue
            windowed = seg * hanning_win
            rms = float(np.sqrt(np.mean(windowed ** 2) + eps))
            ref_db = 20.0 * np.log10(max(rms, 1e-10))
            padded_buf[:] = 0.0
            padded_buf[:len(windowed)] = windowed
            fft_db    = 20.0 * np.log10(np.abs(np.fft.rfft(padded_buf)) + eps)
            spec_shape = fft_db - ref_db

            mse_s1 = float(np.mean((spec_shape[mask] - mu_s1_db[mask]) ** 2))
            mse_s2 = float(np.mean((spec_shape[mask] - mu_s2_db[mask]) ** 2))

            # log-sum-exp stable softmax over [−mse_s1, −mse_s2]
            logit_s1 = -mse_s1 / tau
            logit_s2 = -mse_s2 / tau
            m = max(logit_s1, logit_s2)
            e1 = np.exp(logit_s1 - m)
            e2 = np.exp(logit_s2 - m)
            denom = e1 + e2 + eps
            frame_p_s1.append(float(e1 / denom))
            frame_p_s2.append(float(e2 / denom))

            env_idx = int(round(c * float(sample_rate) / float(target_sr)))
            frame_env_ix.append(max(0, min(env_idx, n_samples - 1)))

        for j, env_idx in enumerate(frame_env_ix):
            if bs <= env_idx < be:
                E_S1[env_idx] += frame_p_s1[j]
                E_S2[env_idx] += frame_p_s2[j]
