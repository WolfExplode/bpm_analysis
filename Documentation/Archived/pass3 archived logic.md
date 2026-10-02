# Pass 3 archived logic

Removed from the active codebase: **6.1** S2 spectral-profile alignment, **6.2** Pass A–C correction loop (and related config), **6.5** Pass C phase/sequence fixes (same removal as 6.2 in this codebase), **6.7** Pass 3 emissions for Pass 4 Viterbi. Sources below are from `git show HEAD` (last commit) before unstaged removals.

---

## emissions.py (full file)

```python
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
```

---

## correction.py excerpts

### Spectral / interval helpers through `_choose_s2_spectral` (lines 153–361)

```python
def _s2_index_respects_pass3_intervals(
    s1: int,
    s1_next: int,
    s2_idx: int,
    sample_rate: int,
    intervals: Dict,
    params: Dict,
) -> bool:
    """
    True when S2 at s2_idx yields systole/diastole durations that Pass A would not flag
    as too short (same thresholds as _pass_a_resnap_s2).
    """
    if s2_idx <= s1 or s2_idx >= s1_next:
        return False
    sr = float(sample_rate)
    systole = (s2_idx - s1) / sr
    diastole = (s1_next - s2_idx) / sr
    s1_s2_min = float(intervals.get("s1_s2_min", 0.12))
    diastole_min = float(intervals.get("diastole_min", 0.08))
    sys_slack = float(params.get("pass3_systole_slack_frac", 0.15))
    dia_slack = float(params.get("pass3_diastole_slack_frac", 0.20))
    systole_too_short = systole < (1.0 - sys_slack) * s1_s2_min
    diastole_too_short = diastole < (1.0 - dia_slack) * diastole_min
    return not (systole_too_short or diastole_too_short)


def _choose_s2_near(
    s1: int,
    s1_next: int,
    s2_pred: int,
    half_window_samples: int,
    snap_s2: bool,
    insert_spectrum_ctx: Optional[Dict],
    sample_rate: int,
    n_samples: int,
    params: Dict,
    intervals: Dict,
) -> int:
    """Choose S2 near predicted time by sliding FFT windows against the S2 spectral template.

    If snap_s2 is False, no spectral context is available, or no confident match is found,
    returns s2_pred clamped within [s1+1, s1_next-1] — keeps rhythm at nominal ejection time
    without requiring a peak at that location.

    A spectral winner is rejected if it would make systole or diastole shorter than Pass A's
    minimum plausible durations for this BPM (calculate_bpm_intervals + slack).
    """
    s2 = int(max(s1 + 1, min(s2_pred, s1_next - 1)))
    if not snap_s2 or insert_spectrum_ctx is None:
        return s2
    t_pred_sec = float(s2_pred) / float(sample_rate)
    search_half_sec = float(half_window_samples) / float(sample_rate)
    result = _choose_s2_spectral(
        t_pred_sec, search_half_sec, insert_spectrum_ctx, params, sample_rate, n_samples,
    )
    if result is None:
        return s2
    sp_idx, _ = result
    if not (s1 < sp_idx < s1_next):
        return s2
    if not _s2_index_respects_pass3_intervals(s1, s1_next, int(sp_idx), sample_rate, intervals, params):
        return s2
    return int(sp_idx)


def _choose_s1_near(
    t_expected_sec: float,
    half_window_samples: int,
    min_sep_samples: int,
    all_raw_peaks: np.ndarray,
    pc: Dict,
    n_samples: int,
    sample_rate: int,
) -> Optional[int]:
    """Choose best S1 near expected time using label_scores['S1']."""
    if len(all_raw_peaks) == 0:
        return None
    center = int(round(t_expected_sec * sample_rate))
    lo = max(0, center - half_window_samples)
    hi = min(n_samples - 1, center + half_window_samples)
    if hi <= lo:
        return None
    cand = [int(p) for p in all_raw_peaks if lo <= int(p) <= hi]
    if not cand:
        return None
    best: Optional[int] = None
    best_score: Optional[float] = None
    for p in cand:
        entry = pc.get(int(p)) or {}
        ls = entry.get("label_scores") if isinstance(entry, dict) else None
        s1_score = float(ls.get("S1", 0.0)) if isinstance(ls, dict) else 0.0
        noise_score = float(ls.get("noise", 0.0)) if isinstance(ls, dict) else 0.0
        dist_sec = abs(p - center) / float(sample_rate)
        score = (2.0 * s1_score) - (1.0 * noise_score) - (0.75 * dist_sec)
        if best is None or score > best_score:
            best, best_score = p, score
    return int(best) if best is not None else None


def _insert_spectrum_envelope_ok(
    env_idx: int,
    audio_envelope: np.ndarray,
    analysis_data: Dict,
    params: Dict,
) -> bool:
    """Return True if envelope at env_idx meets the noise-floor margin."""
    margin = float(params.get("pass3_insert_spectrum_envelope_margin", 0.0))
    if margin <= 0:
        return True
    nfs = analysis_data.get("dynamic_noise_floor_series")
    if nfs is None or getattr(nfs, "empty", True):
        return True
    try:
        ei = int(max(0, min(env_idx, len(audio_envelope) - 1)))
        e = float(audio_envelope[ei])
        nf = float(nfs.reindex([ei], method="nearest").iloc[0])
        return e >= margin * nf
    except Exception:
        return True


def _find_sensitive_peaks_near(
    t_expected_sec: float,
    window_samples: int,
    sensitivity_factor: float,
    audio_envelope: np.ndarray,
    analysis_data: Dict,
    n_samples: int,
    sample_rate: int,
    params: Dict,
) -> Optional[int]:
    """
    Re-scan audio_envelope in a narrow window with a lower height threshold to find
    faint peaks missed by the main detector.  Returns the highest-amplitude peak or None.
    """
    nfs = analysis_data.get("dynamic_noise_floor_series")
    center = int(round(t_expected_sec * sample_rate))
    lo = max(0, center - window_samples)
    hi = min(n_samples - 1, center + window_samples)
    if hi <= lo:
        return None
    segment = audio_envelope[lo: hi + 1]
    if len(segment) == 0:
        return None
    if nfs is not None and not getattr(nfs, "empty", True):
        try:
            indices = np.arange(lo, hi + 1)
            nf_vals = nfs.reindex(indices, method="nearest").values.astype(np.float64)
            height_thresh = sensitivity_factor * nf_vals
        except Exception:
            height_thresh = sensitivity_factor * float(np.median(audio_envelope))
    else:
        height_thresh = sensitivity_factor * float(np.median(audio_envelope))
    min_dist = max(1, int(float(params.get("min_peak_distance_sec", 0.10)) * sample_rate // 2))
    try:
        local_peaks, _ = _scipy_find_peaks(segment, height=height_thresh, distance=min_dist)
    except Exception:
        return None
    if len(local_peaks) == 0:
        return None
    best_local = int(local_peaks[np.argmax(segment[local_peaks])])
    return lo + best_local


def _choose_s2_spectral(
    t_expected_sec: float,
    search_half_sec: float,
    insert_spectrum_ctx: Optional[Dict],
    params: Dict,
    sample_rate: int,
    n_samples: int,
) -> Optional[Tuple[int, float]]:
    """
    Search for S2 using the spectral S2 template from insert_spectrum_ctx.

    LIMITATION: the S2 template is built from Pass 2 paired S2 peaks.  If Pass 2 made
    systematic labeling errors those may bias the template (confirmation-bias risk).
    Only call after _find_sensitive_peaks_near has already failed.

    Returns (envelope_index, score) or None.
    """
    if insert_spectrum_ctx is None:
        return None
    mu_s2 = insert_spectrum_ctx.get("mu_s2_db")
    if mu_s2 is None or not isinstance(mu_s2, np.ndarray) or len(mu_s2) == 0:
        return None
    n_s2_tpl = int(insert_spectrum_ctx.get("n_s2_template", 0))
    min_tpl = int(params.get("pass3_s2_spectral_min_templates", 3))
    if n_s2_tpl < min_tpl:
        return None
    try:
        result = spectrum_template_search_envelope_index(
            insert_spectrum_ctx["bandpass_audio"],
            int(insert_spectrum_ctx["full_sr"]),
            t_expected_sec,
            search_half_sec,
            mu_s2,
            insert_spectrum_ctx["freqs"],
            int(insert_spectrum_ctx["n_fft"]),
            int(insert_spectrum_ctx["half_samples"]),
            sample_rate,
            n_samples,
            params,
        )
    except Exception:
        return None
    return result


```

### `_rebuild_s2_events` with spectral + HF snap (lines 675–732)

```python
def _rebuild_s2_events(
    s1_list: List[int],
    lt_series: Optional[pd.Series],
    fallback_bpm: float,
    sample_rate: int,
    params: Dict,
    snap_s2: bool,
    snap_half: int,
    n_samples: int,
    insert_spectrum_ctx: Optional[Dict],
    noise_ivs: Optional[List[Tuple[int, int]]] = None,
    seed_s2_events: Optional[List[int]] = None,
) -> List[int]:
    """Rebuild s2_events from scratch given the current s1_list.

    When ``seed_s2_events`` is set and a cycle overlaps HF-noise, keep that seed
    index so pass3_align_s2_to_s2_spectral_profile does not run on bad audio.
    """
    niv = noise_ivs or []
    seed = seed_s2_events
    s2_events: List[int] = []
    for j in range(len(s1_list) - 1):
        a = int(s1_list[j])
        b = int(s1_list[j + 1])
        if b <= a:
            s2_events.append(int(a))
            continue
        if (
            seed is not None
            and j < len(seed)
            and niv
            and _hf_noise_disables_s2_snap(a, b, niv, s2_check=int(seed[j]))
        ):
            s2_keep = int(max(a + 1, min(int(seed[j]), b - 1)))
            s2_events.append(s2_keep)
            continue
        t_a = a / float(sample_rate)
        bpm_a = _bpm_at_time(t_a, lt_series, fallback_bpm)
        ivs_a = calculate_bpm_intervals(bpm_a, params)
        s1_s2_nominal_a = float(ivs_a.get("s1_s2_nominal", 0.30))
        s2_pred_a = int(round(a + s1_s2_nominal_a * sample_rate))
        snap_here = _effective_snap_s2(snap_s2, a, b, niv, s2_check=None)
        # When spectral S2 alignment is off, keep the Pass-2 / nominal seed from
        # _seed_s2_from_pass2_pairs instead of re-deriving only the BPM clamp (which
        # can disagree with an in-window labeled S2).
        if seed is not None and j < len(seed) and (not snap_here):
            sj = int(seed[j])
            if a < sj < b:
                s2_events.append(int(max(a + 1, min(sj, b - 1))))
                continue
        s2_a = _choose_s2_near(
            a, b, s2_pred_a, snap_half,
            snap_here, insert_spectrum_ctx, sample_rate, n_samples, params, ivs_a,
        )
        s2_events.append(int(s2_a))
    return s2_events


```

### `_hf_noise_disables_s2_snap`, `_effective_snap_s2` (lines 1668–1699)

```python
def _hf_noise_disables_s2_snap(
    s1: int,
    s1_next: int,
    noise_ivs: List[Tuple[int, int]],
    s2_check: Optional[int] = None,
) -> bool:
    """
    True → do not use pass3_align_s2_to_s2_spectral_profile (spectral / sliding template) for this cycle.
    HF-noise segments make the underlying audio unreliable; use nominal timing only.
    """
    if not noise_ivs:
        return False
    for lo, hi in noise_ivs:
        if lo < s1_next and hi > s1:
            return True
        if s2_check is not None and lo <= int(s2_check) < hi:
            return True
    return False


def _effective_snap_s2(
    snap_s2: bool,
    s1: int,
    s1_next: int,
    noise_ivs: List[Tuple[int, int]],
    s2_check: Optional[int] = None,
) -> bool:
    """Spectral S2 snap allowed only when globally on and cycle not in HF-noise."""
    if not snap_s2:
        return False
    return not _hf_noise_disables_s2_snap(s1, s1_next, noise_ivs, s2_check)

```

### `_build_state_boundaries_before_from_cycles`, `_pass_a_resnap_s2`, `_pass_c_phase_correction` (lines 1737–2108)

```python
def _build_state_boundaries_before_from_cycles(
    s1_list: List[int],
    s2_events: List[int],
    s1_half: int,
    s2_half: int,
    n_samples: int,
) -> List[Tuple]:
    """Initial S1/systole/S2/diastole boundary list (no transient edge detection)."""
    state_boundaries_before: List[Tuple] = []
    for i in range(len(s1_list) - 1):
        s1 = int(s1_list[i])
        s1_next = int(s1_list[i + 1])
        if s1_next <= s1:
            continue
        s2 = int(max(s1 + 1, min(
            int(s2_events[i]) if i < len(s2_events) else s1,
            s1_next - 1,
        )))
        s1_start, s1_end, s2_start, s2_end = _paint_state_boundaries(
            s1, s2, s1_next, s1_half, s2_half, n_samples,
            use_transient_detection=False,
        )
        if s1_end > s1_start:
            state_boundaries_before.append((s1_start, s1_end, "S1", {"s1": s1}))
        if s2_start > s1_end:
            state_boundaries_before.append((s1_end, s2_start, "systole", {"s1": s1, "s2": s2}))
        if s2_end > s2_start:
            state_boundaries_before.append((s2_start, s2_end, "S2", {"s2": s2}))
        if s1_next > s2_end:
            state_boundaries_before.append((s2_end, s1_next, "diastole", {"s2": s2, "s1_next": s1_next}))
    return state_boundaries_before


# ─────────────────────────────────────────────────────────────────────────────
# Pass A — re-snap S2 for timing plausibility
# ─────────────────────────────────────────────────────────────────────────────

def _pass_a_resnap_s2(
    s1_list: List[int],
    s2_events: List[int],
    lt_series: Optional[pd.Series],
    fallback_bpm: float,
    sample_rate: int,
    params: Dict,
    snap_s2: bool,
    resnap_half: int,
    n_samples: int,
    insert_spectrum_ctx: Optional[Dict],
    systole_slack: float,
    diastole_slack: float,
    noise_ivs: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[List[int], List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """
    Re-snap S2 when systole/diastole are out of plausible range.

    Returns (new_s2_events, new_corrections, cycle_diagnostics, changed).
    s1_list is unchanged.
    """
    new_s2_events = list(s2_events)
    new_corrections: List[Dict[str, Any]] = []
    cycle_diagnostics: List[Dict[str, Any]] = []
    changed = False
    niv = noise_ivs or []

    for i in range(len(s1_list) - 1):
        s1 = int(s1_list[i])
        s1_next = int(s1_list[i + 1])
        if s1_next <= s1:
            continue
        s2 = int(new_s2_events[i]) if i < len(new_s2_events) else s1
        s2 = int(max(s1 + 1, min(s2, s1_next - 1)))

        t_s1 = s1 / float(sample_rate)
        bpm = _bpm_at_time(t_s1, lt_series, fallback_bpm)
        intervals = calculate_bpm_intervals(bpm, params)
        s1_s2_min     = float(intervals.get("s1_s2_min",     0.12))
        s1_s2_nominal = float(intervals.get("s1_s2_nominal", 0.30))
        s1_s2_max     = float(intervals.get("s1_s2_max",     0.40))

        systole  = (s2 - s1)     / float(sample_rate)
        rr       = (s1_next - s1) / float(sample_rate)
        diastole = rr - systole
        expected_rr      = float(intervals.get("rr_interval", 60.0 / bpm if bpm > 0 else 0.75))
        diastole_nominal = float(intervals.get("s2_s1_nominal", max(0.0, expected_rr - s1_s2_nominal)))
        diastole_min     = float(intervals.get("diastole_min", 0.08))
        diastole_max     = float(intervals.get("diastole_max", diastole_nominal * 2.0))

        s1_min_evt      = float(intervals.get("s1_min",      0.010))
        s1_nominal_evt  = float(intervals.get("s1_nominal",  0.040))
        s1_max_evt      = float(intervals.get("s1_max",      0.080))
        s2_min_evt      = float(intervals.get("s2_min",      0.010))
        s2_nominal_evt  = float(intervals.get("s2_nominal",  0.030))
        s2_max_evt      = float(intervals.get("s2_max",      0.060))
        min_feasible    = float(intervals.get(
            "min_feasible_cycle", s1_min_evt + s1_s2_min + s2_min_evt + diastole_min,
        ))

        too_short        = systole < (1.0 - systole_slack)  * s1_s2_min
        too_long         = systole > (1.0 + systole_slack)  * s1_s2_max
        far_from_nominal = abs(systole - s1_s2_nominal) > max(0.12, 0.5 * (s1_s2_max - s1_s2_min))
        diastole_too_short = diastole < (1.0 - diastole_slack) * diastole_min

        cycle_diagnostics.append({
            "i": int(i), "s1": int(s1), "s2": int(s2), "s1_next": int(s1_next),
            "bpm": float(bpm), "rr_sec": float(rr),
            "systole_sec": float(systole), "diastole_sec": float(diastole),
            "expected_rr_sec": float(expected_rr),
            "diastole_nominal_sec": float(diastole_nominal),
            "diastole_min_sec": float(diastole_min),
            "diastole_max_sec": float(diastole_max),
            "s1_min_sec": float(s1_min_evt), "s1_nominal_sec": float(s1_nominal_evt),
            "s1_max_sec": float(s1_max_evt),
            "s2_min_sec": float(s2_min_evt), "s2_nominal_sec": float(s2_nominal_evt),
            "s2_max_sec": float(s2_max_evt),
            "min_feasible_cycle_sec": float(min_feasible),
            "s1_s2_min": float(s1_s2_min), "s1_s2_nominal": float(s1_s2_nominal),
            "s1_s2_max": float(s1_s2_max),
            "flags": {
                "systole_too_short": bool(too_short),
                "systole_too_long": bool(too_long),
                "systole_far_from_nominal": bool(far_from_nominal),
                "diastole_too_short": bool(diastole_too_short),
            },
        })

        snap_here = _effective_snap_s2(snap_s2, s1, s1_next, niv, s2_check=s2)
        if (too_short or too_long or far_from_nominal) and snap_here and insert_spectrum_ctx is not None:
            s2_pred = int(round(s1 + s1_s2_nominal * sample_rate))
            new_s2 = _choose_s2_near(
                s1, s1_next, s2_pred, resnap_half,
                snap_here, insert_spectrum_ctx, sample_rate, n_samples, params, intervals,
            )
            new_s2 = int(max(s1 + 1, min(new_s2, s1_next - 1)))
            if new_s2 != s2:
                new_corrections.append({
                    "type": "resnap_s2",
                    "cycle": int(i), "s1": int(s1),
                    "old_s2": int(s2), "new_s2": int(new_s2), "s2_pred": int(s2_pred),
                })
                new_s2_events[i] = int(new_s2)
                changed = True

    return new_s2_events, new_corrections, cycle_diagnostics, changed


# ─────────────────────────────────────────────────────────────────────────────
# Pass C — phase-shift cascade corrections
# ─────────────────────────────────────────────────────────────────────────────

def _pass_c_phase_correction(
    s1_list: List[int],
    s2_events: List[int],
    cycle_diagnostics: List[Dict[str, Any]],
    all_raw_peaks: np.ndarray,
    pc: Dict,
    lt_series: Optional[pd.Series],
    fallback_bpm: float,
    audio_envelope: np.ndarray,
    analysis_data: Dict,
    n_samples: int,
    sample_rate: int,
    params: Dict,
    enable_phase_correction: bool,
    phase_min_score_delta: float,
    local_peak_window_samples: int,
    local_peak_window_ms: float,
    local_peak_sensitivity: float,
    s1_search_half: int,
    min_sep_samples: int,
    snap_s2: bool,
    snap_half: int,
    insert_spectrum_ctx: Optional[Dict],
    noise_ivs: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[List[int], List[int], List[Dict[str, Any]], bool]:
    """
    Phase-shift cascade corrections (one fix per call; outer loop handles multiples).

    C.1  Remove false S1 (both systole + diastole too short).
    C.2  Demote S1_next to S2 (diastole too short, S1_next looks like S2).
    C.3  Find faint S2 (systole too long, Pass A already failed).

    Returns (new_s1_list, new_s2_events, new_corrections, changed).
    """
    if not enable_phase_correction or not cycle_diagnostics:
        return s1_list, s2_events, [], False

    niv = noise_ivs or []
    new_s1_list = list(s1_list)
    new_s2_events = list(s2_events)

    # ── C.1: Remove false S1 ─────────────────────────────────────────────────
    for diag in cycle_diagnostics:
        i = diag["i"]
        if i + 1 >= len(new_s1_list):
            continue
        s1     = diag["s1"]
        s1_next = diag["s1_next"]
        systole  = diag["systole_sec"]
        diastole = diag["diastole_sec"]
        diastole_min_c = diag.get("diastole_min_sec", 0.0)
        s1_s2_min_c    = diag["s1_s2_min"]
        if not (systole < s1_s2_min_c and diastole < diastole_min_c):
            continue
        suspect = int(s1_next)
        entry = pc.get(suspect) or {}
        ls = entry.get("label_scores") if isinstance(entry, dict) else None
        if not isinstance(ls, dict):
            continue
        noise_score = float(ls.get("noise", 0.0))
        s1_score    = float(ls.get("S1",    0.0))
        if noise_score - s1_score < phase_min_score_delta:
            continue
        min_feasible = diag.get("min_feasible_cycle_sec", 0.0)
        merged_next  = int(new_s1_list[i + 2]) if i + 2 < len(new_s1_list) else n_samples
        merged_span  = (merged_next - int(s1)) / float(sample_rate)
        if min_feasible > 0 and merged_span < min_feasible:
            continue
        new_s1_list = [p for p in new_s1_list if p != suspect]
        new_s2_events = _rebuild_s2_events(
            new_s1_list, lt_series, fallback_bpm,
            sample_rate, params, snap_s2, snap_half, n_samples, insert_spectrum_ctx,
            noise_ivs=niv,
        )
        corr = {
            "type": "remove_false_s1", "cycle": int(i), "s1": int(s1),
            "removed_s1": int(suspect),
            "systole_sec": float(systole), "diastole_sec": float(diastole),
            "diastole_min_sec": float(diastole_min_c),
            "noise_score": float(noise_score), "s1_score": float(s1_score),
        }
        logging.info(
            "Pass 3 C.1: removed false S1 at sample %d "
            "(cycle %d, systole=%.3fs, diastole=%.3fs/min=%.3fs).",
            suspect, i, systole, diastole, diastole_min_c,
        )
        return new_s1_list, new_s2_events, [corr], True

    # ── C.2: Demote S1_next to S2 ────────────────────────────────────────────
    for diag in cycle_diagnostics:
        i = diag["i"]
        if i + 1 >= len(new_s1_list):
            continue
        if not diag["flags"].get("diastole_too_short", False):
            continue
        s1     = diag["s1"]
        s1_next = diag["s1_next"]
        entry = pc.get(int(s1_next)) or {}
        ls = entry.get("label_scores") if isinstance(entry, dict) else None
        if not isinstance(ls, dict):
            continue
        s2_score = float(ls.get("S2", 0.0))
        s1_score = float(ls.get("S1", 0.0))
        if s2_score - s1_score < phase_min_score_delta:
            continue
        new_s2 = int(s1_next)
        upper_bound = int(new_s1_list[i + 2]) if i + 2 < len(new_s1_list) else n_samples
        bpm_here = _bpm_at_time(int(s1) / float(sample_rate), lt_series, fallback_bpm)
        ivs_here = calculate_bpm_intervals(bpm_here, params)
        s2_min_here      = float(ivs_here.get("s2_min",      0.010))
        diastole_min_here = float(ivs_here.get("diastole_min", 0.08))
        earliest_new_s1  = new_s2 + max(1, int(s2_min_here * sample_rate))
        expected_dia_here = max(diastole_min_here, float(ivs_here.get("s2_s1_nominal", 0.35)))
        t_new_s1 = earliest_new_s1 / float(sample_rate) + max(0.0, expected_dia_here - s2_min_here)
        new_s1_cand: Optional[int] = None
        new_s1_cand = _choose_s1_near(
            t_new_s1, s1_search_half, min_sep_samples,
            all_raw_peaks, pc, n_samples, sample_rate,
        )
        if new_s1_cand is not None and (new_s1_cand < earliest_new_s1 or new_s1_cand >= upper_bound):
            new_s1_cand = None
        if new_s1_cand is None:
            sens = _find_sensitive_peaks_near(
                t_new_s1, local_peak_window_samples, local_peak_sensitivity,
                audio_envelope, analysis_data, n_samples, sample_rate, params,
            )
            if sens is not None and earliest_new_s1 <= sens < upper_bound:
                new_s1_cand = sens
        if new_s1_cand is None or new_s1_cand < earliest_new_s1 or new_s1_cand >= upper_bound:
            continue
        new_s2_events[i] = new_s2
        new_s1_list = [p for p in new_s1_list if p != int(s1_next)]
        new_s1_list.append(new_s1_cand)
        new_s1_list = sorted(list(dict.fromkeys(new_s1_list)))
        new_s2_events = _rebuild_s2_events(
            new_s1_list, lt_series, fallback_bpm,
            sample_rate, params, snap_s2, snap_half, n_samples, insert_spectrum_ctx,
            noise_ivs=niv,
        )
        corr = {
            "type": "flip_demote_s1", "cycle": int(i), "s1": int(s1),
            "old_s1_next": int(s1_next), "new_s2_for_cycle": int(new_s2),
            "new_s1_next": int(new_s1_cand),
            "s2_score": float(s2_score), "s1_score": float(s1_score),
        }
        logging.info(
            "Pass 3 C.2: flipped S1@%d\u2192S2, new S1 at %d (cycle %d, diastole was %.3fs).",
            s1_next, new_s1_cand, i, diag["diastole_sec"],
        )
        return new_s1_list, new_s2_events, [corr], True

    # ── C.3: Find faint S2 ───────────────────────────────────────────────────
    for diag in cycle_diagnostics:
        i = diag["i"]
        if not diag["flags"].get("systole_too_long", False):
            continue
        s1     = diag["s1"]
        s1_next = diag["s1_next"]
        if _hf_noise_disables_s2_snap(int(s1), int(s1_next), niv, s2_check=int(diag["s2"])):
            continue
        bpm_c  = diag["bpm"]
        ivs_c  = calculate_bpm_intervals(bpm_c, params)
        s1_s2_nominal_c = float(ivs_c.get("s1_s2_nominal", 0.30))
        t_s2_pred       = int(s1) / float(sample_rate) + s1_s2_nominal_c
        search_half_sec = local_peak_window_ms / 2000.0

        new_s2: Optional[int] = None
        method_used: Optional[str] = None
        spectral_score: Optional[float] = None

        sens = _find_sensitive_peaks_near(
            t_s2_pred, local_peak_window_samples, local_peak_sensitivity,
            audio_envelope, analysis_data, n_samples, sample_rate, params,
        )
        if sens is not None and int(s1) < sens < int(s1_next):
            new_s2 = sens
            method_used = "sensitive_peak"

        if new_s2 is None:
            sp_result = _choose_s2_spectral(
                t_s2_pred, search_half_sec, insert_spectrum_ctx, params, sample_rate, n_samples,
            )
            if sp_result is not None:
                sp_idx, sp_score = sp_result
                if int(s1) < sp_idx < int(s1_next):
                    new_s2 = sp_idx
                    spectral_score = sp_score
                    method_used = "spectral_s2"

        if new_s2 is None:
            continue

        new_systole  = (new_s2 - int(s1))     / float(sample_rate)
        new_diastole = (int(s1_next) - new_s2) / float(sample_rate)
        s2_min_c      = float(ivs_c.get("s2_min",     0.010))
        diastole_min_c = float(ivs_c.get("diastole_min", 0.08))
        if new_systole < float(ivs_c.get("s1_s2_min", 0.12)):
            continue
        if new_diastole < s2_min_c + diastole_min_c:
            continue

        new_s2_events[i] = new_s2
        corr: Dict[str, Any] = {
            "type": method_used, "cycle": int(i), "s1": int(s1),
            "new_s2": int(new_s2), "t_s2_pred_sec": float(t_s2_pred),
            "new_systole_sec": float(new_systole), "new_diastole_sec": float(new_diastole),
        }
        if spectral_score is not None:
            corr["spectral_score"] = float(spectral_score)
        logging.info(
            "Pass 3 C.3: placed faint S2 at sample %d via %s "
            "(cycle %d, systole %.3fs\u2192%.3fs).",
            new_s2, method_used, i, diag["systole_sec"], new_systole,
        )
        return new_s1_list, new_s2_events, [corr], True

    return new_s1_list, new_s2_events, [], False


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

```

## run_pass3_correction — spectral seed + Pass A / Pass C loop (excerpt)

```python
    # ── S2 placement: Pass 2 pair seed → spectral snap (_rebuild_s2_events) ────
    _pairs = analysis_data.get("s1_s2_pairs") or []
    s2_seed = _seed_s2_from_pass2_pairs(
        s1_list, _pairs, lt_pass3, fallback_bpm, sample_rate, params, n_samples,
    )
    _pair_set = {int(s1) for s1, _ in _pairs}
    logging.info(
        "Pass 3 Step 1: seeded %d beats from Pass 2 pairs (%d paired, %d BPM fallback).",
        len(s1_list),
        sum(1 for s in s1_list if s in _pair_set),
        sum(1 for s in s1_list if s not in _pair_set),
    )

    s2_events = _rebuild_s2_events(
        s1_list, lt_pass3, fallback_bpm,
        sample_rate, params, snap_s2, snap_half, n_samples, insert_spectrum_ctx,
        noise_ivs=noise_ivs_pass3,
        seed_s2_events=s2_seed,
    )

    # ── Before-correction snapshot for HTML before/after visualization ────────
    state_boundaries_before = _build_state_boundaries_before_from_cycles(
        s1_list, s2_events, s1_half, s2_half, n_samples,
    )

    # ── Correction loop ───────────────────────────────────────────────────────
    corrections: List[Dict[str, Any]] = []
    cycle_diagnostics: List[Dict[str, Any]] = []

    for _iter in range(max_iters):
        # Sync s2_events length with current s1_list.
        n_cycles = max(0, len(s1_list) - 1)
        if len(s2_events) != n_cycles:
            s2_events = s2_events[:n_cycles]
            while len(s2_events) < n_cycles:
                s2_events.append(int(s1_list[len(s2_events)]))

        s2_events, corrs_a, cycle_diagnostics, changed_a = _pass_a_resnap_s2(
            s1_list, s2_events, lt_pass3, fallback_bpm,
            sample_rate, params, snap_s2, resnap_half, n_samples, insert_spectrum_ctx,
            systole_slack, diastole_slack, noise_ivs=noise_ivs_pass3,
        )
        corrections.extend(corrs_a)

        s1_list, s2_events, corrs_c, changed_c = _pass_c_phase_correction(
            s1_list, s2_events, cycle_diagnostics,
            all_raw_peaks, pc, lt_pass3, fallback_bpm,
            audio_envelope, analysis_data, n_samples, sample_rate, params,
            enable_phase_corr, phase_min_score_delta,
            local_peak_win_samp, local_peak_window_ms, local_peak_sens,
            0, min_sep_samples,
            snap_s2, snap_half, insert_spectrum_ctx, noise_ivs=noise_ivs_pass3,
        )
        corrections.extend(corrs_c)

        if not (changed_a or changed_c):
            break

    peaks_out = np.asarray(s1_list, dtype=np.int64)

    noise_ivs_final: List[Tuple[int, int]] = list(noise_ivs_pass3)

    # ── Debug lookup tables for reasoning payload ─────────────────────────────
    _sr_f = float(sample_rate)
    _before_s2_by_s1:   Dict[int, int] = {}
    _s2_to_s1_before:   Dict[int, int] = {}
    _before_s1next_by_s1: Dict[int, int] = {}
    for _bs, _be, _bst, _bm in state_boundaries_before:
        if _bst == "systole":
            _s1k = _bm.get("s1"); _s2k = _bm.get("s2")
            if _s1k is not None and _s2k is not None:
                _before_s2_by_s1[int(_s1k)] = int(_s2k)
                _s2_to_s1_before[int(_s2k)] = int(_s1k)
        elif _bst == "diastole":
```

## pipeline.py hook (removed)

```python
    # Pass 3 continuous emissions (optional, guarded by config).
    if params.get("pass3_generate_emissions", True):
        from emissions import generate_pass3_emissions
        generate_pass3_emissions(analysis_data, algorithm_envelope, sample_rate, params, wav_file_path)
```


---

## fft_profiles.py — Pass 3 spectral insert + template search (lines 242–401)

Archived helpers used by old Pass 3 S2 alignment / insertion.

```python
def prepare_pass3_s1_insert_context(
    audio_path: str,
    peak_classifications: Dict,
    envelope_sample_rate: int,
    audio_envelope: np.ndarray,
    params: Optional[Dict] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build bandpass audio + mean S1 shape spectrum (same construction as compute_fft_profiles)
    for Pass 3 missed-beat insertion when no raw peak exists in the search window.

    Returns None if audio is empty or there are no paired S1 peaks to form a template.
    Caller should hold bandpass_audio only for the duration of Pass 3 (can be large).
    """
    params = params or {}
    window_ms = float(params.get("fft_window_ms", 100.0))
    max_peaks_per_type = int(params.get("fft_max_peaks_per_type", 100))
    target_sr = params.get("pass3_insert_spectrum_target_sr")
    if target_sr is not None:
        target_sr = int(target_sr)

    if target_sr is not None:
        audio_raw, full_sr = librosa.load(audio_path, sr=target_sr, mono=True)
    else:
        audio_raw, full_sr = librosa.load(audio_path, sr=None, mono=True)
    if audio_raw.size == 0:
        logging.warning("Pass 3 spectrum insert: empty audio file.")
        return None

    s1_indices, s2_indices = _collect_s1_s2_indices(peak_classifications, paired_s1_only=True)
    s1_selected, s2_selected = _select_top_peaks_by_confidence(
        peak_classifications, s1_indices, s2_indices, max_per_type=max_peaks_per_type
    )
    s1_full = _peak_indices_to_full_rate(s1_selected, envelope_sample_rate, full_sr)
    s2_full = _peak_indices_to_full_rate(s2_selected, envelope_sample_rate, full_sr)

    env = np.asarray(audio_envelope)
    s1_amps = np.array(
        [env[min(int(idx), len(env) - 1)] for idx in s1_selected], dtype=np.float64
    )
    s2_amps = np.array(
        [env[min(int(idx), len(env) - 1)] for idx in s2_selected], dtype=np.float64
    )

    window_samples = int(round(window_ms * 0.001 * full_sr))
    half_samples = window_samples // 2
    n_fft = 1 << (window_samples - 1).bit_length()
    if n_fft < window_samples:
        n_fft *= 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / full_sr)

    audio_preproc = apply_bandpass_only(audio_raw, full_sr, params)
    preproc_s1_db, preproc_s2_db, n_s1_pre, n_s2_pre = _compute_profiles_from_audio(
        audio_preproc, full_sr, s1_full, s2_full, s1_amps, s2_amps, n_fft, half_samples
    )

    neutral_low = float(params.get("fft_neutral_band_low_hz", 3000.0))
    neutral_high = float(params.get("fft_neutral_band_high_hz", 5000.0))
    preproc_s1_db, preproc_s2_db = _align_s2_to_s1_in_band(
        freqs, preproc_s1_db, preproc_s2_db, neutral_low, neutral_high
    )

    if n_s1_pre < 1:
        logging.info(
            "Pass 3 spectrum insert: no paired S1 template (n_s1_pre=%s); skipping spectral insertion.",
            n_s1_pre,
        )
        return None

    return {
        "bandpass_audio": audio_preproc,
        "full_sr": int(full_sr),
        "freqs": freqs,
        "mu_s1_db": preproc_s1_db,
        "mu_s2_db": preproc_s2_db,
        "n_fft": int(n_fft),
        "half_samples": int(half_samples),
        "window_ms": window_ms,
        "n_s1_template": int(n_s1_pre),
        "n_s2_template": int(n_s2_pre),
    }


def spectrum_template_search_envelope_index(
    bandpass_audio: np.ndarray,
    full_sr: int,
    t_expected_sec: float,
    search_half_sec: float,
    mu_template_db: np.ndarray,
    freqs: np.ndarray,
    n_fft: int,
    half_samples: int,
    envelope_sample_rate: int,
    n_samples_envelope: int,
    params: Optional[Dict] = None,
) -> Optional[Tuple[int, float]]:
    """
    Slide short-time spectra over bandpass audio near t_expected; pick center that best matches
    mu_template_db in fft_separation band (negative mean squared error in dB shape space).
    Works for any template (S1, S2, or other).

    Returns (envelope_sample_index, best_score) or None if no confident winner.
    """
    params = params or {}
    low_hz = float(params.get("fft_separation_low_hz", 10.0))
    high_hz = float(params.get("fft_separation_high_hz", 15000.0))
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask) or len(mu_template_db) != len(freqs):
        return None

    stride_samples = max(
        1, int(round(float(params.get("pass3_insert_spectrum_stride_ms", 8.0)) * full_sr / 1000.0))
    )
    margin_req = float(params.get("pass3_insert_spectrum_min_margin", 0.15))
    eps = 1e-10

    center_full = int(round(float(t_expected_sec) * full_sr))
    span = int(round(float(search_half_sec) * full_sr))
    lo = max(half_samples, center_full - span)
    hi = min(len(bandpass_audio) - half_samples, center_full + span)
    if hi <= lo:
        return None

    scores: List[float] = []
    centers: List[int] = []
    for c in range(lo, hi + 1, stride_samples):
        w = _extract_window(bandpass_audio, c, half_samples)
        if w is None:
            continue
        windowed = w * np.hanning(len(w))
        rms = np.sqrt(np.mean(windowed ** 2) + eps)
        ref_db = 20.0 * np.log10(max(rms, 1e-10))
        padded = np.zeros(n_fft, dtype=np.float64)
        padded[: len(windowed)] = windowed
        fft_mag = np.abs(np.fft.rfft(padded))
        fft_db = 20.0 * np.log10(fft_mag + eps)
        spec_shape = fft_db - ref_db
        diff = spec_shape[mask] - mu_template_db[mask]
        score = -float(np.mean(diff ** 2))
        scores.append(score)
        centers.append(int(c))

    if not scores:
        return None

    scores_arr = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores_arr)[::-1]
    best_i = int(order[0])
    best_score = float(scores_arr[best_i])
    best_center = centers[best_i]

    if len(scores_arr) >= 2:
        second_best = float(scores_arr[int(order[1])])
        if best_score - second_best < margin_req:
            return None

    env_idx = int(round(best_center * float(envelope_sample_rate) / float(full_sr)))
    env_idx = max(0, min(env_idx, n_samples_envelope - 1))
    return env_idx, best_score

```
