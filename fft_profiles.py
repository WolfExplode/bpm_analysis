# fft_profiles.py
# Computes aggregate FFT profiles for S1 and S2 peaks from raw and bandpass-only audio.
# Uses full sample rate to preserve all frequency information.

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import librosa

from audio_preprocessing import apply_bandpass_only
from peak_utils import PeakType, _get_peak_type_from_debug


def _get_pairing_confidence(entry) -> Optional[float]:
    """Prefer label_scores for the peak's role; else final score from confidence_trace."""
    if not isinstance(entry, dict):
        return None
    ls = entry.get("label_scores")
    if isinstance(ls, dict):
        pt = _get_peak_type_from_debug(entry) or ""
        if PeakType.is_s1(pt):
            v = ls.get("S1")
            if isinstance(v, (int, float)):
                return float(v)
        elif PeakType.is_s2(pt):
            v = ls.get("S2")
            if isinstance(v, (int, float)):
                return float(v)
    for sec in entry.get("sections", []):
        if sec.get("type") != "confidence_trace":
            continue
        steps = sec.get("steps", [])
        if not steps:
            continue
        result = steps[-1].get("result")
        if result is not None and isinstance(result, (int, float)):
            return float(result)
    return None


def _collect_s1_s2_indices(
    peak_classifications: Dict, paired_s1_only: bool = False
) -> Tuple[list, list]:
    """Extract S1 and S2 peak indices (at envelope sample rate) from peak_classifications.
    If paired_s1_only is True, only include paired S1s (exclude Lone S1)."""
    s1_indices = []
    s2_indices = []
    for peak_idx, entry in peak_classifications.items():
        pt = _get_peak_type_from_debug(entry) or ""
        if PeakType.is_s1(pt):
            if paired_s1_only and pt.strip().startswith("Lone S1"):
                continue
            s1_indices.append(peak_idx)
        elif PeakType.is_s2(pt):
            s2_indices.append(peak_idx)
    return s1_indices, s2_indices


def _select_top_peaks_by_confidence(
    peak_classifications: Dict,
    s1_indices: list,
    s2_indices: list,
    max_per_type: int = 100,
) -> Tuple[list, list]:
    """Return up to max_per_type S1 and S2 indices with highest pairing confidence."""
    def sort_and_cap(indices: list) -> list:
        with_conf = [
            (idx, _get_pairing_confidence(peak_classifications.get(idx)) or 0.0)
            for idx in indices
        ]
        with_conf.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in with_conf[:max_per_type]]

    return sort_and_cap(s1_indices), sort_and_cap(s2_indices)


def _peak_indices_to_full_rate(
    indices: list, envelope_sample_rate: int, full_sample_rate: int
) -> np.ndarray:
    """Convert peak indices from envelope (600 Hz) to full-rate sample indices."""
    if not indices:
        return np.array([], dtype=np.int64)
    times_sec = np.array(indices, dtype=float) / envelope_sample_rate
    full_indices = (times_sec * full_sample_rate).astype(np.int64)
    return full_indices


def _extract_window(
    audio: np.ndarray, center_idx: int, half_samples: int
) -> Optional[np.ndarray]:
    """Extract a window centered at center_idx. Returns None if out of bounds."""
    start = center_idx - half_samples
    end = center_idx + half_samples
    if start < 0 or end > len(audio):
        return None
    return audio[start:end].astype(np.float64)


def _compute_profiles_from_audio(
    audio: np.ndarray,
    full_sr: int,
    s1_full: np.ndarray,
    s2_full: np.ndarray,
    s1_amps: np.ndarray,
    s2_amps: np.ndarray,
    n_fft: int,
    half_samples: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Compute FFT profiles from a single audio array.
    Normalizes in dB: each peak's spectrum is expressed relative to the RMS energy
    of the same window (fft_db - ref_db), then averaged. Both profiles are then
    shifted to a common reference (mean of S1 and S2 aggregate amplitude) so
    S1 and S2 are on the same scale and only spectral shape differences remain.
    Returns (profile_s1_db, profile_s2_db, n_s1, n_s2).
    """
    profile_s1_sum = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    profile_s2_sum = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    eps = 1e-10
    rms_floor = 1e-10  # Minimum RMS for dB conversion to avoid -inf

    def _add_to_profile(
        indices: np.ndarray, amps: np.ndarray, profile_sum: np.ndarray
    ) -> Tuple[int, float]:
        """Returns (count, sum_ref_db) for aggregate amplitude normalization."""
        count = 0
        sum_ref_db = 0.0
        for i, idx in enumerate(indices):
            window = _extract_window(audio, idx, half_samples)
            if window is None:
                continue
            windowed = window * np.hanning(len(window))
            rms = np.sqrt(np.mean(windowed ** 2) + eps)
            ref_db = 20.0 * np.log10(max(rms, rms_floor))
            padded = np.zeros(n_fft, dtype=np.float64)
            padded[: len(windowed)] = windowed
            fft_mag = np.abs(np.fft.rfft(padded))
            fft_db = 20.0 * np.log10(fft_mag + eps)
            profile_sum += fft_db - ref_db
            sum_ref_db += ref_db
            count += 1
        return count, sum_ref_db

    n_s1, sum_ref_s1 = _add_to_profile(s1_full, s1_amps, profile_s1_sum)
    n_s2, sum_ref_s2 = _add_to_profile(s2_full, s2_amps, profile_s2_sum)

    if n_s1 > 0:
        profile_s1_db = profile_s1_sum / n_s1
    else:
        profile_s1_db = profile_s1_sum.copy()
    if n_s2 > 0:
        profile_s2_db = profile_s2_sum / n_s2
    else:
        profile_s2_db = profile_s2_sum.copy()

    # Normalize both profiles to a common scale using aggregate S1/S2 amplitude
    if n_s1 > 0 and n_s2 > 0:
        mean_ref_s1 = sum_ref_s1 / n_s1
        mean_ref_s2 = sum_ref_s2 / n_s2
        common_ref = (mean_ref_s1 + mean_ref_s2) / 2.0
        profile_s1_db = profile_s1_db + (mean_ref_s1 - common_ref)
        profile_s2_db = profile_s2_db + (mean_ref_s2 - common_ref)

    return profile_s1_db, profile_s2_db, n_s1, n_s2


def compute_fft_profiles(
    audio_path: str,
    peak_classifications: Dict,
    envelope_sample_rate: int,
    audio_envelope: np.ndarray,
    params: Optional[Dict] = None,
    target_sr: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Compute aggregate FFT profiles for S1 and S2 from raw and bandpass-only audio.
    Normalizes by RMS energy of each peak's window so each peak contributes equally.

    target_sr: If set (e.g. 32000), resample audio to this rate so all files share
        the same frequency grid for aggregation. If None, use native sample rate.

    Returns:
        (freqs, raw_s1_db, raw_s2_db, bandpass_s1_db, bandpass_s2_db, n_s1, n_s2)
    """
    params = params or {}
    window_ms = float(params.get("fft_window_ms", 100.0))
    max_peaks_per_type = int(params.get("fft_max_peaks_per_type", 100))

    if target_sr is not None:
        audio_raw, full_sr = librosa.load(audio_path, sr=target_sr, mono=True)
    else:
        audio_raw, full_sr = librosa.load(audio_path, sr=None, mono=True)
    if audio_raw.size == 0:
        logging.warning("Empty audio file for FFT profiles.")
        empty = np.array([])
        return (empty, empty, empty, empty, empty, 0, 0)

    s1_indices, s2_indices = _collect_s1_s2_indices(peak_classifications, paired_s1_only=True)
    s1_selected, s2_selected = _select_top_peaks_by_confidence(
        peak_classifications, s1_indices, s2_indices, max_per_type=max_peaks_per_type
    )
    s1_full = _peak_indices_to_full_rate(s1_selected, envelope_sample_rate, full_sr)
    s2_full = _peak_indices_to_full_rate(s2_selected, envelope_sample_rate, full_sr)

    # Envelope amplitudes at each peak (for amplitude normalization)
    env = np.asarray(audio_envelope)
    s1_amps = np.array([env[min(int(idx), len(env) - 1)] for idx in s1_selected], dtype=np.float64)
    s2_amps = np.array([env[min(int(idx), len(env) - 1)] for idx in s2_selected], dtype=np.float64)

    window_samples = int(round(window_ms * 0.001 * full_sr))
    half_samples = window_samples // 2
    n_fft = 1 << (window_samples - 1).bit_length()
    if n_fft < window_samples:
        n_fft *= 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / full_sr)

    raw_s1_db, raw_s2_db, n_s1_raw, n_s2_raw = _compute_profiles_from_audio(
        audio_raw, full_sr, s1_full, s2_full, s1_amps, s2_amps, n_fft, half_samples
    )
    audio_preproc = apply_bandpass_only(audio_raw, full_sr, params)
    preproc_s1_db, preproc_s2_db, n_s1_pre, n_s2_pre = _compute_profiles_from_audio(
        audio_preproc, full_sr, s1_full, s2_full, s1_amps, s2_amps, n_fft, half_samples
    )

    # Align S2 to S1 in a neutral high band so curves converge there (shape-only difference in 0–500 Hz)
    neutral_low = float(params.get("fft_neutral_band_low_hz", 3000.0))
    neutral_high = float(params.get("fft_neutral_band_high_hz", 5000.0))
    raw_s1_db, raw_s2_db = _align_s2_to_s1_in_band(freqs, raw_s1_db, raw_s2_db, neutral_low, neutral_high)
    preproc_s1_db, preproc_s2_db = _align_s2_to_s1_in_band(freqs, preproc_s1_db, preproc_s2_db, neutral_low, neutral_high)

    logging.info(
        f"FFT profiles: raw ({n_s1_raw} S1, {n_s2_raw} S2), bandpass ({n_s1_pre} S1, {n_s2_pre} S2), "
        f"{full_sr} Hz, {window_ms} ms window"
    )
    return freqs, raw_s1_db, raw_s2_db, preproc_s1_db, preproc_s2_db, n_s1_raw, n_s2_raw


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
        "n_fft": int(n_fft),
        "half_samples": int(half_samples),
        "window_ms": window_ms,
        "n_s1_template": int(n_s1_pre),
        "n_s2_template": int(n_s2_pre),
    }


def spectrum_s1_search_envelope_index(
    bandpass_audio: np.ndarray,
    full_sr: int,
    t_expected_sec: float,
    search_half_sec: float,
    mu_s1_db: np.ndarray,
    freqs: np.ndarray,
    n_fft: int,
    half_samples: int,
    envelope_sample_rate: int,
    n_samples_envelope: int,
    params: Optional[Dict] = None,
) -> Optional[Tuple[int, float]]:
    """
    Slide short-time spectra over bandpass audio near t_expected; pick center that best matches
    mu_s1_db in fft_separation band (negative mean squared error in dB shape space).

    Returns (envelope_sample_index, best_score) or None if no confident winner.
    """
    params = params or {}
    low_hz = float(params.get("fft_separation_low_hz", 10.0))
    high_hz = float(params.get("fft_separation_high_hz", 15000.0))
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask) or len(mu_s1_db) != len(freqs):
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
        diff = spec_shape[mask] - mu_s1_db[mask]
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


def _align_s2_to_s1_in_band(
    freqs: np.ndarray,
    profile_s1_db: np.ndarray,
    profile_s2_db: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shift S2 profile so its mean dB in [low_hz, high_hz] equals S1's (neutral-band alignment)."""
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return profile_s1_db, profile_s2_db
    mean_s1 = float(np.nanmean(profile_s1_db[mask]))
    mean_s2 = float(np.nanmean(profile_s2_db[mask]))
    offset = mean_s1 - mean_s2
    return profile_s1_db, profile_s2_db + offset


def compute_frequency_separation(
    freqs: np.ndarray,
    profile_s1_db: np.ndarray,
    profile_s2_db: np.ndarray,
    params: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Build the S1 vs S2 frequency comparison vector over a configurable band (e.g. 10–15000 Hz).
    separation_db = profile_s1_db - profile_s2_db; positive means S1 stronger at that frequency.
    Not used for any logic yet; infrastructure for future pass 3 / classifier use.
    Returns dict with keys "freqs" and "separation_db" (1d arrays in band), or None if invalid.
    """
    if freqs is None or len(freqs) == 0 or profile_s1_db is None or profile_s2_db is None:
        return None
    if len(profile_s1_db) != len(freqs) or len(profile_s2_db) != len(freqs):
        return None
    params = params or {}
    low_hz = float(params.get("fft_separation_low_hz", 10.0))
    high_hz = float(params.get("fft_separation_high_hz", 15000.0))
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return None
    freqs_band = np.asarray(freqs[mask], dtype=np.float64)
    separation_db = np.asarray(profile_s1_db[mask] - profile_s2_db[mask], dtype=np.float64)
    return {"freqs": freqs_band, "separation_db": separation_db}


def aggregate_fft_profiles(
    file_results: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]],
    params: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate FFT profiles from multiple files (same frequency grid assumed, e.g. target_sr=32000).
    Each file_result is (freqs, raw_s1_db, raw_s2_db, bandpass_s1_db, bandpass_s2_db, n_s1, n_s2).
    Returns (freqs, agg_raw_s1, agg_raw_s2, agg_bandpass_s1, agg_bandpass_s2) weighted by peak counts.
    Applies neutral-band alignment (S2 to S1) after aggregation when params has fft_neutral_band_*.
    """
    if not file_results:
        empty = np.array([])
        return (empty, empty, empty, empty, empty)
    freqs = file_results[0][0]
    n_bins = len(freqs)
    sum_raw_s1 = np.zeros(n_bins, dtype=np.float64)
    sum_raw_s2 = np.zeros(n_bins, dtype=np.float64)
    sum_bandpass_s1 = np.zeros(n_bins, dtype=np.float64)
    sum_bandpass_s2 = np.zeros(n_bins, dtype=np.float64)
    total_n_s1 = 0
    total_n_s2 = 0
    for (f, r1, r2, b1, b2, n1, n2) in file_results:
        if len(f) != n_bins:
            continue
        sum_raw_s1 += r1 * n1
        sum_raw_s2 += r2 * n2
        sum_bandpass_s1 += b1 * n1
        sum_bandpass_s2 += b2 * n2
        total_n_s1 += n1
        total_n_s2 += n2
    if total_n_s1 == 0 and total_n_s2 == 0:
        return (freqs, sum_raw_s1, sum_raw_s2, sum_bandpass_s1, sum_bandpass_s2)
    if total_n_s1 > 0:
        sum_raw_s1 /= total_n_s1
        sum_bandpass_s1 /= total_n_s1
    if total_n_s2 > 0:
        sum_raw_s2 /= total_n_s2
        sum_bandpass_s2 /= total_n_s2
    params = params or {}
    neutral_low = float(params.get("fft_neutral_band_low_hz", 3000.0))
    neutral_high = float(params.get("fft_neutral_band_high_hz", 5000.0))
    sum_raw_s1, sum_raw_s2 = _align_s2_to_s1_in_band(freqs, sum_raw_s1, sum_raw_s2, neutral_low, neutral_high)
    sum_bandpass_s1, sum_bandpass_s2 = _align_s2_to_s1_in_band(freqs, sum_bandpass_s1, sum_bandpass_s2, neutral_low, neutral_high)
    return (freqs, sum_raw_s1, sum_raw_s2, sum_bandpass_s1, sum_bandpass_s2)


def _build_fft_figure(
    freqs: np.ndarray,
    raw_s1_db: np.ndarray,
    raw_s2_db: np.ndarray,
    bandpass_s1_db: np.ndarray,
    bandpass_s2_db: np.ndarray,
    window_ms: float = 100.0,
) -> go.Figure:
    """Build the shared FFT profile Plotly figure (dB + linear, 0–500 Hz)."""
    trace_style = dict(width=1.5)
    raw_s1_lin = np.power(10.0, raw_s1_db / 20.0)
    raw_s2_lin = np.power(10.0, raw_s2_db / 20.0)
    bandpass_s1_lin = np.power(10.0, bandpass_s1_db / 20.0)
    bandpass_s2_lin = np.power(10.0, bandpass_s2_db / 20.0)
    linear_vals = np.concatenate([raw_s1_lin, raw_s2_lin, bandpass_s1_lin, bandpass_s2_lin])
    linear_min = float(np.min(linear_vals))
    linear_max = float(np.max(linear_vals))
    pad = (linear_max - linear_min) * 0.05 if linear_max > linear_min else 0.1
    linear_range = [max(0, linear_min - pad), linear_max + pad]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=raw_s1_db, name="S1 (raw)", line=trace_style))
    fig.add_trace(go.Scatter(x=freqs, y=raw_s2_db, name="S2 (raw)", line=trace_style))
    fig.add_trace(go.Scatter(x=freqs, y=bandpass_s1_db, name="S1 (bandpass)", line=trace_style, visible="legendonly"))
    fig.add_trace(go.Scatter(x=freqs, y=bandpass_s2_db, name="S2 (bandpass)", line=trace_style, visible="legendonly"))
    fig.add_trace(go.Scatter(x=freqs, y=raw_s1_lin, name="S1 (raw)", line=trace_style, visible=False))
    fig.add_trace(go.Scatter(x=freqs, y=raw_s2_lin, name="S2 (raw)", line=trace_style, visible=False))
    fig.add_trace(go.Scatter(x=freqs, y=bandpass_s1_lin, name="S1 (bandpass)", line=trace_style, visible=False))
    fig.add_trace(go.Scatter(x=freqs, y=bandpass_s2_lin, name="S2 (bandpass)", line=trace_style, visible=False))
    fig.update_layout(
        template="plotly_dark",
        title=f"S1 / S2 FFT Profiles ({window_ms:.0f} ms window)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (dB)",
        margin=dict(t=60, b=60, l=70, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        dragmode="pan",
        uirevision="layout-stable",
        autosize=True,
        updatemenus=[
            dict(
                active=0,
                type="dropdown",
                direction="down",
                x=0.01,
                y=1.08,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(label="dB", method="update", args=[
                        {"visible": [True, True, "legendonly", "legendonly", False, False, False, False]},
                        {"yaxis.title": "Magnitude (dB)", "yaxis.range": [-50, 90]},
                    ]),
                    dict(label="Linear", method="update", args=[
                        {"visible": [False, False, False, False, True, True, "legendonly", "legendonly"]},
                        {"yaxis.title": "Magnitude", "yaxis.range": linear_range},
                    ]),
                ],
            ),
        ],
    )
    fig.update_xaxes(range=[0, 500], automargin=False)
    fig.update_yaxes(range=[0, 70], automargin=False)
    return fig


def _write_fft_html(fig: go.Figure, output_path: str, page_title: str, header_label: str) -> None:
    """Write FFT figure to HTML with shared wrapper."""
    plot_config = {"scrollZoom": True, "showTips": False}
    plotly_html = fig.to_html(full_html=False, config=plot_config, include_plotlyjs="cdn")
    # If CDN is unavailable, fall back to a local plotly.min.js beside the HTML (if present).
    plotly_html = re.sub(
        r'<script\s+src="(https://cdn\.plot\.ly/plotly[^"]+)"\s*></script>',
        r'<script src="\1" onerror="this.onerror=null;this.src=\'plotly.min.js\';"></script>',
        plotly_html,
        count=1,
    )
    wrapper_html = f"""<!DOCTYPE html>
<html style="height:100%;margin:0;padding:0;">
<head>
    <meta charset="utf-8" />
    <title>{page_title}</title>
    <style>
        html, body {{ height: 100%; margin: 0; padding: 0; background-color: #111; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #e0e0e0; }}
        body {{ display: flex; flex-direction: column; }}
        #fft-header {{ display: flex; align-items: center; gap: 8px; padding: 4px 8px; background: rgba(40, 40, 50, 0.6); border-bottom: 1px solid #333; font-size: 12px; flex-shrink: 0; }}
        #fft-header .chart-toolbar-title {{ color: #aaa; white-space: nowrap; }}
        #fft-header #audio-file-name {{ font-size: 12px; color: #ccc; white-space: nowrap; }}
        #fft-plot-container {{ flex: 1 1 0; min-height: 0; display: flex; flex-direction: column; }}
        #fft-plot-container > div {{ flex: 1 1 0 !important; min-height: 0 !important; height: 100% !important; }}
    </style>
</head>
<body>
    <div id="fft-header">
        <span class="chart-toolbar-title">S1 / S2 FFT Profiles – </span>
        <span id="audio-file-name" title="{header_label}">{header_label}</span>
    </div>
    <div id="fft-plot-container">
        {plotly_html}
    </div>
</body>
</html>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(wrapper_html)
    logging.info(f"FFT profiles saved to {output_path}")


def save_fft_profiles_html(
    audio_path: str,
    peak_classifications: Dict,
    envelope_sample_rate: int,
    output_path: str,
    audio_envelope: np.ndarray,
    params: Optional[Dict] = None,
    fft_result: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]] = None,
) -> None:
    """
    Compute FFT profiles and save a minimal HTML plot.

    Args:
        audio_path: Path to the WAV file.
        peak_classifications: From analysis_data["peak_classifications"] (pass 2 classification result).
        envelope_sample_rate: Sample rate used for peak detection (e.g. 600 Hz).
        output_path: Path for the output HTML file.
        audio_envelope: Envelope used for peak detection; amplitudes used for normalization.
        params: Optional config dict for fft_window_ms.
        fft_result: If provided, use this 7-tuple from compute_fft_profiles instead of computing (e.g. for aggregate collection).
    """
    params = params or {}
    if fft_result is not None:
        freqs, raw_s1_db, raw_s2_db, preproc_s1_db, preproc_s2_db = fft_result[0], fft_result[1], fft_result[2], fft_result[3], fft_result[4]
        if freqs.size == 0:
            logging.warning("No FFT data to plot; skipping FFT profiles HTML.")
            return
        window_ms = float(params.get("fft_window_ms", 100.0))
        fig = _build_fft_figure(freqs, raw_s1_db, raw_s2_db, preproc_s1_db, preproc_s2_db, window_ms)
        _write_fft_html(fig, output_path, f"S1 / S2 FFT Profiles - {os.path.basename(audio_path)}", os.path.basename(audio_path))
        return
    s1_indices, s2_indices = _collect_s1_s2_indices(peak_classifications, paired_s1_only=True)
    n_s1, n_s2 = len(s1_indices), len(s2_indices)
    if n_s1 == 0:
        logging.info("FFT profiles: no paired S1 labels; skipping FFT profiles HTML.")
        return
    if n_s2 <= 0.5 * n_s1:
        logging.warning(
            "FFT profiles: S2 labels (%d) are not > 50%% of S1 labels (%d); skipping (insufficient data).",
            n_s2,
            n_s1,
        )
        return

    result = compute_fft_profiles(
        audio_path, peak_classifications, envelope_sample_rate, audio_envelope, params
    )
    freqs, raw_s1_db, raw_s2_db, preproc_s1_db, preproc_s2_db = result[0], result[1], result[2], result[3], result[4]
    if freqs.size == 0:
        logging.warning("No FFT data to plot; skipping FFT profiles HTML.")
        return

    window_ms = float(params.get("fft_window_ms", 100.0))
    fig = _build_fft_figure(freqs, raw_s1_db, raw_s2_db, preproc_s1_db, preproc_s2_db, window_ms)
    audio_file_name = os.path.basename(audio_path)
    _write_fft_html(fig, output_path, f"S1 / S2 FFT Profiles - {audio_file_name}", audio_file_name)


def save_aggregate_fft_profiles_html(
    freqs: np.ndarray,
    raw_s1_db: np.ndarray,
    raw_s2_db: np.ndarray,
    bandpass_s1_db: np.ndarray,
    bandpass_s2_db: np.ndarray,
    output_path: str,
    params: Optional[Dict] = None,
) -> None:
    """Save aggregated FFT profiles (from aggregate_fft_profiles) to HTML."""
    params = params or {}
    window_ms = float(params.get("fft_window_ms", 100.0))
    if freqs.size == 0:
        logging.warning("No FFT data for aggregate; skipping aggregate FFT HTML.")
        return
    fig = _build_fft_figure(freqs, raw_s1_db, raw_s2_db, bandpass_s1_db, bandpass_s2_db, window_ms)
    _write_fft_html(fig, output_path, "Aggregated S1 / S2 FFT Profiles", "Aggregated")
