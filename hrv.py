import datetime
import logging
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, lombscargle
from typing import List, Dict, Tuple, Optional

from time_utils import dense_time_grid, STANDARD_DT_SEC
from config import param

_LOMB_FREQS: Optional[np.ndarray] = None
_LOMB_ANGULAR: Optional[np.ndarray] = None


def _lomb_frequency_grid() -> Tuple[np.ndarray, np.ndarray]:
    """Cached 0.001–0.5 Hz grid (1000 points) for Lomb–Scargle band integration."""
    global _LOMB_FREQS, _LOMB_ANGULAR
    if _LOMB_FREQS is None:
        _LOMB_FREQS = np.linspace(0.001, 0.5, 1000, dtype=np.float64)
        _LOMB_ANGULAR = (2.0 * np.pi * _LOMB_FREQS).astype(np.float64)
    return _LOMB_FREQS, _LOMB_ANGULAR


def median_mad_keep_mask_time_window(
    times_sec: np.ndarray,
    values: np.ndarray,
    half_window_sec: float,
    mad_k: float,
) -> np.ndarray:
    """
    Outlier mask: same rule as |t - t_i| <= half_window on sorted time axis,
    using searchsorted for the window bounds (times_sec must be non-decreasing).

    Public: shared local-window MAD filter used by both hrv (BPM scatter) and
    correction (Pass 3 phase durations); the global variant below stays hrv-internal.
    """
    n = len(values)
    keep = np.ones(n, dtype=bool)
    t = np.asarray(times_sec, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    hw = float(half_window_sec)
    for i in range(n):
        lo = np.searchsorted(t, t[i] - hw, side="left")
        hi = np.searchsorted(t, t[i] + hw, side="right")
        if hi <= lo:
            continue
        window_vals = v[lo:hi]
        local_median = np.median(window_vals)
        local_mad = np.median(np.abs(window_vals - local_median))
        if local_mad > 1e-9:
            keep[i] = np.abs(v[i] - local_median) <= mad_k * local_mad
    return keep


def _median_mad_keep_mask_global(values: np.ndarray, mad_k: float) -> np.ndarray:
    """Keep points within global median ± mad_k * MAD (robust z-score). If MAD ~ 0, keep all."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n == 0:
        return np.array([], dtype=bool)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    if mad <= 1e-9:
        return np.ones(n, dtype=bool)
    return np.abs(v - med) <= float(mad_k) * mad


def filter_interval_durations_by_limits(
    times_sec: np.ndarray,
    intervals_sec: np.ndarray,
    *,
    kind: str,
    params: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop (time, duration) samples outside wide plausible bounds before MAD / other outlier logic.

    kind: "systole" (S1→S2 duration) or "diastole" (S2→next S1 duration).

    Bounds default to very wide values so only obviously broken data is removed.
    """
    pc = params or {}
    t = np.asarray(times_sec, dtype=np.float64)
    v = np.asarray(intervals_sec, dtype=np.float64)
    if len(t) == 0 or len(v) == 0 or len(t) != len(v):
        return t, v

    if kind == "systole":
        lo = float(param(pc, "systole_duration_clamp_min_sec"))
        hi = float(param(pc, "systole_duration_clamp_max_sec"))
    elif kind == "diastole":
        lo = float(param(pc, "diastole_duration_clamp_min_sec"))
        hi = float(param(pc, "diastole_duration_clamp_max_sec"))
    else:
        return t, v

    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return t, v

    keep = np.isfinite(t) & np.isfinite(v) & (v >= lo) & (v <= hi)
    return t[keep], v[keep]


def _lombscargle_band_powers(
    times_sec: np.ndarray, rr_ms: np.ndarray, include_vlf: bool = False
) -> Optional[Dict[str, float]]:
    """
    Compute power in Task Force bands (VLF, LF, HF) via Lomb-Scargle periodogram.
    times_sec: start time of each RR interval (same length as rr_ms).
    rr_ms: RR intervals in milliseconds.
    Returns dict with lf_power, hf_power, total_power, lf_hf_ratio; if include_vlf, also vlf_power (ms²).
    """
    if len(times_sec) < 10 or len(rr_ms) < 10:
        logging.debug(
            "Lomb-Scargle: skipping (too few points): len(times_sec)=%d, len(rr_ms)=%d",
            len(times_sec), len(rr_ms),
        )
        return None
    if len(times_sec) != len(rr_ms):
        logging.warning(
            "Lomb-Scargle: length mismatch (times_sec=%d, rr_ms=%d). Check window slice.",
            len(times_sec), len(rr_ms),
        )
        return None
    freqs, angular_freqs = _lomb_frequency_grid()
    ts = np.ascontiguousarray(np.asarray(times_sec, dtype=np.float64))
    rr = np.ascontiguousarray(np.asarray(rr_ms, dtype=np.float64))
    try:
        periodogram = lombscargle(ts, rr, angular_freqs, normalize=True)
    except Exception as e:
        logging.warning("Lomb-Scargle: lombscargle() failed: %s", e)
        return None
    # Task Force bands: VLF 0.003-0.04, LF 0.04-0.15, HF 0.15-0.40 Hz
    # With normalize=True the periodogram is dimensionless; scale by RR variance to get power in ms² (Task Force convention).
    vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
    raw_vlf = float(np.trapezoid(periodogram[vlf_mask], freqs[vlf_mask])) if np.any(vlf_mask) else 0.0
    raw_lf = float(np.trapezoid(periodogram[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else 0.0
    raw_hf = float(np.trapezoid(periodogram[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else 0.0
    raw_total = raw_vlf + raw_lf + raw_hf
    var_rr = float(np.var(rr_ms))
    if raw_total > 1e-20 and var_rr > 0:
        scale = var_rr / raw_total
        vlf_power = raw_vlf * scale
        lf_power = raw_lf * scale
        hf_power = raw_hf * scale
        total_power = var_rr
    else:
        vlf_power, lf_power, hf_power = raw_vlf, raw_lf, raw_hf
        total_power = raw_total
    lf_hf_ratio = (lf_power / hf_power) if hf_power > 0 else 0.0
    out = {
        "lf_power": lf_power,
        "hf_power": hf_power,
        "total_power": total_power,
        "lf_hf_ratio": lf_hf_ratio,
    }
    if include_vlf:
        out["vlf_power"] = vlf_power
    return out


def calculate_windowed_hrv(s1_peaks: np.ndarray, sample_rate: int, params: Dict) -> pd.DataFrame:
    """ Calculates HRV metrics using R-R intervals based on changing heart rate """
    window_size_beats = params['hrv_window_size_beats']
    step_size_beats = params['hrv_step_size_beats']
    enable_freq = param(params, "enable_hrv_frequency_domain")

    # First, calculate all R-R intervals from the S1 peaks
    if len(s1_peaks) < window_size_beats:
        logging.warning("Not enough beats (%d) to perform windowed HRV analysis with a window of %d beats.", len(s1_peaks), window_size_beats)
        return pd.DataFrame(columns=['time', 'rmssdc', 'sdnn', 'bpm'])

    rr_intervals_sec = np.diff(s1_peaks) / sample_rate
    s1_times_sec = s1_peaks / sample_rate

    results = []
    # Iterate through the R-R intervals with a sliding window
    for i in range(0, len(rr_intervals_sec) - window_size_beats + 1, step_size_beats):
        window_rr_sec = rr_intervals_sec[i : i + window_size_beats]
        window_rr_ms = window_rr_sec * 1000
        start_time = s1_times_sec[i]
        end_time = s1_times_sec[i + window_size_beats]
        window_mid_time = (start_time + end_time) / 2.0

        # --- Calculate HRV Metrics for the Window ---
        mean_rr_ms = np.mean(window_rr_ms)
        sdnn = np.std(window_rr_ms)
        successive_diffs_ms = np.diff(window_rr_ms)
        rmssd = np.sqrt(np.mean(successive_diffs_ms**2))

        # --- Calculate Corrected RMSSD (RMSSDc) ---
        mean_rr_sec = mean_rr_ms / 1000.0
        rmssdc = rmssd / mean_rr_sec if mean_rr_sec > 0 else 0

        # Calculate the average BPM within this specific window
        window_bpm = 60 / mean_rr_sec if mean_rr_sec > 0 else 0

        row = {
            'time': window_mid_time,
            'rmssdc': rmssdc,
            'sdnn': sdnn,
            'bpm': window_bpm
        }
        if enable_freq:
            # Interval start times for this window (one per RR: peak i starts interval 0, ..., peak i+window_size_beats-1 starts last)
            window_times_sec = s1_times_sec[i : i + window_size_beats]
            band_powers = _lombscargle_band_powers(window_times_sec, window_rr_ms, include_vlf=False)
            if band_powers is not None:
                row["lf_power"] = band_powers["lf_power"]
                row["hf_power"] = band_powers["hf_power"]
                row["total_power"] = band_powers["total_power"]
                row["lf_hf_ratio"] = band_powers["lf_hf_ratio"]
            else:
                row["lf_power"] = np.nan
                row["hf_power"] = np.nan
                row["total_power"] = np.nan
                row["lf_hf_ratio"] = np.nan
        results.append(row)

    if enable_freq and results:
        freq_ok = sum(1 for r in results if "lf_hf_ratio" in r and not np.isnan(r["lf_hf_ratio"]))
        if freq_ok == 0:
            logging.warning(
                "Windowed HRV frequency: all %d windows had no valid Lomb-Scargle result (check logs above for length mismatch or lombscargle errors).",
                len(results),
            )
        elif freq_ok < len(results):
            logging.info(
                "Windowed HRV frequency: %d/%d windows had valid LF/HF; %d had NaN.",
                freq_ok, len(results), len(results) - freq_ok,
            )

    if not results:
        logging.warning("Could not perform windowed HRV analysis. Recording may be too short or have too few beats.")
        return pd.DataFrame(columns=['time', 'rmssdc', 'sdnn', 'bpm'])

    logging.info("Beat-based windowed HRV analysis complete. Generated %d data points.", len(results))
    return pd.DataFrame(results)


def calculate_global_hrv_frequency(
    s1_peaks: np.ndarray, sample_rate: int, params: Dict
) -> Optional[Dict[str, float]]:
    """Compute one Lomb-Scargle spectrum over the full recording. Returns VLF/LF/HF (ms²) and LF/HF when duration >= hrv_global_min_duration_sec."""
    if len(s1_peaks) < 2:
        return None
    rr_sec = np.diff(s1_peaks) / float(sample_rate)
    rr_ms = rr_sec * 1000.0
    times_sec = s1_peaks[:-1] / float(sample_rate)
    duration_sec = float(times_sec[-1] - times_sec[0]) + (rr_sec[-1] if len(rr_sec) else 0)
    min_duration = param(params, "hrv_global_min_duration_sec")
    if duration_sec < min_duration or len(rr_ms) < 20:
        return None
    band_powers = _lombscargle_band_powers(times_sec, rr_ms, include_vlf=True)
    if band_powers is None:
        return None
    logging.info(
        "Global HRV spectrum (%.1f min): VLF=%.2f, LF=%.2f, HF=%.2f ms² ; total=%.2f ms² ; LF/HF=%.2f",
        duration_sec / 60.0,
        band_powers.get("vlf_power", 0),
        band_powers["lf_power"],
        band_powers["hf_power"],
        band_powers["total_power"],
        band_powers["lf_hf_ratio"],
    )
    return {
        "vlf_power": band_powers["vlf_power"],
        "lf_power": band_powers["lf_power"],
        "hf_power": band_powers["hf_power"],
        "total_power": band_powers["total_power"],
        "lf_hf_ratio": band_powers["lf_hf_ratio"],
    }


def _gaussian_kernel_smooth(
    t_evals: np.ndarray,
    t_data: np.ndarray,
    y_data: np.ndarray,
    sigma_sec: float,
) -> np.ndarray:
    """
    Gaussian kernel regression for irregular time series.

    For each t in t_evals: y(t) = sum_i exp(-0.5 * ((t - t_i)/sigma)^2) * y_i / sum_i w_i
    """
    t_evals = np.asarray(t_evals, dtype=float)
    t_data = np.asarray(t_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    if len(t_data) == 0 or len(t_data) != len(y_data):
        return np.zeros(len(t_evals), dtype=float)
    sigma = float(sigma_sec)
    if not np.isfinite(sigma) or sigma <= 1e-9:
        # No smoothing; nearest-neighbor via interpolation fallback.
        return np.interp(t_evals, t_data, y_data, left=float(y_data[0]), right=float(y_data[-1]))
    y_out = np.empty(len(t_evals), dtype=float)
    for i, t in enumerate(t_evals):
        d = (t_data - float(t)) / sigma
        w = np.exp(-0.5 * d * d)
        ws = float(np.sum(w))
        if ws <= 1e-12:
            y_out[i] = float(np.mean(y_data))
        else:
            y_out[i] = float(np.dot(w, y_data) / ws)
    return y_out


def _gaussian_sigma_from_frac_and_spacing(
    t_data: np.ndarray,
    frac: float,
) -> float:
    """
    Map a legacy curve \"frac\" to an approximate Gaussian sigma in seconds.
    Uses median sample spacing * (frac*n) as an effective span; sigma ~ span/3.
    """
    t_data = np.asarray(t_data, dtype=float)
    if len(t_data) < 3:
        return 1.0
    dt = np.diff(np.sort(t_data))
    dt = dt[np.isfinite(dt) & (dt > 1e-9)]
    med_dt = float(np.median(dt)) if len(dt) else 1.0
    n = len(t_data)
    k = max(3, int(np.ceil(float(frac) * n)))
    span = med_dt * float(k)
    return max(0.25 * med_dt, span / 3.0)


def compute_pass1_bpm_curve(
    anchor_beats: np.ndarray, sample_rate: int, params: Dict
) -> Optional[Dict[str, np.ndarray]]:
    """
    Canonical pass 1 BPM curve: instant BPM from anchor beats, local then global outlier removal
    (median+MAD; global pass skipped if pass1_bpm_global_outlier_mad_k <= 0), then light Gaussian smoothing. Used for the
    time-varying prior, recovery phase, and all plots so display
    matches algorithm input.
    Returns dict with curve_times, curve_bpm (dense Gaussian-smoothed), scatter_times, scatter_bpm (filtered instant),
    raw_scatter_times, raw_scatter_bpm (instant BPM before any outlier removal), or None if insufficient data.
    """
    if anchor_beats is None or len(anchor_beats) < 2:
        return None
    peak_times = anchor_beats.astype(float) / sample_rate
    rr_sec = np.diff(peak_times)
    valid = rr_sec > 1e-6
    if not np.any(valid):
        return None
    instant_bpm = 60.0 / rr_sec[valid]
    times_sec = peak_times[1:][valid]
    raw_scatter_times = np.asarray(times_sec, dtype=float)
    raw_scatter_bpm = np.asarray(instant_bpm, dtype=float)

    # Outlier removal: keep point if within median ± k*MAD in local window
    half_window_sec = float(param(params, "pass1_bpm_outlier_window_sec"))
    mad_k = float(param(params, "pass1_bpm_outlier_mad_k"))
    keep = median_mad_keep_mask_time_window(times_sec, instant_bpm, half_window_sec, mad_k)
    scatter_bpm = np.asarray(instant_bpm[keep], dtype=float)
    scatter_times = np.asarray(times_sec[keep], dtype=float)

    global_mad_k = float(param(params, "pass1_bpm_global_outlier_mad_k"))
    if global_mad_k > 0 and len(scatter_bpm) > 0:
        gkeep = _median_mad_keep_mask_global(scatter_bpm, global_mad_k)
        scatter_bpm = np.asarray(scatter_bpm[gkeep], dtype=float)
        scatter_times = np.asarray(scatter_times[gkeep], dtype=float)

    if len(scatter_times) < 3:
        return None

    gaussian_frac = float(param(params, "pass1_bpm_gaussian_frac"))
    # Canonical dense curve on the standardized dt raster.
    curve_times = dense_time_grid(float(scatter_times.max()), STANDARD_DT_SEC)
    curve_times = curve_times[curve_times >= float(scatter_times.min())]
    sigma_sec = _gaussian_sigma_from_frac_and_spacing(scatter_times, gaussian_frac)
    curve_bpm = _gaussian_kernel_smooth(curve_times, scatter_times, scatter_bpm, sigma_sec)

    return {
        "curve_times": curve_times,
        "curve_bpm": curve_bpm,
        "scatter_times": scatter_times,
        "scatter_bpm": scatter_bpm,
        "raw_scatter_times": raw_scatter_times,
        "raw_scatter_bpm": raw_scatter_bpm,
    }


def filter_instant_bpm_mad(
    bpm_times: np.ndarray, instant_bpm: np.ndarray, params: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply MAD-based outlier removal to instantaneous BPM (pass 2 and pass 3):
    run the local-window MAD filter twice (second pass uses a wider window and a less
    aggressive threshold to catch remaining non-local spikes without a global rule).
    Returns (filtered_bpm_times, filtered_instant_bpm).
    """
    if bpm_times is None or instant_bpm is None or len(bpm_times) != len(instant_bpm) or len(bpm_times) < 2:
        return np.array([]), np.array([])
    bpm_times = np.asarray(bpm_times, dtype=float)
    instant_bpm = np.asarray(instant_bpm, dtype=float)

    def _apply_local_mad(
        t_in: np.ndarray, v_in: np.ndarray, half_window_sec: float, mad_k: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        keep = median_mad_keep_mask_time_window(t_in, v_in, half_window_sec, mad_k)
        return t_in[keep], v_in[keep]

    # Pass 1: standard local filter
    half_window_sec = float(param(params, "pass2_instant_bpm_outlier_window_sec"))
    mad_k = float(param(params, "pass2_instant_bpm_outlier_mad_k"))
    t_out, b_out = _apply_local_mad(bpm_times, instant_bpm, half_window_sec, mad_k)

    # Pass 2: wider window, gentler threshold
    if len(b_out) > 0:
        half_window_sec2 = 5.0 * half_window_sec
        mad_k2 = 2.0 * mad_k
        t_out, b_out = _apply_local_mad(t_out, b_out, half_window_sec2, mad_k2)

    return t_out, b_out


def smooth_bpm_series_from_instant(
    bpm_times: np.ndarray, instant_bpm: np.ndarray, params: Dict
) -> Tuple[pd.Series, np.ndarray, np.ndarray]:
    """
    Build smoothed BPM series from given (bpm_times, instant_bpm) using the same
    rolling window as calculate_bpm_series. Returns (smoothed_bpm, bpm_times, instant_bpm).
    """
    if bpm_times is None or instant_bpm is None or len(bpm_times) != len(instant_bpm) or len(bpm_times) == 0:
        return pd.Series(dtype=np.float64), np.array([]), np.array([])
    bpm_times = np.asarray(bpm_times, dtype=float)
    instant_bpm = np.asarray(instant_bpm, dtype=float)
    smoothing_window_sec = float(param(params, "output_smoothing_window_sec"))
    # Gaussian sigma chosen so that ±3σ spans roughly the same width as the old rolling window.
    sigma_sec = max(0.05, smoothing_window_sec / 3.0)
    smoothed_vals = _gaussian_kernel_smooth(bpm_times, bpm_times, instant_bpm, sigma_sec)
    start_time = datetime.datetime(1970, 1, 1)
    valid_peak_times_dt = [start_time + datetime.timedelta(seconds=float(t)) for t in bpm_times]
    smoothed_bpm = pd.Series(smoothed_vals, index=valid_peak_times_dt)
    return smoothed_bpm, bpm_times, instant_bpm


def calculate_bpm_series(peaks: np.ndarray, sample_rate: int, params: Dict) -> Tuple[pd.Series, np.ndarray, np.ndarray]:
    """Calculates and smooths the final BPM series from S1 peaks. Returns (smoothed_bpm, bpm_times, instant_bpm)."""
    if len(peaks) < 2:
        return pd.Series(dtype=np.float64), np.array([]), np.array([])
    peak_times = peaks / sample_rate
    time_diffs = np.diff(peak_times)
    valid_diffs = time_diffs > 1e-6
    if not np.any(valid_diffs):
        return pd.Series(dtype=np.float64), np.array([]), np.array([])

    instant_bpm = np.asarray(60.0 / time_diffs[valid_diffs], dtype=float)
    bpm_times = peak_times[1:][valid_diffs]
    avg_heart_rate = np.median(instant_bpm)
    if avg_heart_rate <= 0:
        return pd.Series(dtype=np.float64), bpm_times, instant_bpm

    smoothing_window_sec = float(param(params, "output_smoothing_window_sec"))
    sigma_sec = max(0.05, smoothing_window_sec / 3.0)
    smoothed_vals = _gaussian_kernel_smooth(bpm_times, bpm_times, instant_bpm, sigma_sec)
    start_time = datetime.datetime(1970, 1, 1)
    valid_peak_times_dt = [start_time + datetime.timedelta(seconds=float(t)) for t in bpm_times]
    smoothed_bpm = pd.Series(smoothed_vals, index=valid_peak_times_dt)
    return smoothed_bpm, bpm_times, instant_bpm


def calculate_bpm_series_from_s1_state_labels(
    state_labels: np.ndarray,
    sample_rate: int,
    params: Dict,
    state_s1_code: int = 0,
) -> Tuple[pd.Series, np.ndarray, np.ndarray]:
    """
    Same BPM series as calculate_bpm_series, but beat times come from the **start sample**
    of each contiguous S1 region in the dense state timeline (Pass 3: 0 = S1).
    RR interval = start(S1_i) → start(S1_{i+1}); instant BPM timestamps align with
    calculate_bpm_series (second beat of each pair).
    """
    if state_labels is None or len(state_labels) < 2:
        return pd.Series(dtype=np.float64), np.array([]), np.array([])
    sl = np.asarray(state_labels)
    is_s1 = sl == int(state_s1_code)
    if not np.any(is_s1):
        return pd.Series(dtype=np.float64), np.array([]), np.array([])
    # Start index of each contiguous S1 run
    starts = np.where(is_s1 & ~np.concatenate(([False], is_s1[:-1])))[0].astype(np.int64)
    if len(starts) < 2:
        return pd.Series(dtype=np.float64), np.array([]), np.array([])

    peak_times = starts / float(sample_rate)
    time_diffs = np.diff(peak_times)
    valid_diffs = time_diffs > 1e-6
    if not np.any(valid_diffs):
        return pd.Series(dtype=np.float64), np.array([]), np.array([])

    instant_bpm = np.asarray(60.0 / time_diffs[valid_diffs], dtype=float)
    bpm_times = peak_times[1:][valid_diffs]
    avg_heart_rate = np.median(instant_bpm)
    if avg_heart_rate <= 0:
        return pd.Series(dtype=np.float64), bpm_times, instant_bpm

    smoothing_window_sec = float(param(params, "output_smoothing_window_sec"))
    sigma_sec = max(0.05, smoothing_window_sec / 3.0)
    smoothed_vals = _gaussian_kernel_smooth(bpm_times, bpm_times, instant_bpm, sigma_sec)
    start_time = datetime.datetime(1970, 1, 1)
    valid_peak_times_dt = [start_time + datetime.timedelta(seconds=float(t)) for t in bpm_times]
    smoothed_bpm = pd.Series(smoothed_vals, index=valid_peak_times_dt)
    return smoothed_bpm, bpm_times, instant_bpm


def detect_bpm_failure(
    bpm_times: np.ndarray,
    instant_bpm: np.ndarray,
    total_duration_sec: float,
    params: Dict,
) -> Dict:
    """
    Algorithm-agnostic post-hoc plausibility gate on a raw (pre-smoothing) instant-BPM
    series. Flags likely tracking failure (lost lock, double-counted or missed beats)
    using only the beat-to-beat BPM sequence and its time coverage — no ground truth,
    no algorithm-specific internals — so the same check applies to native-pipeline and
    Springer output alike. Does not alter which algorithm ran; purely diagnostic.

    Returns {"failed": bool, "reasons": [str, ...], "metrics": {...}}.
    """
    bpm_times = np.asarray(bpm_times, dtype=np.float64) if bpm_times is not None else np.array([])
    instant_bpm = np.asarray(instant_bpm, dtype=np.float64) if instant_bpm is not None else np.array([])

    if len(instant_bpm) == 0 or total_duration_sec is None or total_duration_sec <= 0:
        return {"failed": False, "reasons": [], "metrics": {}}

    bpm_min = float(param(params, "bpm_min_physiological"))
    bpm_max = float(param(params, "bpm_max_physiological"))
    jump_ratio_threshold = float(param(params, "bpm_jump_ratio_threshold"))
    anomaly_frac_threshold = float(param(params, "bpm_anomaly_fraction_threshold"))
    max_gap_sec = float(param(params, "bpm_coverage_gap_sec"))
    trailing_frac_threshold = float(param(params, "bpm_trailing_coverage_frac"))
    min_beats_for_fractions = int(param(params, "bpm_min_beats_for_fraction_checks"))

    reasons: List[str] = []
    metrics: Dict = {}

    # With too few beats, a single anomaly dominates the fraction (e.g. 1 bad beat out of 2
    # is "100% jump") — rules 1 and 2 below need a minimum sample size to mean anything.
    enough_beats = len(instant_bpm) >= min_beats_for_fractions

    # 1. Physiological range violation.
    out_of_range = (instant_bpm < bpm_min) | (instant_bpm > bpm_max)
    range_violation_frac = float(np.mean(out_of_range))
    metrics["range_violation_frac"] = range_violation_frac
    if enough_beats and range_violation_frac > anomaly_frac_threshold:
        reasons.append(
            f"{range_violation_frac * 100:.0f}% of beats outside {bpm_min:.0f}-{bpm_max:.0f} BPM"
        )

    # 2. Beat-to-beat jump ratio: real HRV rarely swings this fast between adjacent beats;
    #    a high rate of jumps points to double-counted or missed beats.
    if len(instant_bpm) >= 2:
        raw_ratio = instant_bpm[1:] / np.maximum(instant_bpm[:-1], 1e-9)
        symmetric_ratio = np.maximum(raw_ratio, 1.0 / np.maximum(raw_ratio, 1e-9))
        jump_frac = float(np.mean(symmetric_ratio > jump_ratio_threshold))
        metrics["jump_fraction"] = jump_frac
        if enough_beats and jump_frac > anomaly_frac_threshold:
            reasons.append(
                f"{jump_frac * 100:.0f}% of beat-to-beat intervals jump by "
                f">{(jump_ratio_threshold - 1) * 100:.0f}%"
            )
    else:
        metrics["jump_fraction"] = 0.0

    # 3. Coverage gap: a long silent stretch mid-recording means the tracker lost lock.
    if len(bpm_times) >= 2:
        gaps = np.diff(bpm_times)
        gap_idx = int(np.argmax(gaps))
        max_gap = float(gaps[gap_idx])
        metrics["max_gap_sec"] = max_gap
        if max_gap > max_gap_sec:
            reasons.append(
                f"{max_gap:.1f}s silent gap with no detected beat (at {bpm_times[gap_idx]:.1f}s)"
            )
    else:
        metrics["max_gap_sec"] = 0.0

    # 4. Trailing coverage: detections dying out early gets padded flat by downstream
    #    rasterization, which reads as a real plateau unless caught here. Gated on the
    #    *absolute* trailing gap too, so a missed beat at the tail of a 2s clip (small
    #    absolute gap, but a big fraction of a short recording) doesn't trip this.
    last_beat_time = float(bpm_times[-1]) if len(bpm_times) else 0.0
    coverage_frac = last_beat_time / total_duration_sec
    trailing_gap_sec = total_duration_sec - last_beat_time
    metrics["trailing_coverage_frac"] = coverage_frac
    if coverage_frac < trailing_frac_threshold and trailing_gap_sec > max_gap_sec:
        reasons.append(
            f"beat detection stops at {last_beat_time:.1f}s, only {coverage_frac * 100:.0f}% of the "
            f"{total_duration_sec:.1f}s recording"
        )

    return {"failed": bool(reasons), "reasons": reasons, "metrics": metrics}


def _find_major_hr_trends(
    smoothed_bpm_series: pd.Series,
    min_duration_sec: int,
    min_bpm_change: int,
    rising: bool,
) -> List[Dict]:
    """
    Shared algorithm for finding sustained HR inclines (rising=True) or declines (rising=False).

    For inclines, iterates from each trough to its first following peak; for declines, from
    each peak to its first following trough. Only trends that meet both the minimum duration
    and the minimum BPM change threshold are returned. Results are sorted by steepness.
    """
    if smoothed_bpm_series.empty or len(smoothed_bpm_series) < 2:
        return []

    direction = "inclines" if rising else "declines"
    change_label = "increase" if rising else "decrease"
    logging.info(
        f"Searching for major HR {direction} (min_duration={min_duration_sec}s, "
        f"min_{change_label}={min_bpm_change} BPM)..."
    )

    time_diffs_sec = smoothed_bpm_series.index.to_series().diff().dt.total_seconds()
    mean_time_diff = np.nanmean(time_diffs_sec)
    distance_samples = (
        5 if np.isnan(mean_time_diff) or mean_time_diff == 0
        else int((min_duration_sec / 2) / mean_time_diff)
    )

    peaks, _ = find_peaks(smoothed_bpm_series.values, prominence=5, distance=distance_samples)
    troughs, _ = find_peaks(-smoothed_bpm_series.values, prominence=5, distance=distance_samples)
    if len(troughs) == 0 or len(peaks) == 0:
        return []

    # For inclines: start=trough, end=next peak; for declines: start=peak, end=next trough.
    starts, ends = (troughs, peaks) if rising else (peaks, troughs)
    logging.info(
        f"Found {len(starts)} potential start points and {len(ends)} potential end points "
        f"for {direction}."
    )

    results = []
    for start_idx in starts:
        following = ends[ends > start_idx]
        if len(following) == 0:
            continue
        end_idx = following[0]
        start_time = smoothed_bpm_series.index[start_idx]
        end_time = smoothed_bpm_series.index[end_idx]
        start_bpm = smoothed_bpm_series.values[start_idx]
        end_bpm = smoothed_bpm_series.values[end_idx]
        duration = (end_time - start_time).total_seconds()
        bpm_change = (end_bpm - start_bpm) if rising else (start_bpm - end_bpm)

        if duration >= min_duration_sec and bpm_change >= min_bpm_change:
            slope = (end_bpm - start_bpm) / duration  # positive for inclines, negative for declines
            entry = {
                'start_time': start_time, 'end_time': end_time,
                'start_bpm': start_bpm, 'end_bpm': end_bpm,
                'duration_sec': duration, 'slope_bpm_per_sec': slope,
            }
            entry['bpm_increase' if rising else 'bpm_decrease'] = bpm_change
            results.append(entry)

    # Steepest first: descending slope for inclines, ascending (most negative) for declines.
    results.sort(key=lambda x: x['slope_bpm_per_sec'], reverse=rising)
    return results


def find_major_hr_inclines(smoothed_bpm_series: pd.Series, min_duration_sec: int = 10, min_bpm_increase: int = 15) -> List[Dict]:
    """Identifies significant, sustained periods of heart rate increase."""
    return _find_major_hr_trends(smoothed_bpm_series, min_duration_sec, min_bpm_increase, rising=True)


def find_major_hr_declines(smoothed_bpm_series: pd.Series, min_duration_sec: int = 10, min_bpm_decrease: int = 15) -> List[Dict]:
    """Identifies significant, sustained periods of heart rate decrease (recovery)."""
    return _find_major_hr_trends(smoothed_bpm_series, min_duration_sec, min_bpm_decrease, rising=False)


def _find_steepest_slope(series: pd.Series, window_sec: int, rising: bool) -> Optional[Dict]:
    """Sliding-window search for the steepest sustained slope within *series*.

    Args:
        series:     BPM time-series to search (index must be datetime-like).
        window_sec: Minimum window width in seconds for each slope measurement.
        rising:     True → find the steepest positive slope (exertion).
                    False → find the steepest negative slope (recovery).
    """
    if series.empty or len(series) < 2:
        return None
    times_sec = (series.index - series.index[0]).total_seconds()
    if times_sec[-1] < window_sec:
        return None

    bpm_values = series.values
    steepest_slope, best_period = 0, None
    for i in range(len(times_sec) - 1):
        target_t = times_sec[i] + window_sec
        end_idx = int(np.searchsorted(times_sec, target_t, side="left"))
        if end_idx >= len(times_sec):
            break
        duration = times_sec[end_idx] - times_sec[i]
        if duration > 0:
            slope = (bpm_values[end_idx] - bpm_values[i]) / duration
            if (rising and slope > steepest_slope) or (not rising and slope < steepest_slope):
                steepest_slope = slope
                best_period = {
                    'start_time': series.index[i], 'end_time': series.index[end_idx],
                    'start_bpm': bpm_values[i], 'end_bpm': bpm_values[end_idx],
                    'slope_bpm_per_sec': slope, 'duration_sec': duration,
                }
    return best_period


def find_peak_recovery_rate(smoothed_bpm_series: pd.Series, window_sec: int = 20) -> Optional[Dict]:
    """Finds the steepest slope of heart rate decline after the peak BPM."""
    if smoothed_bpm_series.empty or len(smoothed_bpm_series) < 2:
        return None
    recovery_series = smoothed_bpm_series[smoothed_bpm_series.idxmax():]
    if recovery_series.empty:
        return None
    return _find_steepest_slope(recovery_series, window_sec, rising=False)


def find_peak_exertion_rate(smoothed_bpm_series: pd.Series, window_sec: int = 20) -> Optional[Dict]:
    """Finds the steepest slope of heart rate increase across the entire recording."""
    if smoothed_bpm_series.empty or len(smoothed_bpm_series) < 2:
        return None
    return _find_steepest_slope(smoothed_bpm_series, window_sec, rising=True)


def calculate_hrr(smoothed_bpm_series: pd.Series, interval_sec: int = 60) -> Optional[Dict]:
    """Calculates the standard Heart Rate Recovery (HRR) over a fixed interval."""
    if smoothed_bpm_series.empty or len(smoothed_bpm_series) < 2:
        return None
    peak_bpm, peak_time = smoothed_bpm_series.max(), smoothed_bpm_series.idxmax()
    recovery_check_time = peak_time + pd.Timedelta(seconds=interval_sec)
    if recovery_check_time > smoothed_bpm_series.index.max():
        return None

    recovery_bpm = np.interp(
        recovery_check_time.timestamp(),
        (smoothed_bpm_series.index.astype(np.int64) // 10**9).to_numpy(dtype=float),
        np.asarray(smoothed_bpm_series.values, dtype=float))
    return {'peak_bpm': peak_bpm, 'peak_time': peak_time, 'recovery_bpm': recovery_bpm,
            'recovery_check_time': recovery_check_time, 'hrr_value_bpm': peak_bpm - recovery_bpm,
            'interval_sec': interval_sec}


def find_recovery_phase(bpm_values: np.ndarray, bpm_times_sec: np.ndarray, params: Dict) -> Tuple[Optional[float], Optional[float]]:
    """Analyzes a pass 1 BPM series to find the peak heart rate and define the subsequent recovery phase window.
    Returns (None, None) if BPM stays low (no exertion/recovery), so recovery-phase adjust is not applied."""
    if bpm_times_sec is None or len(bpm_times_sec) < 2:
        logging.warning("Not enough pass 1 beats to determine a recovery phase.")
        return None, None
    bpm_values = np.asarray(bpm_values, dtype=np.float64)
    bpm_times_sec = np.asarray(bpm_times_sec, dtype=np.float64)
    if len(bpm_values) < 2 or len(bpm_values) != len(bpm_times_sec):
        logging.warning("Pass 1 BPM curve has inconsistent lengths; cannot determine recovery phase.")
        return None, None
    peak_idx = np.argmax(bpm_values)
    peak_bpm = float(bpm_values[peak_idx])
    min_peak_bpm = param(params, "recovery_phase_min_peak_bpm")
    if peak_bpm < min_peak_bpm:
        logging.info(
            f"Recovery phase not used: peak BPM in pass 1 is {peak_bpm:.1f} (below {min_peak_bpm:.0f}). "
            "BPM remains low throughout -- no exertion/recovery assumed."
        )
        return None, None
    peak_time_sec = float(bpm_times_sec[peak_idx])
    recovery_end_time_sec = peak_time_sec + param(params, "recovery_phase_duration_sec")
    logging.info("Peak BPM detected in pass 1 at %.2fs (%.1f BPM). High-contractility state defined until %.2fs.", peak_time_sec, peak_bpm, recovery_end_time_sec)
    return peak_time_sec, recovery_end_time_sec
