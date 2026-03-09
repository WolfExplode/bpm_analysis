import datetime
import logging
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, lombscargle
from typing import List, Dict, Tuple, Optional


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
    freqs = np.linspace(0.001, 0.5, 1000)
    angular_freqs = 2.0 * np.pi * freqs
    try:
        periodogram = lombscargle(times_sec, rr_ms, angular_freqs, normalize=True)
    except Exception as e:
        logging.warning(f"Lomb-Scargle: lombscargle() failed: {e}")
        return None
    # Task Force bands: VLF 0.003-0.04, LF 0.04-0.15, HF 0.15-0.40 Hz
    # With normalize=True the periodogram is dimensionless; scale by RR variance to get power in ms² (Task Force convention).
    vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
    raw_vlf = float(np.trapz(periodogram[vlf_mask], freqs[vlf_mask])) if np.any(vlf_mask) else 0.0
    raw_lf = float(np.trapz(periodogram[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else 0.0
    raw_hf = float(np.trapz(periodogram[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else 0.0
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
    enable_freq = params.get("enable_hrv_frequency_domain", False)

    # First, calculate all R-R intervals from the S1 peaks
    if len(s1_peaks) < window_size_beats:
        logging.warning(f"Not enough beats ({len(s1_peaks)}) to perform windowed HRV analysis with a window of {window_size_beats} beats.")
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

    logging.info(f"Beat-based windowed HRV analysis complete. Generated {len(results)} data points.")
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
    min_duration = params.get("hrv_global_min_duration_sec", 300.0)
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


def _loess(
    t_evals: np.ndarray,
    t_data: np.ndarray,
    y_data: np.ndarray,
    frac: float = 0.2,
    degree: int = 1,
) -> np.ndarray:
    """LOESS: weighted local polynomial fit. For each t in t_evals, fit polynomial to nearby (t_data, y_data) with tricube weights; return fitted values."""
    t_evals = np.asarray(t_evals, dtype=float)
    t_data = np.asarray(t_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    n = len(t_data)
    k = max(degree + 1, min(n, int(np.ceil(frac * n))))
    y_out = np.zeros(len(t_evals), dtype=float)
    for i, t in enumerate(t_evals):
        dist = np.abs(t_data - t)
        idx = np.argsort(dist)[:k]
        d_max = float(dist[idx[-1]])
        if d_max < 1e-9:
            y_out[i] = float(np.mean(y_data[idx]))
        else:
            w = (1 - (dist[idx] / d_max) ** 3) ** 3
            p = np.polyfit(t_data[idx], y_data[idx], degree, w=w)
            y_out[i] = float(np.polyval(p, t))
    return y_out


def compute_pass1_bpm_curve(
    anchor_beats: np.ndarray, sample_rate: int, params: Dict
) -> Optional[Dict[str, np.ndarray]]:
    """
    Canonical pass 1 BPM curve: instant BPM from anchor beats, outlier removal (median+MAD), then LOESS.
    Used for the time-varying prior, recovery phase, and all plots so display matches algorithm input.
    Returns dict with curve_times, curve_bpm (dense LOESS), scatter_times, scatter_bpm (filtered instant), or None if insufficient data.
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

    # Outlier removal: keep point if within median ± k*MAD in local window
    half_window_sec = float(params.get("pass1_bpm_outlier_window_sec", 10.0))
    mad_k = float(params.get("pass1_bpm_outlier_mad_k", 2.5))
    keep = np.ones(len(instant_bpm), dtype=bool)
    for i in range(len(instant_bpm)):
        in_window = np.abs(times_sec - times_sec[i]) <= half_window_sec
        if not np.any(in_window):
            continue
        local_median = np.median(instant_bpm[in_window])
        local_mad = np.median(np.abs(instant_bpm[in_window] - local_median))
        if local_mad > 1e-9:
            keep[i] = np.abs(instant_bpm[i] - local_median) <= mad_k * local_mad
    scatter_bpm = np.asarray(instant_bpm[keep], dtype=float)
    scatter_times = np.asarray(times_sec[keep], dtype=float)

    if len(scatter_times) < 3:
        return None

    loess_frac = float(params.get("pass1_bpm_loess_frac", 0.2))
    curve_times = np.linspace(float(scatter_times.min()), float(scatter_times.max()), 200)
    curve_bpm = _loess(curve_times, scatter_times, scatter_bpm, frac=loess_frac)

    return {
        "curve_times": curve_times,
        "curve_bpm": curve_bpm,
        "scatter_times": scatter_times,
        "scatter_bpm": scatter_bpm,
    }


def filter_instant_bpm_mad(
    bpm_times: np.ndarray, instant_bpm: np.ndarray, params: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply MAD-based outlier removal to instantaneous BPM (e.g. for pass 2).
    Keeps points where |BPM - local_median| <= k * local_MAD in a time window.
    Returns (filtered_bpm_times, filtered_instant_bpm).
    """
    if bpm_times is None or instant_bpm is None or len(bpm_times) != len(instant_bpm) or len(bpm_times) < 2:
        return np.array([]), np.array([])
    bpm_times = np.asarray(bpm_times, dtype=float)
    instant_bpm = np.asarray(instant_bpm, dtype=float)
    half_window_sec = float(params.get("pass2_instant_bpm_outlier_window_sec", 10.0))
    mad_k = float(params.get("pass2_instant_bpm_outlier_mad_k", 2.5))
    keep = np.ones(len(instant_bpm), dtype=bool)
    for i in range(len(instant_bpm)):
        in_window = np.abs(bpm_times - bpm_times[i]) <= half_window_sec
        if not np.any(in_window):
            continue
        local_median = np.median(instant_bpm[in_window])
        local_mad = np.median(np.abs(instant_bpm[in_window] - local_median))
        if local_mad > 1e-9:
            keep[i] = np.abs(instant_bpm[i] - local_median) <= mad_k * local_mad
    return bpm_times[keep], instant_bpm[keep]


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
    start_time = datetime.datetime.fromtimestamp(0)
    valid_peak_times_dt = [start_time + datetime.timedelta(seconds=float(t)) for t in bpm_times]
    bpm_series = pd.Series(instant_bpm, index=valid_peak_times_dt)
    smoothing_window_sec = params["output_smoothing_window_sec"]
    smoothing_window_str = f"{smoothing_window_sec}s"
    smoothed_bpm = bpm_series.rolling(window=smoothing_window_str, min_periods=1, center=True).mean()
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
    start_time = datetime.datetime.fromtimestamp(0)
    valid_peak_times_dt = [start_time + datetime.timedelta(seconds=t) for t in bpm_times]
    bpm_series = pd.Series(instant_bpm, index=valid_peak_times_dt)
    avg_heart_rate = np.median(instant_bpm)
    if avg_heart_rate > 0:
        smoothing_window_sec = params['output_smoothing_window_sec']
        smoothing_window_str = f"{smoothing_window_sec}s"
        smoothed_bpm = bpm_series.rolling(window=smoothing_window_str, min_periods=1, center=True).mean()
    else:
        smoothed_bpm = pd.Series(dtype=np.float64)

    return smoothed_bpm, bpm_times, instant_bpm


def detect_trapezoid_discontinuities(smoothed_bpm: pd.Series, bpm_times_sec: np.ndarray, params: Dict) -> List[Dict]:
    """
    Detects trapezoid-shaped discontinuities in the average BPM series that are
    characteristic of a brief extra-beat artifact:
      - A very fast rise
      - A sustained (possibly slightly sloped) plateau
      - A very fast fall that returns to baseline

    Detection only -- results are not yet used to correct labels. See
    Documentation.md "Trapezoid Artifacts" for design rationale and future plans.
    """
    if (
        smoothed_bpm is None
        or smoothed_bpm.empty
        or bpm_times_sec is None
        or len(bpm_times_sec) != len(smoothed_bpm)
    ):
        return []

    # Build working DataFrame equivalent to the CSV used in detectTrapezoid.py
    df = pd.DataFrame(
        {
            "Time (s)": bpm_times_sec.astype(float),
            "Average BPM": smoothed_bpm.to_numpy(dtype=float),
        }
    ).dropna(subset=["Time (s)", "Average BPM"])

    if len(df) < 4:
        return []

    # Calculate differences and instantaneous rate of BPM change
    df["Δt"] = df["Time (s)"].diff()
    df["ΔBPM"] = df["Average BPM"].diff()
    df["Rate"] = df["ΔBPM"] / df["Δt"]

    # --- CONFIGURATION (now driven by params, defaults mirror detectTrapezoid.py) ---
    RATE_THRESHOLD = params.get("trapezoid_rate_threshold", 7.0)                 # BPM/s
    MAX_EDGE_DURATION = params.get("trapezoid_max_edge_duration_sec", 1.5)       # seconds
    MIN_PLATEAU_DURATION = params.get("trapezoid_min_plateau_duration_sec", 1.5) # seconds
    MAX_PLATEAU_DURATION = params.get("trapezoid_max_plateau_duration_sec", 15.0)# seconds
    BASELINE_TOLERANCE = params.get("trapezoid_baseline_tolerance_bpm", 5.0)     # BPM
    MIN_JUMP = params.get("trapezoid_min_jump_bpm", 6.0)                         # BPM
    MIN_FALL_DELTA = params.get("trapezoid_min_fall_delta_bpm", 5.0)             # BPM

    # Step 1: Identify edge intervals (second point of each edge)
    df["is_rise"] = (df["Rate"] > RATE_THRESHOLD) & (df["Δt"] < MAX_EDGE_DURATION)
    df["is_fall"] = (df["Rate"] < -RATE_THRESHOLD) & (df["Δt"] < MAX_EDGE_DURATION)

    rise_indices = df[df["is_rise"]].index.tolist()
    fall_indices = df[df["is_fall"]].index.tolist()

    trapezoids: List[Dict] = []

    # Step 2: Match edges into trapezoids
    for rise_idx in rise_indices:
        # Need at least one sample before the rise edge for t1 / baseline
        if rise_idx <= 0:
            continue

        rise_time = float(df.loc[rise_idx, "Time (s)"])

        for fall_idx in list(fall_indices):
            # Need at least one sample before fall edge for t3
            if fall_idx <= 0:
                continue

            fall_time = float(df.loc[fall_idx, "Time (s)"])

            # Timing constraints: plateau must be long enough but not absurdly long
            plateau_duration = fall_time - rise_time
            if not (MIN_PLATEAU_DURATION <= plateau_duration <= MAX_PLATEAU_DURATION):
                continue

            # --- Validate plateau (region strictly between rise and fall) ---
            plateau_mask = (df["Time (s)"] > rise_time) & (df["Time (s)"] < fall_time)
            plateau_df = df[plateau_mask]
            if plateau_df.empty:
                continue

            # Allow sloped plateaus: median absolute rate should be modest
            if plateau_df["Rate"].abs().median() > RATE_THRESHOLD / 3.0:
                continue

            # --- Validate baseline recovery ---
            # Baseline before: up to 3 points before the rise edge
            before_start_idx = max(0, rise_idx - 3)
            before_end_idx = rise_idx - 1
            if before_end_idx < before_start_idx:
                continue
            baseline_before = float(
                df.loc[before_start_idx:before_end_idx, "Average BPM"].mean()
            )

            # Baseline after: up to 3 points starting at fall edge
            after_end_idx = min(fall_idx + 2, df.index[-1])
            baseline_after = float(
                df.loc[fall_idx:after_end_idx, "Average BPM"].mean()
            )

            if abs(baseline_after - baseline_before) > BASELINE_TOLERANCE:
                continue

            # --- Calculate the four key timestamps ---
            t1 = float(df.loc[rise_idx - 1, "Time (s)"])
            t2 = rise_time
            t3 = float(df.loc[fall_idx - 1, "Time (s)"])
            t4 = fall_time

            # Validate edge intervals are brief
            if (t2 - t1) > MAX_EDGE_DURATION or (t4 - t3) > MAX_EDGE_DURATION:
                continue

            # Enforce a minimum BPM change across the fall edge itself.
            # If the fall barely changes BPM, don't treat it as a trapezoid artifact.
            fall_start_bpm = float(df.loc[fall_idx - 1, "Average BPM"])
            fall_end_bpm = float(df.loc[fall_idx, "Average BPM"])
            if abs(fall_start_bpm - fall_end_bpm) < MIN_FALL_DELTA:
                continue

            # Calculate jump from baseline to plateau median
            plateau_median = float(plateau_df["Average BPM"].median())
            jump_size = plateau_median - baseline_before
            if jump_size < MIN_JUMP:
                continue

            plateau_slope = float(
                plateau_df["Average BPM"].iloc[-1] - plateau_df["Average BPM"].iloc[0]
            )

            # Store both timestamps and BPM values for debugging/plotting
            trap = {
                "t_start_rise": t1,
                "t_end_rise": t2,
                "t_start_fall": t3,
                "t_end_fall": t4,
                "bpm_start_rise": float(df.loc[rise_idx - 1, "Average BPM"]),
                "bpm_end_rise": float(df.loc[rise_idx, "Average BPM"]),
                "bpm_start_fall": fall_start_bpm,
                "bpm_end_fall": fall_end_bpm,
                "baseline_before": baseline_before,
                "plateau_median": plateau_median,
                "plateau_slope": plateau_slope,
                "jump_size": jump_size,
                "baseline_after": baseline_after,
                "baseline_diff": baseline_after - baseline_before,
                "plateau_duration": plateau_duration,
                "plateau_points": int(len(plateau_df)),
            }
            trapezoids.append(trap)

            # Remove used fall index so it cannot be reused by another rise
            fall_indices.remove(fall_idx)
            break

    if trapezoids:
        logging.info(f"Detected {len(trapezoids)} trapezoid HR artifacts (sudden plateau jumps):")
        for i, trap in enumerate(trapezoids, 1):
            logging.info(
                "  Trapezoid #%d: "
                "Rise %.3fs (%.1f BPM) → %.3fs (%.1f BPM); "
                "Plateau %.3fs → %.3fs (Δ%.3fs, %d pts); "
                "Fall %.3fs (%.1f BPM) → %.3fs (%.1f BPM)",
                i,
                trap["t_start_rise"],
                trap["bpm_start_rise"],
                trap["t_end_rise"],
                trap["bpm_end_rise"],
                trap["t_end_rise"],
                trap["t_start_fall"],
                trap["plateau_duration"],
                trap["plateau_points"],
                trap["t_start_fall"],
                trap["bpm_start_fall"],
                trap["t_end_fall"],
                trap["bpm_end_fall"],
            )
    else:
        logging.info("No trapezoid-like HR artifacts detected in average BPM series.")

    return trapezoids


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
        end_idx_candidates = np.where(times_sec >= times_sec[i] + window_sec)[0]
        if len(end_idx_candidates) == 0:
            break
        end_idx = end_idx_candidates[0]
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


def find_recovery_phase(bpm_series: pd.Series, bpm_times_sec: np.ndarray, params: Dict) -> Tuple[Optional[float], Optional[float]]:
    """Analyzes a pass 1 BPM series to find the peak heart rate and define the subsequent recovery phase window.
    Returns (None, None) if BPM stays low (no exertion/recovery), so recovery-phase adjust is not applied."""
    if bpm_times_sec is None or len(bpm_times_sec) < 2:
        logging.warning("Not enough pass 1 beats to determine a recovery phase.")
        return None, None
    bpm_values = bpm_series.to_numpy()
    peak_idx = np.argmax(bpm_values)
    peak_bpm = float(bpm_values[peak_idx])
    min_peak_bpm = params.get("recovery_phase_min_peak_bpm", 95.0)
    if peak_bpm < min_peak_bpm:
        logging.info(
            f"Recovery phase not used: peak BPM in pass 1 is {peak_bpm:.1f} (below {min_peak_bpm:.0f}). "
            "BPM remains low throughout -- no exertion/recovery assumed."
        )
        return None, None
    peak_time_sec = float(bpm_times_sec[peak_idx])
    recovery_end_time_sec = peak_time_sec + params.get("recovery_phase_duration_sec", 120.0)
    logging.info(f"Peak BPM detected in pass 1 at {peak_time_sec:.2f}s ({peak_bpm:.1f} BPM). High-contractility state defined until {recovery_end_time_sec:.2f}s.")
    return peak_time_sec, recovery_end_time_sec
