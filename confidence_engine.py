import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Callable
from peak_utils import (
    PeakType,
    get_peak_prominence_details,
    calculate_peak_prominence,
    _RHYTHM_DEVIATION_XPOINTS,
    _RHYTHM_SCORE_YPOINTS,
    _AMPLITUDE_RATIO_XPOINTS,
    _AMPLITUDE_SCORE_YPOINTS,
)


@dataclass
class AnalysisState:
    """
    Holds all mutable state used by the peak classification loop.

    Attributes:
        dynamic_noise_floor: Rolling estimate of the background noise floor at each sample,
            derived from sanitized audio troughs. Used to threshold peaks and compute
            peak "strength" relative to the local noise environment.
        trough_indices: Indices of sanitized troughs in the audio envelope. These anchor
            prominence calculations by defining the key cols around each peak.
        all_peaks: Indices of all raw peaks above the dynamic height threshold before
            any S1/S2/Noise classification or correction passes.
        smoothed_dev_series: Time-indexed series of normalized peak-to-peak amplitude
            deviations, smoothed over time. This captures rhythm stability and is used
            as context when reasoning about sudden changes in the waveform.
        long_term_bpm: BPM value used when computing expected systole (S1→S2) and R-R intervals.
            Set each iteration from the pass 1 BPM prior (time-varying) when available;
            otherwise the initial start_bpm and not updated.
        analysis_data: Bag of analysis artifacts that downstream plotting/reporting
            relies on (e.g., `dynamic_noise_floor_series`, `trough_indices`,
            `deviation_series`, `peak_classifications`, `long_term_bpm_series`).
        candidate_beats: Sample indices of peaks that have been accepted as S1 heartbeats
            (either paired S1 or validated Lone S1) during the main loop.
        peak_classifications: Mapping from raw peak index to a structured record
            (peak_type, sections, label_scores) for how that peak was classified (S1/S2/Lone S1/Noise).
            label_scores: heuristic dict mapping 'S1', 'S2', 'noise' to floats in [0,1] from the greedy pass;
            not calibrated probabilities. Used by pass 3+, FFT ranking, debug, and plot tooltips.
        long_term_bpm_history: Sequence of `(time_sec, bpm)` tuples for plotting (BPM trend).
        pass1_bpm_prior: Optional callable time_sec -> bpm from the pass 1 curve.
        sorted_troughs: Sorted list of trough indices mirroring `trough_indices`,
            kept in list form for fast neighbor lookups and iteration.
        loop_idx: Current index into `all_peaks` for the main classification loop. This
            is the loop counter that drives progression through raw peaks.
        s1_s2_interval_history: Rolling window of the last N accepted systole (S1→S2) intervals
            (in seconds). Used by `PairingEngine` to build an empirical expected
            systole (S1→S2) interval once enough paired beats have been observed.
        s1_s2_contractility_history: Rolling window of the last N accepted S1/S2
            prominence ratios. Used by `PairingEngine` to build an empirical
            contractility expectation once enough paired beats have been observed.
        recent_s1_outcomes: (time_sec, was_paired) for each S1 we decided on; used to
            compute pair rate in the last N seconds for blending history vs BPM expected.
        noise_floor_values: Same noise floor as dynamic_noise_floor, as a contiguous float64
            array indexed by sample (avoids pandas iloc in hot loops).
        peak_prominence_detail_cache: Optional precomputed prominence details for raw peak indices.
    """

    dynamic_noise_floor: pd.Series
    noise_floor_values: np.ndarray
    trough_indices: np.ndarray
    all_peaks: np.ndarray
    smoothed_dev_series: pd.Series
    long_term_bpm: float
    analysis_data: Dict[str, Any] = field(default_factory=dict)
    candidate_beats: List[int] = field(default_factory=list)
    peak_classifications: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    long_term_bpm_history: List[Tuple[float, float]] = field(default_factory=list)
    pass1_bpm_prior: Optional[Callable[[float], float]] = None
    sorted_troughs: List[int] = field(default_factory=list)
    loop_idx: int = 0
    s1_s2_interval_history: List[float] = field(default_factory=list)  # Last N accepted systole (S1→S2) intervals (sec)
    s1_s2_contractility_history: List[float] = field(default_factory=list)  # Last N accepted S1/S2 prominence ratios for expected contractility
    recent_s1_outcomes: List[Tuple[float, bool]] = field(default_factory=list)  # (time_sec, was_paired) for pair-rate window
    peak_prominence_detail_cache: Optional[Dict[int, Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# Interval / timing helpers
# ---------------------------------------------------------------------------

def calculate_bpm_intervals(bpm: float, params: Dict) -> Dict[str, float]:
    """
    Given a BPM value, computes key timing intervals (in seconds) implied by that rate.

    Returns a dictionary with:
      - 'rr_interval'         : full S1→S1 (R-R) interval
      - 's1_s2_min'           : minimum plausible S1→S2 (systole) interval
      - 's1_s2_nominal'       : expected S1→S2 (Weissler: ET = ref_et - slope*(BPM - ref_bpm))
      - 's1_s2_max'           : maximum plausible S1→S2 interval (capped)
      - 's2_s1_nominal'       : nominal diastole (S2→next S1 = RR minus systole nominal)
      - 'diastole_min'        : minimum plausible diastole duration
      - 'diastole_max'        : maximum plausible diastole duration
      - 's1_min'              : minimum S1 acoustic event duration (BPM-independent)
      - 's1_nominal'          : nominal S1 acoustic event duration
      - 's1_max'              : maximum S1 acoustic event duration
      - 's2_min'              : minimum S2 acoustic event duration (BPM-independent)
      - 's2_nominal'          : nominal S2 acoustic event duration
      - 's2_max'              : maximum S2 acoustic event duration
      - 'min_feasible_cycle'  : smallest time that can hold one valid S1+systole+S2+diastole
    """
    # Guard against zero or negative BPM
    bpm = float(max(bpm, 1e-6))
    rr_interval = 60.0 / bpm

    min_frac = params.get("min_s1_s2_interval_rr_fraction", 0.35)
    min_abs = params.get("min_s1_s2_interval_sec", 0.15)
    cap_abs = params.get("s1_s2_interval_cap_sec", 0.4)

    s1_s2_min = max(rr_interval * min_frac, min_abs)

    # Weissler-style: ejection time shortens linearly with rising BPM
    ref_et_ms = params.get("s1_s2_expected_weissler_ref_et_ms", 300)
    ref_bpm = params.get("s1_s2_expected_weissler_ref_bpm", 60)
    slope = params.get("s1_s2_expected_weissler_slope_ms_per_bpm", 1.0)
    expected_et_ms = ref_et_ms - slope * (bpm - ref_bpm)
    s1_s2_nominal = np.clip(expected_et_ms / 1000.0, min_abs, cap_abs)

    s1_s2_max = min(cap_abs, max(s1_s2_nominal, s1_s2_min))
    s2_s1_nominal = max(0.0, rr_interval - s1_s2_nominal)

    # Diastole plausibility bounds (S2→next S1).
    # min: diastole can compress at high BPM but needs some filling time.
    # max: only relevant for dropout/gap detection (large multiples of nominal).
    diastole_min_frac = float(params.get("min_diastole_nominal_frac", 0.35))
    diastole_max_frac = float(params.get("max_diastole_nominal_frac", 2.0))
    diastole_min_abs = float(params.get("min_diastole_sec", 0.08))
    diastole_min = max(diastole_min_abs, s2_s1_nominal * diastole_min_frac)
    diastole_max = s2_s1_nominal * diastole_max_frac

    # S1 and S2 acoustic event duration bounds — BPM-independent physiological constants.
    # These are how long the heart sound impulse itself lasts, not the inter-sound intervals.
    s1_min = float(params.get("s1_min_sec", 0.010))
    s1_nominal = float(params.get("s1_nominal_sec", 0.040))
    s1_max = float(params.get("s1_max_sec", 0.080))
    s2_min = float(params.get("s2_min_sec", 0.010))
    s2_nominal = float(params.get("s2_nominal_sec", 0.030))
    s2_max = float(params.get("s2_max_sec", 0.060))

    # Smallest span that can physically hold one complete cardiac cycle at minimum compression.
    min_feasible_cycle = s1_min + s1_s2_min + s2_min + diastole_min

    return {
        "rr_interval": rr_interval,
        "s1_s2_min": s1_s2_min,
        "s1_s2_nominal": s1_s2_nominal,
        "s1_s2_max": s1_s2_max,
        "s2_s1_nominal": s2_s1_nominal,
        "diastole_min": diastole_min,
        "diastole_max": diastole_max,
        "s1_min": s1_min,
        "s1_nominal": s1_nominal,
        "s1_max": s1_max,
        "s2_min": s2_min,
        "s2_nominal": s2_nominal,
        "s2_max": s2_max,
        "min_feasible_cycle": min_feasible_cycle,
    }


# Exponent for reactivity power law; higher = sharper slowdown near HR_max. Kept as variable for future tuning.
REACTIVITY_POWER_EXPONENT = 2.0


def hr_reactivity_factor(hr: float, hr_max: float, hr_rest: float, C: float = REACTIVITY_POWER_EXPONENT) -> float:
    """
    Power-law reactivity factor: as HR approaches HR_max, allowed rate of increase goes to zero.

    reactivity_factor(HR) = ((HR_max - HR) / (HR_max - HR_rest))^C

    When HR is low (near rest), factor is 1 (full reactivity). When HR is near max, factor -> 0
    (heart cannot increase much more). C controls sharpness; default 2.
    """
    reserve = hr_max - hr_rest
    if reserve <= 0:
        return 1.0
    ratio = (hr_max - hr) / reserve
    ratio = np.clip(ratio, 0.0, 1.0)
    return float(ratio ** C)


def update_long_term_bpm(new_rr_sec: float, current_long_term_bpm: float, params: Dict) -> float:
    """Updates the long-term BPM belief from a new R-R interval. Only used when no pass 1 BPM prior (e.g. pass 1 run)."""
    instant_bpm = 60.0 / new_rr_sec
    lr = params.get("bpm_belief_learning_rate", 0.05)
    max_change_per_beat = params.get("bpm_belief_max_change_per_beat", 3.0)
    target_bpm = ((1 - lr) * current_long_term_bpm) + (lr * instant_bpm)
    max_change = max_change_per_beat * new_rr_sec
    proposed_change = target_bpm - current_long_term_bpm
    limited_change = np.clip(proposed_change, -max_change, max_change)
    new_bpm = current_long_term_bpm + limited_change
    return max(params['min_bpm'], min(new_bpm, params['max_bpm']))


# ---------------------------------------------------------------------------
# Contractility / confidence helpers
# ---------------------------------------------------------------------------

def _contractility_expected_ratio_bpm(bpm: float, params: Dict) -> float:
    """Expected S1/S2 ratio from BPM using a power curve (non-linear; steep rise at low BPM then flatter)."""
    bpm_min = params.get("contractility_bpm_min", 60)
    bpm_max = params.get("contractility_bpm_max", 200)
    low_ratio = params.get("contractility_low_ratio", 0.9)
    high_ratio = params.get("contractility_high_ratio", 6.0)
    exponent = params.get("contractility_power_exponent", 0.7)
    bpm_clipped = np.clip(bpm, bpm_min, bpm_max)
    t = (bpm_clipped - bpm_min) / max(bpm_max - bpm_min, 1e-9)
    return low_ratio + (high_ratio - low_ratio) * (t ** exponent)


def adjust_confidence_with_contractility(
    base_confidence: float,
    s1_prominence: float,
    s2_prominence: float,
    bpm: float,
    params: Dict,
    state: Optional[AnalysisState] = None,
    current_time_sec: Optional[float] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Contractility / prominence adjustment. S1/S2 ratio; expected from history or BPM power curve.
    When contractility_expected_use_history is True, expected is blended by recent pair rate:
    pair_rate in last N sec → weight history; (1 - pair_rate) → weight BPM (so many Lone S1s → more BPM).
    Single deviation from expected: inside band → boost (tent); outside → penalty (linear ramp).

    Returns (updated_confidence, step_dict) where step_dict is a confidence_trace step.
    """
    # --- 1. Expected S1/S2: history vs BPM, blended by pair rate in last N seconds ---
    history = getattr(state, "s1_s2_contractility_history", []) if state else []
    n_use = params.get("contractility_expected_history_count", 10)
    min_for_history = max(1, n_use // 2)
    use_history_flag = params.get("contractility_expected_use_history", True) and len(history) >= min_for_history

    expected_ratio_bpm = _contractility_expected_ratio_bpm(bpm, params)
    if use_history_flag:
        arr = np.array(history[-n_use:])
        if len(arr) > 2:
            arr = np.sort(arr)[1:-1]  # drop highest and lowest to reduce outlier impact
        expected_ratio_history = float(np.mean(arr))
    else:
        expected_ratio_history = expected_ratio_bpm  # not used when not use_history_flag

    # Pair rate in last N sec: 100% pairs → trust history; many Lone S1s → blend toward BPM
    pair_rate = 1.0
    if state is not None and current_time_sec is not None:
        outcomes = getattr(state, "recent_s1_outcomes", [])
        window_sec = params.get("contractility_pair_rate_window_sec", 5.0)
        cutoff = current_time_sec - window_sec
        in_window = [(t, p) for t, p in outcomes if t >= cutoff]
        if in_window:
            n_paired = sum(1 for _, p in in_window if p)
            pair_rate = n_paired / len(in_window)

    if use_history_flag:
        expected_ratio = pair_rate * expected_ratio_history + (1.0 - pair_rate) * expected_ratio_bpm
        expected_source = f"blend: {pair_rate:.0%} history + {1 - pair_rate:.0%} BPM (pair rate in last {params.get('contractility_pair_rate_window_sec', 5):.0f}s)"
    else:
        expected_ratio = expected_ratio_bpm
        expected_source = "BPM power curve"

    # --- 2. Measured S1/S2 prominence ratio ---
    actual_ratio = s1_prominence / (s2_prominence + 1e-9)

    # --- 3. Asymmetric piecewise-linear curve: L2, L1, R_exp, R1, R2 ---
    a_low = params.get("contractility_zero_crossing_low", 0.2)
    a_high = params.get("contractility_zero_crossing_high", 0.2)
    r_low = params.get("contractility_penalty_ramp_fraction_low", 0.4)
    r_high = params.get("contractility_penalty_ramp_fraction_high", 0.4)
    boost_max = params.get("contractility_boost_max", 0.10)
    penalty_max = params.get("contractility_penalty_max", 0.30)

    L2 = expected_ratio * (1.0 - r_low)
    L1 = expected_ratio * (1.0 - a_low)
    R1 = expected_ratio * (1.0 + a_high)
    R2 = expected_ratio * (1.0 + r_high)

    contractility_boost = 0.0
    contractility_penalty = 0.0

    if actual_ratio < L2:
        # Left of left ramp: full penalty
        contractility_penalty = penalty_max
    elif actual_ratio < L1:
        # Left penalty ramp: (L2, -P_max) to (L1, 0)
        span = L1 - L2
        if span > 1e-9:
            t = (L1 - actual_ratio) / span
            contractility_penalty = penalty_max * float(np.clip(t, 0, 1))
    elif actual_ratio <= R1:
        # Boost zone: (L1, 0) to (R_exp, B_max) to (R1, 0)
        if actual_ratio <= expected_ratio:
            span = expected_ratio - L1
            if span > 1e-9:
                t = (actual_ratio - L1) / span
                contractility_boost = boost_max * float(np.clip(t, 0, 1))
        else:
            span = R1 - expected_ratio
            if span > 1e-9:
                t = (R1 - actual_ratio) / span
                contractility_boost = boost_max * float(np.clip(t, 0, 1))
    elif actual_ratio <= R2:
        # Right penalty ramp: (R1, 0) to (R2, -P_max)
        span = R2 - R1
        if span > 1e-9:
            t = (actual_ratio - R1) / span
            contractility_penalty = penalty_max * float(np.clip(t, 0, 1))
    else:
        # Right of right ramp: full penalty
        contractility_penalty = penalty_max

    confidence = base_confidence
    if contractility_boost > 0:
        confidence = min(1.0, confidence * (1.0 + contractility_boost))
    if contractility_penalty > 0:
        confidence = max(0.0, confidence * (1.0 - contractility_penalty))

    detail = (
        f"S1={s1_prominence:.3f}, S2={s2_prominence:.3f}, S1/S2={actual_ratio:.2f} "
        f"(expected {expected_ratio:.2f} from {expected_source})"
    )
    if contractility_boost > 0:
        detail += f", boost +{contractility_boost:.2f} (×{(1.0 + contractility_boost):.2f})"
    elif contractility_penalty > 0:
        detail += f", penalty -{contractility_penalty:.2f} (×{(1.0 - contractility_penalty):.2f})"

    return confidence, {"step": "Contractility", "detail": detail, "result": confidence}


def _count_consecutive_noise_raw_peaks_before(
    all_peaks: np.ndarray,
    current_pos: int,
    peak_classifications: Dict[int, Any],
) -> int:
    """How many consecutive raw peaks immediately before all_peaks[current_pos] are NOISE."""
    n = 0
    j = current_pos - 1
    while j >= 0:
        idx = int(all_peaks[j])
        pc = peak_classifications.get(idx) or {}
        if pc.get("peak_type") != PeakType.NOISE.value:
            break
        n += 1
        j -= 1
    return n


def _try_lone_s1_missed_beat_rhythm(
    actual_rr_backward_sec: float,
    expected_rr_sec: float,
    current_peak_idx: int,
    all_peaks: np.ndarray,
    peak_classifications: Dict[int, Any],
    params: Dict,
) -> Optional[Tuple[float, str]]:
    """
    If the k >= 1 raw peaks immediately before the current peak are all NOISE, treat k missed
    beats: expect span last_S1→current ≈ (k+1)× expected RR, score (span/(k+1)) vs expected RR.
    """
    tol = float(params.get("lone_s1_missed_beat_tolerance_frac", 0.22))
    if expected_rr_sec <= 1e-12:
        return None

    pos = int(np.searchsorted(all_peaks, current_peak_idx))
    if pos <= 0 or pos >= len(all_peaks) or int(all_peaks[pos]) != int(current_peak_idx):
        return None

    n_noise = _count_consecutive_noise_raw_peaks_before(all_peaks, pos, peak_classifications)
    if n_noise < 1:
        return None

    mult = float(n_noise + 1)
    target_span = mult * expected_rr_sec
    rel_err = abs(actual_rr_backward_sec - target_span) / target_span
    if rel_err > tol:
        return None

    effective_rr = actual_rr_backward_sec / mult
    rhythm_deviation_pct = abs(effective_rr - expected_rr_sec) / expected_rr_sec
    score = float(np.interp(rhythm_deviation_pct, _RHYTHM_DEVIATION_XPOINTS, _RHYTHM_SCORE_YPOINTS))
    reason = (
        f"Rhythm Fit (missed beat): {n_noise} consecutive noise peak(s) before current → expect "
        f"{mult:.0f}× RR; span {actual_rr_backward_sec:.3f}s ≈ {mult:.0f}× expected {expected_rr_sec:.3f}s "
        f"(tolerance {tol:.0%}) → effective interval {effective_rr:.3f}s (deviation {rhythm_deviation_pct:.0%}) "
        f"→ score {score:.2f}"
    )
    return score, reason


def calculate_lone_s1_confidence(
    current_peak_idx: int,
    last_s1_idx: int,
    long_term_bpm: float,
    audio_envelope: np.ndarray,
    noise_floor_values: np.ndarray,
    sample_rate: int,
    params: Dict,
    all_peaks: Optional[np.ndarray] = None,
    peak_classifications: Optional[Dict[int, Any]] = None,
) -> Tuple[float, List[str]]:
    """
    Calculates a confidence score for a Lone S1 candidate based on a weighted average of
    its rhythmic timing and its amplitude consistency with the previous beat, and returns
    human-readable detail lines explaining the calculation.

    When the backward interval (last S1 → current) is poor, may use: (1) missed-beat recovery
    if consecutive raw peaks before current are noise and span ≈ (k+1)× expected RR for k misses;
    (2) forward interval (current → next-next peak) if the score is still zero.
    """
    # --- 1. Calculate Rhythmic Fit Score (backward: last S1 → current) ---
    expected_rr_sec = calculate_bpm_intervals(long_term_bpm, params)["rr_interval"]
    actual_rr_backward_sec = (current_peak_idx - last_s1_idx) / sample_rate
    rhythm_deviation_pct = abs(actual_rr_backward_sec - expected_rr_sec) / expected_rr_sec

    # Piecewise-linear map: deviation fraction → confidence score.
    # 0 % off → 1.0, 15 % off → 0.8 (minor tolerance), 40 % off → 0.4, >=60 % off → 0.0.
    rhythm_score_bw = float(np.interp(rhythm_deviation_pct, _RHYTHM_DEVIATION_XPOINTS, _RHYTHM_SCORE_YPOINTS))
    rhythm_score = rhythm_score_bw
    rhythm_reason = (
        f"Rhythm Fit: interval {actual_rr_backward_sec:.3f}s vs expected {expected_rr_sec:.3f}s "
        f"(deviation {rhythm_deviation_pct:.0%}; map 0/15/40/60% → 1.00/0.80/0.40/0.00) "
        f"→ score {rhythm_score_bw:.2f} → {rhythm_score_bw:.2f}"
    )

    if (
        peak_classifications is not None
        and all_peaks is not None
        and len(all_peaks) >= 2
    ):
        missed = _try_lone_s1_missed_beat_rhythm(
            actual_rr_backward_sec,
            expected_rr_sec,
            current_peak_idx,
            np.asarray(all_peaks, dtype=np.int64),
            peak_classifications,
            params,
        )
        if missed is not None:
            score_mb, reason_mb = missed
            if score_mb > rhythm_score:
                rhythm_score = score_mb
                rhythm_reason = reason_mb

    # If backward/missed-beat still gives zero, try forward interval (current → next-next peak) as fallback
    if rhythm_score <= 0.0 and all_peaks is not None and len(all_peaks) >= 3:
        pos = np.searchsorted(all_peaks, current_peak_idx)
        if pos < len(all_peaks) and all_peaks[pos] == current_peak_idx and pos + 2 < len(all_peaks):
            next_next_peak_idx = int(all_peaks[pos + 2])
            actual_rr_forward_sec = (next_next_peak_idx - current_peak_idx) / sample_rate
            rhythm_deviation_forward_pct = abs(actual_rr_forward_sec - expected_rr_sec) / expected_rr_sec
            rhythm_score_forward = np.interp(
                rhythm_deviation_forward_pct,
                _RHYTHM_DEVIATION_XPOINTS,
                _RHYTHM_SCORE_YPOINTS,
            )
            if rhythm_score_forward > 0.0:
                rhythm_score = rhythm_score_forward
                rhythm_reason = (
                    f"Rhythm Fit (forward fallback): backward interval {actual_rr_backward_sec:.3f}s too far from expected "
                    f"{expected_rr_sec:.3f}s → used forward interval {actual_rr_forward_sec:.3f}s (current→next-next) "
                    f"(deviation {rhythm_deviation_forward_pct:.0%}) → score {rhythm_score:.2f}"
                )
            else:
                rhythm_reason = (
                    f"Rhythm Fit: interval {actual_rr_backward_sec:.3f}s vs expected {expected_rr_sec:.3f}s "
                    f"(deviation {rhythm_deviation_pct:.0%}; map 0/15/40/60% → 1.00/0.80/0.40/0.00) "
                    f"→ score {rhythm_score:.2f}; forward fallback interval {actual_rr_forward_sec:.3f}s also poor → {rhythm_score:.2f}"
                )
        else:
            rhythm_reason = (
                f"Rhythm Fit: interval {actual_rr_backward_sec:.3f}s vs expected {expected_rr_sec:.3f}s "
                f"(deviation {rhythm_deviation_pct:.0%}; map 0/15/40/60% → 1.00/0.80/0.40/0.00) "
                f"→ score {rhythm_score:.2f} → {rhythm_score:.2f}"
            )

    # --- 2. Calculate Amplitude Fit Score ---
    last_s1_strength = max(0, audio_envelope[last_s1_idx] - noise_floor_values[last_s1_idx])
    current_peak_strength = max(0, audio_envelope[current_peak_idx] - noise_floor_values[current_peak_idx])
    amplitude_ratio = current_peak_strength / (last_s1_strength + 1e-9)

    # Piecewise-linear map: current/history amplitude ratio → confidence score.
    # A ratio near 1.0 means amplitude is consistent with recent history → full confidence.
    amplitude_score = np.interp(amplitude_ratio, _AMPLITUDE_RATIO_XPOINTS, _AMPLITUDE_SCORE_YPOINTS)
    amplitude_reason = (
        f"Amplitude Fit: strength ratio {amplitude_ratio:.2f}x "
        f"(map 0/0.4/0.7/1.0 → 0/0.4/0.7/1.0) → score {amplitude_score:.2f} → {amplitude_score:.2f}"
    )

    # --- 3. Combine Scores with Weights ---
    rhythm_weight = params.get('lone_s1_rhythm_weight', 0.65)
    amplitude_weight = params.get('lone_s1_amplitude_weight', 0.35)
    final_confidence = (rhythm_score * rhythm_weight) + (amplitude_score * amplitude_weight)

    reason_lines = [
        rhythm_reason,
        amplitude_reason,
        (
            f"Weighted Score: (Rhythm {rhythm_score:.2f}×{rhythm_weight:.2f}) + "
            f"(Amplitude {amplitude_score:.2f}×{amplitude_weight:.2f}) → combined → {final_confidence:.3f}"
        ),
    ]
    return final_confidence, reason_lines


# ---------------------------------------------------------------------------
# State history helpers
# ---------------------------------------------------------------------------

def _append_s1_s2_interval(state: AnalysisState, interval_sec: float, params: Dict) -> None:
    """Append an accepted systole (S1→S2) interval to history and cap to last N for history-based expected systole."""
    state.s1_s2_interval_history.append(interval_sec)
    n_keep = params.get("s1_s2_expected_history_count", 10)
    if len(state.s1_s2_interval_history) > n_keep:
        state.s1_s2_interval_history = state.s1_s2_interval_history[-n_keep:]


def _append_s1_s2_contractility(
    state: AnalysisState,
    s1_idx: int,
    s2_idx: int,
    audio_envelope: np.ndarray,
    trough_indices: np.ndarray,
    sample_rate: int,
    params: Dict,
) -> None:
    """Append an accepted pair's S1/S2 prominence ratio to history for expected contractility."""
    s1_details = get_peak_prominence_details(
        s1_idx, audio_envelope, trough_indices, sample_rate,
        detail_cache=state.peak_prominence_detail_cache,
    )
    s2_details = get_peak_prominence_details(
        s2_idx, audio_envelope, trough_indices, sample_rate,
        detail_cache=state.peak_prominence_detail_cache,
    )
    s1_prom = s1_details["prominence"]
    s2_prom = s2_details["prominence"]
    ratio = s1_prom / (s2_prom + 1e-9)
    state.s1_s2_contractility_history.append(ratio)
    n_keep = params.get("contractility_expected_history_count", 10)
    if len(state.s1_s2_contractility_history) > n_keep:
        state.s1_s2_contractility_history = state.s1_s2_contractility_history[-n_keep:]


def record_s1_outcome(state: AnalysisState, time_sec: float, was_paired: bool, params: Dict) -> None:
    """Record an S1 decision (paired or lone) and trim recent_s1_outcomes to the pair-rate window."""
    state.recent_s1_outcomes.append((time_sec, was_paired))
    window_sec = params.get("contractility_pair_rate_window_sec", 5.0)
    cutoff = time_sec - window_sec
    state.recent_s1_outcomes = [(t, p) for t, p in state.recent_s1_outcomes if t >= cutoff]


def _get_recent_s1_prominences_for_state(
    state: AnalysisState,
    audio_envelope: np.ndarray,
    trough_indices: np.ndarray,
) -> List[float]:
    """
    Helper to compute recent validated S1 prominences from an AnalysisState instance.

    Kept outside of PeakClassifier so it can be reused by PairingEngine without
    giving it write access to classifier internals.
    """
    recent_s1_types = [
        state.peak_classifications.get(idx, {}).get("peak_type")
        for idx in state.candidate_beats[-50:]
    ]
    cache = state.peak_prominence_detail_cache
    out: List[float] = []
    for idx, typ in zip(state.candidate_beats[-50:], recent_s1_types):
        if typ not in (PeakType.S1_PAIRED.value, PeakType.LONE_S1_VALIDATED.value):
            continue
        if cache is not None and idx in cache:
            out.append(float(cache[idx]["prominence"]))
        else:
            out.append(calculate_peak_prominence(idx, audio_envelope, trough_indices))
    return out


# ---------------------------------------------------------------------------
# Pairing engine
# ---------------------------------------------------------------------------

class PairingEngine:
    """
    Scores candidate S1→S2 (systole) pairs and returns a pairing decision plus debug context.

    This class is intentionally stateless (mostly stateless) with respect to the main analysis loop:
    it never mutates `AnalysisState` and instead relies on the caller (`PeakClassifier`)
    to own all state updates. This keeps the confidence model self-contained and
    easier to reason about in isolation.
    """

    def __init__(
        self,
        audio_envelope: np.ndarray,
        sample_rate: int,
        params: Dict,
        peak_bpm_time_sec: Optional[float],
        recovery_end_time_sec: Optional[float],
    ) -> None:
        self.audio_envelope = audio_envelope
        self.sample_rate = sample_rate
        self.params = params
        self.peak_bpm_time_sec = peak_bpm_time_sec
        self.recovery_end_time_sec = recovery_end_time_sec

    def attempt_pair(
        self,
        state: AnalysisState,
        s1_candidate_idx: int,
        s2_candidate_idx: int,
        pairing_ratio: float,
    ) -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Calculates the confidence score for pairing two candidate peaks.

        Returns:
            (is_paired, steps, prominence_context)
        where:
            - is_paired: True if the final confidence exceeds the configured threshold
            - steps: list of confidence_trace step dicts explaining each scoring decision
            - prominence_context: raw prominence details for S1/S2 used by debug tools
        """
        interval_sec = (s2_candidate_idx - s1_candidate_idx) / self.sample_rate
        bpm = state.long_term_bpm

        intervals = calculate_bpm_intervals(bpm, self.params)
        history = getattr(state, "s1_s2_interval_history", []) or []
        n_use = self.params.get("s1_s2_expected_history_count", 10)
        # Use BPM-based expected until queue is at least half the window (e.g. first 5 of 10); then use history.
        min_for_history = max(1, n_use // 2)
        use_history = self.params.get("s1_s2_expected_use_history", True) and len(history) >= min_for_history
        if use_history:
            arr = np.array(history[-n_use:])
            if len(arr) > 2:
                arr = np.sort(arr)[1:-1]  # drop highest and lowest to reduce outlier impact
            expected_s1_s2 = float(np.mean(arr))
            expected_s1_s2_source = f"past {len(history[-n_use:])} pairs"
        else:
            expected_s1_s2 = intervals["s1_s2_nominal"]
            expected_s1_s2_source = "BPM"
        short_cutoff = expected_s1_s2 * self.params.get("interval_v_short_ramp_end_fraction", 0.2)
        long_reject = expected_s1_s2 * self.params.get("interval_v_long_reject_fraction", 3.0)

        if interval_sec < short_cutoff:
            implied_total_cycle = interval_sec * 2.0
            implied_bpm = 60.0 / implied_total_cycle if implied_total_cycle > 0 else float('inf')
            detail = (
                f"Systole (S1→S2) interval {interval_sec:.3f}s < short cutoff {short_cutoff:.3f}s "
                f"(expected {expected_s1_s2:.3f}s @ {bpm:.0f} BPM; implies {implied_bpm:.0f} BPM)"
            )
            return False, [{"step": "Interval Reject", "detail": detail, "result": 0.0}], {}

        if interval_sec >= long_reject:
            detail = (
                f"Systole (S1→S2) interval {interval_sec:.3f}s >= long reject {long_reject:.3f}s "
                f"(expected {expected_s1_s2:.3f}s @ {bpm:.0f} BPM)"
            )
            return False, [{"step": "Interval Reject", "detail": detail, "result": 0.0}], {}

        steps: List[Dict[str, Any]] = []

        # --- Base confidence: neutral starting point (contractility handled by prominence adjustment) ---
        base_confidence = 0.60
        steps.append({"step": "Base", "detail": "starting confidence", "result": base_confidence})

        # --- RR vs BPM penalty (penalty only) ---
        # If the proposed S1 would create an implausible R-R interval vs the current BPM belief,
        # penalize pairing confidence. This helps avoid pairing when the "S1 candidate" is actually noise.
        confidence = base_confidence
        if state.candidate_beats:
            last_s1_idx = state.candidate_beats[-1]
            expected_rr_sec = float(intervals.get("rr_interval", 0.0))
            observed_rr_sec = (s1_candidate_idx - last_s1_idx) / self.sample_rate
            if expected_rr_sec > 1e-9 and observed_rr_sec > 0:
                rr_deviation_frac = abs(observed_rr_sec - expected_rr_sec) / expected_rr_sec
                # Allow a small dead zone: no penalty for deviations ≤ 5%; ramp starts beyond that.
                rr_deviation_for_penalty = max(0.0, rr_deviation_frac - 0.05)
                rr_score = float(np.interp(rr_deviation_for_penalty, _RHYTHM_DEVIATION_XPOINTS, _RHYTHM_SCORE_YPOINTS))
                rr_penalty_max = float(self.params.get("pairing_rr_penalty_max", 0.25))
                rr_penalty = rr_penalty_max * float(np.clip(1.0 - rr_score, 0.0, 1.0))
                if rr_penalty > 0:
                    confidence *= max(0.0, 1.0 - rr_penalty)
                    steps.append({
                        "step": "RR vs BPM",
                        "detail": (
                            f"{observed_rr_sec:.3f}s vs {expected_rr_sec:.3f}s "
                            f"(deviation {rr_deviation_frac:.0%}) → -{rr_penalty:.2f} (×{(1.0 - rr_penalty):.2f})"
                        ),
                        "result": confidence,
                    })

        # --- Contractility / prominence-based adjustment (S1 vs S2) ---
        s1_details = get_peak_prominence_details(
            s1_candidate_idx,
            self.audio_envelope,
            state.trough_indices,
            sample_rate=self.sample_rate,
            detail_cache=state.peak_prominence_detail_cache,
        )
        s2_details = get_peak_prominence_details(
            s2_candidate_idx,
            self.audio_envelope,
            state.trough_indices,
            sample_rate=self.sample_rate,
            detail_cache=state.peak_prominence_detail_cache,
        )
        s1_prominence = s1_details["prominence"]
        s2_prominence = s2_details["prominence"]

        shared_s1_right_s2_left = (
            s1_details.get("right_trough_idx") is not None
            and s2_details.get("left_trough_idx") is not None
            and s1_details["right_trough_idx"] == s2_details["left_trough_idx"]
        )
        prominence_context = {
            "s1": s1_details,
            "s2": s2_details,
            "shared_s1_right_s2_left": shared_s1_right_s2_left,
        }

        # --- Contractility model based on S1/S2 prominence ratio ---
        current_time_sec = s1_candidate_idx / self.sample_rate
        confidence, contractility_step = adjust_confidence_with_contractility(
            confidence,
            s1_prominence,
            s2_prominence,
            bpm,
            self.params,
            state=state,
            current_time_sec=current_time_sec,
        )
        steps.append(contractility_step)

        # --- Absolute S1 prominence guardrail (shared with Lone S1 logic) ---
        # Protect against tiny noise bumps being interpreted as heartbeats in a "high contractility" S1/S2 pair.
        # We compare the current S1 prominence against a recent high-quality S1 baseline.
        recent_prominences = _get_recent_s1_prominences_for_state(
            state, self.audio_envelope, state.trough_indices
        )
        if len(recent_prominences) >= 5:
            reference_prominence = np.percentile(recent_prominences, 80)  # Top 20% as adaptive reference
            if reference_prominence > 0:
                # Re-use Lone S1 ratio setting for now to keep behavior consistent
                min_ratio = self.params.get(
                    "paired_s1_min_prominence_ratio",
                    self.params.get("lone_s1_min_prominence_ratio", 0.4),  # Tuned magic number, see docs.
                )
                prominence_ratio = s1_prominence / (reference_prominence + 1e-9)

                if prominence_ratio < min_ratio:
                    # Linear penalty, mirroring Lone S1 behavior:
                    #   at ratio == min_ratio -> no penalty
                    #   at ratio  == 0       -> full veto
                    penalty_factor = float(np.clip(prominence_ratio / (min_ratio + 1e-9), 0.0, 1.0))
                    confidence *= penalty_factor
                    steps.append({
                        "step": "Abs. S1 Prominence",
                        "detail": (
                            f"{s1_prominence:.3f} < {min_ratio:.1f}× reference "
                            f"({reference_prominence:.3f}) → ×{penalty_factor:.2f}"
                        ),
                        "result": confidence,
                    })

        # --- V-shaped interval: linear boost when close to expected, linear penalty outside the boost zone ---
        # Hard cutoffs already applied above.
        interval_v_penalty = 0.0
        interval_v_boost = 0.0
        v_penalty_max = self.params.get("interval_v_penalty_max", 0.2)
        v_boost_max = self.params.get("interval_v_boost_max", 0.10)
        zero_crossing_fraction = self.params.get("interval_zero_crossing_fraction", 0.2)
        long_ramp_end = expected_s1_s2 * self.params.get("interval_v_long_ramp_end_fraction", 2.0)
        left_ramp_start = expected_s1_s2 * (1.0 - zero_crossing_fraction)   # below this: left penalty ramp
        right_ramp_start = expected_s1_s2 * (1.0 + zero_crossing_fraction)  # above this: right penalty ramp

        if interval_sec < left_ramp_start:
            # Left penalty ramp: from short_cutoff to left_ramp_start, penalty 0 at left_ramp_start, v_penalty_max at short_cutoff
            ramp_span = left_ramp_start - short_cutoff
            if ramp_span > 1e-9:
                t = (left_ramp_start - interval_sec) / ramp_span
                interval_v_penalty = v_penalty_max * float(np.clip(t, 0, 1))
        elif interval_sec > right_ramp_start:
            # Right penalty ramp: from right_ramp_start to long_ramp_end; flat v_penalty_max beyond long_ramp_end
            if interval_sec <= long_ramp_end:
                ramp_span = long_ramp_end - right_ramp_start
                if ramp_span > 1e-9:
                    t = (interval_sec - right_ramp_start) / ramp_span
                    interval_v_penalty = v_penalty_max * float(np.clip(t, 0, 1))
            else:
                interval_v_penalty = v_penalty_max
        else:
            # Boost zone [left_ramp_start, right_ramp_start]: linear boost from 0 at edges to v_boost_max at expected
            if interval_sec <= expected_s1_s2:
                span = expected_s1_s2 - left_ramp_start
                if span > 1e-9:
                    t = (interval_sec - left_ramp_start) / span
                    interval_v_boost = v_boost_max * float(np.clip(t, 0, 1))
            else:
                span = right_ramp_start - expected_s1_s2
                if span > 1e-9:
                    t = (right_ramp_start - interval_sec) / span
                    interval_v_boost = v_boost_max * float(np.clip(t, 0, 1))

        # Always show actual vs expected interval on hover (even when no penalty/boost)
        interval_base_detail = f"{interval_sec:.3f}s (expected {expected_s1_s2:.3f}s from {expected_s1_s2_source})"
        if interval_v_penalty > 0:
            confidence *= max(0.0, 1.0 - interval_v_penalty)
            steps.append({
                "step": "Systole (S1→S2) interval",
                "detail": f"{interval_base_detail}. Too far → -{interval_v_penalty:.2f} (×{(1.0 - interval_v_penalty):.2f})",
                "result": confidence,
            })
        elif interval_v_boost > 0:
            confidence = min(1.0, confidence * (1.0 + interval_v_boost))
            steps.append({
                "step": "Systole (S1→S2) interval",
                "detail": f"{interval_base_detail}. Near expected → +{interval_v_boost:.2f} (×{(1.0 + interval_v_boost):.2f})",
                "result": confidence,
            })
        else:
            steps.append({"step": "Systole (S1→S2) interval", "detail": f"{interval_base_detail} → no change", "result": confidence})

        # --- Forward-Looking Contextual Penalty ---
        # If pairing systole (S1→S2) causes the next S2→S1 transition to be implausible, penalize it.
        # Guardrail: only trust this check if the "next S1" peak is strong enough to be
        # a plausible beat; otherwise it may just be noise and should not veto the pair.
        if state.loop_idx + 2 < len(state.all_peaks):
            next_next_peak_idx = state.all_peaks[state.loop_idx + 2]
            next_s1_details = get_peak_prominence_details(
                next_next_peak_idx,
                self.audio_envelope,
                state.trough_indices,
                sample_rate=self.sample_rate,
                detail_cache=state.peak_prominence_detail_cache,
            )
            next_s1_prominence = next_s1_details["prominence"]

            if s2_prominence > 1e-9 and next_s1_prominence > 1e-9:
                noise_thresh = self.params.get('noise_prominence_threshold', 0.35)
                if next_s1_prominence >= s2_prominence * noise_thresh:
                    # Next peak is strong enough to evaluate; apply penalty if needed.
                    drop_ratio = next_s1_prominence / (s2_prominence + 1e-9)
                    threshold = self.params.get('forward_look_drop_threshold', 0.4)
                    if drop_ratio < threshold:
                        severity = (threshold - drop_ratio) / threshold
                        forward_look_penalty = severity * self.params.get('forward_look_max_penalty', 0.3)
                        confidence = max(0.0, confidence - forward_look_penalty)
                        steps.append({
                            "step": "Forward-Look",
                            "detail": f"S2→S1 drop {drop_ratio:.2f}x < threshold {threshold:.1f}x → -{forward_look_penalty:.2f}",
                            "result": confidence,
                        })

        # --- Stability (pairing ratio) multiplier applied last ---
        # Scale confidence by recent pairing success so we don't over-penalize when S2 was absent at high BPM.
        if len(state.candidate_beats) >= 5:
            floor = self.params.get("stability_confidence_floor", 0.85)
            ceiling = self.params.get("stability_confidence_ceiling", 1.10)
            current_time_sec = s1_candidate_idx / self.sample_rate
            # During recovery, pairing_ratio can be low because S2 was physiologically absent at high BPM.
            # Use a higher floor so we don't penalize valid systole (S1→S2) pairs when S2 re-emerges.
            if (self.peak_bpm_time_sec is not None and
                self.recovery_end_time_sec is not None and
                self.peak_bpm_time_sec <= current_time_sec <= self.recovery_end_time_sec):
                recovery_floor = self.params.get("recovery_phase_stability_floor", 0.95)
                floor = max(floor, recovery_floor)
                steps.append({"step": "Recovery Phase", "detail": f"using floor {floor:.2f}", "result": confidence})
            stability_factor = np.interp(pairing_ratio, [0.0, 1.0], [floor, ceiling])
            confidence = max(0.0, min(1.0, confidence * stability_factor))
            steps.append({
                "step": "Stability",
                "detail": f"pairing ratio {pairing_ratio:.0%} → ×{stability_factor:.2f}",
                "result": confidence,
            })

        is_paired = confidence >= self.params['pairing_confidence_threshold']
        steps.append({
            "step": "Final",
            "detail": (
                f"score {confidence:.2f} vs threshold {self.params['pairing_confidence_threshold']:.2f} "
                f"→ {'Paired' if is_paired else 'Not Paired'}"
            ),
            "result": confidence,
        })
        return is_paired, steps, prominence_context


# ---------------------------------------------------------------------------
# Lookahead skipper
# ---------------------------------------------------------------------------

class LookaheadSkipper:
    """
    Encapsulates the "skip weak middle peak" logic used before standard pairing.

    This component proposes an alternative S1→S2′ interpretation when a middle peak
    is likely noise. It never mutates `AnalysisState`; instead it returns a structured
    decision and lets the caller update state and debug records.
    """

    def __init__(
        self,
        audio_envelope: np.ndarray,
        sample_rate: int,
        params: Dict,
        pairing_engine: PairingEngine,
    ) -> None:
        self.audio_envelope = audio_envelope
        self.sample_rate = sample_rate
        self.params = params
        self.pairing_engine = pairing_engine

    def maybe_skip(
        self,
        state: AnalysisState,
        loop_idx: int,
        pairing_ratio: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Returns a decision dict if lookahead wants to reinterpret a middle peak as noise.

        The returned dict contains:
            - s1_idx, middle_idx, s2_idx: indices of the three peaks
            - steps: confidence_trace step dicts from the pairing attempt
            - prominence_context: raw prominence context for debug tooltips
            - lookahead_msg: human-readable summary of *why* the middle was skipped
            - middle_noise_msg: explanation attached to the middle peak marked as Noise

        If no lookahead skip is appropriate, returns None.
        """
        if not self.params.get('enable_lookahead_skipping', True):
            return None

        all_peaks = state.all_peaks
        if loop_idx + 2 >= len(all_peaks):
            return None

        current_peak_idx = all_peaks[loop_idx]
        next_peak_idx = all_peaks[loop_idx + 1]
        middle_peak_idx = next_peak_idx
        next_next_peak_idx = all_peaks[loop_idx + 2]

        # Heuristic: treat the middle peak as noise if its prominence is much weaker than the S1 candidate
        dc = state.peak_prominence_detail_cache
        current_prom = calculate_peak_prominence(
            current_peak_idx, self.audio_envelope, state.trough_indices, detail_cache=dc
        )
        middle_prom = calculate_peak_prominence(
            middle_peak_idx, self.audio_envelope, state.trough_indices, detail_cache=dc
        )
        next_next_prom = calculate_peak_prominence(
            next_next_peak_idx, self.audio_envelope, state.trough_indices, detail_cache=dc
        )
        noise_thresh = self.params.get('noise_prominence_threshold', 0.35)

        bpm = state.long_term_bpm
        intervals = calculate_bpm_intervals(bpm, self.params)
        min_s1_s2_interval = intervals["s1_s2_min"]
        s1_s2_max_interval = intervals["s1_s2_max"]

        s2_to_next_s1_interval_sec = (next_next_peak_idx - middle_peak_idx) / self.sample_rate
        alt_s1_s2_interval_sec = (next_next_peak_idx - current_peak_idx) / self.sample_rate

        # Determine if middle is skippable based on BOTH intervals AND intensity.
        middle_is_weak = middle_prom < current_prom * noise_thresh
        interval_is_impossible = s2_to_next_s1_interval_sec < min_s1_s2_interval
        next_next_is_strong = next_next_prom > middle_prom
        alt_interval_plausible = (
            alt_s1_s2_interval_sec >= min_s1_s2_interval
            and alt_s1_s2_interval_sec <= s1_s2_max_interval
        )

        # --- Mode 1: interval-aware reinterpretation (impossible S2→S1 interval) ---
        if middle_is_weak and interval_is_impossible and next_next_is_strong and alt_interval_plausible:
            is_paired_interval, steps_interval, prominence_context_interval = self.pairing_engine.attempt_pair(
                state, current_peak_idx, next_next_peak_idx, pairing_ratio
            )

            if is_paired_interval:
                lookahead_msg_interval = (
                    "LOOKAHEAD INTERVAL: Reinterpreted middle peak as noise because the implied "
                    f"S2→S1 interval {s2_to_next_s1_interval_sec:.3f}s is below the minimum "
                    f"{min_s1_s2_interval:.3f}s for BPM {bpm:.0f}, the alternative S1→S2′ interval "
                    f"{alt_s1_s2_interval_sec:.3f}s is within [{min_s1_s2_interval:.3f}, "
                    f"{s1_s2_max_interval:.3f}]s, and the middle peak is weak "
                    f"({middle_prom:.3f} < {noise_thresh:.2f} × S1 {current_prom:.3f}) while the "
                    f"next candidate is stronger ({next_next_prom:.3f} > {middle_prom:.3f})."
                )
                middle_noise_msg_interval = (
                    "Middle peak treated as noise due to impossible S2→S1 interval "
                    f"({s2_to_next_s1_interval_sec:.3f}s < {min_s1_s2_interval:.3f}s), with a "
                    f"plausible alternative S1→S2′ interval ({alt_s1_s2_interval_sec:.3f}s "
                    f"within [{min_s1_s2_interval:.3f}, {s1_s2_max_interval:.3f}]s), "
                    f"weak prominence ({middle_prom:.3f} < {noise_thresh:.2f} × S1 "
                    f"{current_prom:.3f}), and a stronger following candidate "
                    f"({next_next_prom:.3f} > {middle_prom:.3f})."
                )

                return {
                    "s1_idx": current_peak_idx,
                    "middle_idx": middle_peak_idx,
                    "s2_idx": next_next_peak_idx,
                    "steps": steps_interval,
                    "prominence_context": prominence_context_interval,
                    "lookahead_msg": lookahead_msg_interval,
                    "middle_noise_msg": middle_noise_msg_interval,
                }

        # --- Mode 2: simpler prominence-based reinterpretation ---
        if (
            middle_prom < current_prom * noise_thresh
            and middle_prom < next_next_prom
            and alt_interval_plausible
        ):
            is_paired_skip, steps_skip, prominence_context_skip = self.pairing_engine.attempt_pair(
                state, current_peak_idx, next_next_peak_idx, pairing_ratio
            )

            if is_paired_skip:
                lookahead_msg = (
                    "LOOKAHEAD SUCCESS: Skipped intermediate weak peak "
                    f"(middle prominence {middle_prom:.3f} < {noise_thresh:.2f} × "
                    f"S1 prominence {current_prom:.3f} and next candidate prominence "
                    f"{next_next_prom:.3f} > middle) with plausible S1→S2′ interval "
                    f"{alt_s1_s2_interval_sec:.3f}s within "
                    f"[{min_s1_s2_interval:.3f}, {s1_s2_max_interval:.3f}]s"
                )
                middle_noise_msg = (
                    "Middle peak treated as noise due to weak prominence "
                    f"({middle_prom:.3f} < {noise_thresh:.2f} × "
                    f"S1 prominence {current_prom:.3f}) and the following "
                    f"candidate is stronger (next prominence {next_next_prom:.3f})."
                )

                return {
                    "s1_idx": current_peak_idx,
                    "middle_idx": middle_peak_idx,
                    "s2_idx": next_next_peak_idx,
                    "steps": steps_skip,
                    "prominence_context": prominence_context_skip,
                    "lookahead_msg": lookahead_msg,
                    "middle_noise_msg": middle_noise_msg,
                }

        return None
