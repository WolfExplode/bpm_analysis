import os
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any, Callable, List

from scipy.interpolate import interp1d

from audio_preprocessing import preprocess_audio
from config import DEFAULT_OUTPUT_OPTIONS, output_stem_from_path
from plotting import Plotter, prewarm_kaleido_png_export
from reporting import ReportGenerator
from validation import (
    _load_manual_labels_csv,
    _build_predicted_labels_for_validation,
    _append_validation_results_row,
)
from classifier import PeakClassifier
from confidence_engine import calculate_bpm_intervals
from hrv import (
    calculate_bpm_series,
    calculate_bpm_series_from_s1_state_labels,
    compute_pass1_bpm_curve,
    filter_instant_bpm_mad,
    find_recovery_phase,
    smooth_bpm_series_from_instant,
    find_major_hr_inclines,
    find_major_hr_declines,
    calculate_hrr,
    find_peak_recovery_rate,
    find_peak_exertion_rate,
    calculate_windowed_hrv,
    calculate_global_hrv_frequency,
)
from fft_profiles import (
    compute_fft_profiles,
    compute_frequency_separation,
    prepare_pass3_s1_insert_context,
    save_fft_profiles_html,
    spectrum_s1_search_envelope_index,
)


class _NoisyAlgorithmLogFilter(logging.Filter):
    """
    Filters out very chatty INFO-level messages that make benchmarking hard.
    WARNING/ERROR always pass through.
    """

    # Substrings that identify "noisy" algorithm-detail logs.
    _NOISY_SUBSTRINGS = (
        "LOOKAHEAD ",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True

        try:
            msg = record.getMessage()
        except Exception:
            return True

        return not any(s in msg for s in self._NOISY_SUBSTRINGS)


def _run_pass1(audio_envelope: np.ndarray, sample_rate: int, params: Dict,
               noise_floor: pd.Series, troughs: np.ndarray,
               start_bpm_hint: Optional[float],
               ) -> Tuple[float, Optional[float], Optional[float], np.ndarray, Optional[Dict], Dict]:
    """
    Runs pass 1 (high-confidence anchor-finding) to estimate global BPM and find the recovery phase.
    Returns (start_bpm, peak_bpm_time_sec, recovery_end_time_sec, anchor_beats, pass1_bpm, pass1_analysis_data).
    pass1_bpm is the canonical curve (outlier-filtered + LOESS) used for prior and all plots, or None if insufficient data.
    """
    logging.info("--- STAGE 2: Pass 1 — high-confidence anchor beats ---")
    params_pass1 = params.copy()
    params_pass1["pairing_confidence_threshold"] = params.get(
        "pass1_pairing_confidence_threshold", 0.7
    )

    classifier = PeakClassifier(audio_envelope, sample_rate, params_pass1, start_bpm_hint,
                               noise_floor, troughs, None, None)
    anchor_beats, _, pass1_analysis_data = classifier.classify_peaks()

    global_bpm_estimate = None
    if len(anchor_beats) >= 10:
        median_rr_sec = np.median(np.diff(anchor_beats) / sample_rate)
        if median_rr_sec > 0:
            global_bpm_estimate = 60.0 / median_rr_sec
            logging.info(f"Automatically determined Global BPM Estimate: {global_bpm_estimate:.1f} BPM")

    start_bpm = start_bpm_hint or global_bpm_estimate or 80.0

    # Canonical pass 1 BPM curve (outlier filter + LOESS) — same data used for prior and all plots
    pass1_bpm = compute_pass1_bpm_curve(anchor_beats, sample_rate, params)
    if pass1_bpm is not None:
        curve_series = pd.Series(pass1_bpm["curve_bpm"])
        peak_bpm_time_sec, recovery_end_time_sec = find_recovery_phase(curve_series, pass1_bpm["curve_times"], params)
    else:
        pass1_fallback_series, pass1_fallback_times, _ = calculate_bpm_series(anchor_beats, sample_rate, params)
        peak_bpm_time_sec, recovery_end_time_sec = find_recovery_phase(pass1_fallback_series, pass1_fallback_times, params)

    return start_bpm, peak_bpm_time_sec, recovery_end_time_sec, anchor_beats, pass1_bpm, pass1_analysis_data


def _build_pass1_bpm_prior(
    pass1_bpm_times: np.ndarray,
    pass1_bpm_series: pd.Series,
) -> Optional[Callable[[float], float]]:
    """Build a time -> BPM callable from the pass 1 BPM curve for use as a time-varying prior. Returns None if insufficient data."""
    if pass1_bpm_times is None or pass1_bpm_series is None or len(pass1_bpm_times) < 2 or pass1_bpm_series.empty:
        return None
    times = np.asarray(pass1_bpm_times, dtype=float)
    values = np.asarray(pass1_bpm_series.values, dtype=float)
    if len(times) != len(values) or len(times) < 2:
        return None
    try:
        interp = interp1d(
            times,
            values,
            kind="linear",
            bounds_error=False,
            fill_value=(float(values[0]), float(values[-1])),
        )
        return lambda t_sec: float(interp(t_sec))
    except Exception:
        return None


def _refine_and_correct_peaks(s1_peaks: np.ndarray, all_raw_peaks: np.ndarray,
                              analysis_data: Dict, audio_envelope: np.ndarray,
                              sample_rate: int, params: Dict,
                              wav_file_path: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
    """
    Phase 3 template.
    Replace with new Stage 4/5 correction logic.
    """
    if "peak_classifications" not in analysis_data or analysis_data["peak_classifications"] is None:
        analysis_data["peak_classifications"] = {}
    if "s1_s2_pairs" not in analysis_data:
        analysis_data["s1_s2_pairs"] = []

    peaks_out = np.asarray(s1_peaks)
    if len(peaks_out) < 2:
        return peaks_out, analysis_data

    # ---------------------------------------------------------------------
    # Pass 3 (bridge): correction + dense per-sample cardiac-state timeline
    # ---------------------------------------------------------------------
    n_samples = int(len(audio_envelope))
    if n_samples <= 0:
        return peaks_out, analysis_data

    # State encoding (ints) for compact storage and easy plotting/debugging.
    # 0: S1, 1: systole, 2: S2, 3: diastole
    STATE_S1 = 0
    STATE_SYSTOLE = 1
    STATE_S2 = 2
    STATE_DIASTOLE = 3

    state_labels = np.full(n_samples, STATE_DIASTOLE, dtype=np.int8)
    state_boundaries = []

    # Event window sizes.
    s1_window_ms = float(params.get("pass3_state_s1_window_ms", 80.0))
    s2_window_ms = float(params.get("pass3_state_s2_window_ms", 80.0))
    s1_half = max(1, int(round(0.5 * s1_window_ms * sample_rate / 1000.0)))
    s2_half = max(1, int(round(0.5 * s2_window_ms * sample_rate / 1000.0)))

    # Optional: snap predicted S2 to the best nearby raw peak using label_scores["S2"].
    snap_s2 = bool(params.get("pass3_snap_s2_to_peak", True))
    snap_window_ms = float(params.get("pass3_snap_s2_window_ms", 120.0))
    snap_half = max(1, int(round(0.5 * snap_window_ms * sample_rate / 1000.0)))
    max_snap_dist_sec = float(params.get("pass3_snap_s2_max_dist_sec", 0.12))

    pc = analysis_data.get("peak_classifications") or {}
    lt = analysis_data.get("long_term_bpm_series")

    # Fallback BPM if long-term series is missing.
    fallback_bpm = None
    try:
        rr = np.diff(peaks_out) / float(sample_rate)
        rr = rr[np.isfinite(rr) & (rr > 0)]
        if len(rr) > 0:
            fallback_bpm = float(60.0 / np.median(rr))
    except Exception:
        fallback_bpm = None
    if fallback_bpm is None or not np.isfinite(fallback_bpm):
        fallback_bpm = 80.0

    def _bpm_at_time(t_sec: float) -> float:
        if lt is None or getattr(lt, "empty", True):
            return fallback_bpm
        try:
            times = np.asarray(lt.index.values, dtype=np.float64)
            values = np.asarray(lt.values, dtype=np.float64)
            if len(times) < 2 or len(times) != len(values):
                return fallback_bpm
            bpm = float(np.interp(float(t_sec), times, values, left=values[0], right=values[-1]))
            if not np.isfinite(bpm) or bpm <= 0:
                return fallback_bpm
            return bpm
        except Exception:
            return fallback_bpm

    # --- Helper: choose best S2 near predicted time ---
    def _choose_s2_near(s1: int, s1_next: int, s2_pred: int, half_window_samples: int) -> int:
        s2 = int(max(s1 + 1, min(s2_pred, s1_next - 1)))
        if (not snap_s2) or (len(all_raw_peaks) == 0) or (s1_next <= s1 + 2):
            return s2
        lo = max(s1 + 1, s2_pred - half_window_samples)
        hi = min(s1_next - 1, s2_pred + half_window_samples)
        if hi <= lo:
            return s2
        cand = [int(p) for p in all_raw_peaks if lo <= int(p) <= hi]
        if not cand:
            return s2
        best = None
        best_score = None
        for p in cand:
            entry = pc.get(int(p)) or {}
            ls = entry.get("label_scores") if isinstance(entry, dict) else None
            s2_score = float(ls.get("S2", 0.0)) if isinstance(ls, dict) else 0.0
            noise_score = float(ls.get("noise", 0.0)) if isinstance(ls, dict) else 0.0
            dist_sec = abs(p - s2_pred) / float(sample_rate)
            if dist_sec > max_snap_dist_sec:
                continue
            score = (2.0 * s2_score) - (1.0 * noise_score) - (0.75 * dist_sec)
            if best is None or score > best_score:
                best, best_score = p, score
        return int(best) if best is not None else s2

    # --- Helper: choose best S1 near expected time ---
    def _choose_s1_near(t_expected_sec: float, half_window_samples: int, min_sep_samples: int) -> Optional[int]:
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
        best = None
        best_score = None
        for p in cand:
            # Keep away from existing chosen S1 peaks
            # (caller also checks, but we avoid obvious near-duplicates early).
            entry = pc.get(int(p)) or {}
            ls = entry.get("label_scores") if isinstance(entry, dict) else None
            s1_score = float(ls.get("S1", 0.0)) if isinstance(ls, dict) else 0.0
            noise_score = float(ls.get("noise", 0.0)) if isinstance(ls, dict) else 0.0
            dist_sec = abs(p - center) / float(sample_rate)
            score = (2.0 * s1_score) - (1.0 * noise_score) - (0.75 * dist_sec)
            if best is None or score > best_score:
                best, best_score = p, score
        if best is None:
            return None
        # Min separation gate (avoid snapping to something essentially on top of an existing beat)
        # Caller provides min_sep_samples based on min_peak_distance_sec.
        return int(best)

    def _insert_spectrum_envelope_ok(env_idx: int) -> bool:
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

    def _find_sensitive_peaks_near(t_expected_sec: float, window_samples: int, sensitivity_factor: float) -> Optional[int]:
        """
        Re-scan audio_envelope in a narrow window with a lower height threshold to find
        faint peaks that the main detector missed. Returns the best candidate sample index
        (highest amplitude among found peaks) or None. Does not consult label_scores —
        this is a pure signal-level check to guard against over-reliance on spectral search.
        """
        from scipy.signal import find_peaks as _find_peaks
        nfs = analysis_data.get("dynamic_noise_floor_series")
        center = int(round(t_expected_sec * sample_rate))
        lo = max(0, center - window_samples)
        hi = min(n_samples - 1, center + window_samples)
        if hi <= lo:
            return None
        segment = audio_envelope[lo:hi + 1]
        if len(segment) == 0:
            return None
        # Build a per-sample sensitivity-adjusted height threshold for the segment.
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
            local_peaks, _ = _find_peaks(segment, height=height_thresh, distance=min_dist)
        except Exception:
            return None
        if len(local_peaks) == 0:
            return None
        # Return the highest-amplitude peak in the window.
        best_local = int(local_peaks[np.argmax(segment[local_peaks])])
        return lo + best_local

    def _choose_s2_spectral(t_expected_sec: float, search_half_sec: float) -> Optional[Tuple[int, float]]:
        """
        Search for S2 using the spectral S2 template from insert_spectrum_ctx.
        Falls back on spectrum_s1_search_envelope_index called with mu_s2_db.

        Returns (envelope_index, score) or None.

        LIMITATION: the S2 template is built from Pass 2 paired S2 peaks. If Pass 2
        made systematic labeling errors, those may bias the template, causing spectral
        search to confirm the same mistakes (confirmation-bias risk). Only call this
        after _find_sensitive_peaks_near has already failed.
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
            result = spectrum_s1_search_envelope_index(
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

    # ---------------------------------------------------------------------
    # 1) Build initial S1/S2 events from the current S1 list + BPM prior
    # ---------------------------------------------------------------------
    min_sep_samples = int(float(params.get("min_peak_distance_sec", 0.10)) * sample_rate)
    s1_list = [int(x) for x in peaks_out.tolist()]
    s1_list = sorted(list(dict.fromkeys([x for x in s1_list if 0 <= x < n_samples])))

    # Spectral template + bandpass audio for missed-beat insertion when no raw peak exists.
    insert_spectrum_ctx: Optional[Dict[str, Any]] = None
    if (
        bool(params.get("pass3_insert_use_spectrum", True))
        and wav_file_path
        and os.path.isfile(wav_file_path)
    ):
        try:
            insert_spectrum_ctx = prepare_pass3_s1_insert_context(
                wav_file_path,
                pc,
                sample_rate,
                audio_envelope,
                params,
            )
            if insert_spectrum_ctx is not None:
                logging.info(
                    "Pass 3: spectral S1 insert context ready (n_s1_template=%s, sr=%s).",
                    insert_spectrum_ctx.get("n_s1_template"),
                    insert_spectrum_ctx.get("full_sr"),
                )
        except Exception as e:
            logging.warning("Pass 3: could not build spectral insert context: %s", e)
            insert_spectrum_ctx = None

    s2_events: List[int] = []
    for i in range(len(s1_list) - 1):
        s1 = int(s1_list[i])
        s1_next = int(s1_list[i + 1])
        if s1_next <= s1:
            s2_events.append(int(s1))
            continue
        t_s1 = s1 / float(sample_rate)
        bpm = _bpm_at_time(t_s1)
        intervals = calculate_bpm_intervals(bpm, params)
        s1_s2_nominal = float(intervals.get("s1_s2_nominal", 0.30))
        s2_pred = int(round(s1 + s1_s2_nominal * sample_rate))
        s2 = _choose_s2_near(s1, s1_next, s2_pred, snap_half)
        s2_events.append(int(s2))

    # Snapshot "before correction" state boundaries for HTML visualization.
    state_boundaries_before: List[Tuple[int, int, str, Dict[str, Any]]] = []
    for i in range(len(s1_list) - 1):
        s1 = int(s1_list[i])
        s1_next = int(s1_list[i + 1])
        if s1_next <= s1:
            continue
        s2 = int(s2_events[i]) if i < len(s2_events) else s1
        s2 = int(max(s1 + 1, min(s2, s1_next - 1)))

        s1_start = max(0, s1 - s1_half)
        s1_end = min(n_samples, s1 + s1_half + 1)
        s2_start = max(0, s2 - s2_half)
        s2_end = min(n_samples, s2 + s2_half + 1)

        if s2_start < s1_end:
            mid = (s1_end + s2_start) // 2
            s1_end = max(s1_start + 1, min(s1_end, mid))
            s2_start = max(s2_start, s1_end)
        if s2_end >= s1_next:
            s2_end = min(s2_end, s1_next)

        if s1_end > s1_start:
            state_boundaries_before.append((s1_start, s1_end, "S1", {"s1": s1}))
        if s2_start > s1_end:
            state_boundaries_before.append((s1_end, s2_start, "systole", {"s1": s1, "s2": s2}))
        if s2_end > s2_start:
            state_boundaries_before.append((s2_start, s2_end, "S2", {"s2": s2}))
        if s1_next > s2_end:
            state_boundaries_before.append((s2_end, s1_next, "diastole", {"s2": s2, "s1_next": s1_next}))

    # ---------------------------------------------------------------------
    # 2) Correction loop: fix implausible systole/diastole and insert missing S1s
    # ---------------------------------------------------------------------
    corrections: List[Dict[str, Any]] = []
    cycle_diagnostics: List[Dict[str, Any]] = []

    max_iters = int(params.get("pass3_correction_max_iters", 6))
    enable_insert = bool(params.get("pass3_enable_insert_missing_s1", True))
    rr_too_long_frac = float(params.get("pass3_rr_too_long_frac", 1.7))
    max_fill_gap_sec = float(params.get("pass3_gap_fill_max_duration_sec", 5.0))
    s1_search_window_ms = float(params.get("pass3_insert_s1_search_window_ms", 180.0))
    s1_search_half = max(1, int(round(0.5 * s1_search_window_ms * sample_rate / 1000.0)))

    # When systole is implausible, try re-snapping with a wider window.
    resnap_window_ms = float(params.get("pass3_resnap_s2_window_ms", 220.0))
    resnap_half = max(1, int(round(0.5 * resnap_window_ms * sample_rate / 1000.0)))
    systole_slack = float(params.get("pass3_systole_slack_frac", 0.15))
    diastole_slack = float(params.get("pass3_diastole_slack_frac", 0.20))

    # Pass C params.
    enable_phase_correction = bool(params.get("pass3_enable_phase_correction", True))
    phase_min_score_delta = float(params.get("pass3_phase_min_score_delta", 0.15))
    local_peak_window_ms = float(params.get("pass3_local_peak_window_ms", 160.0))
    local_peak_window_samples = max(1, int(round(0.5 * local_peak_window_ms * sample_rate / 1000.0)))
    local_peak_sensitivity = float(params.get("pass3_local_peak_sensitivity_factor", 0.6))

    for _iter in range(max_iters):
        changed = False
        cycle_diagnostics.clear()

        # Ensure s2_events length matches cycles
        if len(s2_events) != max(0, len(s1_list) - 1):
            s2_events = s2_events[: max(0, len(s1_list) - 1)]
            while len(s2_events) < max(0, len(s1_list) - 1):
                s2_events.append(int(s1_list[len(s2_events)]))

        # --- Pass A: re-snap S2 for timing plausibility ---
        for i in range(len(s1_list) - 1):
            s1 = int(s1_list[i])
            s1_next = int(s1_list[i + 1])
            if s1_next <= s1:
                continue
            s2 = int(s2_events[i])
            s2 = int(max(s1 + 1, min(s2, s1_next - 1)))

            t_s1 = s1 / float(sample_rate)
            bpm = _bpm_at_time(t_s1)
            intervals = calculate_bpm_intervals(bpm, params)
            s1_s2_min = float(intervals.get("s1_s2_min", 0.12))
            s1_s2_nominal = float(intervals.get("s1_s2_nominal", 0.30))
            s1_s2_max = float(intervals.get("s1_s2_max", 0.40))

            systole = (s2 - s1) / float(sample_rate)
            rr = (s1_next - s1) / float(sample_rate)
            diastole = rr - systole
            expected_rr = float(intervals.get("rr_interval", 60.0 / bpm if bpm > 0 else 0.75))
            diastole_nominal = float(intervals.get("s2_s1_nominal", max(0.0, expected_rr - s1_s2_nominal)))
            diastole_min = float(intervals.get("diastole_min", 0.08))
            diastole_max = float(intervals.get("diastole_max", diastole_nominal * 2.0))

            too_short = systole < (1.0 - systole_slack) * s1_s2_min
            too_long = systole > (1.0 + systole_slack) * s1_s2_max
            far_from_nominal = abs(systole - s1_s2_nominal) > max(0.12, 0.5 * (s1_s2_max - s1_s2_min))
            diastole_too_short = diastole < (1.0 - diastole_slack) * diastole_min

            s1_min_evt = float(intervals.get("s1_min", 0.010))
            s1_nominal_evt = float(intervals.get("s1_nominal", 0.040))
            s1_max_evt = float(intervals.get("s1_max", 0.080))
            s2_min_evt = float(intervals.get("s2_min", 0.010))
            s2_nominal_evt = float(intervals.get("s2_nominal", 0.030))
            s2_max_evt = float(intervals.get("s2_max", 0.060))
            min_feasible_cycle = float(intervals.get("min_feasible_cycle", s1_min_evt + s1_s2_min + s2_min_evt + diastole_min))

            diag = {
                "i": int(i),
                "s1": int(s1),
                "s2": int(s2),
                "s1_next": int(s1_next),
                "bpm": float(bpm),
                "rr_sec": float(rr),
                "systole_sec": float(systole),
                "diastole_sec": float(diastole),
                "expected_rr_sec": float(expected_rr),
                "diastole_nominal_sec": float(diastole_nominal),
                "diastole_min_sec": float(diastole_min),
                "diastole_max_sec": float(diastole_max),
                "s1_min_sec": float(s1_min_evt),
                "s1_nominal_sec": float(s1_nominal_evt),
                "s1_max_sec": float(s1_max_evt),
                "s2_min_sec": float(s2_min_evt),
                "s2_nominal_sec": float(s2_nominal_evt),
                "s2_max_sec": float(s2_max_evt),
                "min_feasible_cycle_sec": float(min_feasible_cycle),
                "s1_s2_min": float(s1_s2_min),
                "s1_s2_nominal": float(s1_s2_nominal),
                "s1_s2_max": float(s1_s2_max),
                "flags": {
                    "systole_too_short": bool(too_short),
                    "systole_too_long": bool(too_long),
                    "systole_far_from_nominal": bool(far_from_nominal),
                    "diastole_too_short": bool(diastole_too_short),
                },
            }
            cycle_diagnostics.append(diag)

            if (too_short or too_long or far_from_nominal) and len(all_raw_peaks) > 0:
                s2_pred = int(round(s1 + s1_s2_nominal * sample_rate))
                new_s2 = _choose_s2_near(s1, s1_next, s2_pred, resnap_half)
                new_s2 = int(max(s1 + 1, min(new_s2, s1_next - 1)))
                if new_s2 != s2:
                    corrections.append(
                        {
                            "type": "resnap_s2",
                            "cycle": int(i),
                            "s1": int(s1),
                            "old_s2": int(s2),
                            "new_s2": int(new_s2),
                            "s2_pred": int(s2_pred),
                        }
                    )
                    s2_events[i] = int(new_s2)
                    changed = True

        # --- Pass B: insert missing S1 when RR is implausibly long vs BPM prior ---
        if enable_insert and len(s1_list) >= 2:
            inserted_any = False
            can_use_peaks = len(all_raw_peaks) > 0
            for i in range(len(s1_list) - 1):
                s1 = int(s1_list[i])
                s1_next = int(s1_list[i + 1])
                rr = (s1_next - s1) / float(sample_rate)
                if rr <= 0 or rr > max_fill_gap_sec:
                    continue
                t_s1 = s1 / float(sample_rate)
                bpm = _bpm_at_time(t_s1)
                intervals = calculate_bpm_intervals(bpm, params)
                expected_rr = float(intervals.get("rr_interval", 60.0 / bpm if bpm > 0 else 0.75))
                if expected_rr <= 0:
                    continue
                if rr <= rr_too_long_frac * expected_rr:
                    continue

                # How many beats likely missing in this span?
                n_missing = max(0, int(round(rr / expected_rr) - 1))
                if n_missing < 1:
                    continue

                # Insert at most 1 per iteration to keep behavior stable.
                k = 0
                t_expected = (s1 / float(sample_rate)) + (k + 1) * (rr / (n_missing + 1))
                cand: Optional[int] = None
                insert_method = "raw_peak"
                spectrum_score: Optional[float] = None
                if can_use_peaks:
                    cand = _choose_s1_near(t_expected, s1_search_half, min_sep_samples)
                if cand is None and insert_spectrum_ctx is not None:
                    half_sec = float(s1_search_window_ms) / 2000.0
                    sp = spectrum_s1_search_envelope_index(
                        insert_spectrum_ctx["bandpass_audio"],
                        int(insert_spectrum_ctx["full_sr"]),
                        t_expected,
                        half_sec,
                        insert_spectrum_ctx["mu_s1_db"],
                        insert_spectrum_ctx["freqs"],
                        int(insert_spectrum_ctx["n_fft"]),
                        int(insert_spectrum_ctx["half_samples"]),
                        sample_rate,
                        n_samples,
                        params,
                    )
                    if sp is not None:
                        cand_ix, spectrum_score = sp
                        if _insert_spectrum_envelope_ok(cand_ix):
                            cand = int(cand_ix)
                            insert_method = "spectrum_s1"
                if cand is None:
                    continue
                # Reject if too close to adjacent existing S1
                if (cand - s1) < min_sep_samples or (s1_next - cand) < min_sep_samples:
                    continue
                s1_list.append(int(cand))
                s1_list = sorted(list(dict.fromkeys(s1_list)))
                corr: Dict[str, Any] = {
                    "type": "insert_s1",
                    "between_cycle": int(i),
                    "s1_prev": int(s1),
                    "s1_next": int(s1_next),
                    "inserted_s1": int(cand),
                    "t_expected_sec": float(t_expected),
                    "expected_rr_sec": float(expected_rr),
                    "rr_sec": float(rr),
                    "method": insert_method,
                }
                if spectrum_score is not None:
                    corr["spectrum_score"] = float(spectrum_score)
                corrections.append(corr)
                inserted_any = True
                changed = True
                break

            if inserted_any:
                # Rebuild s2_events after insertion
                s2_events = []
                for j in range(len(s1_list) - 1):
                    a = int(s1_list[j])
                    b = int(s1_list[j + 1])
                    if b <= a:
                        s2_events.append(int(a))
                        continue
                    t_a = a / float(sample_rate)
                    bpm_a = _bpm_at_time(t_a)
                    intervals_a = calculate_bpm_intervals(bpm_a, params)
                    s1_s2_nominal_a = float(intervals_a.get("s1_s2_nominal", 0.30))
                    s2_pred_a = int(round(a + s1_s2_nominal_a * sample_rate))
                    s2_a = _choose_s2_near(a, b, s2_pred_a, snap_half)
                    s2_events.append(int(s2_a))

        # --- Pass C: phase-shift cascade correction ---
        # Handles two failure modes that Pass A/B cannot fix because they never question
        # whether a peak in s1_list is correctly labeled as S1.
        #
        # C.1 Remove false S1 (RR + systole both too short):
        #   A noise peak promoted to S1 creates a very short cycle. Demote it.
        # C.2 Demote S1_next to S2 (diastole too short):
        #   The "next S1" arrived too soon after S2 — it's likely the real S2 of this
        #   cycle. Re-seat it as S2 and search for the real next S1 after it.
        # C.3 Find faint S2 (systole too long, Pass A already failed):
        #   Real S2 was missed by peak detection. Try local sensitive re-detection then
        #   spectral fingerprint matching.
        #
        # One fix per outer iteration; cascade is handled by the outer max_iters loop.
        if enable_phase_correction and not changed and len(cycle_diagnostics) > 0:

            # --- Pass C.1: Remove false S1 ---
            for diag in cycle_diagnostics:
                i = diag["i"]
                if i + 1 >= len(s1_list):
                    continue
                s1 = diag["s1"]
                s1_next = diag["s1_next"]
                systole = diag["systole_sec"]
                diastole = diag["diastole_sec"]
                diastole_min_c = diag.get("diastole_min_sec", 0.0)
                s1_s2_min_c = diag["s1_s2_min"]
                # Trigger: both systole AND diastole are implausibly short (both states squeezed).
                # This means a false S1 has been sandwiched into the cycle, compressing both.
                if not (systole < s1_s2_min_c and diastole < diastole_min_c):
                    continue
                # Check: s1_next is the suspect false S1 — should look more like noise.
                suspect = int(s1_next)
                entry = pc.get(suspect) or {}
                ls = entry.get("label_scores") if isinstance(entry, dict) else None
                if not isinstance(ls, dict):
                    continue
                noise_score = float(ls.get("noise", 0.0))
                s1_score = float(ls.get("S1", 0.0))
                if noise_score - s1_score < phase_min_score_delta:
                    continue
                # Feasibility gate: the merged cycle (s1 → original s1_list[i+2]) must
                # be large enough to hold a minimum valid S1+systole+S2+diastole.
                min_feasible = diag.get("min_feasible_cycle_sec", 0.0)
                merged_next = int(s1_list[i + 2]) if i + 2 < len(s1_list) else n_samples
                merged_span_sec = (merged_next - int(s1)) / float(sample_rate)
                if min_feasible > 0 and merged_span_sec < min_feasible:
                    continue
                # Commit: remove the false S1 from s1_list.
                s1_list = [p for p in s1_list if p != suspect]
                # Rebuild s2_events from scratch.
                s2_events = []
                for j in range(len(s1_list) - 1):
                    a = int(s1_list[j])
                    b = int(s1_list[j + 1])
                    if b <= a:
                        s2_events.append(int(a))
                        continue
                    t_a = a / float(sample_rate)
                    bpm_a = _bpm_at_time(t_a)
                    ivs_a = calculate_bpm_intervals(bpm_a, params)
                    s2_pred_a = int(round(a + float(ivs_a.get("s1_s2_nominal", 0.30)) * sample_rate))
                    s2_events.append(_choose_s2_near(a, b, s2_pred_a, snap_half))
                corrections.append({
                    "type": "remove_false_s1",
                    "cycle": int(i),
                    "s1": int(s1),
                    "removed_s1": int(suspect),
                    "systole_sec": float(systole),
                    "diastole_sec": float(diastole),
                    "diastole_min_sec": float(diastole_min_c),
                    "noise_score": float(noise_score),
                    "s1_score": float(s1_score),
                })
                logging.info(
                    "Pass 3 C.1: removed false S1 at sample %d (cycle %d, systole=%.3fs, diastole=%.3fs/min=%.3fs).",
                    suspect, i, systole, diastole, diastole_min_c,
                )
                changed = True
                break

            # --- Pass C.2: Demote S1_next to S2 (diastole too short) ---
            if not changed:
                for diag in cycle_diagnostics:
                    i = diag["i"]
                    if i + 1 >= len(s1_list):
                        continue
                    if not diag["flags"].get("diastole_too_short", False):
                        continue
                    s1 = diag["s1"]
                    s1_next = diag["s1_next"]
                    # Check: s1_next looks more like S2 than S1.
                    entry = pc.get(int(s1_next)) or {}
                    ls = entry.get("label_scores") if isinstance(entry, dict) else None
                    if not isinstance(ls, dict):
                        continue
                    s2_score = float(ls.get("S2", 0.0))
                    s1_score = float(ls.get("S1", 0.0))
                    if s2_score - s1_score < phase_min_score_delta:
                        continue
                    # Candidate new S2 for cycle i is s1_next itself.
                    new_s2 = int(s1_next)
                    # Now search for a new real S1 after new_s2.
                    # The new S1 must arrive before s1_list[i+2] (if it exists).
                    upper_bound = int(s1_list[i + 2]) if i + 2 < len(s1_list) else n_samples
                    bpm_here = _bpm_at_time(int(s1) / float(sample_rate))
                    ivs_here = calculate_bpm_intervals(bpm_here, params)
                    s2_min_here = float(ivs_here.get("s2_min", 0.010))
                    diastole_min_here = float(ivs_here.get("diastole_min", 0.08))
                    # New S1 must be at least s2_min past the new S2 (can't overlap the S2 event).
                    earliest_new_s1 = new_s2 + max(1, int(s2_min_here * sample_rate))
                    # Expected next S1 is ~diastole after new_s2.
                    expected_diastole_here = max(diastole_min_here, float(ivs_here.get("s2_s1_nominal", 0.35)))
                    t_new_s1 = earliest_new_s1 / float(sample_rate) + max(0.0, expected_diastole_here - s2_min_here)
                    new_s1_cand: Optional[int] = None
                    # Priority 1: raw peak near expected time.
                    new_s1_cand = _choose_s1_near(t_new_s1, s1_search_half, min_sep_samples)
                    if new_s1_cand is not None and (new_s1_cand < earliest_new_s1 or new_s1_cand >= upper_bound):
                        new_s1_cand = None
                    # Priority 2: local sensitive re-detection.
                    if new_s1_cand is None:
                        sens_cand = _find_sensitive_peaks_near(t_new_s1, local_peak_window_samples, local_peak_sensitivity)
                        if sens_cand is not None and earliest_new_s1 <= sens_cand < upper_bound:
                            new_s1_cand = sens_cand
                    if new_s1_cand is None:
                        continue
                    if new_s1_cand < earliest_new_s1 or new_s1_cand >= upper_bound:
                        continue
                    # Commit: update s2_events[i] and insert new S1.
                    s2_events[i] = new_s2
                    # Replace old s1_next with new_s1_cand (remove old, insert new).
                    s1_list = [p for p in s1_list if p != int(s1_next)]
                    s1_list.append(new_s1_cand)
                    s1_list = sorted(list(dict.fromkeys(s1_list)))
                    # Rebuild s2_events fully to stay consistent.
                    s2_events = []
                    for j in range(len(s1_list) - 1):
                        a = int(s1_list[j])
                        b = int(s1_list[j + 1])
                        if b <= a:
                            s2_events.append(int(a))
                            continue
                        t_a = a / float(sample_rate)
                        bpm_a = _bpm_at_time(t_a)
                        ivs_a = calculate_bpm_intervals(bpm_a, params)
                        s2_pred_a = int(round(a + float(ivs_a.get("s1_s2_nominal", 0.30)) * sample_rate))
                        s2_events.append(_choose_s2_near(a, b, s2_pred_a, snap_half))
                    corrections.append({
                        "type": "flip_demote_s1",
                        "cycle": int(i),
                        "s1": int(s1),
                        "old_s1_next": int(s1_next),
                        "new_s2_for_cycle": int(new_s2),
                        "new_s1_next": int(new_s1_cand),
                        "s2_score": float(s2_score),
                        "s1_score": float(s1_score),
                    })
                    logging.info(
                        "Pass 3 C.2: flipped S1@%d→S2, new S1 at %d (cycle %d, diastole was %.3fs).",
                        s1_next, new_s1_cand, i, diag["diastole_sec"],
                    )
                    changed = True
                    break

            # --- Pass C.3: Find faint S2 (systole too long, Pass A already failed) ---
            if not changed:
                for diag in cycle_diagnostics:
                    i = diag["i"]
                    if not diag["flags"].get("systole_too_long", False):
                        continue
                    s1 = diag["s1"]
                    s1_next = diag["s1_next"]
                    bpm_c = diag["bpm"]
                    ivs_c = calculate_bpm_intervals(bpm_c, params)
                    s1_s2_nominal_c = float(ivs_c.get("s1_s2_nominal", 0.30))
                    expected_diastole_c = max(0.0, float(ivs_c.get("rr_interval", 0.75)) - s1_s2_nominal_c)
                    t_s2_pred = int(s1) / float(sample_rate) + s1_s2_nominal_c
                    search_half_sec = local_peak_window_ms / 2000.0

                    new_s2: Optional[int] = None
                    method_used: Optional[str] = None
                    spectral_score: Optional[float] = None

                    # Priority 1: local sensitive re-detection.
                    sens_cand = _find_sensitive_peaks_near(t_s2_pred, local_peak_window_samples, local_peak_sensitivity)
                    if sens_cand is not None and int(s1) < sens_cand < int(s1_next):
                        new_s2 = sens_cand
                        method_used = "sensitive_peak"

                    # Priority 2: spectral S2 search (label-dependent, confirmation-bias risk noted).
                    if new_s2 is None:
                        sp_result = _choose_s2_spectral(t_s2_pred, search_half_sec)
                        if sp_result is not None:
                            sp_idx, sp_score = sp_result
                            if int(s1) < sp_idx < int(s1_next):
                                new_s2 = sp_idx
                                spectral_score = sp_score
                                method_used = "spectral_s2"

                    if new_s2 is None:
                        continue

                    # Validate: placing S2 here must produce a plausible diastole
                    # AND leave room for the S2 event itself before the diastole starts.
                    new_systole = (new_s2 - int(s1)) / float(sample_rate)
                    new_diastole = (int(s1_next) - new_s2) / float(sample_rate)
                    s2_min_c = float(ivs_c.get("s2_min", 0.010))
                    diastole_min_c = float(ivs_c.get("diastole_min", 0.08))
                    if new_systole < float(ivs_c.get("s1_s2_min", 0.12)):
                        continue
                    # Must fit at least the S2 event window + minimum diastole after new_s2.
                    if new_diastole < s2_min_c + diastole_min_c:
                        continue

                    # Commit.
                    s2_events[i] = new_s2
                    corr: Dict[str, Any] = {
                        "type": method_used,
                        "cycle": int(i),
                        "s1": int(s1),
                        "new_s2": int(new_s2),
                        "t_s2_pred_sec": float(t_s2_pred),
                        "new_systole_sec": float(new_systole),
                        "new_diastole_sec": float(new_diastole),
                    }
                    if spectral_score is not None:
                        corr["spectral_score"] = float(spectral_score)
                    corrections.append(corr)
                    logging.info(
                        "Pass 3 C.3: placed faint S2 at sample %d via %s (cycle %d, systole %.3fs→%.3fs).",
                        new_s2, method_used, i, diag["systole_sec"], new_systole,
                    )
                    changed = True
                    break

        if not changed:
            break

    # Final peaks after pass 3 correction (still S1 list).
    peaks_out = np.asarray(s1_list, dtype=np.int64)

    # ---------------------------------------------------------------------
    # Pre-build lookups used by the debug reasoning attached to each segment.
    # ---------------------------------------------------------------------
    _sr_f = float(sample_rate)

    # before_s2_by_s1[s1] = s2 sample before any Pass 3 corrections.
    # before_s1next_by_s1[s1] = s1_next sample before any Pass 3 corrections.
    _before_s2_by_s1: Dict[int, int] = {}
    _s2_to_s1_before: Dict[int, int] = {}
    _before_s1next_by_s1: Dict[int, int] = {}
    for _bs, _be, _bst, _bm in state_boundaries_before:
        if _bst == "systole":
            _s1k = _bm.get("s1"); _s2k = _bm.get("s2")
            if _s1k is not None and _s2k is not None:
                _before_s2_by_s1[int(_s1k)] = int(_s2k)
                _s2_to_s1_before[int(_s2k)] = int(_s1k)
        elif _bst == "diastole":
            _s2k = _bm.get("s2"); _s1nk = _bm.get("s1_next")
            if _s2k is not None and _s1nk is not None:
                _par = _s2_to_s1_before.get(int(_s2k))
                if _par is not None:
                    _before_s1next_by_s1[_par] = int(_s1nk)

    # corrections keyed by the s1 sample of the affected cycle.
    _corrs_by_s1: Dict[int, List[Dict[str, Any]]] = {}
    for _c in corrections:
        _ck = _c.get("s1") if _c.get("s1") is not None else _c.get("s1_prev")
        if _ck is not None:
            _corrs_by_s1.setdefault(int(_ck), []).append(_c)

    # Sorted (time_sec, correction) for cascade attribution.
    _all_corr_sorted = sorted(
        [(float(_c.get("s1", _c.get("s1_prev", 0))) / _sr_f, _c) for _c in corrections],
        key=lambda x: x[0],
    )

    def _fmt_corr_note(c: Dict[str, Any]) -> str:
        """Return a ⚠ warning line describing the direct correction applied."""
        ctype = c.get("type", "")
        if ctype == "resnap_s2":
            new_sys = round((c.get("new_s2", 0) - c.get("s1", 0)) / _sr_f * 1000)
            return f"⚠ Systole out of range — S2 repositioned, systole now {new_sys}ms"
        if ctype == "insert_s1":
            return f"⚠ RR gap too long — missing beat inserted at {c.get('t_expected_sec', 0):.2f}s"
        if ctype == "remove_false_s1":
            return f"⚠ Both systole+diastole too short — false beat at {c.get('removed_s1', 0) / _sr_f:.2f}s removed"
        if ctype == "flip_demote_s1":
            old_t = c.get("old_s1_next", 0) / _sr_f
            new_t = c.get("new_s1_next", 0) / _sr_f
            return f"⚠ Diastole too short — beat at {old_t:.2f}s re-labeled as S2, new S1 at {new_t:.2f}s"
        if ctype in ("sensitive_peak", "spectral_s2"):
            new_sys = round(c.get("new_systole_sec", 0) * 1000)
            method = "sensitive peak detection" if ctype == "sensitive_peak" else "spectral fingerprint"
            return f"⚠ Systole too long — faint S2 found via {method}, systole now {new_sys}ms"
        return f"⚠ Correction applied ({ctype})"

    def _fmt_cascade_note(src: Dict[str, Any]) -> str:
        """Return a ℹ info line describing an upstream correction that shifted this segment."""
        ctype = src.get("type", "")
        src_t = src.get("s1", src.get("s1_prev", 0)) / _sr_f
        label = {
            "resnap_s2": "S2 resnap",
            "insert_s1": "beat insertion",
            "remove_false_s1": "false beat removal",
            "flip_demote_s1": "S1/S2 flip correction",
            "sensitive_peak": "faint-S2 detection",
            "spectral_s2": "spectral S2 detection",
        }.get(ctype, ctype)
        return f"ℹ Shifted by {label} at {src_t:.2f}s; duration still within expected range"

    # ---------------------------------------------------------------------
    # 3) Paint the final state timeline from corrected events
    # ---------------------------------------------------------------------
    state_labels[:] = STATE_DIASTOLE
    state_boundaries.clear()
    s2_events_final: List[int] = []

    _SHIFT_THRESH_SAMP = int(0.020 * sample_rate)  # 20 ms shift threshold for cascade note

    for i in range(len(peaks_out) - 1):
        s1 = int(peaks_out[i])
        s1_next = int(peaks_out[i + 1])
        if s1_next <= s1:
            continue

        # Reuse corrected s2_events when available; otherwise predict.
        if i < len(s2_events):
            s2 = int(s2_events[i])
        else:
            t_s1 = s1 / _sr_f
            bpm = _bpm_at_time(t_s1)
            intervals = calculate_bpm_intervals(bpm, params)
            s1_s2_nominal = float(intervals.get("s1_s2_nominal", 0.30))
            s2_pred = int(round(s1 + s1_s2_nominal * _sr_f))
            s2 = _choose_s2_near(s1, s1_next, s2_pred, snap_half)

        s2 = int(max(s1 + 1, min(s2, s1_next - 1)))
        s2_events_final.append(int(s2))

        # Define event windows and paint states.
        s1_start = max(0, s1 - s1_half)
        s1_end = min(n_samples, s1 + s1_half + 1)
        s2_start = max(0, s2 - s2_half)
        s2_end = min(n_samples, s2 + s2_half + 1)

        # Ensure ordering (no negative-length spans).
        if s2_start < s1_end:
            mid = (s1_end + s2_start) // 2
            s1_end = max(s1_start + 1, min(s1_end, mid))
            s2_start = max(s2_start, s1_end)
        if s2_end >= s1_next:
            s2_end = min(s2_end, s1_next)

        # ---- Build debug reasoning payload for each state in this cycle ----
        _t_s1_r = s1 / _sr_f
        _ivs_r = calculate_bpm_intervals(_bpm_at_time(_t_s1_r), params)

        _exp_s1_ms  = round(_ivs_r.get("s1_nominal",    0.040) * 1000)
        _exp_sys_ms = round(_ivs_r.get("s1_s2_nominal", 0.300) * 1000)
        _exp_s2_ms  = round(_ivs_r.get("s2_nominal",    0.030) * 1000)
        _exp_dia_ms = round(_ivs_r.get("s2_s1_nominal", 0.400) * 1000)

        _meas_s1_ms  = round((s1_end  - s1_start) / _sr_f * 1000) if s1_end  > s1_start else 0
        _meas_sys_ms = round((s2_start - s1_end)  / _sr_f * 1000) if s2_start > s1_end   else 0
        _meas_s2_ms  = round((s2_end  - s2_start) / _sr_f * 1000) if s2_end  > s2_start  else 0
        _meas_dia_ms = round((s1_next - s2_end)   / _sr_f * 1000) if s1_next > s2_end    else 0

        # Before-correction S2 and S1_next positions for this cycle.
        _bef_s2    = _before_s2_by_s1.get(s1)
        _bef_s1nxt = _before_s1next_by_s1.get(s1)
        _bef_sys_ms = round((_bef_s2 - s1) / _sr_f * 1000)        if _bef_s2    is not None else None
        _bef_dia_ms = round((_bef_s1nxt - _bef_s2) / _sr_f * 1000) if (_bef_s2 is not None and _bef_s1nxt is not None) else None

        # Direct corrections recorded for this s1.
        _direct = _corrs_by_s1.get(s1, [])

        # Most recent correction strictly before this cycle (cascade source).
        _cascade_src = None
        for _ct, _cc in _all_corr_sorted:
            if _ct >= _t_s1_r - 0.010:
                break
            if _cc not in _direct:
                _cascade_src = _cc

        # Per-state notes.
        _s1_notes: List[str] = [_fmt_corr_note(c) for c in _direct if c.get("type") == "remove_false_s1"]
        _sys_notes: List[str] = [_fmt_corr_note(c) for c in _direct
                                  if c.get("type") in ("resnap_s2", "flip_demote_s1", "sensitive_peak", "spectral_s2")]
        _s2_notes:  List[str] = [_fmt_corr_note(c) for c in _direct
                                  if c.get("type") in ("resnap_s2", "sensitive_peak", "spectral_s2")]
        _dia_notes: List[str] = [_fmt_corr_note(c) for c in _direct
                                  if c.get("type") in ("insert_s1", "flip_demote_s1")]

        # Cascade shift notes for states that moved but have no direct correction.
        if _cascade_src:
            if not _sys_notes and _bef_s2 is not None and abs(_bef_s2 - s2) > _SHIFT_THRESH_SAMP:
                _sys_notes.append(_fmt_cascade_note(_cascade_src))
            if not _dia_notes and _bef_s1nxt is not None and abs(_bef_s1nxt - s1_next) > _SHIFT_THRESH_SAMP:
                _dia_notes.append(_fmt_cascade_note(_cascade_src))

        _reasoning: Dict[str, Any] = {
            "S1":       {"expected_ms": _exp_s1_ms,  "measured_ms": _meas_s1_ms,  "notes": _s1_notes},
            "systole":  {"expected_ms": _exp_sys_ms, "measured_ms": _meas_sys_ms, "notes": _sys_notes},
            "S2":       {"expected_ms": _exp_s2_ms,  "measured_ms": _meas_s2_ms,  "notes": _s2_notes},
            "diastole": {"expected_ms": _exp_dia_ms, "measured_ms": _meas_dia_ms, "notes": _dia_notes},
        }

        if s1_end > s1_start:
            state_labels[s1_start:s1_end] = STATE_S1
            state_boundaries.append((s1_start, s1_end, "S1", {"s1": s1, "reasoning": _reasoning["S1"]}))
        if s2_start > s1_end:
            state_labels[s1_end:s2_start] = STATE_SYSTOLE
            state_boundaries.append((s1_end, s2_start, "systole", {"s1": s1, "s2": s2, "reasoning": _reasoning["systole"]}))
        if s2_end > s2_start:
            state_labels[s2_start:s2_end] = STATE_S2
            state_boundaries.append((s2_start, s2_end, "S2", {"s2": s2, "reasoning": _reasoning["S2"]}))
        if s1_next > s2_end:
            state_labels[s2_end:s1_next] = STATE_DIASTOLE
            state_boundaries.append((s2_end, s1_next, "diastole", {"s2": s2, "s1_next": s1_next, "reasoning": _reasoning["diastole"]}))

    analysis_data["pass3_state_labels"] = state_labels
    analysis_data["pass3_state_labels_encoding"] = {
        "S1": int(STATE_S1),
        "systole": int(STATE_SYSTOLE),
        "S2": int(STATE_S2),
        "diastole": int(STATE_DIASTOLE),
    }
    analysis_data["pass3_state_boundaries"] = state_boundaries
    analysis_data["pass3_state_boundaries_before"] = state_boundaries_before
    analysis_data["pass3_s2_events"] = np.asarray(s2_events_final, dtype=np.int64)
    analysis_data["pass3_corrections"] = corrections
    analysis_data["pass3_cycle_diagnostics"] = cycle_diagnostics

    logging.info(
        "Pass 3: corrected peaks=%d, generated state timeline (n=%d samples; %d cycles; %d corrections).",
        int(len(peaks_out)),
        n_samples,
        max(0, len(peaks_out) - 1),
        int(len(corrections)),
    )

    return peaks_out, analysis_data


def _calculate_metrics_from_peaks(peaks: np.ndarray, sample_rate: int, params: Dict) -> Dict:
    """Calculates BPM, HRV, and slope metrics from a peak list. Used by any pass (pass 2, pass 3, etc.)."""
    metrics = {}
    metrics['smoothed_bpm'], metrics['bpm_times'], metrics['instant_bpm'] = calculate_bpm_series(peaks, sample_rate, params)
    metrics['major_inclines'] = find_major_hr_inclines(metrics['smoothed_bpm'])
    metrics['major_declines'] = find_major_hr_declines(metrics['smoothed_bpm'])
    metrics['hrr_stats'] = calculate_hrr(metrics['smoothed_bpm'])
    metrics['peak_recovery_stats'] = find_peak_recovery_rate(metrics['smoothed_bpm'])
    metrics['peak_exertion_stats'] = find_peak_exertion_rate(metrics['smoothed_bpm'])
    metrics['windowed_hrv_df'] = calculate_windowed_hrv(peaks, sample_rate, params)
    if params.get("enable_hrv_frequency_domain", False):
        metrics['hrv_global_freq'] = calculate_global_hrv_frequency(peaks, sample_rate, params)
    else:
        metrics['hrv_global_freq'] = None

    hrv_summary_stats = {}
    if not metrics['smoothed_bpm'].empty:
        hrv_summary_stats['avg_bpm'] = metrics['smoothed_bpm'].mean()
        hrv_summary_stats['min_bpm'] = metrics['smoothed_bpm'].min()
        hrv_summary_stats['max_bpm'] = metrics['smoothed_bpm'].max()
    if not metrics['windowed_hrv_df'].empty:
        hrv_summary_stats['avg_rmssdc'] = metrics['windowed_hrv_df']['rmssdc'].mean()
        hrv_summary_stats['avg_sdnn'] = metrics['windowed_hrv_df']['sdnn'].mean()
        if params.get("enable_hrv_frequency_domain", False) and "lf_hf_ratio" in metrics['windowed_hrv_df'].columns:
            wdf = metrics['windowed_hrv_df']
            hrv_summary_stats['avg_lf_power'] = wdf['lf_power'].mean()
            hrv_summary_stats['avg_hf_power'] = wdf['hf_power'].mean()
            avg_lf_hf = wdf['lf_hf_ratio'].mean()
            hrv_summary_stats['avg_lf_hf_ratio'] = avg_lf_hf
            if np.isnan(avg_lf_hf):
                valid = wdf['lf_hf_ratio'].notna().sum()
                logging.warning(
                    "Avg. LF/HF (windowed) is NaN: %d/%d windows had valid lf_hf_ratio. See earlier logs for Lomb-Scargle failures.",
                    int(valid), len(wdf),
                )
    if metrics.get('hrv_global_freq') is not None:
        hrv_summary_stats['global_freq'] = metrics['hrv_global_freq']
    metrics['hrv_summary'] = hrv_summary_stats

    return metrics


def _apply_pass3_state_timeline_bpm(
    metrics: Dict[str, Any],
    analysis_data: Dict,
    sample_rate: int,
    params: Dict,
) -> None:
    """
    Replace instant/smoothed BPM (and derived HR stats) using S1→S1 intervals from
    pass3_state_labels (contiguous S1 run starts). Uses the same MAD + rolling smooth
    params as peak-based BPM (pass2_instant_bpm_*, output_smoothing_window_sec).
    HRV-on-peaks and other metrics are unchanged.
    """
    sl = analysis_data.get("pass3_state_labels")
    if sl is None:
        return
    enc = analysis_data.get("pass3_state_labels_encoding") or {}
    s1_code = int(enc.get("S1", 0))
    _, bt, ib = calculate_bpm_series_from_s1_state_labels(
        sl, sample_rate, params, state_s1_code=s1_code
    )
    if bt is None or ib is None or len(bt) < 2:
        return
    bt = np.asarray(bt, dtype=np.float64)
    ib = np.asarray(ib, dtype=np.float64)
    metrics["bpm_times_raw"] = bt.copy()
    metrics["instant_bpm_raw"] = ib.copy()
    t_filt, b_filt = filter_instant_bpm_mad(bt, ib, params)
    if len(t_filt) == 0:
        logging.warning(
            "Pass 3: state-timeline BPM dropped all points after MAD; keeping peak-based BPM curve."
        )
        return
    smoothed_bpm, bpm_times, instant_bpm = smooth_bpm_series_from_instant(t_filt, b_filt, params)
    metrics["smoothed_bpm"] = smoothed_bpm
    metrics["bpm_times"] = bpm_times
    metrics["instant_bpm"] = instant_bpm
    metrics["major_inclines"] = find_major_hr_inclines(smoothed_bpm)
    metrics["major_declines"] = find_major_hr_declines(smoothed_bpm)
    metrics["hrr_stats"] = calculate_hrr(smoothed_bpm)
    metrics["peak_recovery_stats"] = find_peak_recovery_rate(smoothed_bpm)
    metrics["peak_exertion_stats"] = find_peak_exertion_rate(smoothed_bpm)
    if not smoothed_bpm.empty:
        hrv_summary = dict(metrics.get("hrv_summary") or {})
        hrv_summary["avg_bpm"] = float(smoothed_bpm.mean())
        hrv_summary["min_bpm"] = float(smoothed_bpm.min())
        hrv_summary["max_bpm"] = float(smoothed_bpm.max())
        metrics["hrv_summary"] = hrv_summary
    logging.info("Pass 3: BPM curve from state timeline (S1 run starts → same MAD/smooth as peaks).")


def analyze_wav_file(
    wav_file_path: str,
    params: Dict,
    start_bpm_hint: Optional[float],
    original_file_path: str,
    output_directory: str,
    output_options: Optional[Dict] = None,
    collect_fft_for_aggregate: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
):
    """Main analysis pipeline that orchestrates the refactored classes.

    Returns (plotly_figure, fft_aggregate_data, bpm_rename_summary). On early exit or failure,
    returns (None, None, None). bpm_rename_summary is a dict with start_bpm, min_bpm, max_bpm
    all from the final pass smoothed BPM series (first point in time, then min/max), or None if unavailable.
    """
    def _ui(label: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(label)
            except Exception:
                pass

    # Honor optional verbose logging flag from params to control how noisy the console is.
    # When disabled, we keep stage-level INFO logs but suppress very chatty algorithm-detail INFO logs.
    verbose_logging = bool(
        params.get("algorithm_console_logging", params.get("verbose_console_logging", True))
    )
    root_logger = logging.getLogger()
    active_filters = []

    if not verbose_logging:
        filt = _NoisyAlgorithmLogFilter()
        for handler in root_logger.handlers:
            handler.addFilter(filt)
            active_filters.append((handler, filt))

    start_time = time.time()
    logging.info(f"--- Processing file: {os.path.basename(original_file_path)} ---")

    # STAGE 1: Initialization
    _ui("Preprocessing audio...")
    audio_envelope, sample_rate, noise_floor, troughs = preprocess_audio(wav_file_path, params, output_directory, output_options)

    _ui("Pass 1: detecting anchor beats...")
    start_bpm, peak_time, recovery_time, anchor_beats, pass1_bpm, pass1_analysis_data = _run_pass1(
        audio_envelope, sample_rate, params, noise_floor, troughs, start_bpm_hint
    )

    # Pass 1 plot (envelope + anchor beats + BPM scatter/curve + BPM Trend (Belief)); skip when only last pass requested
    _opts = output_options if output_options is not None else DEFAULT_OUTPUT_OPTIONS.copy()
    if _opts.get("html", True) and _opts.get("output_all_passes", True):
        _ui("Generating pass 1 HTML report...")
        plotter_pass1 = Plotter(
            original_file_path,
            params,
            sample_rate,
            output_directory,
            source_audio_path=wav_file_path,
        )
        base_name = output_stem_from_path(original_file_path)
        pass1_html_path = os.path.join(output_directory, f"{base_name}_pass1.html")
        plotter_pass1.plot_pass1_save(
            audio_envelope,
            anchor_beats,
            _opts,
            pass1_html_path,
            pass1_analysis_data=pass1_analysis_data,
            pass1_bpm_data=pass1_bpm,
        )

    # STAGE 3: Pass 2 — main analysis with time-varying BPM prior from pass 1 curve
    logging.info("--- STAGE 3: Pass 2 — main analysis ---")
    _ui("Pass 2: classifying peaks...")
    pass1_bpm_prior = (
        _build_pass1_bpm_prior(pass1_bpm["curve_times"], pd.Series(pass1_bpm["curve_bpm"]))
        if pass1_bpm is not None
        else None
    )
    classifier = PeakClassifier(
        audio_envelope,
        sample_rate,
        params,
        start_bpm,
        noise_floor,
        troughs,
        peak_time,
        recovery_time,
        pass1_bpm_prior=pass1_bpm_prior,
    )
    s1_peaks, all_raw_peaks, analysis_data = classifier.classify_peaks()

    # Set default output options if none provided (needed for pass 2/pass 3 plot decisions)
    if output_options is None:
        output_options = DEFAULT_OUTPUT_OPTIONS.copy()

    # Pre-warm Kaleido so Chromium startup can overlap with analysis.
    try:
        if output_options.get("png", False):
            prewarm_kaleido_png_export()
    except Exception:
        pass
    needs_plot_outputs = any([
        output_options.get('html', True),
        output_options.get('png', False),
        output_options.get('csv', True),
    ])
    plotter = None
    metrics_pass2 = None

    # Compute pass 2 metrics when we might need them (pass 2 plot and/or pass 3 prior curve)
    output_all_passes = output_options.get("output_all_passes", True)
    if needs_plot_outputs and len(s1_peaks) >= 2:
        _ui("Pass 2: computing heart rate metrics...")
        metrics_pass2 = _calculate_metrics_from_peaks(s1_peaks, sample_rate, params)
        bt0 = metrics_pass2.get("bpm_times")
        ib0 = metrics_pass2.get("instant_bpm")
        if (
            bt0 is not None
            and ib0 is not None
            and len(bt0) == len(ib0)
            and len(bt0) > 0
        ):
            metrics_pass2["bpm_times_raw"] = np.asarray(bt0, dtype=np.float64).copy()
            metrics_pass2["instant_bpm_raw"] = np.asarray(ib0, dtype=np.float64).copy()
        # Pass 2: BPM curve and all derived stats from MAD-filtered instantaneous BPM (same logic as algorithm input)
        bt = metrics_pass2.get("bpm_times")
        ib = metrics_pass2.get("instant_bpm")
        if bt is not None and ib is not None and len(bt) == len(ib) and len(bt) >= 2:
            t_filt, b_filt = filter_instant_bpm_mad(bt, ib, params)
            if len(t_filt) > 0:
                smoothed_bpm, bpm_times, instant_bpm = smooth_bpm_series_from_instant(t_filt, b_filt, params)
                metrics_pass2["smoothed_bpm"] = smoothed_bpm
                metrics_pass2["bpm_times"] = bpm_times
                metrics_pass2["instant_bpm"] = instant_bpm
                metrics_pass2["major_inclines"] = find_major_hr_inclines(smoothed_bpm)
                metrics_pass2["major_declines"] = find_major_hr_declines(smoothed_bpm)
                metrics_pass2["hrr_stats"] = calculate_hrr(smoothed_bpm)
                metrics_pass2["peak_recovery_stats"] = find_peak_recovery_rate(smoothed_bpm)
                metrics_pass2["peak_exertion_stats"] = find_peak_exertion_rate(smoothed_bpm)
                if not smoothed_bpm.empty:
                    hrv_summary = metrics_pass2.get("hrv_summary") or {}
                    hrv_summary["avg_bpm"] = float(smoothed_bpm.mean())
                    hrv_summary["min_bpm"] = float(smoothed_bpm.min())
                    hrv_summary["max_bpm"] = float(smoothed_bpm.max())
                    metrics_pass2["hrv_summary"] = hrv_summary
        if output_all_passes:
            _ui("Pass 2: saving HTML / PNG / CSV...")
            plotter = Plotter(
                original_file_path,
                params,
                sample_rate,
                output_directory,
                source_audio_path=wav_file_path,
            )
            plotter.plot_and_save(
                audio_envelope,
                all_raw_peaks,
                analysis_data,
                metrics_pass2,
                output_options,
                output_suffix="_pass2",
                pass1_bpm_series=pd.Series(pass1_bpm["curve_bpm"]) if pass1_bpm is not None else None,
                pass1_bpm_times=pass1_bpm["curve_times"] if pass1_bpm is not None else None,
            )

    # Pass 3: takes pass 2 output (s1_peaks) as input; outputs refined peaks for reporting/plots
    peaks_after_pass2 = s1_peaks
    _ui("Pass 3: refining peaks...")
    peaks_after_pass3, analysis_data = _refine_and_correct_peaks(
        peaks_after_pass2,
        all_raw_peaks,
        analysis_data,
        audio_envelope,
        sample_rate,
        params,
        wav_file_path=wav_file_path,
    )

    # STAGE 6: Metrics from latest pass (pass 3). Peak-based metrics + optional state-timeline BPM override.
    if len(peaks_after_pass3) < 2:
        logging.warning("Not enough S1 peaks detected to generate full report.")
        _ui("Stopped: not enough detected heartbeat peaks.")
        return None, None, None

    logging.info("--- STAGE 6: Calculating Metrics and Generating Outputs ---")
    _ui("Pass 3: computing heart rate metrics...")
    reuse_pass2_metrics = (
        metrics_pass2 is not None
        and len(peaks_after_pass3) == len(s1_peaks)
        and np.array_equal(np.asarray(peaks_after_pass3), np.asarray(s1_peaks))
    )
    if reuse_pass2_metrics:
        # Shallow copy so Pass 3 BPM overrides do not mutate metrics_pass2 in place.
        metrics_after_pass3 = dict(metrics_pass2)
    else:
        metrics_after_pass3 = _calculate_metrics_from_peaks(peaks_after_pass3, sample_rate, params)
        bt0 = metrics_after_pass3.get("bpm_times")
        ib0 = metrics_after_pass3.get("instant_bpm")
        if (
            bt0 is not None
            and ib0 is not None
            and len(bt0) == len(ib0)
            and len(bt0) > 0
        ):
            metrics_after_pass3["bpm_times_raw"] = np.asarray(bt0, dtype=np.float64).copy()
            metrics_after_pass3["instant_bpm_raw"] = np.asarray(ib0, dtype=np.float64).copy()
        # Apply MAD-based BPM (same params as pass 2) on peak-derived instant BPM
        bt = metrics_after_pass3.get("bpm_times")
        ib = metrics_after_pass3.get("instant_bpm")
        if bt is not None and ib is not None and len(bt) == len(ib) and len(bt) >= 2:
            t_filt, b_filt = filter_instant_bpm_mad(bt, ib, params)
            if len(t_filt) > 0:
                smoothed_bpm, bpm_times, instant_bpm = smooth_bpm_series_from_instant(t_filt, b_filt, params)
                metrics_after_pass3["smoothed_bpm"] = smoothed_bpm
                metrics_after_pass3["bpm_times"] = bpm_times
                metrics_after_pass3["instant_bpm"] = instant_bpm
                metrics_after_pass3["major_inclines"] = find_major_hr_inclines(smoothed_bpm)
                metrics_after_pass3["major_declines"] = find_major_hr_declines(smoothed_bpm)
                metrics_after_pass3["hrr_stats"] = calculate_hrr(smoothed_bpm)
                metrics_after_pass3["peak_recovery_stats"] = find_peak_recovery_rate(smoothed_bpm)
                metrics_after_pass3["peak_exertion_stats"] = find_peak_exertion_rate(smoothed_bpm)
                if not smoothed_bpm.empty:
                    hrv_summary = metrics_after_pass3.get("hrv_summary") or {}
                    hrv_summary["avg_bpm"] = float(smoothed_bpm.mean())
                    hrv_summary["min_bpm"] = float(smoothed_bpm.min())
                    hrv_summary["max_bpm"] = float(smoothed_bpm.max())
                    metrics_after_pass3["hrv_summary"] = hrv_summary

    _apply_pass3_state_timeline_bpm(metrics_after_pass3, analysis_data, sample_rate, params)

    # OPTIONAL: Validation against manually labeled peaks (if a CSV exists next to the WAV).
    # This lets you batch-run a dataset and get an objective error count per file
    # without changing the main analysis workflow or outputs.
    try:
        manual_labels = _load_manual_labels_csv(original_file_path)
        if manual_labels:
            _ui("Validating against manual peak labels...")
            predicted_labels = _build_predicted_labels_for_validation(
                analysis_data, sample_rate
            )
            regression_log_path = None
            if output_options is not None:
                regression_log_path = output_options.get("regression_log_path")
            _append_validation_results_row(
                regression_log_path, original_file_path, manual_labels, predicted_labels
            )
        else:
            logging.info(
                "No manual labels CSV found for '%s'; skipping validation for this file.",
                os.path.basename(original_file_path),
            )
    except Exception as e:
        logging.error(
            "Manual label validation step failed for '%s': %s",
            os.path.basename(original_file_path),
            e,
        )

    plotly_figure = None

    # Pass 3 plot: after refinement (uses metrics_after_pass3; prior curve = BPM from pass 2)
    if needs_plot_outputs and len(peaks_after_pass3) >= 2:
        _ui("Pass 3: saving HTML / PNG / CSV...")
        if plotter is None:
            plotter = Plotter(
                original_file_path,
                params,
                sample_rate,
                output_directory,
                source_audio_path=wav_file_path,
            )
        # Pass 3 plot: show BPM (Pass 2) as the prior curve, not BPM (Pass 1)
        prior_bpm_series = None
        prior_bpm_times = None
        if metrics_pass2 is not None and metrics_pass2.get("smoothed_bpm") is not None and not metrics_pass2["smoothed_bpm"].empty:
            prior_bpm_series = metrics_pass2["smoothed_bpm"]
            prior_bpm_times = metrics_pass2.get("bpm_times")
        if prior_bpm_series is None and pass1_bpm is not None:
            prior_bpm_series = pd.Series(pass1_bpm["curve_bpm"])
            prior_bpm_times = pass1_bpm["curve_times"]
        # Pass 3 plot: include peak/recovery times for systolic shift (exertion vs all-time averaging)
        metrics_after_pass3["peak_bpm_time_sec"] = peak_time
        metrics_after_pass3["recovery_end_time_sec"] = recovery_time
        plotly_figure = plotter.plot_and_save(
            audio_envelope,
            all_raw_peaks,
            analysis_data,
            metrics_after_pass3,
            output_options,
            output_suffix="_pass3",
            filename_suffix="_pass3" if output_all_passes else "_bpm_plot",
            pass1_bpm_series=prior_bpm_series,
            pass1_bpm_times=prior_bpm_times,
        )
    elif not needs_plot_outputs:
        logging.info("Skipping all plot outputs (HTML/PNG/CSV) as requested.")

    # Generate other outputs if requested
    needs_reporter = any([
        output_options.get('summary', True),
        output_options.get('debug', True),
    ])

    if needs_reporter:
        reporter = ReportGenerator(original_file_path, output_directory)

        if output_options.get('summary', True):
            _ui("Writing summary report (Markdown)...")
            reporter.save_analysis_summary(metrics_after_pass3)
        else:
            logging.info("Skipping summary generation as requested.")

        if output_options.get('debug', True):
            _ui("Writing debug log (Markdown)...")
            reporter.create_chronological_log(audio_envelope, sample_rate, all_raw_peaks, analysis_data, metrics_after_pass3)
        else:
            logging.info("Skipping debug log generation as requested.")
    else:
        logging.info("Skipping all report generation as requested.")

    # FFT profiles: aggregate S1/S2 frequency spectra from raw audio (separate minimal HTML)
    fft_aggregate_data = None
    if params.get("enable_fft_profiles", True) and output_options.get("fft_profiles", True):
        _ui("Generating FFT profiles (HTML)...")
        try:
            base_name = output_stem_from_path(original_file_path)
            fft_output_path = os.path.join(output_directory, f"{base_name}_fft_profiles.html")
            if collect_fft_for_aggregate:
                target_sr = int(params.get("fft_aggregate_sr", 32000))
                fft_result = compute_fft_profiles(
                    wav_file_path,
                    analysis_data.get("peak_classifications", {}),
                    sample_rate,
                    audio_envelope,
                    params,
                    target_sr=target_sr,
                )
                save_fft_profiles_html(
                    wav_file_path,
                    analysis_data.get("peak_classifications", {}),
                    sample_rate,
                    fft_output_path,
                    audio_envelope,
                    params,
                    fft_result=fft_result,
                )
                fft_aggregate_data = fft_result
            else:
                fft_result = compute_fft_profiles(
                    wav_file_path,
                    analysis_data.get("peak_classifications", {}),
                    sample_rate,
                    audio_envelope,
                    params,
                )
                save_fft_profiles_html(
                    wav_file_path,
                    analysis_data.get("peak_classifications", {}),
                    sample_rate,
                    fft_output_path,
                    audio_envelope,
                    params,
                    fft_result=fft_result,
                )
            # Store S1 vs S2 frequency separation (10–15000 Hz) for future use; not used by any logic yet.
            if fft_result is not None and len(fft_result[0]) > 0:
                freqs, raw_s1_db, raw_s2_db = fft_result[0], fft_result[1], fft_result[2]
                analysis_data["fft_separation"] = compute_frequency_separation(
                    freqs, raw_s1_db, raw_s2_db, params
                )
            else:
                analysis_data["fft_separation"] = None
        except Exception as e:
            logging.warning(f"FFT profiles generation failed: {e}")

    duration = time.time() - start_time
    logging.info(f"--- Analysis stage finished in {duration:.2f} seconds (post-conversion). ---")

    # Remove filters so this setting is scoped to the analysis call.
    for handler, filt in active_filters:
        try:
            handler.removeFilter(filt)
        except Exception:
            pass

    bpm_rename_summary = None
    sb = metrics_after_pass3.get("smoothed_bpm")
    if sb is not None and not sb.empty:
        vals = sb.dropna()
        if not vals.empty:
            vals_time = vals.sort_index()
            bpm_rename_summary = {
                "start_bpm": float(vals_time.iloc[0]),
                "min_bpm": float(vals.min()),
                "max_bpm": float(vals.max()),
            }

    return plotly_figure, fft_aggregate_data, bpm_rename_summary
