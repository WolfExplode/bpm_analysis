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

            too_short = systole < (1.0 - systole_slack) * s1_s2_min
            too_long = systole > (1.0 + systole_slack) * s1_s2_max
            far_from_nominal = abs(systole - s1_s2_nominal) > max(0.12, 0.5 * (s1_s2_max - s1_s2_min))

            diag = {
                "i": int(i),
                "s1": int(s1),
                "s2": int(s2),
                "s1_next": int(s1_next),
                "bpm": float(bpm),
                "rr_sec": float(rr),
                "systole_sec": float(systole),
                "diastole_sec": float(diastole),
                "s1_s2_min": float(s1_s2_min),
                "s1_s2_nominal": float(s1_s2_nominal),
                "s1_s2_max": float(s1_s2_max),
                "flags": {
                    "systole_too_short": bool(too_short),
                    "systole_too_long": bool(too_long),
                    "systole_far_from_nominal": bool(far_from_nominal),
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

        if not changed:
            break

    # Final peaks after pass 3 correction (still S1 list).
    peaks_out = np.asarray(s1_list, dtype=np.int64)

    # ---------------------------------------------------------------------
    # 3) Paint the final state timeline from corrected events
    # ---------------------------------------------------------------------
    state_labels[:] = STATE_DIASTOLE
    state_boundaries.clear()
    s2_events_final: List[int] = []

    for i in range(len(peaks_out) - 1):
        s1 = int(peaks_out[i])
        s1_next = int(peaks_out[i + 1])
        if s1_next <= s1:
            continue

        # Reuse corrected s2_events when available; otherwise predict.
        if i < len(s2_events):
            s2 = int(s2_events[i])
        else:
            t_s1 = s1 / float(sample_rate)
            bpm = _bpm_at_time(t_s1)
            intervals = calculate_bpm_intervals(bpm, params)
            s1_s2_nominal = float(intervals.get("s1_s2_nominal", 0.30))
            s2_pred = int(round(s1 + s1_s2_nominal * sample_rate))
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

        if s1_end > s1_start:
            state_labels[s1_start:s1_end] = STATE_S1
            state_boundaries.append((s1_start, s1_end, "S1", {"s1": s1}))
        if s2_start > s1_end:
            state_labels[s1_end:s2_start] = STATE_SYSTOLE
            state_boundaries.append((s1_end, s2_start, "systole", {"s1": s1, "s2": s2}))
        if s2_end > s2_start:
            state_labels[s2_start:s2_end] = STATE_S2
            state_boundaries.append((s2_start, s2_end, "S2", {"s2": s2}))
        if s1_next > s2_end:
            state_labels[s2_end:s1_next] = STATE_DIASTOLE
            state_boundaries.append((s2_end, s1_next, "diastole", {"s2": s2, "s1_next": s1_next}))

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

    # STAGE 6: Metrics from latest pass (pass 3). Use same MAD-based BPM as pass 2 so curves match when no correction.
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
        metrics_after_pass3 = metrics_pass2
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
        # Apply MAD-based BPM (same as pass 2) so BPM (Pass 3) is consistent and matches pass 2 when peaks unchanged
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
