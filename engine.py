"""Analysis engine: PCG recording in, beats / states / BPM metrics out.

`run_analysis` is the single entry point. It is pure computation: it reads the WAV
it is given and nothing else — no plots, reports, output directories, output
toggles, GUI, or console configuration. Anything that renders or saves results
(pipeline.py, debug_helpers, benchmarking) sits on top of it:

- intermediate passes are exposed through `on_stage(name, data)` so a caller can
  render them at the moment they exist (see STAGE_* below);
- debug audio is handed to `debug_audio_sink` instead of being written to disk.

tests/test_engine_boundary.py enforces that this module and everything it imports
stay free of UI/output code.
"""
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from audio_preprocessing import DebugAudioSink, preprocess_audio
from noise_segments import compute_noise_event_segments
from config import param, validate_params
from time_utils import dense_time_grid, rasterize_timeseries_linear, STANDARD_DT_SEC
from classifier import PeakClassifier
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
    detect_bpm_failure,
)
from correction import run_pass3_correction

# on_stage event names, in the order they fire. pass1/pass2 fire only on the native
# path, and fire once per algorithm attempt when auto-switch retries.
STAGE_PREPROCESSED = "preprocessed"  # {sample_rate, duration_sec, algorithm_envelope}
STAGE_PASS1 = "pass1"  # {anchor_beats, analysis_data, pass1_bpm}
STAGE_PASS2 = "pass2"  # {all_raw_peaks, analysis_data, metrics, pass1_bpm}; only when compute_pass2_metrics

StageCallback = Callable[[str, Dict[str, Any]], None]


@dataclass
class EngineResult:
    """Everything one analysis produced. `metrics` is None when fewer than 2 S1 peaks were found."""

    peaks: np.ndarray  # final S1 peaks (Pass 3, or Pass 4 when enabled), sample indices
    all_raw_peaks: np.ndarray
    analysis_data: Dict[str, Any]
    metrics: Optional[Dict[str, Any]]
    metrics_pass2: Optional[Dict[str, Any]]  # native path with compute_pass2_metrics only
    pass1_bpm: Optional[Dict[str, Any]]
    peak_bpm_time_sec: Optional[float]
    recovery_end_time_sec: Optional[float]
    algorithm_used: str  # "native" | "springer"
    algorithm_switch_reason: Optional[str]
    bpm_failure_report: Dict[str, Any]
    algorithm_envelope: np.ndarray
    sample_rate: int
    duration_sec: float
    # start/min/max of the final smoothed BPM (used for BPM-annotated renames), or None.
    bpm_summary: Optional[Dict[str, float]]

    @property
    def ok(self) -> bool:
        return self.metrics is not None


def _run_springer_mode(
    wav_file_path: str,
    algorithm_envelope: np.ndarray,
    sample_rate: int,
    params: Dict,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Replace native passes 1-3 with the Springer 2015 pretrained HSMM segmenter.
    Loads raw audio, resamples to sample_rate, runs the HSMM, converts per-sample
    state assignments (1=S1,2=systole,3=S2,4=diastole) to pass3_state_boundaries and
    a peak list, then returns (s1_peaks, all_raw_peaks, analysis_data).
    """
    import sys
    import soundfile as sf

    _SPRINGER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "springer2015")
    if _SPRINGER_DIR not in sys.path:
        sys.path.insert(0, _SPRINGER_DIR)

    from springer_hsmm.run import run_springer_segmentation_algorithm  # noqa: PLC0415
    from springer_hsmm.model_io import load_springer_model              # noqa: PLC0415
    from springer_hsmm.options import default_springer_hsmm_options     # noqa: PLC0415

    model_file = param(params, "springer_model") or "cristhian_potes_model.npz"
    model_path = os.path.join(_SPRINGER_DIR, model_file)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Springer model not found: {model_path}. "
            f"Set springer_model in config.py to a .npz file in springer2015/."
        )

    # Run springer at native sample rate — it does a 25-400 Hz bandpass internally,
    # so the audio must be at ≥800 Hz to avoid Nyquist truncation of that filter.
    # After decoding, nearest-neighbor resample the integer state labels to the
    # pipeline's sample_rate (600 Hz) so all downstream sample indices match.
    audio_raw, native_fs = sf.read(wav_file_path, always_2d=False)
    audio_raw = np.asarray(audio_raw, dtype=np.float64).flatten()

    analysis_start_sec = max(0.0, float(param(params, "analysis_start_sec")))
    if analysis_start_sec > 0.0:
        skip_n = int(round(analysis_start_sec * native_fs))
        skip_n = min(skip_n, max(0, audio_raw.size - 1))
        if skip_n > 0:
            audio_raw = audio_raw[skip_n:]

    model = load_springer_model(model_path)
    opts = default_springer_hsmm_options()
    assigned_states_native, _ = run_springer_segmentation_algorithm(
        audio_raw, float(native_fs),
        model["B_matrix"], model["pi_vector"], model["total_obs_distribution"],
        opts,
    )
    assigned_states_native = np.asarray(assigned_states_native, dtype=np.int32)

    # Nearest-neighbor resample integer state labels from native_fs to sample_rate.
    env_len = len(algorithm_envelope)
    if int(native_fs) == int(sample_rate):
        assigned_states = assigned_states_native[:env_len] if len(assigned_states_native) > env_len else assigned_states_native
    else:
        tgt_indices = np.linspace(0, len(assigned_states_native) - 1, env_len)
        nn_idx = np.clip(np.round(tgt_indices).astype(np.int64), 0, len(assigned_states_native) - 1)
        assigned_states = assigned_states_native[nn_idx]
    assigned_states = np.asarray(assigned_states, dtype=np.int32)

    # Final length guard.
    if len(assigned_states) != env_len:
        pad = int(assigned_states[-1]) if len(assigned_states) > 0 else 4
        if len(assigned_states) < env_len:
            assigned_states = np.concatenate(
                [assigned_states, np.full(env_len - len(assigned_states), pad, dtype=np.int32)]
            )
        else:
            assigned_states = assigned_states[:env_len]

    _CODE_TO_NAME = {1: "S1", 2: "systole", 3: "S2", 4: "diastole"}
    _STATE_ENCODING = {"S1": 1, "systole": 2, "S2": 3, "diastole": 4}

    state_boundaries = []
    s1_peaks_list = []
    i = 0
    while i < len(assigned_states):
        code = int(assigned_states[i])
        j = i
        while j < len(assigned_states) and int(assigned_states[j]) == code:
            j += 1
        name = _CODE_TO_NAME.get(code)
        if name is not None:
            mid = (i + j) // 2
            meta: Dict = {"s1": mid} if name == "S1" else {}
            state_boundaries.append((i, j, name, meta))
            if name == "S1":
                s1_peaks_list.append(mid)
        i = j

    s1_peaks = np.asarray(s1_peaks_list, dtype=np.int64)
    logging.info(
        "Springer: %d S1 peaks, %d boundary segments, %.1f s at %d Hz",
        len(s1_peaks), len(state_boundaries),
        len(assigned_states) / float(sample_rate), sample_rate,
    )
    analysis_data: Dict = {
        "pass3_state_labels": assigned_states,
        "pass3_state_labels_encoding": _STATE_ENCODING,
        "pass3_state_boundaries": state_boundaries,
        "pass3_state_boundaries_before": state_boundaries,
    }
    return s1_peaks, np.array([], dtype=np.int64), analysis_data


def _run_pass1(audio_envelope: np.ndarray, sample_rate: int, params: Dict,
               noise_floor: pd.Series, troughs: np.ndarray,
               start_bpm_hint: Optional[float],
               ) -> Tuple[float, Optional[float], Optional[float], np.ndarray, Optional[Dict], Dict]:
    """
    Runs pass 1 (high-confidence anchor-finding) to estimate global BPM and find the recovery phase.
    Returns (start_bpm, peak_bpm_time_sec, recovery_end_time_sec, anchor_beats, pass1_bpm, pass1_analysis_data).
    pass1_bpm is the canonical curve (outlier-filtered + light Gaussian smoothing) used for prior and all plots, or None if insufficient data.
    """
    logging.info("--- STAGE 2: Pass 1 — high-confidence anchor beats ---")
    params_pass1 = params.copy()
    params_pass1["pairing_confidence_threshold"] = param(params, "pass1_pairing_confidence_threshold")

    classifier = PeakClassifier(audio_envelope, sample_rate, params_pass1, start_bpm_hint,
                               noise_floor, troughs, None, None)
    anchor_beats, _, pass1_analysis_data = classifier.classify_peaks()

    global_bpm_estimate = None
    if len(anchor_beats) >= 10:
        median_rr_sec = np.median(np.diff(anchor_beats) / sample_rate)
        if median_rr_sec > 0:
            global_bpm_estimate = 60.0 / median_rr_sec
            logging.info("Automatically determined Global BPM Estimate: %.1f BPM", global_bpm_estimate)

    start_bpm = start_bpm_hint or global_bpm_estimate or 80.0

    # Canonical pass 1 BPM curve (outlier filter + light Gaussian smoothing) — same data used for prior and all plots
    pass1_bpm = compute_pass1_bpm_curve(anchor_beats, sample_rate, params)
    if pass1_bpm is not None:
        peak_bpm_time_sec, recovery_end_time_sec = find_recovery_phase(
            np.asarray(pass1_bpm["curve_bpm"], dtype=np.float64),
            np.asarray(pass1_bpm["curve_times"], dtype=np.float64),
            params,
        )
    else:
        pass1_fallback_series, pass1_fallback_times, _ = calculate_bpm_series(anchor_beats, sample_rate, params)
        peak_bpm_time_sec, recovery_end_time_sec = find_recovery_phase(
            np.asarray(pass1_fallback_series.values, dtype=np.float64),
            np.asarray(pass1_fallback_times, dtype=np.float64),
            params,
        )

    return start_bpm, peak_bpm_time_sec, recovery_end_time_sec, anchor_beats, pass1_bpm, pass1_analysis_data


def _build_pass1_bpm_prior(
    pass1_bpm_times: np.ndarray,
    pass1_bpm_values: np.ndarray,
) -> Optional[Callable[[float], float]]:
    """Build a time -> BPM callable from the pass 1 BPM curve for use as a time-varying prior. Returns None if insufficient data."""
    if pass1_bpm_times is None or pass1_bpm_values is None or len(pass1_bpm_times) < 2 or len(pass1_bpm_values) < 2:
        return None
    times = np.asarray(pass1_bpm_times, dtype=float)
    values = np.asarray(pass1_bpm_values, dtype=float)
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


def _calculate_metrics_from_peaks(peaks: np.ndarray, sample_rate: int, params: Dict) -> Dict:
    """Calculates BPM, HRV, and slope metrics from a peak list. Used by any pass (pass 2, pass 3, etc.)."""
    metrics = {}
    smoothed_bpm, bpm_times, instant_bpm = calculate_bpm_series(peaks, sample_rate, params)
    # Peak-derived raw series; overwritten by _apply_pass3_state_timeline_bpm below when
    # pass3_state_labels is available (state-timeline BPM is the more accurate final source,
    # used by both native and Springer paths). Kept here as the fallback / native-only case.
    metrics['bpm_times_raw'] = bpm_times
    metrics['instant_bpm_raw'] = instant_bpm
    metrics['major_inclines'] = find_major_hr_inclines(smoothed_bpm)
    metrics['major_declines'] = find_major_hr_declines(smoothed_bpm)
    metrics['hrr_stats'] = calculate_hrr(smoothed_bpm)
    metrics['peak_recovery_stats'] = find_peak_recovery_rate(smoothed_bpm)
    metrics['peak_exertion_stats'] = find_peak_exertion_rate(smoothed_bpm)
    metrics['windowed_hrv_df'] = calculate_windowed_hrv(peaks, sample_rate, params)
    if param(params, "enable_hrv_frequency_domain"):
        metrics['hrv_global_freq'] = calculate_global_hrv_frequency(peaks, sample_rate, params)
    else:
        metrics['hrv_global_freq'] = None

    hrv_summary_stats = {}
    if smoothed_bpm is not None and not getattr(smoothed_bpm, "empty", True):
        hrv_summary_stats['avg_bpm'] = smoothed_bpm.mean()
        hrv_summary_stats['min_bpm'] = smoothed_bpm.min()
        hrv_summary_stats['max_bpm'] = smoothed_bpm.max()
    if not metrics['windowed_hrv_df'].empty:
        hrv_summary_stats['avg_rmssdc'] = metrics['windowed_hrv_df']['rmssdc'].mean()
        hrv_summary_stats['avg_sdnn'] = metrics['windowed_hrv_df']['sdnn'].mean()
        if param(params, "enable_hrv_frequency_domain") and "lf_hf_ratio" in metrics['windowed_hrv_df'].columns:
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

    # Canonical BPM representation for all downstream code: dense raster at STANDARD_DT_SEC.
    # We keep peak-derived time axis (bpm_times, in seconds) but store only the raster.
    if bpm_times is not None and smoothed_bpm is not None and not getattr(smoothed_bpm, "empty", True):
        try:
            dur = float(np.max(peaks) / sample_rate) if len(peaks) else float(bpm_times[-1])
        except Exception:
            dur = float(bpm_times[-1]) if bpm_times is not None and len(bpm_times) else 0.0
        t_grid = dense_time_grid(dur, STANDARD_DT_SEC)
        bpm_vals = np.asarray(smoothed_bpm.values, dtype=np.float64)
        bpm_grid = rasterize_timeseries_linear(np.asarray(bpm_times, dtype=np.float64), bpm_vals, t_grid, fallback=float(bpm_vals[0]))
        metrics["bpm_times"] = t_grid
        metrics["smoothed_bpm"] = bpm_grid
    else:
        metrics["bpm_times"] = np.asarray([], dtype=np.float64)
        metrics["smoothed_bpm"] = np.asarray([], dtype=np.float64)

    return metrics


def _write_hr_stats(metrics: Dict[str, Any], smoothed_bpm) -> None:
    """Write all derived HR stats from a smoothed BPM series into metrics."""
    metrics["major_inclines"] = find_major_hr_inclines(smoothed_bpm)
    metrics["major_declines"] = find_major_hr_declines(smoothed_bpm)
    metrics["hrr_stats"] = calculate_hrr(smoothed_bpm)
    metrics["peak_recovery_stats"] = find_peak_recovery_rate(smoothed_bpm)
    metrics["peak_exertion_stats"] = find_peak_exertion_rate(smoothed_bpm)
    if not getattr(smoothed_bpm, "empty", True):
        hrv_summary = dict(metrics.get("hrv_summary") or {})
        hrv_summary["avg_bpm"] = float(smoothed_bpm.mean())
        hrv_summary["min_bpm"] = float(smoothed_bpm.min())
        hrv_summary["max_bpm"] = float(smoothed_bpm.max())
        metrics["hrv_summary"] = hrv_summary


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
    # Store canonical dense raster (dt=STANDARD_DT_SEC) instead of irregular points/Series.
    try:
        dur = float(len(sl) / float(sample_rate))
    except Exception:
        dur = float(bpm_times[-1]) if bpm_times is not None and len(bpm_times) else 0.0
    t_grid = dense_time_grid(dur, STANDARD_DT_SEC)
    bpm_vals = np.asarray(smoothed_bpm.values, dtype=np.float64)
    bpm_grid = rasterize_timeseries_linear(np.asarray(bpm_times, dtype=np.float64), bpm_vals, t_grid, fallback=float(bpm_vals[0]))
    metrics["smoothed_bpm"] = bpm_grid
    metrics["bpm_times"] = t_grid
    metrics.pop("instant_bpm", None)
    _write_hr_stats(metrics, smoothed_bpm)
    logging.info("Pass 3: BPM curve from state timeline (S1 run starts → same MAD/smooth as peaks).")


def _should_switch_algorithm(alt_failed: bool, primary_reason_count: int, alt_reason_count: int) -> bool:
    """Decide whether an auto-switch retry should replace the primary result.

    Only called when the primary run has already failed the BPM plausibility gate
    (auto_switch_algorithm's trigger condition), so this just compares the retry
    against that known-failed primary: switch if the alternate passes outright, or
    if both still fail but the alternate has strictly fewer failure reasons. Ties
    keep the primary, so a file where both algorithms struggle equally doesn't
    flip-flop between runs.
    """
    if not alt_failed:
        return True
    return alt_reason_count < primary_reason_count


def _attach_envelopes(analysis_data: Dict, envelopes: Dict[str, Any]) -> None:
    """Copy the preprocessing envelopes/segments onto a pass's analysis_data (skipping absent ones)."""
    analysis_data["bandpass_envelope"] = envelopes["bandpass_envelope"]
    for key in ("inverse_band_envelope", "noise_removed_envelope"):
        if envelopes[key] is not None:
            analysis_data[key] = envelopes[key]
    if envelopes["noise_event_segments"]:
        analysis_data["noise_event_segments"] = envelopes["noise_event_segments"]


def _run_algorithm_pass(
    use_springer: bool,
    wav_file_path: str,
    algorithm_envelope: np.ndarray,
    sample_rate: int,
    params: Dict,
    noise_floor,
    troughs,
    start_bpm_hint,
    envelopes: Dict[str, Any],
    duration_sec: float,
    compute_pass2_metrics: bool,
    _emit: StageCallback,
    _ui: Callable[[str], None],
) -> Dict[str, Any]:
    """Run one full algorithm branch (native multi-pass or Springer 2015 HSMM) through
    Pass 3/4, then compute metrics_after_pass3 and the BPM plausibility gate result.

    Split out of run_analysis so auto-switch (see 'auto_switch_algorithm' param) can
    call this twice — once per algorithm — and compare results without duplicating the
    STAGE 1 preprocessing that's shared regardless of which algorithm runs.

    Returns a dict: peaks_after_pass4, all_raw_peaks, analysis_data, metrics_after_pass3
    (None if too few peaks), metrics_pass2 (native only, else None), s1_peaks (native only,
    else None), pass1_bpm, peak_time, recovery_time, algorithm_name, bpm_failure_report.
    """
    metrics_pass2 = None
    s1_peaks = None

    if use_springer:
        # ── Springer 2015 HSMM path: replaces native passes 1/2/3/4 ─────────────
        logging.info("--- STAGE 2-5: Springer 2015 HSMM (replaces native passes 1-3) ---")
        _ui("Springer: running HSMM segmentation...")
        peaks_after_pass4, all_raw_peaks, analysis_data = _run_springer_mode(
            wav_file_path, algorithm_envelope, sample_rate, params
        )
        _attach_envelopes(analysis_data, envelopes)
        pass1_bpm = None
        peak_time = None
        recovery_time = None
    else:
        # ── Native multi-pass pipeline ────────────────────────────────────────────
        _ui("Pass 1: detecting anchor beats...")
        start_bpm, peak_time, recovery_time, anchor_beats, pass1_bpm, pass1_analysis_data = _run_pass1(
            algorithm_envelope, sample_rate, params, noise_floor, troughs, start_bpm_hint
        )
        _attach_envelopes(pass1_analysis_data, envelopes)
        _emit(STAGE_PASS1, {
            "anchor_beats": anchor_beats,
            "analysis_data": pass1_analysis_data,
            "pass1_bpm": pass1_bpm,
        })

        # STAGE 3: Pass 2 — main analysis with time-varying BPM prior from pass 1 curve
        logging.info("--- STAGE 3: Pass 2 — main analysis ---")
        _ui("Pass 2: classifying peaks...")
        pass1_bpm_prior = (
            _build_pass1_bpm_prior(
                np.asarray(pass1_bpm["curve_times"], dtype=np.float64),
                np.asarray(pass1_bpm["curve_bpm"], dtype=np.float64),
            )
            if pass1_bpm is not None
            else None
        )
        classifier = PeakClassifier(
            algorithm_envelope,
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
        _attach_envelopes(analysis_data, envelopes)

        # Pass 2 metrics are optional: callers want them for pass 2 output / as the
        # prior curve on the final plot; when computed they also let the final pass
        # reuse them if Pass 3/4 left the peaks unchanged.
        if compute_pass2_metrics and len(s1_peaks) >= 2:
            _ui("Pass 2: computing heart rate metrics...")
            metrics_pass2 = _calculate_metrics_from_peaks(s1_peaks, sample_rate, params)
            # Fired before Pass 3 mutates analysis_data, so observers see the Pass 2 state.
            _emit(STAGE_PASS2, {
                "all_raw_peaks": all_raw_peaks,
                "analysis_data": analysis_data,
                "metrics": metrics_pass2,
                "pass1_bpm": pass1_bpm,
            })

        # Pass 3: takes pass 2 output (s1_peaks) as input; outputs refined peaks for reporting/plots
        _ui("Pass 3: refining peaks...")
        peaks_after_pass3, analysis_data = run_pass3_correction(
            s1_peaks, all_raw_peaks, analysis_data,
            algorithm_envelope, sample_rate, params, wav_file_path,
        )

        # Pass 4: holistic Viterbi decoder (guarded by config; off by default).
        peaks_after_pass4 = peaks_after_pass3
        if param(params, "enable_pass4"):
            from viterbi import run_pass4_viterbi
            _ui("Pass 4: Viterbi holistic decode...")
            peaks_after_pass4, analysis_data = run_pass4_viterbi(
                peaks_after_pass3, analysis_data, algorithm_envelope, sample_rate, params,
            )

    metrics_after_pass3 = None
    if len(peaks_after_pass4) < 2:
        bpm_failure_report = {
            "failed": True,
            "reasons": ["fewer than 2 S1 peaks detected"],
            "metrics": {},
        }
    else:
        reuse_pass2_metrics = (
            metrics_pass2 is not None
            and len(peaks_after_pass4) == len(s1_peaks)
            and np.array_equal(np.asarray(peaks_after_pass4), np.asarray(s1_peaks))
        )
        if reuse_pass2_metrics:
            # Shallow copy so Pass 3/4 BPM overrides do not mutate metrics_pass2 in place.
            metrics_after_pass3 = dict(metrics_pass2)
        else:
            metrics_after_pass3 = _calculate_metrics_from_peaks(peaks_after_pass4, sample_rate, params)

        _apply_pass3_state_timeline_bpm(metrics_after_pass3, analysis_data, sample_rate, params)

        bpm_failure_report = detect_bpm_failure(
            metrics_after_pass3.get("bpm_times_raw"),
            metrics_after_pass3.get("instant_bpm_raw"),
            duration_sec,
            params,
        )
        metrics_after_pass3["bpm_failure_report"] = bpm_failure_report
        # Also mirrored onto analysis_data: that's the dict callers/tools outside the plotter
        # and reporter (debug_helpers, benchmarking) actually get back.
        analysis_data["bpm_failure_report"] = bpm_failure_report
        if bpm_failure_report["failed"]:
            logging.warning(
                "BPM plausibility gate flagged the %s run as likely failed: %s",
                "Springer" if use_springer else "native",
                "; ".join(bpm_failure_report["reasons"]),
            )

    return {
        "peaks_after_pass4": peaks_after_pass4,
        "all_raw_peaks": all_raw_peaks,
        "analysis_data": analysis_data,
        "metrics_after_pass3": metrics_after_pass3,
        "metrics_pass2": metrics_pass2,
        "s1_peaks": s1_peaks,
        "pass1_bpm": pass1_bpm,
        "peak_time": peak_time,
        "recovery_time": recovery_time,
        "algorithm_name": "springer" if use_springer else "native",
        "bpm_failure_report": bpm_failure_report,
    }


def _bpm_summary(metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """start/min/max of the final dense smoothed BPM raster, or None if unavailable."""
    if metrics is None:
        return None
    sb = metrics.get("smoothed_bpm")
    if sb is None or len(sb) == 0:
        return None
    arr = np.asarray(sb, dtype=np.float64)
    if not np.any(np.isfinite(arr)):
        return None
    # start_bpm: first finite sample (dense raster)
    start_i = int(np.argmax(np.isfinite(arr)))
    return {
        "start_bpm": float(arr[start_i]),
        "min_bpm": float(np.nanmin(arr)),
        "max_bpm": float(np.nanmax(arr)),
    }


def run_analysis(
    wav_file_path: str,
    params: Dict,
    start_bpm_hint: Optional[float] = None,
    *,
    progress_callback: Optional[Callable[[str], None]] = None,
    on_stage: Optional[StageCallback] = None,
    debug_audio_sink: Optional[DebugAudioSink] = None,
    compute_pass2_metrics: bool = False,
) -> EngineResult:
    """Analyze one WAV: preprocessing → algorithm passes (with optional auto-switch) → metrics.

    progress_callback: short human-readable status strings; exceptions it raises are ignored.
    on_stage: called as on_stage(STAGE_*, data) at each intermediate stage; exceptions propagate.
    debug_audio_sink: receives intermediate audio for debug playback (see preprocess_audio).
    compute_pass2_metrics: also compute Pass 2 BPM/HRV metrics (needed for STAGE_PASS2 and
        EngineResult.metrics_pass2); off by default to save time when nothing renders them.
    """
    def _ui(label: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(label)
            except Exception:
                pass

    def _emit(stage: str, data: Dict[str, Any]) -> None:
        if on_stage is not None:
            on_stage(stage, data)

    validate_params(params)

    # STAGE 1: Initialization
    _ui("Preprocessing audio...")
    (
        bandpass_envelope,
        sample_rate,
        noise_floor,
        troughs,
        inverse_band_envelope,
        noise_removed_envelope,
    ) = preprocess_audio(wav_file_path, params, debug_audio_sink)

    algorithm_envelope = (
        noise_removed_envelope
        if noise_removed_envelope is not None
        else bandpass_envelope
    )
    duration_sec = (
        float(len(algorithm_envelope)) / float(sample_rate)
        if sample_rate and len(algorithm_envelope) > 0
        else 0.0
    )
    _emit(STAGE_PREPROCESSED, {
        "sample_rate": sample_rate,
        "duration_sec": duration_sec,
        "algorithm_envelope": algorithm_envelope,
    })

    noise_event_segments: list = []
    if inverse_band_envelope is not None:
        try:
            noise_event_segments = compute_noise_event_segments(
                inverse_band_envelope, sample_rate, params
            )
        except Exception as e:
            logging.warning("Noise event segmentation failed: %s", e)

    primary_use_springer = bool(param(params, "use_springer_algorithm"))
    _pass_kwargs = dict(
        wav_file_path=wav_file_path,
        algorithm_envelope=algorithm_envelope,
        sample_rate=sample_rate,
        params=params,
        noise_floor=noise_floor,
        troughs=troughs,
        start_bpm_hint=start_bpm_hint,
        envelopes={
            "bandpass_envelope": bandpass_envelope,
            "inverse_band_envelope": inverse_band_envelope,
            "noise_removed_envelope": noise_removed_envelope,
            "noise_event_segments": noise_event_segments,
        },
        duration_sec=duration_sec,
        compute_pass2_metrics=compute_pass2_metrics,
        _emit=_emit,
        _ui=_ui,
    )

    result = _run_algorithm_pass(primary_use_springer, **_pass_kwargs)
    algorithm_switch_reason = None

    if bool(param(params, "auto_switch_algorithm")) and result["bpm_failure_report"]["failed"]:
        alt_use_springer = not primary_use_springer
        logging.info(
            "Auto-switch: '%s' failed the BPM plausibility gate (%s); retrying with '%s'.",
            result["algorithm_name"],
            "; ".join(result["bpm_failure_report"]["reasons"]),
            "springer" if alt_use_springer else "native",
        )
        _ui(f"Primary algorithm flagged; retrying with {'Springer' if alt_use_springer else 'native'}...")
        alt_result = _run_algorithm_pass(alt_use_springer, **_pass_kwargs)

        should_switch = _should_switch_algorithm(
            alt_result["bpm_failure_report"]["failed"],
            len(result["bpm_failure_report"]["reasons"]),
            len(alt_result["bpm_failure_report"]["reasons"]),
        )

        if should_switch:
            algorithm_switch_reason = (
                f"switched from {result['algorithm_name']} to {alt_result['algorithm_name']} "
                f"(gate flagged: {'; '.join(result['bpm_failure_report']['reasons'])})"
            )
            logging.warning("Auto-switch: %s", algorithm_switch_reason)
            result = alt_result
        else:
            logging.info(
                "Auto-switch: kept '%s' (retry with '%s' did not improve).",
                result["algorithm_name"],
                alt_result["algorithm_name"],
            )

    analysis_data = result["analysis_data"]
    metrics = result["metrics_after_pass3"]
    analysis_data["algorithm_used"] = result["algorithm_name"]
    analysis_data["algorithm_switch_reason"] = algorithm_switch_reason
    if metrics is not None:
        metrics["algorithm_used"] = result["algorithm_name"]
        metrics["algorithm_switch_reason"] = algorithm_switch_reason

    return EngineResult(
        peaks=result["peaks_after_pass4"],
        all_raw_peaks=result["all_raw_peaks"],
        analysis_data=analysis_data,
        metrics=metrics,
        metrics_pass2=result["metrics_pass2"],
        pass1_bpm=result["pass1_bpm"],
        peak_bpm_time_sec=result["peak_time"],
        recovery_end_time_sec=result["recovery_time"],
        algorithm_used=result["algorithm_name"],
        algorithm_switch_reason=algorithm_switch_reason,
        bpm_failure_report=result["bpm_failure_report"],
        algorithm_envelope=algorithm_envelope,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        bpm_summary=_bpm_summary(metrics),
    )
