import os
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any, Callable

from scipy.interpolate import interp1d

from audio_preprocessing import preprocess_audio
from noise_segments import compute_noise_event_segments
from config import DEFAULT_OUTPUT_OPTIONS, validate_params
from file_io import output_stem_from_path
from time_utils import dense_time_grid, rasterize_timeseries_linear, STANDARD_DT_SEC
from plotting import Plotter, prewarm_kaleido_png_export
from reporting import ReportGenerator
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
from fft_profiles import (
    compute_fft_profiles,
    compute_frequency_separation,
    save_fft_profiles_html,
)
from correction import run_pass3_correction
from config import param

# Recordings longer than this skip PNG export (Kaleido), but still allow HTML/CSV/etc.
# Independent of optimize_long_plots (that flag only trims heavy traces in Plotter when plots are produced).
LONG_RECORDING_DISABLE_PNG_SEC = 3600.0


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
        src_indices = np.arange(len(assigned_states_native), dtype=np.float64)
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


def _refine_and_correct_peaks(
    s1_peaks: np.ndarray,
    all_raw_peaks: np.ndarray,
    analysis_data: Dict,
    audio_envelope: np.ndarray,
    sample_rate: int,
    params: Dict,
    wav_file_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict]:
    """Pass 3: thin wrapper — delegates to correction.run_pass3_correction."""
    return run_pass3_correction(
        s1_peaks, all_raw_peaks, analysis_data,
        audio_envelope, sample_rate, params, wav_file_path,
    )


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

    Returns (plotly_figure, fft_aggregate_data, bpm_rename_summary, analysis_data). On early exit
    or failure, returns (None, None, None, None). bpm_rename_summary is a dict with start_bpm,
    min_bpm, max_bpm all from the final pass smoothed BPM series (first point in time, then
    min/max), or None if unavailable. analysis_data is the full AnalysisData dict from the last
    completed pass (pass3 unless pass4 is enabled).
    """
    def _ui(label: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(label)
            except Exception:
                pass

    validate_params(params)

    # Honor optional verbose logging flag from params to control how noisy the console is.
    # When disabled, we keep stage-level INFO logs but suppress very chatty algorithm-detail INFO logs.
    verbose_logging = bool(param(params, "algorithm_console_logging"))
    root_logger = logging.getLogger()
    active_filters = []

    if not verbose_logging:
        filt = _NoisyAlgorithmLogFilter()
        for handler in root_logger.handlers:
            handler.addFilter(filt)
            active_filters.append((handler, filt))

    start_time = time.time()
    logging.info("--- Processing file: %s ---", os.path.basename(original_file_path))

    # STAGE 1: Initialization
    _ui("Preprocessing audio...")
    (
        bandpass_envelope,
        sample_rate,
        noise_floor,
        troughs,
        inverse_band_envelope,
        noise_removed_envelope,
    ) = preprocess_audio(wav_file_path, params, output_directory, output_options)

    algorithm_envelope = (
        noise_removed_envelope
        if noise_removed_envelope is not None
        else bandpass_envelope
    )

    output_options = {**DEFAULT_OUTPUT_OPTIONS, **(output_options or {})}
    duration_sec = (
        float(len(algorithm_envelope)) / float(sample_rate)
        if sample_rate and len(algorithm_envelope) > 0
        else 0.0
    )
    long_recording = duration_sec > LONG_RECORDING_DISABLE_PNG_SEC
    if long_recording:
        output_options = dict(output_options)
        output_options["png"] = False
        logging.info(
            "Recording length %.1f min exceeds %.0f min — disabling PNG export (Kaleido).",
            duration_sec / 60.0,
            LONG_RECORDING_DISABLE_PNG_SEC / 60.0,
        )

    noise_event_segments: list = []
    if inverse_band_envelope is not None:
        try:
            noise_event_segments = compute_noise_event_segments(
                inverse_band_envelope, sample_rate, params
            )
        except Exception as e:
            logging.warning("Noise event segmentation failed: %s", e)

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
    output_all_passes = output_options.get("output_all_passes", True)
    plotter = None
    metrics_pass2 = None

    if bool(param(params, "use_springer_algorithm")):
        # ── Springer 2015 HSMM path: replaces native passes 1/2/3/4 ─────────────
        logging.info("--- STAGE 2-5: Springer 2015 HSMM (replaces native passes 1-3) ---")
        _ui("Springer: running HSMM segmentation...")
        peaks_after_pass4, all_raw_peaks, analysis_data = _run_springer_mode(
            wav_file_path, algorithm_envelope, sample_rate, params
        )
        analysis_data["bandpass_envelope"] = bandpass_envelope
        if inverse_band_envelope is not None:
            analysis_data["inverse_band_envelope"] = inverse_band_envelope
        if noise_removed_envelope is not None:
            analysis_data["noise_removed_envelope"] = noise_removed_envelope
        if noise_event_segments:
            analysis_data["noise_event_segments"] = noise_event_segments
        pass1_bpm = None
        peak_time = None
        recovery_time = None
    else:
        # ── Native multi-pass pipeline ────────────────────────────────────────────
        _ui("Pass 1: detecting anchor beats...")
        start_bpm, peak_time, recovery_time, anchor_beats, pass1_bpm, pass1_analysis_data = _run_pass1(
            algorithm_envelope, sample_rate, params, noise_floor, troughs, start_bpm_hint
        )
        pass1_analysis_data["bandpass_envelope"] = bandpass_envelope
        if inverse_band_envelope is not None:
            pass1_analysis_data["inverse_band_envelope"] = inverse_band_envelope
        if noise_removed_envelope is not None:
            pass1_analysis_data["noise_removed_envelope"] = noise_removed_envelope
        if noise_event_segments:
            pass1_analysis_data["noise_event_segments"] = noise_event_segments

        # Pass 1 plot (envelope + anchor beats + BPM scatter/curve + BPM Trend (Belief)); skip when only last pass requested
        _opts = output_options
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
                algorithm_envelope,
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
        analysis_data["bandpass_envelope"] = bandpass_envelope
        if inverse_band_envelope is not None:
            analysis_data["inverse_band_envelope"] = inverse_band_envelope
        if noise_removed_envelope is not None:
            analysis_data["noise_removed_envelope"] = noise_removed_envelope
        if noise_event_segments:
            analysis_data["noise_event_segments"] = noise_event_segments

        # Compute pass 2 metrics when we might need them (pass 2 plot and/or pass 3 prior curve)
        if needs_plot_outputs and len(s1_peaks) >= 2:
            _ui("Pass 2: computing heart rate metrics...")
            metrics_pass2 = _calculate_metrics_from_peaks(s1_peaks, sample_rate, params)
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
                    algorithm_envelope,
                    all_raw_peaks,
                    analysis_data,
                    metrics_pass2,
                    output_options,
                    output_suffix="_pass2",
                    pass1_bpm_series=np.asarray(pass1_bpm["curve_bpm"], dtype=np.float64) if pass1_bpm is not None else None,
                    pass1_bpm_times=np.asarray(pass1_bpm["curve_times"], dtype=np.float64) if pass1_bpm is not None else None,
                    is_final_pass=False,
                )

        # Pass 3: takes pass 2 output (s1_peaks) as input; outputs refined peaks for reporting/plots
        peaks_after_pass2 = s1_peaks
        _ui("Pass 3: refining peaks...")
        peaks_after_pass3, analysis_data = _refine_and_correct_peaks(
            peaks_after_pass2,
            all_raw_peaks,
            analysis_data,
            algorithm_envelope,
            sample_rate,
            params,
            wav_file_path=wav_file_path,
        )

        # Pass 4: holistic Viterbi decoder (guarded by config; off by default).
        peaks_after_pass4 = peaks_after_pass3
        if param(params, "enable_pass4"):
            from viterbi import run_pass4_viterbi
            _ui("Pass 4: Viterbi holistic decode...")
            peaks_after_pass4, analysis_data = run_pass4_viterbi(
                peaks_after_pass3, analysis_data, algorithm_envelope, sample_rate, params,
            )

    # STAGE 6: Metrics from latest pass (peaks_after_pass4 = pass3 when pass4 disabled).
    if len(peaks_after_pass4) < 2:
        logging.warning("Not enough S1 peaks detected to generate full report.")
        _ui("Stopped: not enough detected heartbeat peaks.")
        return None, None, None, None

    logging.info("--- STAGE 6: Calculating Metrics and Generating Outputs ---")
    _ui("Pass 3: computing heart rate metrics...")
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

    metrics_after_pass3["bpm_failure_report"] = detect_bpm_failure(
        metrics_after_pass3.get("bpm_times_raw"),
        metrics_after_pass3.get("instant_bpm_raw"),
        duration_sec,
        params,
    )
    # Also mirrored onto analysis_data: that's the dict callers/tools outside the plotter
    # and reporter (debug_helpers, benchmarking) actually get back from analyze_wav_file.
    analysis_data["bpm_failure_report"] = metrics_after_pass3["bpm_failure_report"]
    if metrics_after_pass3["bpm_failure_report"]["failed"]:
        logging.warning(
            "BPM plausibility gate flagged this analysis as likely failed: %s",
            "; ".join(metrics_after_pass3["bpm_failure_report"]["reasons"]),
        )

    plotly_figure = None

    # Pass 3/4 plot: after refinement (uses metrics_after_pass3; prior curve = BPM from pass 2)
    if needs_plot_outputs and len(peaks_after_pass4) >= 2:
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
        if metrics_pass2 is not None and metrics_pass2.get("smoothed_bpm") is not None and len(metrics_pass2["smoothed_bpm"]) > 0:
            prior_bpm_series = metrics_pass2["smoothed_bpm"]
            prior_bpm_times = metrics_pass2.get("bpm_times")
        if prior_bpm_series is None and pass1_bpm is not None:
            prior_bpm_series = np.asarray(pass1_bpm["curve_bpm"], dtype=np.float64)
            prior_bpm_times = np.asarray(pass1_bpm["curve_times"], dtype=np.float64)
        # Pass 3 plot: include peak/recovery times for systolic shift (exertion vs all-time averaging)
        metrics_after_pass3["peak_bpm_time_sec"] = peak_time
        metrics_after_pass3["recovery_end_time_sec"] = recovery_time
        plotly_figure = plotter.plot_and_save(
            algorithm_envelope,
            all_raw_peaks,
            analysis_data,
            metrics_after_pass3,
            output_options,
            output_suffix="_pass3",
            filename_suffix="_pass3" if output_all_passes else "_bpm_plot",
            pass1_bpm_series=prior_bpm_series,
            pass1_bpm_times=prior_bpm_times,
            is_final_pass=True,
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
            reporter.create_chronological_log(algorithm_envelope, sample_rate, all_raw_peaks, analysis_data, metrics_after_pass3)
        else:
            logging.info("Skipping debug log generation as requested.")
    else:
        logging.info("Skipping all report generation as requested.")

    # FFT profiles: aggregate S1/S2 frequency spectra from raw audio (separate minimal HTML)
    fft_aggregate_data = None
    if param(params, "enable_fft_profiles") and output_options.get("fft_profiles", True):
        _ui("Generating FFT profiles (HTML)...")
        try:
            base_name = output_stem_from_path(original_file_path)
            fft_output_path = os.path.join(output_directory, f"{base_name}_fft_profiles.html")
            if collect_fft_for_aggregate:
                target_sr = int(param(params, "fft_aggregate_sr"))
                fft_result = compute_fft_profiles(
                    wav_file_path,
                    analysis_data.get("peak_classifications", {}),
                    sample_rate,
                    algorithm_envelope,
                    params,
                    target_sr=target_sr,
                )
                save_fft_profiles_html(
                    wav_file_path,
                    analysis_data.get("peak_classifications", {}),
                    sample_rate,
                    fft_output_path,
                    algorithm_envelope,
                    params,
                    fft_result=fft_result,
                )
                fft_aggregate_data = fft_result
            else:
                fft_result = compute_fft_profiles(
                    wav_file_path,
                    analysis_data.get("peak_classifications", {}),
                    sample_rate,
                    algorithm_envelope,
                    params,
                )
                save_fft_profiles_html(
                    wav_file_path,
                    analysis_data.get("peak_classifications", {}),
                    sample_rate,
                    fft_output_path,
                    algorithm_envelope,
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
            logging.warning("FFT profiles generation failed: %s", e)

    duration = time.time() - start_time
    logging.info("--- Analysis stage finished in %.2f seconds (post-conversion). ---", duration)

    # Remove filters so this setting is scoped to the analysis call.
    for handler, filt in active_filters:
        try:
            handler.removeFilter(filt)
        except Exception:
            pass

    bpm_rename_summary = None
    sb = metrics_after_pass3.get("smoothed_bpm")
    if sb is not None and len(sb) > 0:
        arr = np.asarray(sb, dtype=np.float64)
        if np.any(np.isfinite(arr)):
            # start_bpm: first finite sample (dense raster)
            start_i = int(np.argmax(np.isfinite(arr)))
            bpm_rename_summary = {
                "start_bpm": float(arr[start_i]),
                "min_bpm": float(np.nanmin(arr)),
                "max_bpm": float(np.nanmax(arr)),
            }

    return plotly_figure, fft_aggregate_data, bpm_rename_summary, analysis_data
