import os
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any, Callable

from scipy.interpolate import interp1d

from audio_preprocessing import preprocess_audio
from config import DEFAULT_OUTPUT_OPTIONS
from plotting import Plotter
from reporting import ReportGenerator
from validation import (
    _load_manual_labels_csv,
    _build_predicted_labels_for_validation,
    _append_validation_results_row,
)
from classifier import PeakClassifier
from hrv import (
    calculate_bpm_series,
    compute_preliminary_bpm_curve,
    find_recovery_phase,
    detect_trapezoid_discontinuities,
    find_major_hr_inclines,
    find_major_hr_declines,
    calculate_hrr,
    find_peak_recovery_rate,
    find_peak_exertion_rate,
    calculate_windowed_hrv,
    calculate_global_hrv_frequency,
)
from correction import correct_peaks_by_rhythm, fix_rhythmic_discontinuities
from fft_profiles import compute_fft_profiles, save_fft_profiles_html


class _NoisyAlgorithmLogFilter(logging.Filter):
    """
    Filters out very chatty INFO-level messages that make benchmarking hard.
    WARNING/ERROR always pass through.
    """

    # Substrings that identify "noisy" algorithm-detail logs.
    _NOISY_SUBSTRINGS = (
        "KICK-START:",
        "CASCADE RESET:",
        "Trapezoid #",
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


def _run_preliminary_pass(audio_envelope: np.ndarray, sample_rate: int, params: Dict,
                          noise_floor: pd.Series, troughs: np.ndarray,
                          start_bpm_hint: Optional[float],
                          band_envelopes: Optional[Dict[str, np.ndarray]] = None,
                          ) -> Tuple[float, Optional[float], Optional[float], np.ndarray, Optional[Dict], Dict]:
    """
    Runs a high-confidence first pass to estimate global BPM and find the recovery phase.
    Returns (start_bpm, peak_bpm_time_sec, recovery_end_time_sec, anchor_beats, prelim_bpm, prelim_analysis_data).
    prelim_bpm is the canonical curve (outlier-filtered + LOESS) used for prior and all plots, or None if insufficient data.
    """
    logging.info("--- STAGE 2: Running High-Confidence pass to find anchor beats ---")
    params_pass_1 = params.copy()
    # Use a higher threshold for a more confident initial beat detection
    params_pass_1["pairing_confidence_threshold"] = params.get("preliminary_pass_confidence_threshold", 0.75)

    # Use the classifier for a high-confidence dry run
    classifier = PeakClassifier(audio_envelope, sample_rate, params_pass_1, start_bpm_hint,
                                noise_floor, troughs, None, None, band_envelopes)
    anchor_beats, _, prelim_analysis_data = classifier.classify_peaks()

    global_bpm_estimate = None
    if len(anchor_beats) >= 10:
        median_rr_sec = np.median(np.diff(anchor_beats) / sample_rate)
        if median_rr_sec > 0:
            global_bpm_estimate = 60.0 / median_rr_sec
            logging.info(f"Automatically determined Global BPM Estimate: {global_bpm_estimate:.1f} BPM")

    # Determine the starting BPM for the main analysis
    start_bpm = start_bpm_hint or global_bpm_estimate or 80.0

    # Canonical preliminary BPM curve (outlier filter + LOESS) — same data used for prior and all plots
    prelim_bpm = compute_preliminary_bpm_curve(anchor_beats, sample_rate, params)
    if prelim_bpm is not None:
        curve_series = pd.Series(prelim_bpm["curve_bpm"])
        peak_bpm_time_sec, recovery_end_time_sec = find_recovery_phase(curve_series, prelim_bpm["curve_times"], params)
    else:
        prelim_bpm_series, prelim_bpm_times = calculate_bpm_series(anchor_beats, sample_rate, params)
        peak_bpm_time_sec, recovery_end_time_sec = find_recovery_phase(prelim_bpm_series, prelim_bpm_times, params)

    return start_bpm, peak_bpm_time_sec, recovery_end_time_sec, anchor_beats, prelim_bpm, prelim_analysis_data


def _build_prelim_bpm_prior(
    prelim_bpm_times: np.ndarray,
    prelim_bpm_series: pd.Series,
) -> Optional[Callable[[float], float]]:
    """Build a time -> BPM callable from the preliminary BPM series for use as a time-varying prior. Returns None if insufficient data."""
    if prelim_bpm_times is None or prelim_bpm_series is None or len(prelim_bpm_times) < 2 or prelim_bpm_series.empty:
        return None
    times = np.asarray(prelim_bpm_times, dtype=float)
    values = np.asarray(prelim_bpm_series.values, dtype=float)
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
                              sample_rate: int, params: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Applies rhythmic and iterative contextual correction passes to refine S1 peaks.
    """
    logging.info("--- STAGES 4 & 5: Refining peaks with rhythmic and contextual correction ---")

    # STAGE 4: Simple rhythmic correction (e.g., remove beats that are too close)
    s1_peaks_rhythm_corrected = correct_peaks_by_rhythm(s1_peaks, audio_envelope, sample_rate, params)

    # Prepare data for the iterative pass
    dynamic_noise_floor = analysis_data['dynamic_noise_floor_series']
    current_debug_info = analysis_data["beat_debug_info"].copy()
    final_peaks = s1_peaks_rhythm_corrected

    # iterative correction loop
    # Safety cap: in practice 1-2 passes suffice; 5 prevents a runaway loop if
    # the corrections destabilize and re-introduce the same errors each time.
    max_iterations = 1
    for i in range(max_iterations):
        logging.info(f"Correction Pass Iteration {i + 1}...")

        new_peaks, new_debug_info, corrections_made = fix_rhythmic_discontinuities(
            s1_peaks=final_peaks,
            all_raw_peaks=all_raw_peaks,
            debug_info=current_debug_info,
            audio_envelope=audio_envelope,
            dynamic_noise_floor=dynamic_noise_floor,
            params=params,
            sample_rate=sample_rate
        )

        final_peaks = new_peaks
        current_debug_info = new_debug_info

        if corrections_made == 0:
            logging.info("Correction process stabilized. Exiting loop.")
            break
        else:
            logging.info(f"Made {corrections_made} corrections in iteration {i + 1}.")
    else:
        logging.warning("Correction process reached max iterations without stabilizing.")

    analysis_data["beat_debug_info"] = current_debug_info
    return final_peaks, analysis_data


def _calculate_final_metrics(final_peaks: np.ndarray, sample_rate: int, params: Dict) -> Dict:
    """Calculates all final BPM, HRV, and slope metrics for reporting."""
    metrics = {}
    metrics['smoothed_bpm'], metrics['bpm_times'] = calculate_bpm_series(final_peaks, sample_rate, params)
    metrics['trapezoids'] = detect_trapezoid_discontinuities(metrics['smoothed_bpm'], metrics['bpm_times'], params)
    metrics['major_inclines'] = find_major_hr_inclines(metrics['smoothed_bpm'])
    metrics['major_declines'] = find_major_hr_declines(metrics['smoothed_bpm'])
    metrics['hrr_stats'] = calculate_hrr(metrics['smoothed_bpm'])
    metrics['peak_recovery_stats'] = find_peak_recovery_rate(metrics['smoothed_bpm'])
    metrics['peak_exertion_stats'] = find_peak_exertion_rate(metrics['smoothed_bpm'])
    metrics['windowed_hrv_df'] = calculate_windowed_hrv(final_peaks, sample_rate, params)
    if params.get("enable_hrv_frequency_domain", False):
        metrics['hrv_global_freq'] = calculate_global_hrv_frequency(final_peaks, sample_rate, params)
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


def analyze_wav_file(wav_file_path: str, params: Dict, start_bpm_hint: Optional[float], original_file_path: str, output_directory: str, output_options: Optional[Dict] = None, collect_fft_for_aggregate: bool = False):
    """Main analysis pipeline that orchestrates the refactored classes."""
    # Honor optional verbose logging flag from params to control how noisy the console is.
    # When disabled, we keep stage-level INFO logs but suppress very chatty algorithm-detail INFO logs.
    verbose_logging = bool(params.get("verbose_console_logging", True))
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
    audio_envelope, sample_rate, band_envelopes, noise_floor, troughs = preprocess_audio(wav_file_path, params, output_directory, output_options)

    start_bpm, peak_time, recovery_time, anchor_beats, prelim_bpm, prelim_analysis_data = _run_preliminary_pass(
        audio_envelope, sample_rate, params, noise_floor, troughs, start_bpm_hint, band_envelopes
    )

    # Preliminary pass plot (envelope + anchor beats + BPM scatter/curve + BPM Trend (Belief)); skip when only last pass requested
    _opts = output_options if output_options is not None else DEFAULT_OUTPUT_OPTIONS.copy()
    if _opts.get("html", True) and _opts.get("output_all_passes", True):
        plotter_prelim = Plotter(
            original_file_path,
            params,
            sample_rate,
            output_directory,
            source_audio_path=wav_file_path,
        )
        base_name = os.path.basename(os.path.splitext(original_file_path)[0])
        prelim_html_path = os.path.join(output_directory, f"{base_name}_preliminary.html")
        plotter_prelim.plot_preliminary_save(
            audio_envelope,
            anchor_beats,
            _opts,
            prelim_html_path,
            prelim_analysis_data=prelim_analysis_data,
            prelim_bpm_data=prelim_bpm,
        )

    # STAGE 3: Main Analysis, now informed by the preliminary pass (time-varying BPM prior from canonical curve)
    logging.info("--- STAGE 3: Running Main Analysis Pass ---")
    prelim_bpm_prior = (
        _build_prelim_bpm_prior(prelim_bpm["curve_times"], pd.Series(prelim_bpm["curve_bpm"]))
        if prelim_bpm is not None
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
        band_envelopes,
        prelim_bpm_prior=prelim_bpm_prior,
    )
    s1_peaks, all_raw_peaks, analysis_data = classifier.classify_peaks()

    # Attach band envelopes to analysis_data for plotting (S1/S2 band debug traces)
    if band_envelopes is not None:
        analysis_data["s1_band"] = band_envelopes.get("s1_band")
        analysis_data["s2_band"] = band_envelopes.get("s2_band")

    # Set default output options if none provided (needed for pass1/pass2 plot decisions)
    if output_options is None:
        output_options = DEFAULT_OUTPUT_OPTIONS.copy()
    needs_plot_outputs = any([
        output_options.get('html', True),
        output_options.get('png', False),
        output_options.get('csv', True),
    ])
    plotter = None

    # Pass1 plot: main classifier output (before refinement); skip when only last pass requested
    output_all_passes = output_options.get("output_all_passes", True)
    if needs_plot_outputs and output_all_passes and len(s1_peaks) >= 2:
        plotter = Plotter(
            original_file_path,
            params,
            sample_rate,
            output_directory,
            source_audio_path=wav_file_path,
        )
        metrics_pass1 = _calculate_final_metrics(s1_peaks, sample_rate, params)
        plotter.plot_and_save(
            audio_envelope,
            all_raw_peaks,
            analysis_data,
            metrics_pass1,
            output_options,
            output_suffix="_pass1",
            prelim_bpm_series=pd.Series(prelim_bpm["curve_bpm"]) if prelim_bpm is not None else None,
            prelim_bpm_times=prelim_bpm["curve_times"] if prelim_bpm is not None else None,
        )

    # STAGE 4 & 5: Correction and Refinement
    final_peaks, analysis_data = _refine_and_correct_peaks(
        s1_peaks, all_raw_peaks, analysis_data, audio_envelope, sample_rate, params
    )

    # STAGE 6: Final Reporting
    if len(final_peaks) < 2:
        logging.warning("Not enough S1 peaks detected to generate full report.")
        return None

    logging.info("--- STAGE 6: Calculating Metrics and Generating Outputs ---")
    final_metrics = _calculate_final_metrics(final_peaks, sample_rate, params)

    # OPTIONAL: Validation against manually labeled peaks (if a CSV exists next to the WAV).
    # This lets you batch-run a dataset and get an objective error count per file
    # without changing the main analysis workflow or outputs.
    try:
        manual_labels = _load_manual_labels_csv(original_file_path)
        if manual_labels:
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

    # Pass2 plot: after refinement
    if needs_plot_outputs and len(final_peaks) >= 2:
        if plotter is None:
            plotter = Plotter(
                original_file_path,
                params,
                sample_rate,
                output_directory,
                source_audio_path=wav_file_path,
            )
        plotly_figure = plotter.plot_and_save(
            audio_envelope,
            all_raw_peaks,
            analysis_data,
            final_metrics,
            output_options,
            output_suffix="_pass2",
            prelim_bpm_series=pd.Series(prelim_bpm["curve_bpm"]) if prelim_bpm is not None else None,
            prelim_bpm_times=prelim_bpm["curve_times"] if prelim_bpm is not None else None,
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
            reporter.save_analysis_summary(final_metrics)
        else:
            logging.info("Skipping summary generation as requested.")

        if output_options.get('debug', True):
            reporter.create_chronological_log(audio_envelope, sample_rate, all_raw_peaks, analysis_data, final_metrics)
        else:
            logging.info("Skipping debug log generation as requested.")
    else:
        logging.info("Skipping all report generation as requested.")

    # FFT profiles: aggregate S1/S2 frequency spectra from raw audio (separate minimal HTML)
    fft_aggregate_data = None
    if params.get("enable_fft_profiles", True) and output_options.get("fft_profiles", True):
        try:
            base_name = os.path.basename(os.path.splitext(original_file_path)[0])
            fft_output_path = os.path.join(output_directory, f"{base_name}_fft_profiles.html")
            if collect_fft_for_aggregate:
                target_sr = int(params.get("fft_aggregate_sr", 32000))
                fft_result = compute_fft_profiles(
                    wav_file_path,
                    analysis_data.get("beat_debug_info", {}),
                    sample_rate,
                    audio_envelope,
                    params,
                    target_sr=target_sr,
                )
                save_fft_profiles_html(
                    wav_file_path,
                    analysis_data.get("beat_debug_info", {}),
                    sample_rate,
                    fft_output_path,
                    audio_envelope,
                    params,
                    fft_result=fft_result,
                )
                fft_aggregate_data = fft_result
            else:
                save_fft_profiles_html(
                    wav_file_path,
                    analysis_data.get("beat_debug_info", {}),
                    sample_rate,
                    fft_output_path,
                    audio_envelope,
                    params,
                )
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

    return plotly_figure, fft_aggregate_data
