"""Output layer: runs the analysis engine on one WAV and writes the requested artifacts.

The algorithm itself lives in engine.py (pure computation, no file output). This
module decides what to render and where — Plotly HTML/PNG/CSV per pass, Markdown
summary/debug reports, FFT profile HTML, and debug WAVs — driven by output_options.
"""
import os
import logging
import time
from typing import Dict, Optional, Callable

import numpy as np

from app_settings import DEFAULT_OUTPUT_OPTIONS
from audio_io import write_peak_normalized_debug_wav, write_peak_normalized_wav_native_rate
from pcg.engine.config import param
from pcg.engine.run import STAGE_PASS1, STAGE_PASS2, STAGE_PREPROCESSED, run_analysis
from file_io import output_stem_from_path
from plotting import Plotter, prewarm_kaleido_png_export
from reporting import ReportGenerator
from fft_profiles import (
    compute_fft_profiles,
    compute_frequency_separation,
    save_fft_profiles_html,
)

# Recordings longer than this skip PNG export (Kaleido), but still allow HTML/CSV/etc.
# Independent of optimize_long_plots (that flag only trims heavy traces in Plotter when plots are produced).
LONG_RECORDING_DISABLE_PNG_SEC = 3600.0

# Sample rate of the bandpass debug WAV embedded for HTML playback.
DEBUG_WAV_SAMPLE_RATE = 10000


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


def _debug_wav_writer(original_file_path: str, output_directory: str):
    """Build an engine debug_audio_sink that writes the filtered debug WAVs next to the outputs."""
    base_name = output_stem_from_path(original_file_path)

    def _sink(kind: str, signal: np.ndarray, sample_rate: int) -> None:
        if kind == "filtered":
            path = os.path.join(output_directory, f"{base_name}_filtered_debug.wav")
            try:
                write_peak_normalized_debug_wav(path, signal, sample_rate, DEBUG_WAV_SAMPLE_RATE)
                logging.info(
                    "Saved filtered audio WAV debug file (%s, %d Hz, int16) for HTML playback.",
                    path,
                    DEBUG_WAV_SAMPLE_RATE,
                )
            except Exception as e:
                logging.error("Failed to write filtered debug WAV file %s: %s", path, e)
        elif kind == "filtered_inverse":
            path = os.path.join(output_directory, f"{base_name}_filtered_inverse_debug.wav")
            try:
                write_peak_normalized_wav_native_rate(path, signal, sample_rate)
                logging.info("Saved inverse-band debug WAV (%s, %d Hz, int16).", path, sample_rate)
            except Exception as e:
                logging.error("Failed to write inverse-band debug WAV file %s: %s", path, e)

    return _sink


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
    """Run the engine on one WAV and write the outputs selected in output_options.

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

    try:
        return _analyze_and_write(
            wav_file_path,
            params,
            start_bpm_hint,
            original_file_path,
            output_directory,
            output_options,
            collect_fft_for_aggregate,
            _ui,
        )
    finally:
        # Remove filters so this setting is scoped to the analysis call.
        for handler, filt in active_filters:
            try:
                handler.removeFilter(filt)
            except Exception:
                pass


def _analyze_and_write(
    wav_file_path: str,
    params: Dict,
    start_bpm_hint: Optional[float],
    original_file_path: str,
    output_directory: str,
    requested_output_options: Optional[Dict],
    collect_fft_for_aggregate: bool,
    _ui: Callable[[str], None],
):
    start_time = time.time()
    logging.info("--- Processing file: %s ---", os.path.basename(original_file_path))

    # Debug WAVs: a caller-supplied dict without "filtered_wav" means "write them";
    # only a missing dict falls back to DEFAULT_OUTPUT_OPTIONS.
    wav_opts = requested_output_options if requested_output_options is not None else DEFAULT_OUTPUT_OPTIONS
    debug_audio_sink = None
    if param(params, "save_filtered_wav"):
        if wav_opts.get("filtered_wav", True):
            debug_audio_sink = _debug_wav_writer(original_file_path, output_directory)
        else:
            logging.info("Skipping filtered audio WAV generation as requested.")

    output_options = {**DEFAULT_OUTPUT_OPTIONS, **(requested_output_options or {})}
    output_all_passes = output_options.get("output_all_passes", True)

    def _wants_plots() -> bool:
        # Evaluated at use: the long-recording guard may switch PNG off after preprocessing.
        return any([
            output_options.get('html', True),
            output_options.get('png', False),
            output_options.get('csv', True),
        ])

    preprocessed: Dict = {}  # sample_rate / algorithm_envelope, filled at STAGE_PREPROCESSED

    def _new_plotter() -> Plotter:
        return Plotter(
            original_file_path,
            params,
            preprocessed["sample_rate"],
            output_directory,
            source_audio_path=wav_file_path,
        )

    def _on_stage(stage: str, data: Dict) -> None:
        if stage == STAGE_PREPROCESSED:
            preprocessed["sample_rate"] = data["sample_rate"]
            preprocessed["algorithm_envelope"] = data["algorithm_envelope"]
            duration_sec = data["duration_sec"]
            if duration_sec > LONG_RECORDING_DISABLE_PNG_SEC:
                output_options["png"] = False
                logging.info(
                    "Recording length %.1f min exceeds %.0f min — disabling PNG export (Kaleido).",
                    duration_sec / 60.0,
                    LONG_RECORDING_DISABLE_PNG_SEC / 60.0,
                )
            # Pre-warm Kaleido so Chromium startup can overlap with analysis.
            try:
                if output_options.get("png", False):
                    prewarm_kaleido_png_export()
            except Exception:
                pass

        elif stage == STAGE_PASS1:
            # Pass 1 plot (envelope + anchor beats + BPM scatter/curve + BPM Trend (Belief)); skip when only last pass requested
            if output_options.get("html", True) and output_options.get("output_all_passes", True):
                _ui("Generating pass 1 HTML report...")
                base_name = output_stem_from_path(original_file_path)
                pass1_html_path = os.path.join(output_directory, f"{base_name}_pass1.html")
                _new_plotter().plot_pass1_save(
                    preprocessed["algorithm_envelope"],
                    data["anchor_beats"],
                    output_options,
                    pass1_html_path,
                    pass1_analysis_data=data["analysis_data"],
                    pass1_bpm_data=data["pass1_bpm"],
                )

        elif stage == STAGE_PASS2:
            if output_all_passes and _wants_plots():
                _ui("Pass 2: saving HTML / PNG / CSV...")
                pass1_bpm = data["pass1_bpm"]
                _new_plotter().plot_and_save(
                    preprocessed["algorithm_envelope"],
                    data["all_raw_peaks"],
                    data["analysis_data"],
                    data["metrics"],
                    output_options,
                    output_suffix="_pass2",
                    pass1_bpm_series=np.asarray(pass1_bpm["curve_bpm"], dtype=np.float64) if pass1_bpm is not None else None,
                    pass1_bpm_times=np.asarray(pass1_bpm["curve_times"], dtype=np.float64) if pass1_bpm is not None else None,
                    is_final_pass=False,
                )

    result = run_analysis(
        wav_file_path,
        params,
        start_bpm_hint,
        progress_callback=_ui,
        on_stage=_on_stage,
        debug_audio_sink=debug_audio_sink,
        # Pass 2 metrics feed the pass 2 plot and the prior curve on the final plot.
        compute_pass2_metrics=_wants_plots(),
    )

    # STAGE 6: Metrics from latest pass (pass3, or pass4 when enabled).
    if not result.ok:
        logging.warning("Not enough S1 peaks detected to generate full report.")
        _ui("Stopped: not enough detected heartbeat peaks.")
        return None, None, None, None

    logging.info("--- STAGE 6: Calculating Metrics and Generating Outputs ---")

    analysis_data = result.analysis_data
    metrics = result.metrics
    algorithm_envelope = result.algorithm_envelope
    sample_rate = result.sample_rate
    plotly_figure = None

    # Pass 3/4 plot: after refinement (prior curve = BPM from pass 2, else pass 1)
    if _wants_plots():
        _ui("Pass 3: saving HTML / PNG / CSV...")
        prior_bpm_series = None
        prior_bpm_times = None
        metrics_pass2 = result.metrics_pass2
        if metrics_pass2 is not None and metrics_pass2.get("smoothed_bpm") is not None and len(metrics_pass2["smoothed_bpm"]) > 0:
            prior_bpm_series = metrics_pass2["smoothed_bpm"]
            prior_bpm_times = metrics_pass2.get("bpm_times")
        if prior_bpm_series is None and result.pass1_bpm is not None:
            prior_bpm_series = np.asarray(result.pass1_bpm["curve_bpm"], dtype=np.float64)
            prior_bpm_times = np.asarray(result.pass1_bpm["curve_times"], dtype=np.float64)
        # Pass 3 plot: include peak/recovery times for systolic shift (exertion vs all-time averaging)
        metrics["peak_bpm_time_sec"] = result.peak_bpm_time_sec
        metrics["recovery_end_time_sec"] = result.recovery_end_time_sec
        plotly_figure = _new_plotter().plot_and_save(
            algorithm_envelope,
            result.all_raw_peaks,
            analysis_data,
            metrics,
            output_options,
            output_suffix="_pass3",
            filename_suffix="_pass3" if output_all_passes else "_bpm_plot",
            pass1_bpm_series=prior_bpm_series,
            pass1_bpm_times=prior_bpm_times,
            is_final_pass=True,
        )
    else:
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
            reporter.save_analysis_summary(metrics)
        else:
            logging.info("Skipping summary generation as requested.")

        if output_options.get('debug', True):
            _ui("Writing debug log (Markdown)...")
            reporter.create_chronological_log(algorithm_envelope, sample_rate, result.all_raw_peaks, analysis_data, metrics)
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
            peak_classifications = analysis_data.get("peak_classifications", {})
            fft_kwargs = {"target_sr": int(param(params, "fft_aggregate_sr"))} if collect_fft_for_aggregate else {}
            fft_result = compute_fft_profiles(
                wav_file_path,
                peak_classifications,
                sample_rate,
                algorithm_envelope,
                params,
                **fft_kwargs,
            )
            save_fft_profiles_html(
                wav_file_path,
                peak_classifications,
                sample_rate,
                fft_output_path,
                algorithm_envelope,
                params,
                fft_result=fft_result,
            )
            if collect_fft_for_aggregate:
                fft_aggregate_data = fft_result
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

    return plotly_figure, fft_aggregate_data, result.bpm_summary, analysis_data
