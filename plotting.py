import os
import logging
import urllib.parse
import re
from time_utils import seconds_to_datetime
import csv
import shutil
import json
from typing import Dict, Optional, List, Any, Tuple
from peak_utils import PeakType, _get_peak_type_from_debug, format_debug_entry, get_peak_prominence_details
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import DEFAULT_OUTPUT_OPTIONS
from file_io import find_companion_wav, normalize_output_filename_stem, output_stem_from_path
from confidence_engine import calculate_bpm_intervals

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for spectrogram generation
import matplotlib.pyplot as plt
from config import param


# --- Trace audience registry (single source of truth for the HTML "Show:" filter) ---
# Maps every plot trace name to the audience it serves:
#   "analysis" = end-user derived results (BPM/HRV/contractility/final systole+diastole)
#   "debug"    = raw signal + per-peak reasoning (envelopes, markers, scores, intermediates)
#   "both"     = always visible regardless of view (the BPM result curves)
# This dict is injected into the HTML as JSON; interactive_plot.js / html_inline_minimal.js
# read it instead of hardcoding name lists. Any trace name NOT listed here is treated as
# "debug" by the JS (fail-safe: unknown/new traces never leak into the Analysis Data view).
# tests/test_trace_audience.py guards that every static trace name in this file is registered.
TRACE_AUDIENCE: Dict[str, str] = {
    # both (always visible) — the BPM result curve, named per pass
    "Average BPM": "both",
    "BPM (Pass 2)": "both",
    "BPM (Pass 3)": "both",
    # analysis (end-user derived results)
    "Average S1 contractility": "analysis",
    "Average S2 contractility": "analysis",
    "Average contractility": "analysis",
    "RMSSDc": "analysis",
    "SDNN": "analysis",
    "LF/HF (windowed)": "analysis",
    "Measured systole curve (final)": "analysis",
    "Measured diastole curve (final)": "analysis",
    "Exertion": "analysis",
    "Recovery": "analysis",
    "Peak Recovery Slope": "analysis",
    "Peak Exertion Slope": "analysis",
    # debug — raw signal / envelopes
    "Bandpass Envelope": "debug",
    "Noise Removed Envelope": "debug",
    "Noise Envelope": "debug",
    "Dynamic Noise Floor": "debug",
    # debug — peak/beat markers
    "Troughs": "debug",
    "S1 Beats": "debug",
    "S2 Beats": "debug",
    "Noise/Rejected": "debug",
    "Recovered peaks at large gaps (insensitive)": "debug",
    "Recovered peaks at large gaps (sensitive)": "debug",
    # debug — per-peak classifier reasoning
    "S1 score": "debug",
    "S2 score": "debug",
    "Noise score": "debug",
    # debug — reference / intermediate phase-interval curves
    "Expected systole from BPM": "debug",
    "Expected diastole from BPM": "debug",
    "Measured systole": "debug",
    "Measured diastole": "debug",
    "Measured systole curve (before repair)": "debug",
    "Measured diastole curve (before repair)": "debug",
    # debug — BPM belief / raw instantaneous BPM (dynamic per-pass names enumerated)
    "BPM Trend (Belief)": "debug",
    "Instant BPM (Pass 2)": "debug",
    "Instant BPM (Pass 2) outliers removed": "debug",
    "Instantaneous BPM (Pass 2)": "debug",
    "Instant BPM (Pass 3)": "debug",
    "Instant BPM (Pass 3) outliers removed": "debug",
    "Instantaneous BPM (Pass 3)": "debug",
    # debug — pass 1 plot (whole plot is diagnostic; no selector is shown there)
    "Anchor beats": "debug",
    "Instant BPM (Pass 1)": "debug",
    "Instant BPM (Pass 1) outliers removed": "debug",
    "BPM (pass 1)": "debug",
    # internal invisible axis anchor (kept visible in every view so the BPM axis renders)
    "_axis_anchor": "both",
}

# Trace names produced dynamically (f-strings) rather than as static literals; listed so the
# guard test can verify them too. Static-literal names are scraped from the source directly.
_DYNAMIC_TRACE_NAMES = (
    "Instant BPM (Pass 2)",
    "Instant BPM (Pass 2) outliers removed",
    "Instantaneous BPM (Pass 2)",
    "Instant BPM (Pass 3)",
    "Instant BPM (Pass 3) outliers removed",
    "Instantaneous BPM (Pass 3)",
)


def _category_filter_html(default_view: str) -> str:
    """Build the toolbar 'Show:' category <select>, pre-selecting default_view.
    Returns '' so callers can omit the control entirely on non-final-pass plots."""
    options = (("all", "All"), ("debug", "Debug"), ("analysis", "Analysis Data"))
    options_html = "".join(
        f'<option value="{value}"{" selected" if value == default_view else ""}>{label}</option>'
        for value, label in options
    )
    return (
        '<label for="legend-category-filter" class="chart-toolbar-label">Show:</label>'
        '<select id="legend-category-filter" title="Filter legend and visible traces by category">'
        f"{options_html}</select>"
    )


def _elapsed_seconds_to_plot_datetimes(seconds: np.ndarray) -> np.ndarray:
    """
    Vectorized equivalent of
    pd.to_datetime([seconds_to_datetime(float(t)) for t in seconds]).
    Same local-epoch convention as time_utils.seconds_to_datetime (for Plotly x).

    Returns numpy datetime64[ns] (not pandas DatetimeIndex) so figure JSON for
    Kaleido PNG export does not contain non-JSON-serializable pandas.Timestamp.
    """
    arr = np.asarray(seconds, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.array([], dtype="datetime64[ns]")
    base = pd.Timestamp(seconds_to_datetime(0.0))
    idx = base + pd.to_timedelta(arr, unit="s")
    return idx.to_numpy(dtype="datetime64[ns]")


def _bpm_axis_times_to_plot_x_coords(times_like: List[Any]) -> np.ndarray:
    """
    Map HRV slope dict times to the same numpy datetime64 x-axis as other traces.
    Values may be elapsed seconds (float) or datetime-like (from smoothed_bpm index);
    pandas.Timestamp is not JSON-safe for Kaleido unless converted here.
    """
    origin = seconds_to_datetime(0.0)
    secs: List[float] = []
    for t in times_like:
        if isinstance(t, (int, float, np.integer, np.floating)) and not isinstance(t, bool):
            secs.append(float(t))
        else:
            secs.append(float((pd.Timestamp(t) - pd.Timestamp(origin)).total_seconds()))
    return _elapsed_seconds_to_plot_datetimes(np.asarray(secs, dtype=np.float64))


def _json_for_html_inline_script(json_str: str) -> str:
    """
    If config JSON is embedded literally inside <script>...</script>, any '</script>'
    substring (case-insensitive) ends the HTML script element early and breaks parsing.
    Break the token with a backslash so the browser keeps the script open; JSON still
    decodes the same when assigned as a JS object literal (same as json.dumps output).
    """
    return re.sub(
        r"(?i)</script\s*>",
        lambda m: "<\\/" + m.group(0)[2:],
        json_str,
    )


def prewarm_kaleido_png_export() -> None:
    """
    Start Kaleido's persistent sync server (Kaleido >= 1.1) so Chromium stays
    alive across multiple fig.write_image() calls. Plotly routes write_image to
    kaleido.calc_fig_sync, which uses the global server when it is running.

    Safe to call multiple times. No-op if Kaleido is missing or too old.
    Kaleido registers its own atexit handler to stop the server on shutdown.
    """
    try:
        logging.debug("Kaleido: prewarm_kaleido_png_export()")
        import kaleido  # noqa: PLC0415

        if not hasattr(kaleido, "start_sync_server"):
            logging.debug("Kaleido: start_sync_server missing (install kaleido>=1.1 for persistent export)")
            return
        # Align with plotly.io.defaults so the server loads the same plotly.js /
        # MathJax as Plotly would pass via kopts on one-shot exports (kopts are
        # ignored per call while the server is running).
        kw = {"silence_warnings": True}
        try:
            import plotly.io as pio  # noqa: PLC0415

            d = pio.defaults
            if getattr(d, "plotlyjs", None):
                kw["plotlyjs"] = d.plotlyjs
            if getattr(d, "mathjax", None):
                kw["mathjax"] = d.mathjax
        except Exception:
            pass
        try:
            from kaleido import _sync_server as _kaleido_sync  # noqa: PLC0415

            _kaleido_was_running = _kaleido_sync.GlobalKaleidoServer().is_running()
        except Exception:
            _kaleido_was_running = False
        kaleido.start_sync_server(**kw)
        if _kaleido_was_running:
            logging.debug("Kaleido: sync server already running (batch — reuse persistent Chromium)")
        else:
            logging.debug("Kaleido: start_sync_server() started (persistent Chromium for PNG export)")
    except Exception:
        logging.debug(
            "Kaleido: prewarm failed (one-shot export per write_image)",
            exc_info=True,
        )


def _compute_systolic_interval_data(
    analysis_data: Dict,
    pass_metrics: Dict,
    sample_rate: int,
    params: Dict,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Compute measured systole intervals from labeled pairs and BPM-expected systolic curve.
    Returns (observed_times, observed_intervals, expected_times, expected_intervals).
    """
    # Measured systole intervals from labeled pairs
    pairs = analysis_data.get("s1_s2_pairs") or []
    observed_times: List[float] = []
    observed_intervals: List[float] = []
    for s1_idx, s2_idx in pairs:
        t = (s1_idx + s2_idx) / 2.0 / sample_rate
        interval = (s2_idx - s1_idx) / sample_rate
        observed_times.append(t)
        observed_intervals.append(interval)

    # BPM-expected systolic curve from pass metrics
    smoothed_bpm = pass_metrics.get("smoothed_bpm")
    bpm_times = pass_metrics.get("bpm_times")
    expected_times: List[float] = []
    expected_intervals: List[float] = []
    if smoothed_bpm is not None and bpm_times is not None and len(bpm_times) == len(smoothed_bpm):
        for t, bpm in zip(np.asarray(bpm_times, dtype=float), np.asarray(smoothed_bpm, dtype=float)):
            intervals = calculate_bpm_intervals(float(bpm), params)
            expected_times.append(float(t))
            expected_intervals.append(intervals["s1_s2_nominal"])

    return observed_times, observed_intervals, expected_times, expected_intervals


def _compute_measured_systole_from_state_boundaries(
    analysis_data: Dict,
    sample_rate: int,
    *,
    prefer: str = "after",
) -> Tuple[List[float], List[float]]:
    """
    Compute measured systole durations from Pass 3 dense state boundaries.
    Returns (times, durations) in seconds, where time is the segment midpoint.
    """
    if prefer == "before":
        boundaries = analysis_data.get("pass3_state_boundaries_before") or []
    else:
        boundaries = analysis_data.get("pass3_state_boundaries") or []
        if not boundaries:
            boundaries = analysis_data.get("pass3_state_boundaries_before") or []

    measured_times: List[float] = []
    measured_durations: List[float] = []
    for s0, s1, state_name, _meta in (boundaries or []):
        try:
            if str(state_name).lower() != "systole":
                continue
            s0i = int(s0)
            s1i = int(s1)
            if s1i <= s0i:
                continue
            t_mid = (s0i + s1i) / 2.0 / float(sample_rate)
            dur = (s1i - s0i) / float(sample_rate)
            if not np.isfinite(t_mid) or not np.isfinite(dur):
                continue
            measured_times.append(float(t_mid))
            measured_durations.append(float(dur))
        except Exception:
            continue

    return measured_times, measured_durations


def _compute_measured_diastole_from_state_boundaries(
    analysis_data: Dict,
    sample_rate: int,
    *,
    prefer: str = "after",
) -> Tuple[List[float], List[float]]:
    """
    Compute measured diastole durations from Pass 3 dense state boundaries.
    Returns (times, durations) in seconds, where time is the segment midpoint.
    """
    if prefer == "before":
        boundaries = analysis_data.get("pass3_state_boundaries_before") or []
    else:
        boundaries = analysis_data.get("pass3_state_boundaries") or []
        if not boundaries:
            boundaries = analysis_data.get("pass3_state_boundaries_before") or []

    measured_times: List[float] = []
    measured_durations: List[float] = []
    for s0, s1, state_name, _meta in (boundaries or []):
        try:
            if str(state_name).lower() != "diastole":
                continue
            s0i = int(s0)
            s1i = int(s1)
            if s1i <= s0i:
                continue
            t_mid = (s0i + s1i) / 2.0 / float(sample_rate)
            dur = (s1i - s0i) / float(sample_rate)
            if not np.isfinite(t_mid) or not np.isfinite(dur):
                continue
            measured_times.append(float(t_mid))
            measured_durations.append(float(dur))
        except Exception:
            continue

    return measured_times, measured_durations


def _compute_expected_diastole_from_bpm(
    pass_metrics: Dict,
    params: Dict,
) -> Tuple[List[float], List[float]]:
    """
    Compute expected diastole (S2→next S1) from BPM over time.
    Returns (times, durations) in seconds.
    """
    smoothed_bpm = pass_metrics.get("smoothed_bpm")
    bpm_times = pass_metrics.get("bpm_times")
    expected_times: List[float] = []
    expected_durations: List[float] = []
    if smoothed_bpm is not None and bpm_times is not None and len(bpm_times) == len(smoothed_bpm):
        for t, bpm in zip(np.asarray(bpm_times, dtype=float), np.asarray(smoothed_bpm, dtype=float)):
            intervals = calculate_bpm_intervals(float(bpm), params)
            expected_times.append(float(t))
            expected_durations.append(float(intervals["s2_s1_nominal"]))
    return expected_times, expected_durations


def _compute_systolic_shift(
    obs_t: List[float],
    obs_iv: List[float],
    exp_t: List[float],
    exp_iv: List[float],
    peak_bpm_time_sec: Optional[float],
) -> Optional[float]:
    """
    Compute shift to align expected systole curve to measured data.
    If peak_bpm_time_sec: use exertion only (t < peak). Else: average across all time.
    Returns shift (measured_avg - expected_avg) or None if insufficient data.
    """
    if not obs_t or not obs_iv or not exp_t or not exp_iv or len(exp_t) < 2:
        return None
    obs_t_arr = np.array(obs_t, dtype=float)
    obs_iv_arr = np.array(obs_iv, dtype=float)
    exp_t_arr = np.array(exp_t, dtype=float)
    exp_iv_arr = np.array(exp_iv, dtype=float)

    if peak_bpm_time_sec is not None:
        mask = obs_t_arr < peak_bpm_time_sec
    else:
        mask = np.ones(len(obs_t_arr), dtype=bool)

    if not np.any(mask):
        return None

    measured_avg = float(np.mean(obs_iv_arr[mask]))
    expected_at_measured = np.interp(obs_t_arr[mask], exp_t_arr, exp_iv_arr)
    expected_avg = float(np.mean(expected_at_measured))
    return measured_avg - expected_avg


class Plotter:
    """Handles the creation and generation of the final analysis plot."""

    def __init__(
        self,
        file_name: str,
        params: Dict,
        sample_rate: int,
        output_directory: str,
        source_audio_path: Optional[str] = None,
    ):
        self.file_name = file_name
        self.output_stem = output_stem_from_path(file_name)
        self.params = params
        self.sample_rate = sample_rate
        self.output_directory = output_directory
        self.audio_source_path = source_audio_path or file_name
        self.fig = make_subplots(specs=[[{"secondary_y": True}]])
        self.audio_duration_sec = None  # Will be set during plot_and_save
        # Optional spectrogram image filenames (saved in output dir); filtered generated on demand.
        self.spectrogram_original_filename: Optional[str] = None
        self.bpm_axis_center: float = float(params.get("default_bpm_axis_center", 125))
        self.bpm_axis_span: float = float(params.get("bpm_axis_span", 150))

    def _generate_spectrogram_image(self, audio_path: str, output_path: str) -> Optional[str]:
        """
        Generate a spectrogram image from the audio file and save as PNG to output_path.
        Returns the basename of the saved file (for use in HTML/config), or None on failure.
        """
        try:
            # Load audio at a reasonable sample rate for spectrogram
            audio_data, sr = librosa.load(audio_path, sr=22050, mono=True)

            if audio_data is None or len(audio_data) == 0:
                logging.warning("Could not load audio for spectrogram generation")
                return None

            # Compute mel spectrogram for better visual representation
            n_fft = 2048
            hop_length = 128
            n_mels = 256  # Slightly smaller for reasonable file size

            # Generate mel spectrogram
            S = librosa.feature.melspectrogram(
                y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
            )

            # Convert to dB scale
            S_dB = librosa.power_to_db(S, ref=np.max)

            # Calculate figure dimensions based on audio duration; cap width to limit file size
            duration = len(audio_data) / sr
            fig_width = min(max(20, duration / 10), 80)
            fig_height = 6

            # Create figure with transparent background
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)

            # Display spectrogram with a colormap that works well as background
            librosa.display.specshow(
                S_dB,
                sr=sr,
                hop_length=hop_length,
                x_axis="time",
                y_axis="mel",
                ax=ax,
                cmap="magma",
            )

            # Remove axes, labels, and all decorations for clean overlay
            ax.axis("off")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")

            # Remove all margins
            plt.tight_layout(pad=0)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # Save to file as PNG with transparency (dpi 72 for smaller file size)
            fig.savefig(
                output_path,
                format="png",
                transparent=True,
                dpi=72,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close(fig)

            basename = os.path.basename(output_path)
            logging.info("Generated spectrogram image for background overlay: %s", basename)
            return basename

        except Exception as e:
            logging.warning("Failed to generate spectrogram image: %s", e)
            return None

    def plot_and_save(
        self,
        audio_envelope: np.ndarray,
        all_raw_peaks: np.ndarray,
        analysis_data: Dict,
        pass_metrics: Dict,
        output_options: Optional[Dict] = None,
        output_suffix: Optional[str] = None,
        filename_suffix: Optional[str] = None,
        pass1_bpm_series: Optional[pd.Series] = None,
        pass1_bpm_times: Optional[np.ndarray] = None,
        is_final_pass: bool = True,
    ):
        """Generates and saves the main analysis plot by calling helper methods.
        is_final_pass: True only for the last pass plotted (currently Pass 3). The Debug/Analysis Data
        category selector is shown only on the final pass; per-pass output is itself a debug feature,
        so non-final passes render every trace with no selector.
        pass_metrics: BPM/HRV/slope metrics for the pass being plotted (pass 2, pass 3, etc.).
        output_suffix: pass id for trace labels and systolic logic ('_pass2', '_pass3').
        filename_suffix: if set, used for HTML/PNG/CSV file names; if None, uses output_suffix or '_bpm_plot'.
        """
        self.fig = make_subplots(specs=[[{"secondary_y": True}]])
        self.time_axis_sec = np.arange(len(audio_envelope)) / self.sample_rate
        self.audio_duration_sec = self.time_axis_sec[-1] if len(self.time_axis_sec) > 0 else 0
        self._has_systolic_traces = False
        # Optional pass 3 dense state timeline (S1/systole/S2/diastole) for HTML overlay.
        self._pass3_state_boundaries = analysis_data.get("pass3_state_boundaries") or []
        self._pass3_state_boundaries_before = analysis_data.get("pass3_state_boundaries_before") or []
        self._pass3_state_labels_encoding = analysis_data.get("pass3_state_labels_encoding") or {}
        self._noise_event_segments = analysis_data.get("noise_event_segments") or []
        self._pass3_noise_unreliable_windows_samples = analysis_data.get("pass3_noise_unreliable_windows_samples") or []
        self._pass3_large_gap_windows_samples = analysis_data.get("pass3_large_gap_windows_samples") or []
        self._pass3_gap_quiet_windows_samples = analysis_data.get("pass3_gap_quiet_windows_samples") or []
        self._pass3_large_gap_recovered_peaks_insensitive = analysis_data.get("pass3_large_gap_recovered_peaks_insensitive") or []
        self._pass3_large_gap_recovered_peaks_sensitive = analysis_data.get("pass3_large_gap_recovered_peaks_sensitive") or []
        self._pass3_gap_decision_peaks_sensitive = analysis_data.get("pass3_gap_decision_peaks_sensitive") or []

        # Long-plot optimization: optionally skip heavy debug traces for very long recordings.
        optimize_long_plots = bool(param(self.params, "optimize_long_plots"))
        long_threshold_sec = float(param(self.params, "long_plot_duration_threshold_sec"))
        # Only skip details if the recording is longer than the threshold; shorter files always show full detail.
        self.skip_detailed_debug_traces = optimize_long_plots and self.audio_duration_sec > long_threshold_sec

        self._add_line_traces(audio_envelope, analysis_data, all_raw_peaks)
        self._add_trough_markers(audio_envelope, analysis_data)
        self._add_peak_traces(
            all_raw_peaks,
            analysis_data.get("peak_classifications", {}),
            audio_envelope,
            analysis_data.get("trough_indices"),
        )
        self._add_pass3_large_gap_recovered_peak_markers(audio_envelope)
        self._add_bpm_hrv_traces(
            pass_metrics.get("smoothed_bpm"),
            analysis_data,
            pass_metrics.get("windowed_hrv_df"),
            output_suffix=output_suffix,
            pass1_bpm_series=pass1_bpm_series,
            pass1_bpm_times=pass1_bpm_times,
            instant_bpm=pass_metrics.get("instant_bpm"),
            bpm_times=pass_metrics.get("bpm_times"),
            instant_bpm_raw=pass_metrics.get("instant_bpm_raw"),
            bpm_times_raw=pass_metrics.get("bpm_times_raw"),
        )
        self._add_systolic_interval_traces(
            analysis_data, pass_metrics, output_suffix,
        )
        self._add_slope_traces(
            pass_metrics.get("major_inclines"),
            pass_metrics.get("major_declines"),
            pass_metrics.get("peak_recovery_stats"),
            pass_metrics.get("peak_exertion_stats"),
        )
        self._add_annotations_and_summary(
            pass_metrics.get("bpm_times"),
            pass_metrics.get("smoothed_bpm"),
            pass_metrics.get("hrv_summary"),
            pass_metrics.get("hrr_stats"),
            pass_metrics.get("peak_recovery_stats"),
            pass_metrics.get("bpm_failure_report"),
        )
        self._prepare_bpm_axis_center(pass_metrics)

        self._configure_layout()

        base_name = self.output_stem
        file_suffix = (
            filename_suffix
            if filename_suffix is not None
            else (output_suffix if output_suffix is not None else "_bpm_plot")
        )
        output_html_path = os.path.join(self.output_directory, f"{base_name}{file_suffix}.html")
        output_png_path = os.path.join(self.output_directory, f"{base_name}{file_suffix}.png")
        plot_title = f"Heartbeat Analysis - {os.path.basename(self.file_name)}"
        plot_config = {
            "scrollZoom": True,
            "toImageButtonOptions": {"filename": plot_title, "format": "png", "scale": 2},
            "showTips": False,
        }

        html_requested = True if output_options is None else output_options.get("html", True)
        png_requested = False if output_options is None else output_options.get("png", False)

        if html_requested:
            # Determine whether spectrogram generation is enabled (can be disabled via GUI/output options).
            self.spectrogram_enabled = True
            if output_options is not None:
                self.spectrogram_enabled = output_options.get("spectrogram", True)

            # Generate spectrogram image for optional background overlay (original audio only).
            # Filtered spectrograms are generated later in _generate_custom_html if needed.
            if self.spectrogram_enabled:
                try:
                    spec_path = os.path.join(self.output_directory, f"{base_name}_spectrogram.png")
                    self.spectrogram_original_filename = self._generate_spectrogram_image(
                        self.audio_source_path or self.file_name, spec_path
                    )
                except Exception as e:
                    logging.warning("Failed to generate original spectrogram: %s", e)
            else:
                logging.info("Skipping original spectrogram generation as requested (spectrogram output disabled).")

            # Generate the base Plotly HTML
            plotly_html = self.fig.to_html(config=plot_config, full_html=False, include_plotlyjs='cdn')
            # If CDN is unavailable, fall back to a local plotly.min.js beside the HTML (if present).
            plotly_html = re.sub(
                r'<script\s+src="(https://cdn\.plot\.ly/plotly[^"]+)"\s*></script>',
                r'<script src="\1" onerror="this.onerror=null;this.src=\'plotly.min.js\';"></script>',
                plotly_html,
                count=1,
            )

            # Generate custom HTML with audio player and playhead
            custom_html = self._generate_custom_html(
                plotly_html, plot_title, base_name, output_options=output_options,
                is_final_pass=is_final_pass,
            )

            with open(output_html_path, 'w', encoding='utf-8') as f:
                f.write(custom_html)
            logging.info("Interactive plot with audio player saved to %s", output_html_path)
        else:
            logging.info("Skipping HTML plot generation as requested.")

        if png_requested:
            # Use a large default canvas so the graph itself is comfortably sized in the PNG.
            opts = output_options or {}
            png_scale = int(opts.get("png_scale", 2) or 2)
            png_width = int(opts.get("png_width") or 2100)
            png_height = int(opts.get("png_height") or 1200)
            write_kwargs = {
                "format": "png",
                "scale": png_scale,
                "width": png_width,
                "height": png_height,
            }
            try:
                self.fig.write_image(output_png_path, **write_kwargs)
                logging.info("Plot PNG exported via Kaleido to %s", output_png_path)
            except Exception as e:
                logging.warning("Failed to export Plot PNG (requires kaleido): %s", e)
                logging.debug("Kaleido PNG export traceback (set log level DEBUG for details)", exc_info=True)

        if output_options is None or output_options.get("csv", True):
            smoothed_bpm = pass_metrics.get("smoothed_bpm")
            bpm_times = pass_metrics.get("bpm_times")
            if smoothed_bpm is not None and bpm_times is not None and len(bpm_times) == len(smoothed_bpm) and len(bpm_times) > 0:
                csv_path = os.path.join(self.output_directory, f"{base_name}{file_suffix}.csv")
                csv_bpm_header = (
                    "BPM (Pass 2)"
                    if output_suffix == "_pass2"
                    else "BPM (Pass 3)"
                    if output_suffix == "_pass3"
                    else "Average BPM"
                )
                try:
                    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["Time (s)", csv_bpm_header])
                        for t, bpm in zip(np.asarray(bpm_times, dtype=float), np.asarray(smoothed_bpm, dtype=float)):
                            if not np.isnan(bpm):
                                writer.writerow([f"{t:.3f}", f"{bpm:.3f}"])
                    logging.info("BPM plot data saved to %s", csv_path)
                except Exception as e:
                    logging.error("Failed to write BPM plot CSV: %s", e)
        else:
            logging.info("Skipping CSV generation as requested.")

        return self.fig

    def plot_pass1_save(
        self,
        audio_envelope: np.ndarray,
        anchor_beats: np.ndarray,
        output_options: Optional[Dict] = None,
        output_html_path: Optional[str] = None,
        pass1_analysis_data: Optional[Dict] = None,
        pass1_bpm_data: Optional[Dict] = None,
    ):
        """
        Builds and saves the pass 1 plot: envelope, anchor beats, BPM scatter + curve (canonical, same as algorithm), and BPM Trend (Belief).
        pass1_bpm_data: dict from compute_pass1_bpm_curve: raw_scatter_* (instant BPM), scatter_* (outlier-filtered), curve_* (Gaussian smoothing on filtered).
        """
        self.time_axis_sec = np.arange(len(audio_envelope), dtype=float) / self.sample_rate
        self.audio_duration_sec = float(self.time_axis_sec[-1]) if len(self.time_axis_sec) > 0 else 0.0
        base_name = self.output_stem
        if output_html_path is None:
            output_html_path = os.path.join(self.output_directory, f"{base_name}_pass1.html")

        html_requested = True if output_options is None else output_options.get("html", True)
        if not html_requested:
            logging.info("Skipping pass 1 HTML as requested.")
            return

        self.spectrogram_enabled = False
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        optimize_long_plots_pass1 = bool(param(self.params, "optimize_long_plots"))
        long_plot_threshold_sec = float(param(self.params, "long_plot_duration_threshold_sec"))
        skip_heavy_pass1_traces = optimize_long_plots_pass1 and self.audio_duration_sec > long_plot_threshold_sec
        if skip_heavy_pass1_traces:
            logging.info("Pass 1 HTML: skipping envelope and anchor beat traces (optimize_long_plots on long recording).")

        if not skip_heavy_pass1_traces:
            factor = param(self.params, "plot_downsample_factor")
            n = len(audio_envelope)
            if factor > 1 and n >= factor:
                plot_secs = np.arange(0, n, factor, dtype=np.float64) / self.sample_rate
                plot_envelope = audio_envelope[::factor]
            else:
                plot_secs = np.arange(n, dtype=np.float64) / self.sample_rate
                plot_envelope = audio_envelope
            plot_time = _elapsed_seconds_to_plot_datetimes(plot_secs)
            use_nr_main = (pass1_analysis_data or {}).get("noise_removed_envelope") is not None
            main_env_name = "Noise Removed Envelope" if use_nr_main else "Bandpass Envelope"
            fig.add_trace(
                go.Scatter(x=plot_time, y=plot_envelope, name=main_env_name, line=dict(color="#47a5c4")),
                secondary_y=False,
            )

            bp_pass1 = (pass1_analysis_data or {}).get("bandpass_envelope")
            if (
                use_nr_main
                and bp_pass1 is not None
                and isinstance(bp_pass1, np.ndarray)
                and len(bp_pass1) == n
            ):
                plot_bp = bp_pass1[::factor] if factor > 1 and n >= factor else bp_pass1
                fig.add_trace(
                    go.Scatter(
                        x=plot_time,
                        y=plot_bp,
                        name="Bandpass Envelope",
                        line=dict(color="#3498db", width=1.25, dash="dot"),
                        visible="legendonly",
                        hovertemplate="Bandpass Envelope: %{y:.4f}<extra></extra>",
                    ),
                    secondary_y=False,
                )

            inv_env = (pass1_analysis_data or {}).get("inverse_band_envelope")
            if inv_env is not None and isinstance(inv_env, np.ndarray) and len(inv_env) == n:
                plot_inv = inv_env[::factor] if factor > 1 and n >= factor else inv_env
                fig.add_trace(
                    go.Scatter(
                        x=plot_time,
                        y=plot_inv,
                        name="Noise Envelope",
                        line=dict(color="#b85c9e", width=1.25),
                        visible="legendonly",
                        hovertemplate="Noise Envelope: %{y:.4f}<extra></extra>",
                    ),
                    secondary_y=False,
                )

            nr_env = (pass1_analysis_data or {}).get("noise_removed_envelope")
            if (
                not use_nr_main
                and nr_env is not None
                and isinstance(nr_env, np.ndarray)
                and len(nr_env) == n
            ):
                plot_nr = nr_env[::factor] if factor > 1 and n >= factor else nr_env
                fig.add_trace(
                    go.Scatter(
                        x=plot_time,
                        y=plot_nr,
                        name="Noise Removed Envelope",
                        line=dict(color="#e67e22", width=1.25),
                        visible="legendonly",
                        hovertemplate="Noise Removed Envelope: %{y:.4f}<extra></extra>",
                    ),
                    secondary_y=False,
                )

            if len(anchor_beats) > 0:
                in_bounds = (anchor_beats >= 0) & (anchor_beats < len(audio_envelope))
                ab = anchor_beats[in_bounds]
                if len(ab) > 0:
                    anchor_times_sec = ab.astype(np.float64) / self.sample_rate
                    anchor_times_dt = _elapsed_seconds_to_plot_datetimes(anchor_times_sec)
                    y_at_beats = np.asarray(audio_envelope)[ab]
                    fig.add_trace(
                        go.Scatter(
                            x=anchor_times_dt,
                            y=y_at_beats,
                            name="Anchor beats",
                            mode="markers",
                            marker=dict(symbol="diamond", size=8, color="orange"),
                        ),
                        secondary_y=False,
                    )

        # Pass 1 instant BPM: raw 60/RR, then local median±MAD in time window, then global median±MAD (pass1_bpm_global_outlier_mad_k; <=0 skips)
        if pass1_bpm_data and "raw_scatter_times" in pass1_bpm_data and "raw_scatter_bpm" in pass1_bpm_data:
            rt = pass1_bpm_data["raw_scatter_times"]
            rb = pass1_bpm_data["raw_scatter_bpm"]
            if len(rt) > 0 and len(rt) == len(rb):
                raw_dt = _elapsed_seconds_to_plot_datetimes(np.asarray(rt, dtype=np.float64))
                fig.add_trace(
                    go.Scatter(
                        x=raw_dt,
                        y=rb,
                        name="Instant BPM (Pass 1)",
                        mode="markers",
                        marker=dict(size=6, color="#e74c3c", symbol="circle"),
                        visible="legendonly",
                    ),
                    secondary_y=True,
                )
                self.bpm_axis_center = float(np.median(rb))
        if pass1_bpm_data and "scatter_times" in pass1_bpm_data and "scatter_bpm" in pass1_bpm_data:
            st = pass1_bpm_data["scatter_times"]
            sb = pass1_bpm_data["scatter_bpm"]
            if len(st) > 0 and len(st) == len(sb):
                scatter_dt = _elapsed_seconds_to_plot_datetimes(np.asarray(st, dtype=np.float64))
                fig.add_trace(
                    go.Scatter(
                        x=scatter_dt,
                        y=sb,
                        name="Instant BPM (Pass 1) outliers removed",
                        mode="markers",
                        marker=dict(size=6, color="#9b59b6", symbol="circle-open"),
                    ),
                    secondary_y=True,
                )
                self.bpm_axis_center = float(np.median(sb))
        if pass1_bpm_data and "curve_times" in pass1_bpm_data and "curve_bpm" in pass1_bpm_data:
            ct = pass1_bpm_data["curve_times"]
            cb = pass1_bpm_data["curve_bpm"]
            if len(ct) > 0 and len(ct) == len(cb):
                curve_dt = _elapsed_seconds_to_plot_datetimes(np.asarray(ct, dtype=np.float64))
                fig.add_trace(
                    go.Scatter(
                        x=curve_dt,
                        y=cb,
                        name="BPM (pass 1)",
                        mode="lines",
                        line=dict(color="orange", width=2),
                    ),
                    secondary_y=True,
                )
        has_pass1_bpm_markers = pass1_bpm_data and (
            ("raw_scatter_bpm" in pass1_bpm_data and len(pass1_bpm_data["raw_scatter_bpm"]) > 0)
            or ("scatter_bpm" in pass1_bpm_data and len(pass1_bpm_data["scatter_bpm"]) > 0)
        )
        if not has_pass1_bpm_markers:
            self.bpm_axis_center = float(self.params.get("default_bpm_axis_center", self.bpm_axis_center))

        # BPM Trend (Belief) (canonical: dense raster at STANDARD_DT_SEC).
        if pass1_analysis_data and "pass2_lt_bpm_times" in pass1_analysis_data and "pass2_lt_bpm" in pass1_analysis_data:
            lt_times = np.asarray(pass1_analysis_data["pass2_lt_bpm_times"], dtype=np.float64)
            lt_vals = np.asarray(pass1_analysis_data["pass2_lt_bpm"], dtype=np.float64)
            if len(lt_times) >= 2 and len(lt_times) == len(lt_vals):
                lt_times_dt = _elapsed_seconds_to_plot_datetimes(lt_times)
                fig.add_trace(
                    go.Scatter(
                        x=lt_times_dt,
                        y=lt_vals,
                        name="BPM Trend (Belief)",
                        mode="lines",
                        line=dict(color="orange", width=2),
                        visible="legendonly",
                    ),
                    secondary_y=True,
                )

        self.fig = fig
        self._configure_layout()

        plot_title = f"Pass 1 - {os.path.basename(self.file_name)}"
        plot_config = {
            "scrollZoom": True,
            "toImageButtonOptions": {"filename": plot_title, "format": "png", "scale": 2},
            "showTips": False,
        }
        plotly_html = self.fig.to_html(config=plot_config, full_html=False, include_plotlyjs="cdn")
        # If CDN is unavailable, fall back to a local plotly.min.js beside the HTML (if present).
        plotly_html = re.sub(
            r'<script\s+src="(https://cdn\.plot\.ly/plotly[^"]+)"\s*></script>',
            r'<script src="\1" onerror="this.onerror=null;this.src=\'plotly.min.js\';"></script>',
            plotly_html,
            count=1,
        )
        self._noise_event_segments = (pass1_analysis_data or {}).get("noise_event_segments") or []
        self._pass3_state_boundaries = []
        self._pass3_state_boundaries_before = []
        custom_html = self._generate_custom_html(
            plotly_html, plot_title, base_name, output_options=output_options,
            is_final_pass=False,
        )
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(custom_html)
        logging.info("Preliminary pass plot saved to %s", output_html_path)

        self.fig = make_subplots(specs=[[{"secondary_y": True}]])

    def _prepare_bpm_axis_center(self, pass_metrics: Dict):
        """Use detected BPM stats to keep the BPM axis centered without altering the per-file zoom."""
        hrv_summary = pass_metrics.get("hrv_summary") or {}
        avg_bpm = hrv_summary.get("avg_bpm")
        smoothed_bpm = pass_metrics.get("smoothed_bpm")
        if avg_bpm is None and smoothed_bpm is not None and len(smoothed_bpm) > 0:
            avg_bpm = float(np.nanmean(np.asarray(smoothed_bpm, dtype=float)))
        if avg_bpm is None:
            avg_bpm = float(self.params.get("default_bpm_axis_center", self.bpm_axis_center))
        self.bpm_axis_center = float(avg_bpm)

    def _configure_layout(self):
        """Sets up the plot layout, titles, and axes.

        X-axis uses automatic ticks with a modest nticks cap. Supplying hundreds of explicit
        tickvals/ticktext for long files made Plotly pan/zoom lag badly (layout cost per frame).
        """
        self.fig.update_layout(
            template="plotly_dark",
            dragmode="pan",
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
            margin=dict(t=90, b=30, l=40, r=00), # borders/margins
            hovermode="x unified",
            autosize=True,
            uirevision="layout-stable",
        )

        self.fig.update_xaxes(
            title_text="Time",
            tickmode="auto",
            nticks=24,
            tickformat="%H:%M:%S",
            hoverformat="%H:%M:%S.%L",
            automargin=False,
            title_standoff=4,
            domain=[0, 0.95],
        )

        # Use the audio envelope trace, if present, to scale the amplitude axis.
        robust_upper_limit = 1
        if self.fig.data:
            envelope_values = None
            for trace in self.fig.data:
                tname = getattr(trace, "name", "")
                if tname in ("Bandpass Envelope", "Noise Removed Envelope") and hasattr(trace, "y"):
                    try:
                        envelope_values = np.asarray(trace.y, dtype=float)
                    except Exception:
                        envelope_values = None
                    break
            if envelope_values is not None and envelope_values.size > 0:
                robust_upper_limit = float(np.quantile(envelope_values, 0.95))

        amplitude_scale = param(self.params, "plot_amplitude_scale_factor")
        self.fig.update_yaxes(
            title_text="Signal Amplitude",
            secondary_y=False,
            range=[0, robust_upper_limit * amplitude_scale],
            showgrid=False,
            automargin=False,
            title_standoff=4,
        )
        half_span = self.bpm_axis_span / 2.0
        min_bpm = max(self.bpm_axis_center - half_span, 5)
        max_bpm = self.bpm_axis_center + half_span
        self.fig.update_yaxes(
            title_text="BPM / HRV",
            secondary_y=True,
            range=[min_bpm, max_bpm],
            autorange=False,
            automargin=False,
            title_standoff=0,
        )
        if getattr(self, "_has_systolic_traces", False):
            self.fig.update_layout(
                yaxis3=dict(
                    title="Phase Interval (s)",
                    overlaying="y",
                    anchor="free",
                    side="right",
                    position=0.97,
                    range=[0.1, 0.45],
                    showgrid=False,
                    automargin=False,
                    title_standoff=0,
                ),
            )

    def _add_primary_y_anchor_for_secondary_only_plot(self) -> None:
        """
        When optimize_long_plots skips the envelope, there are no traces on the primary y-axis.
        Plotly's y+y2 overlay subplot can then fail to render secondary (BPM) traces in the exported HTML.

        Anchors y with an invisible degenerate line so BPM / HRV on yaxis2 still draws.
        """
        dur = float(self.audio_duration_sec)
        if not np.isfinite(dur) or dur <= 0.0:
            return
        t_dt = _elapsed_seconds_to_plot_datetimes(np.array([0.0, dur], dtype=np.float64))
        self.fig.add_trace(
            go.Scatter(
                x=t_dt,
                y=np.array([0.0, 0.0], dtype=np.float64),
                mode="lines",
                line=dict(color="rgba(0,0,0,0)", width=1),
                showlegend=False,
                hoverinfo="skip",
                name="_axis_anchor",
            ),
            secondary_y=False,
        )

    def _add_line_traces(
        self,
        audio_envelope: np.ndarray,
        analysis_data: Dict,
        all_raw_peaks: Optional[np.ndarray] = None,
    ):
        """Adds audio envelope and noise floor traces. Downsampling (plot_downsample_factor) applies only here
        to these large arrays; contractility, BPM, HRV and markers are never downsampled.
        Note: Do not use dashed lines (dash=...) for line traces--they cause noticeable lag in the plot."""
        if getattr(self, "skip_detailed_debug_traces", False):
            logging.info("Skipping audio envelope and noise floor traces for long file (optimization enabled).")
            self._add_primary_y_anchor_for_secondary_only_plot()
            return
        plot_envelope = audio_envelope
        plot_noise_floor = analysis_data.get("dynamic_noise_floor_series")

        # Downsample only envelope and noise floor for performance; other traces (contractility, BPM, HRV) use full data
        factor = param(self.params, "plot_downsample_factor")
        n = len(audio_envelope)
        if factor > 1 and n >= factor:
            logging.info("Downsampling envelope and noise floor by factor %d for plotting.", factor)
            plot_secs = np.arange(0, n, factor, dtype=np.float64) / self.sample_rate
            plot_time_axis_dt = _elapsed_seconds_to_plot_datetimes(plot_secs)
            plot_envelope = audio_envelope[::factor]
            if plot_noise_floor is not None and not plot_noise_floor.empty:
                plot_noise_floor = plot_noise_floor.iloc[::factor]
        else:
            plot_secs = np.arange(n, dtype=np.float64) / self.sample_rate
            plot_time_axis_dt = _elapsed_seconds_to_plot_datetimes(plot_secs)

        use_nr_main = analysis_data.get("noise_removed_envelope") is not None
        main_env_name = "Noise Removed Envelope" if use_nr_main else "Bandpass Envelope"
        self.fig.add_trace(
            go.Scatter(x=plot_time_axis_dt, y=plot_envelope, name=main_env_name, line=dict(color="#47a5c4")),
            secondary_y=False,
        )
        bp_adata = analysis_data.get("bandpass_envelope")
        if (
            use_nr_main
            and bp_adata is not None
            and isinstance(bp_adata, np.ndarray)
            and len(bp_adata) == n
        ):
            plot_bp = bp_adata[::factor] if factor > 1 and n >= factor else bp_adata
            self.fig.add_trace(
                go.Scatter(
                    x=plot_time_axis_dt,
                    y=plot_bp,
                    name="Bandpass Envelope",
                    line=dict(color="#3498db", width=1.25, dash="dot"),
                    visible="legendonly",
                    hovertemplate="Bandpass Envelope: %{y:.4f}<extra></extra>",
                ),
                secondary_y=False,
            )
        if (
            plot_noise_floor is not None
            and not plot_noise_floor.empty
            and len(plot_noise_floor) >= len(plot_time_axis_dt)
        ):
            self.fig.add_trace(
                go.Scatter(
                    x=plot_time_axis_dt,
                    y=plot_noise_floor.values,
                    name="Dynamic Noise Floor",
                    line=dict(color="green", width=1.5),
                    hovertemplate="Noise Floor: %{y:.4f}<extra></extra>",
                    visible="legendonly",
                ),
                secondary_y=False,
            )

        inv_env = analysis_data.get("inverse_band_envelope")
        if inv_env is not None and isinstance(inv_env, np.ndarray) and len(inv_env) == n:
            plot_inv = inv_env[::factor] if factor > 1 and n >= factor else inv_env
            self.fig.add_trace(
                go.Scatter(
                    x=plot_time_axis_dt,
                    y=plot_inv,
                    name="Noise Envelope",
                    line=dict(color="#b85c9e", width=1.25),
                    visible="legendonly",
                    hovertemplate="Noise Envelope: %{y:.4f}<extra></extra>",
                ),
                secondary_y=False,
            )

        nr_env = analysis_data.get("noise_removed_envelope")
        if (
            not use_nr_main
            and nr_env is not None
            and isinstance(nr_env, np.ndarray)
            and len(nr_env) == n
        ):
            plot_nr = nr_env[::factor] if factor > 1 and n >= factor else nr_env
            self.fig.add_trace(
                go.Scatter(
                    x=plot_time_axis_dt,
                    y=plot_nr,
                    name="Noise Removed Envelope",
                    line=dict(color="#e67e22", width=1.25),
                    visible="legendonly",
                    hovertemplate="Noise Removed Envelope: %{y:.4f}<extra></extra>",
                ),
                secondary_y=False,
            )

    def _add_trough_markers(self, audio_envelope: np.ndarray, analysis_data: Dict):
        """Adds trough markers to the plot using original full-resolution data for accuracy."""
        if getattr(self, "skip_detailed_debug_traces", False):
            logging.info("Skipping trough markers for long file (optimization enabled).")
            return
        trough_indices = analysis_data.get("trough_indices")
        if trough_indices is not None and trough_indices.size > 0:
            trough_times_dt = _elapsed_seconds_to_plot_datetimes(
                np.asarray(trough_indices, dtype=np.float64) / self.sample_rate
            )

            self.fig.add_trace(
                go.Scatter(
                    x=trough_times_dt,
                    y=audio_envelope[trough_indices],
                    mode="markers",
                    name="Troughs",
                    marker=dict(color="green", symbol="circle-open", size=6),
                    visible="legendonly",
                ),
                secondary_y=False,
            )

    def _add_peak_marker_trace(
        self, indices, customdata, name, color, symbol, size, audio_envelope, hovertemplate
    ):
        """Add a single Scatter trace for peak markers (S1, S2, or Noise)."""
        times_dt = _elapsed_seconds_to_plot_datetimes(
            np.asarray(indices, dtype=np.float64) / self.sample_rate
        )
        self.fig.add_trace(
            go.Scatter(
                x=times_dt,
                y=audio_envelope[indices],
                mode="markers",
                name=name,
                marker=dict(color=color, symbol=symbol, size=size),
                customdata=customdata,
                hovertemplate=hovertemplate,
            ),
            secondary_y=False,
        )

    def _add_peak_traces(self, all_raw_peaks, debug_info, audio_envelope, trough_indices=None):
        """Adds S1, S2, and Noise peak markers to the plot with detailed hover info."""
        if getattr(self, "skip_detailed_debug_traces", False):
            logging.info("Skipping S1/S2/Noise peak markers for long file (optimization enabled).")
            return
        s1_peaks = {"indices": [], "customdata": []}
        s2_peaks = {"indices": [], "customdata": []}
        noise_peaks = {"indices": [], "customdata": []}

        classified_indices = set()

        for peak_idx, debug_value in debug_info.items():
            hover_text_parts = []

            peak_type = _get_peak_type_from_debug(debug_value) or "Unknown Peak"
            hover_text_parts.append(f"<b>Type:</b> {peak_type}")
            hover_text_parts.append(f"<b>Time:</b> {peak_idx / self.sample_rate:.2f}s")
            hover_text_parts.append(f"<b>Amp:</b> {audio_envelope[peak_idx]:.0f}")
            hover_text_parts.append("---")

            formatted_lines = format_debug_entry(debug_value)
            if formatted_lines:
                sub_text = "<br>".join(ln.replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;") for ln in formatted_lines)
                hover_text_parts.append(sub_text)

            full_hover_text = "<br>".join(hover_text_parts)
            classified_indices.add(peak_idx)

            if PeakType.is_s1(peak_type):
                s1_peaks["indices"].append(peak_idx)
                s1_peaks["customdata"].append(full_hover_text)
            elif PeakType.is_s2(peak_type):
                s2_peaks["indices"].append(peak_idx)
                s2_peaks["customdata"].append(full_hover_text)
            else:
                noise_peaks["indices"].append(peak_idx)
                noise_peaks["customdata"].append(full_hover_text)

        for peak_idx in all_raw_peaks:
            if peak_idx not in classified_indices:
                hover_text = (
                    f"<b>Type:</b> Unclassified<br>"
                    f"<b>Time:</b> {peak_idx / self.sample_rate:.2f}s<br>"
                    f"<b>Amp:</b> {audio_envelope[peak_idx]:.0f}<br>"
                    "<b>Details:</b> Peak was not evaluated by the classifier."
                )
                noise_peaks["indices"].append(peak_idx)
                noise_peaks["customdata"].append(hover_text)

        hovertemplate = "%{customdata}<extra></extra>"
        for name, peaks, color, symbol, size in (
            ("S1 Beats", s1_peaks, "#e36f6f", "circle", 8),
            ("S2 Beats", s2_peaks, "orange", "circle", 6),
            ("Noise/Rejected", noise_peaks, "grey", "x", 6),
        ):
            if peaks["indices"]:
                self._add_peak_marker_trace(
                    peaks["indices"], peaks["customdata"], name, color, symbol, size,
                    audio_envelope, hovertemplate,
                )

        # Average S1 / S2 contractility traces (prominence-based, averaged over time segments), Analysis Data only
        self._add_s1_s2_amplitude_traces(
            s1_peaks["indices"], s2_peaks["indices"], audio_envelope, trough_indices
        )

    def _add_pass3_large_gap_recovered_peak_markers(self, audio_envelope: np.ndarray) -> None:
        """Pass 3 debug: markers for peaks re-detected at higher sensitivity inside large-gap windows."""
        if getattr(self, "skip_detailed_debug_traces", False):
            return
        def _add(rec, name, color):
            if not rec:
                return
            try:
                indices = np.asarray(rec, dtype=np.int64)
            except Exception:
                return
            if indices.size == 0:
                return
            # Clip to array bounds
            n = int(len(audio_envelope))
            indices = indices[(indices >= 0) & (indices < n)]
            if indices.size == 0:
                return

            customdata = []
            for ix in indices.tolist():
                try:
                    customdata.append(
                        f"<b>Type:</b> {name}<br>"
                        f"<b>Time:</b> {float(ix) / float(self.sample_rate):.2f}s<br>"
                        f"<b>Amp:</b> {float(audio_envelope[int(ix)]):.0f}"
                    )
                except Exception:
                    customdata.append(f"<b>Type:</b> {name}")
            hovertemplate = "%{customdata}<extra></extra>"
            self._add_peak_marker_trace(
                indices=indices.tolist(),
                customdata=customdata,
                name=name,
                color=color,
                symbol="triangle-up-open",
                size=8,
                audio_envelope=audio_envelope,
                hovertemplate=hovertemplate,
            )

        _add(
            getattr(self, "_pass3_large_gap_recovered_peaks_insensitive", None) or [],
            "Recovered peaks at large gaps (insensitive)",
            "#b07cff",
        )
        _add(
            getattr(self, "_pass3_large_gap_recovered_peaks_sensitive", None) or [],
            "Recovered peaks at large gaps (sensitive)",
            "#67d1ff",
        )
        # The peaks find_gap_windows actually used to decide each gap/insert. These
        # drive the decision; the "recovered" sets above are re-detected over the
        # trimmed window and can differ. Plotting both makes that disagreement visible.
        _add(
            getattr(self, "_pass3_gap_decision_peaks_sensitive", None) or [],
            "Gap decision peaks (sensitive)",
            "#ff5b5b",
        )

    def _average_prominence_by_time_segment(
        self, times_sec: np.ndarray, proms: np.ndarray, segment_sec: float
    ) -> tuple:
        """Bin prominence by fixed-duration time segments; return (segment_center_times, mean_prominence)."""
        times_sec = np.asarray(times_sec, dtype=float)
        proms = np.asarray(proms, dtype=float)
        if len(times_sec) == 0 or len(proms) == 0 or len(times_sec) != len(proms):
            return np.array([]), np.array([])
        t_min, t_max = float(np.min(times_sec)), float(np.max(times_sec))
        t0 = np.floor(t_min / segment_sec) * segment_sec
        segment_centers = []
        segment_means = []
        while t0 <= t_max:
            mask = (times_sec >= t0) & (times_sec < t0 + segment_sec)
            if np.any(mask):
                segment_centers.append(t0 + segment_sec / 2.0)
                segment_means.append(float(np.mean(proms[mask])))
            t0 += segment_sec
        return np.array(segment_centers), np.array(segment_means)

    def _smooth_peak_amplitudes(self, amps: np.ndarray, window_size: int = 3) -> np.ndarray:
        """Moving average over window_size points (current and adjacent). Boundaries use fewer points."""
        n = len(amps)
        if n == 0:
            return amps
        half = max(0, (window_size - 1) // 2)
        smoothed = np.empty(n, dtype=float)
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            smoothed[i] = float(np.mean(amps[lo:hi]))
        return smoothed

    def _add_prominence_line_trace(
        self, times_sec, proms, name, color, visible, window_size=1
    ):
        """Add one prominence-based contractility line trace. Use window_size=1 for pre-averaged (e.g. time-segment) data."""
        proms = np.asarray(proms, dtype=float)
        smoothed = self._smooth_peak_amplitudes(proms, window_size=window_size)
        times_dt = _elapsed_seconds_to_plot_datetimes(np.asarray(times_sec, dtype=np.float64))
        self.fig.add_trace(
            go.Scatter(
                x=times_dt,
                y=smoothed,
                mode="lines",
                name=name,
                line=dict(color=color, width=2),
                visible=visible,
            ),
            secondary_y=False,
        )

    def _add_s1_s2_amplitude_traces(self, s1_indices, s2_indices, audio_envelope, trough_indices=None):
        """Add line traces for Average S1, S2, and combined contractility (prominence-based, averaged over time segments).
        Uses a fixed-duration segment (default 2 s) so trends reflect: long-term contractility vs BPM; short-term S1 vs inhale/exhale."""
        segment_sec = float(param(self.params, "contractility_average_window_sec"))
        troughs = np.array(trough_indices) if trough_indices is not None and len(trough_indices) > 0 else np.array([], dtype=np.intp)

        def prominence_at(peak_idx):
            details = get_peak_prominence_details(peak_idx, audio_envelope, troughs)
            return details["prominence"]

        if s1_indices:
            s1_idx = np.array(s1_indices)
            times_sec = s1_idx.astype(float) / self.sample_rate
            proms = np.array([prominence_at(int(i)) for i in s1_idx])
            t_centers, mean_proms = self._average_prominence_by_time_segment(times_sec, proms, segment_sec)
            if len(t_centers) > 0:
                self._add_prominence_line_trace(
                    t_centers, mean_proms, "Average S1 contractility", "#e36f6f", "legendonly", window_size=1
                )
        if s2_indices:
            s2_idx = np.array(s2_indices)
            times_sec = s2_idx.astype(float) / self.sample_rate
            proms = np.array([prominence_at(int(i)) for i in s2_idx])
            t_centers, mean_proms = self._average_prominence_by_time_segment(times_sec, proms, segment_sec)
            if len(t_centers) > 0:
                self._add_prominence_line_trace(
                    t_centers, mean_proms, "Average S2 contractility", "orange", "legendonly", window_size=1
                )
        if s1_indices or s2_indices:
            times_sec = []
            proms = []
            for indices in (s1_indices or [], s2_indices or []):
                if not indices:
                    continue
                idx = np.array(indices)
                times_sec.extend((idx.astype(float) / self.sample_rate).tolist())
                proms.extend([prominence_at(int(i)) for i in idx])
            times_sec = np.array(times_sec)
            proms = np.array(proms)
            t_centers, mean_proms = self._average_prominence_by_time_segment(times_sec, proms, segment_sec)
            if len(t_centers) > 0:
                self._add_prominence_line_trace(
                    t_centers, mean_proms, "Average contractility", "#aaa", "legendonly", window_size=1
                )

    def _add_bpm_hrv_traces(
        self,
        smoothed_bpm,
        analysis_data,
        windowed_hrv_df,
        output_suffix: Optional[str] = None,
        pass1_bpm_series: Optional[pd.Series] = None,
        pass1_bpm_times: Optional[np.ndarray] = None,
        instant_bpm: Optional[np.ndarray] = None,
        bpm_times: Optional[np.ndarray] = None,
        instant_bpm_raw: Optional[np.ndarray] = None,
        bpm_times_raw: Optional[np.ndarray] = None,
    ):
        """Adds BPM, pass 1 BPM curve (when provided), and HRV traces. BPM Trend (Belief) is only on the pass 1 plot."""
        def _add_pass2_label_score_traces() -> None:
            """
            Pass 2 debug: plot per-peak label_scores (S1/S2/noise) as percentage traces (0–100)
            on the same axis as BPM (secondary_y=True).

            Also included on the Pass 3 plot since Pass 3 consumes these scores.
            """
            if output_suffix not in ("_pass2", "_pass3"):
                return
            if getattr(self, "skip_detailed_debug_traces", False):
                return
            debug_info = (analysis_data or {}).get("peak_classifications") or {}
            if not isinstance(debug_info, dict) or not debug_info:
                return
            times_sec = []
            s1_scores = []
            s2_scores = []
            noise_scores = []
            for peak_idx, entry in debug_info.items():
                if not isinstance(peak_idx, (int, np.integer)):
                    continue
                if not isinstance(entry, dict):
                    continue
                ls = entry.get("label_scores")
                if not isinstance(ls, dict):
                    continue
                try:
                    t = float(peak_idx) / float(self.sample_rate)
                    s1 = float(ls.get("S1", 0.0)) * 100.0
                    s2 = float(ls.get("S2", 0.0)) * 100.0
                    nz = float(ls.get("noise", 0.0)) * 100.0
                except Exception:
                    continue
                if not (np.isfinite(t) and np.isfinite(s1) and np.isfinite(s2) and np.isfinite(nz)):
                    continue
                times_sec.append(t)
                s1_scores.append(s1)
                s2_scores.append(s2)
                noise_scores.append(nz)
            if not times_sec:
                return
            order = np.argsort(np.asarray(times_sec, dtype=np.float64))
            t_dt = _elapsed_seconds_to_plot_datetimes(np.asarray(times_sec, dtype=np.float64)[order])
            s1_arr = np.asarray(s1_scores, dtype=np.float64)[order]
            s2_arr = np.asarray(s2_scores, dtype=np.float64)[order]
            nz_arr = np.asarray(noise_scores, dtype=np.float64)[order]

            common = dict(mode="lines", line=dict(width=1), opacity=0.9, visible="legendonly")
            self.fig.add_trace(
                go.Scatter(x=t_dt, y=s1_arr, name="S1 score", line=dict(color="#e36f6f", width=1), **{k: v for k, v in common.items() if k != "line"}),
                secondary_y=True,
            )
            self.fig.add_trace(
                go.Scatter(x=t_dt, y=s2_arr, name="S2 score", line=dict(color="orange", width=1), **{k: v for k, v in common.items() if k != "line"}),
                secondary_y=True,
            )
            self.fig.add_trace(
                go.Scatter(x=t_dt, y=nz_arr, name="Noise score", line=dict(color="grey", width=1), **{k: v for k, v in common.items() if k != "line"}),
                secondary_y=True,
            )

        # Label smoothed BPM by pass when known (Pass 2 / Pass 3), else generic
        if output_suffix == "_pass2":
            bpm_trace_name = "BPM (Pass 2)"
        elif output_suffix == "_pass3":
            bpm_trace_name = "BPM (Pass 3)"
        else:
            bpm_trace_name = "Average BPM"

        def _flatten_float_series(a):
            if a is None:
                return None
            arr = getattr(a, "to_numpy", None)
            if callable(arr):
                vals = np.asarray(a.to_numpy(dtype=np.float64), dtype=np.float64)
            else:
                vals = np.asarray(a, dtype=np.float64)
            return vals.reshape(-1)

        flat_bpm = _flatten_float_series(smoothed_bpm)
        flat_bt = _flatten_float_series(bpm_times)
        if (
            flat_bpm is not None
            and flat_bt is not None
            and flat_bt.shape[0] == flat_bpm.shape[0]
            and flat_bt.shape[0] > 0
        ):
            bpm_dt = _elapsed_seconds_to_plot_datetimes(flat_bt.astype(np.float64, copy=False))
            self.fig.add_trace(
                go.Scatter(
                    x=bpm_dt,
                    y=flat_bpm,
                    name=bpm_trace_name,
                    line=dict(color="#4a4a4a", width=3),
                ),
                secondary_y=True,
            )

        _add_pass2_label_score_traces()

        # Instantaneous BPM: raw 60/RR vs MAD-filtered (local + global), same style as pass 1
        if output_suffix in ("_pass2", "_pass3"):
            pass_instant_label = "Pass 2" if output_suffix == "_pass2" else "Pass 3"
            if (
                instant_bpm_raw is not None
                and bpm_times_raw is not None
                and len(instant_bpm_raw) == len(bpm_times_raw)
                and len(bpm_times_raw) > 0
            ):
                raw_dt = _elapsed_seconds_to_plot_datetimes(np.asarray(bpm_times_raw, dtype=np.float64))
                self.fig.add_trace(
                    go.Scatter(
                        x=raw_dt,
                        y=instant_bpm_raw,
                        name=f"Instant BPM ({pass_instant_label})",
                        mode="markers",
                        marker=dict(size=6, color="#e74c3c", symbol="circle"),
                        visible="legendonly",
                    ),
                    secondary_y=True,
                )
                if (
                    instant_bpm is not None
                    and bpm_times is not None
                    and len(instant_bpm) == len(bpm_times)
                    and len(bpm_times) > 0
                ):
                    filt_dt = _elapsed_seconds_to_plot_datetimes(np.asarray(bpm_times, dtype=np.float64))
                    self.fig.add_trace(
                        go.Scatter(
                            x=filt_dt,
                            y=instant_bpm,
                            name=f"Instant BPM ({pass_instant_label}) outliers removed",
                            mode="markers",
                            marker=dict(size=6, color="#9b59b6", symbol="circle-open"),
                            visible="legendonly",
                        ),
                        secondary_y=True,
                    )
            elif (
                instant_bpm is not None
                and bpm_times is not None
                and len(instant_bpm) == len(bpm_times)
                and len(bpm_times) > 0
            ):
                instant_times_dt = _elapsed_seconds_to_plot_datetimes(np.asarray(bpm_times, dtype=np.float64))
                self.fig.add_trace(
                    go.Scatter(
                        x=instant_times_dt,
                        y=instant_bpm,
                        name=f"Instantaneous BPM ({pass_instant_label})",
                        mode="markers",
                        marker=dict(size=5, color="#4a4a4a", symbol="circle"),
                        visible="legendonly",
                    ),
                    secondary_y=True,
                )

        # Prior BPM curve: pass 1 on pass 2 plot, pass 2 on pass 3 plot
        if (
            pass1_bpm_series is not None
            and len(pass1_bpm_series) > 0
            and pass1_bpm_times is not None
            and len(pass1_bpm_times) == len(pass1_bpm_series)
        ):
            prior_curve_name = "BPM (Pass 2)" if output_suffix == "_pass3" else "BPM (pass 1)"
            pass1_times_dt = _elapsed_seconds_to_plot_datetimes(np.asarray(pass1_bpm_times, dtype=np.float64))
            self.fig.add_trace(
                go.Scatter(
                    x=pass1_times_dt,
                    y=np.asarray(pass1_bpm_series, dtype=np.float64),
                    name=prior_curve_name,
                    line=dict(color="orange", width=2),
                    visible="legendonly",
                ),
                secondary_y=True,
            )
        if (
            windowed_hrv_df is not None
            and not windowed_hrv_df.empty
            and "time" in windowed_hrv_df
            and "rmssdc" in windowed_hrv_df
            and "sdnn" in windowed_hrv_df
        ):
            hrv_times_dt = _elapsed_seconds_to_plot_datetimes(
                np.asarray(windowed_hrv_df["time"], dtype=np.float64)
            )
            self.fig.add_trace(
                go.Scatter(
                    x=hrv_times_dt, y=windowed_hrv_df["rmssdc"], name="RMSSDc", line=dict(color="cyan", width=2), visible="legendonly"
                ),
                secondary_y=True,
            )
            self.fig.add_trace(
                go.Scatter(
                    x=hrv_times_dt, y=windowed_hrv_df["sdnn"], name="SDNN", line=dict(color="magenta", width=2), visible="legendonly"
                ),
                secondary_y=True,
            )
            if "lf_hf_ratio" in windowed_hrv_df.columns:
                self.fig.add_trace(
                    go.Scatter(
                        x=hrv_times_dt,
                        y=windowed_hrv_df["lf_hf_ratio"],
                        name="LF/HF (windowed)",
                        line=dict(color="yellow", width=2),
                        visible="legendonly",
                    ),
                    secondary_y=True,
                )

    def _add_systolic_interval_traces(
        self,
        analysis_data: Dict,
        pass_metrics: Dict,
        output_suffix: Optional[str],
    ) -> None:
        """Add BPM-expected systole curve, measured systole datapoints (outlier-filtered), and best-fit curve on yaxis3 (Pass 2 and Pass 3 only)."""
        if output_suffix not in ("_pass2", "_pass3"):
            return
        if getattr(self, "skip_detailed_debug_traces", False):
            return
        obs_t, obs_iv, exp_t, exp_iv = _compute_systolic_interval_data(
            analysis_data, pass_metrics, self.sample_rate, self.params
        )
        measured_timeline_t, measured_timeline_iv = _compute_measured_systole_from_state_boundaries(
            analysis_data, self.sample_rate, prefer="after"
        )
        measured_diastole_t, measured_diastole_iv = _compute_measured_diastole_from_state_boundaries(
            analysis_data, self.sample_rate, prefer="after"
        )
        exp_dia_t, exp_dia_iv = _compute_expected_diastole_from_bpm(pass_metrics, self.params)
        if not exp_t and not obs_t:
            return
        def to_dt(seq):
            return _elapsed_seconds_to_plot_datetimes(np.asarray(seq, dtype=np.float64))
        if exp_t:
            # Pass 3: shift expected to align with measured (exertion-only if peak exists, else all-time)
            plot_exp_iv = list(exp_iv)
            if output_suffix == "_pass3":
                peak_sec = pass_metrics.get("peak_bpm_time_sec")
                shift = _compute_systolic_shift(obs_t, obs_iv, exp_t, exp_iv, peak_sec)
                if shift is not None:
                    plot_exp_iv = [v + shift for v in exp_iv]
            self.fig.add_trace(
                go.Scatter(
                    x=to_dt(exp_t),
                    y=plot_exp_iv,
                    name="Expected systole from BPM",
                    line=dict(color="cyan", width=2),
                    yaxis="y3",
                    visible="legendonly",
                ),
            )
        if exp_dia_t:
            self.fig.add_trace(
                go.Scatter(
                    x=to_dt(exp_dia_t),
                    y=exp_dia_iv,
                    name="Expected diastole from BPM",
                    line=dict(color="#66d9ff", width=2),
                    yaxis="y3",
                    visible="legendonly",
                ),
            )
        if measured_timeline_t:
            self.fig.add_trace(
                go.Scatter(
                    x=to_dt(measured_timeline_t),
                    y=measured_timeline_iv,
                    name="Measured systole",
                    mode="markers",
                    marker=dict(size=5, color="#b07cff", symbol="diamond"),
                    yaxis="y3",
                    visible="legendonly",
                ),
            )
        if measured_diastole_t:
            self.fig.add_trace(
                go.Scatter(
                    x=to_dt(measured_diastole_t),
                    y=measured_diastole_iv,
                    name="Measured diastole",
                    mode="markers",
                    marker=dict(size=5, color="#55d68d", symbol="triangle-up"),
                    yaxis="y3",
                    visible="legendonly",
                ),
            )
        # Pass 3: cleaned & smoothed measured-phase curves.
        # "Before repair" is computed immediately after the initial state paint, even if there are no noise windows.
        _pre_sys_t   = analysis_data.get("pass3_measured_phase_before_repair_systole_t")
        _pre_sys_dur = analysis_data.get("pass3_measured_phase_before_repair_systole_dur")
        _pre_dia_t   = analysis_data.get("pass3_measured_phase_before_repair_diastole_t")
        _pre_dia_dur = analysis_data.get("pass3_measured_phase_before_repair_diastole_dur")
        if _pre_sys_t is not None and len(_pre_sys_t) >= 2:
            self.fig.add_trace(
                go.Scatter(
                    x=to_dt(_pre_sys_t),
                    y=list(_pre_sys_dur),
                    name="Measured systole curve (before repair)",
                    line=dict(color="#c87cff", width=2),
                    yaxis="y3",
                    visible="legendonly",
                ),
            )
        if _pre_dia_t is not None and len(_pre_dia_t) >= 2:
            self.fig.add_trace(
                go.Scatter(
                    x=to_dt(_pre_dia_t),
                    y=list(_pre_dia_dur),
                    name="Measured diastole curve (before repair)",
                    line=dict(color="#33cc77", width=2),
                    yaxis="y3",
                    visible="legendonly",
                ),
            )

        _post_sys_t   = analysis_data.get("pass3_measured_phase_final_systole_t")
        _post_sys_dur = analysis_data.get("pass3_measured_phase_final_systole_dur")
        _post_dia_t   = analysis_data.get("pass3_measured_phase_final_diastole_t")
        _post_dia_dur = analysis_data.get("pass3_measured_phase_final_diastole_dur")
        if _post_sys_t is not None and len(_post_sys_t) >= 2:
            self.fig.add_trace(
                go.Scatter(
                    x=to_dt(_post_sys_t),
                    y=list(_post_sys_dur),
                    name="Measured systole curve (final)",
                    line=dict(color="#b07cff", width=2, dash="dot"),
                    yaxis="y3",
                    visible="legendonly",
                ),
            )
        if _post_dia_t is not None and len(_post_dia_t) >= 2:
            self.fig.add_trace(
                go.Scatter(
                    x=to_dt(_post_dia_t),
                    y=list(_post_dia_dur),
                    name="Measured diastole curve (final)",
                    line=dict(color="#55d68d", width=2, dash="dot"),
                    yaxis="y3",
                    visible="legendonly",
                ),
            )
        self._has_systolic_traces = True

    def _add_annotations_and_summary(
        self, bpm_times, smoothed_bpm, hrv_summary, hrr_stats, peak_recovery_stats, bpm_failure_report=None
    ):
        """Adds min/max BPM annotations on the plot and builds plain-text summary for the HTML Analysis Summary modal."""
        if bpm_failure_report and bpm_failure_report.get("failed"):
            reasons = bpm_failure_report.get("reasons") or []
            self.fig.add_annotation(
                x=0.5,
                y=1.08,
                xref="paper",
                yref="paper",
                text="⚠ Possible BPM tracking failure: " + "; ".join(reasons),
                showarrow=False,
                font=dict(color="#e36f6f", size=13),
                bgcolor="rgba(60,20,20,0.6)",
                bordercolor="#e36f6f",
                borderwidth=1,
            )
        # smoothed_bpm is stored as a dense raster (array) in pass_metrics.
        if bpm_times is not None and smoothed_bpm is not None and len(bpm_times) > 0 and len(bpm_times) == len(smoothed_bpm):
            arr = np.asarray(smoothed_bpm, dtype=np.float64)
            if not np.any(np.isfinite(arr)):
                return
            max_bpm_val = float(np.nanmax(arr))
            min_bpm_val = float(np.nanmin(arr))
            t = np.asarray(bpm_times, dtype=np.float64)
            max_i = int(np.nanargmax(arr))
            min_i = int(np.nanargmin(arr))
            max_bpm_time = _elapsed_seconds_to_plot_datetimes(np.asarray([t[max_i]], dtype=np.float64))[0]
            min_bpm_time = _elapsed_seconds_to_plot_datetimes(np.asarray([t[min_i]], dtype=np.float64))[0]

            self.fig.add_annotation(
                x=max_bpm_time,
                y=max_bpm_val,
                text=f"Max: {max_bpm_val:.1f} BPM",
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=-40,
                font=dict(color="#e36f6f"),
                yref="y2",
            )
            self.fig.add_annotation(
                x=min_bpm_time,
                y=min_bpm_val,
                text=f"Min: {min_bpm_val:.1f} BPM",
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=40,
                font=dict(color="#a3d194"),
                yref="y2",
            )

        # Build plain-text summary for HTML (Analysis Summary button popup); no longer drawn on plot.
        summary_lines: List[str] = []
        if bpm_failure_report and bpm_failure_report.get("failed"):
            summary_lines.append("⚠ Possible BPM tracking failure:")
            for reason in bpm_failure_report.get("reasons") or []:
                summary_lines.append(f"  - {reason}")
        if hrv_summary:
            if hrv_summary.get("avg_bpm") is not None:
                summary_lines.append(
                    f"Avg/Min/Max BPM: {hrv_summary['avg_bpm']:.1f} / {hrv_summary['min_bpm']:.1f} / {hrv_summary['max_bpm']:.1f}"
                )
            if hrr_stats and hrr_stats.get("hrr_value_bpm") is not None:
                summary_lines.append(f"1-Min HRR: {hrr_stats['hrr_value_bpm']:.1f} BPM Drop")
            if peak_recovery_stats and peak_recovery_stats.get("slope_bpm_per_sec") is not None:
                summary_lines.append(f"Peak Recovery Rate: {peak_recovery_stats['slope_bpm_per_sec']:.2f} BPM/sec")
            if hrv_summary.get("avg_rmssdc") is not None:
                summary_lines.append(f"Avg. Corrected RMSSD: {hrv_summary['avg_rmssdc']:.2f}")
            if hrv_summary.get("avg_sdnn") is not None:
                summary_lines.append(f"Avg. Windowed SDNN: {hrv_summary['avg_sdnn']:.2f} ms")
            if hrv_summary.get("avg_lf_hf_ratio") is not None:
                summary_lines.append(f"Avg. LF/HF (windowed): {hrv_summary['avg_lf_hf_ratio']:.2f}")
            global_freq = hrv_summary.get("global_freq") if hrv_summary else None
            if global_freq:
                summary_lines.append(
                    f"VLF/LF/HF (global, ms²): {global_freq.get('vlf_power', 0):.2f} / {global_freq.get('lf_power', 0):.2f} / {global_freq.get('hf_power', 0):.2f} ; LF/HF: {global_freq.get('lf_hf_ratio', 0):.2f}"
                )
        self.analysis_summary_text = "\n".join(summary_lines) if summary_lines else ""

    def _add_slope_traces(self, major_inclines, major_declines, peak_recovery_stats, peak_exertion_stats):
        """Adds traces for major exertion and recovery periods."""
        if major_inclines:
            for i, incline in enumerate(major_inclines):
                c_data = [incline["duration_sec"], incline["bpm_increase"], incline["slope_bpm_per_sec"]]
                self.fig.add_trace(
                    go.Scatter(
                        x=_bpm_axis_times_to_plot_x_coords([incline["start_time"], incline["end_time"]]),
                        y=[incline["start_bpm"], incline["end_bpm"]],
                        mode="lines",
                        line=dict(color="purple", width=4),
                        name="Exertion",
                        legendgroup="Exertion",
                        showlegend=(i == 0),
                        visible="legendonly",
                        yaxis="y2",
                        hovertemplate="<b>Exertion Period</b><br>Duration: %{customdata[0]:.1f}s<br>BPM Increase: %{customdata[1]:.1f}<br>Slope: %{customdata[2]:.2f} BPM/sec<extra></extra>",
                        customdata=np.array([c_data, c_data]),
                    )
                )

        if major_declines:
            for i, decline in enumerate(major_declines):
                c_data = [decline["duration_sec"], decline["bpm_decrease"], decline["slope_bpm_per_sec"]]
                self.fig.add_trace(
                    go.Scatter(
                        x=_bpm_axis_times_to_plot_x_coords([decline["start_time"], decline["end_time"]]),
                        y=[decline["start_bpm"], decline["end_bpm"]],
                        mode="lines",
                        line=dict(color="#2ca02c", width=4),
                        name="Recovery",
                        legendgroup="Recovery",
                        showlegend=(i == 0),
                        visible="legendonly",
                        yaxis="y2",
                        hovertemplate="<b>Recovery Period</b><br>Duration: %{customdata[0]:.1f}s<br>BPM Decrease: %{customdata[1]:.1f}<br>Slope: %{customdata[2]:.2f} BPM/sec<extra></extra>",
                        customdata=np.array([c_data, c_data]),
                    )
                )

        if peak_recovery_stats:
            stats = peak_recovery_stats
            self.fig.add_trace(
                go.Scatter(
                    x=_bpm_axis_times_to_plot_x_coords([stats["start_time"], stats["end_time"]]),
                    y=[stats["start_bpm"], stats["end_bpm"]],
                    mode="lines",
                    line=dict(color="#ff69b4", width=5),
                    name="Peak Recovery Slope",
                    legendgroup="Steepest Slopes",
                    visible="legendonly",
                    yaxis="y2",
                    hovertemplate="<b>Peak Recovery Slope</b><br>Slope: %{customdata[0]:.2f} BPM/sec<br>Duration: %{customdata[1]:.1f}s<extra></extra>",
                    customdata=np.array([[stats["slope_bpm_per_sec"], stats["duration_sec"]]] * 2),
                )
            )

        if peak_exertion_stats:
            stats = peak_exertion_stats
            self.fig.add_trace(
                go.Scatter(
                    x=_bpm_axis_times_to_plot_x_coords([stats["start_time"], stats["end_time"]]),
                    y=[stats["start_bpm"], stats["end_bpm"]],
                    mode="lines",
                    line=dict(color="#9d32a8", width=5),
                    name="Peak Exertion Slope",
                    legendgroup="Steepest Slopes",
                    visible="legendonly",
                    yaxis="y2",
                    hovertemplate="<b>Peak Exertion Slope</b><br>Slope: +%{customdata[0]:.2f} BPM/sec<br>Duration: %{customdata[1]:.1f}s<extra></extra>",
                    customdata=np.array([[stats["slope_bpm_per_sec"], stats["duration_sec"]]] * 2),
                )
            )

    def _generate_custom_html(
        self,
        plotly_html: str,
        plot_title: str,
        base_name: str,
        *,
        pipeline_steps_html: str = "",
        output_options: Optional[Dict] = None,
        is_final_pass: bool = True,
    ) -> str:
        """
        Generates custom HTML with audio player, timeline scrubber, and synchronized playhead.
        Loads assets/template.html and substitutes %%PLACEHOLDER%% tokens with computed values.
        Uses %%HTML_INTERACTIVE_SCRIPTS%% for config + either interactive_plot.js or inlined html_inline_minimal.js.
        """
        audio_file_name = os.path.basename(self.audio_source_path)
        _stem, _ext = os.path.splitext(audio_file_name)
        if not _ext:
            _ext = ".wav"
        output_audio_basename = normalize_output_filename_stem(_stem) + _ext
        duration_sec = self.audio_duration_sec or 0

        # --- Resolve audio source path ---
        audio_src = ""
        dest_audio_path = os.path.join(self.output_directory, output_audio_basename)
        copy_from = (
            self.audio_source_path
            if os.path.exists(self.audio_source_path)
            else find_companion_wav(
                _stem,
                self.output_directory,
                os.path.dirname(self.audio_source_path),
            )
        )
        if copy_from and os.path.abspath(copy_from) != os.path.abspath(dest_audio_path):
            try:
                shutil.copy2(copy_from, dest_audio_path)
                logging.info("Copied audio file to %s", dest_audio_path)
            except Exception as e:
                logging.error("Could not copy audio file: %s", e)
        elif not copy_from:
            logging.error("Audio source file does NOT exist: %s", self.audio_source_path)

        if os.path.exists(dest_audio_path):
            audio_src = output_audio_basename.replace('\\', '/')
            if not copy_from or os.path.abspath(copy_from) != os.path.abspath(dest_audio_path):
                logging.info("Found audio file in output directory: %s", dest_audio_path)
        else:
            logging.error("Audio file not found anywhere: %s", output_audio_basename)

        filtered_debug_file_name = f"{base_name}_filtered_debug.wav"
        filtered_debug_path = os.path.join(self.output_directory, filtered_debug_file_name)
        filtered_available = os.path.exists(filtered_debug_path)
        filtered_audio_src = filtered_debug_file_name.replace('\\', '/') if filtered_available else ""
        if filtered_available:
            logging.info("Using filtered debug audio: %s", filtered_debug_path)

        filtered_inverse_debug_file_name = f"{base_name}_filtered_inverse_debug.wav"
        filtered_inverse_debug_path = os.path.join(
            self.output_directory, filtered_inverse_debug_file_name
        )
        filtered_inverse_available = os.path.exists(filtered_inverse_debug_path)
        filtered_inverse_audio_src = (
            filtered_inverse_debug_file_name.replace('\\', '/') if filtered_inverse_available else ""
        )
        if filtered_inverse_available:
            logging.info("Using inverse-band debug audio: %s", filtered_inverse_debug_path)

        logging.info("HTML audio source path: '%s'", audio_src)
        audio_src_escaped = urllib.parse.quote(audio_src)
        filtered_audio_src_escaped = urllib.parse.quote(filtered_audio_src) if filtered_audio_src else ""
        filtered_inverse_audio_src_escaped = (
            urllib.parse.quote(filtered_inverse_audio_src) if filtered_inverse_audio_src else ""
        )

        # --- Resolve spectrogram paths ---
        spectrogram_original_src = ""
        spectrogram_filtered_src = ""
        spectrogram_available_original = False
        spectrogram_available_filtered = False

        if getattr(self, "spectrogram_enabled", True):
            if getattr(self, "spectrogram_original_filename", None):
                spectrogram_original_src = self.spectrogram_original_filename
                spectrogram_available_original = True
            else:
                try:
                    if audio_src:
                        spec_path = os.path.join(self.output_directory, f"{base_name}_spectrogram.png")
                        spec_name = self._generate_spectrogram_image(
                            os.path.join(self.output_directory, audio_src), spec_path
                        )
                        if spec_name:
                            spectrogram_original_src = spec_name
                            spectrogram_available_original = True
                except Exception as e:
                    logging.warning("Failed to generate on-demand original spectrogram: %s", e)

            if filtered_available:
                try:
                    spec_filtered_path = os.path.join(
                        self.output_directory, f"{base_name}_filtered_spectrogram.png"
                    )
                    spec_filtered_name = self._generate_spectrogram_image(
                        filtered_debug_path, spec_filtered_path
                    )
                    if spec_filtered_name:
                        spectrogram_filtered_src = spec_filtered_name
                        spectrogram_available_filtered = True
                except Exception as e:
                    logging.warning("Failed to generate filtered spectrogram: %s", e)
        else:
            logging.info("Spectrogram generation disabled; no spectrogram images generated.")

        # --- Build audio source <select> ---
        audio_source_options = ['<option value="original">Original Audio</option>']
        if filtered_available:
            audio_source_options.append('<option value="filtered">Bandpass Audio</option>')
        if filtered_inverse_available:
            audio_source_options.append('<option value="filtered_inverse">Noise Audio</option>')
        audio_source_select_html = (
            '<select id="audio-source-select" class="audio-source-select">'
            + "".join(audio_source_options)
            + '</select>'
        )

        # --- Build JS configuration payload ---
        _oo = output_options if output_options is not None else DEFAULT_OUTPUT_OPTIONS.copy()
        hover_on_by_default = bool(_oo.get("html_s1_s2_hover_on_by_default", False))

        config_payload = {
            "totalDuration": float(duration_sec),
            "spectrogramSources": {
                "original": spectrogram_original_src,
                "filtered": spectrogram_filtered_src,
            },
            "spectrogramAvailable": {
                "original": spectrogram_available_original,
                "filtered": spectrogram_available_filtered,
            },
            "audioSources": {
                "original": audio_src_escaped,
                "filtered": filtered_audio_src_escaped,
                "filtered_inverse": filtered_inverse_audio_src_escaped,
            },
            "audioLabels": {
                "original": audio_file_name,
                "filtered": "Bandpass Audio" if filtered_available else audio_file_name,
                "filtered_inverse": "Noise Audio" if filtered_inverse_available else audio_file_name,
            },
            "analysisSummary": getattr(self, "analysis_summary_text", "") or "",
            "htmlS1S2HoverOnByDefault": hover_on_by_default,
            # Trace audience map + default view for the "Show:" category filter (final pass only).
            "traceAudience": TRACE_AUDIENCE,
            "showCategoryFilter": bool(is_final_pass),
            "defaultLegendView": "debug" if is_final_pass else "all",
            "bpmIntervalParams": {
                "s1_nominal_sec":             float(param(self.params, "s1_nominal_sec")),
                "s2_nominal_sec":             float(param(self.params, "s2_nominal_sec")),
                "weissler_ref_et_ms":         float(param(self.params, "s1_s2_expected_weissler_ref_et_ms")),
                "weissler_ref_bpm":           float(param(self.params, "s1_s2_expected_weissler_ref_bpm")),
                "weissler_slope_ms_per_bpm":  float(param(self.params, "s1_s2_expected_weissler_slope_ms_per_bpm")),
                "min_s1_s2_interval_sec":     float(param(self.params, "min_s1_s2_interval_sec")),
                "s1_s2_interval_cap_sec":     float(param(self.params, "s1_s2_interval_cap_sec")),
                "output_smoothing_window_sec": float(param(self.params, "output_smoothing_window_sec")),
            },
        }
        # Pass 3: state timeline overlay (compact strip above chart).
        def _segments_from_boundaries(boundaries):
            segs_local = []
            for s0, s1, state_name, meta in (boundaries or []):
                try:
                    start_sec = float(s0) / float(self.sample_rate)
                    end_sec = float(s1) / float(self.sample_rate)
                    if not np.isfinite(start_sec) or not np.isfinite(end_sec) or end_sec <= start_sec:
                        continue
                    seg_dict: Dict = {
                        "start": start_sec,
                        "end": end_sec,
                        "state": str(state_name),
                    }
                    if isinstance(meta, dict) and "reasoning" in meta:
                        seg_dict["reasoning"] = meta["reasoning"]
                    segs_local.append(seg_dict)
                except Exception:
                    continue
            return segs_local

        segs_after = _segments_from_boundaries(getattr(self, "_pass3_state_boundaries", None))
        segs_before = _segments_from_boundaries(getattr(self, "_pass3_state_boundaries_before", None))
        if segs_after or segs_before:
            if segs_after:
                config_payload["pass3SegmentsAfter"] = segs_after
                # Back-compat: keep old key as "after" when present.
                config_payload["pass3Segments"] = segs_after
            if segs_before:
                config_payload["pass3SegmentsBefore"] = segs_before
            config_payload["pass3SegmentsEncoding"] = getattr(self, "_pass3_state_labels_encoding", {}) or {}
            config_payload["pass3SegmentsDefaultView"] = "after" if segs_after else "before"
        noise_segs = getattr(self, "_noise_event_segments", None) or []
        calc_noise_strip = bool(param(self.params, "pass3_calculate_noisy_regions"))
        p3_noise_ivs = getattr(self, "_pass3_noise_unreliable_windows_samples", None) or []
        if calc_noise_strip and p3_noise_ivs:
            try:
                sr_n = float(self.sample_rate) if self.sample_rate else 0.0
            except Exception:
                sr_n = 0.0
            if sr_n > 0:
                out_nsegs = []
                for w in p3_noise_ivs:
                    if not isinstance(w, dict):
                        continue
                    try:
                        a = int(w.get("start_sample", -1))
                        b = int(w.get("end_sample", -1))
                    except Exception:
                        continue
                    if a < 0 or b <= a:
                        continue
                    out_nsegs.append({"start": float(a) / sr_n, "end": float(b) / sr_n})
                if out_nsegs:
                    config_payload["noiseEventSegments"] = out_nsegs
        if "noiseEventSegments" not in config_payload and noise_segs:
            config_payload["noiseEventSegments"] = noise_segs

        # Pass 3: debug windows for "large gap" state insert (sample indices → seconds)
        gap_wins = getattr(self, "_pass3_large_gap_windows_samples", None) or []
        if gap_wins and isinstance(gap_wins, list):
            out_gap = []
            try:
                sr = float(self.sample_rate) if self.sample_rate else None
            except Exception:
                sr = None
            if sr and sr > 0:
                for w in gap_wins:
                    if not isinstance(w, dict):
                        continue
                    try:
                        a = int(w.get("start_sample", -1))
                        b = int(w.get("end_sample", -1))
                    except Exception:
                        continue
                    if a < 0 or b <= a:
                        continue
                    d = {"start": float(a) / sr, "end": float(b) / sr}
                    # Optional extra debug fields for tooltip
                    for k in ("gap_region_candidate_state", "source_state", "trigger", "bpm_at_mid", "cycle0_samples", "segment_samples"):
                        if k in w:
                            d[k] = w[k]
                    out_gap.append(d)
            if out_gap:
                config_payload["pass3LargeGapSegments"] = out_gap

        # Pass 3: quiet-prefix windows trimmed from gap regions (sample indices → seconds)
        quiet_wins = getattr(self, "_pass3_gap_quiet_windows_samples", None) or []
        if quiet_wins and isinstance(quiet_wins, list):
            out_quiet = []
            try:
                srq = float(self.sample_rate) if self.sample_rate else None
            except Exception:
                srq = None
            if srq and srq > 0:
                for w in quiet_wins:
                    if not isinstance(w, dict):
                        continue
                    try:
                        a = int(w.get("start_sample", -1))
                        b = int(w.get("end_sample", -1))
                    except Exception:
                        continue
                    if a < 0 or b <= a:
                        continue
                    d = {"start": float(a) / srq, "end": float(b) / srq}
                    for k in ("gap_region_candidate_state", "trigger", "first_sensitive_peak_sample", "s1_expected_samples", "pre_pad_samples"):
                        if k in w:
                            d[k] = w[k]
                    out_quiet.append(d)
            if out_quiet:
                config_payload["pass3GapQuietSegments"] = out_quiet
        config_json = _json_for_html_inline_script(json.dumps(config_payload))

        use_inline_js = bool(_oo.get("html_inline_interactive_script", False))

        # --- Full interactive script (sidecar) vs minimal script embedded in HTML ---
        if not use_inline_js:
            try:
                js_src_path = os.path.join(os.path.dirname(__file__), "assets", "interactive_plot.js")
                js_dest_path = os.path.join(self.output_directory, "interactive_plot.js")
                if os.path.exists(js_src_path):
                    shutil.copy2(js_src_path, js_dest_path)
                    logging.info("Copied interactive_plot.js to %s", js_dest_path)
                else:
                    logging.error(
                        "interactive_plot.js not found at %s; HTML will reference a missing script.", js_src_path
                    )
            except Exception as e:
                logging.error("Failed to copy interactive_plot.js: %s", e)
        else:
            logging.info("HTML uses embedded minimal script (no interactive_plot.js copy).")

        minimal_js_path = os.path.join(os.path.dirname(__file__), "assets", "html_inline_minimal.js")
        if use_inline_js:
            try:
                with open(minimal_js_path, encoding="utf-8") as jf:
                    minimal_js_body = jf.read()
            except OSError as e:
                logging.error("Could not load %s: %s", minimal_js_path, e)
                raise
            minimal_js_body = minimal_js_body.replace("</script>", "<\\/script>")
            scripts_tail = (
                "    <script>\n        window.BPM_ANALYZER_CONFIG = "
                + config_json
                + ";\n    </script>\n    <script>\n"
                + minimal_js_body
                + "\n    </script>\n"
            )
        else:
            scripts_tail = (
                "    <script>\n        window.BPM_ANALYZER_CONFIG = "
                + config_json
                + ';\n    </script>\n    <script src="interactive_plot.js"></script>\n'
            )

        # --- Load template and substitute placeholders ---
        template_path = os.path.join(os.path.dirname(__file__), "assets", "template.html")
        try:
            with open(template_path, encoding="utf-8") as f:
                template = f.read()
        except OSError as e:
            logging.error("Could not load HTML template from %s: %s", template_path, e)
            raise

        total_time_str = f"{int(duration_sec // 60):02d}:{int(duration_sec % 60):02d}"

        # Category 'Show:' selector only on the final pass (per-pass output is a debug feature).
        category_filter_html = (
            _category_filter_html("debug") if is_final_pass else ""
        )

        return (
            template
            .replace("%%PLOT_TITLE%%", plot_title)
            .replace("%%AUDIO_FILE_NAME%%", audio_file_name)
            .replace("%%TOTAL_TIME%%", total_time_str)
            .replace("%%AUDIO_SOURCE_SELECT%%", audio_source_select_html)
            .replace("%%CATEGORY_FILTER%%", category_filter_html)
            .replace("%%SPECTROGRAM_SRC%%", spectrogram_original_src)
            .replace("%%HTML_INTERACTIVE_SCRIPTS%%", scripts_tail)
            .replace("%%PLOTLY_HTML%%", plotly_html)
        )
