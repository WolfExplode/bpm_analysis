import os
import logging
import time
import urllib.parse
import re
from time_utils import seconds_to_datetime
import csv
import shutil
import json
import tempfile
from typing import Dict, Optional, List, Any, Tuple
from peak_utils import PeakType, _get_peak_type_from_debug, format_debug_entry, get_peak_prominence_details
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import DEFAULT_OUTPUT_OPTIONS, output_stem_from_path, strip_output_filename_emojis
from confidence_engine import calculate_bpm_intervals
from hrv import compute_s1_s2_interval_curve

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for spectrogram generation
import matplotlib.pyplot as plt


def _elapsed_seconds_to_plot_datetimes(seconds: np.ndarray) -> pd.DatetimeIndex:
    """
    Vectorized equivalent of
    pd.to_datetime([seconds_to_datetime(float(t)) for t in seconds]).
    Same local-epoch convention as time_utils.seconds_to_datetime (for Plotly x).
    """
    arr = np.asarray(seconds, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    base = pd.Timestamp(seconds_to_datetime(0.0))
    return base + pd.to_timedelta(arr, unit="s")


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
    Compute measured S1-S2 intervals from labeled pairs and BPM-expected systolic curve.
    Returns (observed_times, observed_intervals, expected_times, expected_intervals).
    """
    # Measured S1-S2 intervals from labeled pairs
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
    if smoothed_bpm is not None and bpm_times is not None:
        for t, bpm in zip(bpm_times, smoothed_bpm.values):
            intervals = calculate_bpm_intervals(float(bpm), params)
            expected_times.append(float(t))
            expected_intervals.append(intervals["s1_s2_nominal"])

    return observed_times, observed_intervals, expected_times, expected_intervals


def _compute_systolic_shift(
    obs_t: List[float],
    obs_iv: List[float],
    exp_t: List[float],
    exp_iv: List[float],
    peak_bpm_time_sec: Optional[float],
) -> Optional[float]:
    """
    Compute shift to align expected S1-S2 curve to measured data.
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
            logging.info(f"Generated spectrogram image for background overlay: {basename}")
            return basename

        except Exception as e:
            logging.warning(f"Failed to generate spectrogram image: {e}")
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
    ):
        """Generates and saves the main analysis plot by calling helper methods.
        pass_metrics: BPM/HRV/slope metrics for the pass being plotted (pass 2, pass 3, etc.).
        output_suffix: pass id for trace labels and systolic logic ('_pass2', '_pass3').
        filename_suffix: if set, used for HTML/PNG/CSV file names; if None, uses output_suffix or '_bpm_plot'.
        """
        self.fig = make_subplots(specs=[[{"secondary_y": True}]])
        self.time_axis_sec = np.arange(len(audio_envelope)) / self.sample_rate
        self.audio_duration_sec = self.time_axis_sec[-1] if len(self.time_axis_sec) > 0 else 0
        self._has_systolic_traces = False

        # Long-plot optimization: optionally skip heavy debug traces for very long recordings.
        optimize_long_plots = bool(self.params.get("optimize_long_plots", False))
        long_threshold_sec = float(self.params.get("long_plot_duration_threshold_sec", 600.0))
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
        self._add_bpm_hrv_traces(
            pass_metrics.get("smoothed_bpm"),
            analysis_data,
            pass_metrics.get("windowed_hrv_df"),
            output_suffix=output_suffix,
            pass1_bpm_series=pass1_bpm_series,
            pass1_bpm_times=pass1_bpm_times,
            instant_bpm=pass_metrics.get("instant_bpm"),
            bpm_times=pass_metrics.get("bpm_times"),
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
            pass_metrics.get("smoothed_bpm"),
            pass_metrics.get("hrv_summary"),
            pass_metrics.get("hrr_stats"),
            pass_metrics.get("peak_recovery_stats"),
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
                    logging.warning(f"Failed to generate original spectrogram: {e}")
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
                plotly_html, plot_title, base_name, output_options=output_options
            )

            with open(output_html_path, 'w', encoding='utf-8') as f:
                f.write(custom_html)
            logging.info(f"Interactive plot with audio player saved to {output_html_path}")
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
                logging.info(f"Plot PNG exported via Kaleido to {output_png_path}")
            except Exception as e:
                logging.warning(f"Failed to export Plot PNG (requires kaleido): {e}")

        if output_options is None or output_options.get("csv", True):
            smoothed_bpm = pass_metrics.get("smoothed_bpm")
            bpm_times = pass_metrics.get("bpm_times")
            if smoothed_bpm is not None and not smoothed_bpm.empty and bpm_times is not None:
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
                        for t, bpm in zip(bpm_times, smoothed_bpm.values):
                            if not np.isnan(bpm):
                                writer.writerow([f"{t:.3f}", f"{bpm:.3f}"])
                    logging.info(f"BPM plot data saved to {csv_path}")
                except Exception as e:
                    logging.error(f"Failed to write BPM plot CSV: {e}")
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
        pass1_bpm_data: dict with curve_times, curve_bpm, scatter_times, scatter_bpm from compute_pass1_bpm_curve (display = algorithm input).
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

        factor = self.params.get("plot_downsample_factor", 5)
        n = len(audio_envelope)
        if factor > 1 and n >= factor:
            plot_secs = np.arange(0, n, factor, dtype=np.float64) / self.sample_rate
            plot_envelope = audio_envelope[::factor]
        else:
            plot_secs = np.arange(n, dtype=np.float64) / self.sample_rate
            plot_envelope = audio_envelope
        plot_time = _elapsed_seconds_to_plot_datetimes(plot_secs)
        fig.add_trace(
            go.Scatter(x=plot_time, y=plot_envelope, name="Audio Envelope", line=dict(color="#47a5c4")),
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

        # Canonical pass 1 BPM (same as algorithm): scatter + curve from pass1_bpm_data
        if pass1_bpm_data and "scatter_times" in pass1_bpm_data and "scatter_bpm" in pass1_bpm_data:
            st = pass1_bpm_data["scatter_times"]
            sb = pass1_bpm_data["scatter_bpm"]
            if len(st) > 0 and len(st) == len(sb):
                scatter_dt = _elapsed_seconds_to_plot_datetimes(np.asarray(st, dtype=np.float64))
                fig.add_trace(
                    go.Scatter(
                        x=scatter_dt,
                        y=sb,
                        name="Instant BPM (pass 1)",
                        mode="markers",
                        marker=dict(size=6, color="#e74c3c", symbol="circle"),
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
        if not (pass1_bpm_data and "scatter_bpm" in pass1_bpm_data and len(pass1_bpm_data["scatter_bpm"]) > 0):
            self.bpm_axis_center = float(self.params.get("default_bpm_axis_center", self.bpm_axis_center))

        # BPM Trend (Belief) from pass 1 (EMA from accepted beats during that run)
        if pass1_analysis_data and "long_term_bpm_series" in pass1_analysis_data and not pass1_analysis_data["long_term_bpm_series"].empty:
            lt_series = pass1_analysis_data["long_term_bpm_series"]
            lt_times_dt = _elapsed_seconds_to_plot_datetimes(np.asarray(lt_series.index, dtype=np.float64))
            fig.add_trace(
                go.Scatter(
                    x=lt_times_dt,
                    y=lt_series.values,
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
        custom_html = self._generate_custom_html(
            plotly_html, plot_title, base_name, output_options=output_options
        )
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(custom_html)
        logging.info(f"Preliminary pass plot saved to {output_html_path}")

        self.fig = make_subplots(specs=[[{"secondary_y": True}]])

    def _prepare_bpm_axis_center(self, pass_metrics: Dict):
        """Use detected BPM stats to keep the BPM axis centered without altering the per-file zoom."""
        hrv_summary = pass_metrics.get("hrv_summary") or {}
        avg_bpm = hrv_summary.get("avg_bpm")
        smoothed_bpm = pass_metrics.get("smoothed_bpm")
        if avg_bpm is None and smoothed_bpm is not None and not smoothed_bpm.empty:
            avg_bpm = float(smoothed_bpm.mean())
        if avg_bpm is None:
            avg_bpm = float(self.params.get("default_bpm_axis_center", self.bpm_axis_center))
        self.bpm_axis_center = float(avg_bpm)

    def _configure_layout(self):
        """Sets up the plot layout, titles, and axes with custom x-axis tick labels."""
        self.fig.update_layout(
            template="plotly_dark",
            dragmode="pan",
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
            margin=dict(t=90, b=30, l=40, r=00), # borders/margins
            hovermode="x unified",
            autosize=True,
            uirevision="layout-stable",
        )

        duration_sec = float(self.time_axis_sec[-1])
        tick_interval_sec = 30
        tick_positions_sec = np.arange(0, duration_sec + 1e-6, tick_interval_sec, dtype=float)
        if tick_positions_sec.size > 0 and tick_positions_sec[-1] < duration_sec:
            tick_positions_sec = np.append(tick_positions_sec, duration_sec)
        tickvals = _elapsed_seconds_to_plot_datetimes(tick_positions_sec)
        ticktext = [f"{int(s // 60):02d}:{int(s % 60):02d} ({s:.2f})" for s in tick_positions_sec]

        self.fig.update_xaxes(
            title_text="Time",
            tickvals=tickvals,
            ticktext=ticktext,
            hoverformat="%M:%S.%L",
            automargin=False,
            title_standoff=4,
            domain=[0, 0.95],
        )

        # Use the audio envelope trace, if present, to scale the amplitude axis.
        robust_upper_limit = 1
        if self.fig.data:
            envelope_values = None
            for trace in self.fig.data:
                if getattr(trace, "name", "") == "Audio Envelope" and hasattr(trace, "y"):
                    try:
                        envelope_values = np.asarray(trace.y, dtype=float)
                    except Exception:
                        envelope_values = None
                    break
            if envelope_values is not None and envelope_values.size > 0:
                robust_upper_limit = float(np.quantile(envelope_values, 0.95))

        amplitude_scale = self.params.get("plot_amplitude_scale_factor", 60.0)
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
                    title="Systolic Interval (s)",
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
            return
        plot_envelope = audio_envelope
        plot_noise_floor = analysis_data.get("dynamic_noise_floor_series")

        # Downsample only envelope and noise floor for performance; other traces (contractility, BPM, HRV) use full data
        factor = self.params.get("plot_downsample_factor", 5)
        n = len(audio_envelope)
        if factor > 1 and n >= factor:
            logging.info(f"Downsampling envelope and noise floor by factor {factor} for plotting.")
            plot_secs = np.arange(0, n, factor, dtype=np.float64) / self.sample_rate
            plot_time_axis_dt = _elapsed_seconds_to_plot_datetimes(plot_secs)
            plot_envelope = audio_envelope[::factor]
            if plot_noise_floor is not None and not plot_noise_floor.empty:
                plot_noise_floor = plot_noise_floor.iloc[::factor]
        else:
            plot_secs = np.arange(n, dtype=np.float64) / self.sample_rate
            plot_time_axis_dt = _elapsed_seconds_to_plot_datetimes(plot_secs)

        self.fig.add_trace(
            go.Scatter(x=plot_time_axis_dt, y=plot_envelope, name="Audio Envelope", line=dict(color="#47a5c4")),
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
                sub_text = "<br>".join(l.replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;") for l in formatted_lines)
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
        segment_sec = float(self.params.get("contractility_average_window_sec", 2.0))
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
    ):
        """Adds BPM, pass 1 BPM curve (when provided), and HRV traces. BPM Trend (Belief) is only on the pass 1 plot."""
        # Label smoothed BPM by pass when known (Pass 2 / Pass 3), else generic
        if output_suffix == "_pass2":
            bpm_trace_name = "BPM (Pass 2)"
        elif output_suffix == "_pass3":
            bpm_trace_name = "BPM (Pass 3)"
        else:
            bpm_trace_name = "Average BPM"
        if smoothed_bpm is not None and not smoothed_bpm.empty:
            self.fig.add_trace(
                go.Scatter(
                    x=smoothed_bpm.index, y=smoothed_bpm.values, name=bpm_trace_name, line=dict(color="#4a4a4a", width=3)
                ),
                secondary_y=True,
            )

        # Instantaneous BPM at each beat (pass 2 only)
        if (
            output_suffix == "_pass2"
            and instant_bpm is not None
            and bpm_times is not None
            and len(instant_bpm) == len(bpm_times)
            and len(bpm_times) > 0
        ):
            instant_times_dt = _elapsed_seconds_to_plot_datetimes(np.asarray(bpm_times, dtype=np.float64))
            self.fig.add_trace(
                go.Scatter(
                    x=instant_times_dt,
                    y=instant_bpm,
                    name="Instantaneous BPM (Pass 2)",
                    mode="markers",
                    marker=dict(size=5, color="#4a4a4a", symbol="circle"),
                    visible="legendonly",
                ),
                secondary_y=True,
            )

        # Prior BPM curve: pass 1 on pass 2 plot, pass 2 on pass 3 plot
        if (
            pass1_bpm_series is not None
            and not pass1_bpm_series.empty
            and pass1_bpm_times is not None
            and len(pass1_bpm_times) == len(pass1_bpm_series)
        ):
            prior_curve_name = "BPM (Pass 2)" if output_suffix == "_pass3" else "BPM (pass 1)"
            pass1_times_dt = _elapsed_seconds_to_plot_datetimes(np.asarray(pass1_bpm_times, dtype=np.float64))
            self.fig.add_trace(
                go.Scatter(
                    x=pass1_times_dt,
                    y=pass1_bpm_series.values,
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
        """Add BPM-expected systolic curve, measured S1-S2 datapoints (outlier-filtered), and best-fit curve on yaxis3 (Pass 2 and Pass 3 only)."""
        if output_suffix not in ("_pass2", "_pass3"):
            return
        obs_t, obs_iv, exp_t, exp_iv = _compute_systolic_interval_data(
            analysis_data, pass_metrics, self.sample_rate, self.params
        )
        if not exp_t and not obs_t:
            return
        to_dt = lambda seq: _elapsed_seconds_to_plot_datetimes(np.asarray(seq, dtype=np.float64))
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
                    name="Expected S1-S2 from BPM",
                    line=dict(color="cyan", width=2),
                    yaxis="y3",
                    visible="legendonly",
                ),
            )
        if obs_t:
            # Outlier removal + LOESS best-fit curve (reuses pass1 BPM pattern)
            fit_data = compute_s1_s2_interval_curve(
                np.array(obs_t), np.array(obs_iv), self.params
            )
            if fit_data is not None:
                scatter_t = fit_data["scatter_times"]
                scatter_iv = fit_data["scatter_intervals"]
                curve_t = fit_data["curve_times"]
                curve_iv = fit_data["curve_intervals"]
                self.fig.add_trace(
                    go.Scatter(
                        x=to_dt(curve_t),
                        y=curve_iv,
                        name="S1-S2 Interval",
                        line=dict(color="orange", width=2),
                        yaxis="y3",
                        visible="legendonly",
                    ),
                )
                self.fig.add_trace(
                    go.Scatter(
                        x=to_dt(scatter_t),
                        y=scatter_iv,
                        name="Measured S1-S2",
                        mode="markers",
                        marker=dict(size=6, color="lime", symbol="circle"),
                        yaxis="y3",
                        visible="legendonly",
                    ),
                )
            else:
                # Fallback: plot raw if curve fit fails (e.g. too few points)
                self.fig.add_trace(
                    go.Scatter(
                        x=to_dt(obs_t),
                        y=obs_iv,
                        name="Measured S1-S2",
                        mode="markers",
                        marker=dict(size=6, color="lime", symbol="circle"),
                        yaxis="y3",
                        visible="legendonly",
                    ),
                )
        self._has_systolic_traces = True

    def _add_annotations_and_summary(self, smoothed_bpm, hrv_summary, hrr_stats, peak_recovery_stats):
        """Adds min/max BPM annotations on the plot and builds plain-text summary for the HTML Analysis Summary modal."""
        if smoothed_bpm is not None and not smoothed_bpm.empty:
            max_bpm_val = smoothed_bpm.max()
            min_bpm_val = smoothed_bpm.min()
            max_bpm_time = smoothed_bpm.idxmax()
            min_bpm_time = smoothed_bpm.idxmin()

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
                        x=[incline["start_time"], incline["end_time"]],
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
                        x=[decline["start_time"], decline["end_time"]],
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
                    x=[stats["start_time"], stats["end_time"]],
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
                    x=[stats["start_time"], stats["end_time"]],
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
    ) -> str:
        """
        Generates custom HTML with audio player, timeline scrubber, and synchronized playhead.
        Loads assets/template.html and substitutes %%PLACEHOLDER%% tokens with computed values.
        Uses %%HTML_INTERACTIVE_SCRIPTS%% for config + either interactive_plot.js or inlined html_inline_minimal.js.
        """
        audio_file_name = os.path.basename(self.audio_source_path)
        _stem, _ext = os.path.splitext(audio_file_name)
        output_audio_basename = strip_output_filename_emojis(_stem) + _ext
        duration_sec = self.audio_duration_sec or 0

        # --- Resolve audio source path ---
        audio_src = ""
        if os.path.exists(self.audio_source_path):
            dest_audio_path = os.path.join(self.output_directory, output_audio_basename)
            if os.path.abspath(self.audio_source_path) != os.path.abspath(dest_audio_path):
                try:
                    shutil.copy2(self.audio_source_path, dest_audio_path)
                    logging.info(f"Copied audio file to {dest_audio_path}")
                except Exception as e:
                    logging.error(f"Could not copy audio file: {e}")
            audio_src = output_audio_basename.replace('\\', '/')
        else:
            logging.error(f"Audio source file does NOT exist: {self.audio_source_path}")
            dest_audio_path = os.path.join(self.output_directory, output_audio_basename)
            if os.path.exists(dest_audio_path):
                audio_src = output_audio_basename.replace('\\', '/')
                logging.info(f"Found audio file in output directory: {dest_audio_path}")
            else:
                logging.error(f"Audio file not found anywhere: {output_audio_basename}")

        filtered_debug_file_name = f"{base_name}_filtered_debug.wav"
        filtered_debug_path = os.path.join(self.output_directory, filtered_debug_file_name)
        filtered_available = os.path.exists(filtered_debug_path)
        filtered_audio_src = filtered_debug_file_name.replace('\\', '/') if filtered_available else ""
        if filtered_available:
            logging.info(f"Using filtered debug audio: {filtered_debug_path}")

        logging.info(f"HTML audio source path: '{audio_src}'")
        audio_src_escaped = urllib.parse.quote(audio_src)
        filtered_audio_src_escaped = urllib.parse.quote(filtered_audio_src) if filtered_audio_src else ""

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
                    logging.warning(f"Failed to generate on-demand original spectrogram: {e}")

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
                    logging.warning(f"Failed to generate filtered spectrogram: {e}")
        else:
            logging.info("Spectrogram generation disabled; no spectrogram images generated.")

        # --- Build audio source <select> ---
        audio_source_options = ['<option value="original">Original Audio</option>']
        if filtered_available:
            audio_source_options.append('<option value="filtered">Filtered Debug</option>')
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
            },
            "audioLabels": {
                "original": audio_file_name,
                "filtered": filtered_debug_file_name if filtered_available else audio_file_name,
            },
            "analysisSummary": getattr(self, "analysis_summary_text", "") or "",
            "htmlS1S2HoverOnByDefault": hover_on_by_default,
        }
        config_json = json.dumps(config_payload)

        use_inline_js = bool(_oo.get("html_inline_interactive_script", False))

        # --- Full interactive script (sidecar) vs minimal script embedded in HTML ---
        if not use_inline_js:
            try:
                js_src_path = os.path.join(os.path.dirname(__file__), "assets", "interactive_plot.js")
                js_dest_path = os.path.join(self.output_directory, "interactive_plot.js")
                if os.path.exists(js_src_path):
                    shutil.copy2(js_src_path, js_dest_path)
                    logging.info(f"Copied interactive_plot.js to {js_dest_path}")
                else:
                    logging.error(
                        f"interactive_plot.js not found at {js_src_path}; HTML will reference a missing script."
                    )
            except Exception as e:
                logging.error(f"Failed to copy interactive_plot.js: {e}")
        else:
            logging.info("HTML uses embedded minimal script (no interactive_plot.js copy).")

        minimal_js_path = os.path.join(os.path.dirname(__file__), "assets", "html_inline_minimal.js")
        if use_inline_js:
            try:
                with open(minimal_js_path, encoding="utf-8") as jf:
                    minimal_js_body = jf.read()
            except OSError as e:
                logging.error(f"Could not load {minimal_js_path}: {e}")
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
            logging.error(f"Could not load HTML template from {template_path}: {e}")
            raise

        total_time_str = f"{int(duration_sec // 60):02d}:{int(duration_sec % 60):02d}"

        return (
            template
            .replace("%%PLOT_TITLE%%", plot_title)
            .replace("%%AUDIO_FILE_NAME%%", audio_file_name)
            .replace("%%TOTAL_TIME%%", total_time_str)
            .replace("%%AUDIO_SOURCE_SELECT%%", audio_source_select_html)
            .replace("%%SPECTROGRAM_SRC%%", spectrogram_original_src)
            .replace("%%HTML_INTERACTIVE_SCRIPTS%%", scripts_tail)
            .replace("%%PLOTLY_HTML%%", plotly_html)
        )
