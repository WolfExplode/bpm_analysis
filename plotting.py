import os
import datetime
import logging
import csv
import base64
import shutil
import io
import json
from typing import Dict, Optional, List, Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for spectrogram generation
import matplotlib.pyplot as plt


class Plotter:
    """Handles the creation and generation of the final analysis plot."""

    def __init__(
        self,
        file_name: str,
        params: Dict,
        sample_rate: int,
        output_directory: str,
        source_audio_path: Optional[str] = None,
        peak_type_helper: Optional[Callable[[Any], str]] = None,
        format_debug_entry_func: Optional[Callable[[Dict], List[str]]] = None,
        peak_type_cls: Optional[Any] = None,
    ):
        self.file_name = file_name
        self.params = params
        self.sample_rate = sample_rate
        self.output_directory = output_directory
        self.audio_source_path = source_audio_path or file_name
        self.fig = make_subplots(specs=[[{"secondary_y": True}]])
        self.audio_duration_sec = None  # Will be set during plot_and_save
        # Optional spectrogram image (original audio); filtered spectrograms are generated on demand.
        self.spectrogram_base64: Optional[str] = None
        self.bpm_axis_center: float = float(params.get("default_bpm_axis_center", 125))
        self.bpm_axis_span: float = float(params.get("bpm_axis_span", 150))

        # Debug formatting helpers are injected from the analysis module to avoid
        # importing bpm_analysis here (which would create a circular dependency).
        # If they are not provided, we fall back to no-op implementations.
        self._get_peak_type_from_debug: Callable[[Any], str] = peak_type_helper or (lambda entry: "")
        self._format_debug_entry: Callable[[Dict], List[str]] = format_debug_entry_func or (lambda entry: [])
        self._PeakType = peak_type_cls

    def _generate_spectrogram_image(self, audio_path: str) -> Optional[str]:
        """
        Generate a spectrogram image from the audio file and return as base64 PNG.
        The spectrogram is rendered with a transparent background for overlay.
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
            n_mels = 512

            # Generate mel spectrogram
            S = librosa.feature.melspectrogram(
                y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
            )

            # Convert to dB scale
            S_dB = librosa.power_to_db(S, ref=np.max)

            # Calculate figure dimensions based on audio duration
            duration = len(audio_data) / sr
            # Width should be proportional to duration, height fixed
            fig_width = max(20, duration / 10)  # Scale width with duration
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

            # Save to buffer as PNG with transparency
            buf = io.BytesIO()
            fig.savefig(
                buf,
                format="png",
                transparent=True,
                dpi=100,
                bbox_inches="tight",
                pad_inches=0,
            )
            buf.seek(0)

            # Encode to base64
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")

            plt.close(fig)
            buf.close()

            logging.info("Generated spectrogram image for background overlay")
            return img_base64

        except Exception as e:
            logging.warning(f"Failed to generate spectrogram image: {e}")
            return None

    def plot_and_save(
        self,
        audio_envelope: np.ndarray,
        all_raw_peaks: np.ndarray,
        analysis_data: Dict,
        final_metrics: Dict,
        output_options: Optional[Dict] = None,
    ):
        """Generates and saves the main analysis plot by calling helper methods."""
        self.time_axis_sec = np.arange(len(audio_envelope)) / self.sample_rate
        self.audio_duration_sec = self.time_axis_sec[-1] if len(self.time_axis_sec) > 0 else 0
        self._springer_segments = analysis_data.get("springer_segments") or []
        self._springer_pipeline_viz = analysis_data.get("springer_pipeline_viz")

        # Long-plot optimization: optionally skip heavy debug traces for very long recordings.
        optimize_long_plots = bool(self.params.get("optimize_long_plots", False))
        long_threshold_sec = float(self.params.get("long_plot_duration_threshold_sec", 600.0))
        # Only skip details if the recording is longer than the threshold; shorter files always show full detail.
        self.skip_detailed_debug_traces = optimize_long_plots and self.audio_duration_sec > long_threshold_sec

        time_axis_dt = pd.to_datetime(
            [datetime.datetime.fromtimestamp(0) + datetime.timedelta(seconds=t) for t in self.time_axis_sec]
        )

        self._add_line_traces(time_axis_dt, audio_envelope, analysis_data)
        if analysis_data.get("segmenter") == "springer":
            self._add_springer_pipeline_overlay_traces(time_axis_dt, audio_envelope, analysis_data)
            self._add_springer_s1_s2_markers(audio_envelope, analysis_data)
        else:
            self._add_trough_markers(audio_envelope, analysis_data)
            self._add_peak_traces(
                all_raw_peaks,
                analysis_data.get("beat_debug_info", {}),
                audio_envelope,
                analysis_data.get("trough_indices"),
            )
        self._add_bpm_hrv_traces(
            final_metrics.get("smoothed_bpm"), analysis_data, final_metrics.get("windowed_hrv_df")
        )
        if analysis_data.get("segmenter") != "springer":
            self._add_slope_traces(
                final_metrics.get("major_inclines"),
                final_metrics.get("major_declines"),
                final_metrics.get("peak_recovery_stats"),
                final_metrics.get("peak_exertion_stats"),
            )
            self._add_trapezoid_shapes(final_metrics.get("trapezoids"))
        self._add_annotations_and_summary(
            final_metrics.get("smoothed_bpm"),
            final_metrics.get("hrv_summary"),
            final_metrics.get("hrr_stats"),
            final_metrics.get("peak_recovery_stats"),
        )
        self._prepare_bpm_axis_center(final_metrics)

        self._configure_layout()

        base_name = os.path.basename(os.path.splitext(self.file_name)[0])
        output_html_path = os.path.join(self.output_directory, f"{base_name}_bpm_plot.html")
        output_png_path = os.path.join(self.output_directory, f"{base_name}_bpm_plot.png")
        plot_title = f"Heartbeat Analysis - {os.path.basename(self.file_name)}"
        plot_config = {"scrollZoom": True, "toImageButtonOptions": {"filename": plot_title, "format": "png", "scale": 2}}

        html_requested = True if output_options is None else output_options.get("html", True)
        png_requested = False if output_options is None else output_options.get("png", False)

        if html_requested:
            # Determine whether spectrogram generation is enabled (can be disabled via GUI/output options).
            self.spectrogram_enabled = True
            if output_options is not None:
                self.spectrogram_enabled = output_options.get("spectrogram", True)

            # Generate spectrogram image for optional background overlay (original audio only).
            # Filtered spectrograms are generated later in `_generate_custom_html` if needed.
            if self.spectrogram_enabled:
                try:
                    self.spectrogram_base64 = self._generate_spectrogram_image(
                        self.audio_source_path or self.file_name
                    )
                except Exception as e:
                    logging.warning(f"Failed to generate original spectrogram: {e}")
            else:
                logging.info("Skipping original spectrogram generation as requested (spectrogram output disabled).")

            # Generate the base Plotly HTML
            plotly_html = self.fig.to_html(config=plot_config, full_html=False, include_plotlyjs='cdn')

            # Pipeline steps are overlaid on main plot for Springer; no separate figure
            pipeline_steps_html = ""

            # Generate custom HTML with audio player and playhead
            custom_html = self._generate_custom_html(
                plotly_html, plot_title, base_name, pipeline_steps_html=pipeline_steps_html
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

            try:
                # Note: Kaleido must be installed for write_image() to work.
                write_kwargs = {
                    "format": "png",
                    "scale": png_scale,
                    "width": png_width,
                    "height": png_height,
                }
                self.fig.write_image(output_png_path, **write_kwargs)
                logging.info(f"Plot PNG exported to {output_png_path}")
            except Exception as e:
                logging.warning(f"Failed to export Plot PNG (requires kaleido): {e}")

        if output_options is None or output_options.get("csv", True):
            smoothed_bpm = final_metrics.get("smoothed_bpm")
            bpm_times = final_metrics.get("bpm_times")
            if smoothed_bpm is not None and not smoothed_bpm.empty and bpm_times is not None:
                csv_path = os.path.join(self.output_directory, f"{base_name}_bpm_plot.csv")
                try:
                    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["Time (s)", "Average BPM"])
                        for t, bpm in zip(bpm_times, smoothed_bpm.values):
                            if not np.isnan(bpm):
                                writer.writerow([f"{t:.3f}", f"{bpm:.3f}"])
                    logging.info(f"BPM plot data saved to {csv_path}")
                except Exception as e:
                    logging.error(f"Failed to write BPM plot CSV: {e}")
        else:
            logging.info("Skipping CSV generation as requested.")

        return self.fig

    def _prepare_bpm_axis_center(self, final_metrics: Dict):
        """Use detected BPM stats to keep the BPM axis centered without altering the per-file zoom."""
        hrv_summary = final_metrics.get("hrv_summary") or {}
        avg_bpm = hrv_summary.get("avg_bpm")
        smoothed_bpm = final_metrics.get("smoothed_bpm")
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
            legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1),
            margin=dict(t=80, b=100, l=100, r=10),
            hovermode="x unified",
            autosize=True,
            uirevision="layout-stable",
        )

        duration_sec = float(self.time_axis_sec[-1])
        tick_interval_sec = 30
        tick_positions_sec = np.arange(0, duration_sec + 1e-6, tick_interval_sec, dtype=float)
        if tick_positions_sec.size > 0 and tick_positions_sec[-1] < duration_sec:
            tick_positions_sec = np.append(tick_positions_sec, duration_sec)
        epoch = datetime.datetime.fromtimestamp(0)

        tickvals = [epoch + datetime.timedelta(seconds=s) for s in tick_positions_sec]
        ticktext = [f"{int(s // 60):02d}:{int(s % 60):02d} ({s:.2f})" for s in tick_positions_sec]

        self.fig.update_xaxes(
            title_text="Time",
            tickvals=tickvals,
            ticktext=ticktext,
            hoverformat="%M:%S.%L",
            automargin=False,
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
        )

    def _add_springer_pipeline_overlay_traces(
        self,
        time_axis_dt: pd.Series,
        audio_envelope: np.ndarray,
        analysis_data: Dict,
    ) -> None:
        """Add pipeline step signals as overlaid traces on the main plot (normalized 0–1) with legend entries."""
        viz = getattr(self, "_springer_pipeline_viz", None)
        if not viz or not isinstance(viz, dict):
            return
        filtered_audio = viz.get("filtered_audio")
        homomorphic_env = viz.get("homomorphic_env")
        envelope_50hz = viz.get("envelope_50hz")
        time_50hz = viz.get("time_50hz")
        features_50hz = viz.get("features_50hz")
        fs = viz.get("sample_rate")
        if filtered_audio is None or homomorphic_env is None or envelope_50hz is None or time_50hz is None or not fs:
            return
        filtered_audio = np.asarray(filtered_audio).flatten()
        homomorphic_env = np.asarray(homomorphic_env).flatten()
        envelope_50hz = np.asarray(envelope_50hz).flatten()
        time_50hz = np.asarray(time_50hz).flatten()
        n = len(audio_envelope)
        factor = self.params.get("plot_downsample_factor", 5)
        if factor > 1 and n >= factor:
            plot_time_axis_dt = time_axis_dt[::factor]
            fa = filtered_audio[:n][::factor]
        else:
            plot_time_axis_dt = time_axis_dt
            fa = filtered_audio[:n]
        fs_int = int(fs)
        def norm01(y):
            y = np.asarray(y, dtype=np.float64)
            mn, mx = np.nanmin(y), np.nanmax(y)
            if mx - mn < 1e-12:
                return np.zeros_like(y)
            return (y - mn) / (mx - mn)
        # Filtered PCG only (homomorphic env = main "Audio Envelope", skip redundant overlay)
        self.fig.add_trace(
            go.Scatter(
                x=plot_time_axis_dt,
                y=norm01(fa),
                name=f"Filtered PCG ({fs_int} Hz)",
                line=dict(color="#7f7f7f", width=0.8),
            ),
            secondary_y=False,
        )
        # Use features length for 50 Hz axis (Feature 0 = z-scored homomorphic @ 50 Hz, so skip redundant "Homomorphic env (50 Hz)")
        n_f = min(len(time_50hz), len(envelope_50hz))
        if n_f == 0:
            return
        time_50hz = time_50hz[:n_f]
        time_50_dt = pd.to_datetime(
            [datetime.datetime.fromtimestamp(0) + datetime.timedelta(seconds=float(t)) for t in time_50hz]
        )
        if features_50hz is not None and np.asarray(features_50hz).ndim == 2:
            features_50hz = np.asarray(features_50hz)[:n_f]
            if features_50hz.shape[1] >= 4:
                colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
                labels = [
                    "Feature 0: Homomorphic (50 Hz)",
                    "Feature 1: Hilbert (50 Hz)",
                    "Feature 2: PSD 40–60 Hz (50 Hz)",
                    "Feature 3: DWT (50 Hz)",
                ]
                for i, (label, c) in enumerate(zip(labels, colors)):
                    self.fig.add_trace(
                        go.Scatter(
                            x=time_50_dt,
                            y=norm01(features_50hz[:, i]),
                            name=label,
                            line=dict(color=c, width=0.7),
                        ),
                        secondary_y=False,
                    )
        # Step 4 debug: P(state|O) posteriors (LR output) and decoded state qt at 50 Hz.
        # Posteriors should be high for the correct state at each time (e.g. P(S1|O) high during S1).
        posteriors_50hz = viz.get("posteriors_50hz")
        qt_50hz = viz.get("qt_50hz")
        if posteriors_50hz is not None and qt_50hz is not None:
            posteriors_50hz = np.asarray(posteriors_50hz)
            qt_50hz = np.asarray(qt_50hz).flatten()
            n_em = min(n_f, posteriors_50hz.shape[0] if posteriors_50hz.ndim >= 1 else 0, len(qt_50hz))
            if n_em > 0 and posteriors_50hz.ndim == 2 and posteriors_50hz.shape[1] >= 4:
                time_50_dt_em = time_50_dt[:n_em]
                em_colors = ["#4a90d9", "#e67e22", "#27ae60", "#c0392b"]
                em_labels = [
                    "P(S1|O) — high when S1",
                    "P(Systole|O) — high when systole",
                    "P(S2|O) — high when S2",
                    "P(Diastole|O) — high when diastole",
                ]
                for i, (label, c) in enumerate(zip(em_labels, em_colors)):
                    self.fig.add_trace(
                        go.Scatter(
                            x=time_50_dt_em,
                            y=norm01(posteriors_50hz[:n_em, i]),
                            name=label,
                            line=dict(color=c, width=0.8),
                        ),
                        secondary_y=False,
                    )
                # Decoded state qt: map 1->0, 2->0.33, 3->0.66, 4->1 for visibility
                qt_scaled = (qt_50hz[:n_em].astype(np.float64) - 1.0) / 3.0
                self.fig.add_trace(
                    go.Scatter(
                        x=time_50_dt_em,
                        y=qt_scaled,
                        name="Decoded state qt (50 Hz)",
                        line=dict(color="#2c3e50", width=1.2, shape="hv"),
                    ),
                    secondary_y=False,
                )

    def _build_springer_pipeline_steps_html(self) -> str:
        """Build one multi-panel figure (Step 1 + Step 2 features) with shared x-axis for synced zoom/pan."""
        viz = getattr(self, "_springer_pipeline_viz", None)
        if not viz or not isinstance(viz, dict):
            return ""
        filtered_audio = viz.get("filtered_audio")
        homomorphic_env = viz.get("homomorphic_env")
        envelope_50hz = viz.get("envelope_50hz")
        time_50hz = viz.get("time_50hz")
        features_50hz = viz.get("features_50hz")
        fs = viz.get("sample_rate")
        if filtered_audio is None or homomorphic_env is None or envelope_50hz is None or time_50hz is None or not fs:
            return ""
        filtered_audio = np.asarray(filtered_audio).flatten()
        homomorphic_env = np.asarray(homomorphic_env).flatten()
        envelope_50hz = np.asarray(envelope_50hz).flatten()
        time_50hz = np.asarray(time_50hz).flatten()
        has_features = features_50hz is not None and np.asarray(features_50hz).ndim == 2
        if has_features:
            features_50hz = np.asarray(features_50hz)
            n_f = min(features_50hz.shape[0], len(time_50hz), len(envelope_50hz))
            time_50hz = time_50hz[:n_f]
            envelope_50hz = envelope_50hz[:n_f]
            features_50hz = features_50hz[:n_f]
        max_dur_sec = 60.0
        n_max = min(len(filtered_audio), int(max_dur_sec * fs))
        t_sec = np.arange(n_max) / float(fs)
        fa = filtered_audio[:n_max]
        he = homomorphic_env[:n_max]
        if n_max > 15000:
            step = max(1, n_max // 15000)
            t_sec = t_sec[::step]
            fa = fa[::step]
            he = he[::step]
        mask_50 = time_50hz <= max_dur_sec
        t50 = time_50hz[mask_50]
        e50 = envelope_50hz[mask_50]

        subplot_titles = [
            "Step 1a: Filtered PCG (25–400 Hz)",
            "Step 1b: Homomorphic envelope",
            "Envelope at 50 Hz",
            "Step 2: Feature 0 (homomorphic)",
            "Step 2: Feature 1 (Hilbert)",
            "Step 2: Feature 2 (PSD 40–60 Hz)",
            "Step 2: Feature 3 (wavelet)",
        ]
        n_rows = 7 if has_features else 3
        row_heights = [0.18, 0.18, 0.14, 0.12, 0.12, 0.12, 0.14] if has_features else [0.35, 0.35, 0.3]
        fig = make_subplots(
            rows=n_rows,
            cols=1,
            subplot_titles=subplot_titles[:n_rows],
            vertical_spacing=0.04,
            row_heights=row_heights,
            shared_xaxes=True,
        )
        fig.add_trace(go.Scatter(x=t_sec, y=fa, name="Filtered PCG", line=dict(color="#7f7f7f", width=0.8)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_sec, y=he, name="Homomorphic env", line=dict(color="#47a5c4", width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=t50, y=e50, name="Envelope @ 50 Hz", line=dict(color="#2ca02c", width=1)), row=3, col=1)
        if has_features and features_50hz.shape[1] >= 4:
            f0, f1, f2, f3 = features_50hz[mask_50, 0], features_50hz[mask_50, 1], features_50hz[mask_50, 2], features_50hz[mask_50, 3]
            fig.add_trace(go.Scatter(x=t50, y=f0, line=dict(color="#1f77b4", width=1)), row=4, col=1)
            fig.add_trace(go.Scatter(x=t50, y=f1, line=dict(color="#ff7f0e", width=1)), row=5, col=1)
            fig.add_trace(go.Scatter(x=t50, y=f2, line=dict(color="#2ca02c", width=1)), row=6, col=1)
            fig.add_trace(go.Scatter(x=t50, y=f3, line=dict(color="#d62728", width=1)), row=7, col=1)
        fig.update_layout(
            template="plotly_dark",
            height=720 if has_features else 420,
            margin=dict(t=50, b=40, l=50, r=30),
            showlegend=False,
            title_text="Springer pipeline – steps 1 & 2 (zoom/pan synced)",
        )
        fig.update_xaxes(title_text="Time (s)", row=n_rows, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Envelope", row=2, col=1)
        fig.update_yaxes(title_text="Envelope", row=3, col=1)
        if has_features:
            for r in range(4, 8):
                fig.update_yaxes(title_text="Norm", row=r, col=1)
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _add_line_traces(self, time_axis_dt: pd.Series, audio_envelope: np.ndarray, analysis_data: Dict):
        """Adds downsampled audio envelope and noise floor traces for performance."""
        if getattr(self, "skip_detailed_debug_traces", False):
            logging.info("Skipping audio envelope and noise floor traces for long file (optimization enabled).")
            return
        plot_time_axis_dt = time_axis_dt
        plot_envelope = audio_envelope
        plot_noise_floor = analysis_data.get("dynamic_noise_floor_series")

        factor = self.params.get("plot_downsample_factor", 5)
        if factor > 1 and len(audio_envelope) >= factor:
            logging.info(f"Downsampling line traces by a factor of {factor} for plotting.")
            plot_time_axis_dt = time_axis_dt[::factor]
            plot_envelope = audio_envelope[::factor]
            if plot_noise_floor is not None and not plot_noise_floor.empty:
                plot_noise_floor = plot_noise_floor.iloc[::factor]

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
                ),
                secondary_y=False,
            )

    def _add_springer_s1_s2_markers(self, audio_envelope: np.ndarray, analysis_data: Dict):
        """Adds only S1 and S2 markers from Springer segmenter (segment midpoints)."""
        s1_indices = analysis_data.get("springer_s1_indices")
        s2_indices = analysis_data.get("springer_s2_indices")
        if s1_indices is not None and s1_indices.size > 0:
            s1_indices = np.asarray(s1_indices).flatten()
            times_dt = pd.to_datetime(
                [
                    datetime.datetime.fromtimestamp(0) + datetime.timedelta(seconds=t)
                    for t in (s1_indices / self.sample_rate)
                ]
            )
            self.fig.add_trace(
                go.Scatter(
                    x=times_dt,
                    y=audio_envelope[s1_indices],
                    mode="markers",
                    name="S1 (Springer)",
                    marker=dict(color="#e36f6f", size=8, symbol="diamond"),
                    hovertemplate="S1 (Springer)<br>Time: %{x}<br>Amp: %{y:.0f}<extra></extra>",
                ),
                secondary_y=False,
            )
        if s2_indices is not None and s2_indices.size > 0:
            s2_indices = np.asarray(s2_indices).flatten()
            times_dt = pd.to_datetime(
                [
                    datetime.datetime.fromtimestamp(0) + datetime.timedelta(seconds=t)
                    for t in (s2_indices / self.sample_rate)
                ]
            )
            self.fig.add_trace(
                go.Scatter(
                    x=times_dt,
                    y=audio_envelope[s2_indices],
                    mode="markers",
                    name="S2 (Springer)",
                    marker=dict(color="orange", symbol="circle", size=6),
                    hovertemplate="S2 (Springer)<br>Time: %{x}<br>Amp: %{y:.0f}<extra></extra>",
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
            trough_times_dt = pd.to_datetime(
                [datetime.datetime.fromtimestamp(0) + datetime.timedelta(seconds=t) for t in (trough_indices / self.sample_rate)]
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

            peak_type = self._get_peak_type_from_debug(debug_value) or "Unknown Peak"
            hover_text_parts.append(f"<b>Type:</b> {peak_type}")
            hover_text_parts.append(f"<b>Time:</b> {peak_idx / self.sample_rate:.2f}s")
            hover_text_parts.append(f"<b>Amp:</b> {audio_envelope[peak_idx]:.0f}")
            hover_text_parts.append("---")

            formatted_lines = self._format_debug_entry(debug_value)
            if formatted_lines:
                sub_text = "<br>".join(l.replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;") for l in formatted_lines)
                hover_text_parts.append(sub_text)

            full_hover_text = "<br>".join(hover_text_parts)
            classified_indices.add(peak_idx)

            if self._PeakType is not None and self._PeakType.is_s1(peak_type):
                s1_peaks["indices"].append(peak_idx)
                s1_peaks["customdata"].append(full_hover_text)
            elif self._PeakType is not None and self._PeakType.is_s2(peak_type):
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

        if s1_peaks["indices"]:
            times_dt = pd.to_datetime(
                [
                    datetime.datetime.fromtimestamp(0) + datetime.timedelta(seconds=t)
                    for t in (np.array(s1_peaks["indices"]) / self.sample_rate)
                ]
            )
            self.fig.add_trace(
                go.Scatter(
                    x=times_dt,
                    y=audio_envelope[s1_peaks["indices"]],
                    mode="markers",
                    name="S1 Beats",
                    marker=dict(color="#e36f6f", size=8, symbol="diamond"),
                    customdata=s1_peaks["customdata"],
                    hovertemplate=hovertemplate,
                ),
                secondary_y=False,
            )

        if s2_peaks["indices"]:
            times_dt = pd.to_datetime(
                [
                    datetime.datetime.fromtimestamp(0) + datetime.timedelta(seconds=t)
                    for t in (np.array(s2_peaks["indices"]) / self.sample_rate)
                ]
            )
            self.fig.add_trace(
                go.Scatter(
                    x=times_dt,
                    y=audio_envelope[s2_peaks["indices"]],
                    mode="markers",
                    name="S2 Beats",
                    marker=dict(color="orange", symbol="circle", size=6),
                    customdata=s2_peaks["customdata"],
                    hovertemplate=hovertemplate,
                ),
                secondary_y=False,
            )

        if noise_peaks["indices"]:
            times_dt = pd.to_datetime(
                [
                    datetime.datetime.fromtimestamp(0) + datetime.timedelta(seconds=t)
                    for t in (np.array(noise_peaks["indices"]) / self.sample_rate)
                ]
            )
            self.fig.add_trace(
                go.Scatter(
                    x=times_dt,
                    y=audio_envelope[noise_peaks["indices"]],
                    mode="markers",
                    name="Noise/Rejected",
                    marker=dict(color="grey", symbol="x", size=6),
                    customdata=noise_peaks["customdata"],
                    hovertemplate=hovertemplate,
                ),
                secondary_y=False,
            )

        # Average S1 / S2 contractility traces (prominence-based, 3-point moving average), Analysis Data only
        self._add_s1_s2_amplitude_traces(
            s1_peaks["indices"], s2_peaks["indices"], audio_envelope, trough_indices
        )

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

    def _add_s1_s2_amplitude_traces(self, s1_indices, s2_indices, audio_envelope, trough_indices=None):
        """Add line traces for Average S1, S2, and combined contractility (prominence-based, smoothed)."""
        from bpm_analysis import get_peak_prominence_details

        troughs = np.array(trough_indices) if trough_indices is not None and len(trough_indices) > 0 else np.array([], dtype=np.intp)

        def prominence_at(peak_idx):
            details = get_peak_prominence_details(peak_idx, audio_envelope, troughs)
            return details["prominence"]

        if s1_indices:
            s1_idx = np.array(s1_indices)
            s1_times_sec = s1_idx.astype(float) / self.sample_rate
            s1_prom = np.array([prominence_at(int(i)) for i in s1_idx])
            s1_smooth = self._smooth_peak_amplitudes(s1_prom, window_size=3)
            s1_times_dt = pd.to_datetime(
                [datetime.datetime.fromtimestamp(0) + datetime.timedelta(seconds=t) for t in s1_times_sec]
            )
            self.fig.add_trace(
                go.Scatter(
                    x=s1_times_dt,
                    y=s1_smooth,
                    mode="lines",
                    name="Average S1 contractility",
                    line=dict(color="#e36f6f", width=2),
                    visible=True,
                ),
                secondary_y=False,
            )
        if s2_indices:
            s2_idx = np.array(s2_indices)
            s2_times_sec = s2_idx.astype(float) / self.sample_rate
            s2_prom = np.array([prominence_at(int(i)) for i in s2_idx])
            s2_smooth = self._smooth_peak_amplitudes(s2_prom, window_size=3)
            s2_times_dt = pd.to_datetime(
                [datetime.datetime.fromtimestamp(0) + datetime.timedelta(seconds=t) for t in s2_times_sec]
            )
            self.fig.add_trace(
                go.Scatter(
                    x=s2_times_dt,
                    y=s2_smooth,
                    mode="lines",
                    name="Average S2 contractility",
                    line=dict(color="orange", width=2),
                    visible="legendonly",
                ),
                secondary_y=False,
            )
        # Combined S1+S2 average contractility (12-point window)
        if s1_indices or s2_indices:
            times_sec = []
            proms = []
            if s1_indices:
                s1_idx = np.array(s1_indices)
                times_sec.extend((s1_idx.astype(float) / self.sample_rate).tolist())
                proms.extend([prominence_at(int(i)) for i in s1_idx])
            if s2_indices:
                s2_idx = np.array(s2_indices)
                times_sec.extend((s2_idx.astype(float) / self.sample_rate).tolist())
                proms.extend([prominence_at(int(i)) for i in s2_idx])
            times_sec = np.array(times_sec)
            proms = np.array(proms)
            order = np.argsort(times_sec)
            times_sec = times_sec[order]
            proms = proms[order]
            combined_smooth = self._smooth_peak_amplitudes(proms, window_size=12)
            times_dt = pd.to_datetime(
                [datetime.datetime.fromtimestamp(0) + datetime.timedelta(seconds=t) for t in times_sec]
            )
            self.fig.add_trace(
                go.Scatter(
                    x=times_dt,
                    y=combined_smooth,
                    mode="lines",
                    name="Average contractility",
                    line=dict(color="#aaa", width=2),
                    visible="legendonly",
                ),
                secondary_y=False,
            )

    def _add_bpm_hrv_traces(self, smoothed_bpm, analysis_data, windowed_hrv_df):
        """Adds BPM, BPM trend, and HRV traces."""
        if smoothed_bpm is not None and not smoothed_bpm.empty:
            self.fig.add_trace(
                go.Scatter(
                    x=smoothed_bpm.index, y=smoothed_bpm.values, name="Average BPM", line=dict(color="#4a4a4a", width=3)
                ),
                secondary_y=True,
            )

        if "long_term_bpm_series" in analysis_data and not analysis_data["long_term_bpm_series"].empty:
            lt_series = analysis_data["long_term_bpm_series"]
            start_datetime = datetime.datetime.fromtimestamp(0)
            lt_times_dt = pd.to_datetime([start_datetime + datetime.timedelta(seconds=t) for t in lt_series.index])
            self.fig.add_trace(
                go.Scatter(
                    x=lt_times_dt,
                    y=lt_series.values,
                    name="BPM Trend (Belief)",
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
            hrv_times_dt = pd.to_datetime(
                [datetime.datetime.fromtimestamp(0) + datetime.timedelta(seconds=t) for t in windowed_hrv_df["time"]]
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

    def _add_annotations_and_summary(self, smoothed_bpm, hrv_summary, hrr_stats, peak_recovery_stats):
        """Adds min/max BPM annotations and the main summary box."""
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

        if hrv_summary:
            annotation_text = "<b>Analysis Summary</b><br>"
            if hrv_summary.get("avg_bpm") is not None:
                annotation_text += (
                    f"Avg/Min/Max BPM: {hrv_summary['avg_bpm']:.1f} / {hrv_summary['min_bpm']:.1f} / {hrv_summary['max_bpm']:.1f}<br>"
                )
            if hrr_stats and hrr_stats.get("hrr_value_bpm") is not None:
                annotation_text += f"<b>1-Min HRR: {hrr_stats['hrr_value_bpm']:.1f} BPM Drop</b><br>"
            if peak_recovery_stats and peak_recovery_stats.get("slope_bpm_per_sec") is not None:
                annotation_text += f"<b>Peak Recovery Rate: {peak_recovery_stats['slope_bpm_per_sec']:.2f} BPM/sec</b><br>"
            if hrv_summary.get("avg_rmssdc") is not None:
                annotation_text += f"Avg. Corrected RMSSD: {hrv_summary['avg_rmssdc']:.2f}<br>"
            if hrv_summary.get("avg_sdnn") is not None:
                annotation_text += f"Avg. Windowed SDNN: {hrv_summary['avg_sdnn']:.2f} ms"

            self.fig.add_annotation(
                text=annotation_text,
                align="left",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.98,
                bordercolor="black",
                borderwidth=1,
                bgcolor="rgba(255, 253, 231, 0.4)",
            )

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

    def _seconds_to_datetime(self, seconds: float) -> datetime.datetime:
        """Converts elapsed seconds since epoch to timezone-naive datetime."""
        epoch = datetime.datetime.fromtimestamp(0)
        return epoch + datetime.timedelta(seconds=seconds)

    def _add_trapezoid_shapes(self, trapezoids: Optional[List[Dict]]):
        """Draws trapezoid outlines and markers for detected HR artifacts."""
        if not trapezoids:
            return

        for idx, trap in enumerate(trapezoids, start=1):
            event_sequence = [
                ("Start of rise", trap["t_start_rise"], trap["bpm_start_rise"]),
                ("End of rise", trap["t_end_rise"], trap["bpm_end_rise"]),
                ("Start of fall", trap["t_start_fall"], trap["bpm_start_fall"]),
                ("End of fall", trap["t_end_fall"], trap["bpm_end_fall"]),
            ]
            x_times = [self._seconds_to_datetime(t) for _, t, _ in event_sequence]
            y_values = [bpm for _, _, bpm in event_sequence]
            customdata = [
                f"<b>{label}</b><br>{t:.3f}s<br>{bpm:.1f} BPM" for label, t, bpm in event_sequence
            ]

            self.fig.add_trace(
                go.Scatter(
                    x=x_times,
                    y=y_values,
                    mode="lines+markers",
                    name="Trapezoid Artifacts",
                    marker=dict(symbol="circle-open", size=8, color="#ffd166"),
                    line=dict(color="#ffd166", width=2),
                    customdata=customdata,
                    hovertemplate="%{customdata}<extra></extra>",
                    legendgroup="Trapezoid Artifacts",
                    showlegend=(idx == 1),
                ),
                secondary_y=True,
            )

    def _generate_custom_html(
        self,
        plotly_html: str,
        plot_title: str,
        base_name: str,
        *,
        pipeline_steps_html: str = "",
    ) -> str:
        """
        Generates custom HTML with audio player, timeline scrubber, and synchronized playhead.
        Fixes audio path issues and adds debugging capabilities.
        """
        audio_file_name = os.path.basename(self.audio_source_path)
        duration_sec = self.audio_duration_sec or 0

        audio_src = ""
        if os.path.exists(self.audio_source_path):
            dest_audio_path = os.path.join(self.output_directory, audio_file_name)
            src_abspath = os.path.abspath(self.audio_source_path)
            dest_abspath = os.path.abspath(dest_audio_path)

            if src_abspath != dest_abspath:
                try:
                    shutil.copy2(self.audio_source_path, dest_audio_path)
                    logging.info(f"✅ Copied audio file to {dest_audio_path}")
                except Exception as e:
                    logging.error(f"❌ Could not copy audio file: {e}")
            audio_src = audio_file_name.replace('\\', '/')
        else:
            logging.error(f"❌ Audio source file does NOT exist: {self.audio_source_path}")
            dest_audio_path = os.path.join(self.output_directory, audio_file_name)
            if os.path.exists(dest_audio_path):
                audio_src = audio_file_name.replace('\\', '/')
                logging.info(f"✅ Found audio file in output directory: {dest_audio_path}")
            else:
                logging.error(f"❌ Audio file not found anywhere: {audio_file_name}")
                audio_src = ""

        filtered_debug_file_name = f"{base_name}_filtered_debug.wav"
        filtered_debug_path = os.path.join(self.output_directory, filtered_debug_file_name)
        filtered_available = os.path.exists(filtered_debug_path)
        filtered_audio_src = filtered_debug_file_name.replace('\\', '/') if filtered_available else ""
        if filtered_available:
            logging.info(f"🎧 Using filtered debug audio: {filtered_debug_path}")

        logging.info(f"🎵 HTML audio source path: '{audio_src}'")
        import urllib.parse
        audio_src_escaped = urllib.parse.quote(audio_src)
        filtered_audio_src_escaped = urllib.parse.quote(filtered_audio_src) if filtered_audio_src else ""

        # Prepare spectrogram data (optional, per audio source)
        spectrogram_original_src = ""
        spectrogram_filtered_src = ""
        spectrogram_available_original = "false"
        spectrogram_available_filtered = "false"

        # Respect spectrogram-enabled flag when embedding/generating images.
        spectrogram_enabled = getattr(self, "spectrogram_enabled", True)

        if spectrogram_enabled:
            # Original spectrogram (precomputed in plot_and_save if possible)
            if getattr(self, "spectrogram_base64", None):
                spectrogram_original_src = f"data:image/png;base64,{self.spectrogram_base64}"
                spectrogram_available_original = "true"
            else:
                # Fallback: try to generate on demand from the copied audio in the output directory
                try:
                    if audio_src:
                        orig_audio_path_for_spec = os.path.join(self.output_directory, audio_src)
                        spec_b64 = self._generate_spectrogram_image(orig_audio_path_for_spec)
                        if spec_b64:
                            spectrogram_original_src = f"data:image/png;base64,{spec_b64}"
                            spectrogram_available_original = "true"
                except Exception as e:
                    logging.warning(f"Failed to generate on-demand original spectrogram: {e}")

            # Filtered spectrogram (if filtered debug audio exists)
            if filtered_available:
                try:
                    spec_filtered_b64 = self._generate_spectrogram_image(filtered_debug_path)
                    if spec_filtered_b64:
                        spectrogram_filtered_src = f"data:image/png;base64,{spec_filtered_b64}"
                        spectrogram_available_filtered = "true"
                except Exception as e:
                    logging.warning(f"Failed to generate filtered spectrogram: {e}")
        else:
            logging.info("Spectrogram generation disabled; no spectrogram images embedded in HTML.")

        audio_source_options = ['<option value="original">Original Audio</option>']
        if filtered_available:
            audio_source_options.append('<option value="filtered">Filtered Debug</option>')
        audio_source_select_html = (
            '<select id="audio-source-select" class="audio-source-select">'
            + "".join(audio_source_options)
            + '</select>'
        )
        
        # Build configuration payload for external interactive_plot.js script
        config_payload = {
            "totalDuration": float(duration_sec),
            "spectrogramSources": {
                "original": spectrogram_original_src,
                "filtered": spectrogram_filtered_src,
            },
            "spectrogramAvailable": {
                "original": spectrogram_available_original == "true",
                "filtered": spectrogram_available_filtered == "true",
            },
            "audioSources": {
                "original": audio_src_escaped,
                "filtered": filtered_audio_src_escaped,
            },
            "audioLabels": {
                "original": audio_file_name,
                "filtered": filtered_debug_file_name if filtered_available else audio_file_name,
            },
            "springerSegments": getattr(self, "_springer_segments", []) or [],
        }
        config_json = json.dumps(config_payload)

        # Ensure interactive_plot.js is available next to the HTML file
        try:
            js_src_path = os.path.join(os.path.dirname(__file__), "assets", "interactive_plot.js")
            js_dest_path = os.path.join(self.output_directory, "interactive_plot.js")
            if os.path.exists(js_src_path):
                shutil.copy2(js_src_path, js_dest_path)
                logging.info(f"Copied interactive_plot.js to {js_dest_path}")
            else:
                logging.error(f"interactive_plot.js not found at {js_src_path}; HTML will reference a missing script.")
        except Exception as e:
            logging.error(f"Failed to copy interactive_plot.js: {e}")

        springer_legend_html = ""
        if getattr(self, "_springer_segments", None):
            springer_legend_html = (
                ' <span class="springer-segments-legend" title="Segment strip: S1, systole, S2, diastole">'
                '<span class="springer-legend-dot" style="background:#e36f6f"></span> S1 '
                '<span class="springer-legend-dot" style="background:#666"></span> Systole '
                '<span class="springer-legend-dot" style="background:#f0a030"></span> S2 '
                '<span class="springer-legend-dot" style="background:#999"></span> Diastole</span>'
            )

        pipeline_steps_section = ""
        if pipeline_steps_html:
            pipeline_steps_section = (
                '<div class="springer-pipeline-steps-section">'
                '<h3>Pipeline steps (debug)</h3>'
                f'<div class="springer-pipeline-steps-chart">{pipeline_steps_html}</div>'
                '</div>'
            )

        html_template = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>{plot_title}</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            margin: 0;
            padding: 0;
            background-color: #111;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #e0e0e0;
        }}
        
        /* Main container - full viewport */
        #main-container {{
            width: 100%;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        
        /* Timeline Scrubber - compact bar above the graph */
        #timeline-container {{
            background: linear-gradient(180deg, #1e1e2e 0%, #151520 100%);
            border-bottom: 1px solid #333;
            padding: 4px 10px;
            flex-shrink: 0;
        }}
        
        /* Controls row */
        #controls-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 4px;
            flex-wrap: wrap;
        }}
        
        #current-time {{
            font-size: 12px;
            font-weight: bold;
            color: #00d4ff;
            font-family: 'Consolas', 'Monaco', monospace;
            min-width: 140px;
        }}
        
        #audio-controls {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        #audio-controls button {{
            background: #2a2a3a;
            border: 1px solid #444;
            color: #e0e0e0;
            padding: 3px 10px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
            transition: all 0.15s;
        }}
        
        #audio-controls button:hover {{
            background: #3a3a4a;
            border-color: #00d4ff;
        }}
        
        #audio-controls button.active {{
            background: #00d4ff;
            color: #111;
        }}

        #grid-controls {{
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: 11px;
            color: #aaa;
        }}

        #grid-controls .grid-label {{
            font-size: 11px;
            font-weight: 600;
            color: #cfdcff;
        }}

        .grid-toggle-button {{
            background: #2a2a3a;
            border: 1px solid #444;
            color: #e0e0e0;
            padding: 3px 8px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
            transition: all 0.15s;
        }}

        .grid-toggle-button:hover {{
            background: #3a3a4a;
            border-color: #00d4ff;
        }}

        .grid-toggle-button.active {{
            background: #00d4ff;
            color: #111;
        }}
        
        #volume-control {{
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: 12px;
        }}
        
        #volume-slider {{
            width: 60px;
            height: 3px;
            -webkit-appearance: none;
            background: #333;
            border-radius: 2px;
            outline: none;
        }}
        
        #volume-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 10px;
            height: 10px;
            background: #00d4ff;
            border-radius: 50%;
            cursor: pointer;
        }}

        #audio-source-select {{
            background: #1e1e2e;
            border: 1px solid #333;
            color: #e0e0e0;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            font-family: 'Segoe UI', sans-serif;
        }}
        
        #total-time {{
            font-size: 12px;
            color: #888;
            font-family: 'Consolas', 'Monaco', monospace;
        }}
        

        /* Labeling controls */
        #labeling-controls {{
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: 11px;
            color: #aaa;
            flex-wrap: wrap;
        }}

        #label-type-select {{
            background: #1e1e2e;
            border: 1px solid #333;
            color: #e0e0e0;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            font-family: 'Segoe UI', sans-serif;
        }}

        #apply-label-btn,
        #download-labels-btn {{
            background: #2a2a3a;
            border: 1px solid #444;
            color: #e0e0e0;
            padding: 3px 8px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
            transition: all 0.15s;
        }}

        #apply-label-btn:hover,
        #download-labels-btn:hover {{
            background: #3a3a4a;
            border-color: #00d4ff;
        }}
        
        /* Timeline scrubber bar */
        #timeline-scrubber {{
            position: relative;
            height: 18px;
            background: #1a1a2a;
            border-radius: 3px;
            cursor: pointer;
            overflow: hidden;
            border: 1px solid #333;
        }}
        
        #timeline-progress {{
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            background: linear-gradient(90deg, #00d4ff44 0%, #00d4ff22 100%);
            pointer-events: none;
        }}
        
        #timeline-playhead {{
            position: absolute;
            top: 0;
            width: 2px;
            height: 100%;
            background: #00d4ff;
            box-shadow: 0 0 6px #00d4ff;
            pointer-events: none;
        }}
        
        #timeline-ticks {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            pointer-events: none;
        }}
        
        .timeline-tick {{
            position: absolute;
            top: 0;
            width: 1px;
            background: #333;
        }}
        
        .timeline-tick.major {{
            height: 100%;
            background: #444;
        }}
        
        .timeline-tick.minor {{
            height: 40%;
            top: 60%;
        }}
        
        .tick-label {{
            position: absolute;
            top: 1px;
            font-size: 8px;
            color: #666;
            transform: translateX(-50%);
            pointer-events: none;
        }}
        
        /* Chart container - fills remaining space */
        #chart-container {{
            flex: 1;
            position: relative;
            min-height: 0;
            display: flex;
            flex-direction: column;
        }}
        
        /* Chart toolbar - title, audio filename, legend filter */
        #chart-toolbar {{
            flex-shrink: 0;
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 4px 8px;
            background: rgba(40, 40, 50, 0.6);
            border-bottom: 1px solid #333;
            font-size: 12px;
        }}
        #chart-toolbar .chart-toolbar-title {{
            color: #aaa;
            white-space: nowrap;
        }}
        #chart-toolbar .chart-toolbar-label {{
            color: #aaa;
            white-space: nowrap;
            margin-left: auto;
        }}
        #chart-toolbar #audio-file-name {{
            font-size: 12px;
            color: #ccc;
            white-space: nowrap;
            flex-shrink: 0;
        }}
        #legend-category-filter {{
            padding: 2px 6px;
            background: #2a2a35;
            color: #ddd;
            border: 1px solid #444;
            border-radius: 3px;
            font-size: 12px;
            cursor: pointer;
        }}
        #legend-category-filter:hover {{
            border-color: #666;
        }}
        
        .plotly-chart-wrapper {{
            position: relative;
            flex: 1;
            min-height: 0;
            width: 100%;
        }}

        #plotly-chart {{
            width: 100%;
            height: 100%;
        }}

        #plotly-chart > div {{
            width: 100% !important;
            height: 100% !important;
        }}
        
        .springer-segments-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            overflow: visible;
        }}

        .springer-segments-legend {{
            margin-left: 0.75em;
            font-size: 0.85em;
            color: #aaa;
        }}

        .springer-legend-dot {{
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 1px;
            vertical-align: middle;
            margin-right: 2px;
        }}

        .springer-pipeline-steps-section {{
            margin-top: 1em;
            padding: 0.5em 0;
            border-top: 1px solid #333;
        }}
        .springer-pipeline-steps-section h3 {{
            margin: 0 0 0.5em 0;
            font-size: 0.95em;
            color: #aaa;
        }}
        .springer-pipeline-steps-chart {{
            width: 100%;
            min-height: 400px;
        }}

        /* Vertical playhead line on chart */
        #chart-playhead {{
            position: absolute;
            top: 0;
            width: 2px;
            height: 100%;
            background: #ff4757;
            box-shadow: 0 0 8px #ff4757;
            pointer-events: none;
            z-index: 100;
            display: none;
        }}
        
        /* Hidden audio element */
        #audio-player {{
            display: none;
        }}
        
        /* Spectrogram overlay */
        #spectrogram-container {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1;
            overflow: hidden;
        }}
        
        #spectrogram-image {{
            position: absolute;
            height: 100%;
            opacity: 0.4;
            pointer-events: none;
            image-rendering: auto;
            transition: opacity 0.2s ease;
        }}
        
        #spectrogram-image.hidden {{
            opacity: 0;
        }}
        
        /* Audio error overlay */
        #audio-error {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #ff4757;
            color: white;
            padding: 20px;
            border-radius: 5px;
            display: none;
            z-index: 2000;
            font-family: monospace;
        }}
        
        /* Keyboard shortcuts hint */
        #shortcuts-hint {{
            position: fixed;
            bottom: 8px;
            right: 8px;
            background: rgba(30, 30, 40, 0.85);
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 10px;
            color: #777;
            z-index: 1000;
        }}
        
        #shortcuts-hint kbd {{
            background: #333;
            padding: 1px 4px;
            border-radius: 2px;
            margin: 0 1px;
            font-size: 9px;
        }}
    </style>
</head>
<body>
    <div id="main-container">
        <!-- Timeline Scrubber - compact bar above graph -->
        <div id="timeline-container">
            <div id="controls-row">
                <span id="current-time">00:00.000 (0.00s)</span>
                <div id="audio-controls">
                    <button id="play-btn" title="Play/Pause (Space)">▶ Play</button>
                    <button id="stop-btn" title="Stop (S)">⏹ Stop</button>
                    <button id="sync-btn" class="active" title="Sync playhead">🔗</button>
                    <button id="spectrogram-btn" title="Toggle Spectrogram (G)">📊 Spectrogram</button>
                    <div id="spectrogram-controls">
                        <input type="range" id="spectrogram-opacity" min="0.1" max="0.8" step="0.05" value="0.4" title="Spectrogram opacity">
                    </div>
                    <div id="volume-control">
                        <span>🔊</span>
                        <input type="range" id="volume-slider" min="0" max="1" step="0.05" value="1">
                    </div>
                    {audio_source_select_html}
                </div>
                <div id="grid-controls">
                    <span class="grid-label">Grid:</span>
                    <button class="grid-toggle-button" data-grid-axis="yaxis" title="Toggle signal amplitude gridlines">Signal</button>
                    <button class="grid-toggle-button active" data-grid-axis="yaxis2" title="Toggle BPM/HRV gridlines">BPM</button>
                </div>
                <div id="labeling-controls">
                    <span class="grid-label">Label:</span>
                    <select id="label-type-select" title="Desired label for nearest peak">
                        <option value="S1">S1</option>
                        <option value="S2">S2</option>
                        <option value="Noise">Noise</option>
                    </select>
                    <button id="apply-label-btn" title="Relabel nearest peak to current playhead time">Apply</button>
                    <button id="flip-labels-right-btn" title="Flip all S1/S2 labels to the right of the playhead">Flip Right</button>
                    <button id="download-labels-btn" title="Download current labels as CSV">Download CSV</button>
                    <button id="import-labels-btn" title="Import manually labeled peaks CSV">Import CSV</button>
                    <input type="file" id="import-labels-input" accept=".csv" style="display:none" />
                </div>
                <span id="total-time">{int(duration_sec // 60):02d}:{int(duration_sec % 60):02d}</span>
            </div>
            <div id="timeline-scrubber">
                <div id="timeline-ticks"></div>
                <div id="timeline-progress"></div>
                <div id="timeline-playhead"></div>
            </div>
        </div>
        
        <!-- Chart container - takes up remaining space -->
        <div id="chart-container">
            <div id="chart-toolbar">
                <span class="chart-toolbar-title">Heartbeat Analysis – </span>
                <span id="audio-file-name" title="{audio_file_name}">{audio_file_name}</span>
                <label for="legend-category-filter" class="chart-toolbar-label">Show:</label>
                <select id="legend-category-filter" title="Filter legend and visible traces by category">
                    <option value="all">All</option>
                    <option value="debug">Debug</option>
                    <option value="analysis">Analysis Data</option>
                </select>
                {springer_legend_html}
            </div>
            <div id="spectrogram-container">
                <img id="spectrogram-image" class="hidden" src="{spectrogram_original_src}" alt="Spectrogram" />
            </div>
            <div id="chart-playhead"></div>
            <div class="plotly-chart-wrapper">
                <div id="plotly-chart">
                    {plotly_html}
                </div>
                <svg id="springer-segments-layer" class="springer-segments-overlay" aria-hidden="true">
                    <g id="springer-segments-group"></g>
                </svg>
            </div>
            {pipeline_steps_section}
        </div>
    </div>
    
    <!-- Hidden audio player -->
    <audio id="audio-player" preload="auto">
        Your browser does not support audio playback.
    </audio>
    
    <!-- Audio error overlay -->
    <div id="audio-error"></div>
    
    <!-- Keyboard shortcuts hint -->
    <div id="shortcuts-hint">
        <kbd>Space</kbd> Play &nbsp;
        <kbd>S</kbd> Stop &nbsp;
        <kbd>←→</kbd> Seek &nbsp;
        <kbd>G</kbd> Spectrogram
    </div>
    
    <script>
        window.BPM_ANALYZER_CONFIG = {config_json};
    </script>
    <script src="interactive_plot.js"></script>
</body>
</html>'''
        
        return html_template
