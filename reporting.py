import os
import datetime
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from bpm_analysis import _get_peak_type_from_debug, format_debug_entry


class ReportGenerator:
    """Handles the creation of text-based analysis reports."""

    def __init__(self, file_name: str, output_directory: str):
        self.file_name = file_name
        self.output_directory = output_directory
        self.file_name_no_ext = os.path.splitext(file_name)[0]
        self.base_name = os.path.basename(self.file_name_no_ext)

    def save_analysis_summary(self, final_metrics: Dict):
        """Saves a comprehensive Markdown summary of the analysis results."""
        output_path = os.path.join(self.output_directory, f"{self.base_name}_Analysis_Summary.md")

        with open(output_path, "w", encoding="utf-8") as f:
            self._write_summary_header(f)
            self._write_overall_summary(f, final_metrics.get("hrv_summary"), final_metrics.get("hrr_stats"))
            self._write_steepest_slopes(
                f, final_metrics.get("peak_exertion_stats"), final_metrics.get("peak_recovery_stats")
            )
            self._write_significant_changes(
                f, final_metrics.get("major_inclines"), final_metrics.get("major_declines")
            )
            self._write_heartbeat_data_table(f, final_metrics.get("smoothed_bpm"), final_metrics.get("bpm_times"))

        logging.info(f"Markdown analysis summary saved to {output_path}")

    def create_chronological_log(
        self,
        audio_envelope: np.ndarray,
        sample_rate: int,
        all_raw_peaks: np.ndarray,
        analysis_data: Dict,
        final_metrics: Dict,
    ):
        """Creates a detailed, readable debug log file."""
        output_log_path = os.path.join(self.output_directory, f"{self.base_name}_Debug_Log.md")
        logging.info(f"Generating readable debug log at '{output_log_path}'...")
        merged_df = self._prepare_log_data(
            audio_envelope,
            sample_rate,
            all_raw_peaks,
            analysis_data,
            final_metrics.get("smoothed_bpm"),
            final_metrics.get("bpm_times"),
        )
        springer_debug = analysis_data.get("springer_debug")
        with open(output_log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"# Chronological Debug Log for {os.path.basename(self.file_name)}\n")
            log_file.write(f"Analysis performed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            if springer_debug is not None:
                self._write_springer_debug_section(log_file, springer_debug)
                log_file.write("\n---\n\n## Chronological events\n\n")
            if merged_df is None or merged_df.empty:
                log_file.write("No significant events detected to log.\n")
            else:
                self._write_log_events(log_file, merged_df)
        logging.info("Debug log generation complete.")

    def _prepare_log_data(self, audio_envelope, sample_rate, all_raw_peaks, analysis_data, smoothed_bpm, bpm_times):
        """Prepares and merges all data sources into a single DataFrame for logging."""
        events = []
        debug_info = analysis_data.get("beat_debug_info", {})

        for p in all_raw_peaks:
            reason = debug_info.get(p)
            if reason:
                events.append({"time": p / sample_rate, "type": "Peak", "amp": audio_envelope[p], "reason": reason})
        if "trough_indices" in analysis_data:
            for p in analysis_data["trough_indices"]:
                events.append({"time": p / sample_rate, "type": "Trough", "amp": audio_envelope[p], "reason": ""})

        if not events:
            return None
        events_df = pd.DataFrame(events).sort_values(by="time").set_index("time")

        master_df = pd.DataFrame(index=np.arange(len(audio_envelope)) / sample_rate)
        noise_floor_series = analysis_data.get("dynamic_noise_floor_series")
        if noise_floor_series is not None and hasattr(noise_floor_series, "values"):
            master_df["noise_floor"] = noise_floor_series.values
        if smoothed_bpm is not None and not smoothed_bpm.empty:
            smoothed_bpm_sec_index = pd.Series(data=smoothed_bpm.values, index=bpm_times).groupby(level=0).mean()
            master_df["smoothed_bpm"] = smoothed_bpm_sec_index
        if "long_term_bpm_series" in analysis_data and not analysis_data["long_term_bpm_series"].empty:
            master_df["lt_bpm"] = analysis_data["long_term_bpm_series"].groupby(level=0).mean()

        master_df.ffill(inplace=True)

        return pd.merge_asof(
            left=events_df,
            right=master_df,
            left_index=True,
            right_index=True,
            direction="nearest",
            tolerance=pd.Timedelta(seconds=0.5).total_seconds(),
        )

    def _write_log_events(self, log_file, merged_df):
        """Write chronological peak/trough events (header is written by create_chronological_log)."""
        for row in merged_df.itertuples(name="LogEvent"):
            log_file.write(f"## Time: `{row.Index:.4f}s`\n")

            if row.type == "Trough":
                log_file.write("**Trough Detected**\n")
            else:
                raw_reason = getattr(row, "reason", "")
                if not raw_reason or raw_reason == "Unknown":
                    log_file.write("**Unclassified Peak**\n")
                else:
                    peak_type = _get_peak_type_from_debug(raw_reason) or "Unclassified Peak"
                    log_file.write(f"**{peak_type}.**\n")

                    formatted_lines = format_debug_entry(raw_reason)
                    for ln in formatted_lines:
                        log_file.write(f"{ln}\n")

            metric_values = {
                "Raw Amp": getattr(row, "amp", None),
                "Noise Floor": getattr(row, "noise_floor", None),
                "Average BPM (Smoothed)": getattr(row, "smoothed_bpm", None),
                "Long-Term BPM (Belief)": getattr(row, "lt_bpm", None),
            }
            metric_formats = {
                "Raw Amp": "{value:.3f}",
                "Noise Floor": "{value:.3f}",
                "Average BPM (Smoothed)": "{value:.1f}",
                "Long-Term BPM (Belief)": "{value:.1f}",
            }
            for name, value in metric_values.items():
                if pd.notna(value):
                    fmt = metric_formats.get(name, "{value}")
                    formatted = fmt.format(value=value)
                    log_file.write(f"- **{name}**: `{formatted}`\n")

            log_file.write("\n\n")

    def _write_springer_debug_section(self, log_file, springer_debug: Dict) -> None:
        """Writes Springer pipeline debug info (HR, features, state counts) to the debug log."""
        log_file.write("\n---\n\n# Springer Pipeline Debug\n\n")
        log_file.write("| Parameter | Value |\n|:---|:---|\n")
        if springer_debug.get("heart_rate") is not None:
            log_file.write(f"| **Heart rate (BPM)** | {springer_debug['heart_rate']:.1f} |\n")
        if springer_debug.get("systolic_time_interval") is not None:
            log_file.write(f"| **Systolic time (s)** | {springer_debug['systolic_time_interval']:.3f} |\n")
        if springer_debug.get("features_shape") is not None:
            sh = springer_debug["features_shape"]
            log_file.write(f"| **PCG_Features shape** | {sh} (frames × features) |\n")
        log_file.write("\n## Viterbi state counts (at 50 Hz, before expand)\n\n")
        qt_counts = springer_debug.get("qt_state_counts") or {}
        state_names = {1: "S1", 2: "Systole", 3: "S2", 4: "Diastole"}
        for state in (1, 2, 3, 4):
            count = qt_counts.get(state, 0)
            name = state_names.get(state, f"State {state}")
            log_file.write(f"- **{name}**: {count} frames\n")
        log_file.write("\n## Assigned states (after expand to audio length)\n\n")
        assigned_counts = springer_debug.get("assigned_states_state_counts") or {}
        for state in (1, 2, 3, 4):
            count = assigned_counts.get(state, 0)
            name = state_names.get(state, f"State {state}")
            log_file.write(f"- **{name}**: {count} samples\n")
        num_seg = springer_debug.get("num_segments")
        if num_seg is not None:
            log_file.write(f"\n**Total segments**: {num_seg}\n")
        feature_mean = springer_debug.get("feature_mean")
        feature_std = springer_debug.get("feature_std")
        if feature_mean is not None and feature_std is not None:
            log_file.write("\n## Feature summary (per channel, over time)\n\n")
            log_file.write("| Feature index | Mean | Std |\n|:---:|:---:|:---:|\n")
            for i, (m, s) in enumerate(zip(feature_mean, feature_std)):
                log_file.write(f"| {i} | {m:.4f} | {s:.4f} |\n")
        # Per-state feature means (do emissions differ by state?)
        mean_by_state = springer_debug.get("feature_mean_by_state")
        if mean_by_state:
            log_file.write("\n## Per-state feature means (Viterbi frames)\n\n")
            log_file.write("If S1/S2 rows are similar to Systole/Diastole, the model may not be discriminating.\n\n")
            log_file.write("| State | Feat 0 | Feat 1 | Feat 2 | Feat 3 |\n|:---:|:---:|:---:|:---:|:---:|\n")
            for st in (1, 2, 3, 4):
                name = state_names.get(st, f"S{st}")
                row = mean_by_state.get(st, [])
                if row:
                    log_file.write(f"| {name} | {row[0]:.4f} | {row[1]:.4f} | {row[2]:.4f} | {row[3]:.4f} |\n")
                else:
                    log_file.write(f"| {name} | — | — | — | — |\n")
        # Segment duration stats
        seg_dur = springer_debug.get("segment_duration_stats")
        if seg_dur:
            log_file.write("\n## Segment duration by state\n\n")
            log_file.write("Expected roughly: S1 ~50–100 ms, Systole ~250 ms, S2 ~50–100 ms, Diastole variable.\n\n")
            log_file.write("| State | Mean (s) | Std (s) | Count |\n|:---:|:---:|:---:|:---:|\n")
            for st in (1, 2, 3, 4):
                name = state_names.get(st, f"S{st}")
                d = seg_dur.get(st, {})
                mean_s = d.get("mean_sec")
                std_s = d.get("std_sec")
                cnt = d.get("count", 0)
                if mean_s is not None:
                    log_file.write(f"| {name} | {mean_s:.3f} | {std_s:.3f} | {cnt} |\n")
                else:
                    log_file.write(f"| {name} | — | — | {cnt} |\n")
        # First segments
        first_seg = springer_debug.get("first_segments")
        if first_seg:
            log_file.write("\n## First segments (start, end, state, duration)\n\n")
            log_file.write("| # | Start (s) | End (s) | State | Duration (s) |\n|:---:|:---:|:---:|:---:|:---:|\n")
            for i, seg in enumerate(first_seg, 1):
                log_file.write(f"| {i} | {seg['s']:.3f} | {seg['e']:.3f} | {seg['state']} | {seg['duration_sec']:.3f} |\n")
        # Envelope vs Springer alignment
        mean_s1 = springer_debug.get("mean_abs_offset_s1_ms")
        mean_s2 = springer_debug.get("mean_abs_offset_s2_ms")
        if mean_s1 is not None or mean_s2 is not None:
            log_file.write("\n## Envelope vs Springer alignment\n\n")
            log_file.write("Offset = Springer S1/S2 midpoint time minus nearest envelope peak time. ")
            log_file.write("Small values mean states line up with envelope peaks.\n\n")
            if mean_s1 is not None:
                log_file.write(f"- **Mean |offset| S1 → nearest envelope peak**: {mean_s1:.1f} ms\n")
            if mean_s2 is not None:
                log_file.write(f"- **Mean |offset| S2 → nearest envelope peak**: {mean_s2:.1f} ms\n")
        env_peaks = springer_debug.get("envelope_peak_times_sec")
        s1_mid = springer_debug.get("s1_midpoint_times_sec")
        s2_mid = springer_debug.get("s2_midpoint_times_sec")
        if env_peaks and (s1_mid or s2_mid):
            log_file.write("\nFirst envelope peak times (s): ")
            log_file.write(", ".join(f"{t:.2f}" for t in env_peaks[:10]) + "\n")
            if s1_mid:
                log_file.write("First S1 midpoints (s): ")
                log_file.write(", ".join(f"{t:.2f}" for t in s1_mid[:10]) + "\n")
            if s2_mid:
                log_file.write("First S2 midpoints (s): ")
                log_file.write(", ".join(f"{t:.2f}" for t in s2_mid[:10]) + "\n")
        log_file.write("\n")

    def _write_summary_header(self, f):
        f.write(f"# Analysis Report for: {os.path.basename(self.file_name)}\n")
        f.write(f"*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

    def _write_overall_summary(self, f, hrv_summary, hrr_stats):
        """Writes the main summary table to the markdown report file."""
        f.write("## Overall Summary\n\n| Metric | Value |\n|:---|:---|\n")
        if hrv_summary:
            if hrv_summary.get("avg_bpm") is not None:
                f.write(f"| **Average BPM** | {hrv_summary['avg_bpm']:.1f} BPM |\n")
                f.write(f"| **BPM Range** | {hrv_summary['min_bpm']:.1f} to {hrv_summary['max_bpm']:.1f} BPM |\n")
            if hrv_summary.get("avg_rmssdc") is not None:
                f.write(f"| **Avg. Corrected RMSSD** | {hrv_summary['avg_rmssdc']:.2f} |\n")
            if hrv_summary.get("avg_sdnn") is not None:
                f.write(f"| **Avg. Windowed SDNN** | {hrv_summary['avg_sdnn']:.2f} ms |\n")
        if hrr_stats and hrr_stats.get("hrr_value_bpm") is not None:
            f.write(f"| **1-Minute HRR** | {hrr_stats['hrr_value_bpm']:.1f} BPM Drop |\n")
        f.write("\n")

    def _write_steepest_slopes(self, f, peak_exertion_stats, peak_recovery_stats):
        """Writes the peak exertion and recovery slope data to the markdown report."""
        f.write("## Steepest Slopes Analysis\n\n### Peak Exertion (Fastest HR Increase)\n\n")
        if peak_exertion_stats:
            pes = peak_exertion_stats
            f.write("| Attribute | Value |\n|:---|:---|\n")
            f.write(f"| **Rate** | `+{pes['slope_bpm_per_sec']:.2f}` BPM/second |\n")
            f.write(f"| **Period** | {pes['start_time'].strftime('%M:%S')} to {pes['end_time'].strftime('%M:%S')} |\n")
            f.write(f"| **Duration** | {pes['duration_sec']:.1f} seconds |\n")
            f.write(f"| **BPM Change** | {pes['start_bpm']:.1f} to {pes['end_bpm']:.1f} BPM |\n\n")
        else:
            f.write("*No significant peak exertion period found.*\n\n")

        f.write("### Peak Recovery (Fastest HR Decrease)\n\n")
        if peak_recovery_stats:
            prs = peak_recovery_stats
            f.write("| Attribute | Value |\n|:---|:---|\n")
            f.write(f"| **Rate** | `{prs['slope_bpm_per_sec']:.2f}` BPM/second |\n")
            f.write(f"| **Period** | {prs['start_time'].strftime('%M:%S')} to {prs['end_time'].strftime('%M:%S')} |\n")
            f.write(f"| **Duration** | {prs['duration_sec']:.1f} seconds |\n")
            f.write(f"| **BPM Change** | {prs['start_bpm']:.1f} to {prs['end_bpm']:.1f} BPM |\n\n")
        else:
            f.write("*No significant peak recovery period found post-peak.*\n\n")

    def _write_significant_changes(self, f, major_inclines, major_declines):
        """Writes the sections on sustained heart rate increases and decreases to the report file."""
        f.write("## All Significant HR Changes\n\n### Exertion Periods (Sustained HR Increase)\n\n")
        if major_inclines:
            epoch = datetime.datetime.fromtimestamp(0)
            for incline in major_inclines:
                start_sec = (incline["start_time"] - epoch).total_seconds()
                end_sec = (incline["end_time"] - epoch).total_seconds()
                f.write(
                    f"- **From {start_sec:.1f}s to {end_sec:.1f}s:** Duration={incline['duration_sec']:.1f}s, Change=`+{incline['bpm_increase']:.1f}` BPM\n"
                )
        else:
            f.write("*No significant exertion periods detected.*\n")

        f.write("\n### Recovery Periods (Sustained HR Decrease)\n\n")
        if major_declines:
            epoch = datetime.datetime.fromtimestamp(0)
            for decline in major_declines:
                start_sec = (decline["start_time"] - epoch).total_seconds()
                end_sec = (decline["end_time"] - epoch).total_seconds()
                f.write(
                    f"- **From {start_sec:.1f}s to {end_sec:.1f}s:** Duration={decline['duration_sec']:.1f}s, Change=`-{decline['bpm_decrease']:.1f}` BPM\n"
                )
        else:
            f.write("*No significant recovery periods detected.*\n")
        f.write("\n")

    def _write_heartbeat_data_table(self, f, smoothed_bpm, bpm_times):
        """Writes the final time-series BPM data to a markdown table in the report file."""
        f.write("## Heartbeat Data (BPM over Time)\n\n| Time (s) | Average BPM |\n|:---:|:---:|\n")
        if smoothed_bpm is not None and not smoothed_bpm.empty and bpm_times is not None:
            for t, bpm in zip(bpm_times, smoothed_bpm.values):
                if not np.isnan(bpm):
                    f.write(f"| {t:.2f} | {bpm:.1f} |\n")
        else:
            f.write("| *No data* | *No data* |\n")

    def save_bpm_time_txt(self, smoothed_bpm, bpm_times):
        """Exports the time-aligned BPM series using tilde delimiters."""
        if smoothed_bpm is None or smoothed_bpm.empty or bpm_times is None:
            logging.info("Skipping BPM text export; no BPM data available.")
            return

        output_path = os.path.join(self.output_directory, f"{self.base_name}_HbPerBpm.txt")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for t, bpm in zip(bpm_times, smoothed_bpm.values):
                    if np.isnan(bpm):
                        continue
                    f.write(f"{t:.1f}~{bpm:.1f}~\n")
            logging.info(f"Tilde-delimited BPM export saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save BPM text export: {e}")

