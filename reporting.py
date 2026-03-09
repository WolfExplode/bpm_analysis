import os
import logging
import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from peak_utils import _get_peak_type_from_debug, format_debug_entry
from time_utils import timestamp_str


class ReportGenerator:
    """Handles the creation of text-based analysis reports."""

    _LOG_METRICS = (
        ("Raw Amp", "amp", ".3f"),
        ("Noise Floor", "noise_floor", ".3f"),
        ("Average BPM (Smoothed)", "smoothed_bpm", ".1f"),
        ("Long-Term BPM (Belief)", "lt_bpm", ".1f"),
    )

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
        with open(output_log_path, "w", encoding="utf-8") as log_file:
            if merged_df is None or merged_df.empty:
                log_file.write("# No significant events detected to log.\n")
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
        if "dynamic_noise_floor_series" in analysis_data:
            master_df["noise_floor"] = analysis_data["dynamic_noise_floor_series"].values
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
            tolerance=0.5,
        )

    def _write_log_events(self, log_file, merged_df):
        log_file.write(f"# Chronological Debug Log for {os.path.basename(self.file_name)}\n")
        log_file.write(f"Analysis performed on: {timestamp_str()}\n\n")

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

            for name, attr, fmt in self._LOG_METRICS:
                value = getattr(row, attr, None)
                if pd.notna(value):
                    log_file.write(f"- **{name}**: `{value:{fmt}}`\n")

            log_file.write("\n\n")

    def _write_summary_header(self, f):
        f.write(f"# Analysis Report for: {os.path.basename(self.file_name)}\n")
        f.write(f"*Generated on: {timestamp_str()}*\n\n")

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
            if hrv_summary.get("avg_lf_hf_ratio") is not None:
                f.write(f"| **Avg. Windowed LF/HF Ratio** | {hrv_summary['avg_lf_hf_ratio']:.2f} |\n")
            global_freq = hrv_summary.get("global_freq")
            if global_freq:
                f.write(f"| **VLF Power (global, ms²)** | {global_freq.get('vlf_power', 0):.2f} |\n")
                f.write(f"| **LF Power (global, ms²)** | {global_freq.get('lf_power', 0):.2f} |\n")
                f.write(f"| **HF Power (global, ms²)** | {global_freq.get('hf_power', 0):.2f} |\n")
                f.write(f"| **LF/HF Ratio (global)** | {global_freq.get('lf_hf_ratio', 0):.2f} |\n")
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
        epoch = datetime.datetime.fromtimestamp(0)

        def _write_period_list(items, empty_msg, duration_key, change_key, change_prefix):
            if not items:
                f.write(empty_msg + "\n")
                return
            for item in items:
                start_sec = (item["start_time"] - epoch).total_seconds()
                end_sec = (item["end_time"] - epoch).total_seconds()
                f.write(
                    f"- **From {start_sec:.1f}s to {end_sec:.1f}s:** Duration={item[duration_key]:.1f}s, Change=`{change_prefix}{item[change_key]:.1f}` BPM\n"
                )

        f.write("## All Significant HR Changes\n\n### Exertion Periods (Sustained HR Increase)\n\n")
        _write_period_list(
            major_inclines or [],
            "*No significant exertion periods detected.*",
            "duration_sec",
            "bpm_increase",
            "+",
        )
        f.write("\n### Recovery Periods (Sustained HR Decrease)\n\n")
        _write_period_list(
            major_declines or [],
            "*No significant recovery periods detected.*",
            "duration_sec",
            "bpm_decrease",
            "-",
        )
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


