#!/usr/bin/env python3
"""
Generate the bpm_analysis HTML plot from Octave Springer segmentation export.

Usage (from repo root):
  python plot_octave_springer_results.py path/to/base_springer_export.mat

The .mat file must contain: assigned_states, Fs, export_audio_wav (filename of WAV in same dir).
These are produced by run_Springer_and_export_for_plot.m in the MATLAB codebase.

Output: processed_files/base_bpm_plot.html (and optional CSV), viewable via the GUI "View Reports".
"""

import argparse
import os
import sys
import logging
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Run from repo root so we can import bpm_analysis and plotting
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bpm_analysis import (
    _springer_states_to_peaks,
    _springer_states_to_segments,
    _get_peak_type_from_debug,
    format_debug_entry,
    PeakType,
)
from audio_io import load_audio_mono
from plotting import Plotter
from config import DEFAULT_PARAMS


def _parse_mat_string(arr):
    """Extract string from MATLAB char array or cell."""
    if arr is None:
        return None
    if isinstance(arr, np.ndarray) and arr.dtype.kind == "U":
        return str(arr.flat[0]) if arr.size else None
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return str(arr.flat[0]) if arr.size else None
    return str(arr).strip()


def _compute_envelope(audio: np.ndarray, sample_rate: float, low_hz: float = 15, high_hz: float = 250) -> np.ndarray:
    """Bandpass + Hilbert envelope for display (matches config bandpass range)."""
    nyq = sample_rate / 2.0
    low = max(1.0, low_hz) / nyq
    high = min(nyq - 1, high_hz) / nyq
    if low >= high:
        return np.abs(hilbert(audio))
    b, a = butter(2, [low, high], btype="band")
    filtered = filtfilt(b, a, audio.astype(np.float64))
    return np.abs(hilbert(filtered))


def _bpm_from_s1_intervals(s1_indices: np.ndarray, fs: float) -> tuple:
    """Compute simple BPM and time series from S1–S1 intervals for the plot."""
    if s1_indices is None or len(s1_indices) < 2:
        return None, None
    rr_samples = np.diff(s1_indices.astype(float))
    rr_sec = rr_samples / float(fs)
    bpm = 60.0 / rr_sec
    t_sec = (s1_indices[1:] + s1_indices[:-1]) / 2.0 / float(fs)
    times = np.concatenate([[0], t_sec, [s1_indices[-1] / float(fs)]])
    bpms = np.concatenate([[bpm[0]], bpm, [bpm[-1]]])
    idx = pd.to_datetime(times, unit="s")
    smoothed = pd.Series(bpms, index=idx)
    return smoothed, times


def main():
    parser = argparse.ArgumentParser(description="Plot Octave Springer export in bpm_analysis interface.")
    parser.add_argument("mat_path", help="Path to base_springer_export.mat")
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory for HTML/CSV (default: processed_files in current dir)",
    )
    args = parser.parse_args()

    mat_path = os.path.abspath(args.mat_path)
    if not os.path.isfile(mat_path):
        logging.error("Not a file: %s", mat_path)
        return 1

    export_dir = os.path.dirname(mat_path)
    data = loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    assigned_states = data.get("assigned_states")
    if assigned_states is None:
        logging.error("Missing 'assigned_states' in %s", mat_path)
        return 1
    assigned_states = np.asarray(assigned_states).flatten()

    Fs = data.get("Fs") or data.get("fs")
    if Fs is None:
        logging.error("Missing 'Fs' in %s", mat_path)
        return 1
    fs = int(np.round(np.squeeze(Fs)))

    wav_name = _parse_mat_string(data.get("export_audio_wav"))
    if not wav_name:
        logging.error("Missing 'export_audio_wav' in %s", mat_path)
        return 1
    wav_path = os.path.join(export_dir, wav_name)
    if not os.path.isfile(wav_path):
        logging.error("WAV not found: %s", wav_path)
        return 1

    audio, sr = load_audio_mono(wav_path, fs)
    if len(audio) != len(assigned_states):
        # Trim to match (Octave may have padded)
        n = min(len(audio), len(assigned_states))
        audio = audio[:n]
        assigned_states = assigned_states[:n]
        logging.warning("Trimmed audio/assigned_states to length %d", n)

    envelope = _compute_envelope(audio, float(fs))
    s1_indices, s2_indices = _springer_states_to_peaks(assigned_states)
    springer_segments = _springer_states_to_segments(assigned_states, float(fs))
    all_raw_peaks = np.unique(np.concatenate([s1_indices, s2_indices])) if (len(s1_indices) or len(s2_indices)) else np.array([], dtype=np.int64)

    analysis_data = {
        "segmenter": "springer",
        "springer_segments": springer_segments,
        "springer_s1_indices": s1_indices,
        "springer_s2_indices": s2_indices,
        "springer_pipeline_viz": None,
        "beat_debug_info": {},
        "trough_indices": np.array([], dtype=np.int64),
    }
    for i in s1_indices:
        analysis_data["beat_debug_info"][int(i)] = {"peak_type": "S1 (Springer)", "sections": []}
    for i in s2_indices:
        analysis_data["beat_debug_info"][int(i)] = {"peak_type": "S2 (Springer)", "sections": []}

    smoothed_bpm, bpm_times = _bpm_from_s1_intervals(s1_indices, fs)
    if smoothed_bpm is None:
        smoothed_bpm = pd.Series(dtype=float)
        bpm_times = np.array([])
    final_metrics = {
        "smoothed_bpm": smoothed_bpm,
        "bpm_times": bpm_times,
        "hrv_summary": {},
        "windowed_hrv_df": None,
        "hrr_stats": None,
        "peak_recovery_stats": None,
    }

    output_dir = args.output_dir or os.path.join(os.getcwd(), "processed_files")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(mat_path))[0].replace("_springer_export", "")
    file_name = base_name + ".wav"

    params = dict(DEFAULT_PARAMS)
    plotter = Plotter(
        file_name,
        params,
        fs,
        output_dir,
        source_audio_path=wav_path,
        peak_type_helper=_get_peak_type_from_debug,
        format_debug_entry_func=format_debug_entry,
        peak_type_cls=PeakType,
    )
    output_options = {"html": True, "csv": True, "png": False}
    plotter.plot_and_save(envelope, all_raw_peaks, analysis_data, final_metrics, output_options)

    out_html = os.path.join(output_dir, f"{base_name}_bpm_plot.html")
    logging.info("Wrote %s", out_html)
    return 0


if __name__ == "__main__":
    sys.exit(main())
