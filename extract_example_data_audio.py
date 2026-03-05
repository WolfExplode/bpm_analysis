#!/usr/bin/env python3
"""
Extract PCG recordings from example_data.mat to WAV files (1000 Hz).

Usage (from repo root):
  python extract_example_data_audio.py path/to/example_data.mat [output_dir]
  python extract_example_data_audio.py path/to/example_data.mat -n 10   # first 10 only

Output: output_dir/recording_001.wav, recording_002.wav, ... (or recording_000.wav, ...)
"""

import argparse
import os
import sys

import numpy as np
from scipy.io import wavfile

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from springer_hsmm.example_data import load_example_data


def main():
    parser = argparse.ArgumentParser(description="Extract audio from example_data.mat to WAV files.")
    parser.add_argument("mat_path", help="Path to example_data.mat")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Output directory for WAV files (default: same dir as .mat)",
    )
    parser.add_argument(
        "-n", "--max-recordings",
        type=int,
        default=None,
        help="Export only the first N recordings (default: all)",
    )
    args = parser.parse_args()

    mat_path = os.path.abspath(args.mat_path)
    if not os.path.isfile(mat_path):
        print(f"Error: not a file: {mat_path}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or os.path.join(os.path.dirname(mat_path), "example_data_wav")
    os.makedirs(output_dir, exist_ok=True)

    audio_list, _, fs = load_example_data(mat_path)
    fs = int(fs)
    n_total = len(audio_list)
    n_export = min(n_total, args.max_recordings) if args.max_recordings is not None else n_total

    for i in range(n_export):
        rec = np.asarray(audio_list[i], dtype=np.float64).flatten()
        rec = rec / (np.abs(rec).max() + 1e-10)
        wav_path = os.path.join(output_dir, f"recording_{i + 1:03d}.wav")
        wavfile.write(wav_path, fs, rec.astype(np.float32))

    print(f"Exported {n_export} recordings to {output_dir} (fs={fs} Hz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
