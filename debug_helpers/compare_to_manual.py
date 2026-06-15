"""
Regression check: compare the algorithm's Pass-3 cardiac state output against a
hand-marked ground-truth CSV (<wav>_manual_state_sequence.csv).

Builds per-sample state-label arrays for both (S1 / systole / S2 / diastole),
then reports:
  * overall per-sample agreement
  * per-state recall (how much of each manual state the algorithm reproduced)
  * beat counts (manual vs algorithm S1 segments) — catches missed or phantom beats

Use it to prove a change didn't regress segmentation on a file with trusted labels.

Usage (from repo root):
    python debug_helpers/compare_to_manual.py "inputs/.../file.wav"
    python debug_helpers/compare_to_manual.py "inputs/.../file.wav" --csv "<path>.csv"
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

from debug_helpers._common import (  # noqa: E402
    env_sample_rate, params, reconfigure_stdio, run_pipeline,
)

_STATES = ("S1", "systole", "S2", "diastole")
_CODE = {s: i + 1 for i, s in enumerate(_STATES)}  # 0 = unlabeled


def _run(wav, run_params):
    data = run_pipeline(wav, run_params)
    if data is None:
        return None, 0.0, 0
    labels = data.get("pass3_state_labels")
    n = 0 if labels is None else len(labels)
    sr = env_sample_rate(wav, n) or 0.0
    return data, sr, n


def _algo_labels(data, n):
    arr = np.zeros(n, dtype=np.int8)
    for seg in (data.get("pass3_state_boundaries") or []):
        a0, a1, name = int(seg[0]), int(seg[1]), seg[2]
        c = _CODE.get(name, 0)
        if c:
            arr[max(0, a0):min(n, a1)] = c
    return arr


def _manual_labels(csv_path, sr, n):
    arr = np.zeros(n, dtype=np.int8)
    rows = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                a0 = int(round(float(row["start_sec"]) * sr))
                a1 = int(round(float(row["end_sec"]) * sr))
            except (TypeError, ValueError, KeyError):
                continue
            c = _CODE.get((row.get("state") or "").strip(), 0)
            if c:
                arr[max(0, a0):min(n, a1)] = c
                rows += 1
    return arr, rows


def _count_s1(arr):
    """Number of S1 runs = beat count."""
    s1 = (arr == _CODE["S1"]).astype(np.int8)
    return int(np.sum((np.diff(np.concatenate([[0], s1])) == 1)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("--csv", default=None, help="Manual CSV (default: <wav>_manual_state_sequence.csv).")
    ns = ap.parse_args(argv)

    reconfigure_stdio()

    csv_path = ns.csv or (ns.wav + "_manual_state_sequence.csv")
    if not os.path.exists(csv_path):
        print(f"Manual CSV not found: {csv_path}", file=sys.stderr)
        return 2

    data, sr, n = _run(ns.wav, params())
    if data is None or not sr or not n:
        print("Could not determine sample rate / length.", file=sys.stderr)
        return 2

    algo = _algo_labels(data, n)
    man, mrows = _manual_labels(csv_path, sr, n)

    both = (algo != 0) & (man != 0)
    agree = int(np.sum((algo == man) & both))
    total = int(np.sum(both))
    overall = 100.0 * agree / total if total else 0.0

    print(f"# {os.path.basename(ns.wav)}  (sr~{sr:.1f}Hz, n={n}, manual rows={mrows})")
    print(f"# overlap samples (both labeled) = {total}\n")
    print(f"OVERALL per-sample agreement: {overall:.2f}%  ({agree}/{total})\n")

    print("per-state recall (manual state reproduced by algo):")
    for s in _STATES:
        c = _CODE[s]
        m = man == c
        mtot = int(np.sum(m))
        hit = int(np.sum(m & (algo == c)))
        r = 100.0 * hit / mtot if mtot else float("nan")
        print(f"  {s:9} recall={r:6.2f}%   manual_samples={mtot}")

    mb, ab = _count_s1(man), _count_s1(algo)
    print(f"\nbeat count (S1 runs):  manual={mb}  algo={ab}  diff={ab - mb:+d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
