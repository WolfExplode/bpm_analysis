"""
Regression check: compare the algorithm's Pass-3 cardiac state output against the
recording's Annotation (found by fingerprint next to the recording, or --annotation).

Builds per-sample state-label arrays for both (S1 / systole / S2 / diastole; the
Annotation's systole/diastole are the derived gaps between its sounds),
then reports:
  * overall per-sample agreement
  * per-state recall (how much of each manual state the algorithm reproduced)
  * beat counts (manual vs algorithm S1 segments) — catches missed or phantom beats

Use it to prove a change didn't regress segmentation on a file with trusted labels.

Usage (from repo root):
    python debug_helpers/compare_to_annotation.py "inputs/.../file.wav"
    python debug_helpers/compare_to_annotation.py "inputs/.../file.wav" --annotation "<path>.annotation.json"
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

from debug_helpers._common import (  # noqa: E402
    env_sample_rate, params, reconfigure_stdio, run_pipeline,
)
from pcg import annotation, recording  # noqa: E402

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


def _manual_labels(annotation_path, sr, n):
    arr = np.zeros(n, dtype=np.int8)
    rows = 0
    for a, b, state in annotation.derived_states(annotation.load(annotation_path).spans):
        c = _CODE.get(state, 0)
        if c:
            arr[max(0, int(round(a * sr))):min(n, int(round(b * sr)))] = c
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
    ap.add_argument("--annotation", default=None, help="Annotation file (default: found by fingerprint).")
    ns = ap.parse_args(argv)

    reconfigure_stdio()

    ann_path = ns.annotation or annotation.find_for_recording(ns.wav, recording.fingerprint(ns.wav))
    if not ann_path or not os.path.exists(ann_path):
        print(f"No Annotation found for {ns.wav}", file=sys.stderr)
        return 2

    data, sr, n = _run(ns.wav, params())
    if data is None or not sr or not n:
        print("Could not determine sample rate / length.", file=sys.stderr)
        return 2

    algo = _algo_labels(data, n)
    man, mrows = _manual_labels(ann_path, sr, n)

    both = (algo != 0) & (man != 0)
    agree = int(np.sum((algo == man) & both))
    total = int(np.sum(both))
    overall = 100.0 * agree / total if total else 0.0

    print(f"# {os.path.basename(ns.wav)}  (sr~{sr:.1f}Hz, n={n}, annotation states={mrows})")
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
