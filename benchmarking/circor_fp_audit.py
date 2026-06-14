#!/usr/bin/env python3
"""Characterize CirCor false-positive S1s: where do they land in the manual timeline?

For each predicted S1 (in a labeled window) that is NOT within 60 ms of a manual
S1, find which manual state its center falls in: S2 (octave/flip), systole,
diastole, or a gap. Tells us whether over-detection is mislabeled real sounds
(S2) or spurious detections in the quiet intervals.

    python benchmarking/circor_fp_audit.py [--n 0] [--jobs N]
"""

import argparse
import bisect
import os
import sys
from collections import Counter
from multiprocessing import Pool

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, "adapters"))
sys.path.insert(0, _THIS)

from adapters import circor as cc  # noqa: E402
from bench_scoring import filter_to_windows, s1_centers, s2_centers  # noqa: E402

TOL = 0.06


def _state_at(spans_sorted, starts, t):
    """Manual state name covering time t, or 'gap'."""
    i = bisect.bisect_right(starts, t) - 1
    if 0 <= i < len(spans_sorted):
        s, e, st = spans_sorted[i]
        if s <= t <= e:
            return st
    return "gap"


def _audit_one(task):
    wav, tsv = task
    gt = cc.load_tsv(tsv)
    manual_s1 = s1_centers(gt)
    if not manual_s1:
        return None
    windows = cc.labeled_windows(gt)
    pred = cc.predict_centers(wav, cc._W_PARAMS or cc._params(), cc._W_SR)
    if pred is None:
        return None
    ps1 = sorted(filter_to_windows(pred[0], windows))
    msorted = sorted(manual_s1)

    spans_sorted = sorted((s, e, st) for s, e, st in gt)
    starts = [s for s, _, _ in spans_sorted]

    c = Counter()
    for t in ps1:
        # matched to a manual S1?
        j = bisect.bisect_left(msorted, t)
        near = min(
            (abs(msorted[k] - t) for k in (j - 1, j) if 0 <= k < len(msorted)),
            default=9e9,
        )
        if near <= TOL:
            c["matched"] += 1
        else:
            c["FP_" + _state_at(spans_sorted, starts, t)] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=cc.DEFAULT_ROOT)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    recs = cc.collect_recordings(args.root)
    if args.n:
        import random
        random.Random(0).shuffle(recs)
        recs = recs[:args.n]
    jobs = max(1, min(args.jobs, len(recs)))
    print(f"FP audit — {len(recs)} recordings, jobs={jobs}\n")

    total = Counter()
    with Pool(processes=jobs, initializer=cc._init_worker) as pool:
        for c in pool.imap_unordered(_audit_one, recs, chunksize=4):
            if c:
                total += c

    fp = {k: v for k, v in total.items() if k.startswith("FP_")}
    fp_total = sum(fp.values())
    print(f"matched S1: {total['matched']}")
    print(f"false-positive S1: {fp_total}\n  by manual state they land in:")
    for k in sorted(fp, key=lambda x: -fp[x]):
        print(f"    {k[3:]:10s}: {fp[k]:6d}  ({fp[k]/fp_total*100:4.1f}%)")


if __name__ == "__main__":
    main()
