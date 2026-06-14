#!/usr/bin/env python3
"""Phase-audit tool: is CirCor flip mass a whole-recording S1<->S2 inversion?

For every recording, score normally (manual S1 vs predicted S1) and *swapped*
(manual S1 vs predicted S2). If a file's swapped F1 is much higher than its
normal F1, the tool got the global phase backwards on that file. Quantifies the
ceiling of a per-recording phase-correction fix.

    python benchmarking/circor_phase_audit.py [--n 0] [--jobs N]
"""

import argparse
import os
import sys
from multiprocessing import Pool

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, "adapters"))
sys.path.insert(0, _THIS)

from adapters import circor as cc  # noqa: E402
from bench_scoring import derive_metrics, filter_to_windows, s1_centers, s2_centers, score_file  # noqa: E402

TOL = 0.06


def _audit_one(task):
    wav, tsv = task
    name = os.path.basename(wav)
    gt = cc.load_tsv(tsv)
    manual = s1_centers(gt)
    if not manual:
        return None
    windows = cc.labeled_windows(gt)
    pred = cc.predict_centers(wav, cc._W_PARAMS or cc._params(), cc._W_SR)
    if pred is None:
        return None
    ps1 = filter_to_windows(pred[0], windows)
    ps2 = filter_to_windows(pred[1], windows)

    normal = score_file(manual, ps1, ps2, (TOL,))[TOL]
    swapped = score_file(manual, ps2, ps1, (TOL,))[TOL]

    # Unsupervised phase cue from the tool's OWN labels: median systole (S1->S2)
    # vs median diastole (S2->next S1). At rest systole < diastole.
    seq = sorted([(c, "S1") for c in ps1] + [(c, "S2") for c in ps2])
    syst, dias = [], []
    for (t0, l0), (t1, l1) in zip(seq, seq[1:]):
        if l0 == "S1" and l1 == "S2":
            syst.append(t1 - t0)
        elif l0 == "S2" and l1 == "S1":
            dias.append(t1 - t0)
    import statistics
    med_s = statistics.median(syst) if syst else None
    med_d = statistics.median(dias) if dias else None
    # predicted inverted when systole looks longer than diastole
    pred_inv = (med_s is not None and med_d is not None and med_s > med_d)
    return {
        "file": name,
        "manual": len(manual),
        "normal": normal,
        "swapped": swapped,
        "f1_normal": derive_metrics(normal)["f1"],
        "f1_swapped": derive_metrics(swapped)["f1"],
        "pred_inv": pred_inv,
        "med_s": med_s,
        "med_d": med_d,
    }


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
    print(f"Phase audit — {len(recs)} recordings, jobs={jobs}\n")

    rows = []
    with Pool(processes=jobs, initializer=cc._init_worker) as pool:
        for r in pool.imap_unordered(_audit_one, recs, chunksize=4):
            if r:
                rows.append(r)

    # Per-file phase decision: would swapping help this file?
    inverted = [r for r in rows if r["f1_swapped"] > r["f1_normal"] + 0.2]
    strong_inv = [r for r in rows if r["f1_normal"] < 0.3 and r["f1_swapped"] > 0.7]

    # Oracle phase: pick the better orientation per file, sum counts.
    def agg(key):
        tp = sum(r[key]["tp"] for r in rows)
        fn = sum(r[key]["fn"] for r in rows)
        fp = sum(r[key]["fp"] for r in rows)
        return tp, fn, fp

    tp_n, fn_n, fp_n = agg("normal")
    # oracle: per file choose orientation with higher f1
    o_tp = o_fn = o_fp = 0
    for r in rows:
        best = r["normal"] if r["f1_normal"] >= r["f1_swapped"] else r["swapped"]
        o_tp += best["tp"]; o_fn += best["fn"]; o_fp += best["fp"]

    # UNSUPERVISED rule: swap whenever pred_inv (median systole > diastole)
    u_tp = u_fn = u_fp = 0
    for r in rows:
        chosen = r["swapped"] if r.get("pred_inv") else r["normal"]
        u_tp += chosen["tp"]; u_fn += chosen["fn"]; u_fp += chosen["fp"]

    # how well does the unsupervised cue agree with the oracle?
    oracle_swap = [r for r in rows if r["f1_swapped"] > r["f1_normal"]]
    tp_rule = sum(1 for r in oracle_swap if r.get("pred_inv"))
    fp_rule = sum(1 for r in rows if r.get("pred_inv") and r["f1_swapped"] <= r["f1_normal"])
    print(f"cue: caught {tp_rule}/{len(oracle_swap)} oracle-swaps, "
          f"{fp_rule} false swaps\n")

    def line(tag, tp, fn, fp):
        m = derive_metrics({"tp": tp, "fn": fn, "fp": fp})
        print(f"  {tag:22s} Se={m['se']*100:5.1f}%  PPV={m['ppv']*100:5.1f}%  "
              f"F1={m['f1']*100:5.1f}%  (TP={tp} FN={fn} FP={fp})")

    print(f"files scored: {len(rows)}")
    print(f"files where swap helps (>+0.2 F1): {len(inverted)} "
          f"({len(inverted)/len(rows)*100:.1f}%)")
    print(f"strong global inversions (norm<0.3, swap>0.7): {len(strong_inv)}\n")
    line("current (normal):", tp_n, fn_n, fp_n)
    line("UNSUPERVISED rule:", u_tp, u_fn, u_fp)
    line("ORACLE per-file phase:", o_tp, o_fn, o_fp)
    print("\noracle = ceiling if global per-recording phase were always correct.")
    print("unsupervised = implementable: swap when median systole > diastole.")


if __name__ == "__main__":
    main()
