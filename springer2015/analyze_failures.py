"""Analyze hard-failing files across all Springer model variants.

Runs all 4 models on N sampled CirCor recordings, compares per-file F1,
and correlates failure patterns with CirCor metadata.

Usage:
    python springer2015/analyze_failures.py [--n 200] [--seed 0] [--threshold 0.5]
"""

import argparse
import glob
import os
import random
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_BENCH = os.path.join(_ROOT, "benchmarking")
for _p in (_HERE, _ROOT, _BENCH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from springer_hsmm.model_io import load_springer_model           # noqa: E402
from springer_hsmm.options import default_springer_hsmm_options  # noqa: E402
from springer_hsmm.run import run_springer_segmentation_algorithm  # noqa: E402
from bench_scoring import (                                       # noqa: E402
    DEFAULT_TOLERANCES_SEC, derive_metrics,
    filter_to_windows, s1_centers, s2_centers, score_file,
)

CIRCOR_ROOT = (
    r"G:\HB other\PCG Datasets"
    r"\the-circor-digiscope-phonocardiogram-dataset-1.0.3"
)
TRAINING_DATA = os.path.join(CIRCOR_ROOT, "training_data")
METADATA_CSV  = os.path.join(CIRCOR_ROOT, "training_data.csv")

CODE_TO_STATE = {1: "S1", 2: "systole", 3: "S2", 4: "diastole"}
TOL = 0.06

MODELS = {
    "original": os.path.join(_HERE, "springer_original.npz"),
    "potes":    os.path.join(_HERE, "cristhian_potes_model.npz"),
    "circor":   os.path.join(_HERE, "springer_circor_trained.npz"),
    "pn2016":   os.path.join(_HERE, "springer_pn2016_trained.npz"),
}


def collect_recordings(root):
    found = []
    for tsv in sorted(glob.glob(os.path.join(root, "*.tsv"))):
        wav = tsv[:-4] + ".wav"
        if os.path.isfile(wav):
            found.append((wav, tsv))
    return found


def load_tsv(path):
    spans = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                start, end, code = float(parts[0]), float(parts[1]), int(float(parts[2]))
            except ValueError:
                continue
            state = CODE_TO_STATE.get(code)
            if state and end > start:
                spans.append((start, end, state))
    return spans


def states_to_spans(states, fs):
    spans = []
    states = np.asarray(states).astype(int)
    n = len(states)
    i = 0
    while i < n:
        s = states[i]
        j = i
        while j < n and states[j] == s:
            j += 1
        name = CODE_TO_STATE.get(int(s))
        if name:
            spans.append((i / fs, j / fs, name))
        i = j
    return spans


def labeled_windows(spans):
    if not spans:
        return []
    ordered = sorted((s, e) for s, e, _ in spans)
    cur_s, cur_e = ordered[0]
    windows = []
    for s, e in ordered[1:]:
        if s <= cur_e + 1e-6:
            cur_e = max(cur_e, e)
        else:
            windows.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    windows.append((cur_s, cur_e))
    return windows


def f1_score(gt, pred):
    manual_s1 = s1_centers(gt)
    if not manual_s1:
        return None
    windows = labeled_windows(gt)
    ps1 = filter_to_windows(s1_centers(pred), windows)
    ps2 = filter_to_windows(s2_centers(pred), windows)
    counts = score_file(manual_s1, ps1, ps2, [TOL])
    return derive_metrics(counts[TOL])["f1"]


def parse_filename(name):
    m = re.match(r"^(\d+)_([A-Z]+)", name)
    if m:
        return int(m.group(1)), m.group(2)
    return None, None


def ratio_table(df, col, threshold):
    low  = df[df["f1"] < threshold][col].fillna("nan")
    all_ = df[col].fillna("nan")
    low_c, all_c = Counter(low), Counter(all_)
    n_low, n_all = len(low), len(all_)
    rows = []
    for k in sorted(set(low_c) | set(all_c), key=lambda x: -low_c.get(x, 0)):
        lp = low_c.get(k, 0) / n_low if n_low else 0
        ap = all_c.get(k, 0) / n_all if n_all else 0
        ratio = lp / ap if ap > 0 else float("inf")
        rows.append((k, lp, ap, ratio))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",         type=int,   default=200)
    ap.add_argument("--seed",      type=int,   default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    recs = collect_recordings(TRAINING_DATA)
    random.Random(args.seed).shuffle(recs)
    if args.n:
        recs = recs[: args.n]

    loaded = {}
    for name, path in MODELS.items():
        if os.path.isfile(path):
            loaded[name] = load_springer_model(path)
        else:
            print(f"SKIP {name}: not found")

    opts = default_springer_hsmm_options()

    # Score every file with every model
    rows = []
    for i, (wav, tsv) in enumerate(recs, 1):
        name = os.path.splitext(os.path.basename(wav))[0]
        pid, pos = parse_filename(name)
        gt = load_tsv(tsv)
        if not s1_centers(gt):
            continue
        try:
            audio, fs = sf.read(wav)
            audio = np.asarray(audio, dtype=np.float64).flatten()
        except Exception:
            continue
        rec = {"name": name, "patient_id": pid, "position": pos}
        for mname, model in loaded.items():
            try:
                states, _ = run_springer_segmentation_algorithm(
                    audio, fs, model["B_matrix"], model["pi_vector"],
                    model["total_obs_distribution"], opts,
                )
                pred = states_to_spans(states, float(fs))
                rec[mname] = f1_score(gt, pred)
            except Exception:
                rec[mname] = None
        rows.append(rec)
        if i % 50 == 0:
            print(f"  {i}/{len(recs)}")

    df_base = pd.DataFrame(rows)

    # Join metadata
    meta = pd.read_csv(METADATA_CSV).rename(columns={"Patient ID": "patient_id"})
    df = df_base.merge(meta, on="patient_id", how="left")

    # ---- Summary ----
    print(f"\n{'='*65}")
    print(f"Files scored: {len(df)}  |  threshold: {args.threshold}  |  tol: {int(TOL*1000)}ms")
    print()

    model_names = list(loaded.keys())
    hdr = f"{'':25}" + "".join(f"{m:>12}" for m in model_names)
    print(hdr)
    print("-" * len(hdr))

    # Mean F1
    row = f"{'Mean F1':25}"
    for m in model_names:
        vals = df[m].dropna()
        row += f"  {vals.mean()*100:>8.1f}%"
    print(row)

    # Hard-fail rate
    row = f"{'Hard failures (<50%)':25}"
    for m in model_names:
        vals = df[m].dropna()
        rate = (vals < args.threshold).mean()
        row += f"  {rate*100:>8.1f}%"
    print(row)

    # Zero-F1 count
    row = f"{'Zero F1 (0%)':25}"
    for m in model_names:
        vals = df[m].dropna()
        row += f"  {(vals == 0).sum():>9d}"
    print(row)

    # ---- Per-model risk factor analysis ----
    META_COLS = [
        ("Age",                    "Age"),
        ("Murmur",                 "Murmur"),
        ("Systolic murmur timing", "Systolic murmur timing"),
        ("Systolic murmur quality","Systolic murmur quality"),
        ("Most audible location",  "Most audible location"),
        ("Campaign",               "Campaign"),
    ]

    print(f"\n{'='*65}")
    print("RISK FACTORS (ratio = share in hard-fails / share overall)")
    print(f"Only showing ratio > 1.4x\n")

    for mname in model_names:
        sub = df[["patient_id", "position", mname] + [c for _, c in META_COLS if c in df.columns]].copy()
        sub = sub.rename(columns={mname: "f1"}).dropna(subset=["f1"])
        fail_rate = (sub["f1"] < args.threshold).mean()
        print(f"[{mname}]  fail_rate={fail_rate*100:.1f}%  n={len(sub)}")
        for label, col in META_COLS:
            if col not in sub.columns:
                continue
            rows_r = ratio_table(sub, col, args.threshold)
            notable = [(k, lp, ap, r) for k, lp, ap, r in rows_r if r > 1.4 and lp > 0.05]
            if notable:
                for k, lp, ap, r in notable:
                    print(f"  {label:<30} {str(k):<20} {lp*100:>5.1f}% vs {ap*100:>5.1f}%  ({r:.2f}x)")
        print()

    # ---- Position breakdown per model ----
    print(f"{'='*65}")
    print("MEAN F1 BY AUSCULTATION POSITION")
    print()
    pos_col = "position"
    positions = sorted(df[pos_col].dropna().unique())
    hdr2 = f"{'Position':8}" + "".join(f"{m:>12}" for m in model_names)
    print(hdr2)
    for pos in positions:
        sub = df[df[pos_col] == pos]
        row = f"{pos:<8}"
        for m in model_names:
            v = sub[m].dropna()
            row += f"  {v.mean()*100:>8.1f}%" if len(v) else f"  {'—':>9}"
        print(row)


if __name__ == "__main__":
    main()
