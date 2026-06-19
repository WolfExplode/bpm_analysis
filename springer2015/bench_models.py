"""Compare all Springer model variants on CirCor recordings.

Samples N recordings, runs all 4 trained models, prints aggregate F1 table.

Usage:
    python springer2015/bench_models.py [--n 50] [--seed 0]
"""

import argparse
import glob
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_BENCH = os.path.join(_ROOT, "benchmarking")
for _p in (_HERE, _ROOT, _BENCH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from springer_hsmm.model_io import load_springer_model       # noqa: E402
from springer_hsmm.options import default_springer_hsmm_options  # noqa: E402
from springer_hsmm.run import run_springer_segmentation_algorithm  # noqa: E402
from bench_scoring import (                                  # noqa: E402
    DEFAULT_TOLERANCES_SEC, Span,
    derive_metrics, filter_to_windows, s1_centers, s2_centers, score_file,
)

DEFAULT_ROOT = (
    r"G:\HB other\PCG Datasets"
    r"\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data"
)
CODE_TO_STATE = {1: "S1", 2: "systole", 3: "S2", 4: "diastole"}

MODELS = {
    "original":  os.path.join(_HERE, "springer_original.npz"),
    "potes":     os.path.join(_HERE, "cristhian_potes_model.npz"),
    "circor":    os.path.join(_HERE, "springer_circor_trained.npz"),
    "pn2016":    os.path.join(_HERE, "springer_pn2016_trained.npz"),
}


def collect_recordings(root: str) -> List[Tuple[str, str]]:
    found = []
    for tsv in sorted(glob.glob(os.path.join(root, "*.tsv"))):
        wav = tsv[:-4] + ".wav"
        if os.path.isfile(wav):
            found.append((wav, tsv))
    return found


def load_tsv(path: str) -> List[Span]:
    spans: List[Span] = []
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
            if state is None or end <= start:
                continue
            spans.append((start, end, state))
    return spans


def states_to_spans(states: np.ndarray, fs: float) -> List[Span]:
    spans: List[Span] = []
    states = np.asarray(states).astype(int)
    n = len(states)
    i = 0
    while i < n:
        s = states[i]
        j = i
        while j < n and states[j] == s:
            j += 1
        name = CODE_TO_STATE.get(int(s))
        if name is not None:
            spans.append((i / fs, j / fs, name))
        i = j
    return spans


def labeled_windows(spans: List[Span]) -> List[Tuple[float, float]]:
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


def f1_vs_gt(gt: List[Span], pred: List[Span]) -> Dict[float, Dict[str, float]]:
    manual_s1 = s1_centers(gt)
    windows = labeled_windows(gt)
    ps1 = filter_to_windows(s1_centers(pred), windows)
    ps2 = filter_to_windows(s2_centers(pred), windows)
    counts = score_file(manual_s1, ps1, ps2, DEFAULT_TOLERANCES_SEC)
    return {tol: {**counts[tol], **derive_metrics(counts[tol])} for tol in DEFAULT_TOLERANCES_SEC}


def run_model(wav_path: str, model: Dict, opts: Dict) -> List[Span]:
    audio, fs = sf.read(wav_path)
    audio = np.asarray(audio, dtype=np.float64).flatten()
    states, _ = run_springer_segmentation_algorithm(
        audio, fs, model["B_matrix"], model["pi_vector"],
        model["total_obs_distribution"], opts,
    )
    return states_to_spans(states, float(fs))


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark Springer model variants on CirCor.")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--n", type=int, default=50, help="Recordings to sample (0=all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = collect_recordings(args.root)
    if not recs:
        print(f"No recordings under {args.root}", file=sys.stderr)
        sys.exit(1)
    random.Random(args.seed).shuffle(recs)
    if args.n:
        recs = recs[: args.n]
    print(f"Benchmarking {len(recs)} recordings...")

    # Load available models
    loaded = {}
    for name, path in MODELS.items():
        if os.path.isfile(path):
            loaded[name] = load_springer_model(path)
            print(f"  loaded {name}: {os.path.basename(path)}")
        else:
            print(f"  SKIP {name}: {path} not found")

    if not loaded:
        print("No models found.", file=sys.stderr)
        sys.exit(1)

    opts = default_springer_hsmm_options()
    agg = {name: {t: {"tp": 0, "fn": 0, "fp": 0} for t in DEFAULT_TOLERANCES_SEC}
           for name in loaded}

    for i, (wav, tsv) in enumerate(recs, 1):
        name = os.path.splitext(os.path.basename(wav))[0]
        gt = load_tsv(tsv)
        if not s1_centers(gt):
            print(f"  [{i}/{len(recs)}] SKIP {name} (no S1 in GT)")
            continue
        row_parts = [f"[{i:3d}/{len(recs)}] {name:14s}"]
        ok = True
        per_model: Dict[str, Dict] = {}
        for mname, model in loaded.items():
            try:
                pred = run_model(wav, model, opts)
                m = f1_vs_gt(gt, pred)
                per_model[mname] = m
            except Exception as exc:
                print(f"  [{i}/{len(recs)}] ERROR {name} [{mname}]: {exc}")
                ok = False
                break
        if not ok:
            continue
        for mname, m in per_model.items():
            for tol in DEFAULT_TOLERANCES_SEC:
                for k in ("tp", "fn", "fp"):
                    agg[mname][tol][k] += m[tol][k]
            row_parts.append(f"{mname}={m[0.06]['f1']*100:5.1f}%")
        print("  " + "  ".join(row_parts))

    tol_main = 0.06
    tols = list(DEFAULT_TOLERANCES_SEC)
    print(f"\n{'Model':<12}  " + "  ".join(f"F1@{int(t*1000)}ms" for t in tols))
    print("-" * (12 + 14 * len(tols)))
    for mname in loaded:
        parts = [f"{mname:<12}"]
        for tol in tols:
            m = derive_metrics(agg[mname][tol])
            parts.append(f"  {m['f1']*100:6.1f}%")
        print("".join(parts))

    print(f"\n(primary tol={int(tol_main*1000)}ms, micro-aggregated over {len(recs)} files)")


if __name__ == "__main__":
    main()
