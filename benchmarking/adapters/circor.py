#!/usr/bin/env python3
"""CirCor DigiScope adapter — benchmark the segmenter against published Se/PPV/F1.

Source: PhysioNet `circor-heart-sound` v1.0.3 (George B. Moody Challenge 2022).
Ground truth: one `.tsv` per recording, rows `start_sec end_sec code` with
    1=S1  2=systole  3=S2  4=diastole  0=unlabeled
Recordings are only annotated in a middle window; ~46.5% of total time is
code-0 (mostly leading/trailing). We clip to labeled spans: predictions outside
them are neither FP nor FN. Murmur labels are ignored — the tool segments only.

Reports per-beat Sensitivity / PPV / F1 at 60 ms and 100 ms (see
benchmarking/bench_scoring.py and docs/adr/0003).

Usage:
    python benchmarking/adapters/circor.py [DATASET_ROOT] [--n 200] [--seed 0]

DATASET_ROOT defaults to the training_data folder of the PhysioNet release.
"""

import argparse
import glob
import json
import logging
import os
import random
import sys
import tempfile
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

# Pin BLAS/FFT to one thread per process: workers parallelize across files, so
# per-file numpy threads would oversubscribe the pool. Must precede numpy import
# (pulled in by config/pipeline below); with spawn, children re-run this module.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BENCH_DIR = os.path.dirname(_THIS_DIR)
_ROOT = os.path.dirname(_BENCH_DIR)
for _p in (_BENCH_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bench_scoring import (  # noqa: E402
    DEFAULT_TOLERANCES_SEC,
    Span,
    derive_metrics,
    filter_to_windows,
    s1_centers,
    s2_centers,
    score_file,
    span_center,
)
from config import DEFAULT_PARAMS  # noqa: E402
from pipeline import analyze_wav_file  # noqa: E402

DEFAULT_ROOT = (
    r"G:\HB other\PCG Datasets"
    r"\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data"
)

CODE_TO_STATE = {1: "S1", 2: "systole", 3: "S2", 4: "diastole"}


# ---------------------------------------------------------------------------
# CirCor-specific ground-truth loading
# ---------------------------------------------------------------------------

def load_tsv(path: str) -> List[Span]:
    """Parse a CirCor .tsv into labeled (start, end, state) spans.

    Drops code-0 (unlabeled) and any out-of-range code (e.g. a stray 28).
    """
    spans: List[Span] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
                code = int(float(parts[2]))
            except ValueError:
                continue
            state = CODE_TO_STATE.get(code)
            if state is None or end <= start:
                continue
            spans.append((start, end, state))
    return spans


def labeled_windows(spans: List[Span]) -> List[Tuple[float, float]]:
    """Merge labeled spans into contiguous time windows, splitting at gaps.

    A gap between consecutive spans marks an excised code-0 region; predictions
    landing there are ignored by filter_to_windows.
    """
    if not spans:
        return []
    ordered = sorted((s, e) for s, e, _ in spans)
    windows: List[Tuple[float, float]] = []
    cur_s, cur_e = ordered[0]
    for s, e in ordered[1:]:
        if s <= cur_e + 1e-6:  # touching/overlapping -> same window
            cur_e = max(cur_e, e)
        else:
            windows.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    windows.append((cur_s, cur_e))
    return windows


def collect_recordings(root: str) -> List[Tuple[str, str]]:
    """Return (wav_path, tsv_path) for every CirCor recording under root."""
    found = []
    for tsv in sorted(glob.glob(os.path.join(root, "*.tsv"))):
        wav = tsv[:-4] + ".wav"
        if os.path.isfile(wav):
            found.append((wav, tsv))
    return found


# ---------------------------------------------------------------------------
# Pipeline run -> predicted S1/S2 centers
# ---------------------------------------------------------------------------

_OUTPUT_OPTIONS = {
    "html": False, "png": False, "csv": False, "summary": False, "debug": False,
    "filtered_wav": False, "spectrogram": False, "fft_profiles": False,
    "output_all_passes": False, "working_wav_in_output": False,
}


def _params() -> Dict:
    p = {**DEFAULT_PARAMS, "save_filtered_wav": False, "enable_fft_profiles": False}
    # Optional param overrides via env (so sweeps reach spawned workers).
    # Format: BENCH_PARAM_OVERRIDES="key=val,key=val" (numeric/bool values).
    raw = os.environ.get("BENCH_PARAM_OVERRIDES", "")
    for item in (s for s in raw.split(",") if s.strip()):
        k, _, v = item.partition("=")
        k, v = k.strip(), v.strip()
        if not k:
            continue
        if v.lower() in ("true", "false"):
            p[k] = v.lower() == "true"
        else:
            try:
                p[k] = float(v)
            except ValueError:
                p[k] = v
    return p


def predict_centers(
    wav_path: str, params: Dict, sample_rate: int
) -> Optional[Tuple[List[float], List[float]]]:
    """Run the pipeline; return (pred_s1_centers, pred_s2_centers) in seconds."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _, _, _, data = analyze_wav_file(
                wav_path, params, None,
                original_file_path=wav_path,
                output_directory=tmp,
                output_options=_OUTPUT_OPTIONS,
                collect_fft_for_aggregate=False,
            )
        except Exception as exc:  # noqa: BLE001
            logging.error("pipeline error on %s: %s", os.path.basename(wav_path), exc)
            return None
    if data is None:
        return None

    sr = float(sample_rate)
    # pass3_state_boundaries: (start_sample, end_sample, state_name, meta)
    boundaries = data.get("pass3_state_boundaries") or []
    pred_spans: List[Span] = [
        (b[0] / sr, b[1] / sr, str(b[2])) for b in boundaries
    ]
    return s1_centers(pred_spans), s2_centers(pred_spans)


# ---------------------------------------------------------------------------
# Parallel worker — one recording per call, fully independent
# ---------------------------------------------------------------------------

_W_PARAMS: Optional[Dict] = None
_W_SR: int = 600


def _init_worker() -> None:
    """Build pipeline params once per worker process."""
    global _W_PARAMS, _W_SR
    _W_PARAMS = _params()
    _W_SR = int(_W_PARAMS.get("preprocess_target_sample_rate", 600))


def _score_one(task: Tuple[str, str]) -> Dict:
    """Load GT, run pipeline, score one recording. Returns a result/skip dict."""
    wav_path, tsv_path = task
    name = os.path.basename(wav_path)
    gt_spans = load_tsv(tsv_path)
    manual_s1 = s1_centers(gt_spans)
    if not manual_s1:
        return {"file": name, "skip": "empty labels"}
    windows = labeled_windows(gt_spans)

    pred = predict_centers(wav_path, _W_PARAMS or _params(), _W_SR)
    if pred is None:
        return {"file": name, "skip": "pipeline error / no data"}
    pred_s1 = filter_to_windows(pred[0], windows)
    pred_s2 = filter_to_windows(pred[1], windows)

    counts = score_file(manual_s1, pred_s1, pred_s2, DEFAULT_TOLERANCES_SEC)
    return {
        "file": name,
        "manual_s1": len(manual_s1),
        "pred_s1": len(pred_s1),
        "counts": counts,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(root: str, n: Optional[int], seed: int, jobs: int) -> None:
    recordings = collect_recordings(root)
    if not recordings:
        print(f"No CirCor recordings found under: {root}", file=sys.stderr)
        sys.exit(1)

    total = len(recordings)
    if n is not None and n < total:
        random.Random(seed).shuffle(recordings)
        recordings = recordings[:n]

    tolerances = DEFAULT_TOLERANCES_SEC
    jobs = max(1, min(jobs, len(recordings)))

    # Summed counts per tolerance across all scored files.
    totals: Dict[float, Dict[str, int]] = {
        t: {"tp": 0, "fn": 0, "fp": 0, "flip": 0} for t in tolerances
    }
    per_file: List[Dict] = []
    scored = skipped = 0
    done = 0
    n_in = len(recordings)

    print(f"CirCor benchmark — {n_in}/{total} recordings  (seed={seed}, jobs={jobs})\n")

    def _consume(res: Dict) -> None:
        nonlocal scored, skipped, done
        done += 1
        name = res["file"]
        if "skip" in res:
            skipped += 1
            print(f"  [{done}/{n_in}] SKIP {name}  ({res['skip']})")
            return
        counts = res["counts"]
        for t in tolerances:
            for k in ("tp", "fn", "fp", "flip"):
                totals[t][k] += counts[t][k]
        row = {
            "file": name,
            "manual_s1": res["manual_s1"],
            "pred_s1": res["pred_s1"],
            "counts": {f"{t:.2f}": counts[t] for t in tolerances},
            "f1_60ms": round(derive_metrics(counts[0.06])["f1"], 4),
        }
        per_file.append(row)
        scored += 1
        f1s = "  ".join(
            f"{int(t*1000)}ms F1={derive_metrics(counts[t])['f1']:.3f}"
            for t in tolerances
        )
        print(f"  [{done}/{n_in}] {name:24s} s1={res['manual_s1']:3d} "
              f"pred={res['pred_s1']:3d}  {f1s}")

    if jobs == 1:
        _init_worker()
        for task in recordings:
            _consume(_score_one(task))
    else:
        with Pool(processes=jobs, initializer=_init_worker) as pool:
            for res in pool.imap_unordered(_score_one, recordings, chunksize=4):
                _consume(res)

    # ---- Aggregate report ----
    sep = "=" * 64
    print(f"\n{sep}\nGRAND TOTAL  (scored={scored}  skipped={skipped})")
    summary_metrics: Dict[str, Dict] = {}
    for t in tolerances:
        c = totals[t]
        m = derive_metrics(c)
        label = f"{int(t*1000)}ms"
        summary_metrics[label] = {**c, **{k: round(v, 4) for k, v in m.items()}}
        print(
            f"  tol={label:>6}  Se={m['se']*100:5.1f}%  PPV={m['ppv']*100:5.1f}%  "
            f"F1={m['f1']*100:5.1f}%   "
            f"(TP={c['tp']} FN={c['fn']} FP={c['fp']} flip={c['flip']})"
        )
    print("  reference: Springer HSMM Se ~= 97.4% on CirCor (60ms)")
    print(sep)

    out = {
        "dataset": "circor-1.0.3",
        "root": root,
        "scored": scored,
        "skipped": skipped,
        "tolerances_sec": list(tolerances),
        "totals": summary_metrics,
        "per_file": per_file,
    }
    out_path = os.path.join(_BENCH_DIR, "circor_benchmark_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"JSON -> {out_path}")


def main() -> None:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="CirCor segmentation benchmark.")
    ap.add_argument("root", nargs="?", default=DEFAULT_ROOT, help="training_data dir")
    ap.add_argument("--n", type=int, default=200, help="subset size (default 200; 0=all)")
    ap.add_argument("--seed", type=int, default=0, help="subset RNG seed")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="worker processes (default: CPU count - 1; 1 = serial)")
    args = ap.parse_args()
    n = None if args.n == 0 else args.n
    if not os.path.isdir(args.root):
        print(f"Not a directory: {args.root}", file=sys.stderr)
        sys.exit(1)
    run(args.root, n, args.seed, args.jobs)


if __name__ == "__main__":
    main()
