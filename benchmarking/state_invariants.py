#!/usr/bin/env python3
"""
State-timeline invariant gate.

The Pass 3 cardiac-state timeline is supposed to satisfy structural invariants
that the algorithm never enforces directly. This gate *measures* them across a
corpus so that a change which trades accuracy for structural breakage becomes
visible — the "fitness term for structure" the accuracy benchmark lacks.

Unlike run_benchmark.py this needs **no manual ground truth**: every metric is an
internal consistency check on the pipeline's own output.

Metrics (lower is better; all should trend to 0 except where noted):
    overlaps_gap_rebuild   two spans claim the same samples, >=1 from a gap fill
    overlaps_edge_paint    two real spans whose painted edges overlap (minor)
    coverage_desync_runs   dense labels say a real state but the boundary strip
                           shows no matching band
    seq_missing_s1         diastole -> S2 (an S2 with no S1 before it)
    seq_bad_transition     any other illegal cardiac-cycle transition
    swap_mismatches        an S1 peak under an S2 band (or vice-versa)

Usage (from repo root):
    python benchmarking/state_invariants.py                         # compare to baseline
    python benchmarking/state_invariants.py --write-baseline        # (re)write baseline
    python benchmarking/state_invariants.py inputs/Difficulty 3 -j 8
    python benchmarking/state_invariants.py --json out.json

Exit code: 0 if no metric regressed vs the baseline (or no baseline / --write-baseline);
1 if any total metric increased. With no baseline file it just reports.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_SCRIPT_DIR)
for _p in (_REPO, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import soundfile as sf  # noqa: E402

from config import DEFAULT_PARAMS  # noqa: E402
from pipeline import analyze_wav_file  # noqa: E402
from debug_helpers.overlap_detector import find_overlapping_states  # noqa: E402
from debug_helpers.coverage_detector import find_label_boundary_desync  # noqa: E402
from debug_helpers.state_sequence_detector import find_sequence_violations  # noqa: E402
from debug_helpers.peak_state_mismatch_detector import find_peak_state_mismatches  # noqa: E402
from debug_helpers.scan_sequence import _bpm_hint_from_name  # noqa: E402

_BASELINE = os.path.join(_SCRIPT_DIR, "state_invariants_baseline.json")
_OO = {k: False for k in (
    "html", "png", "csv", "summary", "debug", "filtered_wav",
    "spectrogram", "fft_profiles", "output_all_passes", "working_wav_in_output",
)}
_METRICS = (
    "overlaps_gap_rebuild", "overlaps_edge_paint", "coverage_desync_runs",
    "seq_missing_s1", "seq_bad_transition", "swap_mismatches",
)


def _params():
    return {**DEFAULT_PARAMS, "save_filtered_wav": False, "enable_fft_profiles": False}


def _collect_wavs(paths):
    wavs = []
    for p in paths:
        if os.path.isfile(p) and p.lower().endswith(".wav"):
            wavs.append(p)
        elif os.path.isdir(p):
            wavs.extend(sorted(glob.glob(os.path.join(p, "**", "*.wav"), recursive=True)))
    return wavs


def _metrics_for_file(wav, params):
    """Run the pipeline once and apply all four detectors. Returns a metrics dict
    (or None on pipeline failure)."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _, _, _, data = analyze_wav_file(
                wav, params, _bpm_hint_from_name(wav),
                original_file_path=wav, output_directory=tmp,
                output_options=_OO, collect_fft_for_aggregate=False,
            )
        except Exception as exc:  # noqa: BLE001
            logging.error("pipeline error on %s: %s", os.path.basename(wav), exc)
            return None
    if not data:
        return None
    labels = data.get("pass3_state_labels")
    bounds = data.get("pass3_state_boundaries") or []
    enc = data.get("pass3_state_labels_encoding")
    pc = data.get("peak_classifications") or {}
    n = 0 if labels is None else len(labels)
    try:
        info = sf.info(wav)
        sr = n / (info.frames / float(info.samplerate)) if info.frames else None
    except Exception:  # noqa: BLE001
        sr = None

    ov = find_overlapping_states(bounds, sample_rate=sr)
    cov = find_label_boundary_desync(labels, bounds, enc, sample_rate=sr) if labels is not None else []
    seq = find_sequence_violations(bounds, sample_rate=sr)
    sw = find_peak_state_mismatches(pc, bounds, sample_rate=sr)
    return {
        "overlaps_gap_rebuild": sum(1 for r in ov if r["kind"] == "gap_rebuild"),
        "overlaps_edge_paint": sum(1 for r in ov if r["kind"] == "edge_paint"),
        # Edge-of-file desync (1-2 samples at t=0/end) is a benign artefact; ignore it.
        "coverage_desync_runs": sum(1 for r in cov if r["run_samples"] > 3),
        "seq_missing_s1": sum(1 for r in seq if r["kind"] == "missing_s1"),
        "seq_bad_transition": sum(1 for r in seq if r["kind"] == "bad_transition"),
        "swap_mismatches": len(sw),
    }


_WORKER_PARAMS = None


def _worker_init(params):
    global _WORKER_PARAMS
    _WORKER_PARAMS = params
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    logging.getLogger().setLevel(logging.ERROR)


def _worker(wav):
    return wav, _metrics_for_file(wav, _WORKER_PARAMS)


def _aggregate(per_file):
    totals = {m: 0 for m in _METRICS}
    files_affected = {m: 0 for m in _METRICS}
    for m in per_file.values():
        if not m:
            continue
        for k in _METRICS:
            totals[k] += m[k]
            if m[k] > 0:
                files_affected[k] += 1
    return {"totals": totals, "files_affected": files_affected, "n_files": len(per_file)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=["inputs"], help="WAV files or dirs (default: inputs)")
    ap.add_argument("--write-baseline", action="store_true", help="Write the result as the new baseline and exit 0.")
    ap.add_argument("--baseline", default=_BASELINE, help="Baseline JSON path.")
    ap.add_argument("--json", metavar="FILE", help="Also write the full per-file result here.")
    ap.add_argument("--jobs", "-j", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="Parallel worker processes (default: CPU count - 1).")
    ns = ap.parse_args(argv)

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="%(message)s")

    wavs = _collect_wavs(ns.paths or ["inputs"])
    if not wavs:
        print("No WAV files found.", file=sys.stderr)
        return 2

    params = _params()
    jobs = max(1, int(ns.jobs))
    print(f"Checking state invariants over {len(wavs)} file(s) ({jobs} worker(s))...", flush=True)
    per_file = {}
    errors = 0
    if jobs == 1:
        for i, wav in enumerate(wavs, 1):
            per_file[os.path.relpath(wav, _REPO)] = _metrics_for_file(wav, params)
    else:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_worker_init, initargs=(params,)) as ex:
            for i, (wav, m) in enumerate(ex.map(_worker, wavs), 1):
                per_file[os.path.relpath(wav, _REPO)] = m
                if i % 50 == 0:
                    print(f"  ...{i}/{len(wavs)}", flush=True)
    errors = sum(1 for m in per_file.values() if m is None)

    agg = _aggregate(per_file)
    result = {"aggregate": agg, "errors": errors, "per_file": {k: v for k, v in per_file.items() if v}}

    print("\n=== invariant totals (files affected) ===")
    for m in _METRICS:
        print(f"  {m:22} {agg['totals'][m]:7}  ({agg['files_affected'][m]} files)")
    print(f"  {'pipeline errors':22} {errors:7}")

    if ns.json:
        with open(ns.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"Wrote {ns.json}")

    if ns.write_baseline:
        with open(ns.baseline, "w", encoding="utf-8") as fh:
            json.dump({"aggregate": agg, "errors": errors}, fh, indent=2)
        print(f"Wrote baseline {ns.baseline}")
        return 0

    if not os.path.exists(ns.baseline):
        print(f"\nNo baseline at {ns.baseline}; run with --write-baseline to create one.")
        return 0

    base = json.load(open(ns.baseline, encoding="utf-8"))["aggregate"]["totals"]
    print("\n=== vs baseline ===")
    regressed = False
    for m in _METRICS:
        cur, b = agg["totals"][m], base.get(m, 0)
        delta = cur - b
        tag = "OK" if delta <= 0 else "REGRESSION"
        if delta > 0:
            regressed = True
        sign = f"+{delta}" if delta > 0 else str(delta)
        print(f"  {m:22} {cur:7} (baseline {b:7})  {sign:>7}  {tag}")
    if regressed:
        print("\nFAIL: a structural invariant regressed against the baseline.")
        return 1
    print("\nPASS: no structural invariant regressed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
