"""
Run the real pipeline on input WAVs and report overlapping cardiac states.

Usage (from repo root):
    python debug_helpers/scan_overlaps.py                 # scan inputs/**/*.wav
    python debug_helpers/scan_overlaps.py "inputs/Difficulty 5"   # a subtree
    python debug_helpers/scan_overlaps.py path/to/one.wav         # single file
    python debug_helpers/scan_overlaps.py --json out.json inputs

For every file it runs ``analyze_wav_file`` (no output artifacts written), pulls
``analysis_data["pass3_state_boundaries"]``, and runs the overlap detector. Files
with overlaps are printed with the worst offending segment pairs; a JSON dump of
all overlap records is written when --json is given.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Allow running as `python debug_helpers/scan_overlaps.py` from repo root.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from debug_helpers._common import (  # noqa: E402
    collect_wavs, default_jobs, parallel_scan, params, reconfigure_stdio, run_pipeline,
)
from debug_helpers.overlap_detector import find_overlapping_states, summarize  # noqa: E402


def scan_file(wav_path, run_params):
    """Return (sample_rate, records) or (None, None) on pipeline failure."""
    data = run_pipeline(wav_path, run_params)
    if data is None:
        return None, None
    sr = int(data.get("sample_rate") or 0) or None
    bounds = data.get("pass3_state_boundaries") or []
    return sr, find_overlapping_states(bounds, sample_rate=sr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", default=["inputs"], help="WAV files or directories (default: inputs)")
    ap.add_argument("--json", metavar="FILE", help="Write all overlap records to this JSON file.")
    ap.add_argument("--max-pairs", type=int, default=3, help="Offending pairs printed per file.")
    ap.add_argument("--jobs", "-j", type=int, default=default_jobs(),
                    help="Parallel worker processes (default: CPU count - 1).")
    ns = ap.parse_args(argv)

    reconfigure_stdio()
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="%(message)s")

    wavs = collect_wavs(ns.paths or ["inputs"])
    if not wavs:
        print("No WAV files found.", file=sys.stderr)
        return 2

    run_params = params()
    all_results = {}
    n_with_overlap = 0
    grand_total = 0
    grand_gap_rebuild = 0
    grand_edge_paint = 0

    jobs = max(1, int(ns.jobs))
    print(f"Scanning {len(wavs)} file(s) for overlapping cardiac states "
          f"({jobs} worker process(es))...\n", flush=True)

    def _handle(idx, wav, sr, records):
        nonlocal n_with_overlap, grand_total, grand_gap_rebuild, grand_edge_paint
        rel = os.path.relpath(wav, _REPO)
        if idx % 25 == 0 or idx == len(wavs):
            print(f"  ...progress {idx}/{len(wavs)} ({n_with_overlap} flagged so far)", flush=True)
        if records is None:
            print(f"  ERROR   {rel}", flush=True)
            return
        if not records:
            return
        n_with_overlap += 1
        grand_total += len(records)
        s = summarize(records)
        grand_gap_rebuild += s["gap_rebuild_overlaps"]
        grand_edge_paint += s["edge_paint_overlaps"]
        all_results[rel] = {"summary": s, "records": records}
        print(f"  OVERLAP {rel}")
        print(f"          {s['total_overlaps']} overlap(s): "
              f"{s['gap_rebuild_overlaps']} gap-rebuild, "
              f"{s['edge_paint_overlaps']} edge-paint; "
              f"worst {s['worst_overlap_samples']} samples; pairs={s['overlap_state_pairs']}")
        worst = sorted(records, key=lambda r: r["overlap_samples"], reverse=True)[: ns.max_pairs]
        for r in worst:
            ta = r.get("overlap_t_lo_sec")
            tloc = f"@{ta:.2f}s " if ta is not None else ""
            print(f"            {tloc}{r['seg_a'][2]}[{r['seg_a'][0]}:{r['seg_a'][1]} "
                  f"src={r['seg_a'][3] or '-'}]  x  "
                  f"{r['seg_b'][2]}[{r['seg_b'][0]}:{r['seg_b'][1]} src={r['seg_b'][3] or '-'}]  "
                  f"overlap={r['overlap_samples']}")
        print(flush=True)

    for idx, (wav, (sr, records)) in enumerate(parallel_scan(wavs, scan_file, run_params, jobs), 1):
        _handle(idx, wav, sr, records)

    print("=" * 60)
    print(f"Files scanned : {len(wavs)}")
    print(f"With overlaps : {n_with_overlap}")
    print(f"Total overlaps: {grand_total}  (gap-rebuild={grand_gap_rebuild}, edge-paint={grand_edge_paint})")

    if ns.json:
        with open(ns.json, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"Wrote {ns.json}")

    return 1 if n_with_overlap else 0


if __name__ == "__main__":
    raise SystemExit(main())
