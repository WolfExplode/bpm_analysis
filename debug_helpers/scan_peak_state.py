"""
Run the pipeline on input WAVs and report peak/state label mismatches —
S1 peaks sitting under S2 state bands (and vice-versa), i.e. regional S1<->S2
label swaps (the "missing S1 state" bug).

Usage (from repo root):
    python debug_helpers/scan_peak_state.py                 # scan inputs/**/*.wav
    python debug_helpers/scan_peak_state.py "inputs/Difficulty 3"
    python debug_helpers/scan_peak_state.py path/to/one.wav
    python debug_helpers/scan_peak_state.py inputs --json out.json -j 8

Exit code is 1 when any file has mismatches, else 0. Runs a process pool
(--jobs, default CPU - 1) since analyze_wav_file is CPU-bound.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from debug_helpers._common import (  # noqa: E402
    bpm_hint_from_name, collect_wavs, default_jobs, env_sample_rate,
    parallel_scan, params, reconfigure_stdio, run_pipeline,
)
from debug_helpers.peak_state_mismatch_detector import (  # noqa: E402
    find_peak_state_mismatches,
    summarize,
)


def scan_file(wav_path, run_params):
    """Return (sample_rate, records) or (None, None) on pipeline failure."""
    data = run_pipeline(wav_path, run_params, bpm_hint=bpm_hint_from_name(wav_path))
    if data is None:
        return None, None
    labels = data.get("pass3_state_labels")
    n = 0 if labels is None else len(labels)
    sr = env_sample_rate(wav_path, n)
    records = find_peak_state_mismatches(
        data.get("peak_classifications") or {}, data.get("pass3_state_boundaries") or [],
        sample_rate=sr,
    )
    return sr, records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", default=["inputs"], help="WAV files or directories (default: inputs)")
    ap.add_argument("--json", metavar="FILE", help="Write all mismatch records to this JSON file.")
    ap.add_argument("--max-show", type=int, default=4, help="Mismatches printed per file.")
    ap.add_argument("--min-mismatches", type=int, default=1, help="Only flag files with >= this many.")
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
    n_flagged = 0
    grand_total = 0

    jobs = max(1, int(ns.jobs))
    print(f"Scanning {len(wavs)} file(s) for S1<->S2 peak/state swaps "
          f"({jobs} worker process(es))...\n", flush=True)

    def _handle(idx, wav, sr, records):
        nonlocal n_flagged, grand_total
        rel = os.path.relpath(wav, _REPO)
        if idx % 25 == 0 or idx == len(wavs):
            print(f"  ...progress {idx}/{len(wavs)} ({n_flagged} flagged so far)", flush=True)
        if records is None:
            print(f"  ERROR   {rel}", flush=True)
            return
        if len(records) < ns.min_mismatches:
            return
        n_flagged += 1
        grand_total += len(records)
        s = summarize(records)
        all_results[rel] = {"summary": s, "records": records}
        print(f"  SWAP {rel}")
        print(f"          {s['total_mismatches']} mismatch(es): "
              f"{s['s1_peak_under_s2_band']} S1-under-S2, "
              f"{s['s2_peak_under_s1_band']} S2-under-S1", flush=True)
        for r in records[: ns.max_show]:
            ts = f"@{r['peak_sec']:.2f}s " if "peak_sec" in r else ""
            print(f"            {ts}peak {r['peak']} {r['peak_kind']} "
                  f"under {r['state']} band {r['state_span']}", flush=True)
        print(flush=True)

    for idx, (wav, (sr, records)) in enumerate(parallel_scan(wavs, scan_file, run_params, jobs), 1):
        _handle(idx, wav, sr, records)

    print("=" * 60)
    print(f"Files scanned : {len(wavs)}")
    print(f"With swaps    : {n_flagged}")
    print(f"Total swaps   : {grand_total}")

    if ns.json:
        with open(ns.json, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"Wrote {ns.json}")

    return 1 if n_flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
