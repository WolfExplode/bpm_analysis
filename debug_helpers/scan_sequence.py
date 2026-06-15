"""
Run the real pipeline on input WAVs and report cardiac state-sequence violations.

A correct timeline cycles S1 -> systole -> S2 -> diastole. This scanner flags any
boundary list that breaks the cycle, in particular ``diastole -> S2`` (an S2 with
no S1 before it — the "missing S1 state" bug).

Usage (from repo root):
    python debug_helpers/scan_sequence.py                 # scan inputs/**/*.wav
    python debug_helpers/scan_sequence.py "inputs/Difficulty 3"
    python debug_helpers/scan_sequence.py path/to/one.wav
    python debug_helpers/scan_sequence.py inputs --json out.json

Exit code is 1 when any file has violations, else 0.
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
from debug_helpers.state_sequence_detector import find_sequence_violations, summarize  # noqa: E402

# Back-compat alias: benchmarking/state_invariants.py and other helpers import
# this name from here.
_bpm_hint_from_name = bpm_hint_from_name


def scan_file(wav_path, run_params):
    """Return (sample_rate, records) or (None, None) on pipeline failure."""
    data = run_pipeline(wav_path, run_params, bpm_hint=bpm_hint_from_name(wav_path))
    if data is None:
        return None, None
    bounds = data.get("pass3_state_boundaries") or []
    labels = data.get("pass3_state_labels")
    n = 0 if labels is None else len(labels)
    sr = env_sample_rate(wav_path, n)
    return sr, find_sequence_violations(bounds, sample_rate=sr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", default=["inputs"], help="WAV files or directories (default: inputs)")
    ap.add_argument("--json", metavar="FILE", help="Write all violation records to this JSON file.")
    ap.add_argument("--max-show", type=int, default=4, help="Violations printed per file.")
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
    grand_missing_s1 = 0

    jobs = max(1, int(ns.jobs))
    print(f"Scanning {len(wavs)} file(s) for state-sequence violations "
          f"({jobs} worker process(es))...\n", flush=True)

    def _handle(idx, wav, sr, records):
        nonlocal n_flagged, grand_total, grand_missing_s1
        rel = os.path.relpath(wav, _REPO)
        if idx % 25 == 0 or idx == len(wavs):
            print(f"  ...progress {idx}/{len(wavs)} ({n_flagged} flagged so far)", flush=True)
        if records is None:
            print(f"  ERROR   {rel}", flush=True)
            return
        if not records:
            return
        n_flagged += 1
        grand_total += len(records)
        s = summarize(records)
        grand_missing_s1 += s["missing_s1"]
        all_results[rel] = {"summary": s, "records": records}
        print(f"  VIOLATION {rel}")
        print(f"          {s['total_violations']} violation(s): "
              f"{s['missing_s1']} missing-S1; transitions={s['by_transition']}", flush=True)
        for r in records[: ns.max_show]:
            ts = f"@{r['at_sec']:.2f}s " if "at_sec" in r else ""
            print(f"            {ts}{r['prev_state']}->{r['cur_state']} "
                  f"(expected {r['expected']}) [{r['kind']}]", flush=True)
        print(flush=True)

    for idx, (wav, (sr, records)) in enumerate(parallel_scan(wavs, scan_file, run_params, jobs), 1):
        _handle(idx, wav, sr, records)

    print("=" * 60)
    print(f"Files scanned   : {len(wavs)}")
    print(f"With violations : {n_flagged}")
    print(f"Total violations: {grand_total}  (missing-S1={grand_missing_s1})")

    if ns.json:
        with open(ns.json, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"Wrote {ns.json}")

    return 1 if n_flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
