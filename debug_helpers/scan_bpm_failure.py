"""
Run the real pipeline on input WAVs and report BPM-plausibility-gate failures
(hrv.detect_bpm_failure, wired in pipeline.py STAGE 6). Used to validate the
bpm_min_physiological / bpm_jump_ratio_threshold / bpm_coverage_gap_sec /
bpm_trailing_coverage_frac defaults in config.py against the real corpus before
trusting them, the same way state_invariants.py validates structural invariants.

Usage (from repo root):
    python debug_helpers/scan_bpm_failure.py                 # scan inputs/**/*.wav
    python debug_helpers/scan_bpm_failure.py "inputs/Difficulty 3"
    python debug_helpers/scan_bpm_failure.py inputs --json out.json

Exit code is 1 when any file is flagged, else 0.
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
    bpm_hint_from_name, collect_wavs, default_jobs, parallel_scan,
    params, reconfigure_stdio, run_pipeline,
)


def scan_file(wav_path, run_params):
    """Return the bpm_failure_report dict, or None on pipeline failure."""
    data = run_pipeline(wav_path, run_params, bpm_hint=bpm_hint_from_name(wav_path))
    if data is None:
        return None
    return data.get("bpm_failure_report") or {"failed": False, "reasons": [], "metrics": {}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", default=["inputs"], help="WAV files or directories (default: inputs)")
    ap.add_argument("--json", metavar="FILE", help="Write all flagged records to this JSON file.")
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
    n_errors = 0

    jobs = max(1, int(ns.jobs))
    print(f"Scanning {len(wavs)} file(s) for BPM plausibility-gate failures "
          f"({jobs} worker process(es))...\n", flush=True)

    for idx, (wav, report) in enumerate(parallel_scan(wavs, scan_file, run_params, jobs), 1):
        rel = os.path.relpath(wav, _REPO)
        if idx % 25 == 0 or idx == len(wavs):
            print(f"  ...progress {idx}/{len(wavs)} ({n_flagged} flagged so far)", flush=True)
        if report is None:
            n_errors += 1
            print(f"  ERROR   {rel}", flush=True)
            continue
        if not report.get("failed"):
            continue
        n_flagged += 1
        all_results[rel] = report
        print(f"  FLAGGED {rel}")
        for reason in report.get("reasons") or []:
            print(f"            {reason}")
        print(flush=True)

    print("=" * 60)
    print(f"Files scanned : {len(wavs)}")
    print(f"Flagged       : {n_flagged}")
    print(f"Errors        : {n_errors}")

    if ns.json:
        with open(ns.json, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"Wrote {ns.json}")

    return 1 if n_flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
