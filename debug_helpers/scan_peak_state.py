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
import glob
import json
import logging
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import soundfile as sf  # noqa: E402

from config import DEFAULT_PARAMS  # noqa: E402
from pipeline import analyze_wav_file  # noqa: E402
from debug_helpers.scan_sequence import _bpm_hint_from_name  # noqa: E402
from debug_helpers.peak_state_mismatch_detector import (  # noqa: E402
    find_peak_state_mismatches,
    summarize,
)

_OO = {k: False for k in (
    "html", "png", "csv", "summary", "debug", "filtered_wav",
    "spectrogram", "fft_profiles", "output_all_passes", "working_wav_in_output",
)}


def _collect_wavs(paths):
    wavs = []
    for p in paths:
        if os.path.isfile(p) and p.lower().endswith(".wav"):
            wavs.append(p)
        elif os.path.isdir(p):
            wavs.extend(sorted(glob.glob(os.path.join(p, "**", "*.wav"), recursive=True)))
    return wavs


def _params():
    return {**DEFAULT_PARAMS, "save_filtered_wav": False, "enable_fft_profiles": False}


def scan_file(wav_path, params):
    """Return (sample_rate, records) or (None, None) on pipeline failure."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _, _, _, data = analyze_wav_file(
                wav_path, params, _bpm_hint_from_name(wav_path),
                original_file_path=wav_path,
                output_directory=tmp,
                output_options=_OO,
                collect_fft_for_aggregate=False,
            )
        except Exception as exc:  # noqa: BLE001
            logging.error("pipeline error on %s: %s", os.path.basename(wav_path), exc)
            return None, None
    if not data:
        return None, None
    _labels = data.get("pass3_state_labels")
    n = 0 if _labels is None else len(_labels)
    try:
        info = sf.info(wav_path)
        sr = n / (info.frames / float(info.samplerate)) if info.frames else None
    except Exception:  # noqa: BLE001
        sr = None
    records = find_peak_state_mismatches(
        data.get("peak_classifications") or {}, data.get("pass3_state_boundaries") or [],
        sample_rate=sr,
    )
    return sr, records


# Module-level worker for the process pool (Windows spawn).
_WORKER_PARAMS = None


def _worker_init(params):
    global _WORKER_PARAMS
    _WORKER_PARAMS = params
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    logging.getLogger().setLevel(logging.ERROR)


def _worker(wav):
    sr, records = scan_file(wav, _WORKER_PARAMS)
    return wav, sr, records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", default=["inputs"], help="WAV files or directories (default: inputs)")
    ap.add_argument("--json", metavar="FILE", help="Write all mismatch records to this JSON file.")
    ap.add_argument("--max-show", type=int, default=4, help="Mismatches printed per file.")
    ap.add_argument("--min-mismatches", type=int, default=1, help="Only flag files with >= this many.")
    ap.add_argument("--jobs", "-j", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="Parallel worker processes (default: CPU count - 1).")
    ns = ap.parse_args(argv)

    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="%(message)s")

    wavs = _collect_wavs(ns.paths or ["inputs"])
    if not wavs:
        print("No WAV files found.", file=sys.stderr)
        return 2

    params = _params()
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

    if jobs == 1:
        for idx, wav in enumerate(wavs, 1):
            sr, records = scan_file(wav, params)
            _handle(idx, wav, sr, records)
    else:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_worker_init, initargs=(params,)) as ex:
            for idx, (wav, sr, records) in enumerate(ex.map(_worker, wavs), 1):
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
