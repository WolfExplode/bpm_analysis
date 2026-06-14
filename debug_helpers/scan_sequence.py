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
from debug_helpers.state_sequence_detector import find_sequence_violations, summarize  # noqa: E402

_OUTPUT_OPTIONS = {
    "html": False, "png": False, "csv": False, "summary": False, "debug": False,
    "filtered_wav": False, "spectrogram": False, "fft_profiles": False,
    "output_all_passes": False, "working_wav_in_output": False,
}


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


def _bpm_hint_from_name(path):
    """Parse a leading starting BPM from a "[107,71-108bpm]" style name, else None.
    Mirrors the GUI/batch default (bpm_from_filename) so results match real runs."""
    import re
    m = re.search(r"\[(\d+(?:\.\d+)?)\s*,", os.path.basename(path))
    return float(m.group(1)) if m else None


def _env_sample_rate(wav_path, n_samples):
    """Envelope sample rate = analysed samples / wall-clock duration of the WAV."""
    try:
        info = sf.info(wav_path)
        dur = info.frames / float(info.samplerate)
        return (n_samples / dur) if dur > 0 else None
    except Exception:  # noqa: BLE001
        return None


def scan_file(wav_path, params):
    """Return (sample_rate, records) or (None, None) on pipeline failure."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _, _, _, data = analyze_wav_file(
                wav_path, params, _bpm_hint_from_name(wav_path),
                original_file_path=wav_path,
                output_directory=tmp,
                output_options=_OUTPUT_OPTIONS,
                collect_fft_for_aggregate=False,
            )
        except Exception as exc:  # noqa: BLE001
            logging.error("pipeline error on %s: %s", os.path.basename(wav_path), exc)
            return None, None
    if not data:
        return None, None
    bounds = data.get("pass3_state_boundaries") or []
    _labels = data.get("pass3_state_labels")
    n = 0 if _labels is None else len(_labels)
    sr = _env_sample_rate(wav_path, n)
    records = find_sequence_violations(bounds, sample_rate=sr)
    return sr, records


# Module-level worker so it is picklable for the process pool (Windows spawn).
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
    ap.add_argument("--json", metavar="FILE", help="Write all violation records to this JSON file.")
    ap.add_argument("--max-show", type=int, default=4, help="Violations printed per file.")
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

    if jobs == 1:
        for idx, wav in enumerate(wavs, 1):
            sr, records = scan_file(wav, params)
            _handle(idx, wav, sr, records)
    else:
        # ProcessPoolExecutor: analyze_wav_file is CPU-bound, so processes (not
        # threads) give real speedup. Results stream back in submission order.
        with ProcessPoolExecutor(max_workers=jobs, initializer=_worker_init, initargs=(params,)) as ex:
            for idx, (wav, sr, records) in enumerate(ex.map(_worker, wavs), 1):
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
