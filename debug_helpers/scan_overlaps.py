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
import glob
import json
import logging
import os
import sys
import tempfile

# Allow running as `python debug_helpers/scan_overlaps.py` from repo root.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from config import DEFAULT_PARAMS  # noqa: E402
from pipeline import analyze_wav_file  # noqa: E402
from debug_helpers.overlap_detector import find_overlapping_states, summarize  # noqa: E402

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


def scan_file(wav_path, params):
    """Return (sample_rate, records) or (None, None) on pipeline failure."""
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
            return None, None
    if not data:
        return None, None
    sr = int(data.get("sample_rate") or 0) or None
    bounds = data.get("pass3_state_boundaries") or []
    records = find_overlapping_states(bounds, sample_rate=sr)
    return sr, records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", default=["inputs"], help="WAV files or directories (default: inputs)")
    ap.add_argument("--json", metavar="FILE", help="Write all overlap records to this JSON file.")
    ap.add_argument("--max-pairs", type=int, default=3, help="Offending pairs printed per file.")
    ns = ap.parse_args(argv)

    # Input filenames contain CJK/emoji; force UTF-8 so prints survive a redirect
    # to a cp1252 file on Windows (otherwise main() dies on the first such name).
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
    n_with_overlap = 0
    grand_total = 0
    grand_gap_rebuild = 0
    grand_edge_paint = 0

    print(f"Scanning {len(wavs)} file(s) for overlapping cardiac states...\n", flush=True)
    for idx, wav in enumerate(wavs, 1):
        rel = os.path.relpath(wav, _REPO)
        if idx % 25 == 0 or idx == len(wavs):
            print(f"  ...progress {idx}/{len(wavs)} ({n_with_overlap} flagged so far)", flush=True)
        sr, records = scan_file(wav, params)
        if records is None:
            print(f"  ERROR   {rel}")
            continue
        if not records:
            continue
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
