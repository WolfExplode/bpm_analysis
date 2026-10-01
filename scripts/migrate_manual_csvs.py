"""One-time migration: `<recording>.wav_manual_state_sequence.csv` -> `<recording>.annotation.json`.

Keeps S1 / S2 / noisy spans (systole, diastole and bpm_at_mid are derived, so dropped),
resolves overlaps by cutting at the midpoint, fingerprints each recording, and writes a
CSV report of every cut for spot-checking. Converted Annotations are treated as complete.
The original CSVs are left untouched.

    python scripts/migrate_manual_csvs.py                 # dry run over inputs/
    python scripts/migrate_manual_csvs.py --write         # write the Annotations + report
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import soundfile as sf  # noqa: E402

from pcg import annotation as an  # noqa: E402
from pcg import batch, recording  # noqa: E402


# Long names were truncated on disk ("_manual_state_sequ.csv"), so match any "_manual_state_se*.csv".
SUFFIX_RE = re.compile(r"_manual_state_se[a-z]*\.csv$")
ORIGINS = {"manual": an.ORIGIN_HAND}  # auto / regenerated -> algorithm


def find_audio(csv_path: Path):
    """The recording a CSV belongs to: same name, or (after a BPM rename) the one audio file
    in the folder whose name matches once BPM tags are stripped and whitespace collapsed."""
    wav = csv_path.with_name(SUFFIX_RE.sub("", csv_path.name))
    if wav.is_file():
        return wav
    def key(stem: str) -> str:  # whitespace-insensitive (double / full-width spaces differ between copies)
        return " ".join(batch.strip_bpm_tags(stem).split())

    want = key(Path(wav.name).stem)
    hits = [p for p in csv_path.parent.iterdir() if recording.is_audio_file(p) and key(p.stem) == want]
    return hits[0] if len(hits) == 1 else None


def read_rows(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            state = (r.get("state") or "").strip()
            kind = {"S1": an.S1, "S2": an.S2, "noisy": an.NOISY}.get(state)
            if kind is None:
                continue
            try:
                a, b = float(r["start_sec"]), float(r["end_sec"])
            except (KeyError, ValueError):
                continue
            rows.append((a, b, kind, ORIGINS.get((r.get("source") or "").strip(), an.ORIGIN_ALGORITHM)))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=[str(REPO / "inputs")])
    ap.add_argument("--write", action="store_true", help="write Annotations (default: dry run)")
    ap.add_argument("--overwrite", action="store_true", help="replace existing .annotation.json files")
    ap.add_argument("--report", default=str(REPO / "debug_helpers" / "manual_csv_migration_report.csv"))
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    csvs = sorted(p for root in args.roots for p in Path(root).rglob("*_manual_state_se*.csv") if SUFFIX_RE.search(p.name))
    report_rows = []
    written = skipped = 0
    for csv_path in csvs:
        wav = find_audio(csv_path)
        if wav is None:
            print(f"MISSING AUDIO  {csv_path.name}")
            skipped += 1
            continue
        rows = read_rows(csv_path)
        clips: list = []
        spans = an.from_spans(rows, clips)
        info = sf.info(str(wav))
        ann = an.Annotation(recording.fingerprint(wav), float(info.duration), wav.name, spans)
        for c in clips:
            report_rows.append({
                "recording": wav.name, "at_sec": f"{c['at']:.3f}", "overlap_ms": f"{c['overlap_sec'] * 1000:.1f}",
                "first": c["first"], "first_span": "%.3f-%.3f" % c["first_span"],
                "second": c["second"], "second_span": "%.3f-%.3f" % c["second_span"],
            })
        kinds = {k: sum(1 for s in spans if s.kind == k) for k in an.KINDS}
        target = an.sidecar_path(wav)
        status = "dry-run"
        if args.write:
            if target.exists() and not args.overwrite:
                status = "exists, skipped"
                skipped += 1
            else:
                an.save(ann, target)
                status = "written"
                written += 1
        print(f"{status:16} {wav.name}: {kinds}  {len(clips)} clips")

    pairs: dict = {}
    for r in report_rows:
        key = "/".join(sorted((r["first"], r["second"])))
        pairs[key] = pairs.get(key, 0) + 1
    print(f"\n{len(csvs)} CSVs, {len(report_rows)} clips {pairs}; written {written}, skipped {skipped}")
    if args.write:
        with open(args.report, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["recording", "at_sec", "overlap_ms", "first", "first_span",
                                              "second", "second_span"])
            w.writeheader()
            w.writerows(report_rows)
        print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
