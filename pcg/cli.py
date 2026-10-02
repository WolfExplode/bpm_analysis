"""Command line: `python -m pcg <command>`.

    analyze  PATHS...           run the engine, store Analyses in the library
    inspect  RECORDING --from S --to S   text dump of a window (same as Ctrl+Shift+C)
    export   RECORDING ...      BPM CSV / BPM chart (from the Analysis or the Annotation), summary
    rename   PATHS...           write each recording's BPM into its filename
    app      [PATHS...] [--analyze]  open the workspace (the default with no command)

For breakpoint debugging of the engine, run `analyze` under a debugger.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from pcg import analysis, annotation, batch, context, recording
from pcg.clock import clock_text


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _library(args) -> analysis.Library:
    return analysis.Library(args.library)


def open_latest(path: str, lib: analysis.Library, channel: Optional[str] = None) -> analysis.Analysis:
    fp = lib.fingerprints.get(path)
    found = lib.latest(fp, channel)
    if found is None:
        raise SystemExit(f"no Analysis for {Path(path).name} — run `python -m pcg analyze` first")
    return lib.open(found)


# ─────────────────────────────────────────────────────────────────────────────

def cmd_analyze(args) -> int:
    lib = _library(args)
    paths = batch.collect(args.paths)
    if not paths:
        print("no audio files found", file=sys.stderr)
        return 1
    settings = batch.RunSettings(springer=args.springer, auto_switch=args.auto_switch,
                                 channel=args.channel, jobs=args.jobs)
    jobs = [batch.Job(str(p), args.bpm, args.start) for p in paths]
    if settings.effective_jobs < settings.jobs:
        print("note: Springer/auto-switch runs one recording at a time (memory)", file=sys.stderr)

    def on_progress(path: str, msg: str) -> None:
        if args.progress_json:
            print(json.dumps({"progress": msg}), flush=True)
        elif args.verbose:
            print(f"  {Path(path).name}: {msg}", flush=True)

    failed = 0

    def on_done(r: batch.JobResult) -> None:
        nonlocal failed
        name = Path(r.path).name
        if args.progress_json:
            print(json.dumps({"done": r.analyses, "error": r.error}), flush=True)
        if not r.ok:
            failed += 1
            print(f"FAILED  {name}: {r.error}", flush=True)
            return
        for bpm, gate_failed, reasons in zip(r.bpm, r.gate_failed, r.gate_reasons):
            rng = f"{bpm['start_bpm']:.0f}, {bpm['min_bpm']:.0f}-{bpm['max_bpm']:.0f} BPM" if bpm else "no BPM"
            gate = f"  GATE: {'; '.join(reasons)}" if gate_failed else ""
            if not args.progress_json:
                print(f"ok      {name}: {rng}{gate}", flush=True)
        if args.rename and len(r.bpm) == 1:
            new, why = batch.rename_with_bpm(r.path, r.bpm[0])
            if not args.progress_json:
                print(f"        renamed -> {new.name}" if new else f"        not renamed: {why}", flush=True)

    batch.run(jobs, settings, lib, on_progress=on_progress, on_done=on_done)
    return 1 if failed else 0


def cmd_inspect(args) -> int:
    lib = _library(args)
    a = open_latest(args.recording, lib, args.channel)
    ann = None
    dis = None
    found = annotation.find_for_recording(args.recording, a.fingerprint)
    if found:
        ann = annotation.load(found)
        algo = annotation.from_states(a.states.start, a.states.end, a.states.state)
        dis = annotation.disagreements(ann.spans, algo)
    traces = args.traces.split(",") if args.traces else None
    sys.stdout.write(context.window_text(
        a, args.t_from, args.t_to, recording_path=args.recording, visible_traces=traces,
        annotation=ann, disagreements=dis,
    ))
    return 0


def write_bpm_csv(path: Path, times, bpm, header: str, time_format: str = "seconds") -> None:
    """*time_format*: "seconds", "clock" (hh:mm:ss.xxx) or "both"."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow({"seconds": ["time_sec"], "clock": ["time"], "both": ["time_sec", "time"]}[time_format] + [header])
        for t, b in zip(times, bpm):
            if b == b:  # skip NaN
                stamps = {"seconds": [f"{t:.3f}"], "clock": [clock_text(t)], "both": [f"{t:.3f}", clock_text(t)]}
                w.writerow(stamps[time_format] + [f"{b:.3f}"])


def summary_text(a: analysis.Analysis) -> str:
    s = a.summary
    bpm = s.get("bpm") or {}
    lines = [
        f"Recording: {a.recording_name}",
        f"Fingerprint: {a.fingerprint} ({a.channel})",
        f"Algorithm: {a.algorithm_used}" + (f" — {a.algorithm_switch_reason}" if a.algorithm_switch_reason else ""),
        f"Duration: {a.duration_sec:.1f} s",
    ]
    if bpm:
        lines.append(f"BPM: start {bpm['start_bpm']:.1f}, range {bpm['min_bpm']:.1f}–{bpm['max_bpm']:.1f}")
    for k, v in (s.get("hrv") or {}).items():
        lines.append(f"{k}: {v:.4g}")
    gate = s.get("gate") or {}
    lines.append("Plausibility gate: " + ("FAILED — " + "; ".join(gate.get("reasons") or []) if gate.get("failed") else "passed"))
    lines.append(f"Defects: {len(a.defects)}")
    return "\n".join(lines) + "\n"


def _bpm_series(args, a: analysis.Analysis):
    """(times, BPM) from the Annotation next to the recording or from the Analysis, per --source."""
    if args.source == "annotation":
        found = annotation.find_for_recording(args.recording, a.fingerprint)
        if not found:
            raise SystemExit("no Annotation found next to the recording")
        return annotation.bpm_series(annotation.load(found).spans)
    tr = a.trace("BPM")
    if tr is None:
        raise SystemExit("the Analysis has no BPM curve")
    return tr.times(), tr.y


def cmd_export(args) -> int:
    lib = _library(args)
    a = open_latest(args.recording, lib, args.channel)
    if args.bpm_csv:
        t, b = _bpm_series(args, a)
        write_bpm_csv(Path(args.bpm_csv), t, b, "bpm_annotation" if args.source == "annotation" else "bpm")
        print(f"wrote {args.bpm_csv}")
    if args.summary:
        Path(args.summary).write_text(summary_text(a), encoding="utf-8")
        print(f"wrote {args.summary}")
    if args.chart:
        from pcg.app.chart import annotation_bpm_curve, render_bpm_chart

        if args.source == "annotation":
            found = annotation.find_for_recording(args.recording, a.fingerprint)
            if not found:
                raise SystemExit("no Annotation found next to the recording")
            t, b = annotation_bpm_curve(annotation.load(found).spans, a.params)
        else:
            t, b = _bpm_series(args, a)
        env = a.trace("Algorithm envelope")
        if env is None:
            raise SystemExit("the Analysis has no envelope")
        if not render_bpm_chart(t, b, env.t0, env.dt, env.y, 0.0, a.duration_sec).save(args.chart):
            raise SystemExit(f"could not write {args.chart}")
        print(f"wrote {args.chart}")
    return 0


def cmd_rename(args) -> int:
    lib = _library(args)
    for p in batch.collect(args.paths):
        fp = lib.fingerprints.get(p)
        found = lib.latest(fp, recording.CHANNEL_MIXED) or lib.latest(fp)
        if not found:
            print(f"skip    {p.name}: not analysed")
            continue
        new, why = batch.rename_with_bpm(p, analysis.load(found).summary.get("bpm"))
        print(f"renamed {p.name} -> {new.name}" if new else f"skip    {p.name}: {why}")
    return 0


def cmd_app(args) -> int:
    from pcg.app.main import main as app_main

    return app_main(list(args.paths), library=args.library, analyze=args.analyze)


# ─────────────────────────────────────────────────────────────────────────────

COMMANDS = ("analyze", "inspect", "export", "rename", "app")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m pcg", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", default=None, help="Analysis library folder (default: ./library or $PCG_LIBRARY)")
    sub = ap.add_subparsers(dest="command")

    p = sub.add_parser("analyze", help="run the engine on recordings")
    p.add_argument("paths", nargs="+")
    p.add_argument("--springer", action="store_true", help="use the Springer 2015 HSMM instead of the native passes")
    p.add_argument("--auto-switch", action="store_true", help="retry with the other algorithm when the gate fails")
    p.add_argument("--channel", choices=recording.CHANNEL_MODES, default=recording.CHANNEL_MIXED)
    p.add_argument("--start", type=float, default=0.0, help="skip this many seconds at the start")
    p.add_argument("--bpm", type=float, default=None, help="start BPM hint (default: from the filename tag)")
    p.add_argument("-j", "--jobs", type=int, default=1)
    p.add_argument("--rename", action="store_true", help="write the BPM into each input filename")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--progress-json", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("inspect", help="text dump of a time window")
    p.add_argument("recording")
    p.add_argument("--from", dest="t_from", type=float, required=True)
    p.add_argument("--to", dest="t_to", type=float, required=True)
    p.add_argument("--channel", default=None)
    p.add_argument("--traces", default=None, help="comma-separated trace names (default: signal/BPM lines)")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("export", help="export BPM CSV / summary / BPM chart")
    p.add_argument("recording")
    p.add_argument("--bpm-csv")
    p.add_argument("--source", choices=("analysis", "annotation"), default="analysis")
    p.add_argument("--summary")
    p.add_argument("--chart", help="PNG: the BPM curve (0-230 BPM) over the waveform, whole recording")
    p.add_argument("--channel", default=None)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("rename", help="write BPM tags into filenames from existing Analyses")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_rename)

    p = sub.add_parser("app", help="open the workspace")
    p.add_argument("paths", nargs="*")
    p.add_argument("--analyze", action="store_true",
                   help="start analysing the given recordings right away (one file: in the workspace)")
    p.set_defaults(func=cmd_app)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    _utf8_stdio()
    argv = list(sys.argv[1:] if argv is None else argv)
    head = argv[:2] if argv[:1] == ["--library"] else []  # global option stays in front
    rest = argv[len(head):]
    if not rest or (rest[0] not in COMMANDS and rest[0] not in ("-h", "--help")):
        argv = head + ["app"] + rest  # no command given: open the app
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
