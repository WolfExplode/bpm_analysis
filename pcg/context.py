"""Plain-text dump of a time window, for pasting into an LLM conversation.

Used by the workspace (Ctrl+Shift+C, with the traces currently visible) and by
`python -m pcg inspect` (all line traces of the signal/BPM lanes).
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np

from pcg import annotation as ann_mod
from pcg.analysis import Analysis
from pcg.engine.traces import KIND_LINE, KIND_POINTS, KIND_SPANS, Trace

BELIEF_TRACES = ("BPM belief (Pass 2)", "BPM prior (Pass 3)", "BPM")


def _f(v: Optional[float], nd: int = 3) -> str:
    return "-" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def default_inspect_traces(a: Analysis) -> List[str]:
    return [t.name for t in a.traces if t.lane in ("signal", "bpm") and t.kind == KIND_LINE]


def window_text(
    a: Analysis,
    t0: float,
    t1: float,
    *,
    recording_path: str = "",
    visible_traces: Optional[Iterable[str]] = None,
    annotation: Optional[ann_mod.Annotation] = None,
    disagreements: Optional[Sequence[ann_mod.Disagreement]] = None,
) -> str:
    names = list(visible_traces) if visible_traces is not None else default_inspect_traces(a)
    traces: List[Trace] = [t for t in (a.trace(n) for n in names) if t is not None]
    lines: List[str] = []
    w = lines.append

    w(f"# Window {t0:.3f}s – {t1:.3f}s ({t1 - t0:.3f}s)")
    w(f"recording: {recording_path or a.recording_name}")
    w(f"fingerprint: {a.fingerprint}  channel: {a.channel}")
    w(f"algorithm: {a.algorithm_used}" + (f"  (auto-switch: {a.algorithm_switch_reason})" if a.algorithm_switch_reason else ""))
    w(f"run settings: {a.run_settings}  start BPM hint: {a.start_bpm_hint}")
    w("analysis: " + ("STALE — " + "; ".join(a.stale_reasons) if a.stale_reasons else "current") + f"  (created {a.created})")
    gate = a.summary.get("gate") or {}
    if gate.get("failed"):
        w("plausibility gate: FAILED — " + "; ".join(gate.get("reasons") or []))

    w("")
    w("## BPM belief at window edges")
    for name in BELIEF_TRACES:
        t = a.trace(name)
        if t is not None:
            w(f"{name}: {_f(t.value_at(t0), 1)} → {_f(t.value_at(t1), 1)}")

    p = a.peaks
    idx = np.nonzero((p.time >= t0) & (p.time <= t1))[0]
    w("")
    w(f"## Peaks ({len(idx)})")
    line_traces = [t for t in traces if t.kind == KIND_LINE]
    if line_traces:
        w("trace values per peak: " + ", ".join(t.name for t in line_traces))
    for i in idx:
        scores = ""
        if np.isfinite(p.s1_score[i]):
            scores = f"  scores S1 {p.s1_score[i]:.2f} S2 {p.s2_score[i]:.2f} noise {p.noise_score[i]:.2f}"
        w(f"- {p.time[i]:.3f}s  amp {p.amp[i]:.4g}  pass2: {p.label[i]}  final state: {p.state[i] or '-'}"
          f"{'  [final S1]' if p.final_s1[i] else ''}{scores}")
        if line_traces:
            w("    values: " + ", ".join(f"{t.name}={_f(t.value_at(float(p.time[i])), 4)}" for t in line_traces))
        for r in p.reasoning[i]:
            w(f"    {r}")

    for t in traces:
        if t.kind == KIND_LINE:
            continue
        sel = t.window(t0, t1)
        if not len(sel):
            continue
        w("")
        w(f"## {t.name} ({t.group}, {len(sel)})")
        for j in sel:
            text = f"  {t.text[j]}" if t.text else ""
            if t.kind == KIND_POINTS:
                w(f"- {t.x[j]:.3f}s  {t.y[j]:.4g}{t.unit and ' ' + t.unit}{text}".replace("\n", "; "))
            elif t.kind == KIND_SPANS:
                w(f"- {t.x[j]:.3f}–{t.x_end[j]:.3f}s{text}".replace("\n", "; "))

    s = a.states
    sel = np.nonzero((s.end >= t0) & (s.start <= t1))[0]
    w("")
    w(f"## Algorithm states ({len(sel)})")
    for j in sel:
        src = f"  [{s.source[j]}]" if s.source[j] else ""
        w(f"- {s.start[j]:.3f}–{s.end[j]:.3f}s  {s.state[j]}{src}")
        for r in s.reasoning[j]:
            w(f"    {r}")

    if annotation is not None:
        spans = [x for x in annotation.spans if x.end >= t0 and x.start <= t1]
        w("")
        w(f"## Annotation spans ({len(spans)})")
        for x in spans:
            w(f"- {x.start:.3f}–{x.end:.3f}s  {x.kind}  ({x.origin}{', clipped' if x.clipped else ''})")
        dis = [d for d in (disagreements or []) if d.end >= t0 and d.start <= t1]
        w("")
        w(f"## Disagreements ({len(dis)})")
        for d in dis:
            w(f"- {d.start:.3f}–{d.end:.3f}s  {d.kind}: {d.message}")

    defects = [d for d in a.defects if d.end >= t0 and d.start <= t1]
    w("")
    w(f"## Defects ({len(defects)})")
    for d in defects:
        w(f"- {d.start:.3f}–{d.end:.3f}s  {d.kind}: {d.message}")
    return "\n".join(lines) + "\n"
