"""Defects: places where a run violates the algorithm's own invariants.

Each checker module is pure (boundaries / labels in, records out) and unit-tested
on its own. `find_defects` runs them all at the end of an analysis and converts
the records into uniform `Defect`s (kind, time span, one-line message) for the
Analysis file and the workspace's Defects panel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .coverage import find_label_boundary_desync
from .overlaps import find_overlapping_states
from .peak_state import find_peak_state_mismatches
from .sequence import find_sequence_violations

KIND_OVERLAP = "overlap"
KIND_PEAK_STATE = "peak/state swap"
KIND_SEQUENCE = "sequence"
KIND_COVERAGE = "coverage"
KIND_PLAUSIBILITY = "plausibility gate"

KINDS = (KIND_OVERLAP, KIND_PEAK_STATE, KIND_SEQUENCE, KIND_COVERAGE, KIND_PLAUSIBILITY)


@dataclass(frozen=True)
class Defect:
    kind: str
    start: float  # seconds
    end: float
    message: str

    def shifted(self, offset_sec: float) -> "Defect":
        return Defect(self.kind, self.start + offset_sec, self.end + offset_sec, self.message)


def find_defects(
    analysis_data: Dict[str, Any],
    sample_rate: int,
    duration_sec: float,
    bpm_failure_report: Dict[str, Any],
) -> List[Defect]:
    """All invariant violations of one run, sorted by time."""
    sr = float(sample_rate)
    bounds = analysis_data.get("pass3_state_boundaries") or []
    out: List[Defect] = []

    for r in find_overlapping_states(bounds):
        a, b = r["seg_a"], r["seg_b"]
        out.append(Defect(
            KIND_OVERLAP, r["overlap_lo"] / sr, r["overlap_hi"] / sr,
            f"{a[2]} and {b[2]} overlap by {r['overlap_samples'] / sr * 1000:.0f} ms ({r['kind']})",
        ))

    for r in find_peak_state_mismatches(analysis_data.get("peak_classifications") or {}, bounds):
        lo, hi = r["state_span"]
        out.append(Defect(
            KIND_PEAK_STATE, lo / sr, hi / sr,
            f"{r['peak_type']} peak at {r['peak'] / sr:.2f}s sits under an {r['state']} state",
        ))

    for r in find_sequence_violations(bounds):
        out.append(Defect(
            KIND_SEQUENCE, r["prev_lo"] / sr, r["cur_hi"] / sr,
            f"{r['prev_state']} -> {r['cur_state']} (expected {r['expected']})"
            + (" — missing S1" if r["kind"] == "missing_s1" else ""),
        ))

    labels = analysis_data.get("pass3_state_labels")
    if labels is not None and len(labels):
        for r in find_label_boundary_desync(np.asarray(labels), bounds, analysis_data.get("pass3_state_labels_encoding")):
            out.append(Defect(
                KIND_COVERAGE, r["lo"] / sr, r["hi"] / sr,
                f"labels say {r['label_state']} but the state list shows {r['strip_state']} ({r['kind']})",
            ))

    report = bpm_failure_report or {}
    if report.get("failed"):
        out.append(Defect(KIND_PLAUSIBILITY, 0.0, float(duration_sec), "; ".join(report.get("reasons") or [])))

    out.sort(key=lambda d: (d.start, d.end, d.kind))
    return out
