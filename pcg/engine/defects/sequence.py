"""
Detector for cardiac state-sequence (grammar) violations.

A correct Pass 3 timeline walks one fixed cycle:

    S1 -> systole -> S2 -> diastole -> S1 -> ...

Each state has exactly one legal successor. When the boundary list breaks that
cycle, a beat is mislabelled relative to its neighbours. The most visible case is
``diastole -> S2``: an S2 appears with no S1 (and no systole) before it, so the
state strip shows the S2 band where an S1 cycle belongs — i.e. "the S1 state is
missing at this beat" even though the dense labels agree with the boundaries.

Unlike the overlap / coverage detectors this is a *sequence* check: labels and
boundaries can be perfectly in sync and still encode an illegal cycle.

Pure module (no pipeline import) — unit-testable and reusable.

Gaps are not violations: a run of ``unknown`` (or any uncovered span between two
real runs) legitimately breaks the cycle, so transitions across a gap are skipped.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# The one legal successor of each real state.
_NEXT = {"S1": "systole", "systole": "S2", "S2": "diastole", "diastole": "S1"}
_REAL = frozenset(_NEXT)


def _real_runs(state_boundaries: List[Tuple]) -> List[Tuple[int, int, str]]:
    """Sort, drop empty/unknown, and collapse consecutive same-name spans into
    runs of (start, end, name). Only real cardiac states are kept."""
    segs = []
    for seg in (state_boundaries or []):
        try:
            a0, a1, name = int(seg[0]), int(seg[1]), str(seg[2])
        except (TypeError, ValueError, IndexError):
            continue
        if a1 > a0 and name in _REAL:
            segs.append((a0, a1, name))
    segs.sort(key=lambda s: (s[0], s[1]))
    runs: List[Tuple[int, int, str]] = []
    for a0, a1, name in segs:
        if runs and name == runs[-1][2] and a0 <= runs[-1][1]:
            runs[-1] = (runs[-1][0], max(runs[-1][1], a1), name)
        else:
            runs.append((a0, a1, name))
    return runs


def find_sequence_violations(
    state_boundaries: List[Tuple],
    *,
    sample_rate: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return one record per illegal transition between abutting real-state runs.

    Two consecutive real runs are checked only when they abut (no gap between).
    A transition is illegal when ``cur != _NEXT[prev]``.

    Each record:
      prev_state, cur_state   — the two run names
      expected                — the legal successor of prev_state
      kind                    — "missing_s1" for a ``diastole -> S2`` step (the
                                S1 band is absent between diastole and S2 — the
                                visible bug); else "bad_transition"
      at_sample               — start sample of the offending (cur) run
      prev_lo, prev_hi, cur_lo, cur_hi
      at_sec, prev/cur spans in seconds (if sample_rate given)
    """
    runs = _real_runs(state_boundaries)
    out: List[Dict[str, Any]] = []
    for (p0, p1, pn), (c0, c1, cn) in zip(runs, runs[1:]):
        if c0 != p1:
            continue  # a gap separates them — legitimate cycle break, not a violation
        if _NEXT[pn] == cn:
            continue  # legal
        kind = "missing_s1" if pn == "diastole" and cn == "S2" else "bad_transition"
        rec: Dict[str, Any] = {
            "prev_state": pn,
            "cur_state": cn,
            "expected": _NEXT[pn],
            "kind": kind,
            "at_sample": c0,
            "prev_lo": p0, "prev_hi": p1,
            "cur_lo": c0, "cur_hi": c1,
        }
        if sample_rate:
            rec["at_sec"] = c0 / float(sample_rate)
        out.append(rec)
    return out


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Roll up violations for a one-line verdict."""
    by_kind: Dict[str, int] = {}
    by_transition: Dict[str, int] = {}
    for r in records:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        key = f"{r['prev_state']}->{r['cur_state']}"
        by_transition[key] = by_transition.get(key, 0) + 1
    return {
        "total_violations": len(records),
        "missing_s1": by_kind.get("missing_s1", 0),
        "by_kind": by_kind,
        "by_transition": by_transition,
    }
