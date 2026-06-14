"""
Detector for overlapping cardiac states.

The Pass 3 state timeline (``analysis_data["pass3_state_boundaries"]``) is meant to
be a *dense, non-overlapping* partition of time into S1 / systole / S2 / diastole
spans. A known bug lets the gap-fill paths emit segments that overlap each other
(two cardiac meanings claiming the same instant), most often inside gap regions
where old boundaries were not removed before new ones were painted.

This module is pure (no pipeline import) so it can be unit-tested and reused.

A "state boundary" is a tuple ``(start_sample, end_sample, name, meta_dict)``
representing the half-open span ``[start_sample, end_sample)``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

Segment = Tuple[int, int, str, Dict[str, Any]]

# rebuild_source values written by the three Pass 3 gap-fill paths. Any segment
# carrying one of these was *painted into a gap*, not detected from the signal.
GAP_REBUILD_SOURCES = frozenset({"gap_insert", "gap_label_pass3", "noise_repair"})


def _norm(seg: Any) -> Optional[Tuple[int, int, str, Dict[str, Any]]]:
    """Coerce a boundary tuple to (a0, a1, name, meta); drop empty/degenerate."""
    try:
        a0 = int(seg[0])
        a1 = int(seg[1])
        name = str(seg[2])
        meta = seg[3] if len(seg) > 3 and isinstance(seg[3], dict) else {}
    except (TypeError, ValueError, IndexError):
        return None
    if a1 <= a0:
        return None
    return a0, a1, name, meta


def find_overlapping_states(
    state_boundaries: List[Segment],
    *,
    sample_rate: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return a list of overlap records found in *state_boundaries*.

    Two spans overlap when ``a0 < b1 and b0 < a1`` (strict; abutting spans where
    ``a1 == b0`` are fine). Detection is a sweep: sort by start, and for each
    segment compare against every earlier segment whose end exceeds this start.

    Each record:
      seg_a, seg_b            — the two overlapping (a0, a1, name, meta) tuples
      overlap_lo, overlap_hi  — the overlapping sample span
      overlap_samples         — hi - lo
      overlap_sec             — seconds (only if sample_rate given)
      rebuild_sources         — set (as sorted list) of the two meta rebuild_source
      gap_related             — True if either segment was painted by a gap-fill path
      kind                    — "gap_rebuild" (>=1 side painted into a gap; the
                                targeted bug) or "edge_paint" (two real detected
                                segments whose painted edges overlap)
    """
    segs = [s for s in (_norm(x) for x in (state_boundaries or [])) if s is not None]
    segs.sort(key=lambda s: (s[0], s[1]))

    out: List[Dict[str, Any]] = []
    # Sweep with an "active" list of spans whose end is still ahead of cursor.
    active: List[Tuple[int, int, str, Dict[str, Any]]] = []
    for cur in segs:
        c0, c1, _, _ = cur
        active = [a for a in active if a[1] > c0]  # keep only spans that can overlap
        for prev in active:
            p0, p1, _, _ = prev
            lo = max(p0, c0)
            hi = min(p1, c1)
            if hi > lo:  # genuine overlap
                src_a = str(prev[3].get("rebuild_source", "")) if prev[3] else ""
                src_b = str(cur[3].get("rebuild_source", "")) if cur[3] else ""
                gap_related = bool(GAP_REBUILD_SOURCES & {src_a, src_b})
                rec: Dict[str, Any] = {
                    "seg_a": (p0, p1, prev[2], src_a),
                    "seg_b": (c0, c1, cur[2], src_b),
                    "overlap_lo": lo,
                    "overlap_hi": hi,
                    "overlap_samples": hi - lo,
                    "rebuild_sources": sorted({src_a, src_b}),
                    "gap_related": gap_related,
                    "kind": "gap_rebuild" if gap_related else "edge_paint",
                }
                if sample_rate:
                    rec["overlap_sec"] = (hi - lo) / float(sample_rate)
                    rec["overlap_t_lo_sec"] = lo / float(sample_rate)
                out.append(rec)
        active.append(cur)
    return out


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Roll up overlap records into counts for a one-line verdict."""
    total = len(records)
    gap_related = sum(1 for r in records if r["gap_related"])
    edge_paint = total - gap_related
    worst = max((r["overlap_samples"] for r in records), default=0)
    pair_kinds: Dict[str, int] = {}
    for r in records:
        key = "|".join(sorted((r["seg_a"][2], r["seg_b"][2])))
        pair_kinds[key] = pair_kinds.get(key, 0) + 1
    return {
        "total_overlaps": total,
        "gap_rebuild_overlaps": gap_related,
        "edge_paint_overlaps": edge_paint,
        "worst_overlap_samples": worst,
        "overlap_state_pairs": pair_kinds,
    }
