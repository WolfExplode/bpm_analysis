"""
Detector: does the cardiac-state band under a peak agree with the peak's label?

Each detected peak is classified S1 / S2 / Noise (``peak_classifications[idx].peak_type``).
The state band covering that same sample should agree: an S1 peak should sit under
an S1 state, an S2 peak under an S2 state. When an S1 peak sits under an **S2** band
(or vice-versa) the cycle's labels are swapped — the visible "the marker says S1 but
the strip shows S2" bug, which in turn produces the ``diastole -> S2`` sequence
violation (the missing S1 state).

Pure module (no pipeline import). Reads the same two structures the plots do.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Local copy of the peak_type -> kind rule so this module stays pipeline-free.
def _peak_kind(peak_type: str) -> str:
    s = (peak_type or "").strip()
    if s.startswith("S1") or s.startswith("Lone S1"):
        return "S1"
    if s.startswith("S2"):
        return "S2"
    if s.startswith("Noise"):
        return "NOISE"
    return "other"


def _covering_state(boundaries_sorted: List[Tuple], sample: int) -> Optional[Tuple]:
    """Return the (last) boundary segment covering *sample*, or None."""
    hit = None
    for seg in boundaries_sorted:
        a0, a1 = int(seg[0]), int(seg[1])
        if a0 <= sample < a1:
            hit = seg  # last writer wins, matching strip draw order
        elif a0 > sample:
            break
    return hit


def find_peak_state_mismatches(
    peak_classifications: Dict[Any, Any],
    state_boundaries: List[Tuple],
    *,
    sample_rate: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Flag every S1/S2 peak whose covering state band is the *other* heart sound.

    Only S1 and S2 peaks are checked (Noise/other peaks legitimately fall in any
    state). A mismatch requires the covering band to also be a heart sound (S1/S2)
    that differs from the peak's kind — so a peak landing in systole/diastole or a
    gap is not flagged.

    Each record: peak, peak_type, peak_kind, state, state_span, (peak_sec).
    """
    segs = sorted(
        (s for s in (state_boundaries or []) if int(s[1]) > int(s[0])),
        key=lambda s: int(s[0]),
    )
    out: List[Dict[str, Any]] = []
    for idx, info in (peak_classifications or {}).items():
        try:
            sample = int(idx)
        except (TypeError, ValueError):
            continue
        pt = info.get("peak_type", "") if isinstance(info, dict) else str(info)
        kind = _peak_kind(pt)
        if kind not in ("S1", "S2"):
            continue
        seg = _covering_state(segs, sample)
        if seg is None:
            continue
        state = str(seg[2])
        if state in ("S1", "S2") and state != kind:
            rec: Dict[str, Any] = {
                "peak": sample,
                "peak_type": pt,
                "peak_kind": kind,
                "state": state,
                "state_span": (int(seg[0]), int(seg[1])),
            }
            if sample_rate:
                rec["peak_sec"] = sample / float(sample_rate)
            out.append(rec)
    out.sort(key=lambda r: r["peak"])
    return out


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Roll up; ``swapped_pairs`` counts S1-under-S2 paired with a nearby S2-under-S1."""
    s1_under_s2 = sum(1 for r in records if r["peak_kind"] == "S1")
    s2_under_s1 = sum(1 for r in records if r["peak_kind"] == "S2")
    return {
        "total_mismatches": len(records),
        "s1_peak_under_s2_band": s1_under_s2,
        "s2_peak_under_s1_band": s2_under_s1,
    }
