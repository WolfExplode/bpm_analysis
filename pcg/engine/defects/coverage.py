"""
Detector for label/boundary desync in the Pass 3 state timeline.

There are two parallel representations of the state timeline:
  * ``pass3_state_labels``     — dense per-sample int array (the source of truth)
  * ``pass3_state_boundaries`` — list of (start, end, name, meta) segments

The HTML state strip renders from the *boundary list*. When a sample is labelled
a real state (S1/systole/S2/diastole) but no matching boundary segment covers it,
the strip shows a hole even though the labels are correct — e.g. an S1 beat with
no S1 band beneath it.

This module is pure (no pipeline import) so it can be unit-tested and reused.

Encoding (``pass3_state_labels_encoding``): name -> int. Default in this codebase:
S1=0, systole=1, S2=2, diastole=3, unknown=4.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_DEFAULT_ENCODING = {"S1": 0, "systole": 1, "S2": 2, "diastole": 3, "unknown": 4}


def find_label_boundary_desync(
    state_labels: np.ndarray,
    state_boundaries: List[Tuple],
    encoding: Optional[Dict[str, int]] = None,
    *,
    sample_rate: Optional[int] = None,
    min_run: int = 1,
) -> List[Dict[str, Any]]:
    """Return runs where the dense labels and the boundary list disagree.

    For every sample, compute the state implied by the boundary list (the last
    boundary covering it; ``unknown``/uncovered count as "no real band"). Compare
    to the dense label. A *desync run* is a maximal run of samples where the label
    is a **real** state but the boundary strip shows no matching band, split into:

      kind = "uncovered"     — no boundary segment covers the sample at all
      kind = "covered_other" — a boundary covers it but with a different state
                               name (includes an "unknown"-named segment)

    Each record: label_state, strip_state, kind, lo, hi, run_samples, and
    (if sample_rate given) t_lo_sec / run_sec.
    """
    enc = dict(encoding or _DEFAULT_ENCODING)
    inv = {int(v): k for k, v in enc.items()}
    unknown_code = int(enc.get("unknown", 4))

    labels = np.asarray(state_labels)
    n = int(labels.shape[0]) if labels.ndim else 0
    if n == 0:
        return []

    # Rasterize the boundary list. -1 = uncovered; otherwise the state code of the
    # last segment painted over that sample (matches strip draw order).
    strip = np.full(n, -1, dtype=np.int64)
    for seg in (state_boundaries or []):
        try:
            a0 = max(0, int(seg[0]))
            a1 = min(n, int(seg[1]))
            code = enc.get(str(seg[2]), unknown_code)
        except (TypeError, ValueError, IndexError):
            continue
        if a1 > a0:
            strip[a0:a1] = int(code)

    real = labels != unknown_code
    covered = strip >= 0
    matches = covered & (strip == labels)
    # A desync sample: real label, but the strip does not show that same state.
    desync = real & ~matches

    out: List[Dict[str, Any]] = []
    if not np.any(desync):
        return out

    # Run-length encode the desync mask.
    padded = np.concatenate([[False], desync, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for lo, hi in zip(starts.tolist(), ends.tolist()):
        run = hi - lo
        if run < min_run:
            continue
        label_code = int(labels[lo])
        strip_code = int(strip[lo])
        kind = "uncovered" if strip_code < 0 else "covered_other"
        rec: Dict[str, Any] = {
            "lo": lo,
            "hi": hi,
            "run_samples": run,
            "label_state": inv.get(label_code, str(label_code)),
            "strip_state": "<none>" if strip_code < 0 else inv.get(strip_code, str(strip_code)),
            "kind": kind,
        }
        if sample_rate:
            rec["t_lo_sec"] = lo / float(sample_rate)
            rec["run_sec"] = run / float(sample_rate)
        out.append(rec)
    return out


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Roll up desync records for a one-line verdict."""
    by_label: Dict[str, int] = {}
    by_kind: Dict[str, int] = {}
    for r in records:
        by_label[r["label_state"]] = by_label.get(r["label_state"], 0) + 1
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
    return {
        "total_runs": len(records),
        "worst_run_samples": max((r["run_samples"] for r in records), default=0),
        "by_label_state": by_label,
        "by_kind": by_kind,
    }
