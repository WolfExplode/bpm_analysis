#!/usr/bin/env python3
"""Canonical per-beat S1 detection scoring, shared by all dataset adapters.

Each research database ships ground truth in its own format; a per-database
adapter (benchmarking/adapters/) translates that into the canonical inputs here
so every benchmark reports the same literature-standard metrics. See
docs/adr/0003-per-database-benchmark-adapters.md.

Metric definitions follow the heart-sound segmentation literature
(Springer et al. 2016). A reference S1 is a true positive when a predicted S1
center falls within `tolerance` seconds; matching is greedy nearest, one-to-one.

    Se  = TP / (TP + FN)   FN = reference S1 with no predicted S1 in tolerance
    PPV = TP / (TP + FP)   FP = predicted S1 with no reference S1 in tolerance
    F1  = 2*Se*PPV / (Se + PPV)

A phase-flipped beat (right time, predicted as S2 instead of S1) is a plain
false negative here, exactly as the literature would score it. A `flip`
diagnostic count is reported separately but never alters Se/PPV/F1.
"""

from typing import Dict, List, Optional, Tuple

# Literature-standard tolerance (Springer ±60 ms) plus the looser 100 ms used by
# the native inputs/ benchmark, for cross-reference.
DEFAULT_TOLERANCES_SEC: Tuple[float, ...] = (0.06, 0.10)

Span = Tuple[float, float, str]  # (start_sec, end_sec, state)


def span_center(start: float, end: float) -> float:
    return (start + end) / 2.0


def _in_any_span(center: float, windows: List[Tuple[float, float]]) -> bool:
    return any(s <= center <= e for s, e in windows)


def s1_centers(spans: List[Span]) -> List[float]:
    return [span_center(s, e) for s, e, st in spans if st.upper() == "S1"]


def s2_centers(spans: List[Span]) -> List[float]:
    return [span_center(s, e) for s, e, st in spans if st.upper() == "S2"]


def _greedy_match(
    ref: List[float],
    pred: List[float],
    tolerance: float,
) -> Tuple[List[Optional[int]], List[bool]]:
    """One-to-one nearest match of ref centers to pred centers within tolerance.

    Returns (ref_to_pred, pred_used): ref_to_pred[i] is the matched pred index
    for ref[i] or None; pred_used[j] marks a consumed prediction.
    """
    pred_used = [False] * len(pred)
    ref_to_pred: List[Optional[int]] = []
    for rc in ref:
        best_j: Optional[int] = None
        best_d = tolerance
        for j, pc in enumerate(pred):
            if pred_used[j]:
                continue
            d = abs(pc - rc)
            if d <= best_d:
                best_d = d
                best_j = j
        if best_j is not None:
            pred_used[best_j] = True
        ref_to_pred.append(best_j)
    return ref_to_pred, pred_used


def score_file(
    manual_s1: List[float],
    pred_s1: List[float],
    pred_s2: List[float],
    tolerances: Tuple[float, ...] = DEFAULT_TOLERANCES_SEC,
) -> Dict[float, Dict[str, int]]:
    """Per-tolerance TP/FN/FP/flip counts for one recording.

    Aggregate these raw counts across files, then derive Se/PPV/F1 from the
    totals (micro-average) — that is how the literature pools beats.
    """
    out: Dict[float, Dict[str, int]] = {}
    for tol in tolerances:
        ref_to_pred, pred_used = _greedy_match(manual_s1, pred_s1, tol)
        tp = sum(1 for j in ref_to_pred if j is not None)
        fn = len(manual_s1) - tp
        fp = sum(1 for used in pred_used if not used)

        # Diagnostic only: of the FN reference S1s, how many had a predicted S2
        # sitting within tolerance (i.e. a phase flip rather than a true miss).
        flip = 0
        for i, j in enumerate(ref_to_pred):
            if j is not None:
                continue
            rc = manual_s1[i]
            if any(abs(pc - rc) <= tol for pc in pred_s2):
                flip += 1

        out[tol] = {"tp": tp, "fn": fn, "fp": fp, "flip": flip}
    return out


def derive_metrics(counts: Dict[str, int]) -> Dict[str, float]:
    """Se/PPV/F1 from summed tp/fn/fp counts."""
    tp, fn, fp = counts["tp"], counts["fn"], counts["fp"]
    se = tp / (tp + fn) if (tp + fn) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * se * ppv / (se + ppv) if (se + ppv) else 0.0
    return {"se": se, "ppv": ppv, "f1": f1}


def filter_to_windows(
    centers: List[float],
    windows: List[Tuple[float, float]],
) -> List[float]:
    """Keep only predicted centers that fall inside a labeled window.

    Predictions outside annotated time have no ground truth and must not be
    counted as false positives.
    """
    if not windows:
        return list(centers)
    return [c for c in centers if _in_any_span(c, windows)]
