"""
Per-peak label_scores (S1, S2, noise) attached to peak_classifications.

These are heuristic scores derived from the greedy forward pass (pairing / lone-S1),
not calibrated probabilities or global marginals. They are path-dependent and intended
for pass 3+ and tooling—not as a generative model over labels.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Keys stored on each classification dict under "label_scores".
LABEL_SCORE_S1 = "S1"
LABEL_SCORE_S2 = "S2"
LABEL_SCORE_NOISE = "noise"


def extract_pairing_final_confidence_from_steps(steps: List[Dict[str, Any]]) -> float:
    """Final pairing confidence from attempt_pair steps (prefers step name 'Final', else last result)."""
    if not steps:
        return 0.0
    for s in reversed(steps):
        if s.get("step") == "Final":
            try:
                return float(s.get("result", 0.0))
            except (TypeError, ValueError):
                return 0.0
    try:
        return float(steps[-1].get("result", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def label_scores_paired_s1(final_pair_confidence: float) -> Dict[str, float]:
    """Peak is labeled paired S1; final_pair_confidence is the pairing engine final score."""
    c = _clip01(final_pair_confidence)
    return {LABEL_SCORE_S1: c, LABEL_SCORE_S2: 0.0, LABEL_SCORE_NOISE: max(0.0, 1.0 - c)}


def label_scores_paired_s2(final_pair_confidence: float) -> Dict[str, float]:
    """Peak is labeled paired S2; same pairing run as its S1 partner."""
    c = _clip01(final_pair_confidence)
    return {LABEL_SCORE_S1: 0.0, LABEL_SCORE_S2: c, LABEL_SCORE_NOISE: max(0.0, 1.0 - c)}


def label_scores_lone_s1_validated(final_confidence: float) -> Dict[str, float]:
    """Peak is validated lone S1 (combined lone-S1 score)."""
    c = _clip01(final_confidence)
    return {LABEL_SCORE_S1: c, LABEL_SCORE_S2: 0.0, LABEL_SCORE_NOISE: max(0.0, 1.0 - c)}


def label_scores_noise_rejected_lone(
    final_lone_confidence: float,
    pairing_failure_confidence: Optional[float] = None,
) -> Dict[str, float]:
    """
    Peak labeled noise after failed pair and failed lone-S1 threshold.
    final_lone_confidence: last combined lone score before rejection.
    pairing_failure_confidence: if set, adds a small S2 mass when pairing was weak (mis-timed S2 hint).
    """
    lc = _clip01(final_lone_confidence)
    s2 = 0.0
    if pairing_failure_confidence is not None:
        pc = _clip01(pairing_failure_confidence)
        s2 = 0.25 * max(0.0, 1.0 - pc)
    return {
        LABEL_SCORE_S1: lc,
        LABEL_SCORE_S2: s2,
        LABEL_SCORE_NOISE: max(0.0, 1.0 - lc),
    }


def label_scores_noise_middle_structural() -> Dict[str, float]:
    """Middle peak labeled noise (lookahead skip-one or skip-one pair); no pairing score on this index."""
    return {LABEL_SCORE_S1: 0.15, LABEL_SCORE_S2: 0.15, LABEL_SCORE_NOISE: 0.70}


def label_scores_lone_s1_last_peak() -> Dict[str, float]:
    """Final peak forced as Lone S1 (last in sequence) without lone-S1 scoring."""
    return {LABEL_SCORE_S1: 0.75, LABEL_SCORE_S2: 0.05, LABEL_SCORE_NOISE: 0.20}


def get_label_scores(entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Return label_scores dict from a peak_classifications entry, or None."""
    if not entry or not isinstance(entry, dict):
        return None
    ls = entry.get("label_scores")
    if not isinstance(ls, dict):
        return None
    return ls
