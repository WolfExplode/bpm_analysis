"""
Tests for the cardiac state-sequence (grammar) detector.

A correct timeline cycles S1 -> systole -> S2 -> diastole. The detector flags
illegal transitions between abutting real-state runs; ``diastole -> S2`` (an S2
with no S1 before it) is the "missing S1 state" bug.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_helpers.state_sequence_detector import (  # noqa: E402
    find_sequence_violations,
    summarize,
)


def _cycle(start, step=10, n=2, names=("S1", "systole", "S2", "diastole")):
    """Build n back-to-back full cardiac cycles as (lo, hi, name, {}) tuples."""
    segs = []
    t = start
    for _ in range(n):
        for nm in names:
            segs.append((t, t + step, nm, {}))
            t += step
    return segs


def test_clean_cycle_has_no_violations():
    assert find_sequence_violations(_cycle(0, n=4)) == []


def test_missing_s1_flagged():
    # ... S2, diastole, S2, diastole ...  — the second S2 has no S1 before it.
    segs = [
        (0, 10, "S1", {}), (10, 20, "systole", {}), (20, 30, "S2", {}),
        (30, 40, "diastole", {}),
        (40, 50, "S2", {}),          # <-- diastole -> S2 : missing S1
        (50, 60, "diastole", {}),
        (60, 70, "S1", {}),
    ]
    recs = find_sequence_violations(segs, sample_rate=100)
    assert len(recs) == 1
    r = recs[0]
    assert r["prev_state"] == "diastole" and r["cur_state"] == "S2"
    assert r["expected"] == "S1"
    assert r["kind"] == "missing_s1"
    assert r["at_sample"] == 40
    assert r["at_sec"] == 0.4
    assert summarize(recs)["missing_s1"] == 1


def test_gap_between_runs_is_not_a_violation():
    # A gap (unknown / uncovered) between two real runs legitimately breaks the
    # cycle, so an out-of-order pair separated by a gap is NOT flagged.
    segs = [
        (0, 10, "S1", {}), (10, 20, "systole", {}),
        # gap 20..40 (no segment)
        (40, 50, "S2", {}), (50, 60, "diastole", {}),
    ]
    assert find_sequence_violations(segs) == []


def test_unknown_run_breaks_chain_without_flagging():
    segs = [
        (0, 10, "diastole", {}),
        (10, 20, "unknown", {}),
        (20, 30, "S2", {}),  # abuts unknown, not a real predecessor -> skipped
    ]
    assert find_sequence_violations(segs) == []


def test_generic_bad_transition_classified():
    # S1 -> S2 (skipping systole) is illegal but not the missing-S1 shape.
    segs = [(0, 10, "S1", {}), (10, 20, "S2", {}), (20, 30, "diastole", {})]
    recs = find_sequence_violations(segs)
    assert len(recs) == 1
    assert recs[0]["kind"] == "bad_transition"
    assert recs[0]["cur_state"] == "S2" and recs[0]["prev_state"] == "S1"
