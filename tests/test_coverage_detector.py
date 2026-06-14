"""
Tests for the label/boundary coverage desync detector.

`pass3_state_labels` (dense) is the source of truth; the HTML strip renders from
`pass3_state_boundaries`. A desync run is where a sample is labelled a real state
but the boundary list shows no matching band (uncovered, or covered by a different
state name — e.g. an "unknown" segment).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_helpers.coverage_detector import find_label_boundary_desync, summarize  # noqa: E402

_ENC = {"S1": 0, "systole": 1, "S2": 2, "diastole": 3, "unknown": 4}


def test_in_sync_has_no_desync():
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    bounds = [(0, 2, "S1", {}), (2, 4, "systole", {}), (4, 6, "S2", {}), (6, 8, "diastole", {})]
    assert find_label_boundary_desync(labels, bounds, _ENC) == []


def test_uncovered_s1_flagged():
    # Labels say S1 over [2,4) but no boundary covers it -> uncovered desync.
    labels = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
    bounds = [(0, 2, "S1", {}), (4, 6, "systole", {})]
    recs = find_label_boundary_desync(labels, bounds, _ENC, sample_rate=100)
    assert len(recs) == 1
    r = recs[0]
    assert r["label_state"] == "S1" and r["strip_state"] == "<none>"
    assert r["kind"] == "uncovered" and (r["lo"], r["hi"]) == (2, 4)
    assert summarize(recs)["total_runs"] == 1


def test_covered_by_other_state_flagged():
    # Labels say S1 but the only boundary there is "unknown" -> covered_other.
    labels = np.array([0, 0, 0, 0], dtype=np.int64)
    bounds = [(0, 4, "unknown", {})]
    recs = find_label_boundary_desync(labels, bounds, _ENC)
    assert len(recs) == 1
    assert recs[0]["kind"] == "covered_other"
    assert recs[0]["strip_state"] == "unknown"
