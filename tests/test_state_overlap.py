"""
Tests for the overlapping-states detector.

The Pass 3 state timeline should be a non-overlapping partition. This detector
finds where two spans claim the same samples and classifies the overlap.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_helpers.overlap_detector import find_overlapping_states  # noqa: E402


def test_detector_flags_strict_overlap_only():
    # Abutting spans are fine; a 10-sample overlap is flagged and classified.
    abut = [(0, 100, "S1", {}), (100, 200, "systole", {})]
    assert find_overlapping_states(abut) == []
    bad = [(0, 100, "S1", {}), (90, 200, "systole", {"rebuild_source": "noise_repair"})]
    recs = find_overlapping_states(bad)
    assert len(recs) == 1
    assert recs[0]["overlap_samples"] == 10
    assert recs[0]["kind"] == "gap_rebuild"


def test_edge_paint_vs_gap_rebuild_classification():
    # Neither side carries a rebuild_source -> edge_paint; one side does -> gap_rebuild.
    edge = [(0, 100, "S2", {}), (95, 150, "S1", {})]
    recs = find_overlapping_states(edge)
    assert len(recs) == 1 and recs[0]["kind"] == "edge_paint"
