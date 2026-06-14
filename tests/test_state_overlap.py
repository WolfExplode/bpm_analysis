"""
Regression tests for the gap-region overlapping-states fix.

The Pass 3 state timeline must be a non-overlapping partition. Two helpers in
correction.py keep it that way when gap-fill segments are merged with kept real
boundaries: `_pass3_clip_boundaries_outside_spans` (clip a kept neighbour whose
painted edge bled into a filled region) and `_clamp_segments_sequential` (stop
adjacent rebuilt beats overlapping each other). The pure detector in
debug_helpers mirrors how the bug is found in the wild.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from correction import (  # noqa: E402
    _clamp_segments_sequential,
    _merge_spans,
    _pass3_clip_boundaries_outside_spans,
)
from debug_helpers.overlap_detector import find_overlapping_states  # noqa: E402


def test_merge_spans_overlapping_and_adjacent():
    assert _merge_spans([(0, 10), (5, 15), (20, 25)]) == [(0, 15), (20, 25)]
    assert _merge_spans([(10, 10), (3, 5)]) == [(3, 5)]  # empty dropped


def test_clip_trims_bled_edge():
    # A kept real S1 whose painted start bled into a filled gap ending at 281462.
    seg = [(281438, 281486, "S1", {"s1": 281450})]
    out = _pass3_clip_boundaries_outside_spans(seg, [(281419, 281462)])
    assert out == [(281462, 281486, "S1", {"s1": 281450})]


def test_clip_splits_and_drops():
    assert _pass3_clip_boundaries_outside_spans([(0, 100, "X", {})], [(40, 60)]) == [
        (0, 40, "X", {}),
        (60, 100, "X", {}),
    ]
    assert _pass3_clip_boundaries_outside_spans([(10, 20, "Y", {})], [(0, 100)]) == []


def test_clamp_sequential_resolves_self_overlap():
    # Adjacent rebuilt S2 / next S1 collide by 23 samples; clamp truncates the S2.
    segs = [(36322, 36360, "S2", {}), (36337, 36384, "S1", {})]
    out = _clamp_segments_sequential(segs)
    assert out == [(36322, 36337, "S2", {}), (36337, 36384, "S1", {})]
    assert find_overlapping_states(out) == []


def test_detector_flags_strict_overlap_only():
    # Abutting spans are fine; a 10-sample overlap is flagged and classified.
    abut = [(0, 100, "S1", {}), (100, 200, "systole", {})]
    assert find_overlapping_states(abut) == []
    bad = [(0, 100, "S1", {}), (90, 200, "systole", {"rebuild_source": "noise_repair"})]
    recs = find_overlapping_states(bad)
    assert len(recs) == 1
    assert recs[0]["overlap_samples"] == 10
    assert recs[0]["kind"] == "gap_rebuild"
