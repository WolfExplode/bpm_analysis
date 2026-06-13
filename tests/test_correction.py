"""Characterization tests for the pure helpers in correction.py (Pass 3).

These lock current behavior of the deterministic geometry/interval/interp
helpers so the Pass-3 logic can be refactored or split into modules safely.
"""
import numpy as np
import pandas as pd
import pytest

import correction as corr


# --- interval utilities -----------------------------------------------------

def test_merge_sorted_intervals_merges_overlap_and_touch():
    assert corr._merge_sorted_intervals([(0, 5), (3, 8), (10, 12)]) == [(0, 8), (10, 12)]
    # touching at the boundary (a <= lb) merges
    assert corr._merge_sorted_intervals([(0, 5), (5, 8)]) == [(0, 8)]
    # unsorted input is sorted first
    assert corr._merge_sorted_intervals([(10, 12), (0, 5)]) == [(0, 5), (10, 12)]
    assert corr._merge_sorted_intervals([]) == []


def test_half_open_intervals_intersect():
    assert corr._half_open_intervals_intersect(0, 5, 4, 10)
    assert not corr._half_open_intervals_intersect(0, 5, 5, 10)   # touching != overlap
    assert not corr._half_open_intervals_intersect(0, 5, 6, 10)


def test_span_intersects_merged_noise():
    merged = [(100, 200), (400, 500)]
    assert corr._span_intersects_merged_noise(150, 160, merged, 1000)
    assert corr._span_intersects_merged_noise(190, 410, merged, 1000)
    assert not corr._span_intersects_merged_noise(210, 390, merged, 1000)


def test_noise_sample_intervals_seconds_to_samples():
    segs = [{"start": 1.0, "end": 2.0}, {"start": 3.0, "end": 3.5}]
    out = corr._noise_sample_intervals(segs, sample_rate=100, n_samples=1000)
    assert out == [(100, 200), (300, 350)]


def test_noise_sample_intervals_guards():
    assert corr._noise_sample_intervals(None, 100, 1000) == []
    assert corr._noise_sample_intervals([], 100, 1000) == []
    # bad/degenerate segments dropped
    assert corr._noise_sample_intervals([{"start": 2.0, "end": 1.0}], 100, 1000) == []
    assert corr._noise_sample_intervals(["notadict"], 100, 1000) == []


# --- interpolation ----------------------------------------------------------

def test_interp_piecewise_linear_basic_and_edges():
    t = np.array([0.0, 10.0])
    y = np.array([0.0, 100.0])
    assert corr._interp_piecewise_linear(5.0, t, y) == pytest.approx(50.0)
    assert corr._interp_piecewise_linear(-5.0, t, y) == pytest.approx(0.0)    # left clamp
    assert corr._interp_piecewise_linear(20.0, t, y) == pytest.approx(100.0)  # right clamp


def test_interp_piecewise_linear_guards():
    assert corr._interp_piecewise_linear(5.0, np.array([1.0]), np.array([1.0])) is None
    assert corr._interp_piecewise_linear(5.0, np.array([0.0, 1.0]), np.array([1.0])) is None
    assert corr._interp_piecewise_linear(5.0, None, None) is None


def test_bpm_at_time_accepts_tuple_dict_series():
    times = [0.0, 10.0]
    bpm = [60.0, 120.0]
    assert corr._bpm_at_time(5.0, (times, bpm), fallback_bpm=0) == pytest.approx(90.0)
    assert corr._bpm_at_time(5.0, {"times": times, "bpm": bpm}, fallback_bpm=0) == pytest.approx(90.0)
    series = pd.Series(bpm, index=times)
    assert corr._bpm_at_time(5.0, series, fallback_bpm=0) == pytest.approx(90.0)


def test_bpm_at_time_fallbacks():
    assert corr._bpm_at_time(5.0, None, fallback_bpm=77.0) == 77.0
    assert corr._bpm_at_time(5.0, ([0.0], [60.0]), fallback_bpm=77.0) == 77.0   # < 2 points
    assert corr._bpm_at_time(5.0, "garbage", fallback_bpm=77.0) == 77.0
    assert corr._bpm_at_time(5.0, pd.Series(dtype=float), fallback_bpm=77.0) == 77.0


def test_bpm_at_time_edge_extrapolation_constant():
    times, bpm = [0.0, 10.0], [60.0, 120.0]
    assert corr._bpm_at_time(-5.0, (times, bpm), 0) == pytest.approx(60.0)
    assert corr._bpm_at_time(99.0, (times, bpm), 0) == pytest.approx(120.0)


# --- boundary geometry ------------------------------------------------------

def test_resolve_boundary_overlap_splits_overlap_at_midpoint():
    s1s, s1e, s2s, s2e = corr._resolve_boundary_overlap(90, 111, 95, 116, s1_next=300)
    assert s1e <= s2s          # overlap resolved
    assert s2e <= 300


def test_resolve_boundary_overlap_clips_s2_to_next_cycle():
    _, _, _, s2e = corr._resolve_boundary_overlap(90, 100, 200, 350, s1_next=300)
    assert s2e == 300


def test_paint_state_boundaries_fixed_window_no_overlap():
    s1s, s1e, s2s, s2e = corr._paint_state_boundaries(
        s1=100, s2=200, s1_next=300, s1_half=10, s2_half=10, n_samples=1000,
        use_transient_detection=False,
    )
    assert (s1s, s1e, s2s, s2e) == (90, 111, 190, 211)


def test_paint_state_boundaries_resolves_close_peaks():
    s1s, s1e, s2s, s2e = corr._paint_state_boundaries(
        s1=100, s2=105, s1_next=300, s1_half=10, s2_half=10, n_samples=1000,
        use_transient_detection=False,
    )
    assert s1e <= s2s          # close peaks no longer overlap
    assert s1s >= 0 and s2e <= 1000


# --- state-boundary list transforms -----------------------------------------

def test_trim_diastole_ends_on_next_s1():
    boundaries = [
        (0, 10, "S1", {}),
        (10, 50, "diastole", {}),
        (45, 55, "S1", {}),
    ]
    out = corr._pass3_trim_diastole_ends_on_next_s1(boundaries)
    dia = [b for b in out if b[2] == "diastole"][0]
    assert dia[1] == 45        # trimmed to next S1 start


def test_trim_diastole_drops_segment_that_would_invert():
    # next S1 starts before the diastole even begins -> trimmed away entirely
    boundaries = [(20, 50, "diastole", {}), (10, 15, "S1", {})]
    out = corr._pass3_trim_diastole_ends_on_next_s1(boundaries)
    assert all(b[2] != "diastole" for b in out)


def test_remove_boundaries_overlapping_span():
    boundaries = [(0, 100, "S1", {}), (100, 200, "S2", {}), (200, 300, "S1", {})]
    out = corr._pass3_remove_boundaries_overlapping_span(boundaries, lo=150, hi=250)
    # middle and last overlap [150,250); first does not
    assert out == [(0, 100, "S1", {})]
