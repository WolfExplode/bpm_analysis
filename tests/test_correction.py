"""Characterization tests for the pure helpers in correction.py (Pass 3).

These lock current behavior of the deterministic geometry/interval/interp
helpers so the Pass-3 logic can be refactored or split into modules safely.
"""
import numpy as np
import pandas as pd
import pytest

from pcg.engine import correction as corr


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

def test_make_sequential_later_segment_wins_shared_samples():
    boundaries = [
        (45, 55, "S1", {}),
        (0, 10, "S1", {}),
        (10, 50, "diastole", {}),
        (30, 48, "S2", {}),
    ]
    out = corr._pass3_make_sequential(boundaries)
    assert [b[:3] for b in out] == [(0, 10, "S1"), (10, 30, "diastole"), (30, 45, "S2"), (45, 55, "S1")]


def test_make_sequential_sounds_win_over_gaps():
    # A diastole that starts inside an S1 resumes after it; one lying wholly inside a
    # sound (an S2 that ran into the next S1) vanishes instead of cutting the S1 short.
    boundaries = [
        (0, 40, "S2", {}),
        (30, 60, "S1", {}),
        (40, 50, "diastole", {}),
        (55, 90, "systole", {"k": 1}),
    ]
    out = corr._pass3_make_sequential(boundaries)
    assert out == [(0, 30, "S2", {}), (30, 60, "S1", {}), (60, 90, "systole", {"k": 1})]


def _beat(s1, length=100):
    """One beat starting at *s1*: S1 10, systole 30, S2 10, diastole rest (empty meta, as after the phase decision)."""
    return [(s1, s1 + 10, "S1", {}), (s1 + 10, s1 + 40, "systole", {}),
            (s1 + 40, s1 + 50, "S2", {}), (s1 + 50, s1 + length, "diastole", {})]


def _labels_for(boundaries, n):
    code = {"S1": corr.STATE_S1, "systole": corr.STATE_SYSTOLE, "S2": corr.STATE_S2, "diastole": corr.STATE_DIASTOLE}
    labels = np.full(n, corr.STATE_DIASTOLE, dtype=np.int8)
    for a, b, name, _m in boundaries:
        labels[a:b] = code[name]
    return labels


def test_clear_hf_noise_on_s1_drops_whole_beat_and_clears_exactly_it():
    bd = _beat(0) + _beat(100) + _beat(200)
    labels, out = corr._pass3_clear_states_in_hf_noise(_labels_for(bd, 300), bd, [(105, 108)], 300)
    assert [b[0] for b in out] == [0, 10, 40, 50, 200, 210, 240, 250]
    assert (labels[100:200] == corr.STATE_UNKNOWN).all()
    assert not (labels[:100] == corr.STATE_UNKNOWN).any() and not (labels[200:] == corr.STATE_UNKNOWN).any()


def test_clear_hf_noise_in_diastole_keeps_s1():
    bd = _beat(0) + _beat(100) + _beat(200)
    labels, out = corr._pass3_clear_states_in_hf_noise(_labels_for(bd, 300), bd, [(170, 180)], 300)
    assert [b[:3] for b in out if 100 <= b[0] < 200] == [(100, 110, "S1")]
    assert (labels[110:200] == corr.STATE_UNKNOWN).all() and (labels[100:110] == corr.STATE_S1).all()


def test_clear_hf_noise_ignores_last_beat_and_noise_at_file_start():
    bd = _beat(0) + _beat(100)
    labels0 = _labels_for(bd, 200)
    labels, out = corr._pass3_clear_states_in_hf_noise(labels0.copy(), bd, [(0, 5), (150, 160)], 200)
    assert out == bd and (labels == labels0).all()


def test_rebuild_after_clear_never_overlaps_kept_segments():
    from pcg.engine.config import DEFAULT_PARAMS
    from pcg.engine.defects.overlaps import find_overlapping_states
    bd = _beat(0, 300) + _beat(300, 300) + _beat(600, 300) + _beat(900, 300)
    labels, out = corr._pass3_clear_states_in_hf_noise(_labels_for(bd, 1200), bd, [(305, 308), (850, 860)], 1200)
    labels, out = corr._pass3_rebuild_unknown_runs(labels, out, 1200, None, 60.0, 300, dict(DEFAULT_PARAMS))
    assert not (labels == corr.STATE_UNKNOWN).any()
    assert find_overlapping_states(out) == []


@pytest.mark.parametrize("width", [290, 300, 307, 311, 589, 604])
def test_rebuild_fills_gap_without_sliver_s1(width):
    from pcg.engine.config import DEFAULT_PARAMS
    # A gap that is not a whole number of (rounded) cycles must not end in a sliver S1.
    n = 100 + width + 50
    labels = np.full(n, corr.STATE_DIASTOLE, dtype=np.int8)
    labels[100:100 + width] = corr.STATE_UNKNOWN
    labels, out = corr._pass3_rebuild_unknown_runs(labels, [], n, None, 60.0, 300, dict(DEFAULT_PARAMS))
    s1 = [b for b in out if b[2] == "S1"]
    assert s1 and min(b[1] - b[0] for b in s1) >= 5
    assert out[-1][2] == "diastole" and out[-1][1] == 100 + width


def test_remove_boundaries_overlapping_span():
    boundaries = [(0, 100, "S1", {}), (100, 200, "S2", {}), (200, 300, "S1", {})]
    out = corr._pass3_remove_boundaries_overlapping_span(boundaries, lo=150, hi=250)
    # middle and last overlap [150,250); first does not
    assert out == [(0, 100, "S1", {})]
