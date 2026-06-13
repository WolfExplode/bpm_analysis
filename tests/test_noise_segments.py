"""HF-noise interval gating + merging."""
import numpy as np

import config
import noise_segments as ns
from noise_segments import _merge_intervals


def test_merge_intervals_bridges_within_gap():
    # gap between (0,2) and (4,6) is 2 -> merged when max_gap >= 2
    assert _merge_intervals([(0, 2), (4, 6)], max_gap=2) == [(0, 6)]
    assert _merge_intervals([(0, 2), (4, 6)], max_gap=1) == [(0, 2), (4, 6)]


def test_merge_intervals_sorts_and_coalesces_overlaps():
    assert _merge_intervals([(5, 8), (0, 3), (2, 6)], max_gap=0) == [(0, 8)]


def test_merge_intervals_empty():
    assert _merge_intervals([], 5) == []


def _params(**over):
    p = dict(config.DEFAULT_PARAMS)
    p.update(over)
    return p


def test_compute_segments_empty_or_degenerate_inputs():
    p = _params()
    assert ns.compute_noise_event_segments(None, 600, p) == []
    assert ns.compute_noise_event_segments(np.array([1.0]), 600, p) == []  # n < 2
    assert ns.compute_noise_event_segments(np.zeros(100), 0, p) == []      # bad sr
    assert ns.compute_noise_event_segments(np.full(100, np.nan), 600, p) == []


def test_compute_segments_detects_a_noisy_burst():
    sr = 1000
    env = np.full(sr, 0.001)          # 1 s of quiet
    env[400:600] = 1.0                # 200 ms loud burst
    p = _params(
        noise_segment_gate_quantile=0.80,
        noise_segment_gate_min_amplitude=0.01,
        noise_segment_merge_gap_ms=50.0,
        noise_segment_min_duration_ms=20.0,
        noise_segment_expand_ms=0.0,
        noise_segment_max_coverage=0.9,
    )
    segs = ns.compute_noise_event_segments(env, sr, p)
    assert len(segs) == 1
    s = segs[0]
    assert s["state"] == "noise"
    # Burst is samples [400,600) -> [0.4, 0.6) s (expand=0).
    assert s["start"] == 0.4
    assert s["end"] == 0.6
    assert s["peak"] == 1.0


def test_compute_segments_discarded_when_coverage_too_high(caplog):
    sr = 1000
    env = np.linspace(0.0, 1.0, sr)   # ramp -> top 50% all exceed the 0.5 quantile
    p = _params(
        noise_segment_gate_quantile=0.5,
        noise_segment_gate_min_amplitude=0.0,
        noise_segment_expand_ms=0.0,
        noise_segment_min_duration_ms=0.0,
        noise_segment_max_coverage=0.40,   # ~50% coverage > limit -> []
    )
    segs = ns.compute_noise_event_segments(env, sr, p)
    assert segs == []
