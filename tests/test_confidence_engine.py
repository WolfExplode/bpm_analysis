"""Pure physiology/timing helpers in the confidence engine.

Assertions are invariant-based (monotonicity, bounds, identities) rather than
magic numbers, so they survive parameter retuning in config.py.
"""
import numpy as np
import pytest

import config
import confidence_engine as ce

PARAMS = dict(config.DEFAULT_PARAMS)


# --- calculate_bpm_intervals ------------------------------------------------

def test_bpm_intervals_rr_is_60_over_bpm():
    for bpm in (40, 60, 120, 200):
        iv = ce.calculate_bpm_intervals(bpm, PARAMS)
        assert iv["rr_interval"] == pytest.approx(60.0 / bpm)


def test_bpm_intervals_systole_within_abs_bounds():
    min_abs = PARAMS["min_s1_s2_interval_sec"]
    cap = PARAMS["s1_s2_interval_cap_sec"]
    for bpm in (40, 80, 150, 240):
        iv = ce.calculate_bpm_intervals(bpm, PARAMS)
        assert min_abs <= iv["s1_s2_nominal"] <= cap
        assert iv["s1_s2_min"] >= min_abs


def test_bpm_intervals_systole_nominal_shrinks_with_bpm():
    # Weissler: ejection time decreases as heart rate rises.
    nominals = [ce.calculate_bpm_intervals(b, PARAMS)["s1_s2_nominal"] for b in (60, 100, 160)]
    assert nominals[0] > nominals[1] > nominals[2]


def test_bpm_intervals_diastole_identity():
    iv = ce.calculate_bpm_intervals(75, PARAMS)
    assert iv["s2_s1_nominal"] == pytest.approx(iv["rr_interval"] - iv["s1_s2_nominal"])


def test_bpm_intervals_guards_nonpositive_bpm():
    iv = ce.calculate_bpm_intervals(0.0, PARAMS)
    assert np.isfinite(iv["rr_interval"]) and iv["rr_interval"] > 0
    iv2 = ce.calculate_bpm_intervals(-50, PARAMS)
    assert np.isfinite(iv2["rr_interval"])


def test_bpm_intervals_min_feasible_cycle_components():
    iv = ce.calculate_bpm_intervals(90, PARAMS)
    expected = iv["s1_min"] + iv["s1_s2_min"] + iv["s2_min"] + iv["diastole_min"]
    assert iv["min_feasible_cycle"] == pytest.approx(expected)


# --- hr_reactivity_factor ---------------------------------------------------

def test_reactivity_is_one_at_rest_zero_at_max():
    assert ce.hr_reactivity_factor(60, hr_max=200, hr_rest=60) == pytest.approx(1.0)
    assert ce.hr_reactivity_factor(200, hr_max=200, hr_rest=60) == pytest.approx(0.0)


def test_reactivity_monotonic_decreasing():
    vals = [ce.hr_reactivity_factor(hr, 200, 60) for hr in range(60, 201, 20)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_reactivity_degenerate_reserve_returns_one():
    assert ce.hr_reactivity_factor(150, hr_max=100, hr_rest=100) == 1.0
    assert ce.hr_reactivity_factor(150, hr_max=80, hr_rest=120) == 1.0


def test_reactivity_clamps_below_rest_and_above_max():
    # HR outside [rest, max] must not produce values outside [0, 1].
    assert ce.hr_reactivity_factor(40, 200, 60) == pytest.approx(1.0)
    assert ce.hr_reactivity_factor(250, 200, 60) == pytest.approx(0.0)


# --- update_long_term_bpm ---------------------------------------------------

def test_update_long_term_bpm_clamped_to_config_range():
    # Instant BPM implied by a tiny RR is enormous; belief must stay <= max_bpm.
    out = ce.update_long_term_bpm(0.2, current_long_term_bpm=120, params=PARAMS)
    assert PARAMS["min_bpm"] <= out <= PARAMS["max_bpm"]


def test_update_long_term_bpm_rate_limited_per_beat():
    # One beat cannot move the belief more than max_change_per_beat * rr.
    rr = 1.0
    start = 100.0
    out = ce.update_long_term_bpm(rr, current_long_term_bpm=start, params=PARAMS)
    max_change = PARAMS["bpm_belief_max_change_per_beat"] * rr
    assert abs(out - start) <= max_change + 1e-9


def test_update_long_term_bpm_moves_toward_instant():
    # instant = 60/0.5 = 120 > 100 belief -> belief should rise.
    out = ce.update_long_term_bpm(0.5, current_long_term_bpm=100, params=PARAMS)
    assert out > 100


# --- contractility expected ratio (BPM power curve) -------------------------

def test_contractility_expected_ratio_endpoints_and_monotonic():
    lo = PARAMS["contractility_bpm_min"]
    hi = PARAMS["contractility_bpm_max"]
    assert ce._contractility_expected_ratio_bpm(lo, PARAMS) == pytest.approx(PARAMS["contractility_low_ratio"])
    assert ce._contractility_expected_ratio_bpm(hi, PARAMS) == pytest.approx(PARAMS["contractility_high_ratio"])
    ratios = [ce._contractility_expected_ratio_bpm(b, PARAMS) for b in range(lo, hi + 1, 20)]
    assert all(a <= b for a, b in zip(ratios, ratios[1:]))


# --- adjust_confidence_with_contractility (BPM-curve path, no state) ---------

def test_contractility_boost_when_ratio_matches_expected():
    bpm = PARAMS["contractility_bpm_min"]
    expected = ce._contractility_expected_ratio_bpm(bpm, PARAMS)
    conf, step = ce.adjust_confidence_with_contractility(
        base_confidence=0.5,
        s1_prominence=expected,   # actual ratio == expected -> peak of boost tent
        s2_prominence=1.0,
        bpm=bpm,
        params=PARAMS,
    )
    assert conf > 0.5
    assert isinstance(step, dict)


def test_contractility_penalty_when_ratio_far_off():
    bpm = PARAMS["contractility_bpm_min"]
    conf, _ = ce.adjust_confidence_with_contractility(
        base_confidence=0.5,
        s1_prominence=50.0,   # actual ratio huge, far right of penalty ramp
        s2_prominence=1.0,
        bpm=bpm,
        params=PARAMS,
    )
    assert conf < 0.5


def test_contractility_confidence_stays_in_unit_interval():
    for s1, s2 in [(0.0, 1.0), (1.0, 0.0), (5.0, 5.0), (100.0, 0.001)]:
        conf, _ = ce.adjust_confidence_with_contractility(0.9, s1, s2, bpm=120, params=PARAMS)
        assert 0.0 <= conf <= 1.0
