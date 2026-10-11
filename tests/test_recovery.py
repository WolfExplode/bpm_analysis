"""Recovery endpoints, data coverage and renewed-exertion detection."""
import numpy as np
import pytest

from pcg.app.recovery import measure_recovery, window_mean


def test_decline_has_exact_durations_signed_change_and_slope():
    t = np.arange(0, 1901, 0.5)
    y = 190 - 0.04 * t
    lines, _ = measure_recovery(t, y, 10)
    assert [line.end - line.start for line in lines] == [30, 60, 120, 180, 300, 600, 900, 1200, 1800]
    for line in lines:
        assert line.slope == pytest.approx(-0.04)
        assert line.change == pytest.approx(-0.04 * (line.end - line.start))


def test_endpoint_is_time_weighted_average_not_peak_or_sample_weighted():
    t = np.array([0, 1, 2, 2.1, 3, 4], dtype=float)
    y = 100 + t
    assert window_mean(t, y, 2) == pytest.approx(102)
    t = np.arange(0, 200, 0.1)
    y = np.full_like(t, 100)
    y[np.argmin(abs(t - 60))] = 120
    assert window_mean(t, y, 60) == pytest.approx(100.5)


def test_sustained_rise_between_endpoints_stops_later_lines_even_after_hr_falls():
    t = np.arange(0, 700, 0.5)
    y = 180 - 0.1 * t
    y[(t >= 140) & (t < 180)] += 20
    lines, cutoff = measure_recovery(t, y, 0)
    assert [line.end for line in lines] == [30, 60, 120]
    assert 136 <= cutoff <= 144


def test_brief_spike_and_small_rebound_do_not_end_recovery():
    t = np.arange(0, 650, 0.5)
    y = 180 - 0.03 * t
    y[(t >= 140) & (t < 141)] += 20
    y[(t >= 210) & (t < 230)] += 3
    lines, _ = measure_recovery(t, y, 0)
    assert lines[-1].end == 600


@pytest.mark.parametrize("gap", ["nan", "missing"])
def test_gap_stops_recovery_instead_of_interpolating_across_it(gap):
    t = np.arange(0, 400, 0.5)
    y = 180 - 0.1 * t
    mask = (t > 80) & (t < 95)
    if gap == "nan":
        y[mask] = np.nan
    else:
        t, y = t[~mask], y[~mask]
    lines, cutoff = measure_recovery(t, y, 0)
    assert [line.end for line in lines] == [30, 60]
    assert cutoff <= 80


def test_full_endpoint_window_required_and_next_start_is_a_hard_boundary():
    t = np.arange(0, 401, 0.5)
    y = 180 - 0.1 * t
    lines, cutoff = measure_recovery(t, y, 0, stop=180)
    assert [line.end for line in lines] == [30, 60, 120]
    assert cutoff == 180
    assert [line.end for line in measure_recovery(t[:121], y[:121], 0)[0]] == [30]
    assert not measure_recovery(t, y, -1)[0]
