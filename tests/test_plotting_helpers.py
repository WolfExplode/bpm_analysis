"""plotting: pure data-transform helpers that feed the interactive HTML report
(systole/diastole curves, epoch->datetime64 mapping, inline-script JSON escaping).
"""
import numpy as np
import pandas as pd
import pytest

from plotting import (
    _compute_expected_diastole_from_bpm,
    _compute_measured_diastole_from_state_boundaries,
    _compute_measured_systole_from_state_boundaries,
    _compute_systolic_interval_data,
    _compute_systolic_shift,
    _elapsed_seconds_to_plot_datetimes,
    _json_for_html_inline_script,
)
from time_utils import seconds_to_datetime


# --- _elapsed_seconds_to_plot_datetimes ---------------------------------------

def test_elapsed_seconds_matches_seconds_to_datetime_scalar_path():
    seconds = np.array([0.0, 1.5, 10.0])
    result = _elapsed_seconds_to_plot_datetimes(seconds)
    expected = pd.to_datetime([seconds_to_datetime(float(t)) for t in seconds]).to_numpy()
    np.testing.assert_array_equal(result, expected)


def test_elapsed_seconds_empty_input_returns_empty_datetime64_array():
    result = _elapsed_seconds_to_plot_datetimes(np.array([]))
    assert result.size == 0
    assert result.dtype == np.dtype("datetime64[ns]")


def test_elapsed_seconds_returns_plain_numpy_not_pandas_index():
    result = _elapsed_seconds_to_plot_datetimes(np.array([1.0]))
    assert isinstance(result, np.ndarray)
    assert not isinstance(result, pd.DatetimeIndex)


# --- _json_for_html_inline_script ---------------------------------------------

def test_json_escape_breaks_closing_script_tag():
    raw = '{"a": "</script>"}'
    escaped = _json_for_html_inline_script(raw)
    assert "</script>" not in escaped
    assert "<\\/script>" in escaped


def test_json_escape_is_case_insensitive():
    raw = "</SCRIPT>"
    escaped = _json_for_html_inline_script(raw)
    assert "</SCRIPT>" not in escaped
    assert escaped == "<\\/SCRIPT>"


def test_json_escape_tolerates_whitespace_before_close():
    raw = "</script  >"
    escaped = _json_for_html_inline_script(raw)
    assert "</script  >" not in escaped


def test_json_escape_no_match_is_unchanged():
    raw = '{"a": 1, "b": [1, 2, 3]}'
    assert _json_for_html_inline_script(raw) == raw


def test_json_escape_preserves_json_semantics_after_unescaping():
    import json

    raw = json.dumps({"note": "</script> tag"})
    escaped = _json_for_html_inline_script(raw)
    # A JS engine assigning this as a literal would see "<\/script>" which JSON-decodes
    # back to the original string (backslash before '/' is a valid, no-op JSON escape).
    assert json.loads(escaped) == {"note": "</script> tag"}


# --- _compute_systolic_interval_data -------------------------------------------

def test_compute_systolic_interval_data_observed_from_pairs():
    analysis_data = {"s1_s2_pairs": [(100, 200), (1100, 1200)]}
    obs_t, obs_iv, exp_t, exp_iv = _compute_systolic_interval_data(
        analysis_data, {}, sample_rate=1000, params={}
    )
    assert obs_t == [0.15, 1.15]
    assert obs_iv == [0.1, 0.1]
    assert exp_t == []
    assert exp_iv == []


def test_compute_systolic_interval_data_expected_from_bpm():
    pass_metrics = {"smoothed_bpm": [60.0, 120.0], "bpm_times": [0.0, 1.0]}
    obs_t, obs_iv, exp_t, exp_iv = _compute_systolic_interval_data(
        {}, pass_metrics, sample_rate=1000, params={}
    )
    assert exp_t == [0.0, 1.0]
    assert len(exp_iv) == 2
    assert all(v > 0 for v in exp_iv)


def test_compute_systolic_interval_data_missing_pairs_and_bpm_returns_empty():
    obs_t, obs_iv, exp_t, exp_iv = _compute_systolic_interval_data({}, {}, sample_rate=1000, params={})
    assert (obs_t, obs_iv, exp_t, exp_iv) == ([], [], [], [])


def test_compute_systolic_interval_data_mismatched_bpm_lengths_skips_expected():
    pass_metrics = {"smoothed_bpm": [60.0, 120.0], "bpm_times": [0.0]}
    _, _, exp_t, exp_iv = _compute_systolic_interval_data({}, pass_metrics, sample_rate=1000, params={})
    assert exp_t == []
    assert exp_iv == []


# --- _compute_measured_systole_from_state_boundaries ---------------------------

def test_measured_systole_filters_by_state_name_case_insensitive():
    analysis_data = {
        "pass3_state_boundaries": [
            (0, 100, "Systole", None),
            (100, 300, "diastole", None),
            (300, 400, "systole", None),
        ]
    }
    times, durations = _compute_measured_systole_from_state_boundaries(analysis_data, sample_rate=1000)
    assert len(times) == 2
    assert durations == pytest.approx([0.1, 0.1])


def test_measured_systole_skips_non_positive_duration():
    analysis_data = {"pass3_state_boundaries": [(100, 100, "systole", None), (100, 50, "systole", None)]}
    times, durations = _compute_measured_systole_from_state_boundaries(analysis_data, sample_rate=1000)
    assert times == []
    assert durations == []


def test_measured_systole_swallows_malformed_boundary_tuples():
    analysis_data = {"pass3_state_boundaries": [("not", "numeric", "systole", None), (0, 100, "systole", None)]}
    times, durations = _compute_measured_systole_from_state_boundaries(analysis_data, sample_rate=1000)
    assert len(times) == 1
    assert durations == pytest.approx([0.1])


def test_measured_systole_falls_back_to_before_boundaries_when_after_missing():
    analysis_data = {
        "pass3_state_boundaries": [],
        "pass3_state_boundaries_before": [(0, 100, "systole", None)],
    }
    times, durations = _compute_measured_systole_from_state_boundaries(analysis_data, sample_rate=1000)
    assert len(times) == 1


def test_measured_systole_prefer_before_uses_before_boundaries_even_if_after_present():
    analysis_data = {
        "pass3_state_boundaries": [(0, 100, "systole", None)],
        "pass3_state_boundaries_before": [(0, 200, "systole", None)],
    }
    times, durations = _compute_measured_systole_from_state_boundaries(
        analysis_data, sample_rate=1000, prefer="before"
    )
    assert durations == pytest.approx([0.2])


def test_measured_systole_no_boundaries_returns_empty():
    times, durations = _compute_measured_systole_from_state_boundaries({}, sample_rate=1000)
    assert times == []
    assert durations == []


# --- _compute_measured_diastole_from_state_boundaries ---------------------------

def test_measured_diastole_filters_by_state_name():
    analysis_data = {
        "pass3_state_boundaries": [
            (0, 100, "systole", None),
            (100, 300, "diastole", None),
        ]
    }
    times, durations = _compute_measured_diastole_from_state_boundaries(analysis_data, sample_rate=1000)
    assert durations == pytest.approx([0.2])


# --- _compute_expected_diastole_from_bpm ---------------------------------------

def test_expected_diastole_from_bpm_basic():
    pass_metrics = {"smoothed_bpm": [60.0], "bpm_times": [5.0]}
    times, durations = _compute_expected_diastole_from_bpm(pass_metrics, params={})
    assert times == [5.0]
    assert durations[0] > 0


def test_expected_diastole_from_bpm_missing_keys_returns_empty():
    times, durations = _compute_expected_diastole_from_bpm({}, params={})
    assert times == []
    assert durations == []


# --- _compute_systolic_shift -----------------------------------------------

def test_systolic_shift_returns_none_when_insufficient_data():
    assert _compute_systolic_shift([], [], [], [], None) is None
    assert _compute_systolic_shift([1.0], [0.1], [1.0], [0.1], None) is None  # exp_t < 2


def test_systolic_shift_computes_average_offset_all_time():
    obs_t = [0.0, 1.0, 2.0]
    obs_iv = [0.5, 0.5, 0.5]
    exp_t = [0.0, 2.0]
    exp_iv = [0.3, 0.3]
    shift = _compute_systolic_shift(obs_t, obs_iv, exp_t, exp_iv, peak_bpm_time_sec=None)
    assert shift == pytest.approx(0.2)


def test_systolic_shift_exertion_only_masks_by_peak_time():
    obs_t = [0.0, 1.0, 5.0]
    obs_iv = [1.0, 1.0, 100.0]  # the point at t=5 is after the peak and should be excluded
    exp_t = [0.0, 5.0]
    exp_iv = [0.0, 0.0]
    shift = _compute_systolic_shift(obs_t, obs_iv, exp_t, exp_iv, peak_bpm_time_sec=2.0)
    assert shift == pytest.approx(1.0)


def test_systolic_shift_empty_mask_after_peak_filter_returns_none():
    obs_t = [5.0, 6.0]
    obs_iv = [1.0, 1.0]
    exp_t = [0.0, 10.0]
    exp_iv = [0.0, 0.0]
    shift = _compute_systolic_shift(obs_t, obs_iv, exp_t, exp_iv, peak_bpm_time_sec=0.0)
    assert shift is None
