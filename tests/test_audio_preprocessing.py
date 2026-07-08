"""audio_preprocessing: channel-mode validation and the numeric rolling-window helpers."""
import numpy as np
import pandas as pd
import pytest

from audio_preprocessing import (
    CHANNEL_MODE_ALL,
    CHANNEL_MODE_LEFT,
    CHANNEL_MODE_MIXED,
    CHANNEL_MODE_RIGHT,
    _centered_moving_average,
    _dense_troughs_linear_interpolate,
    _rolling_quantile_center_bfill_ffill,
    normalize_channel_mode,
)


# --- normalize_channel_mode ---------------------------------------------------

def test_normalize_channel_mode_defaults_to_mixed_for_none():
    assert normalize_channel_mode(None) == CHANNEL_MODE_MIXED


def test_normalize_channel_mode_defaults_to_mixed_for_empty_string():
    assert normalize_channel_mode("") == CHANNEL_MODE_MIXED


def test_normalize_channel_mode_lowercases_and_strips():
    assert normalize_channel_mode("  LEFT  ") == CHANNEL_MODE_LEFT


@pytest.mark.parametrize("mode", [CHANNEL_MODE_MIXED, CHANNEL_MODE_LEFT, CHANNEL_MODE_RIGHT, CHANNEL_MODE_ALL])
def test_normalize_channel_mode_accepts_all_known_modes(mode):
    assert normalize_channel_mode(mode) == mode


def test_normalize_channel_mode_rejects_unknown_value():
    with pytest.raises(ValueError):
        normalize_channel_mode("surround")


# --- _centered_moving_average --------------------------------------------------

def _pandas_reference(x, window):
    return pd.Series(x).rolling(window, min_periods=1, center=True).mean().to_numpy()


def test_centered_moving_average_window_one_returns_copy_not_same_array():
    x = np.array([1.0, 2.0, 3.0])
    result = _centered_moving_average(x, window=1)
    np.testing.assert_array_equal(result, x)
    assert result is not x


@pytest.mark.parametrize("window", [2, 3, 4, 5])
def test_centered_moving_average_matches_pandas_odd_and_even_windows(window):
    rng = np.random.default_rng(0)
    x = rng.normal(size=37)
    expected = _pandas_reference(x, window)
    actual = _centered_moving_average(x, window)
    np.testing.assert_allclose(actual, expected)


def test_centered_moving_average_matches_pandas_at_edges():
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    for window in (2, 3, 4):
        np.testing.assert_allclose(_centered_moving_average(x, window), _pandas_reference(x, window))


def test_centered_moving_average_window_larger_than_array():
    x = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(_centered_moving_average(x, window=10), _pandas_reference(x, 10))


def test_centered_moving_average_single_element():
    x = np.array([5.0])
    np.testing.assert_allclose(_centered_moving_average(x, window=3), [5.0])


# --- _dense_troughs_linear_interpolate -----------------------------------------

def test_dense_troughs_interpolates_between_two_points():
    result = _dense_troughs_linear_interpolate(5, np.array([0, 4]), np.array([0.0, 40.0]))
    np.testing.assert_allclose(result, [0.0, 10.0, 20.0, 30.0, 40.0])


def test_dense_troughs_flat_extends_past_last_trough():
    # pandas .interpolate(method="linear") does NOT extrapolate past the last known
    # point by default; it forward-fills as flat (holds the last value).
    result = _dense_troughs_linear_interpolate(6, np.array([1, 3]), np.array([10.0, 30.0]))
    expected = pd.Series([np.nan, 10.0, np.nan, 30.0, np.nan, np.nan]).interpolate(method="linear").to_numpy()
    np.testing.assert_allclose(result, expected)


def test_dense_troughs_single_trough_flat_fills_forward_leaves_leading_nan():
    # With only one known point, pandas can't interpolate a slope: it leaves values
    # before the point as NaN and flat-extends the point's value forward.
    result = _dense_troughs_linear_interpolate(3, np.array([1]), np.array([5.0]))
    assert np.isnan(result[0])
    assert result[1] == 5.0
    assert result[2] == 5.0


# --- _rolling_quantile_center_bfill_ffill --------------------------------------

def test_rolling_quantile_matches_pandas_reference():
    rng = np.random.default_rng(1)
    y = rng.normal(size=30)
    expected = (
        pd.Series(y).rolling(window=5, min_periods=3, center=True).quantile(0.5).bfill().ffill().to_numpy()
    )
    actual = _rolling_quantile_center_bfill_ffill(y, window=5, quantile_val=0.5, min_periods=3)
    np.testing.assert_allclose(actual, expected)


def test_rolling_quantile_has_no_nans_after_bfill_ffill():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    result = _rolling_quantile_center_bfill_ffill(y, window=3, quantile_val=0.25, min_periods=2)
    assert not np.any(np.isnan(result))
