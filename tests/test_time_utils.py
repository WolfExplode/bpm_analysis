"""Pure time/grid/raster helpers."""
import datetime

import numpy as np

import time_utils as tu


def test_seconds_to_datetime_uses_fixed_epoch():
    # Must be timezone-independent (no fromtimestamp local shift).
    assert tu.seconds_to_datetime(0.0) == datetime.datetime(1970, 1, 1)
    assert tu.seconds_to_datetime(3661.0) == datetime.datetime(1970, 1, 1, 1, 1, 1)


def test_dense_time_grid_basic():
    g = tu.dense_time_grid(1.0, 0.25)
    np.testing.assert_allclose(g, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_dense_time_grid_invalid_inputs_return_empty_or_default_dt():
    assert tu.dense_time_grid(0.0).size == 0
    assert tu.dense_time_grid(-5.0).size == 0
    assert tu.dense_time_grid(float("nan")).size == 0
    # Bad dt falls back to STANDARD_DT_SEC, still produces a grid.
    g = tu.dense_time_grid(1.0, 0.0)
    assert g.size > 0
    np.testing.assert_allclose(g[1] - g[0], tu.STANDARD_DT_SEC)


def test_rasterize_linear_interpolates_and_extrapolates_constant():
    t = np.array([1.0, 2.0, 3.0])
    y = np.array([10.0, 20.0, 30.0])
    grid = np.array([0.0, 1.5, 2.0, 5.0])
    out = tu.rasterize_timeseries_linear(t, y, grid)
    # left edge held at y[0]=10, mid interpolated, right edge held at y[-1]=30
    np.testing.assert_allclose(out, [10.0, 15.0, 20.0, 30.0])


def test_rasterize_linear_sorts_unsorted_input():
    t = np.array([3.0, 1.0, 2.0])
    y = np.array([30.0, 10.0, 20.0])
    grid = np.array([1.5])
    out = tu.rasterize_timeseries_linear(t, y, grid)
    np.testing.assert_allclose(out, [15.0])


def test_rasterize_linear_too_few_points_uses_fallback():
    grid = np.array([0.0, 1.0])
    out = tu.rasterize_timeseries_linear(np.array([1.0]), np.array([5.0]), grid, fallback=7.0)
    np.testing.assert_allclose(out, [7.0, 7.0])
    # No fallback -> NaN
    out2 = tu.rasterize_timeseries_linear(np.array([]), np.array([]), grid)
    assert np.all(np.isnan(out2))


def test_rasterize_empty_grid_returns_empty():
    out = tu.rasterize_timeseries_linear(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([]))
    assert out.size == 0
