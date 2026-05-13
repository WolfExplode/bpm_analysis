"""Time conversion and formatting for reports and plots."""
import datetime
from typing import Tuple, Optional

import numpy as np


def timestamp_str() -> str:
    """Return current local time as 'YYYY-MM-DD HH:MM:SS' for reports and logs."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def seconds_to_datetime(seconds: float) -> datetime.datetime:
    """Elapsed seconds since Unix epoch -> timezone-naive datetime (for Plotly/pandas)."""
    # Use a fixed epoch instead of fromtimestamp(0), which can shift to 1969 in local time zones.
    return datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=seconds)


# ─────────────────────────────────────────────────────────────────────────────
# Standardized dense time raster (dt=0.05s)
# ─────────────────────────────────────────────────────────────────────────────

STANDARD_DT_SEC = 0.05


def dense_time_grid(duration_sec: float, dt_sec: float = STANDARD_DT_SEC) -> np.ndarray:
    """Return a dense time grid [0, duration] with step dt_sec."""
    dur = float(duration_sec)
    dt = float(dt_sec)
    if not np.isfinite(dur) or dur <= 0:
        return np.asarray([], dtype=np.float64)
    if not np.isfinite(dt) or dt <= 0:
        dt = STANDARD_DT_SEC
    return np.arange(0.0, dur + 1e-12, dt, dtype=np.float64)


def rasterize_timeseries_linear(
    t_data: np.ndarray,
    y_data: np.ndarray,
    t_grid: np.ndarray,
    *,
    fallback: Optional[float] = None,
) -> np.ndarray:
    """
    Linear interpolation of irregular samples onto a dense grid with constant edge extrapolation.
    If fallback is provided, non-finite outputs are replaced with it.
    """
    t_data = np.asarray(t_data, dtype=np.float64)
    y_data = np.asarray(y_data, dtype=np.float64)
    t_grid = np.asarray(t_grid, dtype=np.float64)
    if len(t_grid) == 0:
        return np.asarray([], dtype=np.float64)
    if len(t_data) < 2 or len(t_data) != len(y_data):
        if fallback is None:
            return np.full_like(t_grid, np.nan, dtype=np.float64)
        return np.full_like(t_grid, float(fallback), dtype=np.float64)
    order = np.argsort(t_data)
    t = t_data[order]
    y = y_data[order]
    out = np.interp(t_grid, t, y, left=float(y[0]), right=float(y[-1])).astype(np.float64)
    if fallback is not None:
        out[~np.isfinite(out)] = float(fallback)
    return out
