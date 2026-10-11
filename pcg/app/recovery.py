"""Recovery measurements and a conservative, sustained-rise cutoff (seconds/BPM)."""
from dataclasses import dataclass

import numpy as np

INTERVALS = (30, 60, 120, 180, 300, 600, 900, 1200, 1800)
END_RADIUS = 2.0
MAX_GAP = 5.0
RISE_BPM = 5.0
RISE_SECONDS = 10.0


@dataclass(frozen=True)
class RecoveryLine:
    start: float
    end: float
    start_bpm: float
    end_bpm: float

    @property
    def change(self):
        return self.end_bpm - self.start_bpm

    @property
    def slope(self):
        return self.change / (self.end - self.start)


def window_mean(t, y, center, radius=END_RADIUS):
    """Time-weighted mean, without extrapolation or bridging missing samples."""
    a, b = center - radius, center + radius
    lo, hi = np.searchsorted(t, [a, b], side="right")
    lo = max(0, lo - 1)
    hi = min(len(t), hi + 1)
    tt, yy = t[lo:hi], y[lo:hi]
    if len(tt) < 2 or tt[0] > a or tt[-1] < b:
        return None
    if not np.isfinite(yy).all() or np.any(np.diff(tt) > MAX_GAP):
        return None
    xs = np.r_[a, tt[(tt > a) & (tt < b)], b]
    values = np.interp(xs, tt, yy)
    return float(np.sum(np.diff(xs) * (values[:-1] + values[1:]) / 2) / (b - a))


def recovery_start_bpm(t, y, start):
    """BPM at the exact marker, only within a continuous finite source interval."""
    if not np.isfinite(start) or not len(t) or start < t[0] or start > t[-1]:
        return None
    i = max(0, np.searchsorted(t, start, side="right") - 1)
    j = min(i + 1, len(t) - 1)
    if not np.isfinite(y[i:j + 1]).all() or t[j] - t[i] > MAX_GAP:
        return None
    return float(np.interp(start, t, y))


def measure_recovery(t, y, start, stop=np.inf):
    """Return lines and cutoff. Stop on missing data, next marker, or sustained rise.

    The rise detector scans every two seconds using +/-2 s means; it ends recovery
    at the first threshold crossing once the rise persists for ten seconds.
    """
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    initial = recovery_start_bpm(t, y, start)
    if initial is None:
        return (), start
    i = max(0, np.searchsorted(t, start, side="right") - 1)
    cutoff = min(stop, t[-1], start + INTERVALS[-1] + END_RADIUS)
    # Never connect a recovery chord across a hole in the source curve.
    for k in range(i, len(t) - 1):
        if t[k] >= cutoff:
            break
        if not np.isfinite(y[k:k + 2]).all() or t[k + 1] - t[k] > MAX_GAP:
            cutoff = min(cutoff, t[k])
            break
    low, rising = initial, None
    for center in np.arange(start + END_RADIUS, cutoff - END_RADIUS + 1e-9, 2.0):
        value = window_mean(t, y, center)
        if value is None:
            cutoff = min(cutoff, center - END_RADIUS)
            break
        low = min(low, value)
        if value >= low + RISE_BPM:
            if rising is None:
                rising = center
            if center - rising >= RISE_SECONDS:
                cutoff = rising - END_RADIUS
                break
        else:
            rising = None
    lines = []
    for seconds in INTERVALS:
        end = start + seconds
        if end + END_RADIUS > cutoff:
            break
        value = window_mean(t, y, end)
        if value is None:
            break
        lines.append(RecoveryLine(start, end, initial, value))
    return tuple(lines), cutoff
