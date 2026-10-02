"""The BPM chart export: the BPM curve over the recording's waveform, as a PNG.

The BPM axis is fixed at 0–230 and the image size is fixed, so charts of different recordings line
up and compare at a glance. The waveform has no scale of its own: it is fitted to the bottom of the
chart (its ordinary peaks fill WAVE_SHARE of the height; rare spikes are clipped).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui

from pcg import annotation as an
from pcg.engine.config import param

from . import theme
from .theme import qcolor

BPM_MAX = 230.0
BPM_STEP = 50.0
SIZE = (1920, 660)
WAVE_SHARE = 0.25          # of the plot height, for the waveform's ordinary peaks
MARGINS = (20, 26, 62, 34)  # left, top, right (BPM labels), bottom (time labels)
_TIME_STEPS = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200)

_app: Optional[QtGui.QGuiApplication] = None


def _ensure_app() -> None:
    """Text rendering needs a Qt application; the CLI has none."""
    global _app
    if QtGui.QGuiApplication.instance() is None:
        _app = QtGui.QGuiApplication([])


def _time_label(t: float, hours: bool) -> str:
    t = int(round(t))
    h, rest = divmod(t, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if hours else f"{m + 60 * h}:{s:02d}"


def _column_peaks(env_t0: float, env_dt: float, env_y: np.ndarray, t0: float, t1: float, cols: int) -> np.ndarray:
    """Per pixel column, the waveform's largest magnitude in that column's time span."""
    y = np.abs(np.asarray(env_y, dtype=np.float64))
    if not len(y) or env_dt <= 0:
        return np.zeros(cols)
    edges = ((np.linspace(t0, t1, cols + 1) - env_t0) / env_dt).astype(np.int64)
    edges = np.clip(edges, 0, len(y))
    out = np.zeros(cols)
    nonempty = edges[1:] > edges[:-1]
    if nonempty.any():
        # Each non-empty column runs to the next one's start; the last one stops at the window's end.
        with np.errstate(all="ignore"):
            out[nonempty] = np.fmax.reduceat(y[: edges[-1]], edges[:-1][nonempty])
    return np.nan_to_num(out)


def annotation_bpm_curve(spans, params):
    """(times, BPM) of an Annotation as the workspace draws it: per-beat BPM smoothed with the same
    Gaussian the engine applies to its own curve, so the chart agrees with the app's min and max."""
    t, instant = an.bpm_series(spans)
    window = float(param(params, "output_smoothing_window_sec"))
    return t, an.smooth_series(t, instant, max(0.05, window / 3.0))


def render_bpm_chart(bpm_t, bpm, env_t0: float, env_dt: float, env_y, t0: float, t1: float,
                     size: Tuple[int, int] = SIZE) -> QtGui.QImage:
    """The chart for [t0, t1] (seconds): *bpm* over *bpm_t*, and the uniformly sampled envelope.
    Min and max are labelled from the curve as drawn."""
    _ensure_app()
    w, h = size
    left, top, right, bottom = MARGINS
    plot = QtCore.QRectF(left, top, w - left - right, h - top - bottom)
    bpm_t, bpm = np.asarray(bpm_t, dtype=np.float64), np.asarray(bpm, dtype=np.float64)
    keep = (bpm_t >= t0) & (bpm_t <= t1)
    bpm_t, bpm = bpm_t[keep], bpm[keep]
    span = max(t1 - t0, 1e-9)

    def x_of(t):
        return plot.left() + (np.asarray(t) - t0) / span * plot.width()

    def y_of(b):
        return plot.bottom() - np.asarray(b) / BPM_MAX * plot.height()

    img = QtGui.QImage(w, h, QtGui.QImage.Format.Format_ARGB32)
    img.fill(qcolor(theme.SURFACE_0))
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
    small = theme.mono_font(9)
    p.setFont(small)
    fm = QtGui.QFontMetricsF(small)

    # Grid and axes: BPM on the right, time along the bottom.
    p.setPen(QtGui.QPen(qcolor(theme.LINE), 1))
    b = 0.0
    while b <= BPM_MAX:
        y = float(y_of(b))
        p.drawLine(QtCore.QPointF(plot.left(), y), QtCore.QPointF(plot.right(), y))
        p.setPen(qcolor(theme.TEXT_3))
        p.drawText(QtCore.QPointF(plot.right() + 8, y + fm.ascent() / 2 - 1), f"{b:.0f}")
        p.setPen(QtGui.QPen(qcolor(theme.LINE), 1))
        b += BPM_STEP
    p.setPen(qcolor(theme.TEXT_3))
    p.drawText(QtCore.QPointF(plot.right() + 8, plot.top() - 9), "BPM")
    step = next((s for s in _TIME_STEPS if span / s <= 12), _TIME_STEPS[-1])
    hours = t1 >= 3600
    tick = np.ceil(t0 / step) * step
    while tick <= t1 + 1e-9:
        x = float(x_of(tick))
        p.setPen(QtGui.QPen(qcolor(theme.LINE), 1))
        p.drawLine(QtCore.QPointF(x, plot.top()), QtCore.QPointF(x, plot.bottom()))
        label = _time_label(tick, hours)
        p.setPen(qcolor(theme.TEXT_3))
        lx = min(max(x - fm.horizontalAdvance(label) / 2, 0), w - fm.horizontalAdvance(label))
        p.drawText(QtCore.QPointF(lx, plot.bottom() + fm.height() + 6), label)
        tick += step

    p.save()
    p.setClipRect(plot)
    # Waveform, filled up from the bottom.
    cols = max(1, int(plot.width()))
    peaks = _column_peaks(env_t0, env_dt, env_y, t0, t1, cols)
    if peaks.any():
        ref = float(np.percentile(peaks[peaks > 0], 99.5)) or 1.0
        heights = np.minimum(peaks / ref, 1.15) * WAVE_SHARE * plot.height()
        xs = plot.left() + np.arange(cols) + 0.5
        poly = QtGui.QPolygonF([QtCore.QPointF(plot.left(), plot.bottom())]
                               + [QtCore.QPointF(x, plot.bottom() - hh) for x, hh in zip(xs, heights)]
                               + [QtCore.QPointF(plot.right(), plot.bottom())])
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(qcolor(theme.CHART_WAVE))
        p.drawPolygon(poly)

    # BPM curve, broken where the BPM is missing.
    path = QtGui.QPainterPath()
    pen_down = False
    for x, y, ok in zip(x_of(bpm_t), y_of(bpm), np.isfinite(bpm)):
        if not ok:
            pen_down = False
        elif pen_down:
            path.lineTo(float(x), float(y))
        else:
            path.moveTo(float(x), float(y))
            pen_down = True
    pen = QtGui.QPen(qcolor(theme.CHART_BPM), 2.6)
    pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    p.drawPath(path)
    p.restore()

    # Min and max, each with a leader line to its label.
    finite = np.isfinite(bpm)
    if finite.any():
        idx = np.nonzero(finite)[0]
        i_max, i_min = idx[np.argmax(bpm[idx])], idx[np.argmin(bpm[idx])]
        for i, text, color, prefer_up in ((i_max, f"Max: {bpm[i_max]:.1f} BPM", theme.TONES["danger"][0], True),
                                          (i_min, f"Min: {bpm[i_min]:.1f} BPM", theme.TONES["success"][0], False)):
            _label_point(p, fm, plot, QtCore.QPointF(float(x_of(bpm_t[i])), float(y_of(min(bpm[i], BPM_MAX)))),
                         text, color, prefer_up)
    p.end()
    return img


def _label_point(p: QtGui.QPainter, fm: QtGui.QFontMetricsF, plot: QtCore.QRectF, pt: QtCore.QPointF, text: str,
                 color: str, prefer_up: bool) -> None:
    """A dot on *pt* and a label diagonally away from it (above or below, whichever fits, preferring
    *prefer_up*), on a backing so the curve never runs through the text."""
    dy, dx, pad = 46.0, 34.0, 5.0
    tw, th = fm.horizontalAdvance(text) + 2 * pad, fm.height() + 2 * pad
    fits_up = pt.y() - dy - th > plot.top()
    fits_down = pt.y() + dy + th < plot.bottom()
    up = fits_up if prefer_up else not fits_down
    right = pt.x() + dx + tw < plot.right()  # lean right unless that leaves the plot
    tx = pt.x() + dx if right else pt.x() - dx - tw
    box = QtCore.QRectF(max(plot.left(), tx), pt.y() - dy - th if up else pt.y() + dy, tw, th)
    corner = QtCore.QPointF(box.left() if right else box.right(), box.bottom() if up else box.top())
    p.setPen(QtGui.QPen(qcolor(theme.TEXT_2), 1.2))
    p.drawLine(pt, corner)
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(qcolor(theme.SURFACE_0, 225))
    p.drawRoundedRect(box, 4, 4)
    p.setBrush(qcolor(color))
    p.drawEllipse(pt, 4, 4)
    p.setPen(qcolor(color))
    p.drawText(box, QtCore.Qt.AlignmentFlag.AlignCenter, text)
