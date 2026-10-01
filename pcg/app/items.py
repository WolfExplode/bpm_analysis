"""PyQtGraph items for spans: state rows and translucent time bands.

Both draw only what is in view. When more spans are visible than there are pixels,
rows are rasterised per pixel column (one QImage per paint) instead of painting
tens of thousands of rectangles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui

STATE_COLORS = {
    "S1": "#e36f6f",
    "S2": "#f0a040",
    "systole": "#5b4a75",
    "diastole": "#2f5f4a",
    "unknown": "#555555",
    "noisy": "#8a8a8a",
}


def qcolor(c: str, alpha: int = 255) -> QtGui.QColor:
    col = QtGui.QColor(c or "#cccccc")
    col.setAlpha(alpha)
    return col


@dataclass
class SpanRow:
    """One horizontal row of spans between y0 and y1 (data coordinates)."""
    starts: np.ndarray
    ends: np.ndarray
    colors: List[QtGui.QColor]     # palette
    color_idx: np.ndarray          # palette index per span
    y0: float
    y1: float
    outline: Optional[QtGui.QColor] = None  # drawn around each span (rect mode only)

    def __post_init__(self) -> None:
        order = np.argsort(self.starts, kind="stable")
        self.starts = np.asarray(self.starts, dtype=np.float64)[order]
        self.ends = np.asarray(self.ends, dtype=np.float64)[order]
        self.color_idx = np.asarray(self.color_idx, dtype=np.int32)[order]
        self._cummax_end = np.maximum.accumulate(self.ends) if len(self.ends) else self.ends

    def visible(self, t0: float, t1: float) -> slice:
        lo = int(np.searchsorted(self._cummax_end, t0, side="right"))
        hi = int(np.searchsorted(self.starts, t1, side="left"))
        return slice(lo, max(lo, hi))


def palette_row(starts, ends, labels: Sequence[str], y0: float, y1: float,
                colors: Optional[dict] = None, alpha: int = 255, outline: Optional[str] = None) -> SpanRow:
    colors = colors or STATE_COLORS
    names = sorted(set(labels)) or ["unknown"]
    pal = [qcolor(colors.get(n, "#999999"), alpha) for n in names]
    lookup = {n: i for i, n in enumerate(names)}
    return SpanRow(np.asarray(starts), np.asarray(ends), pal, np.array([lookup[x] for x in labels], dtype=np.int32),
                   y0, y1, qcolor(outline) if outline else None)


class SpanRowsItem(pg.GraphicsObject):
    """Rows of coloured spans; x in seconds, y in the lane's data coordinates."""

    def __init__(self, rows: Optional[List[SpanRow]] = None):
        super().__init__()
        self.rows: List[SpanRow] = rows or []
        self._bounds = QtCore.QRectF()
        self._update_bounds()

    def set_rows(self, rows: List[SpanRow]) -> None:
        self.prepareGeometryChange()
        self.rows = rows
        self._update_bounds()
        self.update()

    def _update_bounds(self) -> None:
        xs = [r.starts[0] for r in self.rows if len(r.starts)] + [r._cummax_end[-1] for r in self.rows if len(r.ends)]
        ys = [r.y0 for r in self.rows] + [r.y1 for r in self.rows]
        if xs and ys:
            self._bounds = QtCore.QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        else:
            self._bounds = QtCore.QRectF()

    def boundingRect(self) -> QtCore.QRectF:
        return self._bounds

    def paint(self, p: QtGui.QPainter, *_args) -> None:
        vb = self.getViewBox()
        if vb is None:
            return
        (t0, t1), _ = vb.viewRange()
        px_w = max(1, int(vb.width()))
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        for row in self.rows:
            paint_row(p, row, t0, t1, px_w, row.y0, row.y1)


class BandsItem(pg.GraphicsObject):
    """Translucent full-height time bands (debug windows, Noisy spans, the loop region...)."""

    def __init__(self, starts, ends, color: str, alpha: int = 60):
        super().__init__()
        self.row = SpanRow(np.asarray(starts), np.asarray(ends), [qcolor(color, alpha)],
                           np.zeros(len(starts), dtype=np.int32), 0.0, 1.0)
        self.setZValue(-10)

    def boundingRect(self) -> QtCore.QRectF:
        vb = self.getViewBox()
        if vb is None or not len(self.row.starts):
            return QtCore.QRectF()
        (_, _), (y0, y1) = vb.viewRange()
        return QtCore.QRectF(float(self.row.starts[0]), y0, float(self.row._cummax_end[-1] - self.row.starts[0]), y1 - y0)

    def viewRangeChanged(self) -> None:
        self.prepareGeometryChange()
        super().viewRangeChanged()

    def paint(self, p: QtGui.QPainter, *_args) -> None:
        vb = self.getViewBox()
        if vb is None:
            return
        (t0, t1), (y0, y1) = vb.viewRange()
        paint_row(p, self.row, t0, t1, max(1, int(vb.width())), y0, y1)


def paint_row(p: QtGui.QPainter, row: SpanRow, t0: float, t1: float, px_w: int, y0: float, y1: float) -> None:
    """Paint the visible part of a row between y0 and y1: rectangles, or a per-pixel raster when dense."""
    sl = row.visible(t0, t1)
    n = sl.stop - sl.start
    if n <= 0 or t1 <= t0:
        return
    if n > px_w // 2:
        _paint_raster(p, row, sl, t0, t1, px_w, y0, y1)
        return
    p.setPen(pg.mkPen(row.outline, width=0) if row.outline is not None else QtCore.Qt.PenStyle.NoPen)
    min_w = (t1 - t0) / px_w  # keep sub-pixel spans visible
    for i in range(sl.start, sl.stop):
        a, b = row.starts[i], row.ends[i]
        if b <= t0:
            continue
        p.setBrush(row.colors[row.color_idx[i]])
        p.drawRect(QtCore.QRectF(a, y0, max(b - a, min_w), y1 - y0))


def _paint_raster(p, row: SpanRow, sl: slice, t0, t1, px_w, y0, y1) -> None:
    centers = t0 + (np.arange(px_w) + 0.5) * (t1 - t0) / px_w
    k = np.searchsorted(row.starts, centers, side="right") - 1
    kk = np.clip(k, 0, max(0, len(row.starts) - 1))
    valid = (k >= 0) & (row.ends[kk] > centers)
    argb = np.array([c.rgba() for c in row.colors], dtype=np.uint32)
    img = np.zeros(px_w, dtype=np.uint32)
    img[valid] = argb[row.color_idx[kk[valid]]]
    # Spans narrower than a pixel can fall between centres: mark the pixel of each span start.
    starts_px = ((row.starts[sl] - t0) / (t1 - t0) * px_w).astype(np.int64)
    ok = (starts_px >= 0) & (starts_px < px_w)
    px, idx = starts_px[ok], row.color_idx[sl][ok]
    empty = img[px] == 0
    img[px[empty]] = argb[idx[empty]]
    buf = img.tobytes()  # QImage does not copy: keep the buffer alive while drawing
    qimg = QtGui.QImage(buf, px_w, 1, QtGui.QImage.Format.Format_ARGB32)
    p.drawImage(QtCore.QRectF(t0, y0, t1 - t0, y1 - y0), qimg)


class TimeAxis(pg.AxisItem):
    """Seconds shown as m:ss(.fff) depending on zoom."""

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            sign = "-" if v < 0 else ""
            v = abs(v)
            m, s = divmod(v, 60.0)
            if spacing >= 1:
                out.append(f"{sign}{int(m)}:{s:02.0f}")
            elif spacing >= 0.1:
                out.append(f"{sign}{int(m)}:{s:04.1f}")
            else:
                out.append(f"{sign}{int(m)}:{s:06.3f}")
        return out


class EnvelopeItem(pg.GraphicsObject):
    """A uniformly sampled line (an envelope) drawn as per-pixel min/max bars when dense.

    Stroking a long zig-zag path through the view transform is what makes zoomed-out
    envelopes slow in Qt's raster engine; per-pixel bars drawn in device space are not.
    A block min/max pyramid keeps the zoomed-out reduction cheap for hour-long data.
    """

    _BLOCK = 256

    def __init__(self, t0: float, dt: float, y: np.ndarray, color: str):
        # Attribute names must not shadow QGraphicsItem methods (x(), y(), ...): Qt calls those.
        super().__init__()
        self.t0, self.dt = float(t0), float(dt)
        self._y = np.ascontiguousarray(y, dtype=np.float32)
        self.pen = pg.mkPen(color, width=1)
        self.pen.setCosmetic(True)
        n = len(self._y) // self._BLOCK
        blocks = self._y[: n * self._BLOCK].reshape(n, self._BLOCK) if n else np.zeros((0, self._BLOCK), np.float32)
        finite = np.where(np.isfinite(blocks), blocks, np.nan)
        with np.errstate(all="ignore"):
            self._bmin = np.nanmin(finite, axis=1) if n else np.zeros(0, np.float32)
            self._bmax = np.nanmax(finite, axis=1) if n else np.zeros(0, np.float32)
        fin = self._y[np.isfinite(self._y)]
        self._ylim = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)

    @property
    def t_end(self) -> float:
        return self.t0 + max(0, len(self._y) - 1) * self.dt

    def boundingRect(self) -> QtCore.QRectF:
        lo, hi = self._ylim
        return QtCore.QRectF(self.t0, lo, self.t_end - self.t0, hi - lo)

    def _index(self, t: float) -> int:
        return int(np.clip(np.floor((t - self.t0) / self.dt), 0, len(self._y)))

    def _minmax(self, i0: int, i1: int, bins: int):
        """Per-bin min and max over y[i0:i1] in `bins` contiguous bins (edges, mins, maxs)."""
        edges = np.linspace(i0, i1, bins + 1).astype(np.int64)
        if (i1 - i0) / bins >= 4 * self._BLOCK and len(self._bmin):
            # Coarse: whole blocks only (a bin spans >= 4 blocks, so edge blocks barely matter).
            b = np.clip(edges // self._BLOCK, 0, len(self._bmin))
            b[1:] = np.maximum(b[1:], b[:-1] + 1)
            b = np.clip(b, 0, len(self._bmin))
            lo, hi = int(b[0]), int(b[-1])
            starts = np.minimum(b[:-1] - lo, max(0, hi - lo - 1))
            return edges, np.minimum.reduceat(self._bmin[lo:hi], starts), np.maximum.reduceat(self._bmax[lo:hi], starts)
        seg = self._y[i0:i1]
        idx = np.clip(edges[:-1] - i0, 0, max(0, len(seg) - 1))
        return edges, np.minimum.reduceat(seg, idx), np.maximum.reduceat(seg, idx)

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if ax == 0:
            return (self.t0, self.t_end)
        if orthoRange is None:
            return self._ylim
        i0, i1 = self._index(orthoRange[0]), self._index(orthoRange[1]) + 1
        if i1 - i0 < 2:
            return self._ylim
        _e, mins, maxs = self._minmax(i0, i1, min(2048, i1 - i0))
        with np.errstate(all="ignore"):
            lo, hi = float(np.nanmin(mins)), float(np.nanmax(maxs))
        return (lo, hi) if np.isfinite(lo) and np.isfinite(hi) else self._ylim

    def _paint_dense(self, p, vb, i0, i1, px, sx, ox, sy, oy) -> None:
        """One column per pixel, filled from the bin's min to its max, rasterised with numpy."""
        edges, mins, maxs = self._minmax(i0, i1, px)
        # Join each column to the previous bin's last sample so steep edges stay continuous.
        prev = self._y[np.clip(edges[:-1] - 1, 0, len(self._y) - 1)]
        mins, maxs = np.fmin(mins, prev), np.fmax(maxs, prev)
        (_, _), (yv0, yv1) = vb.viewRange()
        top_dev, bot_dev = sorted((yv1 * sy + oy, yv0 * sy + oy))
        h = int(min(4096, max(1, np.ceil(bot_dev - top_dev))))
        a = mins * sy + oy - top_dev
        b = maxs * sy + oy - top_dev
        lo = np.floor(np.minimum(a, b))
        hi = np.ceil(np.maximum(a, b))
        lo = np.where(np.isfinite(lo), lo, h + 1)
        hi = np.where(np.isfinite(hi), hi, -1)
        rows = np.arange(h, dtype=np.float64)[:, None]
        mask = (rows >= lo[None, :] - 0.5) & (rows <= hi[None, :] + 0.5)
        img = np.where(mask, np.uint32(self.pen.color().rgba()), np.uint32(0)).astype(np.uint32)
        buf = np.ascontiguousarray(img).tobytes()  # QImage does not copy: keep alive while drawing
        qimg = QtGui.QImage(buf, img.shape[1], h, QtGui.QImage.Format.Format_ARGB32)
        x_left = (self.t0 + edges[0] * self.dt) * sx + ox
        x_right = (self.t0 + edges[-1] * self.dt) * sx + ox
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, False)
        p.drawImage(QtCore.QRectF(x_left, top_dev, x_right - x_left, h), qimg)

    def paint(self, p: QtGui.QPainter, *_args) -> None:
        vb = self.getViewBox()
        if vb is None or len(self._y) < 2:
            return
        (v0, v1), _ = vb.viewRange()
        i0, i1 = self._index(v0), min(len(self._y), self._index(v1) + 2)
        if i1 - i0 < 2:
            return
        tr = p.transform()  # item -> device; (self.deviceTransform() crashes inside paint on PySide6 6.11)
        px = max(1, int(vb.width()))
        p.save()
        p.resetTransform()
        p.setPen(self.pen)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        sx, ox, sy, oy = tr.m11(), tr.m31(), tr.m22(), tr.m32()
        if i1 - i0 > 2 * px:
            self._paint_dense(p, vb, i0, i1, px, sx, ox, sy, oy)
        else:
            t = self.t0 + np.arange(i0, i1) * self.dt
            path = pg.arrayToQPath(t * sx + ox, self._y[i0:i1] * sy + oy, connect="finite")
            p.drawPath(path)
        p.restore()
