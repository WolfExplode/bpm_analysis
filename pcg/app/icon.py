"""The app icon, drawn in code from the theme: three heartbeats as bars, a tall coral S1 then a short
cyan S2, on a dark rounded square. Drawn per size so small sizes stay crisp."""
from __future__ import annotations

import sys
from typing import Iterable

from PySide6 import QtCore, QtGui

from . import theme
from .theme import qcolor

SIZES = (16, 24, 32, 48, 64, 128, 256)
BEATS = 3
APP_ID = "pcg.workspace"  # Windows groups taskbar buttons by this; without it they take python.exe's icon


def render(size: int) -> QtGui.QImage:
    img = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32)
    img.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    u = size / 100.0
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(qcolor(theme.SURFACE_1))
    p.drawRoundedRect(QtCore.QRectF(0, 0, size, size), 22 * u, 22 * u)
    p.setPen(QtGui.QPen(qcolor(theme.BORDER_STRONG), max(1.0, 1.5 * u)))
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QtCore.QRectF(0.5 * u, 0.5 * u, size - u, size - u), 21.5 * u, 21.5 * u)

    # Bars: each beat is S1 (tall) + S2 (short), sharing a baseline; fewer, fatter beats when tiny.
    beats = 2 if size <= 24 else BEATS
    bar_w, gap, beat_gap = (14, 4, 12) if size <= 24 else (9, 3, 9)
    content = beats * (2 * bar_w + gap) + (beats - 1) * beat_gap
    x = (100 - content) / 2
    base = 80
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    for _ in range(beats):
        for height, color in ((46, theme.S1), (28, theme.S2)):
            p.setBrush(qcolor(color))
            p.drawRoundedRect(QtCore.QRectF(x * u, (base - height) * u, bar_w * u, height * u), bar_w / 2 * u, bar_w / 2 * u)
            x += bar_w + gap
        x += beat_gap - gap
    # The BPM line the whole tool exists to produce, riding over the beats.
    if size >= 32:
        path = QtGui.QPainterPath()
        pts = [(14, 25), (32, 21), (48, 27), (64, 18), (80, 23), (86, 20)]
        path.moveTo(pts[0][0] * u, pts[0][1] * u)
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            path.quadTo(x1 * u + (x2 - x1) * u / 2, y1 * u, x2 * u, y2 * u)
        pen = QtGui.QPen(qcolor(theme.METRIC), 3.4 * u)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawPath(path)
    p.end()
    return img


def app_icon(sizes: Iterable[int] = SIZES) -> QtGui.QIcon:
    icon = QtGui.QIcon()
    for s in sizes:
        icon.addPixmap(QtGui.QPixmap.fromImage(render(s)))
    return icon


def set_taskbar_identity() -> None:
    """Give the process its own Windows identity so the taskbar shows our icon, not python's. Call
    before the first window is shown."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
