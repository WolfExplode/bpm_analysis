"""Signal-lane gain handles and audition controls. Edits last for the open recording."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from . import theme
from .gain import GainEdit, gain_envelope
from .theme import qcolor


class _Handles(pg.ScatterPlotItem):
    def mouseClickEvent(self, ev):
        # Let the lane route clicks and drags through the same hit testing.
        ev.ignore()


class GainEditor(QtWidgets.QWidget):
    def __init__(self, ws):
        super().__init__(ws)
        self.ws = ws
        self.plot = ws.lanes["signal"].plot
        self.vb = self.plot.getViewBox()
        self.edits: tuple[GainEdit, ...] = ()
        self.selected = None
        self.add_btn = QtWidgets.QPushButton("+", checkable=True)
        self.add_btn.setFixedSize(30, 30)
        self.add_btn.setFont(theme.mono_font(14))
        self.add_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.add_btn.setToolTip("Add gain point: click +, then click the Signal graph. Click + again to cancel.")
        self.buttons = []
        self.row = QtWidgets.QHBoxLayout(self)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(5)
        self.row.addWidget(self.add_btn)
        self.curve = pg.PlotDataItem(pen=pg.mkPen(theme.GAIN, width=2), fillLevel=0,
                                     brush=qcolor(theme.GAIN_FILL, 70))
        self.rings = _Handles()
        self.dots = _Handles()
        self.label = pg.TextItem("Playback gain: 0 dB", color=theme.GAIN, anchor=(0, 0))
        self.readout = pg.TextItem(color=theme.GAIN, anchor=(0.5, 0))
        self.band_curves = []
        self.numbers = []
        for item in (self.curve, self.rings, self.dots, self.label, self.readout):
            item.setZValue(40)
            self.plot.addItem(item, ignoreBounds=True)
        self.vb.sigRangeChanged.connect(self.draw)
        self.add_btn.toggled.connect(self.show_editor)
        self.draw()

    def show_editor(self, on):
        if on:
            self.ws.set_lane_visible("signal", True)
            self.ensure_space()
        self.ws.lanes["signal"].widget.viewport().setCursor(
            QtCore.Qt.CursorShape.CrossCursor if on else QtCore.Qt.CursorShape.ArrowCursor)
        self.draw()

    def ensure_space(self):
        """Keep the actual signal zero as baseline, with room below it for cuts."""
        lo, hi = self.vb.viewRange()[1]
        hi = max(hi, 1e-6)
        if lo > -hi * 0.65:
            self.plot.setYRange(-hi * 0.65, hi, padding=0)

    def _scales(self):
        lo, hi = self.vb.viewRange()[1]
        span = max(hi - lo, 1e-12)
        return max(-lo, span * 0.1) * 0.85 / 60, max(hi, span * 0.1) * 0.85 / 12

    def y_of(self, db):
        down, up = self._scales()
        db = np.asarray(db)
        return db * np.where(db < 0, down, up)

    def db_of(self, y):
        down, up = self._scales()
        return float(np.clip(y / (down if y < 0 else up), -60, 12))

    def hit(self, scene_pos):
        for i, edit in enumerate(self.edits):
            pos = self.vb.mapViewToScene(QtCore.QPointF(edit.center, float(self.y_of(edit.db))))
            if (pos - scene_pos).manhattanLength() <= 24:
                return i
        return None

    def select(self, index):
        self.selected = index
        if index is not None:
            if self.ws.selected is not None:
                self.ws.selected = None
                self.ws._draw_annotation()
            self.ws.lanes["signal"].widget.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
        self.draw()

    def add(self, x, db=0.0, width=0.2):
        if not self.ws.player.loaded:
            return None
        self.ensure_space()
        edit = GainEdit(float(np.clip(x, 0, self.ws.player.duration)),
                        width, db, shape="bell")
        self.edits += (edit,)
        self.select(len(self.edits) - 1)
        self.add_btn.setChecked(False)
        self.publish()
        return self.selected

    def toggle(self, index):
        edit = self.edits[index]
        self.edits = self.edits[:index] + (replace(edit, enabled=not edit.enabled),) + self.edits[index + 1:]
        self.publish()

    def wheel(self, ev):
        if not self.edits:
            return False
        index = self.hit(ev.scenePos())
        if index is None:
            return False
        ev.accept()
        if not ev.delta():
            return True
        edit = self.edits[index]
        q = float(np.clip(edit.q * 1.15 ** (ev.delta() / 120), 0.1, 100))
        self.edits = self.edits[:index] + (replace(edit, q=q, shape="bell"),) + self.edits[index + 1:]
        self.select(index)
        self.publish()
        return True

    def move(self, index, x, y):
        edit = replace(self.edits[index], center=float(np.clip(x, 0, self.ws.player.duration)), db=self.db_of(y))
        self.edits = self.edits[:index] + (edit,) + self.edits[index + 1:]
        self.select(index)
        self.publish()

    def remove(self, index):
        self.edits = self.edits[:index] + self.edits[index + 1:]
        self.select(None)
        self.publish()

    def delete_selected(self):
        if self.selected is not None:
            self.remove(self.selected)
            return True
        return False

    def clear_recording(self):
        self.edits = ()
        self.select(None)
        self.add_btn.setChecked(False)
        self.publish()

    def publish(self, *_):
        self.ws.player.set_gain_edits(self.edits)
        self.draw()

    def draw(self, *_):
        self._draw_buttons()
        visible = self.add_btn.isChecked() or bool(self.edits)
        for item in (self.curve, self.rings, self.dots, self.label, *self.band_curves, *self.numbers):
            item.setVisible(visible)
        self.readout.setVisible(visible and self.selected is not None)
        if not visible:
            return
        a, b = self.vb.viewRange()[0]
        # Sample each bell independently so high-Q bands stay smooth at any zoom.
        edges = []
        for e in self.edits:
            if e.shape == "bell":
                edges.extend(e.center + np.linspace(-4, 4, 121) * e.bandwidth)
            else:
                fade = min(0.01, e.width / 4)
                edges.extend((e.center - e.width / 2, e.center - e.width / 2 + fade,
                              e.center, e.center + e.width / 2 - fade, e.center + e.width / 2))
        times = np.unique(np.r_[np.linspace(a, b, 700), [t for t in edges if a <= t <= b]])
        gain = gain_envelope(times, self.edits)
        db = np.clip(20 * np.log10(np.maximum(gain, 0.001)), -60, 12)
        self.curve.setData(times, self.y_of(db))
        while len(self.numbers) > len(self.edits):
            self.plot.removeItem(self.numbers.pop())
            self.plot.removeItem(self.band_curves.pop())
        while len(self.numbers) < len(self.edits):
            number = pg.TextItem(anchor=(0.5, 0.5))
            curve = pg.PlotDataItem(fillLevel=0)
            for item, z in ((number, 43), (curve, 39)):
                item.setZValue(z)
                self.plot.addItem(item, ignoreBounds=True)
            self.numbers.append(number)
            self.band_curves.append(curve)
        colors = [theme.GAIN_COLORS[i % len(theme.GAIN_COLORS)] if e.enabled else theme.TEXT_3
                  for i, e in enumerate(self.edits)]
        xs, ys = [e.center for e in self.edits], [float(self.y_of(e.db)) for e in self.edits]
        self.rings.setData(xs, ys, size=[36 if i == self.selected else 30 for i in range(len(xs))],
                           brush=qcolor(theme.SURFACE_0, 220),
                           pen=[pg.mkPen(qcolor(c, 170), width=2) for c in colors])
        self.dots.setData(xs, ys, size=23, brush=qcolor(theme.SURFACE_1, 230),
                          pen=[pg.mkPen(c, width=2) for c in colors])
        for i, (e, color, number, curve) in enumerate(zip(self.edits, colors, self.numbers, self.band_curves)):
            number.setText(str(i + 1), color=color)
            number.setPos(xs[i], ys[i])
            band_gain = gain_envelope(times, (e,))
            band_db = np.clip(20 * np.log10(np.maximum(band_gain, 0.001)), -60, 12)
            curve.setData(times, self.y_of(band_db))
            curve.setPen(pg.mkPen(qcolor(color, 100), width=1))
            curve.setBrush(qcolor(color, 25))
            curve.setVisible(e.enabled)
        text = "Playback gain: 0 dB"
        if self.selected is not None:
            e = self.edits[self.selected]
            level = "Off" if not e.enabled else "Mute" if e.db <= -60 else f"{e.db:+.1f} dB"
            self.readout.setText(f"{e.center:.3f} s · {level}\nQ {e.q:.2f} · {e.bandwidth:.3f} s" if e.shape == "bell"
                                 else f"{e.center:.3f} s · {level}\nRegion {e.width:.3f} s", color=colors[self.selected])
            self.readout.setPos(e.center, float(self.y_of(e.db)) - (self.vb.viewRange()[1][1]
                               - self.vb.viewRange()[1][0]) * 22 / max(1, self.vb.height()))
        self.label.setText(text)
        self.label.setPos(a + (b - a) * 0.01, self.vb.viewRange()[1][1])

    def _draw_buttons(self):
        while len(self.buttons) > len(self.edits):
            button = self.buttons.pop()
            self.row.removeWidget(button)
            button.hide()
            button.deleteLater()
        while len(self.buttons) < len(self.edits):
            index = len(self.buttons)
            button = QtWidgets.QPushButton(str(index + 1), checkable=True)
            button.setFixedSize(30, 30)
            button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _=False, i=index: self.toggle(i))
            self.row.addWidget(button)
            self.buttons.append(button)
        for i, (button, edit) in enumerate(zip(self.buttons, self.edits)):
            button.setChecked(edit.enabled)
            color = theme.GAIN_COLORS[i % len(theme.GAIN_COLORS)] if edit.enabled else theme.TEXT_3
            button.setStyleSheet(f"color: {color}; border: 1px solid {color}; border-radius: 5px;"
                                f"background: {theme.SURFACE_3 if i == self.selected else theme.SURFACE_1};")
            button.setToolTip(f"Gain point {i + 1}: {'on' if edit.enabled else 'off'}. Click to toggle its effect.")
