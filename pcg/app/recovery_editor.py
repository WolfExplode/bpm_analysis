"""Editable recovery starts on the heart-rate lane, remembered per recording."""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from pcg.clock import clock_text
from . import theme
from .gain_editor import _Handles
from .icon import tool_icon
from .recovery import measure_recovery


class _RecoveryHandles(_Handles):
    def __init__(self, editor):
        super().__init__()
        self.editor = editor

    def mouseClickEvent(self, ev):
        index = self.editor.hit(ev.scenePos())
        if index is None or ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            ev.ignore()
        elif ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            self.editor.select(index)
        else:
            ev.ignore()


class RecoveryEditor:
    def __init__(self, ws):
        self.ws = ws
        self.plot = ws.lanes["bpm"].plot
        self.vb = self.plot.getViewBox()
        self.starts = ()
        self.selected = None
        self.lines = []
        self.items = []
        self.positions = []
        self.source = ""
        self.add_btn = QtWidgets.QPushButton(checkable=True)
        self.add_btn.setIcon(tool_icon("recovery"))
        self.add_btn.setFixedSize(30, 30)
        self.add_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.add_btn.setAccessibleName("Add recovery start")
        self.add_btn.setToolTip("Add recovery start: click here, then click the Heart rate graph. "
                                "Drag a marker to reposition; select and press Delete/Backspace to remove. "
                                "Click again or press Escape to cancel.")
        self.markers = _RecoveryHandles(self)
        self.markers.setZValue(45)
        self.plot.addItem(self.markers, ignoreBounds=True)
        self.hover_label = QtWidgets.QLabel(ws.lanes["bpm"].widget.viewport())
        self.hover_label.setObjectName("cursorReadout")
        self.hover_label.setFont(theme.mono_font())
        self.hover_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hover_label.hide()
        self.add_btn.toggled.connect(self.arm)

    def show_hover(self, scene_pos, text):
        """A stationary hover readout without QToolTip's automatic timeout."""
        widget = self.ws.lanes["bpm"].widget
        viewport = widget.viewport()
        pos = widget.mapFromScene(scene_pos)
        self.hover_label.setText(text)
        self.hover_label.adjustSize()
        x, y = pos.x() + 14, pos.y() + 10
        if x + self.hover_label.width() > viewport.width():
            x = pos.x() - self.hover_label.width() - 14
        if y + self.hover_label.height() > viewport.height():
            y = pos.y() - self.hover_label.height() - 10
        self.hover_label.move(max(0, x), max(0, y))
        self.hover_label.show()
        self.hover_label.raise_()

    def arm(self, on):
        if on:
            self.ws.gain_editor.add_btn.setChecked(False)
            self.ws.set_lane_visible("bpm", True)
        self.ws.lanes["bpm"].widget.viewport().setCursor(
            QtCore.Qt.CursorShape.CrossCursor if on else QtCore.Qt.CursorShape.ArrowCursor)

    def load_recording(self):
        self.add_btn.setChecked(False)
        every = self.ws.settings.get_json("recovery_starts", {})
        self.starts = tuple(sorted(float(t) for t in every.get(self.ws.fingerprint, []) if np.isfinite(t)))
        self.selected = None
        self.draw()

    def publish(self):
        if self.ws.fingerprint:
            every = self.ws.settings.get_json("recovery_starts", {})
            if self.starts:
                every[self.ws.fingerprint] = list(self.starts)
            else:
                every.pop(self.ws.fingerprint, None)
            self.ws.settings.set_json("recovery_starts", every)
        self.draw()

    def select(self, index):
        self.selected = index
        if index is not None:
            self.ws.gain_editor.select(None)
            if self.ws.selected is not None:
                self.ws.selected = None
                self.ws._draw_annotation()
            self.ws.lanes["bpm"].widget.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
        self.draw()

    def add(self, x):
        src = self.ws._bpm_extreme_source() if self.ws.analysis is not None else None
        if src is None:
            return
        t, y = src[:2]
        if not len(t) or x < t[0] or x > t[-1]:
            return
        _, cutoff = measure_recovery(t, y, x)
        if cutoff <= x:
            return
        self.starts += (float(x),)
        self.selected = len(self.starts) - 1
        self.add_btn.setChecked(False)
        self.select(self.selected)
        self.publish()

    def move(self, index, x):
        if self.ws.analysis is None:
            return
        x = float(np.clip(x, 0, self.ws.analysis.duration_sec))
        self.starts = self.starts[:index] + (x,) + self.starts[index + 1:]
        self.selected = index
        self.publish()

    def remove(self, index):
        self.starts = self.starts[:index] + self.starts[index + 1:]
        self.selected = None
        self.publish()

    def delete_selected(self):
        if self.selected is None:
            return False
        self.remove(self.selected)
        return True

    def hit(self, pos):
        for index, x, y in self.positions:
            point = self.vb.mapViewToScene(QtCore.QPointF(x, y))
            if (point - pos).manhattanLength() <= 18:
                return index
        return None

    def draw(self):
        self.hover_label.hide()
        for item in self.items:
            self.plot.removeItem(item)
        self.items.clear()
        self.lines.clear()
        self.positions.clear()
        src = self.ws._bpm_extreme_source() if self.ws.analysis is not None else None
        if src is None:
            self.markers.setData([], [])
            return
        t, y, _, suffix = src
        self.source = "Annotation BPM" if suffix else "Result BPM"
        color = theme.GAIN_COLORS[1]
        for index, start in enumerate(self.starts):
            stop = min((s for s in self.starts if s > start), default=np.inf)
            lines, _ = measure_recovery(t, y, start, stop)
            if not len(t) or start < t[0] or start > t[-1]:
                continue
            bpm = float(np.interp(start, t, y))
            if not np.isfinite(bpm):
                continue
            self.positions.append((index, start, bpm))
            label = pg.TextItem(f"R{index + 1}", color=color, anchor=(0.5, 1.3))
            label.setPos(start, bpm)
            self._add_item(label)
            for line in lines:
                item = pg.PlotDataItem([line.start, line.end], [line.start_bpm, line.end_bpm],
                                      pen=pg.mkPen(theme.qcolor(color, 102), width=1.6))
                self._add_item(item)
                endpoint = _Handles([line.end], [line.end_bpm], symbol="o", size=12,
                                    brush=theme.qcolor(color, 40), pen=pg.mkPen(color, width=2))
                self._add_item(endpoint)
                self.lines.append((index, line))
        self.markers.setData([x for _, x, _ in self.positions], [y for _, _, y in self.positions],
                             symbol="d", size=[16 if i == self.selected else 11 for i, _, _ in self.positions],
                             pen=pg.mkPen(color, width=2), brush=theme.SURFACE_0)

    def _add_item(self, item):
        item.setZValue(42)
        self.plot.addItem(item, ignoreBounds=True)
        self.items.append(item)

    def hover_text(self, pos):
        index = self.hit(pos)
        if index is not None:
            return (f"Recovery {index + 1} start · {clock_text(self.starts[index])}\n{self.source}\n"
                    "Drag to reposition · Select and press Delete/Backspace to remove\n"
                    "Cutoff: sustained rise ≥5 BPM for 10 s, missing data, or next marker")
        best, distance = None, 8.0
        for index, line in self.lines:
            b = self.vb.mapViewToScene(QtCore.QPointF(line.end, line.end_bpm))
            d = np.hypot(pos.x() - b.x(), pos.y() - b.y())
            if d < distance:
                best, distance = (index, line), d
        if best is None:
            return ""
        _, line = best
        return f"Slope: {line.slope:+.3f} bpm/s\nBPM change: {line.change:+.2f} BPM"
