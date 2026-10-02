"""Small widgets of the design language: lane legends, stage chips, buttons with key hints, status pills."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from pcg.engine.traces import KIND_LINE, KIND_POINTS, KIND_SPANS

from . import theme
from .theme import TraceStyle, qcolor

LEGEND_WIDTH = 210


# ─────────────────────────────────────────────────────────────────────────────
# Swatches
# ─────────────────────────────────────────────────────────────────────────────

def _symbol_path(symbol: str, c: QtCore.QPointF, r: float) -> QtGui.QPainterPath:
    path = QtGui.QPainterPath()
    x, y = c.x(), c.y()
    if symbol == "t":
        path.addPolygon(QtGui.QPolygonF([QtCore.QPointF(x, y - r), QtCore.QPointF(x + r, y + r * 0.8),
                                         QtCore.QPointF(x - r, y + r * 0.8)]))
        path.closeSubpath()
    elif symbol == "s":
        path.addRect(QtCore.QRectF(x - r * 0.85, y - r * 0.85, r * 1.7, r * 1.7))
    elif symbol == "d":
        path.addPolygon(QtGui.QPolygonF([QtCore.QPointF(x, y - r), QtCore.QPointF(x + r, y),
                                         QtCore.QPointF(x, y + r), QtCore.QPointF(x - r, y)]))
        path.closeSubpath()
    elif symbol == "x":
        path.moveTo(x - r, y - r)
        path.lineTo(x + r, y + r)
        path.moveTo(x - r, y + r)
        path.lineTo(x + r, y - r)
    else:
        path.addEllipse(c, r, r)
    return path


def paint_swatch(p: QtGui.QPainter, rect: QtCore.QRectF, style: TraceStyle, kind: str) -> None:
    """A miniature of how the trace is drawn: its colour, dash pattern, marker or fill."""
    p.save()
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    color = qcolor(style.color)
    mid = rect.center()
    if kind == KIND_SPANS:
        p.fillRect(rect.adjusted(0, 1, 0, -1), qcolor(style.color, 110))
    elif kind == KIND_POINTS:
        path = _symbol_path(style.symbol, mid, 3.2)
        if style.hollow or style.symbol == "x":
            p.setPen(QtGui.QPen(color, 1.2))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        else:
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(color)
        p.drawPath(path)
    elif style.fill:
        body = QtCore.QRectF(rect.left(), mid.y() - 2, rect.width(), rect.bottom() - mid.y() + 1)
        p.fillRect(body, qcolor(theme.ENVELOPE_FILL))
        p.setPen(QtGui.QPen(color, 1))
        p.drawLine(QtCore.QPointF(rect.left(), mid.y() - 2), QtCore.QPointF(rect.right(), mid.y() - 2))
    else:
        pen = theme.pen(style)
        pen.setWidthF(max(1.2, style.width))
        p.setPen(pen)
        p.drawLine(QtCore.QPointF(rect.left(), mid.y()), QtCore.QPointF(rect.right(), mid.y()))
    p.restore()


# ─────────────────────────────────────────────────────────────────────────────
# Lane legend
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LegendEntry:
    name: str           # the trace's name (key for toggling)
    label: str          # what the legend shows (the name without its pass)
    tag: str            # the stage, short ("P2", "result")
    kind: str
    style: TraceStyle
    shown: bool
    edge: str = ""      # stage colour for debug-stage entries; "" for everyday ones
    divider: bool = False  # first entry of a new stage


class _EntryRow(QtWidgets.QWidget):
    HEIGHT = 20

    def __init__(self, entry: LegendEntry, legend: "LaneLegend"):
        super().__init__()
        self.entry = entry
        self.legend = legend
        self._hover = False
        self.setFixedHeight(self.HEIGHT)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setToolTip(entry.name)
        self.setMouseTracking(True)

    def enterEvent(self, ev) -> None:
        self._hover = True
        self.update()
        self.legend.hovered.emit(self.entry.name)

    def leaveEvent(self, ev) -> None:
        self._hover = False
        self.update()
        self.legend.hovered.emit(None)

    def mousePressEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.entry.shown = not self.entry.shown
            self.update()
            self.legend.toggled.emit(self.entry.name, self.entry.shown)

    def paintEvent(self, _ev) -> None:
        e = self.entry
        p = QtGui.QPainter(self)
        r = self.rect()
        if self._hover:
            p.fillRect(r, qcolor(theme.SURFACE_3))
        if e.edge:
            p.fillRect(QtCore.QRect(0, 2, 2, r.height() - 4), qcolor(e.edge, 255 if e.shown else 110))
        if not e.shown:
            p.setOpacity(0.35)
        paint_swatch(p, QtCore.QRectF(10, 5, 16, r.height() - 10), e.style, e.kind)
        p.setOpacity(1.0)
        tag_w = 0
        if e.tag:
            p.setFont(theme.mono_font(7.5))
            p.setPen(qcolor(theme.TEXT_4 if not e.shown else theme.TEXT_3))
            fm = p.fontMetrics()
            tag_w = fm.horizontalAdvance(e.tag) + 8
            p.drawText(QtCore.QRect(r.width() - tag_w, 0, tag_w - 6, r.height()),
                       QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter, e.tag)
        p.setFont(self.font())
        p.setPen(qcolor(theme.TEXT if e.shown else theme.TEXT_4))
        text_rect = QtCore.QRect(34, 0, r.width() - 34 - tag_w - 4, r.height())
        text = p.fontMetrics().elidedText(e.label, QtCore.Qt.TextElideMode.ElideRight, text_rect.width())
        p.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, text)


class _Divider(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(7)

    def paintEvent(self, _ev) -> None:
        p = QtGui.QPainter(self)
        p.fillRect(QtCore.QRect(8, 3, self.width() - 16, 1), qcolor(theme.LINE))


class _DragHeader(QtWidgets.QWidget):
    """The lane title row: dragging it (past a few pixels) reports the pointer's global y."""

    moved = QtCore.Signal(int)
    dropped = QtCore.Signal(int)

    def __init__(self):
        super().__init__()
        self._press: Optional[QtCore.QPoint] = None
        self._dragging = False
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        self.setToolTip("Drag to reorder this lane")

    def mousePressEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self._press = ev.globalPosition().toPoint()

    def mouseMoveEvent(self, ev) -> None:
        if self._press is None:
            return
        pos = ev.globalPosition().toPoint()
        if not self._dragging and abs(pos.y() - self._press.y()) >= 4:
            self._dragging = True
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        if self._dragging:
            self.moved.emit(pos.y())

    def mouseReleaseEvent(self, ev) -> None:
        if self._dragging:
            self.dropped.emit(ev.globalPosition().toPoint().y())
        self._press, self._dragging = None, False
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)


class _LinkLabel(QtWidgets.QLabel):
    """A line of text that runs a callback when clicked."""

    def __init__(self, text: str, tip: str, on_click: Callable[[], None]):
        super().__init__(text)
        self._on_click = on_click
        self.setToolTip(tip)
        self.setFont(theme.mono_font(8.5))
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setContentsMargins(10, 1, 4, 1)

    def mousePressEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self._on_click()


class LaneLegend(QtWidgets.QFrame):
    """The column left of a lane: its title and one entry per trace it can draw.

    Hovering an entry isolates that trace in the lane; clicking toggles it."""

    toggled = QtCore.Signal(str, bool)
    hovered = QtCore.Signal(object)   # trace name, or None when the pointer leaves
    hide_requested = QtCore.Signal()
    drag_moved = QtCore.Signal(int)    # global y while the title is being dragged
    drag_dropped = QtCore.Signal(int)

    def __init__(self, title: str, unit: str = ""):
        super().__init__()
        self.setObjectName("laneLegend")
        self.setFixedWidth(LEGEND_WIDTH)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 2)
        lay.setSpacing(2)
        header = _DragHeader()
        header.moved.connect(self.drag_moved)
        header.dropped.connect(self.drag_dropped)
        head = QtWidgets.QHBoxLayout(header)
        head.setContentsMargins(10, 0, 4, 0)
        head.setSpacing(6)
        t = QtWidgets.QLabel(title)
        t.setObjectName("laneTitle")
        head.addWidget(t)
        if unit:
            u = QtWidgets.QLabel(unit)
            u.setProperty("tone", "muted")
            head.addWidget(u)
        head.addStretch(1)
        close = QtWidgets.QToolButton(text="×")
        close.setToolTip("Hide this lane")
        close.clicked.connect(self.hide_requested)
        head.addWidget(close)
        lay.addWidget(header)
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.viewport().setAutoFillBackground(False)
        self._body = QtWidgets.QWidget()
        self._body.setAutoFillBackground(False)
        self._rows = QtWidgets.QVBoxLayout(self._body)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(0)
        self._rows.addStretch(1)
        self._scroll.setWidget(self._body)
        lay.addWidget(self._scroll, 1)
        self._footer = QtWidgets.QVBoxLayout()
        self._footer.setContentsMargins(0, 0, 0, 2)
        self._footer.setSpacing(0)
        lay.addLayout(self._footer)
        self._footer_key: list = []
        self.entries: List[LegendEntry] = []

    def set_footer(self, rows: List[Tuple[str, str, Callable[[], None]]]) -> None:
        """Clickable readout lines under the entries: (text, tooltip, on_click). [] clears them."""
        key = [(text, tip) for text, tip, _ in rows]
        if key == self._footer_key:
            return
        self._footer_key = key
        while self._footer.count():
            w = self._footer.takeAt(0).widget()
            if w is not None:
                w.hide()  # a deleteLater() widget is still painted until the event loop runs
                w.setParent(None)
                w.deleteLater()
        for text, tip, on_click in rows:
            self._footer.addWidget(_LinkLabel(text, tip, on_click))

    def set_entries(self, entries: List[LegendEntry]) -> None:
        self.entries = entries
        while self._rows.count() > 1:
            w = self._rows.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        for i, e in enumerate(entries):
            if e.divider and i > 0:
                self._rows.insertWidget(self._rows.count() - 1, _Divider())
            self._rows.insertWidget(self._rows.count() - 1, _EntryRow(e, self))


# ─────────────────────────────────────────────────────────────────────────────
# Buttons
# ─────────────────────────────────────────────────────────────────────────────

class KeyButton(QtWidgets.QPushButton):
    """A button whose keyboard shortcut is shown as a dim hint after its label."""

    def __init__(self, text: str, key: str = "", **kwargs):
        super().__init__(text, **kwargs)
        self.key = key

    def _key_width(self) -> int:
        return QtGui.QFontMetrics(theme.mono_font(8)).horizontalAdvance(self.key) + 8 if self.key else 0

    def sizeHint(self) -> QtCore.QSize:
        s = super().sizeHint()
        return QtCore.QSize(s.width() + self._key_width(), s.height())

    def minimumSizeHint(self) -> QtCore.QSize:
        return self.sizeHint()

    def paintEvent(self, ev) -> None:
        if not self.key:
            return super().paintEvent(ev)
        opt = QtWidgets.QStyleOptionButton()
        self.initStyleOption(opt)
        text, opt.text = opt.text, ""
        p = QtWidgets.QStylePainter(self)
        p.drawControl(QtWidgets.QStyle.ControlElement.CE_PushButton, opt)
        kw = self._key_width()
        fm = self.fontMetrics()
        tw = fm.horizontalAdvance(text)
        x = (self.width() - tw - kw) // 2
        r = self.rect()
        active = self.isEnabled() and (self.underMouse() or self.isChecked())
        p.setPen(qcolor(theme.TEXT if active else theme.TEXT_2 if self.isEnabled() else theme.TEXT_4))
        p.drawText(QtCore.QRect(x, 0, tw, r.height()), QtCore.Qt.AlignmentFlag.AlignVCenter, text)
        p.setFont(theme.mono_font(8))
        p.setPen(qcolor(theme.TEXT_3 if self.isEnabled() else theme.TEXT_4))
        p.drawText(QtCore.QRect(x + tw + 8, 0, kw, r.height()), QtCore.Qt.AlignmentFlag.AlignVCenter, self.key)


class StageChip(QtWidgets.QAbstractButton):
    """A pill toggling one debug stage, with a dot in the stage's colour."""

    def __init__(self, stage: str):
        super().__init__()
        self.stage = stage
        self.setText(stage)
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"Offer the traces {stage} computes in the lane legends")

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self.fontMetrics().horizontalAdvance(self.text()) + 30, 22)

    def enterEvent(self, ev) -> None:
        self.update()

    def leaveEvent(self, ev) -> None:
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        r = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        on, hover = self.isChecked(), self.underMouse()
        p.setPen(QtGui.QPen(qcolor(theme.BORDER_STRONG if on or hover else theme.BORDER), 1))
        p.setBrush(qcolor(theme.SURFACE_3) if on else QtCore.Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(qcolor(theme.stage_color(self.stage), 255 if on else 90))
        p.drawEllipse(QtCore.QPointF(12, r.center().y()), 3.5, 3.5)
        p.setPen(qcolor(theme.TEXT if on else theme.TEXT_2 if hover else theme.TEXT_3))
        p.drawText(QtCore.QRectF(21, 0, r.width() - 24, r.height()), QtCore.Qt.AlignmentFlag.AlignVCenter,
                   self.text())


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────

class PillDelegate(QtWidgets.QStyledItemDelegate):
    """Draws a cell's text as a tinted pill; *tone_of* maps the text to a theme tone (None = plain)."""

    def __init__(self, tone_of: Callable[[str], Optional[str]], parent=None):
        super().__init__(parent)
        self.tone_of = tone_of

    def paint(self, p: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex) -> None:
        text = index.data() or ""
        tone = self.tone_of(text) if text else None
        if tone is None:
            return super().paint(p, option, index)
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        style = opt.widget.style() if opt.widget else QtWidgets.QApplication.style()
        style.drawControl(QtWidgets.QStyle.ControlElement.CE_ItemViewItem, opt, p, opt.widget)
        fg, bg = theme.tone_colors(tone)
        fm = option.fontMetrics
        r = option.rect.adjusted(8, 0, -8, 0)
        w = min(r.width(), fm.horizontalAdvance(text) + 14)
        pill = QtCore.QRectF(r.left(), r.center().y() - fm.height() / 2 - 1, w, fm.height() + 2)
        p.save()
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(qcolor(bg))
        p.drawRoundedRect(pill, 4, 4)
        p.setPen(qcolor(fg))
        p.drawText(pill.adjusted(7, 0, -7, 0), QtCore.Qt.AlignmentFlag.AlignVCenter,
                   fm.elidedText(text, QtCore.Qt.TextElideMode.ElideRight, int(pill.width()) - 14))
        p.restore()
