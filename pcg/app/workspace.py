"""Workspace screen: one Recording, its Analysis and Annotation on stacked, linked lanes."""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from pcg import analysis as A
from pcg import batch
from pcg import annotation as an
from pcg import context, recording
from pcg.engine.config import param
from pcg.clock import clock_text
from pcg.engine.traces import KIND_LINE, KIND_POINTS, KIND_SPANS, ROLE_ENVELOPE, ROLE_NOISE, ROLE_S1, ROLE_S2, Trace

from . import theme
from .audio import SOURCE_FILTERED, Player, load_playback_audio, spectrogram
from .chart import annotation_bpm_curve, render_bpm_chart
from .export_dialog import ExportDialog
from .items import BandsItem, EnvelopeItem, SpanRow, SpanRowsItem, TimeAxis, palette_row
from .state import AnalyzeWorker, AnnotationDoc, Settings, downloads_dir, reveal_in_folder
from .theme import TraceMeta, qcolor
from .widgets import LEGEND_WIDTH, KeyButton, LaneLegend, LegendEntry, StageChip

STATES_DEFAULT_HEIGHT = 130  # px, until the user drags a handle; the other lanes share the rest
LANE_ORDER = ["states", "signal", "bpm", "intervals", "scores", "hrv", "contractility", "spectrogram"]
LANE_TITLES = {  # lane -> (title, unit)
    "states": ("States", ""), "signal": ("Signal", ""), "bpm": ("Heart rate", "BPM"),
    "intervals": ("Intervals", "s"), "scores": ("Classifier scores", "%"), "hrv": ("HRV", ""),
    "contractility": ("Contractility", ""), "spectrogram": ("Spectrogram", "Hz"),
}
DEFAULT_LANES = {"states", "signal", "bpm"}
AXIS_WIDTH = 64  # px; equal on every lane so the plots line up

# Pseudo-traces the workspace draws itself (not from the Analysis's trace catalog).
PEAKS_S1, PEAKS_S2, PEAKS_NOISE = "Peaks: S1", "Peaks: S2", "Peaks: noise"
ANN_BPM = "BPM from Annotation"                 # smoothed like the result curve
ANN_BPM_INSTANT = "Instant BPM from Annotation"  # one value per S1-S1 interval
BPM_TRACE = "BPM"  # the result heart-rate line the high/low marks follow
PSEUDO_TRACES = (
    TraceMeta(PEAKS_S1, "signal", KIND_POINTS, theme.RESULT, ROLE_S1, visible=True),
    TraceMeta(PEAKS_S2, "signal", KIND_POINTS, theme.RESULT, ROLE_S2, visible=True),
    TraceMeta(PEAKS_NOISE, "signal", KIND_POINTS, theme.RESULT, ROLE_NOISE),
    TraceMeta(ANN_BPM, "bpm", KIND_LINE, theme.ANNOTATION_GROUP, theme.ROLE_ANNOTATION, estimate=True, visible=True),
    TraceMeta(ANN_BPM_INSTANT, "bpm", KIND_POINTS, theme.ANNOTATION_GROUP, theme.ROLE_ANNOTATION),
)

# States lane rows (y ranges)
ROW_BEFORE = (-1.0, -0.1)
ROW_ANN = (0.0, 1.0)
ROW_DIS = (1.02, 1.13)
ROW_ALG = (1.15, 2.15)
ROW_DEF = (2.17, 2.28)
HOVER_PX = 8
HRV_LABELS = {
    "avg_bpm": "Mean BPM", "min_bpm": "Min BPM", "max_bpm": "Max BPM", "avg_rmssdc": "RMSSDc",
    "avg_sdnn": "SDNN", "avg_lf_power": "LF power", "avg_hf_power": "HF power", "avg_lf_hf_ratio": "LF/HF",
}
RUN_SETTING_LABELS = {"auto_switch_algorithm": "Auto-switch allowed", "analysis_start_sec": "Skipped start (s)"}


def legend_label(name: str) -> str:
    """A trace name without its pass, which the legend shows as a tag: "BPM (Pass 1)" -> "BPM",
    "Instant BPM (Pass 1, outliers removed)" -> "Instant BPM (outliers removed)"."""
    out = re.sub(r"\(Pass \d+(?:, )?", "(", name)
    return re.sub(r"\s*\(\)", "", out).strip()


@dataclass
class Lane:
    name: str
    plot: pg.PlotItem
    widget: Optional[pg.PlotWidget] = None
    legend: Optional[LaneLegend] = None
    box: Optional[QtWidgets.QWidget] = None  # legend + plot: what the lane splitter holds
    items: Dict[str, object] = field(default_factory=dict)
    playhead: Optional[pg.InfiniteLine] = None
    region: Optional[pg.LinearRegionItem] = None


def _whole_data_bounds(item: pg.PlotDataItem):
    """dataBounds over the full series: with view clipping on, pyqtgraph's own bounds follow the visible
    window, which would make an auto-ranged y-axis rescale while panning."""
    cache: Dict[int, Tuple[Optional[float], Optional[float]]] = {}

    def bounds(ax, frac=1.0, orthoRange=None):
        if ax not in cache:
            data = item.getOriginalDataset()[ax]
            finite = data[np.isfinite(data)] if data is not None and len(data) else data
            cache[ax] = (float(finite.min()), float(finite.max())) if finite is not None and len(finite) else (None, None)
        return cache[ax]

    return bounds


class LaneViewBox(pg.ViewBox):
    """X-only pan/zoom, plus the workspace's drag gestures (shift = region, N = noisy, noisy edges).

    The wheel zooms time over the plot; over the y-axis (or with Ctrl) it scales the y-axis around
    the cursor. Double-clicking the y-axis returns to auto-range.
    """

    def __init__(self, ws: "Workspace", lane: str):
        super().__init__()
        self.ws = ws
        self.lane = lane
        self.setMouseEnabled(x=True, y=False)
        self.setMenuEnabled(False)
        self._drag_mode: Optional[str] = None
        self._drag_ref: object = None

    def wheelEvent(self, ev, axis=None):
        if self.lane == "states" or not (axis == 1 or ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
            return super().wheelEvent(ev, axis)
        ev.accept()
        delta = ev.delta()  # QGraphicsSceneWheelEvent: no angleDelta()
        if delta:
            self.ws.zoom_y(self.lane, self.mapSceneToView(ev.scenePos()).y(), 0.9 ** (delta / 120))

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.MouseButton.RightButton:  # pyqtgraph's right-drag zoom is not wanted
            ev.ignore()
            return
        if axis == 1 and self.lane != "states" and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()  # dragging the y-axis slides the lane's y-range up and down
            dy = self.mapSceneToView(ev.scenePos()).y() - self.mapSceneToView(ev.lastScenePos()).y()
            self.ws.pan_y(self.lane, -dy, final=ev.isFinish())
            return
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return super().mouseDragEvent(ev, axis)
        x = self.mapSceneToView(ev.scenePos()).x()
        if ev.isStart():
            x0 = self.mapSceneToView(ev.buttonDownScenePos()).x()
            y0 = self.mapSceneToView(ev.buttonDownScenePos()).y()
            if ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                self._drag_mode, self._drag_ref = "region", x0
            elif self.ws.n_held:
                self._drag_mode, self._drag_ref = "noisy", x0
            else:
                edge = self.ws.noisy_edge_at(self, x0, y0) if self.lane == "states" else None
                self._drag_mode, self._drag_ref = ("resize", edge) if edge else (None, None)
        if self._drag_mode is None:
            return super().mouseDragEvent(ev, axis)
        ev.accept()
        if self._drag_mode == "region":
            self.ws.set_region((self._drag_ref, x), final=ev.isFinish())
        elif self._drag_mode == "noisy":
            self.ws.preview_noisy((self._drag_ref, x), final=ev.isFinish())
        elif self._drag_mode == "resize":
            span, side = self._drag_ref
            a, b = (x, span.end) if side == "start" else (span.start, x)
            self.ws.preview_noisy((a, b), final=ev.isFinish(), replacing=span)
        if ev.isFinish():
            self._drag_mode = None

    def mouseClickEvent(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        ev.accept()
        if not self.sceneBoundingRect().contains(ev.scenePos()):  # forwarded from the y-axis
            if ev.double():
                self.ws.reset_y(self.lane)
            return
        pos = self.mapSceneToView(ev.scenePos())
        if ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            self.ws.set_region(None, final=True)
            return
        self.ws.click(self.lane, pos.x(), pos.y())


class OverviewViewBox(pg.ViewBox):
    """Click or drag anywhere on the overview to centre the view window on the cursor; the wheel
    grows/shrinks the window."""

    def __init__(self, ws: "Workspace"):
        super().__init__(enableMenu=False)
        self.ws = ws
        self.setMouseEnabled(x=False, y=False)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            self.ws.center_view_at(self.mapSceneToView(ev.scenePos()).x())

    def wheelEvent(self, ev, axis=None):
        ev.accept()
        if ev.delta():
            self.ws.scale_view(0.9 ** (-ev.delta() / 120))  # scroll up grows the window (opposite of the lanes)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        ev.accept()
        self.ws.center_view_at(self.mapSceneToView(ev.scenePos()).x())


class Workspace(QtWidgets.QWidget):
    title_changed = QtCore.Signal(str)
    analyses_changed = QtCore.Signal()
    # Background audio loading reports back through queued signals.
    _audio_loaded = QtCore.Signal(int, object)
    _filtered_loaded = QtCore.Signal(int, object)
    _spectrogram_loaded = QtCore.Signal(int, object)
    _audio_error = QtCore.Signal(int, str)

    def __init__(self, settings: Settings, library: A.Library, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.library = library
        self.path: Optional[str] = None
        self.fingerprint = ""
        self.analysis: Optional[A.Analysis] = None
        self.doc: Optional[AnnotationDoc] = None
        self.algo_spans: an.Spans = ()
        self.disagreements: List[an.Disagreement] = []
        self.selected: Optional[an.Span] = None
        self.region: Optional[Tuple[float, float]] = None
        self.n_held = False
        self.player = Player()
        self.worker = AnalyzeWorker(self)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.trace_vis: Dict[str, bool] = settings.trace_visibility()
        self.lane_vis: Dict[str, bool] = settings.lane_visibility()
        self.stages_on: Dict[str, bool] = settings.stages()
        self.lane_order: List[str] = settings.lane_order()
        self._ann_bpm: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None  # t, instant, smoothed
        self._ghost: Optional[QtWidgets.QLabel] = None  # snapshot of the lane being dragged
        self._dim: Optional[QtWidgets.QFrame] = None    # covers the lane while it is dragged
        self._ghost_grab = 0
        self._metas: Dict[str, TraceMeta] = {}   # every trace of the open Analysis, incl. the pseudo-traces
        self._styles: Dict[str, theme.TraceStyle] = {}
        self.lanes: Dict[str, Lane] = {}
        self._issues: List[Tuple[str, str, float, float, str]] = []  # (source, kind, start, end, message)
        self._syncing = False
        self._audio_token = 0
        self._peak_kinds = np.array([], dtype=object)
        self._audio_loaded.connect(self._audio_ready)
        self._filtered_loaded.connect(self._filtered_ready)
        self._spectrogram_loaded.connect(self._spectrogram_ready)
        self._spec = None  # (t0, dt, f_top, dB) of the open recording, computed when its lane is shown
        self._y_manual: Dict[str, List[float]] = {}  # lane -> [lo, hi] the user set, for the open Recording
        self._spec_token = 0  # bumped to drop an in-flight spectrogram (new audio, or the lane turned off)
        self._audio_error.connect(self._audio_failed)
        self._build_ui()
        self._build_shortcuts()
        self._timer = QtCore.QTimer(self, interval=33, timeout=self._tick)
        self._timer.start()
        self._issues_timer = QtCore.QTimer(self, singleShot=True, interval=150, timeout=self._refresh_issues)
        QtWidgets.QApplication.instance().installEventFilter(self)

    # ─────────────────────────────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        header = QtWidgets.QFrame()
        header.setObjectName("header")
        top = QtWidgets.QHBoxLayout(header)
        top.setContentsMargins(12, 6, 10, 6)
        top.setSpacing(8)
        self.name_label = QtWidgets.QLabel("No recording")
        self.name_label.setObjectName("title")
        self.name_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.algo_badge = theme.badge("", "info")
        self.algo_badge.hide()
        self.info_label = QtWidgets.QLabel()
        self.info_label.setFont(theme.mono_font(8.5))
        self.info_label.setProperty("tone", "secondary")
        self.gate_badge = theme.badge("Gate failed", "warning")
        self.gate_badge.hide()
        self.stale_badge = theme.badge("Stale", "warning")
        self.stale_badge.hide()
        self.progress_label = QtWidgets.QLabel()
        self.progress_label.setProperty("tone", "accent")
        self.rerun_btn = KeyButton("Re-run", "Ctrl R", clicked=self.rerun)
        self.cancel_btn = QtWidgets.QPushButton("Cancel", clicked=self.worker.cancel)
        self.cancel_btn.hide()
        self.flip_btn = KeyButton("Flip S1/S2 after playhead", clicked=self.flip_after_playhead)
        self.source_btn = KeyButton("", "T", clicked=self.toggle_source)
        self.source_btn.setToolTip("Switch between the original and the band-passed audio")
        self._show_source("Audio loading…", enabled=False)
        for w in (self.name_label, self.algo_badge, self.info_label, self.gate_badge, self.stale_badge):
            top.addWidget(w)
        top.addStretch(1)
        for w in (self.progress_label, self.cancel_btn, self.rerun_btn, self.flip_btn, self.source_btn):
            top.addWidget(w)

        # Second row: which debug stages the legends offer (left), the side panel (right).
        sub = QtWidgets.QFrame()
        sub.setObjectName("header")
        chips = QtWidgets.QHBoxLayout(sub)
        chips.setContentsMargins(12, 4, 10, 4)
        chips.setSpacing(6)
        debug = QtWidgets.QLabel("Debug stages")
        debug.setProperty("tone", "muted")
        chips.addWidget(debug)
        self.stage_chips: Dict[str, StageChip] = {}
        for stage in theme.DEBUG_STAGES:
            self._add_stage_chip(stage, chips)
        self._chip_row = chips
        chips.addStretch(1)
        self.issues_btn = QtWidgets.QPushButton("Issues", checkable=True)
        self.summary_btn = QtWidgets.QPushButton("Summary", checkable=True)
        self.issues_btn.clicked.connect(lambda on: self._show_side("issues" if on else ""))
        self.summary_btn.clicked.connect(lambda on: self._show_side("summary" if on else ""))
        chips.addWidget(self.issues_btn)
        chips.addWidget(self.summary_btn)

        # One widget per lane in a vertical splitter, so lanes can be resized by dragging between them.
        self.lane_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.lane_split.setHandleWidth(3)
        self.lane_split.setChildrenCollapsible(False)
        self.lane_split.splitterMoved.connect(self._save_lane_heights)
        # Drag feedback lives in their own frameless windows: child widgets painted over the plot
        # views are not repainted when they move and leave a trail.
        self.drop_line = self._overlay_window(QtWidgets.QFrame())
        self.drop_line.setObjectName("dropLine")
        self.drop_line.setFixedHeight(3)

        self.overview = pg.PlotItem(viewBox=OverviewViewBox(self), axisItems={"bottom": TimeAxis("bottom")})
        self.overview.setMouseEnabled(x=False, y=False)
        self.overview.setMenuEnabled(False)
        self.overview.hideButtons()
        self.overview.getAxis("left").setWidth(AXIS_WIDTH)
        self.overview.getAxis("left").setStyle(showValues=False)
        for ax in ("left", "bottom"):
            theme.style_axis(self.overview.getAxis(ax))
        # Its own fixed-height widget: a maximum height inside the lanes' layout collapses every row.
        self.overview_widget = pg.PlotWidget(plotItem=self.overview)
        self.overview_widget.setFixedHeight(64)
        self.overview_region = pg.LinearRegionItem(brush=qcolor(theme.ACCENT, 34), pen=qcolor(theme.ACCENT, 150),
                                                   movable=False)
        self.overview_curve = pg.PlotDataItem(pen=pg.mkPen(theme.ENVELOPE), fillLevel=0,
                                              brush=qcolor(theme.ENVELOPE_FILL))
        self.overview_playhead = pg.InfiniteLine(angle=90, pen=pg.mkPen(theme.PLAYHEAD, width=1))
        for it in (self.overview_curve, self.overview_region, self.overview_playhead):
            self.overview.addItem(it)
        overview_row = QtWidgets.QWidget()
        orow = QtWidgets.QHBoxLayout(overview_row)
        orow.setContentsMargins(0, 0, 0, 0)
        orow.setSpacing(0)
        ov_label = QtWidgets.QLabel("Overview")
        ov_label.setObjectName("overviewLegend")
        ov_label.setProperty("tone", "muted")
        ov_label.setFixedWidth(LEGEND_WIDTH)
        ov_label.setContentsMargins(10, 0, 0, 0)
        orow.addWidget(ov_label)
        orow.addWidget(self.overview_widget, 1)

        for name in LANE_ORDER:
            self._make_lane(name)

        # Hidden lanes, one chip each: click to bring the lane back.
        self.lane_bar = QtWidgets.QFrame()
        self.lane_bar.setObjectName("laneBar")
        self._lane_bar_lay = QtWidgets.QHBoxLayout(self.lane_bar)
        self._lane_bar_lay.setContentsMargins(10, 4, 10, 4)
        self._lane_bar_lay.setSpacing(6)

        issues = QtWidgets.QWidget()
        il = QtWidgets.QVBoxLayout(issues)
        il.setContentsMargins(8, 8, 8, 6)
        self.issue_filter = QtWidgets.QComboBox()
        self.issue_filter.currentIndexChanged.connect(self._fill_issue_list)
        self.issue_list = QtWidgets.QListWidget()
        self.issue_list.setFont(theme.mono_font(8.5))
        self.issue_list.itemActivated.connect(lambda it: self._jump_to_issue(self.issue_list.row(it)))
        self.issue_list.itemClicked.connect(lambda it: self._jump_to_issue(self.issue_list.row(it)))
        hint = QtWidgets.QLabel("]  next     [  previous")
        hint.setProperty("tone", "muted")
        il.addWidget(self.issue_filter)
        il.addWidget(self.issue_list, 1)
        il.addWidget(hint)

        self.summary_text = QtWidgets.QTextBrowser()
        self.summary_text.setOpenLinks(False)
        self.side = QtWidgets.QFrame()
        self.side.setObjectName("sidePanel")
        self.side_stack = QtWidgets.QStackedLayout(self.side)
        self.side_stack.addWidget(issues)
        self.side_stack.addWidget(self.summary_text)

        split = QtWidgets.QSplitter()
        split.setHandleWidth(1)
        self.lanes_box = lanes_box = QtWidgets.QWidget()
        lb = QtWidgets.QVBoxLayout(lanes_box)
        lb.setContentsMargins(0, 0, 0, 0)
        lb.setSpacing(0)
        lb.addWidget(overview_row)
        lb.addWidget(self.lane_split, 1)
        lb.addWidget(self.lane_bar)
        split.addWidget(lanes_box)
        split.addWidget(self.side)
        split.setStretchFactor(0, 1)
        split.setSizes([1500, 340])

        self.status = QtWidgets.QLabel()
        self.status.setFont(theme.mono_font(8.5))
        self.status.setProperty("tone", "secondary")
        self.status.setContentsMargins(12, 3, 12, 3)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(header)
        lay.addWidget(sub)
        lay.addWidget(split, 1)
        lay.addWidget(self.status)
        self._show_side(self.settings.get_json("side_panel", ""))
        self._layout_lanes()

    def _add_stage_chip(self, stage: str, row: QtWidgets.QHBoxLayout, index: int = -1) -> None:
        chip = StageChip(stage)
        chip.setChecked(bool(self.stages_on.get(stage)))
        chip.toggled.connect(lambda on, s=stage: self.set_stage(s, on))
        self.stage_chips[stage] = chip
        if index < 0:
            row.addWidget(chip)
        else:
            row.insertWidget(index, chip)

    def _show_side(self, which: str) -> None:
        """Open the side panel on Issues or Summary, or close it ("")."""
        self.side.setVisible(bool(which))
        if which:
            self.side_stack.setCurrentIndex(0 if which == "issues" else 1)
        self.issues_btn.setChecked(which == "issues")
        self.summary_btn.setChecked(which == "summary")
        self.settings.set_json("side_panel", which)

    def _make_lane(self, name: str) -> None:
        vb = LaneViewBox(self, name)
        plot = pg.PlotItem(viewBox=vb, axisItems={"bottom": TimeAxis("bottom")})
        plot.hideButtons()
        left = plot.getAxis("left")
        left.setWidth(AXIS_WIDTH)
        left.enableAutoSIPrefix(False)
        for ax in ("left", "bottom"):
            theme.style_axis(plot.getAxis(ax))
        plot.showGrid(x=True, y=True, alpha=0.5)
        if name == "spectrogram":
            plot.setYRange(0, 1000, padding=0)
        elif name == "states":
            plot.setYRange(ROW_ANN[0] - 0.05, ROW_DEF[1] + 0.05, padding=0)
            plot.showGrid(x=True, y=False, alpha=0.5)
            theme.style_axis(left, mono=False)
            left.setTicks([[(0.5, "Annotation"), (1.65, "Analysis"), (1.075, "Disagree"), (2.225, "Defects"),
                            (-0.55, "Before repair")]])
        else:
            plot.enableAutoRange(axis="y", enable=True)
            plot.setAutoVisible(y=False)  # fit the whole recording, not the visible window, so panning never rescales
        lane = Lane(name, plot)
        lane.widget = pg.PlotWidget(plotItem=plot)
        lane.widget.setMinimumHeight(40)
        lane.widget.scene().sigMouseMoved.connect(lambda pos, lane=lane: self._on_mouse_moved(pos, lane))
        lane.playhead = pg.InfiniteLine(angle=90, pen=pg.mkPen(theme.PLAYHEAD, width=1))
        lane.playhead.setZValue(50)
        lane.region = pg.LinearRegionItem(brush=qcolor(theme.ACCENT, 40), pen=qcolor(theme.ACCENT, 120), movable=False)
        lane.region.hide()
        plot.addItem(lane.playhead, ignoreBounds=True)
        plot.addItem(lane.region, ignoreBounds=True)
        if name == "bpm":
            vb.sigXRangeChanged.connect(lambda *_: self._update_bpm_extremes())
        title, unit = LANE_TITLES.get(name, (name.replace("_", " ").capitalize(), ""))
        lane.legend = LaneLegend(title, unit)
        lane.legend.toggled.connect(self._on_legend_toggled)
        lane.legend.hovered.connect(lambda trace, lane=lane: self._highlight(lane, trace))
        lane.legend.hide_requested.connect(lambda n=name: self.set_lane_visible(n, False))
        lane.legend.drag_moved.connect(lambda y, n=name: self._drag_lane(n, y))
        lane.legend.drag_dropped.connect(lambda y, n=name: self._drop_lane(n, y))
        lane.box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(lane.box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(lane.legend)
        row.addWidget(lane.widget, 1)
        if self.lanes:
            plot.setXLink(self.lanes["states"].plot)
        else:
            vb.sigXRangeChanged.connect(self._main_range_changed)
        self.lanes[name] = lane

    def _extra_lane(self, name: str) -> Lane:
        if name not in self.lanes:
            self._make_lane(name)
        return self.lanes[name]

    def _lane_visible(self, name: str) -> bool:
        return self.lane_vis.get(name, name in DEFAULT_LANES)

    def _ordered_lanes(self) -> List[str]:
        """Lane names in display order: the user's saved order, then any lane not in it yet."""
        known = [n for n in self.lane_order if n in self.lanes]
        return known + [n for n in self.lanes if n not in known]

    def _drop_target(self, global_y: int) -> Tuple[Optional[str], int]:
        """(lane to insert before, or None for the end; y in lane_split coordinates of the drop line)."""
        y = self.lane_split.mapFromGlobal(QtCore.QPoint(0, global_y)).y()
        shown = [self.lanes[n].box for n in self._ordered_lanes() if self._lane_visible(n)]
        for n in self._ordered_lanes():
            box = self.lanes[n].box
            if self._lane_visible(n) and y < box.geometry().center().y():
                return n, box.geometry().top()
        return None, shown[-1].geometry().bottom() if shown else 0

    @staticmethod
    def _overlay_window(w: QtWidgets.QWidget) -> QtWidgets.QWidget:
        w.setWindowFlags(QtCore.Qt.WindowType.Tool | QtCore.Qt.WindowType.FramelessWindowHint
                         | QtCore.Qt.WindowType.WindowStaysOnTopHint | QtCore.Qt.WindowType.WindowTransparentForInput)
        w.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
        return w

    def _drag_lane(self, name: str, global_y: int) -> None:
        box = self.lanes[name].box
        origin = box.mapToGlobal(QtCore.QPoint(0, 0))
        if self._ghost is None:  # first move: snapshot the lane, dim the original
            self._ghost = self._overlay_window(QtWidgets.QLabel())
            self._ghost.setPixmap(box.grab())
            self._ghost.setStyleSheet(f"border: 1px solid {theme.ACCENT};")
            self._ghost.resize(box.size())
            self._ghost.setWindowOpacity(0.75)
            self._dim = QtWidgets.QFrame(box)
            self._dim.setStyleSheet(f"background: {theme.qcolor(theme.SURFACE_0, 170).name(QtGui.QColor.NameFormat.HexArgb)};")
            self._dim.setGeometry(box.rect())
            self._dim.show()
            self._dim.raise_()
            self._ghost_grab = global_y - origin.y()
        top_limit = self.lane_split.mapToGlobal(QtCore.QPoint(0, 0)).y()
        bottom_limit = top_limit + self.lane_split.height() - self._ghost.height()
        self._ghost.move(origin.x(), int(min(max(top_limit, global_y - self._ghost_grab), max(top_limit, bottom_limit))))
        self._ghost.show()
        _before, line_y = self._drop_target(global_y)
        at = self.lane_split.mapToGlobal(QtCore.QPoint(0, max(0, line_y - 1)))
        self.drop_line.setGeometry(at.x(), at.y(), self.lane_split.width(), 3)
        self.drop_line.show()

    def _end_lane_drag(self) -> None:
        self.drop_line.hide()
        if self._ghost is not None:
            self._ghost.deleteLater()
            self._ghost = None
            self._dim.deleteLater()

    def _drop_lane(self, name: str, global_y: int) -> None:
        self._end_lane_drag()
        before, _ = self._drop_target(global_y)
        order = [n for n in self._ordered_lanes() if n != name]
        order.insert(order.index(before) if before in order else self._after_last_visible(order), name)
        if order == self._ordered_lanes():
            return
        self.lane_order = order
        self.settings.set_lane_order(order)
        self._layout_lanes()

    def _after_last_visible(self, order: List[str]) -> int:
        last = max((i for i, n in enumerate(order) if self._lane_visible(n)), default=-1)
        return last + 1

    def _layout_lanes(self) -> None:
        names = self._ordered_lanes()
        visible = [n for n in names if self._lane_visible(n)]
        saved = self.settings.lane_heights()
        total = self.lane_split.height() if self.lane_split.height() > 200 else 800
        fixed = {n: saved.get(n, STATES_DEFAULT_HEIGHT if n == "states" else None) for n in visible}
        free = [n for n in visible if fixed[n] is None]
        share = max(100, (total - sum(h for h in fixed.values() if h is not None)) // max(1, len(free)))
        sizes = []
        for i, n in enumerate(names):
            lane = self.lanes[n]
            self.lane_split.insertWidget(i, lane.box)
            lane.box.setVisible(n in visible)
            self.lane_split.setStretchFactor(i, 0 if n == "states" else 1)  # window resizes go to the plots
            if n in visible:
                lane.plot.getAxis("bottom").setStyle(showValues=(n == visible[-1]))
            sizes.append((fixed[n] if fixed[n] is not None else share) if n in visible else 0)
        self.lane_split.setSizes(sizes)
        self._fill_lane_bar()

    def _fill_lane_bar(self) -> None:
        lay = self._lane_bar_lay
        while lay.count():
            w = lay.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        hidden = [n for n in self._ordered_lanes() if not self._lane_visible(n)]
        self.lane_bar.setVisible(bool(hidden))
        if not hidden:
            return
        label = QtWidgets.QLabel("Show lane")
        label.setProperty("tone", "muted")
        lay.addWidget(label)
        for n in hidden:
            chip = QtWidgets.QPushButton(f"+ {LANE_TITLES.get(n, (n.capitalize(), ''))[0]}")
            chip.setProperty("chip", "lane")
            chip.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            chip.clicked.connect(lambda _=False, n=n: self.set_lane_visible(n, True))
            lay.addWidget(chip)
        lay.addStretch(1)

    def _save_lane_heights(self) -> None:
        sizes = dict(zip(self._ordered_lanes(), self.lane_split.sizes()))
        self.settings.set_lane_heights({n: h for n, h in sizes.items() if h > 0})

    def _build_shortcuts(self) -> None:
        def sc(key, fn):
            s = QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=fn)
            s.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)

        sc("Space", self.player_toggle)
        sc("T", self.toggle_source)
        sc("1", lambda: self.place(an.S1))
        sc("2", lambda: self.place(an.S2))
        sc("X", self.delete_span)
        sc("Delete", self.delete_span)
        sc("F", self.relabel_span)
        sc("A", self.replace_region_from_algorithm)
        sc("]", lambda: self.step_issue(+1))
        sc("[", lambda: self.step_issue(-1))
        sc("G", lambda: self.set_lane_visible("spectrogram", not self._lane_visible("spectrogram")))
        sc("Ctrl+Z", self.undo)
        sc("Ctrl+Shift+Z", self.redo)
        sc("Ctrl+Y", self.redo)
        sc("Ctrl+S", self.save_annotation)
        sc("Ctrl+R", self.rerun)
        sc("Ctrl+Shift+C", self.copy_context)
        sc("Ctrl+Shift+E", self.export_dialog)

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t in (QtCore.QEvent.Type.KeyPress, QtCore.QEvent.Type.KeyRelease) and ev.key() == QtCore.Qt.Key.Key_N:
            if not ev.isAutoRepeat() and self.isVisible():
                self.n_held = t == QtCore.QEvent.Type.KeyPress
        elif t == QtCore.QEvent.Type.ApplicationDeactivate:
            self.n_held = False
        return False

    # ─────────────────────────────────────────────────────────────────────
    # Opening
    # ─────────────────────────────────────────────────────────────────────
    def maybe_close(self) -> bool:
        """Ask about unsaved Annotation changes. False = the user cancelled."""
        if self.doc is None or not self.doc.dirty:
            return True
        r = QtWidgets.QMessageBox.question(
            self, "Unsaved Annotation", "Save the Annotation changes first?",
            QtWidgets.QMessageBox.StandardButton.Save | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel)
        if r == QtWidgets.QMessageBox.StandardButton.Cancel:
            return False
        if r == QtWidgets.QMessageBox.StandardButton.Save:
            return self.save_annotation()
        return True

    def open_recording(self, path: str, analysis_path: Optional[str] = None) -> None:
        self.player.clear()
        self.worker.cancel()
        self.path = str(Path(path).resolve())
        self.fingerprint = self.library.fingerprints.get(self.path)
        self.analysis = None
        self.doc = None
        self.selected = None
        self.region = None
        self.name_label.setText(Path(self.path).name)
        found = analysis_path or self.library.latest(self.fingerprint)
        if found:
            self._load_analysis(str(found), keep_view=False)
        else:
            self._clear_plots()
            self.rerun()
        self._load_audio()

    def _load_analysis(self, analysis_path: str, keep_view: bool) -> None:
        view = self.lanes["states"].plot.getViewBox().viewRange()[0] if keep_view else None
        self.analysis = self.library.open(analysis_path)
        self.algo_spans = an.from_states(self.analysis.states.start, self.analysis.states.end, self.analysis.states.state)
        if self.doc is None:
            self._open_annotation_for_recording()
        self._rebuild_plots()
        dur = self.analysis.duration_sec
        for lane in self.lanes.values():
            lane.plot.getViewBox().setLimits(xMin=0, xMax=dur, minXRange=0.05)
        self.overview.setXRange(0, dur, padding=0)
        if view is None:
            view = (0.0, min(dur, 30.0))
        self.lanes["states"].plot.setXRange(*view, padding=0)
        self._update_header()
        self._on_doc_changed()

    def _open_annotation_for_recording(self) -> None:
        a = self.analysis
        found = an.find_for_recording(self.path, self.fingerprint)
        if found:
            ann = an.load(found)
            self._set_doc(AnnotationDoc(ann, found, draft=False))
        else:
            ann = an.Annotation(self.fingerprint, a.duration_sec, Path(self.path).name, self.algo_spans)
            self._set_doc(AnnotationDoc(ann, None, draft=True))

    def _set_doc(self, doc: AnnotationDoc) -> None:
        self.doc = doc
        doc.changed.connect(self._on_doc_changed)
        self.selected = None

    def open_annotation_file(self) -> None:
        if self.analysis is None or not self.maybe_close():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Annotation", str(Path(self.path).parent),
                                                        f"Annotations (*{an.SUFFIX});;JSON (*.json)")
        if not path:
            return
        ann = an.load(path)
        if ann.fingerprint != self.fingerprint:
            r = QtWidgets.QMessageBox.warning(
                self, "Different recording",
                "This Annotation was made on different audio (fingerprint mismatch). Open it anyway?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
            if r != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            ann = an.Annotation(self.fingerprint, ann.duration_sec, ann.filename_hint, ann.spans)
        self._set_doc(AnnotationDoc(ann, Path(path), draft=False))
        self._on_doc_changed()

    def _load_audio(self) -> None:
        self._audio_token += 1
        token = self._audio_token
        path = self.path
        channel = self.analysis.channel if self.analysis else recording.CHANNEL_MIXED
        params = dict(self.analysis.params) if self.analysis else A.effective_params({})
        self._show_source("Audio: loading…", enabled=False)

        def work():
            try:
                sig, sr = load_playback_audio(path, channel)
                self._audio_loaded.emit(token, (sig, sr, params))
            except Exception as e:  # noqa: BLE001
                self._audio_error.emit(token, str(e))

        threading.Thread(target=work, daemon=True).start()

    def _audio_ready(self, token: int, payload) -> None:
        if token != self._audio_token:
            return
        sig, sr, params = payload
        self.player.set_audio(sig, sr)
        self._spec = None
        self._spec_token += 1
        if self._lane_visible("spectrogram"):
            self._compute_spectrogram()
        self._show_source("Original audio  (filtered loading…)")

        def work():
            self._filtered_loaded.emit(token, recording.bandpassed(sig, sr, params))

        threading.Thread(target=work, daemon=True).start()

    def _filtered_ready(self, token: int, filt) -> None:
        if token == self._audio_token:
            self.player.set_filtered(filt)
            self._show_source(self._source_text())

    def _compute_spectrogram(self) -> None:
        src = self.player.samples()
        if src is None:
            return
        self._spec_token += 1
        token, sr = self._spec_token, self.player.sample_rate
        self.status.setText("Computing spectrogram…")

        def work():
            self._spectrogram_loaded.emit(token, spectrogram(src, sr))

        threading.Thread(target=work, daemon=True).start()

    def _spectrogram_ready(self, token: int, payload) -> None:
        if token != self._spec_token:
            return
        self._spec = payload
        self.status.setText("")
        self._draw_spectrogram()

    def _drop_spectrogram(self) -> None:
        """Lane turned off: discard any in-flight result and free the image."""
        self._spec_token += 1
        self._spec = None
        self._set_internal(self.lanes["spectrogram"], "_spec", None)
        if self.status.text().startswith("Computing spectrogram"):
            self.status.setText("")

    def _draw_spectrogram(self) -> None:
        lane = self.lanes["spectrogram"]
        if self._spec is None:
            self._set_internal(lane, "_spec", None)
            return
        t0, dt, f_top, db = self._spec
        img = pg.ImageItem(db, autoDownsample=True)
        img.setColorMap(pg.colormap.get("inferno"))
        lo, hi = np.percentile(db[:: max(1, len(db) // 2000)], [5, 99.5]) if len(db) else (0.0, 1.0)
        img.setLevels((float(lo), float(hi)))
        img.setRect(QtCore.QRectF(t0 - dt / 2, 0, len(db) * dt, f_top))
        img.setZValue(-20)
        self._set_internal(lane, "_spec", img)

    def _audio_failed(self, token: int, msg: str) -> None:
        if token != self._audio_token:
            return
        self._show_source("Audio: unavailable", enabled=False)
        self.status.setText(f"Could not load audio: {msg}")

    # ─────────────────────────────────────────────────────────────────────
    # Plots
    # ─────────────────────────────────────────────────────────────────────
    def _clear_plots(self) -> None:
        self._ann_bpm = None
        for lane in self.lanes.values():
            for item in lane.items.values():
                lane.plot.removeItem(item)
            lane.items.clear()
            lane.legend.set_entries([])
        self.overview_curve.setData([], [])

    # Which traces are offered and drawn. A trace is *offered* (listed in its lane's legend) when its
    # stage is always listed (Result, Annotation), it is shown by default, or its debug stage is
    # switched on. It is *drawn* when offered and toggled on (the toggle is remembered either way).
    def _offered(self, m: TraceMeta) -> bool:
        return m.group in theme.ALWAYS_LISTED or m.visible or bool(self.stages_on.get(m.group))

    def _wanted(self, name: str) -> bool:
        m = self._metas[name]
        return self._offered(m) and bool(self.trace_vis.get(name, m.visible))

    def _rebuild_plots(self) -> None:
        self._clear_plots()
        a = self.analysis
        self._peak_kinds = a.peaks.kind()
        metas = [TraceMeta.of(t) for t in a.traces]
        pseudo = {m.lane: [] for m in PSEUDO_TRACES}
        for m in PSEUDO_TRACES:
            pseudo[m.lane].append(m)
        # The workspace's own traces lead their lane; catalog order otherwise.
        ordered = [m for lane in pseudo.values() for m in lane] + metas
        self._metas = {m.name: m for m in ordered}
        self._styles = theme.assign_styles(ordered)
        self._sync_stage_chips()
        for t in a.traces:
            if t.lane not in self.lanes:
                self._extra_lane(t.lane)
        self._layout_lanes()
        for name in self._metas:
            self._show(name, self._wanted(name))
        if self._lane_visible("spectrogram"):
            self._draw_spectrogram()
        self._draw_states()
        self._draw_overview()
        self._fill_legends()
        self._restore_y()
        self._update_bpm_extremes()

    def _sync_stage_chips(self) -> None:
        """A chip for every debug stage of the open Analysis (emit_trace may invent groups)."""
        extra = [g for g in theme.stage_order(m.group for m in self._metas.values())
                 if g not in self.stage_chips and g not in theme.ALWAYS_LISTED]
        for stage in extra:
            index = self._chip_row.indexOf(self.stage_chips[theme.DEBUG_STAGES[-1]]) + 1
            self._add_stage_chip(stage, self._chip_row, index)

    def set_stage(self, stage: str, on: bool) -> None:
        """Offer (or withdraw) the traces one debug stage computed, in every lane."""
        self.stages_on[stage] = on
        self.settings.set_stages(self.stages_on)
        if self.analysis is None:
            return
        for name, m in self._metas.items():
            if m.group == stage:
                self._show(name, self._wanted(name))
        self._fill_legends()

    def _show(self, name: str, shown: bool) -> None:
        if name in (PEAKS_S1, PEAKS_S2, PEAKS_NOISE):
            self._set_peaks_shown(name, shown)
        elif name in (ANN_BPM, ANN_BPM_INSTANT):
            self._set_ann_bpm_shown(name, shown)
        else:
            t = self.analysis.trace(name)
            if t is not None:
                self._set_trace_shown(t, shown)
        if name in (BPM_TRACE, ANN_BPM):
            self._update_bpm_extremes()

    def _bpm_extreme_source(self) -> Optional[Tuple[np.ndarray, np.ndarray, str, str]]:
        """(times, values, ring colour, label suffix) of the line the high/low marks follow: the smoothed
        BPM from the Annotation when it is drawn and has data, else the result BPM line."""
        lane = self.lanes["bpm"]
        if not self._lane_visible("bpm"):
            return None
        ann = lane.items.get(ANN_BPM)
        if self._ann_bpm is not None and len(self._ann_bpm[0]) and ann is not None and ann.isVisible():
            return self._ann_bpm[0], self._ann_bpm[2], theme.ANNOTATION, "  annotation"
        t, line = self.analysis.trace(BPM_TRACE), lane.items.get(BPM_TRACE)
        if t is not None and line is not None and line.isVisible():
            return t.times(), t.y, theme.METRIC, ""
        return None

    def _update_bpm_extremes(self) -> None:
        """Mark the highest and lowest BPM of the whole recording on the plot, and list them (click to
        jump there) under the lane's legend. Follows the smoothed Annotation line when it is drawn."""
        if self.analysis is None:
            return
        lane = self.lanes["bpm"]
        marks = [lane.items.get(k) for k in ("_bpm_marks", "_bpm_max", "_bpm_min")]
        src = self._bpm_extreme_source()
        idx = np.array([], dtype=int)
        if src is not None:
            idx = np.nonzero(np.isfinite(src[1]))[0]
        if not len(idx):
            for m in marks:
                if m is not None:
                    m.setVisible(False)
            lane.legend.set_footer([])
            return
        if marks[0] is None:
            scatter = pg.ScatterPlotItem(brush=None, size=9, symbol="o")
            scatter.setZValue(30)
            lane.items["_bpm_marks"] = scatter
            lane.plot.addItem(scatter, ignoreBounds=True)
            for key in ("_bpm_max", "_bpm_min"):
                label = pg.TextItem(color=theme.TEXT, fill=qcolor(theme.SURFACE_0, 215))
                label.setFont(theme.mono_font(8.5))
                label.setZValue(31)
                lane.items[key] = label
                lane.plot.addItem(label, ignoreBounds=True)
            marks = [lane.items[k] for k in ("_bpm_marks", "_bpm_max", "_bpm_min")]
        times, vals = src[0][idx], src[1][idx]
        picks = (("_bpm_max", "max", int(np.argmax(vals))), ("_bpm_min", "min", int(np.argmin(vals))))
        marks[0].setPen(pg.mkPen(src[2], width=1.5))
        marks[0].setData([times[i] for *_, i in picks], [vals[i] for *_, i in picks])
        marks[0].setVisible(True)
        (v0, v1), _ = lane.plot.getViewBox().viewRange()
        footer = []
        for key, word, i in picks:
            at, bpm = float(times[i]), float(vals[i])
            label = lane.items[key]
            label.setText(f"{word} {bpm:.1f} bpm · {clock_text(at)[:8]}{src[3]}")
            # Above the high, below the low; slid sideways near either edge so the text stays in view.
            frac = (at - v0) / max(1e-9, v1 - v0)
            label.setAnchor((0.0 if frac < 0.12 else 1.0 if frac > 0.88 else 0.5, 1.35 if word == "max" else -0.35))
            label.setPos(at, bpm)
            label.setVisible(True)
            footer.append((f"{word} {bpm:.1f} bpm · {clock_text(at)[:8]}",
                           f"{'Highest' if word == 'max' else 'Lowest'} BPM of the whole recording"
                           f"{' (smoothed Annotation)' if src[3] else ''}. Click to jump there.",
                           lambda at=at: self.center_on(at, at)))
        lane.legend.set_footer(footer)

    @staticmethod
    def _add_item(lane: Lane, item) -> None:
        """Add to the lane, then enable view clipping/decimation (pyqtgraph needs the view first)."""
        lane.plot.addItem(item)
        if isinstance(item, pg.PlotDataItem):
            # ~1 bin per pixel (peak mode keeps each bin's min and max); the default 5 makes
            # zoomed-out envelopes draw ~10 points per pixel and dominate the frame time.
            item.opts["autoDownsampleFactor"] = 1.0
            item.setClipToView(True)
            item.setDownsampling(auto=True, method="peak" if item.opts.get("symbol") is None else "subsample")
            item.dataBounds = _whole_data_bounds(item)

    def _points_item(self, name: str, x, y, size: float = 5) -> pg.PlotDataItem:
        st = self._styles[name]
        hollow = st.hollow or st.symbol == "x"
        return pg.PlotDataItem(x, y, pen=None, symbol=st.symbol, symbolSize=size,
                               symbolPen=pg.mkPen(st.color, width=1.2) if hollow else None,
                               symbolBrush=None if hollow else qcolor(st.color))

    def _make_trace_item(self, t: Trace):
        st = self._styles[t.name]
        if t.uniform:
            return EnvelopeItem(t.t0, t.dt, t.y, theme.pen(st), fill=qcolor(theme.ENVELOPE_FILL) if st.fill else None)
        if t.kind == KIND_LINE:
            return pg.PlotDataItem(t.times(), t.y, pen=theme.pen(st), connect="finite")
        if t.kind == KIND_POINTS:
            return self._points_item(t.name, t.x, t.y, size=4)
        if t.lane == "states":
            labels = list(t.text) if t.text else ["unknown"] * t.size
            return SpanRowsItem([palette_row(t.x, t.x_end, labels, *ROW_BEFORE)])
        return BandsItem(t.x, t.x_end, st.color, alpha=theme.BAND_ALPHA)

    def _set_trace_shown(self, t: Trace, shown: bool) -> None:
        lane = self.lanes[t.lane]
        item = lane.items.get(t.name)
        if not self._lane_visible(t.lane):
            # Lanes outside the layout have no view; their items are created when the lane is shown.
            if item is not None:
                item.setVisible(shown)
            return
        if shown and item is None:
            item = self._make_trace_item(t)
            lane.items[t.name] = item
            self._add_item(lane, item)
        if item is not None:
            item.setVisible(shown)
        if t.lane == "states":
            self._update_states_range()

    def _update_states_range(self) -> None:
        extra = any(it.isVisible() for name, it in self.lanes["states"].items.items() if not name.startswith("_"))
        self.lanes["states"].plot.setYRange((ROW_BEFORE[0] if extra else ROW_ANN[0]) - 0.05, ROW_DEF[1] + 0.05,
                                            padding=0)

    def _peak_mask(self, name: str) -> np.ndarray:
        kinds = self._peak_kinds
        return kinds == {PEAKS_S1: "S1", PEAKS_S2: "S2", PEAKS_NOISE: "noise"}[name]

    def _set_peaks_shown(self, name: str, shown: bool) -> None:
        lane = self.lanes["signal"]
        item = lane.items.get(name)
        if not self._lane_visible("signal"):
            if item is not None:
                item.setVisible(shown)
            return
        if shown and item is None:
            m = self._peak_mask(name)
            p = self.analysis.peaks
            item = self._points_item(name, p.time[m], p.amp[m], size=5)
            item.setZValue(5)
            lane.items[name] = item
            self._add_item(lane, item)
        if item is not None:
            item.setVisible(shown)

    def _set_ann_bpm_shown(self, name: str, shown: bool) -> None:
        lane = self.lanes["bpm"]
        item = lane.items.get(name)
        if not self._lane_visible("bpm"):
            if item is not None:
                item.setVisible(shown)
            return
        if item is None:
            if name == ANN_BPM:
                item = pg.PlotDataItem(pen=theme.pen(self._styles[name], alpha=200))
            else:
                item = self._points_item(name, [], [], size=4)
            item.setZValue(4)
            lane.items[name] = item
            self._add_item(lane, item)
            if self._ann_bpm is not None:
                t, instant, smooth = self._ann_bpm
                item.setData(t, smooth if name == ANN_BPM else instant)
        item.setVisible(shown)

    def _draw_states(self) -> None:
        lane = self.lanes["states"]
        s = self.analysis.states
        start, end, names = s.start, s.end, list(s.state)
        order = np.argsort(start, kind="stable")
        later = np.zeros(len(start), dtype=bool)
        earlier = np.zeros(len(start), dtype=bool)
        run_end, run_idx = -np.inf, -1
        for i in order:
            if start[i] < run_end - 1e-9:
                later[i] = True
                earlier[run_idx] = True
            if end[i] > run_end:
                run_end, run_idx = end[i], i
        earlier &= ~later
        full = ~(later | earlier)
        mid = (ROW_ALG[0] + ROW_ALG[1]) / 2
        rows = []
        for mask, y0, y1, outline in ((full, ROW_ALG[0], ROW_ALG[1], None), (earlier, mid, ROW_ALG[1], theme.ISSUE),
                                      (later, ROW_ALG[0], mid, theme.ISSUE)):
            if mask.any():
                rows.append(palette_row(start[mask], end[mask], [names[i] for i in np.nonzero(mask)[0]], y0, y1,
                                        outline=outline))
        d = self.analysis.defects
        if d:
            rows.append(palette_row([x.start for x in d], [max(x.end, x.start + 0.02) for x in d],
                                    ["defect"] * len(d), *ROW_DEF, colors=theme.DEFECT_FILLS))
        self._set_internal(lane, "_alg", SpanRowsItem(rows))

    def _set_internal(self, lane: Lane, key: str, item) -> None:
        old = lane.items.pop(key, None)
        if old is not None:
            lane.plot.removeItem(old)
        if item is not None and self._lane_visible(lane.name):
            lane.items[key] = item
            self._add_item(lane, item)

    def _draw_annotation(self) -> None:
        lane = self.lanes["states"]
        spans = self.doc.spans if self.doc else ()
        rows = []
        if spans:
            rows.append(palette_row([s.start for s in spans], [s.end for s in spans],
                                    [f"{s.kind}|{s.origin}" for s in spans], *ROW_ANN, colors=theme.ANNOTATION_FILLS))
            clipped = [s for s in spans if s.clipped]
            if clipped:
                rows.append(palette_row([s.start for s in clipped], [s.end for s in clipped], ["c"] * len(clipped),
                                        ROW_ANN[1] - 0.12, ROW_ANN[1], colors={"c": theme.TEXT}))
        if self.disagreements:
            d = self.disagreements
            rows.append(palette_row([x.start for x in d], [x.end for x in d], [x.kind for x in d], *ROW_DIS,
                                    colors=theme.DISAGREEMENT_FILLS))
        if self.selected is not None:
            s = self.selected
            rows.append(SpanRow(np.array([s.start]), np.array([s.end]), [qcolor(theme.ACCENT, 0)], np.array([0]),
                                ROW_ANN[0] - 0.04, ROW_ANN[1] + 0.04, outline=qcolor(theme.ACCENT)))
        self._set_internal(lane, "_ann", SpanRowsItem(rows))

    def _envelope_trace(self) -> Optional[Trace]:
        found = next((t for t in self.analysis.traces if t.role == ROLE_ENVELOPE and t.uniform), None)
        return found or self.analysis.trace("Algorithm envelope")  # Analyses saved before trace roles

    def _draw_overview(self) -> None:
        t = self._envelope_trace()
        if t is not None and t.size:
            n_bins = 3000
            y = np.abs(t.y)
            per = max(1, len(y) // n_bins)
            k = len(y) // per
            yy = y[: k * per].reshape(k, per).max(axis=1)
            xx = t.t0 + (np.arange(k) * per + per / 2) * t.dt
            self.overview_curve.setData(xx, yy)
            top = float(np.percentile(yy, 99.5)) if len(yy) else 1.0
            self.overview.setYRange(0, top * 1.15, padding=0)
        else:
            self.overview_curve.setData([], [])

    def _fill_legends(self) -> None:
        """Each lane's legend: its offered traces in pipeline order (stage, then catalog order)."""
        if self.analysis is None:
            return
        rank = {g: i for i, g in enumerate(theme.stage_order(m.group for m in self._metas.values()))}
        by_lane: Dict[str, List[TraceMeta]] = {n: [] for n in self.lanes}
        for i, m in enumerate(self._metas.values()):
            if self._offered(m):
                by_lane.setdefault(m.lane, []).append(m)
        for lane_name, metas in by_lane.items():
            metas.sort(key=lambda m: rank.get(m.group, len(rank)))
            entries, prev = [], None
            for m in metas:
                debug = m.group not in theme.ALWAYS_LISTED
                entries.append(LegendEntry(
                    name=m.name, label=legend_label(m.name), tag=theme.STAGE_TAGS.get(m.group, m.group),
                    kind=m.kind, style=self._styles[m.name], shown=bool(self.trace_vis.get(m.name, m.visible)),
                    edge=theme.stage_color(m.group) if debug and not m.visible else "",
                    divider=prev is not None and m.group != prev))
                prev = m.group
            self.lanes[lane_name].legend.set_entries(entries)

    def _on_legend_toggled(self, name: str, on: bool) -> None:
        self.trace_vis[name] = on
        self.settings.set_trace_visibility(self.trace_vis)
        self._show(name, on)

    def _highlight(self, lane: Lane, name: Optional[str]) -> None:
        """Hovering a legend entry isolates its trace: the lane's other traces fade."""
        for key, item in lane.items.items():
            if key.startswith("_") and lane.name != "states":
                continue
            item.setOpacity(1.0 if name is None or key == name else theme.DIM_OPACITY)

    def set_lane_visible(self, name: str, on: bool) -> None:
        """The one path for showing/hiding a lane (lane legends, the lane bar and the G key)."""
        if self._lane_visible(name) == on:
            return
        self.lane_vis[name] = on
        self.settings.set_lane_visibility(self.lane_vis)
        self._layout_lanes()
        if name == "spectrogram" and not on:
            self._drop_spectrogram()
        if self.analysis is not None and on:
            self._realize_lane(name)

    def _realize_lane(self, lane_name: str) -> None:
        """Create the items of a lane that just became visible."""
        for name, m in self._metas.items():
            if m.lane == lane_name:
                self._show(name, self._wanted(name))
        if lane_name == "states":
            self._draw_states()
        if lane_name in ("bpm", "states"):
            self._on_doc_changed()
        if lane_name == "spectrogram":
            if self._spec is None:
                self._compute_spectrogram()
            else:
                self._draw_spectrogram()
        if lane_name not in self._y_manual:
            self._auto_y(lane_name)

    def visible_trace_names(self) -> List[str]:
        out = []
        for lane in self.lanes.values():
            if not self._lane_visible(lane.name):
                continue
            for name, item in lane.items.items():
                if not name.startswith("_") and item.isVisible() and self.analysis.trace(name) is not None:
                    out.append(name)
        return out

    # ─────────────────────────────────────────────────────────────────────
    # View sync, playhead
    # ─────────────────────────────────────────────────────────────────────
    def view_range(self) -> Tuple[float, float]:
        return tuple(self.lanes["states"].plot.getViewBox().viewRange()[0])

    def _main_range_changed(self, _vb, rng) -> None:
        if self._syncing:
            return
        self._syncing = True
        self.overview_region.setRegion(rng)
        self._syncing = False

    def center_view_at(self, x: float) -> None:
        """Move the view window (keeping its width) so it is centred on x, inside the recording."""
        if self.analysis is None:
            return
        t0, t1 = self.view_range()
        a = min(max(x - (t1 - t0) / 2, 0.0), max(0.0, self.analysis.duration_sec - (t1 - t0)))
        self.lanes["states"].plot.setXRange(a, a + (t1 - t0), padding=0)

    def scale_view(self, factor: float) -> None:
        """Grow (>1) or shrink (<1) the view window around its centre, within the recording."""
        if self.analysis is None:
            return
        t0, t1 = self.view_range()
        dur = self.analysis.duration_sec
        width = min(max((t1 - t0) * factor, 0.5), dur)
        a = min(max((t0 + t1) / 2 - width / 2, 0.0), dur - width)
        self.lanes["states"].plot.setXRange(a, a + width, padding=0)

    def center_on(self, a: float, b: float) -> None:
        t0, t1 = self.view_range()
        w = t1 - t0
        if b - a > w * 0.8:
            w = (b - a) * 1.4
        c = (a + b) / 2
        self.lanes["states"].plot.setXRange(c - w / 2, c + w / 2, padding=0)

    def _tick(self) -> None:
        if not self.player.loaded:
            return
        t = self.player.position
        for lane in self.lanes.values():
            lane.playhead.setValue(t)
        self.overview_playhead.setValue(t)
        if self.player.playing:
            t0, t1 = self.view_range()
            if t > t1 - 0.02 * (t1 - t0) or t < t0:
                w = t1 - t0
                self.lanes["states"].plot.setXRange(t - 0.02 * w, t + 0.98 * w, padding=0)

    def player_toggle(self) -> None:
        self.player.toggle()

    def toggle_source(self) -> None:
        if not self.source_btn.isEnabled():
            return
        self.player.toggle_source()
        self._show_source(self._source_text())

    def _source_text(self) -> str:
        loading = "" if self.player.has(SOURCE_FILTERED) else "  (filtered loading…)"
        return f"{self.player.source.capitalize()} audio{loading}"

    def _show_source(self, text: str, enabled: bool = True) -> None:
        self.source_btn.setText(text)
        self.source_btn.setEnabled(enabled)
        self.source_btn.updateGeometry()

    # ─────────────────────────────────────────────────────────────────────
    # Mouse
    # ─────────────────────────────────────────────────────────────────────
    def zoom_y(self, lane_name: str, center: float, factor: float) -> None:
        plot = self.lanes[lane_name].plot
        lo, hi = plot.getViewBox().viewRange()[1]
        new = [center - (center - lo) * factor, center + (hi - center) * factor]
        plot.setYRange(*new, padding=0)  # also switches y auto-range off
        self._y_manual[lane_name] = new
        self.settings.set_y_ranges(self.fingerprint, self._y_manual)

    def pan_y(self, lane_name: str, dy: float, *, final: bool) -> None:
        plot = self.lanes[lane_name].plot
        lo, hi = plot.getViewBox().viewRange()[1]
        plot.setYRange(lo + dy, hi + dy, padding=0)
        self._y_manual[lane_name] = [lo + dy, hi + dy]
        if final:
            self.settings.set_y_ranges(self.fingerprint, self._y_manual)

    def reset_y(self, lane_name: str) -> None:
        self._y_manual.pop(lane_name, None)
        self.settings.set_y_ranges(self.fingerprint, self._y_manual)
        self._auto_y(lane_name)

    def _auto_y(self, lane_name: str) -> None:
        plot = self.lanes[lane_name].plot
        if lane_name == "spectrogram":
            plot.setYRange(0, 1000, padding=0)
        elif lane_name == "signal" and self.analysis is not None and self._envelope_trace() is not None:
            plot.setYRange(*self._signal_y_range(), padding=0)
        elif lane_name != "states":
            plot.enableAutoRange(axis="y", enable=True)

    def _signal_y_range(self) -> Tuple[float, float]:
        """Fit the envelope's ordinary peaks, not its rare artifact spikes: the 99th percentile of
        per-block maxima, plus headroom for the peak markers."""
        y = self._envelope_trace().y
        per = max(1, len(y) // 4000)
        k = len(y) // per
        with np.errstate(all="ignore"):
            maxima = np.nanmax(y[: k * per].reshape(k, per), axis=1) if k else y
            top = float(np.nanpercentile(maxima, 99)) if len(maxima) else 1.0
            lo = float(min(0.0, np.nanmin(y))) if len(y) else 0.0
        return lo, (top * 1.12 if np.isfinite(top) and top > lo else lo + 1.0)

    def _restore_y(self) -> None:
        """Re-apply the y-ranges saved for this Recording; every other lane gets its automatic range."""
        saved = self.settings.y_ranges(self.fingerprint)
        for name, lane in self.lanes.items():
            if name in saved:
                lane.plot.setYRange(*saved[name], padding=0)
            else:
                self._auto_y(name)
        self._y_manual = dict(saved)

    def click(self, lane: str, x: float, y: float) -> None:
        self.player.seek(x)
        self._tick()
        if lane == "states" and self.doc is not None and ROW_ANN[0] <= y <= ROW_ANN[1]:
            self.selected = an.span_at(self.doc.spans, x)
            self._draw_annotation()

    def set_region(self, region: Optional[Tuple[float, float]], final: bool) -> None:
        if region is not None:
            a, b = sorted(region)
            region = (a, b) if b - a > 1e-3 else None
        self.region = region
        for lane in self.lanes.values():
            if region is None:
                lane.region.hide()
            else:
                lane.region.setRegion(region)
                lane.region.show()
        if final:
            self.player.set_loop(region)

    def preview_noisy(self, region: Tuple[float, float], final: bool, replacing: Optional[an.Span] = None) -> None:
        lane = self.lanes["states"]
        a, b = sorted(region)
        if not final:
            self._set_internal(lane, "_noisy_preview", SpanRowsItem([
                SpanRow(np.array([a]), np.array([b]), [qcolor(theme.NOISE, 140)], np.array([0]), *ROW_ANN)]))
            return
        self._set_internal(lane, "_noisy_preview", None)
        if self.doc is None:
            return
        dur = self.analysis.duration_sec
        if replacing is not None:
            self.doc.apply(lambda s: an.resize_noisy(s, replacing, a, b, dur))
        else:
            self.doc.apply(lambda s: an.paint_noisy(s, a, b, dur))

    def noisy_edge_at(self, vb: pg.ViewBox, x: float, y: float) -> Optional[Tuple[an.Span, str]]:
        if self.doc is None or not (ROW_ANN[0] <= y <= ROW_ANN[1]):
            return None
        px = self._sec_per_px(vb)
        best = None
        for s in self.doc.spans:
            if s.kind != an.NOISY or s.end < x - 10 * px or s.start > x + 10 * px:
                continue
            for side, t in (("start", s.start), ("end", s.end)):
                d = abs(t - x) / px
                if d <= 6 and (best is None or d < best[0]):
                    best = (d, s, side)
        return (best[1], best[2]) if best else None

    @staticmethod
    def _sec_per_px(vb: pg.ViewBox) -> float:
        (t0, t1), _ = vb.viewRange()
        return (t1 - t0) / max(1.0, vb.width())

    def _on_mouse_moved(self, scene_pos, lane: Lane) -> None:
        if self.analysis is None or not lane.plot.sceneBoundingRect().contains(scene_pos):
            return
        vb = lane.plot.getViewBox()
        pt = vb.mapSceneToView(scene_pos)
        self.status.setText(f"{clock_text(pt.x())}   {pt.x():.3f} s   {lane.name}: {pt.y():.4g}")
        text = self._hover_text(lane, vb, pt.x(), pt.y(), scene_pos)
        view = lane.widget
        if text:
            gpos = view.mapToGlobal(view.mapFromScene(scene_pos))
            QtWidgets.QToolTip.showText(gpos + QtCore.QPoint(14, 10), text, view)
        else:
            QtWidgets.QToolTip.hideText()

    def _hover_text(self, lane: Lane, vb, x: float, y: float, scene_pos) -> str:
        spp = self._sec_per_px(vb)
        if lane.name == "signal":
            p = self.analysis.peaks
            lo = np.searchsorted(p.time, x - HOVER_PX * spp)
            hi = np.searchsorted(p.time, x + HOVER_PX * spp)
            best, best_d = None, HOVER_PX
            for i in range(lo, hi):
                name = {"S1": PEAKS_S1, "S2": PEAKS_S2, "noise": PEAKS_NOISE}[self._peak_kinds[i]]
                item = lane.items.get(name)
                if item is None or not item.isVisible():
                    continue
                sp = vb.mapViewToScene(QtCore.QPointF(p.time[i], p.amp[i]))
                d = ((sp.x() - scene_pos.x()) ** 2 + (sp.y() - scene_pos.y()) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = i, d
            if best is not None:
                return "\n".join(A.peak_reasoning_text(self.analysis, best))
        if lane.name == "states":
            if ROW_ALG[0] <= y <= ROW_ALG[1]:
                s = self.analysis.states
                hits = np.nonzero((s.start <= x) & (s.end >= x))[0]
                parts = []
                for j in hits[:3]:
                    src = f" [{s.source[j]}]" if s.source[j] else ""
                    parts.append(f"{s.state[j]} {s.start[j]:.3f}–{s.end[j]:.3f}s{src}\n" + "\n".join(s.reasoning[j]))
                return "\n\n".join(parts)
            if ROW_ANN[0] <= y <= ROW_ANN[1] and self.doc is not None:
                sp = an.span_at(self.doc.spans, x)
                if sp is not None:
                    return f"{sp.kind} {sp.start:.3f}–{sp.end:.3f}s ({sp.origin}{', clipped' if sp.clipped else ''})"
            if ROW_DIS[0] - 0.05 <= y <= ROW_DIS[1] + 0.05:
                d = [x_ for x_ in self.disagreements if x_.start - spp * 3 <= x <= x_.end + spp * 3]
                return "\n".join(f"{x_.kind}: {x_.message}" for x_ in d[:5])
            if ROW_DEF[0] - 0.05 <= y <= ROW_DEF[1] + 0.05:
                d = [x_ for x_ in self.analysis.defects if x_.start - spp * 3 <= x <= x_.end + spp * 3]
                return "\n".join(f"{x_.kind}: {x_.message}" for x_ in d[:5])
        for name, item in lane.items.items():
            t = self.analysis.trace(name)
            if t is None or not item.isVisible() or not t.text:
                continue
            if t.kind == KIND_SPANS:
                hit = np.nonzero((t.x <= x) & (t.x_end >= x))[0]
                if len(hit):
                    return f"{t.name}\n{t.text[hit[0]]}"
            elif t.kind == KIND_POINTS:
                hit = np.nonzero(np.abs(t.x - x) <= HOVER_PX * spp)[0]
                if len(hit):
                    return f"{t.name} {t.x[hit[0]]:.3f}s: {t.y[hit[0]]:.4g}\n{t.text[hit[0]]}"
        return ""

    # ─────────────────────────────────────────────────────────────────────
    # Annotation editing
    # ─────────────────────────────────────────────────────────────────────
    def _edit_target(self) -> Optional[an.Span]:
        if self.doc is None:
            return None
        if self.selected is not None and self.selected in self.doc.spans:
            return self.selected
        return an.span_at(self.doc.spans, self.player.position)

    def place(self, kind: str) -> None:
        if self.doc is None:
            return
        length = float(param(self.analysis.params, "s1_nominal_sec" if kind == an.S1 else "s2_nominal_sec"))
        t = self.player.position
        self.doc.apply(lambda s: an.place_sound(s, kind, t, length, self.analysis.duration_sec))

    def delete_span(self) -> None:
        target = self._edit_target()
        if target is not None:
            self.selected = None
            self.doc.apply(lambda s: an.delete(s, target))

    def relabel_span(self) -> None:
        target = self._edit_target()
        if target is not None and target.kind in an.SOUNDS:
            self.selected = None
            self.doc.apply(lambda s: an.relabel(s, target))

    def flip_after_playhead(self) -> None:
        if self.doc is not None:
            t = self.player.position
            self.doc.apply(lambda s: an.flip_after(s, t))

    def replace_region_from_algorithm(self) -> None:
        if self.doc is None or self.region is None:
            self.status.setText("Shift+drag a region first, then press A to replace it with the algorithm's output.")
            return
        a, b = self.region
        self.doc.apply(lambda s: an.replace_region(s, a, b, self.algo_spans))

    def undo(self) -> None:
        if self.doc:
            self.selected = None
            self.doc.undo()

    def redo(self) -> None:
        if self.doc:
            self.selected = None
            self.doc.redo()

    def _on_doc_changed(self) -> None:
        if self.doc is None or self.analysis is None:
            return
        self.disagreements = an.disagreements(self.doc.spans, self.algo_spans)
        if self.selected is not None and self.selected not in self.doc.spans:
            self.selected = None
        self._draw_annotation()
        t, instant = an.bpm_series(self.doc.spans)
        window = float(param(self.analysis.params, "output_smoothing_window_sec"))
        smooth = an.smooth_series(t, instant, max(0.05, window / 3.0))
        self._ann_bpm = (t, instant, smooth)
        for name, values in ((ANN_BPM, smooth), (ANN_BPM_INSTANT, instant)):
            item = self.lanes["bpm"].items.get(name)
            if item is not None:
                item.setData(t, values)
        self._update_bpm_extremes()
        self._issues_timer.start()
        self._emit_title()

    def _emit_title(self) -> None:
        if not self.path:
            self.title_changed.emit("")
            return
        name = Path(self.path).name
        mark = ""
        if self.doc is not None:
            mark = " [draft]" if self.doc.draft else ""
            mark += " *" if self.doc.dirty else ""
        self.title_changed.emit(f"{name}{mark}")

    def save_annotation(self) -> bool:
        if self.doc is None:
            return False
        protected = recording.is_inside(self.path, self.settings.protected_folders)
        target = self.doc.path
        if protected:
            start = downloads_dir() / an.sidecar_path(self.path).name
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Annotation (recording is in a protected folder)", str(start),
                f"Annotations (*{an.SUFFIX})")
            if not path:
                return False
            target = Path(path)
        elif target is None:
            target = an.sidecar_path(self.path)
        self.doc.save_to(Path(target))
        self.status.setText(f"Saved {target}")
        return True

    # ─────────────────────────────────────────────────────────────────────
    # Issues panel
    # ─────────────────────────────────────────────────────────────────────
    def _refresh_issues(self) -> None:
        if self.analysis is None:
            return
        issues = [("Defect", d.kind, d.start, d.end, d.message) for d in self.analysis.defects]
        issues += [("Disagreement", d.kind, d.start, d.end, d.message) for d in self.disagreements]
        issues.sort(key=lambda r: r[2])
        self._issues = issues
        current = self.issue_filter.currentData()
        counts: Dict[Tuple[str, str], int] = {}
        for src, kind, *_ in issues:
            counts[(src, kind)] = counts.get((src, kind), 0) + 1
        self.issues_btn.setText(f"Issues  {len(issues)}" if issues else "Issues")
        self.issue_filter.blockSignals(True)
        self.issue_filter.clear()
        self.issue_filter.addItem(f"All ({len(issues)})", None)
        for src in ("Defect", "Disagreement"):
            n = sum(v for (s, _), v in counts.items() if s == src)
            self.issue_filter.addItem(f"{src}s ({n})", (src, None))
            for (s, kind), v in sorted(counts.items()):
                if s == src:
                    self.issue_filter.addItem(f"    {kind} ({v})", (s, kind))
        idx = self.issue_filter.findData(current)
        self.issue_filter.setCurrentIndex(max(0, idx))
        self.issue_filter.blockSignals(False)
        self._fill_issue_list()

    def _filtered_issues(self):
        f = self.issue_filter.currentData()
        if not f:
            return self._issues
        src, kind = f
        return [r for r in self._issues if r[0] == src and (kind is None or r[1] == kind)]

    def _fill_issue_list(self) -> None:
        rows = self._filtered_issues()
        self.issue_list.setUpdatesEnabled(False)
        self.issue_list.clear()
        shown = rows[:5000]
        self.issue_list.addItems([f"{clock_text(s)}  {kind}: {msg}" for _src, kind, s, _e, msg in shown])
        if len(rows) > len(shown):
            self.issue_list.addItem(f"… {len(rows) - len(shown)} more (use ] / [)")
        self.issue_list.setUpdatesEnabled(True)

    def _jump_to_issue(self, row: int) -> None:
        rows = self._filtered_issues()
        if 0 <= row < len(rows):
            _src, _kind, a, b, msg = rows[row]
            self.center_on(a, b)
            self.status.setText(f"{a:.3f}–{b:.3f}s  {msg}")

    def step_issue(self, direction: int) -> None:
        rows = self._filtered_issues()
        if not rows:
            return
        t0, t1 = self.view_range()
        c = (t0 + t1) / 2
        starts = np.array([r[2] for r in rows])
        if direction > 0:
            i = int(np.searchsorted(starts, c + 1e-6, side="right"))
            i = i if i < len(rows) else 0
        else:
            i = int(np.searchsorted(starts, c - 1e-6, side="left")) - 1
            i = i if i >= 0 else len(rows) - 1
        self._jump_to_issue(i)
        if i < self.issue_list.count():
            self.issue_list.setCurrentRow(i)

    # ─────────────────────────────────────────────────────────────────────
    # Header / summary
    # ─────────────────────────────────────────────────────────────────────
    def _update_header(self) -> None:
        a = self.analysis
        bpm = (a.summary or {}).get("bpm") or {}
        gate = (a.summary or {}).get("gate") or {}
        self.algo_badge.setText(f"Algorithm: {a.algorithm_used}")
        self.algo_badge.show()
        parts = []
        # The filename often carries the same BPM tag (written by "rename"); show the BPM here only when
        # it adds something: the name has no tag, or an older run's tag that no longer matches.
        if bpm and batch.format_bpm_tag(bpm["start_bpm"], bpm["min_bpm"], bpm["max_bpm"]) not in Path(self.path).stem:
            parts.append(f"{bpm['start_bpm']:.0f} bpm · {bpm['min_bpm']:.0f}–{bpm['max_bpm']:.0f}")
        if a.channel != recording.CHANNEL_MIXED:
            parts.append(f"{a.channel} channel")
        self.info_label.setText("  ·  ".join(parts))
        self.gate_badge.setVisible(bool(gate.get("failed")))
        self.gate_badge.setToolTip("; ".join(gate.get("reasons") or []))
        self.stale_badge.setVisible(bool(a.stale_reasons))
        self.stale_badge.setToolTip("\n".join(a.stale_reasons) + "\nCtrl+R to re-run" if a.stale_reasons else "")
        self.summary_text.setHtml(self._summary_html())

    def _summary_html(self) -> str:
        a = self.analysis
        bpm = (a.summary or {}).get("bpm") or {}
        gate = (a.summary or {}).get("gate") or {}
        hrv = (a.summary or {}).get("hrv") or {}
        mono = f"font-family:{', '.join(theme.MONO_FONTS)};"  # unquoted: it goes inside quoted attributes

        def card(label: str, value: str, color: str = theme.TEXT) -> str:
            # QTextDocument styles spans, not divs, inside table cells.
            return (f"<td bgcolor='{theme.SURFACE_3}'><span style='color:{theme.TEXT_3}'>{label}</span><br>"
                    f"<span style='{mono} font-size:14pt; color:{color}'>{value}</span></td>")

        def section(title: str, rows: List[Tuple[str, str, bool]]) -> str:
            body = "".join(
                f"<tr><td style='color:{theme.TEXT_2}; padding:2px 14px 2px 4px'>{escape(k)}</td>"
                f"<td style='{mono if is_num else ''} color:{theme.TEXT}; padding:2px 0'>{escape(v)}</td></tr>"
                for k, v, is_num in rows)
            return (f"<p style='color:{theme.TEXT_3}; margin:16px 0 4px 4px'>{title}</p>"
                    f"<table cellspacing='0' cellpadding='0'>{body}</table>")

        fmt = lambda v: f"{v:.1f}" if isinstance(v, (int, float)) else "—"  # noqa: E731
        cards = "<table width='100%' cellspacing='4' cellpadding='8'><tr>" + "".join((
            card("Mean BPM", fmt(hrv.get("avg_bpm")), theme.METRIC),
            card("Range", f"{bpm['min_bpm']:.0f}–{bpm['max_bpm']:.0f}" if bpm else "—"),
            card("Defects", str(len(a.defects)), theme.TONES["issue"][0] if a.defects else theme.TEXT),
        )) + "</tr></table>"
        hrv_rows = [(HRV_LABELS.get(k, k.replace("_", " ").capitalize()), f"{v:.4g}", True) for k, v in hrv.items()]
        gate_text = "Failed: " + "; ".join(gate.get("reasons") or []) if gate.get("failed") else "Passed"
        run_rows = [
            ("Algorithm", a.algorithm_used, False),
            ("Auto-switch", a.algorithm_switch_reason or "Off", False),
            ("Plausibility gate", gate_text, False),
            ("Channel", a.channel.capitalize(), False),
            ("Start BPM", f"{bpm['start_bpm']:.1f}" if bpm else "—", True),
            ("Start BPM hint", f"{a.start_bpm_hint:g}" if a.start_bpm_hint else "—", True),
        ]
        run_rows += [(RUN_SETTING_LABELS.get(k, k.replace("_", " ").capitalize()),
                      ("On" if v else "Off") if isinstance(v, bool) else f"{v:g}" if isinstance(v, float) else str(v),
                      isinstance(v, (int, float)) and not isinstance(v, bool))
                     for k, v in (a.run_settings or {}).items() if k not in ("use_springer_algorithm",)]
        try:
            created = datetime.fromisoformat(a.created)
            created_text = f"{created.day} {created:%b %Y, %H:%M}"
        except (TypeError, ValueError):
            created_text = str(a.created)
        file_rows = [
            ("Recording", Path(self.path).name, False),
            ("Analysed", created_text, False),
            ("Status", "Stale: " + "; ".join(a.stale_reasons) if a.stale_reasons else "Current", False),
            ("Fingerprint", a.fingerprint[:16], True),
        ]
        return (f"<body style='color:{theme.TEXT}'>{cards}"
                + section("HRV", hrv_rows) + section("Run", run_rows) + section("File", file_rows) + "</body>")

    # ─────────────────────────────────────────────────────────────────────
    # Re-run, context, export
    # ─────────────────────────────────────────────────────────────────────
    def rerun(self) -> None:
        if not self.path or self.worker.running:
            return
        a = self.analysis
        rs = a.run_settings if a else {}
        self.worker.start(
            self.path, str(self.library.root),
            springer=bool(rs.get("use_springer_algorithm", False)),
            auto_switch=bool(rs.get("auto_switch_algorithm", False)),
            channel=a.channel if a else recording.CHANNEL_MIXED,
            start_sec=float(rs.get("analysis_start_sec", 0.0)),
            bpm_hint=a.start_bpm_hint if a else None,
        )
        self.progress_label.setText("Analyzing…")
        self.cancel_btn.show()
        self.rerun_btn.setEnabled(False)

    def _on_worker_progress(self, msg: str) -> None:
        self.progress_label.setText(msg)

    def _on_worker_finished(self, paths: list, error: str) -> None:
        self.cancel_btn.hide()
        self.rerun_btn.setEnabled(True)
        if error or not paths:
            self.progress_label.setText(f"Analysis failed: {error}" if error != "cancelled" else "Cancelled")
            return
        self.progress_label.setText("")
        current = self.analysis.channel if self.analysis else None
        chosen = next((p for p in paths if current and p.endswith(f"-{current}.analysis.zip")), paths[0])
        self._load_analysis(chosen, keep_view=self.analysis is not None)
        if not self.player.loaded:
            self._load_audio()
        self.analyses_changed.emit()

    def copy_context(self) -> None:
        if self.analysis is None:
            return
        t0, t1 = self.view_range()
        text = context.window_text(
            self.analysis, t0, t1, recording_path=self.path, visible_traces=self.visible_trace_names(),
            annotation=self.doc.ann if self.doc else None, disagreements=self.disagreements,
        )
        QtWidgets.QApplication.clipboard().setText(text)
        self.status.setText(f"Copied context for {t0:.2f}–{t1:.2f}s ({len(text.splitlines())} lines)")

    def export_dialog(self) -> None:
        if self.analysis is None:
            return
        dlg = ExportDialog(self, self.settings, Path(self.path).stem, has_annotation=self.doc is not None,
                           has_region=self.region is not None)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        try:
            self._export(dlg.fmt, Path(dlg.path), dlg.opts)
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", f"{dlg.path}\n\n{e}")
            return
        self.status.setText(f"Exported {dlg.path}")
        if dlg.open_after:
            reveal_in_folder(dlg.path)

    def _export_range(self, which: str) -> Tuple[float, float]:
        if which == "region" and self.region is not None:
            return tuple(sorted(self.region))
        if which == "whole":
            return 0.0, float("inf")
        return self.view_range()

    def _export(self, fmt: str, path: Path, opts: dict) -> None:
        from pcg.cli import summary_text, write_bpm_csv

        a = self.analysis
        t0, t1 = self._export_range(opts["range"])
        if fmt in ("csv", "chart"):
            if opts["source"] == "annotation" and self.doc is not None:
                t, bpm = an.bpm_series(self.doc.spans)
                header = "bpm_annotation"
            else:
                tr = a.trace("BPM")
                if tr is None:
                    raise OSError("This Analysis has no BPM trace.")
                t, bpm, header = tr.times(), tr.y, "bpm"
            t, bpm = np.asarray(t), np.asarray(bpm)
            if fmt == "chart":
                env = self._envelope_trace()
                if env is None:
                    raise OSError("This Analysis has no envelope to draw the waveform from.")
                if header == "bpm_annotation":
                    t, bpm = annotation_bpm_curve(self.doc.spans, a.params)
                img = render_bpm_chart(t, bpm, env.t0, env.dt, env.y, max(t0, 0.0), min(t1, a.duration_sec))
                if not img.save(str(path)):
                    raise OSError("Could not write the image.")
                return
            keep = (t >= t0) & (t <= t1)
            write_bpm_csv(path, t[keep], bpm[keep], header, opts["time_format"])
        elif fmt == "summary":
            path.write_text(summary_text(a), encoding="utf-8")
        elif fmt == "context":
            path.write_text(context.window_text(
                a, t0, t1, recording_path=self.path, visible_traces=self.visible_trace_names(),
                annotation=self.doc.ann if self.doc else None, disagreements=self.disagreements), encoding="utf-8")
        elif fmt == "image":
            target = self.lanes_box if opts["overview"] else self.lane_split
            if not target.grab().save(str(path)):
                raise OSError("Could not write the image.")

    def shutdown(self) -> None:
        self.player.stop()
        self.worker.cancel()
