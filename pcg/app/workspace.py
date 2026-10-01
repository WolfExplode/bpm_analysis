"""Workspace screen: one Recording, its Analysis and Annotation on stacked, linked lanes."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from pcg import analysis as A
from pcg import batch
from pcg import annotation as an
from pcg import context, recording
from pcg.engine.traces import KIND_LINE, KIND_POINTS, KIND_SPANS, Trace

from .audio import SOURCE_FILTERED, Player, load_playback_audio, spectrogram
from .items import BandsItem, EnvelopeItem, SpanRow, SpanRowsItem, TimeAxis, clock_text, palette_row, qcolor
from .state import AnalyzeWorker, AnnotationDoc, Settings, downloads_dir

STATES_DEFAULT_HEIGHT = 160  # px, until the user drags a handle; the other lanes share the rest
LANE_ORDER = ["states", "signal", "bpm", "intervals", "scores", "hrv", "contractility", "spectrogram"]
LANE_TITLES = {
    "states": "States", "signal": "Signal", "bpm": "Heart rate (BPM)", "intervals": "Intervals (s)",
    "scores": "Classifier scores (%)", "hrv": "HRV", "contractility": "Contractility",
    "spectrogram": "Spectrogram (Hz)",
}
DEFAULT_LANES = {"states", "signal", "bpm"}

# Pseudo-traces the workspace draws itself (not from the Analysis's trace catalog).
PEAKS_S1, PEAKS_S2, PEAKS_NOISE = "Peaks: S1", "Peaks: S2", "Peaks: noise"
ANN_BPM = "BPM from Annotation"
PEAK_COLORS = {PEAKS_S1: "#ff5c5c", PEAKS_S2: "#ffb347", PEAKS_NOISE: "#9e9e9e"}

# States lane rows (y ranges)
ROW_BEFORE = (-1.0, -0.1)
ROW_ANN = (0.0, 1.0)
ROW_DIS = (1.02, 1.13)
ROW_ALG = (1.15, 2.15)
ROW_DEF = (2.17, 2.28)

ANN_COLORS = {
    "S1|hand": "#ff6b6b", "S2|hand": "#ffb347", "noisy|hand": "#8a8a8a",
    "S1|algorithm": "#c45656", "S2|algorithm": "#c98a3a", "noisy|algorithm": "#707070",
}
DIS_COLORS = {"missed": "#ff3d7f", "swapped": "#ffe14d", "extra": "#4dd2ff"}
DEFECT_COLOR = "#ff2b2b"
HOVER_PX = 8


@dataclass
class Lane:
    name: str
    plot: pg.PlotItem
    widget: Optional[pg.PlotWidget] = None
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
    _audio_error = QtCore.Signal(str)

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
        pg.setConfigOptions(antialias=False, background="#121212", foreground="#cfcfcf")
        top = QtWidgets.QHBoxLayout()
        self.name_label = QtWidgets.QLabel("No recording")
        self.name_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        f = self.name_label.font()
        f.setBold(True)
        self.name_label.setFont(f)
        self.stale_badge = QtWidgets.QLabel()
        self.stale_badge.setStyleSheet("background:#8a5a00;color:white;padding:1px 6px;border-radius:3px;")
        self.stale_badge.hide()
        self.info_label = QtWidgets.QLabel()
        self.progress_label = QtWidgets.QLabel()
        self.progress_label.setStyleSheet("color:#7fc7ff;")
        self.rerun_btn = QtWidgets.QPushButton("Re-run (Ctrl+R)", clicked=self.rerun)
        self.cancel_btn = QtWidgets.QPushButton("Cancel", clicked=self.worker.cancel)
        self.cancel_btn.hide()
        self.flip_btn = QtWidgets.QPushButton("Flip S1/S2 right of playhead", clicked=self.flip_after_playhead)
        self.follow_btn = QtWidgets.QPushButton("Follow playhead (L)", checkable=True, checked=True)
        self.source_btn = QtWidgets.QPushButton(clicked=self.toggle_source)
        self.source_btn.setToolTip("Switch between the original and the band-passed audio (T)")
        self._show_source("Audio: loading…", enabled=False)
        for w in (self.name_label, self.stale_badge, self.info_label):
            top.addWidget(w)
        top.addStretch(1)
        for w in (self.progress_label, self.cancel_btn, self.rerun_btn, self.flip_btn, self.follow_btn,
                  self.source_btn):
            top.addWidget(w)

        # One widget per lane in a vertical splitter, so lanes can be resized by dragging between them.
        self.lane_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.lane_split.setHandleWidth(5)
        self.lane_split.setChildrenCollapsible(False)
        self.lane_split.setStyleSheet("QSplitter::handle { background: #2e2e2e; } "
                                      "QSplitter::handle:hover { background: #5a7fb5; }")
        self.lane_split.splitterMoved.connect(self._save_lane_heights)

        self.overview = pg.PlotItem(viewBox=OverviewViewBox(self), axisItems={"bottom": TimeAxis("bottom")})
        self.overview.setMouseEnabled(x=False, y=False)
        self.overview.setMenuEnabled(False)
        self.overview.hideButtons()
        self.overview.getAxis("left").setWidth(70)
        # Its own fixed-height widget: a maximum height inside the lanes' layout collapses every row.
        self.overview_widget = pg.PlotWidget(plotItem=self.overview)
        self.overview_widget.setFixedHeight(80)
        self.overview_region = pg.LinearRegionItem(brush=(80, 140, 255, 50), movable=False)
        self.overview_curve = pg.PlotDataItem(pen=pg.mkPen("#47a5c4"))
        self.overview_playhead = pg.InfiniteLine(angle=90, pen=pg.mkPen("#ffffff", width=1))
        for it in (self.overview_curve, self.overview_region, self.overview_playhead):
            self.overview.addItem(it)

        for name in LANE_ORDER:
            self._make_lane(name)

        side = QtWidgets.QTabWidget()
        self.trace_tree = QtWidgets.QTreeWidget()
        self.trace_tree.setHeaderHidden(True)
        self.trace_tree.itemChanged.connect(self._on_tree_changed)
        side.addTab(self.trace_tree, "Traces")

        issues = QtWidgets.QWidget()
        il = QtWidgets.QVBoxLayout(issues)
        il.setContentsMargins(2, 2, 2, 2)
        self.issue_filter = QtWidgets.QComboBox()
        self.issue_filter.currentIndexChanged.connect(self._fill_issue_list)
        self.issue_list = QtWidgets.QListWidget()
        self.issue_list.itemActivated.connect(lambda it: self._jump_to_issue(self.issue_list.row(it)))
        self.issue_list.itemClicked.connect(lambda it: self._jump_to_issue(self.issue_list.row(it)))
        il.addWidget(self.issue_filter)
        il.addWidget(self.issue_list)
        il.addWidget(QtWidgets.QLabel("] / [  next / previous"))
        side.addTab(issues, "Defects && Disagreements")

        self.summary_text = QtWidgets.QTextBrowser()
        side.addTab(self.summary_text, "Summary")

        split = QtWidgets.QSplitter()
        lanes_box = QtWidgets.QWidget()
        lb = QtWidgets.QVBoxLayout(lanes_box)
        lb.setContentsMargins(0, 0, 0, 0)
        lb.setSpacing(0)
        lb.addWidget(self.overview_widget)
        lb.addWidget(self.lane_split, 1)
        split.addWidget(lanes_box)
        split.addWidget(side)
        split.setStretchFactor(0, 1)
        split.setSizes([1400, 360])

        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet("color:#aaaaaa;")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addLayout(top)
        lay.addWidget(split, 1)
        lay.addWidget(self.status)
        self._layout_lanes()

    def _make_lane(self, name: str) -> None:
        vb = LaneViewBox(self, name)
        plot = pg.PlotItem(viewBox=vb, axisItems={"bottom": TimeAxis("bottom")})
        plot.hideButtons()
        plot.getAxis("left").setWidth(70)
        plot.getAxis("left").enableAutoSIPrefix(False)
        plot.setLabel("left", LANE_TITLES.get(name, name))
        plot.showGrid(x=True, y=False, alpha=0.15)
        if name == "spectrogram":
            plot.setYRange(0, 1000, padding=0)
        elif name == "states":
            plot.setYRange(ROW_ANN[0] - 0.05, ROW_DEF[1] + 0.05, padding=0)
            plot.getAxis("left").setTicks([[(0.5, "Annotation"), (1.65, "Analysis"), (-0.55, "Before repair")]])
            plot.setLabel("left", "")
        else:
            plot.enableAutoRange(axis="y", enable=True)
            plot.setAutoVisible(y=False)  # fit the whole recording, not the visible window, so panning never rescales
        lane = Lane(name, plot)
        lane.widget = pg.PlotWidget(plotItem=plot)
        lane.widget.setMinimumHeight(40)
        lane.widget.scene().sigMouseMoved.connect(lambda pos, lane=lane: self._on_mouse_moved(pos, lane))
        lane.playhead = pg.InfiniteLine(angle=90, pen=pg.mkPen("#ffffff", width=1))
        lane.region = pg.LinearRegionItem(brush=(255, 255, 255, 30), movable=False)
        lane.region.hide()
        plot.addItem(lane.playhead, ignoreBounds=True)
        plot.addItem(lane.region, ignoreBounds=True)
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

    def _layout_lanes(self) -> None:
        visible = [n for n in self.lanes if self._lane_visible(n)]
        saved = self.settings.lane_heights()
        total = self.lane_split.height() if self.lane_split.height() > 200 else 800
        fixed = {n: saved.get(n, STATES_DEFAULT_HEIGHT if n == "states" else None) for n in visible}
        free = [n for n in visible if fixed[n] is None]
        share = max(100, (total - sum(h for h in fixed.values() if h is not None)) // max(1, len(free)))
        sizes = []
        for i, (n, lane) in enumerate(self.lanes.items()):
            self.lane_split.insertWidget(i, lane.widget)
            lane.widget.setVisible(n in visible)
            self.lane_split.setStretchFactor(i, 0 if n == "states" else 1)  # window resizes go to the plots
            if n in visible:
                lane.plot.getAxis("bottom").setStyle(showValues=(n == visible[-1]))
            sizes.append((fixed[n] if fixed[n] is not None else share) if n in visible else 0)
        self.lane_split.setSizes(sizes)

    def _save_lane_heights(self) -> None:
        sizes = dict(zip(self.lanes, self.lane_split.sizes()))
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
        sc("L", self.follow_btn.toggle)
        sc("G", lambda: self.set_lane_visible("spectrogram", not self._lane_visible("spectrogram")))
        sc("Ctrl+Z", self.undo)
        sc("Ctrl+Shift+Z", self.redo)
        sc("Ctrl+Y", self.redo)
        sc("Ctrl+S", self.save_annotation)
        sc("Ctrl+R", self.rerun)
        sc("Ctrl+Shift+C", self.copy_context)

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
                self._audio_error.emit(str(e))

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
        self._show_source("Listening to: original  (filtered loading…)")

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

    def _audio_failed(self, msg: str) -> None:
        self._show_source("Audio: unavailable", enabled=False)
        self.status.setText(f"Could not load audio: {msg}")

    # ─────────────────────────────────────────────────────────────────────
    # Plots
    # ─────────────────────────────────────────────────────────────────────
    def _clear_plots(self) -> None:
        for lane in self.lanes.values():
            for item in lane.items.values():
                lane.plot.removeItem(item)
            lane.items.clear()
        self.overview_curve.setData([], [])
        self.trace_tree.clear()

    def _trace_visible(self, name: str, default: bool) -> bool:
        return bool(self.trace_vis.get(name, default))

    def _rebuild_plots(self) -> None:
        self._clear_plots()
        a = self.analysis
        self._peak_kinds = a.peaks.kind()
        for t in a.traces:
            if t.lane not in self.lanes:
                self._extra_lane(t.lane)
        self._layout_lanes()
        for t in a.traces:
            self._set_trace_shown(t, self._trace_visible(t.name, t.visible))
        for name in (PEAKS_S1, PEAKS_S2, PEAKS_NOISE):
            self._set_peaks_shown(name, self._trace_visible(name, name != PEAKS_NOISE))
        self._set_ann_bpm_shown(self._trace_visible(ANN_BPM, True))
        if self._lane_visible("spectrogram"):
            self._draw_spectrogram()
        self._draw_states()
        self._draw_overview()
        self._fill_tree()
        self._restore_y()

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

    def _make_trace_item(self, t: Trace):
        color = t.color or "#cccccc"
        if t.uniform:
            return EnvelopeItem(t.t0, t.dt, t.y, color)
        if t.kind == KIND_LINE:
            item = pg.PlotDataItem(t.times(), t.y, pen=pg.mkPen(color, width=1), connect="finite")
            return item
        if t.kind == KIND_POINTS:
            item = pg.PlotDataItem(t.x, t.y, pen=None, symbol="o", symbolSize=5, symbolPen=None,
                                   symbolBrush=qcolor(color))
            return item
        if t.lane == "states":
            labels = list(t.text) if t.text else ["unknown"] * t.size
            return SpanRowsItem([palette_row(t.x, t.x_end, labels, *ROW_BEFORE)])
        return BandsItem(t.x, t.x_end, color, alpha=55)

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
            symbol = "x" if name == PEAKS_NOISE else "o"
            item = pg.PlotDataItem(p.time[m], p.amp[m], pen=None, symbol=symbol, symbolSize=7 if symbol == "o" else 6,
                                   symbolPen=None if symbol == "o" else pg.mkPen(PEAK_COLORS[name]),
                                   symbolBrush=qcolor(PEAK_COLORS[name]))
            item.setZValue(5)
            lane.items[name] = item
            self._add_item(lane, item)
        if item is not None:
            item.setVisible(shown)

    def _set_ann_bpm_shown(self, shown: bool) -> None:
        lane = self.lanes["bpm"]
        item = lane.items.get(ANN_BPM)
        if not self._lane_visible("bpm"):
            if item is not None:
                item.setVisible(shown)
            return
        if item is None:
            item = pg.PlotDataItem(pen=pg.mkPen("#7CFC9A", width=1), symbol="o", symbolSize=3,
                                   symbolBrush=qcolor("#7CFC9A"), symbolPen=None)
            item.setZValue(4)
            lane.items[ANN_BPM] = item
            self._add_item(lane, item)
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
        for mask, y0, y1, outline in ((full, ROW_ALG[0], ROW_ALG[1], None), (earlier, mid, ROW_ALG[1], DEFECT_COLOR),
                                      (later, ROW_ALG[0], mid, DEFECT_COLOR)):
            if mask.any():
                rows.append(palette_row(start[mask], end[mask], [names[i] for i in np.nonzero(mask)[0]], y0, y1,
                                        outline=outline))
        d = self.analysis.defects
        if d:
            rows.append(palette_row([x.start for x in d], [max(x.end, x.start + 0.02) for x in d],
                                    ["defect"] * len(d), *ROW_DEF, colors={"defect": DEFECT_COLOR}))
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
                                    [f"{s.kind}|{s.origin}" for s in spans], *ROW_ANN, colors=ANN_COLORS))
            clipped = [s for s in spans if s.clipped]
            if clipped:
                rows.append(palette_row([s.start for s in clipped], [s.end for s in clipped], ["c"] * len(clipped),
                                        ROW_ANN[1] - 0.12, ROW_ANN[1], colors={"c": "#ffffff"}))
        if self.disagreements:
            d = self.disagreements
            rows.append(palette_row([x.start for x in d], [x.end for x in d], [x.kind for x in d], *ROW_DIS,
                                    colors=DIS_COLORS))
        if self.selected is not None:
            s = self.selected
            rows.append(SpanRow(np.array([s.start]), np.array([s.end]), [qcolor("#ffffff", 0)], np.array([0]),
                                ROW_ANN[0] - 0.04, ROW_ANN[1] + 0.04, outline=qcolor("#ffffff")))
        self._set_internal(lane, "_ann", SpanRowsItem(rows))

    def _draw_overview(self) -> None:
        a = self.analysis
        t = a.trace("Algorithm envelope")
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

    def _fill_tree(self) -> None:
        self.trace_tree.blockSignals(True)
        self.trace_tree.clear()
        groups: Dict[str, List[Tuple[str, str, bool]]] = {n: [] for n in self.lanes}
        for t in self.analysis.traces:
            groups[t.lane].append((t.name, t.group, self._trace_visible(t.name, t.visible)))
        for name in (PEAKS_S1, PEAKS_S2, PEAKS_NOISE):
            groups["signal"].insert(0, (name, "Pass 2", self._trace_visible(name, name != PEAKS_NOISE)))
        groups["bpm"].insert(0, (ANN_BPM, "Annotation", self._trace_visible(ANN_BPM, True)))
        for lane_name, entries in groups.items():
            if not entries and lane_name not in ("states", "spectrogram"):
                continue
            top = QtWidgets.QTreeWidgetItem([LANE_TITLES.get(lane_name, lane_name)])
            top.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("lane", lane_name))
            top.setFlags(top.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            top.setCheckState(0, QtCore.Qt.CheckState.Checked if self._lane_visible(lane_name)
                              else QtCore.Qt.CheckState.Unchecked)
            self.trace_tree.addTopLevelItem(top)
            for name, group, vis in entries:
                child = QtWidgets.QTreeWidgetItem([f"{name}   ·  {group}"])
                child.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("trace", name))
                child.setFlags(child.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                child.setCheckState(0, QtCore.Qt.CheckState.Checked if vis else QtCore.Qt.CheckState.Unchecked)
                top.addChild(child)
            top.setExpanded(self._lane_visible(lane_name))
        self.trace_tree.blockSignals(False)

    def _on_tree_changed(self, item: QtWidgets.QTreeWidgetItem, _col: int) -> None:
        kind, name = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        on = item.checkState(0) == QtCore.Qt.CheckState.Checked
        if kind == "lane":
            self.set_lane_visible(name, on)
            return
        self.trace_vis[name] = on
        self.settings.set_trace_visibility(self.trace_vis)
        if name in (PEAKS_S1, PEAKS_S2, PEAKS_NOISE):
            self._set_peaks_shown(name, on)
        elif name == ANN_BPM:
            self._set_ann_bpm_shown(on)
        else:
            t = self.analysis.trace(name)
            if t is not None:
                self._set_trace_shown(t, on)

    def set_lane_visible(self, name: str, on: bool) -> None:
        """The one path for showing/hiding a lane (Traces tree and the G key)."""
        if self._lane_visible(name) == on:
            return
        self.lane_vis[name] = on
        self.settings.set_lane_visibility(self.lane_vis)
        self._layout_lanes()
        if name == "spectrogram" and not on:
            self._drop_spectrogram()
        if self.analysis is not None:
            self._sync_tree_lane(name, on)
            if on:
                self._realize_lane(name)

    def _sync_tree_lane(self, name: str, on: bool) -> None:
        state = QtCore.Qt.CheckState.Checked if on else QtCore.Qt.CheckState.Unchecked
        for i in range(self.trace_tree.topLevelItemCount()):
            top = self.trace_tree.topLevelItem(i)
            if top.data(0, QtCore.Qt.ItemDataRole.UserRole) == ("lane", name) and top.checkState(0) != state:
                self.trace_tree.blockSignals(True)
                top.setCheckState(0, state)
                self.trace_tree.blockSignals(False)

    def _realize_lane(self, lane_name: str) -> None:
        """Create the items of a lane that just became visible."""
        for t in self.analysis.traces:
            if t.lane == lane_name:
                self._set_trace_shown(t, self._trace_visible(t.name, t.visible))
        if lane_name == "signal":
            for name in (PEAKS_S1, PEAKS_S2, PEAKS_NOISE):
                self._set_peaks_shown(name, self._trace_visible(name, name != PEAKS_NOISE))
        if lane_name == "bpm":
            self._set_ann_bpm_shown(self._trace_visible(ANN_BPM, True))
        if lane_name == "states":
            self._draw_states()
        if lane_name in ("bpm", "states"):
            self._on_doc_changed()
        if lane_name == "spectrogram":
            if self._spec is None:
                self._compute_spectrogram()
            else:
                self._draw_spectrogram()

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
        if self.player.playing and self.follow_btn.isChecked():
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
        return f"Listening to: {self.player.source}  (T){loading}"

    def _show_source(self, text: str, enabled: bool = True) -> None:
        self.source_btn.setText(text)
        self.source_btn.setEnabled(enabled)

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

    def reset_y(self, lane_name: str) -> None:
        self._y_manual.pop(lane_name, None)
        self.settings.set_y_ranges(self.fingerprint, self._y_manual)
        self._auto_y(lane_name)

    def _auto_y(self, lane_name: str) -> None:
        plot = self.lanes[lane_name].plot
        if lane_name == "spectrogram":
            plot.setYRange(0, 1000, padding=0)
        else:
            plot.enableAutoRange(axis="y", enable=True)

    def _restore_y(self) -> None:
        """Re-apply the y-ranges saved for this Recording (lanes the previous one had scaled go back to auto)."""
        saved = self.settings.y_ranges(self.fingerprint)
        for name, lane in self.lanes.items():
            if name in saved:
                lane.plot.setYRange(*saved[name], padding=0)
            elif name in self._y_manual and name != "states":
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
                SpanRow(np.array([a]), np.array([b]), [qcolor("#bbbbbb", 140)], np.array([0]), *ROW_ANN)]))
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
        length = float(self.analysis.params.get("s1_nominal_sec" if kind == an.S1 else "s2_nominal_sec", 0.08))
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
        t, b = an.bpm_series(self.doc.spans)
        item = self.lanes["bpm"].items.get(ANN_BPM)
        if item is not None:
            item.setData(t, b)
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
        self.issue_list.addItems([f"{s:8.2f}s  {kind}: {msg}" for _src, kind, s, _e, msg in shown])
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
        txt = f"{a.algorithm_used}"
        # The filename often carries the same BPM tag (written by "rename"); show the BPM here only when
        # it adds something: the name has no tag, or an older run's tag that no longer matches.
        if bpm and batch.format_bpm_tag(bpm["start_bpm"], bpm["min_bpm"], bpm["max_bpm"]) not in Path(self.path).stem:
            txt += f"  ·  {bpm['start_bpm']:.0f}, {bpm['min_bpm']:.0f}–{bpm['max_bpm']:.0f} BPM"
        if gate.get("failed"):
            txt += "  ·  ⚠ gate failed"
        if a.channel != recording.CHANNEL_MIXED:
            txt += f"  ·  {a.channel} channel"
        self.info_label.setText(txt)
        if a.stale_reasons:
            self.stale_badge.setText("STALE")
            self.stale_badge.setToolTip("\n".join(a.stale_reasons) + "\nCtrl+R to re-run")
            self.stale_badge.show()
        else:
            self.stale_badge.hide()
        hrv = "".join(f"<tr><td>{k}</td><td>{v:.4g}</td></tr>" for k, v in (a.summary.get("hrv") or {}).items())
        rows = [
            ("Recording", Path(self.path).name), ("Fingerprint", a.fingerprint), ("Channel", a.channel),
            ("Algorithm", a.algorithm_used), ("Auto-switch", a.algorithm_switch_reason or "—"),
            ("BPM", f"start {bpm.get('start_bpm', float('nan')):.1f}, {bpm.get('min_bpm', float('nan')):.1f}–"
                    f"{bpm.get('max_bpm', float('nan')):.1f}" if bpm else "—"),
            ("Plausibility gate", "FAILED: " + "; ".join(gate.get("reasons") or []) if gate.get("failed") else "passed"),
            ("Defects", str(len(a.defects))), ("Run settings", str(a.run_settings)),
            ("Start BPM hint", str(a.start_bpm_hint)), ("Created", a.created),
            ("Status", "stale — " + "; ".join(a.stale_reasons) if a.stale_reasons else "current"),
        ]
        html = "<table>" + "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in rows) + "</table>"
        self.summary_text.setHtml(html + "<h4>HRV</h4><table>" + hrv + "</table>")

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

    def export_bpm_csv(self, source: str) -> None:
        from pcg.cli import write_bpm_csv

        if self.analysis is None:
            return
        stem = Path(self.path).stem
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export BPM CSV",
                                                        str(downloads_dir() / f"{stem}_{source}_bpm.csv"), "CSV (*.csv)")
        if not path:
            return
        if source == "annotation":
            t, b = an.bpm_series(self.doc.spans)
            write_bpm_csv(Path(path), t, b, "bpm_annotation")
        else:
            tr = self.analysis.trace("BPM")
            if tr is None:
                return
            write_bpm_csv(Path(path), tr.times(), tr.y, "bpm")
        self.status.setText(f"Exported {path}")

    def export_summary(self) -> None:
        from pcg.cli import summary_text

        if self.analysis is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export summary", str(downloads_dir() / f"{Path(self.path).stem}_summary.txt"), "Text (*.txt)")
        if path:
            Path(path).write_text(summary_text(self.analysis), encoding="utf-8")

    def export_image(self) -> None:
        if self.analysis is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export view as image", str(downloads_dir() / f"{Path(self.path).stem}_view.png"), "PNG (*.png)")
        if path:
            self.lane_split.grab().save(path)

    def shutdown(self) -> None:
        self.player.stop()
        self.worker.cancel()
