"""Run screen: a queue of Recordings -> Analyses."""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from pcg import analysis as A
from pcg import batch, recording

from . import theme
from .state import AnalyzeWorker, Settings
from .widgets import PillDelegate

COLS = ["Recording", "Start BPM", "Skip (s)", "Status", "Result", ""]
C_NAME, C_BPM, C_START, C_STATUS, C_RESULT, C_STALE = range(len(COLS))

# Engine stderr: log records are "LEVEL logger: message" (pcg.cli's logging format). The engine
# logs through the root logger; Springer through its own module loggers.
_LOG_RECORD = re.compile(r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL) (\S+): ")
_ALGORITHM_LOGGERS = ("pcg.engine", "springer")
_PY_WARNING = re.compile(r"^\S.*:\d+: \w*Warning: ")


@dataclass
class Row:
    path: str
    bpm_override: Optional[float] = None
    start_sec: float = 0.0
    status: str = "queued"
    result: str = ""
    stale: str = ""
    analyses: List[str] = field(default_factory=list)
    bpm: Optional[dict] = None
    log: List[Tuple[str, bool, bool]] = field(default_factory=list)  # (line, in Basic, in Verbose)
    in_py_warning: bool = False  # inside a Python warning's multi-line stderr output


class RunScreen(QtWidgets.QWidget):
    open_requested = QtCore.Signal(str, str)  # recording path, analysis path ("" = latest)
    _status_ready = QtCore.Signal(str, str, str, object)  # path, result, stale, bpm

    def __init__(self, settings: Settings, library: A.Library, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.library = library
        self.rows: List[Row] = []
        self.workers: Dict[str, AnalyzeWorker] = {}
        self.pending: List[Row] = []
        self._log_row: Optional[Row] = None  # whose console log the panel shows
        self.setAcceptDrops(True)
        self._status_ready.connect(self._apply_status)
        self._build_ui()

    def _build_ui(self) -> None:
        header = QtWidgets.QFrame()
        header.setObjectName("header")
        bar = QtWidgets.QHBoxLayout(header)
        bar.setContentsMargins(12, 6, 12, 6)
        bar.setSpacing(6)
        for text, fn in (("Add files…", self.add_files_dialog), ("Add folder…", self.add_folder_dialog),
                         ("Remove", self.remove_selected), ("Clear", self.clear)):
            bar.addWidget(QtWidgets.QPushButton(text, clicked=fn))
        bar.addWidget(_separator())
        self.algo = QtWidgets.QComboBox()
        self.algo.addItems(["Native", "Springer 2015"])
        self.auto_switch = QtWidgets.QCheckBox("Auto-switch on gate failure")
        self.jobs = QtWidgets.QSpinBox(minimum=1, maximum=32, value=self.settings.get_json("run_jobs", 4))
        self.channel = QtWidgets.QComboBox()
        self.channel.addItems(list(recording.CHANNEL_MODES))
        saved = self.settings.get_json("run_settings", {})
        self.algo.setCurrentIndex(1 if saved.get("springer") else 0)
        self.auto_switch.setChecked(bool(saved.get("auto_switch")))
        self.channel.setCurrentText(saved.get("channel", recording.CHANNEL_MIXED))
        for label, w in (("Algorithm", self.algo), (None, self.auto_switch), ("Channel", self.channel),
                         ("Jobs", self.jobs)):
            if label:
                lab = QtWidgets.QLabel(label)
                lab.setProperty("tone", "muted")
                bar.addWidget(lab)
            bar.addWidget(w)
            bar.addSpacing(6)
        bar.addStretch(1)
        self.rename_btn = QtWidgets.QPushButton("Write BPM into filenames", clicked=self.rename_selected)
        self.stop_btn = QtWidgets.QPushButton("Stop", clicked=self.stop)
        self.run_btn = QtWidgets.QPushButton("Run", clicked=self.run)
        self.run_btn.setProperty("primary", True)
        bar.addWidget(self.rename_btn)
        bar.addWidget(_separator())
        for w in (self.stop_btn, self.run_btn):
            bar.addWidget(w)

        self.table = QtWidgets.QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.horizontalHeader().setSectionResizeMode(C_NAME, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(C_RESULT, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft
                                                          | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.table.setItemDelegateForColumn(C_STATUS, PillDelegate(_status_tone, self.table))
        self.table.setItemDelegateForColumn(C_STALE, PillDelegate(lambda _t: "warning", self.table))
        self.table.cellDoubleClicked.connect(self._double_clicked)
        self.table.viewport().installEventFilter(self)  # to tell which button double-clicked
        self._dbl_button = QtCore.Qt.MouseButton.LeftButton
        self.table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        self.table.itemChanged.connect(self._item_changed)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(header)
        lay.addWidget(self._build_console_panel(), 1)

    def _build_console_panel(self) -> QtWidgets.QSplitter:
        self.console_title = QtWidgets.QLabel("Console")
        self.console_title.setProperty("tone", "muted")
        self.console_title.setContentsMargins(12, 6, 12, 4)
        consoles = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        consoles.setChildrenCollapsible(False)
        self.console, self.verbose_console = (self._console_pane(consoles, title)
                                              for title in ("Basic", "Verbose — algorithm logging"))
        panel = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(0)
        pl.addWidget(self.console_title)
        pl.addWidget(consoles, 1)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        split.addWidget(self.table)
        split.addWidget(panel)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setChildrenCollapsible(False)
        return split

    @staticmethod
    def _console_pane(parent: QtWidgets.QSplitter, title: str) -> QtWidgets.QPlainTextEdit:
        label = QtWidgets.QLabel(title)
        label.setProperty("tone", "muted")
        label.setContentsMargins(12, 2, 12, 2)
        text = QtWidgets.QPlainTextEdit(readOnly=True)
        text.setFont(theme.mono_font())
        text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        text.setMaximumBlockCount(20000)
        text.setPlaceholderText("Double-click a Status to show that run's console log.")
        pane = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(pane)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(label)
        lay.addWidget(text, 1)
        parent.addWidget(pane)
        return text

    # ── queue ────────────────────────────────────────────────────────────
    def dragEnterEvent(self, ev: QtGui.QDragEnterEvent) -> None:
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev: QtGui.QDropEvent) -> None:
        self.add_paths([u.toLocalFile() for u in ev.mimeData().urls()])

    def add_files_dialog(self) -> None:
        exts = " ".join(f"*{e}" for e in recording.AUDIO_EXTENSIONS)
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add recordings", "", f"Audio ({exts})")
        self.add_paths(paths)

    def add_folder_dialog(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Add folder")
        if d:
            self.add_paths([d])

    def add_paths(self, paths: List[str]) -> None:
        known = {r.path for r in self.rows}
        new = [Row(str(p)) for p in batch.collect(paths) if str(p) not in known]
        self.rows.extend(new)
        self._refresh_table()
        self._check_status([r.path for r in new])

    def remove_selected(self) -> None:
        sel = {i.row() for i in self.table.selectionModel().selectedRows()}
        self.rows = [r for i, r in enumerate(self.rows) if i not in sel or r.path in self.workers]
        self._refresh_table()

    def clear(self) -> None:
        self.rows = [r for r in self.rows if r.path in self.workers]
        self._refresh_table()

    def _refresh_table(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.rows))
        for i, r in enumerate(self.rows):
            self._fill_row(i, r)
        self.table.blockSignals(False)
        if self._log_row not in self.rows:
            self._show_log(self.rows[0] if self.rows else None)

    def _fill_row(self, i: int, r: Row) -> None:
        def cell(col, text, editable=False, tip=""):
            it = QtWidgets.QTableWidgetItem(text)
            flags = QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled
            if editable:
                flags |= QtCore.Qt.ItemFlag.ItemIsEditable
            it.setFlags(flags)
            if tip:
                it.setToolTip(tip)
            self.table.setItem(i, col, it)
            return it

        cell(C_NAME, Path(r.path).name, tip=r.path)
        hint = batch.start_bpm_from_filename(r.path)
        bpm_item = cell(C_BPM, "" if r.bpm_override is None else f"{r.bpm_override:g}", editable=True,
                        tip=f"from filename: {hint:g}" if hint else "no filename tag")
        if r.bpm_override is None and hint:
            bpm_item.setText(f"{hint:g}")
            bpm_item.setForeground(theme.qcolor(theme.TEXT_3))
        start = cell(C_START, f"{r.start_sec:g}", editable=True)
        for it in (bpm_item, start):
            it.setFont(theme.mono_font())
            it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        cell(C_STATUS, r.status)
        res = cell(C_RESULT, r.result)
        if "⚠" in r.result or r.status == "failed":
            res.setForeground(theme.qcolor(theme.TONES["warning" if "⚠" in r.result else "danger"][0]))
        cell(C_STALE, r.stale)

    def _row_index(self, path: str) -> int:
        return next((i for i, r in enumerate(self.rows) if r.path == path), -1)

    def _item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        r = self.rows[item.row()]
        text = item.text().strip()
        try:
            if item.column() == C_BPM:
                hint = batch.start_bpm_from_filename(r.path)
                r.bpm_override = None if not text or (hint and float(text) == hint) else float(text)
            elif item.column() == C_START:
                r.start_sec = max(0.0, float(text or 0))
        except ValueError:
            pass
        self.table.blockSignals(True)
        self._fill_row(item.row(), r)
        self.table.blockSignals(False)

    # ── status (fingerprint, existing analysis, staleness) in the background ─
    def _check_status(self, paths: List[str]) -> None:
        if not paths:
            return

        def work():
            config, code = A.current_config_params(), A.code_fingerprint()
            for p in paths:
                try:
                    fp = self.library.fingerprints.get(p)
                    found = self.library.latest(fp)
                    if not found:
                        self._status_ready.emit(p, "", "", None)
                        continue
                    a = A.load(found)
                    reasons = A.stale_reasons(a, config, code)
                    self._status_ready.emit(p, _result_text(a.summary), "stale" if reasons else "",
                                            a.summary.get("bpm"))
                except Exception as e:  # noqa: BLE001
                    self._status_ready.emit(p, f"error: {e}", "", None)

        threading.Thread(target=work, daemon=True).start()

    def _apply_status(self, path: str, result: str, stale: str, bpm) -> None:
        i = self._row_index(path)
        if i < 0 or path in self.workers:
            return
        r = self.rows[i]
        r.result, r.stale, r.bpm = result, stale, bpm
        if result and r.status == "queued" and r not in self.pending:
            r.status = "analysed"
        self.table.blockSignals(True)
        self._fill_row(i, r)
        self.table.blockSignals(False)

    # ── running ──────────────────────────────────────────────────────────
    def run_settings(self) -> batch.RunSettings:
        s = batch.RunSettings(springer=self.algo.currentIndex() == 1, auto_switch=self.auto_switch.isChecked(),
                              channel=self.channel.currentText(), jobs=self.jobs.value())
        self.settings.set_json("run_settings", {"springer": s.springer, "auto_switch": s.auto_switch,
                                                "channel": s.channel})
        self.settings.set_json("run_jobs", s.jobs)
        return s

    def run(self) -> None:
        sel = self._selected_rows()
        self._run_rows(sel if len(sel) > 1 else list(self.rows))

    def _selected_rows(self) -> List[Row]:
        return [self.rows[i] for i in sorted({i.row() for i in self.table.selectionModel().selectedRows()})]

    def _run_rows(self, targets: List[Row]) -> None:
        self.pending = [r for r in targets if r.path not in self.workers]
        for r in self.pending:
            r.status = "queued"
            r.log.clear()
        self._refresh_table()
        if self._log_row in self.pending:
            self._show_log(self._log_row)  # the stored log was cleared; clear the panel too
        self._settings = self.run_settings()
        self._pump()

    def stop(self) -> None:
        self.pending.clear()
        for w in list(self.workers.values()):
            w.cancel()

    def _pump(self) -> None:
        limit = self._settings.effective_jobs
        while self.pending and len(self.workers) < limit:
            r = self.pending.pop(0)
            w = AnalyzeWorker(self)
            w.progress.connect(lambda msg, p=r: self._log_line(p, msg))
            w.progress.connect(lambda msg, p=r.path: self._set_status(p, msg))
            w.log.connect(lambda line, p=r: self._log_stderr(p, line))
            w.finished.connect(lambda paths, err, p=r.path: self._finished(p, paths, err))
            self.workers[r.path] = w
            hint = r.bpm_override if r.bpm_override is not None else batch.start_bpm_from_filename(r.path)
            w.start(r.path, str(self.library.root), springer=self._settings.springer,
                    auto_switch=self._settings.auto_switch, channel=self._settings.channel,
                    start_sec=r.start_sec, bpm_hint=hint)
            self._set_status(r.path, "starting…")

    # ── console log ──────────────────────────────────────────────────────
    def _show_log(self, row: Optional[Row]) -> None:
        self._log_row = row
        self.console_title.setText(f"Console — {Path(row.path).name}" if row else "Console")
        entries = row.log if row else []
        for view, verbose in ((self.console, False), (self.verbose_console, True)):
            view.setPlainText("\n".join(line for line, basic, verb in entries if (verb if verbose else basic)))
            view.verticalScrollBar().setValue(view.verticalScrollBar().maximum())

    def _log_line(self, row: Row, text: str, basic: bool = True, verbose: bool = True) -> None:
        line = f"{time.strftime('%H:%M:%S')}  {text}"
        row.log.append((line, basic, verbose))
        if row is self._log_row:
            if basic:
                self.console.appendPlainText(line)
            if verbose:
                self.verbose_console.appendPlainText(line)

    def _log_stderr(self, row: Row, line: str) -> None:
        """Route an engine stderr line: Basic gets warnings and anything that isn't a log record
        (tracebacks); Verbose gets every algorithm log record plus tracebacks, but not
        other libraries' logging or Python warnings."""
        m = _LOG_RECORD.match(line)
        if m:
            row.in_py_warning = False
            level, name = m.group(1), m.group(2)
            self._log_line(row, line, basic=level not in ("DEBUG", "INFO"),
                           verbose=name == "root" or name.startswith(_ALGORITHM_LOGGERS))
            return
        if _PY_WARNING.match(line):
            row.in_py_warning = True  # its source line follows, indented
        elif not line[:1].isspace():
            row.in_py_warning = False
        self._log_line(row, line, verbose=not row.in_py_warning)

    def _set_status(self, path: str, msg: str) -> None:
        i = self._row_index(path)
        if i >= 0:
            self.rows[i].status = msg
            self.table.blockSignals(True)
            self._fill_row(i, self.rows[i])
            self.table.blockSignals(False)

    def _finished(self, path: str, paths: list, err: str) -> None:
        w = self.workers.pop(path, None)
        if w is not None:
            w.deleteLater()
        i = self._row_index(path)
        if i >= 0:
            r = self.rows[i]
            if err or not paths:
                r.status, r.result = ("cancelled" if err == "cancelled" else "failed"), err
                if err:
                    self._log_line(r, f"{r.status}: {err}")
            else:
                r.status, r.analyses, r.stale = "done", list(paths), ""
                summaries = [A.load(p).summary for p in paths]
                r.result = " | ".join(_result_text(s) for s in summaries)
                r.bpm = summaries[0].get("bpm") if len(summaries) == 1 else None
            self.table.blockSignals(True)
            self._fill_row(i, r)
            self.table.blockSignals(False)
        self._pump()

    def rename_selected(self) -> None:
        sel = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        rows = [self.rows[i] for i in sel] or list(self.rows)
        todo = [r for r in rows if r.bpm and r.path not in self.workers]
        if not todo:
            QtWidgets.QMessageBox.information(self, "Rename", "No analysed single-channel recordings selected.")
            return
        skipped = []
        for r in todo:
            new, why = batch.rename_with_bpm(r.path, r.bpm)
            if new:
                r.path = str(new)
            else:
                skipped.append(f"{Path(r.path).name}: {why}")
        self._refresh_table()
        if skipped:
            QtWidgets.QMessageBox.information(self, "Some files were not renamed", "\n".join(skipped[:30]))

    def eventFilter(self, obj, ev) -> bool:
        if obj is self.table.viewport() and ev.type() == QtCore.QEvent.Type.MouseButtonDblClick:
            self._dbl_button = ev.button()
        return super().eventFilter(obj, ev)

    def _context_menu(self, pos: QtCore.QPoint) -> None:
        i = self.table.rowAt(pos.y())
        if i < 0:
            return
        if not self.table.selectionModel().isRowSelected(i):
            self.table.selectRow(i)
        menu = QtWidgets.QMenu(self)
        menu.addAction("Run", lambda: self._run_and_show(self.rows[i]))
        menu.addAction("Remove", self.remove_selected)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _run_and_show(self, clicked: Row) -> None:
        """Run the selected rows and put the right-clicked one in the consoles."""
        self._run_rows(self._selected_rows())
        self._show_log(clicked)

    def _double_clicked(self, row: int, col: int) -> None:
        if self._dbl_button != QtCore.Qt.MouseButton.LeftButton:
            return
        r = self.rows[row]
        if col == C_STATUS:
            self._show_log(r)
            return
        if col != C_NAME:  # the other cells are editable
            return
        self.open_requested.emit(r.path, r.analyses[0] if r.analyses else "")

    def busy(self) -> bool:
        return bool(self.workers)


def _separator() -> QtWidgets.QFrame:
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.Shape.VLine)
    line.setFixedSize(9, 18)
    line.setStyleSheet(f"color: {theme.BORDER};")
    return line


def _status_tone(status: str) -> str:
    if status in ("done", "analysed"):
        return "success"
    if status == "failed":
        return "danger"
    if status in ("queued", "cancelled"):
        return "muted"
    return "info"  # starting / a progress message


def _result_text(summary: dict) -> str:
    bpm = summary.get("bpm")
    gate = summary.get("gate") or {}
    txt = f"{bpm['start_bpm']:.0f}, {bpm['min_bpm']:.0f}–{bpm['max_bpm']:.0f} BPM" if bpm else "no BPM"
    if gate.get("failed"):
        txt += "  ⚠ " + "; ".join(gate.get("reasons") or [])
    return txt
