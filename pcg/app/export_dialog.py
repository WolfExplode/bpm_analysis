"""The export dialog (Ctrl+Shift+E): a file browser on the left, the chosen format's options on the right."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from . import theme
from .state import Settings, downloads_dir

# format key -> (label, extension, name filter)
FORMATS: Dict[str, Tuple[str, str, str]] = {
    "csv": ("BPM CSV", ".csv", "CSV (*.csv)"),
    "summary": ("Summary", ".txt", "Text (*.txt)"),
    "context": ("Window text for an LLM", ".txt", "Text (*.txt)"),
    "image": ("Workspace image", ".png", "PNG (*.png)"),
}
NAME_SUFFIX = {"csv": "_bpm", "summary": "_summary", "context": "_window", "image": "_view"}
RANGES = (("whole", "Whole recording"), ("visible", "Visible window"), ("region", "Loop region"))
TIME_FORMATS = (("seconds", "Seconds"), ("clock", "hh:mm:ss.xxx"), ("both", "Both"))
SOURCES = (("analysis", "Analysis"), ("annotation", "Annotation"))


class ExportDialog(QtWidgets.QDialog):
    """After exec() == Accepted: .fmt, .path and .opts describe the export to run."""

    def __init__(self, parent, settings: Settings, stem: str, *, has_annotation: bool, has_region: bool):
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.settings, self.stem = settings, stem
        self.has_annotation, self.has_region = has_annotation, has_region
        self.fmt, self.path, self.opts = "csv", "", {}
        saved = settings.get_json("export", {})

        self.files = QtWidgets.QFileDialog(self)
        self.files.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)
        self.files.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self.files.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        self.files.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
        self.files.setLabelText(QtWidgets.QFileDialog.DialogLabel.Accept, "Export")
        folder = saved.get("folder")
        self.files.setDirectory(folder if folder and Path(folder).is_dir() else str(downloads_dir()))
        self._use_recycle_bin()
        self.files.accepted.connect(self._accepted)
        self.files.rejected.connect(self.reject)

        # A pasteable folder field, like Explorer's address bar (the browser's own is read-only).
        self.folder_edit = QtWidgets.QLineEdit(self.files.directory().absolutePath())
        self.folder_edit.setPlaceholderText("Paste a folder path and press Enter")
        self.folder_edit.setClearButtonEnabled(True)
        completer = QtWidgets.QCompleter(self)
        completer.setModel(QtWidgets.QFileSystemModel(completer))
        completer.model().setRootPath("")
        completer.model().setFilter(QtCore.QDir.Filter.Dirs | QtCore.QDir.Filter.Drives | QtCore.QDir.Filter.NoDotAndDotDot)
        completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        self.folder_edit.setCompleter(completer)
        self.folder_edit.installEventFilter(self)
        self.files.directoryEntered.connect(self.folder_edit.setText)

        self.format_box = self._combo({k: v[0] for k, v in FORMATS.items()}.items())
        self.range_box = self._combo(RANGES)
        self.time_box = self._combo(TIME_FORMATS)
        self.source_box = self._combo(SOURCES)
        self.overview_box = QtWidgets.QCheckBox("Include the overview strip")
        self.empty_note = QtWidgets.QLabel("No options for this format.")
        self.empty_note.setProperty("tone", "muted")

        form = QtWidgets.QFormLayout()
        form.addRow("Format", self.format_box)
        self.rows = {
            "source": self._row(form, "Source", self.source_box),
            "range": self._row(form, "Range", self.range_box),
            "time": self._row(form, "Time column", self.time_box),
            "overview": self._row(form, "", self.overview_box),
            "empty": self._row(form, "", self.empty_note),
        }
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(300)
        pl = QtWidgets.QVBoxLayout(panel)
        pl.addLayout(form)
        pl.addStretch(1)

        browser = QtWidgets.QVBoxLayout()
        browser.addWidget(self.folder_edit)
        browser.addWidget(self.files, 1)
        lay = QtWidgets.QHBoxLayout(self)
        lay.addLayout(browser, 1)
        lay.addWidget(panel)
        self.resize(1150, 640)

        self._restore(saved)
        self.format_box.currentIndexChanged.connect(self._format_changed)
        self._format_changed()

    # ── layout helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _combo(items) -> QtWidgets.QComboBox:
        box = QtWidgets.QComboBox()
        for key, label in items:
            box.addItem(label, key)
        return box

    @staticmethod
    def _row(form: QtWidgets.QFormLayout, label: str, widget: QtWidgets.QWidget):
        form.addRow(label, widget)
        return form.labelForField(widget), widget

    def _set_row(self, name: str, shown: bool) -> None:
        for w in self.rows[name]:
            if w is not None:
                w.setVisible(shown)

    @staticmethod
    def _pick(box: QtWidgets.QComboBox, key) -> None:
        i = box.findData(key)
        if i >= 0:
            box.setCurrentIndex(i)

    @staticmethod
    def _enable(box: QtWidgets.QComboBox, key: str, enabled: bool) -> None:
        i = box.findData(key)
        if i >= 0:
            box.model().item(i).setEnabled(enabled)

    # ── state ────────────────────────────────────────────────────────────────
    def _restore(self, saved: dict) -> None:
        self._pick(self.format_box, saved.get("format", "csv"))
        o = saved.get("opts", {})
        self._pick(self.range_box, o.get("range", "whole"))
        self._pick(self.time_box, o.get("time_format", "seconds"))
        self._pick(self.source_box, o.get("source", "annotation" if self.has_annotation else "analysis"))
        self.overview_box.setChecked(bool(o.get("overview", True)))

    def _format_changed(self) -> None:
        fmt = self.format_box.currentData()
        _label, ext, flt = FORMATS[fmt]
        self._set_row("source", fmt == "csv")
        self._set_row("range", fmt in ("csv", "context"))
        self._set_row("time", fmt == "csv")
        self._set_row("overview", fmt == "image")
        self._set_row("empty", fmt == "summary")
        # Offer only what exists: a loop region, an open Annotation; "whole" is for the CSV only.
        self._enable(self.range_box, "region", self.has_region)
        self._enable(self.range_box, "whole", fmt == "csv")
        if not self.range_box.model().item(self.range_box.currentIndex()).isEnabled():
            self._pick(self.range_box, "visible")
        self._enable(self.source_box, "annotation", self.has_annotation)
        if not self.has_annotation:
            self._pick(self.source_box, "analysis")
        self.files.setNameFilter(flt)
        self.files.setDefaultSuffix(ext.lstrip("."))
        self.files.selectFile(f"{self.stem}{NAME_SUFFIX[fmt]}{ext}")

    def _use_recycle_bin(self) -> None:
        """Qt's browser deletes permanently; route its Delete menu item to the Recycle Bin instead."""
        for action in self.files.findChildren(QtGui.QAction):
            if action.text().replace("&", "") == "Delete":
                action.triggered.disconnect()
                action.triggered.connect(self._trash_selected)

    def _trash_selected(self) -> None:
        for view in self.files.findChildren(QtWidgets.QAbstractItemView):
            if view.objectName() in ("listView", "treeView") and view.isVisible():
                break
        else:
            return
        paths = {i.data(QtWidgets.QFileSystemModel.Roles.FilePathRole) for i in view.selectionModel().selectedIndexes()}
        for path in sorted(p for p in paths if p):
            name = Path(path).name
            ask = QtWidgets.QMessageBox.question(self, "Move to Recycle Bin", f"Move '{name}' to the Recycle Bin?")
            if ask != QtWidgets.QMessageBox.StandardButton.Yes:
                continue
            if not QtCore.QFile.moveToTrash(path):
                QtWidgets.QMessageBox.warning(self, "Recycle Bin", f"Could not move '{name}' to the Recycle Bin.")

    def eventFilter(self, obj, ev):
        # Enter in the folder field navigates; left alone it would reach the dialog's default Export button.
        if obj is self.folder_edit and ev.type() == QtCore.QEvent.Type.KeyPress and ev.key() in (
                QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            if not self.folder_edit.completer().popup().isVisible():
                self._folder_typed()
                return True
        return super().eventFilter(obj, ev)

    def _folder_typed(self) -> None:
        """Enter in the folder field: go there. A pasted file path goes to its folder and fills the name."""
        text = self.folder_edit.text().strip().strip('"')
        path = Path(text).expanduser()
        if path.is_dir():
            self.files.setDirectory(str(path))
        elif path.parent.is_dir() and path.suffix:
            self.files.setDirectory(str(path.parent))
            self.files.selectFile(path.name)
        else:
            theme.set_prop(self.folder_edit, "invalid", True)
            return
        theme.set_prop(self.folder_edit, "invalid", False)
        self.folder_edit.setText(self.files.directory().absolutePath())

    def _accepted(self) -> None:
        picked = self.files.selectedFiles()
        if not picked:
            return
        self.fmt = self.format_box.currentData()
        self.path = picked[0]
        self.opts = {
            "range": self.range_box.currentData(),
            "time_format": self.time_box.currentData(),
            "source": self.source_box.currentData(),
            "overview": self.overview_box.isChecked(),
        }
        self.settings.set_json("export", {"format": self.fmt, "folder": str(Path(self.path).parent), "opts": self.opts})
        self.accept()
