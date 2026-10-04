"""The export dialog (Ctrl+Shift+E): pick a format and its options, then where the file goes.

Folders are chosen by typing or pasting a path (recent ones are in the drop-down) or with
"Browse…", which opens the system's own folder picker.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

from PySide6 import QtCore, QtWidgets

from . import theme
from .state import Settings, downloads_dir
from .widgets import FormatCard, Segmented

# format key -> (label, description, extension, file-name suffix)
FORMATS: Dict[str, Tuple[str, str, str, str]] = {
    "csv": ("BPM CSV", "Time and BPM rows", ".csv", "_bpm"),
    "chart": ("BPM/time graph", "BPM over the waveform, 0–230", ".png", "_chart"),
    "context": ("Window text", "The visible window, for an LLM", ".txt", "_window"),
}
RANGES = (("whole", "Whole recording"), ("visible", "Visible window"), ("region", "Loop region"))
TIME_FORMATS = (("seconds", "Seconds"), ("clock", "hh:mm:ss.xxx"), ("both", "Both"))
SOURCES = (("analysis", "Analysis"), ("annotation", "Annotation"))
RECENT_FOLDERS = 8


class ExportDialog(QtWidgets.QDialog):
    """After exec() == Accepted: .fmt, .path, .opts and .open_after describe the export to run."""

    def __init__(self, parent, settings: Settings, stem: str, *, has_annotation: bool, has_region: bool):
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.settings, self.stem = settings, stem
        self.has_annotation, self.has_region = has_annotation, has_region
        self.fmt, self.path, self.opts, self.open_after = "csv", "", {}, False
        self._saved = settings.get_json("export", {})

        self.cards: Dict[str, FormatCard] = {}
        cards = QtWidgets.QHBoxLayout()
        cards.setSpacing(8)
        for key, (label, desc, _ext, _suffix) in FORMATS.items():
            card = FormatCard(label, desc)
            card.toggled.connect(lambda on, k=key: on and self._format_changed(k))
            self.cards[key] = card
            cards.addWidget(card)

        self.source = Segmented(SOURCES)
        self.range = Segmented(RANGES)
        self.time = Segmented(TIME_FORMATS)
        self.options = QtWidgets.QFormLayout()
        self.options.setHorizontalSpacing(14)
        self.options.setVerticalSpacing(8)
        self.options.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.rows = {
            "source": self._row("Source", self.source),
            "range": self._row("Range", self.range),
            "time": self._row("Time column", self.time),
        }

        self.folder = QtWidgets.QComboBox(editable=True)
        self.folder.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.folder.lineEdit().setPlaceholderText("Paste a folder path")
        self.folder.lineEdit().setFont(theme.mono_font(8.5))
        self.folder.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.folder.setMinimumContentsLength(30)
        completer = QtWidgets.QCompleter(self)
        model = QtWidgets.QFileSystemModel(completer)
        model.setRootPath("")
        model.setFilter(QtCore.QDir.Filter.Dirs | QtCore.QDir.Filter.Drives | QtCore.QDir.Filter.NoDotAndDotDot)
        completer.setModel(model)
        completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        self.folder.setCompleter(completer)
        self.folder.lineEdit().textEdited.connect(lambda _t: theme.set_prop(self.folder.lineEdit(), "invalid", False))
        browse = QtWidgets.QPushButton("Browse…", clicked=self._browse)
        folder_row = QtWidgets.QHBoxLayout()
        folder_row.setSpacing(6)
        folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(browse)

        self.name = QtWidgets.QLineEdit()
        self.name.setFont(theme.mono_font(8.5))
        self.ext = QtWidgets.QLabel()
        self.ext.setFont(theme.mono_font(8.5))
        self.ext.setProperty("tone", "muted")
        name_row = QtWidgets.QHBoxLayout()
        name_row.setSpacing(6)
        name_row.addWidget(self.name, 1)
        name_row.addWidget(self.ext)

        dest = QtWidgets.QFormLayout()
        dest.setHorizontalSpacing(14)
        dest.setVerticalSpacing(8)
        dest.addRow(self._label("Folder"), folder_row)
        dest.addRow(self._label("File name"), name_row)

        line = QtWidgets.QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet(f"background: {theme.LINE};")
        body = QtWidgets.QVBoxLayout()
        body.setContentsMargins(16, 16, 16, 14)
        body.setSpacing(14)
        body.addLayout(cards)
        body.addLayout(self.options)
        body.addWidget(line)
        body.addLayout(dest)

        footer = QtWidgets.QFrame()
        footer.setObjectName("footer")
        fl = QtWidgets.QHBoxLayout(footer)
        fl.setContentsMargins(16, 10, 16, 10)
        self.open_after_box = QtWidgets.QCheckBox("Show in folder afterwards")
        cancel = QtWidgets.QPushButton("Cancel", clicked=self.reject)
        export = QtWidgets.QPushButton("Export", clicked=self._export)
        export.setProperty("primary", True)
        export.setDefault(True)
        for b in (cancel, browse):
            b.setAutoDefault(False)
        fl.addWidget(self.open_after_box)
        fl.addStretch(1)
        fl.addWidget(cancel)
        fl.addWidget(export)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addLayout(body, 1)
        lay.addWidget(footer)
        self.setMinimumWidth(640)

        self._restore()

    # ── layout helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _label(text: str) -> QtWidgets.QLabel:
        lab = QtWidgets.QLabel(text)
        lab.setProperty("tone", "muted")
        lab.setMinimumWidth(90)
        return lab

    def _row(self, label: str, widget: QtWidgets.QWidget):
        lab = self._label(label)
        self.options.addRow(lab, widget)
        return lab, widget

    def _set_row(self, name: str, shown: bool) -> None:
        for w in self.rows[name]:
            w.setVisible(shown)

    # ── state ────────────────────────────────────────────────────────────────
    def _restore(self) -> None:
        saved = self._saved
        o = saved.get("opts", {})
        self.source.set_value(o.get("source", "annotation" if self.has_annotation else "analysis"))
        self.range.set_value(o.get("range", "whole"))
        self.time.set_value(o.get("time_format", "seconds"))
        self.open_after_box.setChecked(bool(saved.get("open_after", False)))
        recent = [f for f in saved.get("recent", []) if Path(f).is_dir()]
        folder = saved.get("folder")
        if folder and Path(folder).is_dir() and folder not in recent:
            recent.insert(0, folder)
        self.folder.addItems(recent or [str(downloads_dir())])
        self.folder.setCurrentIndex(0)
        self.cards.get(saved.get("format"), self.cards["csv"]).setChecked(True)

    def _format_changed(self, fmt: str) -> None:
        self.fmt = fmt
        self._set_row("source", fmt in ("csv", "chart"))
        self._set_row("range", fmt in ("csv", "chart", "context"))
        self._set_row("time", fmt == "csv")
        # Offer only what exists: a loop region, an open Annotation; "whole" is for the BPM exports only.
        self.range.set_enabled("region", self.has_region, "Shift+drag a loop region first")
        self.range.set_enabled("whole", fmt in ("csv", "chart"), "Only the BPM exports can cover the whole recording")
        if not self.range.is_enabled(self.range.value()):
            self.range.set_value("visible")
        self.source.set_enabled("annotation", self.has_annotation, "No Annotation is open")
        if not self.has_annotation:
            self.source.set_value("analysis")
        _label, _desc, ext, suffix = FORMATS[fmt]
        self.ext.setText(ext)
        self.name.setText(f"{self.stem}{suffix}")

    def _browse(self) -> None:
        start = self.folder.currentText().strip() or str(downloads_dir())
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Export to folder", start)
        if d:
            self.folder.setEditText(str(Path(d)))
            theme.set_prop(self.folder.lineEdit(), "invalid", False)

    def _target(self) -> Optional[Path]:
        """The file to write, or None (with the folder field marked) when the folder doesn't exist.
        A full file path pasted into the folder field is split into folder and name."""
        folder = Path(self.folder.currentText().strip().strip('"')).expanduser()
        name = self.name.text().strip() or f"{self.stem}{FORMATS[self.fmt][3]}"
        if not folder.is_dir() and folder.suffix and folder.parent.is_dir():
            folder, name = folder.parent, folder.name
        if not folder.is_dir():
            theme.set_prop(self.folder.lineEdit(), "invalid", True)
            self.folder.setFocus()
            return None
        ext = FORMATS[self.fmt][2]
        if not name.lower().endswith(ext):
            name += ext
        return folder / name

    def _export(self) -> None:
        path = self._target()
        if path is None:
            return
        if path.exists():
            ask = QtWidgets.QMessageBox.question(self, "Replace file", f"{path.name} already exists. Replace it?")
            if ask != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self.path = str(path)
        self.open_after = self.open_after_box.isChecked()
        self.opts = {
            "range": self.range.value(),
            "time_format": self.time.value(),
            "source": self.source.value(),
        }
        folder = str(path.parent)
        recent = [folder] + [f for f in self._saved.get("recent", []) if f != folder]
        self.settings.set_json("export", {"format": self.fmt, "folder": folder, "opts": self.opts,
                                          "recent": recent[:RECENT_FOLDERS], "open_after": self.open_after})
        self.accept()
