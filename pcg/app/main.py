"""The desktop app: run screen + workspace in one window."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from pcg import analysis as A
from pcg import recording

from .run_screen import RunScreen
from .state import Settings
from .workspace import Workspace

HELP = """\
<h3>Workspace keys</h3>
<table>
<tr><td><b>Space</b></td><td>play / pause</td></tr>
<tr><td><b>T</b></td><td>toggle original / filtered audio</td></tr>
<tr><td><b>Click</b></td><td>move playhead (on the Annotation row: select span)</td></tr>
<tr><td><b>Shift+drag / Shift+click</b></td><td>set loop region / clear it</td></tr>
<tr><td><b>Drag, wheel</b></td><td>pan, zoom (time only)</td></tr>
<tr><td><b>L</b></td><td>follow playhead on/off</td></tr>
<tr><td><b>1 / 2</b></td><td>place S1 / S2 at the playhead (replaces what it overlaps)</td></tr>
<tr><td><b>X / Del</b></td><td>delete selected span (else the one under the playhead)</td></tr>
<tr><td><b>F</b></td><td>relabel S1 ↔ S2</td></tr>
<tr><td><b>N + drag</b></td><td>paint a Noisy span; drag a Noisy span's edge to resize it</td></tr>
<tr><td><b>A</b></td><td>replace the loop region with the algorithm's output</td></tr>
<tr><td><b>] / [</b></td><td>next / previous Defect or Disagreement (panel filter applies)</td></tr>
<tr><td><b>Ctrl+Z / Ctrl+Shift+Z</b></td><td>undo / redo</td></tr>
<tr><td><b>Ctrl+S</b></td><td>save Annotation (protected folders: Save As, starting in Downloads)</td></tr>
<tr><td><b>Ctrl+R</b></td><td>re-run the engine (fresh process — picks up code edits)</td></tr>
<tr><td><b>Ctrl+Shift+C</b></td><td>copy the visible window as text for an LLM</td></tr>
</table>
"""


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, library: A.Library):
        super().__init__()
        self.settings = Settings()
        self.library = library
        self.run_screen = RunScreen(self.settings, library)
        self.workspace = Workspace(self.settings, library)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.run_screen, "Run")
        self.tabs.addTab(self.workspace, "Workspace")
        self.setCentralWidget(self.tabs)
        self.run_screen.open_requested.connect(self.open_in_workspace)
        self.workspace.title_changed.connect(self._title)
        self._title("")
        self._menus()
        geo = self.settings.q.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1600, 950)

    def _menus(self) -> None:
        ws = self.workspace
        m = self.menuBar().addMenu("&File")
        m.addAction("Open recording…", self.open_recording_dialog, QtGui.QKeySequence("Ctrl+O"))
        m.addAction("Open Annotation…", ws.open_annotation_file)
        m.addAction("Save Annotation", ws.save_annotation)
        m.addSeparator()
        e = m.addMenu("Export")
        e.addAction("BPM CSV (Analysis)…", lambda: ws.export_bpm_csv("analysis"))
        e.addAction("BPM CSV (Annotation)…", lambda: ws.export_bpm_csv("annotation"))
        e.addAction("Summary…", ws.export_summary)
        e.addAction("Current view as image…", ws.export_image)
        m.addSeparator()
        m.addAction("Protected folders…", self.edit_protected_folders)
        m.addAction("Open library folder", lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(self.library.root))))
        m.addSeparator()
        m.addAction("Quit", self.close, QtGui.QKeySequence("Ctrl+Q"))
        h = self.menuBar().addMenu("&Help")
        h.addAction("Keys", lambda: QtWidgets.QMessageBox.information(self, "Keys", HELP))

    def _title(self, text: str) -> None:
        self.setWindowTitle(f"{text} — PCG Workspace" if text else "PCG Workspace")

    def open_recording_dialog(self) -> None:
        exts = " ".join(f"*{e}" for e in recording.AUDIO_EXTENSIONS)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open recording", "", f"Audio ({exts})")
        if path:
            self.open_in_workspace(path, "")

    def open_in_workspace(self, path: str, analysis_path: str = "") -> None:
        if not self.workspace.maybe_close():
            return
        self.tabs.setCurrentWidget(self.workspace)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            self.workspace.open_recording(path, analysis_path or None)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Could not open recording", f"{Path(path).name}\n\n{e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.workspace.glw.setFocus()

    def edit_protected_folders(self) -> None:
        text, ok = QtWidgets.QInputDialog.getMultiLineText(
            self, "Protected folders",
            "Annotations for recordings inside these folders are saved via Save As (starting in Downloads).\n"
            "One folder per line:", "\n".join(self.settings.protected_folders))
        if ok:
            self.settings.protected_folders = [ln.strip() for ln in text.splitlines() if ln.strip()]

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        if not self.workspace.maybe_close():
            ev.ignore()
            return
        if self.run_screen.busy():
            r = QtWidgets.QMessageBox.question(self, "Analyses running", "Stop the running analyses and quit?")
            if r != QtWidgets.QMessageBox.StandardButton.Yes:
                ev.ignore()
                return
            self.run_screen.stop()
        self.workspace.shutdown()
        self.settings.q.setValue("geometry", self.saveGeometry())
        ev.accept()


def main(paths: Optional[List[str]] = None, library: Optional[str] = None) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("PCG Workspace")
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    for role, color in ((QtGui.QPalette.ColorRole.Window, "#1e1e1e"), (QtGui.QPalette.ColorRole.WindowText, "#dddddd"),
                        (QtGui.QPalette.ColorRole.Base, "#151515"), (QtGui.QPalette.ColorRole.AlternateBase, "#222222"),
                        (QtGui.QPalette.ColorRole.Text, "#dddddd"), (QtGui.QPalette.ColorRole.Button, "#2b2b2b"),
                        (QtGui.QPalette.ColorRole.ButtonText, "#dddddd"), (QtGui.QPalette.ColorRole.Highlight, "#2f6db5"),
                        (QtGui.QPalette.ColorRole.ToolTipBase, "#202020"), (QtGui.QPalette.ColorRole.ToolTipText, "#eeeeee")):
        pal.setColor(role, QtGui.QColor(color))
    app.setPalette(pal)
    win = MainWindow(A.Library(library))
    win.show()
    paths = paths or []
    files = [p for p in paths if Path(p).is_file()]
    if len(files) == 1 and len(paths) == 1:
        win.open_in_workspace(files[0])
    elif paths:
        win.run_screen.add_paths(paths)
    return app.exec()
