"""Files opened from the CLI remain visible in Run, including Workspace analyses."""
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import soundfile as sf
from PySide6 import QtCore, QtWidgets
from shiboken6 import isValid

from pcg import analysis, annotation as an
from pcg.app import main, state
from pcg.app.run_screen import RunScreen


@pytest.fixture
def app_setup(tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(state.Settings, "__init__", lambda self: setattr(
        self, "q", QtCore.QSettings(str(tmp_path / "ui.ini"), QtCore.QSettings.Format.IniFormat)))
    monkeypatch.setattr(RunScreen, "_check_status", lambda self, paths: None)
    return app


@pytest.mark.parametrize("analyze", [False, True])
def test_single_file_launch_populates_run_and_uses_one_worker(app_setup, tmp_path, monkeypatch, analyze):
    path = tmp_path / "recording.wav"
    sf.write(str(path), np.zeros(400), 4000)
    calls = []
    windows = []
    real_window = main.MainWindow

    def make_window(library):
        win = real_window(library)
        windows.append(win)
        return win

    def start_process(worker, args):
        calls.append(args)
        worker.test_running = True

    monkeypatch.setattr(main, "MainWindow", make_window)
    monkeypatch.setattr(state.AnalyzeWorker, "_start_process", start_process)
    monkeypatch.setattr(state.AnalyzeWorker, "running", property(lambda self: getattr(self, "test_running", False)))
    monkeypatch.setattr(main.Workspace, "_load_audio", lambda self: None)

    def exec_app(self):
        win = windows[0]
        run = win.run_screen
        assert win.tabs.currentWidget() is win.workspace
        assert [r.path for r in run.rows] == [str(path)]
        assert run.rows[0].status == "starting…"
        assert run.workers[str(path)] is win.workspace.worker
        assert len(calls) == 1  # Opening starts analysis; --analyze must not duplicate it.
        win.workspace.worker.progress.emit("Pass 1")
        assert run.rows[0].status == "Pass 1"
        assert "Pass 1" in run.console.toPlainText()
        return 0

    monkeypatch.setattr(QtWidgets.QApplication, "exec", exec_app)
    try:
        assert main.main([str(path)], library=str(tmp_path / "library"), analyze=analyze) == 0
    finally:
        for win in windows:
            win.run_screen.workers.clear()
            win.workspace.worker.test_running = False
            win.workspace.shutdown()
            win.close()
        app_setup.processEvents()


@pytest.mark.parametrize("error,status", [("", "done"), ("decoder failed", "failed"), ("cancelled", "cancelled")])
def test_workspace_worker_reports_result_and_can_be_reused(app_setup, tmp_path, monkeypatch, error, status):
    run = RunScreen(state.Settings(), analysis.Library(tmp_path / "library"))
    owner = QtCore.QObject()
    worker = state.AnalyzeWorker(owner)
    run.watch_workspace_worker(worker)
    path = str(tmp_path / "recording.wav")
    bpm = {"start_bpm": 100, "min_bpm": 90, "max_bpm": 110}
    monkeypatch.setattr(analysis, "load", lambda path: SimpleNamespace(summary={"bpm": bpm}))
    try:
        worker.started.emit(path)
        worker.log.emit("WARNING root: example warning")
        assert "example warning" in run.console.toPlainText()
        worker.finished.emit([] if error else ["saved.analysis.zip"], error)
        assert run.rows[0].status == status
        assert not run.busy()
        if not error:
            assert run.rows[0].bpm == bpm
            assert run.rows[0].analyses == ["saved.analysis.zip"]
            assert "100, 90–110 BPM" in run.rows[0].result
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        assert isValid(worker)  # Workspace owns this worker and needs it for Ctrl+R.
        worker.started.emit(path)
        assert len(run.rows) == 1
        assert run.busy()
        assert not run.rows[0].log
        assert run.rows[0].status == "starting…"
    finally:
        run.workers.clear()
        run.close()


@pytest.mark.parametrize("recording,error,refresh", [
    ("recording.wav", "", True),
    ("other.wav", "", False),
    ("recording.wav", "decoder failed", False),
    ("recording.wav", "cancelled", False),
])
def test_run_completion_refreshes_only_matching_workspace(app_setup, tmp_path, monkeypatch,
                                                        recording, error, refresh):
    win = main.MainWindow(analysis.Library(tmp_path / "library"))
    ws = win.workspace
    ws.path = str(tmp_path / "recording.wav")
    ws._set_doc(state.AnnotationDoc(an.Annotation("fp", 10, "recording.wav", (
        an.Span("S1", 1, 1.1),)), None, draft=False))
    ws.doc.apply(lambda spans: spans + (an.Span("S2", 2, 2.1),))
    doc, spans = ws.doc, ws.doc.spans
    old = SimpleNamespace(channel="right")
    ws.analysis = old
    ws.region = (2, 4)
    ws.selected = spans[0]
    ws.lanes["states"].plot.setXRange(2, 4, padding=0)
    view = ws.lanes["states"].plot.getViewBox().viewRange()[0]
    new = SimpleNamespace(
        channel="right", duration_sec=10,
        states=SimpleNamespace(start=np.array([3.0]), end=np.array([3.1]), state=["S1"]),
        summary={"bpm": {"start_bpm": 100, "min_bpm": 90, "max_bpm": 110}},
    )
    opened, updated = [], []

    def open_analysis(path):
        opened.append(path)
        return new

    monkeypatch.setattr(win.library, "open", open_analysis)
    monkeypatch.setattr(analysis, "load", lambda path: new)
    for method in ("_rebuild_plots", "_update_header", "_on_doc_changed"):
        monkeypatch.setattr(ws, method, lambda method=method: updated.append(method))
    monkeypatch.setattr(ws, "_load_audio", lambda: None)
    run_path = str(tmp_path / recording)
    paths = ["new-left.analysis.zip", "new-right.analysis.zip"]
    win.run_screen.add_paths([run_path])
    try:
        win.run_screen._finished(run_path, [] if error else paths, error)
        assert ws.analysis is (new if refresh else old)
        assert opened == ([paths[1]] if refresh else [])
        assert updated == (["_rebuild_plots", "_update_header", "_on_doc_changed"] if refresh else [])
        assert ws.doc is doc and ws.doc.spans == spans and ws.doc.dirty
        assert ws.selected == spans[0] and ws.region == (2, 4)
        np.testing.assert_allclose(ws.lanes["states"].plot.getViewBox().viewRange()[0], view)
        assert win.tabs.currentWidget() is win.run_screen
        if refresh:
            assert ws.algo_spans[0].start == 3.0
    finally:
        ws.doc = None  # Avoid an unsaved-Annotation prompt during teardown.
        win.close()
        app_setup.processEvents()
