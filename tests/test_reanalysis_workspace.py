"""Selected-range edits are one undo step; stale, failed and cancelled jobs do not edit."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import numpy as np
import soundfile as sf
from PySide6 import QtCore, QtTest, QtWidgets

from pcg import analysis, annotation as an
from pcg.app.state import AnnotationDoc, Settings
from pcg.app.workspace import Workspace
from pcg.reanalysis import RangeResult


@pytest.fixture
def workspace(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = Settings.__new__(Settings)
    settings.q = QtCore.QSettings(str(tmp_path / "ui.ini"), QtCore.QSettings.Format.IniFormat)
    ws = Workspace(settings, analysis.Library(tmp_path / "library"))
    ws.path = str(tmp_path / "recording.wav")
    ws._set_doc(AnnotationDoc(an.Annotation("fp", 10, "recording.wav", (
        an.Span("S1", .9, 1.1), an.Span("S2", 1.3, 1.4), an.Span("S1", 3, 3.1))), None, draft=False))
    ws.region = (1, 2)
    yield ws
    ws.shutdown()
    ws.close()
    app.processEvents()


def target(ws):
    ws._range_target = (ws.path, ws.doc, 1, 2, tuple(s for s in ws.doc.spans if s.overlaps(1, 2)))


def result():
    return RangeResult(1, 2, 0, 10, "springer", (an.Span("S1", 1.5, 1.6, an.ORIGIN_ALGORITHM),))


def test_range_edit_undo_redo(workspace):
    ws = workspace
    original = ws.doc.spans
    target(ws)
    ws._on_range_finished(result(), "")
    modified = ws.doc.spans
    assert modified != original
    assert ws.doc.dirty
    assert [(s.start, s.end) for s in modified] == [(.9, 1), (1.5, 1.6), (3, 3.1)]
    ws.doc.undo()
    assert ws.doc.spans == original and not ws.doc.dirty
    ws.doc.redo()
    assert ws.doc.spans == modified


@pytest.mark.parametrize("error", ["cancelled", "algorithm failed"])
def test_failed_or_cancelled_job_keeps_labels(workspace, error):
    original = workspace.doc.spans
    target(workspace)
    workspace._on_range_finished(None, error)
    assert workspace.doc.spans == original
    assert not workspace.doc.dirty


def test_edit_inside_selection_discards_inflight_result(workspace):
    target(workspace)
    workspace.doc.apply(lambda spans: an.place_sound(spans, "S2", 1.6, .1, 10))
    modified = workspace.doc.spans
    workspace._on_range_finished(result(), "")
    assert workspace.doc.spans == modified
    assert "edited while" in workspace.status.text()


def test_edit_outside_selection_survives_inflight_result(workspace):
    target(workspace)
    workspace.doc.apply(lambda spans: an.place_sound(spans, "S2", 8, .1, 10))
    workspace._on_range_finished(result(), "")
    assert any(s.center == 8 for s in workspace.doc.spans)
    assert any(s.start == 1.5 for s in workspace.doc.spans)


def test_switching_annotation_discards_inflight_result(workspace):
    target(workspace)
    workspace._set_doc(AnnotationDoc(an.Annotation("other", 10, "other.wav"), None, draft=True))
    workspace._on_range_finished(result(), "")
    assert workspace.doc.spans == ()


def test_dialog_routes_algorithm_and_hint(workspace, monkeypatch):
    requests = []

    def accept(dialog):
        dialog.findChild(QtWidgets.QComboBox).setCurrentIndex(1)
        dialog.findChild(QtWidgets.QDoubleSpinBox).setValue(100)
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(QtWidgets.QDialog, "exec", accept)
    monkeypatch.setattr(workspace.range_worker, "start_range",
                        lambda *args, **kwargs: requests.append((args, kwargs)))
    workspace.rerun_selection()
    assert requests[0][0] == (workspace.path, 1, 2)
    assert requests[0][1] == {"algorithm": "native", "channel": "mixed", "bpm_hint": 100}
    assert not workspace.rerun_btn.isEnabled()
    assert not workspace.range_btn.isEnabled()


@pytest.mark.parametrize("algorithm", ["springer", "native"])
def test_worker_runs_cli_in_fresh_process(workspace, tmp_path, algorithm):
    sr = 2000
    times = np.arange(sr * 20) / sr
    signal = np.zeros_like(times)
    for s1 in np.arange(.5, 19.5, .75):
        for center, frequency, amplitude in ((s1, 65, 1), (s1 + .25, 110, .65)):
            relative = times - center
            signal += amplitude * np.exp(-(relative / .025) ** 2) * np.sin(2 * np.pi * frequency * relative)
    path = tmp_path / "synthetic.wav"
    sf.write(path, signal, sr, subtype="FLOAT")
    worker = workspace.range_worker
    # Isolate protocol verification from the workspace's result application.
    worker.range_finished.disconnect(workspace._on_range_finished)
    spy = QtTest.QSignalSpy(worker.range_finished)
    worker.start_range(str(path), 6, 8, algorithm=algorithm, channel="mixed", bpm_hint=80)
    assert spy.wait(30000), "selected-range worker did not finish"
    result, error = spy.at(0)
    assert error == ""
    assert result.algorithm == algorithm
    assert (result.start, result.end) == (6, 8)
    assert all(6 <= s.start < s.end <= 8 for s in result.spans)
    assert not list(tmp_path.glob("*.annotation.json"))
