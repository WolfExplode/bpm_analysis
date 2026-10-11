"""Offscreen recovery marker routing, recalculation and recording persistence."""
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from pcg.analysis import Library
from pcg.app.state import Settings
from pcg.app.workspace import Workspace, ANN_BPM
from pcg.engine.traces import Trace


@pytest.fixture
def editor(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = Settings.__new__(Settings)
    settings.q = QtCore.QSettings(str(tmp_path / "ui.ini"), QtCore.QSettings.Format.IniFormat)
    ws = Workspace(settings, Library(tmp_path / "library"))
    t = np.arange(0, 701, 0.5)
    trace = Trace("BPM", "Result", "bpm", "line", t, 180 - 0.05 * t)
    ws.analysis = SimpleNamespace(trace=lambda name: trace if name == "BPM" else None, duration_sec=700)
    ws.fingerprint = "recording-one"
    ws.lanes["bpm"].items["BPM"] = pg.PlotDataItem(trace.x, trace.y)
    ws.lanes["bpm"].plot.addItem(ws.lanes["bpm"].items["BPM"])
    ws.resize(1500, 900)
    ws.show()
    ws.lanes["states"].plot.setXRange(0, 700, padding=0)
    ws.lanes["bpm"].plot.setYRange(130, 190, padding=0)
    app.processEvents()
    yield ws.recovery_editor
    ws.close()
    app.processEvents()


def pixel(editor, x):
    widget = editor.ws.lanes["bpm"].widget
    return widget.mapFromScene(editor.vb.mapViewToScene(QtCore.QPointF(x, 180 - 0.05 * x)))


@pytest.mark.parametrize("key", [QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace])
def test_click_to_add_drag_select_hover_and_delete(editor, key):
    ws = editor.ws
    widget = ws.lanes["bpm"].widget
    viewport = widget.viewport()
    QtTest.QTest.mouseClick(viewport, QtCore.Qt.MouseButton.LeftButton, pos=pixel(editor, 10))
    assert not editor.starts
    ws.gain_editor.add_btn.setChecked(True)
    editor.add_btn.click()
    assert not ws.gain_editor.add_btn.isChecked()
    QtTest.QTest.mouseClick(viewport, QtCore.Qt.MouseButton.LeftButton, pos=pixel(editor, 10))
    assert len(editor.starts) == 1
    assert not editor.add_btn.isChecked()
    start, end = pixel(editor, editor.starts[0]), pixel(editor, 40)
    QtTest.QTest.mousePress(viewport, QtCore.Qt.MouseButton.LeftButton, pos=start)
    event = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseMove, QtCore.QPointF(end),
                             QtCore.QPointF(viewport.mapToGlobal(end)), QtCore.Qt.MouseButton.NoButton,
                             QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier)
    QtWidgets.QApplication.sendEvent(viewport, event)
    QtTest.QTest.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=end)
    assert editor.starts[0] == pytest.approx(40, abs=1)
    line = editor.lines[-1][1]
    pos = editor.vb.mapViewToScene(QtCore.QPointF(line.end, line.end_bpm))
    text = editor.hover_text(pos)
    assert "bpm/s" in text and "BPM change" in text
    assert len(text.splitlines()) == 2
    editor.select(None)
    QtTest.QTest.mouseClick(viewport, QtCore.Qt.MouseButton.LeftButton, pos=pixel(editor, editor.starts[0]))
    assert editor.selected == 0
    ws.activateWindow()
    QtTest.QTest.qWait(20)
    QtTest.QTest.keyClick(widget, key)
    assert not editor.starts and not editor.lines


def test_persistence_next_marker_source_switch_and_right_click_readout(editor):
    editor.add(10)
    editor.add(150)
    assert [line.end - line.start for i, line in editor.lines if i == 0] == [30, 60, 120]
    editor.load_recording()
    assert editor.starts == (10, 150)
    editor.ws.fingerprint = "recording-two"
    editor.load_recording()
    assert not editor.starts
    editor.ws.fingerprint = "recording-one"
    editor.load_recording()
    t = np.arange(0, 701, 0.5)
    editor.ws._ann_bpm = (t, 170 - 0.1 * t, 170 - 0.1 * t)
    item = pg.PlotDataItem(t, editor.ws._ann_bpm[2])
    editor.ws.lanes["bpm"].items[ANN_BPM] = item
    editor.plot.addItem(item)
    editor.draw()
    assert editor.source == "Annotation BPM"
    assert editor.lines[0][1].slope == pytest.approx(-0.1)
    item.hide()
    editor.draw()
    assert editor.lines[0][1].slope == pytest.approx(-0.05)
    widget = editor.ws.lanes["bpm"].widget
    pos = pixel(editor, 10)
    QtTest.QTest.mousePress(widget.viewport(), QtCore.Qt.MouseButton.RightButton, pos=pos)
    assert editor.ws._bpm_readout.isVisible()
    assert editor.starts == (10, 150)
    QtTest.QTest.mouseRelease(widget.viewport(), QtCore.Qt.MouseButton.RightButton, pos=pos)
    assert editor.starts == (10, 150)
    editor.select(0)
    editor.delete_selected()
    assert editor.starts == (150,)
    assert editor.ws.settings.get_json("recovery_starts", {})["recording-one"] == [150]


def test_recovery_hover_stays_visible_until_pointer_leaves(editor):
    editor.add(10)
    ws = editor.ws
    lane = ws.lanes["bpm"]
    pos = editor.vb.mapViewToScene(QtCore.QPointF(10, 179.5))
    ws._on_mouse_moved(pos, lane)
    assert editor.hover_label.isVisible()
    assert "Recovery 1 start" in editor.hover_label.text()
    QtTest.QTest.qWait(11000)
    assert editor.hover_label.isVisible()
    QtWidgets.QApplication.sendEvent(lane.widget.viewport(), QtCore.QEvent(QtCore.QEvent.Type.Leave))
    assert editor.hover_label.isHidden()
    line = editor.lines[-1][1]
    pos = editor.vb.mapViewToScene(QtCore.QPointF(line.end, line.end_bpm))
    ws._on_mouse_moved(pos, lane)
    assert editor.hover_label.isVisible()
    assert "bpm/s" in editor.hover_label.text()
    ws._on_mouse_moved(editor.vb.mapViewToScene(QtCore.QPointF(300, 185)), lane)
    assert editor.hover_label.isHidden()


def test_only_endpoints_show_measurements_on_overlapping_lines(editor):
    editor.add(10)
    for index, line in editor.lines:
        endpoint = editor.vb.mapViewToScene(QtCore.QPointF(line.end, line.end_bpm))
        text = editor.hover_text(endpoint)
        assert f"BPM change: {-0.05 * (line.end - line.start):+.2f} BPM" in text
        assert "bpm/s" in text
        # The linear fixture makes every recovery chord overlap here.
        midpoint = editor.vb.mapViewToScene(QtCore.QPointF(35, 180 - 0.05 * 35))
        assert editor.hover_text(midpoint) == ""
        assert editor.hover_text(endpoint + QtCore.QPointF(9, 0)) == ""


def test_metadata_comparison_saves_and_reloads_without_cross_recording_leak(editor):
    from pcg import recording_metadata

    panel = editor.ws.recording_panel
    panel.load_recording()
    editor.add(10)
    panel.age.setValue(40)
    panel.seated_rest.setValue(80)
    text = panel.results.toPlainText()
    assert "179.5 bpm" in text  # Marker BPM, not recording-start BPM.
    assert "60 s: 134.9 bpm" in text
    assert "120 s: 120.3 bpm" in text
    saved = recording_metadata.load(editor.ws.library.root, editor.ws.fingerprint)
    assert saved["person"]["age"] == 40
    assert saved["person"]["seated_resting_hr"] == 80
    editor.ws.fingerprint = "recording-two"
    editor.load_recording()
    panel.load_recording()
    assert panel.age.value() == 0
    assert "60 s: 134.9 bpm" not in panel.results.toPlainText()
    editor.ws.fingerprint = "recording-one"
    editor.load_recording()
    panel.load_recording()
    assert panel.age.value() == 40 and panel.seated_rest.value() == 80
    assert "60 s: 134.9 bpm" in panel.results.toPlainText()


def test_recording_tab_and_annotation_source_refresh(editor):
    panel = editor.ws.recording_panel
    panel.load_recording()
    editor.ws.recording_btn.click()
    assert editor.ws.side_stack.currentWidget() is panel
    assert editor.ws.recording_btn.isChecked() and not editor.ws.summary_btn.isChecked()
    editor.add(10)
    assert "Result BPM" in panel.results.toPlainText()
    t = np.arange(0, 701, 0.5)
    editor.ws._ann_bpm = (t, 150 - 0.03 * t, 150 - 0.03 * t)
    editor.ws.lanes["bpm"].items[ANN_BPM] = pg.PlotDataItem(t, 150 - 0.03 * t)
    editor.plot.addItem(editor.ws.lanes["bpm"].items[ANN_BPM])
    editor.draw()
    text = panel.results.toPlainText()
    assert "Annotation BPM" in text and "149.7 bpm" in text
    editor.ws.summary_btn.click()
    assert editor.ws.side_stack.currentWidget() is editor.ws.summary_text
    assert not editor.ws.recording_btn.isChecked()


def test_corrupt_metadata_shows_error_and_cannot_overwrite(editor):
    from pcg import recording_metadata

    path = recording_metadata.metadata_path(editor.ws.library.root, editor.ws.fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("bad JSON", encoding="utf-8")
    panel = editor.ws.recording_panel
    panel.load_recording()
    assert not panel.form.isEnabled()
    assert "Cannot read recording metadata" in panel.save_status.text()
    panel.age.setValue(40)
    editor.add(10)
    assert path.read_text() == "bad JSON"


def test_typing_metadata_does_not_trigger_graph_shortcuts(editor):
    panel = editor.ws.recording_panel
    panel.load_recording()
    editor.add(10)
    editor.ws._show_side("recording")
    panel.age.setFocus()
    panel.age.selectAll()
    QtTest.QTest.keyClicks(panel.age, "21")
    QtTest.QTest.keyClick(panel.age, QtCore.Qt.Key.Key_Return)
    assert panel.age.value() == 21
    panel.age.selectAll()
    QtTest.QTest.keyClicks(panel.age, "21")
    QtTest.QTest.keyClick(panel.age, QtCore.Qt.Key.Key_Backspace)
    QtTest.QTest.keyClick(panel.age, QtCore.Qt.Key.Key_Return)
    assert panel.age.value() == 2
    assert editor.starts == (10,)


def test_seated_estimate_defaults_and_needs_its_own_resting_hr(editor):
    panel = editor.ws.recording_panel
    panel.load_recording()
    editor.add(10)
    panel.data["person"]["resting_hr"] = 80  # Legacy supine HR is not silently repurposed.
    assert "Enter resting heart rate" in panel.results.toPlainText()
    panel.seated_rest.setValue(80)
    text = panel.results.toPlainText()
    assert "179.5 bpm" in text and "Estimated drop" in text
    assert "60 s: 127.4 bpm" in text
    assert "300 s: 105.1 bpm" in text
    assert "Beyond model" not in text and "Unavailable" not in text
    assert "Median" not in text and "P25" not in text
    assert panel.data["person"]["seated_resting_hr"] == 80
    panel.load_recording()
    assert panel.seated_rest.value() == 80
    assert "60 s: 127.4 bpm" in panel.results.toPlainText()


def test_seated_estimate_recalculates_from_marker_and_source(editor):
    panel = editor.ws.recording_panel
    panel.load_recording()
    editor.add(10)
    panel.seated_rest.setValue(80)
    editor.move(0, 100)
    text = panel.results.toPlainText()
    assert "175.0 bpm" in text and "60 s: 125.2 bpm" in text
    t = np.arange(0, 701, 0.5)
    editor.ws._ann_bpm = (t, 150 - 0.03 * t, 150 - 0.03 * t)
    item = pg.PlotDataItem(t, editor.ws._ann_bpm[2])
    editor.ws.lanes["bpm"].items[ANN_BPM] = item
    editor.plot.addItem(item)
    editor.draw()
    text = panel.results.toPlainText()
    assert "Annotation BPM" in text and "147.0 bpm" in text
    assert "60 s: 111.9 bpm" in text
    panel.seated_rest.setValue(160)
    assert "no greater than starting BPM" in panel.results.toPlainText()
    assert "60 s: 111.9 bpm" not in panel.results.toPlainText()


def test_seated_age_sex_changes_refresh_and_reload(editor):
    panel = editor.ws.recording_panel
    panel.load_recording()
    editor.add(10)
    panel.seated_rest.setValue(80)
    panel.age.setValue(20)
    assert "60 s: 131.3 bpm" in panel.results.toPlainText()
    assert "300 s: 105.1 bpm" not in panel.results.toPlainText()
    panel.sex.setCurrentIndex(panel.sex.findData("female"))
    assert "60 s: 127.4 bpm" in panel.results.toPlainText()
    panel.age.setValue(40)
    assert "60 s: 131.0 bpm" in panel.results.toPlainText()
    panel.load_recording()
    assert panel.sex.currentData() == "female"
    assert panel.age.value() == 40
    assert "60 s: 131.0 bpm" in panel.results.toPlainText()
    panel.age.setValue(70)
    assert "greater than 16 and less than 70" in panel.results.toPlainText()
    assert "Estimated endpoint BPM" not in panel.results.toPlainText()
    panel.age.setValue(0)
    assert "300 s: 105.1 bpm" in panel.results.toPlainText()
