"""Offscreen checks of zero anchoring and handle wheel routing, with synthetic audio."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from pcg.analysis import Library
from pcg.app.state import Settings
from pcg.app.workspace import Workspace
from pcg.app.workspace import ORIGINAL_ENVELOPE


@pytest.fixture
def editor(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = Settings.__new__(Settings)
    settings.q = QtCore.QSettings(str(tmp_path / "ui.ini"), QtCore.QSettings.Format.IniFormat)
    ws = Workspace(settings, Library(tmp_path / "library"))
    ws.resize(1500, 900)
    ws.show()
    ws.player.set_audio(np.ones(32000, np.float32), 16000)
    ws.lanes["states"].plot.setXRange(0, 2, padding=0)
    ws.lanes["signal"].plot.setYRange(0, 1, padding=0)
    ws.gain_editor.add_btn.setChecked(True)
    app.processEvents()
    yield ws.gain_editor
    ws.close()
    app.processEvents()


class Wheel:
    def __init__(self, pos, delta):
        self.pos, self.amount, self.accepted = pos, delta, False

    def scenePos(self):
        return self.pos

    def delta(self):
        return self.amount

    def accept(self):
        self.accepted = True


def test_zero_anchor_survives_y_zoom_and_gains_round_trip(editor):
    for lo, hi in ((-0.65, 1), (-2, 3), (-0.1, 0.2)):
        editor.plot.setYRange(lo, hi, padding=0)
        assert editor.y_of(0) == 0
        for db in (-60, -18, 0, 6, 12):
            assert editor.db_of(float(editor.y_of(db))) == pytest.approx(db)


def test_wheel_on_center_changes_q_not_timeline(editor):
    editor.add(0.5, -12)
    original = editor.edits[0]
    view = editor.vb.viewRange()
    pos = editor.vb.mapViewToScene(QtCore.QPointF(original.center, float(editor.y_of(original.db))))
    event = Wheel(pos, 120)
    editor.vb.wheelEvent(event)
    assert event.accepted
    assert editor.edits[0].q > original.q
    assert editor.edits[0].bandwidth < original.bandwidth
    assert editor.edits[0].db == original.db
    assert editor.vb.viewRange() == view
    assert editor.ws.player._gain_edits == editor.edits
    assert editor.wheel(Wheel(pos, -120))
    assert editor.edits[0].q == pytest.approx(original.q)
    assert editor.wheel(Wheel(pos, -120))
    assert editor.edits[0].q < original.q
    assert not editor.wheel(Wheel(editor.vb.mapViewToScene(QtCore.QPointF(1.5, 0)), 120))


@pytest.mark.parametrize("key", [QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace])
def test_add_requires_plus_then_graph_select_drag_and_keyboard_delete(editor, key):
    widget = editor.ws.lanes["signal"].widget
    viewport = widget.viewport()

    def pixel(x, db):
        return widget.mapFromScene(editor.vb.mapViewToScene(QtCore.QPointF(x, float(editor.y_of(db)))))

    editor.add_btn.setChecked(False)
    QtTest.QTest.mouseClick(viewport, QtCore.Qt.MouseButton.LeftButton, pos=pixel(0.5, 0))
    assert not editor.edits
    editor.add_btn.click()
    QtTest.QTest.mouseClick(viewport, QtCore.Qt.MouseButton.LeftButton, pos=pixel(0.5, 0))
    assert len(editor.edits) == 1
    assert not editor.add_btn.isChecked()
    start, end = pixel(editor.edits[0].center, 0), pixel(0.6, -18)
    QtTest.QTest.mousePress(viewport, QtCore.Qt.MouseButton.LeftButton, pos=start)
    event = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseMove, QtCore.QPointF(end),
                             QtCore.QPointF(viewport.mapToGlobal(end)), QtCore.Qt.MouseButton.NoButton,
                             QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier)
    QtWidgets.QApplication.sendEvent(viewport, event)
    QtTest.QTest.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=end)
    assert editor.edits[0].db < -15
    editor.select(None)
    QtTest.QTest.mouseClick(viewport, QtCore.Qt.MouseButton.LeftButton,
                           pos=pixel(editor.edits[0].center, editor.edits[0].db))
    assert editor.selected == 0
    # Shortcut dispatch requires an active offscreen window.
    editor.ws.activateWindow()
    QtTest.QTest.qWait(20)
    QtTest.QTest.keyClick(widget, key)
    assert not editor.edits
    assert not editor.buttons


def test_number_buttons_toggle_only_their_point_and_renumber_after_deletion(editor):
    editor.add(0.5, -12)
    editor.add(1.0, 6)
    editor.buttons[0].click()
    assert not editor.edits[0].enabled
    assert editor.edits[1].enabled
    assert editor.ws.player._gain_edits == editor.edits
    editor.buttons[0].click()
    assert editor.edits[0].enabled
    editor.remove(0)
    assert editor.buttons[0].text() == "1"
    editor.buttons[0].click()
    assert not editor.edits[0].enabled
    editor.clear_recording()
    assert not editor.edits and not editor.buttons


@pytest.mark.parametrize("lane_name", ["signal", "bpm", "states"])
@pytest.mark.parametrize("side,target", [("start", 0.3), ("end", 1.5)])
def test_selection_border_drag_resizes_all_lanes_and_playback_loop(editor, lane_name, side, target):
    ws = editor.ws
    editor.add_btn.setChecked(False)
    ws.set_region((0.5, 1.2), final=True)
    lane = ws.lanes[lane_name]
    vb, widget = lane.plot.getViewBox(), lane.widget
    view = vb.viewRange()[0][:]
    lo, hi = vb.viewRange()[1]
    y = lo + (hi - lo) * 0.35

    def pixel(t):
        return widget.mapFromScene(vb.mapViewToScene(QtCore.QPointF(t, y)))

    # Start four pixels away from the line, rather than directly on it.
    offset = QtCore.QPoint(4, 0)
    start = pixel(0.5 if side == "start" else 1.2) + offset
    end = pixel(target) + offset
    scene_pos = widget.mapToScene(start)
    ws._on_mouse_moved(scene_pos, lane)
    assert widget.viewport().cursor().shape() == QtCore.Qt.CursorShape.SizeHorCursor
    QtTest.QTest.mousePress(widget.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=start)
    event = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseMove, QtCore.QPointF(end),
                             QtCore.QPointF(widget.viewport().mapToGlobal(end)),
                             QtCore.Qt.MouseButton.NoButton, QtCore.Qt.MouseButton.LeftButton,
                             QtCore.Qt.KeyboardModifier.NoModifier)
    QtWidgets.QApplication.sendEvent(widget.viewport(), event)
    QtTest.QTest.mouseRelease(widget.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=end)
    expected = (target, 1.2) if side == "start" else (0.5, target)
    assert ws.region == pytest.approx(expected, abs=0.003)
    for other in ws.lanes.values():
        assert other.region.getRegion() == pytest.approx(ws.region)
    assert ws.player._loop == tuple(int(t * ws.player.sample_rate) for t in ws.region)
    assert vb.viewRange()[0] == view
    assert not editor.edits


def test_selection_resize_clamps_to_recording_and_prevents_crossing(editor):
    ws = editor.ws
    ws.resize_region("start", 1.2, -1, final=True)
    assert ws.region == (0, 1.2)
    ws.resize_region("end", 0.5, 5, final=True)
    assert ws.region == (0.5, 2)
    ws.resize_region("start", 1.2, 1.8, final=True)
    assert ws.region == pytest.approx((1.198, 1.2))
    ws.resize_region("end", 0.5, 0.1, final=True)
    assert ws.region == pytest.approx((0.5, 0.502))


def test_original_envelope_display_is_toggleable_without_analysis_and_drops_stale_results(editor):
    ws = editor.ws
    original = np.full(400, 0.3, np.float32)
    ws._original_envelope_ready(ws._audio_token, (0.005, original))
    assert ORIGINAL_ENVELOPE not in ws.lanes["signal"].items
    assert not ws._wanted(ORIGINAL_ENVELOPE)
    ws._on_legend_toggled(ORIGINAL_ENVELOPE, True)
    item = ws.lanes["signal"].items[ORIGINAL_ENVELOPE]
    assert item.isVisible()
    assert ws.analysis is None
    assert ws._original_envelope.t0 == 0
    ws._on_legend_toggled(ORIGINAL_ENVELOPE, False)
    assert not item.isVisible()
    ws._on_legend_toggled(ORIGINAL_ENVELOPE, True)
    assert item.isVisible()
    trace = ws._original_envelope
    ws._original_envelope_ready(ws._audio_token - 1, (0.005, np.ones(400)))
    assert ws._original_envelope is trace
    editor.add(0.5, -12)
    ws.player.toggle_source()
    np.testing.assert_array_equal(ws._original_envelope.y, original)
    ws._clear_plots()
    ws._show(ORIGINAL_ENVELOPE, True)
    assert ws.lanes["signal"].items[ORIGINAL_ENVELOPE].isVisible()
    ws._clear_plots()
    ws.set_lane_visible("signal", False)
    ws._original_envelope_ready(ws._audio_token, (0.005, original))
    assert ORIGINAL_ENVELOPE not in ws.lanes["signal"].items
    ws.set_lane_visible("signal", True)
    assert ws.lanes["signal"].items[ORIGINAL_ENVELOPE].isVisible()
