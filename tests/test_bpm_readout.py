"""Right-button BPM inspection follows the cursor without changing the plot view."""
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from pcg import analysis
from pcg.app import theme
from pcg.app.state import Settings
from pcg.app.workspace import Workspace


@pytest.fixture
def workspace(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.apply(app)
    settings = Settings.__new__(Settings)
    settings.q = QtCore.QSettings(str(tmp_path / "ui.ini"), QtCore.QSettings.Format.IniFormat)
    ws = Workspace(settings, analysis.Library(tmp_path / "library"))
    ws.analysis = SimpleNamespace(trace=lambda name: None)  # Readout uses plot coordinates.
    ws.resize(1500, 900)
    ws.show()
    ws.lanes["states"].plot.setXRange(0, 10, padding=0)
    ws.lanes["bpm"].plot.setYRange(50, 200, padding=0)
    app.processEvents()
    yield ws
    ws.close()
    app.processEvents()


def pixel(ws, x, y):
    lane = ws.lanes["bpm"]
    return lane.widget.mapFromScene(lane.plot.getViewBox().mapViewToScene(QtCore.QPointF(x, y)))


def move_held(viewport, pos):
    event = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseMove, QtCore.QPointF(pos),
                             QtCore.QPointF(viewport.mapToGlobal(pos)), QtCore.Qt.MouseButton.NoButton,
                             QtCore.Qt.MouseButton.RightButton, QtCore.Qt.KeyboardModifier.NoModifier)
    QtWidgets.QApplication.sendEvent(viewport, event)


def test_readout_shows_immediately_tracks_and_hides_on_release(workspace):
    ws = workspace
    viewport = ws.lanes["bpm"].widget.viewport()
    vb = ws.lanes["bpm"].plot.getViewBox()
    view = vb.viewRange()
    start, end = pixel(ws, 3, 100), pixel(ws, 6, 160)
    assert ws._bpm_readout.isHidden()
    QtTest.QTest.mousePress(viewport, QtCore.Qt.MouseButton.RightButton, pos=start)
    assert ws._bpm_readout.isVisible()
    assert float(ws._bpm_readout.text().split()[0]) == pytest.approx(100, abs=2)
    first_position = ws._bpm_readout.pos()
    move_held(viewport, end)
    assert float(ws._bpm_readout.text().split()[0]) == pytest.approx(160, abs=2)
    assert ws._bpm_readout.pos() != first_position
    assert abs(ws._bpm_readout.x() - end.x()) < ws._bpm_readout.width() + 20
    assert vb.viewRange() == view
    QtTest.QTest.mouseRelease(viewport, QtCore.Qt.MouseButton.RightButton, pos=end)
    assert ws._bpm_readout.isHidden()
    assert not ws._bpm_held


def test_readout_fits_near_plot_edge_and_hides_outside(workspace):
    ws = workspace
    viewport = ws.lanes["bpm"].widget.viewport()
    pos = pixel(ws, 9.9, 55)
    QtTest.QTest.mousePress(viewport, QtCore.Qt.MouseButton.RightButton, pos=pos)
    assert ws._bpm_readout.isVisible()
    assert viewport.rect().contains(ws._bpm_readout.geometry())
    move_held(viewport, QtCore.QPoint(-10, -10))
    assert ws._bpm_readout.isHidden()
    move_held(viewport, pos)
    assert ws._bpm_readout.isVisible()
    QtWidgets.QApplication.sendEvent(ws, QtCore.QEvent(QtCore.QEvent.Type.ApplicationDeactivate))
    assert ws._bpm_readout.isHidden()
    assert not ws._bpm_held


def test_right_press_without_recording_does_not_show_readout(workspace):
    ws = workspace
    ws.analysis = None
    viewport = ws.lanes["bpm"].widget.viewport()
    pos = pixel(ws, 3, 100)
    QtTest.QTest.mouseClick(viewport, QtCore.Qt.MouseButton.RightButton, pos=pos)
    assert ws._bpm_readout.isHidden()
    assert not ws._bpm_held
