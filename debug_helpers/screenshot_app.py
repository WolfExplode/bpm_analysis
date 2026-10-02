"""Render the app offscreen and save PNGs of the Run screen and the workspace (for UI review).

    python debug_helpers/screenshot_app.py RECORDING OUT_DIR [--from S --to S] [--stages "Pass 2,Pass 3"]

Uses a throwaway QSettings scope, so the user's saved lanes/traces/y-ranges are untouched.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")  # the offscreen platform has no font database
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6 import QtCore, QtWidgets  # noqa: E402

from pcg import analysis as A  # noqa: E402
from pcg.app import state, theme  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("out_dir")
    ap.add_argument("--from", dest="t0", type=float, default=30.0)
    ap.add_argument("--to", dest="t1", type=float, default=150.0)
    ap.add_argument("--stages", default="")
    ap.add_argument("--side", default="", choices=["", "issues", "summary"])
    ap.add_argument("--lanes", default="", help="extra lanes to show, comma separated")
    args = ap.parse_args()

    QtCore.QCoreApplication.setOrganizationName("pcg-screenshot")
    real_init = state.Settings.__init__

    def scratch_settings(self) -> None:
        real_init(self)
        self.q = QtCore.QSettings("pcg-screenshot", "workspace")

    state.Settings.__init__ = scratch_settings
    app = QtWidgets.QApplication(sys.argv[:1])
    theme.apply(app)
    from pcg.app.main import MainWindow

    s = state.Settings()
    s.q.clear()
    s.set_stages({st: True for st in args.stages.split(",") if st})
    s.set_json("side_panel", args.side)
    s.set_lane_visibility({n: True for n in args.lanes.split(",") if n})
    win = MainWindow(A.Library(None))
    win.resize(1900, 1060)
    win.show()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    app.processEvents()
    win.grab().save(str(out / "run.png"))
    win.open_in_workspace(args.recording, "")
    ws = win.workspace
    for _ in range(20):
        app.processEvents()
    ws.lanes["states"].plot.setXRange(args.t0, args.t1, padding=0)
    for _ in range(20):
        app.processEvents()
    win.grab().save(str(out / "workspace.png"))
    print("saved to", out)


if __name__ == "__main__":
    main()
