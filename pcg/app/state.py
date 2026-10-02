"""App state that isn't drawing: the open Annotation (undo/redo, saving), settings, the worker."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, List, Optional

from PySide6 import QtCore

from pcg import annotation as an

REPO_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

class Settings:
    """Per-user settings (QSettings): trace/lane/stage visibility, protected folders, run settings."""

    def __init__(self) -> None:
        self.q = QtCore.QSettings("pcg", "workspace")

    def get_json(self, key: str, default):
        raw = self.q.value(key)
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return default

    def set_json(self, key: str, value) -> None:
        self.q.setValue(key, json.dumps(value))

    @property
    def protected_folders(self) -> List[str]:
        return self.get_json("protected_folders", [str(REPO_ROOT / "inputs")])

    @protected_folders.setter
    def protected_folders(self, folders: List[str]) -> None:
        self.set_json("protected_folders", folders)

    def trace_visibility(self) -> dict:
        return self.get_json("trace_visibility", {})

    def set_trace_visibility(self, vis: dict) -> None:
        self.set_json("trace_visibility", vis)

    def y_ranges(self, fingerprint: str) -> dict:
        """Manual y-ranges {lane: [lo, hi]} the user set for one Recording."""
        return self.get_json("y_ranges", {}).get(fingerprint, {})

    def set_y_ranges(self, fingerprint: str, ranges: dict) -> None:
        every = self.get_json("y_ranges", {})
        if ranges:
            every[fingerprint] = ranges
        else:
            every.pop(fingerprint, None)
        self.set_json("y_ranges", every)

    def lane_heights(self) -> dict:
        return self.get_json("lane_heights", {})

    def set_lane_heights(self, heights: dict) -> None:
        self.set_json("lane_heights", heights)

    def stages(self) -> dict:
        """Debug stages {group: on} whose traces the lane legends offer."""
        return self.get_json("stages", {})

    def set_stages(self, stages: dict) -> None:
        self.set_json("stages", stages)

    def lane_order(self) -> list:
        return self.get_json("lane_order", [])

    def set_lane_order(self, order: list) -> None:
        self.set_json("lane_order", order)

    def lane_visibility(self) -> dict:
        return self.get_json("lane_visibility", {})

    def set_lane_visibility(self, vis: dict) -> None:
        self.set_json("lane_visibility", vis)


def downloads_dir() -> Path:
    d = Path.home() / "Downloads"
    return d if d.is_dir() else Path.home()


# ─────────────────────────────────────────────────────────────────────────────
# Annotation document
# ─────────────────────────────────────────────────────────────────────────────

class AnnotationDoc(QtCore.QObject):
    """The Annotation being edited: spans + undo/redo + where it saves.

    A Recording without an Annotation gets a draft seeded from the Analysis; it
    lives in memory until the first save.
    """

    changed = QtCore.Signal()

    def __init__(self, ann: an.Annotation, path: Optional[Path], *, draft: bool):
        super().__init__()
        self.ann = ann
        self.path = path
        self.draft = draft
        self._undo: List[an.Spans] = []
        self._redo: List[an.Spans] = []
        # A draft is 'clean' until edited; its title still says [draft] until the first save.
        self._saved_spans: an.Spans = ann.spans

    @property
    def spans(self) -> an.Spans:
        return self.ann.spans

    @property
    def dirty(self) -> bool:
        return self.ann.spans != self._saved_spans

    def apply(self, edit: Callable[[an.Spans], an.Spans]) -> bool:
        new = edit(self.ann.spans)
        if new == self.ann.spans:
            return False
        self._undo.append(self.ann.spans)
        self._redo.clear()
        self.ann = self.ann.with_spans(new)
        self.changed.emit()
        return True

    def undo(self) -> None:
        if self._undo:
            self._redo.append(self.ann.spans)
            self.ann = self.ann.with_spans(self._undo.pop())
            self.changed.emit()

    def redo(self) -> None:
        if self._redo:
            self._undo.append(self.ann.spans)
            self.ann = self.ann.with_spans(self._redo.pop())
            self.changed.emit()

    def save_to(self, path: Path) -> None:
        an.save(self.ann, path)
        self.path = path
        self.draft = False
        self._saved_spans = self.ann.spans
        self.changed.emit()


# ─────────────────────────────────────────────────────────────────────────────
# Worker: the engine in a fresh process (picks up code edits)
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeWorker(QtCore.QObject):
    """Runs `python -m pcg analyze --progress-json` for one recording in a fresh process."""

    progress = QtCore.Signal(str)
    finished = QtCore.Signal(list, str)  # analysis paths, error

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.proc: Optional[QtCore.QProcess] = None
        self._paths: List[str] = []
        self._error = ""
        self._buf = b""

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.state() != QtCore.QProcess.ProcessState.NotRunning

    def start(self, recording_path: str, library: str, *, springer: bool, auto_switch: bool, channel: str,
              start_sec: float, bpm_hint: Optional[float]) -> None:
        if self.running:
            return
        args = ["-m", "pcg", "--library", library, "analyze", recording_path, "--progress-json",
                "--channel", channel, "--start", str(start_sec)]
        if springer:
            args.append("--springer")
        if auto_switch:
            args.append("--auto-switch")
        if bpm_hint is not None:
            args += ["--bpm", str(bpm_hint)]
        self._paths, self._error, self._buf = [], "", b""
        self.proc = QtCore.QProcess(self)
        self.proc.setWorkingDirectory(str(REPO_ROOT))
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)
        self.proc.start(sys.executable, args)

    def cancel(self) -> None:
        if self.running:
            self._error = "cancelled"
            self.proc.kill()

    def _read(self) -> None:
        self._buf += bytes(self.proc.readAllStandardOutput())
        *lines, self._buf = self._buf.split(b"\n")
        for raw in lines:
            try:
                msg = json.loads(raw.decode("utf-8", "replace"))
            except ValueError:
                continue
            if "progress" in msg:
                self.progress.emit(msg["progress"])
            if "done" in msg:
                self._paths = msg["done"] or []
                self._error = msg.get("error") or self._error

    def _done(self, code: int, _status) -> None:
        self._read()
        err = self._error
        if not self._paths and not err:
            tail = bytes(self.proc.readAllStandardError()).decode("utf-8", "replace").strip().splitlines()
            err = tail[-1] if tail else f"worker exited with code {code}"
        self.finished.emit(self._paths, err)
