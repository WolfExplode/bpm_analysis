"""
Shared stdout logging setup for GUI and batch_cli.

When root is DEBUG (general console / debug mode), third-party libraries otherwise flood
the console (Numba bytecode, matplotlib, pydub/ffmpeg, Kaleido/choreographer). The GUI
has always gated those; batch_cli must use the same rules.
"""
from __future__ import annotations

import logging
import sys
from typing import List, Optional, TextIO, Tuple

# Loggers to keep at WARNING while root is DEBUG (match gui._apply_general_console_logging).
_NOISY_LOGGER_NAMES = (
    "numba",
    "llvmlite",
    "asyncio",
    "pydub",
    "pydub.utils",
    "matplotlib",
    "PIL",
    "choreographer",
    "chrome_wrapper",
    "kaleido",
    "browser_proc",
)


def make_stream_unicode_safe(stream: TextIO) -> None:
    """Stop StreamHandler from raising UnicodeEncodeError on a legacy console encoding.

    On Windows the console is often cp1252, which cannot encode the Unicode that shows up
    in log records: emoji in input file names, and arrow / minus / ≤ glyphs baked into log
    strings. Without this, each such record makes logging.emit throw and spam the console
    with "--- Logging error ---" tracebacks (the analysis itself is unaffected).

    UTF-8 with backslashreplace never raises; on a UTF-8-capable terminal (VS Code, Windows
    Terminal) it renders cleanly, and on a legacy one the rare odd glyph degrades to an
    escape instead of crashing the handler. Best-effort: silently no-op if the stream cannot
    be reconfigured (e.g. already wrapped, or not a TextIOWrapper).
    """
    try:
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass


class SuppressKaleidoChoreographerRootNoiseFilter(logging.Filter):
    """
    choreographer uses logistro.getLogger() (root) in utils/_which.py; Chrome stderr uses logger
    name browser_proc. Named loggers are raised to WARNING separately; this catches root + any
    site-packages choreographer/kaleido call site still on root.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "root":
            return True
        path = (getattr(record, "pathname", None) or "").replace("\\", "/").lower()
        if path.endswith("choreographer/utils/_which.py"):
            return record.levelno >= logging.WARNING
        if "/site-packages/" in path or "/dist-packages/" in path:
            if "/choreographer/" in path or "/kaleido/" in path:
                return record.levelno >= logging.WARNING
        return True


def configure_analysis_console_logging(
    *,
    general_debug: bool,
    quiet: bool = False,
    log_format: str = "%(asctime)s - [%(levelname)s] - %(message)s",
    stream: TextIO = sys.stdout,
    tracked_filters: Optional[List[Tuple[logging.Handler, logging.Filter]]] = None,
) -> None:
    """
    Configure root + StreamHandler levels, noisy loggers, and optional Kaleido root filter.

    If root has no handlers yet, adds a single StreamHandler with log_format.

    tracked_filters: when general_debug is True, Kaleido/choreographer filters are appended
    as (handler, filter) pairs so the GUI can remove them when toggling settings off.
    Pass None for one-shot CLI runs (filters stay on the handlers for process lifetime).
    """
    make_stream_unicode_safe(stream)
    root = logging.getLogger()
    if not root.handlers:
        h = logging.StreamHandler(stream)
        h.setFormatter(logging.Formatter(log_format))
        root.addHandler(h)

    if tracked_filters is not None:
        for h, filt in tracked_filters:
            try:
                h.removeFilter(filt)
            except Exception:
                pass
        tracked_filters.clear()

    if quiet:
        root_level = logging.WARNING
        debug_mode = False
    elif general_debug:
        root_level = logging.DEBUG
        debug_mode = True
    else:
        root_level = logging.INFO
        debug_mode = False

    root.setLevel(root_level)
    for h in root.handlers:
        try:
            h.setLevel(root_level)
        except Exception:
            pass

    for name in _NOISY_LOGGER_NAMES:
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING if debug_mode else logging.NOTSET)

    if debug_mode:
        nf = SuppressKaleidoChoreographerRootNoiseFilter()
        for h in root.handlers:
            h.addFilter(nf)
            if tracked_filters is not None:
                tracked_filters.append((h, nf))
