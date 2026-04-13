"""
Rename analyzed audio inputs with a trailing BPM tag (shared by GUI and headless batch_cli).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Dict, Optional

# Trailing BPM tags stripped from the stem before appending a fresh annotation (end of stem only).
_BPM_FILENAME_TAIL_RE = re.compile(
    r"(?:\s+(?:\d+\s*,\s*\d+\s*-\s*\d+\s*bpm|\d+\s*to\s*\d+\s*bpm|\d+\s*bpm))+$",
    re.IGNORECASE,
)
_MAX_BASENAME_LEN = 255
_MAX_FULL_PATH_LEN = 260


def strip_trailing_bpm_filename_annotations(stem: str) -> str:
    return _BPM_FILENAME_TAIL_RE.sub("", stem).rstrip()


def format_bpm_filename_annotation(start_bpm: float, min_bpm: float, max_bpm: float) -> str:
    """start_bpm = first-in-time smoothed BPM from the final analysis pass; min/max from that same series."""
    a = int(round(start_bpm))
    lo = int(round(min_bpm))
    hi = int(round(max_bpm))
    return f"{a},{lo}-{hi}bpm"


def _warn(
    warn: Optional[Callable[[str, str], None]],
    reason: str,
    basename: str,
) -> None:
    if warn is not None:
        warn(reason, basename)
    else:
        logging.warning("BPM rename skipped for '%s': %s", basename, reason)


def try_rename_input_with_bpm_annotation(
    file_path: str,
    info: Optional[Dict[str, Any]],
    *,
    warn: Optional[Callable[[str, str], None]] = None,
) -> Optional[str]:
    """
    Rename file_path to append/replace trailing start,min-maxbpm tag.
    Returns new absolute path, or None if skipped/failed.
    """
    if not info:
        return None
    try:
        start_bpm = float(info["start_bpm"])
        min_bpm = float(info["min_bpm"])
        max_bpm = float(info["max_bpm"])
    except (KeyError, TypeError, ValueError):
        return None

    base = os.path.basename(file_path)
    stem, ext = os.path.splitext(base)
    clean_stem = strip_trailing_bpm_filename_annotations(stem)
    tag = format_bpm_filename_annotation(start_bpm, min_bpm, max_bpm)
    new_stem = f"{clean_stem} {tag}".strip() if clean_stem else tag
    new_basename = new_stem + ext
    if len(new_basename) > _MAX_BASENAME_LEN:
        _warn(
            warn,
            f"The new file name would be too long ({len(new_basename)} characters; max {_MAX_BASENAME_LEN}).",
            base,
        )
        return None

    parent = os.path.dirname(file_path) or "."
    new_path = os.path.join(parent, new_basename)
    abs_old = os.path.abspath(file_path)
    abs_new = os.path.abspath(new_path)
    if len(abs_new) > _MAX_FULL_PATH_LEN:
        _warn(
            warn,
            f"The full path would be too long ({len(abs_new)} characters; max {_MAX_FULL_PATH_LEN}).",
            base,
        )
        return None
    if os.path.normcase(abs_old) == os.path.normcase(abs_new):
        return None
    if os.path.exists(new_path):
        _warn(warn, f"A file already exists with the target name:\n{new_basename}", base)
        return None
    try:
        os.rename(file_path, new_path)
        logging.info("Renamed input file to %s", new_basename)
        return abs_new
    except OSError as e:
        _warn(warn, str(e), base)
        return None
