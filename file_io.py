"""
Output-path naming and companion audio file discovery (emoji/whitespace tolerant).
"""
from __future__ import annotations

import os
import re
from typing import Optional

_FILENAME_WS_RE = re.compile(r"\s+")

# Emojis stripped from stems used for generated outputs (input files on disk are unchanged).
_OUTPUT_FILENAME_EMOJIS = ("⭐", "🌟", "💦")


def strip_output_filename_emojis(stem: str) -> str:
    """Remove configured emojis from a file name stem."""
    for ch in _OUTPUT_FILENAME_EMOJIS:
        stem = stem.replace(ch, "")
    return stem


def normalize_output_filename_stem(stem: str) -> str:
    """Emoji-strip and collapse whitespace for output names and companion-file matching."""
    return _FILENAME_WS_RE.sub(" ", strip_output_filename_emojis(stem)).strip()


def output_stem_from_path(file_path: str) -> str:
    """Normalized stem for analysis artifacts derived from an input path."""
    return normalize_output_filename_stem(os.path.basename(os.path.splitext(file_path)[0]))


def find_companion_wav(source_stem: str, *directories: str) -> Optional[str]:
    """
    Find an existing .wav whose normalized stem matches source_stem (emoji/space tolerant).
    Checks literal and normalized names, then scans each directory.
    """
    target = normalize_output_filename_stem(source_stem).casefold()
    if not target:
        return None
    seen_dirs: set[str] = set()
    for directory in directories:
        if not directory:
            continue
        dir_abs = os.path.abspath(directory)
        if dir_abs in seen_dirs or not os.path.isdir(dir_abs):
            continue
        seen_dirs.add(dir_abs)
        for name in (source_stem + ".wav", normalize_output_filename_stem(source_stem) + ".wav"):
            path = os.path.join(dir_abs, name)
            if os.path.isfile(path):
                return path
        try:
            for name in os.listdir(dir_abs):
                if not name.lower().endswith(".wav"):
                    continue
                file_stem = os.path.splitext(name)[0]
                if normalize_output_filename_stem(file_stem).casefold() == target:
                    return os.path.join(dir_abs, name)
        except OSError:
            continue
    return None
