"""
Rename analyzed audio inputs with a trailing BPM tag (shared by GUI and headless batch_cli).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Dict, Optional
from urllib.parse import quote

from config import output_stem_from_path, strip_output_filename_emojis

# Suffixes of analysis artifacts under output_directory that use output_stem_from_path(input).
# Used to rename outputs when the input file is BPM-tagged (single-channel only); order matters for HTML patching.
_OUTPUT_STEM_SUFFIXES: tuple[str, ...] = (
    "_pass1.html",
    "_pass2.html",
    "_pass2.png",
    "_pass2.csv",
    "_pass3.html",
    "_pass3.png",
    "_pass3.csv",
    "_bpm_plot.html",
    "_bpm_plot.png",
    "_bpm_plot.csv",
    "_spectrogram.png",
    "_filtered_spectrogram.png",
    "_filtered_inverse_spectrogram.png",
    "_fft_profiles.html",
    "_Analysis_Summary.md",
    "_Debug_Log.md",
    "_filtered_debug.wav",
    "_filtered_inverse_debug.wav",
)

# Trailing BPM tags stripped from the stem before appending a fresh annotation (end of stem only).
_BPM_FILENAME_TAIL_RE = re.compile(
    r"(?:\s+(?:\d+\s*,\s*\d+\s*-\s*\d+\s*bpm|\d+\s*to\s*\d+\s*bpm|\d+\s*bpm))+$",
    re.IGNORECASE,
)
_MAX_BASENAME_LEN = 255
_MAX_FULL_PATH_LEN = 260


def strip_trailing_bpm_filename_annotations(stem: str) -> str:
    return _BPM_FILENAME_TAIL_RE.sub("", stem).rstrip()


def _output_audio_basename_for_html(input_basename: str) -> str:
    """Match plotting._generate_custom_html: emoji-strip stem, keep extension (copied WAV name in output dir)."""
    stem, ext = os.path.splitext(input_basename)
    return strip_output_filename_emojis(stem) + ext


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


def _patch_html_output_references(
    file_path: str,
    old_stem: str,
    new_stem: str,
    old_basename: str,
    new_basename: str,
) -> None:
    """Update embedded artifact names after on-disk renames (audio, spectrograms, etc.)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return
    # Stem-prefixed outputs first so paths like name_spectrogram.png track the new stem.
    updated = content.replace(old_stem + "_", new_stem + "_")
    updated = updated.replace(old_basename, new_basename)
    # Copied working WAV in the output folder uses stem.wav (even when input was e.g. .mp3).
    updated = updated.replace(old_stem + ".wav", new_stem + ".wav")
    # BPM_ANALYZER_CONFIG stores audioSources / filtered paths with urllib.parse.quote(...) — plain
    # replace() misses those; the on-disk WAV/HTML renames succeed but the <audio> URL stayed stale.
    old_audio_bn = _output_audio_basename_for_html(old_basename)
    new_audio_bn = _output_audio_basename_for_html(new_basename)
    updated = updated.replace(quote(old_audio_bn), quote(new_audio_bn))
    updated = updated.replace(
        quote(old_stem + "_filtered_debug.wav"),
        quote(new_stem + "_filtered_debug.wav"),
    )
    updated = updated.replace(
        quote(old_stem + "_filtered_inverse_debug.wav"),
        quote(new_stem + "_filtered_inverse_debug.wav"),
    )
    if updated != content:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(updated)
            logging.info("Updated embedded file references in %s", os.path.basename(file_path))
        except OSError as e:
            logging.warning("Could not update HTML references in %s: %s", file_path, e)


def rename_analysis_outputs_after_input_bpm_rename(
    old_input_path: str,
    new_input_path: str,
    output_directory: str,
) -> None:
    """
    After try_rename_input_with_bpm_annotation, rename outputs that used the old input stem
    so they match the new file name (e.g. name_bpm_plot.png -> name 100,120-206bpm_bpm_plot.png).
    """
    if not output_directory or not os.path.isdir(output_directory):
        return
    old_stem = output_stem_from_path(old_input_path)
    new_stem = output_stem_from_path(new_input_path)
    old_basename = os.path.basename(old_input_path)
    new_basename = os.path.basename(new_input_path)
    if old_stem == new_stem and old_basename == new_basename:
        return

    for suffix in _OUTPUT_STEM_SUFFIXES:
        old_path = os.path.join(output_directory, old_stem + suffix)
        new_path = os.path.join(output_directory, new_stem + suffix)
        if not os.path.isfile(old_path):
            continue
        if os.path.exists(new_path):
            logging.warning(
                "Skipping output rename (target exists): %s",
                os.path.basename(new_path),
            )
            continue
        try:
            os.rename(old_path, new_path)
            logging.info(
                "Renamed output %s -> %s",
                os.path.basename(old_path),
                os.path.basename(new_path),
            )
        except OSError as e:
            logging.warning("Could not rename output %s: %s", old_path, e)
            continue

    # Working / copied WAV in the output folder (same stem as original for typical .wav inputs).
    old_plain_wav = os.path.join(output_directory, old_stem + ".wav")
    new_plain_wav = os.path.join(output_directory, new_stem + ".wav")
    if os.path.isfile(old_plain_wav) and not os.path.exists(new_plain_wav):
        try:
            os.rename(old_plain_wav, new_plain_wav)
            logging.info(
                "Renamed output %s -> %s",
                os.path.basename(old_plain_wav),
                os.path.basename(new_plain_wav),
            )
        except OSError as e:
            logging.warning("Could not rename %s: %s", old_plain_wav, e)
    elif os.path.isfile(old_plain_wav) and os.path.exists(new_plain_wav):
        logging.warning(
            "Skipping WAV rename in output dir (target exists): %s",
            os.path.basename(new_plain_wav),
        )

    for suffix in _OUTPUT_STEM_SUFFIXES:
        if not suffix.endswith(".html"):
            continue
        html_path = os.path.join(output_directory, new_stem + suffix)
        if os.path.isfile(html_path):
            _patch_html_output_references(html_path, old_stem, new_stem, old_basename, new_basename)
