"""Pure filename-tag helpers: BPM tag formatting + trailing-tag stripping.

These guard the rename round-trip (strip -> format -> strip is idempotent) and,
just as important, that stripping never eats a legitimate filename that merely
contains digits or the substring "bpm".
"""
from bpm_input_rename import (
    format_bpm_filename_annotation,
    strip_trailing_bpm_filename_annotations,
)


def test_format_rounds_to_int_and_uses_compact_form():
    assert format_bpm_filename_annotation(99.6, 80.4, 205.5) == "[100,80-206bpm]"


def test_strip_removes_current_bracketed_tag():
    assert strip_trailing_bpm_filename_annotations("recording [100,120-206bpm]") == "recording"


def test_strip_removes_legacy_tag_forms():
    # Older/looser shapes the regex still has to clean before re-tagging.
    assert strip_trailing_bpm_filename_annotations("rec 120bpm") == "rec"
    assert strip_trailing_bpm_filename_annotations("rec [120bpm]") == "rec"
    assert strip_trailing_bpm_filename_annotations("rec 120to206bpm") == "rec"
    assert strip_trailing_bpm_filename_annotations("rec 120 bpm") == "rec"


def test_strip_removes_repeated_stacked_tags():
    # Re-running rename on an already-double-tagged stem must collapse to the base.
    assert (
        strip_trailing_bpm_filename_annotations("rec [100,120-206bpm] [90,80-150bpm]")
        == "rec"
    )


def test_strip_preserves_legitimate_names():
    # Digits and embedded "bpm" that are NOT a trailing tag must survive untouched.
    for name in ("recording", "my 4k video", "128bpm song title", "song 174bpm remix"):
        assert strip_trailing_bpm_filename_annotations(name) == name


def test_strip_is_idempotent_after_a_rename():
    # strip(format(...) appended) recovers the original base, and re-stripping is a no-op.
    tag = format_bpm_filename_annotation(100, 120, 206)
    for base in ("recording", "rec [120bpm]", "my 4k video", "128bpm song"):
        clean = strip_trailing_bpm_filename_annotations(base)
        renamed = f"{clean} {tag}".strip() if clean else tag
        assert strip_trailing_bpm_filename_annotations(renamed) == clean
        assert strip_trailing_bpm_filename_annotations(clean) == clean
