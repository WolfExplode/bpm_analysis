"""file_io: output stem normalization and companion-WAV discovery."""
import os

import file_io as fio


# --- normalize_output_filename_stem -----------------------------------------

def test_normalize_collapses_internal_whitespace():
    assert fio.normalize_output_filename_stem("a   b\tc") == "a b c"


def test_normalize_strips_leading_and_trailing_whitespace():
    assert fio.normalize_output_filename_stem("  padded  ") == "padded"


def test_normalize_passthrough_when_no_whitespace():
    # _OUTPUT_FILENAME_EMOJIS is currently () (stripping disabled) - pin the
    # no-op-emoji-strip behavior so a silent re-population doesn't go unnoticed.
    assert fio.normalize_output_filename_stem("plain_name") == "plain_name"


def test_normalize_empty_string():
    assert fio.normalize_output_filename_stem("") == ""


# --- output_stem_from_path ----------------------------------------------------

def test_output_stem_from_path_strips_dir_and_extension():
    assert fio.output_stem_from_path(os.path.join("a", "b", "song.wav")) == "song"


def test_output_stem_from_path_normalizes_whitespace():
    assert fio.output_stem_from_path(os.path.join("dir", "a   b.wav")) == "a b"


# --- find_companion_wav ---------------------------------------------------

def test_find_companion_wav_literal_match(tmp_path):
    (tmp_path / "recording.wav").write_bytes(b"")
    result = fio.find_companion_wav("recording", str(tmp_path))
    assert result == str(tmp_path / "recording.wav")


def test_find_companion_wav_case_and_whitespace_tolerant(tmp_path):
    (tmp_path / "My   Song.WAV").write_bytes(b"")
    result = fio.find_companion_wav("my song", str(tmp_path))
    assert result == str(tmp_path / "My   Song.WAV")


def test_find_companion_wav_no_match_returns_none(tmp_path):
    (tmp_path / "other.wav").write_bytes(b"")
    assert fio.find_companion_wav("recording", str(tmp_path)) is None


def test_find_companion_wav_ignores_non_wav_files(tmp_path):
    (tmp_path / "recording.mp3").write_bytes(b"")
    assert fio.find_companion_wav("recording", str(tmp_path)) is None


def test_find_companion_wav_empty_stem_returns_none(tmp_path):
    (tmp_path / "  .wav").write_bytes(b"")
    assert fio.find_companion_wav("   ", str(tmp_path)) is None


def test_find_companion_wav_skips_missing_directory(tmp_path):
    missing = str(tmp_path / "does_not_exist")
    (tmp_path / "recording.wav").write_bytes(b"")
    result = fio.find_companion_wav("recording", missing, str(tmp_path))
    assert result == str(tmp_path / "recording.wav")


def test_find_companion_wav_skips_duplicate_directories(tmp_path):
    (tmp_path / "recording.wav").write_bytes(b"")
    # Passing the same directory twice must not raise or scan it twice.
    result = fio.find_companion_wav("recording", str(tmp_path), str(tmp_path))
    assert result == str(tmp_path / "recording.wav")


def test_find_companion_wav_searches_directories_in_order(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (second / "recording.wav").write_bytes(b"")
    # Not present in `first`; must fall through to `second`.
    result = fio.find_companion_wav("recording", str(first), str(second))
    assert result == str(second / "recording.wav")


def test_find_companion_wav_ignores_falsy_directory_entries(tmp_path):
    (tmp_path / "recording.wav").write_bytes(b"")
    result = fio.find_companion_wav("recording", "", str(tmp_path))
    assert result == str(tmp_path / "recording.wav")
