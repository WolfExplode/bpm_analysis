"""batch_runner: filename BPM parsing, input dedupe, and working-WAV resolution."""
import os

import pytest

import batch_runner as br


# --- extract_start_bpm_from_filename -------------------------------------------

def test_extract_bpm_comma_range_pattern():
    assert br.extract_start_bpm_from_filename("clip [120,60-150bpm].wav") == 120.0


def test_extract_bpm_to_range_pattern():
    assert br.extract_start_bpm_from_filename("clip 90to132bpm.wav") == 90.0


def test_extract_bpm_simple_pattern():
    assert br.extract_start_bpm_from_filename("clip 150bpm.wav") == 150.0


def test_extract_bpm_case_insensitive():
    assert br.extract_start_bpm_from_filename("clip 150BPM.wav") == 150.0


def test_extract_bpm_no_match_returns_none():
    assert br.extract_start_bpm_from_filename("clip.wav") is None


def test_extract_bpm_rightmost_comma_match_wins():
    name = "a [100,50-140bpm] b [200,80-220bpm].wav"
    assert br.extract_start_bpm_from_filename(name) == 200.0


def test_extract_bpm_comma_pattern_takes_priority_over_simple():
    # Both a comma-range tag and a bare "bpm" number are present; comma-range wins.
    name = "clip [120,60-150bpm] extra 999bpm.wav"
    assert br.extract_start_bpm_from_filename(name) == 120.0


def test_extract_bpm_to_pattern_priority_over_simple():
    name = "clip 90to132bpm extra 999bpm.wav"
    assert br.extract_start_bpm_from_filename(name) == 90.0


def test_extract_bpm_uses_basename_only():
    path = os.path.join("150bpm_folder", "clip.wav")
    assert br.extract_start_bpm_from_filename(path) is None


# --- dedupe_input_files -----------------------------------------------------

def test_dedupe_prefers_wav_over_lossy_regardless_of_order():
    paths = ["a.mp3", "a.wav"]
    result = br.dedupe_input_files(paths)
    assert result == ["a.wav"]


def test_dedupe_prefers_wav_when_wav_listed_first():
    paths = ["a.wav", "a.mp3"]
    result = br.dedupe_input_files(paths)
    assert result == ["a.wav"]


def test_dedupe_keeps_unrelated_files():
    paths = ["a.wav", "b.wav"]
    assert br.dedupe_input_files(paths) == ["a.wav", "b.wav"]


def test_dedupe_is_case_and_whitespace_tolerant():
    paths = ["My Song.mp3", "my   song.wav"]
    result = br.dedupe_input_files(paths)
    assert result == ["my   song.wav"]


def test_dedupe_equal_score_keeps_first_seen():
    paths = ["a.mp3", "a.flac"]  # both score 1
    result = br.dedupe_input_files(paths)
    assert result == ["a.mp3"]


def test_dedupe_unknown_extension_scores_zero_and_loses_to_known():
    paths = ["a.xyz", "a.wav"]
    result = br.dedupe_input_files(paths)
    assert result == ["a.wav"]


# --- resolve_working_wav -----------------------------------------------------

def test_resolve_working_wav_wav_input_in_output_dir_returned_as_is(tmp_path):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"")
    result = br.resolve_working_wav(str(wav), str(tmp_path), str(tmp_path), {})
    assert result == str(wav)


def test_resolve_working_wav_wav_input_outside_output_dir_is_copied(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    wav = input_dir / "clip.wav"
    wav.write_bytes(b"hello")
    result = br.resolve_working_wav(str(wav), str(output_dir), str(output_dir), {})
    assert result == str(output_dir / "clip.wav")
    assert (output_dir / "clip.wav").read_bytes() == b"hello"


def test_resolve_working_wav_non_wav_reuses_companion_in_source_dir(tmp_path, monkeypatch):
    source_dir = tmp_path / "src"
    output_dir = tmp_path / "out"
    source_dir.mkdir()
    output_dir.mkdir()
    companion = source_dir / "clip.wav"
    companion.write_bytes(b"existing")
    non_wav = source_dir / "clip.mp3"
    non_wav.write_bytes(b"")

    def _fail_convert(*a, **kw):
        raise AssertionError("convert_to_wav should not be called when a companion WAV exists")

    monkeypatch.setattr(br, "convert_to_wav", _fail_convert)
    result = br.resolve_working_wav(
        str(non_wav), str(output_dir), str(output_dir), {"working_wav_in_output": False}
    )
    assert result == str(companion)


def test_resolve_working_wav_non_wav_companion_copied_when_working_wav_in_output(tmp_path, monkeypatch):
    source_dir = tmp_path / "src"
    output_dir = tmp_path / "out"
    source_dir.mkdir()
    output_dir.mkdir()
    companion = source_dir / "clip.wav"
    companion.write_bytes(b"existing")
    non_wav = source_dir / "clip.mp3"
    non_wav.write_bytes(b"")

    monkeypatch.setattr(br, "convert_to_wav", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not convert")))
    result = br.resolve_working_wav(
        str(non_wav), str(output_dir), str(output_dir), {"working_wav_in_output": True}
    )
    assert result == str(output_dir / "clip.wav")
    assert (output_dir / "clip.wav").read_bytes() == b"existing"


def test_resolve_working_wav_non_wav_reuses_existing_wav_in_working_dir(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    non_wav = tmp_path / "clip.mp3"
    non_wav.write_bytes(b"")
    existing_candidate = output_dir / "clip.wav"
    existing_candidate.write_bytes(b"already converted")

    monkeypatch.setattr(br, "convert_to_wav", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not convert")))
    result = br.resolve_working_wav(str(non_wav), str(output_dir), str(output_dir), {})
    assert result == str(existing_candidate)
    assert existing_candidate.read_bytes() == b"already converted"  # reused, not overwritten


def test_resolve_working_wav_non_wav_no_companion_converts(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    non_wav = tmp_path / "clip.mp3"
    non_wav.write_bytes(b"")

    calls = []

    def _fake_convert(src, dst):
        calls.append((src, dst))
        with open(dst, "wb") as f:
            f.write(b"converted")
        return True

    monkeypatch.setattr(br, "convert_to_wav", _fake_convert)
    result = br.resolve_working_wav(str(non_wav), str(output_dir), str(output_dir), {})
    assert result == str(output_dir / "clip.wav")
    assert calls == [(str(non_wav), str(output_dir / "clip.wav"))]


def test_resolve_working_wav_conversion_failure_raises(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    non_wav = tmp_path / "clip.mp3"
    non_wav.write_bytes(b"")

    monkeypatch.setattr(br, "convert_to_wav", lambda *a, **kw: False)
    with pytest.raises(RuntimeError):
        br.resolve_working_wav(str(non_wav), str(output_dir), str(output_dir), {})
