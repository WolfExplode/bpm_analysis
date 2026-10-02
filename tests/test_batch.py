"""pcg.batch: filename BPM tags, BPM rename, input collection, run settings."""
import os

import numpy as np
import soundfile as sf

from pcg import analysis, annotation, batch

# --- start BPM from the filename ------------------------------------------------

def test_start_bpm_patterns():
    assert batch.start_bpm_from_filename("clip [120,60-150bpm].wav") == 120.0
    assert batch.start_bpm_from_filename("clip 90to132bpm.wav") == 90.0
    assert batch.start_bpm_from_filename("clip 150bpm.wav") == 150.0
    assert batch.start_bpm_from_filename("clip 150BPM.wav") == 150.0
    assert batch.start_bpm_from_filename("clip.wav") is None


def test_start_bpm_rightmost_and_priority():
    assert batch.start_bpm_from_filename("a [100,50-140bpm] b [200,80-220bpm].wav") == 200.0
    assert batch.start_bpm_from_filename("clip [120,60-150bpm] extra 999bpm.wav") == 120.0
    assert batch.start_bpm_from_filename("clip 90to132bpm extra 999bpm.wav") == 90.0


def test_start_bpm_uses_basename_only():
    assert batch.start_bpm_from_filename(os.path.join("150bpm_folder", "clip.wav")) is None


# --- tags ------------------------------------------------------------------------

def test_format_rounds_to_int():
    assert batch.format_bpm_tag(99.6, 80.4, 205.5) == "[100,80-206bpm]"


def test_strip_removes_current_and_legacy_tags():
    for name, clean in (("recording [100,120-206bpm]", "recording"), ("rec 120bpm", "rec"), ("rec [120bpm]", "rec"),
                        ("rec 120to206bpm", "rec"), ("rec 120 bpm", "rec"),
                        ("rec [100,120-206bpm] [90,80-150bpm]", "rec")):
        assert batch.strip_bpm_tags(name) == clean


def test_strip_preserves_legitimate_names():
    for name in ("recording", "my 4k video", "128bpm song title", "song 174bpm remix"):
        assert batch.strip_bpm_tags(name) == name


def test_rename_with_bpm_moves_annotation_sidecar(tmp_path):
    rec = tmp_path / "rec [90,80-100bpm].wav"
    sf.write(str(rec), np.zeros(100, dtype=np.int16), 4000, subtype="PCM_16")
    annotation.save(annotation.Annotation("fp", 1.0, rec.name, ()), annotation.sidecar_path(rec))
    new, why = batch.rename_with_bpm(rec, {"start_bpm": 101.4, "min_bpm": 70, "max_bpm": 140.6})
    assert why == "" and new.name == "rec [101,70-141bpm].wav" and new.exists()
    assert annotation.sidecar_path(new).exists() and not annotation.sidecar_path(rec).exists()
    assert batch.rename_with_bpm(new, {"start_bpm": 101, "min_bpm": 70, "max_bpm": 141}) == (None, "already tagged")
    assert batch.rename_with_bpm(new, None) == (None, "no BPM result")


# --- inputs ------------------------------------------------------------------------

def test_collect_prefers_wav_and_recurses(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ("a.mp3", "a.wav", "b.flac", "sub/c.wav", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    got = sorted(p.relative_to(tmp_path).as_posix() for p in batch.collect([tmp_path]))
    assert got == ["a.wav", "b.flac", "sub/c.wav"]


def test_effective_jobs_serialises_springer():
    assert batch.RunSettings(jobs=8).effective_jobs == 8
    assert batch.RunSettings(jobs=8, springer=True).effective_jobs == 1
    assert batch.RunSettings(jobs=8, auto_switch=True).effective_jobs == 1


def test_annotated_recordings_follows_renames(tmp_path):
    lib = analysis.Library(tmp_path / "lib")
    rec = tmp_path / "renamed [80,70-90bpm].wav"
    sf.write(str(rec), np.arange(200, dtype=np.int16), 4000, subtype="PCM_16")
    annotation.save(annotation.Annotation(lib.fingerprints.get(rec), 0.05, "old.wav", ()),
                    tmp_path / "old.annotation.json")
    assert batch.annotated_recordings([tmp_path], lib) == [(rec, tmp_path / "old.annotation.json")]
