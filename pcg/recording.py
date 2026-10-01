"""Recordings: identity by audio fingerprint, audio loading, channels, playback audio.

A Recording is identified by a hash of its decoded audio samples (ADR 0004): renames
and metadata edits never change it, re-encoding/resampling/trimming does. Nothing
is ever written into the audio file.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import soundfile as sf

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aiff", ".aif", ".mp4", ".mkv", ".mov")

CHANNEL_MIXED = "mixed"
CHANNEL_LEFT = "left"
CHANNEL_RIGHT = "right"
CHANNEL_ALL = "all"  # one Analysis per channel (left and right)
CHANNEL_MODES = (CHANNEL_MIXED, CHANNEL_LEFT, CHANNEL_RIGHT, CHANNEL_ALL)

_FINGERPRINT_VERSION = "a1"
_BLOCK_FRAMES = 1 << 20


class AudioError(RuntimeError):
    pass


@dataclass
class Audio:
    """Decoded audio: float32 samples shaped (frames, channels)."""
    samples: np.ndarray
    sample_rate: int

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_sec(self) -> float:
        return self.samples.shape[0] / float(self.sample_rate)

    def channel(self, channel: str) -> np.ndarray:
        """Mono float32 signal for a channel mode (mixed = mean of channels)."""
        if channel == CHANNEL_MIXED or self.channels == 1:
            return self.samples.mean(axis=1, dtype=np.float32) if self.channels > 1 else self.samples[:, 0]
        if self.channels > 2:
            raise AudioError(f"{self.channels}-channel audio: left/right analysis supports stereo only")
        return self.samples[:, 0 if channel == CHANNEL_LEFT else 1]


def is_audio_file(path: os.PathLike | str) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXTENSIONS


def _soundfile_dtype(info) -> str:
    sub = (info.subtype or "").upper()
    if sub.startswith("PCM_") or sub in ("ULAW", "ALAW"):
        return "int16" if sub in ("PCM_S8", "PCM_U8", "PCM_16", "ULAW", "ALAW") else "int32"
    return "float64" if sub == "DOUBLE" else "float32"


def _pydub_segment(path: Path):
    try:
        from pydub import AudioSegment
    except ImportError as e:  # pragma: no cover - environment dependent
        raise AudioError(f"cannot decode {path.name}: soundfile failed and pydub is not installed") from e
    try:
        return AudioSegment.from_file(str(path))
    except Exception as e:
        raise AudioError(f"cannot decode {path.name}: {e}") from e


def _decoded_blocks(path: Path) -> Tuple[int, int, Iterator[np.ndarray]]:
    """(sample_rate, channels, native-dtype sample blocks) — the canonical decode."""
    try:
        info = sf.info(str(path))
        dtype = _soundfile_dtype(info)
        blocks = sf.blocks(str(path), blocksize=_BLOCK_FRAMES, dtype=dtype, always_2d=True)
        return int(info.samplerate), int(info.channels), blocks
    except (sf.LibsndfileError, RuntimeError):
        seg = _pydub_segment(path)
        width = {1: np.int8, 2: np.int16, 4: np.int32}.get(seg.sample_width)
        if width is None:
            raise AudioError(f"{path.name}: unsupported sample width {seg.sample_width}")
        arr = np.frombuffer(seg.raw_data, dtype=width).reshape(-1, seg.channels)
        return int(seg.frame_rate), int(seg.channels), iter([arr])


def fingerprint(path: os.PathLike | str) -> str:
    """Hash of the decoded samples (plus rate and channel count)."""
    sr, ch, blocks = _decoded_blocks(Path(path))
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{_FINGERPRINT_VERSION}:{sr}:{ch}:".encode())
    for block in blocks:
        h.update(np.ascontiguousarray(block).tobytes())
    return h.hexdigest()


class FingerprintCache:
    """Remembers fingerprints by (path, size, mtime) so reopening a file is instant."""

    def __init__(self, cache_file: Path):
        self.cache_file = Path(cache_file)
        self._entries: Dict[str, Dict] = {}
        try:
            self._entries = json.loads(self.cache_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass

    def get(self, path: os.PathLike | str) -> str:
        p = Path(path).resolve()
        st = p.stat()
        key = os.path.normcase(str(p))
        hit = self._entries.get(key)
        if hit and hit.get("size") == st.st_size and hit.get("mtime_ns") == st.st_mtime_ns:
            return hit["fingerprint"]
        fp = fingerprint(p)
        self._entries[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "fingerprint": fp}
        self._save()
        return fp

    def _save(self) -> None:
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._entries, indent=0), encoding="utf-8")
            os.replace(tmp, self.cache_file)
        except OSError:
            pass


def load(path: os.PathLike | str) -> Audio:
    """Decode the whole file to float32 (frames, channels)."""
    path = Path(path)
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return Audio(np.ascontiguousarray(data), int(sr))
    except (sf.LibsndfileError, RuntimeError):
        seg = _pydub_segment(path)
        scale = float(1 << (8 * seg.sample_width - 1))
        width = {1: np.int8, 2: np.int16, 4: np.int32}[seg.sample_width]
        arr = np.frombuffer(seg.raw_data, dtype=width).reshape(-1, seg.channels).astype(np.float32) / scale
        return Audio(arr, int(seg.frame_rate))


def engine_inputs(path: os.PathLike | str, channel_mode: str, workdir: os.PathLike | str) -> List[Tuple[str, str]]:
    """[(channel, wav_path)] for the engine, which reads WAV.

    Mixed WAVs are passed through untouched; anything else (other containers,
    single channels) is decoded and written as a float WAV into `workdir`.
    """
    if channel_mode not in CHANNEL_MODES:
        raise ValueError(f"channel mode {channel_mode!r}; expected one of {CHANNEL_MODES}")
    path = Path(path)
    if channel_mode == CHANNEL_MIXED and path.suffix.lower() == ".wav":
        return [(CHANNEL_MIXED, str(path))]
    audio = load(path)
    if audio.channels == 1 or channel_mode == CHANNEL_MIXED:
        wanted = [CHANNEL_MIXED]
    elif channel_mode == CHANNEL_ALL:
        wanted = [CHANNEL_LEFT, CHANNEL_RIGHT]
    else:
        wanted = [channel_mode]
    out = []
    for ch in wanted:
        target = Path(workdir) / f"input-{ch}.wav"
        sf.write(str(target), audio.channel(ch), audio.sample_rate, subtype="FLOAT")
        out.append((ch, str(target)))
    return out


def temp_workdir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="pcg-")


def bandpassed(signal: np.ndarray, sample_rate: int, params: Dict) -> np.ndarray:
    """The engine's bandpass filter applied at the playback rate, peak-normalised (for listening)."""
    from pcg.engine.audio_preprocessing import apply_bandpass_only

    filtered = apply_bandpass_only(np.asarray(signal, dtype=np.float64), int(sample_rate), params)
    peak = float(np.max(np.abs(filtered))) if filtered.size else 0.0
    return (filtered / peak if peak > 0 else filtered).astype(np.float32)


def find_recordings(paths: List[os.PathLike | str]) -> List[Path]:
    """Expand files/folders into audio files (recursive), keeping order, without duplicates."""
    seen, out = set(), []
    for p in map(Path, paths):
        candidates = sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else [p]
        for q in candidates:
            key = os.path.normcase(str(q.resolve()))
            if is_audio_file(q) and key not in seen:
                seen.add(key)
                out.append(q)
    return out


def is_inside(path: os.PathLike | str, folders: List[os.PathLike | str]) -> bool:
    """True if `path` lies inside any of `folders` (used for protected folders)."""
    p = os.path.normcase(str(Path(path).resolve()))
    for f in folders:
        root = os.path.normcase(str(Path(f).resolve()))
        if p == root or p.startswith(root.rstrip("\\/") + os.sep):
            return True
    return False


def time_offset_sec(params: Dict, analysis_sample_rate: int) -> float:
    """Where the engine's time zero sits in the recording (analysis_start_sec, snapped to its grid)."""
    from pcg.engine import param

    start = max(0.0, float(param(params, "analysis_start_sec")))
    return round(start * analysis_sample_rate) / float(analysis_sample_rate) if analysis_sample_rate else start
