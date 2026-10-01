"""Playback: original and bandpass-filtered audio, playhead, loop region.

Playback audio is mono at no more than PLAYBACK_RATE_MAX, decoded block by block,
so an hour-long recording costs ~230 MB rather than gigabytes of full-rate stereo.
"""
from __future__ import annotations

import threading
from math import gcd
from typing import Dict, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from pcg import recording

PLAYBACK_RATE_MAX = 16000
SOURCE_ORIGINAL = "original"
SOURCE_FILTERED = "filtered"


def load_playback_audio(path: str, channel: str) -> Tuple[np.ndarray, int]:
    """Mono float32 at min(native rate, PLAYBACK_RATE_MAX)."""
    try:
        info = sf.info(path)
    except Exception:
        audio = recording.load(path)
        mono = audio.channel(channel)
        return _resample(mono, audio.sample_rate)
    sr = int(info.samplerate)
    target = min(sr, PLAYBACK_RATE_MAX)
    if target == sr:
        data, _ = sf.read(path, dtype="float32", always_2d=True)
        return recording.Audio(data, sr).channel(channel).astype(np.float32), sr
    g = gcd(sr, target)
    up, down = target // g, sr // g
    pad = 4 * down * 64
    block = down * max(1, (sr * 30) // down)  # ~30 s, a multiple of `down`
    pieces = []
    total = info.frames
    start = 0
    while start < total:
        lo = max(0, start - pad)
        hi = min(total, start + block + pad)
        data, _ = sf.read(path, start=lo, stop=hi, dtype="float32", always_2d=True)
        mono = recording.Audio(data, sr).channel(channel)
        out = resample_poly(mono.astype(np.float32), up, down)
        head = (start - lo) * up // down
        n = (min(start + block, total) - start) * up // down
        pieces.append(out[head:head + n].astype(np.float32))
        start += block
    return np.concatenate(pieces) if pieces else np.zeros(0, np.float32), target


def _resample(mono: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    target = min(sr, PLAYBACK_RATE_MAX)
    if target == sr:
        return mono.astype(np.float32), sr
    g = gcd(sr, target)
    return resample_poly(mono, target // g, sr // g).astype(np.float32), target


class Player:
    """sounddevice output with a shared position; all times in recording seconds."""

    def __init__(self) -> None:
        self._sources: Dict[str, np.ndarray] = {}
        self.sample_rate = 0
        self.source = SOURCE_ORIGINAL
        self._pos = 0
        self._loop: Optional[Tuple[int, int]] = None
        self._lock = threading.Lock()
        self._stream = None
        self.gain = 1.0

    # sources ------------------------------------------------------------------
    def set_audio(self, original: np.ndarray, sample_rate: int) -> None:
        self.stop()
        peak = float(np.max(np.abs(original))) if original.size else 0.0
        with self._lock:
            self._sources = {SOURCE_ORIGINAL: original / peak if peak > 0 else original}
            self.sample_rate = int(sample_rate)
            self._pos = 0
            self.source = SOURCE_ORIGINAL

    def set_filtered(self, filtered: np.ndarray) -> None:
        with self._lock:
            self._sources[SOURCE_FILTERED] = filtered

    def has(self, source: str) -> bool:
        return source in self._sources

    def clear(self) -> None:
        self.stop()
        with self._lock:
            self._sources = {}
            self.sample_rate = 0

    def samples(self, source: str = SOURCE_ORIGINAL) -> Optional[np.ndarray]:
        return self._sources.get(source)

    @property
    def loaded(self) -> bool:
        return bool(self._sources)

    @property
    def duration(self) -> float:
        a = self._sources.get(SOURCE_ORIGINAL)
        return len(a) / self.sample_rate if a is not None and self.sample_rate else 0.0

    # transport ----------------------------------------------------------------
    @property
    def playing(self) -> bool:
        return self._stream is not None and self._stream.active

    @property
    def position(self) -> float:
        return self._pos / self.sample_rate if self.sample_rate else 0.0

    def seek(self, t: float) -> None:
        if self.sample_rate:
            with self._lock:
                self._pos = int(np.clip(t * self.sample_rate, 0, len(self._sources[SOURCE_ORIGINAL])))

    def set_loop(self, region: Optional[Tuple[float, float]]) -> None:
        with self._lock:
            if region is None or not self.sample_rate:
                self._loop = None
            else:
                a, b = sorted(region)
                self._loop = (int(a * self.sample_rate), int(b * self.sample_rate))
                if not self._loop[0] <= self._pos < self._loop[1]:
                    self._pos = self._loop[0]

    def toggle_source(self) -> str:
        nxt = SOURCE_FILTERED if self.source == SOURCE_ORIGINAL else SOURCE_ORIGINAL
        if nxt in self._sources:
            self.source = nxt
        return self.source

    def play(self) -> None:
        if not self.loaded or self.playing:
            return
        import sounddevice as sd

        self._stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype="float32",
                                       callback=self._callback, blocksize=1024)
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def toggle(self) -> None:
        self.stop() if self.playing else self.play()

    def _callback(self, outdata, frames, _time, _status) -> None:
        import sounddevice as sd

        with self._lock:
            src = self._sources.get(self.source)
            if src is None:
                src = self._sources[SOURCE_ORIGINAL]
            lo, hi = self._loop if self._loop else (0, len(src))
            written = 0
            while written < frames:
                if self._pos >= hi:
                    if self._loop:
                        self._pos = lo
                    else:
                        outdata[written:, 0] = 0
                        raise sd.CallbackStop
                n = min(frames - written, hi - self._pos)
                outdata[written:written + n, 0] = src[self._pos:self._pos + n] * self.gain
                self._pos += n
                written += n


def spectrogram(signal: np.ndarray, sample_rate: int, fmax: float = 1000.0, hop_sec: float = 0.032,
                n_fft: int = 1024) -> Tuple[float, float, float, np.ndarray]:
    """Log-power spectrogram up to fmax: (t0, dt, f_top, dB array shaped (frames, bins)).

    Computed in chunks so an hour of audio needs ~30 MB, not a full STFT in memory.
    """
    hop = max(1, int(round(hop_sec * sample_rate)))
    n_fft = int(min(n_fft, max(64, len(signal))))
    n_bins = min(n_fft // 2 + 1, max(1, int(fmax / (sample_rate / n_fft)) + 1))  # fmax may exceed Nyquist
    n_frames = max(0, 1 + (len(signal) - n_fft) // hop)
    window = np.hanning(n_fft).astype(np.float32)
    out = np.empty((n_frames, n_bins), dtype=np.float32)
    x = np.ascontiguousarray(signal, dtype=np.float32)
    chunk = 4096
    for c0 in range(0, n_frames, chunk):
        c1 = min(n_frames, c0 + chunk)
        idx = (np.arange(c0, c1) * hop)[:, None] + np.arange(n_fft)[None, :]
        spec = np.abs(np.fft.rfft(x[idx] * window, axis=1)[:, :n_bins]) ** 2
        out[c0:c1] = 10.0 * np.log10(spec + 1e-12)
    f_top = n_bins * sample_rate / n_fft
    return (n_fft / 2) / sample_rate, hop / sample_rate, f_top, out
