"""Non-destructive, time-local playback gain; no changes to Analysis or source samples."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GainEdit:
    center: float
    width: float
    db: float
    q: float = 1.0
    shape: str = "region"
    enabled: bool = True

    @property
    def bandwidth(self) -> float:
        return self.width / self.q


def gain_envelope(times: np.ndarray, edits: tuple[GainEdit, ...]) -> np.ndarray:
    """Multiply time-local bell curves or flat regions with cosine edge fades.

    -60 dB means mute. Fades occupy at most 10 ms at each end (or 1/4 of a tiny region).
    Work and allocations depend on the playback block, never the recording's length.
    """
    gain = np.ones_like(times, dtype=np.float32)
    for edit in edits:
        if not edit.enabled:
            continue
        if edit.shape == "bell":
            offset = (times - edit.center) / edit.bandwidth
            inside = np.abs(offset) < 4
            if not inside.any():
                continue
            # Width is the full width at half the requested dB change; higher Q narrows it.
            weight = np.exp(-4 * np.log(2) * offset[inside] ** 2)
            factor = 10 ** (edit.db * weight / 20)
            if edit.db <= -60:
                factor = np.maximum(0, factor - 0.001 * weight)
            gain[inside] *= factor
            continue
        half = edit.width / 2
        distance = half - np.abs(times - edit.center)
        inside = distance > 0
        if not inside.any():
            continue
        fade = min(0.01, edit.width / 4)
        weight = 0.5 - 0.5 * np.cos(np.pi * np.clip(distance[inside] / fade, 0, 1))
        target = 0.0 if edit.db <= -60 else 10 ** (edit.db / 20)
        gain[inside] *= 1 + (target - 1) * weight
    return gain
