# noise_segments.py — HF noise envelope → contiguous "noisy event" intervals (inverse noise gate).
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

# HF envelope must exceed both the file quantile threshold and this absolute floor to count as noisy.
_NOISE_GATE_QUANTILE = 0.85 # higher = Less marked as noise
_NOISE_GATE_MIN_AMPLITUDE = 0.02 # lower = more sensitive to noise


def compute_noise_event_segments(
    inverse_band_envelope: np.ndarray,
    sample_rate: int,
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Mark samples where HF envelope >= quantile threshold and > min amplitude (see module constants),
    take contiguous runs,
    merge nearby runs (noise_segment_merge_gap_ms), drop very short ones, pad each interval by
    noise_segment_expand_ms, then merge again with the same gap so bridging applies in expanded time.

    Returns segments: start, end (seconds), state \"noise\", plus tooltip fields.
    """
    if inverse_band_envelope is None:
        return []
    env = np.asarray(inverse_band_envelope, dtype=np.float64).ravel()
    n = int(env.size)
    if n < 2 or sample_rate <= 0:
        return []

    fin = np.isfinite(env)
    if not np.any(fin):
        return []

    T = float(np.quantile(env[fin], _NOISE_GATE_QUANTILE))
    noisy = fin & (env >= T) & (env > _NOISE_GATE_MIN_AMPLITUDE)

    runs: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if not noisy[i]:
            i += 1
            continue
        j = i + 1
        while j < n and noisy[j]:
            j += 1
        runs.append((i, j))
        i = j

    merge_gap_ms = float(params.get("noise_segment_merge_gap_ms", 100.0))
    merge_gap = max(0, int(round(merge_gap_ms * sample_rate / 1000.0)))
    merged = _merge_intervals(runs, merge_gap)

    min_ms = float(params.get("noise_segment_min_duration_ms", 40.0))
    min_samples = max(1, int(round(min_ms * sample_rate / 1000.0)))

    expand_ms = float(params.get("noise_segment_expand_ms", 10.0))
    pad = max(0, int(round(expand_ms * sample_rate / 1000.0)))

    expanded_prep: List[Tuple[int, int]] = []
    for a, b in merged:
        if b - a < min_samples:
            continue
        a2 = max(0, a - pad)
        b2 = min(n, b + pad)
        if b2 <= a2:
            continue
        expanded_prep.append((a2, b2))

    # Same merge gap as pre-expand: coalesce pad-overlaps and bridge quiet gaps in expanded time.
    merged_expanded = _merge_intervals(expanded_prep, merge_gap)

    sr_f = float(sample_rate)
    out: List[Dict[str, Any]] = []
    for a, b in merged_expanded:
        start_sec = a / sr_f
        end_sec = b / sr_f
        if end_sec <= start_sec:
            continue
        chunk = env[a:b]
        chunk = chunk[np.isfinite(chunk)]
        peak = float(np.max(chunk)) if chunk.size > 0 else 0.0
        mean_e = float(np.mean(chunk)) if chunk.size > 0 else 0.0
        dur_ms = (b - a) / sr_f * 1000.0
        out.append(
            {
                "start": start_sec,
                "end": end_sec,
                "state": "noise",
                "duration_ms": round(dur_ms, 1),
                "peak": round(peak, 4),
                "mean": round(mean_e, 4),
                "threshold": round(T, 6),
                "min_amplitude_gate": _NOISE_GATE_MIN_AMPLITUDE,
            }
        )

    if out:
        logging.info(
            "Noise event segments: %d intervals (HF > %.2f and ≥ %.0f%% quantile).",
            len(out),
            _NOISE_GATE_MIN_AMPLITUDE,
            100.0 * _NOISE_GATE_QUANTILE,
        )
    return out


def _merge_intervals(runs: List[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
    if not runs:
        return []
    runs = sorted(runs, key=lambda t: (t[0], t[1]))
    out = [runs[0]]
    for a, b in runs[1:]:
        pa, pb = out[-1]
        if a - pb <= max_gap:
            out[-1] = (pa, max(pb, b))
        else:
            out.append((a, b))
    return out
