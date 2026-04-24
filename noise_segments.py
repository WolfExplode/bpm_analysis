# noise_segments.py — HF noise envelope → contiguous "noisy event" intervals (inverse noise gate).
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

def compute_noise_event_segments(
    inverse_band_envelope: np.ndarray,
    sample_rate: int,
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Mark samples where HF envelope >= quantile threshold and > min amplitude
    (noise_segment_gate_quantile, noise_segment_gate_min_amplitude in params),
    take contiguous runs,
    merge nearby runs (noise_segment_merge_gap_ms), drop very short ones, pad each interval by
    noise_segment_expand_ms, then merge again with the same gap so bridging applies in expanded time.

    Returns segments: start, end (seconds), state \"noise\", plus tooltip fields.
    If merged+expanded intervals cover more than noise_segment_max_coverage of the file,
    returns [] and logs a warning (unreliable HF gate).
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

    gate_q = float(params.get("noise_segment_gate_quantile", 0.85))
    gate_q = float(np.clip(gate_q, 0.0, 1.0))
    min_amp = float(params.get("noise_segment_gate_min_amplitude", 0.02))
    min_amp = max(0.0, min_amp)

    T = float(np.quantile(env[fin], gate_q))
    noisy = fin & (env >= T) & (env > min_amp)

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
    max_cov = float(params.get("noise_segment_max_coverage", 0.40))
    max_cov = float(np.clip(max_cov, 0.0, 1.0))
    if merged_expanded and n > 0:
        noise_samples = sum(int(b) - int(a) for a, b in merged_expanded)
        coverage = noise_samples / float(n)
        if coverage > max_cov:
            logging.warning(
                "Noisy segments could not be determined: HF-noise intervals cover %.1f%% of the "
                "recording (limit %.0f%%), so the inverse-band gate is treated as unreliable "
                "(e.g. clipping, overload, or insufficient HF headroom). Skipping noise segment "
                "labeling; downstream steps will not use HF noise windows.",
                100.0 * coverage,
                100.0 * max_cov,
            )
            return []

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
                "min_amplitude_gate": min_amp,
            }
        )

    if out:
        logging.info(
            "Noise event segments: %d intervals (HF > %.4f gate amp and ≥ %.0f%% quantile gate).",
            len(out),
            min_amp,
            100.0 * gate_q,
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
