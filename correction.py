"""
correction.py — Pass 3 cardiac-cycle correction and state-timeline generation.

Entry point: run_pass3_correction().

Section map (Ctrl+F the section title to jump)
-----------------------------------------------
  STATE CONSTANTS                           line ~43
  Gap peak recovery                         line ~63
  Boundary snapping                         line ~162
  BPM interpolation / transient edge        line ~366
  Hover / correction note formatters        line ~467
  Boundary geometry (paint, overlap)        line ~513
  Reasoning payload builder                 line ~627
  S2 events rebuild                         line ~723
  Noise interval utilities                  line ~758
  BPM raster construction                   line ~822
  Noise repair (clear + rebuild)            line ~967
  Correction reasoning record builders     line ~1094
  Duration series cleaning + measurement   line ~1152
  Boundary geometry helpers (trim/interp)  line ~1312
  Gap detection and filling                line ~1362
  S2 seeding from Pass 2 pairs             line ~2203
  MAIN ENTRY POINT                         line ~2276
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from analysis_data_schema import AnalysisData

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from classifier import PeakClassifier
from confidence_engine import calculate_bpm_intervals
from hrv import _median_mad_keep_mask_time_window, filter_interval_durations_by_limits


# ─────────────────────────────────────────────────────────────────────────────
# State encoding constants (shared with pipeline / plotting)
# ─────────────────────────────────────────────────────────────────────────────

STATE_S1 = 0
STATE_SYSTOLE = 1
STATE_S2 = 2
STATE_DIASTOLE = 3
# Samples inside HF-noise windows when pass3_enable_noise_repair clears untrusted labels.
STATE_UNKNOWN = 4

STATE_LABELS_ENCODING: Dict[str, int] = {
    "S1": STATE_S1,
    "systole": STATE_SYSTOLE,
    "S2": STATE_S2,
    "diastole": STATE_DIASTOLE,
    "unknown": STATE_UNKNOWN,
}

# ─────────────────────────────────────────────────────────────────────────────
# Pass 3: large-gap peak recovery (debug/visualization)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_sensitive_peaks_in_large_gap_windows(
    audio_envelope: np.ndarray,
    sample_rate: int,
    gap_windows_samples: List[Dict[str, Any]],
    params: Dict,
    *,
    prominence_quantile: Optional[float] = None,
    height_scale_override: Optional[float] = None,
    dynamic_noise_floor_series: Optional[pd.Series] = None,
) -> np.ndarray:
    """
    Rerun a more sensitive peak detector inside Pass 3 "large gap" windows.

    The recovered peaks are stored in analysis_data for plotting/debug and are also
    consumed by _pass3_snap_rebuilt_states_to_recovered_peaks to shift synthetic S1/S2
    boundaries onto real acoustic events after the gap fill is complete.
    """
    if sample_rate <= 0:
        return np.asarray([], dtype=np.int64)
    env = np.asarray(audio_envelope, dtype=np.float64)
    n = int(len(env))
    if n <= 0:
        return np.asarray([], dtype=np.int64)
    if not gap_windows_samples or not isinstance(gap_windows_samples, list):
        return np.asarray([], dtype=np.int64)

    min_peak_dist_samples = int(float(params.get("min_peak_distance_sec", 0.18)) * float(sample_rate))
    min_peak_dist_samples = max(1, int(min_peak_dist_samples))

    if prominence_quantile is None:
        q = float(params.get("pass3_gap_recovery_peak_prominence_quantile", 0.50))
    else:
        q = float(prominence_quantile)
    q = float(np.clip(q, 0.0, 1.0))
    if height_scale_override is None:
        height_scale = float(params.get("pass3_gap_recovery_height_scale", 0.85))
    else:
        height_scale = float(height_scale_override)
    height_scale = float(np.clip(height_scale, 0.0, 1.0))

    nf = None
    if dynamic_noise_floor_series is not None and isinstance(dynamic_noise_floor_series, pd.Series):
        try:
            nf_arr = dynamic_noise_floor_series.to_numpy(dtype=np.float64, copy=False)
            if len(nf_arr) == n:
                nf = nf_arr
        except Exception:
            nf = None

    out: List[int] = []
    for w in gap_windows_samples:
        if not isinstance(w, dict):
            continue
        try:
            lo = int(w.get("start_sample", -1))
            hi = int(w.get("end_sample", -1))
        except Exception:
            continue
        lo = int(max(0, min(lo, n)))
        hi = int(max(0, min(hi, n)))
        if hi <= lo + 3:
            continue

        seg = env[lo:hi]
        if seg.size < 4:
            continue

        try:
            prom_thresh = float(np.quantile(seg, q))
        except Exception:
            prom_thresh = 0.0
        if not np.isfinite(prom_thresh) or prom_thresh < 0:
            prom_thresh = 0.0

        height = None
        if nf is not None:
            ht = nf[lo:hi] * height_scale
            height = ht

        try:
            pk, _ = find_peaks(
                seg,
                height=height,
                prominence=prom_thresh,
                distance=min_peak_dist_samples,
            )
        except Exception:
            continue
        if pk is None or len(pk) == 0:
            continue
        out.extend((lo + np.asarray(pk, dtype=np.int64)).tolist())

    if not out:
        return np.asarray([], dtype=np.int64)
    # Unique + sorted
    out_arr = np.asarray(sorted(set(int(x) for x in out if 0 <= int(x) < n)), dtype=np.int64)
    return out_arr


# ─────────────────────────────────────────────────────────────────────────────
# Boundary snapping — shift rebuilt state edges to align with recovered peaks
# ─────────────────────────────────────────────────────────────────────────────

def _pass3_snap_rebuilt_states_to_recovered_peaks(
    state_labels: np.ndarray,
    state_boundaries: List[Tuple],
    recovered_peaks_insensitive: np.ndarray,
    recovered_peaks_sensitive: np.ndarray,
    audio_envelope: np.ndarray,
    sample_rate: int,
    snap_window_samples: int,
) -> Tuple[np.ndarray, List[Tuple]]:
    """
    Post-processing: after the gap fill is complete, shift rebuilt S1 and S2 segment
    boundaries to align with acoustically recovered peaks where available.

    "Fill first, then shift": the existing rebuild produced a valid non-overlapping
    partition; this function makes small position adjustments on top of it. The
    neighboring systole or diastole segment absorbs the shift on each side, so the
    partition stays contiguous and non-overlapping by construction.

    Snapping is S1-first and cycle-aware:
      Pass 1 — all rebuilt S1 segments snap first (forward order), reserving peaks.
      Pass 2 — rebuilt S2 segments snap to remaining unresolved peaks only, and are
               also blocked from using any peak within snap_window_samples of the next
               rebuilt S1 center (first right of refusal), preventing S2 from stealing
               peaks that should anchor S1s.
    Each recovered peak is assigned to at most one segment (tracked via used_peaks).

    State segments represent durations [start, end). The center of an S1 or S2 segment
    is only used here to locate a nearby recovered peak and compute the shift delta.

    For start_after_s1 gaps, the first rebuilt segment is systole (not S1), so the kept
    real S1 anchor before the gap is never touched by this function.
    """
    if (
        (recovered_peaks_insensitive is None or len(recovered_peaks_insensitive) == 0)
        and (recovered_peaks_sensitive is None or len(recovered_peaks_sensitive) == 0)
    ):
        return state_labels, state_boundaries

    # NOTE: don't use `arr or []` with numpy arrays (ambiguous truth value).
    rec_ins = np.asarray(
        recovered_peaks_insensitive if recovered_peaks_insensitive is not None else [],
        dtype=np.int64,
    )
    rec_sens = np.asarray(
        recovered_peaks_sensitive if recovered_peaks_sensitive is not None else [],
        dtype=np.int64,
    )
    n = int(len(state_labels))
    env = np.asarray(audio_envelope, dtype=np.float64)
    sw = int(snap_window_samples)

    used_peaks: set = set()

    def _snap(expected: int, exclude_near: Optional[int] = None, *, allow_sensitive: bool = False) -> Optional[int]:
        # Due to how gap detection works, there shouldn't be any already-known raw peaks
        # inside the gap; recovered peaks are the only candidate source here.
        # Scoring: closest in time; tie-break by higher envelope amplitude.
        # Recovered peaks are generated with find_peaks(distance=...) so adjacent
        # candidates shouldn't appear — the tie-break is just a safety net.
        lo = expected - sw
        hi = expected + sw
        base = rec_ins
        if allow_sensitive and rec_sens.size:
            # S2 can use leftover insensitive peaks plus any additional sensitive peaks.
            base = np.unique(np.concatenate([rec_ins, rec_sens]))
        mask = (base >= lo) & (base <= hi)
        cands = base[mask]
        # Exclude peaks already consumed by a prior snap.
        cands = cands[np.array([int(c) not in used_peaks for c in cands], dtype=bool)]
        # Exclude peaks near the next S1 center (first right of refusal for S2 snapping).
        if exclude_near is not None:
            cands = cands[np.abs(cands - exclude_near) > sw]
        if len(cands) == 0:
            return None
        dists = np.abs(cands - expected)
        min_d = int(np.min(dists))
        closest = cands[dists == min_d]
        if len(closest) == 1:
            return int(closest[0])
        amps = env[np.clip(closest, 0, n - 1)]
        return int(closest[np.argmax(amps)])

    def _apply_shift(i: int, snapped: int) -> bool:
        """Shift segment i to be centered on snapped. Updates segs in place. Returns True if applied."""
        a0, a1 = int(segs[i][0]), int(segs[i][1])
        width = a1 - a0
        if width <= 0:
            return False
        center = (a0 + a1) // 2
        delta = snapped - center
        if delta == 0:
            return False

        new_a0 = a0 + delta
        new_a1 = a1 + delta  # = new_a0 + width (segment width is preserved)

        # Lower bound: can't start before the previous segment ends.
        prev_end = int(segs[i - 1][1]) if i > 0 else a0
        new_a0 = max(prev_end, new_a0)
        new_a1 = new_a0 + width  # re-derive after clamping start

        # Upper bound: the next segment must keep at least 1 sample (systole / diastole).
        next_end = int(segs[i + 1][1]) if i + 1 < ns else a1
        if new_a1 >= next_end:
            # Not enough room after the shift; keep synthetic placement.
            return False

        if new_a1 <= new_a0:
            return False

        segs[i][0] = new_a0
        segs[i][1] = new_a1
        name = segs[i][2]
        new_meta = dict(segs[i][3])
        new_meta["s1" if name == "S1" else "s2"] = snapped
        new_meta["snapped"] = True
        try:
            if isinstance(new_meta.get("reasoning"), dict):
                r = dict(new_meta["reasoning"])
                notes = r.get("notes")
                if not isinstance(notes, list):
                    notes = []
                notes = list(notes)
                notes.append("Shifted to recovered peak at gap.")
                r["notes"] = notes
                new_meta["reasoning"] = r
        except Exception:
            pass
        segs[i][3] = new_meta

        # Adjacent segments absorb the shift on each side.
        if i > 0:
            segs[i - 1][1] = new_a0
        if i + 1 < ns:
            segs[i + 1][0] = new_a1
        return True

    # Work on a mutable copy (list-of-lists) sorted by start.
    segs = [list(s) for s in sorted(state_boundaries, key=lambda t: int(t[0]))]
    ns = len(segs)
    changed = False

    def _is_rebuilt(i: int) -> bool:
        m = segs[i][3]
        return isinstance(m, dict) and m.get("rebuild_source") in ("gap_insert", "noise_repair")

    # --- Pass 1: Snap all rebuilt S1 segments first ---
    # S1 is the primary cardiac anchor; it gets first pick of recovered peaks.
    for i in range(ns):
        if segs[i][2] != "S1" or not _is_rebuilt(i):
            continue
        center = (int(segs[i][0]) + int(segs[i][1])) // 2
        snapped = _snap(center, allow_sensitive=False)
        if snapped is None:
            continue
        if _apply_shift(i, snapped):
            used_peaks.add(snapped)
            changed = True

    # --- Pass 2: Snap rebuilt S2 segments using only remaining peaks ---
    # Build rebuilt S1 centers (post-Pass-1 positions) for first-right-of-refusal:
    # any peak within snap_window_samples of a future S1 center is off-limits for S2.
    rebuilt_s1_centers = [
        (int(segs[i][0]) + int(segs[i][1])) // 2
        for i in range(ns)
        if segs[i][2] == "S1" and _is_rebuilt(i)
    ]
    for i in range(ns):
        if segs[i][2] != "S2" or not _is_rebuilt(i):
            continue
        center = (int(segs[i][0]) + int(segs[i][1])) // 2
        next_s1_center = next((c for c in rebuilt_s1_centers if c > center), None)
        snapped = _snap(center, exclude_near=next_s1_center, allow_sensitive=True)
        if snapped is None:
            continue
        if _apply_shift(i, snapped):
            used_peaks.add(snapped)
            changed = True

    if not changed:
        return state_labels, state_boundaries

    new_boundaries = [tuple(s) for s in segs]

    # Repaint state_labels from the updated boundary list.
    _state_code = {
        "S1": STATE_S1, "systole": STATE_SYSTOLE, "S2": STATE_S2, "diastole": STATE_DIASTOLE,
    }
    for _a0, _a1, _name, _ in new_boundaries:
        code = _state_code.get(_name)
        if code is None:
            continue
        lo = int(max(0, min(int(_a0), n)))
        hi = int(max(0, min(int(_a1), n)))
        if hi > lo:
            state_labels[lo:hi] = code

    return state_labels, new_boundaries


# ─────────────────────────────────────────────────────────────────────────────
# BPM interpolation and transient edge detection
# ─────────────────────────────────────────────────────────────────────────────

def _bpm_at_time(
    t_sec: float,
    lt_source: Any,
    fallback_bpm: float,
) -> float:
    """
    Interpolate BPM at time t_sec from a raster-backed source.

    Accepts:
    - (times, bpm) tuple/list of arrays
    - dict with {"times": ..., "bpm": ...}
    - pd.Series indexed by time (legacy; allowed for now)
    """
    if lt_source is None:
        return fallback_bpm
    try:
        times = None
        values = None
        if isinstance(lt_source, (tuple, list)) and len(lt_source) == 2:
            times, values = lt_source
        elif isinstance(lt_source, dict):
            times = lt_source.get("times")
            values = lt_source.get("bpm")
        elif isinstance(lt_source, pd.Series):
            if getattr(lt_source, "empty", True):
                return fallback_bpm
            times = lt_source.index.values
            values = lt_source.values
        else:
            return fallback_bpm

        times = np.asarray(times, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if len(times) < 2 or len(times) != len(values):
            return fallback_bpm
        bpm = float(np.interp(float(t_sec), times, values, left=float(values[0]), right=float(values[-1])))
        if not np.isfinite(bpm) or bpm <= 0:
            return fallback_bpm
        return bpm
    except Exception:
        return fallback_bpm


def _find_transient_bounds(
    peak_idx: int,
    half_window: int,
    min_half: int,
    max_half: int,
    n_samples: int,
    audio_envelope: np.ndarray,
    edge_alpha: float,
    edge_n_exp: float,
) -> Tuple[int, int]:
    """
    Return (start, end) as a [start, end) sample slice for the transient at peak_idx.

    Walks outward through a super-Gaussian-weighted envelope until the weighted value
    drops to edge_alpha * peak_weighted_value.  half_window is a hard cap; min/max_half
    clamp the result to physiology bounds.  Falls back to fixed half_window on error.
    """
    try:
        left = max(0, peak_idx - half_window)
        right = min(n_samples - 1, peak_idx + half_window)
        n_win = right - left + 1
        sigma = max(1.0, n_win / 4.0)
        dist = np.abs(np.arange(left, right + 1, dtype=np.float64) - peak_idx)
        weights = np.exp(-((dist / sigma) ** edge_n_exp))
        weighted_env = weights * audio_envelope[left: right + 1]
        center_offset = peak_idx - left
        peak_val = float(weighted_env[center_offset])
        if peak_val <= 0.0:
            raise ValueError("zero peak")
        thresh = edge_alpha * peak_val

        start = left
        for k in range(center_offset - 1, -1, -1):
            if weighted_env[k] <= thresh:
                start = left + k + 1
                break

        end = right
        for k in range(center_offset + 1, len(weighted_env)):
            if weighted_env[k] <= thresh:
                end = left + k - 1
                break

        start = max(start, peak_idx - max_half)
        end = min(end, peak_idx + max_half)
        start = min(start, peak_idx - min_half)
        end = max(end, peak_idx + min_half)
        start = max(0, start)
        end = min(n_samples - 1, end)
        return int(start), int(end + 1)
    except Exception:
        return max(0, peak_idx - half_window), min(n_samples, peak_idx + half_window + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Hover / correction note formatters (used by _build_reasoning_payload)
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_corr_note(c: Dict[str, Any], sample_rate: float) -> str:
    """Return a warning line describing the direct correction applied."""
    ctype = c.get("type", "")
    if ctype == "resnap_s2":
        new_sys = round((c.get("new_s2", 0) - c.get("s1", 0)) / sample_rate * 1000)
        return f"\u26a0 Systole out of range \u2014 S2 repositioned, systole now {new_sys}ms"
    if ctype == "insert_s1":
        return f"\u26a0 RR gap too long \u2014 missing beat inserted at {c.get('t_expected_sec', 0):.2f}s"
    if ctype == "remove_false_s1":
        return (
            f"\u26a0 Both systole+diastole too short \u2014 false beat at "
            f"{c.get('removed_s1', 0) / sample_rate:.2f}s removed"
        )
    if ctype == "flip_demote_s1":
        old_t = c.get("old_s1_next", 0) / sample_rate
        new_t = c.get("new_s1_next", 0) / sample_rate
        return (
            f"\u26a0 Diastole too short \u2014 beat at {old_t:.2f}s re-labeled as S2, "
            f"new S1 at {new_t:.2f}s"
        )
    if ctype in ("sensitive_peak", "spectral_s2"):
        new_sys = round(c.get("new_systole_sec", 0) * 1000)
        method = "sensitive peak detection" if ctype == "sensitive_peak" else "spectral fingerprint"
        return f"\u26a0 Systole too long \u2014 faint S2 found via {method}, systole now {new_sys}ms"
    return f"\u26a0 Correction applied ({ctype})"


def _fmt_cascade_note(src: Dict[str, Any], sample_rate: float) -> str:
    """Return an info line describing an upstream correction that shifted this segment."""
    ctype = src.get("type", "")
    src_t = src.get("s1", src.get("s1_prev", 0)) / sample_rate
    label = {
        "resnap_s2": "S2 resnap",
        "insert_s1": "beat insertion",
        "remove_false_s1": "false beat removal",
        "flip_demote_s1": "S1/S2 flip correction",
        "sensitive_peak": "faint-S2 detection",
        "spectral_s2": "spectral S2 detection",
    }.get(ctype, ctype)
    return f"\u2139 Shifted by {label} at {src_t:.2f}s; duration still within expected range"


# ─────────────────────────────────────────────────────────────────────────────
# Boundary geometry — paint state spans from S1/S2 peaks, resolve overlaps
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_boundary_overlap(
    s1_start: int, s1_end: int,
    s2_start: int, s2_end: int,
    s1_next: int,
) -> Tuple[int, int, int, int]:
    """Resolve S1/S2 window overlaps and clip S2 end to the next cycle boundary."""
    if s2_start < s1_end:
        mid = (s1_end + s2_start) // 2
        s1_end = max(s1_start + 1, min(s1_end, mid))
        s2_start = max(s2_start, s1_end)
    if s2_end >= s1_next:
        s2_end = min(s2_end, s1_next)
    return s1_start, s1_end, s2_start, s2_end


def _paint_state_boundaries(
    s1: int,
    s2: int,
    s1_next: int,
    s1_half: int,
    s2_half: int,
    n_samples: int,
    audio_envelope: Optional[np.ndarray] = None,
    edge_alpha: float = 0.03,
    edge_n_exp: float = 4.0,
    min_s1_half: int = 1,
    max_s1_half: int = 50,
    min_s2_half: int = 1,
    max_s2_half: int = 50,
    use_transient_detection: bool = False,
) -> Tuple[int, int, int, int]:
    """
    Compute (s1_start, s1_end, s2_start, s2_end) for one cardiac cycle.

    use_transient_detection=True (final timeline): envelope-based edge detection via
    _find_transient_bounds; requires audio_envelope.
    use_transient_detection=False (before-correction snapshot): fixed ±half-window.
    """
    if use_transient_detection and audio_envelope is not None:
        s1_start, s1_end = _find_transient_bounds(
            s1, s1_half, min_s1_half, max_s1_half, n_samples, audio_envelope, edge_alpha, edge_n_exp,
        )
        s2_start, s2_end = _find_transient_bounds(
            s2, s2_half, min_s2_half, max_s2_half, n_samples, audio_envelope, edge_alpha, edge_n_exp,
        )
    else:
        s1_start = max(0, s1 - s1_half)
        s1_end = min(n_samples, s1 + s1_half + 1)
        s2_start = max(0, s2 - s2_half)
        s2_end = min(n_samples, s2 + s2_half + 1)

    s1_start, s1_end, s2_start, s2_end = _resolve_boundary_overlap(
        s1_start, s1_end, s2_start, s2_end, s1_next,
    )
    return s1_start, s1_end, s2_start, s2_end


def _hover_audio_time_mmssmmm(sample_idx: int, sample_rate: float) -> str:
    """Wall-clock style time from start of audio: M:SS.mmm (minutes may exceed two digits)."""
    sr = float(max(sample_rate, 1e-9))
    t = float(sample_idx) / sr
    m = int(t // 60)
    s = t - 60.0 * m
    return f"{m}:{s:06.3f}"


def _envelope_align_duration_note(phase: str, before_ms: int, after_ms: int) -> str:
    """Hover line: fixed ±window duration vs envelope-aligned duration (Pass 3 final paint)."""
    label = {"S1": "S1", "S2": "S2", "systole": "Systole", "diastole": "Diastole"}.get(phase, phase)
    if before_ms == after_ms:
        return (
            f"\u2139 {label}: duration unchanged at {after_ms}ms "
            f"(envelope bounds matched the fixed \u00b1window span)."
        )
    return (
        f"\u2139 {label}: duration altered to align with envelope bounds "
        f"(from {before_ms}ms to {after_ms}ms)."
    )


def _s2_index_hover_note(
    s1: int,
    s2: int,
    s1_next: int,
    s1_s2_pairs: Optional[List[Tuple[int, int]]],
    ivs: Dict,
    sample_rate: float,
) -> str:
    """Short S2 peak-time line for hover (Pass 2 pair vs BPM nominal)."""
    pair_map = {int(a): int(b) for a, b in (s1_s2_pairs or [])}
    s2_p2 = pair_map.get(int(s1))
    in_w = s2_p2 is not None and s1 < int(s2_p2) < s1_next
    s1_s2_nom = float(ivs.get("s1_s2_nominal", 0.30))
    sr = float(sample_rate)
    s2i = int(s2)
    t_here = _hover_audio_time_mmssmmm(s2i, sr)

    if in_w and s2i == int(s2_p2):
        return f"\u2139 S2: location unchanged at {t_here}."

    if in_w:
        t_from = _hover_audio_time_mmssmmm(int(s2_p2), sr)
        t_to = _hover_audio_time_mmssmmm(s2i, sr)
        return f"\u2139 S2: location altered from {t_from} to {t_to}."

    return (
        f"\u2139 S2: at {t_here} from BPM nominal s1_s2 ({s1_s2_nom:.3f}s); no Pass 2 pair in this RR window."
    )


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# Reasoning payload builder (assembles per-cycle debug hover data)
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def _build_reasoning_payload(
    s1: int,
    s1_start: int,
    s1_end: int,
    s2: int,
    s2_start: int,
    s2_end: int,
    s1_next: int,
    ivs: Dict,
    direct_corrections: List[Dict],
    cascade_src: Optional[Dict],
    before_s2: Optional[int],
    before_s1nxt: Optional[int],
    sample_rate: float,
    shift_thresh_samp: int,
    *,
    hover_debug_geometry: bool = False,
    hover_s1_s2_pairs: Optional[List[Tuple[int, int]]] = None,
    hover_s1_half: int = 0,
    hover_s2_half: int = 0,
    hover_n_samples: int = 0,
) -> Dict[str, Any]:
    """Build expected/measured ms + warning notes per state for the HTML overlay (cardiac strip hover)."""
    _exp_s1_ms  = round(ivs.get("s1_nominal",    0.040) * 1000)
    _exp_sys_ms = round(ivs.get("s1_s2_nominal", 0.300) * 1000)
    _exp_s2_ms  = round(ivs.get("s2_nominal",    0.030) * 1000)
    _exp_dia_ms = round(ivs.get("s2_s1_nominal", 0.400) * 1000)

    _meas_s1_ms  = round((s1_end   - s1_start) / sample_rate * 1000) if s1_end   > s1_start else 0
    _meas_sys_ms = round((s2_start - s1_end)   / sample_rate * 1000) if s2_start > s1_end   else 0
    _meas_s2_ms  = round((s2_end   - s2_start) / sample_rate * 1000) if s2_end   > s2_start else 0
    _meas_dia_ms = round((s1_next  - s2_end)   / sample_rate * 1000) if s1_next  > s2_end   else 0

    _s1_notes  = [_fmt_corr_note(c, sample_rate) for c in direct_corrections
                  if c.get("type") == "remove_false_s1"]
    _sys_notes = [_fmt_corr_note(c, sample_rate) for c in direct_corrections
                  if c.get("type") in ("resnap_s2", "flip_demote_s1", "sensitive_peak", "spectral_s2")]
    _s2_notes  = [_fmt_corr_note(c, sample_rate) for c in direct_corrections
                  if c.get("type") in ("resnap_s2", "flip_demote_s1", "sensitive_peak", "spectral_s2")]
    _dia_notes = [_fmt_corr_note(c, sample_rate) for c in direct_corrections
                  if c.get("type") in ("insert_s1", "flip_demote_s1")]

    if cascade_src:
        if not _sys_notes and before_s2 is not None and abs(before_s2 - s2) > shift_thresh_samp:
            _sys_notes.append(_fmt_cascade_note(cascade_src, sample_rate))
        if not _dia_notes and before_s1nxt is not None and abs(before_s1nxt - s1_next) > shift_thresh_samp:
            _dia_notes.append(_fmt_cascade_note(cascade_src, sample_rate))
        if not _s2_notes and before_s2 is not None and abs(before_s2 - s2) > shift_thresh_samp:
            _s2_notes.append(_fmt_cascade_note(cascade_src, sample_rate))

    if hover_debug_geometry:
        if hover_n_samples > 0 and hover_s1_half >= 1 and hover_s2_half >= 1:
            fs1s, fs1e, fs2s, fs2e = _paint_state_boundaries(
                s1, s2, s1_next, hover_s1_half, hover_s2_half, hover_n_samples,
                use_transient_detection=False,
            )
            bf_s1 = round((fs1e - fs1s) / sample_rate * 1000) if fs1e > fs1s else 0
            bf_s2 = round((fs2e - fs2s) / sample_rate * 1000) if fs2e > fs2s else 0
            bf_sys = round((fs2s - fs1e) / sample_rate * 1000) if fs2s > fs1e else 0
            bf_dia = round((s1_next - fs2e) / sample_rate * 1000) if s1_next > fs2e else 0
            _s1_notes = [_envelope_align_duration_note("S1", bf_s1, _meas_s1_ms)] + _s1_notes
            _sys_notes = [_envelope_align_duration_note("systole", bf_sys, _meas_sys_ms)] + _sys_notes
            _s2_notes = (
                [_envelope_align_duration_note("S2", bf_s2, _meas_s2_ms)]
                + [_s2_index_hover_note(s1, s2, s1_next, hover_s1_s2_pairs, ivs, sample_rate)]
                + _s2_notes
            )
            _dia_notes = [_envelope_align_duration_note("diastole", bf_dia, _meas_dia_ms)] + _dia_notes
        else:
            _s1_notes = [
                f"\u2139 S1: duration {_meas_s1_ms}ms (envelope-aligned; fixed-window baseline unavailable).",
            ] + _s1_notes
            _sys_notes = [
                f"\u2139 Systole: duration {_meas_sys_ms}ms (gap after envelope paint).",
            ] + _sys_notes
            _s2_notes = (
                [f"\u2139 S2: duration {_meas_s2_ms}ms (envelope-aligned; fixed-window baseline unavailable)."]
                + [_s2_index_hover_note(s1, s2, s1_next, hover_s1_s2_pairs, ivs, sample_rate)]
                + _s2_notes
            )
            _dia_notes = [
                f"\u2139 Diastole: duration {_meas_dia_ms}ms (gap after envelope paint).",
            ] + _dia_notes

    return {
        "S1":       {"expected_ms": _exp_s1_ms,  "measured_ms": _meas_s1_ms,  "notes": _s1_notes},
        "systole":  {"expected_ms": _exp_sys_ms, "measured_ms": _meas_sys_ms, "notes": _sys_notes},
        "S2":       {"expected_ms": _exp_s2_ms,  "measured_ms": _meas_s2_ms,  "notes": _s2_notes},
        "diastole": {"expected_ms": _exp_dia_ms, "measured_ms": _meas_dia_ms, "notes": _dia_notes},
    }


# ─────────────────────────────────────────────────────────────────────────────
# S2 events rebuild — synthesize S2 positions from S1 list + BPM prior
# ─────────────────────────────────────────────────────────────────────────────

def _rebuild_s2_events(
    s1_list: List[int],
    lt_series: Optional[pd.Series],
    fallback_bpm: float,
    sample_rate: int,
    params: Dict,
    seed_s2_events: Optional[List[int]] = None,
) -> List[int]:
    """One S2 index per RR interval: Pass 2 seed when valid, else BPM nominal clamp."""
    seed = seed_s2_events
    s2_events: List[int] = []
    for j in range(len(s1_list) - 1):
        a = int(s1_list[j])
        b = int(s1_list[j + 1])
        if b <= a:
            s2_events.append(int(a))
            continue
        if seed is not None and j < len(seed):
            sj = int(seed[j])
            if a < sj < b:
                s2_events.append(int(max(a + 1, min(sj, b - 1))))
                continue
        t_a = a / float(sample_rate)
        bpm_a = _bpm_at_time(t_a, lt_series, fallback_bpm)
        ivs_a = calculate_bpm_intervals(bpm_a, params)
        s1_s2_nominal_a = float(ivs_a.get("s1_s2_nominal", 0.30))
        s2_pred_a = int(round(a + s1_s2_nominal_a * sample_rate))
        s2_events.append(int(max(a + 1, min(s2_pred_a, b - 1))))
    return s2_events


# ─────────────────────────────────────────────────────────────────────────────
# Noise interval utilities — convert segments to sample ranges, merge, query
# ─────────────────────────────────────────────────────────────────────────────

def _noise_sample_intervals(
    noise_event_segments: Optional[List[Dict[str, Any]]],
    sample_rate: int,
    n_samples: int,
) -> List[Tuple[int, int]]:
    """Convert noise_event_segments (seconds) to half-open sample intervals [lo, hi)."""
    out: List[Tuple[int, int]] = []
    if not noise_event_segments or sample_rate <= 0 or n_samples <= 0:
        return out
    for seg in noise_event_segments:
        if not isinstance(seg, dict):
            continue
        try:
            t0 = float(seg.get("start", float("nan")))
            t1 = float(seg.get("end", float("nan")))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(t0) and math.isfinite(t1)) or t1 <= t0:
            continue
        lo = int(math.floor(t0 * sample_rate))
        hi = int(math.ceil(t1 * sample_rate))
        lo = max(0, lo)
        hi = min(n_samples, max(lo + 1, hi))
        out.append((lo, hi))
    return out


def _merge_sorted_intervals(ivs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ivs:
        return []
    s = sorted(ivs, key=lambda t: (t[0], t[1]))
    out: List[Tuple[int, int]] = [s[0]]
    for a, b in s[1:]:
        la, lb = out[-1]
        if a <= lb:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def _half_open_intervals_intersect(a0: int, a1: int, b0: int, b1: int) -> bool:
    """True iff [a0, a1) and [b0, b1) overlap (half-open)."""
    return int(b0) < int(a1) and int(b1) > int(a0)


def _span_intersects_merged_noise(
    lo: int, hi: int, merged: List[Tuple[int, int]], n_samples: int,
) -> bool:
    """True if [lo, hi) overlaps any merged HF-noise interval."""
    for nlo, nhi in merged:
        nlo = max(0, int(nlo))
        nhi = min(n_samples, int(nhi))
        if nhi <= nlo:
            continue
        if _half_open_intervals_intersect(lo, hi, nlo, nhi):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# BPM raster construction — build smooth BPM series from clean RR intervals
# ─────────────────────────────────────────────────────────────────────────────

def _build_lt_bpm_series_from_clean_rr(
    s1_peaks: np.ndarray,
    noise_ivs: List[Tuple[int, int]],
    sample_rate: int,
    n_samples: int,
    params: Optional[Dict] = None,
) -> Optional[pd.Series]:
    """
    Build a time-indexed BPM series from S1→S1 intervals that do NOT intersect HF-noise.

    Each retained interval contributes one (t_mid, bpm) point:
      t_mid = midpoint time of the RR interval, bpm = 60 / RR_sec.

    Applies the same light cleaning used for measured systole/diastole curves:
      - local MAD outlier removal in a rolling time window
      - small Gaussian-weighted rolling mean smoothing

    Returns None if there are not enough clean intervals to interpolate (need >=2).
    """
    if sample_rate <= 0:
        return None
    peaks = np.asarray(s1_peaks, dtype=np.int64)
    if len(peaks) < 2:
        return None
    merged = _merge_sorted_intervals(noise_ivs or [])
    if merged and int(merged[0][0]) == 0:
        # Mirror _pass3_clear_states_in_hf_noise: ignore the first noise span at file start.
        merged = merged[1:]

    t_list: List[float] = []
    bpm_list: List[float] = []
    sr = float(sample_rate)
    for i in range(len(peaks) - 1):
        a = int(peaks[i])
        b = int(peaks[i + 1])
        if b <= a:
            continue
        if merged and _span_intersects_merged_noise(a, b, merged, n_samples):
            continue
        rr_sec = (b - a) / sr
        if not np.isfinite(rr_sec) or rr_sec <= 0:
            continue
        bpm = 60.0 / rr_sec
        if not np.isfinite(bpm) or bpm <= 0:
            continue
        t_mid = 0.5 * (a + b) / sr
        t_list.append(float(t_mid))
        bpm_list.append(float(bpm))

    if len(t_list) < 2:
        return None
    t = np.asarray(t_list, dtype=np.float64)
    y = np.asarray(bpm_list, dtype=np.float64)
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    # Reuse the measured-phase cleaner (MAD + light Gaussian smoothing).
    # We pass noise_ivs=None because RR intervals were already excluded by intersection tests.
    t2, y2 = _pass3_clean_duration_series(t, y, noise_ivs=None, sample_rate=sample_rate, params=(params or {}))
    if len(t2) < 2:
        return None
    return pd.Series(y2, index=t2, dtype=float)


def _dense_bpm_raster_from_series(
    lt_series: Optional[pd.Series],
    n_samples: int,
    sample_rate: int,
    fallback_bpm: float,
    dt_sec: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a dense BPM raster (times, bpm_values) sampled every dt_sec across the file.

    This makes the BPM prior explicit and continuous-in-time for later consumers.
    Values are obtained by linear interpolation of lt_series with constant edge extrapolation.
    """
    sr = float(sample_rate)
    if sr <= 0 or n_samples <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    dur_sec = float(n_samples) / sr
    if not np.isfinite(dur_sec) or dur_sec <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    dt = float(dt_sec)
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.05

    t_grid = np.arange(0.0, dur_sec + 1e-12, dt, dtype=np.float64)
    if lt_series is None or getattr(lt_series, "empty", True):
        return t_grid, np.full_like(t_grid, float(fallback_bpm), dtype=np.float64)
    try:
        times = np.asarray(lt_series.index.values, dtype=np.float64)
        values = np.asarray(lt_series.values, dtype=np.float64)
        if len(times) < 2 or len(times) != len(values):
            return t_grid, np.full_like(t_grid, float(fallback_bpm), dtype=np.float64)
        bpm_grid = np.interp(t_grid, times, values, left=float(values[0]), right=float(values[-1])).astype(np.float64)
        bpm_grid[~np.isfinite(bpm_grid)] = float(fallback_bpm)
        bpm_grid[bpm_grid <= 0] = float(fallback_bpm)
        return t_grid, bpm_grid
    except Exception:
        return t_grid, np.full_like(t_grid, float(fallback_bpm), dtype=np.float64)


def _dense_raster_from_points(
    t_points: np.ndarray,
    y_points: np.ndarray,
    n_samples: int,
    sample_rate: int,
    *,
    dt_sec: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a dense raster (times, values) sampled every dt_sec across the file from sparse points.

    Uses linear interpolation with constant edge extrapolation (np.interp).
    Returns empty arrays when there are not enough points to interpolate (need >=2).
    """
    sr = float(sample_rate)
    if sr <= 0 or n_samples <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    dur_sec = float(n_samples) / sr
    if not np.isfinite(dur_sec) or dur_sec <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    dt = float(dt_sec)
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.05

    t_points = np.asarray(t_points, dtype=np.float64)
    y_points = np.asarray(y_points, dtype=np.float64)
    if len(t_points) < 2 or len(t_points) != len(y_points):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    t_grid = np.arange(0.0, dur_sec + 1e-12, dt, dtype=np.float64)
    order = np.argsort(t_points)
    t_points = t_points[order]
    y_points = y_points[order]
    y_grid = np.interp(t_grid, t_points, y_points, left=float(y_points[0]), right=float(y_points[-1])).astype(np.float64)
    return t_grid, y_grid


# ─────────────────────────────────────────────────────────────────────────────
# Noise repair — clear state labels inside HF-noise windows and rebuild
# ─────────────────────────────────────────────────────────────────────────────

def _pass3_clear_states_in_hf_noise(
    state_labels: np.ndarray,
    state_boundaries: List[Tuple],
    noise_ivs: List[Tuple[int, int]],
    n_samples: int,
    s1_peaks: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple]]:
    """
    HF noise repair (merged preprocessing windows):

    - If the **S1** phase of a beat intersects noise, clear dense labels from painted S1 start
      through the next S1 peak and drop all four boundary segments for that beat.

    - If **diastole** intersects noise **and S1 does not**, clear only **after** that beat's
      own S1 phase through the next S1 peak ``[s1_end, s1_next)``, keep the **S1** segment,
      and drop systole / S2 / diastole segments for that beat only (anchor = this beat's S1).

    The **first merged** HF-noise interval **starting at sample 0** is ignored for this
    repair only: clearing there would remove early S1 runs and Pass 3 state-timeline BPM would
    have no points for the start of the file. Later noise windows still trigger clearing.
    """
    merged = _merge_sorted_intervals(noise_ivs)
    if not merged:
        return state_labels, state_boundaries
    if int(merged[0][0]) == 0:
        merged = merged[1:]
    if not merged:
        return state_labels, state_boundaries

    peaks = np.asarray(s1_peaks, dtype=np.int64)
    n_cyc = max(0, len(peaks) - 1)

    # One pass over boundaries: O(n) maps for cycle lookups (avoids re-scanning per beat).
    s1_span: Dict[int, Tuple[int, int]] = {}
    dia_span: Dict[int, Tuple[int, int]] = {}
    for seg in state_boundaries:
        st = seg[2]
        if st not in ("S1", "diastole"):
            continue
        bm = seg[3] if isinstance(seg[3], dict) else {}
        pk = bm.get("s1")
        if pk is None:
            continue
        pk_i = int(pk)
        a0 = max(0, min(int(seg[0]), n_samples))
        a1 = max(0, min(int(seg[1]), n_samples))
        if a1 <= a0:
            continue
        if st == "S1":
            s1_span[pk_i] = (a0, a1)
        else:
            dia_span[pk_i] = (a0, a1)

    s1_touch: set = set()
    dia_touch: set = set()
    for i in range(n_cyc):
        pk = int(peaks[i])
        s1_next = int(peaks[i + 1])
        if s1_next <= pk:
            continue
        se = s1_span.get(pk)
        if se is None:
            continue
        s0, e0 = se
        if _span_intersects_merged_noise(s0, e0, merged, n_samples):
            s1_touch.add(pk)
        de = dia_span.get(pk)
        if de is not None:
            d0, d1 = de
            if _span_intersects_merged_noise(d0, d1, merged, n_samples):
                dia_touch.add(pk)

    # Full beat only when S1 touches noise; diastole-only → clear states after the S1 for that beat.
    dia_only = dia_touch - s1_touch
    if not s1_touch and not dia_only:
        return state_labels, state_boundaries

    # Dense: S1 noise → clear [S1 start .. next peak)
    for i in range(n_cyc):
        pk = int(peaks[i])
        if pk not in s1_touch:
            continue
        s1_next = int(peaks[i + 1])
        se = s1_span.get(pk)
        if se is None or s1_next <= se[0]:
            continue
        lo_clr = max(0, se[0])
        hi_clr = min(n_samples, s1_next)
        if hi_clr > lo_clr:
            state_labels[lo_clr:hi_clr] = STATE_UNKNOWN

    # Dense: diastole noise only (S1 clean) → clear [s1_end .. next peak), keep S1 labels
    for i in range(n_cyc):
        pk = int(peaks[i])
        if pk not in dia_only:
            continue
        s1_next = int(peaks[i + 1])
        se = s1_span.get(pk)
        if se is None:
            continue
        _, s1_end = se
        if s1_next <= s1_end:
            continue
        lo_p = max(0, s1_end)
        hi_p = min(n_samples, s1_next)
        if hi_p > lo_p:
            state_labels[lo_p:hi_p] = STATE_UNKNOWN

    new_bd: List[Tuple] = []
    for seg in state_boundaries:
        st_name = seg[2]
        meta = seg[3] if isinstance(seg[3], dict) else {}
        s1_key = meta.get("s1")
        if s1_key is not None:
            sk = int(s1_key)
            if sk in s1_touch:
                continue
            if sk in dia_only and st_name in ("systole", "S2", "diastole"):
                continue
        new_bd.append(seg)
    return state_labels, new_bd


# ─────────────────────────────────────────────────────────────────────────────
# Correction reasoning record builders (for debug overlay in HTML plots)
# ─────────────────────────────────────────────────────────────────────────────

def _pass3_rebuild_reasoning(
    phase: str,
    ivs: Dict,
    sample_rate: int,
    lo: int,
    hi: int,
    note: str,
) -> Dict[str, Any]:
    """Per-phase hover payload for segments filled by state rebuild (matches cardiac strip tooltip)."""
    sr = float(sample_rate)
    meas_ms = int(round((hi - lo) / sr * 1000.0)) if hi > lo else 0
    if phase == "S1":
        exp_ms = int(round(float(ivs.get("s1_nominal", 0.040)) * 1000.0))
    elif phase == "systole":
        exp_ms = int(round(float(ivs.get("s1_s2_nominal", 0.300)) * 1000.0))
    elif phase == "S2":
        exp_ms = int(round(float(ivs.get("s2_nominal", 0.030)) * 1000.0))
    else:
        exp_ms = int(round(float(ivs.get("s2_s1_nominal", 0.400)) * 1000.0))
    return {
        "expected_ms": exp_ms,
        "measured_ms": meas_ms,
        "notes": [
            str(note),
        ],
    }


def _pass3_noise_rebuild_reasoning(
    phase: str,
    ivs: Dict,
    sample_rate: int,
    lo: int,
    hi: int,
) -> Dict[str, Any]:
    return _pass3_rebuild_reasoning(
        phase, ivs, sample_rate, lo, hi,
        "Regenerated in noisy region (noise repair).",
    )


def _pass3_gap_insert_rebuild_reasoning(
    phase: str,
    ivs: Dict,
    sample_rate: int,
    lo: int,
    hi: int,
) -> Dict[str, Any]:
    return _pass3_rebuild_reasoning(
        phase, ivs, sample_rate, lo, hi,
        "Inserted missing cardiac cycle(s) in a large gap.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Duration series cleaning and phase-curve measurement
# ─────────────────────────────────────────────────────────────────────────────

def _pass3_clean_duration_series(
    t_raw: np.ndarray,
    d_raw: np.ndarray,
    noise_ivs: Optional[List[Tuple[int, int]]],
    sample_rate: int,
    params: Optional[Dict] = None,
    *,
    duration_kind: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shared post-processing for measured systole / diastole series:
      0. Optional: drop durations outside wide systole/diastole clamps (when duration_kind is set).
      1. Drop points whose time falls inside a noise interval.
      2. Local median ± k*MAD in a rolling time window (same helper as systole interval curve).
      3. Gaussian-weighted rolling mean (sigma=1 beat, half-window=3 beats) — light smoothing
         that keeps values close to measured data while removing beat-to-beat jitter.

    When duration_kind is None, step 0 is skipped (used e.g. for BPM series cleaning).
    Input arrays must already be sorted by time.
    """
    if len(t_raw) == 0:
        return t_raw.copy(), d_raw.copy()

    t = t_raw.copy()
    d = d_raw.copy()
    pc = params or {}

    if duration_kind in ("systole", "diastole"):
        t, d = filter_interval_durations_by_limits(
            t, d, kind=str(duration_kind), params=pc,
        )
        if len(t) == 0:
            return t, d

    # 1. Remove points inside noise intervals.
    if noise_ivs:
        sr = float(sample_rate)
        keep = np.ones(len(t), dtype=bool)
        for lo_samp, hi_samp in noise_ivs:
            lo_sec = lo_samp / sr
            hi_sec = hi_samp / sr
            keep &= ~((t >= lo_sec) & (t < hi_sec))
        t = t[keep]
        d = d[keep]

    if len(t) == 0:
        return t, d

    # 2. Local MAD outlier removal (time window around each point).
    half_win = float(pc.get(
        "pass3_measured_phase_outlier_window_sec",
        pc.get("systole_outlier_window_sec", pc.get("s1_s2_outlier_window_sec", 8.0)),
    ))
    mad_k = float(pc.get(
        "pass3_measured_phase_outlier_mad_k",
        pc.get("systole_outlier_mad_k", pc.get("s1_s2_outlier_mad_k", 2.5)),
    ))
    if len(t) >= 2:
        keep = _median_mad_keep_mask_time_window(t, d, half_win, mad_k)
        t = t[keep]
        d = d[keep]

    if len(t) == 0:
        return t, d

    # 3. Gaussian-weighted rolling mean (light smoothing — keeps data close to measured values).
    #    Kernel sigma = 1 beat (window = 5 beats each side at ≥ 3σ), edge samples handled via
    #    explicit per-point weighted sum so boundaries don't pull toward zero.
    if len(d) >= 3:
        sigma = 1.0  # beats; small → stays close to data
        half_kern = max(1, int(math.ceil(3.0 * sigma)))
        kernel_idx = np.arange(-half_kern, half_kern + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (kernel_idx / sigma) ** 2)
        d_smooth = np.empty_like(d)
        for i in range(len(d)):
            lo = max(0, i - half_kern)
            hi = min(len(d), i + half_kern + 1)
            k_lo = lo - (i - half_kern)
            k_hi = k_lo + (hi - lo)
            w = kernel[k_lo:k_hi]
            d_smooth[i] = float(np.dot(d[lo:hi], w) / w.sum())
        d = d_smooth

    return t, d


def _pass3_measured_systole_series_from_boundaries(
    state_boundaries: List[Tuple],
    sample_rate: int,
    noise_ivs: Optional[List[Tuple[int, int]]] = None,
    params: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (times_sec, durations_sec) for *measured systole* from the current state boundaries.
    Noisy points are dropped, outliers removed, and the series is rolling-median smoothed.
    """
    sr = float(sample_rate)
    t_list: List[float] = []
    d_list: List[float] = []
    for s0, s1, st, _meta in (state_boundaries or []):
        if st != "systole":
            continue
        a0 = float(s0); a1 = float(s1)
        if not np.isfinite(a0) or not np.isfinite(a1) or a1 <= a0:
            continue
        t_mid = (a0 + a1) / 2.0 / sr
        dur   = (a1 - a0) / sr
        if not np.isfinite(t_mid) or not np.isfinite(dur) or dur <= 0:
            continue
        t_list.append(float(t_mid))
        d_list.append(float(dur))
    if not t_list:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    t = np.asarray(t_list, dtype=np.float64)
    d = np.asarray(d_list, dtype=np.float64)
    order = np.argsort(t)
    return _pass3_clean_duration_series(
        t[order], d[order], noise_ivs, sample_rate, params=params, duration_kind="systole",
    )


def _pass3_measured_diastole_series_from_boundaries(
    state_boundaries: List[Tuple],
    sample_rate: int,
    noise_ivs: Optional[List[Tuple[int, int]]] = None,
    params: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (times_sec, durations_sec) for *measured diastole* from the current state boundaries.
    Noisy points are dropped, outliers removed, and the series is rolling-median smoothed.
    """
    sr = float(sample_rate)
    t_list: List[float] = []
    d_list: List[float] = []
    for s0, s1, st, _meta in (state_boundaries or []):
        if st != "diastole":
            continue
        a0 = float(s0); a1 = float(s1)
        if not np.isfinite(a0) or not np.isfinite(a1) or a1 <= a0:
            continue
        t_mid = (a0 + a1) / 2.0 / sr
        dur   = (a1 - a0) / sr
        if not np.isfinite(t_mid) or not np.isfinite(dur) or dur <= 0:
            continue
        t_list.append(float(t_mid))
        d_list.append(float(dur))
    if not t_list:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    t = np.asarray(t_list, dtype=np.float64)
    d = np.asarray(d_list, dtype=np.float64)
    order = np.argsort(t)
    return _pass3_clean_duration_series(
        t[order], d[order], noise_ivs, sample_rate, params=params, duration_kind="diastole",
    )


# ─────────────────────────────────────────────────────────────────────────────
# State boundary geometry helpers (piecewise interpolation, trim, removal)
# ─────────────────────────────────────────────────────────────────────────────

def _interp_piecewise_linear(
    t_query: float,
    t: np.ndarray,
    y: np.ndarray,
) -> Optional[float]:
    """1D linear interpolation with constant edge extrapolation; returns None if empty."""
    if t is None or y is None or len(t) < 2 or len(t) != len(y):
        return None
    tq = float(t_query)
    val = float(np.interp(tq, t, y, left=float(y[0]), right=float(y[-1])))
    if not np.isfinite(val):
        return None
    return val


def _pass3_trim_diastole_ends_on_next_s1(state_boundaries: List[Tuple]) -> List[Tuple]:
    """Shorten diastole segment ends so they do not overlap the next S1 segment (same as main Pass 3 trim)."""
    _trimmed: List[Tuple] = []
    for _bi, _seg in enumerate(state_boundaries):
        if _seg[2] == "diastole":
            _dia_start, _dia_end, _dia_name, _dia_meta = _seg
            _next_s1_start = _dia_end
            for _fwd in range(_bi + 1, len(state_boundaries)):
                if state_boundaries[_fwd][2] == "S1":
                    _next_s1_start = state_boundaries[_fwd][0]
                    break
            _new_end = min(_dia_end, _next_s1_start)
            if _new_end > _dia_start:
                _trimmed.append((_dia_start, _new_end, _dia_name, _dia_meta))
        else:
            _trimmed.append(_seg)
    return _trimmed


def _pass3_remove_boundaries_overlapping_span(
    state_boundaries: List[Tuple], lo: int, hi: int,
) -> List[Tuple]:
    out: List[Tuple] = []
    for seg in state_boundaries:
        a0, a1 = int(seg[0]), int(seg[1])
        if a0 < hi and a1 > lo:
            continue
        out.append(seg)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Gap detection and filling — find, label, and rebuild long unknown spans
# ─────────────────────────────────────────────────────────────────────────────

def _pass3_find_gap_windows(
    state_boundaries: List[Tuple],
    n_samples: int,
    bpm_prior_raster: Tuple[np.ndarray, np.ndarray],
    fallback_bpm: float,
    sample_rate: int,
    params: Dict,
    audio_envelope: np.ndarray,
    *,
    measured_systole_t: Optional[np.ndarray] = None,
    measured_systole_dur: Optional[np.ndarray] = None,
    measured_diastole_t: Optional[np.ndarray] = None,
    measured_diastole_dur: Optional[np.ndarray] = None,
    dynamic_noise_floor_series: Optional[pd.Series] = None,
    quiet_windows_out: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Identify candidate gap windows from state_boundaries using the richer extra-cycle
    width test and quiet/sensitive-peak prefix trim. Does NOT mutate state_labels.
    No wall-time duration filter is applied here — callers filter by
    gap_region_duration_sec (> threshold for labeling path, <= threshold for insert path).

    Returned dict keys per window:
      start_sample            — trimmed gap_region_lo (after quiet prefix trim)
      end_sample              — gap_region_hi (= source segment end, before trim)
      source_seg_start        — original segment start (a0)
      source_seg_end          — original segment end (a1)
      segment_name            — phase name
      segment_duration_sec    — wall duration of original segment
      gap_region_duration_sec — wall duration of trimmed window
      bpm_at_mid              — BPM at segment midpoint
      cyc0_samples            — nominal cycle length in samples
      n_sensitive_peaks       — sensitive peaks inside trimmed window (used to cap rebuild cycles)
      first_sensitive_peak_sample
      meta                    — original boundary meta dict
      s1_right                — meta["s1_next"] if present (next S1 anchor for labeling path)
    """
    if n_samples <= 0 or sample_rate <= 0:
        return []

    t_r, bpm_r = bpm_prior_raster
    t_r = np.asarray(t_r, dtype=np.float64)
    bpm_r = np.asarray(bpm_r, dtype=np.float64)
    if len(t_r) < 2 or len(t_r) != len(bpm_r):
        return []

    SR = float(sample_rate)
    MAX_SCALE = 0.30  # must match _pass3_rebuild_unknown_runs
    _q_sens = float(params.get("pass3_gap_recovery_peak_prominence_quantile_sensitive", 0.50))

    def _bpm_at_sample(ix: int) -> float:
        t = float(ix) / SR
        v = _interp_piecewise_linear(t, t_r, bpm_r)
        if v is None:
            return float(fallback_bpm)
        if not np.isfinite(v) or v <= 0:
            return float(fallback_bpm)
        return float(v)

    result: List[Dict[str, Any]] = []

    for seg in state_boundaries:
        a0, a1, name, meta = seg
        a0 = int(max(0, min(int(a0), n_samples)))
        a1 = int(max(0, min(int(a1), n_samples)))
        if a1 <= a0:
            continue
        if name not in ("S1", "systole", "S2", "diastole"):
            continue

        seg_dur_sec = (a1 - a0) / SR
        width = a1 - a0
        t_mid = 0.5 * (a0 + a1) / SR
        bpm = _bpm_at_sample(int((a0 + a1) // 2))
        ivs = calculate_bpm_intervals(bpm, params)

        s1_sec = float(ivs.get("s1_nominal", 0.040))
        s2_sec = float(ivs.get("s2_nominal", 0.030))
        sys_sec = _interp_piecewise_linear(t_mid, measured_systole_t, measured_systole_dur)
        if sys_sec is None:
            sys_sec = float(ivs.get("s1_s2_nominal", 0.300))
        dia_sec = _interp_piecewise_linear(t_mid, measured_diastole_t, measured_diastole_dur)
        if dia_sec is None:
            rr_sec = 60.0 / bpm if bpm > 0 else 0.75
            dia_sec = float(ivs.get("s2_s1_nominal", max(0.0, rr_sec - float(sys_sec))))
        cyc0 = (
            max(1, int(round(s1_sec * SR)))
            + max(1, int(round(sys_sec * SR)))
            + max(1, int(round(s2_sec * SR)))
            + max(1, int(round(dia_sec * SR)))
        )

        min_extra = int(round((1.0 - MAX_SCALE) * float(max(1, cyc0))))
        need_wide = max(4, min_extra)
        if width < need_wide:
            continue

        gap_region_lo = a0
        gap_region_hi = a1
        gap_region_lo_orig = int(gap_region_lo)

        try:
            _pk_sens = _detect_sensitive_peaks_in_large_gap_windows(
                audio_envelope, sample_rate,
                [{"start_sample": int(gap_region_lo), "end_sample": int(gap_region_hi)}],
                params,
                prominence_quantile=_q_sens,
                dynamic_noise_floor_series=dynamic_noise_floor_series,
            )
        except Exception:
            _pk_sens = np.asarray([], dtype=np.int64)

        if _pk_sens is None or len(_pk_sens) == 0:
            if quiet_windows_out is not None:
                try:
                    quiet_windows_out.append({
                        "start_sample": int(gap_region_lo_orig),
                        "end_sample": int(gap_region_hi),
                        "gap_region_candidate_state": str(name),
                        "trigger": "quiet_entire_gap_region",
                    })
                except Exception:
                    pass
            continue

        try:
            first_pk_i = int(np.min(np.asarray(_pk_sens, dtype=np.int64)))
        except Exception:
            continue

        s1_sec_nom = float(ivs.get("s1_nominal", 0.040))
        s1_expected = max(1, int(round(s1_sec_nom * SR)))
        pre_pad = max(0, int(s1_expected // 2))
        new_gap_region_lo = int(first_pk_i) - int(pre_pad)
        new_gap_region_lo = int(max(int(a0), min(int(new_gap_region_lo), int(gap_region_hi))))

        if new_gap_region_lo > gap_region_lo_orig and quiet_windows_out is not None:
            try:
                quiet_windows_out.append({
                    "start_sample": int(gap_region_lo_orig),
                    "end_sample": int(new_gap_region_lo),
                    "gap_region_candidate_state": str(name),
                    "trigger": "trim_quiet_prefix",
                    "first_sensitive_peak_sample": int(first_pk_i),
                    "s1_expected_samples": int(s1_expected),
                    "pre_pad_samples": int(pre_pad),
                })
            except Exception:
                pass

        gap_region_lo = int(new_gap_region_lo)

        if gap_region_hi - gap_region_lo < 4:
            continue

        try:
            _pk_sens_arr = np.asarray(_pk_sens, dtype=np.int64)
            n_sens = int(np.sum(
                (_pk_sens_arr >= int(gap_region_lo)) & (_pk_sens_arr < int(gap_region_hi))
            ))
        except Exception:
            n_sens = 0

        s1_right: Optional[int] = None
        if isinstance(meta, dict) and meta.get("s1_next") is not None:
            s1_right = int(meta["s1_next"])

        gap_dur_sec = (gap_region_hi - gap_region_lo) / SR

        result.append({
            "start_sample": int(gap_region_lo),
            "end_sample": int(gap_region_hi),
            "source_seg_start": int(a0),
            "source_seg_end": int(a1),
            "segment_name": str(name),
            "segment_duration_sec": float(seg_dur_sec),
            "gap_region_duration_sec": float(gap_dur_sec),
            "bpm_at_mid": float(bpm),
            "cyc0_samples": int(cyc0),
            "n_sensitive_peaks": int(n_sens),
            "first_sensitive_peak_sample": int(first_pk_i),
            "meta": dict(meta) if isinstance(meta, dict) else {},
            "s1_right": s1_right,
        })

    return result


def _pass3_apply_peaks_labeling_in_large_gaps(
    state_labels: np.ndarray,
    state_boundaries: List[Tuple],
    *,
    gap_windows: List[Dict[str, Any]],
    n_samples: int,
    sample_rate: int,
    params: Dict,
    audio_envelope: np.ndarray,
    analysis_data: Dict,
    s1_list: List[int],
    lt_pass3: Any,
    fallback_bpm: float,
    s1_s2_pairs: List,
    corrections: List[Dict],
    before_s2_by_s1: Dict,
    before_s1next_by_s1: Dict,
    s1_half: int,
    s2_half: int,
    edge_alpha: float,
    edge_n_exp: float,
    s1_min_half: int,
    s1_max_half: int,
    s2_min_half: int,
    s2_max_half: int,
) -> Tuple[np.ndarray, List[Tuple]]:
    """
    Pass 3 — for large gaps (> threshold) from the pre-computed gap_windows list, run
    insensitive peaks + the same S1/S2/Noise labeling as Pass 2, then paint cardiac states
    into the gap. Runs after noise repair, consuming shared gap windows.
    """
    if not bool(params.get("pass3_enable_peaks_labeling_in_large_gaps", True)):
        return state_labels, state_boundaries
    if not gap_windows:
        return state_labels, state_boundaries

    min_sec = float(params.get("pass3_peaks_labeling_in_large_gaps_min_sec", 10.0))
    nf = analysis_data.get("dynamic_noise_floor_series")
    tr = analysis_data.get("trough_indices")
    if nf is None or tr is None or not isinstance(nf, pd.Series):
        return state_labels, state_boundaries
    tr_arr = np.asarray(tr, dtype=np.int64)
    if tr_arr.size == 0:
        return state_labels, state_boundaries

    # Filter for gaps exceeding the large-gap threshold.
    large_gaps = [
        gw for gw in gap_windows
        if float(gw.get("gap_region_duration_sec", 0.0)) > min_sec
        and gw.get("s1_right") is not None
    ]
    if not large_gaps:
        return state_labels, state_boundaries

    large_gaps.sort(key=lambda g: int(g.get("start_sample", 0)))

    # Mark all large-gap regions as STATE_UNKNOWN so state_boundaries no longer covers them.
    for gw in large_gaps:
        lo = int(gw["start_sample"])
        hi = int(gw["end_sample"])
        if hi > lo:
            state_labels[lo:hi] = STATE_UNKNOWN

    _sr_f = float(sample_rate)
    _SHIFT_THRESH = int(0.020 * sample_rate)
    s1_s2_pairs = s1_s2_pairs or []
    _corrs_by_s1: Dict[int, List[Dict]] = {}
    for _c in corrections:
        _ck = _c.get("s1") if _c.get("s1") is not None else _c.get("s1_prev")
        if _ck is not None:
            _corrs_by_s1.setdefault(int(_ck), []).append(_c)
    all_corr_sorted = sorted(
        [(float(_c.get("s1", _c.get("s1_prev", 0))) / _sr_f, _c) for _c in corrections],
        key=lambda x: x[0],
    )
    q_ins = float(params.get("pass3_gap_recovery_peak_prominence_quantile_insensitive", 0.70))
    bd: List[Tuple] = list(state_boundaries)

    for gw in large_gaps:
        lo = int(gw["start_sample"])
        hi = int(gw["end_sample"])
        s1_right = int(gw["s1_right"])  # safe: filtered above
        src_lo = int(gw.get("source_seg_start", lo))
        src_name = str(gw.get("segment_name", "diastole"))
        src_meta = dict(gw.get("meta", {}))
        if hi <= lo or s1_right <= lo:
            continue

        # Keep quiet-trimmed prefix from the original source segment so boundaries remain
        # contiguous when labeling starts at trimmed `lo` (common for diastole).
        prefix_seg: Optional[Tuple] = None
        if src_lo < lo:
            prefix_meta = dict(src_meta)
            if "reasoning" in prefix_meta and isinstance(prefix_meta["reasoning"], dict):
                _r = dict(prefix_meta["reasoning"])
                _r["measured_ms"] = round((lo - src_lo) / _sr_f * 1000.0)
                prefix_meta["reasoning"] = _r
            if src_name == "diastole" and "s1_next" in prefix_meta:
                prefix_meta["s1_next"] = int(lo)
            prefix_seg = (int(src_lo), int(lo), src_name, prefix_meta)

        w = [{"start_sample": lo, "end_sample": hi}]
        pkins = _detect_sensitive_peaks_in_large_gap_windows(
            audio_envelope, sample_rate, w, params,
            prominence_quantile=q_ins,
            dynamic_noise_floor_series=analysis_data.get("dynamic_noise_floor_series"),
        )
        if pkins is None or len(pkins) == 0:
            bd = _pass3_remove_boundaries_overlapping_span(bd, lo, hi)
            if prefix_seg is not None and prefix_seg[1] > prefix_seg[0]:
                bd.append(prefix_seg)
                _st_prefix = {
                    "S1": STATE_S1, "systole": STATE_SYSTOLE, "S2": STATE_S2, "diastole": STATE_DIASTOLE,
                }.get(src_name)
                if _st_prefix is not None:
                    state_labels[int(prefix_seg[0]):int(prefix_seg[1])] = _st_prefix
            bd.append((lo, hi, "unknown", {"rebuild_source": "gap_label_pass3"}))
            bd = sorted(bd, key=lambda s: s[0])
            continue

        def _bpm_callable(t_sec: float) -> float:
            return _bpm_at_time(float(t_sec), lt_pass3, float(fallback_bpm))

        cl = PeakClassifier(
            audio_envelope,
            sample_rate,
            params,
            float(fallback_bpm),
            nf,
            tr_arr,
            None,
            None,
            pass1_bpm_prior=_bpm_callable,
            raw_peaks_override=pkins,
        )
        cl._pass3_large_gap = True
        cl.classify_peaks()

        s2_by_s1 = {int(a): int(b) for a, b in cl._build_s1_s2_pairs()}

        cand = sorted(
            {
                int(x) for x in cl.state.candidate_beats
                if lo <= int(x) < min(hi, s1_right)
            }
        )
        if not cand:
            bd = _pass3_remove_boundaries_overlapping_span(bd, lo, hi)
            if prefix_seg is not None and prefix_seg[1] > prefix_seg[0]:
                bd.append(prefix_seg)
                _st_prefix = {
                    "S1": STATE_S1, "systole": STATE_SYSTOLE, "S2": STATE_S2, "diastole": STATE_DIASTOLE,
                }.get(src_name)
                if _st_prefix is not None:
                    state_labels[int(prefix_seg[0]):int(prefix_seg[1])] = _st_prefix
            bd.append((lo, hi, "unknown", {"rebuild_source": "gap_label_pass3"}))
            bd = sorted(bd, key=lambda s: s[0])
            continue

        chain = [int(x) for x in cand if int(x) < s1_right]
        if not chain:
            bd = _pass3_remove_boundaries_overlapping_span(bd, lo, hi)
            if prefix_seg is not None and prefix_seg[1] > prefix_seg[0]:
                bd.append(prefix_seg)
                _st_prefix = {
                    "S1": STATE_S1, "systole": STATE_SYSTOLE, "S2": STATE_S2, "diastole": STATE_DIASTOLE,
                }.get(src_name)
                if _st_prefix is not None:
                    state_labels[int(prefix_seg[0]):int(prefix_seg[1])] = _st_prefix
            bd.append((lo, hi, "unknown", {"rebuild_source": "gap_label_pass3"}))
            bd = sorted(bd, key=lambda s: s[0])
            continue
        chain.append(s1_right)

        bd = _pass3_remove_boundaries_overlapping_span(bd, lo, hi)
        if prefix_seg is not None and prefix_seg[1] > prefix_seg[0]:
            bd.append(prefix_seg)
            _st_prefix = {
                "S1": STATE_S1, "systole": STATE_SYSTOLE, "S2": STATE_S2, "diastole": STATE_DIASTOLE,
            }.get(src_name)
            if _st_prefix is not None:
                state_labels[int(prefix_seg[0]):int(prefix_seg[1])] = _st_prefix
        new_segs: List[Tuple] = []
        for j in range(len(chain) - 1):
            s1 = chain[j]
            s1_next = chain[j + 1]
            if s1_next <= s1:
                continue
            t_s1 = s1 / _sr_f
            bpm = _bpm_at_time(t_s1, lt_pass3, float(fallback_bpm))
            ivs = calculate_bpm_intervals(bpm, params)
            _cascade: Optional[Dict] = None
            for _ct, _cc in all_corr_sorted:
                if _ct >= t_s1 - 0.010:
                    break
                if _cc not in _corrs_by_s1.get(s1, []):
                    _cascade = _cc
            s2g = s2_by_s1.get(s1)
            if s2g is None:
                s2_pred = int(round(s1 + float(ivs.get("s1_s2_nominal", 0.30)) * _sr_f))
                s2g = int(max(s1 + 1, min(s2_pred, s1_next - 1)))
            else:
                s2g = int(max(s1 + 1, min(int(s2g), s1_next - 1)))

            s1_start, s1_end, s2_start, s2_end = _paint_state_boundaries(
                s1, s2g, s1_next, s1_half, s2_half, n_samples,
                audio_envelope=audio_envelope,
                edge_alpha=edge_alpha, edge_n_exp=edge_n_exp,
                min_s1_half=s1_min_half, max_s1_half=s1_max_half,
                min_s2_half=s2_min_half, max_s2_half=s2_max_half,
                use_transient_detection=True,
            )
            if j == 0 and s1_start > lo:
                _rs_p = {
                    "s1": s1, "s2": s2g, "s1_next": s1,
                    "rebuild_source": "gap_label_pass3",
                }
                new_segs.append(
                    (lo, min(s1_start, n_samples), "diastole", _rs_p),
                )
                if min(s1_start, n_samples) > lo:
                    state_labels[lo:min(s1_start, n_samples)] = STATE_DIASTOLE
            _reasoning = _build_reasoning_payload(
                s1, s1_start, s1_end, s2g, s2_start, s2_end, s1_next,
                ivs, _corrs_by_s1.get(s1, []), _cascade,
                before_s2_by_s1.get(s1), before_s1next_by_s1.get(s1),
                _sr_f, _SHIFT_THRESH,
                hover_debug_geometry=True, hover_s1_s2_pairs=s1_s2_pairs, hover_s1_half=s1_half, hover_s2_half=s2_half, hover_n_samples=n_samples,
            )
            _st_code = {
                "S1": STATE_S1, "systole": STATE_SYSTOLE, "S2": STATE_S2, "diastole": STATE_DIASTOLE,
            }
            for ph in (
                (s1_start, s1_end, "S1", {"s1": s1, "reasoning": _reasoning["S1"], "rebuild_source": "gap_label_pass3"}),
                (s1_end, s2_start, "systole", {"s1": s1, "s2": s2g, "reasoning": _reasoning["systole"], "rebuild_source": "gap_label_pass3"}),
                (s2_start, s2_end, "S2", {"s1": s1, "s2": s2g, "reasoning": _reasoning["S2"], "rebuild_source": "gap_label_pass3"}),
                (s2_end, s1_next, "diastole", {
                    "s1": s1, "s2": s2g, "s1_next": s1_next, "reasoning": _reasoning["diastole"], "rebuild_source": "gap_label_pass3",
                }),
            ):
                a, b, nm, m = ph
                if b > a:
                    new_segs.append((a, b, nm, m))
                    c0 = _st_code.get(nm)
                    if c0 is not None:
                        state_labels[a:min(b, n_samples)] = c0
        bd.extend(new_segs)
        bd = sorted(bd, key=lambda s: s[0])
        bd = _pass3_trim_diastole_ends_on_next_s1(bd)

    return state_labels, bd


def _pass3_rebuild_unknown_runs(
    state_labels: np.ndarray,
    state_boundaries: List[Tuple],
    n_samples: int,
    lt_series: Optional[pd.Series],
    fallback_bpm: float,
    sample_rate: int,
    params: Dict,
    measured_systole_t: Optional[np.ndarray] = None,
    measured_systole_dur: Optional[np.ndarray] = None,
    measured_diastole_t: Optional[np.ndarray] = None,
    measured_diastole_dur: Optional[np.ndarray] = None,
    *,
    rebuild_source: str = "noise_repair",
    max_cycles_by_gap: Optional[Dict[Tuple[int, int], int]] = None,
) -> Tuple[np.ndarray, List[Tuple]]:
    """
    Fill every STATE_UNKNOWN run in *state_labels* with a scaled cardiac sequence.

    Strategy (simple, gap-first)
    ----------------------------
    For each gap [gap_lo, gap_hi):
      - Determine which state immediately precedes gap_lo. If it is S1, the gap must start
        with systole (we kept the S1 but cleared the rest). Otherwise, the gap starts with S1.
      - Generate forward from gap_lo by advancing a cursor, painting states back-to-back,
        and stopping exactly at gap_hi (no precomputed cycle boundaries).
      - Choose how many full cycles to pack into the gap and apply a single scale factor
        (±30% preferred) so the generated sequence exactly fills the gap.

    State segments represent durations [start, end). The metadata fields "s1" and "s2"
    are peak *center* indices used as anchors for downstream derivation and hover display;
    they are not the segment boundaries themselves.

    Rebuilt segments carry ``rebuild_source`` metadata so the UI can distinguish
    noise repair vs gap repair on hover (and other downstream debug consumers).
    If the gap is smaller than one minimum feasible cardiac cycle it is left
    as STATE_UNKNOWN.
    """
    MAX_SCALE = 0.30
    SR = float(sample_rate)

    unknown_mask = (state_labels == STATE_UNKNOWN)
    if not np.any(unknown_mask):
        return state_labels, state_boundaries

    # Fast numpy run-length encoding
    padded = np.concatenate([[False], unknown_mask, [False]])
    diff   = np.diff(padded.astype(np.int8))
    starts = np.where(diff ==  1)[0]
    ends   = np.where(diff == -1)[0]
    gaps   = list(zip(starts.tolist(), ends.tolist()))

    new_segs: List[Tuple] = []

    for gap_lo, gap_hi in gaps:
        gap_width = gap_hi - gap_lo
        if gap_width <= 0:
            continue

        # Step A: determine which state precedes the gap (S1 => start at systole).
        start_after_s1 = (gap_lo > 0 and int(state_labels[gap_lo - 1]) == STATE_S1)

        # Find the "anchor" S1 peak index when the gap begins right after an S1 segment.
        anchor_s1_pk: Optional[int] = None
        if start_after_s1:
            for _seg in reversed(state_boundaries):
                if _seg[2] == "S1" and int(_seg[1]) == gap_lo and isinstance(_seg[3], dict) and "s1" in _seg[3]:
                    anchor_s1_pk = int(_seg[3]["s1"])
                    break

        # Read expected durations from BPM + measured curves at gap midpoint.
        t_mid  = (gap_lo + gap_hi) / 2.0 / SR
        bpm    = _bpm_at_time(t_mid, lt_series, fallback_bpm)
        ivs    = calculate_bpm_intervals(bpm, params)
        rr_sec = 60.0 / bpm

        s1_sec = float(ivs.get("s1_nominal", 0.040))
        s2_sec = float(ivs.get("s2_nominal", 0.030))
        sys_sec = _interp_piecewise_linear(t_mid, measured_systole_t, measured_systole_dur)
        if sys_sec is None:
            sys_sec = float(ivs.get("s1_s2_nominal", 0.300))
        dia_sec = _interp_piecewise_linear(t_mid, measured_diastole_t, measured_diastole_dur)
        if dia_sec is None:
            dia_sec = float(ivs.get("s2_s1_nominal", max(0.0, rr_sec - float(sys_sec))))

        # Nominal (unscaled) phase lengths in samples.
        p0_s1  = max(1, int(round(s1_sec  * SR)))
        p0_sys = max(1, int(round(sys_sec * SR)))
        p0_s2  = max(1, int(round(s2_sec  * SR)))
        p0_dia = max(1, int(round(dia_sec * SR)))
        cyc0   = p0_s1 + p0_sys + p0_s2 + p0_dia
        part0  = p0_sys + p0_s2 + p0_dia  # when gap starts after S1

        # Skip gaps that are too small to hold even one minimal phase sequence.
        if gap_width < 4:
            continue

        # Choose how many full cycles to pack in, with one global scale factor for the whole gap.
        # Pattern length = (partial? part0 : 0) + K * cyc0.
        best_K = -1
        best_scale = 1.0
        best_abs = 1e9
        base_len = part0 if start_after_s1 else 0
        # Candidate K around gap_width / cyc0 (plus a few neighbors).
        k_center = int(max(0, round(max(0.0, (gap_width - base_len) / float(max(1, cyc0)))))) if cyc0 > 0 else 0
        # Optional safety cap: if the sensitive peak detector saw N peaks in this gap window,
        # do not generate more than N cardiac cycles. (Computed upstream before any states are generated.)
        k_cap = None
        if max_cycles_by_gap is not None:
            try:
                k_cap = int(max_cycles_by_gap.get((int(gap_lo), int(gap_hi))))
            except Exception:
                k_cap = None
        k_lo = max(0, k_center - 2)
        k_hi = k_center + 3
        if k_cap is not None and k_cap >= 0:
            # Hard cap: cycles <= n_sensitive_peaks for this gap window.
            k_lo = min(k_lo, k_cap)
            k_hi = min(k_hi, k_cap + 1)
        for K in range(k_lo, k_hi):
            pat0 = base_len + K * cyc0
            if pat0 <= 0:
                continue
            sc = gap_width / float(pat0)
            abs_sc = abs(sc - 1.0)
            if abs_sc < best_abs:
                best_abs = abs_sc
                best_scale = sc
                best_K = K
        if best_K < 0:
            continue

        # Prefer staying within ±30% (if not, still proceed with closest).
        if abs(best_scale - 1.0) > MAX_SCALE:
            logging.warning(
                "Pass 3 rebuild: gap [%d, %d) scale %.1f%% exceeds ±30%% (K=%s).",
                gap_lo, gap_hi, (best_scale - 1.0) * 100.0, str(best_K),
            )

        # Scale phase lengths (integers) and paint forward with a cursor until gap_hi.
        def _scaled_phase_samples(sec_val: float) -> int:
            return max(1, int(round(float(sec_val) * SR * best_scale)))

        p_s1  = _scaled_phase_samples(s1_sec)
        p_sys = _scaled_phase_samples(sys_sec)
        p_s2  = _scaled_phase_samples(s2_sec)
        p_dia = _scaled_phase_samples(dia_sec)

        cursor = int(gap_lo)
        end = int(gap_hi)

        # Helper to append a segment, clipped to [gap_lo, gap_hi).
        def _emit(a0: int, a1: int, name: str, meta: Dict[str, Any]) -> None:
            a0 = max(int(gap_lo), int(a0))
            a1 = min(int(gap_hi), int(a1))
            if a1 <= a0:
                return
            state_bound = (a0, a1, name, meta)
            new_segs.append(state_bound)
            if name == "S1":
                state_labels[a0:a1] = STATE_S1
            elif name == "systole":
                state_labels[a0:a1] = STATE_SYSTOLE
            elif name == "S2":
                state_labels[a0:a1] = STATE_S2
            elif name == "diastole":
                state_labels[a0:a1] = STATE_DIASTOLE

        # If we start after a kept S1, generate the remainder of that cycle:
        # systole → S2 → diastole, then continue with full cycles.
        current_s1_pk = anchor_s1_pk if anchor_s1_pk is not None else cursor
        _reason = _pass3_noise_rebuild_reasoning if rebuild_source == "noise_repair" else _pass3_gap_insert_rebuild_reasoning
        if start_after_s1 and cursor < end:
            sys0 = cursor
            sys1 = min(end, sys0 + p_sys)
            s2_1 = min(end, sys1 + p_s2)
            dia1 = min(end, s2_1 + p_dia)
            _emit(sys0, sys1, "systole", {
                "s1": int(current_s1_pk),
                "s2": int(sys1),
                "rebuild_source": str(rebuild_source),
                "reasoning": _reason("systole", ivs, sample_rate, sys0, sys1),
            })
            _emit(sys1, s2_1, "S2", {
                "s1": int(current_s1_pk),
                "s2": int(sys1),
                "rebuild_source": str(rebuild_source),
                "reasoning": _reason("S2", ivs, sample_rate, sys1, s2_1),
            })
            _emit(s2_1, dia1, "diastole", {
                "s1": int(current_s1_pk),
                "s2": int(sys1),
                "s1_next": int(dia1),
                "rebuild_source": str(rebuild_source),
                "reasoning": _reason("diastole", ivs, sample_rate, s2_1, dia1),
            })
            cursor = dia1

        # Otherwise, generate full cycles S1 → systole → S2 → diastole until we reach gap_hi.
        # (We always clamp and stop exactly at gap_hi.)
        while cursor < end:
            s1_pk = cursor
            s1_end = min(end, cursor + p_s1)
            sys_end = min(end, s1_end + p_sys)
            s2_end = min(end, sys_end + p_s2)
            dia_end = min(end, s2_end + p_dia)

            _emit(cursor, s1_end, "S1", {
                "s1": int(s1_pk),
                "rebuild_source": str(rebuild_source),
                "reasoning": _reason("S1", ivs, sample_rate, cursor, s1_end),
            })
            _emit(s1_end, sys_end, "systole", {
                "s1": int(s1_pk),
                "s2": int(sys_end),
                "rebuild_source": str(rebuild_source),
                "reasoning": _reason("systole", ivs, sample_rate, s1_end, sys_end),
            })
            _emit(sys_end, s2_end, "S2", {
                "s1": int(s1_pk),
                "s2": int(sys_end),
                "rebuild_source": str(rebuild_source),
                "reasoning": _reason("S2", ivs, sample_rate, sys_end, s2_end),
            })
            _emit(s2_end, dia_end, "diastole", {
                "s1": int(s1_pk),
                "s2": int(sys_end),
                "s1_next": int(dia_end),
                "rebuild_source": str(rebuild_source),
                "reasoning": _reason("diastole", ivs, sample_rate, s2_end, dia_end),
            })
            cursor = dia_end

    if not new_segs:
        return state_labels, state_boundaries

    combined = state_boundaries + new_segs
    combined.sort(key=lambda t: t[0])
    return state_labels, combined


def _pass3_insert_missing_states_in_large_gaps(
    state_labels: np.ndarray,
    state_boundaries: List[Tuple],
    n_samples: int,
    bpm_prior_raster: Tuple[np.ndarray, np.ndarray],
    fallback_bpm: float,
    sample_rate: int,
    params: Dict,
    gap_windows: List[Dict[str, Any]],
    *,
    measured_systole_t: Optional[np.ndarray] = None,
    measured_systole_dur: Optional[np.ndarray] = None,
    measured_diastole_t: Optional[np.ndarray] = None,
    measured_diastole_dur: Optional[np.ndarray] = None,
    debug_windows_out: Optional[List[Dict[str, Any]]] = None,
    dry_run: bool = False,
) -> Tuple[np.ndarray, List[Tuple]]:
    """
    Pass 3 Step 3 — Insert missing *states* in small-to-medium gaps (≤ threshold).

    Consumes pre-computed gap_windows from _pass3_find_gap_windows and filters for
    windows whose gap_region_duration_sec is at or below pass3_peaks_labeling_in_large_gaps_min_sec.
    Marks those regions STATE_UNKNOWN then rebuilds via _pass3_rebuild_unknown_runs.

    When dry_run=True, only fills debug_windows_out; does not mutate state_labels or run rebuild.
    """
    if n_samples <= 0 or sample_rate <= 0:
        logging.info(
            "Pass 3 gap insert: skipped (n_samples=%s, sample_rate=%s).",
            n_samples, sample_rate,
        )
        return state_labels, state_boundaries

    t_r, bpm_r = bpm_prior_raster
    t_r = np.asarray(t_r, dtype=np.float64)
    bpm_r = np.asarray(bpm_r, dtype=np.float64)
    if len(t_r) < 2 or len(t_r) != len(bpm_r):
        logging.info(
            "Pass 3 gap insert: skipped (invalid BPM prior raster: len(t)=%d, len(bpm)=%d).",
            len(t_r), len(bpm_r),
        )
        return state_labels, state_boundaries

    SR = float(sample_rate)
    _gap_insert_max_sec = float(params.get("pass3_peaks_labeling_in_large_gaps_min_sec", 10.0))
    if not np.isfinite(_gap_insert_max_sec) or _gap_insert_max_sec <= 0:
        _gap_insert_max_sec = 10.0

    # Filter for windows at or below the threshold (insert path handles ≤ threshold).
    insert_windows = [
        gw for gw in gap_windows
        if float(gw.get("gap_region_duration_sec", 0.0)) <= _gap_insert_max_sec
    ]

    logging.info(
        "Pass 3 gap insert: %d candidate window(s) (≤ %.1fs) from %d total gap windows.",
        len(insert_windows), _gap_insert_max_sec, len(gap_windows),
    )

    if debug_windows_out is not None:
        for gw in insert_windows:
            try:
                debug_windows_out.append({
                    "start_sample": int(gw["start_sample"]),
                    "end_sample": int(gw["end_sample"]),
                    "sensitive_peaks_count": int(gw.get("n_sensitive_peaks", 0)),
                    "sensitive_first_peak_sample": int(gw.get("first_sensitive_peak_sample", -1)),
                    "gap_region_candidate_state": str(gw.get("segment_name", "")),
                    "trigger": "can_fit_extra_cycle",
                    "bpm_at_mid": float(gw.get("bpm_at_mid", 0.0)),
                    "cycle0_samples": int(gw.get("cyc0_samples", 0)),
                    "segment_samples": int(
                        int(gw.get("source_seg_end", 0)) - int(gw.get("source_seg_start", 0))
                    ),
                    "dry_run": bool(dry_run),
                })
            except Exception:
                logging.debug("Pass 3 gap insert: failed to build window diagnostics", exc_info=True)

    if not insert_windows:
        logging.info("Pass 3 gap insert: no qualifying windows — nothing to insert.")
        return state_labels, state_boundaries

    if dry_run:
        logging.info(
            "Pass 3 gap insert: dry-run complete — %d gap window(s).",
            len(insert_windows),
        )
        return state_labels, state_boundaries

    # Mark each qualifying window as STATE_UNKNOWN and build prefix segments for quiet-trimmed starts.
    max_cycles_by_gap: Dict[Tuple[int, int], int] = {}
    prefix_segs: List[Tuple] = []

    for gw in insert_windows:
        lo = int(gw["start_sample"])
        hi = int(gw["end_sample"])
        if hi <= lo:
            continue
        state_labels[lo:hi] = STATE_UNKNOWN
        max_cycles_by_gap[(lo, hi)] = int(max(0, int(gw.get("n_sensitive_peaks", 0))))

        a0 = int(gw["source_seg_start"])
        name = str(gw["segment_name"])
        meta = dict(gw.get("meta", {}))
        if lo > a0:
            new_meta = dict(meta)
            new_dur_ms = round((lo - a0) / SR * 1000)
            if "reasoning" in new_meta and isinstance(new_meta["reasoning"], dict):
                r = dict(new_meta["reasoning"])
                r["measured_ms"] = new_dur_ms
                new_meta["reasoning"] = r
            if name == "diastole" and "s1_next" in new_meta:
                new_meta["s1_next"] = lo
            prefix_segs.append((a0, lo, name, new_meta))

    # Drop any existing segments overlapping STATE_UNKNOWN regions, then add prefix remnants.
    filtered: List[Tuple] = []
    for seg in state_boundaries:
        a0_, a1_ = int(max(0, min(int(seg[0]), n_samples))), int(max(0, min(int(seg[1]), n_samples)))
        if a1_ <= a0_:
            continue
        if np.any(state_labels[a0_:a1_] == STATE_UNKNOWN):
            continue
        filtered.append(seg)
    filtered.extend(prefix_segs)
    filtered.sort(key=lambda t: int(t[0]))

    state_labels, rebuilt = _pass3_rebuild_unknown_runs(
        state_labels,
        filtered,
        n_samples,
        lt_series=pd.Series(bpm_r, index=t_r, dtype=float),
        fallback_bpm=fallback_bpm,
        sample_rate=sample_rate,
        params=params,
        measured_systole_t=measured_systole_t,
        measured_systole_dur=measured_systole_dur,
        measured_diastole_t=measured_diastole_t,
        measured_diastole_dur=measured_diastole_dur,
        rebuild_source="gap_insert",
        max_cycles_by_gap=max_cycles_by_gap,
    )
    _gap_ins = sum(
        1 for s in rebuilt
        if s[2] == "S1" and isinstance(s[3], dict) and s[3].get("rebuild_source") == "gap_insert"
    )
    logging.info(
        "Pass 3 gap insert: rebuild done (%d window(s), %d rebuilt S1 from gap_insert).",
        len(insert_windows),
        _gap_ins,
    )
    return state_labels, rebuilt


# ─────────────────────────────────────────────────────────────────────────────
# S2 seeding from Pass 2 pairs and pre-correction boundary snapshot
# ─────────────────────────────────────────────────────────────────────────────

def _seed_s2_from_pass2_pairs(
    s1_list: List[int],
    s1_s2_pairs: List[Tuple[int, int]],
    lt_series: Optional[pd.Series],
    fallback_bpm: float,
    sample_rate: int,
    params: Dict,
    n_samples: int,
) -> List[int]:
    """
    Build the initial s2_events list from Pass 2's acoustically-detected S1→S2 (systole) pairs.

    For each beat (s1_list[j] → s1_list[j+1]):
    - If Pass 2 paired an S2 with this S1 and it falls inside the RR window, use it.
    - Otherwise fall back to BPM-nominal ejection time (lone S1, missed S2, etc.).
    """
    pair_map: Dict[int, int] = {int(s1): int(s2) for s1, s2 in (s1_s2_pairs or [])}
    s2_events: List[int] = []
    for j in range(len(s1_list) - 1):
        s1 = int(s1_list[j])
        s1_next = int(s1_list[j + 1])
        if s1_next <= s1:
            s2_events.append(s1)
            continue
        s2_p2 = pair_map.get(s1)
        if s2_p2 is not None and s1 < int(s2_p2) < s1_next:
            s2_events.append(int(s2_p2))
        else:
            t_s1 = s1 / float(sample_rate)
            bpm = _bpm_at_time(t_s1, lt_series, fallback_bpm)
            ivs = calculate_bpm_intervals(bpm, params)
            s2_pred = int(round(s1 + float(ivs.get("s1_s2_nominal", 0.30)) * sample_rate))
            s2_events.append(int(max(s1 + 1, min(s2_pred, s1_next - 1))))
    return s2_events


def _build_state_boundaries_before_from_cycles(
    s1_list: List[int],
    s2_events: List[int],
    s1_half: int,
    s2_half: int,
    n_samples: int,
) -> List[Tuple]:
    """Initial S1/systole/S2/diastole boundary list (no transient edge detection)."""
    state_boundaries_before: List[Tuple] = []
    for i in range(len(s1_list) - 1):
        s1 = int(s1_list[i])
        s1_next = int(s1_list[i + 1])
        if s1_next <= s1:
            continue
        s2 = int(max(s1 + 1, min(
            int(s2_events[i]) if i < len(s2_events) else s1,
            s1_next - 1,
        )))
        s1_start, s1_end, s2_start, s2_end = _paint_state_boundaries(
            s1, s2, s1_next, s1_half, s2_half, n_samples,
            use_transient_detection=False,
        )
        if s1_end > s1_start:
            state_boundaries_before.append((s1_start, s1_end, "S1", {"s1": s1}))
        if s2_start > s1_end:
            state_boundaries_before.append((s1_end, s2_start, "systole", {"s1": s1, "s2": s2}))
        if s2_end > s2_start:
            state_boundaries_before.append((s2_start, s2_end, "S2", {"s2": s2}))
        if s1_next > s2_end:
            state_boundaries_before.append((s2_end, s1_next, "diastole", {"s2": s2, "s1_next": s1_next}))
    return state_boundaries_before


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_pass3_correction(
    s1_peaks: np.ndarray,
    all_raw_peaks: np.ndarray,
    analysis_data: AnalysisData,
    audio_envelope: np.ndarray,
    sample_rate: int,
    params: Dict,
    wav_file_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Pass 3: dense per-sample cardiac-state timeline from Pass 2 peaks and pairs.

    Mutates and returns analysis_data with keys:
      pass3_state_labels, pass3_state_labels_encoding,
      pass3_state_boundaries, pass3_state_boundaries_before,
      pass3_corrections (empty), pass3_cycle_diagnostics (empty),
      pass3_spectral_context (None),
      pass3_noise_unreliable_windows_samples (when noise_event_segments exist).
      When pass3_enable_noise_repair: labels in HF windows may be cleared and rebuilt.
      pass3_calculate_large_gaps: when True, scan for large-gap / quiet windows (and optional
      recovered peaks) for HTML even if pass3_enable_gap_state_insert is False (dry-run; no insert/snap).
      pass3_calculate_noisy_regions: HTML noise strip prefers pass3_noise_unreliable_windows_samples
      when set (see plotting).
    """
    if "peak_classifications" not in analysis_data or analysis_data["peak_classifications"] is None:
        analysis_data["peak_classifications"] = {}
    if "s1_s2_pairs" not in analysis_data:
        analysis_data["s1_s2_pairs"] = []

    peaks_out = np.asarray(s1_peaks)
    if len(peaks_out) < 2:
        return peaks_out, analysis_data

    n_samples = int(len(audio_envelope))
    if n_samples <= 0:
        return peaks_out, analysis_data

    state_labels = np.full(n_samples, STATE_DIASTOLE, dtype=np.int8)

    # ── Read params ──────────────────────────────────────────────────────────
    s1_window_ms = float(params.get("pass3_state_s1_window_ms", 80.0))
    s2_window_ms = float(params.get("pass3_state_s2_window_ms", 80.0))
    s1_half = max(1, int(round(0.5 * s1_window_ms * sample_rate / 1000.0)))
    s2_half = max(1, int(round(0.5 * s2_window_ms * sample_rate / 1000.0)))

    edge_alpha  = float(params.get("pass3_state_edge_alpha",  0.03))
    edge_n_exp  = float(params.get("pass3_state_edge_n_exp",  4.0))
    s1_min_half = max(1, int(round(float(params.get("s1_min_sec", 0.030)) * 0.5 * sample_rate)))
    s1_max_half = max(1, int(round(float(params.get("s1_max_sec", 0.080)) * 0.5 * sample_rate)))
    s2_min_half = max(1, int(round(float(params.get("s2_min_sec", 0.030)) * 0.5 * sample_rate)))
    s2_max_half = max(1, int(round(float(params.get("s2_max_sec", 0.080)) * 0.5 * sample_rate)))

    fallback_bpm = 80.0
    try:
        rr_arr = np.diff(peaks_out) / float(sample_rate)
        rr_arr = rr_arr[np.isfinite(rr_arr) & (rr_arr > 0)]
        if len(rr_arr) > 0:
            fb = float(60.0 / np.median(rr_arr))
            if np.isfinite(fb) and fb > 0:
                fallback_bpm = fb
    except Exception:
        pass

    s1_list = sorted(list(dict.fromkeys(
        [int(x) for x in peaks_out.tolist() if 0 <= int(x) < n_samples]
    )))

    noise_segs_raw = analysis_data.get("noise_event_segments") or []
    noise_ivs_pass3: List[Tuple[int, int]] = []
    if noise_segs_raw:
        noise_ivs_pass3 = _noise_sample_intervals(noise_segs_raw, sample_rate, n_samples)
        analysis_data["pass3_noise_unreliable_windows_samples"] = [
            {"start_sample": int(lo), "end_sample": int(hi)} for lo, hi in noise_ivs_pass3
        ]
    logging.info(
        "Pass 3: HF-noise from pipeline — raw noise_event_segments=%d, sample intervals [lo,hi)=%d.",
        len(noise_segs_raw),
        len(noise_ivs_pass3),
    )

    # Build the BPM prior from clean RR intervals (exclude any S1→S1 that intersects HF noise).
    # If we can't build it (too few clean intervals), Pass 3 falls back to fallback_bpm.
    lt_pass3 = _build_lt_bpm_series_from_clean_rr(
        peaks_out,
        noise_ivs_pass3,
        sample_rate,
        n_samples,
        params=params,
    )

    # Build BPM prior raster unconditionally here — needed by noise repair, gap labeling,
    # and gap insert. All three paths consume (_bpm_prior_t, _bpm_prior) set below.
    _lt_for_raster = lt_pass3
    _bpm_prior_t, _bpm_prior = _dense_bpm_raster_from_series(
        _lt_for_raster, n_samples, sample_rate, fallback_bpm, dt_sec=0.05,
    )
    analysis_data["pass3_bpm_prior_times"] = _bpm_prior_t
    analysis_data["pass3_bpm_prior"] = _bpm_prior

    # ── S2 placement: Pass 2 pair seed → nominal fallback (_rebuild_s2_events) ─
    _pairs = analysis_data.get("s1_s2_pairs") or []
    s2_seed = _seed_s2_from_pass2_pairs(
        s1_list, _pairs, lt_pass3, fallback_bpm, sample_rate, params, n_samples,
    )
    _pair_set = {int(s1) for s1, _ in _pairs}
    logging.info(
        "Pass 3 Step 1: seeded %d beats from Pass 2 pairs (%d paired, %d BPM fallback).",
        len(s1_list),
        sum(1 for s in s1_list if s in _pair_set),
        sum(1 for s in s1_list if s not in _pair_set),
    )

    s2_events = _rebuild_s2_events(
        s1_list, lt_pass3, fallback_bpm, sample_rate, params, seed_s2_events=s2_seed,
    )

    # ── Before-correction snapshot for HTML before/after visualization ────────
    state_boundaries_before = _build_state_boundaries_before_from_cycles(
        s1_list, s2_events, s1_half, s2_half, n_samples,
    )

    corrections: List[Dict[str, Any]] = []
    cycle_diagnostics: List[Dict[str, Any]] = []

    peaks_out = np.asarray(s1_list, dtype=np.int64)

    noise_ivs_final: List[Tuple[int, int]] = list(noise_ivs_pass3)

    # ── Debug lookup tables for reasoning payload ─────────────────────────────
    _sr_f = float(sample_rate)
    _before_s2_by_s1:   Dict[int, int] = {}
    _s2_to_s1_before:   Dict[int, int] = {}
    _before_s1next_by_s1: Dict[int, int] = {}
    for _bs, _be, _bst, _bm in state_boundaries_before:
        if _bst == "systole":
            _s1k = _bm.get("s1"); _s2k = _bm.get("s2")
            if _s1k is not None and _s2k is not None:
                _before_s2_by_s1[int(_s1k)] = int(_s2k)
                _s2_to_s1_before[int(_s2k)] = int(_s1k)
        elif _bst == "diastole":
            _s2k = _bm.get("s2"); _s1nk = _bm.get("s1_next")
            if _s2k is not None and _s1nk is not None:
                _par = _s2_to_s1_before.get(int(_s2k))
                if _par is not None:
                    _before_s1next_by_s1[_par] = int(_s1nk)

    _corrs_by_s1: Dict[int, List[Dict]] = {}
    for _c in corrections:
        _ck = _c.get("s1") if _c.get("s1") is not None else _c.get("s1_prev")
        if _ck is not None:
            _corrs_by_s1.setdefault(int(_ck), []).append(_c)

    _all_corr_sorted = sorted(
        [(float(_c.get("s1", _c.get("s1_prev", 0))) / _sr_f, _c) for _c in corrections],
        key=lambda x: x[0],
    )
    _SHIFT_THRESH_SAMP = int(0.020 * sample_rate)

    # ── Paint final state timeline ────────────────────────────────────────────
    state_labels[:] = STATE_DIASTOLE
    state_boundaries: List[Tuple] = []
    # NOTE: we intentionally do not persist per-beat S2 indices; the state sequence
    # (labels + boundaries) is the canonical Pass 3 output.

    for i in range(len(peaks_out) - 1):
        s1     = int(peaks_out[i])
        s1_next = int(peaks_out[i + 1])
        if s1_next <= s1:
            continue

        if i < len(s2_events):
            s2 = int(s2_events[i])
        else:
            t_s1 = s1 / _sr_f
            bpm  = _bpm_at_time(t_s1, lt_pass3, fallback_bpm)
            ivs  = calculate_bpm_intervals(bpm, params)
            s2_pred = int(round(s1 + float(ivs.get("s1_s2_nominal", 0.30)) * _sr_f))
            s2 = int(max(s1 + 1, min(s2_pred, s1_next - 1)))
        s2 = int(max(s1 + 1, min(s2, s1_next - 1)))

        s1_start, s1_end, s2_start, s2_end = _paint_state_boundaries(
            s1, s2, s1_next, s1_half, s2_half, n_samples,
            audio_envelope=audio_envelope,
            edge_alpha=edge_alpha, edge_n_exp=edge_n_exp,
            min_s1_half=s1_min_half, max_s1_half=s1_max_half,
            min_s2_half=s2_min_half, max_s2_half=s2_max_half,
            use_transient_detection=True,
        )

        _t_s1_r = s1 / _sr_f
        _bpm_r = _bpm_at_time(_t_s1_r, lt_pass3, fallback_bpm)
        _ivs_r  = calculate_bpm_intervals(_bpm_r, params)
        _bef_s2    = _before_s2_by_s1.get(s1)
        _bef_s1nxt = _before_s1next_by_s1.get(s1)
        _direct    = _corrs_by_s1.get(s1, [])
        _cascade   = None
        for _ct, _cc in _all_corr_sorted:
            if _ct >= _t_s1_r - 0.010:
                break
            if _cc not in _direct:
                _cascade = _cc

        _reasoning = _build_reasoning_payload(
            s1, s1_start, s1_end, s2, s2_start, s2_end, s1_next,
            _ivs_r, _direct, _cascade, _bef_s2, _bef_s1nxt,
            _sr_f, _SHIFT_THRESH_SAMP,
            hover_debug_geometry=True,
            hover_s1_s2_pairs=_pairs,
            hover_s1_half=s1_half,
            hover_s2_half=s2_half,
            hover_n_samples=n_samples,
        )

        if s1_end > s1_start:
            state_labels[s1_start:s1_end] = STATE_S1
            state_boundaries.append(
                (s1_start, s1_end, "S1", {"s1": s1, "reasoning": _reasoning["S1"]}),
            )
        if s2_start > s1_end:
            state_labels[s1_end:s2_start] = STATE_SYSTOLE
            state_boundaries.append(
                (s1_end, s2_start, "systole", {"s1": s1, "s2": s2, "reasoning": _reasoning["systole"]}),
            )
        if s2_end > s2_start:
            state_labels[s2_start:s2_end] = STATE_S2
            state_boundaries.append(
                (s2_start, s2_end, "S2", {"s1": s1, "s2": s2, "reasoning": _reasoning["S2"]}),
            )
        if s1_next > s2_end:
            state_labels[s2_end:s1_next] = STATE_DIASTOLE
            state_boundaries.append(
                (s2_end, s1_next, "diastole", {
                    "s1": s1, "s2": s2, "s1_next": s1_next, "reasoning": _reasoning["diastole"],
                }),
            )

    # ── Trim diastole ends so boundary list is a strict non-overlapping partition ──
    # Each diastole was appended ending at s1_next (the next peak index), but the next
    # cycle's S1 starts *before* that peak (edge-detection start < peak). That left a
    # region claimed by both diastole and S1 in the boundary list, even though
    # state_labels itself (last-write-wins) is correct. Fix: walk the list once and
    # shorten any diastole whose end exceeds the start of the very next S1 segment.
    _trimmed: List[Tuple] = []
    for _bi, _seg in enumerate(state_boundaries):
        if _seg[2] == "diastole":
            _dia_start, _dia_end, _dia_name, _dia_meta = _seg
            # Find the start of the immediately following S1 segment.
            _next_s1_start = _dia_end  # default: no change
            for _fwd in range(_bi + 1, len(state_boundaries)):
                if state_boundaries[_fwd][2] == "S1":
                    _next_s1_start = state_boundaries[_fwd][0]
                    break
            _new_end = min(_dia_end, _next_s1_start)
            if _new_end > _dia_start:
                _trimmed.append((_dia_start, _new_end, _dia_name, _dia_meta))
        else:
            _trimmed.append(_seg)
    state_boundaries = _trimmed

    def _compute_and_store_measured_phase_curves(
        *,
        key_prefix: str,
        boundaries: List[Tuple],
        noise_ivs: Optional[List[Tuple[int, int]]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute cleaned (MAD + light smoothing) measured systole/diastole series from boundaries,
        exclude points inside noise windows, and store both sparse points and dense rasters.

        key_prefix examples:
          - "pass3_measured_phase_before_repair"
          - "pass3_measured_phase_final"
        """
        _ms_t, _ms_d = _pass3_measured_systole_series_from_boundaries(
            boundaries, sample_rate, noise_ivs=noise_ivs, params=params,
        )
        _md_t, _md_d = _pass3_measured_diastole_series_from_boundaries(
            boundaries, sample_rate, noise_ivs=noise_ivs, params=params,
        )
        _ms_t_r, _ms_d_r = _dense_raster_from_points(
            _ms_t, _ms_d, n_samples, sample_rate, dt_sec=0.05,
        )
        _md_t_r, _md_d_r = _dense_raster_from_points(
            _md_t, _md_d, n_samples, sample_rate, dt_sec=0.05,
        )

        analysis_data[f"{key_prefix}_systole_t"] = _ms_t
        analysis_data[f"{key_prefix}_systole_dur"] = _ms_d
        analysis_data[f"{key_prefix}_diastole_t"] = _md_t
        analysis_data[f"{key_prefix}_diastole_dur"] = _md_d
        analysis_data[f"{key_prefix}_systole_times"] = _ms_t_r
        analysis_data[f"{key_prefix}_systole"] = _ms_d_r
        analysis_data[f"{key_prefix}_diastole_times"] = _md_t_r
        analysis_data[f"{key_prefix}_diastole"] = _md_d_r
        return _ms_t_r, _ms_d_r, _md_t_r, _md_d_r

    # ── HF noise: clear then rebuild cardiac labels in unreliable audio ─────────
    _noise_repair_on = bool(
        params.get("pass3_enable_noise_repair", params.get("pass3_enable_noise_s2_repair", True)),
    )
    _gap_apply = bool(params.get("pass3_enable_gap_state_insert", True))
    _gap_calc = bool(params.get("pass3_calculate_large_gaps", True))
    _did_noise_repair = bool(_noise_repair_on and noise_ivs_final)

    # Always compute and store measured curves from the initially-painted boundaries.
    # These are the canonical "before repair" curves (repairs may modify boundaries later).
    _ms_t_r = np.asarray([], dtype=np.float64)
    _ms_d_r = np.asarray([], dtype=np.float64)
    _md_t_r = np.asarray([], dtype=np.float64)
    _md_d_r = np.asarray([], dtype=np.float64)
    try:
        _ms_t_r, _ms_d_r, _md_t_r, _md_d_r = _compute_and_store_measured_phase_curves(
            key_prefix="pass3_measured_phase_before_repair",
            boundaries=state_boundaries,
            noise_ivs=list(noise_ivs_final) if noise_ivs_final else [],
        )
    except Exception:
        logging.debug("Pass 3: failed to compute before-repair phase curves", exc_info=True)

    if not _noise_repair_on:
        logging.info(
            "Pass 3: noise repair disabled — skipping HF-noise clear/rebuild.",
        )
    elif not noise_ivs_final:
        logging.info(
            "Pass 3: no HF-noise sample intervals — skipping HF-noise clear/rebuild.",
        )
    if _did_noise_repair:
        # BPM prior raster was already computed unconditionally above; reuse it here.
        state_labels, state_boundaries = _pass3_clear_states_in_hf_noise(
            state_labels, state_boundaries, noise_ivs_final, n_samples, peaks_out,
        )
        logging.info(
            "Pass 3 noise repair: S1 noise → full beat clear; diastole-only noise → post-S1 clear "
            "for that beat; %d merged HF-noise window(s).",
            len(_merge_sorted_intervals(list(noise_ivs_final))),
        )
        state_labels, state_boundaries = _pass3_rebuild_unknown_runs(
            state_labels, state_boundaries, n_samples,
            (_bpm_prior_t, _bpm_prior),
            fallback_bpm, sample_rate, params,
            measured_systole_t=_ms_t_r,
            measured_systole_dur=_ms_d_r,
            measured_diastole_t=_md_t_r,
            measured_diastole_dur=_md_d_r,
        )
        _n_rebuilt = sum(
            1 for seg in state_boundaries
            if seg[2] == "S1" and isinstance(seg[3], dict) and seg[3].get("rebuild_source") == "noise_repair"
        )
        logging.info("Pass 3 rebuild: %d rebuilt beat(s) in noise gap(s).", _n_rebuilt)

    # ── Gap logic: detect windows once, then route to labeling (>threshold) and insert (≤threshold) ──
    # _pass3_find_gap_windows runs once on post-repair boundaries and is the canonical source
    # for both paths. BPM prior raster was already computed unconditionally above.
    _lg_label_on = bool(params.get("pass3_enable_peaks_labeling_in_large_gaps", True))
    _gap_peak_apply = bool(params.get("pass3_enable_gap_snap_to_peaks", True))

    if _gap_calc or _gap_apply or _lg_label_on:
        # Refresh measured rasters from post-repair boundaries.
        try:
            _ms_t_r, _ms_d_r, _md_t_r, _md_d_r = _compute_and_store_measured_phase_curves(
                key_prefix="pass3_measured_phase_for_gap_insert",
                boundaries=state_boundaries,
                noise_ivs=list(noise_ivs_final) if noise_ivs_final else [],
            )
        except Exception:
            logging.debug("Pass 3: failed to compute for-gap-insert phase curves", exc_info=True)

        # ── Single canonical gap detection ────────────────────────────────────
        _gap_quiet_debug: List[Dict[str, Any]] = []
        _gap_windows: List[Dict[str, Any]] = _pass3_find_gap_windows(
            state_boundaries,
            n_samples,
            (_bpm_prior_t, _bpm_prior),
            fallback_bpm,
            sample_rate,
            params,
            audio_envelope,
            measured_systole_t=_ms_t_r,
            measured_systole_dur=_ms_d_r,
            measured_diastole_t=_md_t_r,
            measured_diastole_dur=_md_d_r,
            dynamic_noise_floor_series=analysis_data.get("dynamic_noise_floor_series"),
            quiet_windows_out=_gap_quiet_debug,
        )

        # Populate debug keys from canonical list (exclude the raw meta dict).
        if _gap_windows:
            analysis_data["pass3_large_gap_windows_samples"] = [
                {k: v for k, v in gw.items() if k != "meta"}
                for gw in _gap_windows
            ]
        else:
            analysis_data.setdefault("pass3_large_gap_windows_samples", [])
        if _gap_quiet_debug:
            analysis_data["pass3_gap_quiet_windows_samples"] = list(_gap_quiet_debug)

        # ── Large-gap peaks labeling (> threshold) ────────────────────────────
        if _lg_label_on:
            state_labels, state_boundaries = _pass3_apply_peaks_labeling_in_large_gaps(
                state_labels,
                state_boundaries,
                gap_windows=_gap_windows,
                n_samples=n_samples,
                sample_rate=sample_rate,
                params=params,
                audio_envelope=audio_envelope,
                analysis_data=analysis_data,
                s1_list=s1_list,
                lt_pass3=lt_pass3,
                fallback_bpm=fallback_bpm,
                s1_s2_pairs=_pairs,
                corrections=corrections,
                before_s2_by_s1=_before_s2_by_s1,
                before_s1next_by_s1=_before_s1next_by_s1,
                s1_half=s1_half,
                s2_half=s2_half,
                edge_alpha=edge_alpha,
                edge_n_exp=edge_n_exp,
                s1_min_half=s1_min_half,
                s1_max_half=s1_max_half,
                s2_min_half=s2_min_half,
                s2_max_half=s2_max_half,
            )

        # ── Regular gap state insert (≤ threshold) ────────────────────────────
        _gap_debug: List[Dict[str, Any]] = []
        state_labels, state_boundaries = _pass3_insert_missing_states_in_large_gaps(
            state_labels,
            state_boundaries,
            n_samples,
            (_bpm_prior_t, _bpm_prior),
            fallback_bpm,
            sample_rate,
            params,
            _gap_windows,
            measured_systole_t=_ms_t_r,
            measured_systole_dur=_ms_d_r,
            measured_diastole_t=_md_t_r,
            measured_diastole_dur=_md_d_r,
            debug_windows_out=_gap_debug,
            dry_run=not _gap_apply,
        )

        # ── Recovered peaks for HTML strip (all gap windows) ──────────────────
        gap_wins_for_peaks = analysis_data.get("pass3_large_gap_windows_samples") or []
        want_peak_detect = len(gap_wins_for_peaks) > 0 and (
            _gap_calc or (_gap_apply and _gap_peak_apply) or _lg_label_on
        )
        if not want_peak_detect:
            analysis_data["pass3_large_gap_recovered_peaks_insensitive"] = []
            analysis_data["pass3_large_gap_recovered_peaks_sensitive"] = []
        if want_peak_detect:
            q_ins = float(params.get("pass3_gap_recovery_peak_prominence_quantile_insensitive", 0.70))
            q_sens = float(params.get("pass3_gap_recovery_peak_prominence_quantile_sensitive", 0.50))
            recovered_ins = _detect_sensitive_peaks_in_large_gap_windows(
                audio_envelope, sample_rate, gap_wins_for_peaks, params,
                prominence_quantile=q_ins,
                dynamic_noise_floor_series=analysis_data.get("dynamic_noise_floor_series"),
            )
            recovered_sens = _detect_sensitive_peaks_in_large_gap_windows(
                audio_envelope, sample_rate, gap_wins_for_peaks, params,
                prominence_quantile=q_sens,
                dynamic_noise_floor_series=analysis_data.get("dynamic_noise_floor_series"),
            )
            analysis_data["pass3_large_gap_recovered_peaks_insensitive"] = sorted(
                set(int(x) for x in recovered_ins.tolist())
            )
            analysis_data["pass3_large_gap_recovered_peaks_sensitive"] = sorted(
                set(int(x) for x in recovered_sens.tolist())
            )

        # ── Snap rebuilt states to detected peaks ─────────────────────────────
        if _gap_apply and _gap_peak_apply and len(gap_wins_for_peaks) > 0:
            recovered_ins_snap = np.asarray(
                analysis_data.get("pass3_large_gap_recovered_peaks_insensitive") or [],
                dtype=np.int64,
            )
            recovered_sens_snap = np.asarray(
                analysis_data.get("pass3_large_gap_recovered_peaks_sensitive") or [],
                dtype=np.int64,
            )
            _snap_window = int(round(
                float(params.get("pass3_gap_snap_window_ms", 80.0)) * sample_rate / 1000.0
            ))
            state_labels, state_boundaries = _pass3_snap_rebuilt_states_to_recovered_peaks(
                state_labels,
                state_boundaries,
                recovered_ins_snap,
                recovered_sens_snap,
                audio_envelope,
                sample_rate,
                _snap_window,
            )
    else:
        logging.info(
            "Pass 3: large-gap scan and insert disabled "
            "(pass3_calculate_large_gaps=False, pass3_enable_gap_state_insert=False, "
            "pass3_enable_peaks_labeling_in_large_gaps=False).",
        )

    # ── Derive peaks_out from state sequence (canonical source of truth) ──────
    # Intent: Pass 3 treats the *state sequence* (pass3_state_labels / pass3_state_boundaries)
    # as the authoritative representation of cardiac timing. When HF-noise repair runs, it
    # may *regenerate* (“synthetic”) states inside unreliable/low-confidence audio windows
    # (see metadata: rebuild_source="noise_repair" or "gap_insert"). Those regenerated states are not just for display:
    # they are promoted into the canonical sequence and therefore replace the corresponding
    # noisy region for all downstream consumers (metrics, plots, Pass 4, etc.).
    _s1_from_states = sorted({
        int(seg[3]["s1"])
        for seg in state_boundaries
        if seg[2] == "S1" and isinstance(seg[3], dict) and "s1" in seg[3]
    })
    if _s1_from_states:
        peaks_out = np.asarray(_s1_from_states, dtype=np.int64)

    # ── Store results ─────────────────────────────────────────────────────────
    # Final cleaned curves + dense rasters for plotting/debug and for any downstream consumers.
    # Kept distinct from "before repair" so the dataflow is unambiguous.
    try:
        _ms_t_r_f, _ms_d_r_f, _md_t_r_f, _md_d_r_f = _compute_and_store_measured_phase_curves(
            key_prefix="pass3_measured_phase_final",
            boundaries=state_boundaries,
            noise_ivs=list(noise_ivs_final) if noise_ivs_final else [],
        )
        analysis_data["pass3_measured_systole_times"] = _ms_t_r_f
        analysis_data["pass3_measured_systole"] = _ms_d_r_f
        analysis_data["pass3_measured_diastole_times"] = _md_t_r_f
        analysis_data["pass3_measured_diastole"] = _md_d_r_f
        analysis_data["pass3_measured_systole_t"] = analysis_data.get("pass3_measured_phase_final_systole_t")
        analysis_data["pass3_measured_systole_dur"] = analysis_data.get("pass3_measured_phase_final_systole_dur")
        analysis_data["pass3_measured_diastole_t"] = analysis_data.get("pass3_measured_phase_final_diastole_t")
        analysis_data["pass3_measured_diastole_dur"] = analysis_data.get("pass3_measured_phase_final_diastole_dur")
    except Exception:
        logging.debug("Pass 3: failed to compute final phase curves", exc_info=True)

    analysis_data["pass3_state_labels"]          = state_labels
    analysis_data["pass3_state_labels_encoding"] = dict(STATE_LABELS_ENCODING)
    analysis_data["pass3_state_boundaries"]        = state_boundaries
    analysis_data["pass3_state_boundaries_before"] = state_boundaries_before
    analysis_data["pass3_corrections"]    = corrections
    analysis_data["pass3_cycle_diagnostics"] = cycle_diagnostics

    analysis_data["pass3_spectral_context"] = None

    logging.info(
        "Pass 3: corrected peaks=%d, state timeline n=%d samples, %d cycles, %d corrections.",
        int(len(peaks_out)), n_samples,
        max(0, len(peaks_out) - 1),
        int(len(corrections)),
    )
    return peaks_out, analysis_data
