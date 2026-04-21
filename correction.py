"""
correction.py — Pass 3 cardiac-cycle correction and state-timeline generation.

Entry point: run_pass3_correction().

Structure
---------
  Module-level helpers  (formerly closures inside _refine_and_correct_peaks)
  Boundary geometry     (_resolve_boundary_overlap, _paint_state_boundaries,
                         _build_reasoning_payload)
  S2-events rebuild     (_rebuild_s2_events — shared by multiple passes)
  Correction passes     (_pass_a_resnap_s2, _pass_c_phase_correction)
  Main entry point      (run_pass3_correction)
"""

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks as _scipy_find_peaks

from confidence_engine import calculate_bpm_intervals
from hrv import _median_mad_keep_mask_time_window
from fft_profiles import (
    prepare_pass3_s1_insert_context,
    spectrum_template_search_envelope_index,
)


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
# Module-level helpers  (formerly closures inside _refine_and_correct_peaks)
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


def _s2_index_respects_pass3_intervals(
    s1: int,
    s1_next: int,
    s2_idx: int,
    sample_rate: int,
    intervals: Dict,
    params: Dict,
) -> bool:
    """
    True when S2 at s2_idx yields systole/diastole durations that Pass A would not flag
    as too short (same thresholds as _pass_a_resnap_s2).
    """
    if s2_idx <= s1 or s2_idx >= s1_next:
        return False
    sr = float(sample_rate)
    systole = (s2_idx - s1) / sr
    diastole = (s1_next - s2_idx) / sr
    s1_s2_min = float(intervals.get("s1_s2_min", 0.12))
    diastole_min = float(intervals.get("diastole_min", 0.08))
    sys_slack = float(params.get("pass3_systole_slack_frac", 0.15))
    dia_slack = float(params.get("pass3_diastole_slack_frac", 0.20))
    systole_too_short = systole < (1.0 - sys_slack) * s1_s2_min
    diastole_too_short = diastole < (1.0 - dia_slack) * diastole_min
    return not (systole_too_short or diastole_too_short)


def _choose_s2_near(
    s1: int,
    s1_next: int,
    s2_pred: int,
    half_window_samples: int,
    snap_s2: bool,
    insert_spectrum_ctx: Optional[Dict],
    sample_rate: int,
    n_samples: int,
    params: Dict,
    intervals: Dict,
) -> int:
    """Choose S2 near predicted time by sliding FFT windows against the S2 spectral template.

    If snap_s2 is False, no spectral context is available, or no confident match is found,
    returns s2_pred clamped within [s1+1, s1_next-1] — keeps rhythm at nominal ejection time
    without requiring a peak at that location.

    A spectral winner is rejected if it would make systole or diastole shorter than Pass A's
    minimum plausible durations for this BPM (calculate_bpm_intervals + slack).
    """
    s2 = int(max(s1 + 1, min(s2_pred, s1_next - 1)))
    if not snap_s2 or insert_spectrum_ctx is None:
        return s2
    t_pred_sec = float(s2_pred) / float(sample_rate)
    search_half_sec = float(half_window_samples) / float(sample_rate)
    result = _choose_s2_spectral(
        t_pred_sec, search_half_sec, insert_spectrum_ctx, params, sample_rate, n_samples,
    )
    if result is None:
        return s2
    sp_idx, _ = result
    if not (s1 < sp_idx < s1_next):
        return s2
    if not _s2_index_respects_pass3_intervals(s1, s1_next, int(sp_idx), sample_rate, intervals, params):
        return s2
    return int(sp_idx)


def _choose_s1_near(
    t_expected_sec: float,
    half_window_samples: int,
    min_sep_samples: int,
    all_raw_peaks: np.ndarray,
    pc: Dict,
    n_samples: int,
    sample_rate: int,
) -> Optional[int]:
    """Choose best S1 near expected time using label_scores['S1']."""
    if len(all_raw_peaks) == 0:
        return None
    center = int(round(t_expected_sec * sample_rate))
    lo = max(0, center - half_window_samples)
    hi = min(n_samples - 1, center + half_window_samples)
    if hi <= lo:
        return None
    cand = [int(p) for p in all_raw_peaks if lo <= int(p) <= hi]
    if not cand:
        return None
    best: Optional[int] = None
    best_score: Optional[float] = None
    for p in cand:
        entry = pc.get(int(p)) or {}
        ls = entry.get("label_scores") if isinstance(entry, dict) else None
        s1_score = float(ls.get("S1", 0.0)) if isinstance(ls, dict) else 0.0
        noise_score = float(ls.get("noise", 0.0)) if isinstance(ls, dict) else 0.0
        dist_sec = abs(p - center) / float(sample_rate)
        score = (2.0 * s1_score) - (1.0 * noise_score) - (0.75 * dist_sec)
        if best is None or score > best_score:
            best, best_score = p, score
    return int(best) if best is not None else None


def _insert_spectrum_envelope_ok(
    env_idx: int,
    audio_envelope: np.ndarray,
    analysis_data: Dict,
    params: Dict,
) -> bool:
    """Return True if envelope at env_idx meets the noise-floor margin."""
    margin = float(params.get("pass3_insert_spectrum_envelope_margin", 0.0))
    if margin <= 0:
        return True
    nfs = analysis_data.get("dynamic_noise_floor_series")
    if nfs is None or getattr(nfs, "empty", True):
        return True
    try:
        ei = int(max(0, min(env_idx, len(audio_envelope) - 1)))
        e = float(audio_envelope[ei])
        nf = float(nfs.reindex([ei], method="nearest").iloc[0])
        return e >= margin * nf
    except Exception:
        return True


def _find_sensitive_peaks_near(
    t_expected_sec: float,
    window_samples: int,
    sensitivity_factor: float,
    audio_envelope: np.ndarray,
    analysis_data: Dict,
    n_samples: int,
    sample_rate: int,
    params: Dict,
) -> Optional[int]:
    """
    Re-scan audio_envelope in a narrow window with a lower height threshold to find
    faint peaks missed by the main detector.  Returns the highest-amplitude peak or None.
    """
    nfs = analysis_data.get("dynamic_noise_floor_series")
    center = int(round(t_expected_sec * sample_rate))
    lo = max(0, center - window_samples)
    hi = min(n_samples - 1, center + window_samples)
    if hi <= lo:
        return None
    segment = audio_envelope[lo: hi + 1]
    if len(segment) == 0:
        return None
    if nfs is not None and not getattr(nfs, "empty", True):
        try:
            indices = np.arange(lo, hi + 1)
            nf_vals = nfs.reindex(indices, method="nearest").values.astype(np.float64)
            height_thresh = sensitivity_factor * nf_vals
        except Exception:
            height_thresh = sensitivity_factor * float(np.median(audio_envelope))
    else:
        height_thresh = sensitivity_factor * float(np.median(audio_envelope))
    min_dist = max(1, int(float(params.get("min_peak_distance_sec", 0.10)) * sample_rate // 2))
    try:
        local_peaks, _ = _scipy_find_peaks(segment, height=height_thresh, distance=min_dist)
    except Exception:
        return None
    if len(local_peaks) == 0:
        return None
    best_local = int(local_peaks[np.argmax(segment[local_peaks])])
    return lo + best_local


def _choose_s2_spectral(
    t_expected_sec: float,
    search_half_sec: float,
    insert_spectrum_ctx: Optional[Dict],
    params: Dict,
    sample_rate: int,
    n_samples: int,
) -> Optional[Tuple[int, float]]:
    """
    Search for S2 using the spectral S2 template from insert_spectrum_ctx.

    LIMITATION: the S2 template is built from Pass 2 paired S2 peaks.  If Pass 2 made
    systematic labeling errors those may bias the template (confirmation-bias risk).
    Only call after _find_sensitive_peaks_near has already failed.

    Returns (envelope_index, score) or None.
    """
    if insert_spectrum_ctx is None:
        return None
    mu_s2 = insert_spectrum_ctx.get("mu_s2_db")
    if mu_s2 is None or not isinstance(mu_s2, np.ndarray) or len(mu_s2) == 0:
        return None
    n_s2_tpl = int(insert_spectrum_ctx.get("n_s2_template", 0))
    min_tpl = int(params.get("pass3_s2_spectral_min_templates", 3))
    if n_s2_tpl < min_tpl:
        return None
    try:
        result = spectrum_template_search_envelope_index(
            insert_spectrum_ctx["bandpass_audio"],
            int(insert_spectrum_ctx["full_sr"]),
            t_expected_sec,
            search_half_sec,
            mu_s2,
            insert_spectrum_ctx["freqs"],
            int(insert_spectrum_ctx["n_fft"]),
            int(insert_spectrum_ctx["half_samples"]),
            sample_rate,
            n_samples,
            params,
        )
    except Exception:
        return None
    return result


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
# Boundary geometry helpers
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
    snap_s2: bool,
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

    # Initial S2 placement uses FFT template matching when pass3_align_s2_to_s2_spectral_profile is on.
    # There is no per-beat correction dict for that path — explain when S2 still differs from the nominal index.
    if snap_s2 and not _s2_notes:
        s2_pred_nom = int(round(s1 + float(ivs.get("s1_s2_nominal", 0.30)) * sample_rate))
        s2_nom_idx = int(max(s1 + 1, min(s2_pred_nom, s1_next - 1)))
        if abs(s2 - s2_nom_idx) > shift_thresh_samp:
            d_ms = round(abs(s2 - s2_nom_idx) / sample_rate * 1000)
            _s2_notes.append(
                f"\u2139 S2 placed by S2 spectral template match ({d_ms} ms from nominal ejection time)."
            )

    return {
        "S1":       {"expected_ms": _exp_s1_ms,  "measured_ms": _meas_s1_ms,  "notes": _s1_notes},
        "systole":  {"expected_ms": _exp_sys_ms, "measured_ms": _meas_sys_ms, "notes": _sys_notes},
        "S2":       {"expected_ms": _exp_s2_ms,  "measured_ms": _meas_s2_ms,  "notes": _s2_notes},
        "diastole": {"expected_ms": _exp_dia_ms, "measured_ms": _meas_dia_ms, "notes": _dia_notes},
    }


# ─────────────────────────────────────────────────────────────────────────────
# S2-events rebuild helper  (used by multiple correction passes)
# ─────────────────────────────────────────────────────────────────────────────

def _rebuild_s2_events(
    s1_list: List[int],
    lt_series: Optional[pd.Series],
    fallback_bpm: float,
    sample_rate: int,
    params: Dict,
    snap_s2: bool,
    snap_half: int,
    n_samples: int,
    insert_spectrum_ctx: Optional[Dict],
    noise_ivs: Optional[List[Tuple[int, int]]] = None,
    seed_s2_events: Optional[List[int]] = None,
) -> List[int]:
    """Rebuild s2_events from scratch given the current s1_list.

    When ``seed_s2_events`` is set and a cycle overlaps HF-noise, keep that seed
    index so pass3_align_s2_to_s2_spectral_profile does not run on bad audio.
    """
    niv = noise_ivs or []
    seed = seed_s2_events
    s2_events: List[int] = []
    for j in range(len(s1_list) - 1):
        a = int(s1_list[j])
        b = int(s1_list[j + 1])
        if b <= a:
            s2_events.append(int(a))
            continue
        if (
            seed is not None
            and j < len(seed)
            and niv
            and _hf_noise_disables_s2_snap(a, b, niv, s2_check=int(seed[j]))
        ):
            s2_keep = int(max(a + 1, min(int(seed[j]), b - 1)))
            s2_events.append(s2_keep)
            continue
        t_a = a / float(sample_rate)
        bpm_a = _bpm_at_time(t_a, lt_series, fallback_bpm)
        ivs_a = calculate_bpm_intervals(bpm_a, params)
        s1_s2_nominal_a = float(ivs_a.get("s1_s2_nominal", 0.30))
        s2_pred_a = int(round(a + s1_s2_nominal_a * sample_rate))
        snap_here = _effective_snap_s2(snap_s2, a, b, niv, s2_check=None)
        s2_a = _choose_s2_near(
            a, b, s2_pred_a, snap_half,
            snap_here, insert_spectrum_ctx, sample_rate, n_samples, params, ivs_a,
        )
        s2_events.append(int(s2_a))
    return s2_events


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
        "Inserted missing cardiac cycle(s) in a large gap (gap repair).",
    )


def _pass3_clean_duration_series(
    t_raw: np.ndarray,
    d_raw: np.ndarray,
    noise_ivs: Optional[List[Tuple[int, int]]],
    sample_rate: int,
    params: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shared post-processing for measured systole / diastole series:
      1. Drop points whose time falls inside a noise interval.
      2. Local median ± k*MAD in a rolling time window (same helper as systole interval curve).
      3. Gaussian-weighted rolling mean (sigma=1 beat, half-window=3 beats) — light smoothing
         that keeps values close to measured data while removing beat-to-beat jitter.
    Input arrays must already be sorted by time.
    """
    if len(t_raw) == 0:
        return t_raw.copy(), d_raw.copy()

    t = t_raw.copy()
    d = d_raw.copy()
    pc = params or {}

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
    return _pass3_clean_duration_series(t[order], d[order], noise_ivs, sample_rate, params=params)


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
    return _pass3_clean_duration_series(t[order], d[order], noise_ivs, sample_rate, params=params)


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
        for K in range(max(0, k_center - 2), k_center + 3):
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
    *,
    measured_systole_t: Optional[np.ndarray] = None,
    measured_systole_dur: Optional[np.ndarray] = None,
    measured_diastole_t: Optional[np.ndarray] = None,
    measured_diastole_dur: Optional[np.ndarray] = None,
    debug_windows_out: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, List[Tuple]]:
    """
    Pass 3 Step 3 — Insert missing *states* in large gaps.

    This replaces the older "insert missing S1 in long RR" event-level logic.
    We operate on the already-generated *state sequence*: if any single state segment is
    long enough to plausibly contain an additional full cardiac cycle (per the same model
    used by _pass3_rebuild_unknown_runs), we mark the surplus tail as STATE_UNKNOWN and
    reuse _pass3_rebuild_unknown_runs to pack in full cycles using the same cursoring.

    Side notes:
      - This runs after HF-noise repair. In practice, "lack of sound" gaps tend to coincide
        with regions already cleared by noise repair, so the two usually do not overlap.
      - Although we do not target any specific phase, the only segment that should commonly
        be able to absorb extra cycles is diastole (the leftover) if upstream timing is sane.
    """
    if n_samples <= 0 or sample_rate <= 0:
        return state_labels, state_boundaries

    t_r, bpm_r = bpm_prior_raster
    t_r = np.asarray(t_r, dtype=np.float64)
    bpm_r = np.asarray(bpm_r, dtype=np.float64)
    if len(t_r) < 2 or len(t_r) != len(bpm_r):
        return state_labels, state_boundaries

    SR = float(sample_rate)
    MAX_SCALE = 0.30  # must match _pass3_rebuild_unknown_runs

    def _bpm_at_sample(ix: int) -> float:
        t = float(ix) / SR
        v = _interp_piecewise_linear(t, t_r, bpm_r)
        if v is None:
            return float(fallback_bpm)
        if not np.isfinite(v) or v <= 0:
            return float(fallback_bpm)
        return float(v)

    def _expected_phase_samples(phase: str, t_mid: float, ivs: Dict) -> int:
        if phase == "S1":
            sec = float(ivs.get("s1_nominal", 0.040))
        elif phase == "S2":
            sec = float(ivs.get("s2_nominal", 0.030))
        elif phase == "systole":
            sec = _interp_piecewise_linear(t_mid, measured_systole_t, measured_systole_dur)
            if sec is None:
                sec = float(ivs.get("s1_s2_nominal", 0.300))
        else:  # diastole
            sec = _interp_piecewise_linear(t_mid, measured_diastole_t, measured_diastole_dur)
            if sec is None:
                bpm_here = float(ivs.get("bpm", 0.0)) if isinstance(ivs, dict) else 0.0
                rr = 60.0 / float(bpm_here) if bpm_here and bpm_here > 0 else 0.75
                sec = float(ivs.get("s2_s1_nominal", rr - float(ivs.get("s1_s2_nominal", 0.300))))
        return max(1, int(round(float(sec) * SR)))

    changed = False
    new_bd: List[Tuple] = []

    for seg in state_boundaries:
        a0, a1, name, meta = seg
        a0 = int(max(0, min(int(a0), n_samples)))
        a1 = int(max(0, min(int(a1), n_samples)))
        if a1 <= a0:
            continue
        if name not in ("S1", "systole", "S2", "diastole"):
            new_bd.append(seg)
            continue

        width = a1 - a0
        t_mid = 0.5 * (a0 + a1) / SR
        bpm = _bpm_at_sample(int((a0 + a1) // 2))
        ivs = calculate_bpm_intervals(bpm, params)
        # stash bpm for _expected_phase_samples diastole fallback
        try:
            ivs = dict(ivs)
            ivs["bpm"] = float(bpm)
        except Exception:
            pass

        # Compute nominal cycle length (unscaled) the same way rebuild does.
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

        exp_phase = _expected_phase_samples(name, t_mid, ivs)
        # "Can fit another cycle" allowing rebuild's ±30% scaling.
        min_extra = int(round((1.0 - MAX_SCALE) * float(max(1, cyc0))))
        if width < exp_phase + max(4, min_extra):
            new_bd.append(seg)
            continue

        # Mark the surplus tail as unknown; keep the expected head of the segment.
        tail_lo = a0 + exp_phase
        tail_hi = a1
        if tail_hi - tail_lo < 4:
            new_bd.append(seg)
            continue

        changed = True
        state_labels[tail_lo:tail_hi] = STATE_UNKNOWN
        if debug_windows_out is not None:
            try:
                debug_windows_out.append({
                    "start_sample": int(tail_lo),
                    "end_sample": int(tail_hi),
                    "source_state": str(name),
                    "trigger": "can_fit_extra_cycle",
                    "bpm_at_mid": float(bpm),
                    "expected_phase_samples": int(exp_phase),
                    "cycle0_samples": int(cyc0),
                    "segment_samples": int(width),
                })
            except Exception:
                pass

        # Keep the truncated original segment.
        kept = (a0, tail_lo, name, meta)
        new_bd.append(kept)

    if not changed:
        return state_labels, state_boundaries

    # Drop any segments that overlap unknown tails (they may have been truncated above).
    # Since we only mark tails inside existing segments, a simple filter on label identity
    # is enough to remove any stale overlaps.
    filtered: List[Tuple] = []
    for seg in new_bd:
        a0, a1, name, meta = seg
        a0 = int(max(0, min(int(a0), n_samples)))
        a1 = int(max(0, min(int(a1), n_samples)))
        if a1 <= a0:
            continue
        if np.any(state_labels[a0:a1] == STATE_UNKNOWN):
            continue
        filtered.append(seg)
    filtered.sort(key=lambda t: int(t[0]))

    # Reuse the same rebuild logic + cursoring to fill unknown tails.
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
    )
    return state_labels, rebuilt


def _hf_noise_disables_s2_snap(
    s1: int,
    s1_next: int,
    noise_ivs: List[Tuple[int, int]],
    s2_check: Optional[int] = None,
) -> bool:
    """
    True → do not use pass3_align_s2_to_s2_spectral_profile (spectral / sliding template) for this cycle.
    HF-noise segments make the underlying audio unreliable; use nominal timing only.
    """
    if not noise_ivs:
        return False
    for lo, hi in noise_ivs:
        if lo < s1_next and hi > s1:
            return True
        if s2_check is not None and lo <= int(s2_check) < hi:
            return True
    return False


def _effective_snap_s2(
    snap_s2: bool,
    s1: int,
    s1_next: int,
    noise_ivs: List[Tuple[int, int]],
    s2_check: Optional[int] = None,
) -> bool:
    """Spectral S2 snap allowed only when globally on and cycle not in HF-noise."""
    if not snap_s2:
        return False
    return not _hf_noise_disables_s2_snap(s1, s1_next, noise_ivs, s2_check)


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
# Pass A — re-snap S2 for timing plausibility
# ─────────────────────────────────────────────────────────────────────────────

def _pass_a_resnap_s2(
    s1_list: List[int],
    s2_events: List[int],
    lt_series: Optional[pd.Series],
    fallback_bpm: float,
    sample_rate: int,
    params: Dict,
    snap_s2: bool,
    resnap_half: int,
    n_samples: int,
    insert_spectrum_ctx: Optional[Dict],
    systole_slack: float,
    diastole_slack: float,
    noise_ivs: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[List[int], List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """
    Re-snap S2 when systole/diastole are out of plausible range.

    Returns (new_s2_events, new_corrections, cycle_diagnostics, changed).
    s1_list is unchanged.
    """
    new_s2_events = list(s2_events)
    new_corrections: List[Dict[str, Any]] = []
    cycle_diagnostics: List[Dict[str, Any]] = []
    changed = False
    niv = noise_ivs or []

    for i in range(len(s1_list) - 1):
        s1 = int(s1_list[i])
        s1_next = int(s1_list[i + 1])
        if s1_next <= s1:
            continue
        s2 = int(new_s2_events[i]) if i < len(new_s2_events) else s1
        s2 = int(max(s1 + 1, min(s2, s1_next - 1)))

        t_s1 = s1 / float(sample_rate)
        bpm = _bpm_at_time(t_s1, lt_series, fallback_bpm)
        intervals = calculate_bpm_intervals(bpm, params)
        s1_s2_min     = float(intervals.get("s1_s2_min",     0.12))
        s1_s2_nominal = float(intervals.get("s1_s2_nominal", 0.30))
        s1_s2_max     = float(intervals.get("s1_s2_max",     0.40))

        systole  = (s2 - s1)     / float(sample_rate)
        rr       = (s1_next - s1) / float(sample_rate)
        diastole = rr - systole
        expected_rr      = float(intervals.get("rr_interval", 60.0 / bpm if bpm > 0 else 0.75))
        diastole_nominal = float(intervals.get("s2_s1_nominal", max(0.0, expected_rr - s1_s2_nominal)))
        diastole_min     = float(intervals.get("diastole_min", 0.08))
        diastole_max     = float(intervals.get("diastole_max", diastole_nominal * 2.0))

        s1_min_evt      = float(intervals.get("s1_min",      0.010))
        s1_nominal_evt  = float(intervals.get("s1_nominal",  0.040))
        s1_max_evt      = float(intervals.get("s1_max",      0.080))
        s2_min_evt      = float(intervals.get("s2_min",      0.010))
        s2_nominal_evt  = float(intervals.get("s2_nominal",  0.030))
        s2_max_evt      = float(intervals.get("s2_max",      0.060))
        min_feasible    = float(intervals.get(
            "min_feasible_cycle", s1_min_evt + s1_s2_min + s2_min_evt + diastole_min,
        ))

        too_short        = systole < (1.0 - systole_slack)  * s1_s2_min
        too_long         = systole > (1.0 + systole_slack)  * s1_s2_max
        far_from_nominal = abs(systole - s1_s2_nominal) > max(0.12, 0.5 * (s1_s2_max - s1_s2_min))
        diastole_too_short = diastole < (1.0 - diastole_slack) * diastole_min

        cycle_diagnostics.append({
            "i": int(i), "s1": int(s1), "s2": int(s2), "s1_next": int(s1_next),
            "bpm": float(bpm), "rr_sec": float(rr),
            "systole_sec": float(systole), "diastole_sec": float(diastole),
            "expected_rr_sec": float(expected_rr),
            "diastole_nominal_sec": float(diastole_nominal),
            "diastole_min_sec": float(diastole_min),
            "diastole_max_sec": float(diastole_max),
            "s1_min_sec": float(s1_min_evt), "s1_nominal_sec": float(s1_nominal_evt),
            "s1_max_sec": float(s1_max_evt),
            "s2_min_sec": float(s2_min_evt), "s2_nominal_sec": float(s2_nominal_evt),
            "s2_max_sec": float(s2_max_evt),
            "min_feasible_cycle_sec": float(min_feasible),
            "s1_s2_min": float(s1_s2_min), "s1_s2_nominal": float(s1_s2_nominal),
            "s1_s2_max": float(s1_s2_max),
            "flags": {
                "systole_too_short": bool(too_short),
                "systole_too_long": bool(too_long),
                "systole_far_from_nominal": bool(far_from_nominal),
                "diastole_too_short": bool(diastole_too_short),
            },
        })

        snap_here = _effective_snap_s2(snap_s2, s1, s1_next, niv, s2_check=s2)
        if (too_short or too_long or far_from_nominal) and snap_here and insert_spectrum_ctx is not None:
            s2_pred = int(round(s1 + s1_s2_nominal * sample_rate))
            new_s2 = _choose_s2_near(
                s1, s1_next, s2_pred, resnap_half,
                snap_here, insert_spectrum_ctx, sample_rate, n_samples, params, intervals,
            )
            new_s2 = int(max(s1 + 1, min(new_s2, s1_next - 1)))
            if new_s2 != s2:
                new_corrections.append({
                    "type": "resnap_s2",
                    "cycle": int(i), "s1": int(s1),
                    "old_s2": int(s2), "new_s2": int(new_s2), "s2_pred": int(s2_pred),
                })
                new_s2_events[i] = int(new_s2)
                changed = True

    return new_s2_events, new_corrections, cycle_diagnostics, changed


# ─────────────────────────────────────────────────────────────────────────────
# Pass C — phase-shift cascade corrections
# ─────────────────────────────────────────────────────────────────────────────

def _pass_c_phase_correction(
    s1_list: List[int],
    s2_events: List[int],
    cycle_diagnostics: List[Dict[str, Any]],
    all_raw_peaks: np.ndarray,
    pc: Dict,
    lt_series: Optional[pd.Series],
    fallback_bpm: float,
    audio_envelope: np.ndarray,
    analysis_data: Dict,
    n_samples: int,
    sample_rate: int,
    params: Dict,
    enable_phase_correction: bool,
    phase_min_score_delta: float,
    local_peak_window_samples: int,
    local_peak_window_ms: float,
    local_peak_sensitivity: float,
    s1_search_half: int,
    min_sep_samples: int,
    snap_s2: bool,
    snap_half: int,
    insert_spectrum_ctx: Optional[Dict],
    noise_ivs: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[List[int], List[int], List[Dict[str, Any]], bool]:
    """
    Phase-shift cascade corrections (one fix per call; outer loop handles multiples).

    C.1  Remove false S1 (both systole + diastole too short).
    C.2  Demote S1_next to S2 (diastole too short, S1_next looks like S2).
    C.3  Find faint S2 (systole too long, Pass A already failed).

    Returns (new_s1_list, new_s2_events, new_corrections, changed).
    """
    if not enable_phase_correction or not cycle_diagnostics:
        return s1_list, s2_events, [], False

    niv = noise_ivs or []
    new_s1_list = list(s1_list)
    new_s2_events = list(s2_events)

    # ── C.1: Remove false S1 ─────────────────────────────────────────────────
    for diag in cycle_diagnostics:
        i = diag["i"]
        if i + 1 >= len(new_s1_list):
            continue
        s1     = diag["s1"]
        s1_next = diag["s1_next"]
        systole  = diag["systole_sec"]
        diastole = diag["diastole_sec"]
        diastole_min_c = diag.get("diastole_min_sec", 0.0)
        s1_s2_min_c    = diag["s1_s2_min"]
        if not (systole < s1_s2_min_c and diastole < diastole_min_c):
            continue
        suspect = int(s1_next)
        entry = pc.get(suspect) or {}
        ls = entry.get("label_scores") if isinstance(entry, dict) else None
        if not isinstance(ls, dict):
            continue
        noise_score = float(ls.get("noise", 0.0))
        s1_score    = float(ls.get("S1",    0.0))
        if noise_score - s1_score < phase_min_score_delta:
            continue
        min_feasible = diag.get("min_feasible_cycle_sec", 0.0)
        merged_next  = int(new_s1_list[i + 2]) if i + 2 < len(new_s1_list) else n_samples
        merged_span  = (merged_next - int(s1)) / float(sample_rate)
        if min_feasible > 0 and merged_span < min_feasible:
            continue
        new_s1_list = [p for p in new_s1_list if p != suspect]
        new_s2_events = _rebuild_s2_events(
            new_s1_list, lt_series, fallback_bpm,
            sample_rate, params, snap_s2, snap_half, n_samples, insert_spectrum_ctx,
            noise_ivs=niv,
        )
        corr = {
            "type": "remove_false_s1", "cycle": int(i), "s1": int(s1),
            "removed_s1": int(suspect),
            "systole_sec": float(systole), "diastole_sec": float(diastole),
            "diastole_min_sec": float(diastole_min_c),
            "noise_score": float(noise_score), "s1_score": float(s1_score),
        }
        logging.info(
            "Pass 3 C.1: removed false S1 at sample %d "
            "(cycle %d, systole=%.3fs, diastole=%.3fs/min=%.3fs).",
            suspect, i, systole, diastole, diastole_min_c,
        )
        return new_s1_list, new_s2_events, [corr], True

    # ── C.2: Demote S1_next to S2 ────────────────────────────────────────────
    for diag in cycle_diagnostics:
        i = diag["i"]
        if i + 1 >= len(new_s1_list):
            continue
        if not diag["flags"].get("diastole_too_short", False):
            continue
        s1     = diag["s1"]
        s1_next = diag["s1_next"]
        entry = pc.get(int(s1_next)) or {}
        ls = entry.get("label_scores") if isinstance(entry, dict) else None
        if not isinstance(ls, dict):
            continue
        s2_score = float(ls.get("S2", 0.0))
        s1_score = float(ls.get("S1", 0.0))
        if s2_score - s1_score < phase_min_score_delta:
            continue
        new_s2 = int(s1_next)
        upper_bound = int(new_s1_list[i + 2]) if i + 2 < len(new_s1_list) else n_samples
        bpm_here = _bpm_at_time(int(s1) / float(sample_rate), lt_series, fallback_bpm)
        ivs_here = calculate_bpm_intervals(bpm_here, params)
        s2_min_here      = float(ivs_here.get("s2_min",      0.010))
        diastole_min_here = float(ivs_here.get("diastole_min", 0.08))
        earliest_new_s1  = new_s2 + max(1, int(s2_min_here * sample_rate))
        expected_dia_here = max(diastole_min_here, float(ivs_here.get("s2_s1_nominal", 0.35)))
        t_new_s1 = earliest_new_s1 / float(sample_rate) + max(0.0, expected_dia_here - s2_min_here)
        new_s1_cand: Optional[int] = None
        new_s1_cand = _choose_s1_near(
            t_new_s1, s1_search_half, min_sep_samples,
            all_raw_peaks, pc, n_samples, sample_rate,
        )
        if new_s1_cand is not None and (new_s1_cand < earliest_new_s1 or new_s1_cand >= upper_bound):
            new_s1_cand = None
        if new_s1_cand is None:
            sens = _find_sensitive_peaks_near(
                t_new_s1, local_peak_window_samples, local_peak_sensitivity,
                audio_envelope, analysis_data, n_samples, sample_rate, params,
            )
            if sens is not None and earliest_new_s1 <= sens < upper_bound:
                new_s1_cand = sens
        if new_s1_cand is None or new_s1_cand < earliest_new_s1 or new_s1_cand >= upper_bound:
            continue
        new_s2_events[i] = new_s2
        new_s1_list = [p for p in new_s1_list if p != int(s1_next)]
        new_s1_list.append(new_s1_cand)
        new_s1_list = sorted(list(dict.fromkeys(new_s1_list)))
        new_s2_events = _rebuild_s2_events(
            new_s1_list, lt_series, fallback_bpm,
            sample_rate, params, snap_s2, snap_half, n_samples, insert_spectrum_ctx,
            noise_ivs=niv,
        )
        corr = {
            "type": "flip_demote_s1", "cycle": int(i), "s1": int(s1),
            "old_s1_next": int(s1_next), "new_s2_for_cycle": int(new_s2),
            "new_s1_next": int(new_s1_cand),
            "s2_score": float(s2_score), "s1_score": float(s1_score),
        }
        logging.info(
            "Pass 3 C.2: flipped S1@%d\u2192S2, new S1 at %d (cycle %d, diastole was %.3fs).",
            s1_next, new_s1_cand, i, diag["diastole_sec"],
        )
        return new_s1_list, new_s2_events, [corr], True

    # ── C.3: Find faint S2 ───────────────────────────────────────────────────
    for diag in cycle_diagnostics:
        i = diag["i"]
        if not diag["flags"].get("systole_too_long", False):
            continue
        s1     = diag["s1"]
        s1_next = diag["s1_next"]
        if _hf_noise_disables_s2_snap(int(s1), int(s1_next), niv, s2_check=int(diag["s2"])):
            continue
        bpm_c  = diag["bpm"]
        ivs_c  = calculate_bpm_intervals(bpm_c, params)
        s1_s2_nominal_c = float(ivs_c.get("s1_s2_nominal", 0.30))
        t_s2_pred       = int(s1) / float(sample_rate) + s1_s2_nominal_c
        search_half_sec = local_peak_window_ms / 2000.0

        new_s2: Optional[int] = None
        method_used: Optional[str] = None
        spectral_score: Optional[float] = None

        sens = _find_sensitive_peaks_near(
            t_s2_pred, local_peak_window_samples, local_peak_sensitivity,
            audio_envelope, analysis_data, n_samples, sample_rate, params,
        )
        if sens is not None and int(s1) < sens < int(s1_next):
            new_s2 = sens
            method_used = "sensitive_peak"

        if new_s2 is None:
            sp_result = _choose_s2_spectral(
                t_s2_pred, search_half_sec, insert_spectrum_ctx, params, sample_rate, n_samples,
            )
            if sp_result is not None:
                sp_idx, sp_score = sp_result
                if int(s1) < sp_idx < int(s1_next):
                    new_s2 = sp_idx
                    spectral_score = sp_score
                    method_used = "spectral_s2"

        if new_s2 is None:
            continue

        new_systole  = (new_s2 - int(s1))     / float(sample_rate)
        new_diastole = (int(s1_next) - new_s2) / float(sample_rate)
        s2_min_c      = float(ivs_c.get("s2_min",     0.010))
        diastole_min_c = float(ivs_c.get("diastole_min", 0.08))
        if new_systole < float(ivs_c.get("s1_s2_min", 0.12)):
            continue
        if new_diastole < s2_min_c + diastole_min_c:
            continue

        new_s2_events[i] = new_s2
        corr: Dict[str, Any] = {
            "type": method_used, "cycle": int(i), "s1": int(s1),
            "new_s2": int(new_s2), "t_s2_pred_sec": float(t_s2_pred),
            "new_systole_sec": float(new_systole), "new_diastole_sec": float(new_diastole),
        }
        if spectral_score is not None:
            corr["spectral_score"] = float(spectral_score)
        logging.info(
            "Pass 3 C.3: placed faint S2 at sample %d via %s "
            "(cycle %d, systole %.3fs\u2192%.3fs).",
            new_s2, method_used, i, diag["systole_sec"], new_systole,
        )
        return new_s1_list, new_s2_events, [corr], True

    return new_s1_list, new_s2_events, [], False


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_pass3_correction(
    s1_peaks: np.ndarray,
    all_raw_peaks: np.ndarray,
    analysis_data: Dict,
    audio_envelope: np.ndarray,
    sample_rate: int,
    params: Dict,
    wav_file_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Pass 3 (bridge): correction + dense per-sample cardiac-state timeline.

    Mutates and returns analysis_data with keys:
      pass3_state_labels, pass3_state_labels_encoding,
      pass3_state_boundaries, pass3_state_boundaries_before,
      pass3_corrections, pass3_cycle_diagnostics,
      pass3_spectral_context  (template arrays only; bandpass_audio not stored).
      pass3_noise_unreliable_windows_samples  (when noise_event_segments exist: HF intervals as sample indices for HTML).
      When pass3_enable_noise_repair: labels in those windows are set to unknown (encoding includes "unknown").
    """
    # Design intent (Pass 3 architecture)
    # ----------------------------------
    # Pass 2 provides labeled events (S1 peaks + paired S2 candidates + BPM belief). Pass 3's job is to:
    #   1) Generate an initial *state sequence* from those Pass 2 labels, then
    #   2) Correct that *state sequence* (not just isolated events) under timing constraints.
    #
    # Direction we are moving toward:
    #   Treat each cardiac cycle as the canonical half-open interval [S1_i, S1_{i+1}), and make the four
    #   phases (S1, systole, S2, diastole) a strict partition of that interval. All four phases have equal
    #   weighting in the logic (timing plausibility / repairs should reason about the whole cycle, not only S2).
    #
    # Transitional note:
    #   The current implementation includes a "trim diastole overlap" normalization step because transient-edge
    #   detection can start S1 before its peak while diastole is anchored to the next peak. As the codebase
    #   converges on strict [S1_i, S1_{i+1}) partitions, that trimming should become unnecessary.
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

    snap_s2           = bool(params.get(
        "pass3_align_s2_to_s2_spectral_profile",
        True,
    ))
    snap_window_ms    = float(params.get(
        "pass3_align_s2_window_ms",
        120.0,
    ))
    snap_half         = max(1, int(round(0.5 * snap_window_ms * sample_rate / 1000.0)))
    resnap_window_ms  = float(params.get("pass3_resnap_s2_window_ms",  220.0))
    resnap_half       = max(1, int(round(0.5 * resnap_window_ms * sample_rate / 1000.0)))
    systole_slack     = float(params.get("pass3_systole_slack_frac",   0.15))
    diastole_slack    = float(params.get("pass3_diastole_slack_frac",  0.20))

    max_iters          = int(params.get("pass3_correction_max_iters",        32))
    min_sep_samples    = int(float(params.get("min_peak_distance_sec", 0.10)) * sample_rate)

    enable_phase_corr    = bool(params.get("pass3_enable_phase_correction",   True))
    phase_min_score_delta = float(params.get("pass3_phase_min_score_delta",  0.15))
    local_peak_window_ms  = float(params.get("pass3_local_peak_window_ms",   160.0))
    local_peak_win_samp   = max(1, int(round(0.5 * local_peak_window_ms * sample_rate / 1000.0)))
    local_peak_sens       = float(params.get("pass3_local_peak_sensitivity_factor", 0.6))

    pc = analysis_data.get("peak_classifications") or {}

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
    # ── Build spectral context (S1 + S2 templates) ───────────────────────────
    insert_spectrum_ctx: Optional[Dict] = None
    if bool(params.get("pass3_insert_use_spectrum", True)) and wav_file_path and os.path.isfile(wav_file_path):
        try:
            insert_spectrum_ctx = prepare_pass3_s1_insert_context(
                wav_file_path, pc, sample_rate, audio_envelope, params,
            )
            if insert_spectrum_ctx is not None:
                logging.info(
                    "Pass 3: spectral context ready "
                    "(n_s1_template=%s, n_s2_template=%s, sr=%s).",
                    insert_spectrum_ctx.get("n_s1_template"),
                    insert_spectrum_ctx.get("n_s2_template"),
                    insert_spectrum_ctx.get("full_sr"),
                )
        except Exception as exc:
            logging.warning("Pass 3: could not build spectral context: %s", exc)

    # Build the BPM prior from clean RR intervals (exclude any S1→S1 that intersects HF noise).
    # If we can't build it (too few clean intervals), Pass 3 falls back to fallback_bpm.
    lt_pass3 = _build_lt_bpm_series_from_clean_rr(
        peaks_out,
        noise_ivs_pass3,
        sample_rate,
        n_samples,
        params=params,
    )

    # ── S2 placement: Pass 2 pair seed → spectral snap (_rebuild_s2_events) ────
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
        s1_list, lt_pass3, fallback_bpm,
        sample_rate, params, snap_s2, snap_half, n_samples, insert_spectrum_ctx,
        noise_ivs=noise_ivs_pass3,
        seed_s2_events=s2_seed,
    )

    # ── Before-correction snapshot for HTML before/after visualization ────────
    state_boundaries_before = _build_state_boundaries_before_from_cycles(
        s1_list, s2_events, s1_half, s2_half, n_samples,
    )

    # ── Correction loop ───────────────────────────────────────────────────────
    corrections: List[Dict[str, Any]] = []
    cycle_diagnostics: List[Dict[str, Any]] = []

    for _iter in range(max_iters):
        # Sync s2_events length with current s1_list.
        n_cycles = max(0, len(s1_list) - 1)
        if len(s2_events) != n_cycles:
            s2_events = s2_events[:n_cycles]
            while len(s2_events) < n_cycles:
                s2_events.append(int(s1_list[len(s2_events)]))

        s2_events, corrs_a, cycle_diagnostics, changed_a = _pass_a_resnap_s2(
            s1_list, s2_events, lt_pass3, fallback_bpm,
            sample_rate, params, snap_s2, resnap_half, n_samples, insert_spectrum_ctx,
            systole_slack, diastole_slack, noise_ivs=noise_ivs_pass3,
        )
        corrections.extend(corrs_a)

        s1_list, s2_events, corrs_c, changed_c = _pass_c_phase_correction(
            s1_list, s2_events, cycle_diagnostics,
            all_raw_peaks, pc, lt_pass3, fallback_bpm,
            audio_envelope, analysis_data, n_samples, sample_rate, params,
            enable_phase_corr, phase_min_score_delta,
            local_peak_win_samp, local_peak_window_ms, local_peak_sens,
            0, min_sep_samples,
            snap_s2, snap_half, insert_spectrum_ctx, noise_ivs=noise_ivs_pass3,
        )
        corrections.extend(corrs_c)

        if not (changed_a or changed_c):
            break

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
            snap_paint = _effective_snap_s2(snap_s2, s1, s1_next, noise_ivs_final, s2_check=None)
            s2 = _choose_s2_near(
                s1, s1_next, s2_pred, snap_half,
                snap_paint, insert_spectrum_ctx, sample_rate, n_samples, params, ivs,
            )
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

        snap_reason = _effective_snap_s2(snap_s2, s1, s1_next, noise_ivs_final, s2_check=s2)
        _reasoning = _build_reasoning_payload(
            s1, s1_start, s1_end, s2, s2_start, s2_end, s1_next,
            _ivs_r, _direct, _cascade, _bef_s2, _bef_s1nxt,
            _sr_f, _SHIFT_THRESH_SAMP, snap_reason,
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

    # ── HF noise: clear then rebuild cardiac labels in unreliable audio ─────────
    _noise_repair_on = bool(
        params.get("pass3_enable_noise_repair", params.get("pass3_enable_noise_s2_repair", True)),
    )
    if _noise_repair_on and noise_ivs_final:
        _ms_t, _ms_d = _pass3_measured_systole_series_from_boundaries(
            state_boundaries, sample_rate, noise_ivs=noise_ivs_final, params=params,
        )
        _md_t, _md_d = _pass3_measured_diastole_series_from_boundaries(
            state_boundaries, sample_rate, noise_ivs=noise_ivs_final, params=params,
        )
        # Make measured phase-duration curves continuous-in-time (dense rasters) like the BPM prior.
        _ms_t_r, _ms_d_r = _dense_raster_from_points(
            _ms_t, _ms_d, n_samples, sample_rate, dt_sec=0.05,
        )
        _md_t_r, _md_d_r = _dense_raster_from_points(
            _md_t, _md_d, n_samples, sample_rate, dt_sec=0.05,
        )
        analysis_data["pass3_measured_systole_times"] = _ms_t_r
        analysis_data["pass3_measured_systole"] = _ms_d_r
        analysis_data["pass3_measured_diastole_times"] = _md_t_r
        analysis_data["pass3_measured_diastole"] = _md_d_r
        # Build a BPM prior from clean RR intervals only (exclude any S1→S1 that intersects noise).
        # This avoids using possibly-corrupted BPM belief inside the noisy spans.
        _lt_clean = _build_lt_bpm_series_from_clean_rr(
            peaks_out, noise_ivs_final, sample_rate, n_samples, params=params,
        )
        # Store a dense raster for plotting/debug + use it as the rebuild prior.
        _lt_for_raster = _lt_clean if _lt_clean is not None else lt_pass3
        _bpm_prior_t, _bpm_prior = _dense_bpm_raster_from_series(
            _lt_for_raster, n_samples, sample_rate, fallback_bpm, dt_sec=0.05,
        )
        analysis_data["pass3_bpm_prior_times"] = _bpm_prior_t
        analysis_data["pass3_bpm_prior"] = _bpm_prior
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

        # ── Step 3: Insert missing states in large gaps (state-level) ─────────────
        # Uses the same cursoring + scaling logic as _pass3_rebuild_unknown_runs.
        if bool(params.get("pass3_enable_gap_state_insert", True)):
            _gap_debug: List[Dict[str, Any]] = []
            state_labels, state_boundaries = _pass3_insert_missing_states_in_large_gaps(
                state_labels,
                state_boundaries,
                n_samples,
                (_bpm_prior_t, _bpm_prior),
                fallback_bpm,
                sample_rate,
                params,
                measured_systole_t=_ms_t_r,
                measured_systole_dur=_ms_d_r,
                measured_diastole_t=_md_t_r,
                measured_diastole_dur=_md_d_r,
                debug_windows_out=_gap_debug,
            )
            if _gap_debug:
                analysis_data["pass3_large_gap_windows_samples"] = list(_gap_debug)

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
    # Cleaned systole/diastole series (noise-removed + MAD + smoothed) for plotting.
    if _noise_repair_on and noise_ivs_final:
        analysis_data["pass3_measured_systole_t"]   = _ms_t
        analysis_data["pass3_measured_systole_dur"]  = _ms_d
        analysis_data["pass3_measured_diastole_t"]  = _md_t
        analysis_data["pass3_measured_diastole_dur"] = _md_d
    analysis_data["pass3_state_labels"]          = state_labels
    analysis_data["pass3_state_labels_encoding"] = dict(STATE_LABELS_ENCODING)
    analysis_data["pass3_state_boundaries"]        = state_boundaries
    analysis_data["pass3_state_boundaries_before"] = state_boundaries_before
    analysis_data["pass3_corrections"]    = corrections
    analysis_data["pass3_cycle_diagnostics"] = cycle_diagnostics

    # Store lightweight spectral templates (no bandpass_audio) for emissions / Pass 4.
    if insert_spectrum_ctx is not None:
        analysis_data["pass3_spectral_context"] = {
            k: v for k, v in insert_spectrum_ctx.items() if k != "bandpass_audio"
        }
    else:
        analysis_data["pass3_spectral_context"] = None

    logging.info(
        "Pass 3: corrected peaks=%d, state timeline n=%d samples, %d cycles, %d corrections.",
        int(len(peaks_out)), n_samples,
        max(0, len(peaks_out) - 1),
        int(len(corrections)),
    )
    return peaks_out, analysis_data
