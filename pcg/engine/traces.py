"""Self-describing debug traces: everything the engine computed, as named series.

A Trace carries its own presentation metadata (group, lane, kind, unit, colour,
default visibility), so the workspace draws any trace without per-trace code.

Adding a trace from inside the engine is one line wherever `analysis_data` is in
scope:

    emit_trace(analysis_data, "My debug curve", group="Pass 3", lane=LANE_BPM,
               x=times_sec, y=values)

`collect_traces` turns a finished run into the full catalog: the engine's
standard outputs (envelopes, per-pass BPM, scores, intervals, debug windows, HRV,
contractility) plus every emitted trace. Times are seconds from the start of the
analysed audio (i.e. after `analysis_start_sec` is skipped).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .confidence_engine import calculate_bpm_intervals
from .peak_utils import PeakType, get_peak_prominence_details
from .config import param

# Lanes (the workspace's stacked plots). Unknown lanes get their own optional lane.
LANE_SIGNAL = "signal"
LANE_STATES = "states"
LANE_BPM = "bpm"
LANE_INTERVALS = "intervals"
LANE_SCORES = "scores"
LANE_HRV = "hrv"
LANE_CONTRACTILITY = "contractility"

# Kinds
KIND_LINE = "line"      # y over x; uniformly sampled when dt > 0
KIND_POINTS = "points"  # markers at (x, y)
KIND_SPANS = "spans"    # time ranges [x, x_end), optional per-span text

KINDS = (KIND_LINE, KIND_POINTS, KIND_SPANS)

_DEBUG_TRACES_KEY = "debug_traces"


@dataclass
class Trace:
    name: str
    group: str
    lane: str
    kind: str
    x: np.ndarray                      # times (s) / span starts; empty for uniform lines
    y: Optional[np.ndarray] = None     # values (line / points)
    x_end: Optional[np.ndarray] = None  # span ends
    t0: float = 0.0                    # uniform lines: time of y[0]
    dt: float = 0.0                    # uniform lines: sample spacing (0 = use x)
    unit: str = ""
    color: str = ""
    visible: bool = False              # shown by default
    text: Optional[List[str]] = None   # per point / span hover text

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"trace {self.name!r}: unknown kind {self.kind!r}")
        self.x = np.asarray(self.x if self.x is not None else [], dtype=np.float64)
        if self.y is not None:
            self.y = np.asarray(self.y, dtype=np.float64)
        if self.x_end is not None:
            self.x_end = np.asarray(self.x_end, dtype=np.float64)
        n = self.size
        if self.kind != KIND_SPANS and (self.y is None or len(self.y) != n):
            raise ValueError(f"trace {self.name!r}: y must match x")
        if self.kind == KIND_SPANS and (self.x_end is None or len(self.x_end) != n):
            raise ValueError(f"trace {self.name!r}: spans need x_end matching x")
        if self.text is not None and len(self.text) != n:
            raise ValueError(f"trace {self.name!r}: text must have one entry per item")

    @property
    def uniform(self) -> bool:
        return self.kind == KIND_LINE and self.dt > 0

    @property
    def size(self) -> int:
        return len(self.y) if self.uniform else len(self.x)

    def times(self) -> np.ndarray:
        """Item times (s), materialising uniform sampling."""
        if self.uniform:
            return self.t0 + np.arange(len(self.y), dtype=np.float64) * self.dt
        return self.x

    def shifted(self, offset_sec: float) -> "Trace":
        if not offset_sec:
            return self
        if self.uniform:
            return replace(self, t0=self.t0 + offset_sec)
        return replace(
            self,
            x=self.x + offset_sec,
            x_end=None if self.x_end is None else self.x_end + offset_sec,
        )

    def window(self, t_lo: float, t_hi: float) -> np.ndarray:
        """Indices of items intersecting [t_lo, t_hi]."""
        if self.uniform:
            n = len(self.y)
            i0 = int(np.clip(np.floor((t_lo - self.t0) / self.dt), 0, n))
            i1 = int(np.clip(np.ceil((t_hi - self.t0) / self.dt) + 1, 0, n))
            return np.arange(i0, i1)
        if self.kind == KIND_SPANS:
            return np.nonzero((self.x_end >= t_lo) & (self.x <= t_hi))[0]
        return np.nonzero((self.x >= t_lo) & (self.x <= t_hi))[0]

    def value_at(self, t: float) -> Optional[float]:
        """Line value at t (linear interpolation), else None."""
        if self.kind != KIND_LINE or self.size == 0:
            return None
        xs = self.times()
        if t < xs[0] or t > xs[-1]:
            return None
        v = float(np.interp(t, xs, self.y))
        return v if np.isfinite(v) else None


def emit_trace(analysis_data: Dict[str, Any], name: str, *, group: str, lane: str,
               x: Sequence[float], y: Optional[Sequence[float]] = None, kind: Optional[str] = None,
               **kwargs: Any) -> None:
    """Record a debug trace from anywhere in the engine. `kind` defaults to line (y) or spans."""
    if kind is None:
        kind = KIND_LINE if y is not None else KIND_SPANS
    analysis_data.setdefault(_DEBUG_TRACES_KEY, []).append(
        Trace(name=name, group=group, lane=lane, kind=kind, x=x, y=y, **kwargs)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────

def describe(value: Any, indent: str = "") -> List[str]:
    """Flatten a reasoning value (dict / list / scalar) into readable lines."""
    if isinstance(value, dict):
        lines: List[str] = []
        for k, v in value.items():
            if isinstance(v, (dict, list, tuple)) and v:
                lines.append(f"{indent}{k}:")
                lines.extend(describe(v, indent + "  "))
            else:
                lines.append(f"{indent}{k}: {_fmt(v)}")
        return lines
    if isinstance(value, (list, tuple)):
        if all(not isinstance(v, (dict, list, tuple)) for v in value):
            return [indent + ", ".join(_fmt(v) for v in value)]
        out: List[str] = []
        for v in value:
            out.extend(describe(v, indent + "- "))
        return out
    return [indent + _fmt(value)]


def _fmt(v: Any) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4g}"
    if isinstance(v, np.integer):
        return str(int(v))
    return str(v)


# ─────────────────────────────────────────────────────────────────────────────
# Catalog
# ─────────────────────────────────────────────────────────────────────────────

def _arr(a: Any) -> Optional[np.ndarray]:
    if a is None:
        return None
    vals = a.to_numpy(dtype=np.float64) if hasattr(a, "to_numpy") else np.asarray(a, dtype=np.float64)
    return vals.reshape(-1)


def _ok_pair(x: Any, y: Any, min_len: int = 1) -> bool:
    xa, ya = _arr(x), _arr(y)
    return xa is not None and ya is not None and len(xa) == len(ya) and len(xa) >= min_len


def _sample_points(name, group, indices, envelope, sr, **kw) -> Optional[Trace]:
    if indices is None:
        return None
    idx = np.asarray(list(indices), dtype=np.int64).reshape(-1)
    idx = idx[(idx >= 0) & (idx < len(envelope))]
    if idx.size == 0:
        return None
    idx.sort()
    return Trace(name, group, LANE_SIGNAL, KIND_POINTS, x=idx / float(sr), y=envelope[idx], **kw)


def _sample_windows(name, group, windows, sr, text_keys=(), **kw) -> Optional[Trace]:
    starts, ends, texts = [], [], []
    for w in windows or []:
        if isinstance(w, dict):
            a, b = w.get("start_sample", -1), w.get("end_sample", -1)
        else:
            a, b = w[0], w[1]
        try:
            a, b = int(a), int(b)
        except (TypeError, ValueError):
            continue
        if a < 0 or b <= a:
            continue
        starts.append(a / float(sr))
        ends.append(b / float(sr))
        if isinstance(w, dict):
            texts.append("\n".join(f"{k}: {_fmt(w[k])}" for k in text_keys if k in w))
    if not starts:
        return None
    text = texts if text_keys and len(texts) == len(starts) else None
    return Trace(name, group, LANE_SIGNAL, KIND_SPANS, x=starts, x_end=ends, text=text, **kw)


def state_segments(boundaries: Iterable, sr: int) -> List[Dict[str, Any]]:
    """Normalise (start_sample, end_sample, name, meta) boundaries into dicts with seconds."""
    out = []
    for seg in boundaries or []:
        try:
            a0, a1, name = int(seg[0]), int(seg[1]), str(seg[2])
        except (TypeError, ValueError, IndexError):
            continue
        if a1 <= a0:
            continue
        meta = seg[3] if len(seg) > 3 and isinstance(seg[3], dict) else {}
        out.append({
            "start_sample": a0, "end_sample": a1, "state": name,
            "start": a0 / float(sr), "end": a1 / float(sr),
            "rebuild_source": str(meta.get("rebuild_source", "") or ""),
            "reasoning": describe(meta["reasoning"]) if "reasoning" in meta else [],
        })
    out.sort(key=lambda s: (s["start_sample"], s["end_sample"]))
    return out


def interval_series(boundaries: Iterable, sr: int, state: str):
    """(midpoint times, durations) in seconds of every `state` span."""
    segs = [s for s in state_segments(boundaries, sr) if s["state"] == state]
    t = np.array([(s["start"] + s["end"]) / 2.0 for s in segs], dtype=np.float64)
    d = np.array([s["end"] - s["start"] for s in segs], dtype=np.float64)
    return t, d


def expected_intervals_from_bpm(bpm_times, bpm_values, params: Dict):
    """(times, expected systole, expected diastole) in seconds from a BPM curve."""
    t = _arr(bpm_times)
    b = _arr(bpm_values)
    if t is None or b is None or len(t) != len(b) or len(t) == 0:
        return np.array([]), np.array([]), np.array([])
    keep = np.isfinite(b)
    t, b = t[keep], b[keep]
    # The curve is a dense raster; the intervals change slowly, so 1 s resolution is plenty.
    if len(t) > 2:
        step = max(1, int(round(1.0 / max(1e-9, float(np.median(np.diff(t)))))))
        t, b = t[::step], b[::step]
    sys_ = np.empty(len(b))
    dia = np.empty(len(b))
    for i, bpm in enumerate(b):
        iv = calculate_bpm_intervals(float(bpm), params)
        sys_[i] = iv["s1_s2_nominal"]
        dia[i] = iv["s2_s1_nominal"]
    return t, sys_, dia


def _segment_means(times: np.ndarray, values: np.ndarray, segment_sec: float):
    if len(times) == 0:
        return np.array([]), np.array([])
    bins = np.floor(times / segment_sec).astype(np.int64)
    uniq, inv = np.unique(bins, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv)
    return (uniq + 0.5) * segment_sec, sums / counts


def _contractility_traces(peak_classifications, envelope, troughs, sr, params) -> List[Trace]:
    seg = float(param(params, "contractility_average_window_sec"))
    troughs = np.asarray(troughs if troughs is not None else [], dtype=np.int64)
    groups: Dict[str, List[int]] = {"S1": [], "S2": []}
    for idx, entry in (peak_classifications or {}).items():
        pt = entry.get("peak_type", "") if isinstance(entry, dict) else ""
        if PeakType.is_s1(pt):
            groups["S1"].append(int(idx))
        elif PeakType.is_s2(pt):
            groups["S2"].append(int(idx))
    proms = {
        k: np.array([get_peak_prominence_details(i, envelope, troughs)["prominence"] for i in sorted(v)], dtype=np.float64)
        for k, v in groups.items()
    }
    times = {k: np.array(sorted(v), dtype=np.float64) / float(sr) for k, v in groups.items()}
    out = []
    for name, color, keys in (
        ("Average S1 contractility", "#e36f6f", ("S1",)),
        ("Average S2 contractility", "#f0a040", ("S2",)),
        ("Average contractility", "#aaaaaa", ("S1", "S2")),
    ):
        t = np.concatenate([times[k] for k in keys])
        p = np.concatenate([proms[k] for k in keys])
        order = np.argsort(t)
        ct, cv = _segment_means(t[order], p[order], seg)
        if len(ct):
            out.append(Trace(name, "Result", LANE_CONTRACTILITY, KIND_LINE, x=ct, y=cv, color=color))
    return out


def collect_traces(
    analysis_data: Dict[str, Any],
    *,
    sample_rate: int,
    algorithm_envelope: np.ndarray,
    metrics: Optional[Dict[str, Any]],
    params: Dict,
    pass1: Optional[Dict[str, Any]] = None,
    metrics_pass2: Optional[Dict[str, Any]] = None,
) -> List[Trace]:
    """Every trace of one run. `pass1` = {anchor_beats, analysis_data, pass1_bpm} (native path)."""
    sr = float(sample_rate)
    env = np.asarray(algorithm_envelope, dtype=np.float64)
    ad = analysis_data
    traces: List[Trace] = []

    def add(t: Optional[Trace]) -> None:
        if t is not None and t.size > 0:
            traces.append(t)

    def line(name, group, lane, x, y, **kw) -> None:
        if _ok_pair(x, y):
            add(Trace(name, group, lane, KIND_LINE, x=_arr(x), y=_arr(y), **kw))

    def points(name, group, lane, x, y, **kw) -> None:
        if _ok_pair(x, y):
            add(Trace(name, group, lane, KIND_POINTS, x=_arr(x), y=_arr(y), **kw))

    def uniform(name, group, values, **kw) -> None:
        v = _arr(values)
        if v is not None and len(v):
            add(Trace(name, group, LANE_SIGNAL, KIND_LINE, x=[], y=v, dt=1.0 / sr, **kw))

    # ── Preprocessing ────────────────────────────────────────────────────────
    nr = ad.get("noise_removed_envelope")
    uniform("Algorithm envelope", "Preprocessing", env, color="#47a5c4", visible=True)
    if nr is not None:
        uniform("Bandpass envelope", "Preprocessing", ad.get("bandpass_envelope"), color="#3498db")
    uniform("Noise envelope", "Preprocessing", ad.get("inverse_band_envelope"), color="#b85c9e")
    nf = ad.get("dynamic_noise_floor_series")
    if nf is not None and len(nf) == len(env):
        uniform("Dynamic noise floor", "Preprocessing", nf, color="#3cb371")
    add(_sample_points("Troughs", "Preprocessing", ad.get("trough_indices"), env, sr, color="#3cb371"))
    nes = ad.get("noise_event_segments") or []
    if nes:
        add(Trace("HF noise events", "Preprocessing", LANE_SIGNAL, KIND_SPANS,
                  x=[s["start"] for s in nes], x_end=[s["end"] for s in nes],
                  text=[f"duration {s.get('duration_ms')} ms, peak {s.get('peak')}" for s in nes],
                  color="#b85c9e"))

    # ── Pass 1 ───────────────────────────────────────────────────────────────
    if pass1:
        add(_sample_points("Anchor beats", "Pass 1", pass1.get("anchor_beats"), env, sr, color="#ffd24d"))
        p1 = pass1.get("pass1_bpm") or {}
        points("Instant BPM (Pass 1)", "Pass 1", LANE_BPM, p1.get("raw_scatter_times"), p1.get("raw_scatter_bpm"),
               unit="BPM", color="#e74c3c")
        points("Instant BPM (Pass 1, outliers removed)", "Pass 1", LANE_BPM, p1.get("scatter_times"),
               p1.get("scatter_bpm"), unit="BPM", color="#9b59b6")
        line("BPM (Pass 1)", "Pass 1", LANE_BPM, p1.get("curve_times"), p1.get("curve_bpm"), unit="BPM",
             color="#f0a040")
        p1ad = pass1.get("analysis_data") or {}
        line("BPM belief (Pass 1)", "Pass 1", LANE_BPM, p1ad.get("pass2_lt_bpm_times"), p1ad.get("pass2_lt_bpm"),
             unit="BPM", color="#ffb366")

    # ── Pass 2 ───────────────────────────────────────────────────────────────
    line("BPM belief (Pass 2)", "Pass 2", LANE_BPM, ad.get("pass2_lt_bpm_times"), ad.get("pass2_lt_bpm"),
         unit="BPM", color="#ff8c1a", visible=True)
    pcs = ad.get("peak_classifications") or {}
    scored = sorted(
        (int(i), e["label_scores"]) for i, e in pcs.items()
        if isinstance(e, dict) and isinstance(e.get("label_scores"), dict)
    )
    if scored:
        t = np.array([i for i, _ in scored], dtype=np.float64) / sr
        for key, name, color in (("S1", "S1 score", "#e36f6f"), ("S2", "S2 score", "#f0a040"),
                                 ("noise", "Noise score", "#999999")):
            line(name, "Pass 2", LANE_SCORES, t, [100.0 * float(s.get(key, 0.0)) for _, s in scored],
                 unit="%", color=color)
    if metrics_pass2:
        line("BPM (Pass 2)", "Pass 2", LANE_BPM, metrics_pass2.get("bpm_times"), metrics_pass2.get("smoothed_bpm"),
             unit="BPM", color="#c0c0c0")
        points("Instant BPM (Pass 2)", "Pass 2", LANE_BPM, metrics_pass2.get("bpm_times_raw"),
               metrics_pass2.get("instant_bpm_raw"), unit="BPM", color="#e74c3c")
    pairs = ad.get("s1_s2_pairs") or []
    if pairs:
        pt = np.array([(a + b) / 2.0 / sr for a, b in pairs])
        pd_ = np.array([(b - a) / sr for a, b in pairs])
        points("Paired systole (Pass 2)", "Pass 2", LANE_INTERVALS, pt, pd_, unit="s", color="#d9a0ff")

    # ── Pass 3 ───────────────────────────────────────────────────────────────
    line("BPM prior (Pass 3)", "Pass 3", LANE_BPM, ad.get("pass3_bpm_prior_times"), ad.get("pass3_bpm_prior"),
         unit="BPM", color="#66d9ff")
    before = state_segments(ad.get("pass3_state_boundaries_before"), int(sr))
    if before:
        add(Trace("States before repair", "Pass 3", LANE_STATES, KIND_SPANS,
                  x=[s["start"] for s in before], x_end=[s["end"] for s in before],
                  text=[s["state"] for s in before]))
    add(_sample_windows("Noise-unreliable windows", "Pass 3", ad.get("pass3_noise_unreliable_windows_samples"), sr,
                        color="#b85c9e"))
    add(_sample_windows("Large-gap windows", "Pass 3", ad.get("pass3_large_gap_windows_samples"), sr,
                        text_keys=("gap_region_candidate_state", "source_state", "trigger", "bpm_at_mid",
                                   "cycle0_samples", "segment_samples"),
                        color="#ff5b5b"))
    add(_sample_windows("Gap quiet windows", "Pass 3", ad.get("pass3_gap_quiet_windows_samples"), sr,
                        color="#67d1ff"))
    for key, name, color in (
        ("pass3_large_gap_recovered_peaks_insensitive", "Recovered gap peaks (insensitive)", "#b07cff"),
        ("pass3_large_gap_recovered_peaks_sensitive", "Recovered gap peaks (sensitive)", "#67d1ff"),
        ("pass3_gap_decision_peaks_sensitive", "Gap decision peaks (sensitive)", "#ff5b5b"),
    ):
        add(_sample_points(name, "Pass 3", ad.get(key), env, sr, color=color))
    for phase, color in (("systole", "#b07cff"), ("diastole", "#55d68d")):
        for snap, label in (("before_repair", "before repair"), ("final", "final")):
            line(f"Measured {phase} curve ({label})", "Pass 3", LANE_INTERVALS,
                 ad.get(f"pass3_measured_phase_{snap}_{phase}_t"), ad.get(f"pass3_measured_phase_{snap}_{phase}_dur"),
                 unit="s", color=color, visible=(snap == "final"))

    # ── Result ───────────────────────────────────────────────────────────────
    final_bounds = ad.get("pass3_state_boundaries") or ad.get("pass3_state_boundaries_before")
    for phase, color in (("systole", "#b07cff"), ("diastole", "#55d68d")):
        t, d = interval_series(final_bounds, int(sr), phase)
        points(f"Measured {phase}", "Result", LANE_INTERVALS, t, d, unit="s", color=color)
    if metrics:
        line("BPM", "Result", LANE_BPM, metrics.get("bpm_times"), metrics.get("smoothed_bpm"), unit="BPM",
             color="#e0e0e0", visible=True)
        points("Instant BPM", "Result", LANE_BPM, metrics.get("bpm_times_raw"), metrics.get("instant_bpm_raw"),
               unit="BPM", color="#e74c3c")
        et, esys, edia = expected_intervals_from_bpm(metrics.get("bpm_times"), metrics.get("smoothed_bpm"), params)
        line("Expected systole from BPM", "Result", LANE_INTERVALS, et, esys, unit="s", color="#00bcd4")
        line("Expected diastole from BPM", "Result", LANE_INTERVALS, et, edia, unit="s", color="#66d9ff")
        hrv = metrics.get("windowed_hrv_df")
        if hrv is not None and not hrv.empty and "time" in hrv:
            for col, name, color in (("rmssdc", "RMSSDc", "#00e5ff"), ("sdnn", "SDNN", "#ff4dff"),
                                     ("lf_hf_ratio", "LF/HF (windowed)", "#ffeb3b")):
                if col in hrv:
                    line(name, "Result", LANE_HRV, hrv["time"], hrv[col], color=color, visible=(col == "rmssdc"))
    traces.extend(_contractility_traces(pcs, env, ad.get("trough_indices"), sr, params))

    traces.extend(ad.get(_DEBUG_TRACES_KEY) or [])

    names = [t.name for t in traces]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate trace names: {dupes}")
    return traces
