"""Analyses: the saved result of running the engine on one Recording.

File format (`<fingerprint>[-<channel>].analysis.zip` in the library folder):

    manifest.json   fingerprint, parameters, engine code fingerprint, summary,
                    peak/state/trace/defect metadata and per-item text
    arrays/*.npy    every numeric series, referenced by name from the manifest

All times in an Analysis are seconds on the *recording's* clock (the engine's
analysis-start offset is already added), so they line up with the audio and with
Annotations.

An Analysis is **stale** when the current config parameters or engine code differ
from the ones that produced it. Staleness is reported, never acted on silently.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import os
import runpy
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from pcg import recording
from pcg.engine.defects import Defect
from pcg.engine.traces import Trace, describe, state_segments

FORMAT_VERSION = 1
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENGINE_DIR = _REPO_ROOT / "pcg" / "engine"
_SPRINGER_DIR = _REPO_ROOT / "springer2015" / "springer_hsmm"
_CONFIG_FILE = _ENGINE_DIR / "config.py"

# Run settings a user picks per run (run screen / CLI). Everything else comes from config.py.
RUN_SETTING_KEYS = ("use_springer_algorithm", "auto_switch_algorithm", "analysis_start_sec")

ProgressFn = Callable[[str], None]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Peaks:
    """Every detected peak. `label` is the Pass 2 classification; `state` the final state under it."""
    time: np.ndarray
    amp: np.ndarray
    label: List[str]
    state: List[str]
    final_s1: np.ndarray  # bool: in the engine's final S1 list
    s1_score: np.ndarray
    s2_score: np.ndarray
    noise_score: np.ndarray
    reasoning: List[List[str]]

    def __len__(self) -> int:
        return len(self.time)

    def kind(self) -> np.ndarray:
        """'S1' / 'S2' / 'noise' per peak (from the Pass 2 label)."""
        out = np.full(len(self.label), "noise", dtype=object)
        for i, lab in enumerate(self.label):
            s = lab.strip()
            if s.startswith("S1") or s.startswith("Lone S1"):
                out[i] = "S1"
            elif s.startswith("S2"):
                out[i] = "S2"
        return out


@dataclass
class States:
    """The final state timeline (S1 / systole / S2 / diastole / unknown spans). May overlap (a Defect)."""
    start: np.ndarray
    end: np.ndarray
    state: List[str]
    source: List[str]  # rebuild_source ('' = detected from the signal)
    reasoning: List[List[str]]

    def __len__(self) -> int:
        return len(self.start)


@dataclass
class Analysis:
    fingerprint: str
    channel: str
    recording_name: str
    created: str
    params: Dict[str, Any]
    run_settings: Dict[str, Any]
    start_bpm_hint: Optional[float]
    code_fingerprint: str
    algorithm_used: str
    algorithm_switch_reason: Optional[str]
    sample_rate: int
    duration_sec: float        # recording duration
    time_offset_sec: float     # engine time zero on the recording clock
    summary: Dict[str, Any]
    peaks: Peaks
    states: States
    traces: List[Trace]
    defects: List[Defect]
    stale_reasons: List[str] = field(default_factory=list)  # filled on load, not saved

    def trace(self, name: str) -> Optional[Trace]:
        return next((t for t in self.traces if t.name == name), None)

    @property
    def is_stale(self) -> bool:
        return bool(self.stale_reasons)


# ─────────────────────────────────────────────────────────────────────────────
# Staleness
# ─────────────────────────────────────────────────────────────────────────────

def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=_json_default))


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def code_fingerprint() -> str:
    """Hash of the engine's source (config.py excluded: parameters are compared separately)."""
    h = hashlib.blake2b(digest_size=12)
    files = sorted(p for p in _ENGINE_DIR.rglob("*.py") if p != _CONFIG_FILE)
    files += sorted(_SPRINGER_DIR.glob("*.py"))
    for p in files:
        h.update(p.relative_to(_REPO_ROOT).as_posix().encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def current_config_params() -> Dict[str, Any]:
    """DEFAULT_PARAMS as config.py says *now* (re-read from disk, not the imported module)."""
    return runpy.run_path(str(_CONFIG_FILE))["DEFAULT_PARAMS"]


def effective_params(run_settings: Dict[str, Any], config_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = dict(config_params if config_params is not None else current_config_params())
    params.update(run_settings)
    return params


def stale_reasons(analysis: Analysis, config_params: Optional[Dict[str, Any]] = None,
                  code_fp: Optional[str] = None) -> List[str]:
    reasons = []
    now = _jsonable(effective_params(analysis.run_settings, config_params))
    then = analysis.params
    changed = sorted(k for k in set(now) | set(then) if now.get(k) != then.get(k))
    if changed:
        shown = ", ".join(changed[:6]) + (f" (+{len(changed) - 6} more)" if len(changed) > 6 else "")
        reasons.append(f"parameters changed: {shown}")
    if (code_fp or code_fingerprint()) != analysis.code_fingerprint:
        reasons.append("engine code changed")
    return reasons


# ─────────────────────────────────────────────────────────────────────────────
# Building from an engine run
# ─────────────────────────────────────────────────────────────────────────────

def _covering_states(times_samples: np.ndarray, segs: List[Dict[str, Any]]) -> List[str]:
    """State under each sample (last covering segment wins, as the Defect checks assume)."""
    out = [""] * len(times_samples)
    order = np.argsort(times_samples)
    j = 0
    active: List[Dict[str, Any]] = []
    for k in order:
        s = int(times_samples[k])
        while j < len(segs) and segs[j]["start_sample"] <= s:
            active.append(segs[j])
            j += 1
        active = [a for a in active if a["end_sample"] > s]
        if active:
            out[k] = active[-1]["state"]
    return out


def _peaks_from_engine(result, segs, offset: float) -> Peaks:
    from pcg.engine.peak_utils import format_debug_entry

    sr = float(result.sample_rate)
    env = result.algorithm_envelope
    pcs = result.analysis_data.get("peak_classifications") or {}
    idx = sorted({int(i) for i in pcs} | {int(i) for i in np.asarray(result.all_raw_peaks).reshape(-1)}
                 | {int(i) for i in np.asarray(result.peaks).reshape(-1)})
    idx = np.array([i for i in idx if 0 <= i < len(env)], dtype=np.int64)
    final = {int(i) for i in np.asarray(result.peaks).reshape(-1)}
    labels, reasoning = [], []
    scores = np.full((len(idx), 3), np.nan)
    for k, i in enumerate(idx):
        entry = pcs.get(i, pcs.get(np.int64(i)))
        if isinstance(entry, dict):
            labels.append(str(entry.get("peak_type", "") or "Unclassified"))
            reasoning.append([ln.replace("\t", "    ") for ln in format_debug_entry(entry)])
            ls = entry.get("label_scores")
            if isinstance(ls, dict):
                scores[k] = [ls.get("S1", np.nan), ls.get("S2", np.nan), ls.get("noise", np.nan)]
        else:
            labels.append("S1 (final)" if i in final else "Unclassified")
            reasoning.append([])
    return Peaks(
        time=idx / sr + offset,
        amp=env[idx].astype(np.float64) if len(idx) else np.array([]),
        label=labels,
        state=_covering_states(idx, segs),
        final_s1=np.array([i in final for i in idx], dtype=bool),
        s1_score=scores[:, 0], s2_score=scores[:, 1], noise_score=scores[:, 2],
        reasoning=reasoning,
    )


def _summary(result) -> Dict[str, Any]:
    m = result.metrics or {}
    hrv = {k: float(v) for k, v in (m.get("hrv_summary") or {}).items()
           if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v)}
    return _jsonable({
        "bpm": result.bpm_summary,
        "hrv": hrv,
        "gate": result.bpm_failure_report,
        "peak_bpm_time_sec": result.peak_bpm_time_sec,
        "recovery_end_time_sec": result.recovery_end_time_sec,
        "n_final_s1": int(len(result.peaks)),
    })


def build(
    recording_path: os.PathLike | str,
    *,
    fingerprint: str,
    channel: str = recording.CHANNEL_MIXED,
    run_settings: Optional[Dict[str, Any]] = None,
    start_bpm_hint: Optional[float] = None,
    progress: Optional[ProgressFn] = None,
) -> List[Analysis]:
    """Run the engine on a recording; one Analysis per channel (two for channel mode 'all')."""
    from pcg.engine import run_analysis

    run_settings = {k: v for k, v in (run_settings or {}).items() if k in RUN_SETTING_KEYS}
    config_params = current_config_params()
    params = effective_params(run_settings, config_params)
    code_fp = code_fingerprint()
    path = Path(recording_path)
    duration = sf_duration(path)
    out = []
    with recording.temp_workdir() as work:
        for ch, wav in recording.engine_inputs(path, channel, work):
            def _progress(msg: str, _ch=ch) -> None:
                if progress:
                    progress(msg if _ch == recording.CHANNEL_MIXED else f"[{_ch}] {msg}")

            result = run_analysis(wav, dict(params), start_bpm_hint, progress_callback=_progress,
                                  compute_pass2_metrics=True)
            offset = recording.time_offset_sec(params, result.sample_rate)
            segs = state_segments(result.analysis_data.get("pass3_state_boundaries"), result.sample_rate)
            states = States(
                start=np.array([s["start"] for s in segs], dtype=np.float64) + offset,
                end=np.array([s["end"] for s in segs], dtype=np.float64) + offset,
                state=[s["state"] for s in segs],
                source=[s["rebuild_source"] for s in segs],
                reasoning=[s["reasoning"] for s in segs],
            )
            out.append(Analysis(
                fingerprint=fingerprint,
                channel=ch,
                recording_name=path.name,
                created=_dt.datetime.now().isoformat(timespec="seconds"),
                params=_jsonable(params),
                run_settings=_jsonable(run_settings),
                start_bpm_hint=start_bpm_hint,
                code_fingerprint=code_fp,
                algorithm_used=result.algorithm_used,
                algorithm_switch_reason=result.algorithm_switch_reason,
                sample_rate=int(result.sample_rate),
                duration_sec=duration if duration else result.duration_sec + offset,
                time_offset_sec=offset,
                summary=_summary(result),
                peaks=_peaks_from_engine(result, segs, offset),
                states=states,
                traces=[t.shifted(offset) for t in result.traces],
                defects=[d.shifted(offset) for d in result.defects],
            ))
    return out


def sf_duration(path: Path) -> Optional[float]:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# File format
# ─────────────────────────────────────────────────────────────────────────────

class _ArrayWriter:
    def __init__(self, zf: zipfile.ZipFile):
        self.zf = zf
        self.n = 0

    def __call__(self, arr: Optional[np.ndarray], dtype=None) -> Optional[str]:
        if arr is None:
            return None
        a = np.asarray(arr, dtype=dtype) if dtype else np.asarray(arr)
        name = f"arrays/{self.n}.npy"
        self.n += 1
        buf = io.BytesIO()
        np.save(buf, a, allow_pickle=False)
        self.zf.writestr(zipfile.ZipInfo(name), buf.getvalue(), compress_type=zipfile.ZIP_STORED)
        return name


def save(analysis: Analysis, path: os.PathLike | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # per process: identical audio may save in parallel
    with zipfile.ZipFile(tmp, "w") as zf:
        put = _ArrayWriter(zf)
        p, s = analysis.peaks, analysis.states
        manifest = {
            "format": FORMAT_VERSION,
            "fingerprint": analysis.fingerprint,
            "channel": analysis.channel,
            "recording_name": analysis.recording_name,
            "created": analysis.created,
            "params": analysis.params,
            "run_settings": analysis.run_settings,
            "start_bpm_hint": analysis.start_bpm_hint,
            "code_fingerprint": analysis.code_fingerprint,
            "algorithm_used": analysis.algorithm_used,
            "algorithm_switch_reason": analysis.algorithm_switch_reason,
            "sample_rate": analysis.sample_rate,
            "duration_sec": analysis.duration_sec,
            "time_offset_sec": analysis.time_offset_sec,
            "summary": analysis.summary,
            "peaks": {
                "time": put(p.time, np.float64), "amp": put(p.amp, np.float64),
                "final_s1": put(p.final_s1, np.bool_), "s1_score": put(p.s1_score, np.float32),
                "s2_score": put(p.s2_score, np.float32), "noise_score": put(p.noise_score, np.float32),
                "label": p.label, "state": p.state, "reasoning": p.reasoning,
            },
            "states": {
                "start": put(s.start, np.float64), "end": put(s.end, np.float64),
                "state": s.state, "source": s.source, "reasoning": s.reasoning,
            },
            "traces": [
                {
                    "name": t.name, "group": t.group, "lane": t.lane, "kind": t.kind, "unit": t.unit,
                    "role": t.role, "estimate": t.estimate, "visible": t.visible, "t0": t.t0, "dt": t.dt, "text": t.text,
                    "x": None if t.uniform else put(t.x, np.float64),
                    "y": put(t.y, np.float32 if t.uniform else np.float64),
                    "x_end": put(t.x_end, np.float64),
                }
                for t in analysis.traces
            ],
            "defects": [
                {"kind": d.kind, "start": d.start, "end": d.end, "message": d.message} for d in analysis.defects
            ],
        }
        zf.writestr("manifest.json", json.dumps(_jsonable(manifest), ensure_ascii=False),
                    compress_type=zipfile.ZIP_DEFLATED)
    os.replace(tmp, path)


def load(path: os.PathLike | str) -> Analysis:
    with zipfile.ZipFile(path) as zf:
        m = json.loads(zf.read("manifest.json").decode("utf-8"))
        if m.get("format") != FORMAT_VERSION:
            raise ValueError(f"{Path(path).name}: unsupported Analysis format {m.get('format')}")

        def arr(ref: Optional[str]) -> Optional[np.ndarray]:
            if ref is None:
                return None
            with zf.open(ref) as f:
                return np.load(io.BytesIO(f.read()), allow_pickle=False)

        p, s = m["peaks"], m["states"]
        peaks = Peaks(
            time=arr(p["time"]), amp=arr(p["amp"]), label=p["label"], state=p["state"],
            final_s1=arr(p["final_s1"]), s1_score=arr(p["s1_score"]), s2_score=arr(p["s2_score"]),
            noise_score=arr(p["noise_score"]), reasoning=p["reasoning"],
        )
        states = States(start=arr(s["start"]), end=arr(s["end"]), state=s["state"], source=s["source"],
                        reasoning=s["reasoning"])
        traces = [
            Trace(name=t["name"], group=t["group"], lane=t["lane"], kind=t["kind"], unit=t["unit"],
                  role=t.get("role", ""), estimate=t.get("estimate", False), visible=t["visible"], t0=t["t0"], dt=t["dt"], text=t["text"],
                  x=arr(t["x"]) if t["x"] else [], y=arr(t["y"]), x_end=arr(t["x_end"]))
            for t in m["traces"]
        ]
    return Analysis(
        fingerprint=m["fingerprint"], channel=m["channel"], recording_name=m["recording_name"],
        created=m["created"], params=m["params"], run_settings=m["run_settings"],
        start_bpm_hint=m["start_bpm_hint"], code_fingerprint=m["code_fingerprint"],
        algorithm_used=m["algorithm_used"], algorithm_switch_reason=m["algorithm_switch_reason"],
        sample_rate=m["sample_rate"], duration_sec=m["duration_sec"], time_offset_sec=m["time_offset_sec"],
        summary=m["summary"], peaks=peaks, states=states, traces=traces,
        defects=[Defect(d["kind"], d["start"], d["end"], d["message"]) for d in m["defects"]],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Library
# ─────────────────────────────────────────────────────────────────────────────

def default_library_dir() -> Path:
    env = os.environ.get("PCG_LIBRARY")
    return Path(env) if env else _REPO_ROOT / "library"


class Library:
    """App-managed folder of Analyses keyed by Recording fingerprint (never next to the audio)."""

    def __init__(self, root: Optional[os.PathLike | str] = None):
        self.root = Path(root) if root else default_library_dir()
        self.fingerprints = recording.FingerprintCache(self.root / "fingerprints.json")

    def path_for(self, fingerprint: str, channel: str = recording.CHANNEL_MIXED) -> Path:
        suffix = "" if channel == recording.CHANNEL_MIXED else f"-{channel}"
        return self.root / f"{fingerprint}{suffix}.analysis.zip"

    def save(self, analysis: Analysis) -> Path:
        path = self.path_for(analysis.fingerprint, analysis.channel)
        save(analysis, path)
        return path

    def find(self, fingerprint: str) -> List[Path]:
        """Every Analysis of a Recording, newest first."""
        hits = list(self.root.glob(f"{fingerprint}*.analysis.zip"))
        return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)

    def latest(self, fingerprint: str, channel: Optional[str] = None) -> Optional[Path]:
        if channel is not None:
            p = self.path_for(fingerprint, channel)
            return p if p.exists() else None
        hits = self.find(fingerprint)
        return hits[0] if hits else None

    def open(self, path: os.PathLike | str) -> Analysis:
        a = load(path)
        a.stale_reasons = stale_reasons(a)
        return a


def peak_reasoning_text(analysis: Analysis, i: int) -> List[str]:
    """Hover/inspect text of one peak."""
    p = analysis.peaks
    lines = [
        f"{p.label[i]} at {p.time[i]:.3f}s (amp {p.amp[i]:.4g})",
        f"final state: {p.state[i] or '-'}{'  [final S1]' if p.final_s1[i] else ''}",
    ]
    if np.isfinite(p.s1_score[i]):
        lines.append(f"scores S1 {p.s1_score[i]:.2f}  S2 {p.s2_score[i]:.2f}  noise {p.noise_score[i]:.2f}")
    return lines + list(p.reasoning[i])


__all__ = [
    "Analysis", "Library", "Peaks", "States", "build", "code_fingerprint", "current_config_params",
    "describe", "load", "save", "stale_reasons", "peak_reasoning_text",
]
