"""Selected-range analysis for either algorithm, without replacing a full Analysis.

Run on a context window of original audio; map S1/S2 spans to recording time and
clip them to the selection. The caller explicitly applies the returned labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from pcg import analysis, annotation as an, recording
from pcg.engine import run_analysis
from pcg.engine.traces import state_segments

DEFAULT_CONTEXT_SEC = 15.0
ALGORITHMS = ("springer", "native")


@dataclass(frozen=True)
class RangeResult:
    start: float
    end: float
    context_start: float
    context_end: float
    algorithm: str
    spans: an.Spans
    gate_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RangeResult":
        spans = tuple(an.Span(**s) for s in data["spans"])
        an.validate(spans)
        result = cls(**{**data, "spans": spans, "gate_reasons": tuple(data.get("gate_reasons", ()))})
        if not np.isfinite([result.start, result.end, result.context_start, result.context_end]).all():
            raise ValueError("invalid range result times")
        if not 0 <= result.context_start <= result.start < result.end <= result.context_end:
            raise ValueError("invalid range result bounds")
        if result.algorithm not in ALGORITHMS or not spans:
            raise ValueError("range result has no sound labels or an unknown algorithm")
        if any(s.kind not in an.SOUNDS or s.start < result.start or s.end > result.end for s in spans):
            raise ValueError("range result contains labels outside the selection")
        return result


def run_range(recording_path: str | Path, start: float, end: float, *,
              algorithm: str = "springer", channel: str = recording.CHANNEL_MIXED,
              context_sec: float = DEFAULT_CONTEXT_SEC, bpm_hint: Optional[float] = None,
              progress: Optional[Callable[[str], None]] = None) -> RangeResult:
    """Re-detect and classify a selection using either complete algorithm pipeline.

    No audio, Annotation or library files are changed. Start offsets from a full
    Analysis are deliberately ignored: all inputs/outputs use recording time.
    A BPM hint seeds Native and supplies a constant duration prior to Springer.
    """
    if not np.isfinite([start, end, context_sec]).all() or not 0 <= start < end or context_sec < 0:
        raise ValueError("selection needs finite 0 <= start < end and nonnegative context")
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}")
    if channel not in (recording.CHANNEL_MIXED, recording.CHANNEL_LEFT, recording.CHANNEL_RIGHT):
        raise ValueError("choose one channel for a selected-range rerun")
    if bpm_hint is not None and (not np.isfinite(bpm_hint) or not 20 <= bpm_hint <= 300):
        raise ValueError("BPM hint must be between 20 and 300")

    if progress:
        progress("Loading selection and surrounding audio…")
    audio, offset = recording.load_range(recording_path, max(0.0, start - context_sec), end + context_sec)
    context_end = offset + audio.duration_sec
    if end > context_end + 1e-6:
        raise ValueError("selection extends beyond the recording")
    end = min(end, context_end)
    params = analysis.current_config_params()
    params.update(use_springer_algorithm=algorithm == "springer", auto_switch_algorithm=False,
                  analysis_start_sec=0.0)
    heart_rate_curve = None
    if bpm_hint is not None and algorithm == "springer":
        heart_rate_curve = (np.array([0.0, audio.duration_sec]), np.array([bpm_hint, bpm_hint]))
    with recording.temp_workdir() as work:
        wav = Path(work) / "selection.wav"
        sf.write(str(wav), audio.channel(channel), audio.sample_rate, subtype="FLOAT")
        result = run_analysis(str(wav), params, start_bpm_hint=bpm_hint,
                              heart_rate_curve=heart_rate_curve, progress_callback=progress)
    segments = state_segments(result.analysis_data.get("pass3_state_boundaries"), result.sample_rate)
    sounds = an.from_states([s["start"] + offset for s in segments],
                            [s["end"] + offset for s in segments], [s["state"] for s in segments])
    spans = an.clip_sounds(sounds, start, end)
    if not spans:
        raise ValueError("No S1/S2 sounds detected inside the selection; existing labels were kept")
    return RangeResult(start, end, offset, context_end, result.algorithm_used, spans,
                       tuple(result.bpm_failure_report.get("reasons", ())))
