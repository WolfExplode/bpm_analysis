"""
Phase decision — decide S1/S2 once, from stable Beats, via a lens combiner.

This is the evidence-anchored replacement for the two post-hoc swap stages
(`_pass3_global_phase_correction`, `_pass3_interval_phase_relabel`). See
docs/adr/0004-phase-decided-once-from-stable-beats.md.

Shape (CONTEXT vocabulary):
  * The decision is made on a fixed list of Beat centres (sample indices) — the
    detected S1/S2 sounds, frozen before any Pass 3 stage mutates geometry.
  * Each Scoring lens scores one labelled span. Two kinds of evidence:
      - emission (per-sound)   : "does this S1/S2 span sound like S1?" — shape,
                                 and a future spectral lens. (none implemented yet)
      - duration (per-gap)     : "is this gap short → systole, long → diastole?" —
                                 the timing lens.
  * A grammar-constrained decoder (subset-DP) picks the cycle-valid alternating
    S1/S2 labelling that maximises the weighted sum of span scores. Sounds left
    off the chain are spurious (charged a skip penalty).

Higher score = more likely. Lenses return 0.0 to abstain (wrong span kind, or
outside their validity regime). A lens that is not `active` for a recording is
dropped entirely; if no lens is active the decoder makes no decision (returns []),
preserving the "don't relabel where we have no trustworthy signal" behaviour.

Pure module — numpy only, no pipeline imports — so it is unit-testable in
isolation (the timing lens needs no audio at all).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

# A span carries one cardiac meaning. Emission lenses read S1/S2; duration lenses
# read systole/diastole.
Kind = str  # one of: "S1", "systole", "S2", "diastole"


@dataclass
class PhaseContext:
    """Fixed, per-recording evidence the lenses read. Built once from the stable
    Beat centres before any geometry is mutated."""
    sample_rate: float
    sys0: float          # expected systole duration, samples (gap-distribution prior)
    dia0: float          # expected diastole duration, samples
    cycle: float         # sys0 + dia0
    bpm_est: float       # rate from sound-to-sound gaps (label-independent)
    bpm_ceiling: float   # above this the systole<diastole inequality is unreliable
    audio_envelope: Optional[np.ndarray] = None  # for emission lenses (future)


class Lens(Protocol):
    """A scoring lens. `weight` scales its contribution; `active` gates it by the
    recording's regime; `score_span` scores one labelled span (higher = better,
    0.0 = abstain)."""
    weight: float

    def active(self, ctx: PhaseContext) -> bool: ...

    def score_span(self, kind: Kind, start: float, end: float, ctx: PhaseContext) -> float: ...


class TimingLens:
    """Duration evidence: systole gaps are short, diastole gaps are long.

    Scores systole/diastole spans by how close their duration is to the
    gap-distribution prior; abstains on S1/S2 spans. Lens-local validity gate:
    inactive above the BPM ceiling, where systole<diastole stops holding."""

    def __init__(self, weight: float = 1.0):
        self.weight = float(weight)

    def active(self, ctx: PhaseContext) -> bool:
        return bool(np.isfinite(ctx.bpm_est) and 0.0 < ctx.bpm_est < ctx.bpm_ceiling)

    def score_span(self, kind: Kind, start: float, end: float, ctx: PhaseContext) -> float:
        if kind == "systole":
            exp = ctx.sys0
        elif kind == "diastole":
            exp = ctx.dia0
        else:
            return 0.0  # emission territory — not this lens
        if exp <= 0:
            return 0.0
        return -abs((end - start) - exp) / exp


def build_context(
    centers: Sequence[float],
    sample_rate: float,
    bpm_ceiling: float,
    audio_envelope: Optional[np.ndarray] = None,
) -> Optional[PhaseContext]:
    """Learn the duration priors from this recording's own gap distribution.

    Heart rate comes from sound-to-sound gaps (a full cycle ≈ 2 gaps), which is
    label-independent — robust even when the current S1/S2 guesses are flipped.
    Returns None when there are too few sounds to estimate priors.
    """
    n = len(centers)
    if n < 4:
        return None
    gaps = np.diff(np.asarray(centers, dtype=np.float64))
    if gaps.size == 0:
        return None
    med_gap = float(np.median(gaps))
    if med_gap <= 0.0:
        return None
    bpm_est = 60.0 / (2.0 * med_gap / float(sample_rate)) if sample_rate > 0 else 0.0
    short = gaps[gaps <= med_gap]
    long_ = gaps[gaps > med_gap]
    sys0 = float(np.median(short)) if short.size else 0.4 * 2 * med_gap
    dia0 = float(np.median(long_)) if long_.size else 0.6 * 2 * med_gap
    return PhaseContext(
        sample_rate=float(sample_rate),
        sys0=sys0, dia0=dia0, cycle=sys0 + dia0,
        bpm_est=float(bpm_est), bpm_ceiling=float(bpm_ceiling),
        audio_envelope=audio_envelope,
    )


def _gap_kind(prev_state: int, state: int) -> Optional[Kind]:
    """The span between two consecutive chosen sounds. 0=S1, 1=S2."""
    if prev_state == 0 and state == 1:
        return "systole"
    if prev_state == 1 and state == 0:
        return "diastole"
    return None  # same-state step => a beat was skipped; handled structurally


def decode_phase(
    centers: Sequence[float],
    ctx: PhaseContext,
    lenses: Sequence[Lens],
    skip_penalty: float,
    win: int = 6,
    incumbent: Optional[Sequence[Optional[str]]] = None,
    stick_margin: float = 0.0,
) -> List[Tuple[int, str]]:
    """Pick the highest-score alternating S1/S2 chain through the Beat `centers`.

    Score = Σ emission(sound) + Σ duration(gap) − skip_penalty·(skipped sounds)
            + Σ stick_margin·[label kept from Pass 2].
    A same-state step (a skipped beat, ~one extra cycle) pays a structural cost.

    `incumbent[i]` is the label Pass 2 already assigned sound `i` ("S1"/"S2"/None).
    `stick_margin` *taxes a flip*: keeping the incumbent label costs nothing, but
    relabelling a sound to the opposite kind pays the margin, so timing (or a
    future emission lens) must out-score it to flip — the doc's "boost the
    confidence that S1 and S2 has switched" inequality, made explicit. Taxing the
    flip (rather than rewarding the stay) leaves the keep-vs-drop economics
    untouched, so spurious-sound pruning is preserved. With stick_margin=0 (or
    incumbent=None) this is the pure max-score decode, which with only an active
    TimingLens is the negation of legacy `_phase_subset_dp`.

    Returns [(sound_index, "S1"|"S2"), …] in time order, or [] if no lens is
    active (no trustworthy signal — make no decision).
    """
    n = len(centers)
    if n < 4:
        return []
    active = [l for l in lenses if l.active(ctx)]
    if not active:
        return []
    emit_lenses = active  # lenses self-select by kind via score_span returning 0
    NEG = -1e18
    c = [float(x) for x in centers]
    inc = list(incumbent) if incumbent is not None else None
    margin = float(stick_margin)

    def emission(i: int, s: int) -> float:
        kind = "S1" if s == 0 else "S2"
        e = sum(l.weight * l.score_span(kind, c[i], c[i], ctx) for l in emit_lenses)
        if inc is not None and margin and inc[i] is not None and inc[i] != kind:
            e -= margin  # tax a flip; keeping the incumbent label is free
        return e

    def transition(sj: int, s: int, j: int, i: int) -> float:
        gk = _gap_kind(sj, s)
        if gk is None:  # skipped beat: structural cost on a ~full-cycle gap
            dt = c[i] - c[j]
            return -(1.0 + (abs(dt - ctx.cycle) / ctx.cycle if ctx.cycle > 0 else 0.0))
        return sum(l.weight * l.score_span(gk, c[j], c[i], ctx) for l in active)

    dp = [[NEG, NEG] for _ in range(n)]
    back: List[List[Tuple[int, int]]] = [[(-1, -1), (-1, -1)] for _ in range(n)]
    for i in range(n):
        for s in (0, 1):
            best = -skip_penalty * i + emission(i, s)  # start chain here
            bj = (-1, -1)
            for j in range(max(0, i - win), i):
                skipped = i - j - 1
                for sj in (0, 1):
                    v = dp[j][sj] + transition(sj, s, j, i) - skip_penalty * skipped + emission(i, s)
                    if v > best:
                        best = v
                        bj = (j, sj)
            dp[i][s] = best
            back[i][s] = bj

    ei, es, bv = 0, 0, NEG
    for i in range(n):
        for s in (0, 1):
            v = dp[i][s] - skip_penalty * (n - 1 - i)
            if v > bv:
                bv = v
                ei, es = i, s
    out: List[Tuple[int, str]] = []
    i, s = ei, es
    while i >= 0:
        out.append((i, "S1" if s == 0 else "S2"))
        i, s = back[i][s]
    out.reverse()
    return out


def decide_phase(
    centers: Sequence[float],
    sample_rate: float,
    bpm_ceiling: float,
    skip_penalty: float,
    lenses: Optional[Sequence[Lens]] = None,
    audio_envelope: Optional[np.ndarray] = None,
    win: int = 6,
    incumbent: Optional[Sequence[Optional[str]]] = None,
    stick_margin: float = 0.0,
) -> List[Tuple[int, str]]:
    """Convenience: build the context and decode in one call. Defaults to the
    timing lens only (today's sole evidence source)."""
    ctx = build_context(centers, sample_rate, bpm_ceiling, audio_envelope)
    if ctx is None:
        return []
    if lenses is None:
        lenses = [TimingLens(weight=1.0)]
    return decode_phase(centers, ctx, lenses, skip_penalty, win=win,
                        incumbent=incumbent, stick_margin=stick_margin)
