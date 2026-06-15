"""
Tests for the evidence-anchored phase decision (phase_decision.py).

Covers the timing lens, the gap-distribution context, the grammar-constrained
decoder, and a fidelity check that timing-only decoding reproduces the legacy
`_phase_subset_dp` chain exactly (so wiring it in changes only the substrate and
ordering, never the decoder maths).
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase_decision import (  # noqa: E402
    PhaseContext,
    TimingLens,
    build_context,
    decode_phase,
    decide_phase,
)
from correction import _phase_subset_dp  # noqa: E402


def _ctx(sys0=60.0, dia0=240.0, bpm=120.0, ceiling=200.0):
    return PhaseContext(sample_rate=600.0, sys0=sys0, dia0=dia0, cycle=sys0 + dia0,
                        bpm_est=bpm, bpm_ceiling=ceiling)


def test_timing_lens_scores_gap_spans_only():
    lens = TimingLens()
    ctx = _ctx()
    # systole span of exactly sys0 is perfect (score 0); off by sys0 scores -1.
    assert lens.score_span("systole", 0, 60, ctx) == 0.0
    assert lens.score_span("systole", 0, 120, ctx) == -1.0
    assert lens.score_span("diastole", 0, 240, ctx) == 0.0
    # abstains on sound spans
    assert lens.score_span("S1", 0, 60, ctx) == 0.0
    assert lens.score_span("S2", 0, 60, ctx) == 0.0


def test_timing_lens_inactive_above_ceiling():
    lens = TimingLens()
    assert lens.active(_ctx(bpm=120.0, ceiling=200.0)) is True
    assert lens.active(_ctx(bpm=260.0, ceiling=200.0)) is False


def test_build_context_learns_priors_from_gaps():
    centers = [0, 60, 300, 360, 600, 660, 900]  # short/long alternating gaps
    ctx = build_context(centers, sample_rate=600.0, bpm_ceiling=200.0)
    assert ctx is not None
    assert ctx.sys0 == 60.0 and ctx.dia0 == 240.0
    assert abs(ctx.bpm_est - 120.0) < 1e-6
    assert build_context([0, 60, 300], 600.0, 200.0) is None  # too few sounds


def test_decoder_recovers_alternating_phase():
    centers = [0, 60, 300, 360, 600, 660, 900]
    chain = decide_phase(centers, sample_rate=600.0, bpm_ceiling=200.0, skip_penalty=2.0)
    assert chain == [(0, "S1"), (1, "S2"), (2, "S1"), (3, "S2"),
                     (4, "S1"), (5, "S2"), (6, "S1")]


def test_no_decision_when_lens_inactive():
    centers = [0, 60, 300, 360, 600, 660, 900]
    # ceiling below the recording's bpm -> timing lens inactive -> no decision.
    assert decide_phase(centers, 600.0, bpm_ceiling=50.0, skip_penalty=2.0) == []
    # too few sounds
    assert decide_phase([0, 60, 300], 600.0, 200.0, 2.0) == []


def test_timing_only_matches_legacy_decoder():
    # Fidelity: with only an active TimingLens and no emission lens, the decoder is
    # the negation of the legacy cost DP, so the chosen chain must be identical.
    rng = random.Random(1234)
    for _ in range(40):
        n = rng.randint(4, 14)
        t = 0.0
        centers = []
        for _k in range(n):
            centers.append(t)
            t += rng.choice([55, 65, 235, 245]) + rng.uniform(-8, 8)
        ctx = build_context(centers, sample_rate=600.0, bpm_ceiling=10_000.0)
        if ctx is None:
            continue
        skip_pen = 2.0
        new = decode_phase(centers, ctx, [TimingLens(1.0)], skip_pen)
        old = _phase_subset_dp(centers, ctx.sys0, ctx.dia0, ctx.cycle, skip_pen)
        assert new == old
