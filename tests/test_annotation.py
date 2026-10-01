"""Annotation model: every edit keeps spans sorted, non-overlapping and positive-length."""
import itertools
import random

import numpy as np
import pytest

from pcg import annotation as an
from pcg.annotation import NOISY, S1, S2, Span

DUR = 100.0


def kinds(spans):
    return [(s.kind, round(s.start, 3), round(s.end, 3)) for s in spans]


def test_place_sound_replaces_overlapping_sounds():
    spans = an.place_sound((), S1, 1.0, 0.1, DUR)
    spans = an.place_sound(spans, S2, 1.3, 0.1, DUR)
    spans = an.place_sound(spans, S2, 1.05, 0.1, DUR)  # overlaps the S1
    assert kinds(spans) == [(S2, 1.0, 1.1), (S2, 1.25, 1.35)]


def test_place_sound_cuts_noisy_instead_of_removing_it():
    spans = an.paint_noisy((), 10, 20, DUR)
    spans = an.place_sound(spans, S1, 15, 1.0, DUR)
    assert kinds(spans) == [(NOISY, 10, 14.5), (S1, 14.5, 15.5), (NOISY, 15.5, 20)]


def test_place_sound_clamps_to_recording():
    spans = an.place_sound((), S1, 0.01, 0.1, DUR)
    assert spans[0].start == 0.0


def test_paint_noisy_removes_touched_sounds_and_merges():
    spans = ()
    for t in (1, 2, 3, 4):
        spans = an.place_sound(spans, S1, t, 0.1, DUR)
    spans = an.paint_noisy(spans, 1.95, 3.0, DUR)
    spans = an.paint_noisy(spans, 2.9, 3.5, DUR)
    assert kinds(spans) == [(S1, 0.95, 1.05), (NOISY, 1.95, 3.5), (S1, 3.95, 4.05)]


def test_resize_noisy():
    spans = an.paint_noisy(an.place_sound((), S1, 5, 0.1, DUR), 1, 2, DUR)
    noisy = [s for s in spans if s.kind == NOISY][0]
    spans = an.resize_noisy(spans, noisy, 1, 5.5, DUR)
    assert kinds(spans) == [(NOISY, 1, 5.5)]


def test_relabel_and_flip_after():
    spans = ()
    for t, k in ((1, S1), (1.3, S2), (2, S1), (2.3, S2)):
        spans = an.place_sound(spans, k, t, 0.1, DUR)
    spans = an.relabel(spans, spans[0])
    assert [s.kind for s in spans] == [S2, S2, S1, S2]
    spans = an.flip_after(spans, 1.9)
    assert [s.kind for s in spans] == [S2, S2, S2, S1]


def test_delete_and_span_at():
    spans = an.place_sound((), S1, 1, 0.1, DUR)
    assert an.span_at(spans, 1.02) == spans[0]
    assert an.span_at(spans, 3) is None
    assert an.delete(spans, spans[0]) == ()


def test_from_states_clips_overlaps_at_midpoint_and_merges_touching():
    spans = an.from_states(
        starts=[0.0, 0.08, 0.5, 0.6, 0.65],
        ends=[0.1, 0.2, 0.6, 0.65, 0.7],
        states=["S1", "S2", "S1", "S1", "systole"],
    )
    assert kinds(spans) == [(S1, 0.0, 0.09), (S2, 0.09, 0.2), (S1, 0.5, 0.65)]
    assert [s.clipped for s in spans] == [True, True, False]
    assert all(s.origin == an.ORIGIN_ALGORITHM for s in spans)


def test_replace_region_keeps_outside_and_trims_noisy():
    spans = an.paint_noisy(an.place_sound((), S2, 1, 0.1, DUR), 4, 8, DUR)
    algo = an.from_states([0.95, 2.0, 5.0, 9.9], [1.05, 2.1, 5.1, 10.1], ["S1", "S2", "S1", "S1"])
    spans = an.replace_region(spans, 1.5, 10.0, algo)
    # The noisy span inside the region is gone (region fully overwritten), the S1 crossing 10.0 is left out.
    assert kinds(spans) == [(S2, 0.95, 1.05), (S2, 2.0, 2.1), (S1, 5.0, 5.1)]


def test_random_edits_never_break_invariant():
    rng = random.Random(7)
    spans = ()
    for _ in range(2000):
        op = rng.choice(["s1", "s2", "noisy", "del", "flip", "resize", "replace"])
        t = rng.uniform(0, DUR)
        if op in ("s1", "s2"):
            spans = an.place_sound(spans, S1 if op == "s1" else S2, t, rng.uniform(0.01, 0.3), DUR)
        elif op == "noisy":
            spans = an.paint_noisy(spans, t, t + rng.uniform(-2, 2), DUR)
        elif op == "del" and spans:
            spans = an.delete(spans, rng.choice(spans))
        elif op == "flip":
            spans = an.flip_after(spans, t)
        elif op == "resize":
            noisy = [s for s in spans if s.kind == NOISY]
            if noisy:
                s = rng.choice(noisy)
                spans = an.resize_noisy(spans, s, s.start + rng.uniform(-1, 1), s.end + rng.uniform(-1, 1), DUR)
        elif op == "replace":
            algo = an.from_states(*zip(*[(x, x + 0.1, rng.choice(["S1", "S2"])) for x in sorted(rng.uniform(0, DUR) for _ in range(20))]))
            spans = an.replace_region(spans, t, t + 5, algo)
        an.validate(spans)
        assert all(0 <= s.start < s.end <= DUR for s in spans)
        for a, b in itertools.pairwise(spans):
            assert a.end <= b.start + 1e-6


def test_validate_rejects_overlap():
    with pytest.raises(an.InvariantError):
        an.validate((Span(S1, 0, 1), Span(S2, 0.5, 2)))


def test_bpm_series_skips_intervals_touching_noisy():
    spans = ()
    for t in np.arange(0, 10, 1.0):
        spans = an.place_sound(spans, S1, float(t) + 0.5, 0.1, DUR)
    spans = an.paint_noisy(spans, 4.7, 5.2, DUR)  # covers nothing, sits between beats 4.5 and 5.5
    t, bpm = an.bpm_series(spans)
    assert np.allclose(bpm, 60.0)
    assert 5.0 not in np.round(t, 3)
    assert len(t) == 8


def test_disagreements():
    ann = ()
    for t, k in ((1, S1), (1.3, S2), (2, S1), (2.3, S2), (3, S1)):
        ann = an.place_sound(ann, k, t, 0.1, DUR)
    ann = an.paint_noisy(ann, 5, 6, DUR)
    algo = an.from_states(
        [0.97, 1.97, 2.27, 4.0, 5.5], [1.07, 2.07, 2.37, 4.1, 5.6], ["S1", "S2", "S2", "S1", "S1"]
    )
    d = an.disagreements(ann, algo)
    assert [(x.kind, round(x.start, 2)) for x in d] == [
        ("missed", 1.25), ("swapped", 1.95), ("missed", 2.95), ("extra", 4.0),
    ]


def test_json_round_trip(tmp_path):
    spans = an.paint_noisy(an.place_sound((), S1, 1, 0.1, DUR), 3, 4, DUR)
    ann = an.Annotation("abc", DUR, "rec.wav", spans)
    p = tmp_path / "rec.annotation.json"
    an.save(ann, p)
    assert an.load(p) == ann
    assert b"\r\n" not in p.read_bytes()


def test_find_for_recording_matches_by_fingerprint(tmp_path):
    rec = tmp_path / "renamed [80,70-90bpm].wav"
    an.save(an.Annotation("fp1", DUR, "old name.wav", ()), tmp_path / "old name.annotation.json")
    an.save(an.Annotation("other", DUR, "x.wav", ()), tmp_path / "x.annotation.json")
    assert an.find_for_recording(rec, "fp1") == tmp_path / "old name.annotation.json"
    assert an.find_for_recording(rec, "nope") is None


def _disagreements_reference(ann_spans, algo_spans):
    ann = [s for s in ann_spans if s.kind in an.SOUNDS]
    alg = [s for s in algo_spans if s.kind in an.SOUNDS]
    noisy = [s for s in ann_spans if s.kind == NOISY]
    out = set()
    hit = set()
    for s in ann:
        over = [g for g in alg if g.overlaps(s.start, s.end)]
        hit.update(over)
        if not over:
            out.add(("missed", s.start))
        elif not any(g.kind == s.kind for g in over):
            out.add(("swapped", s.start))
    for g in alg:
        if g not in hit and not any(n.overlaps(g.start, g.end) for n in noisy):
            out.add(("extra", g.start))
    return out


def test_disagreements_match_brute_force():
    rng = random.Random(3)
    for _ in range(50):
        ann = ()
        for _ in range(60):
            t = rng.uniform(0, 30)
            if rng.random() < 0.1:
                ann = an.paint_noisy(ann, t, t + rng.uniform(0.1, 1.5), DUR)
            else:
                ann = an.place_sound(ann, rng.choice([S1, S2]), t, rng.uniform(0.03, 0.2), DUR)
        starts = sorted(rng.uniform(0, 30) for _ in range(60))
        algo = an.from_states(starts, [s + rng.uniform(0.03, 0.2) for s in starts],
                              [rng.choice(["S1", "S2", "systole"]) for _ in starts])
        got = {(d.kind, d.start) for d in an.disagreements(ann, algo)}
        assert got == _disagreements_reference(ann, algo)
