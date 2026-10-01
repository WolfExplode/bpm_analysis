"""Annotations: the human-verified ground truth for one Recording.

An Annotation stores only heart sounds (S1 / S2 spans) and Noisy spans, never
overlapping. Systole and diastole are the gaps between sounds; an S1 followed by
another S1 is a cycle whose S2 was inaudible. Once saved it is trusted in full
and frozen — re-analysis never changes it.

Every edit is a pure function `spans -> spans` that keeps the invariant (sorted,
non-overlapping, positive length), so undo/redo is just a stack of span tuples.

File: `<recording stem>.annotation.json` next to the recording, matched to its
Recording by fingerprint, not by name.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

S1 = "S1"
S2 = "S2"
NOISY = "noisy"
SOUNDS = (S1, S2)
KINDS = (S1, S2, NOISY)

ORIGIN_ALGORITHM = "algorithm"
ORIGIN_HAND = "hand"

FORMAT_VERSION = 1
SUFFIX = ".annotation.json"
_EPS = 1e-6
_MIN_SPAN = 1e-3


@dataclass(frozen=True)
class Span:
    kind: str
    start: float
    end: float
    origin: str = ORIGIN_HAND
    clipped: bool = False  # cut at a midpoint when seeded from overlapping algorithm states

    @property
    def center(self) -> float:
        return 0.5 * (self.start + self.end)

    def overlaps(self, a: float, b: float) -> bool:
        return self.start < b - _EPS and a < self.end - _EPS


Spans = Tuple[Span, ...]


@dataclass(frozen=True)
class Annotation:
    fingerprint: str
    duration_sec: float
    filename_hint: str
    spans: Spans = ()

    def with_spans(self, spans: Spans) -> "Annotation":
        return replace(self, spans=spans)


class InvariantError(ValueError):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Invariant
# ─────────────────────────────────────────────────────────────────────────────

def validate(spans: Sequence[Span]) -> None:
    prev: Optional[Span] = None
    for s in spans:
        if s.kind not in KINDS:
            raise InvariantError(f"unknown span kind {s.kind!r}")
        if not s.end - s.start > 0:
            raise InvariantError(f"empty span {s}")
        if prev is not None and s.start < prev.end - _EPS:
            raise InvariantError(f"overlap: {prev} / {s}")
        prev = s


def _normalize(spans: Iterable[Span]) -> Spans:
    """Sort, drop slivers, merge touching/overlapping Noisy spans, and check the invariant."""
    out: List[Span] = []
    for s in sorted(spans, key=lambda s: (s.start, s.end)):
        if s.end - s.start < _MIN_SPAN:
            continue
        if out and s.kind == NOISY and out[-1].kind == NOISY and s.start <= out[-1].end + _EPS:
            last = out[-1]
            out[-1] = replace(last, end=max(last.end, s.end), origin=ORIGIN_HAND)
            continue
        out.append(s)
    result = tuple(out)
    validate(result)
    return result


def _cut_noisy(span: Span, a: float, b: float) -> List[Span]:
    """What remains of a Noisy span after removing [a, b]."""
    parts = []
    if span.start < a - _EPS:
        parts.append(replace(span, end=min(span.end, a)))
    if span.end > b + _EPS:
        parts.append(replace(span, start=max(span.start, b)))
    return parts


# ─────────────────────────────────────────────────────────────────────────────
# Edits
# ─────────────────────────────────────────────────────────────────────────────

def place_sound(spans: Spans, kind: str, center: float, length: float, duration: float,
                origin: str = ORIGIN_HAND) -> Spans:
    """Place an S1/S2 of `length` centred at `center`; it replaces whatever it overlaps."""
    if kind not in SOUNDS:
        raise ValueError(kind)
    a = max(0.0, center - length / 2.0)
    b = min(duration, center + length / 2.0)
    if b - a < _MIN_SPAN:
        return spans
    kept: List[Span] = []
    for s in spans:
        if not s.overlaps(a, b):
            kept.append(s)
        elif s.kind == NOISY:
            kept.extend(_cut_noisy(s, a, b))
    kept.append(Span(kind, a, b, origin))
    return _normalize(kept)


def paint_noisy(spans: Spans, a: float, b: float, duration: float) -> Spans:
    """Mark [a, b] Noisy; sounds it touches are removed, Noisy spans it touches merge."""
    a, b = max(0.0, min(a, b)), min(duration, max(a, b))
    if b - a < _MIN_SPAN:
        return spans
    kept = [s for s in spans if not (s.kind in SOUNDS and s.overlaps(a, b))]
    kept.append(Span(NOISY, a, b, ORIGIN_HAND))
    return _normalize(kept)


def resize_noisy(spans: Spans, span: Span, start: float, end: float, duration: float) -> Spans:
    """Move a Noisy span's edges (S1/S2 spans are not resizable)."""
    if span.kind != NOISY or span not in spans:
        return spans
    rest = tuple(s for s in spans if s != span)
    return paint_noisy(rest, start, end, duration)


def delete(spans: Spans, span: Span) -> Spans:
    return tuple(s for s in spans if s != span)


def span_at(spans: Spans, t: float) -> Optional[Span]:
    """The span covering t (sounds before Noisy when both touch t)."""
    hits = [s for s in spans if s.start - _EPS <= t <= s.end + _EPS]
    hits.sort(key=lambda s: (s.kind == NOISY, abs(s.center - t)))
    return hits[0] if hits else None


def relabel(spans: Spans, span: Span) -> Spans:
    """S1 <-> S2."""
    if span.kind not in SOUNDS or span not in spans:
        return spans
    flipped = replace(span, kind=S2 if span.kind == S1 else S1, origin=ORIGIN_HAND)
    return tuple(flipped if s == span else s for s in spans)


def flip_after(spans: Spans, t: float) -> Spans:
    """Swap S1 <-> S2 for every sound starting at or after t."""
    return tuple(
        replace(s, kind=S2 if s.kind == S1 else S1, origin=ORIGIN_HAND)
        if s.kind in SOUNDS and s.start >= t - _EPS else s
        for s in spans
    )


def from_states(starts: Sequence[float], ends: Sequence[float], states: Sequence[str],
                origin: str = ORIGIN_ALGORITHM) -> Spans:
    """Sound spans from an algorithm state timeline (systole/diastole/unknown dropped)."""
    return from_spans([(a, b, s, origin) for a, b, s in zip(starts, ends, states) if s in SOUNDS])


def from_spans(rows: Iterable[Tuple[float, float, str, str]], clips: Optional[List[dict]] = None) -> Spans:
    """Valid spans from possibly overlapping (start, end, kind, origin) rows.

    Overlapping spans of different kinds are cut at the midpoint of their overlap and
    marked `clipped`; touching/overlapping spans of the same kind merge. Each cut is
    appended to `clips` (if given) so it can be reported for spot-checking.
    """
    raw = sorted((float(a), float(b), k, o) for a, b, k, o in rows if k in KINDS and float(b) > float(a))
    out: List[Span] = []
    for a, b, kind, origin in raw:
        clipped = False
        merged = False
        while out:
            last = out[-1]
            if kind == last.kind and a <= last.end + _EPS:
                out[-1] = replace(last, end=max(last.end, b), clipped=last.clipped or clipped)
                merged = True
                break
            if a >= last.end - _EPS:
                break
            # Overlaps a different kind: cut both at the midpoint of the overlap.
            lo, hi = max(a, last.start), min(last.end, b)
            mid = 0.5 * (lo + hi)
            clipped = True
            if clips is not None:
                clips.append({"at": lo, "overlap_sec": hi - lo, "first": last.kind, "second": kind,
                              "first_span": (last.start, last.end), "second_span": (a, b)})
            if mid - last.start < _MIN_SPAN:
                out.pop()  # swallowed; keep resolving against the span before it
                continue
            out[-1] = replace(last, end=mid, clipped=True)
            a = max(a, mid)
            break
        if not merged and b - a >= _MIN_SPAN:
            out.append(Span(kind, a, b, origin, clipped))
    return _normalize(out)


def replace_region(spans: Spans, a: float, b: float, algorithm: Spans) -> Spans:
    """Overwrite [a, b] with the algorithm's sounds. Spans crossing the edges: sounds go,
    Noisy spans are trimmed; algorithm sounds crossing the edges are left out."""
    a, b = min(a, b), max(a, b)
    kept: List[Span] = []
    for s in spans:
        if not s.overlaps(a, b):
            kept.append(s)
        elif s.kind == NOISY:
            kept.extend(_cut_noisy(s, a, b))
    kept.extend(s for s in algorithm if s.kind in SOUNDS and s.start >= a - _EPS and s.end <= b + _EPS)
    return _normalize(kept)


def derived_states(spans: Spans) -> List[Tuple[float, float, str]]:
    """The full state sequence: sounds and Noisy spans plus the gaps between sounds —
    S1 -> S2 is systole, S2 -> S1 diastole, S1 -> S1 a cycle with no audible S2 ("cycle").
    Gaps next to a Noisy span (or another kind of transition) are left unlabelled."""
    gap_name = {(S1, S2): "systole", (S2, S1): "diastole", (S1, S1): "cycle"}
    out: List[Tuple[float, float, str]] = []
    for i, s in enumerate(spans):
        out.append((s.start, s.end, s.kind))
        if i + 1 < len(spans):
            nxt = spans[i + 1]
            name = gap_name.get((s.kind, nxt.kind))
            if name and nxt.start > s.end + _EPS:
                out.append((s.end, nxt.start, name))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Derived: BPM and Disagreements
# ─────────────────────────────────────────────────────────────────────────────

def bpm_series(spans: Spans, min_bpm: float = 25.0, max_bpm: float = 260.0) -> Tuple[np.ndarray, np.ndarray]:
    """Instantaneous BPM from consecutive S1 centres (time = interval midpoint).
    Intervals that touch a Noisy span are excluded."""
    s1 = np.array([s.center for s in spans if s.kind == S1])
    if len(s1) < 2:
        return np.array([]), np.array([])
    noisy = [(s.start, s.end) for s in spans if s.kind == NOISY]
    a, b = s1[:-1], s1[1:]
    keep = np.ones(len(a), dtype=bool)
    if noisy:
        ns = np.array(noisy)
        # interval [a, b] touches noisy [c, d] iff c < b and a < d
        first_end_after = np.searchsorted(ns[:, 1], a, side="right")
        has = first_end_after < len(ns)
        keep &= ~(has & (ns[np.minimum(first_end_after, len(ns) - 1), 0] < b))
    d = b - a
    bpm = 60.0 / np.where(d > 0, d, np.nan)
    keep &= np.isfinite(bpm) & (bpm >= min_bpm) & (bpm <= max_bpm)
    return ((a + b) / 2.0)[keep], bpm[keep]


@dataclass(frozen=True)
class Disagreement:
    kind: str  # "missed" | "swapped" | "extra"
    start: float
    end: float
    message: str


DISAGREEMENT_KINDS = ("missed", "swapped", "extra")


def _overlap_ranges(starts: np.ndarray, ends: np.ndarray, a: np.ndarray, b: np.ndarray):
    """For sorted non-overlapping spans (starts, ends): index range [lo, hi) overlapping each [a, b]."""
    lo = np.searchsorted(ends, a + _EPS, side="right")
    hi = np.searchsorted(starts, b - _EPS, side="left")
    return lo, np.maximum(hi, lo)


def disagreements(annotation: Spans, algorithm: Spans) -> List[Disagreement]:
    """Where the Analysis's sounds differ from the Annotation (Noisy spans excluded). Vectorised:
    it runs on every edit, over thousands of spans."""
    ann = [s for s in annotation if s.kind in SOUNDS]
    alg = [s for s in algorithm if s.kind in SOUNDS]
    noisy = [s for s in annotation if s.kind == NOISY]
    a_start = np.array([s.start for s in ann], dtype=np.float64)
    a_end = np.array([s.end for s in ann], dtype=np.float64)
    g_start = np.array([s.start for s in alg], dtype=np.float64)
    g_end = np.array([s.end for s in alg], dtype=np.float64)
    a_s1 = np.array([s.kind == S1 for s in ann], dtype=bool)
    g_s1 = np.array([s.kind == S1 for s in alg], dtype=bool)

    lo, hi = _overlap_ranges(g_start, g_end, a_start, a_end)
    n_over = hi - lo
    s1_cum = np.concatenate([[0], np.cumsum(g_s1)])
    n_s1 = s1_cum[hi] - s1_cum[lo]
    n_same = np.where(a_s1, n_s1, n_over - n_s1)

    out: List[Disagreement] = []
    for i in np.nonzero(n_over == 0)[0]:
        s = ann[i]
        out.append(Disagreement("missed", s.start, s.end, f"algorithm has no sound at the annotated {s.kind}"))
    for i in np.nonzero((n_over > 0) & (n_same == 0))[0]:
        s = ann[i]
        out.append(Disagreement("swapped", s.start, s.end, f"annotated {s.kind}, algorithm says {alg[lo[i]].kind}"))

    # Algorithm sounds touched by no annotated sound and outside every Noisy span.
    cover = np.zeros(len(alg) + 1, dtype=np.int64)
    np.add.at(cover, lo[n_over > 0], 1)
    np.add.at(cover, hi[n_over > 0], -1)
    matched = np.cumsum(cover[:-1]) > 0
    if noisy:
        n_lo, n_hi = _overlap_ranges(np.array([s.start for s in noisy]), np.array([s.end for s in noisy]),
                                     g_start, g_end)
        matched |= n_hi > n_lo
    for i in np.nonzero(~matched)[0]:
        s = alg[i]
        out.append(Disagreement("extra", s.start, s.end, f"algorithm {s.kind} where the annotation has none"))
    out.sort(key=lambda d: d.start)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Files
# ─────────────────────────────────────────────────────────────────────────────

_FIELDS = ["kind", "start", "end", "origin", "clipped"]


def to_json(ann: Annotation) -> str:
    validate(ann.spans)
    doc = {
        "format": FORMAT_VERSION,
        "fingerprint": ann.fingerprint,
        "duration_sec": round(ann.duration_sec, 6),
        "filename_hint": ann.filename_hint,
        "span_fields": _FIELDS,
        "spans": [[s.kind, round(s.start, 5), round(s.end, 5), s.origin, s.clipped] for s in ann.spans],
    }
    return json.dumps(doc, ensure_ascii=False, indent=1)


def from_json(text: str) -> Annotation:
    doc = json.loads(text)
    if doc.get("format") != FORMAT_VERSION:
        raise ValueError(f"unsupported Annotation format {doc.get('format')}")
    fields = doc.get("span_fields", _FIELDS)
    spans = tuple(Span(**dict(zip(fields, row))) for row in doc["spans"])
    validate(spans)
    return Annotation(doc["fingerprint"], float(doc["duration_sec"]), doc.get("filename_hint", ""), spans)


def save(ann: Annotation, path: os.PathLike | str) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(to_json(ann), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def load(path: os.PathLike | str) -> Annotation:
    return from_json(Path(path).read_text(encoding="utf-8"))


def sidecar_path(recording_path: os.PathLike | str) -> Path:
    p = Path(recording_path)
    return p.with_name(p.stem + SUFFIX)


def find_for_recording(recording_path: os.PathLike | str, fingerprint: str) -> Optional[Path]:
    """An Annotation in the recording's folder whose fingerprint matches (any file name)."""
    preferred = sidecar_path(recording_path)
    candidates = [preferred] + sorted(p for p in Path(recording_path).parent.glob(f"*{SUFFIX}") if p != preferred)
    for p in candidates:
        try:
            if p.is_file() and json.loads(p.read_text(encoding="utf-8")).get("fingerprint") == fingerprint:
                return p
        except (OSError, ValueError):
            continue
    return None
