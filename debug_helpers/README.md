# debug_helpers

Throwaway-but-keepable tooling for investigating pipeline bugs. Not part of the
shipped pipeline; safe to run ad hoc.

## Overlapping cardiac states

The Pass 3 state timeline (`analysis_data["pass3_state_boundaries"]`) is supposed
to be a **dense, non-overlapping** partition of time into S1 / systole / S2 /
diastole spans. A bug lets gap-fill paths emit spans that overlap — two cardiac
meanings claiming the same samples.

### Files

- `overlap_detector.py` — pure detector. `find_overlapping_states(boundaries, sample_rate=...)`
  returns one record per overlapping span pair; `summarize(records)` rolls them up.
  No pipeline import, so it is unit-testable and reusable.
- `scan_overlaps.py` — runs the real pipeline (`analyze_wav_file`, no artifacts
  written) on input WAVs and reports overlaps.

### Usage (from repo root)

```
python debug_helpers/scan_overlaps.py                       # scan inputs/**/*.wav
python debug_helpers/scan_overlaps.py "inputs/Difficulty 5" # one subtree
python debug_helpers/scan_overlaps.py path/to/one.wav       # single file
python debug_helpers/scan_overlaps.py inputs --json debug_helpers/overlap_report.json
```

Exit code is `1` when any file has overlaps, else `0` (handy in CI / a guard test).

### How an overlap is classified

Each record has `kind`:

- **`gap_rebuild`** — at least one of the two overlapping spans carries a
  `rebuild_source` of `gap_insert`, `gap_label_pass3`, or `noise_repair`, i.e. it
  was *painted into a gap*. **This is the targeted bug** ("overlapping cardiac
  states during gap regions").
- **`edge_paint`** — both spans are real detected segments whose painted edges
  (the `s1_half` / `s2_half` edge expansion in `_paint_state_boundaries`) bleed
  into each other by a sample or two. Separate, smaller effect.

### Root-cause notes (from observed records)

The gap-fill paths in `correction.py` keep two representations in sync by hand:
the dense `state_labels` array and the `state_boundaries` list.

- `_pass3_rebuild_*` (the cursor forward-paint, ~line 2188) fills each
  `STATE_UNKNOWN` run from `state_labels`, clips new segments to `[gap_lo, gap_hi)`,
  then **concatenates** `state_boundaries + new_segs` without trimming the old
  list (`combined = state_boundaries + new_segs`, ~line 2410).
- The labeling path (`_pass3_apply_peaks_labeling_in_large_gaps`) *does* trim,
  via `_pass3_remove_boundaries_overlapping_span(bd, lo, hi)` — so it is the
  concat-without-trim paths (`noise_repair`, `gap_insert`) that leak overlaps.
- The overlap shows up where a kept *real* neighbour segment's painted span
  (e.g. an S1 whose `s1_start` was expanded backward by edge painting) reaches
  into the just-filled gap window. The gap was derived from `state_labels`
  run-length, which does not see that backward-expanded boundary, so `gap_hi`
  sits *past* where the real S1 boundary already begins → overlap of ~one
  edge-half (~20 samples at the working sample rate).

So: detect by scanning final `pass3_state_boundaries` for strict span overlap;
fix (not done here) would trim boundaries overlapping each gap before the concat,
mirroring what the labeling path already does.
