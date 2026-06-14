# debug_helpers

Throwaway-but-keepable tooling for investigating pipeline bugs. Not part of the
shipped pipeline; safe to run ad hoc.

Three independent state-timeline checks live here, each a pure detector + a
pipeline scanner:

| concern | detector | scanner |
|---|---|---|
| spans overlap each other | `overlap_detector.py` | `scan_overlaps.py` |
| labels vs boundary list disagree | `coverage_detector.py` | (use `coverage_detector` ad hoc) |
| boundary sequence breaks the cycle | `state_sequence_detector.py` | `scan_sequence.py` |
| state band disagrees with peak label | `peak_state_mismatch_detector.py` | (use ad hoc / `inspect_region.py`) |

`inspect_region.py` is a cross-strip correlator: for a recording and a time window
(or each sequence violation) it prints, time-ordered, every **peak** (with its
`peak_type`), every **cardiac-state** segment (with anchor metadata), and which
**noise/quiet/gap** windows cover the region — the same three data sources the two
HTML strips render from. Use it to explain *why* a violation happens.

```
python debug_helpers/inspect_region.py "inputs/.../file.wav"            # each violation
python debug_helpers/inspect_region.py "inputs/.../file.wav" --at 5.65  # a window
python debug_helpers/inspect_region.py "inputs/.../file.wav" --from 5.0 --to 6.2
```

## Missing S1 state (state-sequence violation)

A correct timeline walks one fixed cycle:

```
S1 -> systole -> S2 -> diastole -> S1 -> ...
```

Each state has exactly one legal successor. When the boundary list breaks the
cycle the most visible case is **`diastole -> S2`**: an S2 band sits where an S1
cycle belongs, so the strip appears to be **missing the S1 state** at a beat — even
though the dense `pass3_state_labels` and the boundary list agree with each other
(so neither the overlap nor the coverage check sees it). The S2 itself is correctly
placed; what is absent is the S1 (and systole) span that should precede it.

Confirmed on `#49 …RSA…` at 5.65s and 15.98s (`diastole -> S2`), matching the
reported playhead. A clean recording (`Control`) shows the perfect 4-cycle with
zero violations.

### Files

- `state_sequence_detector.py` — pure. `find_sequence_violations(boundaries, sample_rate=...)`
  collapses the boundary list to real-state runs and flags every illegal transition
  between **abutting** runs (a gap / `unknown` between runs legitimately breaks the
  cycle and is not flagged). `summarize()` rolls up by kind / transition.
- `scan_sequence.py` — runs the pipeline (parsing the starting BPM from each file
  name, matching the GUI's `bpm_from_filename` default) and reports violations.
  Exit code `1` if any.

### Usage (from repo root)

```
python debug_helpers/scan_sequence.py                       # scan inputs/**/*.wav
python debug_helpers/scan_sequence.py "inputs/Difficulty 3" # one subtree
python debug_helpers/scan_sequence.py inputs --json debug_helpers/sequence_report.json
```

Both `scan_sequence.py` and `scan_overlaps.py` run the pipeline across files in a
**process pool** (`analyze_wav_file` is CPU-bound, so processes — not threads —
give real speedup). Defaults to `CPU count - 1` workers; override with
`--jobs N` (`-j N`), or `--jobs 1` for serial debugging.

### Root-cause notes (confirmed mechanism — not yet fixed)

Correlating the three strips with `inspect_region.py` + `peak_state_mismatch_detector.py`
shows the `diastole -> S2` is **not** a missing S1 nor a noise peak landing in S2.
It is a **regional S1<->S2 label swap**:

- On `#49 …RSA…` the Noise peaks (e.g. 5.13 / 5.26 / 5.44s) correctly fall inside
  the long `diastole`; the noise strip marks that span `quiet`. The S2 band at 5.65s
  does **not** sit on a noise peak — it sits on peak 3409, classified **S1 (Paired)**.
- The peak/state detector finds 30 mismatches in that file: S1 peaks under **S2**
  bands and S2 peaks under **S1** bands, alternating, in sustained runs (e.g. 16–22s).
  Geometry is right (band centred on its peak); only the **name** is swapped. At an
  S1 peak the band is named S2, so the strip looks like it is missing the S1 state
  and the sequence reads `diastole -> S2`.
- The swap is **regional**, not whole-recording, so it is *not* the global
  inversion fix in `correction._…global_phase…` (that relabels everything
  uniformly). The local mechanism that flips S1/S2 for these cycles is the next
  thing to pin down.

The fix is therefore about correct S1/S2 *labelling* for these cycles, not about
adding a missing span.

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
