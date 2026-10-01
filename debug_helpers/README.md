# debug_helpers

Throwaway-but-keepable tooling for investigating pipeline bugs. Not part of the
shipped pipeline; safe to run ad hoc.

The four state-timeline checks are **pure detectors** that now live in the engine
(`pcg/engine/defects/`) — every Analysis runs them and stores the results as
**Defects**, which the workspace lists and jumps between. The scanners here run
them across many recordings:

| concern | detector (`pcg/engine/defects/`) | scanner |
|---|---|---|
| spans overlap each other | `overlaps.py` | `scan_overlaps.py` |
| labels vs boundary list disagree | `coverage.py` | (use `coverage` ad hoc) |
| boundary sequence breaks the cycle | `sequence.py` | `scan_sequence.py` |
| state band disagrees with peak label | `peak_state.py` | `scan_peak_state.py` |

Single-file **audit** tools (one WAV in, explanatory dump out — no detector):

| tool | answers |
|---|---|
| `inspect_region.py` | what peaks/states/noise-windows sit in a time window (or each violation) |
| `compare_to_annotation.py` | how well Pass-3 output matches the recording's Annotation (per-sample agreement, per-state recall, beat count) |
| `gap_decision_audit.py` | why each wide diastole became QUIET vs a phantom-insert GAP |
| `phantom_insert_detector.py` | which rebuilt cycles sit over a flat envelope with no real beat |

## Shared plumbing — `_common.py`

Every scanner/audit tool runs the engine (`pcg.engine.run_analysis`) and feeds
the result to a detector. That boilerplate — `params()`, the `run_pipeline()`
wrapper, `env_sample_rate()`, `bpm_hint_from_name()`, `collect_wavs()`,
`reconfigure_stdio()`, and the CPU-bound `parallel_scan()` process pool — lives in
[`_common.py`](./_common.py), imported by all of them. The detectors deliberately do
**not** import it, so they stay engine-run-free and unit-testable.

For a single recording, the workspace (`python -m pcg`) shows the same information
interactively, and `python -m pcg inspect <rec> --from S --to S` dumps it as text.

`inspect_region.py` is a cross-strip correlator: for a recording and a time window
(or each sequence violation) it prints, time-ordered, every **peak** (with its
`peak_type`), every **cardiac-state** segment (with anchor metadata), and which
**noise/quiet/gap** windows cover the region — the same data the workspace's states
and signal lanes draw. Use it to explain *why* a violation happens.

```
python debug_helpers/inspect_region.py "inputs/.../file.wav"            # each violation
python debug_helpers/inspect_region.py "inputs/.../file.wav" --at 5.65  # a window
python debug_helpers/inspect_region.py "inputs/.../file.wav" --from 5.0 --to 6.2
```

## Invariant gate (the point of all this)

These detectors exist because the Pass 3 algorithm never enforces the invariants
they check — so a change can trade accuracy for structural breakage invisibly
(exactly how the reverted overlap fix newly broke 16 files into S1/S2 swaps).

[`benchmarking/state_invariants.py`](../benchmarking/state_invariants.py) runs the
pipeline once per file and applies **all four** detectors, aggregates the totals,
and compares them to a committed baseline — failing if any metric regresses. It
needs no ground truth (unlike `run_benchmark.py`, which scores against Annotations). Run it before/after any
change to Pass 3:

```
python benchmarking/state_invariants.py                 # compare to baseline -> exit 1 on regression
python benchmarking/state_invariants.py --write-baseline  # re-bless after an intended change
python benchmarking/state_invariants.py "inputs/Difficulty 3" -j 8
```

Metrics: `overlaps_gap_rebuild`, `overlaps_edge_paint`, `coverage_desync_runs`,
`seq_missing_s1`, `seq_bad_transition`, `swap_mismatches`. Baseline lives at
`benchmarking/state_invariants_baseline.json`.

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

- `pcg/engine/defects/sequence.py` — pure. `find_sequence_violations(boundaries, sample_rate=...)`
  collapses the boundary list to real-state runs and flags every illegal transition
  between **abutting** runs (a gap / `unknown` between runs legitimately breaks the
  cycle and is not flagged). `summarize()` rolls up by kind / transition.
- `scan_sequence.py` — runs the pipeline (parsing the starting BPM from each file
  name, as the run screen and CLI do by default) and reports violations.
  Exit code `1` if any.

### Usage (from repo root)

```
python debug_helpers/scan_sequence.py                       # scan inputs/**/*.wav
python debug_helpers/scan_sequence.py "inputs/Difficulty 3" # one subtree
python debug_helpers/scan_sequence.py inputs --json debug_helpers/sequence_report.json
```

Both `scan_sequence.py` and `scan_overlaps.py` run the pipeline across files in a
**process pool** (the engine is CPU-bound, so processes — not threads —
give real speedup). Defaults to `CPU count - 1` workers; override with
`--jobs N` (`-j N`), or `--jobs 1` for serial debugging.

### Root-cause notes (confirmed mechanism — not yet fixed)

Correlating the three strips with `inspect_region.py` + `pcg/engine/defects/peak_state.py`
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

- `pcg/engine/defects/overlaps.py` — pure detector. `find_overlapping_states(boundaries, sample_rate=...)`
  returns one record per overlapping span pair; `summarize(records)` rolls them up.
  No pipeline import, so it is unit-testable and reusable.
- `scan_overlaps.py` — runs the engine on input WAVs and reports overlaps.

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
