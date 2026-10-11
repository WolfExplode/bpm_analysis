# Workspace requirements

The Signal lane includes an **Original audio envelope**, off by default and
independently toggleable in its legend with a `display` tag. It is a smoothed
Hilbert magnitude of the decoded, unfiltered original playback channel, sampled
at roughly 200 Hz for display. Background computation uses bounded chunks and
the Analysis's envelope smoothing window. It starts at recording time zero,
including any start the algorithm skips. It applies no peak normalization,
bandpass, noise subtraction or playback gain. This workspace-only trace is not
saved in Analysis and does not change detection, Annotation, or the overview's
algorithm envelope. Existing Analyses need no re-run to display it.

### Recording metadata and simplified recovery estimates

The **Recording** button beside **Summary** opens age at recording, reported sex,
**Resting heart rate**, and a recovery-marker selector. The app uses only the
simplified stop-and-sit model; there is no protocol selector, cohort confirmation,
peak-HR confirmation, pediatric age band, BMI or supine-resting-HR control.
Age and reported sex adjust the approximation at supported checkpoints when age
is supplied. Without age, the original basic estimate remains available.
An optional **Fitness level (approximate)** input autosaves `fitness_level` as
`unknown` (default for existing recordings), `low`, `medium` or `high`.
Unknown retains the current model. For ages 18–79, selected fitness uses
`H=R+(S-R)*[0.25+0.75*exp(-t/tau)]` through 300 s, replacing both adult age
and sex checkpoint corrections; sexes are pooled in the source.
Low means sedentary and High means habitually active; these are activity proxies,
not VO₂max thresholds or an elite-athlete category. Medium is an assumed midpoint.
Muro (2003) sedentary/active tau anchors are 57/37, 85/46 and 108/56 s for
19–39, 40–59 and 60–79 years. The app assigns them to band centers 29, 49.5
and 69.5, interpolates linearly between centers, and holds the nearest endpoint
constant within the supported age range. Both interpolation and the 25% residual
are declared modelling assumptions. Missing age shows an input explanation;
Age 18 uses the youngest adult anchor as an explicit one-year extrapolation
below the seated study's age range. Adult ages outside 18–79 with selected
fitness leave estimates blank. Pediatric
calculations and age gaps ignore the fitness selection. No fitness corrections
are applied on top of pediatric or adult demographic checkpoint estimates.

Metadata autosaves to `<fingerprint>.recording.json` in the Analysis library,
independently of Analysis and Annotation, and survives re-analysis and audio renames.
Copy this sidecar when transferring metadata to another library. Marker positions
continue to use existing per-user settings. Existing metadata remains readable;
legacy study selections have no effect on calculations. The resting input retains
its `seated_resting_hr` key and tooltip describing pre-exercise sitting; legacy
supine `resting_hr` values are not silently repurposed.

The estimate calculates automatically from each marker's BPM and supplied resting
heart rate: `H(t)=R+(S−R)×[0.25+0.75×exp(−t/50)]`, restricted to 0–300 s.
For ages greater than 16 and less than 70, cumulative drops at 30/60/120/180 s
receive `a*(age-20)+b*(female-1)`, anchored to a 20-year-old woman. The respective
`(a,b)` coefficients are `(-0.18,2.15)`, `(-0.18,3.98)`, `(-0.13,6.04)` and
`(-0.12,6.74)`. These are transferred checkpoint associations from
Hughes/Chaturvedi (2017), supplementary Tables S1–S4
(https://pmc.ncbi.nlm.nih.gov/articles/PMC5439512/), not a validated personal curve.
There is no adjusted 300-s estimate or interpolation between checkpoints.
Pediatric estimates instead use `H=S-f*(S-R)` with measured `S>R`. Ages 8–10
use `f60=0.64` for both sexes from Castner et al. (2013), normal-weight cohort
(https://pmc.ncbi.nlm.nih.gov/articles/PMC3674719/). Selecting the inclusive
8–10 range around the cohort's mean age 9.4 is an explicit app approximation,
not a published age-bin boundary or an inferred weight classification.
Ages 12–14 use `(f60,f120)=(0.55,0.68)` for male and `(0.43,0.57)` for female,
from Abu Hanifah et al. (2013), Table 1 (doi:10.1371/journal.pone.0082893).
These are ratios of cohort means scaled to a new starting reserve, not validated
personal predictions. The adolescent source uses peak pulse; using marker BPM
assumes cessation pulse is close to peak. There are no pediatric estimates at
30/180/300 s, no adult corrections applied to pediatric estimates, and no
interpolation across ages. Other ages up to and including 16 leave estimates
blank, while observed measurements remain available.
Out-of-range ages show an input explanation; adjusted drops below zero or above
starting-minus-resting HR are omitted rather than clamped. Changing age or sex
recalculates immediately, and saved selections are used on reload.
The table compares observed and estimated cumulative drops
and lists estimated endpoint BPM. Missing values have blank cells; checkpoints with neither an observation
nor an estimate are omitted entirely, including their time labels.
Starting BPM follows the visible Annotation/Result BPM source used by recovery lines.
Observed endpoint BPM uses time-weighted ±2-s means; missing coverage, sustained
rise or another marker can stop observed measurements. Missing or invalid model
inputs show an explanation rather than assumed values.

This proposed heuristic assumes the marker is exertion cessation followed by
immediate passive sitting; later markers explicitly assume a fresh recovery start.
It supplies no clinical classifications, percentiles, prediction interval
or expected full return-to-rest time. The rounded 50-s decay constant
and 25% remaining excess follow the explicit approximation in
`HeartbeatSimulator/docs/research docs/heart-rate-recovery.md` (2026-10-11), informed
by Barak et al. (2011), Table 2, passive seated submaximal-cycle recovery in young men.
The runtime does not depend on that external checkout.
When no recovery marker exists, the marker control is replaced by a **Starting
heart rate** field and a **Predicted recovery** table of estimated BPM/drop.
The entered value autosaves as `person.preview_start_bpm`; no starting BPM is
invented. The preview uses the same model and missing-checkpoint rules, without
observed values or creating a marker. Adding a marker restores the comparison;
deleting the last marker restores the saved preview. Metadata edits recalculate
both views immediately.

### Playback gain audition

The top-right gain controls contain **+** and numbered point buttons. Click **+**,
then click the Signal graph to place a time-local handle. Click its graph circle
to select it, then drag
up/down to boost or attenuate, or left/right to move it. The gain line's 0 dB
baseline is anchored to Signal y=0. Colored numbered rings and shaded bell curves
show each band, with time/gain/Q readouts beside the selected center. Dragging
sets -60 dB (Mute) to +12 dB. New bells have a 0.2 s half-change bandwidth at Q=1.
Q divides this bandwidth: scrolling up over a band center increases Q and
narrows the bell; scrolling down widens it, without zooming the graph.
Delete or Backspace deletes the selected gain point. Clicking a numbered button
toggles only that point's effect on/off; disabled points remain visible in gray.
Shift-drag selects and loops a region. Left-drag within six pixels of either
selection border to resize it on any lane; a horizontal resize cursor marks the
grab zone. The lanes stay synchronized, and the playback loop updates on release.
Borders stay within the Recording and cannot cross one another. Overlapping
edits multiply; boosted playback is clipped to the output range.

Point toggles work with either original or filtered playback. Live edits and
bypass changes crossfade over 5 ms. The curve is a playback gain overlay,
with its own dB scale, independent of the underlying Signal amplitude scale.
These session edits reset when another Recording is opened; they are not saved
or exported. Processing happens in the audio output callback, without encoding,
modifying source samples, changing Analysis, or changing Annotation. Gain affects
all sounds at a given time; it cannot separate overlapping sounds.

Requirements for the native app that replaces the tkinter GUI and the Plotly HTML
reports. Terms in **bold** are defined in [CONTEXT.md](../CONTEXT.md); the big
decisions are recorded in [ADR 0004](adr/0004-recording-identity-by-audio-fingerprint.md)
and [ADR 0005](adr/0005-native-python-workspace-replaces-html-reports.md).

**Status (2026-10-01): implemented** on branch `new-UI` (`pcg/`). Deviations: trace kind
"band" is called `spans`; `fft_profiles.py` was deleted whole (nothing used its non-HTML
parts once the HTML and cross-file aggregation went); `BPM_Analyzer.spec` (PyInstaller) was
deleted — a frozen exe can't re-run fresh engine code, which is the point of Ctrl+R.
The spectrogram lane is computed from the playback audio (0–1 kHz) when first shown.

The tool is first an instrument for debugging the algorithm, second a tool for
producing **Annotations**. No backwards compatibility with the old GUI, HTML
reports or file formats is required (existing annotation CSVs are converted once).

## 1. Scope

- **In:** workspace screen, run screen, CLI, **Analysis** and **Annotation** files,
  **Recording** fingerprinting, porting `benchmarking/` and `debug_helpers/` to the new formats.
- **Engine changes allowed:** instrumentation only — emit named debug traces, per-peak
  reasoning and **Defects** into the Analysis; move the systole/diastole computations
  out of `plotting.py`. Must be behavior-preserving, verified by an old-vs-new comparison
  of peaks, states and metrics.
- **Out:** any change to what the algorithm decides (overlap fix, S1/S2 phase rework).
  Those follow, using the new workspace.
- **Deleted:** `gui.py`, `plotting.py`, `reporting.py`, HTML parts of `fft_profiles.py`,
  `assets/`, `pipeline.py`, debug-WAV plumbing, cross-file FFT aggregation, the
  Markdown summary/debug logs; dependencies ttkbootstrap, plotly, kaleido.
- **Stack:** Python, Qt (PySide6) + PyQtGraph.

## 2. Code layout

```
pcg/
  engine/        current engine modules, moved as-is; + traces.py, defects.py
  recording.py   fingerprint, audio loading, channel handling, filtered playback audio
  analysis.py    Analysis model, file format, library
  annotation.py  Annotation model, invariants, JSON I/O
  batch.py       run queue, BPM filename rename, start BPM from filename
  cli.py         analyze / inspect / export
  app/           Qt workspace + run screen
benchmarking/, debug_helpers/, springer2015/   imports updated
```

`config.py` (engine parameters) stays the only place parameters are tuned. The app has
no parameter editor.

## 3. Recording identity

- A Recording is identified by a hash of its **decoded audio samples** — not filename,
  not file bytes, and nothing is ever written into the audio file.
- Renaming (including BPM renaming) never breaks the link; re-encoding/resampling/trimming
  makes a different Recording.

## 4. Analysis

- One file per Analysis in an app-managed **library** folder, keyed by fingerprint; never
  written next to the recording.
- Format: zip of a JSON manifest (fingerprint, parameters, engine code fingerprint,
  algorithm used, trace catalog, per-peak reasoning, Defects) + numpy arrays for large series.
- Contains: envelopes, noise floor, troughs, peaks with per-pass labels/scores/reasoning,
  states, **BPM/time belief**, instantaneous BPM, HRV and interval metrics, Pass 1 and
  Pass 2 data, all debug traces, Defects.
- Debug traces are **self-describing**: name, group (pass), unit, suggested lane, kind
  (line / points / band), and a semantic role / estimate flag (never a colour: the app's
  theme, `pcg/app/theme.py`, owns every colour). The viewer has no per-trace code.
- **Stale** when parameters or engine code fingerprint differ from current: shown as a
  badge, re-run only on request. Never re-run silently.
- Only one Analysis per Recording is shown at a time (side-by-side comparison: not now).

## 5. Annotation

- Stores only **S1/S2 spans** and **Noisy spans**, never overlapping (enforced on every
  edit). Systole/diastole are derived gaps; S1→S1 is a cycle with no audible S2.
- Each span records its origin (algorithm / hand-placed) for display only.
- JSON file `<stem>.annotation.json` next to the recording, containing the Recording
  fingerprint, duration and original filename hint. Matched to its Recording by
  fingerprint (same folder, any name); other locations via "Open Annotation…".
- **Saved = complete and trusted in full**, frozen; re-analysis never changes it.
- **Protected folders** (configurable, default `inputs/`): saving an Annotation for a
  Recording inside one opens a Save As dialog starting in Downloads.
- **Starting:** a Recording with no Annotation opens a draft seeded from the Analysis's
  S1/S2 states (overlaps clipped at the midpoint and marked). The draft lives in memory
  until the first Ctrl+S; unsaved changes are indicated in the title.

## 6. Workspace screen

Stacked lanes on one shared, linked time axis; each lane has its own y-axis.

1. **Overview strip:** whole recording, decimated, draggable view window, tick marks at
   Defects.
2. **States lane:** the Analysis's states and the Annotation as two rows; overlapping
   algorithm states are visibly marked, never drawn over each other.
3. **Signal lane:** envelope(s), noise floor, troughs, S1/S2/noise peak markers. Hovering a
   peak shows its labels, scores and reasoning.
4. **Heart-rate lane:** BPM/time belief, instantaneous BPM, BPM rebuilt live from the
   Annotation.
5. **Optional lanes** (hidden by default): HRV, systole/diastole intervals, classifier
   scores, spectrum/spectrogram, any debug trace group.

Each lane has a legend column listing its traces in pipeline order; traces toggle
individually there, and hovering an entry fades the lane's other traces. Debug-stage chips
(Preprocessing, Pass 1–3) decide which stages' traces the legends offer; Result traces and
default-visible ones are always offered. Lane/trace/stage visibility persists per user. Also: a summary
panel (BPM range, HRV, plausibility-gate result, algorithm used / switch reason).

### Audio

- Two sources: original and bandpass-filtered (computed on demand, not stored), toggled
  with one key. No speed control.
- The view always follows the playhead during playback.
- **Shift+drag** sets the region (loops playback inside it); **shift+click** clears it.

### Annotation editing

| Action | Gesture |
|---|---|
| Place S1 / S2 (nominal length) centered on playhead; replaces anything it overlaps | **1 / 2** |
| Delete span (selected, else under playhead) | **X / Del** |
| Relabel S1↔S2 | **F** |
| Paint a Noisy span (removes sounds it covers) | **N** + drag |
| Resize a Noisy span (S1/S2 are not resizable) | drag its edge |
| Replace the region with the algorithm's output | **A** on the shift+drag region |
| Flip S1↔S2 right of playhead | button |
| Undo / redo / save | **Ctrl+Z / Ctrl+Shift+Z / Ctrl+S** |

### Defects and Disagreements

- Engine checks (from today's `debug_helpers` detectors: overlaps, peak/state swaps,
  sequence violations, coverage, plausibility gate) run at the end of each analysis and
  are stored as **Defects** in the Analysis.
- **Disagreements** (Analysis states vs Annotation) are computed live as you edit.
- A side panel lists both, filterable by kind, with counts; **] / [** jump next/previous.

### Re-analysis (debug loop)

- **Ctrl+R** re-runs the engine on the open Recording in a fresh worker process, so code
  edits are picked up without restarting the app. Cancellable; UI never blocks.
- After a re-run the view (zoom window, visible traces, Annotation) stays exactly in place.
- **Ctrl+Shift+R / Re-label selection…** re-detects the Shift+drag selection with
  Springer 2015 or Native in a fresh, cancellable worker process. An optional BPM
  hint seeds Native or supplies Springer's duration prior; Auto derives rhythm
  from the audio. Both paths analyse original audio, on the current channel,
  with up to 15 seconds of context on each side. Only the selected Annotation
  range is replaced, in one undo step, with no automatic save. Boundary sounds
  are clipped to preserve all portions outside the selection. The full Analysis
  and its diagnostics stay available for comparison. Failure, cancellation,
  no sounds in the selection, opening another recording/Annotation, or editing
  the selected spans during the run never applies its labels.

### LLM context

- **Ctrl+Shift+C** copies a text dump of the visible window: header (recording name +
  fingerprint, non-default parameters, algorithm, window); one row per peak (time, label
  per pass, scores, pairing confidence, reasoning, values of currently visible debug
  traces); algorithm states and Annotation spans with Disagreements marked; belief values
  at the window edges. No screenshot — the user screenshots what they choose to show.
- The same dump is available as `cli inspect <recording> --from S --to S`.

## 7. Run screen

- A queue of Recordings → Analyses. Add by drop or picker.
- Run-wide settings: algorithm, auto-switch, parallel jobs, channel mode.
- Per-Recording overrides: start-BPM hint (or from filename), analysis start offset.
- Rows show progress, result (BPM range or gate failure), stale badge; double-click opens
  the workspace.
- Explicit action: write BPM into input filenames.
- The CLI provides the same headless (`analyze`), plus `inspect` and `export`.

## 8. Exports

- Workspace: BPM CSV and BPM/time graph (from the Analysis or from the Annotation), window text.
- CLI: BPM CSV, BPM chart, summary.
- No per-run side outputs; everything is an explicit export.

## 9. Performance targets

Measured on a real 1-hour recording (dev machine: Ryzen 9 7900X, RTX 3090, 64 GB):

- Pan/zoom smooth (≈30+ fps) with default lanes + ~10 debug traces, at every zoom level
  from the whole recording to 1 s.
- Opening a Recording with an existing Analysis: first draw < 2 s.
- Annotation edit → redraw (including Disagreements and rebuilt BPM) < 50 ms.
- 2-hour recordings must work (may degrade slightly). Multi-hour recordings are out of scope.

## 10. One-time migration

After the workspace exists: convert the 69 `*.wav_manual_state_sequence.csv` files to
Annotations — keep S1/S2/noisy spans, drop systole/diastole and `bpm_at_mid`, clip the
481 existing overlaps at the midpoint, compute fingerprints, and write a report listing
every clip for spot-checking. Converted files are treated as complete.
