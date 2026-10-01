# Workspace requirements

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
  (line / points / band). The viewer has no per-trace code.
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

Traces toggle individually; lane/trace visibility persists per user. Also: a summary
panel (BPM range, HRV, plausibility-gate result, algorithm used / switch reason).

### Audio

- Two sources: original and bandpass-filtered (computed on demand, not stored), toggled
  with one key. No speed control.
- Playhead follows playback; auto-scroll can be unlocked.
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

- BPM CSV (from the Analysis or from the Annotation), summary, current view as image.
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
