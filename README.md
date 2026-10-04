<p align="center">
  <a href="README.md">English</a> |
  <a href="README-JP.md">日本語</a>
</p>

# Heartbeat BPM Analyzer

This tool is a heuristic based algorithm for phonocardiogram (PCG) Analysis.
It analyzes audio recordings of heart sounds to detect heartbeats and graphs the Beats Per Minute (BPM) over time.

## Overview

```
pcg/
  engine/        the algorithm: PCG audio in, beats / states / BPM out (pure computation)
  recording.py   Recording identity (fingerprint of the decoded audio), audio loading, channels
  analysis.py    Analysis files: the saved result of one run, kept in the library folder
  annotation.py  Annotations: hand-verified S1 / S2 / noisy spans (the ground truth)
  batch.py       run queue, BPM filename tags
  cli.py         python -m pcg analyze / inspect / export / rename
  app/           the desktop app: run screen + workspace
```

Terms (Recording, Analysis, Annotation, Defect, Disagreement, ...) are defined in
[CONTEXT.md](CONTEXT.md); the design is in [docs/workspace-requirements.md](docs/workspace-requirements.md).

- **Analyses** are stored in `library/` (or `$PCG_LIBRARY`), keyed by a fingerprint of the
  recording's audio samples, so renaming a file (including writing BPM into its name) never
  loses its Analysis. Nothing is written next to your recordings except Annotations.
- An Analysis shows a **stale** badge when `pcg/engine/config.py` or the engine code changed since
  it was made. It is only re-run when you ask (Ctrl+R / Run).
- **Annotations** are `<recording>.annotation.json` next to the recording. For recordings inside a
  protected folder (default `inputs/`) saving opens Save As, starting in Downloads.

## Configuration
All engine parameters live in `pcg/engine/config.py` — the only place they are tuned; the app has
no parameter editor. Per-run settings (algorithm, auto-switch, channel, start offset, start-BPM
hint) are chosen on the run screen or the CLI. `tests/test_engine_boundary.py` keeps UI and file
code out of the engine.

## Installation

```bash
pip install -r requirements.txt
```

FFmpeg on your PATH is only needed for formats libsndfile can't decode (e.g. m4a / video files).

## How to Run

```bash
python -m pcg                      # the app (run screen + workspace)
python -m pcg app path/to/rec.wav  # open one recording in the workspace
```

**Run screen:** drop recordings or folders, pick the algorithm / auto-switch / channel / parallel
jobs, optionally set a start BPM or skip per row, then Run. Double-click a row to open it.
"Write BPM into filenames" renames the selected recordings with their `[start,min-maxbpm]` tag.

**Workspace:** stacked lanes on one time axis — states (Analysis vs Annotation, with
Disagreements and Defects), signal (envelopes, peaks; hover a peak for its labels, scores and
reasoning), heart rate, and optional lanes for every debug trace the engine emits (toggle them in
the Traces panel; choices persist). Keys (also under Help → Keys):

| Key | Action |
|---|---|
| Space / T | play-pause / toggle original vs filtered audio |
| Click · Shift+drag · Shift+click | move playhead · loop region · clear it |
| 1 / 2 | place S1 / S2 at the playhead |
| X or Del / F | delete / relabel S1↔S2 |
| N + drag, drag a noisy edge | paint / resize a Noisy span |
| A | replace the loop region with the algorithm's output |
| ] / [ | next / previous Defect or Disagreement |
| Ctrl+Z / Ctrl+Shift+Z / Ctrl+S | undo / redo / save Annotation |
| Ctrl+R | re-run the engine in a fresh process (picks up code edits) |
| Ctrl+Shift+R | re-detect and re-label the selected range with Springer or Native |
| Ctrl+Shift+C | copy the visible window as text for an LLM |

To re-label just part of a recording, **Shift+drag** a range and click **Re-label
selection…** (Ctrl+Shift+R). Choose Springer 2015 or Native, optionally supply a BPM
hint, then run. Both algorithms process original audio with up to 15 seconds of
context on each side. Only Annotation labels inside the selection are replaced;
sounds crossing its edges are clipped, preserving their portions outside it.
The edit is one undo step (Ctrl+Z) and is saved only with Ctrl+S. The full Analysis
remains available for comparison. Failed/cancelled runs and selections with no
detected sounds keep the existing labels. If you edit the selection while a run
is in progress, its result is discarded so your edits are preserved.

## Command Line

```bash
python -m pcg analyze inputs/ -j 8 --rename      # analyze, store in the library, tag filenames
python -m pcg analyze rec.wav --springer --start 5
python -m pcg reanalyze-range rec.wav --from 120 --to 126 --algorithm springer --bpm 90
python -m pcg inspect rec.wav --from 120 --to 126 # the same text as Ctrl+Shift+C
python -m pcg export rec.wav --bpm-csv bpm.csv [--source annotation] --summary summary.txt
python -m pcg rename inputs/                      # write BPM tags from existing Analyses
```

`reanalyze-range` also accepts `--algorithm native`, `--channel mixed|left|right`
and `--context SECONDS` (default 15 per side). It returns the selected S1/S2 spans
as JSON on the recording's clock and writes no Annotation or library files.

The start-BPM hint defaults to the filename tag. Springer and auto-switch runs always go one at a
time: the Springer HSMM needs several GB of memory per minute of audio. For breakpoint debugging,
run `python -m pcg analyze <file>` under a debugger.

## Testing

**Unit tests** — fast, deterministic, no audio fixtures:
```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```
See [tests/README.md](tests/README.md) for the per-module coverage table.

**Benchmark runner** — runs the engine on every annotated recording and compares predicted S1
states against the Annotation, reporting `phase_flip` / `miss` / `extra` per file:
```bash
python benchmarking/run_benchmark.py [input_dir] -j 4   # default: inputs/
```

**Invariant gate** — `python benchmarking/state_invariants.py` (no ground truth needed; see
[debug_helpers/README.md](debug_helpers/README.md)).

## Extra Features:
Import the generated heart rate graph into Blender to easily calculate the change in bpm over time.
Blender file and scripts are located in Blender BPM tool folder

<img src="https://github.com/user-attachments/assets/20130a36-d990-43ba-9cb2-c4d4d248d069" alt="Import BlenderAsj3vbrst4v" width="360" />

Select the Geometry Nodes object and enter edit mode. This will allow you to calculate:
- Heart Rate Recovery (HRR)
- maximal rate of heart rate increase

<img src="https://github.com/user-attachments/assets/f41d8e27-f525-4736-b67a-18de4e4b98e5" alt="Place BlenderAsj3zdst4v" width="360" />
<img src="https://github.com/user-attachments/assets/5d033948-f5b8-485f-9ebe-e9b87a6ee94c" alt="Adjust BlenderAsj3zny4v" width="360" />

You can also make any BPM/Time graph and export it out of blender using the `Export graph data.py` script

Import any CSV file with format: Time(Seconds), Beats Per Minute — e.g. File → Export → BPM CSV in the
workspace, or `python -m pcg export rec.wav --bpm-csv bpm.csv`.
