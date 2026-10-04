# Tests

Fast, deterministic unit tests — no real recordings, no FFmpeg, no GUI. They guard the
math/logic that algorithm tuning and refactors are most likely to silently break, and the
file formats and editing rules of the workspace.

Run:

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

## Coverage

| File | Module under test | What it pins down |
|------|-------------------|-------------------|
| `test_engine_boundary.py`   | `pcg/engine`        | engine files import only the engine (relative) and third-party compute packages; importing `pcg.engine` loads no `pcg.*` app module and no UI/output package |
| `test_no_param_drift.py`    | whole repo          | parameters are read with `param(params, key)`, never `params.get(key, <literal>)` |
| `test_config.py`            | `engine.config`     | param schema, `validate_params` warns once per unknown key, `param` fallback |
| `test_engine.py`            | `engine.run`        | auto-switch-algorithm decision (switch on pass / fewer reasons, keep on tie) |
| `test_time_utils.py`        | `engine.time_utils` | fixed-epoch datetime, dense grid edge cases, linear raster interp/extrap |
| `test_peak_utils.py`        | `engine.peak_utils` | `PeakType` classification, prominence math, scalar vs vectorized cache parity |
| `test_peak_label_scores.py` | `engine.peak_label_scores` | label-mass clipping, S2-hint logic, final-confidence extraction |
| `test_confidence_engine.py` | `engine.confidence_engine` | interval expectations and pairing-confidence lenses |
| `test_hrv.py`               | `engine.hrv`        | MAD outlier masks, duration clamps, windowed RMSSD/SDNN |
| `test_noise_segments.py`    | `engine.noise_segments` | HF noise-segment detection and interval merging |
| `test_correction.py`, `test_phase_decision.py` | `engine.correction`, `engine.phase_decision` | Pass 3 helpers and the phase-subset DP |
| `test_viterbi.py`           | `engine.viterbi`    | log-domain decode, forbidden transitions, emission/transition normalization |
| `test_audio_preprocessing.py` | `engine.audio_preprocessing` | centered moving average (bit-exact vs pandas), sparse-trough interpolation, rolling quantile |
| `test_state_overlap.py`, `test_coverage_detector.py`, `test_state_sequence.py`, `test_peak_state_mismatch.py` | `engine.defects.*` | the four invariant checks behind Defects |
| `test_analysis_file.py`     | `pcg.analysis`, `pcg.recording`, `engine.traces` | Analysis zip round trip, staleness (params / engine code, not run settings), library paths, Trace validation / shifting / windowing, fingerprint stable across rename and WAV↔FLAC but not trimming, fingerprint cache, channel extraction, protected folders |
| `test_annotation.py`        | `pcg.annotation`    | every edit keeps spans sorted and non-overlapping (incl. 2000 random edits), seeding/clipping from algorithm states, replace-region, BPM from S1 (Noisy intervals excluded), Disagreements (vs a brute-force reference), derived states, JSON round trip, fingerprint matching |
| `test_batch.py`             | `pcg.batch`         | filename BPM tags (parse priority, strip/format), BPM rename incl. the Annotation sidecar, input collection (WAV preferred), Springer runs serialised, annotated recordings found after renames |
| `test_reanalysis.py`        | `pcg.reanalysis`, `recording`, `annotation`, `cli` | selected audio/context/channel extraction, recording-clock alignment, boundary preservation, empty/invalid results, both complete algorithm pipelines on synthetic PCG, JSON without saving |
| `test_reanalysis_workspace.py` | `app.workspace`, `app.state` | range algorithm/BPM choice, undo/redo, cancellation/failure, in-flight edits, stale Annotation results, real CLI subprocess protocol for both algorithms |

## Scope (intentional)

No real audio: a few tests write tiny synthetic WAV/FLAC files to `tmp_path`. The
end-to-end engine is validated separately against the Annotations by
`benchmarking/run_benchmark.py` and, without ground truth, by
`benchmarking/state_invariants.py`. Qt interaction checks run offscreen, including
selected-range reanalysis, gain editing and waveform display; most app logic lives
in the tested modules (`annotation`, `analysis`, `context`, `batch`).
