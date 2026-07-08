# Tests

Fast, deterministic unit tests for the **pure** helper functions — no audio
fixtures, no FFmpeg, no GUI. These guard the math/logic that the heavy
algorithm tuning is most likely to silently break during refactors.

Run:

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

## Coverage

| File | Module under test | What it pins down |
|------|-------------------|-------------------|
| `test_config.py`            | `config`            | param/output schema, `validate_params` warns-once-per-unknown-key |
| `test_time_utils.py`        | `time_utils`        | fixed-epoch datetime, dense grid edge cases, linear raster interp/extrap |
| `test_peak_utils.py`        | `peak_utils`        | `PeakType` classification, prominence math, **scalar vs vectorized cache parity** |
| `test_peak_label_scores.py` | `peak_label_scores` | label-mass clipping, S2-hint logic, final-confidence extraction |
| `test_hrv.py`               | `hrv`               | MAD outlier masks (incl. MAD==0 keep-all degeneracy), duration clamps, windowed RMSSD/SDNN |
| `test_viterbi.py`           | `viterbi`           | log-domain decode, forbidden-transition handling, emission/transition normalization |
| `test_bpm_input_rename.py`  | `bpm_input_rename`  | BPM tag formatting/rounding, trailing-tag strip (legacy + stacked forms), strip idempotence, no false-strip of legit names |
| `test_file_io.py`           | `file_io`            | output-stem whitespace/emoji normalization, companion-WAV lookup (case/whitespace tolerant, missing/duplicate dirs) |
| `test_ui_settings_loader.py`| `ui_settings_loader`  | legacy-key migration, `ui_settings.json` load (missing/malformed/non-dict), starting-BPM parsing, CLI defaulting from UI settings |
| `test_console_logging.py`   | `console_logging`     | Kaleido/choreographer root-noise filter branches, unicode-safe stream reconfigure (missing attr / raises) |
| `test_fft_profiles.py`      | `fft_profiles`        | pairing-confidence lookup, S1/S2 index collection & top-N selection, envelope->full-rate index scaling, neutral-band alignment, frequency-separation guards, weighted aggregation across files |
| `test_audio_preprocessing.py` | `audio_preprocessing` | channel-mode validation, centered moving average (bit-exact parity vs pandas rolling), sparse-trough interpolation, rolling-quantile helper |
| `test_batch_runner.py`      | `batch_runner`        | filename BPM-tag parsing (tier priority + rightmost match), WAV-preferring dedupe, working-WAV resolution (reuse/copy/convert branches) |
| `test_plotting_helpers.py`  | `plotting`            | epoch-seconds->datetime64 mapping, `</script>` JSON escaping, measured/expected systole & diastole curve extraction, systolic-shift alignment |

## Scope (intentional)

Deterministic, unit-level functions only — no real audio, no FFmpeg, no GUI.
Most tests are fully fixture-free; a few (`test_file_io.py`,
`test_ui_settings_loader.py`, `test_batch_runner.py`) use pytest's `tmp_path`
for the handful of functions that read/write small files or dirs (settings
JSON, WAV lookup/copy) — still fast and hermetic, just not literally
in-memory-only. The end-to-end pipeline (preprocessing -> classification ->
correction) is validated separately by `run_benchmark.py` against manually
labelled recordings.

Deliberately not covered: `gui.py` and `main.py` (no independently testable
logic — event wiring / thin entry point), and anything that only becomes
meaningful with real or synthetic audio, Plotly figure objects, or a full
`AnalysisState` (most of `classifier.py`, `pipeline.py`, and the Plotly/HTML
figure-building methods on `plotting.Plotter`).
