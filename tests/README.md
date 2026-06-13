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

## Scope (intentional)

Only deterministic, fixture-free functions are covered. The end-to-end
pipeline (preprocessing -> classification -> correction) is validated separately
by `run_regression.py` against manually labelled recordings.
