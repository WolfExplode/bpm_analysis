# Per-database benchmark adapters

Benchmarking the segmenter against external PCG datasets is split into a shared
scoring core plus one thin **adapter** per research database. Each adapter
translates that group's ground-truth format into a canonical input and reports
the literature-standard metric that database is compared by.

## Why

Every research group ships ground truth differently: CirCor uses per-recording
`.tsv` state codes, PhysioNet-2016 uses `.mat` sample-indexed states, datasets
with simultaneous ECG (ephnogram, senssmarttech) carry no S1/S2 labels at all.
Folding all of that into one runner would bloat the core with dataset quirks.
An adapter isolates each format so the scoring core stays clean and reusable as
new databases are added.

## Shape

- **Core** (`benchmarking/bench_scoring.py`): pure per-beat S1 matching →
  Sensitivity / PPV / F1 at fixed tolerances. No dataset knowledge.
- **Adapter** (`benchmarking/adapters/<db>.py`): loads that database's GT,
  emits canonical `(wav, S1/S2 spans, eval window)`, runs the pipeline, calls
  the core, prints + writes JSON.

## Decisions baked in (CirCor, the first adapter)

- **Metric = literature per-beat Se/PPV/F1** at **60 ms** (Springer-standard,
  the comparison number) and **100 ms** — *not* the native flip=1 `error_rate`
  used by the `inputs/` benchmark. This makes CirCor results directly
  comparable to published segmentation work (Springer Se ≈ 97.4 %), at the cost
  of the root-cause flip attribution that helps tuning. Accepted: phase flips
  are a known, separately-tracked defect to be fixed later, so reporting them as
  plain false negatives is the honest baseline.
- **Code-0 (unlabeled, 46.5 % of CirCor time) is clipped out**: predicted S1
  outside labeled spans is neither FP nor FN — no ground truth exists there.
- **Murmur is never scored.** The tool labels cardiac states only; the
  Challenge-2022 murmur/outcome metric is a classifier score and out of scope.

## Consequences

- The native `inputs/` benchmark (flip=1, `run_benchmark.py`) and the CirCor
  adapter deliberately treat phase flips oppositely. Two numbers, two purposes:
  tuning vs external comparison. Do not reconcile them into one.
- Adding a database = adding one adapter file; the core does not change.
