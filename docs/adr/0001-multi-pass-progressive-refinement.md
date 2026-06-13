# Multi-pass progressive refinement over a single global decoder

The segmentation pipeline is built as a sequence of [[CONTEXT#Pass|passes]] that each refine a single evolving [[CONTEXT#BPM/time belief|BPM/time belief]] and the per-peak guesses, trending from greedy/local (Pass 1–2 decide peak-by-peak) toward holistic/global (Pass 3 reviews and repairs the whole sequence). A global decoder (Viterbi) is intentionally left disabled.

## Why greedy, and why it stayed

The greedy sequential approach was simply the first method that came to mind. After reading the PCG-segmentation literature it was kept deliberately, for two reasons:

1. **Avoid training an ML model.** A heuristic needs no labelled training set to get started.
2. **It is a data-labelling tool first.** The immediate need was to label a large existing PCG database quickly. A heuristic that runs on any file with no per-recording training met that need.

## Why Viterbi / Springer-style HSMM was deferred

The reference implementation (Springer 2015, logistic-regression HSMM) assumes a *constant* heart rate per recording and a homogeneous, clinically-recorded input distribution. This project's database is the opposite: consumer-grade recordings with *changing* heart rate (exercise/recovery) and recording-specific acoustic quirks. Springer's priors do not transfer, and without a strong prior a Viterbi pass is non-trivial to parameterise — so Pass 4 (Viterbi) exists as a stub but is disabled for now.

## Consequences

- The belief is **re-estimated each pass from the corrected beats**, not carried forward, so later passes benefit from denser evidence. Pass 2 reads its BPM prior from Pass 1's curve rather than from its own running decisions, so the belief never feeds off the very labels it is steering — one local mislabel cannot derail the rest of the run.
- This is explicitly a stepping stone: now that the heuristic can produce manually-verifiable labels at scale, those labels become ground truth for a future ML-based segmenter, at which point a learned-prior Viterbi pass becomes feasible.
