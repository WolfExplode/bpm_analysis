# CirCor benchmark optimization log

> **STATUS: LOCKED at F1 80.1% @60ms** (from 63.9% baseline, +16.2 pts; native
> benchmark improved 2.9%→2.5%). 90% target deferred — it requires acoustic
> S1/S2 discrimination (classifier-layer work with native-corpus regression
> risk), a scope explicitly declined for now. Two changes shipped: global phase
> swap + subset-DP phase decoder.

Goal: push CirCor S1 segmentation **F1 ≥ 90%** (Springer HSMM ≈ 97.4% Se).
Metric: per-beat Se/PPV/F1 @60 ms (literature standard). See
[bench_scoring.py](bench_scoring.py), [adapters/circor.py](adapters/circor.py),
[ADR 0003](../docs/adr/0003-per-database-benchmark-adapters.md).

Rule: optimize the **algorithm**, never the benchmark. No peeking at ground
truth at inference time.

---

## Baseline (full 2624 recordings, parallel run, 31.6 s)

| Tol | Se | PPV | F1 |
|---|---|---|---|
| 60 ms | 65.5% | 62.5% | **63.9%** |
| 100 ms | 66.9% | 63.9% | 65.4% |

Counts @60 ms: TP=34413 FN=18152 FP=20653 **flip=13298**.

### Failure distribution (per-file, @60 ms)
- clean F1≥0.9 : 913 (34.8%)
- majority-flip : 588 (22.4%)
- partial-flip : 650 (24.8%)
- over-detect : 282 (10.7%)
- true-miss : 191 ( 7.3%)
- median per-file F1 = 0.765, mean = 0.611
- **388 files at F1 = 0**, 707 files < 0.3.

### Read
Phase flips (right time, S1/S2 swapped) = **73% of all FN**. The 388 F1=0 files
look like whole-recording inversions — a single per-file decision. Fixing label
stability is the dominant lever: TP+flip / manual ⇒ Se ≈ 90.8% ceiling.

CirCor caveat: clean resting pediatric clinical PCG (stationary rate) — Springer's
home turf, the opposite of this tool's design corpus
([ADR 0002](../docs/adr/0002-pcg-database-characteristics.md)). So this benchmark
stresses label-stability, not BPM tracking.

---

## Investigation 1 — is the flip mass a global per-recording inversion? ✅

Tool: [circor_phase_audit.py](circor_phase_audit.py). For each file, score normally
vs with S1↔S2 swapped; an oracle picks the better orientation per file.

| Orientation | Se | PPV | F1 |
|---|---|---|---|
| current (normal) | 65.5% | 62.5% | 63.9% |
| **oracle per-file phase** | 82.9% | 79.4% | **81.1%** |

- **458 strong global inversions** (normal F1<0.3, swapped F1>0.7).
- 592 files (22.6%) improve if swapped.

**Confirmed:** most flip mass is whole-recording S1↔S2 inversion. Oracle ceiling 81%.

### Unsupervised decision rule (no ground truth)
Physiology: at rest systole (S1→S2) < diastole (S2→S1). So swap when
median(systole) > median(diastole), measured from the tool's own S1/S2 peak
centers. Validated in the audit: **F1 76.6%** (Se 78.2 / PPV 75.1) — captures
most of the oracle. Fails only for fast recordings (>~120 bpm) where diastole
genuinely falls below systole and timing can't disambiguate.

---

## Change 1 — global phase-correction pass (shipped)

[correction.py](../correction.py) `_pass3_global_phase_correction`, wired at the
end of `run_pass3_correction`; params in [config.py](../config.py)
(`pass3_global_phase_correct`, `pass3_global_phase_bpm_ceiling`,
`pass3_global_phase_margin`). Pure name-swap S1↔S2 + systole↔diastole on the
final timeline; only fires below the BPM ceiling (protects the native
high-BPM corpus, where the inequality can legitimately reverse).

| Step | Se | PPV | F1 | flip |
|---|---|---|---|---|
| baseline | 65.5% | 62.5% | 63.9% | 13298 |
| span-duration detector | 71.9% | 68.9% | 70.4% | 9896 |
| **center-interval detector** | 74.0% | 70.9% | **72.4%** | 8827 |

Center-to-center intervals (not span widths) match the audit rule and detect
more inversions. Param sweep on margin/ceiling: in progress.

### Committed config + safety
margin 1.0, ceiling 125. **Official CirCor F1 = 74.2%** @60ms (76.0% @100ms),
Se 75.7 / PPV 72.7. Native 38-file benchmark regression check (phase OFF vs ON):
515→499 errors — **no regression, slightly better**. Default-on is safe.

## Investigation 2 — what are the false positives? ([circor_fp_audit.py](circor_fp_audit.py))

Of 14,379 false-positive S1s, by the manual state they land in:
- **S2 : 9346 (65.0%)** — partial within-recording S1/S2 swap (not global)
- diastole : 3371 (23.4%) — spurious quiet-interval detections
- systole : 1403 ( 9.8%)
- S1 : 259 ( 1.8%) — double-detect of one S1

Pred S1 (54740) ≈ manual S1 (52565): not over-labeling, it's **beat-level phase
confusion**. The 9346 FP-on-S2 and the 7918 flip-FN are the same defect. Fixing
within-recording phase ⇒ Se ≈ 92%, PPV ≈ 89%, **F1 ≈ 90%** — the lever to goal.

## Investigation 3 — phase ceiling, and what the "misses" really are

- Per-file oracle phase swap: **F1 81.1%** (uses GT to pick orientation).
- Of 12,453 S1 "misses", **61% (7609) have a detected sound within 60 ms** —
  mislabelled S2, i.e. phase errors. Only **21% (2663) truly absent** (>120 ms).
- Tried three ground-truth-free phase decoders on the detected sounds:
  pairwise interval rule 76.2%, 2-state Viterbi 77.0%, **subset-DP 81.7%**
  (SKIP=0.4). All ceiling near ~82% — the detected-sound stream (spurious +
  missed + jitter) caps timing-only phase. **90% needs cleaner detection.**

## Change 2 — subset-DP phase decoder (shipped, replaces the per-beat relabel)

[correction.py](../correction.py) `_phase_subset_dp` + `_pass3_interval_phase_relabel`;
param `pass3_phase_skip_penalty=0.4`. Picks the lowest-cost alternating S1/S2
chain through the detected sounds (S1→S2≈systole, S2→S1≈diastole, same-state≈a
skipped beat); off-chain sounds are spurious and dropped. Gated by the BPM ceiling.

| Step | Se | PPV | F1 | flip |
|---|---|---|---|---|
| baseline | 65.5 | 62.5 | 63.9 | 13298 |
| + global phase swap | 75.7 | 72.7 | 74.2 | 7918 |
| + per-beat relabel | 76.3 | 74.0 | 75.1 | 7610 |
| **+ subset-DP decoder** | **80.0** | **80.1** | **80.1** | 5206 |
| (@100 ms) | 81.9 | 81.9 | 81.9 | 5741 |

**Native regression check** (36 files): errors 515→**432** (2.9%→2.5%). DP prunes
spurious (extra 431→302) at the cost of a few dropped faint beats (miss 58→109);
net clearly positive. Default-on safe.

## Status: 63.9% → 80.1% F1 @60ms (81.9% @100ms). Native: improved (2.9%→2.5%).

Two shipped, native-safe changes (global phase swap + subset-DP decoder) closed
**16.2 points**. All scratch probes removed; permanent tools kept
([circor_phase_audit.py](circor_phase_audit.py), [circor_fp_audit.py](circor_fp_audit.py)).

## The wall before 90% — timing vs acoustics

Every timing-based phase method ceilings at **~82%** (oracle 81.1, subset-DP
81.7). Post-DP false positives are still **60% on manual S2** (6244): residual
phase errors concentrated in recordings **>125 bpm**, where systole ≥ diastole so
the timing rule cannot orient S1 vs S2 and the decoder is gated off. ~12% of S1
mass lives there.

**To break 82% → 90% the last points must come from *acoustic* S1/S2
discrimination** (amplitude/spectral), not timing:
- >125 bpm files: orient S1/S2 by loudness/spectrum (S1 louder, lower-freq).
- ~2663 truly-absent (faint) S1: detection sensitivity, gated to clean/high-SNR.
- −18 ms systematic S1-center bias (convention) — minor; not pursued (gaming-adjacent).

This is classifier-layer work tuned for pediatric clinical PCG, with real
regression risk to the native consumer corpus the tool is built for
([ADR 0002](../docs/adr/0002-pcg-database-characteristics.md)). Scope decision
required before proceeding. **Decision: locked at 80.1%; 90% path declined.**

---

## Appendix — full change inventory

### Algorithm (ship-affecting, default-on)
- [config.py](../config.py): `pass3_global_phase_correct` (True),
  `pass3_global_phase_bpm_ceiling` (125), `pass3_global_phase_margin` (1.0),
  `pass3_interval_phase_relabel` (True), `pass3_phase_skip_penalty` (0.4).
- [correction.py](../correction.py): `_pass3_global_phase_correction`,
  `_phase_subset_dp`, `_pass3_interval_phase_relabel`; two call sites wired into
  `run_pass3_correction` (after timeline finalize, before publish), plus a
  measured systole/diastole curve swap to stay consistent after a global flip.
- All gated below 125 bpm so the native high-/changing-BPM corpus is untouched;
  native benchmark verified non-regressing (improved, 2.9%→2.5%).

### Benchmark infrastructure (no effect on tool output)
- [bench_scoring.py](bench_scoring.py): shared per-beat Se/PPV/F1 scorer.
- [adapters/circor.py](adapters/circor.py): CirCor adapter. Parallelised with
  `multiprocessing.Pool` + `--jobs` (default CPU−1); BLAS/FFT pinned to 1
  thread/process to avoid oversubscription (full run ~31 s vs minutes serial).
  `BENCH_PARAM_OVERRIDES` env hook so param sweeps reach spawned workers.
  **Windows note:** parallel runs must live in importable modules, never inline
  `python - <<EOF` heredocs — spawn re-imports the module to find the worker, and
  a heredoc has none, so the pool deadlocks.
- Tools: [circor_phase_audit.py](circor_phase_audit.py) (normal-vs-swapped phase,
  oracle ceiling, unsupervised rule), [circor_fp_audit.py](circor_fp_audit.py)
  (false-positive S1 by manual state).

### Observed, not changed
- [analysis_data_schema.py:110](../analysis_data_schema.py) docstring is stale:
  claims `pass3_state_boundaries` = `(state_name, start_sec, end_sec, meta)`; real
  format is `(start_sample, end_sample, state_name, meta)` (see
  `run_pass3_correction` append sites). Left as-is; flag for a future doc fix.
- `circor_benchmark_result.json` is a generated artifact (tracked in git);
  consider gitignoring.
