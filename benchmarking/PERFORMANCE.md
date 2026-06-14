# Pipeline performance optimization

Goal: cut pipeline execution time (GPU or otherwise), within reason, stopping when
returns no longer justify the risk. Method: profile first, change the biggest safe
hotspot, verify the CirCor benchmark stays **bit-identical** (F1 80.1%) and unit
tests pass after each change.

## Baseline (steady-state, warm imports)
- **183 ms/file**; full CirCor 2624-file run **31.6 s** wall (23 parallel workers).
- cProfile (40 files, self-time) top hotspots:

| self-time | what | note |
|---|---|---|
| **2.07 s (28%)** | `gc.collect()` | explicit, once per file, in `preprocess_audio` |
| 0.40 s | pandas rolling `.calc` | mostly the rolling-**quantile** noise floor |
| 0.40 s | scipy FFT (`c2c`) | Hilbert envelope + spectra |
| 0.18 s + tail | `confidence_engine.attempt_pair` | sequential Python pairing loop |
| — | many small numpy reductions | spread thin |

## Changes shipped

### 1. Drop per-file `gc.collect()` — the big win
[audio_preprocessing.py](../audio_preprocessing.py) forced a full `gc.collect()`
after freeing the HF buffers on **every** normal run. The preceding `del`s already
release those arrays, so the collection was wasted — and it dominated runtime
(28% directly, more via cache thrash). Now gated behind `preprocess_force_gc`
(default **False**); opt-in only for memory-tight single-file runs.

- **183 → 83 ms/file (2.2×)**; full run 31.6 → ~22 s. Benchmark + tests unchanged.

### 2. Exact O(n) centered moving average
[audio_preprocessing.py](../audio_preprocessing.py) `_centered_moving_average`
replaces two `pd.Series(...).rolling(min_periods=1, center=True).mean()` envelope
smooths with a cumulative-sum form, **bit-exact** to pandas (≤1e-13, verified for
even/odd windows). Minor (the costly rolling was the quantile, not the mean), but
exact and removes a pandas dependency from the hot path. Full run → **20.8 s**.

## GPU verdict: not worth it — rejected with reason

GPU acceleration was evaluated and **declined**:
- **Arrays are small.** Analysis runs at 600 Hz → a few-second recording is only
  ~2–9k samples. A 3k-point FFT is ~µs–ms on CPU; host↔device transfer + kernel
  launch overhead per call would *exceed* the compute. Net slower.
- **The hot logic isn't vectorizable.** `confidence_engine` pairing is sequential,
  branchy, stateful Python over beats — no array kernel maps onto it.
- **The real parallelism is across files**, and it is already exploited:
  `multiprocessing.Pool` over recordings (≈ core-count speedup, the dominant lever
  for batch runs). That is the correct "other method," already shipped.

A GPU helps when one op crunches large contiguous batches; this workload is the
opposite (many tiny, heterogeneous, partly-sequential steps).

## Stopping point
Banked the large, zero-risk wins: **31.6 s → 20.8 s** full run (and 2.2×/file
single-file), benchmark bit-identical, all tests green. What remains —
rolling-quantile noise floor (exact replacement is non-trivial), FFT length
padding (~5%), and the Python pairing loop (algorithmic rewrite) — each offers
<10% for rising correctness risk. Past the point of reasonable return; stopping
here.
