# Springer HSMM — Documentation

## Datasets

### springer_original (example_data.mat)
792 recordings. Annotations derived from ECG (5-detector consensus), already at 50 Hz feature rate — no conversion needed. Aligns natively with Springer's labeling method. Adult patients; mostly normal/mild cardiac sounds.

### PhysioNet 2016 (PN2016)
2873 recordings. Annotations are state transition indices at audio sample rate, converted to feat_fs by taking midpoints of adjacent transitions. Adult clinical recordings; label conversion introduces noise.

### CirCor DigiScope
3163 recordings. Annotations are acoustic span boundaries in seconds (TSV: start, end, state code). All 4 states (S1/systole/S2/diastole) labeled directly; converted to per-frame labels at feat_fs via direct span mapping. Mixed ages including infants; all murmur types represented.

---

## Model Benchmark

Micro-aggregated S1 detection F1 within GT-labeled windows. 50 CirCor files, seed=0. See `bench_models.py`.

| Model | Training data | F1@60ms | F1@100ms |
|---|---|---|---|
| springer_original.npz | 792 recs, example_data.mat | 84.5% | 86.6% |
| cristhian_potes_model.npz | Same weights, pretrained | 84.9% | 86.8% |
| springer_circor_trained.npz | 3162 recs, CirCor | 81.5% | 82.3% |
| springer_pn2016_trained.npz | 2873 recs, PN2016 | 80.6% | 86.0% |

More training data does not improve performance. The 4 Springer features saturate with the original 792-rec dataset. Annotation quality matters more than dataset size.

---

## Limitations and Failure Modes

### Failure rates across models (200 CirCor files, seed=0)

Distribution is strongly bimodal — files either decode well or fail completely. Zero-F1 failures are Viterbi lock-on errors (wrong cardiac cycle phase), not gradual degradation.

| Model | Mean F1 | Hard failures (<50%) | Zero F1 |
|---|---|---|---|
| original | 76.7% | 21.0% | 11 |
| potes | 77.2% | 20.5% | 11 |
| circor | 75.8% | 24.0% | 14 |
| pn2016 | 71.1% | 28.5% | 16 |

### Risk factors by model

Ratio = share of that factor in hard-failing files vs overall. Shown where ratio > 1.4x.

| Factor | original | potes | circor | pn2016 |
|---|---|---|---|---|
| Age: Infant | 2.86x | 2.93x | 2.08x | 2.28x |
| Murmur: Unknown | 2.93x | 3.00x | 2.56x | 2.43x |
| Holosystolic timing | 1.98x | 1.83x | 1.91x | 2.05x |
| Harsh murmur quality | 1.76x | 1.63x | 1.70x | 1.69x |
| MV most audible | 2.38x | 2.44x | 1.67x | 2.11x |
| Murmur: Present | — | — | 1.42x | 1.52x |

### Mean F1 by auscultation position

| Position | original | potes | circor | pn2016 |
|---|---|---|---|---|
| MV | 68.5% | 67.6% | 68.7% | 61.8% |
| AV | 75.3% | 77.1% | 72.9% | 71.9% |
| PV | 78.0% | 80.0% | 78.1% | 73.1% |
| TV | 85.0% | 84.5% | 83.2% | 78.1% |

MV consistently worst across all models; TV best. This likely reflects acoustic overlap and signal clarity at each position.

### Correlation with training data

**original/potes** perform best despite the smallest training set (792 recs). ECG-derived annotations at 50 Hz align exactly with Springer's feature rate, producing clean emission probability estimates. Adult-only data means infant cases are out-of-distribution, but the clean labels prevent over-fitting elsewhere.

**circor** (3162 recs): Infant risk factor is lower (2.08x vs 2.86x for original) — the model has seen infant data during training. However, overall failure rate is higher (24%) and `Murmur: Present` becomes a new risk factor. CirCor contains more acoustic variation and the model appears to struggle with some murmur patterns it didn't encounter adequately during training.

**pn2016** (2873 recs): Worst across the board (28.5% failure rate, lowest mean F1). PN2016 annotation converts state transition indices at audio rate to feat_fs via midpoint interpolation, which introduces label timing noise. This degrades the emission model. TV also becomes a weak position (1.50x), which is unusual and likely reflects distribution differences in the PN2016 population vs CirCor eval data.

### Root cause

Springer's 4 features (homomorphic envelope, Hilbert envelope, PSD 40–60 Hz, wavelet rbio3.9) are tuned for normal adult cardiac acoustics. Infants have higher heart rates and shorter S1–S2 intervals; holosystolic murmurs fill systole with noise that overwhelms S1/S2 transients. These cases are out-of-distribution regardless of which model is used. Retraining on more data cannot overcome this — the features themselves do not capture the relevant signal variation.

Analysis script: `analyze_failures.py`
