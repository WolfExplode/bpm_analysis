# Holistic feature: confidence-gated global phase decoding

Goal: add one feature that makes the segmenter more *holistic* (whole-recording
reasoning) and raises the CirCor benchmark, without regressing the native corpus.

**Result: CirCor F1 80.1% → 81.1% @60ms (83.0% @100ms); native unchanged
(431 vs 432 errors, 2.5%).** Shipped + default-on.

## Research

- [Documentation/springer2015.md](../Documentation/springer2015.md) — modified-Springer
  HSMM: global HR + time-varying Gaussian **duration model**, LR emissions,
  duration-dependent Viterbi over the whole sequence. The holistic decoder lineage.
- [Sepehri, pediatric ECG-free segmentation](https://www.sciencedirect.com/science/article/abs/pii/S0169260709002909):
  *"respiration affects diastolic duration far more than systolic"* — the
  variance asymmetry that, in principle, orients S1/S2.
- The standing decoder ([correction.py](../correction.py) `_phase_subset_dp`) is
  already a whole-sequence DP; it is **gated below 125 bpm** because above that
  systole ≈ diastole and timing can no longer orient S1 vs S2. That gate (added to
  protect the native exercise corpus) is what caps CirCor — 60% of residual false
  positives live in the gated >125 bpm files.

## The core tension (measured)

Removing the gate gains CirCor (+1.7 pt) but **wrecks native** (errors 2.5%→3.6%):
CirCor's >125 bpm files are *regular resting pediatric* recordings the decoder
fixes; native's are *irregular exercise* the decoder mis-orients. Same operation,
opposite effect — and no per-file label tells them apart.

## Experiments (all measured on full CirCor + the 36-file native set)

| # | Idea | CirCor F1 | Native | Verdict |
|---|---|---|---|---|
| 0 | baseline (gated 125) | 80.1 | 2.5% | — |
| 1 | variance orientation (CV systole<diastole) | — | — | cue only ~60% reliable >100bpm — rejected |
| 2 | amplitude orientation (S1 louder) | — | — | ~48% reliable (CirCor multi-site) — rejected |
| 3 | autocorr cycle → duration anchor | 79.8 | — | worse |
| 4 | autocorr cycle → robust BPM gate | 79.5 | — | worse |
| 5 | ungate decoder fully | 81.8 | **3.6%** | native breaks |
| 6 | prune-only above ceiling | 80.2 | **3.6%** | native breaks (over-drops real beats) |
| 7 | envelope-confirmed missed-S1 recovery | 80.1 | — | neutral (absent beats have no envelope peak) |
| 8 | HSMM Gaussian duration cost | ≤80.1 | — | trades Se↔PPV, no F1 gain |
| **9** | **confidence-gated global decode** | **81.1** | **2.5%** | **shipped** |

## The shipped feature (#9)

The decoder now runs and reports its **mean per-beat duration-fit cost** — a
whole-recording regularity signal: low for a clean periodic rhythm, high for an
irregular/changing-HR one. Above the BPM ceiling the decode is accepted **only
when that cost is below `pass3_phase_confidence_max_cost` (0.08)**:

- CirCor >125 bpm (regular pediatric) → low cost → **accepted**, partial flips and
  spurious cleaned → +1.0 pt.
- Native >125 bpm (irregular exercise) → high cost → **rejected**, labels untouched
  → no regression.

This is "more holistic" in the intended sense: the decoder's confidence in its own
whole-sequence fit decides whether its global correction is trustworthy, instead
of a blunt heart-rate cutoff.

Code: [correction.py](../correction.py) `_phase_subset_dp` (returns mean fit cost),
`_pass3_interval_phase_relabel` (confidence gate). Params in
[config.py](../config.py): `pass3_phase_confidence_gate`,
`pass3_phase_confidence_max_cost`. Native-safety verified by threshold sweep:
errors stay 431–432 for thr ≤ 0.08, climb to 500+ above it.

## Banked negatives
Experiments 1–8 left documented `(Evaluated, off: …)` params in config so the
findings aren't re-discovered. Headline conclusion they establish: **no single
hand-crafted cue orients S1/S2 above ~125 bpm** — that needs the learned
multi-feature HSMM emission model (springer2015.md), a larger future build.
