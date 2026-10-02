# Springer HSMM — Python port

Self-contained Python implementation of the **Logistic Regression–HSMM-based Heart Sound Segmentation** method from:

- D. Springer et al., "Logistic Regression-HSMM-based Heart Sound Segmentation," *IEEE Trans. Biomed. Eng.*, vol. 63, no. 4, pp. 822–832, 2016.

Original MATLAB code: [PhysioNet — Logistic Regression-HSMM-based Heart Sound Segmentation](https://physionet.org/content/hss/).

## What it does

Segments a phonocardiogram (PCG) into four states per sample:

1. **S1** (first heart sound)  
2. **Systole** (S1–S2 interval)  
3. **S2** (second heart sound)  
4. **Diastole** (S2–next S1)

The pipeline: bandpass + spike removal → homomorphic/Hilbert/PSD (and optional wavelet) features at 50 Hz → heart rate and systolic interval from autocorrelation → duration-dependent Viterbi decoding → state sequence expanded back to original sampling rate.

## Install

From the repo root (or the directory containing `springer_hsmm`):

```bash
pip install -r springer_hsmm/requirements.txt
```

Optional: install [PyWavelets](https://pywavelets.readthedocs.io/) for the wavelet feature (`include_wavelet_feature=True` in options). If not installed, set `include_wavelet_feature=False` or the feature step will raise.

## Train and run (example script)

Using the example data from the original MATLAB release (e.g. `example_data.mat` next to the MATLAB code):

```bash
# Train on 5 recordings and run on the 6th; save model
python -m springer_hsmm.run_example --mat path/to/example_data.mat --save-model springer_model.npz --plot

# Later: load saved model and segment one recording
python -m springer_hsmm.run_example --mat path/to/example_data.mat --model springer_model.npz --test-index 5 --plot
```

If `example_data.mat` is in `logistic-regression-hsmm-based-heart-sound-segmentation-1.0/` next to this repo, you can omit `--mat`.

## Use as a library

```python
from springer_hsmm import run_springer_segmentation_algorithm, default_springer_hsmm_options
from springer_hsmm.model_io import load_springer_model
import numpy as np

options = default_springer_hsmm_options()
model = load_springer_model("springer_model.npz")
assigned_states = run_springer_segmentation_algorithm(
    audio_data, fs,
    model["B_matrix"], model["pi_vector"], model["total_obs_distribution"],
    options,
)
# assigned_states: 1=S1, 2=systole, 3=S2, 4=diastole per sample
```

Training on your own annotated PCGs (R-peak and end-T-wave positions in samples at audio Fs):

```python
from springer_hsmm.train import train_springer_segmentation_algorithm
from springer_hsmm.model_io import save_springer_model

B_matrix, pi_vector, total_obs = train_springer_segmentation_algorithm(
    list_of_pcg_arrays,
    list_of_(s1_positions, s2_positions)_per_recording,
    fs,
)
save_springer_model("my_model.npz", B_matrix, pi_vector, total_obs)
```

## Package layout

- `options.py` — default options (e.g. 50 Hz feature rate, wavelet on/off)  
- `signal.py` — bandpass, spike removal, homomorphic/Hilbert envelopes  
- `features.py` — PSD, DWT, `get_springer_pcg_features`  
- `heart_rate.py` — `get_heart_rate_schmidt`  
- `durations.py` — `get_duration_distributions`  
- `label_states.py` — frame-level state labels from R-peak/end-T  
- `train.py` — train logistic regression B matrix and total-observation stats  
- `viterbi.py` — duration-dependent Viterbi decode  
- `expand_qt.py` — expand state sequence to original length  
- `run.py` — full pipeline `run_springer_segmentation_algorithm`  
- `model_io.py` — save/load `.npz` model  
- `example_data.py` — load `example_data.mat`

## References

- Springer et al., IEEE TBME 2016 (paper above).  
- Original MATLAB code: PhysioNet / logistic-regression-hsmm-based-heart-sound-segmentation.  
- Schmidt et al., "Segmentation of heart sound recordings by a duration-dependent hidden Markov model," *Physiol. Meas.*, 2010 (duration model and preprocessing).
