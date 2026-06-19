# GNU Octave and .wav Dataset Support

This folder contains the original Springer MATLAB code with changes for **GNU Octave** compatibility and optional loading from **.wav** files.

## Running under GNU Octave

1. **Install Octave** and the required packages:
   ```bash
   # Install Octave (e.g. Windows: octave from MSYS2 or installer; Linux: apt install octave)
   octave-cli --eval "pkg install -forge statistics; pkg install -forge signal"
   ```

2. **Run the example script** (from this directory):
   ```matlab
   cd path/to/logistic-regression-hsmm-based-heart-sound-segmentation-1.0
   run_Example_Springer_Script
   ```

   The script automatically:
   - Loads the **statistics** package (for `mnrval`, `mvnpdf`, `mnrfit`)
   - Loads the **signal** package (for `resample`, etc.)
   - Disables MEX (uses pure MATLAB/Octave Viterbi) when running under Octave
   - Disables the **wavelet** feature under Octave (the signal package does not provide `wavedec`); the algorithm runs with 3 features (homomorphic envelope, Hilbert envelope, PSD) instead of 4

3. **Example data**: Place `example_data.mat` in this directory (download from [PhysioNet](https://physionet.org/content/hss/) if needed). The .mat includes:
   - **example_audio_data**: 792 PCG recordings at 1000 Hz
   - **example_annotations**: R-peak and end-T-wave positions (samples) per recording
   - **binary_diagnosis** (optional): 0 = normal, 1 = pathology
   - **patient_number** (optional): patient ID per recording

   After loading with `load_springer_data('example_data.mat')`, you can use `data.binary_diagnosis` and `data.patient_number` for benchmarking or stratified splits.

## Using your own .wav dataset

Use `load_pcg_from_wav` to load PCG recordings from .wav files (resampled to 1000 Hz by default):

```matlab
% From a directory of .wav files (same order as dir listing)
[audio_cell, annotations_cell, Fs] = load_pcg_from_wav('path/to/wav_folder', [], 1000);

% With annotations from a .mat file (variables: s1_positions, s2_positions, cell arrays)
[audio_cell, annotations_cell, Fs] = load_pcg_from_wav('path/to/wav_folder', 'annotations.mat', 1000);

% With annotations from a .csv (one row per file; columns: s1_samples, s2_samples, space- or comma-separated)
[audio_cell, annotations_cell, Fs] = load_pcg_from_wav('path/to/wav_folder', 'annotations.csv', 1000);
```

Then train and run as in the example script:

```matlab
pkg load statistics
pkg load signal
springer_options = default_Springer_HSMM_options();

% e.g. train on first 50, test on 51
train_idx = 1:50;
test_idx = 51;
train_recordings = audio_cell(train_idx);
train_annotations = annotations_cell(train_idx, :);
[B_matrix, pi_vector, total_obs_distribution] = trainSpringerSegmentationAlgorithm(...
    train_recordings, train_annotations, Fs, false);

test_audio = audio_cell{test_idx};
assigned_states = runSpringerSegmentationAlgorithm(...
    test_audio, Fs, B_matrix, pi_vector, total_obs_distribution, true);
```

**Annotations format**: R-peak and end-T-wave positions must be in **samples** at the same sampling rate as the audio (e.g. 1000 Hz after resampling). If you have times in seconds, multiply by `Fs`.

## Viewing Octave results in the bpm_analysis plotting interface

To generate the same HTML report that the GUI uses (interactive plot with audio, S1/S2 markers, BPM):

1. **From Octave**, run the export script (from this MATLAB folder):
   ```matlab
   run_Springer_and_export_for_plot
   ```
   This runs the example, then writes to `../processed_files/` (relative to this folder):
   - `octave_springer.wav` — the test recording
   - `octave_springer_springer_export.mat` — assigned_states, Fs, and the WAV filename

2. **From the bpm_analysis repo root**, run the Python plotter:
   ```bash
   python plot_octave_springer_results.py processed_files/octave_springer_springer_export.mat
   ```
   This creates `processed_files/octave_springer_bpm_plot.html` (and optional CSV).

3. Open the HTML via the GUI (**View Reports** / **Open Last HTML Report**) or by opening `processed_files/octave_springer_bpm_plot.html` in a browser.

You can change the export folder or base name by editing `export_dir` and `base_name` in `run_Springer_and_export_for_plot.m`. Use `-o path` with the Python script to write the HTML to a different directory.

## Files added or changed for Octave / .wav

| File | Purpose |
|------|--------|
| `default_Springer_HSMM_options.m` | Sets `use_mex = false` when `OCTAVE_VERSION` is present |
| `run_Example_Springer_Script.m` | Loads `statistics` and `signal` under Octave; uses `load_springer_data` |
| `load_springer_data.m` | Loads `example_data.mat` and normalizes fields (audio_data, annotations, optional diagnosis/patient_number) |
| `load_pcg_from_wav.m` | Loads .wav files (and optional .mat/.csv annotations) for use with the pipeline |
| `wkeep1.m` | Octave compatibility for wavelet code (replaces MATLAB Wavelet Toolbox `wkeep1`) |
| `spectrogram_octave_fallback.m` | PSD feature: Octave fallback when the signal package has no `spectrogram` |
| `get_PSD_feature_Springer_HMM.m` | Uses the fallback under Octave automatically |
| `mnrval_octave_fallback.m` | 2-class logistic regression evaluation when `mnrval` is not available |
| `viterbiDecodePCG_Springer.m` | Calls the fallback under Octave for observation probabilities |
| `export_springer_for_plotting.m` | Saves assigned_states + WAV for the Python plotter |
| `run_Springer_and_export_for_plot.m` | Runs Springer and exports to processed_files for plotting |
