# The PCG database the algorithm is built around
The algorithm's design is shaped by the specific corpus it must handle. Recording this here because none of it is visible in the code, yet it explains most of the architectural choices (see [[0001-multi-pass-progressive-refinement]]).

## What the database is
- **Provenance: collected from public sources**, not self-recorded. Many different subjects, recorded on unknown and varied consumer equipment (phone mics, contact mics, etc.) under uncontrolled conditions. There is no clean, homogeneous clinical capture to rely on.
- **Heart rate changes within a recording.** A large share captures exercise → peak exertion → recovery, so BPM is non-stationary across a single file. This is the central reason a constant-rate model (Springer-style HSMM) does not fit.
- **Hard real-world phenomena are present**, not edge cases: faint or fully inaudible S2 at high BPM, [[CONTEXT#Spectral fingerprint|recording-specific]] acoustic character, split S1/S2, respiratory sinus arrhythmia (RSA), and other arrhythmias.

### What this database contains
- **Physiological states**: resting, exercise (squats, stationary bike, stairs, running), peak exertion, post-exercise recovery, breath-hold, fear, arousal. This wide range of autonomic states is the primary driver of non-stationary BPM and varying S1/S2 prominence within a single recording.
- **Stethoscope placements**: apex, tricuspid, mitral, and pulmonic valve areas — each with different S1/S2 amplitude ratios and acoustic character. Most files do not contain any indication of which auscultation location it was recorded at, if any at all.
- **Pathological conditions**: arrhythmia (irregular rhythm, RSA, PSVT), tachycardia (including idiopathic), mitral valve prolapse. These are present as incidental findings, not recruited subjects.
- **Demographic range**: predominantly female subjects of varying ages; some male; multiple ethnic backgrounds consisting primarily of (English, Japanese, and Chinese).
- **Format variety**: WAV, WMV, WMA, WEBM sourced from social media, YouTube, Discord, and dedicated audio communities. Recording equipment is unknown and inconsistent. Phone microphones, contact mics, stethoscopes, and Doppler devices are all represented.
- **Recording quality**: widely variable. Noise levels, background sound, mic placement quality, and signal-to-noise ratio differ substantially across files. This variability is a primary feature of the corpus. The algorithm must handle poor recordings as well as clean ones.


### What this database does not contain
- **Clinical ground truth**: No ECG, no simultaneous phonocardiograph reference, no controlled recording protocol. The only ground truth is the manually-labelled `_manual_state_sequence.csv` files created for this project.
- **Homogeneous equipment**: No standard stethoscope, microphone, or recording setup. Clinical PCG databases are typically recorded with consistent, calibrated equipment; this one is not.
- **Stationary-BPM recordings as the primary case**: Unlike most PCG research datasets, recordings of subjects at a fixed resting heart rate are a minority. The algorithm is optimized for the changing-BPM case.
## Organization




## Organization
- **Difficulty tiers (1–5, plus "Impossible")** stratify files by *segmentation hardness* — how hard the recording is to label correctly (noise, faint/absent S2, extreme BPM, irregular rhythm). 1 = clean and easy; 5 = very hard; Impossible = no reliably recoverable signal. The regression runner defaults to a representative tier.
- **Filename convention** encodes metadata: `title [MM-DD-YYYY][tags] [primary_bpm, min-max bpm].wav`, where tags flag conditions (`[fast]`, `[irregular]`, `[RSA]`, …) and the bracketed numbers give the labelled primary BPM and observed range.
- **Manual ground truth** lives beside select files as `<name>.wav_manual_state_sequence.csv`. These hand-labelled state sequences are the only ground truth in the project and drive the end-to-end regression test.

## Why this matters
This corpus is also the asset that makes the long-term plan viable: as the heuristic labels more of it (verified by the manual ground truth), the labelled set grows into a training corpus for a future ML-based segmenter — the eventual exit from the heuristic approach.

## Accessing the database
This database is not publicly accessible.
