# Heartbeat BPM Analyzer

A heuristic algorithm for phonocardiogram (PCG) analysis. It detects heart sounds in audio recordings of a heartbeat, labels each as S1 or S2, and plots beats-per-minute over time. Built to extract clinical-grade timing from consumer-grade recordings.

## Language

### Foundations

**Phonocardiogram (PCG)**:
A recording of heart sounds captured by a microphone on the chest. The sole input to this tool — here, sourced from consumer-grade equipment rather than clinical hardware.
_Avoid_: heart audio, recording (when the PCG signal specifically is meant)

**Pass**:
One sweep of the pipeline that refines the [[#BPM/time belief]] and the per-peak guesses. Passes trend from greedy/local toward holistic/global across Pass 1 → N: early passes decide peak-by-peak in local context, later passes review the whole sequence and repair it.
_Avoid_: stage, round, iteration

### Units of detection

**Peak**:
A single point in time flagged as a candidate heart sound — the raw output of envelope peak detection, before any S1/S2 meaning is assigned. A peak says "there is sound *at* this instant".
_Avoid_: spike, hit

**Beat**:
A peak that has been confirmed and labelled as a real cardiac event (an [[#S1]] or [[#S2]]) — the discrete moment the heart makes a sound. Every beat is a peak; not every peak becomes a beat.
_Avoid_: pulse, heartbeat (colloquial for beat)

**State**:
A *span* of time carrying one cardiac meaning — S1, systole, S2, or diastole. Where a peak says "S1 *at* this instant", a state says "S1 *during* this time window". The dense S1/systole/S2/diastole timeline is the state representation introduced in Pass 3.
_Avoid_: phase, segment, label

### Classification outcomes

Every detected [[#Peak]] receives one label from the classifier. Nothing here is ground truth — every label (and every [[#State]]) is the algorithm's *current best guess*, which later passes iterate on and refine. "S1 (Paired)" does not mean "known to be a paired S1"; it means "the algorithm currently guesses this is a paired S1".

**Noise**:
A peak judged to be acoustic noise rather than a heart sound — unwanted energy that happened to be peak-shaped enough to be detected, then rejected. Physically the same stuff as the acoustic-noise terms below; this entry is the *verdict on a peak*, those are *measured quantities*.
_Avoid_: bare lowercase "noise" (always qualify which sense — the label, or one of the acoustic terms)

**S1 (Paired)**:
A peak the algorithm currently guesses is an [[#S1]] *and* associates with a following [[#S2]] in the same cardiac cycle. Higher [[#Pairing confidence]] than a lone guess, but still provisional.
_Avoid_: matched S1

**Lone S1**:
A peak the algorithm guesses is an S1 with no [[#S2]] partner found. A first-class, valid outcome — not a degraded one — because at high BPM S2 genuinely goes inaudible, so long runs of Lone S1 are physiologically expected.
_Avoid_: unpaired S1, orphan S1, single S1

### Heart sounds & cardiac cycle

**S1**:
The first heart sound, produced by closure of the atrioventricular (mitral/tricuspid) valves at the start of ventricular contraction. The louder, lower-frequency sound and the primary anchor of every beat.

**S2**:
The second heart sound, produced by closure of the semilunar (aortic/pulmonary) valves at the end of systole. Higher-frequency and softer than S1; can vanish entirely at high BPM.

**Systole**:
The interval *from S1 to S2* within one cardiac cycle — ventricular contraction and ejection. Relatively fixed in duration as heart rate changes.
_Avoid_: S1-S2 gap (use when naming the measured interval specifically)

**Diastole**:
The interval *from S2 to the next S1* — ventricular relaxation and filling. Shortens dramatically as heart rate rises, which is why S2 weakens at high BPM.

**Pair**:
An S1 and its associated S2, guessed to belong to the same cardiac cycle. Pairing is the *act* of forming this association; a [[#Lone S1]] is what remains when no pair can be formed.
_Avoid_: couple, S1-S2 couple

### Confidence & scoring

**Pairing confidence**:
The score for a *single* pair attempt — how strongly the algorithm currently believes one specific S1 and S2 belong to the same cycle. Momentary, about one pair.
_Avoid_: pair score, match confidence

**Historical pair rate**:
The algorithm's recent success rate at forming pairs, over a rolling window of ~20 beats. A macro feedback signal about overall rhythm quality, *not* about the current pair. Distinct from [[#Pairing confidence]]: that is momentary, this is historical.
_Avoid_: pairing ratio, pair_rate

**Scoring lens**:
One of the independent checks combined into a [[#Pairing confidence]], each deliberately blind to what the others measure so they catch different failure modes. The four lenses: *shape* (amplitude/contractility ratio of S1 vs S2), *timing* (S1-S2 interval plausibility), *stability* (the [[#Historical pair rate]]), and *physical-reality* (hard min/max interval bounds).
_Avoid_: check, scorer, filter

### Belief & BPM estimation

**BPM/time belief**:
The algorithm's single evolving best-guess of the true beats-per-minute curve over the whole recording. Not a measurement — a belief that every pass sharpens as better evidence arrives. The umbrella concept; the two BPM terms below are derived from it.
_Avoid_: BPM estimate, bpm curve (when the *belief* specifically is meant)

**Long-term BPM**:
The slow-adapting heart-rate value sampled from the [[#BPM/time belief]] at a given moment. Stable and deliberately unreactive; it sets expectations (e.g. the pairing window) so one local mislabel can't derail the run.
_Avoid_: average BPM, smoothed BPM

**Instantaneous BPM**:
The raw heart rate implied by the most recent single R-R interval. Reactive and noisy; used only as *evidence* to update the [[#BPM/time belief]], never trusted directly.
_Avoid_: instant BPM, current BPM, live BPM

**Anchor beat**:
A strong, clean S1-S2 pair kept by the first pass at high confidence. Anchor beats are the sparse, reliable points from which the initial [[#BPM/time belief]] and rhythm context are built.
_Avoid_: seed beat, reference beat

### Spectral discrimination

**Spectral fingerprint**:
The recording-specific frequency signature that distinguishes S1 from S2. The premise: within one recording every S1 sounds alike and differs from every S2, even though that difference varies recording to recording. (Exploited cautiously — separation is weak or inconsistent in much of the dataset.)
_Avoid_: spectral profile, frequency signature


