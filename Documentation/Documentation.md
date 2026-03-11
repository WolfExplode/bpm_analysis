# **BPM Analysis documentation**

# Design theory
The goal is to take PCG data recorded via consumer household equipment and convert it into usable data. to plot changes in heart rate, beats per minute (bpm) over time. Normally, clinical PCG data is captured via advanced and expensive equipment that may not be easily accessible to the general public such as digital stethoscopes etc. My goal is to extract information by using a algorithm to compensate for inadequate hardware. 

The script will track bpm over time like a heart rate monitor, but using a recording of a patient's heartbeat instead. The resulting plot should accurately reflect fast changes in heart rate.

I don't like the idea of using a `start_bpm_hint` as a user input since it makes batch processing impossible. I want the algorithm to be automatic and independent. 
In a ideal world, the user should only need to enter a file, press run, and get the data out. All other logic should be handled by the script

> [!say]
> I want to write my code to be "self documenting". I need to decrease the "mental entropy" required to determine if a future change in the code will clash or make previous bits of code redundant or not. Ideally, a lot of these "band aid fixes" that are currently in the code will be replaced with a more robust algorithm if I can come up with the proper solution. 
> 
> To improve maintainability of my codebase, this documentation.md organizes my brainstorming notes. This documentation file will be used to store the entire explanation behind design choices while brief explanations are written in the code as comments. Documentation should be reserved for reasoning and not the specific details on how the idea is implemented. This keeps our documentation flexible to massive algorithm changes with no changes in documentation. 
> I'm treating the documentation like a developer's journal since I'm currently the only user of the tool. One of the best ways to make my codebase more maintainable is to write documentation to help explain the design decisions that led to the current state of the codebase.


## Design requirements:
It's important to lay out the design requirements and scope of this project. Initially I just made this project to plot bpm/time but we have since move way beyond that. simply plotting bpm/time now seems trivial.
Now I'm updating the design requirement to:
**Find the locations of every S1 and S2 heart sound**
Primary challenges include but are not limited to: 
- noise
- lack of audible S2
- very faint S2
- split S1 or split S2
- S3 heart sound

Future design requirements:
**Find the locations of every S1 and S2 heart sound and identify anomalies such as murmur etc..**


# General knowledge:
## Phonocardiography (PCG)
**Phonocardiography (PCG)** is a non-invasive technique that records and analyzes heart sounds and murmurs. It uses a sensitive microphone (phonocardiograph) placed on the chest wall to convert acoustic vibrations into electronic signals, which are then displayed as a waveform (phonocardiogram).
**Key points:**
- **Purpose**: Provides a visual representation of heart sounds (S1, S2, S3, S4) and any pathological murmurs, allowing detailed timing and frequency analysis.
- **Advantage**: Unlike auscultation with a stethoscope, PCG creates a permanent, objective record that can be replayed, measured, and compared over time.
- **Applications**: Diagnosing valvular diseases (e.g., stenosis, regurgitation), congenital heart defects, and cardiac arrhythmias; also used in research and teaching.
- **Modern use**: Often integrated with digital systems and can be combined with ECG for precise correlation with the cardiac cycle.
https://youtu.be/3pwZTYIwBks?t=460



**Audio depends on positioning of the heart relative to the recording device:** 
This is relevant when the patient moves their body, rotates their chest, Inhales or exhales. 
As the diaphragm descends, the heart itself shifts slightly in the chest cavity. It can rotate or move downwards, changing its orientation relative to the fixed position of the stethoscope. 
For example, the heart's rotation might simultaneously move the pulmonic valve closer or into a better acoustic alignment with the stethoscope resulting in a S1 being louder or quieter throughout the recording. 


## Existing methods
It's important to acknowledge and aggregate existing methods to do what I'm trying to accomplish
```embed
title: "Logistic Regression-HSMM-based Heart Sound Segmentation v1.0"
image: "https://physionet.org/static/images/physionet-logo.svg"
description: "Heart sound segmentation code, based on a duration-dependent hidden Markov model, extended with the use of logistic regression for emission probability estimation and an enhanced Viterbi algorithm."
url: "https://physionet.org/content/hss/1.0/"
favicon: ""
aspectRatio: "24.12831241283124"
```

```embed
title: "             An Open Access Database for the Evaluation of Heart Sound Algorithms - PMC         "
image: "https://cdn.ncbi.nlm.nih.gov/pmc/banners/logo-nihpa.png"
description: "In the past few decades, analysis of heart sound signals (i.e., the phonocardiogram or PCG), especially for automated heart sound segmentation and classification, has been widely studied and has been reported to have the potential value to detect ..."
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC7199391/"
favicon: ""
aspectRatio: "15"
```

```embed
title: "Classification of Heart Sound Recordings: The PhysioNet/Computing in Cardiology Challenge 2016 v1.0.0"
image: "https://physionet.org/files/challenge-2016/1.0.0/figure1.png"
description: ""
url: "https://physionet.org/content/challenge-2016/1.0.0/"
favicon: ""
aspectRatio: "62.5"
```



## Comparing my method with existing heart sound segmentation algorithms
[springer2015](https://physionet.org/content/hss/1.0/)
Springer's pipeline was not made to process PCG information at different/changing heart rates. It expects the bpm to be constant across time.
- **Springer:** Calculates **one global HR estimate per recording** before segmentation begins. It uses this single value to parameterize the Gaussian duration distributions (S1, Systole, S2, Diastole) for the **entire sequence**.

[A Wavelet Transform-Based Neural Network Denoising Algorithm for Mobile Phonocardiography](https://doi.org/10.3390/s19040957)

### [Heart sound datasets:](https://pmc.ncbi.nlm.nih.gov/articles/PMC11461928/)
https://pmc.ncbi.nlm.nih.gov/articles/PMC7199391/


## Existing PCG code/libraries
```embed
title: "biosppy.signals — BioSPPy 2.2.2 documentation"
image: "https://biosppy.readthedocs.io/en/stable/_images/math/8045a195c2f29ae10ca11ed1a5cd3dd2801a79d3.png"
description: ""
url: "https://biosppy.readthedocs.io/en/stable/biosppy.signals.html"
favicon: ""
aspectRatio: "14.615384615384617"
```

```embed
title: "Segmentation — pyPCG 0.1b5 documentation"
image: ""
description: ""
url: "https://pypcg-toolbox.readthedocs.io/en/latest/segment.html"
favicon: ""
```



## Background Info on time frequency analysis
```embed
title: "Time and frequency domains"
image: "https://i.ytimg.com/vi/fYtVHhk3xJ0/maxresdefault.jpg"
description: "This video lesson is part of a complete course on neuroscience time series analyses.The full course includes   - over 47 hours of video instruction  - lots a..."
url: "https://youtu.be/fYtVHhk3xJ0"
favicon: ""
aspectRatio: "56.25"
```
Time and frequency domains are two extremes of this uncertainty. You can either know exactly what a value of function is at every time point, but at the cost of being completely ignorant about what frequencies are at that time point. On the other end of the spectrum in the frequency domain we know exactly what frequencies are present in the signal but we have no idea about the temporal dynamics of them.
### Fourier transform
The power spectral density is that graph shown in [FL studio Fruity Parametric EQ 2](https://youtu.be/YrGxCRlCvQI?t=124) when I play the audio. 
The Fourier transform converts data from the time domain to the frequency domain. 

```embed
title: "What is Windowing in Signal Processing?"
image: "https://i.ytimg.com/vi/1Hd72RpMFlQ/maxresdefault.jpg"
description: "Explains the role of Windowing in signal processing, starting with an example of basic audio compression.* If you would like to support me to make these vide..."
url: "https://youtu.be/1Hd72RpMFlQ"
favicon: ""
aspectRatio: "56.25"
```

```embed
title: "Windows and Spectral Leakage"
image: "https://i.ytimg.com/vi/pD7f6X9-_Kg/maxresdefault.jpg?sqp=-oaymwEmCIAKENAF8quKqQMa8AEB-AHUBoAC4AOKAgwIABABGE8gZShlMA8=&rs=AOn4CLAStUjGo6bYNkwzG5OKfC55rSb0UQ"
description: "More information on the Simcenter Testing community: https://community.sw.siemens.com/s/article/windows-and-spectral-leakageAdditional resources from the Sim..."
url: "https://youtu.be/pD7f6X9-_Kg?t=259"
favicon: ""
aspectRatio: "56.25"
```

```embed
title: "How to use the FFT like a pro, 3 essential signal prep tips"
image: "https://i.ytimg.com/vi/tjYMprOD3GI/maxresdefault.jpg"
description: "Unsure how to use the FFT to get meaningful results from your data? Join me as I unveil 3 crucial signal preparation tips to ensure accurate frequency analys..."
url: "https://youtu.be/tjYMprOD3GI?t=238"
favicon: ""
aspectRatio: "56.25"
```





[Nyquist-Shannon Sampling Theorem](https://youtu.be/vrXGaFV1AmE)


### [Application of time frequency analysis](https://www.youtube.com/playlist?list=PLn0OLiymPak2BYu--bR0ADNBJsC4kuRWs)
### What is a Wavelet Transform?
In order to do time series analysis, we must understand these fundamental concepts:
```embed
title: "Wavelets: a mathematical microscope"
image: "https://i.ytimg.com/vi/jnxqHcObNK4/maxresdefault.jpg"
description: "=====My name is Artem, I'm a neuroscience PhD student at Harvard University. 🌎 Website and Social links: https://kirsanov.ai/📥 \"Receptive Field\" neuro-news..."
url: "https://youtu.be/jnxqHcObNK4"
favicon: ""
aspectRatio: "56.25"
```

What is wavelet [convolution](https://youtu.be/jnxqHcObNK4?t=1282)?
We need to find the contribution of a certain frequency around a timepoint. 
By applying wavelet transform, to generate a wavelet scalogram, we can [view a sound's structure](https://youtu.be/jnxqHcObNK4?t=1813)

Side note: [Mike X Cohen](https://youtu.be/ljw3gW-nL0E?t=1721) is fkin intelligent:
We should maintain a holistic view. [If you have a real finding in your data, it will be robust to a reasonable range of parameters](https://youtu.be/ljw3gW-nL0E?t=1676)
aka a good algorithm should have a large range of configuration values that will work on any input data. aka, if you find yourself tuning config/parameters often, your algorithm is not robust. 
This is why it's a good idea to use wavelet convolution for time series analysis

```embed
title: "Morlet wavelets in time and in frequency"
image: "https://i.ytimg.com/vi/7ahrcB5HL0k/maxresdefault.jpg"
description: "This video lesson is part of a complete course on neuroscience time series analyses.The full course includes   - over 47 hours of video instruction  - lots a..."
url: "https://youtu.be/7ahrcB5HL0k"
favicon: ""
aspectRatio: "56.25"
```
Convolution in the time domain acts as a band pass filter:
```embed
title: "Convolution in the time domain"
image: "https://i.ytimg.com/vi/9Hk-RAIzOaw/maxresdefault.jpg"
description: "This video lesson is part of a complete course on neuroscience time series analyses.The full course includes   - over 47 hours of video instruction  - lots a..."
url: "https://youtu.be/9Hk-RAIzOaw"
favicon: ""
aspectRatio: "56.25"
```
Also here's the [Parameters of Morlet wavelet](https://youtu.be/LMqTM7EYlqY?list=PLn0OLiymPak2BYu--bR0ADNBJsC4kuRWs&t=308)


### Hilbert transform
What is [Hilbert transform](https://youtu.be/NMR7PR7M4Iw)?
[What is a Hilbert Space?](https://youtu.be/FFPXm-tuOt8?t=153)

The Hilbert transform can give you a clean separation between envelope and oscillation. This can only be done if the signal has slow changes riding on fast oscillations.
If you have a signal where something is changing slowly while something else is wiggling quickly, and these two aspects don't overlap in their frequency content, then the Hilbert transform can pull them apart accurately.

This is applicable to our application because we don't care about the raw waveform, only the larger underlying envelope. As you can see, the the "carrier", fast oscillation within the envelope, are visible
[![|490x215](https://imgur.com/AUfZXGh.jpg) 
this image shows the resulting envelope
[![|225x141](https://imgur.com/Tm2fXqo.jpg)


### What is a Spectrogram?
```embed
title: "The short-time Fourier transform (STFFT)"
image: "https://i.ytimg.com/vi/T9x2rvdhaIE/maxresdefault.jpg"
description: "This video lesson is part of a complete course on neuroscience time series analyses.The full course includes   - over 47 hours of video instruction  - lots a..."
url: "https://youtu.be/T9x2rvdhaIE"
favicon: ""
aspectRatio: "56.25"
```


```embed
title: "Comparing wavelet, filter-Hilbert, and STFFT"
image: "https://i.ytimg.com/vi/6x3iFs_j5j8/maxresdefault.jpg"
description: "This video lesson is part of a complete course on neuroscience time series analyses.The full course includes   - over 47 hours of video instruction  - lots a..."
url: "https://youtu.be/6x3iFs_j5j8"
favicon: ""
aspectRatio: "56.25"
```




## Find a way to help the algorithm identify S1 and S2
[Transfer Learning in Heart Sound Classification using Mel Spectrogram](https://cinc.org/archives/2022/pdf/CinC2022-046.pdf)
> [!think]
> Right now, the algorithms confidence is artificially boosted
> If we look at the input data the algorithm is receiving. There are many instances where logically there should be no way the algorithm can determine a answer. 
> But right now it does
> I've done this intentionally as a band aid fix. I wanted more labeling but I must acknowledge that many peaks are being labeled correctly because the algorithm is simply making a lucky guess

### Multiple bandpass filters to isolate the different frequency ranges of S1 and S2
> [!say]
> I wonder, is it possible to record a spectral fingerprint for S1 and S2 heart sounds?
> Like, each recording has a different audio characteristic
> But within the file, every S1 should sound similar
> And be distinct from S2
> If this is true, there must be a way to exploit this

> [!think] Multiple frequency bands
> what if we preprocess using multiple frequency bands for the explicit propose of comparing the confidence that a peak is S1 vs S2 across the two frequency bands.
> the idea is, we generate a profile of what we think S1 sound, sounds like and when we encounter something that sound similar we give it a confidence score. then we pre process the audio again with a different frequency band and generate a new confidence score. Then we compare what we did vs how it changed the confidence score to generate a final confidence. 
> S1 is typically found between 20–80 Hz
> S2 is typically found between 60–200 Hz
> 
> if we do 60–200 Hz refilter and run analysis, if confidence for S2 increases, maybe we should increase confidence again since it aligns with what we expect?
- [x] Implemented

S1 tends to have more energy in the lower frequency band, S2 in the higher frequency band.
We calculate the ratio between  these energies and adjust pairing confidence accordingly. If the first peak is more S1-like and the second more S2-like, boost confidence; if the pattern is reversed, penalize.

We use a Gaussian‑weighted sum of the band energy to get the "energy of the whole beat". 100ms is average duration of the S1 heart sound, the width of the beat's Hilbert envelope. We make the gaussian this width and place it at the detected peak. Then use it to mask out the section of time we are interested in. 


#### Does frequency separation exists?
> [!think]
> Each beat, S1 and S2 should have a distinct sound that gives it its unique "profile". S2's higher frequency components (up to 250 Hz) are different S1's lower frequencies. I think there should be a way to exploit this. I need to confirm whether or not this frequency separation exists in my dataset before I try to isolate it. 

After implementing multiple frequency bands, it appears that there's minimal or inconsistent frequency separation in my dataset. I'm not sure what do do about this.

going back to the idea of a spectral fingerprint, what if we can be more flexible than a preconfigured band for S1 and S2? What if we can look at already labeled peaks of high confidence and determine what characteristics they share, then build our band around that? This is another iterative idea though. 


#### Fourier transform Profile for S1 and S2
I'm plotting a frequency "profile" for S1 and S2 by taking the already labeled peaks and sampling around that peak with FFT to generate the frequency. Then adding up all the frequency values to average them up and plotting it. This gives me the pure frequency domain profile of my S1 and S2 beats. 

I want to determine if frequency separation exists in my dataset. If I see frequency separation, then it validates that S1 and S2 are discernable in the frequency domain. 

I cannot do this if the amplitudes of S1 and S2 differ because that would cause one to be constantly higher than the other. I tried to fix this by normalizing the data to make both graphs overlap but I'm having trouble doing this. 
in a ideal world, we would see the frequency of 
S1 be higher 0-100hz and lower at 100-500hz
and S2 to be lower at 0-100hz and higher at 100-500hz
then both graphs should converge to zero or whatever dB value after 500+hz

Right now, I observe limited separation within the 0-500hz range and they do not converge at the 500-22,000 hz range. this should be opposite of what I want to see. separation should be higher within the 0-500hz range and lower within the 500-22,000 hz range. 

The idea is, the FFT graph of S1 and S2 should always overlap except with in the 100-500hz range where S1 starts to separate and have a higher value. then within the 100-500hz range, S1 should also separate and be lower while S2 becomes higher. If we can figure out how to get our dataset to do this with the FFT, we can reverse our process and apply it back to the peaks labeling algorithm.

Since we are aggregating all labeled S1 and S2 peaks into one FFT profile. we should be doing the same for the amplitude to generate a S1 amplitude and S2 amplitude profile. this will allow us to normalize the data and remove all amplitude data from our FFT profile. 
- [x] implemented


I think we should only select up to 100 labeled peaks, for each type, depending on how confident the pairing confidence was when the algoritm did the pairing.
- [x] implemented

I want to be able to aggregate the spectral profile data across multiple input files to get a more general idea of what distinguishes S1 vs S2. 
- [x] implemented 











### I still need to research:
- [mel-spectrogram](https://www.youtube.com/watch?v=9GHCiiDLHQ4&t=280s)
- Wavelet Scalogram



> [!think]
> I want to generate a profile over time:
> so we average the features of the past 5 labeled S1 peaks and build a understanding of what S1 should look like in the context of this recording. 
> 
> what if we further this idea by sampling at random indices, we take 10 random marked S1 peaks after labeling is done and re-run the labeling using this "trained model"
> like we are training a model in real time to label, then label to train the model, recursively. 
> 
> for the features, I want to sample the entire beat, not just the time at the peak. this means finding the bounds of the envelope and sampling the data within that time frame maybe with increasing significance towards the center?
> like we use a gaussian to mask the... or maybe not, do the edges of the beat contain important data? if not we should mask it out... but what if it does?









# Explanation of PCG Preprocessing
## Generating our audio envelope:
### How was the Audio Envelope calculated?
Previously, the Audio Envelope is abs(filtered audio) then a centered 100ms rolling mean at 500 Hz. Peak detection currently returns the top of that envelope
#### Trapezoidal waveforms causes the middle of the sound to not be the peak amplitude
I've noticed from the peak detection algorithm, It places the peak at the highest point, but sometimes the audio envelope of a wave might look like a cut mountain and sometimes slanted. we basically need to capture the middle of the peak of a trapezoidal wave. 
[![|508x155](https://i.imgur.com/xkhzAQS.png)]
From this example, we can see how the placement of the detected peak is not center with the mass of the waveform resulting in a very small inaccuracy.

![|398x161](Y1rGxdpPAq4.png)

**Why this is a issue:**
HRV is calculated from R-R intervals. Since HRV is so sensitive (standard deviation of ~10-50ms) we cannot afford any noise in the HRV calculation. The HRV function expects accurate data so if we feed it incorrect peak timings the the HRV calculated will be wrong. 

> [!think]
> Maybe we can solve this in a simple way:
> we just take the integral of the waveform between two troughs and place the peak such that it divides the area under the curve in half. This should result in a slightly more accurate placement of the labels.


> [!say]
> The algorithm never shifts the detected peaks after they have been detected. If the detected location is not center of mass for the volume of sound around it, then it's not at the center of the "beat"
> right now the peaks detector only labels peaks at the peak of the envelope. This is fine but after we label the peaks we should go back and adjust the location slightly if needed.
> how can this be done? we can use a gaussian to mask around that peak and integrate the amplitude around that peak. then find the center of mass for that beat and shift the peak to that location. this can be done as a post processing step. or maybe a pre processing step? I'm not sure
> find_peaks returns the same local maxima for the detected peaks position. then we find center of mass after integrating around ~100ms of the local maxima. that becomes the location of the detected peak. 
> so it kinda makes sense to do this before we label peaks step
> 
> I think HRV would improve from this actually. sometimes there's a split S1 and the peaks are placed at M1 and sometimes at T1, our idea will average them out giving CoM at the center for a more consistent HRV
> we should use a Super-Gaussian (with flat top), a mix between full rectangle and fully smooth gradient. 
- [x] implemented

> [!think]
> What if the peak we are shifting is a bit of noise and not S1 or S2?, what if it's really noisy and it causes the noise peak to wander? would this be a good thing? 



### Hilbert Envelope:
[![|1645|1027x175](https://imgur.com/69ibaD7.jpg)

#### Issues with the Hilbert Envelope
- Linear dynamic range results in loss of weak S2 amplitudes at high BPM. "At 200 bpm, S1 is very loud and prominent... S2 becomes very soft or even inaudible"
- heart sounds are transient, not sinusoidal
- Hilbert envelope might show artificial peaks between S1 and S2 due to Spectral overlap (S1 tail and S2 onset interfere)




### Homomorphic envelope
Homomorphic filtering is a nonlinear signal processing technique that separates multiplicative components in a signal by transforming them into additive components via logarithms. For phonocardiography, this is powerful because heart sounds are approximately the product of:
- Slowly varying amplitude modulation (the envelope)
- Rapid oscillations (the "carrier" valve vibration frequencies)

Homomorphic filtering explicitly separates the envelope from the carrier. 

> [!think]
> Perhaps we can use **Hilbert** for timing precision and **homomorphic** for amplitude comparison.

After implementing Homomorphic envelope, it doesn't really make a improvement... I think I'll just use Hilbert



# Explanation of algorithm logic decision making
## Temporal Features

### Brainstorming:
> [!think]
> the initial bpm I input is not being respected enough.
> `s1_s2_max_interval_sec` is being updated too frequently and with too much sensitivity. `s1_s2_max_interval_sec` should be determined by the bpm, but shouldn't there be two types of bpm being calculated here?
> a instantaneous bpm, and a more averaged out, long term bpm.
> we can use this long term bpm to determine what the expected bpm should be.
> 
> If I input a suggested bpm of 120, then the script calculates the interval between beats to be at 240bpm. then we can obviously conclude that it counted a S2 as a lone S1, it missed the pairing. So we can also use this discrepancy to work with the normal deviation we calculated in the other step, to definitively conclude that the beat is supposed to be S2.
> obviously my starting bpm suggestion should be value for the long term bpm.
> 
the long term bpm should change slowly and within physiological limits. limited by the `max_bpm` the slope 
I input 90bpm as the start, then the `Long-Term BPM` goes from 90 to 176bpm in 18seconds (86bpm/18s)
since everything is based on the `Long-Term BPM`, we need to make sure it's accurate but not reactive.

> [!think] here's my interpretation of the issue
> - **Hinted 90 BPM**: The `s1_s2_max_interval_sec` is set to approximately 0.33 seconds. The logic assumes any two peaks closer than this are an S1-S2 pair within a _single_ heartbeat.
> - **Actual 170 BPM**: The true interval between _separate_ heartbeats is 60/170≈0.35 seconds.
> - **Actual 181 BPM**: The true interval between _separate_ heartbeats is 60/181≈0.33 seconds.
> 
> when the actual interval between two distinct heartbeats ($0.35$s or $0.33$s) becomes equal to or shorter than the set `s1_s2_max_interval_sec` (0.33s), the algorithm misinterprets them. It sees two separate heartbeats, but because their time difference falls within the "S1-S2" window, it incorrectly considers them as components of a single heartbeat and likely discards one of the peaks. This results in an undercounting of the actual heartbeats and an artificially lower BPM calculation.
> #### Proposed Solution:
> we can use this long term bpm to determine what the expected bpm should be. that way, if the script calculates a bpm that's completely off, it should correct itself.
> for example, If I input a suggested bpm of 120, then the script calculates the interval between beats to be at 240bpm. then we can obviously conclude that it counted a S2 as a lone S1, it missed the pairing. So we can also use this discrepancy to work with the normal deviation we calculated in the other step, to definitively conclude that the beat is supposed to be S2.
> obviously my starting bpm suggestion should be value for the long term bpm.

### Long-Term BPM vs Instantaneous BPM
**Problem:** A mislabeled beat could drastically shrink s1_s2_max_interval, causing all subsequent beats to be misclassified.
**Solution:** Maintain two BPM values:
- `long_term_bpm`: Slowly adapting belief (0.05 learning rate) that stabilizes the S1-S2 window
- `instant_bpm`: Raw calculation from last interval, used only to update the belief
**Why this works:** Allows the algorithm to self-correct. If instant BPM spikes to 240 but long-term is 120, we know we double-counted S2 and can trigger corrective logic.

### Dynamic S1-S2 Pairing Window
**Problem:** At 90 BPM, true S1-S1 interval is 0.67s. But if s1_s2_max_interval is 0.33s, then at 170 BPM (true interval 0.35s), the algorithm merges separate beats.
**Solution:** The maximum allowed S1-S2 interval is derived from the stable long-term BPM belief rather than the last instantaneous interval.
**Physiological basis:** S1-S2 interval is ~35-50% of total R-R interval and adapts with heart rate.


### We add a S1-S2 pairing penalty for the timing of the S1-S2 interval
If the current peak is S1, we take a look at the current heart rate and determine the distance S2 should be. If the next peak is too far or too close to the current peak, we apply a penalty to the pairing confidence. 
The pairing logic uses a V-shaped penalty based on how far the observed S1–S2 interval deviates from the expected interval. 

The center of this V is where we expect S2 to be, so no penalty. The further away we get from the expected interval, the higher the penalty we apply. There's a bit of leeway before the ramp starts applying the penalty. It's just a linear function right now. 

Also, if the timing deviates too much from expected, we hard reject the pairing and move onto Lone S1 decision. 
#### How is the expected interval calculated from BPM?
using the Weissler regression (ET vs HR slope of approximately -1.0 to -1.7 ms per bpm increase)
for example, At 135 bpm: 300 - (75 × 1.0) ≈ 225 ms 
#### Math for nerds:
$I_{obs}$ = observed S1–S2 interval in ms
$I_{exp}$ = expected S1–S2 interval in ms
$B_{max}$ = max boost
$P_{max}$ = max penalty
$a$ = interval_zero_crossing_fraction
$a_{LongEnd}$ = interval_v_long_ramp_end_fraction
$a_{ShortEnd}$ = interval_v_short_ramp_end_fraction

The math for calculating the Δboost should be:
$(L_{2}, -P_{max})$
$(L_{1}, 0)$
$(I_{exp}, B_{max})$
$(R_{1}, 0)$
$(R_{2}, -P_{max})$

where
$L_{2}=I_{exp}\cdot a_{ShortEnd}$
$L_{1}=I_{exp}\left(1-a\right)$
$R_{1}=I_{exp}\left(1+a\right)$
$R_{2}=I_{exp}\cdot a_{LongEnd}$

Then we draw a line connecting these points and that forms our function $f(I_{obs})$
```embed
title: "V-Shaped Interval"
image: "https://www.desmos.com/calc_thumbs/production/version/t4k6ecobne/915e5a30-1a7d-11f1-a974-81ba6c8c6f9b.png"
description: ""
url: "https://www.desmos.com/calculator/t4k6ecobne"
favicon: ""
aspectRatio: "100"
```


> [!think] 📌
> so at runtime, using the belief BPM to calculate intervals is a bit flawed. belief determines intervals and intervals determine belief. this type of sequential decision making is too greedy. 
> If I ever update the iterative code/ second pass correction etc. I should pass the calculated Average BPM from the first pass and remove outliers, then make a line of best fit and pass that new assumed BPM into this function to make the algorithm behave more holistically
- [ ] implemented

> [!think]
> Let's only use weissler, remove the other calculation 
- [ ] implemented

> [!think]
> we should make this a function of past S1-S2 pairs instead of BPM. I want to test how that would look. Maybe average the past 10 S1-S2 pairs to get a expected value. 
- [x] implemented








## As BPM increases, the S1-S2 prominence differential increases
> [!ask]
> At higher bpm, the loudness of S1 is much louder than S2. why is this?
> I'm looking at the waveform of a heartbeat recording and I can't see S2 at all. at 200 bpm, S1 Is very loud and prominent. 

**at extremely high heart rates (like 200 bpm), diastole shortens dramatically more than systole, leading to a much softer S2 and making S1 appear disproportionately loud.**
1. **Dramatic Shortening of Diastole:**
    - The cardiac cycle has two main phases: **Systole** (ventricular contraction/ejection) and **Diastole** (ventricular relaxation/filling).
    - As heart rate increases, both phases shorten, but diastole shortens proportionally much more than systole.
    - **At 200 bpm:**
        - Cycle length = 60 seconds / 200 beats = **0.3 seconds (300 ms) per beat.**
        - **Systole duration** is relatively fixed by contraction/ejection mechanics. Even at high rates, it might be ~150-200 ms.
        - **Diastole duration** = Cycle length - Systole duration = 300 ms - 150-200 ms ≈ 100-150 ms (or less). Diastole becomes extremely brief.
		Studies of maximal exercise show that systole can occupy 50-60% of the cycle at extreme rates. 
2. **Impact on S2 Loudness (Aortic & Pulmonary Valve Closure):**
    - S2 is produced by the closure of the semilunar valves (aortic and pulmonary) at the end of systole.
    - The loudness of S2 depends significantly on the pressure gradient across these valves at the moment of closure.
    - **Very Short Diastole = Reduced Ventricular Filling (Preload),** With diastole compressed to 100-150 ms, the ventricles have drastically less time to fill with blood before the next contraction.
    - **Reduced Filling = Reduced Stroke Volume,** The amount of blood ejected in the next systole (stroke volume) is much lower.
    - **Lower Stroke Volume = Lower Aortic/Pulmonary Pressure & Weaker Valve Closure:**
        - Less blood is ejected into the arteries.
        - Consequently, the pressure difference forcing the aortic and pulmonary valves shut at the end of systole is much smaller.
        - **Result:** The semilunar valves close much more gently and quietly. **S2 becomes very soft or even inaudible.**
		At 200 bpm, because the heart doesn't have time to fill (low preload), it pumps out very little blood (low stroke volume).
		And because the Stroke Volume is tiny, the system pressure generated is low. S2 is caused by the blood in the aorta snapping back against the valve. If there is less pressure pushing back, the "snap" is weak.
		Also the momentum of the blood, the pressure wave generated by the LV, probably becomes more continuous rather than a pulsing force. the discrete packets of blood excreted by the heart blend together into one continuous flow to some extent. 
3. **Impact on S1 Loudness (Mitral & Tricuspid Valve Closure):**
    - S1 is produced by the closure of the atrioventricular valves (mitral and tricuspid) at the beginning of systole.
    - Its loudness depends on:
        - The **position and tautness of the valve leaflets** at the onset of ventricular contraction.
        - The **force and speed of ventricular contraction** slamming the valves shut.
        - The **pressure gradient** across the closed valves immediately after closure.
    - **High Heart Rate & Sympathetic Tone:** Tachycardia at 200 bpm is almost always driven by high sympathetic nervous system activity (stress, exercise, arrhythmia). This **increases myocardial contractility**.
    - **Forceful Contraction:** Despite reduced filling, the sympathetic drive makes the ventricular contraction **more forceful and rapid**.
    - **Result:** The AV valves are slammed shut vigorously by this forceful contraction. **S1 remains loud, or may even become louder** due to the increased contractility.
    *It's important to note:* 
	- The reduced preload (from ultra-short diastole) results in a lower amplitude sound which should offset the increase in amplitude caused by contractility, but my dataset shows a positive correlation with audio amplitude and BPM. 
4. **Proximity of S2 to the Next S1:**
    - With systole taking ~150-200 ms and the total cycle only 300 ms, S2 occurs very close to the next S1.
    - A soft S2 happening just 100-150 ms before a very loud S1 is easily masked by the louder sound. Your ear (and the waveform) simply can't resolve them as separate events; the loud S1 dominates and overshadows the faint S2.
    - **Primary issue is temporal, not intensity**: At 200 bpm, S2's main problem is being **temporally buried** by the adjacent loud S1, not becoming too soft to exist. The acoustic energy of S1 overwhelms the system before S2's waveform can fully resolve.

**Why S2 gets less prominent at higher heart rates:**
-  **Physiologically Soft:** As explained above, S2 is genuinely much quieter due to reduced valve closure force.
- **Masking by S1:** The loud S1 occurring shortly afterward dominates the acoustic energy and visually overshadows the small S2 deflection on the waveform.
- **Temporal Summation:** The sounds are so close together that their waveforms start to overlap, making S2 indistinguishable from the tail end of S1 or the beginning of the next S1.
- **Equipment Limitations:** Recording equipment (microphones, amplifiers, filters) and display settings (gain, time scale) might not be optimized to capture and resolve such a faint, high-frequency sound occurring so close to a much louder one at an extremely rapid rate. S2 often has higher frequency components than S1, which can be harder to record faithfully.

it's also important to note:
- **Normal Heart Rate:** During long diastole, the ventricles fill, pressures equalize, and the mitral valve leaflets float upward, effectively "pre-closing" before the ventricle contracts. The valve only has to close a tiny distance.
- **200 BPM (Short Diastole):** The ventricle is still actively trying to fill when the next contraction starts. The mitral valve leaflets are still fully open due to the rushing blood flow.
- **The Result:** When the ventricle contracts, the valve leaflets have to travel a large distance to slam shut.
This may also be a contributor for why S1 waveform amplitude increases with BPM.


### Contractility Power Curve
```embed
title: "Heart Contractility Power curve"
image: "https://www.desmos.com/calc_thumbs/production/version/qll0xbsiuk/65328f50-19ea-11f1-bd8a-c516bba7813a.png"
description: ""
url: "https://www.desmos.com/calculator/qll0xbsiuk"
favicon: ""
aspectRatio: "100"
```


For post Post-Exercise contractility, we can increase the Exponent in this function, decaying over time. 







## Post-Exercise, S1 amplitude remains elevated despite BPM decreasing
> [!ask]
> I'm listening to a audio file and I notice something interesting. This is a recording of the heart during a period exercise and recovery. At rest, the volume of S1 and S2 are very similar, but at higher bpm, S1 is significantly louder than S2. After the workout ends and heart rate decreases but it seems like the contractile force of the heart sill causes S1 to be significantly louder than S2. 
> 
> Then If I compare the audio from 90bpm post workout to 90 bpm before workout, the amplitude difference between S1 and S2 is larger for the heart post workout.
> 
> Is this expected physiology (is this patient's heart behaving as expected) or a byproduct of my recording device? 

Your observation is the **expected physiology**. This may be called "Inotropic-Chronotropic Dissociation", the heart's rate (speed) and the heart's contractility (force) do not return to baseline at the same time.
The persistence of a relatively loud S1 during early recovery at 90 BPM (compared to pre-exercise 90 BPM) is a real phenomenon caused by **residual sympathetic activation**. This is well-documented in phonocardiography studies.
### Key Mechanisms Explaining Post-Exercise S1 Dominance
#### 1. Sympathetic Tone Persists Beyond Heart Rate Normalization
This is the central mechanism. Beta-adrenergic stimulation affects S1 amplitude more than heart rate:
- **Duration mismatch**: Heart rate recovery begins within 30-60 seconds, but sympathetic neurotransmitter (norepinephrine) clearance and receptor downregulation take **5-10 minutes**
- Contractility remains elevated while Heart Rate is largely controlled by nerve signals (electrical). It can change instantly.
- During recovery, contractility remains augmented while rate falls, creating your observed mismatch
#### 2. The "Cool Down" Effect (Systemic Vascular Resistance)
**Vasodilation diminishes S2:**
- **During/Post-Exercise:** Arteries dilate to supply muscles and release heat. This lowers the resistance in the aorta (Afterload).
- **Impact on S2:** If the arteries are dilated and pressure is lower (which is common immediately post-exercise), the "snap" of the aortic valve is softer.
#### 3. AV Valve Position Remains Altered
From [Merck Manual](http://ncbi.nlm.nih.gov/books/NBK333/) and [Healio](https://www.healio.com/cardiology/learn-the-heart/cardiology-review/topic-reviews/heart-sounds):
- **Shorter effective PR interval**: Elevated sympathetic tone accelerates AV conduction, keeping mitral/tricuspid leaflets **wider apart** at end-diastole
- **Rapid valve closure**: The leaflets close from a greater distance with more force, increasing S1 intensity
- **Preload effects**: Post-exercise, venous return may still be elevated, affecting diastolic filling dynamics
#### 4. S2 Intensity Fails to "Catch Up"
While S1 remains augmented, S2 normalization lags because:
- **Aortic/pulmonary pressure gradients** recover more slowly as stroke volume remains elevated
- **Residual vasodilation** in skeletal muscle beds affects systemic vascular resistance
- **Splitting patterns** remain abnormal during early recovery

#### Post-Exercise, S1 amplitude remains elevated despite BPM decreasing
**Physiology:** Sympathetic tone persists after HR normalizes → contractility remains elevated → S1 stays loud.
**How it works:** When the algorithm detects the recording is within the recovery window after peak exertion, it relaxes confidence penalties on S1-S2 pairs. This prevents faint, re-emerging S2 peaks from being rejected during early recovery.

#### S2 Re-Emergence After Dropout
**Physiology:** After S2 disappears at high BPM, it reappears faintly as HR drops. Initial faint peaks are easy to miss.
**How it works:**
- If the recent beat history shows the pattern **S1 (Lone) → Noise** repeated several times, it means S2 is trying to reappear but failing confidence thresholds.
- The algorithm temporarily relaxes its pairing threshold to give these faint S2 peaks a chance to anchor the rhythm again.
- This is the "kick-start" mechanism to escape a "Lone S1 only" rut.

#### Contractility vs. Rate Mismatch (Post-Exercise 90 BPM)
**Physiology:** At 90 BPM post-exercise, contractility is still high (loud S1). At 90 BPM pre-exercise, contractility is normal (balanced S1/S2).
**How it works:** The algorithm's pairing expectations are anchored to a slowly-decaying long-term BPM belief rather than instantaneous rate. After peak exertion, the belief decays slowly, so even as measured BPM drops, the algorithm's contractility expectations remain elevated for a period. This correctly models the physiological lag between heart rate and sympathetic tone.
**Example timeline:**
1. Peak exertion at 170 BPM → long-term belief = 170
2. 1 min later: Instant BPM = 90, but long-term belief ≈ 120 (slow decay)
3. Algorithm still in "high contractility" mode → expects S1 > S2 → correctly processes post-exercise audio

#### S1 Amplitude Floor (Preventing Noise as S1)
**Physiology:** Even at high BPM, S1 must exceed a minimum amplitude to be a real heartbeat.
**How it works:**
- The algorithm tracks recent S1 peak amplitudes and establishes a reference from the top percentile.
- A candidate S1 whose amplitude falls too far below this reference has its confidence reduced proportionally.
- This prevents noise spikes from being accepted as valid heartbeats even when their timing is plausible.


## S2 disappears and reappears in some recordings
**Problem:**
In some recordings, when bpm increases, S1 becomes so much louder that S2 is not longer audible in the recording at all. 
When bpm decreases again post exertion, S2 becomes visible again

The mechanism behind this is unclear, but it's a reoccurring trend in my dataset. 
It must be the recording mythology used to capture the data or the position of the stethoscope during exercise etc. 

right now, the pairing history shows 0% pairs made after bpm starts to decrease which is true, but this also causes the current stability adjust logic to tank the confidence making it unlikely to pair S1-S2 afterwards.
Kick-Start Recovery Mechanism feels like a band aid fix to this issue. A more robust solution should be used instead

### Kick-Start Recovery Mechanism
**Problem:** When paring ratio reaches, 0 it's very difficult to begin paring again since lack of pairing decreases pairing ratio. This is a negative feedback loop. 
During recovery, S2 disappears. The algorithm enters a "Lone S1 only" mode and can't exit even when S2 reappears.
**Solution:** Scan last 4 beats. If pattern is S1→Noise repeated 3+ times, temporarily boost pairing ratio to 0.60.
**Why this works:** S2 re-emerges as faint peaks that fail normal confidence thresholds. Kick-start gives them a chance to anchor the rhythm again.




## Explanation of prominence calculation
### Peak Prominence Calculation
**Purpose:** Prominence measures how much a peak stands above its local background, making it robust against global volume changes and temporary noise.
Initially, I only compared peaks by their amplitudes but background noise (the noise floor) needed to be accounted for. 
When noise inflates trough heights, prominence will decrease.

**How it's calculated:**
1. **Find adjacent troughs:** For a given peak at index `i`, locate the nearest trough to the left (`i-1`) and right (`i+1`) using the pre-computed `trough_indices`.
2. **Determine key col:** The key col is the **higher** of the two trough amplitudes (the "shoulder" of the peak).
3. **Calculate prominence:** `prominence = peak_amplitude - key_col_amplitude`

### Trough Sanitization
**Problem:** Temporary noise creates high troughs that inflate the noise floor, reducing prominence of valid peaks that occur after noise ends.
*Temporary noise deviates from the noise floor:*
[![|756x182](https://i.imgur.com/iDTD90X.jpeg)
This specific scenario shown above also creates two "Trapezoid Artifacts" since S1 S2 labeling swaps twice resulting in the correct labeling after. I guess two wrongs do make a right...

**Solution:** During noise floor calculation, reject troughs that exceed 10x the draft floor value. This keeps loud transient noise from affecting the calculated prominence of our peaks during a brief moment of noise. 
**Limitation:** Does not handle noise that occurs between S1 and S2 peaks during the pairing phase.


### Important to note
It's important to note that prominence is calculated against the troughs adjacent to the current peak and not against the noise floor.

> [!think]
> Is it a good idea to have it implemented in this way? I find it a bit questionable but I have no proof that it causes any issues so I'll leave a developer's note here



## HRV calculation
**[Lomb–Scargle periodogram](https://archive.physionet.org/physiotools/lomb/lomb.html)** for HRV calculation to obtain frequency-domain data such as VLF, LF, HF, LF/HF ratio
- FFT assumes evenly spaced samples, R–R (beat-to-beat) data is inherently uneven. 
- (SDNN/RMSSD is time domain data)
- Caveats:
	- Shorter segments: frequency-domain measures become less reliable; minimum lengths in the literature are often ~4–5 minutes for stable VLF/LF/HF; HF can still be usable down to ~10 s; LF is more reliable with at least ~30–60 s.


## Trapezoid Artifacts
### Trapezoid Artifact Detection
**Problem:** A single noise peak causes S1/S2 swap, then another noise swap flips them back. BPM graph shows characteristic "notch."
**This is what It looks like:**
https://github.com/WolfExplode/bpm_analysis/blob/main/Examples/R18%E5%BF%83%E9%9F%B31_bpm_plot.html
![|716x195](https://imgur.com/sIfPq8w.jpg)
![|716x195](https://imgur.com/b7TKFmS.jpg)
**It is caused by these failure modes:**
trapezoid artifact #1
![|422x267](https://imgur.com/jRHjahH.jpg)
trapezoid artifact #2
![|422x246](https://imgur.com/jTFakMB.jpg)
In this case, this file contains two Trapezoid Artifacts. the first one was caused by some noise or PVC during a period where S1 and S2 prominence is similar.
This causes the labeling of S1 and S2 to switch until S1 and S2 prominence deviates enough for the algorithm to correct the mistake resulting in the second trapezoid artifact.
the second trapezoid artifact is caused by peaks labeled LoneS1 followed by S1.

**Solution:**
📌The solution of this issue is only half implemented. I only implemented a way to identify trapezoid artifacts but not a way to use them to make the script more robust

The human eye can easily identify errors in the BPM/time graph so I implemented a function to allow the script to identify them automatically.
Detects trapezoid-shaped discontinuities in the average BPM series that are characteristic of a brief extra-beat artifact:
  - A very fast rise
  - A sustained plateau
  - A very fast fall that returns to baseline


## Parameter Tuning Rationale
Parameters in `config.py` were hand-tuned across multiple PCG recordings from consumer hardware.
A known limitation: tightening a parameter to reduce errors on one file can increase errors on another.
This is partly unavoidable given how much recording conditions vary (microphone position, movement noise, BPM range).
If you find yourself re-tuning frequently, the underlying algorithm is probably not robust enough for that class of recording -- see the "Known Limitations" section.

### Noise Floor Parameters
- `trough_rejection_multiplier=4.0`: Reject troughs >4x draft floor. Calibrated to keep physiological troughs while rejecting movement artifacts.
- `noise_window_sec=4`: Rolling window for noise floor. Long enough to smooth out temporary noise, short enough to track gradual changes in background noise.
### Confidence Thresholds
- `pairing_confidence_threshold=0.55`: Empirically determined. Lower values increase false pairs; higher values miss faint S2s.
- `lone_s1_confidence_threshold=0.50`: Must be strong enough to avoid noise, but lenient enough to catch valid single beats when S2 is absent.
### Lookahead Parameters
- `noise_prominence_threshold=0.35`: Middle peak must be <35% of S1 prominence to be considered skippable. Prevents skipping valid S2s.
- `enable_lookahead_skipping=True`: Master switch because lookahead is aggressive. Can be disabled for clean recordings.



## Known Limitations & Edge Cases

### Sequential Decision Making
Currently, peak labeling is assigned locally and greedily as each peak is classified based on immediate context.
**The problem**: A local decision can force suboptimal future choices. If you label a weak peak as S1, you may miss a stronger S1-S2 pair just milliseconds later.
The majority of the issues in our algorithm come from sequential decision making. If only there was a way to give our code a more holistic view. 



 


**research:**
[Hidden Markov Models](https://youtu.be/RWkHJnFj5rY)
```embed
title: "Introduction to HMMs | Hidden Markov Models Part 1"
image: "https://i.ytimg.com/vi/ZIT2UH6bF38/maxresdefault.jpg"
description: "In this video, we break down Hidden Markov Models (HMMs) in machine learning with intuitive explanations and step-by-step examples. Starting from simple Mark..."
url: "https://youtu.be/ZIT2UH6bF38"
favicon: ""
aspectRatio: "56.25"
```

```embed
title: "Viterbi Algorithm"
image: "https://i.ytimg.com/vi/6JVqutwtzmo/maxresdefault.jpg?sqp=-oaymwEmCIAKENAF8quKqQMa8AEB-AG-B4AC0AWKAgwIABABGGUgZShlMA8=&rs=AOn4CLBxRNlVpBCC0Z53iOa7xpZRxrs2kw"
description: "Short description of the Viterbi Algorithm without equations using a trip planning example.Correction: Viterbi first published this in 1967, not 1968 as stat..."
url: "https://youtu.be/6JVqutwtzmo"
favicon: ""
aspectRatio: "56.25"
```










### Cold Start Problem
The cold start is a consequence of sequential decision making
**Issue:** First 4 seconds often misclassified because long_term_bpm hasn't stabilized. 
**Workaround:** Provide `start_bpm_hint` when possible. The `_kickstart_check` helps but isn't perfect.


**Example of Failure mode:**
![|968x178](https://imgur.com/6BK2wQk.jpg)
![|968x190](https://imgur.com/XEYt7r4.jpg)
In this case, the very first peak was mislabeled as Lone S1 while it was S2. Since this is the first peak of the recording, it's impossible to determine if it's S1 or S2 without scanning backwards and using future knowledge

> [!think]
> we could scan backwards to fix errors that occurred before.
> This is difficult to implement because our algorithm expects forward data. Reversing the logic is a non trivial task.

> [!think]
> What if we just loop the waveform back onto itself and run it continuously?
> Then there would be no start... we just process the start data and then discard it after running it again. Therefore there will be no initial labeling in the final output. 
> 
> But what if the start bpm is 60 and the end bpm is 180? What if the audio ends on S2 and starts on S2? this seems like a bad idea









### Noisy audio
There's no real way to solve this issue... just don't input noisy audio I guess

### Heartbeat audio with Arrhythmia/PVCs
**Issue:** This software isn't built to detect PVCs or Arrhythmia
**Future idea:** I plan to support this in the future but I haven't come up with a idea yet

### S1/S2 Swapping During Breathing
**Issue:** During inhalation, S1-S2 amplitude difference decreases. Can cause temporary S2→S1 misclassification, especially if timing aligns poorly.
**Current mitigation:** Contractility model helps but doesn't fully solve it. The `penalty_waiver` logic for ideal deviation range catches some cases.
**Future idea:** Track a rolling history of S1-S2 deviation. If deviation suddenly drops without BPM increase, suspect breathing inversion rather than true contractility change.






### Missing S2 in High BPM
**Issue:** Above ~180 BPM, S2 may be physically absent from waveform due to temporal merging with next S1.
**Current solution:** Algorithm gracefully degrades to Lone S1 mode when pairing fails consistently.
**Future:** Could add a "S2 dropout detection" that disables pairing entirely when consecutive Lone S1s exceed threshold at high BPM.



## Optimizations:
60% of the script's runtime is conversion time, which is fundamentally unavoidable because it's dominated by FFmpeg decoding the compressed MP4 audio stream
Attempting to optimize the remaining 40% of our script's runtime seems kinda silly in comparison...
**Decode time dominates because:**
- MP4 audio is compressed with complex psychoacoustic models (AAC=O(n²) decode)
- 18 minutes of 44.1kHz stereo AAC = ~120 MB → ~1.9 GB PCM
- FFmpeg must decode every frame sequentially




## Notes about the current state of my codebase
Let's do a **Codebase audit**

So far, our code has
**Technical debt**: 
If I can find a better solution these band aid fixes would not be here: (e.g., "kick-start" recovery mechanisms, cascade resets)





### Release
I've decided to package binaries in a `BPM_Analyzer.exe` for non technical users to run on windows computers
Simply run `pyinstaller BPM_Analyzer.spec` to repackage and update binaries



#### LLM prompts to maintain our codebase
> [!say]
> Let's do a codebase audit. Can you find areas of improvement to my codebase to make it more maintainable?
> 

> [!say]
> There are bits of logic that I implemented to the script that likely do not need to be there. I come up with a idea to solve a problem, then later I come up with a better idea but left the initial solution in the code. Therefore there may be redundancy in logic that's still being used in the call stack but from a practical standpoint the logic is redundant. 
> 
> Can you scan our codebase for such cases if they do exist

> [!say]
> Sometimes, a module name doesn't match its actual contents. Can you scan our codebase for such cases if they do exist

> [!say]
> can you scan the codebase for lingering circular dependancies etc.
> observe the current structure and detemine if it needs to be done the way it is. should we further refactor for a more maintanable structure etc...

> [!say]
> can you scan the codebase for dead/redundant code? any stale or outdated comments?

> [!say]
> We should be writing code/text that makes it easy for AI to decipher and replace. Previously, we had difficulty with, and replacing/editing code using AI. Replace all high-risk characters with their ASCII equivalents so the codebase would be all-ASCII in its source text (string literals + comments). double check areas of concern. 

> [!say]
> that was a large refactor, let's double check if all the decisions we made make sense in the larger context of what we are trying to do.


> [!say]
> I'm using the graphical display to debug and understand how my code works. I don't want anything for display only. display should always show what data the algorithm uses/processes. can you double check that all display data is direct data from the algorithm?





### Other remarks
The code has several issues that don't fundamentally change the bpm estimation:
- There might be recordings where the stethoscope is removed resulting in no heartbeats at all, do we have a case for this?
	- Well, if the user of this tool inputs garbage then they should get garbage out. Not my fault and not my problem to fix. User error. 


## Unimplemented/Incomplete Ideas:

> [!think]
> I mean, if you really just want a accurate enough bpm/time graph, we can just plot all R-R intervals, remove the outliers and plot a line of best fit

Before the second pass, we just remove outliers in the labeled dataset and draw a best fitting curve to the remaining data
Then run second pass


> [!think]
> Should use **Kalman filter**?

> [!think]
> Should we add a **deque** for sliding windows instead of `params['stability_history_window']` manual slicing? Would that simplify my code?

> [!think]
> I might add parameter presets for simple configuration to the end user. so maybe a config for noisy files, one for if the file has PVCs, arrhythmia, If the file was recorded from the aortic or pulmonary auscultation areas (S1 might be louder or quieter than other files). 


> [!think]
what if we have a interactive way for the user to correct the peak labeling output from the script. then re input the data back into the script. This will allow the user to work together with the algorithm to arrive at the correct answer

> [!think]
> I've noticed that the BPM Trend (Belief) seems to be shifted to the right compared to the calculated Average BPM graph
> is this a issue? there a better way to calculate BPM Trend (Belief) that doesn't result in this happening?














### The S1-S2 interval changes post exertion
I'm listening to another audio file and I make another observation. This is a recording the heart during a period exertion and recovery

The S1-S2 interval is the time between S1 and S2, Systolic Time Interval (STI). This should logically decrease as heart rate increases. 
From what we know about Heart Rate Kinetics, contractility remains elevated post exertion despite a drop in heart rate. We previously spoke about this but we never considered the impact of exertion on the temporal aspect, the  S1-S2 interval. 

I've noticed that after exercise, the S1-S2 interval rapidly decreases and stays low. The S1-S2 interval is much lower for the same bpm than compared to before exertion. 
For example, before exertion, the S1_S2_interval was ~0.195s but after exertion, the S1_S2_interval decreased to between 0.145s and 0.160s. 

The **S1-S2 interval** represents **systolic ejection time**, the period from mitral/tricuspid valve closure (S1) to aortic/pulmonary valve closure (S2). This interval is determined by:
1. **Contractility** (how forcefully the ventricle contracts)
2. **Afterload** (pressure the ventricle pumps against)
3. **Preload** (ventricular filling volume)
4. **Heart rate** (via force-frequency relationship)

**This is caused by both increased cardiac contractility and reduced afterload.**
Residual adrenaline increases the rate of calcium cycling within the cells which causes the muscle to contract faster. This "lag" exists because the electrical pacemaker (SA Node) recovers faster than the chemical removal of adrenaline from the blood.
Post-exercise vasodilation in skeletal muscle beds decreases systemic vascular resistance. This reduces afterload (decreased aortic pressure during ejection). This allows for a faster, shorter ejection. 

In clinical cardiology, this specific measurement (shortening of the QS2 interval or LVET) is sometimes used as a sensitive marker for "adrenergic drive." QS2 Index is often used in stress testing for this purpose. 

> [!think]
> With precise timing data, we can calculate the S1-S2 / Cycle Length ratio (the Duty Cycle of the heart). This would show what percentage of time the heart spends "working" vs "resting" at that specific post-workout moment.

> [!think]
> I want to make our software output the heartbeat labels and timestamps so I can further gather evidence to see if this physiology exists in my dataset. I will make a script that can import the bpm/time graph. Then the S1-S2-S1 interval data and make a correlation. 






### Trapezoid Artifacts
I was considering this but I can no longer find a example of the issue this is describing, This might be because I directly or indirectly solved the issue via other methods. I will document the idea here regardless
> [!say]
> The script kinda flips between S1 and S2 when this event occurs:
> when inhaling, the amplitude difference between S1 and S2 decrease, at this moment, there can be a chance that S2 becomes louder than S1. If the timing is just right, the script can start to mark S2 as S1. 
> 
> two conditions should be fulfilled:
> - A trapezoidal discontinuity around that moment. or the bpm just suddenly increases a lot 
> 	- maybe we should have a secondary tracker for BPM increase? 
> 		I've been very conservative with the rhythm gaiting. basically, I made the script detect if the bpm is "impossible" then tank the confidence. but what if we make the script track how off the R-R interval is from the expected interval and then boos the confidence that S1 and S2 has switched places? This should probably be done in the correction pass
> - the amplitudes between S1 and S2 get closer together. so how do we detect this?
> 	- We should store a history of highest S1 peaks and lowest S2 peaks and then get a idea of the maximum deviation between them. then we can do the same with the lowest...
> 
> We need to calculate a history of the lowest 25% quartile of S2 peaks. so we track the past 50 S2 getting the lowest 25% quartile
- [ ] Implemented












### Split S2 
Split S2 visible in the Hilbert transform waveform envelope:
![|1018x182](https://imgur.com/dm8zzsY.jpg)

There should be a way to detect split S2. I'm thinking, After S1 and S2 peaks are labeled, we can go back and find more peaks.
Around the timeframe where S2 peaks are labeled, we should disable or drasticly reduce the min_peak_distance_sec and rescan. Maybe we can pick up the two split peaks.
I think the rescan for new peaks should be at the very end, after the post processing peak labeling stuff etc. because it depends on the labelings for the S2 peaks.

> [!think]
> If we can find a way to identify areas of Split S2, we might be able to calculate RSA from this data.

I'm not sure how to store split S2 in the code. I plan on doing respiratory rate Analysis with split S2 data.
In some files with low noise and a prominent S2 peak(s), it's easy to identify a split S2 but in some audios, split S2 may not be so easy to identify. In those cases, we will pick up many peaks around the S2 area. what shall we do? 











### Breathing:
#### How Heavy Breathing Modifies Heart Sounds
**During Inspiration:**
- Increased venous return causes right ventricular preload to increase
- RV stroke volume increases causes pulmonary flow to increase
- S2 splitting widens (P2 delayed)
- S1 may soften slightly as increased lung volume dampens transmission
	- [ ] Is this observed in our dataset? 
**During Expiration:**
- Decreased venous return results in a drop in RV preload
- S2 becomes single (A2-P2 fuse)
- Heart shifts position to be closer to chest wall, potentially amplifying sounds
	- [ ] is this observed in our dataset? 


Inhale/exhale visible from S1 amplitude fluctuations:
![|861x271](https://imgur.com/ibFbbpx.jpg)
But only sometimes...
It's rare, but respiratory rate can be calculated at some times, it's difficult though, because a person might stop breathing, hold their breath momentarily, or just some random noise in the waveform.
This is a difficult challenge because sometimes the data is extremely clear and easy to parse. other times it's non existent. 


#### Respiratory Sinus Arrhythmia (RSA)
![|487x488](https://imgur.com/83CAXFZ.jpg)
Heart rate increases during inspiration
Heart rate decreases during expiration
- [ ] Is this observed in our dataset? 

Breathing, Lung sounds (bronchial airflow) should create high-frequency noise (100-1000 Hz)
- [ ] Is this observed in our dataset? 





#### unimplemented
I want a way to identify breathing from the audio

graph the temporary noise by taking the troughs and comparing to the noise floor. then graphing that value
the idea is, temporary noise might be from breathing so by doing this, we might be able to visualize breathing. 
What if we do a different EQ filter on the initial audio to scan for breathing sounds instead?


Normal breathing causes physiological splitting of S2, can we use this to our advantage?
first, we need to be able to determine if our dataset contains this phenomenon 

#### unimplemented
Let's brainstorm based on some patterns that we can observe from the data. Take a look at how S1 is substantially louder than S2, but then there is some temporary noise that's spread out. When the noise increases, S1 amplitude decreases. 
we can also observe the inverse for S2. when noise increases S2 increases as well. At first I thought this was because the amplitude of S2 was "riding on top of" the noise but since S1 was decreasing this doesn't make sense.

Then I realized the noise must be from breathing since it's gradual fade in and out. 
Let's try to understand the heart's function and how the audio is being recorded. S1 must be decreasing since the stethoscope is further away from the chest at that time. but then why does S2 increase instead? the `trough-S2-trough` distance increases during breathing?
Is this normal physiology?

**Positional Shift:** As the diaphragm descends, the heart itself shifts slightly in the chest cavity. It can rotate or move downwards, changing its orientation relative to the fixed position of the stethoscope.
The heart's rotation might simultaneously move the pulmonic valve closer or into a better acoustic alignment with the stethoscope.

[![|1069x171](https://i.imgur.com/IFUuykq.png)]
By looking at the noise floor and toughs, we can clearly see the affect of breathing in the waveform:
[![|1667x96](https://i.imgur.com/HuYGxhi.png)
> [!think]
> how can we apply these observations to make our script more robust?



### Modeling Heart contractility
Regarding The idea of contractile force, what if we could display how hard the heart contracts by examining the amplitude deviation between S1 and S2 and mapping a trend.

Nah but that depends on how the audio is recorded. just because S1 is louder than S2 in this recording, doesn't specifically mean that the heart is contracting strongly. 
but there must be a positive correlation with total audio amplitude and contractile force. obviously heart contractility correlates with bpm, but my idea is to further identify when the contractility of the heart is seemingly misaligned with bpm. Such as Post-Exercise S1 Dominance as documented previously. 

I've added `Average S1 contractility`, `Average S2 contractility` , and `Average contractility` traces to the plot. They just take the location of the S1 peaks etc, to make a graph. I averaged over time to give it a bit of smoothing. 
The idea is to visualize how contractility (S1/S2 prominence) changes as BPM increases or decreases over the recording (e.g. exercise and recovery). This allows us to observe how S1 contractility remains high after exercise despite BPM decreasing. 
Contractility can also be used to visualize RSA.
































``` title:"Mermaid diagram for algorithm's logic flow"
flowchart TD
    %% Input Stage
    A[Input Audio File] --> B{Is .wav?}
    B -- NO --> C[Convert to WAV via FFmpeg]
    B -- YES --> D
    C --> D

    %% Stage 1: Preprocessing
    subgraph Stage1[Stage 1: Preprocessing]
        D[Resample & Bandpass Filter] --> D1{Hum removal\nenabled?}
        D1 -- YES --> D2[Detect & Notch-Filter\nStationary Hum]
        D1 -- NO --> E
        D2 --> E
        E[Extract Hilbert Envelope] --> E1[Extract Multi-Band\nS1 & S2 Band Envelopes]
        E1 --> F[Find All Potential Troughs]
        F --> G{Enough troughs?\n>= 5}
        G -- YES --> G1[Calculate Dynamic Noise Floor\n& Sanitize Troughs]
        G -- NO --> G2[Use Static Noise Floor\nFallback]
        G1 --> H
        G2 --> H
        H[Find Raw Peaks Above Noise Floor]
    end

    H --> I

    %% Stage 2: Preliminary Pass
    subgraph Stage2[Stage 2: Preliminary Pass]
        I[Run High-Confidence PeakClassifier\npairing threshold elevated] --> I2[Extract Anchor Beats]
        I2 --> I3{Anchor beats\n>= 10?}
        I3 -- YES --> I4[Estimate Global BPM\nfrom Median R-R]
        I3 -- NO --> I5[Use start_bpm_hint\nor Default 80 BPM]
        I4 --> I6
        I5 --> I6
        I6[Detect Peak BPM Time\n& Recovery Phase Window]
    end

    I6 --> J

    %% Stage 3: Main Classification Loop
    subgraph Stage3[Stage 3: Main Classification Loop]
        J[Initialize PeakClassifier\nwith BPM & recovery context] --> M

        M{More peaks\nremaining?} -- NO --> ZZ[Return S1 peaks,\nAll Raw Peaks, Debug Info]
        M -- YES --> K[Calculate Pairing Ratio\nfrom recent beat history]
        K --> L[Kick-Start Check:\nif stuck in Lone S1 mode,\ntemporarily boost pairing ratio]
        L --> Mlast{Is last peak?}
        Mlast -- YES --> Ylast[Label as Lone S1 Last] --> Z
        Mlast -- NO --> N[Get Current Peak & Next Peak]

        N --> O{Lookahead enabled &\nweak middle peak present?}
        O -- YES --> O1{Middle peak below\nskip threshold?}
        O1 -- YES --> O2[Skip Middle: label as Noise\nPair current & next+1] --> Z
        O1 -- NO --> P
        O -- NO --> P

        P[Attempt S1-S2 Pairing] --> P1{Min S1-S2 interval\nconstraint met?}
        P1 -- NO: too close --> S
        P1 -- YES --> P2["Score Confidence:
1. Base timing score
2. Contractility / S1:S2 ratio
3. Multi-band spectral fingerprint
4. Stability history adjustment
5. V-shape interval penalty
6. Forward-look penalty"]
        P2 --> Q{Confidence >=\npairing threshold?}

        Q -- YES --> R[Label as S1 Paired + S2 Paired\nUpdate interval & contractility history]
        Q -- NO --> S

        S["Validate Lone S1:
- Rhythm plausibility check
- Amplitude vs recent S1 history
- Forward BPM spike check"] --> U{Lone S1 score\n>= threshold?}
        U -- YES --> V[Label as Validated Lone S1]
        U -- NO --> W{Consecutive rejections\n>= cascade threshold?}
        W -- YES --> X[Label as Cascade Lone S1\nReset rejection counter]
        W -- NO --> Y[Label as Noise]

        R --> Z[Update Long-Term BPM Belief]
        V --> Z
        X --> Z
        Y --> Z
        Z --> M
    end

    ZZ --> AA

    %% Stages 4 & 5: Correction & Refinement
    subgraph Stage4["Stages 4 & 5: Correction & Refinement"]
        AA["Stage 4: Rhythm Correction
Remove too-close adjacent S1s
Keep higher amplitude"] --> BB

        BB["Stage 5: Fix Rhythmic Discontinuities
Iteration start"] --> BB1["Pass 1: Find Long Gaps
Search noise-labeled peaks for
strong missed S1-S2 pairs and promote them"]
        BB1 --> BB4["Pass 2: Find Short Conflicts
Remove weaker of adjacent S1 pair"]
        BB4 --> BB6{Corrections made\nthis pass?}
        BB6 -- "YES: iterate up to 5x" --> BB
        BB6 -- "NO: stable" --> CC[Final Corrected S1 Peaks & Debug Info]
    end

    CC --> GUARD{>= 2 S1 peaks\ndetected?}
    GUARD -- NO --> ABORT[Return None:\nInsufficient peaks warning]
    GUARD -- YES --> DD

    %% Stage 6: Final Metrics & Outputs
    subgraph Stage5[Stage 6: Final Metrics & Outputs]
        DD[Calculate Smoothed BPM Series] --> DD1[Detect Trapezoid Artifacts]
        DD1 --> DD2[Find Major HR Inclines & Declines]
        DD2 --> DD3["Calculate HRR
Peak Recovery Rate
Peak Exertion Rate"]
        DD3 --> DD4["Calculate Windowed HRV
RMSSD & SDNN per window"]
        DD4 --> DD5{Frequency domain\nHRV enabled?}
        DD5 -- YES --> DD6["Lomb-Scargle Periodogram
LF / HF / VLF / LF:HF ratio"]
        DD5 -- NO --> EE
        DD6 --> EE

        EE{Manual labels\nCSV found?}
        EE -- YES --> EE1[Compare predictions vs manual labels\nAppend to regression log]
        EE -- NO --> FF
        EE1 --> FF

        FF["Generate Outputs
based on output options"]
        FF --> FF1[HTML Interactive Plot\nwith audio sync & spectrogram]
        FF --> FF2[PNG Static Image]
        FF --> FF3[CSV Data Export]
        FF --> FF4["Analysis Summary
& Debug Chronological Log"]
        FF --> FF5[Filtered WAV]
    end
```
