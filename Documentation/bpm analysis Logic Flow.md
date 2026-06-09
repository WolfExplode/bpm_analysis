---
excalidraw-plugin: parsed
excalidraw-open-md: true
excalidraw-onload-script: |-
  // Set active tool to "hand" on mobile
  if (app.isMobile) {
    ea.getExcalidrawAPI().setActiveTool({ type: 'hand' });
  }

  // Update embedded elements' stroke color, roughness, and stroke width
  const elements = ea.getViewElements();
  elements.forEach((element) => {
    if (element.type === "embeddable") {
      element.strokeColor = "#ced4da";
      element.roughness = 0; 
      element.strokeWidth = 0; 
    }
  });
  ea.updateScene({ elements });
Created on: 2025-07-01 19:05
Last Modified: 2025-07-01 19:39
File Folder:
tags:
  - excalidraw
  - Programming
parent: "[[Programming]]"
children:
---
# **bpm analysis Logic Flow**

Heartbeats have S1 and S2. in the recording, S1 and S2 have similar amplitudes at lower bpm, but when exercising starts S1 gets much louder. this results in double counting at lower bpm.
The audio file's volume increases and decreases depending on stethoscope placement. this means the audio amplitude is not consistent. 
- maybe we can use the amplitude data before normalizing? It's logical that the volume increases as bpm increases, maybe we can use that to know if we are double counting beats when the bpm is supposed to be low or something?
we can measure the trend in volume, not the small outliers (if the user hits the microphone it will result in a large outlier). If the volume trends louder, we can assume bpm should also trend higher in the same timeframe. 

- another idea is, it's simple for a human to understand high vs low bpm from listening to the recording. if there was a way to tell the script the expected bpm range at that specific time, maybe we can get it to output a more accurate graph.

The core idea is that a heartbeat cycle isn't just one peak, it's a pattern. Specifically, it's a pair of sounds (S1 and S2) with a short time gap between them, followed by a longer gap before the next pair begins. I tried to teach the script to recognize this pattern.









At lower bpm, the volume of S1 and S2 are very similar, but at higher bpm, S1 is significantly louder than S2. 

Looking at another file's waveform, S2 completely disappears at high bpm, meaning increasing detection sensitivity would not work. S2 occurs very close and sometimes on top of the next S1, making them merge.
therefore we cannot rely on S2 detection since at extremely high bpm, it does not exist in the waveform. 

One way to correct this is to introduce adaptive logic that de-emphasizes S1-S2 pairing (or even disables it) when the estimated BPM is very high.
**(Very Strict S1-S2 Merging):** we can Set `current_s1_s2_threshold` to a very small, fixed value (e.g., 0.08-0.10 seconds), reflecting only the _absolute shortest_ and unavoidable S1-S2 separation, and essentially making it very unlikely that two distinct beats would be merged. This makes the pairing practically non-existent for high BPMs.


I'm analyzing another file and I notice the morphology of the waveform change. This is a recording of exercise. initially, before workout, low bpm, the volume of S1 and S2 are very similar, but at higher bpm, S1 is significantly louder than S2, to the point that S2 stops getting detected as a peak entirely. but then after the peak workout ends and heart rate decreases again, but the contractile force of the heart sill causes S1 to be significantly louder than S2. This means that the loudness of S2 does not depend entirely on BPM as I previously expected. 







how does the peak detection algorithm work?
we need `scipy.signal.find_peaks` to be more sensitive since it's not detecting some S2

I increased the sensitivity via
`prominence_threshold = np.quantile(audio_envelope, 0.1)`
`height_threshold = np.mean(audio_envelope) * 0.1`
but it's still not detecting S2, I can see in the 'All Detected Peaks (Raw)' the peak is not being detected.
I can visually see that S2 is very prominent and well defined in the waveform, but It's substantially lower in amplitude than S1. 
maybe the `min_peak_distance_samples` parameter might be too restrictive, but the `min_peak_distance_samples` is calculated as `int((60.0 / max_bpm) * sample_rate)`.

we can fix our immediate problem by shortening the`min_peak_distance_samples` window, but it does not solve another problem:
In some recording S2 does not exist due to recording position. In other recordings, S1 and S2 are very distinct as two separate peaks that get detected by the algorithm. Therefore, I our `min_peak_distance_samples` logic is flawed. 
our current implementation of peak detection does not account for recordings that have no S2. 
- If a recording genuinely has no detectable S2, this logic might incorrectly try to pair S1 with noise or even the S1 of the _next_ beat if the `s1_s2_max_interval_sec` is too wide.
- Conversely, if `min_peak_distance_samples` is too large, and `find_peaks` only detects S1, the S1-S2 pairing step essentially does nothing for that beat, which is fine, but the core issue of not detecting S2 in the first place persists.

**maybe we need a more clever way to handle peak detection?**
A heartbeat is composed of S1 and S2, two beats. 
a peak is considered when there is a local maximum, but we must also take into account the value of the next local minimum (a trough) following a local maximum. 
therefore, we can define a lower threshold amplitude that the waveform must reach for a local minimum to be considered.

for example, If the algorithm detects a local min, but the amplitude of that min is still too large, (higher than the lower bound), then it must not consider that as a true local min.
the the lower bound value must be calculated dynamically for noise in the recording. 
after a trough is established, the next peak can either be S1 or S2 (since S2 can sometimes disappear, we can use the same logic as before to determine what it should be considered.. etc)...

This approach is more advanced than what `scipy.signal.find_peaks` directly offers as a single parameter. `find_peaks` primarily focuses on the peaks themselves (their height, prominence, and separation) rather than explicitly validating the depth of the troughs between them against a dynamic noise floor.


> [!think]
> what if we have a dynamic bpm window? If the current detected bpm is 120, then the next bpm cannot differ too greatly from that value. the script will be forced to work its way up and down the bpm just like a real heart would. 
> This should stabilize any crazy swings in bpm. 


we can create two modes for the logic after the peak detection phase, a mode where the script thinks S1 and S2 are both being detected. and a mode where it thinks S2 has dropped out and only S1 is audible.
we can get the script to slowly transition between these two modes depending on the morphology of the waveform. 

for example, at the start of the recording, S2 might be very loud and similar to S1. But as the exercise progresses, S2 will not be as loud as S1 anymore. the script can simply track the amplitude of the detected peaks and if the deviation of the peaks gets larger, we can switch to `S1 only` mode. The script should also learn to recognize a pattern and recognize when the pattern changes. 
In `S1-S2` mode, S1 can be louder or softer than S2, so there would be no pattern, but as S2 becomes quieter than S1, we should start to detect a pattern where we have a louder beat immediately followed by a quieter beat. 
we can also plot the `S1-S2 deviation` and see how it changes over time as a debugging step (logically, this score should change very smoothly)

we should probably calculate the `S1-S2 deviation` like a "preliminary waveform analysis step" where we scan all the peaks to see if we can find this pattern. this pattern should look like 1212121212..... etc alternating high and low intensity beats. after we plot the `S1-S2 deviation` we can cross reference it to determine if a beat was mislabeled 
	for example, if we think that S1-S2 deviates greatly in intensity, we can check before and after the peak to determine if it should be S1 or S2 since S2 will be sandwiched between two peaks that are higher in amplitude.

I think `S1-S2` and `S1 only` mode should be dynamic and linear. not on/off. since the heart slowly transitions from S1-S2 to S2. 
I probably didn't explain my idea that well, can you explain my ideas back to me so I know you're on the same page?
 



currently, the script is failing to get a sense of the rhythm when S1 and S2 have similar intensities. At the start, the script has no way of knowing the bpm, and without knowing the bpm, the script cannot determine the difference between S1 and S2. It's a catch 22 right?

maybe the script can do a Two-Stage Analysis
Before we label peaks as S1 or S2, we will first perform a global analysis of the entire sound clip to get a strong, data-driven estimate of the heart rate. then we can apply a heavy smoothing modifier to this preliminary




The detection algorithm is almost perfect, we just need to modify how the peaks are being processed a little bit. we should give some more logic to the algorithm.
sometimes 'All Detected Peaks (Raw)' is displaying that it detected a peak, but for some reason the algorithm decided to skip it.
We can calculate the instantaneous bpm between two beats and if that bpm differs too greatly from the calculated bpm, then that must be wrong?
When we find a suspiciously long interval, re-check if previously discarded peaks should be included

However, note: Our initial true_beat_indices are built by processing the raw peaks in order... so what do we do?




it's still missing some beats, specifically at very high bpm. setting to `suspicious_interval_threshold_sec = max(median_interval_sec * 1.4, (60.0 / min_bpm) * 0.8)` does not help. our new code seems to be functioning so there must be something else.
I enter the starting bpm at 90bpm, but it reaches 170bpm at the peak, where the algorithm starts to consider peaks as not beats.
I enter the starting bpm at 150bpm, it labels the beats properly at higher bpm but not at lower bpm. 
Here's my interpretation of the issue:
- **Hinted 90 BPM**: The `s1_s2_max_interval_sec` is set to approximately 0.33 seconds. The logic assumes any two peaks closer than this are an S1-S2 pair within a _single_ heartbeat.
- **Actual 170 BPM**: The true interval between _separate_ heartbeats is 60/170≈0.35 seconds.
- **Actual 181 BPM**: The true interval between _separate_ heartbeats is 60/181≈0.33 seconds.

when the actual interval between two distinct heartbeats ($0.35$s or $0.33$s) becomes equal to or shorter than the set `s1_s2_max_interval_sec` (0.33s), the algorithm misinterprets them. It sees two separate heartbeats, but because their time difference falls within the "S1-S2" window, it incorrectly considers them as components of a single heartbeat and likely discards one of the peaks. This results in an undercounting of the actual heartbeats and an artificially lower BPM calculation.




> [!think]
> the initial bpm I input is not being respected enough.
> `s1_s2_max_interval_sec` is being updated too frequently and with too much sensitivity. `s1_s2_max_interval_sec` should be determined by the bpm, but shouldn't there be two types of BPM being calculated here?
> a instantaneous bpm, and a more averaged out, long term bpm.
> we can use this long term bpm to determine what the expected bpm should be. that way, if the script calculates a bpm that's completely off, it should correct itself.
> for example, If I input a suggested bpm of 120, then the script calculates the interval between beats to be at 240bpm. then we can obviously conclude that it counted a S2 as a lone S1, it missed the pairing. So we can also use this discrepancy to work with the normal deviation we calculated in the other step, to definitively conclude that the beat is supposed to be S2.
> obviously my starting bpm suggestion should be value for the long term bpm.

the long term bpm should change slowly and within physiological limits. limited by the `max_bpm` the slope 
I input 90bpm as the start, then the `Long-Term BPM` goes from 90 to 176bpm in 18seconds (86bpm/18s)

since everything is based on the `Long-Term BPM`, we need to make sure it's accurate but not reactive. 


> [!think]
> the long term bpm was stable and then it skyrockets to 240 hitting the ceiling 
> - `Long-Term BPM` is still too sensitive to change.
> - once it **BOOSTED** the confidence, it still failed to pair which starts the chain reaction
> - I updated to `s1_s2_max_interval_sec = min(0.4, expected_rr_interval * 0.6)` to capture more S1-S2 pairs
> - the calculate_dynamic_confidence() piecewise function is not good, try to implement a more gradual transition between modes

### 1. The Core Problem: Lack of a "Memory"
The current script calculates the critical `s1_s2_max_interval_sec` based on `last_s1_interval`. This `last_s1_interval` is updated with every single beat that is identified.
- **If it misidentifies an S2 as an S1:** The interval becomes very short, the calculated BPM spikes, and the `s1_s2_max_interval_sec` for the _next_ beat becomes tiny, making it nearly impossible to find a correct S1-S2 pair.
- **If it misses a beat entirely:** The interval becomes very long, the calculated BPM plummets, and the `s1_s2_max_interval_sec` for the next beat becomes too permissive, increasing the risk of incorrectly pairing a beat with noise.
### 2. Your Solution: Long-Term vs. Instantaneous BPM
Your idea of implementing two different BPMs directly solves this. Here's how it would work in practice:
- **Long-Term BPM (or Expected BPM):**
    - This acts as the algorithm's "memory" or "belief" about what the heart rate _should_ be.
    - It would be initialized with your starting BPM suggestion (e.g., 120 BPM). If no suggestion is given, it could start at a reasonable default (e.g., 80 BPM).
    - Instead of being recalculated from scratch every time, it would be updated slowly using a weighted average. For example: `Long-Term BPM = (0.95 * Old Long-Term BPM) + (0.05 * New Instantaneous BPM)`. This ensures that one or two bad measurements don't derail the entire analysis.
- **Instantaneous BPM:**
    - This is what the script currently calculates—the BPM based on the time between the last two beats.
    - Its role changes from _dictating_ the logic to _informing_ it.
### 3. Using the New Model for Smarter Decisions
With this new two-BPM model, we can make the algorithm far more intelligent, just as you suggested:
**A. Stabilize the S1-S2 Pairing Logic:** The `s1_s2_max_interval_sec` would no longer be based on the last beat interval. Instead, it would be calculated from the stable **Long-Term BPM**.
- **Example:** If the Long-Term BPM is `120`, the expected time between beats is `0.5` seconds. The `s1_s2_max_interval_sec` would be set to a fraction of that (e.g., `0.5 * 0.5 = 0.25` seconds), providing a consistent and stable threshold for finding S1-S2 pairs.
**B. Implement Discrepancy-Based Error Correction:** This is your most powerful suggestion. The algorithm can now actively look for and correct its own mistakes.
- **The Logic:** After finding a potential beat, it calculates the `Instantaneous BPM`. It then compares it to the `Long-Term BPM`.
- **Example Scenario:**
    1. Your `start_bpm_hint` sets the **Long-Term BPM** to `120`.
    2. The script processes the audio and finds two consecutive peaks that it initially labels as "Lone S1". The interval between them gives an `Instantaneous BPM` of `240`.
    3. **Discrepancy Check:** The algorithm compares `240` to `120`. It sees that the instantaneous value is exactly double the expected value.
    4. **Conclusion:** This is a huge red flag. The most likely explanation is that the second peak was not a new beat (S1) but was actually the S2 of the first beat.
    5. **Action:** The algorithm can now use this conclusion to override its initial decision. It can look at the `Norm. Deviation` and `Pairing Confidence` between those two peaks. Even if the confidence was initially too low, the glaring BPM discrepancy gives it a strong reason to re-classify the second peak as an S2 and merge the two into a single heartbeat.
In short, your proposed changes would transform the script from a simple signal processor into a true analysis engine that can form a hypothesis about the heart rate and use that hypothesis to intelligently interpret the incoming data and self-correct when it deviates too far from the expected pattern.



### un-implemented
the idea of using `S1-S2 deviation` is useful but I think we are applying it incorrectly. 
As our patient exercises their heart's contractile force increases and it beats faster. S1 gets louder while S2 does not. This causes `S1-S2 deviation` to increase. Then after workout bpm decreases but `S1-S2 deviation` stays elevated due to the heart's increased contractile force. 

Sometimes when the heart is pounding hard, S2 can completely disappear since it can occur very close and sometimes on top of the next S1, making them merge. It becomes physically impossible to see S2 in the waveform. Only S1 will be recorded and the script will see that as a very low `S1-S2 deviation`. This is because it only recorded S1 and the amplitude of S1 and S1 will be very similar.
but then when the patient was at rest, the S1 and S2 may also be similar, also resulting in `S1-S2 deviation` being low.



### implemented
the script is working much better, but It's still missing S1-S2 pairs and labeling as lone S1
when the `S1-S2 deviation` is > 0.3, it should check for the amplitude pattern high,low,high,low,high etc... 
and see where it lies. Since `S1-S2 deviation` is > 0.3, there must be a consistently distinguishable high-low pattern. 
we can search ± 3 beats in either direction of the current beat to determine the sequence.

we can implement this logic before the other code but disregard it if it can't find a high-low pattern or if it can't fit the peak confidently as either high or low

let's change it from 0.3 to 0.25
the `HIGH-low` pattern matching is so successful maybe we can expand it to include peaks that lie on deviation between 0.2 and 0.25, "0.2 <`S1-S2 deviation`< 0.25".
for example, we have the amplitude before and after amp: 4183
2967
4183
2986
the local calculated `S1-S2 deviation` for this beat should be ~.28
then we compare the local calculated `S1-S2 deviation` and see if it's greater than 0.25, If it is, we can include it as part of the `HIGH-low` pattern matching. 






I've noticed from the waveform, the noise in the recording has a pattern. since Heartbeats are distinct pulses and noise is constant and spread out, I can look at the lowest amplitude of the audio envelope to detect noise.
when there is no noise, the amplitude goes like this: 
150, 2,279(amp of S1), 221(this amp is larger since the end of S1 and start of S2 kind of merge), 426(amp of S2), 150
as you can see, we have 2 peaks, S1 and S2. but we also have two troughs, before S1 and between S1_S2. 

We can define a single noise floor value by measuring the lower quartile the all the local minimums (troughs). I think by doing this, we can get some very useful information for our script to use in its logic. 
we can graph the noise to visualize where there is temporary noise in the recording since temporary noise will have troughs that are higher than the noise floor.

in the previous update, we calculated a noise floor. This still needs some tweaking to fully represent the waveform.
Let's make it dynamically update. The logic of the noise floor is used to detect temporary noise and distinguish it from the background noise (noise floor). therefore we must update the noise floor to reflect changes in background noise, but If we make it too sensitive to change, It will capture the temporary noise as well, which ruins its purpose.


```
---
## Time: `130.0975s`
* **Audio Envelope**: `3749.07`
* **Raw Peak (Amp: 3749.07)**
    * Status: S1 (Paired). Base Conf: 0.79 (Smoothed Dev: 0.52, LT-BPM: 148)
    * BOOSTED to 0.95 (BPM spike: 327>>148)
* **Smoothed BPM: 179.52**
* **Long-Term BPM (Belief): 147.88**
---
## Time: `130.1893s`
* **Audio Envelope**: `1214.48`
* **Norm. Deviation (Smoothed): 52.22%**
---
## Time: `130.2812s`
* **Audio Envelope**: `1574.29`
* **Raw Peak (Amp: 1574.29)**
    * Status: S2 of 130.10s
---
## Time: `130.3413s`
* **Audio Envelope**: `1197.27`
* **Norm. Deviation (Smoothed): 52.13%**
---
## Time: `130.4014s`
* **Audio Envelope**: `1375.67`
* **Raw Peak (Amp: 1375.67)**
    * Status: Lone S1. Base Conf: 0.79 (Smoothed Dev: 0.52, LT-BPM: 148)
    * BOOSTED to 0.95 (BPM spike: 448>>148)
    * SKIPPED PAIRING (Vetoed by lookahead)
* **Smoothed BPM: 178.81**
* **Long-Term BPM (Belief): 147.88**
---
## Time: `130.4683s`
* **Audio Envelope**: `2426.41`
* **Norm. Deviation (Smoothed): 51.86%**
---
## Time: `130.5351s`
* **Audio Envelope**: `3141.75`
* **Raw Peak (Amp: 3141.75)**
    * Status: S1 (Paired). Base Conf: 0.79 (Smoothed Dev: 0.52, LT-BPM: 148)
    * BOOSTED to 0.95 (BPM spike: 383>>148)
    * OVERRIDE (H-L Pattern, Local Dev: 0.46)
* **Smoothed BPM: 178.68**
* **Long-Term BPM (Belief): 147.88**
```

I've noticed that the script's functionality heavily relies on a correct order of operations. If we put the logic in the wrong sequence, the script will make incorrect conclusions. 

the script incorrectly identifies a noise `Raw Peak (Amp: 1375.67)` as a Lone S1. Take a look at the amplitudes, can't we intelligently deduce that it can't be a S1? I mean, just looking at the data a human can logically deduce that this muse be noise. how can we give the same logic to the script?

we can add some more hyper-specific logic to the script,
before it tags a peak as a Lone S1, the script should compare the current amplitude of this peak to the amplitude of the next peak following it. If the amplitude is substantially higher, maybe if `next amp > (current amp*1.5)` or some logic like that, then the script can understand that the current peak is not a Lone S1. It might be noise.

maybe we can get even more specific:
We can analyze the `amplitude of the previous minimum trough` and see if it deviates too greatly from the established `noise floor`. If the amplitude of the trough is > 3*`noise floor`, then we can assume that there must be a high amount of temporary noise. Therefore it should make the script even more confident that this peak is temporary noise. we can have a scores like `Noise confidence`. I remember we used another confidence score for S2 logic. to remain consistent with our programing, we can probably rename that to `S2 confidence`








### implemented
Take a look at this log. try to determine why the peak at `225.7083s` is being categorized as S2.
The next peak at `225.8292s` is the actual S2 and it's being incorrectly labeled as Noise.
We have a lot of filters to determine if something is S1, but not a lot of double checking is done to see if a peak is S2


I think we can catch these edge cases by comparing the amplitudes. If (2*(Current peak - Next trough) <  (Next peak - Next trough)), then we can mark current peak as noise

we can deduce that this is a noise peak if the next peak is substantially higher in amplitude by measuring both amplitudes relative to the trough between them. 
[![](https://i.imgur.com/3eKucJ1.png)]


**How this Functions:** It takes the amplitude of the current peak and multiplies it by 1.5. If the next peak's amplitude is greater than this value, it assumes the current peak is insignificant noise and discards it
```
# --- Lookahead Amplitude Veto ---
# If the next peak is substantially larger, the current one is likely noise.
if i < len(all_peaks) - 1:
	next_peak_idx = all_peaks[i+1]
	if audio_envelope[next_peak_idx] > audio_envelope[current_peak_idx] * 1.5:
		beat_debug_info[current_peak_idx] = f"Noise (Vetoed by larger subsequent peak at {next_peak_idx/sample_rate:.2f}s)"
		i += 1
		continue # Skip this peak entirely
```






> [!Question]
> Is tweaking values the correct way to do things? 
> we may end up tweaking values all day. maybe we can make the code more robust by applying even more case specific logic into the chain?

**Parameter Tweaking (The "Knobs")**
- **Pro:** It's fast. Changing a value like `s1_s2_interval_rr_fraction` is easy and immediately testable. The centralized `DEFAULT_PARAMS` in your v4.5 code is the _correct_ way to manage this.
- **Con:** It's not robust. A value that works for a heart rate of 70 BPM may fail for a rate of 160 BPM. It doesn't capture the underlying "why" of the signal.

**More Case-Specific Logic (The "If-Then" Chains)**
- **Pro:** It can solve complex, known failure modes that simple parameter changes can't. For example, "if a peak is vetoed, but the _next_ peak is extremely large, maybe un-veto it."
- **Con:** This can quickly lead to unmaintainable "spaghetti code." A long chain of `if/elif/else` rules becomes incredibly difficult to debug. A new rule added to fix case A might inadvertently create a new bug in case B. The complexity grows exponentially.




### implemented
We can implement:
**Probabilistic Scoring: Move from "Yes/No" to "Maybe"**
The current code makes hard decisions: `is_paired` is either `True` or `False`. A more robust method is to calculate a **score** or **probability**.
- Instead of a single `pairing_confidence`, calculate a `pairing_score` from multiple sources:
    - `score = (w1 * timing_confidence) + (w2 * amplitude_confidence) + (w3 * width_confidence)`
- You then set a threshold on this combined score. This is more robust because a peak can have a slightly-off timing if its amplitude and width characteristics are perfect, and vice-versa.





### unimplemented
Misinterpretation of Noise as Heartbeats can cause a infinite feedback loop that skyrockets the BPM.
**Incorrect Classification as "Lone S1"**: After bypassing the noise filter, the algorithm attempts to confirm the peak as a paired S2. Since there is no true S2, this pairing always fails. The algorithm's fallback is to classify the peak as a `Lone S1`. The result is that a single heart sound (or a noise artifact) is now logged as two separate `Lone S1` events in rapid succession.

maybe we can do a simple dumb logic to say, If the previous 10 peaks were detected as lone S1, then it's likely that this recording has no S2?
If this is the case, should we disable all detection S2 logic or make it harder to classify something as S2? 
but if the script determines that there is no S2, how do we get out of this state?
your thoughts?






### implemented
along side our `max_bpm_rejection_factor`
heart may beat faster or slower beat by beat meaning rejecting a high instantBPM isn't a good idea. I was thinking about the idea of HRV and realized something.
we can calculate a HRV based on the person's expected bpm. I think HRV decreases as bpm increases? 
then we can use the HRV to determine if our calculated instantaneous BPM makes sense or not. 

do we really need to use so much logic to calculate a Plausibility Window? why not calculate a bpm based HRV and use that instead?
we can also calculate a regular HRV for the entire recording so the we can see the patient's metrics





### implemented
the displayed RMSSD and SDNN doesn't represent HRV as a concept since HRV decreases as bpm increases. We could solve this by capturing a local RMSSD and SDNN and correlating that with heart rate at that time.
then average those correlated measurements together to give a final average RMSSD and SDNN.
then as a visualization tool, we can also graph the temporary, uncorrelated RMSSD and SDNN on the graph. 






### fixed?
there are small spikes in the bpm that are caused by noise. since it's really difficult to see what is S1 and S2 because they are similar in amplitude, noise sneaks its way in.
the noise `104.8854s` is getting marked as S2 causing `104.9917s` to be marked as S1 while it's actually S2.
then `105.1958s` gets marked as S2 but it's actually noise, causing `105.3250s` to be marked as S1, which it is, but `105.4354s` is marked as S2 but it's actually noise. then that causes `105.5896s` to be marked as S1 even though it's S2. Then there's no noise after that so the next S1 gets marked as S2 and the S2 after that gets marked as S1 etc.... so the error never gets fixed since there was a odd number of noises. the end result is the S1 and S2 swap places until it runs into another odd number of noises which swaps them back again. 

This phenomenon is visible by a very small bpm increase(caused by noise) followed by S1 and S2 swapping places, then another very small bpm increase (caused by noise) which swaps them back.




### implemented
I want to create a interactive plotting thing where I can click and drag two points. Then automatically calculate the slope of that line (edited)
The goal is to know the rate of bpm increase/decrease to calculate heal etc

I want the measurement to be most useful for heath analytics.
Do we measure the steepest decrease slope just before the bpm decrease starts to wane, or do we measure a fixed time after the workout to standardize the measurement?

2025-06-27 13:15:33,892 - [INFO] - Evaluating potential decline from 07:30 to 09:08 -> Duration: 98.2s, Decrease: 69.9 BPM.
2025-06-27 13:15:33,892 - [INFO] - ----> FOUND significant decline: Duration=98.2s, Change=-69.9 BPM.
2025-06-27 13:15:33,892 - [INFO] - Calculated 60-second HRR: 59.2 BPM drop.
in this case, we can see -69.9 BPM change, but this is over 98.2s. the heart rate recovers much faster right after exertion ends but this steeper slope is never captured. I want to know this data. 

the exertion and recovery slopes identification needs adjustment.
the real exertion slope should start at ~300s to the peak bpm ~450s.





### unimplemented 
during the post processing steps/iteration, we can check the SDNN graph to see where the errors may be. 
logically, SDNN should not change drastically and if it does, there might be a mistake in that section. 
how can we apply this logic to the script?




### implemented
[![|305x210](https://i.imgur.com/cmEGokB.png)]  [![|500x210](https://imgur.com/DrF7C8m.png)]
At `t=511.5728s`, the algorithm analyzes a peak. The `Base Pairing Conf` is `0.54`. This is just barely below the `pairing_confidence_threshold` of `0.55`. Because it fails this check, the peak is classified as a `Lone S1`, and the S1-S2 pairing is broken.
The existing `calculate_blended_confidence` function makes its decision based on one single input: the `smoothed_deviation`. We can add a second input that compares the amplitudes of the S1 and S2 candidates. 
If the following peak is lower than the S1, then increase the pairing confidence. 
this separates the problem into two distinct questions: "How much did the amplitude change?" and "Did it change in the right direction?"






### implemented
due to breathing, S1 is not always higher than S2.
- At `t=542.8344s`, the algorithm attempts to pair a small peak (a real S2, amp `2203`) with the following large peak (a real S1, amp `3729`).
- The algorithm correctly notes that the S2 candidate is much larger than the S1 candidate and applies the penalty: `PENALIZED (S2 candidate amp 3729 > 1.5x S1 amp 2203)`.
- This penalty drops the confidence from a strong `0.57` to a failing `0.29`, ensuring the rhythm cannot be reacquired.

If the S2 amplitude is abnormally high, we will first check how "perfect" the deviation value is. If the deviation is in the ideal range, we will trust it and **waive the penalty**, assuming the inverted amplitude is a physiological anomaly (like breathing). "The amplitude relationship is unusual, but the underlying rhythmic shape of the signal (the deviation) is so perfect that I will trust the pairing."
If the deviation is _not_ in the ideal range, we will apply the penalty as before.



another issue caused by the Penalty Waived.
I see we've been implementing a lot of conditions that are similar to our previous conditions. this might lead to a lot of reusable code.
maybe we should make our large functions smaller to accommodate situations when code can be reused in this way.
###### just putting this here to make me feel :5head:
The core problem is that our series of fixes has created a complex web of rules within a single, massive function. This makes it difficult to see how the different logical paths interact, leading to unintended consequences like this one.
Your suggestion to break down the large `find_heartbeat_peaks` function is the perfect architectural solution. It addresses the immediate bug and makes the entire algorithm more robust, readable, and easier to maintain for the future.














- [x] Fixed
[![|581x265](https://i.imgur.com/IOV0GIZ.png)
v5.9 of the code introduced a bug. take a look at the log.
`136.6589s` is noise that got marked as S1 which caused `136.7351s` to get marked as S2 even though it's the S1. 

In `bpm_analysis_v5.8.py`, a penalty was unconditionally applied to the pairing confidence if an S2 candidate's amplitude was significantly larger than the S1 candidate's amplitude. This is a critical rule, as the S2 heart sound is physiologically typically quieter than the S1 sound.

In `bpm_analysis_v5.9.py`, this logic was modified to be conditional. The penalty would be waived if the "normalized deviation" between the two peaks fell within a so-called "ideal range."










### Observation:
**Filename:** `Female Heart Beating during sex laying on stomach (climax)`, `bpm_analysis_v6.0.py`
in this recording, At the beginning, S1 and S2 are similar amplitudes, then at ~16s S2 gets much louder than S1. then at around ~28s S1 gets louder than S2.
[![](https://i.imgur.com/isIiFuq.png)
### Brainstorming:
so far we know:
- at low bpm, S2 may be sometimes louder than S1 depending on breathing and stethoscope positioning. 
- due to how the heart functions, at higher bpm, S1 starts becoming much louder than S2. 
- After a period of exertion (how do we know if it's after a period of exertion?) the heart needs time to recover its contractile force stays elevated even though the bpm has dropped back down. 
	- therefore the fact that S1 is louder than S2 isn't always dependent on bpm, but rather, it's dependent if the heart has temporarily increased contractile force. 
		- this may be hard to determine since a recording may start after a workout has ended, or before a workout begins.
- At low bpm, S1-S1 interval is high but the S1-S2 interval remains relatively unchanged. therefore in this recording, we can logically deduce where where the S1-S2 pairs are even though their amplitudes exchange. also this logic is not needed at higher bpm because the body will always make S1 louder than S2 due to the heart's increased contractile force. 
	- so if two S1 are too close together at low bpm, then we can say it cannot be a lone S1. 
	- the S1-S1 interval should be high at lower bpm and low at higher bpm
	- The S1-S2 interval should also be high at lower bpm and low at higher bpm, but it shouldn't change as much as S1-S1 interval. 
	- don't we already have R-R interval logic in our script?
- variable names such as `penalty` etc are ambiguous, maybe we should use the logic of `Contractility` instead.

"Given the current heart rate (our proxy for contractility), what is the **expected** relationship between S1 and S2? Does this beat pair match our **expectation**?
how can we implement these ideas? 
### Implementation:
- [x] Implemented
The script uses a fixed rule (`S2 > 1.5 * S1`) to penalize beat pairs, which fails at lower heart rates where S2 can be physiologically louder than S1. We will replace this with a dynamic model that adjusts its expectations based on the heart's current state, using BPM as a proxy for contractility.
- **Introduce a BPM-Dependent Expectation Framework:**
    - **Low BPM (< 100 BPM):** The script will be much more lenient. It will _expect_ that S2 can be louder than S1 and will not penalize, or will only slightly penalize, pairs where S2 is significantly louder (e.g., > 2.2x S1).
    - **High BPM (> 130 BPM):** The script will have a strong _expectation_ that S1 is louder than S2. It will apply a significant confidence reduction to any pair where S2 is louder than S1, as this is physiologically unlikely during high exertion.
    - **Transition BPM (100-130 BPM):** A sliding scale will be used to moderately adjust the confidence if S2 is louder than S1.
- **Implementation in `evaluate_pairing_confidence`:**
    1. This function must now accept `long_term_bpm` as an argument.
    2. Inside the function, an `if/elif/else` block will check the `long_term_bpm`.
    3. Based on the BPM, it will set a `max_expected_s2_s1_ratio` and an `adjustment_factor`.
    4. The penalty will be applied only if `s2_amp` exceeds `s1_amp * max_expected_s2_s1_ratio`. The `adjustment_factor` itself will be determined by the BPM, making it less of a blunt penalty and more of a context-aware adjustment.

at low bpm, S1-S1 interval is high but the S1-S2 interval remains relatively unchanged.
so if two S1 are too close together at low bpm, then we can say it cannot be a lone S1.
the S1-S1 interval should be high at lower bpm and low at higher bpm
The S1-S2 interval should also be high at lower bpm and low at higher bpm, but it shouldn't change as much as S1-S1 interval. 

don't we already have R-R interval logic in our script?

| Feature            | **The "Contractility" Model**                                                                                                                                  | **The "Rhythm Plausibility" Model**                                                                                                                                    |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Primary Focus**  | The **amplitude (loudness)** relationship between an S1 and its potential S2.                                                                                  | The **timing (intervals)** between consecutive S1 beats (the S1-S1 interval).                                                                                          |
| **Problem Solved** | It fixes the error where valid beats are rejected as `Noise (Rejected: Inverted S1/S2)`. This happens at lower BPMs when S2 is physiologically louder than S1. | It would fix a different potential error: two consecutive peaks being classified as "Lone S1" when the time between them is too short for the current (low) BPM.       |
| **Mechanism**      | It adjusts the _expected S1-to-S2 amplitude ratio_ based on the current BPM. It becomes more lenient about a loud S2 at low BPMs and stricter at high BPMs.    | It enforces rules on the _minimum allowable S1-S1 interval_ based on the current BPM. If the interval is too short, it implies one of the beats must be misclassified. |

#### Do We Already Have R-R Interval Logic?
Yes, the script **does** have R-R interval logic, but your idea points out a way to make it even smarter. Here is what the script has now, and how your idea improves upon it:
1. **`is_rhythmically_plausible()` function**:
    - **What it does:** This function acts as a "speed limiter" on the BPM. It checks if a _new_ S1 candidate would cause the R-R interval (and thus the BPM) to change too drastically compared to the established `long_term_bpm`.
    - **Its Limitation:** It is primarily focused on preventing single outlier beats from skewing the BPM calculation. It doesn't explicitly handle the case of two _consecutive_ beats that are both classified as "Lone S1" but are too close to each other.
2. **`correct_peaks_by_rhythm()` function**:
    - **What it does:** This is a _post-processing_ step. After an initial list of S1 peaks is generated, this function finds S1 peaks that are closer together than a certain percentage of the _median_ R-R interval for the whole recording. When it finds a conflict, it keeps the peak with the higher amplitude and discards the other.
    - **Its Limitation:** Because it runs _after_ the main classification, it's more of a cleanup tool. Your idea could be applied more proactively _during_ the main classification loop.
3. **`s1_s2_interval_rr_fraction` parameter**:
    - **What it does:** This parameter directly links the S1-S2 interval to the R-R interval. It ensures that the maximum time allowed between a paired S1 and S2 shrinks as the heart rate increases.
    - **Its Limitation:** This applies only to the `S1-S2` interval for _pairing_, not the `S1-S1` interval for validating lone beats.

The current script checks if a beat is plausible in the context of the _long-term rhythm_, but it doesn't have a strong rule that says:
> "Wait, the BPM is currently 70. The minimum possible S1-S1 interval should be around 0.85 seconds. These two candidate 'Lone S1' peaks are only 0.4 seconds apart. Therefore, they _cannot_ both be Lone S1s. One of them is likely an S2 or a noise peak."

By combining both ideas, the script becomes far more intelligent:
1. The **Contractility (Amplitude) Model** will correctly handle cases where S2 is louder than S1, preventing incorrect rejections.
2. Your **Rhythm Plausibility (Timing) Model** will correctly handle cases where two beats are too close to both be S1s, preventing incorrect classifications.
Together, they create a system that understands both the expected loudness _and_ the expected timing of heartbeats across a range of physiological states.

```
# --- 4.3. Physiology-Based Confidence Adjustment ---  
"enable_stability_pre_adjustment": True,  # Master switch for the stability adjustment logic.  
"stability_history_window": 20,         # Number of recent beats used to determine rhythm stability.  
"stability_confidence_floor": 0.50,     # At 0% pairing success, confidence is multiplied by this (e.g., a 50% reduction).  
"stability_confidence_ceiling": 1.10,   # At 100% pairing success, confidence is multiplied by this (e.g., a 10% boost).  
"s1_s2_boost_ratio": 1.2,               # S1 strength must be > (S2 strength * this value) to get a confidence boost.  
"boost_amount_min": 0.10,               # Additive confidence boost for a "good" pair in an unstable section.  
"boost_amount_max": 0.35,               # Additive confidence boost for a "good" pair in a stable section.  
"penalty_amount_min": 0.15,             # Subtractive confidence penalty for a "bad" pair in a stable section.  
"penalty_amount_max": 0.40,             # Subtractive confidence penalty for a "bad" pair in an unstable section.  
"s2_s1_ratio_low_bpm": 1.5,             # At low BPM, allows S2 to be up to 1.5x S1 strength before penalty.  
"s2_s1_ratio_high_bpm": 1.1,            # At high BPM, expects S2 to be no more than 1.1x S1 strength.  
"contractility_bpm_low": 120.0,         # Below this BPM, the 'low BPM' confidence model is used.  
"contractility_bpm_high": 140.0,        # Above this BPM, the 'high BPM' confidence model is used.  
"recovery_phase_duration_sec": 120,     # Duration (seconds) of the high-contractility state after peak BPM.
```


#### We must also update the `calculate_blended_confidence` with these new ideas. 
we can make the confidence curve itself dynamic, fully realizing the "Contractility Model."

We will modify `calculate_blended_confidence` to construct a unique confidence curve for every beat, based on the `long_term_bpm`.
- **At Low BPM:** The function will use a curve that assigns the highest confidence to peaks with **low deviation** (i.e., S1 and S2 have similar amplitudes).
- **At High BPM:** It will use a curve that assigns the highest confidence to peaks with **high deviation** (i.e., S1 is significantly louder than S2).
- **In the Transition Zone:** It will smoothly interpolate between the two curves, creating a blended, context-aware model.
This change promotes the pairing logic from a simple rule-checker to an intelligent system with dynamic expectations.


#### but wait, We are using BPM as a _proxy_ for contractility
After a period of exertion (how do we know if it's after a period of exertion?) the heart needs time to recover its contractile force stays elevated even though the bpm has dropped back down. 
- therefore the fact that S1 is louder than S2 isn't always dependent on bpm, but rather, it's dependent if the heart has temporarily increased contractile force.
During recovery, the sympathetic nervous system (which increases heart rate and contractility) is still active, but the parasympathetic system is working to slow the heart rate down. This leads to the exact situation you described:
- **Heart Rate (BPM):** Decreasing.
- **Contractile Force:** Remains temporarily elevated.
Our current model would see the decreasing BPM (e.g., 120 BPM -> 95 BPM) and incorrectly switch its expectation, becoming more lenient about a loud S2. However, the heart is still contracting forcefully, so S1 _should_ still be dominant.

#### How Do We Know If It's After a Period of Exertion?
This is the crucial question. Since the script only has the audio data, we cannot know the listener's activity _externally_. However, we can **infer a "post-exertion state" by analyzing the history of the BPM itself.**
Here is how we can do it:
1. **Find the Peak Exertion Point:** First, we analyze the entire `smoothed_bpm` series to find the absolute maximum BPM value and the time at which it occurred. This marks the end of the exertion period and the beginning of the recovery period.
2. **Define a "Recovery Window":** We can define a "recovery window" as a set amount of time immediately following that peak BPM (e.g., the next 1 to 3 minutes).
3. **Create a "Stateful" Contractility Model:** We can make our `evaluate_pairing_confidence` function "state-aware." It will check if the current beat falls within this recovery window.

> [!think]
> but wait, this model only see the peak of maximum exertion. what if the audio file is very long and there are multiple peaks? 
> 	wait, don't we already have a way to calculate recovery and exertion slopes? I wonder if we can leverage that information to better our decision making

### Checking Implementation:
- [x] fixed
with our new changes, we have a problem. S1 peaks are getting rejected because there was no S2 for that beat in the waveform. 
`166.1291s` **Noise (Rejected: Inverted S1/S2).** even though it's actually a lone S1. 
If the pairing failed because of this inverted ratio, the code immediately rejects the S1 candidate as noise


- [x] fixed
the script asks "Okay, if it's not part of a pair, could the peak at `95.7152s` be a valid Lone S1?", but why did the lone S1 part of the script determine it was a lone S1? 

the script only checks `is_rhythmically_plausible` but it never checks (amplitude - noise floor) to see if it's large enough?
also why does the next peak `95.7980s`get flagged as S1 (Paired) if it's so close to `95.7152s`? wasn't there logic that detects if two S1 are too close together? 

in this case `95.7980s` is correctly marked as S1(paired). 
maybe the order of operations should be, 
is_rhythmically_plausible --> pairing_confidence_threshold --> Is (amplitude - noise floor) Not significantly different than (amplitude - noise floor) of the previous S1 peak? --> is it not very close to the next peak?(doesn't cause instantaneous bpm spike) --> only then it is a Lone S1

	- I've implemented a `correct_peaks_by_rhythm` post processing pass but why didn't it pick up on this bpm spike?

> [!think]
> I've noticed that a lot of our heuristics are defined by comparing to either the peak before or the peak after, but it never takes data from both and averages it together. 
> the idea is that we can know if a peak is a outlier if it gets flagged as something (either S1 or S2) and is significantly different(in amplitude relative to the noise floor, and timing) than a peak of the same type in it's immediate surroundings. 


[![|668x292](https://i.imgur.com/07ds0xY.png)]

Time: `509.3344s`
**Noise (Rejected Lone S1: Causes BPM spike (Forward interval 0.185s < min 0.219s)).**
- Original pairing reason: [Base Pairing Conf: 0.61 (vs Threshold: 0.52)
- Inferred Recovery State 
- ADJUSTED (Next peak is too loud to be a plausible S2 at this BPM).
- Justification: S2/S1 Ratio 1.3x > Expected 1.1x at 165 BPM.
- Result: Confidence adjusted to 0.24.]
**Audio Envelope**: `2814.53`
**Noise Floor**: `549.32`
**Raw Peak** (Amp: 2814.53)
**Long-Term BPM (Belief)**: 164.53
**Norm. Deviation**: 35.56%

This is a edge case caused by breathing. according to logic, S1 should be louder than S2 especially due to the heart's increased contractility at this bpm, but due to the effects of breathing, S2 becomes louder momentarily. 
	what do we do about this?



### partly-implemented
I want the boost to be more intelligent but I can't figure out a condition
```
# Standard Boost Logic (when S1 > S2)  
elif s1_amp > (s2_amp * params.get('s1_s2_boost_ratio', 1.2)):  
    confidence = min(1.0, confidence + params.get('s1_s2_boost_amount', 0.15))  
    reason += f"| BOOSTED to {confidence:.2f}, (S1 amp {s1_amp:.0f} > 1.2x S2 amp {s2_amp:.0f}) "
```
the s1_s2_boost_amount should be larger if.... ?
- [x] implemented
If there has been a history of successful S1-S2 pairs? we can get a count of previous successes and conclude that the next pair should be successful. 
we can say, of the previous 10 beats, how many were paired and how many were lone S1, then linearly increase our boost amount if there have been more pairs in the past.
- [ ] implemented
or we can also add a ideal amp range for the potential S2 by comparing its amp to the amplitude of past S2 pairs. we can calculate a rolling a rolling average of past S2 pairs and compare the current potential S2 amplitude to see how close it is. The further away, up or down, the less we should boost.


- [x] implemented
the High local noise confidence could be expanded upon.
```
if noise_confidence > params['noise_confidence_threshold'] and not is_potential_s2 and not strong_peak_override:  
    beat_debug_info[current_peak_idx] = "Noise (High local noise confidence)"  
    i += 1  
    continue
```











> [!think]
> I've noticed that in our code, we have a lot of "Penalty" "veto" "waiver" etc variables. and in many cases, we may even Penalty these rules by Penalty-ing the Penalty. this creates a PenaltyPenaltyPenalty etc..... forever loop
> This is a concerning trend in my codebase and I don't like it. maybe we can fix it by renaming some of these Penalties by using logical facts we know about how the heart functions. 
> 
> for example, The log shows `Noise (Rejected: Inverted S1/S2)` with the justification `PENALIZED (S2 amp > 1.5x S1 amp)`. This happens at a lower heart rate (~82-93 BPM), where S2 can be physiologically louder than S1. The script incorrectly flags these as noise.
> to fix this, we may need a PenaltyPenalty... which sounds ugly. by understanding how the heart works, we can rename these variables. 
> in this case, should use the logic of `Contractility` instead.


















[![|1527x707](https://i.imgur.com/WfwnOJl.jpeg)]


I implemented a limit on how soon S2 can occur after S1 which is causing the pairing to fail. This only occurs since the highest point of the S2 wave is at the start and not the center, resulting in this failure
	note: This specific case also results Lone S1 decision failing















## iterative_correction_pass:
### Brainstorming:
ok, I think It's about time we wrote the iterative part of this script. I've avoided writing implementing this feature since I want to be absolutely sure I can get the script to be as accurate as possible in the first pass. I wanted to do all obvious implementations first before iterating. 

- At the beginning of the file, we essentially had a "Cold Start", how can our iteration mitigate this?
- It's annoying to find the starting bpm from the user, would iterating help find the starting bpm automatically?
- sometimes I see the peak detector put a trough if there's a small divot in the S1/S2 waveform. These small divots are obviously not real toughs so the next iteration should compare the amplitude of every trough to see if it's > (5 x Noise floor) and mark them as noise? would marking them as noise introduce issues, should we remove the troughs instead? 
- since we have a estimate for our bpm, we can go back and re-examine our detected peaks. If two peaks are too close together, one of them must be noise. 
- you said Initialize the `long_term_bpm` with the `global_bpm_estimate` from Stage 1, but wouldn't it be better to calculate the bpm for time 4s to 8s. it's more accurate and relevant to the start and we don't consider the first 4 seconds since it might be incorrect from cold start. 

### Implementation
- [x] Implemented
Currently, the `run_iterative_correction_pass` isn't finished. Help me finish writing the iterative_correction code. To avoid writing logic in `iterative_correction_pass` that should be put in the main body of the script, we should only put conclusions that can only be done by analyzing the finished data 

The algorithm missed a beat because it miscategorized something at 507.10s. Then the bpm drops from 162.7 to 155.0, It stays like this for a while before jumping back up at 511.90s bpm goes from 154.3 to 160.5.
The resulting bpm graph takes a immediate dip for a few seconds and then come back up immediately. This creates a pattern that looks like someone chiseled a trapezoidal notch out of the bpm graph
How do we teach the script how to recognize this discontinuity? 
[![](https://i.imgur.com/s2TXm9v.png)


















## Trapezoid Artifact
Identify the discontinuity:
- A "notch" would start with a very large **negative** rate of change (the sudden drop).
- Immediately following, there would be a very large **positive** rate of change (the jump back up).
- (keep in mind, a notch could go up or down)

Fix the discontinuity:
the script should scrutinize the labels given to those peas at around the time of the discontinuity. It could shuffle around the labeling order, or make new labeling etc.
so fixing the discontinuity and identifying the discontinuity are not mutually exclusive steps. because during the fixing process, it's still trying to identify what caused the discontinuity. 

One way to fix this is to apply a large smoothing to the bpm like `BPM Trend (Belief)`. I like this idea but it can sometimes blunt the sharpness of the slope in the final graph.



#### Brainstorm
- [ ] Implemented
for the `iterative_correction_pass`. If we see a discontinuity, what if we reorder the labels so that it follows a strict 1212 trend and scan again to see if 
	maybe this isn't a good idea, we need something more intelligent. 
- Fix the discontinuity... the script should scrutinize the labels given to those peas at around the time of the discontinuity. It could shuffle around the labeling order, or make new labeling etc.
	how do we implement this? 

the post process correction should not apply logic for the first few seconds and the last few seconds of a recording
- [x] Implemented














#### Brainstorm
- [x] Implemented
the word boost makes sense because that piece of logic is responsible for brining confidence up.
shouldn't we rename "ADJUSTED" to something more descriptive like penalty?
also I like the boost logic's way of handling a linear `dynamic_boost_amount` maybe we can apply the same concept and have a dynamic_penalty_amount where the amount being diminished is dependent on 
how much higher the S1/S2 ratio is from the expected ratio? 

also, to double check. the S1/S2 ratio is calculated by using the noise floor right? if not, we should do that also

wait, I just realized the boost and penalty logic can be combined into one function since they both compare if S1 amplitude to S2 amplitude.

and currently the `pairing_ratio` logic is only applied for the boost function. we should refactor that out since that bit of logic is far more universal/
the pairing_ratio logic is meant to prioritize stability which is a good thing. I want to expand it's logical reach out of just the boost function so it adjusts the confidence before boost or penalty is applied.

hmm, since pairing_ratio, boost, and penalty all affect the S1-S2 confidence. should we put them all in one function or separate?















> [!think]
> does it make sense to pass the pairing confidence to the post processing pass?

That's a great thought. While it seems intuitive to pass more data along, it's actually more effective to keep the post-processing pass separate and **not** pass the original pairing confidence to it.
The current design is intentional, treating the two stages as having distinct jobs that use different types of evidence.
### The Role of Each Pass
Think of your script as performing a two-step analysis:
1. **The Main Analysis Pass (`find_heartbeat_peaks`)**: This is the **initial detective**. It goes through the audio moment by moment, looking at each peak pair in isolation. It uses a complex set of rules (rhythm, shape, S1/S2 ratio, boosts, penalties) to make its best _local_ decision and assign an initial label (`S1 (Paired)`, `Lone S1`, `Noise`).
2. **The Post-Processing Pass (`_fix_rhythmic_discontinuities`)**: This is the **supervisor reviewing the detective's report**. It doesn't re-read all the case notes. Instead, it looks at the big picture for glaring inconsistencies. Its job is to find the _consequences_ of earlier mistakes, like an unrealistic BPM dip or spike (a rhythmic discontinuity), which are only visible after the initial pass is complete.





> [!think]
> with the implementation of `iterative_correction_pass`, we have reached the stage where the script is capable of correcting itself and then recording the before and after. 
> doesn't this mean our script is capable of self improvement? 
> Since it can detect a discontinuity and then iterate until it finds the correct configuration to remove the discontinuity, It can basically output a list of mistakes the detection algorithm made.
> then I can run many files to this script to get a list of repeated mistakes from the detection algorithm, which will help me fine tune the parameters.

> [!think]
I wonder If I can run this script enough times to get a automatic data tagger that will allow me to pass the audio file and its correct beat labeled output to a AI. then train that AI to do beat detection for me 🤔




- [x] Implemented
I like the stability_confidence idea, but its implementation needs a bit of tweaking. when paring ratio reaches, 0 it's very difficult to begin paring again. since lack of pairing decreases pairing ratio, this is a negative feedback loop

what if it has a kick start mechanism to scan if the previous 3 lone S1s was directly followed by a peak marked as noise. This would give evidence that S2 has started to show up in the waveform again.
if it detects this, we can set pairing confidence to 50% or higher












- [x] Implemented
it seems like we need a gradient for the interval check so it's not on/off
wouldn't it be more intelligent to apply the s1_s2_max_interval logic to the rest of the pairing confidence score?

### Boost/Penalty Logic: The "Shape" Check 🎛️
This logic looks at the **amplitude ratio** of the S1 and S2 peaks.
- **Purpose**: To answer the question, "Are the relative volumes of the two heart sounds physiologically normal?"
- **Mechanism**: It compares the strength of the candidate S2 to the S1.
    - It gives a **boost** if S1 is significantly stronger than S2, which is expected behavior.
    - It applies a **penalty** if S2 is unexpectedly stronger than S1, which is often a sign of noise or an arrhythmia.
- **Vulnerability**: This check knows nothing about time. It would happily boost the confidence of two peaks that have a great amplitude ratio, even if they are a full second apart.
### S1-S2 Max Interval Logic: The "Timing" Check ⏱️
This logic looks at the **time gap** between the S1 and S2 peaks.
- **Purpose**: To answer the question, "Is the second sound happening quickly enough after the first to be part of the same cardiac cycle?"
- **Mechanism**: It measures the seconds between the candidate S1 and S2 peaks and compares it to a dynamic maximum allowed interval (`s1_s2_max_interval`). As heart rate increases, the time for the S1-S2 interval (systole) shortens, and this rule accounts for that.
- **Vulnerability**: This check knows nothing about amplitude. It would not penalize two peaks that are perfectly timed but where the "S2" is a massive noise spike that is 10x louder than the S1.
### Stability Logic: The "Historical Performance" Check 📈
The stability logic doesn't measure the time _between the current peaks_. Instead, it looks at the algorithm's **recent classification history** to answer the question, "How successful have I been at finding clean S1-S2 pairs over the last 20 beats?"
- **High Stability:** If the algorithm has successfully paired most of the recent beats (a high `pairing_ratio`), it becomes more confident and applies a small boost to the next attempt.
- **Low Stability:** If it has been failing to find pairs (e.g., classifying many peaks as "Lone S1" or "Noise"), it becomes more skeptical and applies a penalty to the next attempt.
This is a **macro-level feedback system**. It's about the algorithm's confidence in the overall rhythm quality based on its past performance, not the specific timing of the beat being analyzed _right now_.
### S1-S2 Interval Logic: The "Physical Reality" Check 🔬
The `s1_s2_max_interval` logic is a fundamental physiological rule applied at the **micro-level** to every single potential pair. It ignores past performance and asks a simple, physical question: "Are these two specific peaks close enough in time to possibly be from the same heartbeat?"
Even if the stability has been perfect for the last 100 beats, this rule will still penalize a pair if the time between them is too long, correctly identifying that they must be two separate beats.




- [x] Implemented
It would be great if the script could output user settings to the `Analysis_Summary.md` file so if we run the script again on the same file, It could first check to see if there is already a `Analysis_Summary.md` and pull the settings from there to begin the analysis. 





- [ ] Implemented
Let's implement a pairing_interval_expectation and adjust the confidence depending on how close it is. 
we can apply timing penalty/boost.
let's use a generic physiological based model where the expected interval between S1 and S2 would be 
0.28s @110bpm

Let's rename the boost and penalty to something more descriptive, like "pairing_amplitude_expectation"
so S1-S2 pairing confidence will be a function of 
- stability
- pairing_amplitude_expectation
	- amplitude penalty/boost 
- pairing_interval_expectation
	- timing penalty/boost 

wait, Technically, the S1-S2 interval is **not solely dependent on BPM**
Even if the heart rate is the same (90 bpm at rest vs. 90 bpm during recovery from exercise), the **S1-S2 interval would be shorter after exercise.**
wait... I'm confused. I need to gather more data about the exact physiology of the S1-S2 interval...


















> [!think]
we may need to tweak the EQ to get a better waveform



## The effects of using AI on my codebase:
After refactoring my code, my development speed has significantly slowed. I haven't been able to implement simple changes at all
the AI is "tweaking" My code has become so segmented every little change the AI makes a new helper function.
the AI is making helper functions for simple code changes that I will never reuse

THING INSIDE OF A THING INSIDE OF A THING
EVERY NEW THING NEEDS ITS OWN CONTAINER
EVERY NEW FEATURE NEEDS ITS OWN CLASS
every new addition needs a # Fallback 
AAAAAAAAAA

like pattern recognition is making the AI tweak 
my code has become a maze of abstractions because the AI aggressively applies "best practices" and design patterns. 
gotta re-reformat the code to remove these patterns to make the AI stop tweaking

- **Single Responsibility Principle (SRP):** It sees a function doing two things and immediately splits it into two helper functions. It does this over and over, resulting in fragmentation.
- **Don't Repeat Yourself (DRY):** It sees two slightly similar lines of code and creates a new, abstracted function to handle both, even if it makes the code harder to read.
- **Defensive Programming:** The "every new addition needs a # Fallback" is the AI trying to prevent errors. It adds error handling, type checking, and fallback logic for every conceivable edge case because that's what robust production software does. For a simple project, it's just clutter.


Instructions for AI: 
Do not remove any debugging code unless specified by the user
do not further abstract my code
do not further segment my code
do not over-engineer a solution, keep it simple

can you look at my code and try to identify areas that are too abstracted or segmented?  
there also might be too much Defensive Programming bloat in my code. can you help me identify areas where this could be happening?



> [!think]
config.py has 100 lines. Many of these could have sensible defaults and should be simplified.


> [!think]
> Cannot access attribute "to_numpy" for class undarray[Any, Unknown]" Attribute "to_numpyn is unknown  
> Cannot access attribute "to_numpy" for class "NDArraylUnknown]" Attribute "to_numpy" is unknown  
> Cannot access attribute "to_numpyn for class undarray[Any, Unknown]" Attribute "to_numpyn is unknown  
> 
> I've noticed that cursor linter errors like this happen often. This is probably because your python is using a different interpreter than the python version I'm using which has the libraries installed.  
> we probably need to disable linter handling in cursor. this is causing the AI to go crazy








## Heart Rate Kinetics
I've implemented these thoughts into `hr_reactivity.py`
I'm also using blender to visualize and calculate Heart Rate Kinetics

This is a broader term referring to the study of the time course of the heart rate response at the onset, during, and cessation of exercise. Researchers often model the increase in heart rate using exponential functions to determine the **time constant** (or half-time), which is a measure of how quickly the heart rate reaches a steady state or maximum during constant-intensity exercise. A smaller time constant generally indicates faster heart rate acceleration and better cardiovascular fitness.

- **Rate of Heart Rate Increase (rHRI):** In some exercise protocols, the maximum rate of heart rate increase during an incremental treadmill test is directly measured and used as a variable, particularly in research evaluating cardiorespiratory fitness. measured in units of beats per minute per second (bpm⋅s−1)

rHRI is calculated as the **first derivative maximum** of the heart rate data (in beats per minute) as a function of time (in seconds) during the transition from rest or low-intensity exercise to a higher workload.




I want to develop a python script to analyze smart watch data. The watch saves a recording which gives us data. We only have access to a graph of bpm/time.
Our goal is to calculate heart reactivity as a health metric given a graph of bpm/time. The input data is a .CSV file with (time in seconds, bpm)
Person's Age and resting bpm will be inputs by the user.

MaxHR = 220 - age
HRReserve = (MaxHR) - (RestingHR)

I want to calculate some metrics:
**Heart Rate Recovery (HRR)** this is calculated simply with the industry standard. (we may need to change the
**Heart Rate Acceleration (HRA)** this calculation requires a bit more thought as this requires detecting the start of exercise.
We can measure the time it takes to reach a certain percentage of the heart rate reserve (HRR) from rest.
the exercise start time must be automatically calculated given the bpm/time graph. 
#### **Heart Rate Recovery (HRR)**
- **What it is:** The BPM your heart rate decreases within some amount of time after peak exercise.
- **Why it's great:** A fast heart rate recovery indicates a strong, efficient autonomic nervous system. Your body quickly switches from the "fight or flight" (sympathetic) mode to the "rest and digest" (parasympathetic) mode. 
- **How to calculate it:** HRR = (Peak Heart Rate) - (Heart Rate after 1 minute of rest)
#### **Heart Rate Acceleration (HRA)**
This directly measures how quickly the heart rate ramps up at the onset of exercise.
**What it is:** The slope (in BPM per second or BPM per minute) of the heart rate curve during the initial phase of exertion.
**How to calculate it:**
we need to determine  
    1. time at start of exertion (t₁)
    2. time at peak of exertion (t₂)
    3. Heart rate at the start (HR₁)
    4. Heart rate at the peak/stable point (HR₂).
**HRA = (HR₂ - HR₁) / (t₂ - t₁)**

**What does HRA represent:** A steeper slope indicates a more rapid cardiovascular response, which is characteristic of good fitness. A steep slope should represent high autonomic reactivity. A sluggish rise can indicate poor conditioning or autonomic issues.
### **The Challenges:**
It's difficult to define t₁, the exact start of exertion. we might use a threshold, e.g., the point where HR first rises 5% above the resting baseline. or a combination of other detection methods.
Since we do not know when exercise starts or ends, all the rate data must be inferred from the bpm/time data. 
1. **Look for an Inflection Point:** On the graph, find the point where the HR curve definitively breaks upward from its resting baseline. You can use a threshold (e.g., 10% above resting) or a mathematical derivative (rate of change) to find where the slope significantly increases. so We'll look backwards from t_peak until we find a point where the heart rate is below (resting_bpm + 0.1 * HRRes) and the slope is not positive.
2. **Use a "Step Test":** The transition from rest to a fixed, repetitive workload is very clear on a graph, making it easy to measure the acceleration and cost.
3. some other methods??

Also, given we know the start time of exertion, we must determine what to consider as (t₂). we can consider (t₂) as when the heart rate reaches the highest peak, but note that during sustained exercise the heart rate increase might plateau resulting in a asymptotic graph that approaches steady state. If we measure (t₂) as the BPM peak, we will be considering a lot of extra time. For example, if bpm goes from 80 to 180bpm in 2 minutes, but from 180 to 200bpm in the following 5 minutes. we would be measuring (t₂) at 200bpm which would be 7 minutes. This would result in a much lower HRA Slope. maybe to solve this issue, we could calculate two HRA? 

We can use the point where the heart rate reaches a certain percentage (e.g., 90%) of the peak during the exercise period. so 200bpm is peak, 200*.9 = 180, so when bpm reaches 180 we consider that as (t₂). we can calculate
Then we can also calculate another HRA_peak where (t₂) is the peak bpm.



Our goal is to capture the moment of maximal HRA and HRR since it's likely that it will provide us with the most clean data.

The activity period might have multiple peaks. How should we handle this? If bpm decreases rapidly, but then increases again, then we can assume activity has resumed? should we record such a event and record the HRA slope and HRR slope? Maybe it would be a good idea to record every HRA and HRR slope and average them out and generate a average HRA/HRR for the file. If we decide to do this, we must make sure the change in heart rate must be board and ignore small timescale bpm fluctuations. We should have a "climbing the hill" algorithm. find the forest when you're inside trees. we need to determine the shape of the hill while we are a ant. a small temporal bump in the much more massive hill should be ignored if a larger trend can be uncovered. If not, then we can consider the largest discovered temporal hill to be the extent of the activity. Also, the recording may have multiple exercise periods resulting in multiple large hills. Therefore we must treat each large hill as its own exercise period and calculate the HRA and HRR for each one. 


Sometimes the peak is at the end of exercise, but sometimes the heart rate starts to decrease even during the last part of exercise. It might also be a good idea to include hard physiological limits. so for example, a 20yoF has a average HRR of 18bpm within the first minute of rest, but if the measured HRR is 8bpm and the calculated HRA high and resting heart rate is low. Then we can logically deduce that she is healthy and this is not a phase of halted exertion but rather continued exertion at a lower intensity which simply results in a steady heart rate decrease. 
We can find where the derivative becomes negative and stays negative for a while and mark that time as the start time for our HRR calculation. 

We could compute the derivative (dbpm) as the difference between consecutive bpm values (or over a window).
We can define a threshold for the derivative (e.g., 0.5 BPM per second as determined by age, gender, restingBPM, etc) to indicate the start of exercise.
We'll look for the first point before t_peak where the derivative exceeds 0.5 for at least 10 consecutive seconds. Then, t1 is the start of that sustained increase.

When dbpm is negative enough to be considered within the physiological norm if exertion has completely stopped, then we can consider that time to be the start time for our HRR calculation. 



inc_threshold = calculated as a function of max bpm (from age) and resting bpm
dec_threshold = similarly calculated 

so we can say
is_increasing = dbpm > inc_threshold
is_decreasing = dbpm < dec_threshold

We'll then look for contiguous segments where `is_increasing` is True (with a minimum duration, e.g., 30 seconds) and then followed by a segment where `is_decreasing` is True (with a minimum duration, e.g., 30 seconds). The exercise period is the increasing segment and the decreasing segment is the recovery. 

But note, during exercise, the HR might have short periods of decrease (if the intensity varies) so we might want to allow for small gaps. let's define a allowed discontinuity duration where a discontinuity in a increasing segment does not result in two adjacent increasing segments but rather one large increasing segment. 

t_peak = the time of the peak
hr_peak = the peak heart rate for a given exercise

We can find the start of the exercise (t_start) by going backwards from `t_peak` until we find a point where the HR is below `resting_hr + 0.1 * hr_reserve` OR the derivative is not positive (below `inc_threshold`) for a sustained period (e.g., 10 seconds). 

We can find the end of the exercise (t_end) by going forward from `t_peak` until we find a point where the derivative is negative (below `dec_threshold`) and remains negative for a sustained period (e.g., 10 seconds). 

We can compute two HRA:
- HRA_peak = (hr_peak - hr_start) / (t_peak - t_start)
- HRA_90 = (0.9 * hr_peak - hr_start) / (t_90 - t_start)

Since we are trying to capture the greatest HRA and HRR, we could also calculate t_end as the starting point for HRR. so we can have two HRR?
- HRR_ = hr_peak - hr_at_1min_after_t_end
- HRR_ = hr_at_t_end - hr_at_1min_after_t_end

We can compute the HRA and HRR for each exercise period and then calculate average HRA average HRR. We should also display the max HRA and HRR (most intense exercise). 

`hr_start` may also be extremely high since some users may start the recording at near max exertion so recordings don't necessarily start from rest. 


Give it a ttkbootstrap interface for the user to input the file path for the CSV and the input variables. No need to implement smoothing to the bpm data. 
Also use Plotly for visualization. 






Since we have no way of knowing if each exercise is maximal, we must assume based on the peak heart rate detected and the max possible heart rate for that age we can calculate a estimated exercise intensity%. 
intensity can also be calculated for sustained bpm for a long period of time? maybe this should be given a different name.






## Audio preprocesing

> [!say]
> I tried to denoise my file, but there's a specific low frequency hum in the audio
> I can visualize it in Fl studio and EQ it out of the file, but I want a automatic way to detect it and remove it. my existing denoise method does not handle this case well. 
> 
> Since it is low frequency, ~65Hz, it gets past my filters.
> Do you know of any audio denoising methods that can detect such a case?

- [x] Implemented
> [!think]
> I came a cross a concept named "[power spectral density](https://youtu.be/pfjiwxhqd1M?t=129)" idk what it is, but I hope it works something like this:
> we add up all the frequencies causing it to accumulate
> 
> then we target the highest frequency peak with a thin frequency band filter and EQ it out. Logically, the tallest frequency will be the hum since heartbeats are loud and intense, but very brief. also heartbeats have a much wider and spread out frequency spectrum. therefore the only tall frequency peak we should see is any constant noise










> [!say]
> I think there's a indirect issue with our prominence calculation. 
> the dynamic noise floor could be utilized better in combination with the Troughs Detected. 
> 
> The params I set for noise floor get a good average noise floor to act as normalization, but this idea fails when there is temporary noise. There should be a way to detect temporary noise by using the trough height, Raw Amp vs Noise Floor. 
> 
> the noise floor is calculated from our average tough height, but what if there is a temporary two second noisy moment where many trough heights are higher than usual. this would cause the trough height to separate from the noise floor. 
> 
> when we measure the prominence of a peak that is occurring the moment after the noise ends and compare it with the prominence of a peak just before it. the peak that occurs in a noisy region will have lower prominence since the tough height got pushed up. 

Temporary noise deviates from the noise floor:
[![|756x182](https://i.imgur.com/iDTD90X.jpeg)
During clean audio: `trough_amp ≈ noise_floor` During noisy audio: `trough_amp > noise_floor (sometimes significantly)`During clean audio: `trough_amp ≈ noise_floor` During noisy audio: `trough_amp > noise_floor`

- [x] Implemented
> [!think]
> what if we set a maximum trough height based on the nose floor? this simple rule will prevent transient noise from pushing up the tough height which keeps our prominence calculation more stable to transient noise.


> [!think]
> this discussion has had me thinking, what's the point of the prominence calculation in the first place? 
> If it doesn't exist we would measure everything against the noise floor, which is what's already happening during those areas with non temporary noise.
> 



> [!say]
to expand upon this idea, there is a case where there is a noise between S1 and S2, the S1-S2 pairing is marked as impossible due to this noise and the current peak gets marked as noise instead. To prevent this from happening it should look at the peak after the next peak to determine if that passes the standard confidence criteria.
- [ ] Implemented









> [!say]
> the S1-S2 pairing logic has a flaw. It never compares the current peak to the top 20% of recent recent_s1_prominences. If the current peak is too low to be S1. The logic currently implements the contractility ratio as compared to the next peak which is S2. But what if both peaks are low amplitude noise. This artificially creates the illusion of high relative contractility (relative to the suspected S2).
> 
> we should use a similar logic as we used for the Lone S1. # Penalty if <40% of reference S1 prominence (adaptive threshold)
- [x] Implemented





> [!say]
> Should we make a helper function that calculates intervals based on bpm?
> so it should calculate the R-R interval, the S1-S2 interval, the S2-S1 interval. at the assumed bpm.
- [x] Implemented

> [!think]
> wait, I think we can make this even more robust. what if we take the average of the past 50 S1-S2 intervals? 



























## Optimizations:
60% of the script's runtime is conversion time, which is fundamentally unavoidable because it's dominated by FFmpeg decoding the compressed MP4 audio stream
Attempting to optimize the remaining 40% of our script's runtime seems kinda silly in comparison...
**Decode time dominates because:**
- MP4 audio is compressed with complex psychoacoustic models (AAC=O(n²) decode)
- 18 minutes of 44.1kHz stereo AAC = ~120 MB → ~1.9 GB PCM
- FFmpeg must decode every frame sequentially




## Notes about the current state of my codebase

So far, our code has
**Technical debt**: 
Signs of feature creep (e.g., "kick-start" recovery mechanisms, cascade resets) indicate iterative problem-solving without refactoring

**Architectural Debt:**
- **God classes**: `PeakClassifier` (400+ lines) handles peak finding, confidence scoring, lookahead, kick-start logic—violating Single Responsibility Principle
- **Deep call stacks**: 6-stage pipeline with shared state dictionaries makes debugging challenging

**Code Quality:**
- ⚠️ Sparse inline comments in complex logic (e.g., `classify_peaks` loop)

**Things that might make the software more user friendly**:
- **Simplicity**: Abstract away parameters into "presets" for common scenarios




Actually I'm the developer of this software. I just wanted to know how I stack up in terms of a person with actual programming experience. I started coding because I needed a specific tool that I couldn't find anywhere so I had to make it myself. I'm the customer of my own code. 
You critiqued my code saying there's a lot of "clankerslop" (A term I made up, although I find it humorous). Here's what I have to say on your points:

**1. Parameter Soup**
You mentioned that The `config.py` file is the biggest smoking gun. **80+ parameters** with comments like "I Adjusted This✔" reveal someone tuning by trial-and-error, not first principles. I developed this software for my own personal use and since I can't really read code as well as a normal developer, I decided to make a config file where all the parameters are explained in English. I made it with the intention of reading that instead of the codebase because when I first started, I couldn't read any code. (I still can't but to a lesser degree). I want my code to be self documenting but I do admit you have a point when you say "The AI-generated docstrings describe _what_ code does, but never _why_ those choices were made." I didn't really put my thought process into comments so a lot of the things seem like magic numbers. 

You also mentioned "- `trough_rejection_multiplier: 10.0` - Why 10? Not 8 or 12? This is a magic number that _sounds_ precise but is just what made one test file work." Yeah, that's correct. You called me out on my workflow. I run the tests, then tune the parameters that result in the lowest number of errors and write "I Adjusted This✔" to remind myself that I adjusted the parameter before. 

**2. Architectural Overcompensation**
"The elaborate `PeakClassifier`, `UIMessageType` enums, and threading queues are **AI showing off patterns it learned from production code, applied to a problem that doesn't need them**. The developer accepted this complexity because they couldn't distinguish necessary architecture from AI embellishment."
Yeah I agree. Since I'm a beginner at coding, I have no way to tell if the AI is putting out a necessary architectural change or if it's overkill for my project. I can't distinguish necessary architecture from AI embellishment. 

**3. Hidden Performance Bombs**
"The "optimize long plots" flag is a **band-aid for AI-generated O(n²) algorithms**. An experienced developer would design for linear time from the start. The AI just kept adding features until it got slow, then added another feature to "fix" it."
Actually that's me. I came up with that logic. I kept adding features until it got slow, then added another feature to "fix" it. 

**4. Zero Validation Infrastructure**
You mentioned that there were no "unit tests" idk what unit tests are but I'm pretty sure I've been doing some form of unit testing in my workflow. To test the tool, I need to constantly verify if a change I made to the tool actually yields a improvement in its output so I have a bunch of test files that I feed it. I categorize each file based on difficulty. a easy file would be audio with no noise or variation in bpm. and a hard file would be a heartbeat with arrhythmia (which the tool isn't built to detect) or a noisy bit of audio. Every major version change, I save the outputs to a folder so I can compare progress over time. 






**Is my method of programming "industry standard?"**
I discovered "heuristic programming" or at least, that's what I think my programming style is. Basically, I test the code against my files and see when it makes a error and try to figure out a condition or filter to catch that case. Over time, my code evolved into what it is now. I'm not sure if this is standard practice or whatever but this is how I've been programming. Idk what to call my method but "heuristic programming" is what I've named it. I'm not sure if this programing method is the best way to do things but it's how my brain naturally thinks so I just put thoughts to paper and it resulted in the code we have. 

When I made version one of this tool, I didn't know many things about the dataset. Only by observing the data, did I come to understand the data the code needs to process. I basically made a tool to help me observe the data, then using my new discoveries, I tuned the tool to be better. I made a tool to study data and in the process, learnt about the data as I studied it. Of course the code would be much cleaner if I remade it with the knowledge I have now, but I feel like it's kinda unavoidable. You need the tool to help you analyze the data because you want to know more about your dataset. Then you discover that the tool is flawed and repeat. I think it's a natural workflow and kinda unavoidable. 

I formatted the debug to be easier for me to understand what's going wrong. whenever a bug occurred and I couldn't understand which part of the algorithm was failing I modified the debug to help me see better. The debug/logging for the code is what resulted from this workflow. 

I'm basically using logging to watch the algorithm think since I can't read the code. I read whatever the logging outputs to determine if the code is ordered the way I want it to be. 

"**9. Import Bloat** "import every possible dependency and silently degrade." Real engineers make conscious dependency choices."
yeah, I didn't know what the libraries do so I just hit accept on whatever cursor output.
Although, there are sources of bloat in the codebase. 
There are many parts of the code that I don't utilize because I implemented the feature to solve a specific issue I was having. But the feature didn't solve the issue and I just left it in the code without removing it.
I have since solve the issues using other methods in the code and I just never bothered to remove the unused portions. 
Specifically, the pyPCG denoising and tools which I never use. Also, the RMSSDc, SDNN, Peak Recovery Slope, Peak Exertion Slope, Exertion, and Recovery calculation and logic are flawed, incorrect, and I never finished implementing it. 

I'll probably remove the Peak Recovery Slope, Peak Exertion Slope, Exertion, and Recovery logic since I made a script for blender that can import the bpm/time.csv data and graph it in blender. I use Geo-Nodes to calculate the slopes. Since I'm using blender to analyze the data, there's no need for this code to be in my software anymore. 

the RMSSDc, SDNN calculations were only a proof of concept but I implemented these calculations a long time ago and they need to be redone. 







I want to set up a regression testing type of idea for some of the params in the config file. Currently I change a value, then run the code to see what changed. This algorithm is good enough now that errors are more rare to come by. Therefore I need to run a large batch of files to determine if a change actually made a positive impact or not.

When I batch run my dataset, I want the script to read my corrected output data and compare to its own output data to determine how many errors exists. This will allow me to quickly test a massive amount of data to fine tune parameters easier. Add auto validation by scanning for the `.wav_manually_Labeled_peaks.csv` file in the same folder as the input file and comparing the correct output to what the script outputs.






- now we need the ability to regenerate the bpm/time graph using the new data. 
- and add auto validation by scanning for the `.wav_manually_Labeled_peaks.csv` file in the same folder as the input file and comparing the correct output to what the script outputs.







Another idea, what if we analyze the bpm/time graph that the script calculated. There's already a trapezoidal artefact detection. This is in a similar idea. If there is two points that cause a slope of the bpm/time graph to be too steep for normal human physiology, then there must be something mislabeled.  
  
we should make a new metric like "hindsight confidence" or some other name, where the algorithm judges it own peak labels based off the resulting bpm/time graph. Then it should assign this confidence value to each labeled peak and if the algorithm has low confidence in a peak being labeled correctly, it should change its labeling and recalculate the bpm/time graph again. then reassign confidence. If confidence has increased then it will keep the new labeling.

But how can we implement such a idea? Maybe the script can look at the original confidence score 









- [x] implemented
helper function to model heart rate reactivity as a function of bpm. the idea is that as we get closer to a person's BPM max, the heart should increase slower as a function of time
to model physiological limits. so when bpm is low, bpm is allowed to increase. when bpm is high, bpm should not be allowed to increase as much. 
Power Law
$$\text{reactivity\_factor}(HR) = \left(\frac{HR_{max} - HR}{HR_{max} - HR_{rest}}\right)^C$$
Where $C$ is 2 (higher = sharper slowdown near max). leave C as a non tunable param but as a variable for future tuning maybe. 
https://www.desmos.com/calculator/pzaxdwfwfi


















# Excalidraw Data

## Text Elements
Kinetics of heart rate responses to exercise ^M6JWXfuQ

This can be used to identify different sections of sound ^C3JeGHn0

"modulation spectrum"
wait... This seems like exactly what I'm trying to do ^0RYo1dwr

Exploring the Role of the Modulation Spectrum in
Phoneme Recognition ^Pv7bDctf

https://github.com/scipy/scipy/blob/main/scipy/signal/_peak_finding_utils.pyx ^uQBGN0l5

my python script uses scipy.signal.find_peaks which is documented here: ^lYMfyJZv

here we have a peak right around four which should make sense because a typical speaking rate is about four syllables per second so we should see a peak in the envelope about one every 4 to five to six Hertz because there that's just the rate of modulation that we see.

you might want to divide up the spectrum into very clean octave frequencies like 1 2 4 8 16 Herz as done by Frederick Gallun and Pamela Souza in their classic 2008 paper Exploring the Role of the Modulation Spectrum in
Phoneme Recognition ^Li3BEkA2

in my scipy implementation, we might want to use .argrelmax(df.values, order = 20) to identify troughs ^PS3DYbkP

1. Rhythm-Based Correction
2. Resolve Adjacent S1 Conflicts
Keep Stronger Amplitude
3. Fix Rhythmic Discontinuities ^NrEG6w2S

1. Pass 1: Find Long Gaps
(missed beats)
2. Search for Missed S1-S2 Pairs
in noise-labeled peaks
3. Relabel Corrected Gaps

4. Pass 2: Find Short Conflicts
(adjacent S1s)
5. Resolve Short Conflicts
remove weaker peak
 ^5KazotiQ

Final Corrected S1 Peaks & Debug Info ^HzpSeENV

Yes ^DcyO9gXH

1. Run Preliminary Pass
with High-Confidence Threshold
2. High-Confidence PeakClassifier
3. Extract Anchor Beats
(S1 peaks from pass) ^n6PHHjY0

Estimate Global BPM from Median RR ^ZxVepXOF

Use Default BPM = 80.0
(Fallback) ^tSvclf8k

YES ^mdubgqLI

NO ^HAl1T6Zi

1. Calculate Audio Envelope
2. Find All Potential Troughs ^lIToDB7D

Calculate Dynamic Noise Floor 
& Sanitize Troughs ^3Sw4lB4e

Use Static Noise Floor
(Fallback) ^SDiXnB95

Find Raw Peaks Above Noise Floor ^UWtTMQqn

Yes ^93T8illu

No ^3HCiWlsa

Stage 3: Main Classification Loop ^0yAR2gxB

Stage 2: Preliminary Pass ^EAsqsCEH

Stage 1: Preprocessing ^hswoU9Im

No ^IpQOqtP9

No ^fyqgqYOz

Per-Iteration Kickstart Check
Override Pairing Ratio if Stuck ^rSueQawo

Detect Peak Time & Recovery Phase Boundaries ^cF68YBT7

Enough troughs?
(>= 5) ^4VBXp68W

Anchor beats >= 10? ^TWCkrr37

No ^WYsWHiZP

Stage 5: Final Metrics & Output ^PCZ8mH8k

yes ^VFvsdWa8

yes ^0fV6bDyY

No ^U1cphduw

Label as Noise ^WHkPk2zO

Label peak as Lone S1 ^nEDOs1S9

Label peak as Paired S1 and S2 ^pJR0JPo2

Classify Lone Peak ^BvWYzplu

Attempt S1-S2 Pairing ^5kdtBIz3

Update Long-Term BPM ^THRYTlV8

Return: s1_peaks, all_raw_peaks, analysis_data ^HANC32l7

Stage 4: Correction & Refinement ^FpmoeWMe

Weak Middle Peak?
(lookahead) ^VEjzU7dl

Evaluate Middle Peak Prominence & Interval ^z2aFgYTW

Pair Peaks 1 and 3, Mark middle peak as Noise ^0b0WBqIY

Get Current and Next Peak ^QRVtMOlF

More Peaks Remain? ^JyHoqqXm

Is Valid Lone S1? ^tgQ1Ukop

Cascade Reset Triggered? ^HKWgEmht

Label peak as Lone S1 ^Bb9iUApN

More Corrections Made?
(max 5 iterations) ^jKJDgjZl

Pairing Confidence High Enough? ^Wla2u74n

No ^ZgfOHL8l

No ^Welztzsz

Yes ^fyh0FXNK

Yes ^aIUSRa5d

Yes ^9feQ8Id8

Yes ^04ufW0F3

No ^jEtEkZbZ

## Element Links
MAOteNo0: https://www.eneuro.org/content/10/10/ENEURO.0197-23.2023

85HaS0TK: https://youtu.be/qp1G3a2g8r0?t=35

pTyS7sKQ: https://youtu.be/fsgTrsSVPts?t=250

qiDUZvSJ: https://physionet.org/content/challenge-2016/1.0.0/

elVr6vCy: https://github.com/mohammedlajam/Phonocardiogram-Classification

Vy6TOikC: https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/

QfvA2n1o: [[bpm analysis Logic Flow#Stage 5 Final Metrics & Output]]

fhD6DcOX: [[bpm analysis Logic Flow#Attempt S1-S2 Pairing]]

0aRbPdpX: [[bpm analysis Logic Flow#Classify Lone S1]]

M6JWXfuQ: https://www.researchgate.net/publication/19750910_Kinetics_of_heart_rate_responses_to_exercise#:~:text=A%20maximal%20intermittent%20treadmill%20test,mL%2Fkg%2Fmin).

C3JeGHn0: https://youtu.be/qp1G3a2g8r0?t=618

Pv7bDctf: https://sci-hub.ru/10.1097/AUD.0b013e31817e73ef

## Embedded Files
5af812902fe0104fbde6a75767f224dd8e01579d: [[Pasted Image 20250930133816_391.png]]

80686c0c2fda9020946fab7152d4bd923ad1f031: [[Pasted Image 20251001202857_910.png]]

%%
## Drawing
```compressed-json
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
```
%%