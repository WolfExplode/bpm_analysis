the goal is to track bpm over time like a heart rate monitor, but using a recording of a patient's heartbeat instead. The resulting plot should accurately reflect fast changes in heart rate.
there's a issue with the script

heartbeats have S1 and S2. in the recording, S1 and S2 have similar amplitudes at lower bpm, but when exercising starts S1 gets much louder. this results in double counting at lower bpm.
The audio file's volume increases and decreases depending on stethoscope placement. this means the audio amplitude is not consistent. 

- maybe we can use the amplitude data before normalizing? It's logical that the volume increases as bpm increases, maybe we can use that to know if we are double counting beats when the bpm is supposed to be low or something?
we can measure the trend in volume, not the small outliers (if the user hits the microphone it will result in a large outlier). If the volume trends louder, we can assume bpm should also trend higher in the same timeframe. 

- another idea is, it's simple for a human to understand high vs low bpm from listening to the recording. if there was a way to tell the script the expected bpm range at that specific time, maybe we can get it to output a more accurate graph.

The core idea is that a heartbeat cycle isn't just one peak—it's a pattern. Specifically, it's a pair of sounds (S1 and S2) with a short time gap between them, followed by a longer gap before the next pair begins. I tried to teach the script to recognize this pattern.




[[Heart Sounds#Volume/Loudness of S1 and S2]]
A heartbeat cycle isn't just one peak—it's a pair of sounds (S1 and S2) with a short time gap between them, followed by a longer gap before the next pair begins. I tried to teach the script to recognize this pattern, but 

at lower bpm, the volume of S1 and S2 are very similar, but at higher bpm, S1 is significantly louder than S2. 

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

maybe we need a more clever way to handle peak detection?
	A heartbeat is composed of S1 and S2, two beats. 
	a peak is considered when there is a local maximum, but we must also take into account the value of the next local minimum (a trough) following a local maximum. 
	therefore, we can define a lower threshold amplitude that the waveform must reach for a local minimum to be considered.
	
	for example, If the algorithm detects a local min, but the amplitude of that min is still too large, (higher than the lower bound), then it must not consider that as a true local min.
	the the lower bound value must be calculated dynamically for noise in the recording. 
	after a trough is established, the next peak can either be S1 or S2 (since S2 can sometimes disappear, we can use the same logic as before to determine what it should be considered.. etc)...
	
	This approach is more advanced than what `scipy.signal.find_peaks` directly offers as a single parameter. `find_peaks` primarily focuses on the peaks themselves (their height, prominence, and separation) rather than explicitly validating the depth of the troughs between them against a dynamic noise floor.


*what if we have a dynamic bpm window? If the current detected bpm is 120, then the next bpm cannot differ too greatly from that value. the script will be forced to work its way up and down the bpm just like a real heart would.* 
*This should stabilize any crazy swings in bpm.* 


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
                peak_classifications[current_peak_idx] = f"Noise (Vetoed by larger subsequent peak at {next_peak_idx/sample_rate:.2f}s)"
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




## Breathing:
### unimplemented
graph the temporary noise by taking the troughs and comparing to the noise floor. then graphing that value
	the idea is, temporary noise might be from breathing so by doing this, we might be able to visualize breathing. 
What if we do a different EQ filter on the initial audio to scan for breathing sounds instead?


### unimplemented
Let's brainstorm based on some patterns that we can observe from the data. take a look at how S1 is substantially louder than S2, but then there is some temporary noise that's spread out. and when the noise increases, S1 amp decreases. 
we can also observe the inverse for S2. when noise increases S2 increases as well. At first I thought this was because the amplitude of S2 was "riding on top of" the noise but since S1 was decreasing this doesn't make sense.

Then I realized the noise must be from breathing since it's gradual fade in and out. 
Let's try to understand the heart's function and how the audio is being recorded. S1 must be decreasing since the stethoscope is further away from the chest at that time. but then why does S2 increase instead? the `trough-S2-trough` distance increases during breathing?
Is this normal physiology?

**Positional Shift:** As the diaphragm descends, the heart itself shifts slightly in the chest cavity. It can rotate or move downwards, changing its orientation relative to the fixed position of the stethoscope.
The heart's rotation might simultaneously move the pulmonic valve closer or into a better acoustic alignment with the stethoscope.

[![|1069x171](https://i.imgur.com/IFUuykq.png)]
by looking at the noise floor and toughs, we can clearly see the affect of breathing in the waveform. 
[![|1667x96](https://i.imgur.com/HuYGxhi.png)
how can we apply these observations to make our script more robust?


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




## other remarks
the scrip has several inaccuracies that don't fundamentally change the bpm estimation:
- there might also be recordings where the stethoscope is removed resulting in no heartbeats at all, do we have a case for this?

https://github.com/WolfExplode/bpm_analysis
here's my python script that plots heartbeat bpm/time so you don't need to do it by hand anymore.
I updated the code and it works much more reliably now. It's also much faster than before. 








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
    peak_classifications[current_peak_idx] = "Noise (High local noise confidence)"  
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



> [!think]
> regarding The idea of contractile force, what if we could display how hard the heart contracts by examining the amplitude deviation between S1 and S2 and mapping a trend. 
> 	nah but that depend on how the audio is recorded. just because S1 is louder than S2 in this recording, doesn't specifically mean that the heart is contracting strongly. 


> [!think]
I've noticed from the peak detection algorithm, It places the peak at the highest point, but sometimes a wave might look like a cut mountain and sometimes slanted. we basically need to capture the middle of the peak of a trapezoidal wave. 
[![|508x155](https://i.imgur.com/xkhzAQS.png)]
![|398x161](Y1rGxdpPAq4.png)
do you think this would be difficult to implement? would it even make the results more accurate?







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
How do we teach the script how to recognize this discontinuity? https://i.imgur.com/s2TXm9v.png

- Identify the discontinuity
	- A "notch" would start with a very large **negative** rate of change (the sudden drop).
	- Immediately following, there would be a very large **positive** rate of change (the jump back up).
	- (keep in mind, a notch could go up or down)
- Fix the discontinuity... the script should scrutinize the labels given to those peas at around the time of the discontinuity. It could shuffle around the labeling order, or make new labeling etc.
so fixing the discontinuity and identifying the discontinuity are not mutually exclusive steps. because during the fixing process, it's still trying to identify what caused the discontinuity. 

One way to fix this is to apply a large smoothing to the bpm like `BPM Trend (Belief)`. I like this idea but it can sometimes blunt the sharpness of the slope in the final graph. 

### An Analogy: A Car's Speed 🚗
Imagine you're tracking a car's speed every second.
- **Value Outlier:** If the car is on a highway with a speed limit of 65 mph and you get one reading of 10 mph, that's a value-based outlier. The **magnitude** (10 mph) is abnormally low compared to its neighbors (65, 66, 64...).
- **Slope Outlier:** If the speed goes from 60 mph to 120 mph in a single second, that's a slope-based outlier. While neither 60 nor 120 might be an outrageous speed for the car, the **transition** between them is physically impossible. The acceleration (the slope) is the outlier, not necessarily the speeds themselves.






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




> [!think]
what if we have the algorithm just make shit up? 
this idea goes beyond a `iterative_correction_pass` and instead, I propose 


> [!think]
what if we have a interactive way for the user to correct the peak labeling? then the script will calculate the bpm graph and other statistics after the user input




> [!think]
what if we apply EQ to the waveform to look for noise other than heartbeats. when the amplitude of medium frequency noise is high, we can conclude that there is also noise in the heartbeat recording
how can we use this logic to our advantage?



- [x] Implemented
I like the stability_confidence idea, but its implementation needs a bit of tweaking. when paring ratio reaches, 0 it's very difficult to begin paring again. since pairing increases pairing ratio, this is a negative feedback loop

what if it has a kick start mechanism to scan if the previous 3 lone S1s was directly followed by a peak marked as noise. This would give evidence that S2 has started to show up in the waveform again.
if it detects this, we can set pairing confidence to 50% or higher in anticipation of 








- [ ] Fixed
implemented a `strong_peak_override_ratio`
This should have been corrected by the R-R based rule
[![|485x326](https://i.imgur.com/QmQ13Lq.png)

"The last confirmed S1 beat was at 230.9801s. The peak in question at 231.3974s occurred **0.4173 seconds** later. This perfectly matches the expected S1-to-S1 interval for the Long-Term BPM (Belief) of ~150 BPM, which is 0.4 seconds. This made it a very strong S1 candidate."
this is a powerful idea, but our current script does not apply this type of linear thinking. our S1-S2 pairing confidence does apply this type of thinking.
how about we extend this gradient logic to the lone S1 algorithm?




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



It would be great if the script could output user settings to the `Analysis_Summary.md` file so if we run the script again on the same file, It could first check to see if there is already a `Analysis_Summary.md` and pull the settings from there to begin the analysis. 





