I'm plotting a frequency "profile" for S1 and S2 by taking the already labeled peaks and sampling around that peak with FFT to generate the frequency. Then adding up all the frequency values to average them up and plotting it. This gives me the pure frequency domain profile of my S1 and S2 beats. The goal is to determine the frequency separation for S1 vs S2 beats and use that to affect the algorithm. 

For pass 3, 

we will take `fft_max_peaks_per_type` amount of the highest confidence pairs and do FFT the same way we do for the plotting, run Normalization/alignment etc.. the exact same thing. then we automatically find the frequency separation for S1 vs S2 beats. How can we do this? I don't want to assume S1 is stronger or weaker than S2 at whatever frequency. the code should do this based on the calculated data. From the two profiles, define separation in a purely data-driven way. 

I don't want to restrict to the same range we plot, I want to do this between the range of 
10hz to 15,000hz. Due to the way PCG data was recorded, sometimes S1 or S2 may have some other kind of non standard frequency characteristic that's consistent across that recording. 
- [x] implemented







### Factors that affect S1-S2 interval, systolic interval:
Normal beat-to-beat variation in healthy subjects: ~7–11 ms
Systolic interval remains relatively constant across moderate heart rate changes 60–120 bpm
At 180–200 bpm, systole must shorten significantly to preserve minimal diastolic filling time, and the heart approaches the limits of mechanical efficiency.

**Higher contractility shortens systolic interval**
Because of more rapid ejection of the same stroke volume. shorter ejection time. 

**Higher afterload lengthens systolic interval**
The ventricle must generate higher pressure just to open the valve. Longer ejection time. So higher afterload means it opens later, so closes later. 

Pure epinephrine induced tachycardia, 160bpm, cause a slightly greater decrease in afterload compared to the same tachycardia, 160bpm, caused by exertion

**Increased preload lengthens systolic interval**
because the ventricle must eject a larger stroke volume. Frank-Starling mechanism, greater ventricular stretch leads to more forceful contraction. 

but higher contraction also means shorter systolic interval... 

**Inspiration increases right ventricular preload increasing systolic interval**
also increases A2-P2 split, split S2














> [!say]
> We're doing a lot of outlier detection logic in our code. can you check if outlier logic needs to be refactored so it's more reusable in our code? 

(we should do this after our codebase gets too messy)





## Pass 3
> [!say]
> ok I'm planning on adding a pass 3 to further refine our algoritm.
> I want the algorithm behave more holistically. instead of using greedy sequential Decision Making logic, pass 3 should have access to data from pass 2 which should make it have a more holistic view.
> I want you to brainstorm how this could be done, what logic should we put in pass 3 etc.
> consider what pass 2 is giving us that pass 1 does not, how can we use that extra data to be more confident in fixing mistakes...
> should we modify the main algortim to be looping/iterating etc?

After Pass2, we have access to the following evidence sources:
- Expected R-R from BPM curve
- Global S1 amplitude distribution
- S1-S2 interval history

We should build more infrastructure to help Pass 3 do its job. 
right now we have S1-S2 interval history but would 

> [!think]
> I think doing outlier removal on all of these evidence sources will be a good idea, but it's a case by case basis, how should we handle each feature?


I want the algorithm behave more holistically. Instead of using greedy sequential decision making logic, pass 3 should have access to data from pass 2 which should make it have a more holistic view.
before we do this, I think we need to build more infrastructure in our code to support it.

My idea is to run the main algorithm loop again but pass it more features which are calculated from pass2

But a fundamental issue still remains. 
we are still working off of "peak" data. I want to transition into using state labels (S1 / systole / S2 / diastole)
the issue with peaks is that we are saying that "S1 occurs at this time" 
instead, we should be saying that "S1 occurs during this time" 

That way we can begin to say things like "this section of time has the characteristics of S1" 


> [!think]
> when we are doing the outlier detection, we should probably record where in time that outlier occurs. because it's very likely that there's a error at that location so it needs to be examined more.


After pass 2, we have a good idea of which beats are likely correctly labeled and which ones are not. we also have a decently accurate BPM/time graph. 
we can say, we removed a outlier here, let's mark that peak and the peaks around it to be relabeled.
So for pass 3, for each peak we marked to be re-evaluated, we can say according to the bpm, the systolic interval, the spectral fingerprint of S1/S2 etc... we expect there to be a S1 here at this time and we expect it to sound like this... etc. Like, right here, at this time... for the bpm to make sense, there must be a S1 beat. now we look for it. then if we find it, we can say well there has to be a S2 beat right after at this time and look for that too. then we can say, well we found a S1 looking sound and a S2 looking sound, let's put a peak where the center of that probability distribution is and say that's our S1 and S2 peak. 

I think, thinking of it as a probability distribution based on known features from our pass2 is the correct way to think about it. 

I'm not sure if we should reuse the exact same algorithm as pass2 since that logic is made to be a greedy sequential decision making algorithm. 
maybe we should build a more custom pass 3 pipeline that utilizes some of the same code as pass2, but fundamentally different


> [!think]
> For the files where there's no S2 peak, we can still get a profile of what S1 and S2 sound like by only finding the location of the S1s, then just assuming the S2 location and scanning that area for the sound profile. 





> [!say]
> The pass 3 plot should be used to show all the data that's being fed into the pass 3 algorithm. I haven't decided how to build the pass 3 algorithm logic yet so I'm just making infrastructure to support the future feature. 
> 
> The expected systolic interval based on BPM deviates based on person to person. We should shift the calculated values up or down depending on the measured S1-S2 interval. So we calculate the values first, then shift them all by the same amount up or down.
> 
> We should plot this shifted graph as `Expected S1-S2 from BPM` in the pass 3 plot. 
> 
> But how do we determine how to shift the values? 
> I don't really want to explain the logic, but I'm going to do something that doesn't make sense right now. 
> We have code that determines when exertion and recovery exists. I want to average the `measured S1-S2 intervals` during times of exertion and average the`Expected S1-S2 from BPM` during those same times and find the difference. 
> so for example, the measured average might be 0.300s and the expected might be 0.350s. we just shift all `Expected S1-S2 from BPM` values by .05s so the average lines up. we ignore the data from times of recovery.
- [x] implemented

> [!think]
> I'm planning on using this to determine post exertion sympathetic tone. And heart contractility. I can see how the expected interval is higher than the measured post exercise. It's easy to see the two values separate as BPM recovers














> [!say]
> Let's continue brainstorming
> hmm. I was thinking about our fft implementation in pass3. about how we use it to fill in gaps and assume the missing data.
> the logic was:
> sometimes the very faint S2 sound may not get picked up as a peak. if we label enough peaks, we can be more and more confident where the missing peak should be located and go looking for the expected beat.
> we can say "based on the pattern we are already confident in, we can assume the location of where the s2 feature should express itself". then we can go looking and if we find what we are looking for, or some feature that resembles S2, we can say that must be S2
> let's expand on this logic to construct our continuous emissions. what do you think?

> [!think]
> so instead of looking locally at the existing peaks and seeing if that area looks like S1 S2 etc, we sweep the fft spectral profile across the entire audio file and generate a continuious confidence score output for the two spectral profiles
> this reminds me of convolution?



> [!think]
> to identify when stethoscope lifts cause noise, we can do the reverse of our post processing pass and look at all the frequency range outside where we might expect heartbeats to be. This will give us a inverse of our current Hubert envelope where the sum of both envelopes should in theory be the envelope of the entire audio since it encompasses all frequencies. Then if there is a lot of "not so heartbeat sounding noise at high amplitude at this location" then we know that it's likely to be noise. we can use this to generate our continuous noise score emission.
- [x] implemented










> [!say]
> There's a fundamental issue with our code. Since it's so constrained by our greedy algorithm, we have the following failure modes:
> - A bit of noise gets marked as S1 or S2 causing a suspiciously short interval, subsequently resulting in S1 and S2 labelings being flipped. This happens especially often when the S1 and S2 amplitudes are alike. This also causes a suspiciously long interval as the algorithm eventually has to mark S2 as noise down the line and subsequently therefore fixing the reversal of the labeling. 
> 
> - a suspiciously long state caused by S1 being marked as noise.
> - If we have a suspiciously long state, and there's a bit of noise where S1 or S2 could land, but the only thing preventing us from putting the state there, is the already existing label for the peak after. we should not take the labelings from pass2 that seriously then. it's likely that it was mislabeled and we should see if shifting the labelings by one peak fixes the suspiciously long states.
> 
> To do this, we can look at the calculated S1, S2, and Noise scores. see which score is highest if we were to say it is not noise. 
> 
> 
> instead of thinking about RR interval, we should be thinking in terms of state duration times. so not the entire heart cycle time. just the diastolie time etc.
> 
> | State duration anomaly | Likely cause                                                                                                                                                                  |
> | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
> | Systole too short      | A noise peak near S1 was promoted to S2, pushing S2 too close to S1                                                                                                           |
> | Systole too long       | Real S2 missed entirely, or S2 was poached as the next cycle's S1                                                                                                             |
> | Diastole too short     | The labeled S1_next is actually the S2 of this cycle, a flip or The labeled S1_next is actually a loud noise (so we probably hear the real S1 sometime right after the noise) |
> | Diastole too long      | Missing beat in the diastolic span                                                                                                                                            |
> 
- [ ] implemented











> [!say]
> I want to add some debug info to see what decisions the pass3 algoritm made to end up with the final state sequence.
> I want to hover my mouse over the state in state-strip-plot and it will display on hover info
> how can we do this?
> 
> remember, pass3 generates the state sequence from pass2 labels, then it applies the correction if the duration values don't make sense. 
> So I need the following pass 3 info:
> Expected state duration vs pass2 labeled durations
> S1: XXms vs XXms, 
> Systole: XXms vs XXms, 
> S2: XXms vs XXms, 
> Diastole: XXms vs XXms, 
> 
> And if any of these do not line up, then our pass 3 needs to correct it. so it will say:
> Systole: XXms vs XXms, Reasoning, Correction
> such as:
> Systole: XXms vs XXms, Systole too short, Duration updated to XXms by pushing the next S2 further away.
> Diastole: XXms vs XXms, Diastole too long, Duration updated to XXms by pulling the next s1 closer.
> 
> our html file will output the final state sequence and on hover, it will display the reasoning steps that the pass3 algorithm made to get the sequence to end up like this.
- [x] implemented


> [!say]
> how are the state durations generated in pass3? I think we need to update the S1 and S2 state durations. right now they always generate as 82ms since its's using our placeholder fixed-width event windows for S1 and S2 in Pass 3 we should use the hubert envelope or some kind of local intensity method to measure the actual length of the heart sound now. we just need to know when the S1 and S2 transients start and end, the bounds. 
> 
> what method should we use? and how should we implement it? help me brainstorm. 
> the sound should be parabola shaped right? like a transient, quick rise, quick fall, but ultimately like, gaussian shaped.
- [x] implemented

> [!say]
> we already do something similar in the codebase
>     # Peak position refinement: shift each raw peak to super-Gaussian-weighted center-of-mass (~100 ms window).
>     "peak_refine_window_ms": 150,        # Window (ms) around each peak for CoM; ~100 ms covers typical S1 extent.
>     "peak_refine_max_shift_ms": 40,      # Cap shift so noisy envelope cannot pull peak more than this (ms).
>     "peak_refine_super_gaussian_n": 5,   # Super-Gaussian exponent (flat top); higher = more flat top.
> can we reuse some of this logic?
- [x] implemented






The idea is, if the microphone is hit, it will produce loud sounds that can mess up beat detection. It's impossible for the heart to make one or a group of very loud beats and then go back to normal. this probably indicates that there's just noise.
Right now, the noise envelope is a good approximation of noise since noise is more likely to leak into the higher frequency range.
we should have a way to detect these noisy segments and mark them. store these noisy states somewhere, like how we currently store `S1 / systole / S2 / diastole` states.
your thoughts?
- [x] implemented



ok, now we that have a way to detect when a noisy event happens. This means that the labeled states around that duration of time is not accurate or reliable. I want to remove them and regenerate them by using the bpm/time data.
When we generate the bpm/time data, there will be gaps of nothing because we just removed the staes. we can fill in the gaps by linearly interpolating the bpm/time data. then we regenerate the states using this updated bpm/time data by just spacing the correct states to fill the gap. S1 / systole / S2 / diastole.
remember, we need to make sure we maintain the heartbeat cycle. so it will be invalid if we end up with S2(the last state we generate inside the noisy region), S1(outside the noisy region so we know this is accurate) etc...
cuz then we are skipping diastole. we need to maintain the correct sequence. so if it doesn't fit, we may need to stretch the timings or squash the timings a bit. like imagine we just take the entire sequence and scale it to best fit.
the logic for this might be tricky
we should implement this logic in pass 3. I want to implement this logic before Missing-beat insertion (S1 only) logic.
I didn't explain that the best... do you know what I mean?
- [x] implemented




> [!say]
> shit. how do I get the codebase to align with my intended architecture...
> do you fully understand my original intention? it seems like we need to refactor the codebase to align better with what I intended.
> cuz the entire idea is to generate the state sequence. Then correct it. how can we correct soemthing we haven't generated yet.. our current code makes no sense. It's a marvel of AI how we have generated absolute slop to fool me into thinking the codebase is structured in a logical way. shame on me for not double checking what the AI wrote I guess 😭


ok now let's think, how can you implement the logic so it aligns with my intentions
my intentions, pass3 noise repair should:
- scan for when any part of a state enters the noisy region.
- remove the states
- rebuild the sequence
- [x] implemented

let's brainstorm:
now it's time to rebuild the sequence, how?
we are rebuilding it from the assumed bpm. so we just read the durations and fill in.
but what if the noisy segment is at the start of the file, we need to build backwards.
well we need to make sure:
- the rebuilt sequence fits inside the empty gap
- the sequence lines up with where it re continues after the noisy segment ends. so it's... you know... a proper sequence.
- so if it cannot fit a full sequence, S1 -> systole -> S2 -> diastole -> S1... within that gap then it can either scale the timings a little bit to fit. or remove one cardiac cycle, S1 -> systole -> S2 -> diastole, from the sequence and scale up to see if that fits better etc...
- [x] implemented


Ah I see another issue. when we reconstruct the bpm (Pass 3), if the noisy segment is at the start of the file, there will be no bpm data for that segment of time.
hmm, we can probably use simple logic. we just don't remove the bpm/time data if a noise segment is at the start of a file.
- [x] implemented






> [!say]
> our pass3 gap fill logic should be more robust. help me brainstorm.
> sometimes there's just a very quiet s1 beat that got marked as noise.
> sometimes the beat is so quiet that the peaks detection algorithm didn't even label it as a peak in the first place.
> how can we fix this failure mode?
> 
> due to how gap detection works, there shouldn't be already-known raw peaks in the gap (make a comment noting this)
> Scoring , simple, closest in time and tie-break higher envelope amplitude (also make a comment noting this, due to how the peaks detection algorithm works, there shouldn't be any peaks directly next to each other)
> let's do snap anchors, then keep scale as-is, use simple logic. we can make it more robust later if needed. 
> if gap starts after a kept S1, then yeah, we don't move that s1. we can snap the corresponding S2 though, that does make sense. 
> 
> yeah I want Boundary-aligned snapping so the state timeline visually/structurally agrees with the snapped peak.
> snapping window of 80ms is fine, it's the default s1_nominal_sec. 
> "it does start to interfere with the rebuild geometry (phase widths, overlap/ordering, and how “exact fill” lands at gap_hi), so it’s easier to accidentally introduce weird edge cases." yeah, I want to implement the snapping in such a way that it doesn't generate any overlapping states by design. instead of thinking of it like snapping maybe like "shifting"
> 
> yes, the boundaries in the state timeline represent durations, not centers. if the codebase doesn't make this explicit, add a comment noting that as well. 
> yes, treat it as “I am choosing where the S1 (or S2) center is”, then derive S1/S2 windows centered on that peak. 
- [x] implemented



there's another failure mode. the case when the first s1 doesn't happen until it's outside the snap radius, in this case 140ms further away. so it's just a real but delayed beat for whatever reason. when we generate the state sequences, we end up with a extra cardiac cycle to to this anomaly.
feels like a edge case, but we can use this opportunity to make our code more robust.
I got a idea. What if we run the peaks detection algorithm two times in pass3?
we can run it with pass3_gap_peak_detection_quantile_insensitive = 70
then pass3_gap_peak_detection_quantile_sensitive = 50
so we get two sets of detected sets of peaks.
one is likely to only contain prominent peaks that are real s1 or s2.
one is more likely to contain noise or potentially a indication that there could be a peak there we missed. but the real reason we are doing the more sensitive pass is to scan for anything at all. like literally if there's anything there at all.
in the case where the more sensitive pass doesn't generate any new peaks at that location, we can be confident that we should not be generating a s1 state there. does that make sense?
- [x] implemented




