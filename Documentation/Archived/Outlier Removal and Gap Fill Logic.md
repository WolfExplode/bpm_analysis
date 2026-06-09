# Outlier Removal and Gap Fill Logic
This document holds the code for **timing outlier removal**, **amplitude outlier removal**, and **gap fill (add missing beats)** so it can be re-applied after a revert.
**Order of operations:** timing outliers → amplitude outliers → gap fill. All run on the preliminary-pass anchor beats (or any peak list + raw peaks for gap fill).

---
## 1. Config (add to `config.py` in the correction section)
```python
    "rr_outlier_local_window": 5,             # Number of R-R intervals on each side of a peak for timing-outlier context (boundaries skipped).
    "amplitude_outlier_min_ratio": 0.40,      # Peak is amplitude outlier if < this fraction of median amplitude of 5 nearby peaks each side.
    "gap_fill_max_duration_sec": 5.0,         # Only fill gaps (add missing beats) if longest R-R is <= this; longer gaps are left as-is.
```
---

## 2. `correction.py` — Add these three functions
Place after the existing imports and before `correct_peaks_by_rhythm`. No new imports needed (`numpy`, `logging`, `typing` already used).
### 2.1 Timing outlier removal
```python
def remove_timing_outliers(peaks: np.ndarray, sample_rate: int, params: Dict) -> Tuple[np.ndarray, int]:
    """
    Removes timing outliers (extra beats) by comparing local RR context.
    For each interior peak i, we take 5 RR intervals before (ending at i-1) and 5 after (starting at i+1).
    If removing peak i (replacing the two intervals it creates with one merged interval) is closer
    to the median of that local context, peak i is marked as an outlier. Iterates until no removal.
    Boundary peaks are skipped (not enough context). Returns (filtered_peaks, n_removed).
    """
    window = int(params.get("rr_outlier_local_window", 5))
    min_peaks = 2 * window + 3
    if len(peaks) < min_peaks:
        return peaks, 0
    current_peaks = list(peaks)
    total_removed = 0
    while True:
        n = len(current_peaks)
        if n < min_peaks:
            break
        times = np.array(current_peaks, dtype=float) / sample_rate
        rr_sec = np.diff(times)
        best_i = -1
        best_improvement = -1.0
        for i in range(window + 1, n - window - 1):
            rr_back = rr_sec[i - 1]
            rr_fwd = rr_sec[i]
            rr_merged = rr_back + rr_fwd
            local_before = rr_sec[i - 1 - window : i - 1]
            local_after = rr_sec[i + 1 : i + 1 + window]
            local_rrs = np.concatenate([local_before, local_after])
            rr_local = np.median(local_rrs)
            d_with = max(abs(rr_back - rr_local), abs(rr_fwd - rr_local))
            d_without = abs(rr_merged - rr_local)
            if d_without < d_with:
                improvement = d_with - d_without
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_i = i
        if best_i < 0:
            break
        removed_time = current_peaks[best_i] / sample_rate
        current_peaks.pop(best_i)
        total_removed += 1
        logging.info(
            f"Timing outlier removed at {removed_time:.2f}s (local median RR context; improvement {best_improvement:.3f}s)."
        )
    if total_removed > 0:
        logging.info(f"Timing outlier removal: {total_removed} peak(s) removed. Final count: {len(current_peaks)}")
    return np.array(current_peaks), total_removed
```
### 2.2 Amplitude outlier removal
```python
def remove_amplitude_outliers(
    peaks: np.ndarray, audio_envelope: np.ndarray, sample_rate: int, params: Dict
) -> Tuple[np.ndarray, int]:
    """
    Removes amplitude outliers (weak/noise peaks) using the same window as timing: 5 peaks before, 5 after.
    If a peak's amplitude is less than min_ratio (default 40%) of the median amplitude of those 10 nearby peaks,
    it is removed. Iterates until no removal. Boundary peaks are skipped. Returns (filtered_peaks, n_removed).
    """
    window = int(params.get("rr_outlier_local_window", 5))
    min_ratio = float(params.get("amplitude_outlier_min_ratio", 0.4))
    min_peaks = 2 * window + 1
    if len(peaks) < min_peaks:
        return peaks, 0
    current_peaks = list(peaks)
    total_removed = 0
    while True:
        n = len(current_peaks)
        if n < min_peaks:
            break
        amps = np.array([audio_envelope[p] for p in current_peaks], dtype=float)
        worst_i = -1
        worst_ratio = 2.0
        for i in range(window, n - window):
            local_before = amps[i - window : i]
            local_after = amps[i + 1 : i + 1 + window]
            local_amps = np.concatenate([local_before, local_after])
            median_amp = np.median(local_amps)
            if median_amp <= 0:
                continue
            ratio = amps[i] / median_amp
            if ratio < min_ratio and ratio < worst_ratio:
                worst_ratio = ratio
                worst_i = i
        if worst_i < 0:
            break
        removed_time = current_peaks[worst_i] / sample_rate
        current_peaks.pop(worst_i)
        total_removed += 1
        logging.info(
            f"Amplitude outlier removed at {removed_time:.2f}s ({worst_ratio:.2%} of local median)."
        )
    if total_removed > 0:
        logging.info(f"Amplitude outlier removal: {total_removed} peak(s) removed. Final count: {len(current_peaks)}")
    return np.array(current_peaks), total_removed
```











### Gap fill (add missing beats)
```python
def add_missing_beats_in_gaps(
    peaks: np.ndarray,
    all_raw_peaks: np.ndarray,
    audio_envelope: np.ndarray,
    sample_rate: int,
    params: Dict,
) -> Tuple[np.ndarray, int]:
    """
    Fills gaps where beats were likely missed. Finds the largest R-R interval (gap). If <= max_duration_sec,
    estimates expected RR from 5 intervals before and 5 after, computes how many beats are missing, places
    expected times evenly in the gap, and for each picks the closest raw peak in the gap (tie: higher amplitude).
    Iterates until no fillable gap remains. Boundaries (not enough context) are skipped. Returns (peaks, n_added).
    """
    window = int(params.get("rr_outlier_local_window", 5))
    gap_max_sec = float(params.get("gap_fill_max_duration_sec", 5.0))
    min_peaks = 2 * window + 2
    if len(peaks) < min_peaks:
        return peaks, 0
    current_peaks = list(peaks)
    total_added = 0
    while True:
        n = len(current_peaks)
        if n < min_peaks:
            break
        times_sec = np.array(current_peaks, dtype=float) / sample_rate
        rr_sec = np.diff(times_sec)
        i_gap = int(np.argmax(rr_sec))
        gap_duration = rr_sec[i_gap]
        if gap_duration > gap_max_sec:
            break
        if i_gap < window or i_gap > n - 1 - window - 1:
            break
        local_before = rr_sec[i_gap - window : i_gap]
        local_after = rr_sec[i_gap + 1 : i_gap + 1 + window]
        expected_rr = float(np.median(np.concatenate([local_before, local_after])))
        if expected_rr <= 0:
            break
        n_missing = max(0, int(round(gap_duration / expected_rr) - 1))
        if n_missing < 1:
            break
        gap_start_samp = current_peaks[i_gap]
        gap_end_samp = current_peaks[i_gap + 1]
        raw_in_gap = [p for p in all_raw_peaks if gap_start_samp < p < gap_end_samp]
        if not raw_in_gap:
            break
        t_start = gap_start_samp / sample_rate
        t_end = gap_end_samp / sample_rate
        expected_times = [
            t_start + (k + 1) * (t_end - t_start) / (n_missing + 1) for k in range(n_missing)
        ]
        to_add = []
        remaining = list(raw_in_gap)
        for t in expected_times:
            if not remaining:
                break
            best = remaining[0]
            best_dist = abs(best / sample_rate - t)
            best_amp = audio_envelope[best]
            for p in remaining[1:]:
                d = abs(p / sample_rate - t)
                a = audio_envelope[p]
                if d < best_dist or (d == best_dist and a > best_amp):
                    best, best_dist, best_amp = p, d, a
            to_add.append(best)
            remaining.remove(best)
        if not to_add:
            break
        current_peaks = sorted(set(current_peaks) | set(to_add))
        total_added += len(to_add)
        logging.info(
            f"Gap fill: added {len(to_add)} beat(s) in gap at ~{t_start:.2f}s (gap {gap_duration:.2f}s, expected RR {expected_rr:.3f}s)."
        )
    if total_added > 0:
        logging.info(f"Gap fill: {total_added} peak(s) added. Final count: {len(current_peaks)}")
    return np.array(current_peaks), total_added
```








> [!say]
> I think we can implement this logic for pass3.
> I think there should be a location in the codebase to store labeled confidence amounts
> 
> like for detected peak index #40, pass 2 labeled it as S1, and we store the associated confidence for S1. and we also store the associated confidence that the peak is S2. we only labeled it as S1 because among the 3 options, S1,S2,Noise, it has the highest confidence value after pass 2. but if we can store our confidence values in this way, we can potentially be more holistic in our thinking. we can retroactively relabel a peak from noise to S1 for example.
- [x] implemented 


















> [!think]
> do a post processing pass on the "BPM (Pass1)"?
> what if we clamp the slope of the graph to be no steeper than what is physiological possible?











> [!say]
> let's refine my idea. I previously implemented a way to store the confidence amounts for the 3 features, S1, S2, Noise
> 
> I want to use the high confidence S1 labels and high confidence S1-S2 pairs from pass2 to essentially becomes our new anchors where pass 3 can work off of.
> 
> The primary issue in our codebase right now is that it's written to be entirely greedy and forward looking. 
> 
> I'm not sure if "fill missing gaps" is the correct way to be thinking about this. 
> more like "use high confidence labels to 
> I want pass 3 to act more like a correction pass that acts more holisticly. 




> [!say]
> The primary issue in our codebase right now is that it's written to be entirely greedy and forward looking. To get Pass 2 to spit out equally weighted and holistic confidence emissions for the 3 features, S1,S2,Noise, we would need to fundamentally rewrite Pass2 logic. I don't think it's a good idea to do so. so we must live with the caveat that these initial confidence values may have a large degree of "hindsight blindness"
> 
> therefore I think I want to treat pass3 as a logical gap between our forward greedy algoritm and our more holistic venterbi state machine, which may eventually be pass4
> 
> the previous pass's BPM data is the algorithm's best estimate (so far) of what the bpm is. we can use the smoothed bpm graph to assume the location of beats. 
> we need to use a combination of the most confident labelings and the assumed bpm to "look for where the expected peak should be"
> we can say "based on the pattern we are already confident in, we can assume the location of where the s2 feature should express itself". then we can go looking and if we find what we are looking for, or some feature that resembles S2, we can say that must be S2










