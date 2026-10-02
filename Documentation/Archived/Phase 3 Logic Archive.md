# Phase 3 Logic Archive
This file archives the pre-rewrite Phase 3 (Stages 4 & 5 correction) logic so it can be restored or referenced later.
## `pipeline.py` — `_refine_and_correct_peaks(...)` (archived)
```python
def _refine_and_correct_peaks(s1_peaks: np.ndarray, all_raw_peaks: np.ndarray,
                              analysis_data: Dict, audio_envelope: np.ndarray,
                              sample_rate: int, params: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Applies rhythmic and iterative contextual correction passes to refine S1 peaks.
    """
    logging.info("--- STAGES 4 & 5: Refining peaks with rhythmic and contextual correction ---")
    # STAGE 4: Simple rhythmic correction (e.g., remove beats that are too close)
    s1_peaks_rhythm_corrected = correct_peaks_by_rhythm(s1_peaks, audio_envelope, sample_rate, params)
    # Prepare data for the iterative pass
    dynamic_noise_floor = analysis_data['dynamic_noise_floor_series']
    current_debug_info = analysis_data["beat_debug_info"].copy()
    final_peaks = s1_peaks_rhythm_corrected
    # iterative correction loop
    # Safety cap: in practice 1-2 passes suffice; 5 prevents a runaway loop if
    # the corrections destabilize and re-introduce the same errors each time.
    max_iterations = 1
    for i in range(max_iterations):
        logging.info(f"Correction Pass Iteration {i + 1}...")
        new_peaks, new_debug_info, corrections_made = fix_rhythmic_discontinuities(
            s1_peaks=final_peaks,
            all_raw_peaks=all_raw_peaks,
            debug_info=current_debug_info,
            audio_envelope=audio_envelope,
            dynamic_noise_floor=dynamic_noise_floor,
            params=params,
            sample_rate=sample_rate
        )
        final_peaks = new_peaks
        current_debug_info = new_debug_info
        if corrections_made == 0:
            logging.info("Correction process stabilized. Exiting loop.")
            break
        else:
            logging.info(f"Made {corrections_made} corrections in iteration {i + 1}.")
    else:
        logging.warning("Correction process reached max iterations without stabilizing.")
    analysis_data["beat_debug_info"] = current_debug_info
    return final_peaks, analysis_data
```
## `correction.py` (archived full previous implementation)
```python
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from peak_utils import PeakType, _is_noise_debug
def correct_peaks_by_rhythm(peaks: np.ndarray, audio_envelope: np.ndarray, sample_rate: int, params: Dict) -> np.ndarray:
    """
    Refines a list of S1 peaks by removing rhythmically implausible beats.
    If two beats are too close together, the one with the lower amplitude is discarded.
    """
    # If we have too few peaks, correction is unreliable and unnecessary.
    if len(peaks) < 5:
        return peaks
    logging.info(f"--- STAGE 4: Correcting peaks based on rhythm. Initial count: {len(peaks)} ---")
    # Calculate the median R-R interval to establish a stable rhythmic expectation.
    rr_intervals_sec = np.diff(peaks) / sample_rate
    median_rr_sec = np.median(rr_intervals_sec)
    # Any interval shorter than this threshold is considered a conflict.
    correction_threshold_sec = median_rr_sec * params.get("rr_correction_threshold_pct", 0.6)
    logging.info(f"Median R-R: {median_rr_sec:.3f}s. Correction threshold: {correction_threshold_sec:.3f}s.")
    # We build a new list of corrected peaks. Start with the first peak as a given.
    corrected_peaks = [peaks[0]]
    # Iterate through the original peaks, starting from the second one.
    for i in range(1, len(peaks)):
        current_peak = peaks[i]
        last_accepted_peak = corrected_peaks[-1]
        interval_sec = (current_peak - last_accepted_peak) / sample_rate
        if interval_sec < correction_threshold_sec:
            # CONFLICT: The current peak is too close to the last accepted one.
            # We must decide which one to keep. The one with the higher amplitude wins.
            last_peak_amp = audio_envelope[last_accepted_peak]
            current_peak_amp = audio_envelope[current_peak]
            if current_peak_amp > last_peak_amp:
                # The current peak is stronger, so it REPLACES the last accepted peak.
                logging.info(f"Conflict at {current_peak/sample_rate:.2f}s. Replaced previous peak at {last_accepted_peak/sample_rate:.2f}s due to higher amplitude.")
                corrected_peaks[-1] = current_peak
            else:
                # The last accepted peak was stronger, so we DISCARD the current peak.
                logging.info(f"Conflict at {current_peak/sample_rate:.2f}s. Discarding current peak due to lower amplitude.")
        else:
            # NO CONFLICT: The interval is plausible. Add the peak to our corrected list.
            corrected_peaks.append(current_peak)
    final_peak_count = len(corrected_peaks)
    if final_peak_count < len(peaks):
        logging.info(f"Correction complete. Removed {len(peaks) - final_peak_count} peak(s). Final count: {final_peak_count}")
    else:
        logging.info("Correction pass complete. No rhythmic conflicts found.")
    return np.array(corrected_peaks)
def fix_rhythmic_discontinuities(s1_peaks: np.ndarray, all_raw_peaks: np.ndarray, debug_info: Dict,
                                  audio_envelope: np.ndarray, dynamic_noise_floor: pd.Series, params: Dict,
                                  sample_rate: int) -> Tuple[np.ndarray, Dict, int]:
    """
    Identifies and attempts to fix rhythmic discontinuities by re-evaluating misclassified peaks.
    """
    def log_debug(msg):
        logging.info(f"[Correction DEBUG] {msg}")
    # Exclude this many beats from each end of the recording; boundary beats
    # are unreliable rhythm anchors and skew the expected-interval estimate.
    margin = 3
    if len(s1_peaks) < margin * 2:
        log_debug(f"Skipping correction pass: Not enough S1 peaks ({len(s1_peaks)}) to apply a margin of {margin}.")
        return s1_peaks, debug_info, 0
    rr_intervals_sec = np.diff(s1_peaks) / sample_rate
    q1, q3 = np.percentile(rr_intervals_sec, [25, 75])
    iqr = q3 - q1
    stable_rr_intervals = rr_intervals_sec[
        (rr_intervals_sec > (q1 - 1.5 * iqr)) & (rr_intervals_sec < (q3 + 1.5 * iqr))]
    if len(stable_rr_intervals) < 1:
        log_debug("Not enough stable R-R intervals to determine median. Skipping correction.")
        return s1_peaks, debug_info, 0
    median_rr_sec = np.median(stable_rr_intervals)
    short_conflict_threshold_sec = median_rr_sec * params["rr_correction_threshold_pct"]
    long_conflict_threshold_sec = median_rr_sec * params.get("rr_correction_long_interval_pct", 1.7)
    log_debug(
        f"Median R-R: {median_rr_sec:.3f}s. Short Threshold: < {short_conflict_threshold_sec:.3f}s. Long Threshold: > {long_conflict_threshold_sec:.3f}s.")
    corrected_debug_info = debug_info.copy()
    peaks_to_add = set()
    corrections_made = 0
    # --- Pass 1: Look for LONG intervals (missed beats) ---
    log_debug(f"Checking for long intervals between beat {margin} and beat {len(s1_peaks) - margin}...")
    for i in range(margin, len(s1_peaks) - 1 - margin):
        s1_start_idx, s1_end_idx = s1_peaks[i], s1_peaks[i + 1]
        if (s1_end_idx - s1_start_idx) / sample_rate > long_conflict_threshold_sec:
            log_debug(f"Found LONG interval at {s1_start_idx / sample_rate:.2f}s. Investigating gap...")
            gap_candidates = [
                p for p in all_raw_peaks
                if s1_start_idx < p < s1_end_idx and _is_noise_debug(debug_info.get(p))
            ]
            for candidate_s1 in gap_candidates:
                if candidate_s1 in peaks_to_add: continue
                current_raw_idx = np.searchsorted(all_raw_peaks, candidate_s1)
                if current_raw_idx + 1 >= len(all_raw_peaks): continue
                candidate_s2 = all_raw_peaks[current_raw_idx + 1]
                if candidate_s2 >= s1_end_idx or not _is_noise_debug(debug_info.get(candidate_s2)): continue
                s1_strength = max(0, audio_envelope[candidate_s1] - dynamic_noise_floor.iloc[candidate_s1])
                is_strong_s1 = s1_strength > (
                            params["penalty_waiver_strength_ratio"] * dynamic_noise_floor.iloc[candidate_s1])
                is_ratio_plausible = (audio_envelope[candidate_s2] / (audio_envelope[candidate_s1] + 1e-9)) < params[
                    "penalty_waiver_max_s2_s1_ratio"]
                if is_strong_s1 and is_ratio_plausible:
                    gap_sec = (s1_end_idx - s1_start_idx) / sample_rate
                    s2_s1_ratio = audio_envelope[candidate_s2] / (audio_envelope[candidate_s1] + 1e-9)
                    log_debug(f"  - SUCCESS: Re-labeling S1/S2 pair at {candidate_s1 / sample_rate:.2f}s.")
                    corrections_made += 1
                    peaks_to_add.add(candidate_s1)
                    original_reason_s1 = corrected_debug_info.get(candidate_s1)
                    corrected_debug_info[candidate_s1] = {
                        "peak_type": PeakType.S1_CORRECTED_GAP.value,
                        "sections": [
                            {
                                "type": "correction_reason",
                                "text": (
                                    f"Promoted from Noise: strong S1 (strength {s1_strength:.3f}) found in long gap "
                                    f"({gap_sec:.3f}s > {long_conflict_threshold_sec:.3f}s). "
                                    f"S2/S1 ratio {s2_s1_ratio:.2f} (max {params['penalty_waiver_max_s2_s1_ratio']:.2f})."
                                ),
                            },
                            {"type": "original", "original_debug": original_reason_s1},
                        ],
                    }
                    original_reason_s2 = corrected_debug_info.get(candidate_s2)
                    corrected_debug_info[candidate_s2] = {
                        "peak_type": PeakType.S2_CORRECTED_GAP.value,
                        "sections": [
                            {
                                "type": "correction_reason",
                                "text": f"Promoted from Noise: paired with S1 at {candidate_s1 / sample_rate:.2f}s during gap correction.",
                            },
                            {"type": "original", "original_debug": original_reason_s2},
                        ],
                    }
                    break
    # --- Pass 2: Look for SHORT intervals (adjacent S1s) ---
    temp_s1_list = sorted(list(set(s1_peaks) | peaks_to_add))
    peaks_to_remove = set()
    log_debug("Starting SHORT interval check...")
    # Correctly iterate and compare adjacent beats
    for i in range(margin, len(temp_s1_list) - 1 - margin):
        beat_A_idx = temp_s1_list[i]
        beat_B_idx = temp_s1_list[i + 1]
        # Skip if either beat has already been marked for removal
        if beat_A_idx in peaks_to_remove or beat_B_idx in peaks_to_remove:
            continue
        interval_sec = (beat_B_idx - beat_A_idx) / sample_rate
        if interval_sec < short_conflict_threshold_sec:
            log_debug(
                f"Found SHORT interval of {interval_sec:.3f}s between beats at {beat_A_idx / sample_rate:.2f}s and {beat_B_idx / sample_rate:.2f}s. Resolving...")
            # Decide which beat to remove based on amplitude
            amp_A = audio_envelope[beat_A_idx]
            amp_B = audio_envelope[beat_B_idx]
            if amp_B > amp_A:
                log_debug(f"  - Removing weaker peak at {beat_A_idx / sample_rate:.2f}s.")
                reason_text = (
                    f"Removed: short interval {interval_sec:.3f}s < {short_conflict_threshold_sec:.3f}s "
                    f"and weaker amplitude ({amp_A:.0f} < {amp_B:.0f} at {beat_B_idx / sample_rate:.2f}s)."
                )
                original_debug_a = corrected_debug_info.get(beat_A_idx)
                corrected_debug_info[beat_A_idx] = {
                    "peak_type": PeakType.NOISE.value,
                    "sections": [
                        {"type": "correction_reason", "text": reason_text},
                        {"type": "original", "original_debug": original_debug_a},
                    ],
                }
                peaks_to_remove.add(beat_A_idx)
                corrections_made += 1
            else:
                log_debug(f"  - Removing weaker peak at {beat_B_idx / sample_rate:.2f}s.")
                reason_text = (
                    f"Removed: short interval {interval_sec:.3f}s < {short_conflict_threshold_sec:.3f}s "
                    f"and weaker amplitude ({amp_B:.0f} < {amp_A:.0f} at {beat_A_idx / sample_rate:.2f}s)."
                )
                original_debug_b = corrected_debug_info.get(beat_B_idx)
                corrected_debug_info[beat_B_idx] = {
                    "peak_type": PeakType.NOISE.value,
                    "sections": [
                        {"type": "correction_reason", "text": reason_text},
                        {"type": "original", "original_debug": original_debug_b},
                    ],
                }
                peaks_to_remove.add(beat_B_idx)
                corrections_made += 1
    # Construct the final list of S1 peaks after all corrections
    final_s1_peaks = [p for p in temp_s1_list if p not in peaks_to_remove]
    return np.array(sorted(final_s1_peaks)), corrected_debug_info, corrections_made
```







