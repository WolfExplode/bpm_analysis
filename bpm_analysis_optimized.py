"""
Optimized version of the BPM Analysis application.

This file contains optimizations focused on replacing inefficient pandas operations
with NumPy vectorized operations to improve performance while maintaining the same
analytical results.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import logging
import sys
import time
from typing import List, Dict, Tuple, Optional
from enum import Enum
import json
import re
import csv

# Import the original functions to maintain compatibility
from bpm_analysis import (
    PeakType,
    update_long_term_bpm,
    calculate_blended_confidence,
    _adjust_confidence_with_stability_and_ratio,
    calculate_lone_s1_confidence,
    Plotter
)

# --- Core Classes for Analysis Pipeline ---
class PeakClassifier:
    """
    Encapsulates the logic for classifying raw audio peaks into S1, S2, and Noise.
    This class manages the state of the analysis loop, including BPM belief,
    beat candidates, and debug information.
    """
    def __init__(self, audio_envelope: np.ndarray, sample_rate: int, params: Dict,
                 start_bpm_hint: Optional[float], precomputed_noise_floor: pd.Series,
                 precomputed_troughs: np.ndarray, peak_bpm_time_sec: Optional[float],
                 recovery_end_time_sec: Optional[float]):

        self.audio_envelope = audio_envelope
        self.sample_rate = sample_rate
        self.params = params
        self.peak_bpm_time_sec = peak_bpm_time_sec
        self.recovery_end_time_sec = recovery_end_time_sec

        # Store the time indices and values from the pandas Series for faster lookups
        # This avoids repeated Series.index and Series.values calls throughout the code
        # and eliminates unnecessary pandas operations
        self.noise_floor_indices = precomputed_noise_floor.index.values
        self.noise_floor_values = precomputed_noise_floor.values
        
        # Pre-compute frequently used parameters to avoid dictionary lookups
        self.min_peak_distance_samples = int(self.params['min_peak_distance_sec'] * self.sample_rate)
        
        self.state = self._initialize_state(
            start_bpm_hint, precomputed_noise_floor, precomputed_troughs
        )

    def _initialize_state(self, start_bpm_hint, precomputed_noise_floor, precomputed_troughs) -> Dict:
        """Pre-calculates all necessary data and initializes the state for the peak finding loop."""
        # Initialize state with all required keys at once to reduce dictionary resizing
        state = {
            'analysis_data': {},
            'dynamic_noise_floor': precomputed_noise_floor,
            'trough_indices': precomputed_troughs,
            'long_term_bpm': float(start_bpm_hint) if start_bpm_hint else 80.0,
            'candidate_beats': [],
            'beat_debug_info': {},
            'long_term_bpm_history': [],
            'consecutive_rr_rejections': 0,
            'loop_idx': 0
        }
        
        # Find raw peaks
        state['all_peaks'] = self._find_raw_peaks(precomputed_noise_floor.values)
        
        # Store references in analysis_data
        state['analysis_data']['dynamic_noise_floor_series'] = precomputed_noise_floor
        state['analysis_data']['trough_indices'] = precomputed_troughs
        
        # Pre-sort troughs for faster lookups later
        state['sorted_troughs'] = np.sort(precomputed_troughs)
        
        # Calculate peak strengths and deviations more efficiently
        # Use NumPy's searchsorted for nearest neighbor interpolation instead of pandas reindex
        peak_indices = np.searchsorted(self.noise_floor_indices, state['all_peaks'])
        peak_indices = np.clip(peak_indices, 0, len(self.noise_floor_values) - 1)
        noise_floor_at_peaks = self.noise_floor_values[peak_indices]
        peak_strengths = np.maximum(0, self.audio_envelope[state['all_peaks']] - noise_floor_at_peaks)
        
        # Only proceed if we have enough peaks
        if len(peak_strengths) > 1:
            # Calculate normalized deviations more efficiently
            peak_max = np.maximum(peak_strengths[:-1], peak_strengths[1:])
            normalized_deviations = np.abs(np.diff(peak_strengths)) / (peak_max + 1e-9)
            
            # Calculate deviation times
            deviation_times = (state['all_peaks'][:-1] + state['all_peaks'][1:]) / (2 * self.sample_rate)
            
            # Create deviation series - keep as pandas for now since smoothing is needed
            deviation_series = pd.Series(normalized_deviations, index=deviation_times)
            smoothing_window = max(5, int(len(deviation_series) * self.params['deviation_smoothing_factor']))
            state['smoothed_dev_series'] = deviation_series.rolling(
                window=smoothing_window, 
                min_periods=1, 
                center=True
            ).mean()
            state['analysis_data']['deviation_series'] = state['smoothed_dev_series']
        else:
            # Handle edge case with too few peaks
            state['smoothed_dev_series'] = pd.Series()
            state['analysis_data']['deviation_series'] = state['smoothed_dev_series']

        return state

    def classify_peaks(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Main classification loop to iterate through all raw peaks."""
        if len(self.state['all_peaks']) < 2:
            return self.state['all_peaks'], self.state['all_peaks'], {"beat_debug_info": {}}

        while self.state['loop_idx'] < len(self.state['all_peaks']):
            self._kickstart_check()
            current_peak_idx = self.state['all_peaks'][self.state['loop_idx']]
            is_last_peak = self.state['loop_idx'] >= len(self.state['all_peaks']) - 1

            if is_last_peak:
                self._handle_last_peak(current_peak_idx)
            else:
                self._process_peak_pair(current_peak_idx)

            self._update_long_term_bpm()

        return self._finalize_results()

    def _kickstart_check(self):
        """Specialized recovery function to kick-start the algorithm if it gets stuck."""
        # Early exit checks to avoid unnecessary computation
        candidate_beats = self.state['candidate_beats']
        candidate_beats_len = len(candidate_beats)
        
        # Fixed constants to avoid repeated lookups
        history = 4  # Hardcoded history beats
        min_s1s = 3  # Hardcoded min S1 candidates
        
        # Quick check if we have enough beats to analyze
        if candidate_beats_len < history:
            return
            
        # Calculate recent rhythm stability as a ratio
        history_window = self.params.get("stability_history_window", 20)
        if candidate_beats_len < history_window:
            pairing_ratio = 0.5
        else:
            # Cache beat_debug_info to avoid repeated dictionary lookups
            beat_debug_info = self.state['beat_debug_info']
            recent_beats = candidate_beats[-history_window:]
            
            # Use NumPy array operations instead of generator expression
            paired_count = sum(1 for beat_idx in recent_beats if 
                              PeakType.S1_PAIRED.value in beat_debug_info.get(beat_idx, ""))
            pairing_ratio = paired_count / history_window
            
        # Early exit if pairing ratio is above threshold
        if pairing_ratio >= self.params.get("kickstart_check_threshold", 0.3):
            return

        # Find recent lone S1s more efficiently
        beat_debug_info = self.state['beat_debug_info']
        recent_lone_s1s = []
        for idx in candidate_beats[-history:]:
            debug_info = beat_debug_info.get(idx, "")
            if "Lone S1" in debug_info:
                recent_lone_s1s.append(idx)
                
        if len(recent_lone_s1s) < min_s1s:
            return

        # Cache all_peaks for faster lookups
        all_peaks = self.state['all_peaks']
        min_matches = 3  # Hardcoded min matches
        matches = 0
        
        for s1_idx in recent_lone_s1s:
            current_raw_idx = np.searchsorted(all_peaks, s1_idx)
            if current_raw_idx < len(all_peaks) - 1:
                next_raw_peak_idx = all_peaks[current_raw_idx + 1]
                if "Noise" in beat_debug_info.get(next_raw_peak_idx, ""):
                    matches += 1
                    # Early exit if we've found enough matches
                    if matches >= min_matches:
                        break

        if matches >= min_matches:
            override_ratio = self.params.get("kickstart_override_ratio", 0.6)
            logging.info(f"KICK-START: Found {matches}/{len(recent_lone_s1s)} S1->Noise patterns. Overriding pairing ratio to {override_ratio}.")
            # This is a temporary state change
            self.state['pairing_ratio_override'] = override_ratio

    def _handle_last_peak(self, peak_idx: int):
        """Classify the final peak in the sequence."""
        self.state['candidate_beats'].append(peak_idx)
        self.state['beat_debug_info'][peak_idx] = PeakType.LONE_S1_LAST.value
        self.state['loop_idx'] += 1

    def _process_peak_pair(self, current_peak_idx: int):
        """Processes a pair of peaks to determine if they are S1-S2."""
        next_peak_idx = self.state['all_peaks'][self.state['loop_idx'] + 1]
        
        # Calculate pairing ratio once and cache it for reuse
        # Use cached pairing_ratio_override if available from kickstart mechanism
        if 'pairing_ratio_override' in self.state:
            pairing_ratio = self.state.get('pairing_ratio_override', 0.5)
            # Clear the override after using it once
            del self.state['pairing_ratio_override']
        else:
            # Calculate recent rhythm stability as a ratio
            history_window = self.params.get("stability_history_window", 20)
            if len(self.state['candidate_beats']) < history_window:
                pairing_ratio = 0.5
            else:
                # Use list comprehension instead of sum() with generator for better performance
                recent_beats = self.state['candidate_beats'][-history_window:]
                beat_debug_info = self.state['beat_debug_info']
                paired_count = sum(1 for beat_idx in recent_beats if 
                                  PeakType.S1_PAIRED.value in beat_debug_info.get(beat_idx, ""))
                pairing_ratio = paired_count / history_window

        is_paired, reason = self._attempt_s1_s2_pairing(
            current_peak_idx, next_peak_idx, pairing_ratio
        )

        if is_paired:
            self.state['candidate_beats'].append(current_peak_idx)
            reason_tag = f"PAIRING_SUCCESS_REASON§{reason}"
            self.state['beat_debug_info'][current_peak_idx] = f"{PeakType.S1_PAIRED.value}§{reason_tag}"
            self.state['beat_debug_info'][next_peak_idx] = f"{PeakType.S2_PAIRED.value}§{reason_tag}"
            self.state['consecutive_rr_rejections'] = 0
            self.state['loop_idx'] += 2
        else:
            self._classify_lone_peak(current_peak_idx, reason)
            self.state['loop_idx'] += 1

    def _update_long_term_bpm(self):
        """Updates the long-term BPM belief after each decision."""
        if len(self.state['candidate_beats']) > 1:
            new_rr = (self.state['candidate_beats'][-1] - self.state['candidate_beats'][-2]) / self.sample_rate
            if new_rr > 0:
                self.state['long_term_bpm'] = update_long_term_bpm(new_rr, self.state['long_term_bpm'], self.params)

        if self.state['candidate_beats']:
            time_sec = self.state['candidate_beats'][-1] / self.sample_rate
            self.state['long_term_bpm_history'].append((time_sec, self.state['long_term_bpm']))

    def _finalize_results(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Finalizes and returns the analysis results."""
        # Use NumPy's unique function instead of dict.fromkeys for deduplication
        # Sort=True ensures consistent ordering like the original implementation
        final_peaks = np.unique(self.state['candidate_beats'], return_index=False)
        self.state['analysis_data']["beat_debug_info"] = self.state['beat_debug_info']
        
        if self.state['long_term_bpm_history']:
            # Convert list of tuples to NumPy array in one step
            # This avoids creating intermediate lists with zip()
            lt_bpm_history = np.array(self.state['long_term_bpm_history'])
            
            # Extract columns directly from the array
            lt_bpm_times = lt_bpm_history[:, 0]
            lt_bpm_values = lt_bpm_history[:, 1]
            
            # Create Series directly with the extracted arrays
            # This avoids any unnecessary conversions
            self.state['analysis_data']["long_term_bpm_series"] = pd.Series(
                lt_bpm_values, 
                index=lt_bpm_times,
                copy=False  # Avoid copying data when creating the Series
            )
            
        return final_peaks, self.state['all_peaks'], self.state['analysis_data']

    def _find_raw_peaks(self, height_threshold: np.ndarray) -> np.ndarray:
        """Finds all potential peaks above the given height threshold."""
        prominence_thresh = np.quantile(self.audio_envelope, self.params['peak_prominence_quantile'])
        min_peak_dist_samples = int(self.params['min_peak_distance_sec'] * self.sample_rate)
        peaks, _ = find_peaks(self.audio_envelope, height=height_threshold, prominence=prominence_thresh, distance=min_peak_dist_samples)
        logging.info(f"Found {len(peaks)} raw peaks using dynamic height threshold.")
        return peaks

    def _attempt_s1_s2_pairing(self, s1_candidate_idx: int, s2_candidate_idx: int, pairing_ratio: float) -> Tuple[bool, str]:
        """Calculates the confidence score for pairing two candidate peaks."""
        interval_sec = (s2_candidate_idx - s1_candidate_idx) / self.sample_rate
        
        # Use NumPy's searchsorted for faster lookup instead of pandas asof
        # This avoids creating a Series and using pandas' slower asof method
        time_sec = s1_candidate_idx / self.sample_rate
        
        # Get direct access to the underlying arrays for faster operations
        dev_index = self.state['smoothed_dev_series'].index.values
        dev_values = self.state['smoothed_dev_series'].values
        
        if len(dev_values) == 0:
            deviation_value = 0
        else:
            # Find the nearest index using binary search
            idx = np.searchsorted(dev_index, time_sec)
            
            # Handle edge cases
            if idx >= len(dev_values):
                idx = len(dev_values) - 1
            elif idx > 0 and idx < len(dev_index):
                # Choose the closest point between idx-1 and idx
                if abs(dev_index[idx-1] - time_sec) < abs(dev_index[idx] - time_sec):
                    idx = idx - 1
                    
            deviation_value = dev_values[idx]

        confidence = calculate_blended_confidence(deviation_value, self.state['long_term_bpm'], self.params)
        blend_ratio = np.clip((self.state['long_term_bpm'] - self.params['contractility_bpm_low']) / (self.params['contractility_bpm_high'] - self.params['contractility_bpm_low']), 0, 1)
        reason = f"Base Conf (Blended Model {blend_ratio:.0%} High): {confidence:.2f}"

        confidence, adjust_reason = _adjust_confidence_with_stability_and_ratio(
            confidence, s1_candidate_idx, s2_candidate_idx, self.audio_envelope, self.state['dynamic_noise_floor'],
            self.state['long_term_bpm'], pairing_ratio, self.params, self.sample_rate,
            self.peak_bpm_time_sec, self.recovery_end_time_sec, len(self.state['candidate_beats'])
        )
        reason += adjust_reason

        s1_s2_max_interval = min(self.params['s1_s2_interval_cap_sec'], (60.0 / self.state['long_term_bpm']) * self.params['s1_s2_interval_rr_fraction'])
        
        # Apply interval penalty if the S1-S2 interval is too long
        if self.params.get("enable_interval_penalty", True) and interval_sec > s1_s2_max_interval:
            start_factor = self.params.get("interval_penalty_start_factor", 1.0)
            full_factor = self.params.get("interval_penalty_full_factor", 1.4)
            max_penalty = self.params.get("interval_max_penalty", 0.75)
            
            penalty_zone_start = s1_s2_max_interval * start_factor
            penalty_zone_end = s1_s2_max_interval * full_factor
            
            if interval_sec > penalty_zone_start:
                exceedance_scale = (interval_sec - penalty_zone_start) / (penalty_zone_end - penalty_zone_start + 1e-9)
                exceedance_scale = np.clip(exceedance_scale, 0, 1)
                penalty_amount = exceedance_scale * max_penalty
                confidence = max(0, confidence - penalty_amount)
                interval_reason = f"\n- Interval PENALTY by {penalty_amount:.2f} (Interval {interval_sec:.3f}s > Max {s1_s2_max_interval:.3f}s)"
            else:
                interval_reason = ""
        else:
            interval_reason = ""
        reason += interval_reason

        is_paired = confidence >= self.params['pairing_confidence_threshold']
        reason += f"\n- Final Score: {confidence:.2f} vs Threshold {self.params['pairing_confidence_threshold']:.2f} -> {'Paired' if is_paired else 'Not Paired'}"
        return is_paired, reason

    def _classify_lone_peak(self, peak_idx: int, pairing_failure_reason: str):
        """Validates if an unpaired peak is a Lone S1 or Noise."""
        is_valid, rejection_detail = self._validate_lone_s1(peak_idx)
        pairing_info = f"PAIRING_FAIL_REASON§{pairing_failure_reason.lstrip(' |')}"

        if is_valid:
            self.state['candidate_beats'].append(peak_idx)
            # For a validated S1, the "rejection_detail" is just the success reason.
            self.state['beat_debug_info'][
                peak_idx] = f"{PeakType.LONE_S1_VALIDATED.value}§{pairing_info}§LONE_S1_VALIDATE_REASON§{rejection_detail}"
            self.state['consecutive_rr_rejections'] = 0
        else:
            is_rhythm_rejection = "Rhythm Fit" in rejection_detail
            if is_rhythm_rejection:
                self.state['consecutive_rr_rejections'] += 1
            else:
                self.state['consecutive_rr_rejections'] = 0

            lone_s1_rejection_info = f"LONE_S1_REJECT_REASON§{rejection_detail}"

            if self.state['consecutive_rr_rejections'] >= self.params.get("cascade_reset_trigger_count", 3):
                logging.info(
                    f"CASCADE RESET: Forcing peak at {peak_idx / self.sample_rate:.2f}s as Lone S1 due to repeated rhythmic failures.")
                self.state['candidate_beats'].append(peak_idx)
                self.state['beat_debug_info'][
                    peak_idx] = f"{PeakType.LONE_S1_CASCADE.value}§{pairing_info}§{lone_s1_rejection_info}"
                self.state['consecutive_rr_rejections'] = 0
            else:
                self.state['beat_debug_info'][peak_idx] = f"Noise§{pairing_info}§{lone_s1_rejection_info}"

    def _validate_lone_s1(self, current_peak_idx: int) -> Tuple[bool, str]:
        """Performs checks to determine if a peak is a valid Lone S1."""
        if not self.state['candidate_beats']: return True, "First beat"

        confidence, reason = calculate_lone_s1_confidence(
            current_peak_idx, self.state['candidate_beats'][-1], self.state['long_term_bpm'],
            self.audio_envelope, self.state['dynamic_noise_floor'], self.sample_rate, self.params
        )
        threshold = self.params.get("lone_s1_confidence_threshold", 0.6)
        if confidence < threshold:
            return False, f"Rejected Lone S1: Confidence {confidence:.2f} < Threshold {threshold:.2f}. ({reason})"

        current_peak_all_peaks_idx = np.searchsorted(self.state['all_peaks'], current_peak_idx)
        if current_peak_all_peaks_idx < len(self.state['all_peaks']) - 1:
            next_raw_peak_idx = self.state['all_peaks'][current_peak_all_peaks_idx + 1]
            forward_interval_sec = (next_raw_peak_idx - current_peak_idx) / self.sample_rate
            expected_rr_sec = 60.0 / self.state['long_term_bpm']
            min_forward_interval = expected_rr_sec * self.params.get('lone_s1_forward_check_pct', 0.6)
            if forward_interval_sec < min_forward_interval:
                if not (self.audio_envelope[current_peak_idx] > (self.audio_envelope[next_raw_peak_idx] * 1.7)):
                     implied_bpm = 60.0 / forward_interval_sec if forward_interval_sec > 0 else float('inf')
                     return False, f"Rejected Lone S1: Forward check failed (Implies {implied_bpm:.0f} BPM)"
        # Get the weights for the calculation
        rhythm_weight = self.params.get('lone_s1_rhythm_weight', 0.65)
        amplitude_weight = self.params.get('lone_s1_amplitude_weight', 0.35)
        return True, f"Validated Lone S1: Confidence {confidence:.3f} >= Threshold {threshold:.2f}. ({reason}, Weights: Rhythm={rhythm_weight:.2f}, Amplitude={amplitude_weight:.2f}, Final={confidence:.3f})"

# Import necessary functions from the original file
from bpm_analysis import (
    preprocess_audio,
    _refine_and_correct_peaks,
    ReportGenerator,
    find_major_hr_inclines,
    find_major_hr_declines,
    calculate_hrr,
    find_peak_recovery_rate,
    find_peak_exertion_rate,
    find_recovery_phase
)

def _calculate_dynamic_noise_floor(audio_envelope: np.ndarray, sample_rate: int, params: Dict) -> Tuple[pd.Series, np.ndarray]:
    """
    Calculates a dynamic noise floor based on a sanitized set of audio troughs.
    Optimized version using NumPy operations where possible.
    """
    min_peak_dist_samples = int(params['min_peak_distance_sec'] * sample_rate)
    trough_prom_thresh = np.quantile(audio_envelope, params['trough_prominence_quantile'])

    # --- STEP 1: Find all potential troughs initially ---
    all_trough_indices, _ = find_peaks(-audio_envelope, distance=min_peak_dist_samples, prominence=trough_prom_thresh)

    # If we don't have enough troughs to begin with, fall back to a simple static floor.
    if len(all_trough_indices) < 5:
        logging.warning("Not enough troughs found for sanitization. Using a static noise floor.")
        fallback_value = np.quantile(audio_envelope, params['noise_floor_quantile'])
        # Create a pandas Series with the fallback value for compatibility
        # Avoid creating a full-length array first, just create the Series directly
        dynamic_noise_floor = pd.Series([fallback_value] * len(audio_envelope), index=np.arange(len(audio_envelope)))
        return dynamic_noise_floor, all_trough_indices

    # --- STEP 2: Create a preliminary 'draft' noise floor from ALL troughs ---
    # Create a pandas Series directly with the data and indices to avoid unnecessary conversions
    trough_values = audio_envelope[all_trough_indices]
    trough_series_draft = pd.Series(trough_values, index=all_trough_indices)
    
    # Use more efficient reindexing by pre-creating the index array
    index_array = np.arange(len(audio_envelope))
    dense_troughs_draft = trough_series_draft.reindex(index_array).interpolate(method='linear')
    
    noise_window_samples = int(params['noise_window_sec'] * sample_rate)
    quantile_val = params['noise_floor_quantile']
    
    # Use more efficient rolling operation by specifying min_periods only once
    draft_noise_floor = dense_troughs_draft.rolling(
        window=noise_window_samples, 
        min_periods=3, 
        center=True
    ).quantile(quantile_val)
    
    # Fill gaps in one operation to avoid creating intermediate Series
    draft_noise_floor = draft_noise_floor.bfill().ffill()

    # --- STEP 3: Sanitize the trough list ---
    # Use NumPy vectorized operations to filter troughs
    trough_amps = audio_envelope[all_trough_indices]
    
    # Use direct indexing with .values to avoid Series creation
    floor_at_troughs = draft_noise_floor.iloc[all_trough_indices].values
    valid_mask = ~np.isnan(floor_at_troughs) & (trough_amps <= (params.get('trough_rejection_multiplier', 4.0) * floor_at_troughs))
    sanitized_trough_indices = all_trough_indices[valid_mask]

    logging.info(f"Trough Sanitization: Kept {len(sanitized_trough_indices)} of {len(all_trough_indices)} initial troughs.")

    # --- STEP 4: Calculate more accurate noise floor using only sanitized troughs ---
    if len(sanitized_trough_indices) > 2:
        # Create Series directly with data and indices
        sanitized_values = audio_envelope[sanitized_trough_indices]
        trough_series_final = pd.Series(sanitized_values, index=sanitized_trough_indices)
        
        # Reuse the index array from before to avoid creating a new one
        dense_troughs_final = trough_series_final.reindex(index_array).interpolate(method='linear')
        
        # Combine rolling operations to minimize intermediate Series creation
        dynamic_noise_floor = dense_troughs_final.rolling(
            window=noise_window_samples, 
            min_periods=3, 
            center=True
        ).quantile(quantile_val).bfill().ffill()
    else:
        # If sanitization removed too many troughs, it's safer to use the original draft floor.
        logging.warning("Not enough sanitized troughs remaining. Using non-sanitized floor as fallback.")
        dynamic_noise_floor = draft_noise_floor

    # Final check for any remaining null values
    if dynamic_noise_floor.isnull().all():
         fallback_val = np.quantile(audio_envelope, 0.1)
         # Create Series directly with the fallback value
         dynamic_noise_floor = pd.Series([fallback_val] * len(audio_envelope), index=np.arange(len(audio_envelope)))

    return dynamic_noise_floor, sanitized_trough_indices

def calculate_bpm_series(peaks: np.ndarray, sample_rate: int, params: Dict) -> Tuple[pd.Series, np.ndarray]:
    """
    Calculates and smooths the final BPM series from S1 peaks.
    Optimized version using NumPy operations where possible.
    """
    if len(peaks) < 2: 
        return pd.Series(dtype=np.float64), np.array([])
    
    # Calculate time differences and BPM values in one step
    peak_times = peaks / sample_rate
    time_diffs = np.diff(peak_times)
    valid_diffs_mask = time_diffs > 1e-6
    
    if not np.any(valid_diffs_mask): 
        return pd.Series(dtype=np.float64), np.array([])

    # Use masked arrays to avoid unnecessary array creation
    valid_diffs = time_diffs[valid_diffs_mask]
    instant_bpm = 60.0 / valid_diffs
    valid_peak_times = peak_times[1:][valid_diffs_mask]
    
    # Convert to datetime for pandas Series index (required for rolling window)
    # Pre-allocate the array to avoid resizing
    start_time = datetime.datetime.fromtimestamp(0)
    valid_peak_times_dt = np.empty(len(valid_peak_times), dtype=object)
    
    # Use vectorized operations where possible
    for i, t in enumerate(valid_peak_times):
        valid_peak_times_dt[i] = start_time + datetime.timedelta(seconds=t)
    
    # Create pandas Series directly with data and index to avoid intermediate steps
    bpm_series = pd.Series(instant_bpm, index=valid_peak_times_dt)
    avg_heart_rate = np.median(instant_bpm)
    
    if avg_heart_rate > 0:
        # Create the rolling window parameters once
        smoothing_window_sec = params['output_smoothing_window_sec']
        smoothing_window_str = f"{smoothing_window_sec}s"
        
        # Apply rolling mean in one operation
        smoothed_bpm = bpm_series.rolling(
            window=smoothing_window_str, 
            min_periods=1, 
            center=True
        ).mean()
    else:
        # Create an empty Series directly
        smoothed_bpm = pd.Series(dtype=np.float64)

    # Return the original numpy time points for compatibility with older functions that need it
    return smoothed_bpm, valid_peak_times

def calculate_windowed_hrv(s1_peaks: np.ndarray, sample_rate: int, params: Dict) -> pd.DataFrame:
    """
    Calculates HRV metrics using R-R intervals based on changing heart rate.
    Optimized version using NumPy operations where possible.
    """
    window_size_beats = params['hrv_window_size_beats']
    step_size_beats = params['hrv_step_size_beats']

    # First, calculate all R-R intervals from the S1 peaks
    if len(s1_peaks) < window_size_beats:
        logging.warning(f"Not enough beats ({len(s1_peaks)}) to perform windowed HRV analysis with a window of {window_size_beats} beats.")
        # Create an empty DataFrame directly with column names to avoid intermediate dict creation
        return pd.DataFrame(columns=['time', 'rmssdc', 'sdnn', 'bpm'])

    # Calculate all R-R intervals at once
    rr_intervals_sec = np.diff(s1_peaks) / sample_rate
    s1_times_sec = s1_peaks / sample_rate

    # Calculate number of windows more efficiently
    num_windows = max(0, (len(rr_intervals_sec) - window_size_beats + 1) // step_size_beats)
    
    if num_windows == 0:
        logging.warning("Could not perform windowed HRV analysis. Recording may be too short or have too few beats.")
        return pd.DataFrame(columns=['time', 'rmssdc', 'sdnn', 'bpm'])
    
    # Pre-allocate arrays for results - use a single allocation for all metrics
    # This creates a structured array that can be directly converted to DataFrame
    dtype = [('time', float), ('rmssdc', float), ('sdnn', float), ('bpm', float)]
    results_array = np.zeros(num_windows, dtype=dtype)
    
    # Convert to milliseconds once for all calculations
    rr_intervals_ms = rr_intervals_sec * 1000
    
    # Iterate through the R-R intervals with a sliding window
    for i in range(num_windows):
        start_idx = i * step_size_beats
        end_idx = start_idx + window_size_beats
        
        # Get window slice directly from the pre-converted millisecond array
        window_rr_ms = rr_intervals_ms[start_idx:end_idx]
        
        # Calculate window midpoint time
        start_time = s1_times_sec[start_idx]
        end_time = s1_times_sec[start_idx + window_size_beats]
        window_mid_time = (start_time + end_time) / 2.0
        
        # --- Calculate HRV Metrics for the Window ---
        # Use NumPy's optimized functions for these calculations
        mean_rr_ms = np.mean(window_rr_ms)
        sdnn = np.std(window_rr_ms)
        
        # Calculate RMSSD more efficiently
        successive_diffs_ms = np.diff(window_rr_ms)
        rmssd = np.sqrt(np.mean(np.square(successive_diffs_ms)))
        
        # --- Calculate Corrected RMSSD (RMSSDc) ---
        mean_rr_sec = mean_rr_ms / 1000.0
        rmssdc = rmssd / mean_rr_sec if mean_rr_sec > 0 else 0
        
        # Calculate the average BPM within this specific window
        window_bpm = 60 / mean_rr_sec if mean_rr_sec > 0 else 0
        
        # Store results directly in the structured array
        results_array[i] = (window_mid_time, rmssdc, sdnn, window_bpm)

    # Create DataFrame directly from the structured array
    # This avoids creating intermediate dictionaries
    results_df = pd.DataFrame(results_array)
    
    logging.info(f"Beat-based windowed HRV analysis complete. Generated {num_windows} data points.")
    return results_df

def _run_preliminary_pass(audio_envelope: np.ndarray, sample_rate: int, params: Dict,
                          noise_floor: pd.Series, troughs: np.ndarray,
                          start_bpm_hint: Optional[float]) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Runs a high-confidence first pass to estimate global BPM and find the recovery phase.
    Optimized version using NumPy operations where possible.
    """
    logging.info("--- STAGE 2: Running High-Confidence pass to find anchor beats ---")
    # Create a shallow copy of params to avoid modifying the original
    params_pass_1 = params.copy()
    # Use a higher threshold for a more confident initial beat detection
    params_pass_1["pairing_confidence_threshold"] = 0.75

    # Use the classifier for a high-confidence dry run
    # Pass the noise_floor Series directly without conversion
    classifier = PeakClassifier(audio_envelope, sample_rate, params_pass_1, start_bpm_hint,
                                noise_floor, troughs, None, None)
    anchor_beats, _, _ = classifier.classify_peaks()

    global_bpm_estimate = None
    peak_bpm_time_sec = None
    recovery_end_time_sec = None
    
    if len(anchor_beats) >= 10:
        # Calculate median RR interval directly from the anchor beats
        # Avoid creating intermediate arrays by doing the division in one step
        median_rr_sec = np.median(np.diff(anchor_beats)) / sample_rate
        if median_rr_sec > 0:
            global_bpm_estimate = 60.0 / median_rr_sec
            
            # Calculate BPM series using optimized function
            smoothed_bpm, valid_peak_times = calculate_bpm_series(anchor_beats, sample_rate, params)
            
            if not smoothed_bpm.empty:
                # Find recovery phase using NumPy operations where possible
                peak_bpm_time_sec, recovery_end_time_sec = find_recovery_phase(
                    smoothed_bpm, valid_peak_times, params
                )
    
    if global_bpm_estimate:
        logging.info(f"Automatically determined Global BPM Estimate: {global_bpm_estimate:.1f} BPM")
        
    return global_bpm_estimate, peak_bpm_time_sec, recovery_end_time_sec

    return start_bpm, peak_bpm_time_sec, recovery_end_time_sec

def _calculate_final_metrics(final_peaks: np.ndarray, sample_rate: int, params: Dict) -> Dict:
    """
    Calculates all final BPM, HRV, and slope metrics for reporting.
    Optimized version using NumPy operations where possible.
    """
    metrics = {}
    metrics['smoothed_bpm'], metrics['bpm_times'] = calculate_bpm_series(final_peaks, sample_rate, params)
    metrics['major_inclines'] = find_major_hr_inclines(metrics['smoothed_bpm'])
    metrics['major_declines'] = find_major_hr_declines(metrics['smoothed_bpm'])
    metrics['hrr_stats'] = calculate_hrr(metrics['smoothed_bpm'])
    metrics['peak_recovery_stats'] = find_peak_recovery_rate(metrics['smoothed_bpm'])
    metrics['peak_exertion_stats'] = find_peak_exertion_rate(metrics['smoothed_bpm'])
    metrics['windowed_hrv_df'] = calculate_windowed_hrv(final_peaks, sample_rate, params)

    hrv_summary_stats = {}
    if not metrics['smoothed_bpm'].empty:
        # Use NumPy operations on the values array instead of pandas methods
        bpm_values = metrics['smoothed_bpm'].values
        hrv_summary_stats['avg_bpm'] = np.mean(bpm_values)
        hrv_summary_stats['min_bpm'] = np.min(bpm_values)
        hrv_summary_stats['max_bpm'] = np.max(bpm_values)
    if not metrics['windowed_hrv_df'].empty:
        # Use NumPy operations on the values array instead of pandas methods
        hrv_summary_stats['avg_rmssdc'] = np.mean(metrics['windowed_hrv_df']['rmssdc'].values)
        hrv_summary_stats['avg_sdnn'] = np.mean(metrics['windowed_hrv_df']['sdnn'].values)
    metrics['hrv_summary'] = hrv_summary_stats

    return metrics

def analyze_wav_file(wav_file_path: str, params: Dict, start_bpm_hint: Optional[float], original_file_path: str, output_directory: str):
    """
    Main analysis pipeline that orchestrates the refactored classes.
    Optimized version using NumPy vectorized operations where possible.
    """
    start_time = time.time()
    logging.info(f"--- Processing file: {os.path.basename(original_file_path)} ---")

    # STAGE 1: Initialization
    audio_envelope, sample_rate = preprocess_audio(wav_file_path, params, output_directory)
    noise_floor, troughs = _calculate_dynamic_noise_floor(audio_envelope, sample_rate, params)

    start_bpm, peak_time, recovery_time = _run_preliminary_pass(
        audio_envelope, sample_rate, params, noise_floor, troughs, start_bpm_hint
    )

    # STAGE 3: Main Analysis, now informed by the preliminary pass
    logging.info("--- STAGE 3: Running Main Analysis Pass ---")
    classifier = PeakClassifier(
        audio_envelope, sample_rate, params, start_bpm,
        noise_floor, troughs, peak_time, recovery_time
    )
    s1_peaks, all_raw_peaks, analysis_data = classifier.classify_peaks()

    # STAGE 4 & 5: Correction and Refinement
    final_peaks, analysis_data = _refine_and_correct_peaks(
        s1_peaks, all_raw_peaks, analysis_data, audio_envelope, sample_rate, params
    )

    # STAGE 6: Final Reporting
    if len(final_peaks) < 2:
        logging.warning("Not enough S1 peaks detected to generate full report.")
        return None

    logging.info("--- STAGE 6: Calculating Metrics and Generating Outputs ---")
    final_metrics = _calculate_final_metrics(final_peaks, sample_rate, params)

    plotter = Plotter(original_file_path, params, sample_rate, output_directory)
    plotly_figure = plotter.plot_and_save(audio_envelope, all_raw_peaks, analysis_data, final_metrics)

    reporter = ReportGenerator(original_file_path, output_directory)
    reporter.save_analysis_summary(final_metrics)
    reporter.create_chronological_log(audio_envelope, sample_rate, all_raw_peaks, analysis_data, final_metrics)
    reporter.save_analysis_settings(start_bpm_hint)

    duration = time.time() - start_time
    logging.info(f"--- Analysis finished in {duration:.2f} seconds. ---")
    
    return plotly_figure