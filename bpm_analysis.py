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

# INSTRUCTIONS FOR AI: 
# Do not remove any debugging code unless specified by the user
# Do not further abstract my code
# Do not further segment my code
# Do not over-engineer a solution, keep it simple

# --- Enums and Global Helpers ---
class PeakType(Enum):
    """Enumeration for classifying heartbeat peaks."""
    S1_PAIRED = "S1 (Paired)"
    S2_PAIRED = "S2 (Paired)"
    LONE_S1_VALIDATED = "Lone S1"
    LONE_S1_CASCADE = "Lone S1 (Corrected by Cascade Reset)"
    LONE_S1_LAST = "Lone S1 (Last Peak)"
    NOISE = "Noise/Rejected"
    S1_CORRECTED_GAP = "S1 (Paired - Corrected from Gap)"
    S2_CORRECTED_GAP = "S2 (Paired - Corrected from Gap)"
    S2_CORRECTED_CONFLICT = "S2 (Paired - Corrected from Conflict)"

    @classmethod
    def is_s1(cls, peak_type_str: str) -> bool:
        """Check if a string corresponds to any S1 type."""
        return peak_type_str.strip().startswith("S1") or peak_type_str.strip().startswith("Lone S1")

    @classmethod
    def is_s2(cls, peak_type_str: str) -> bool:
        """Check if a string corresponds to any S2 type."""
        return peak_type_str.strip().startswith("S2")


# --- Setup Professional Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    stream=sys.stdout
)

# --- Audio Conversion (requires pydub/ffmpeg) ---
try:
    from pydub import AudioSegment
except ImportError:
    logging.warning("Pydub library not found. Install with 'pip install pydub'.")
    AudioSegment = None

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
        state['sorted_troughs'] = sorted(precomputed_troughs)
        
        # Calculate peak strengths and deviations more efficiently
        noise_floor_at_peaks = precomputed_noise_floor.reindex(state['all_peaks'], method='nearest').values
        peak_strengths = np.maximum(0, self.audio_envelope[state['all_peaks']] - noise_floor_at_peaks)
        
        # Only proceed if we have enough peaks
        if len(peak_strengths) > 1:
            # Calculate normalized deviations more efficiently
            peak_max = np.maximum(peak_strengths[:-1], peak_strengths[1:])
            normalized_deviations = np.abs(np.diff(peak_strengths)) / (peak_max + 1e-9)
            
            # Calculate deviation times
            deviation_times = (state['all_peaks'][:-1] + state['all_peaks'][1:]) / (2 * self.sample_rate)
            
            # Create and smooth deviation series
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
        if hasattr(self.state, 'pairing_ratio_override'):
            pairing_ratio = self.state.get('pairing_ratio_override', 0.5)
            # Clear the override after using it once
            if 'pairing_ratio_override' in self.state:
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
        final_peaks = np.array(sorted(list(dict.fromkeys(self.state['candidate_beats']))))
        self.state['analysis_data']["beat_debug_info"] = self.state['beat_debug_info']
        if self.state['long_term_bpm_history']:
            lt_bpm_times, lt_bpm_values = zip(*self.state['long_term_bpm_history'])
            self.state['analysis_data']["long_term_bpm_series"] = pd.Series(lt_bpm_values, index=lt_bpm_times)
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
        deviation_value = self.state['smoothed_dev_series'].asof(s1_candidate_idx / self.sample_rate)

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


class Plotter:
    """Handles the creation and generation of the final analysis plot."""
    
    @staticmethod
    def format_pairing_details_list(details_str: str) -> List[str]:
        """Formats S1-S2 pairing details into a list of strings."""
        lines = [line.strip().lstrip('- ') for line in details_str.strip().split('\n') if line.strip()]
        if not lines: return ["- S1-S2 pairing decision:", "    - No details available."]

        output_lines = ["- S1-S2 pairing decision:"]
        confidence = 0.0

        try:
            match = re.search(r'([\d\.]+)$', lines[0])
            if match: confidence = float(match.group(1))
            output_lines.append(f"    - {lines[0]}")

            for line in lines[1:]:
                new_confidence = confidence
                if "Stability Pre-Adjust" in line:
                    match = re.search(r'x([\d\.]+)', line); new_confidence *= float(match.group(1)) if match else 1
                    output_lines.append(f"    - {line} -> {new_confidence:.3f}")
                elif "PENALIZED by" in line:
                    match = re.search(r'by ([\d\.]+)', line); new_confidence -= float(match.group(1)) if match else 0
                    output_lines.append(f"    - {line} -> {new_confidence:.3f}")
                elif "Interval PENALTY by" in line:
                    match = re.search(r'by ([\d\.]+)', line); new_confidence -= float(match.group(1)) if match else 0
                    output_lines.append(f"    - {line} -> {max(0, new_confidence):.3f}")
                else:
                    output_lines.append(f"    - {line}")
                confidence = new_confidence
        except (ValueError, IndexError):
            return ["- S1-S2 pairing decision:", f"    - {details_str}"]
        return output_lines

    @staticmethod
    def format_lone_s1_details_list(details_str: str) -> List[str]:
        """Formats Lone S1 validation details into a readable list of strings."""
        output_lines = ["- Lone S1 decision:"]
        
        # Regex to capture the main components of the validation string
        main_pattern = re.compile(r"(Validated|Rejected) Lone S1: Confidence ([\d\.]+) (>=|<) Threshold ([\d\.]+)\. \((.*)\)")
        main_match = main_pattern.search(details_str)

        if not main_match:
            return ["- Lone S1 decision:", f"\t- {details_str}"]

        try:
            status, final_conf_str, operator, threshold_str, reason_str = main_match.groups()
            final_conf = float(final_conf_str)
            threshold = float(threshold_str)

            # Regex patterns to extract all numerical and descriptive parts
            patterns = {
                'rhythm_fit': r"Rhythm Fit=([\d\.]+)",
                'rhythm_details': r"\(Interval .*?s vs Expected .*?s\)",
                'amp_fit': r"Amplitude Fit=([\d\.]+)",
                'amp_details': r"\(Strength Ratio .*?x\)",
                'rhythm_weight': r"Rhythm=([\d\.]+)",
                'amp_weight': r"Amplitude=([\d\.]+)",
            }
            
            # Extract all matching values from the reason string
            extracted = {key: re.search(p, reason_str) for key, p in patterns.items()}

            rhythm_score = float(extracted['rhythm_fit'].group(1))
            rhythm_details = extracted['rhythm_details'].group(0)
            output_lines.append(f"\t- Rhythm Fit={rhythm_score:.2f} {rhythm_details}")

            amp_score = float(extracted['amp_fit'].group(1))
            amp_details = extracted['amp_details'].group(0)
            output_lines.append(f"\t- Amplitude Fit={amp_score:.2f} {amp_details}")

            # If weight information is available, add the detailed breakdown
            if extracted['rhythm_weight'] and extracted['amp_weight']:
                rhythm_weight = float(extracted['rhythm_weight'].group(1))
                amp_weight = float(extracted['amp_weight'].group(1))
                
                rhythm_contrib = rhythm_score * rhythm_weight
                amp_contrib = amp_score * amp_weight

                output_lines.append("\t- Weighted Calculation:")
                output_lines.append(f"\t\t- Rhythm: {rhythm_score:.2f} × {rhythm_weight:.2f} = {rhythm_contrib:.3f}")
                output_lines.append(f"\t\t- Amplitude: {amp_score:.2f} × {amp_weight:.2f} = {amp_contrib:.3f}")
                output_lines.append(f"\t\t- Final: {rhythm_contrib:.3f} + {amp_contrib:.3f} = {final_conf:.3f}")

            # Add the final summary line
            outcome_status = "Validated" if "Validated" in status else "Rejected"
            output_lines.append(f"- Final Score: Confidence {final_conf:.3f} {operator} {threshold:.2f} -> {outcome_status}")

        except (AttributeError, ValueError, IndexError) as e:
            # Fallback for any parsing error
            logging.warning(f"Could not parse Lone S1 details string: '{details_str}'. Error: {e}")
            return ["- Lone S1 decision:", f"\t- {details_str}"]
        
        return output_lines

    def __init__(self, file_name: str, params: Dict, sample_rate: int, output_directory: str):
        self.file_name = file_name
        self.params = params
        self.sample_rate = sample_rate
        self.output_directory = output_directory # Add this line
        self.fig = make_subplots(specs=[[{"secondary_y": True}]])

    def plot_and_save(self, audio_envelope: np.ndarray, all_raw_peaks: np.ndarray, analysis_data: Dict,
                      final_metrics: Dict):
        """Generates and saves the main analysis plot by calling helper methods."""
        self.time_axis_sec = np.arange(len(audio_envelope)) / self.sample_rate
        time_axis_dt = pd.to_datetime([datetime.datetime.fromtimestamp(0) + datetime.timedelta(seconds=t) for t in self.time_axis_sec])

        self._add_line_traces(time_axis_dt, audio_envelope, analysis_data)
        self._add_trough_markers(audio_envelope, analysis_data)
        self._add_peak_traces(all_raw_peaks, analysis_data.get('beat_debug_info', {}), audio_envelope)
        self._add_bpm_hrv_traces(final_metrics.get('smoothed_bpm'), analysis_data, final_metrics.get('windowed_hrv_df'))
        self._add_slope_traces(final_metrics.get('major_inclines'), final_metrics.get('major_declines'), final_metrics.get('peak_recovery_stats'), final_metrics.get('peak_exertion_stats'))
        self._add_annotations_and_summary(final_metrics.get('smoothed_bpm'), final_metrics.get('hrv_summary'), final_metrics.get('hrr_stats'), final_metrics.get('peak_recovery_stats'))

        self._configure_layout()

        base_name = os.path.basename(os.path.splitext(self.file_name)[0])
        output_html_path = os.path.join(self.output_directory, f"{base_name}_bpm_plot.html")
        plot_title = f"Heartbeat Analysis - {os.path.basename(self.file_name)}"
        plot_config = {'scrollZoom': True, 'toImageButtonOptions': {'filename': plot_title, 'format': 'png', 'scale': 2}}
        self.fig.write_html(output_html_path, config=plot_config)
        logging.info(f"Interactive plot saved to {output_html_path}")

        # --- Output CSV for BPM plot ---
        smoothed_bpm = final_metrics.get('smoothed_bpm')
        bpm_times = final_metrics.get('bpm_times')
        if smoothed_bpm is not None and not smoothed_bpm.empty and bpm_times is not None:
            # Use the raw numpy array of times for the table and match it with the smoothed BPM values
            csv_path = os.path.join(self.output_directory, f"{base_name}_bpm_plot.csv")
            try:
                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Time (s)', 'Average BPM'])
                    for t, bpm in zip(bpm_times, smoothed_bpm.values):
                        if not np.isnan(bpm):
                            writer.writerow([f"{t:.3f}", f"{bpm:.3f}"])
                logging.info(f"BPM plot data saved to {csv_path}")
            except Exception as e:
                logging.error(f"Failed to write BPM plot CSV: {e}")
        
        # Return the figure object for Gradio display
        return self.fig

    def _configure_layout(self):
        """Sets up the plot layout, titles, and axes with custom x-axis tick labels."""
        plot_title = f"Heartbeat Analysis - {os.path.basename(self.file_name)}"

        self.fig.update_layout(
            template="plotly_dark", title_text=plot_title, dragmode='pan',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=140, b=100),
            hovermode='x unified'
        )

        # X-Axis
        tick_positions_sec = np.linspace(0, self.time_axis_sec[-1], num=10)
        epoch = datetime.datetime.fromtimestamp(0)
        
        tickvals = [epoch + datetime.timedelta(seconds=s) for s in tick_positions_sec]
        ticktext = [f"{int(s // 60):02d}:{int(s % 60):02d} ({s:.2f})" for s in tick_positions_sec]

        self.fig.update_xaxes(
            title_text="Time",
            tickvals=tickvals,
            ticktext=ticktext,
            hoverformat='%M:%S.%L' # Adds millisecond precision to the hover label
        )
        
        # Y-axis
        robust_upper_limit = np.quantile(self.fig.data[0].y, 0.95) if self.fig.data else 1
        amplitude_scale = self.params.get("plot_amplitude_scale_factor", 60.0)
        self.fig.update_yaxes(title_text="Signal Amplitude", secondary_y=False, range=[0, robust_upper_limit * amplitude_scale])
        self.fig.update_yaxes(title_text="BPM / HRV", secondary_y=True, range=[50, 200])


    def _add_line_traces(self, time_axis_dt: pd.Series, audio_envelope: np.ndarray, analysis_data: Dict):
        """Adds downsampled audio envelope and noise floor traces for performance."""
        # --- Prepare data for plotting, with optional downsampling for performance ---
        plot_noise_floor = analysis_data.get('dynamic_noise_floor_series')

        # Always downsample for performance (hardcoded behavior)
        factor = self.params.get("plot_downsample_factor", 5)
        if factor > 1 and len(audio_envelope) >= factor:
            logging.info(f"Downsampling line traces by a factor of {factor} for plotting.")
            # Use direct array slicing for better performance
            plot_time_axis_dt = time_axis_dt[::factor]
            plot_envelope = audio_envelope[::factor]
            
            # Handle noise floor downsampling more efficiently
            if plot_noise_floor is not None and not plot_noise_floor.empty:
                # Use efficient resampling for pandas Series
                if len(plot_noise_floor) > len(audio_envelope) // factor:
                    plot_noise_floor = plot_noise_floor.iloc[::factor]
        else:
            plot_time_axis_dt = time_axis_dt
            plot_envelope = audio_envelope

        # --- Add the potentially downsampled line traces ---
        # Pre-allocate memory for traces to avoid multiple reallocations
        self.fig.add_trace(go.Scatter(
            x=plot_time_axis_dt,
            y=plot_envelope,
            name="Audio Envelope",
            line=dict(color="#47a5c4"),
            hoverinfo="skip"),  # Skip hover for better performance on large datasets
            secondary_y=False)
            
        # Only add noise floor if it exists and has valid data
        if plot_noise_floor is not None and not plot_noise_floor.empty:
            # Ensure noise floor length matches time axis
            if len(plot_noise_floor) >= len(plot_time_axis_dt):
                self.fig.add_trace(go.Scatter(
                    x=plot_time_axis_dt,
                    y=plot_noise_floor.values[:len(plot_time_axis_dt)],  # Ensure lengths match
                    name="Dynamic Noise Floor",
                    line=dict(color="green", dash="dot", width=1.5),
                    hovertemplate="Noise Floor: %{y:.2f}<extra></extra>"),
                    secondary_y=False)

    def _add_trough_markers(self, audio_envelope: np.ndarray, analysis_data: Dict):
        """Adds trough markers to the plot using original full-resolution data for accuracy."""
        trough_indices = analysis_data.get('trough_indices')
        if trough_indices is not None and trough_indices.size > 0:
            # Create datetime objects for the trough markers
            trough_times_dt = pd.to_datetime([
                datetime.datetime.fromtimestamp(0) + datetime.timedelta(seconds=t)
                for t in (trough_indices / self.sample_rate)
            ])

            self.fig.add_trace(go.Scatter(
                x=trough_times_dt,
                y=audio_envelope[trough_indices],
                mode='markers',
                name='Troughs',
                marker=dict(color='green', symbol='circle-open', size=6),
                visible='legendonly'),
                secondary_y=False)

    def _add_peak_traces(self, all_raw_peaks, debug_info, audio_envelope):
        """Adds S1, S2, and Noise peak markers to the plot with detailed hover info."""
        s1_peaks = {'indices': [], 'customdata': []}
        s2_peaks = {'indices': [], 'customdata': []}
        noise_peaks = {'indices': [], 'customdata': []}

        classified_indices = set()

        # --- Generate detailed hover text for each classified peak ---
        for peak_idx, reason_str in debug_info.items():
            hover_text_parts = []
            parts = reason_str.split('§')
            final_peak_type, details_list = parts[0], parts[1:]

            # Add basic peak info
            hover_text_parts.append(f"<b>Type:</b> {final_peak_type}")
            hover_text_parts.append(f"<b>Time:</b> {peak_idx / self.sample_rate:.2f}s")
            hover_text_parts.append(f"<b>Amp:</b> {audio_envelope[peak_idx]:.0f}")
            hover_text_parts.append("---")  # Visual separator

            # Add detailed, formatted reasons from the debug string
            i = 0
            while i < len(details_list):
                tag = details_list[i]
                value = details_list[i + 1] if (i + 1) < len(details_list) else ""
                formatted_lines = []

                if "PAIRING" in tag:
                    formatted_lines = self.format_pairing_details_list(value)
                elif "LONE_S1_REJECT_REASON" in tag:
                    formatted_lines = self.format_lone_s1_details_list(value)
                elif "LONE_S1_VALIDATE_REASON" in tag:
                    formatted_lines = self.format_lone_s1_details_list(value)
                elif "ORIGINAL_REASON" in tag:
                    formatted_lines = ["- Original Classification:",
                                       f"&nbsp;&nbsp;&nbsp;&nbsp;- {value.replace('`', '')}"]

                if formatted_lines:
                    # Convert the list of strings to a single HTML block
                    sub_text = "<br>".join(l.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;') for l in formatted_lines)
                    hover_text_parts.append(sub_text)
                i += 2

            # Join all parts into a single HTML string for the tooltip
            full_hover_text = "<br>".join(hover_text_parts)
            classified_indices.add(peak_idx)
            # Parse reason string to extract peak type
            if not reason_str:
                peak_type = "Unknown Peak"
            else:
                separators = ['. Pairing Justification: ', '. Rejection: ', '. Original: ', '. ']
                for sep in separators:
                    if sep in reason_str:
                        parts = reason_str.split(sep, 1)
                        peak_type = parts[0].strip()
                        break
                else:
                    peak_type = reason_str.strip()

            # Assign the peak to the correct category for plotting
            if PeakType.is_s1(peak_type):
                s1_peaks['indices'].append(peak_idx)
                s1_peaks['customdata'].append(full_hover_text)
            elif PeakType.is_s2(peak_type):
                s2_peaks['indices'].append(peak_idx)
                s2_peaks['customdata'].append(full_hover_text)
            else:
                noise_peaks['indices'].append(peak_idx)
                noise_peaks['customdata'].append(full_hover_text)

        # --- Handle any raw peaks that were never classified ---
        for peak_idx in all_raw_peaks:
            if peak_idx not in classified_indices:
                hover_text = (f"<b>Type:</b> Unclassified<br>"
                              f"<b>Time:</b> {peak_idx / self.sample_rate:.2f}s<br>"
                              f"<b>Amp:</b> {audio_envelope[peak_idx]:.0f}<br>"
                              "<b>Details:</b> Peak was not evaluated by the classifier.")
                noise_peaks['indices'].append(peak_idx)
                noise_peaks['customdata'].append(hover_text)

        # A simplified hovertemplate that displays the pre-formatted custom data
        hovertemplate = "%{customdata}<extra></extra>"

        # --- Add traces to the plot ---
        if s1_peaks['indices']:
            times_dt = pd.to_datetime([datetime.datetime.fromtimestamp(0) + datetime.timedelta(seconds=t) for t in
                                       (np.array(s1_peaks['indices']) / self.sample_rate)])
            self.fig.add_trace(
                go.Scatter(x=times_dt, y=audio_envelope[s1_peaks['indices']], mode='markers', name='S1 Beats',
                           marker=dict(color='#e36f6f', size=8, symbol='diamond'),
                           customdata=s1_peaks['customdata'],
                           hovertemplate=hovertemplate), secondary_y=False)

        if s2_peaks['indices']:
            times_dt = pd.to_datetime([datetime.datetime.fromtimestamp(0) + datetime.timedelta(seconds=t) for t in
                                       (np.array(s2_peaks['indices']) / self.sample_rate)])
            self.fig.add_trace(
                go.Scatter(x=times_dt, y=audio_envelope[s2_peaks['indices']], mode='markers', name='S2 Beats',
                           marker=dict(color='orange', symbol='circle', size=6),
                           customdata=s2_peaks['customdata'],
                           hovertemplate=hovertemplate), secondary_y=False)

        if noise_peaks['indices']:
            times_dt = pd.to_datetime([datetime.datetime.fromtimestamp(0) + datetime.timedelta(seconds=t) for t in
                                       (np.array(noise_peaks['indices']) / self.sample_rate)])
            self.fig.add_trace(
                go.Scatter(x=times_dt, y=audio_envelope[noise_peaks['indices']], mode='markers', name='Noise/Rejected',
                           marker=dict(color='grey', symbol='x', size=6),
                           customdata=noise_peaks['customdata'],
                           hovertemplate=hovertemplate), secondary_y=False)

    def _add_bpm_hrv_traces(self, smoothed_bpm, analysis_data, windowed_hrv_df):
        """Adds BPM, BPM trend, and HRV traces."""
        if smoothed_bpm is not None and not smoothed_bpm.empty:
            self.fig.add_trace(go.Scatter(x=smoothed_bpm.index, y=smoothed_bpm.values, name="Average BPM", line=dict(color="#4a4a4a", width=3)), secondary_y=True)

        if "long_term_bpm_series" in analysis_data and not analysis_data["long_term_bpm_series"].empty:
            lt_series = analysis_data["long_term_bpm_series"]
            # Create datetime index for plotting
            start_datetime = datetime.datetime.fromtimestamp(0)
            lt_times_dt = pd.to_datetime([start_datetime + datetime.timedelta(seconds=t) for t in lt_series.index])
            self.fig.add_trace(go.Scatter(
                x=lt_times_dt,
                y=lt_series.values,
                name="BPM Trend (Belief)",
                line=dict(color='orange', width=2, dash='dot'),
                visible='legendonly'),
                secondary_y=True)
        if windowed_hrv_df is not None and not windowed_hrv_df.empty and 'time' in windowed_hrv_df and 'rmssdc' in windowed_hrv_df and 'sdnn' in windowed_hrv_df:
            hrv_times_dt = pd.to_datetime([datetime.datetime.fromtimestamp(0) + datetime.timedelta(seconds=t) for t in windowed_hrv_df['time']])
            self.fig.add_trace(go.Scatter(x=hrv_times_dt, y=windowed_hrv_df['rmssdc'], name="RMSSDc", line=dict(color='cyan', width=2), visible='legendonly'), secondary_y=True)
            self.fig.add_trace(go.Scatter(x=hrv_times_dt, y=windowed_hrv_df['sdnn'], name="SDNN", line=dict(color='magenta', width=2), visible='legendonly'), secondary_y=True)


    def _add_annotations_and_summary(self, smoothed_bpm, hrv_summary, hrr_stats, peak_recovery_stats):
        """Adds min/max BPM annotations and the main summary box."""
        if smoothed_bpm is not None and not smoothed_bpm.empty:
            max_bpm_val = smoothed_bpm.max()
            min_bpm_val = smoothed_bpm.min()
            max_bpm_time = smoothed_bpm.idxmax()
            min_bpm_time = smoothed_bpm.idxmin()

            # Add annotation for the maximum BPM
            self.fig.add_annotation(x=max_bpm_time, y=max_bpm_val,
                                    text=f"Max: {max_bpm_val:.1f} BPM",
                                    showarrow=True, arrowhead=1, ax=20, ay=-40,
                                    font=dict(color="#e36f6f"), yref="y2")

            # Add annotation for the minimum BPM
            self.fig.add_annotation(x=min_bpm_time, y=min_bpm_val,
                                    text=f"Min: {min_bpm_val:.1f} BPM",
                                    showarrow=True, arrowhead=1, ax=20, ay=40,
                                    font=dict(color="#a3d194"), yref="y2")

        if hrv_summary:
            annotation_text = "<b>Analysis Summary</b><br>"
            if hrv_summary.get('avg_bpm') is not None:
                annotation_text += f"Avg/Min/Max BPM: {hrv_summary['avg_bpm']:.1f} / {hrv_summary['min_bpm']:.1f} / {hrv_summary['max_bpm']:.1f}<br>"
            if hrr_stats and hrr_stats.get('hrr_value_bpm') is not None:
                annotation_text += f"<b>1-Min HRR: {hrr_stats['hrr_value_bpm']:.1f} BPM Drop</b><br>"
            if peak_recovery_stats and peak_recovery_stats.get('slope_bpm_per_sec') is not None:
                annotation_text += f"<b>Peak Recovery Rate: {peak_recovery_stats['slope_bpm_per_sec']:.2f} BPM/sec</b><br>"
            if hrv_summary.get('avg_rmssdc') is not None:
                annotation_text += f"Avg. Corrected RMSSD: {hrv_summary['avg_rmssdc']:.2f}<br>"
            if hrv_summary.get('avg_sdnn') is not None:
                annotation_text += f"Avg. Windowed SDNN: {hrv_summary['avg_sdnn']:.2f} ms"

            self.fig.add_annotation(text=annotation_text, align='left', showarrow=False,
                                    xref='paper', yref='paper', x=0.02, y=0.98,
                                    bordercolor='black', borderwidth=1,
                                    bgcolor='rgba(255, 253, 231, 0.4)')

    def _add_slope_traces(self, major_inclines, major_declines, peak_recovery_stats, peak_exertion_stats):
        """Adds traces for major exertion and recovery periods."""
        if major_inclines:
            for i, incline in enumerate(major_inclines):
                c_data = [incline['duration_sec'], incline['bpm_increase'], incline['slope_bpm_per_sec']]
                self.fig.add_trace(go.Scatter(
                    x=[incline['start_time'], incline['end_time']],
                    y=[incline['start_bpm'], incline['end_bpm']],
                    mode='lines', line=dict(color="purple", width=4, dash="dash"),
                    name='Exertion', legendgroup='Exertion',
                    showlegend=(i == 0), visible='legendonly', yaxis='y2',
                    hovertemplate="<b>Exertion Period</b><br>Duration: %{customdata[0]:.1f}s<br>BPM Increase: %{customdata[1]:.1f}<br>Slope: %{customdata[2]:.2f} BPM/sec<extra></extra>",
                    customdata=np.array([c_data, c_data])))

        if major_declines:
            for i, decline in enumerate(major_declines):
                c_data = [decline['duration_sec'], decline['bpm_decrease'], decline['slope_bpm_per_sec']]
                self.fig.add_trace(go.Scatter(
                    x=[decline['start_time'], decline['end_time']],
                    y=[decline['start_bpm'], decline['end_bpm']],
                    mode='lines', line=dict(color="#2ca02c", width=4, dash="dash"),
                    name='Recovery', legendgroup='Recovery',
                    showlegend=(i == 0), visible='legendonly', yaxis='y2',
                    hovertemplate="<b>Recovery Period</b><br>Duration: %{customdata[0]:.1f}s<br>BPM Decrease: %{customdata[1]:.1f}<br>Slope: %{customdata[2]:.2f} BPM/sec<extra></extra>",
                    customdata=np.array([c_data, c_data])))


        if peak_recovery_stats:
            stats = peak_recovery_stats
            self.fig.add_trace(go.Scatter(
                x=[stats['start_time'], stats['end_time']],
                y=[stats['start_bpm'], stats['end_bpm']],
                mode='lines', line=dict(color="#ff69b4", width=5, dash="solid"),
                name='Peak Recovery Slope', legendgroup='Steepest Slopes',
                visible='legendonly', yaxis='y2',
                hovertemplate="<b>Peak Recovery Slope</b><br>Slope: %{customdata[0]:.2f} BPM/sec<br>Duration: %{customdata[1]:.1f}s<extra></extra>",
                customdata=np.array([[stats['slope_bpm_per_sec'], stats['duration_sec']]]*2)))

        if peak_exertion_stats:
            stats = peak_exertion_stats
            self.fig.add_trace(go.Scatter(
                x=[stats['start_time'], stats['end_time']],
                y=[stats['start_bpm'], stats['end_bpm']],
                mode='lines', line=dict(color="#9d32a8", width=5, dash="solid"),
                name='Peak Exertion Slope', legendgroup='Steepest Slopes',
                visible='legendonly', yaxis='y2',
                hovertemplate="<b>Peak Exertion Slope</b><br>Slope: +%{customdata[0]:.2f} BPM/sec<br>Duration: %{customdata[1]:.1f}s<extra></extra>",
                customdata=np.array([[stats['slope_bpm_per_sec'], stats['duration_sec']]]*2)))

class ReportGenerator:
    """Handles the creation of text-based analysis reports."""
    def __init__(self, file_name: str, output_directory: str):
        self.file_name = file_name
        self.output_directory = output_directory
        self.file_name_no_ext = os.path.splitext(file_name)[0]
        self.base_name = os.path.basename(self.file_name_no_ext)

    def save_analysis_settings(self, start_bpm_hint: Optional[float]):
        """Saves the user-configurable settings to a JSON file."""
        settings_path = os.path.join(self.output_directory, f"{self.base_name}_Analysis_Settings.json")
        settings_to_save = {'start_bpm_hint': start_bpm_hint}
        try:
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(settings_to_save, f, indent=4)
            logging.info(f"Analysis settings saved to {settings_path}")
        except Exception as e:
            logging.error(f"Could not save analysis settings file. Error: {e}")

    def save_analysis_summary(self, final_metrics: Dict):
        """Saves a comprehensive Markdown summary of the analysis results."""
        output_path = os.path.join(self.output_directory, f"{self.base_name}_Analysis_Summary.md")

        with open(output_path, "w", encoding="utf-8") as f:
            self._write_summary_header(f)
            self._write_overall_summary(f, final_metrics.get('hrv_summary'), final_metrics.get('hrr_stats'))
            self._write_steepest_slopes(f, final_metrics.get('peak_exertion_stats'),
                                        final_metrics.get('peak_recovery_stats'))
            self._write_significant_changes(f, final_metrics.get('major_inclines'), final_metrics.get('major_declines'))
            self._write_heartbeat_data_table(f, final_metrics.get('smoothed_bpm'), final_metrics.get('bpm_times'))

        logging.info(f"Markdown analysis summary saved to {output_path}")

    def create_chronological_log(self, audio_envelope: np.ndarray, sample_rate: int, all_raw_peaks: np.ndarray, analysis_data: Dict, final_metrics: Dict):
        """Creates a detailed, readable debug log file."""
        output_log_path = os.path.join(self.output_directory, f"{self.base_name}_Debug_Log.md")
        logging.info(f"Generating readable debug log at '{output_log_path}'...")
        merged_df = self._prepare_log_data(audio_envelope, sample_rate, all_raw_peaks, analysis_data, final_metrics.get('smoothed_bpm'), final_metrics.get('bpm_times'))
        with open(output_log_path, "w", encoding="utf-8") as log_file:
            if merged_df is None or merged_df.empty:
                log_file.write("# No significant events detected to log.\n")
            else:
                self._write_log_events(log_file, merged_df)
        logging.info("Debug log generation complete.")

    def _prepare_log_data(self, audio_envelope, sample_rate, all_raw_peaks, analysis_data, smoothed_bpm, bpm_times):
        """Prepares and merges all data sources into a single DataFrame for logging."""
        events = []
        debug_info = analysis_data.get('beat_debug_info', {})

        for p in all_raw_peaks:
            reason = debug_info.get(p)
            if reason:
                events.append({'time': p / sample_rate, 'type': 'Peak', 'amp': audio_envelope[p], 'reason': reason})
        if 'trough_indices' in analysis_data:
            for p in analysis_data['trough_indices']:
                events.append({'time': p / sample_rate, 'type': 'Trough', 'amp': audio_envelope[p], 'reason': ''})

        if not events: return None
        events_df = pd.DataFrame(events).sort_values(by='time').set_index('time')

        master_df = pd.DataFrame(index=np.arange(len(audio_envelope)) / sample_rate)
        if 'dynamic_noise_floor_series' in analysis_data:
            master_df['noise_floor'] = analysis_data['dynamic_noise_floor_series'].values
        if smoothed_bpm is not None and not smoothed_bpm.empty:
            smoothed_bpm_sec_index = pd.Series(data=smoothed_bpm.values, index=bpm_times).groupby(level=0).mean()
            master_df['smoothed_bpm'] = smoothed_bpm_sec_index
        if 'long_term_bpm_series' in analysis_data and not analysis_data['long_term_bpm_series'].empty:
            master_df['lt_bpm'] = analysis_data['long_term_bpm_series'].groupby(level=0).mean()

        master_df.ffill(inplace=True)

        return pd.merge_asof(left=events_df, right=master_df, left_index=True,
                             right_index=True, direction='nearest', tolerance=pd.Timedelta(seconds=0.5).total_seconds())

    def _write_log_events(self, log_file, merged_df):
        log_file.write(f"# Chronological Debug Log for {os.path.basename(self.file_name)}\n")
        log_file.write(f"Analysis performed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for row in merged_df.itertuples(name="LogEvent"):
            log_file.write(f"## Time: `{row.Index:.4f}s`\n")

            if row.type == 'Trough':
                log_file.write("**Trough Detected**\n")
            else:
                raw_reason = getattr(row, 'reason', '')
                if not raw_reason or raw_reason == 'Unknown':
                    log_file.write("**Unclassified Peak**\n")
                else:
                    parts = raw_reason.split('§')
                    final_peak_type, details_list = parts[0], parts[1:]
                    log_file.write(f"**{final_peak_type}.**\n")

                    i = 0
                    while i < len(details_list):
                        tag = details_list[i]
                        value = details_list[i + 1] if (i + 1) < len(details_list) else ""
                        formatted_details = ""

                        if "PAIRING" in tag:
                            formatted_details = "\n".join(Plotter.format_pairing_details_list(value))
                        elif "LONE_S1_REJECT_REASON" in tag:
                            formatted_details = "\n".join(Plotter.format_lone_s1_details_list(value))
                        elif "LONE_S1_VALIDATE_REASON" in tag:
                            formatted_details = "\n".join(Plotter.format_lone_s1_details_list(value))
                        elif "ORIGINAL_REASON" in tag:
                            formatted_details = f"- Original Classification:\n    - `{value}`"

                        if formatted_details:
                            log_file.write(f"{formatted_details}\n")

                        i += 2  # Move past the tag and its value

            # Write all available metrics for every event type
            metrics = {
                "Raw Amp": getattr(row, 'amp', None),
                "Noise Floor": getattr(row, 'noise_floor', None),
                "Average BPM (Smoothed)": getattr(row, 'smoothed_bpm', None),
                "Long-Term BPM (Belief)": getattr(row, 'lt_bpm', None)
            }
            for name, value in metrics.items():
                if pd.notna(value):
                    log_file.write(f"- **{name}**: `{value:.1f}`\n")

            log_file.write("\n\n")

    def _write_summary_header(self, f):
        f.write(f"# Analysis Report for: {os.path.basename(self.file_name)}\n")
        f.write(f"*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

    def _write_overall_summary(self, f, hrv_summary, hrr_stats):
        """Writes the main summary table to the markdown report file."""
        f.write("## Overall Summary\n\n| Metric | Value |\n|:---|:---|\n")
        if hrv_summary:
            if hrv_summary.get('avg_bpm') is not None:
                f.write(f"| **Average BPM** | {hrv_summary['avg_bpm']:.1f} BPM |\n")
                f.write(f"| **BPM Range** | {hrv_summary['min_bpm']:.1f} to {hrv_summary['max_bpm']:.1f} BPM |\n")
            if hrv_summary.get('avg_rmssdc') is not None:
                f.write(f"| **Avg. Corrected RMSSD** | {hrv_summary['avg_rmssdc']:.2f} |\n")
            if hrv_summary.get('avg_sdnn') is not None:
                f.write(f"| **Avg. Windowed SDNN** | {hrv_summary['avg_sdnn']:.2f} ms |\n")
        if hrr_stats and hrr_stats.get('hrr_value_bpm') is not None:
            f.write(f"| **1-Minute HRR** | {hrr_stats['hrr_value_bpm']:.1f} BPM Drop |\n")
        f.write("\n")

    def _write_steepest_slopes(self, f, peak_exertion_stats, peak_recovery_stats):
        """Writes the peak exertion and recovery slope data to the markdown report."""
        f.write("## Steepest Slopes Analysis\n\n### Peak Exertion (Fastest HR Increase)\n\n")
        if peak_exertion_stats:
            pes = peak_exertion_stats
            f.write("| Attribute | Value |\n|:---|:---|\n")
            f.write(f"| **Rate** | `+{pes['slope_bpm_per_sec']:.2f}` BPM/second |\n")
            f.write(f"| **Period** | {pes['start_time'].strftime('%M:%S')} to {pes['end_time'].strftime('%M:%S')} |\n")
            f.write(f"| **Duration** | {pes['duration_sec']:.1f} seconds |\n")
            f.write(f"| **BPM Change** | {pes['start_bpm']:.1f} to {pes['end_bpm']:.1f} BPM |\n\n")
        else:
            f.write("*No significant peak exertion period found.*\n\n")

        f.write("### Peak Recovery (Fastest HR Decrease)\n\n")
        if peak_recovery_stats:
            prs = peak_recovery_stats
            f.write("| Attribute | Value |\n|:---|:---|\n")
            f.write(f"| **Rate** | `{prs['slope_bpm_per_sec']:.2f}` BPM/second |\n")
            f.write(f"| **Period** | {prs['start_time'].strftime('%M:%S')} to {prs['end_time'].strftime('%M:%S')} |\n")
            f.write(f"| **Duration** | {prs['duration_sec']:.1f} seconds |\n")
            f.write(f"| **BPM Change** | {prs['start_bpm']:.1f} to {prs['end_bpm']:.1f} BPM |\n\n")
        else:
            f.write("*No significant peak recovery period found post-peak.*\n\n")

    def _write_significant_changes(self, f, major_inclines, major_declines):
        """Writes the sections on sustained heart rate increases and decreases to the report file."""
        f.write("## All Significant HR Changes\n\n### Exertion Periods (Sustained HR Increase)\n\n")
        if major_inclines:
            epoch = datetime.datetime.fromtimestamp(0)
            for incline in major_inclines:
                # Calculate start and end times in seconds from the datetime objects
                start_sec = (incline['start_time'] - epoch).total_seconds()
                end_sec = (incline['end_time'] - epoch).total_seconds()
                f.write(f"- **From {start_sec:.1f}s to {end_sec:.1f}s:** Duration={incline['duration_sec']:.1f}s, Change=`+{incline['bpm_increase']:.1f}` BPM\n")
        else:
            f.write("*None found.*\n")

        f.write("\n### Recovery Periods (Sustained HR Decrease)\n\n")
        if major_declines:
            epoch = datetime.datetime.fromtimestamp(0)
            for decline in major_declines:
                # Calculate start and end times in seconds from the datetime objects
                start_sec = (decline['start_time'] - epoch).total_seconds()
                end_sec = (decline['end_time'] - epoch).total_seconds()
                f.write(f"- **From {start_sec:.1f}s to {end_sec:.1f}s:** Duration={decline['duration_sec']:.1f}s, Change=`-{decline['bpm_decrease']:.1f}` BPM\n")
        else:
            f.write("*None found.*\n")
        f.write("\n")

    def _write_heartbeat_data_table(self, f, smoothed_bpm, bpm_times):
        """Writes the final time-series BPM data to a markdown table in the report file."""
        f.write("## Heartbeat Data (BPM over Time)\n\n| Time (s) | Average BPM |\n|:---:|:---:|\n")
        if smoothed_bpm is not None and not smoothed_bpm.empty and bpm_times is not None:
            # Use the raw numpy array of times for the table and match it with the smoothed BPM values
            for t, bpm in zip(bpm_times, smoothed_bpm.values):
                if not np.isnan(bpm):
                    f.write(f"| {t:.2f} | {bpm:.1f} |\n")
        else:
            f.write("| *No data* | *No data* |\n")

# --- Standalone Utility & Pipeline Functions ---

def convert_to_wav(file_path: str, target_path: str) -> bool:
    """Converts a given audio file to WAV format."""
    if not AudioSegment:
        raise ImportError("Pydub/FFmpeg is required for audio conversion.")

    logging.info(f"Converting {os.path.basename(file_path)} to WAV format...")
    try:
        # Load the audio file
        sound = AudioSegment.from_file(file_path)
        # Convert to mono
        sound = sound.set_channels(1)
        # Export as WAV
        sound.export(target_path, format="wav")
        return True
    except Exception as e:
        logging.error(f"Could not convert file {file_path}. Error: {e}")
        return False

def preprocess_audio(file_path: str, params: Dict, output_directory: str) -> Tuple[np.ndarray, int]:
    """Reads, filters, and prepares the audio envelope for analysis."""
    downsample_factor = params['downsample_factor']
    save_debug_file = params['save_filtered_wav']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample_rate, audio_data = wavfile.read(file_path)
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    lowcut, highcut = 20, 150  # Hardcoded bandpass frequencies

    # Check if the downsample factor is too aggressive for the filter settings.
    max_safe_downsample = int((sample_rate / (highcut * 2)) - 1)

    if downsample_factor > max_safe_downsample:
        logging.warning(
            f"Original 'downsample_factor' of {downsample_factor} is too high for a "
            f"{highcut}Hz filter with a {sample_rate}Hz sample rate."
        )
        downsample_factor = max(1, max_safe_downsample)
        logging.warning(f"Adjusting 'downsample_factor' to a safe value of {downsample_factor}.")

    if downsample_factor > 1:
        new_sample_rate = sample_rate // downsample_factor
        audio_downsampled = audio_data[::downsample_factor]
    else:
        new_sample_rate = sample_rate
        audio_downsampled = audio_data

    nyquist = 0.5 * new_sample_rate
    low, high = lowcut / nyquist, highcut / nyquist

    if high >= 1.0:
        raise ValueError(f"Cannot create a {highcut}Hz filter. The effective sample rate of {new_sample_rate}Hz is too low.")

    b, a = butter(2, [low, high], btype='band')
    audio_filtered = filtfilt(b, a, audio_downsampled)

    if save_debug_file:
        debug_path = f"{os.path.splitext(file_path)[0]}_filtered_debug.wav"
        normalized_audio = np.int16(audio_filtered / np.max(np.abs(audio_filtered)) * 32767)
        wavfile.write(debug_path, new_sample_rate, normalized_audio)

    audio_abs = np.abs(audio_filtered)
    window_size = new_sample_rate // 10
    audio_envelope = pd.Series(audio_abs).rolling(window=window_size, min_periods=1, center=True).mean().values

    if params['save_filtered_wav']:
        base_name = os.path.basename(os.path.splitext(file_path)[0])
        debug_path = os.path.join(output_directory, f"{base_name}_filtered_debug.wav")
        normalized_audio = np.int16(audio_filtered / np.max(np.abs(audio_filtered)) * 32767)
        wavfile.write(debug_path, new_sample_rate, normalized_audio)

    return audio_envelope, new_sample_rate

def _calculate_dynamic_noise_floor(audio_envelope: np.ndarray, sample_rate: int, params: Dict) -> Tuple[pd.Series, np.ndarray]:
    """Calculates a dynamic noise floor based on a sanitized set of audio troughs."""
    min_peak_dist_samples = int(params['min_peak_distance_sec'] * sample_rate)
    trough_prom_thresh = np.quantile(audio_envelope, params['trough_prominence_quantile'])

    # --- STEP 1: Find all potential troughs initially ---
    all_trough_indices, _ = find_peaks(-audio_envelope, distance=min_peak_dist_samples, prominence=trough_prom_thresh)

    # If we don't have enough troughs to begin with, fall back to a simple static floor.
    if len(all_trough_indices) < 5:
        logging.warning("Not enough troughs found for sanitization. Using a static noise floor.")
        fallback_value = np.quantile(audio_envelope, params['noise_floor_quantile'])
        dynamic_noise_floor = pd.Series(fallback_value, index=np.arange(len(audio_envelope)))
        return dynamic_noise_floor, all_trough_indices

    # --- STEP 2: Create a preliminary 'draft' noise floor from ALL troughs ---
    # This draft version is used only to evaluate the troughs themselves.
    trough_series_draft = pd.Series(index=all_trough_indices, data=audio_envelope[all_trough_indices])
    dense_troughs_draft = trough_series_draft.reindex(np.arange(len(audio_envelope))).interpolate()
    noise_window_samples = int(params['noise_window_sec'] * sample_rate)
    quantile_val = params['noise_floor_quantile']
    draft_noise_floor = dense_troughs_draft.rolling(window=noise_window_samples, min_periods=3, center=True).quantile(quantile_val)
    draft_noise_floor = draft_noise_floor.bfill().ffill() # Fill any gaps

    # --- STEP 3: Sanitize the trough list ---
    # remove any toughs too far from the noise floor
    sanitized_trough_indices = []
    rejection_multiplier = params.get('trough_rejection_multiplier', 4.0)
    for trough_idx in all_trough_indices:
        trough_amp = audio_envelope[trough_idx]
        floor_at_trough = draft_noise_floor.iloc[trough_idx]
        # Keep the trough only if it's not too high above the draft floor
        if not pd.isna(floor_at_trough) and trough_amp <= (rejection_multiplier * floor_at_trough):
            sanitized_trough_indices.append(trough_idx)

    logging.info(f"Trough Sanitization: Kept {len(sanitized_trough_indices)} of {len(all_trough_indices)} initial troughs.")

    # --- STEP 4: Calculate more accurate noise floor using only sanitized troughs ---
    if len(sanitized_trough_indices) > 2:
        trough_series_final = pd.Series(index=sanitized_trough_indices, data=audio_envelope[sanitized_trough_indices])
        dense_troughs_final = trough_series_final.reindex(np.arange(len(audio_envelope))).interpolate()
        dynamic_noise_floor = dense_troughs_final.rolling(window=noise_window_samples, min_periods=3, center=True).quantile(quantile_val)
        dynamic_noise_floor = dynamic_noise_floor.bfill().ffill()
    else:
        # If sanitization removed too many troughs, it's safer to use the original draft floor.
        logging.warning("Not enough sanitized troughs remaining. Using non-sanitized floor as fallback.")
        dynamic_noise_floor = draft_noise_floor

    # Final check for any remaining null values
    if dynamic_noise_floor.isnull().all():
         fallback_val = np.quantile(audio_envelope, 0.1)
         dynamic_noise_floor = pd.Series(fallback_val, index=np.arange(len(audio_envelope)))

    return dynamic_noise_floor, np.array(sanitized_trough_indices)


def calculate_blended_confidence(deviation: float, bpm: float, params: Dict) -> float:
    """
    Calculates a confidence score for pairing two peaks based on amplitude deviation.
    This version dynamically constructs the confidence curve based on the current BPM
    to reflect physiological expectations (heart's contractility).
    """
    # Get the anchor points for our dynamic model from params
    bpm_points = [params['contractility_bpm_low'], params['contractility_bpm_high']]
    deviation_points = [0.0, 0.25, 0.40, 0.80, 1.0]  # Hardcoded deviation points

    # Get the two boundary curves (for low and high BPM)
    curve_low = np.array([0.9, 0.9, 0.7, 0.1, 0.1])  # Hardcoded low BPM curve
    curve_high = np.array([0.1, 0.5, 0.75, 0.65, 0])  # Hardcoded high BPM curve

    # --- Create the Live Confidence Curve ---
    # Calculate how far the current BPM is into the transition zone (0.0 to 1.0)
    blend_ratio = np.clip((bpm - bpm_points[0]) / (bpm_points[1] - bpm_points[0]), 0, 1)

    # Linearly interpolate between the low and high curves to get the live curve
    live_confidence_curve = curve_low + (curve_high - curve_low) * blend_ratio

    final_confidence = np.interp(deviation, deviation_points, live_confidence_curve)

    return final_confidence


def _adjust_confidence_with_stability_and_ratio(confidence: float, s1_idx: int, s2_idx: int, audio_envelope: np.ndarray, dynamic_noise_floor: pd.Series,
                                               long_term_bpm: float, pairing_ratio: float, params: Dict, sample_rate: int,
                                               peak_bpm_time_sec: Optional[float], recovery_end_time_sec: Optional[float], beat_count: int) -> Tuple[float, str]:
    """Applies a full suite of confidence adjustments based on rhythm stability and S1/S2 strength ratio."""
    reason = ""

    # --- 1. Universal Stability Pre-Adjustment ---
    if beat_count >= 5:
        floor = params.get("stability_confidence_floor", 0.85)
        ceiling = params.get("stability_confidence_ceiling", 1.10)
        stability_factor = np.interp(pairing_ratio, [0.0, 1.0], [floor, ceiling])
        confidence *= stability_factor
        reason += f"\n- Stability Pre-Adjust: x{stability_factor:.2f} (Pairing Ratio: {pairing_ratio:.0%})"

    # --- 2. Calculate Peak Strengths and Expected Ratio ---
    s1_strength = max(0, audio_envelope[s1_idx] - dynamic_noise_floor.iloc[s1_idx])
    s2_strength = max(0, audio_envelope[s2_idx] - dynamic_noise_floor.iloc[s2_idx])
    current_s2_s1_strength_ratio = s2_strength / (s1_strength + 1e-9)

    # Determine expected ratio based on BPM and recovery state
    is_in_recovery = (peak_bpm_time_sec is not None and recovery_end_time_sec is not None and
                      peak_bpm_time_sec < (s1_idx / sample_rate) < recovery_end_time_sec)
    effective_bpm = max(long_term_bpm, params['contractility_bpm_low']) if is_in_recovery else long_term_bpm
    max_expected_s2_s1_ratio = np.interp(effective_bpm,
                                       [params['contractility_bpm_low'], params['contractility_bpm_high']],
                                       [params['s2_s1_ratio_low_bpm'], params['s2_s1_ratio_high_bpm']])

    # --- 3. Apply Final Dynamic Boost or Penalty Amount ---
    if current_s2_s1_strength_ratio > max_expected_s2_s1_ratio:
        # PENALTY is scaled by the severity of the violation.
        min_penalty = params.get("penalty_amount_min", 0.15)
        max_penalty = params.get("penalty_amount_max", 0.40)
        violation_severity = current_s2_s1_strength_ratio / max_expected_s2_s1_ratio
        severity_scale = np.clip((violation_severity - 1.0) / 2.0, 0, 1)
        penalty_range = max_penalty - min_penalty
        penalty_amount = min_penalty + (severity_scale * penalty_range)
        confidence -= penalty_amount
        reason += f"\n- PENALIZED by {penalty_amount:.2f} (S2 Str. Ratio {current_s2_s1_strength_ratio:.1f}x > Expected {max_expected_s2_s1_ratio:.1f}x)"

    elif s1_strength > (s2_strength * params.get('s1_s2_boost_ratio', 1.2)):
        # BOOST is now also scaled by severity.
        min_boost = params.get("boost_amount_min", 0.10)
        max_boost = params.get("boost_amount_max", 0.35)
        actual_s1_s2_ratio = s1_strength / (s2_strength + 1e-9)
        boost_threshold_ratio = params.get('s1_s2_boost_ratio', 1.2)
        exceedance_scale = np.clip((actual_s1_s2_ratio - boost_threshold_ratio) / (4.0 - boost_threshold_ratio), 0, 1)
        boost_range = max_boost - min_boost
        boost_amount = min_boost + (exceedance_scale * boost_range)
        confidence += boost_amount
        reason += f"\n- BOOSTED by {boost_amount:.2f} (S1 Str. Ratio {actual_s1_s2_ratio:.1f}x > S2)"

    return max(0.0, min(1.0, confidence)), reason



def calculate_lone_s1_confidence(current_peak_idx: int, last_s1_idx: int, long_term_bpm: float, audio_envelope: np.ndarray,
                                 dynamic_noise_floor: pd.Series, sample_rate: int, params: Dict) -> Tuple[float, str]:
    """
    Calculates a confidence score for a Lone S1 candidate based on a weighted average of
    its rhythmic timing and its amplitude consistency with the previous beat.
    """
    # --- 1. Calculate Rhythmic Fit Score ---
    expected_rr_sec = 60.0 / long_term_bpm
    actual_rr_sec = (current_peak_idx - last_s1_idx) / sample_rate
    rhythm_deviation_pct = abs(actual_rr_sec - expected_rr_sec) / expected_rr_sec

    rhythm_score = np.interp(
        rhythm_deviation_pct,
        [0.0, 0.15, 0.30, 0.50],  # Hardcoded rhythm deviation points
        [1.0, 0.8, 0.4, 0.0]      # Hardcoded rhythm confidence curve
    )
    rhythm_reason = f"Rhythm Fit={rhythm_score:.2f} (Interval {actual_rr_sec:.3f}s vs Expected {expected_rr_sec:.3f}s)"

    # --- 2. Calculate Amplitude Fit Score ---
    last_s1_strength = max(0, audio_envelope[last_s1_idx] - dynamic_noise_floor.iloc[last_s1_idx])
    current_peak_strength = max(0, audio_envelope[current_peak_idx] - dynamic_noise_floor.iloc[current_peak_idx])
    amplitude_ratio = current_peak_strength / (last_s1_strength + 1e-9)

    amplitude_score = np.interp(
        amplitude_ratio,
        [0.0, 0.4, 0.7, 1.0],      # Hardcoded amplitude ratio points
        [0.0, 0.4, 0.8, 1.0]       # Hardcoded amplitude confidence curve
    )
    amplitude_reason = f"Amplitude Fit={amplitude_score:.2f} (Strength Ratio {amplitude_ratio:.2f}x)"

    # --- 3. Combine Scores with Weights ---
    rhythm_weight = params.get('lone_s1_rhythm_weight', 0.65)
    amplitude_weight = params.get('lone_s1_amplitude_weight', 0.35)
    final_confidence = (rhythm_score * rhythm_weight) + (amplitude_score * amplitude_weight)

    reason_str = f"{rhythm_reason}, {amplitude_reason}"
    return final_confidence, reason_str

def update_long_term_bpm(new_rr_sec: float, current_long_term_bpm: float, params: Dict) -> float:
    """Updates the long-term BPM belief based on a new R-R interval."""
    instant_bpm = 60.0 / new_rr_sec
    lr = 0.05  # Hardcoded learning rate
    max_change_per_beat = 3.0  # Hardcoded max change per beat

    # Calculate the target BPM using an exponential moving average
    target_bpm = ((1 - lr) * current_long_term_bpm) + (lr * instant_bpm)

    # Limit how much the BPM can change in a single beat (a "speed limiter")
    max_change = max_change_per_beat * new_rr_sec # Scale limit by interval duration
    proposed_change = target_bpm - current_long_term_bpm
    limited_change = np.clip(proposed_change, -max_change, max_change)

    # Apply the limited change and enforce absolute min/max BPM boundaries
    new_bpm = current_long_term_bpm + limited_change
    return max(params['min_bpm'], min(new_bpm, params['max_bpm']))

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
                pass  # Do nothing, effectively dropping the current_peak
        else:
            # NO CONFLICT: The interval is plausible. Add the peak to our corrected list.
            corrected_peaks.append(current_peak)

    final_peak_count = len(corrected_peaks)
    if final_peak_count < len(peaks):
        logging.info(f"Correction complete. Removed {len(peaks) - final_peak_count} peak(s). Final count: {final_peak_count}")
    else:
        logging.info("Correction pass complete. No rhythmic conflicts found.")
    return np.array(corrected_peaks)


def _fix_rhythmic_discontinuities(s1_peaks: np.ndarray, all_raw_peaks: np.ndarray, debug_info: Dict,
                                  audio_envelope: np.ndarray, dynamic_noise_floor: pd.Series, params: Dict,
                                  sample_rate: int) -> Tuple[np.ndarray, Dict, int]:
    """
    Identifies and attempts to fix rhythmic discontinuities by re-evaluating misclassified peaks.
    """
    def log_debug(msg):
        logging.info(f"[Correction DEBUG] {msg}")

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
            gap_candidates = [p for p in all_raw_peaks if
                              s1_start_idx < p < s1_end_idx and "Noise" in debug_info.get(p, "")]
            for candidate_s1 in gap_candidates:
                if candidate_s1 in peaks_to_add: continue
                current_raw_idx = np.searchsorted(all_raw_peaks, candidate_s1)
                if current_raw_idx + 1 >= len(all_raw_peaks): continue
                candidate_s2 = all_raw_peaks[current_raw_idx + 1]
                if candidate_s2 >= s1_end_idx or "Noise" not in debug_info.get(candidate_s2, ""): continue

                s1_strength = max(0, audio_envelope[candidate_s1] - dynamic_noise_floor.iloc[candidate_s1])
                is_strong_s1 = s1_strength > (
                            params["penalty_waiver_strength_ratio"] * dynamic_noise_floor.iloc[candidate_s1])
                is_ratio_plausible = (audio_envelope[candidate_s2] / (audio_envelope[candidate_s1] + 1e-9)) < params[
                    "penalty_waiver_max_s2_s1_ratio"]

                if is_strong_s1 and is_ratio_plausible:
                    log_debug(f"  - SUCCESS: Re-labeling S1/S2 pair at {candidate_s1 / sample_rate:.2f}s.")
                    corrections_made += 1
                    peaks_to_add.add(candidate_s1)
                    original_reason_s1 = corrected_debug_info.get(candidate_s1, "Noise")
                    corrected_debug_info[
                        candidate_s1] = f"{PeakType.S1_CORRECTED_GAP.value}§ORIGINAL_REASON§{original_reason_s1}"
                    original_reason_s2 = corrected_debug_info.get(candidate_s2, "Noise")
                    corrected_debug_info[
                        candidate_s2] = f"{PeakType.S2_CORRECTED_GAP.value}§ORIGINAL_REASON§{original_reason_s2}"
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
                peaks_to_remove.add(beat_A_idx)
                log_debug(f"  - Removing weaker peak at {beat_A_idx / sample_rate:.2f}s.")
                corrections_made += 1
            else:
                peaks_to_remove.add(beat_B_idx)
                log_debug(f"  - Removing weaker peak at {beat_B_idx / sample_rate:.2f}s.")
                corrections_made += 1

    # Construct the final list of S1 peaks after all corrections
    final_s1_peaks = [p for p in temp_s1_list if p not in peaks_to_remove]

    return np.array(sorted(final_s1_peaks)), corrected_debug_info, corrections_made

def calculate_windowed_hrv(s1_peaks: np.ndarray, sample_rate: int, params: Dict) -> pd.DataFrame:
    """ Calculates HRV metrics using R-R intervals based on changing heart rate """
    window_size_beats = params['hrv_window_size_beats']
    step_size_beats = params['hrv_step_size_beats']

    # First, calculate all R-R intervals from the S1 peaks
    if len(s1_peaks) < window_size_beats:
        logging.warning(f"Not enough beats ({len(s1_peaks)}) to perform windowed HRV analysis with a window of {window_size_beats} beats.")
        return pd.DataFrame(columns=['time', 'rmssdc', 'sdnn', 'bpm'])

    rr_intervals_sec = np.diff(s1_peaks) / sample_rate
    s1_times_sec = s1_peaks / sample_rate

    results = []
    # Iterate through the R-R intervals with a sliding window
    for i in range(0, len(rr_intervals_sec) - window_size_beats + 1, step_size_beats):
        window_rr_sec = rr_intervals_sec[i : i + window_size_beats]
        window_rr_ms = window_rr_sec * 1000
        start_time = s1_times_sec[i]
        end_time = s1_times_sec[i + window_size_beats]
        window_mid_time = (start_time + end_time) / 2.0

        # --- Calculate HRV Metrics for the Window ---
        mean_rr_ms = np.mean(window_rr_ms)
        sdnn = np.std(window_rr_ms)
        successive_diffs_ms = np.diff(window_rr_ms)
        rmssd = np.sqrt(np.mean(successive_diffs_ms**2))

        # --- Calculate Corrected RMSSD (RMSSDc) ---
        mean_rr_sec = mean_rr_ms / 1000.0
        rmssdc = rmssd / mean_rr_sec if mean_rr_sec > 0 else 0

        # Calculate the average BPM within this specific window
        window_bpm = 60 / mean_rr_sec if mean_rr_sec > 0 else 0

        results.append({
            'time': window_mid_time,
            'rmssdc': rmssdc,
            'sdnn': sdnn,
            'bpm': window_bpm
        })

    if not results:
        logging.warning("Could not perform windowed HRV analysis. Recording may be too short or have too few beats.")
        return pd.DataFrame(columns=['time', 'rmssdc', 'sdnn', 'bpm'])

    logging.info(f"Beat-based windowed HRV analysis complete. Generated {len(results)} data points.")
    return pd.DataFrame(results)

def calculate_bpm_series(peaks: np.ndarray, sample_rate: int, params: Dict) -> Tuple[pd.Series, np.ndarray]:
    """Calculates and smooths the final BPM series from S1 peaks."""
    if len(peaks) < 2: return pd.Series(dtype=np.float64), np.array([])
    peak_times = peaks / sample_rate
    time_diffs = np.diff(peak_times)
    valid_diffs = time_diffs > 1e-6
    if not np.any(valid_diffs): return pd.Series(dtype=np.float64), np.array([])

    instant_bpm = 60.0 / time_diffs[valid_diffs]
    start_time = datetime.datetime.fromtimestamp(0)
    valid_peak_times_dt = [start_time + datetime.timedelta(seconds=t) for t in peak_times[1:][valid_diffs]]
    bpm_series = pd.Series(instant_bpm, index=valid_peak_times_dt)
    avg_heart_rate = np.median(instant_bpm)
    if avg_heart_rate > 0:
        smoothing_window_sec = params['output_smoothing_window_sec']
        smoothing_window_str = f"{smoothing_window_sec}s"
        smoothed_bpm = bpm_series.rolling(window=smoothing_window_str, min_periods=1, center=True).mean()
    else:
        smoothed_bpm = pd.Series(dtype=np.float64)

    # Return the original numpy time points for compatibility with older functions that need it
    return smoothed_bpm, peak_times[1:][valid_diffs]

def find_major_hr_inclines(smoothed_bpm_series: pd.Series, min_duration_sec: int = 10, min_bpm_increase: int = 15) -> List[Dict]:
    """Identifies significant, sustained periods of heart rate increase."""
    if smoothed_bpm_series.empty or len( smoothed_bpm_series) < 2:
        return []

    logging.info(f"Searching for major HR inclines (min_duration={min_duration_sec}s, min_increase={min_bpm_increase} BPM)...")
    time_diffs_sec = smoothed_bpm_series.index.to_series().diff().dt.total_seconds()
    mean_time_diff = np.nanmean(time_diffs_sec)
    distance_samples = 5 if np.isnan(mean_time_diff) or mean_time_diff == 0 else int((min_duration_sec / 2) / mean_time_diff)

    peaks, _ = find_peaks(smoothed_bpm_series.values, prominence=5, distance=distance_samples)
    troughs, _ = find_peaks(-smoothed_bpm_series.values, prominence=5, distance=distance_samples)
    logging.info(f"Found {len(troughs)} potential start points (troughs) and {len(peaks)} potential end points (peaks) for inclines.")
    if len(troughs) == 0 or len(peaks) == 0:
        return []

    major_inclines = []
    for trough_idx in troughs:
        following_peaks_indices = peaks[peaks > trough_idx]
        if len(following_peaks_indices) > 0:
            peak_idx = following_peaks_indices[0]
            start_time, end_time = smoothed_bpm_series.index[trough_idx], smoothed_bpm_series.index[peak_idx]
            start_bpm, end_bpm = smoothed_bpm_series.values[trough_idx], smoothed_bpm_series.values[peak_idx]
            duration, bpm_increase = (end_time - start_time).total_seconds(), end_bpm - start_bpm

            if duration >= min_duration_sec and bpm_increase >= min_bpm_increase:
                major_inclines.append({
                    'start_time': start_time, 'end_time': end_time, 'start_bpm': start_bpm, 'end_bpm': end_bpm,
                    'duration_sec': duration, 'bpm_increase': bpm_increase, 'slope_bpm_per_sec': bpm_increase / duration
                })
    major_inclines.sort(key=lambda x: x['slope_bpm_per_sec'], reverse=True)
    return major_inclines

def find_major_hr_declines(smoothed_bpm_series: pd.Series, min_duration_sec: int = 10, min_bpm_decrease: int = 15) -> List[Dict]:
    """Identifies significant, sustained periods of heart rate decrease (recovery)."""
    if smoothed_bpm_series.empty or len(smoothed_bpm_series) < 2:
        return []

    logging.info(f"Searching for major HR declines (min_duration={min_duration_sec}s, min_decrease={min_bpm_decrease} BPM)...")
    time_diffs_sec = smoothed_bpm_series.index.to_series().diff().dt.total_seconds()
    mean_time_diff = np.nanmean(time_diffs_sec)
    distance_samples = 5 if np.isnan(mean_time_diff) or mean_time_diff == 0 else int((min_duration_sec / 2) / mean_time_diff)

    peaks, _ = find_peaks(smoothed_bpm_series.values, prominence=5, distance=distance_samples)
    troughs, _ = find_peaks(-smoothed_bpm_series.values, prominence=5, distance=distance_samples)
    logging.info(f"Found {len(peaks)} potential start points (peaks) and {len(troughs)} potential end points (troughs) for declines.")
    if len(troughs) == 0 or len(peaks) == 0:
        return []

    major_declines = []
    for peak_idx in peaks:
        following_troughs_indices = troughs[troughs > peak_idx]
        if len(following_troughs_indices) > 0:
            trough_idx = following_troughs_indices[0]
            start_time, end_time = smoothed_bpm_series.index[peak_idx], smoothed_bpm_series.index[trough_idx]
            start_bpm, end_bpm = smoothed_bpm_series.values[peak_idx], smoothed_bpm_series.values[trough_idx]
            duration, bpm_decrease = (end_time - start_time).total_seconds(), start_bpm - end_bpm

            if duration >= min_duration_sec and bpm_decrease >= min_bpm_decrease:
                major_declines.append({
                    'start_time': start_time, 'end_time': end_time, 'start_bpm': start_bpm, 'end_bpm': end_bpm,
                    'duration_sec': duration, 'bpm_decrease': bpm_decrease, 'slope_bpm_per_sec': (end_bpm - start_bpm) / duration
                })
    major_declines.sort(key=lambda x: x['slope_bpm_per_sec'])
    return major_declines

def find_peak_recovery_rate(smoothed_bpm_series: pd.Series, window_sec: int = 20) -> Optional[Dict]:
    """Finds the steepest slope of heart rate decline after the peak BPM."""
    if smoothed_bpm_series.empty or len(smoothed_bpm_series) < 2: return None
    recovery_series = smoothed_bpm_series[smoothed_bpm_series.idxmax():]
    if recovery_series.empty: return None

    times_sec = (recovery_series.index - recovery_series.index[0]).total_seconds()
    if times_sec[-1] < window_sec: return None

    bpm_values, steepest_slope, best_period = recovery_series.values, 0, None
    for i in range(len(times_sec) - 1):
        end_idx_candidates = np.where(times_sec >= times_sec[i] + window_sec)[0]
        if len(end_idx_candidates) == 0: break
        end_idx = end_idx_candidates[0]
        duration = times_sec[end_idx] - times_sec[i]
        if duration > 0:
            slope = (bpm_values[end_idx] - bpm_values[i]) / duration
            if slope < steepest_slope:
                steepest_slope = slope
                best_period = {'start_time': recovery_series.index[i], 'end_time': recovery_series.index[end_idx],
                               'start_bpm': bpm_values[i], 'end_bpm': bpm_values[end_idx],
                               'slope_bpm_per_sec': slope, 'duration_sec': duration}
    return best_period

def find_peak_exertion_rate(smoothed_bpm_series: pd.Series, window_sec: int = 20) -> Optional[Dict]:
    """Finds the steepest slope of heart rate increase across the entire recording."""
    if smoothed_bpm_series.empty or len(smoothed_bpm_series) < 2: return None
    times_sec = (smoothed_bpm_series.index - smoothed_bpm_series.index[0]).total_seconds()
    if times_sec[-1] < window_sec: return None

    bpm_values, steepest_slope, best_period = smoothed_bpm_series.values, 0, None
    for i in range(len(times_sec) - 1):
        end_idx_candidates = np.where(times_sec >= times_sec[i] + window_sec)[0]
        if len(end_idx_candidates) == 0: break
        end_idx = end_idx_candidates[0]
        duration = times_sec[end_idx] - times_sec[i]
        if duration > 0:
            slope = (bpm_values[end_idx] - bpm_values[i]) / duration
            if slope > steepest_slope:
                steepest_slope = slope
                best_period = {'start_time': smoothed_bpm_series.index[i], 'end_time': smoothed_bpm_series.index[end_idx],
                               'start_bpm': bpm_values[i], 'end_bpm': bpm_values[end_idx],
                               'slope_bpm_per_sec': slope, 'duration_sec': duration}
    return best_period

def calculate_hrr(smoothed_bpm_series: pd.Series, interval_sec: int = 60) -> Optional[Dict]:
    """Calculates the standard Heart Rate Recovery (HRR) over a fixed interval."""
    if smoothed_bpm_series.empty or len(smoothed_bpm_series) < 2: return None
    peak_bpm, peak_time = smoothed_bpm_series.max(), smoothed_bpm_series.idxmax()
    recovery_check_time = peak_time + pd.Timedelta(seconds=interval_sec)
    if recovery_check_time > smoothed_bpm_series.index.max(): return None

    recovery_bpm = np.interp(
        recovery_check_time.timestamp(),
        (smoothed_bpm_series.index.astype(np.int64) // 10**9).to_numpy(dtype=float),
        np.asarray(smoothed_bpm_series.values, dtype=float))
    return {'peak_bpm': peak_bpm, 'peak_time': peak_time, 'recovery_bpm': recovery_bpm,
            'recovery_check_time': recovery_check_time, 'hrr_value_bpm': peak_bpm - recovery_bpm,
            'interval_sec': interval_sec}

def find_recovery_phase(bpm_series: pd.Series, bpm_times_sec: np.ndarray, params: Dict) -> Tuple[Optional[float], Optional[float]]:
    """Analyzes a preliminary BPM series to find the peak heart rate and define the subsequent recovery phase window."""
    if bpm_times_sec is None or len(bpm_times_sec) < 2:
        logging.warning("Not enough preliminary beats to determine a recovery phase.")
        return None, None
    peak_time_sec = bpm_times_sec[np.argmax(bpm_series.to_numpy())]
    recovery_end_time_sec = peak_time_sec + params.get("recovery_phase_duration_sec", 120.0)
    logging.info(f"Peak BPM detected in preliminary pass at {peak_time_sec:.2f}s. High-contractility state defined until {recovery_end_time_sec:.2f}s.")
    return peak_time_sec, recovery_end_time_sec

# --- Main Analysis Pipeline (Orchestrator) ---
def _run_preliminary_pass(audio_envelope: np.ndarray, sample_rate: int, params: Dict,
                          noise_floor: pd.Series, troughs: np.ndarray,
                          start_bpm_hint: Optional[float]) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Runs a high-confidence first pass to estimate global BPM and find the recovery phase.
    """
    logging.info("--- STAGE 2: Running High-Confidence pass to find anchor beats ---")
    params_pass_1 = params.copy()
    # Use a higher threshold for a more confident initial beat detection
    params_pass_1["pairing_confidence_threshold"] = 0.75

    # Use the classifier for a high-confidence dry run
    classifier = PeakClassifier(audio_envelope, sample_rate, params_pass_1, start_bpm_hint,
                                noise_floor, troughs, None, None)
    anchor_beats, _, _ = classifier.classify_peaks()

    global_bpm_estimate = None
    if len(anchor_beats) >= 10:
        median_rr_sec = np.median(np.diff(anchor_beats) / sample_rate)
        if median_rr_sec > 0:
            global_bpm_estimate = 60.0 / median_rr_sec
            logging.info(f"Automatically determined Global BPM Estimate: {global_bpm_estimate:.1f} BPM")

    # Determine the starting BPM for the main analysis
    start_bpm = start_bpm_hint or global_bpm_estimate or 80.0

    prelim_bpm_series, prelim_bpm_times = calculate_bpm_series(anchor_beats, sample_rate, params)
    peak_bpm_time_sec, recovery_end_time_sec = find_recovery_phase(prelim_bpm_series, prelim_bpm_times, params)

    return start_bpm, peak_bpm_time_sec, recovery_end_time_sec


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
    max_iterations = 5  # Safeguard against infinite loops
    for i in range(max_iterations):
        logging.info(f"Correction Pass Iteration {i + 1}...")

        new_peaks, new_debug_info, corrections_made = _fix_rhythmic_discontinuities(
            s1_peaks=final_peaks,
            all_raw_peaks=all_raw_peaks,
            debug_info=current_debug_info,
            audio_envelope=audio_envelope,
            dynamic_noise_floor=dynamic_noise_floor,
            params=params,
            sample_rate=sample_rate
        )

        final_peaks = new_peaks # s1_peaks_rhythm_corrected
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


def _calculate_final_metrics(final_peaks: np.ndarray, sample_rate: int, params: Dict) -> Dict:
    """Calculates all final BPM, HRV, and slope metrics for reporting."""
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
        hrv_summary_stats['avg_bpm'] = metrics['smoothed_bpm'].mean()
        hrv_summary_stats['min_bpm'] = metrics['smoothed_bpm'].min()
        hrv_summary_stats['max_bpm'] = metrics['smoothed_bpm'].max()
    if not metrics['windowed_hrv_df'].empty:
        hrv_summary_stats['avg_rmssdc'] = metrics['windowed_hrv_df']['rmssdc'].mean()
        hrv_summary_stats['avg_sdnn'] = metrics['windowed_hrv_df']['sdnn'].mean()
    metrics['hrv_summary'] = hrv_summary_stats

    return metrics


def analyze_wav_file(wav_file_path: str, params: Dict, start_bpm_hint: Optional[float], original_file_path: str, output_directory: str):
    """Main analysis pipeline that orchestrates the refactored classes."""
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