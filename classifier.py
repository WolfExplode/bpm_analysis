import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import logging
from typing import List, Dict, Tuple, Optional, Any, Callable

from confidence_engine import (
    AnalysisState,
    PairingEngine,
    LookaheadSkipper,
    calculate_bpm_intervals,
    calculate_lone_s1_confidence,
    update_long_term_bpm,
    _append_s1_s2_interval,
    _append_s1_s2_contractility,
    _get_recent_s1_prominences_for_state,
    record_s1_outcome,
)
from peak_utils import (
    PeakType,
    _is_s1_paired_debug,
    _is_lone_s1_debug,
    _is_noise_debug,
    calculate_peak_prominence,
)


class PeakClassifier:
    """
    Encapsulates the logic for classifying raw audio peaks into S1, S2, and Noise.
    This class manages the state of the analysis loop, including BPM belief,
    beat candidates, and debug information.
    """
    def __init__(self, audio_envelope: np.ndarray, sample_rate: int, params: Dict,
                 start_bpm_hint: Optional[float], precomputed_noise_floor: pd.Series,
                 precomputed_troughs: np.ndarray, peak_bpm_time_sec: Optional[float],
                 recovery_end_time_sec: Optional[float],
                 band_envelopes: Optional[Dict[str, np.ndarray]] = None,
                 prelim_bpm_prior: Optional[Callable[[float], float]] = None):

        self.audio_envelope = audio_envelope
        self.sample_rate = sample_rate
        self.params = params
        self.peak_bpm_time_sec = peak_bpm_time_sec
        self.recovery_end_time_sec = recovery_end_time_sec

        # Helper components that encapsulate specific decision logic.
        self.pairing_engine = PairingEngine(
            audio_envelope, sample_rate, params, peak_bpm_time_sec, recovery_end_time_sec,
            band_envelopes=band_envelopes,
        )
        self.lookahead_skipper = LookaheadSkipper(
            audio_envelope, sample_rate, params, self.pairing_engine
        )

        self.state = self._initialize_state(
            start_bpm_hint, precomputed_noise_floor, precomputed_troughs, prelim_bpm_prior
        )

    def _initialize_state(self, start_bpm_hint, precomputed_noise_floor, precomputed_troughs, prelim_bpm_prior=None) -> AnalysisState:
        """Pre-calculates all necessary data and initializes the state for the peak finding loop."""
        analysis_data: Dict[str, Any] = {}
        dynamic_noise_floor, trough_indices = precomputed_noise_floor, precomputed_troughs
        all_peaks = self._find_raw_peaks(dynamic_noise_floor.values)
        all_peaks = self._refine_peaks_by_center_of_mass(all_peaks)

        analysis_data["dynamic_noise_floor_series"] = dynamic_noise_floor
        analysis_data["trough_indices"] = trough_indices

        noise_floor_at_peaks = dynamic_noise_floor.reindex(all_peaks, method='nearest').values
        peak_strengths = self.audio_envelope[all_peaks] - noise_floor_at_peaks
        peak_strengths[peak_strengths < 0] = 0
        normalized_deviations = np.abs(np.diff(peak_strengths)) / (
            np.maximum(peak_strengths[:-1], peak_strengths[1:]) + 1e-9
        )
        deviation_times = (all_peaks[:-1] + all_peaks[1:]) / 2 / self.sample_rate
        deviation_series = pd.Series(normalized_deviations, index=deviation_times)
        smoothing_window = max(5, int(len(deviation_series) * self.params['deviation_smoothing_factor']))
        smoothed_dev_series = deviation_series.rolling(
            window=smoothing_window, min_periods=1, center=True
        ).mean()
        analysis_data["deviation_series"] = smoothed_dev_series

        long_term_bpm = float(start_bpm_hint) if start_bpm_hint else 80.0

        return AnalysisState(
            dynamic_noise_floor=dynamic_noise_floor,
            trough_indices=trough_indices,
            all_peaks=all_peaks,
            smoothed_dev_series=smoothed_dev_series,
            long_term_bpm=long_term_bpm,
            analysis_data=analysis_data,
            sorted_troughs=sorted(trough_indices),
            prelim_bpm_prior=prelim_bpm_prior,
        )

    def classify_peaks(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Main classification loop to iterate through all raw peaks."""
        if len(self.state.all_peaks) < 2:
            return self.state.all_peaks, self.state.all_peaks, {"beat_debug_info": {}}

        while self.state.loop_idx < len(self.state.all_peaks):
            current_peak_idx = self.state.all_peaks[self.state.loop_idx]
            current_time_sec = current_peak_idx / self.sample_rate
            # Set BPM from preliminary curve (time-varying prior) when available
            if self.state.prelim_bpm_prior is not None:
                self.state.long_term_bpm = float(self.state.prelim_bpm_prior(current_time_sec))

            # Calculate pairing ratio once per iteration so all consumers
            # (kick-start recovery, pairing engine, lookahead skipper) share
            # the same view of recent rhythm stability.
            pairing_ratio = self._calculate_pairing_ratio()

            kickstart_msg = self._kickstart_check(pairing_ratio)
            is_last_peak = self.state.loop_idx >= len(self.state.all_peaks) - 1

            if is_last_peak:
                self._handle_last_peak(current_peak_idx)
            else:
                self._process_peak_pair(current_peak_idx, pairing_ratio, kickstart_msg=kickstart_msg)

            self._append_bpm_history()

        return self._finalize_results()

    def _kickstart_check(self, pairing_ratio: float) -> Optional[str]:
        """
        Recovery function that fires when the algorithm is stuck in Lone-S1-only mode.
        Detected by: low pairing ratio + recent Lone S1 beats each followed by a Noise peak.
        When triggered, overrides the pairing ratio to encourage pairing on the next peak.

        Returns a human-readable message if kick-start fired, otherwise None.
        """
        if pairing_ratio >= self.params.get("kickstart_check_threshold", 0.3):
            return None

        history = self.params.get("kickstart_history_beats", 4)
        if len(self.state.candidate_beats) < history:
            return None

        min_s1s = self.params.get("kickstart_min_lone_s1s", 3)
        recent_lone_s1s = [
            idx for idx in self.state.candidate_beats[-history:]
            if _is_lone_s1_debug(self.state.beat_debug_info.get(idx))
        ]
        if len(recent_lone_s1s) < min_s1s:
            return None

        min_matches = self.params.get("kickstart_min_noise_matches", 3)
        matches = 0
        for s1_idx in recent_lone_s1s:
            current_raw_idx = np.searchsorted(self.state.all_peaks, s1_idx)
            if current_raw_idx < len(self.state.all_peaks) - 1:
                next_raw_peak_idx = self.state.all_peaks[current_raw_idx + 1]
                if _is_noise_debug(self.state.beat_debug_info.get(next_raw_peak_idx)):
                    matches += 1

        if matches >= min_matches:
            override_ratio = self.params.get("kickstart_override_ratio", 0.6)
            msg = (
                f"KICK-START: Found {matches}/{len(recent_lone_s1s)} S1→Noise patterns "
                f"(pairing ratio {pairing_ratio:.0%}). Overriding pairing ratio to {override_ratio:.0%} "
                f"to encourage pairing on this peak."
            )
            logging.info(msg)
            self.state.pairing_ratio_override = override_ratio
            return msg

        return None

    def _handle_last_peak(self, peak_idx: int):
        """Classify the final peak in the sequence."""
        self.state.candidate_beats.append(peak_idx)
        self.state.beat_debug_info[peak_idx] = {
            "peak_type": PeakType.LONE_S1_LAST.value,
            "sections": []
        }
        self.state.loop_idx += 1

    def _calculate_pairing_ratio(self) -> float:
        """Calculate recent rhythm stability ratio."""
        history_window = self.params.get("stability_history_window", 20)
        if len(self.state.candidate_beats) < history_window:
            return 0.5

        recent_beats = self.state.candidate_beats[-history_window:]
        paired_count = sum(
            1 for beat_idx in recent_beats
            if _is_s1_paired_debug(self.state.beat_debug_info.get(beat_idx))
        )
        return paired_count / history_window

    def _process_peak_pair(
        self,
        current_peak_idx: int,
        pairing_ratio: float,
        kickstart_msg: Optional[str] = None,
    ) -> None:
        """Processes a pair of peaks to determine if they are S1-S2."""
        all_peaks = self.state.all_peaks
        loop_idx = self.state.loop_idx

        # We always have at least one "next" peak here (caller guards last-peak case)
        next_peak_idx = all_peaks[loop_idx + 1]

        # --- LOOKAHEAD: optionally skip a weak middle peak between a strong S1 and S2 ---
        decision = self.lookahead_skipper.maybe_skip(self.state, loop_idx, pairing_ratio)
        if decision is not None:
            s1_idx = decision["s1_idx"]
            middle_idx = decision["middle_idx"]
            s2_idx = decision["s2_idx"]
            steps = decision["steps"]
            prominence_context = decision["prominence_context"]
            lookahead_msg = decision["lookahead_msg"]
            middle_noise_msg = decision["middle_noise_msg"]

            pair_sections = [
                {"type": "lookahead", "text": lookahead_msg},
                {"type": "confidence_trace", "steps": steps},
                {"type": "prominence", "details": prominence_context},
            ]
            if kickstart_msg:
                pair_sections.insert(0, {"type": "kickstart", "text": kickstart_msg})

            self.state.candidate_beats.append(s1_idx)
            _append_s1_s2_interval(self.state, (s2_idx - s1_idx) / self.sample_rate, self.params)
            _append_s1_s2_contractility(
                self.state, s1_idx, s2_idx, self.audio_envelope, self.state.trough_indices, self.sample_rate, self.params
            )
            self.state.beat_debug_info[s1_idx] = {
                "peak_type": PeakType.S1_PAIRED.value,
                "sections": pair_sections,
            }

            original_middle_debug = self.state.beat_debug_info.get(middle_idx)
            self.state.beat_debug_info[middle_idx] = {
                "peak_type": PeakType.NOISE.value,
                "sections": [
                    {"type": "lookahead", "text": middle_noise_msg},
                    {"type": "original", "original_debug": original_middle_debug},
                ],
            }

            self.state.beat_debug_info[s2_idx] = {
                "peak_type": PeakType.S2_PAIRED.value,
                "sections": pair_sections,
            }

            self.state.consecutive_rr_rejections = 0
            record_s1_outcome(self.state, s1_idx / self.sample_rate, True, self.params)
            # Skip the S1, middle noise, and S2 peaks
            self.state.loop_idx += 3
            return

        # --- Standard pairing attempt ---
        is_paired, steps, prominence_context = self.pairing_engine.attempt_pair(
            self.state, current_peak_idx, next_peak_idx, pairing_ratio
        )

        current_time_sec = current_peak_idx / self.sample_rate
        if is_paired:
            self.state.candidate_beats.append(current_peak_idx)
            _append_s1_s2_interval(self.state, (next_peak_idx - current_peak_idx) / self.sample_rate, self.params)
            _append_s1_s2_contractility(
                self.state, current_peak_idx, next_peak_idx, self.audio_envelope, self.state.trough_indices, self.sample_rate, self.params
            )
            sections: List[Dict[str, Any]] = [
                {"type": "confidence_trace", "steps": steps},
                {"type": "prominence", "details": prominence_context},
            ]
            if kickstart_msg:
                sections.insert(0, {"type": "kickstart", "text": kickstart_msg})
            self.state.beat_debug_info[current_peak_idx] = {
                "peak_type": PeakType.S1_PAIRED.value,
                "sections": sections,
            }
            self.state.beat_debug_info[next_peak_idx] = {
                "peak_type": PeakType.S2_PAIRED.value,
                "sections": sections,
            }
            self.state.consecutive_rr_rejections = 0
            record_s1_outcome(self.state, current_time_sec, True, self.params)
            self.state.loop_idx += 2
            return

        # --- Skip-one fallback: pair with peak after next (middle may be noise) ---
        if loop_idx + 2 < len(all_peaks):
            next_next_peak_idx = all_peaks[loop_idx + 2]
            is_paired_skip, steps_skip, prominence_context_skip = self.pairing_engine.attempt_pair(
                self.state, current_peak_idx, next_next_peak_idx, pairing_ratio
            )
            if is_paired_skip:
                middle_idx = next_peak_idx
                self.state.candidate_beats.append(current_peak_idx)
                _append_s1_s2_interval(
                    self.state, (next_next_peak_idx - current_peak_idx) / self.sample_rate, self.params
                )
                _append_s1_s2_contractility(
                    self.state, current_peak_idx, next_next_peak_idx, self.audio_envelope,
                    self.state.trough_indices, self.sample_rate, self.params,
                )
                skip_one_sections: List[Dict[str, Any]] = [
                    {"type": "skip_one", "text": "Paired S1 with peak after next (middle labeled as noise)."},
                    {"type": "confidence_trace", "steps": steps_skip},
                    {"type": "prominence", "details": prominence_context_skip},
                ]
                if kickstart_msg:
                    skip_one_sections.insert(0, {"type": "kickstart", "text": kickstart_msg})
                self.state.beat_debug_info[current_peak_idx] = {
                    "peak_type": PeakType.S1_PAIRED.value,
                    "sections": skip_one_sections,
                }
                self.state.beat_debug_info[middle_idx] = {
                    "peak_type": PeakType.NOISE.value,
                    "sections": [{"type": "skip_one", "text": "Skipped as noise (S2 was next peak)."}],
                }
                self.state.beat_debug_info[next_next_peak_idx] = {
                    "peak_type": PeakType.S2_PAIRED.value,
                    "sections": skip_one_sections,
                }
                self.state.consecutive_rr_rejections = 0
                record_s1_outcome(self.state, current_time_sec, True, self.params)
                self.state.loop_idx += 3
                return

        record_s1_outcome(self.state, current_time_sec, False, self.params)
        self._classify_lone_peak(current_peak_idx, steps, kickstart_msg=kickstart_msg)
        self.state.loop_idx += 1

    def _append_bpm_history(self):
        """When no preliminary prior: update long_term_bpm from last R-R (preliminary pass). Then append (time, bpm) for plotting."""
        if self.state.prelim_bpm_prior is None and len(self.state.candidate_beats) > 1:
            new_rr = (self.state.candidate_beats[-1] - self.state.candidate_beats[-2]) / self.sample_rate
            if new_rr > 0:
                self.state.long_term_bpm = update_long_term_bpm(
                    new_rr, self.state.long_term_bpm, self.params
                )
        if self.state.candidate_beats:
            time_sec = self.state.candidate_beats[-1] / self.sample_rate
            self.state.long_term_bpm_history.append((time_sec, self.state.long_term_bpm))

    def _finalize_results(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Finalizes and returns the analysis results."""
        final_peaks = np.array(sorted(list(dict.fromkeys(self.state.candidate_beats))))
        self.state.analysis_data["beat_debug_info"] = self.state.beat_debug_info
        if self.state.long_term_bpm_history:
            lt_bpm_times, lt_bpm_values = zip(*self.state.long_term_bpm_history)
            self.state.analysis_data["long_term_bpm_series"] = pd.Series(lt_bpm_values, index=lt_bpm_times)
        return final_peaks, self.state.all_peaks, self.state.analysis_data

    def _find_raw_peaks(self, height_threshold: np.ndarray) -> np.ndarray:
        """Finds all potential peaks above the given height threshold."""
        prominence_thresh = np.quantile(self.audio_envelope, self.params['peak_prominence_quantile'])
        min_peak_dist_samples = int(self.params['min_peak_distance_sec'] * self.sample_rate)
        peaks, _ = find_peaks(self.audio_envelope, height=height_threshold, prominence=prominence_thresh, distance=min_peak_dist_samples)
        logging.info(f"Found {len(peaks)} raw peaks using dynamic height threshold.")
        return peaks

    def _refine_peaks_by_center_of_mass(self, peaks: np.ndarray) -> np.ndarray:
        """
        Refines each peak to the super-Gaussian-weighted center-of-mass of the envelope
        in a window around it. If two refined positions are within 50 ms, keeps the one
        with higher weighted mass (drops the smaller / noisier one).
        """
        if len(peaks) == 0:
            return peaks
        envelope = self.audio_envelope
        n_samples = len(envelope)
        window_ms = float(self.params.get("peak_refine_window_ms", 100))
        max_shift_ms = float(self.params.get("peak_refine_max_shift_ms", 10))
        n_exp = float(self.params.get("peak_refine_super_gaussian_n", 4))
        window_samples = int(window_ms * self.sample_rate / 1000)
        half = max(1, window_samples // 2)
        max_shift_samples = max(0, int(max_shift_ms * self.sample_rate / 1000))
        # Sigma so that flat top spans center; standard choice ~ window/4
        sigma = max(1.0, window_samples / 4.0)
        min_dist_samples = max(1, int(0.05 * self.sample_rate))  # 50 ms

        refined_with_mass: List[Tuple[int, float, int]] = []
        for peak_idx in peaks:
            left = max(0, peak_idx - half)
            right = min(n_samples - 1, peak_idx + half)
            t = np.arange(left, right + 1, dtype=np.float64)
            dist = np.abs(t - peak_idx)
            weights = np.exp(-((dist / sigma) ** n_exp))
            weighted_env = weights * envelope[left : right + 1]
            mass = float(np.sum(weighted_env))
            if mass <= 0:
                refined_idx = int(peak_idx)
            else:
                com = float(np.sum(t * weighted_env) / mass)
                com_clipped = np.clip(
                    com,
                    peak_idx - max_shift_samples,
                    peak_idx + max_shift_samples,
                )
                refined_idx = int(round(com_clipped))
                refined_idx = max(0, min(n_samples - 1, refined_idx))
            refined_with_mass.append((refined_idx, mass, int(peak_idx)))

        refined_with_mass.sort(key=lambda x: x[0])
        result: List[Tuple[int, float, int]] = []
        for r, m, o in refined_with_mass:
            if not result:
                result.append((r, m, o))
                continue
            prev_r, prev_m, _ = result[-1]
            if (r - prev_r) < min_dist_samples:
                if m > prev_m:
                    result[-1] = (r, m, o)
            else:
                result.append((r, m, o))

        out = np.array([r for r, _, _ in result], dtype=peaks.dtype)
        if len(out) < len(peaks):
            logging.info(
                "Peak refinement: %d -> %d peaks after CoM shift and 50 ms merge.",
                len(peaks),
                len(out),
            )
        return np.sort(out)

    def _classify_lone_peak(
        self,
        peak_idx: int,
        pairing_failure_steps: List[Dict[str, Any]],
        kickstart_msg: Optional[str] = None,
    ):
        """Validates if an unpaired peak is a Lone S1 or Noise."""
        is_valid, lone_s1_lines = self._validate_lone_s1(peak_idx)

        def _build_sections(validated: bool) -> List[Dict[str, Any]]:
            secs: List[Dict[str, Any]] = [
                {"type": "confidence_trace", "steps": pairing_failure_steps},
                {"type": "lone_s1", "lines": lone_s1_lines, "validated": validated},
            ]
            if kickstart_msg:
                secs.insert(0, {"type": "kickstart", "text": kickstart_msg})
            return secs

        if is_valid:
            self.state.candidate_beats.append(peak_idx)
            self.state.beat_debug_info[peak_idx] = {
                "peak_type": PeakType.LONE_S1_VALIDATED.value,
                "sections": _build_sections(validated=True),
            }
            self.state.consecutive_rr_rejections = 0
        else:
            is_rhythm_rejection = any("Rhythm Fit" in ln for ln in lone_s1_lines)
            if is_rhythm_rejection:
                self.state.consecutive_rr_rejections += 1
            else:
                self.state.consecutive_rr_rejections = 0

            if self.state.consecutive_rr_rejections >= self.params.get("cascade_reset_trigger_count", 3):
                logging.info(
                    f"CASCADE RESET: Forcing peak at {peak_idx / self.sample_rate:.2f}s as Lone S1 due to repeated rhythmic failures.")
                self.state.candidate_beats.append(peak_idx)
                self.state.beat_debug_info[peak_idx] = {
                    "peak_type": PeakType.LONE_S1_CASCADE.value,
                    "sections": _build_sections(validated=False),
                }
                self.state.consecutive_rr_rejections = 0
            else:
                self.state.beat_debug_info[peak_idx] = {
                    "peak_type": PeakType.NOISE.value,
                    "sections": _build_sections(validated=False),
                }

    def _validate_lone_s1(self, current_peak_idx: int) -> Tuple[bool, List[str]]:
        """Performs checks to determine if a peak is a valid Lone S1."""
        detail_lines = []

        # --- 1. Basic rhythm & amplitude calculation ---
        if not self.state.candidate_beats:
            return True, ["Outcome: First beat (no prior rhythm to compare) → Validated Lone S1 → --"]

        confidence, detail_lines = calculate_lone_s1_confidence(
            current_peak_idx, self.state.candidate_beats[-1], self.state.long_term_bpm,
            self.audio_envelope, self.state.dynamic_noise_floor, self.sample_rate, self.params,
            all_peaks=self.state.all_peaks
        )

        # --- 2. Absolute prominence guardrail ---
        # Track only high-quality S1s (avoid contaminating reference with noise)
        recent_s1_types = [self.state.beat_debug_info.get(idx, {}).get("peak_type")
                        for idx in self.state.candidate_beats[-20:]]  # Last 20 beats
        recent_prominences = [
            calculate_peak_prominence(idx, self.audio_envelope, self.state.trough_indices)
            for idx, typ in zip(self.state.candidate_beats[-20:], recent_s1_types)
            if typ in (PeakType.S1_PAIRED.value, PeakType.LONE_S1_VALIDATED.value)
        ]

        if len(recent_prominences) >= 5:  # Need minimum history
            # Top 20% quartile as reference (robust to outliers)
            reference_prominence = np.percentile(recent_prominences, 80)
            current_prominence = calculate_peak_prominence(
                current_peak_idx, self.audio_envelope, self.state.trough_indices
            )

            # Penalty if <40% of reference S1 prominence (adaptive threshold)
            min_ratio = self.params.get('lone_s1_min_prominence_ratio', 0.4)
            prominence_ratio = current_prominence / (reference_prominence + 1e-9)

            if prominence_ratio < min_ratio:
                # Linear penalty: 0.5x → 0.5 penalty, 0.2x → 0.2 penalty, etc.
                penalty_factor = np.clip(prominence_ratio / min_ratio, 0.0, 1.0)
                confidence *= penalty_factor
                detail_lines.append(
                    f"Absolute Prominence: {current_prominence:.3f} < {min_ratio:.1f}× reference "
                    f"({reference_prominence:.3f}) → ×{penalty_factor:.2f} → {confidence:.2f}"
                )

        # --- 3. Forward check ---
        current_peak_all_peaks_idx = np.searchsorted(self.state.all_peaks, current_peak_idx)
        if current_peak_all_peaks_idx < len(self.state.all_peaks) - 1:
            next_raw_peak_idx = self.state.all_peaks[current_peak_all_peaks_idx + 1]
            forward_interval_sec = (next_raw_peak_idx - current_peak_idx) / self.sample_rate
            expected_rr_sec = calculate_bpm_intervals(self.state.long_term_bpm, self.params)["rr_interval"]

            if forward_interval_sec < expected_rr_sec * 0.45:  # Too close
                current_amp = self.audio_envelope[current_peak_idx]
                next_amp = self.audio_envelope[next_raw_peak_idx]

                # If not MUCH stronger, it's likely S2, not S1
                if current_amp < next_amp * 1.69: # 1.69 is a random number I tuned, a better implementation would avoid the need for this magic number
                    detail_lines.append(
                        f"Forward check: next peak too close ({forward_interval_sec:.3f}s) and not strong enough "
                        f"→ veto → 0.00"
                    )
                    confidence = 0.0  # Hard veto

        # --- 4. Final threshold check ---
        threshold = self.params.get("lone_s1_confidence_threshold", 0.6)
        if confidence < threshold:
            detail_lines.append(
                f"Outcome: score {confidence:.2f} < threshold {threshold:.2f} → Rejected Lone S1 → {confidence:.2f}"
            )
            return False, detail_lines
        detail_lines.append(
            f"Outcome: score {confidence:.2f} >= threshold {threshold:.2f} → Validated Lone S1 → {confidence:.2f}"
        )
        return True, detail_lines
