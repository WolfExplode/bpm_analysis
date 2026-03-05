import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import datetime
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import csv

from audio_io import load_audio_mono, preprocess_audio

# INSTRUCTIONS FOR AI: 
# Do not remove any debugging code unless specified by the user
# try to avoid further abstracting my code
# try to avoid further segmenting my code
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


def _get_peak_type_from_debug(entry) -> str:
    """
    Helper to safely extract the peak_type string from a debug entry.
    Supports both the new dict-based structure and legacy string values.
    """
    if isinstance(entry, dict):
        return entry.get("peak_type", "")
    if isinstance(entry, str):
        # Legacy packing used '§' as a separator: '<PeakType>§TAG§VALUE...'
        return entry.split('§', 1)[0] if entry else ""
    return ""


def _is_s1_paired_debug(entry) -> bool:
    """Returns True if a debug entry represents a paired S1."""
    return _get_peak_type_from_debug(entry) == PeakType.S1_PAIRED.value


def _is_lone_s1_debug(entry) -> bool:
    """Returns True if a debug entry represents any Lone S1 classification."""
    pt = _get_peak_type_from_debug(entry)
    return pt.startswith("Lone S1")


def _is_noise_debug(entry) -> bool:
    """Returns True if a debug entry represents a Noise/Rejected classification."""
    pt = _get_peak_type_from_debug(entry)
    return "Noise" in pt


def format_debug_entry(debug_entry: Dict) -> List[str]:
    """
    Converts a structured debug entry into a list of human-readable lines.

    Shared formatter used by plotting tooltips and debug logs.
    """
    if not isinstance(debug_entry, dict):
        if not debug_entry:
            return []
        text = str(debug_entry)
        return ["- Details:", f"    - {text}"]

    lines: List[str] = []
    sections = debug_entry.get("sections", [])

    for sec in sections:
        sec_type = sec.get("type")

        if sec_type == "pairing":
            lines.append("- S1-S2 pairing decision:")
            raw_lines = sec.get("lines")
            if raw_lines is None:
                text = sec.get("text", "")
                raw_lines = [t.strip() for t in text.split("\n") if t.strip()]
            for ln in raw_lines:
                lines.append(f"    - {ln}")

        elif sec_type == "lone_s1":
            lines.append("- Lone S1 decision:")
            raw_lines = sec.get("lines") or []
            for ln in raw_lines:
                lines.append(f"    - {ln}")

        elif sec_type == "lookahead":
            msg = sec.get("text") or sec.get("message")
            if msg:
                lines.append(f"- {msg}")

        elif sec_type == "original":
            original = sec.get("original_debug")
            if isinstance(original, dict):
                orig_type = original.get("peak_type", "Unknown")
                lines.append(f"- Original Classification: {orig_type}")
            elif original:
                lines.append("- Original Classification:")
                lines.append(f"    - {original}")

        elif sec_type == "prominence":
            details = sec.get("details", {})
            lines.append("- Prominence context:")

            def _format_trough(label: str, idx, amp, time):
                if idx is None or amp is None:
                    return f"    - {label}: None"
                time_str = f"{time:.3f}s" if time is not None else "unknown time"
                return f"    - {label}: idx={idx} ({time_str}), amp={amp:.3f}"

            for peak_label in ("s1", "s2"):
                peak_data = details.get(peak_label, {})
                if not peak_data:
                    continue
                prom = peak_data.get("prominence")
                peak_time = peak_data.get("peak_time")
                peak_amp = peak_data.get("peak_amp")
                key_col = peak_data.get("key_col_amp")
                time_str = f"{peak_time:.3f}s" if peak_time is not None else "unknown time"
                amp_str = f"{peak_amp:.3f}" if peak_amp is not None else "unknown amp"
                prom_str = f"{prom:.3f}" if prom is not None else "unknown"
                key_col_str = f"{key_col:.3f}" if key_col is not None else "unknown"
                lines.append(
                    f"    - {peak_label.upper()}: prom {prom_str}, peak @ {time_str} (amp {amp_str}), key col {key_col_str}"
                )
                lines.append(
                    _format_trough(
                        "      Left trough", peak_data.get("left_trough_idx"), peak_data.get("left_trough_amp"), peak_data.get("left_trough_time")
                    )
                )
                lines.append(
                    _format_trough(
                        "      Right trough", peak_data.get("right_trough_idx"), peak_data.get("right_trough_amp"), peak_data.get("right_trough_time")
                    )
                )

    return lines


def _simple_label_from_debug(entry: Any) -> str:
    """
    Maps a detailed debug entry to a coarse label used for validation.

    Returns one of: 'S1', 'S2', 'Noise', or 'Unknown'.
    """
    pt = _get_peak_type_from_debug(entry) or ""
    if not pt:
        return "Unknown"

    # Re-use the existing helpers so we stay consistent with plotting/reporting.
    if _is_noise_debug(entry):
        return "Noise"
    if _is_s1_paired_debug(entry) or _is_lone_s1_debug(entry):
        return "S1"
    if PeakType.is_s2(pt):
        return "S2"
    return "Unknown"

# Import helpers for plotting/reporting after they are defined to avoid circular imports
from plotting import Plotter
from reporting import ReportGenerator


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

@dataclass
class AnalysisState:
    """
    Holds all mutable state used by the peak classification loop.

    Attributes:
        dynamic_noise_floor: Rolling estimate of the background noise floor at each sample,
            derived from sanitized audio troughs. Used to threshold peaks and compute
            peak “strength” relative to the local noise environment.
        trough_indices: Indices of sanitized troughs in the audio envelope. These anchor
            prominence calculations by defining the key cols around each peak.
        all_peaks: Indices of all raw peaks above the dynamic height threshold before
            any S1/S2/Noise classification or correction passes.
        smoothed_dev_series: Time‑indexed series of normalized peak‑to‑peak amplitude
            deviations, smoothed over time. This captures rhythm stability and is used
            as context when reasoning about sudden changes in the waveform.
        long_term_bpm: Slowly adapting belief about the underlying heart rate, updated
            by `update_long_term_bpm()`. This is the BPM value the algorithm trusts when
            computing expected S1‑S2 and R‑R intervals.
        analysis_data: Bag of analysis artifacts that downstream plotting/reporting
            relies on (e.g., `dynamic_noise_floor_series`, `trough_indices`,
            `deviation_series`, `beat_debug_info`, `long_term_bpm_series`).
        candidate_beats: Sample indices of peaks that have been accepted as S1 heartbeats
            (either paired S1 or validated Lone S1) during the main loop.
        beat_debug_info: Mapping from raw peak index to a structured debug record
            explaining how that peak was classified. This powers the debug log and
            interactive plot tooltips.
        long_term_bpm_history: Sequence of `(time_sec, bpm)` tuples capturing how the
            long‑term BPM belief evolved over the recording.
        sorted_troughs: Sorted list of trough indices mirroring `trough_indices`,
            kept in list form for fast neighbor lookups and iteration.
        consecutive_rr_rejections: Count of consecutive Lone S1 rhythm rejections, used
            to trigger the “cascade reset” safety mechanism when the rhythm model fails.
        loop_idx: Current index into `all_peaks` for the main classification loop. This
            is the loop counter that drives progression through raw peaks.
        pairing_ratio_override: Optional override for the recent pairing stability ratio
            set by the kick‑start recovery logic when the algorithm gets stuck in
            “Lone S1 only” mode.
    """

    dynamic_noise_floor: pd.Series
    trough_indices: np.ndarray
    all_peaks: np.ndarray
    smoothed_dev_series: pd.Series
    long_term_bpm: float
    analysis_data: Dict[str, Any] = field(default_factory=dict)
    candidate_beats: List[int] = field(default_factory=list)
    beat_debug_info: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    long_term_bpm_history: List[Tuple[float, float]] = field(default_factory=list)
    sorted_troughs: List[int] = field(default_factory=list)
    consecutive_rr_rejections: int = 0
    loop_idx: int = 0
    pairing_ratio_override: Optional[float] = None


class PairingEngine:
    """
    Scores candidate S1–S2 pairs and returns a pairing decision plus debug context.

    This class is intentionally stateless (mostly stateless) with respect to the main analysis loop:
    it never mutates `AnalysisState` and instead relies on the caller (`PeakClassifier`)
    to own all state updates. This keeps the confidence model self‑contained and
    easier to reason about in isolation.
    """

    def __init__(
        self,
        audio_envelope: np.ndarray,
        sample_rate: int,
        params: Dict,
        peak_bpm_time_sec: Optional[float],
        recovery_end_time_sec: Optional[float],
    ) -> None:
        self.audio_envelope = audio_envelope
        self.sample_rate = sample_rate
        self.params = params
        self.peak_bpm_time_sec = peak_bpm_time_sec
        self.recovery_end_time_sec = recovery_end_time_sec

    def attempt_pair(
        self,
        state: AnalysisState,
        s1_candidate_idx: int,
        s2_candidate_idx: int,
        pairing_ratio: float,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Calculates the confidence score for pairing two candidate peaks.

        Returns:
            (is_paired, reason, prominence_context)
        where:
            - is_paired: True if the final confidence exceeds the configured threshold
            - reason: multi‑line human‑readable explanation of the decision
            - prominence_context: raw prominence details for S1/S2 used by debug tools
        """
        interval_sec = (s2_candidate_idx - s1_candidate_idx) / self.sample_rate
        bpm = state.long_term_bpm

        # Minimum S1-S2 as fraction of RR interval (adapts to current BPM)
        intervals = calculate_bpm_intervals(bpm, self.params)
        min_s1_s2_interval = intervals["s1_s2_min"]

        if interval_sec < min_s1_s2_interval:
            # Calculate implied BPM for debugging
            # Assuming S2-S1 is at least as long as S1-S2 (conservative estimate)
            implied_total_cycle = interval_sec * 2.0  # S1-S2 + S2-S1
            implied_bpm = 60.0 / implied_total_cycle if implied_total_cycle > 0 else float('inf')

            debug_msg = (
                f"Impossible: S1-S2 interval {interval_sec:.3f}s < min {min_s1_s2_interval:.3f}s "
                f"(implies {implied_bpm:.0f} BPM vs assumed {bpm:.0f} BPM)"
            )
            return False, debug_msg, {}

        # --- Base confidence: neutral starting point (contractility handled by prominence adjustment) ---
        base_confidence = 0.60
        reason = f"Base Conf: {base_confidence:.2f}"

        # --- Contractility / prominence-based adjustment (S1 vs S2) ---
        s1_details = get_peak_prominence_details(
            s1_candidate_idx,
            self.audio_envelope,
            state.trough_indices,
            sample_rate=self.sample_rate,
        )
        s2_details = get_peak_prominence_details(
            s2_candidate_idx,
            self.audio_envelope,
            state.trough_indices,
            sample_rate=self.sample_rate,
        )
        s1_prominence = s1_details["prominence"]
        s2_prominence = s2_details["prominence"]

        shared_s1_right_s2_left = (
            s1_details.get("right_trough_idx") is not None
            and s2_details.get("left_trough_idx") is not None
            and s1_details["right_trough_idx"] == s2_details["left_trough_idx"]
        )
        prominence_context = {
            "s1": s1_details,
            "s2": s2_details,
            "shared_s1_right_s2_left": shared_s1_right_s2_left,
        }

        # --- Contractility model based on S1/S2 prominence ratio ---
        confidence, contractility_reason = adjust_confidence_with_contractility(
            base_confidence,
            s1_prominence,
            s2_prominence,
            bpm,
            self.params,
        )
        reason += contractility_reason

        # --- Absolute S1 prominence guardrail (shared with Lone S1 logic) ---
        # Protect against tiny noise bumps being interpreted as heartbeats, a "high contractility" S1/S2 pair.
        # We compare the current S1 prominence against a recent high-quality S1 baseline.
        recent_prominences = _get_recent_s1_prominences_for_state(
            state, self.audio_envelope, state.trough_indices
        )
        if len(recent_prominences) >= 5:
            reference_prominence = np.percentile(recent_prominences, 80)  # Top 20% as adaptive reference
            if reference_prominence > 0:
                # Re‑use Lone S1 ratio setting for now to keep behavior consistent
                min_ratio = self.params.get(
                    "paired_s1_min_prominence_ratio",
                    self.params.get("lone_s1_min_prominence_ratio", 0.4),  # Tuned magic number, see docs.
                )
                prominence_ratio = s1_prominence / (reference_prominence + 1e-9)

                if prominence_ratio < min_ratio:
                    # Linear penalty, mirroring Lone S1 behavior:
                    #   at ratio == min_ratio -> no penalty
                    #   at ratio  == 0       -> full veto
                    penalty_factor = float(np.clip(prominence_ratio / (min_ratio + 1e-9), 0.0, 1.0))
                    confidence *= penalty_factor
                    reason += (
                        f"\n- Absolute S1 Prominence Penalty: {s1_prominence:.3f} < {min_ratio:.1f}× reference "
                        f"({reference_prominence:.3f}) → confidence ×{penalty_factor:.2f}"
                    )

        # --- Other adjustments (stability, ratio history, etc.) ---
        confidence, other_reason = _apply_other_pairing_adjustments(
            confidence,
            s1_candidate_idx,
            s2_candidate_idx,
            self.audio_envelope,
            state.dynamic_noise_floor,
            bpm,
            pairing_ratio,
            self.params,
            self.sample_rate,
            self.peak_bpm_time_sec,
            self.recovery_end_time_sec,
            len(state.candidate_beats),
        )
        reason += other_reason

        s1_s2_max_interval = intervals["s1_s2_max"]

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
                interval_reason = (
                    f"\n- Interval penalty by {penalty_amount:.2f} "
                    f"(Interval {interval_sec:.3f}s > Max {s1_s2_max_interval:.3f}s)"
                )
            else:
                interval_reason = ""
        else:
            interval_reason = ""
        reason += interval_reason

        # --- Forward-Looking Contextual Penalty ---
        # If pairing S1-S2 causes the next S2→S1 transition to be implausible, penalize it.
        # Guardrail: only trust this check if the "next S1" peak is strong enough to be
        # a plausible beat; otherwise it may just be noise and should not veto the pair.
        forward_look_penalty = 0.0
        if state.loop_idx + 2 < len(state.all_peaks):
            next_next_peak_idx = state.all_peaks[state.loop_idx + 2]

            # Use prominence for robust amplitude comparison
            next_s1_details = get_peak_prominence_details(
                next_next_peak_idx,
                self.audio_envelope,
                state.trough_indices,
                sample_rate=self.sample_rate,
            )
            next_s1_prominence = next_s1_details["prominence"]

            # Only apply if we have valid data
            if s2_prominence > 1e-9 and next_s1_prominence > 1e-9:
                noise_thresh = self.params.get('noise_prominence_threshold', 0.35)

                # Guardrail: Skip penalty if the "next S1" is too weak to be a credible beat.
                # This prevents noise from vetoing a valid S1-S2 pair.
                if next_s1_prominence < s2_prominence * noise_thresh:
                    # Next peak is likely noise; skip forward-look penalty entirely.
                    pass
                else:
                    # Next peak is strong enough to evaluate; apply penalty if needed.
                    drop_ratio = next_s1_prominence / (s2_prominence + 1e-9)

                    # If the following peak is substantially weaker, it suggests s2_candidate
                    # is actually a strong S1, making this S1-S2 pairing incorrect
                    threshold = self.params.get('forward_look_drop_threshold', 0.4)
                    if drop_ratio < threshold:
                        # Scale penalty by severity of the drop
                        severity = (threshold - drop_ratio) / threshold
                        max_pen = self.params.get('forward_look_max_penalty', 0.3)
                        forward_look_penalty = severity * max_pen

                        confidence = max(0.0, confidence - forward_look_penalty)
                        reason += (
                            f"\n- Penalized by Forward-Look {forward_look_penalty:.2f} "
                            f"(S2→S1 drop {drop_ratio:.2f}x < threshold {threshold:.1f}x)"
                        )

        is_paired = confidence >= self.params['pairing_confidence_threshold']
        reason += (
            f"\n- Final Score: {confidence:.2f} vs Threshold "
            f"{self.params['pairing_confidence_threshold']:.2f} -> "
            f"{'Paired' if is_paired else 'Not Paired'}"
        )
        return is_paired, reason, prominence_context


class LookaheadSkipper:
    """
    Encapsulates the "skip weak middle peak" logic used before standard pairing.

    This component proposes an alternative S1→S2′ interpretation when a middle peak
    is likely noise. It never mutates `AnalysisState`; instead it returns a structured
    decision and lets the caller update state and debug records.
    """

    def __init__(
        self,
        audio_envelope: np.ndarray,
        sample_rate: int,
        params: Dict,
        pairing_engine: PairingEngine,
    ) -> None:
        self.audio_envelope = audio_envelope
        self.sample_rate = sample_rate
        self.params = params
        self.pairing_engine = pairing_engine

    def maybe_skip(
        self,
        state: AnalysisState,
        loop_idx: int,
        pairing_ratio: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Returns a decision dict if lookahead wants to reinterpret a middle peak as noise.

        The returned dict contains:
            - s1_idx, middle_idx, s2_idx: indices of the three peaks
            - reason: full pairing reason string
            - prominence_context: raw prominence context for debug tooltips
            - lookahead_msg: human‑readable summary of *why* the middle was skipped
            - middle_noise_msg: explanation attached to the middle peak marked as Noise

        If no lookahead skip is appropriate, returns None.
        """
        if not self.params.get('enable_lookahead_skipping', True):
            return None

        all_peaks = state.all_peaks
        if loop_idx + 2 >= len(all_peaks):
            return None

        current_peak_idx = all_peaks[loop_idx]
        next_peak_idx = all_peaks[loop_idx + 1]
        middle_peak_idx = next_peak_idx
        next_next_peak_idx = all_peaks[loop_idx + 2]

        # Heuristic: treat the middle peak as noise if its prominence is much weaker than the S1 candidate
        current_prom = calculate_peak_prominence(
            current_peak_idx, self.audio_envelope, state.trough_indices
        )
        middle_prom = calculate_peak_prominence(
            middle_peak_idx, self.audio_envelope, state.trough_indices
        )
        next_next_prom = calculate_peak_prominence(
            next_next_peak_idx, self.audio_envelope, state.trough_indices
        )
        noise_thresh = self.params.get('noise_prominence_threshold', 0.35)

        bpm = state.long_term_bpm
        intervals = calculate_bpm_intervals(bpm, self.params)
        min_s1_s2_interval = intervals["s1_s2_min"]
        s1_s2_max_interval = intervals["s1_s2_max"]

        s2_to_next_s1_interval_sec = (next_next_peak_idx - middle_peak_idx) / self.sample_rate
        alt_s1_s2_interval_sec = (next_next_peak_idx - current_peak_idx) / self.sample_rate

        # Determine if middle is skippable based on BOTH intervals AND intensity.
        middle_is_weak = middle_prom < current_prom * noise_thresh
        interval_is_impossible = s2_to_next_s1_interval_sec < min_s1_s2_interval
        next_next_is_strong = next_next_prom > middle_prom
        alt_interval_plausible = (
            alt_s1_s2_interval_sec >= min_s1_s2_interval
            and alt_s1_s2_interval_sec <= s1_s2_max_interval
        )

        # --- Mode 1: interval-aware reinterpretation (impossible S2→S1 interval) ---
        if middle_is_weak and interval_is_impossible and next_next_is_strong and alt_interval_plausible:
            is_paired_interval, reason_interval, prominence_context_interval = self.pairing_engine.attempt_pair(
                state, current_peak_idx, next_next_peak_idx, pairing_ratio
            )

            if is_paired_interval:
                lookahead_msg_interval = (
                    "LOOKAHEAD INTERVAL: Reinterpreted middle peak as noise because the implied "
                    f"S2→S1 interval {s2_to_next_s1_interval_sec:.3f}s is below the minimum "
                    f"{min_s1_s2_interval:.3f}s for BPM {bpm:.0f}, the alternative S1→S2′ interval "
                    f"{alt_s1_s2_interval_sec:.3f}s is within [{min_s1_s2_interval:.3f}, "
                    f"{s1_s2_max_interval:.3f}]s, and the middle peak is weak "
                    f"({middle_prom:.3f} < {noise_thresh:.2f} × S1 {current_prom:.3f}) while the "
                    f"next candidate is stronger ({next_next_prom:.3f} > {middle_prom:.3f})."
                )
                middle_noise_msg_interval = (
                    "Middle peak treated as noise due to impossible S2→S1 interval "
                    f"({s2_to_next_s1_interval_sec:.3f}s < {min_s1_s2_interval:.3f}s), with a "
                    f"plausible alternative S1→S2′ interval ({alt_s1_s2_interval_sec:.3f}s "
                    f"within [{min_s1_s2_interval:.3f}, {s1_s2_max_interval:.3f}]s), "
                    f"weak prominence ({middle_prom:.3f} < {noise_thresh:.2f} × S1 "
                    f"{current_prom:.3f}), and a stronger following candidate "
                    f"({next_next_prom:.3f} > {middle_prom:.3f})."
                )

                return {
                    "s1_idx": current_peak_idx,
                    "middle_idx": middle_peak_idx,
                    "s2_idx": next_next_peak_idx,
                    "reason": reason_interval,
                    "prominence_context": prominence_context_interval,
                    "lookahead_msg": lookahead_msg_interval,
                    "middle_noise_msg": middle_noise_msg_interval,
                }

        # --- Mode 2: simpler prominence‑based reinterpretation ---
        if (
            middle_prom < current_prom * noise_thresh
            and middle_prom < next_next_prom
            and alt_interval_plausible
        ):
            is_paired_skip, reason_skip, prominence_context_skip = self.pairing_engine.attempt_pair(
                state, current_peak_idx, next_next_peak_idx, pairing_ratio
            )

            if is_paired_skip:
                lookahead_msg = (
                    "LOOKAHEAD SUCCESS: Skipped intermediate weak peak "
                    f"(middle prominence {middle_prom:.3f} < {noise_thresh:.2f} × "
                    f"S1 prominence {current_prom:.3f} and next candidate prominence "
                    f"{next_next_prom:.3f} > middle) with plausible S1→S2′ interval "
                    f"{alt_s1_s2_interval_sec:.3f}s within "
                    f"[{min_s1_s2_interval:.3f}, {s1_s2_max_interval:.3f}]s"
                )
                middle_noise_msg = (
                    "Middle peak treated as noise due to weak prominence "
                    f"({middle_prom:.3f} < {noise_thresh:.2f} × "
                    f"S1 prominence {current_prom:.3f}) and the following "
                    f"candidate is stronger (next prominence {next_next_prom:.3f})."
                )

                return {
                    "s1_idx": current_peak_idx,
                    "middle_idx": middle_peak_idx,
                    "s2_idx": next_next_peak_idx,
                    "reason": reason_skip,
                    "prominence_context": prominence_context_skip,
                    "lookahead_msg": lookahead_msg,
                    "middle_noise_msg": middle_noise_msg,
                }

        return None


def _build_prominence_section(prominence_context: Dict[str, Any]) -> Dict[str, Any]:
    """Wraps prominence context in a debug section structure for tooltips/logs."""
    return {"type": "prominence", "details": prominence_context}


def _get_recent_s1_prominences_for_state(
    state: AnalysisState,
    audio_envelope: np.ndarray,
    trough_indices: np.ndarray,
) -> List[float]:
    """
    Helper to compute recent validated S1 prominences from an AnalysisState instance.

    Kept outside of PeakClassifier so it can be reused by PairingEngine without
    giving it write access to classifier internals.
    """
    recent_s1_types = [
        state.beat_debug_info.get(idx, {}).get("peak_type")
        for idx in state.candidate_beats[-50:]
    ]
    return [
        calculate_peak_prominence(idx, audio_envelope, trough_indices)
        for idx, typ in zip(state.candidate_beats[-50:], recent_s1_types)
        if typ in (PeakType.S1_PAIRED.value, PeakType.LONE_S1_VALIDATED.value)
    ]


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

        # Helper components that encapsulate specific decision logic.
        self.pairing_engine = PairingEngine(
            audio_envelope, sample_rate, params, peak_bpm_time_sec, recovery_end_time_sec
        )
        self.lookahead_skipper = LookaheadSkipper(
            audio_envelope, sample_rate, params, self.pairing_engine
        )

        self.state = self._initialize_state(
            start_bpm_hint, precomputed_noise_floor, precomputed_troughs
        )

    def _initialize_state(self, start_bpm_hint, precomputed_noise_floor, precomputed_troughs) -> AnalysisState:
        """Pre-calculates all necessary data and initializes the state for the peak finding loop."""
        analysis_data: Dict[str, Any] = {}
        dynamic_noise_floor, trough_indices = precomputed_noise_floor, precomputed_troughs
        all_peaks = self._find_raw_peaks(dynamic_noise_floor.values)

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
        )

    def classify_peaks(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Main classification loop to iterate through all raw peaks."""
        if len(self.state.all_peaks) < 2:
            return self.state.all_peaks, self.state.all_peaks, {"beat_debug_info": {}}

        while self.state.loop_idx < len(self.state.all_peaks):
            # Calculate pairing ratio once per iteration so all consumers
            # (kick-start recovery, pairing engine, lookahead skipper) share
            # the same view of recent rhythm stability.
            pairing_ratio = self._calculate_pairing_ratio()

            self._kickstart_check(pairing_ratio)
            current_peak_idx = self.state.all_peaks[self.state.loop_idx]
            is_last_peak = self.state.loop_idx >= len(self.state.all_peaks) - 1

            if is_last_peak:
                self._handle_last_peak(current_peak_idx)
            else:
                self._process_peak_pair(current_peak_idx, pairing_ratio)

            self._update_long_term_bpm()

        return self._finalize_results()

    def _kickstart_check(self, pairing_ratio: float) -> None:
        """
        Specialized recovery function to kick-start the algorithm if it gets stuck.
        This is a "bandaid" fix to help the algorithm recover from pairing failures. but ideally we would have a more robust solution.
        If I manage to get the algorithm good enough, this feature should never activate...
        """
        # pairing_ratio is calculated once per main-loop iteration in classify_peaks.
        if pairing_ratio >= self.params.get("kickstart_check_threshold", 0.3):
            return

        history = 4  # Hardcoded history beats
        if len(self.state.candidate_beats) < history:
            return

        min_s1s = 3  # Hardcoded min S1 candidates
        recent_lone_s1s = [
            idx for idx in self.state.candidate_beats[-history:]
            if _is_lone_s1_debug(self.state.beat_debug_info.get(idx))
        ]
        if len(recent_lone_s1s) < min_s1s:
            return

        min_matches = 3  # Hardcoded min matches
        matches = 0
        for s1_idx in recent_lone_s1s:
            current_raw_idx = np.searchsorted(self.state.all_peaks, s1_idx)
            if current_raw_idx < len(self.state.all_peaks) - 1:
                next_raw_peak_idx = self.state.all_peaks[current_raw_idx + 1]
                if _is_noise_debug(self.state.beat_debug_info.get(next_raw_peak_idx)):
                    matches += 1

        if matches >= min_matches:
            override_ratio = self.params.get("kickstart_override_ratio", 0.6)
            logging.info(f"KICK-START: Found {matches}/{len(recent_lone_s1s)} S1->Noise patterns. Overriding pairing ratio to {override_ratio}.")
            # This is a temporary state change, so we don't store the override ratio in self.state
            self.state.pairing_ratio_override = override_ratio

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

    def _process_peak_pair(self, current_peak_idx: int, pairing_ratio: float) -> None:
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
            reason = decision["reason"]
            prominence_context = decision["prominence_context"]
            lookahead_msg = decision["lookahead_msg"]
            middle_noise_msg = decision["middle_noise_msg"]

            pairing_lines = [
                line.strip().lstrip('- ')
                for line in reason.strip().split("\n")
                if line.strip()
            ]

            prominence_section = _build_prominence_section(prominence_context)

            s1_sections = [
                {"type": "lookahead", "text": lookahead_msg},
                {"type": "pairing", "lines": pairing_lines, "success": True},
                prominence_section,
            ]
            s2_sections = [
                {"type": "lookahead", "text": lookahead_msg},
                {"type": "pairing", "lines": pairing_lines, "success": True},
                prominence_section,
            ]

            self.state.candidate_beats.append(s1_idx)
            self.state.beat_debug_info[s1_idx] = {
                "peak_type": PeakType.S1_PAIRED.value,
                "sections": s1_sections,
            }

            original_middle_debug = self.state.beat_debug_info.get(middle_idx)
            self.state.beat_debug_info[middle_idx] = {
                "peak_type": PeakType.NOISE.value,
                "sections": [
                    {
                        "type": "lookahead",
                        "text": middle_noise_msg,
                    },
                    {
                        "type": "original",
                        "original_debug": original_middle_debug,
                    },
                ],
            }

            self.state.beat_debug_info[s2_idx] = {
                "peak_type": PeakType.S2_PAIRED.value,
                "sections": s2_sections,
            }

            self.state.consecutive_rr_rejections = 0
            # Skip the S1, middle noise, and S2 peaks
            self.state.loop_idx += 3
            return

        # --- Standard pairing attempt ---
        is_paired, reason, prominence_context = self.pairing_engine.attempt_pair(
            self.state, current_peak_idx, next_peak_idx, pairing_ratio
        )

        if is_paired:
            self.state.candidate_beats.append(current_peak_idx)
            pairing_lines = [
                line.strip().lstrip('- ')
                for line in reason.strip().split("\n")
                if line.strip()
            ]
            sections = [{"type": "pairing", "lines": pairing_lines, "success": True}]
            sections.append(_build_prominence_section(prominence_context))
            self.state.beat_debug_info[current_peak_idx] = {
                "peak_type": PeakType.S1_PAIRED.value,
                "sections": sections,
            }
            self.state.beat_debug_info[next_peak_idx] = {
                "peak_type": PeakType.S2_PAIRED.value,
                "sections": sections,
            }
            self.state.consecutive_rr_rejections = 0
            self.state.loop_idx += 2
        else:
            self._classify_lone_peak(current_peak_idx, reason)
            self.state.loop_idx += 1

    def _update_long_term_bpm(self):
        """Updates the long-term BPM belief after each decision."""
        if len(self.state.candidate_beats) > 1:
            new_rr = (self.state.candidate_beats[-1] - self.state.candidate_beats[-2]) / self.sample_rate
            if new_rr > 0:
                self.state.long_term_bpm = update_long_term_bpm(new_rr, self.state.long_term_bpm, self.params)

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

    def _classify_lone_peak(self, peak_idx: int, pairing_failure_reason: str):
        """Validates if an unpaired peak is a Lone S1 or Noise."""
        is_valid, lone_s1_lines = self._validate_lone_s1(peak_idx)
        pairing_lines = [
            line.strip().lstrip('- ')
            for line in pairing_failure_reason.strip().split("\n")
            if line.strip()
        ]

        if is_valid:
            self.state.candidate_beats.append(peak_idx)
            # For a validated S1, the "rejection_detail" is just the success reason.
            self.state.beat_debug_info[peak_idx] = {
                "peak_type": PeakType.LONE_S1_VALIDATED.value,
                "sections": [
                    {"type": "pairing", "lines": pairing_lines, "success": False},
                    {"type": "lone_s1", "lines": lone_s1_lines, "validated": True},
                ],
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
                    "sections": [
                        {"type": "pairing", "lines": pairing_lines, "success": False},
                        {"type": "lone_s1", "lines": lone_s1_lines, "validated": False},
                    ],
                }
                self.state.consecutive_rr_rejections = 0
            else:
                self.state.beat_debug_info[peak_idx] = {
                    "peak_type": PeakType.NOISE.value,
                    "sections": [
                        {"type": "pairing", "lines": pairing_lines, "success": False},
                        {"type": "lone_s1", "lines": lone_s1_lines, "validated": False},
                    ],
                }

    def _validate_lone_s1(self, current_peak_idx: int) -> Tuple[bool, List[str]]:
        """Performs checks to determine if a peak is a valid Lone S1."""
        detail_lines = []
        
        # --- 1. Basic rhythm & amplitude calculation ---
        if not self.state.candidate_beats:
            return True, ["Validated Lone S1: First beat (no prior rhythm to compare)."]
        
        confidence, detail_lines = calculate_lone_s1_confidence(
            current_peak_idx, self.state.candidate_beats[-1], self.state.long_term_bpm,
            self.audio_envelope, self.state.dynamic_noise_floor, self.sample_rate, self.params
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
                    f"\nAbsolute Prominence Veto: {current_prominence:.3f} < {min_ratio:.1f}× reference "
                    f"({reference_prominence:.3f}) → confidence ×{penalty_factor:.2f}"
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
                        f"\nForward check failed: next peak too close ({forward_interval_sec:.3f}s) and not strong enough"
                    )
                    confidence = 0.0  # Hard veto
        
        # --- 4. Final threshold check ---
        threshold = self.params.get("lone_s1_confidence_threshold", 0.6)
        if confidence < threshold:
            detail_lines.append(f"Outcome: Rejected Lone S1 (final score {confidence:.2f} < {threshold:.2f})")
            return False, detail_lines
        
        detail_lines.append(f"Outcome: Validated Lone S1 (final score {confidence:.2f} >= {threshold:.2f})")
        return True, detail_lines


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

def get_peak_prominence_details(
    peak_idx: int,
    audio_envelope: np.ndarray,
    trough_indices: np.ndarray,
    sample_rate: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Returns the details used when computing a peak's prominence.
    The returned dictionary includes the adjacent trough amplitudes, the key col,
    and optional timestamps (if `sample_rate` is provided).
    """
    if len(trough_indices) == 0:
        return {
            "peak_idx": peak_idx,
            "peak_amp": float(audio_envelope[peak_idx]),
            "left_trough_idx": None,
            "left_trough_amp": None,
            "right_trough_idx": None,
            "right_trough_amp": None,
            "key_col_amp": 0.0,
            "prominence": float(audio_envelope[peak_idx]),
        }

    insert_pos = np.searchsorted(trough_indices, peak_idx)

    left_trough_idx = trough_indices[insert_pos - 1] if insert_pos > 0 else None
    right_trough_idx = trough_indices[insert_pos] if insert_pos < len(trough_indices) else None

    peak_amplitude = float(audio_envelope[peak_idx])
    left_trough_amp = float(audio_envelope[left_trough_idx]) if left_trough_idx is not None else None
    right_trough_amp = float(audio_envelope[right_trough_idx]) if right_trough_idx is not None else None

    if left_trough_amp is not None and right_trough_amp is not None:
        key_col_amp = max(left_trough_amp, right_trough_amp)
    elif left_trough_amp is not None:
        key_col_amp = left_trough_amp
    elif right_trough_amp is not None:
        key_col_amp = right_trough_amp
    else:
        key_col_amp = 0.0

    prominence = max(0.0, peak_amplitude - key_col_amp)

    details: Dict[str, Any] = {
        "peak_idx": peak_idx,
        "peak_amp": peak_amplitude,
        "left_trough_idx": left_trough_idx,
        "left_trough_amp": left_trough_amp,
        "right_trough_idx": right_trough_idx,
        "right_trough_amp": right_trough_amp,
        "key_col_amp": key_col_amp,
        "prominence": prominence,
    }

    if sample_rate:
        details["peak_time"] = peak_idx / sample_rate
        details["left_trough_time"] = (
            left_trough_idx / sample_rate if left_trough_idx is not None else None
        )
        details["right_trough_time"] = (
            right_trough_idx / sample_rate if right_trough_idx is not None else None
        )

    return details


def calculate_peak_prominence(peak_idx: int, audio_envelope: np.ndarray, trough_indices: np.ndarray) -> float:
    """
    Calculate the true prominence of a peak by finding adjacent troughs.
    """
    details = get_peak_prominence_details(peak_idx, audio_envelope, trough_indices)
    return details["prominence"]


def calculate_bpm_intervals(bpm: float, params: Dict) -> Dict[str, float]:
    """
    Given a BPM value, computes key timing intervals (in seconds) implied by that rate.

    Returns a dictionary with:
      - 'rr_interval'     : full S1→S1 (R-R) interval
      - 's1_s2_min'       : minimum plausible S1→S2 interval
      - 's1_s2_nominal'   : nominal S1→S2 interval as a fraction of R-R
      - 's1_s2_max'       : maximum plausible S1→S2 interval (capped)
      - 's2_s1_nominal'   : nominal S2→S1 interval (R-R minus S1→S2 nominal)
    """
    # Guard against zero or negative BPM
    bpm = float(max(bpm, 1e-6))
    rr_interval = 60.0 / bpm

    min_frac = params.get("min_s1_s2_interval_rr_fraction", 0.35)
    nominal_frac = params.get("s1_s2_interval_rr_fraction", 0.5)
    min_abs = params.get("min_s1_s2_interval_sec", 0.15)
    cap_abs = params.get("s1_s2_interval_cap_sec", rr_interval * nominal_frac)

    s1_s2_min = max(rr_interval * min_frac, min_abs)
    s1_s2_nominal = rr_interval * nominal_frac
    s1_s2_max = min(cap_abs, s1_s2_nominal)
    s2_s1_nominal = max(0.0, rr_interval - s1_s2_nominal)

    return {
        "rr_interval": rr_interval,
        "s1_s2_min": s1_s2_min,
        "s1_s2_nominal": s1_s2_nominal,
        "s1_s2_max": s1_s2_max,
        "s2_s1_nominal": s2_s1_nominal,
    }

def adjust_confidence_with_contractility(
    base_confidence: float,
    s1_prominence: float,
    s2_prominence: float,
    bpm: float,
    params: Dict,
) -> Tuple[float, str]:
    """
    Contractility / prominence adjustment.

    Compares the measured S2/S1 prominence ratio against a BPM-based expectation:
    - If S2 is too prominent for this BPM → penalize confidence.
    - If S1 is much more prominent than expected → modestly boost confidence.
    """
    reason = ""

    # --- 1. Contractility Model: Expected S2/S1 ratio as a function of BPM ---
    # reminder to add comments to explain how the contractility model's logic was derrived from observing the data.
    # I should probably not document the logic here since it's a long explination that's better left to the documentation
    # but a surface level explanation would be helpful.

    expected_max_ratio = np.interp(
        bpm,
        [params["contractility_bpm_low"], params["contractility_bpm_high"]],
        [params["s2_s1_ratio_low_bpm"], params["s2_s1_ratio_high_bpm"]],
    )

    # --- 2. Reality: Measured S2/S1 prominence ratio ---
    actual_ratio = s2_prominence / (s1_prominence + 1e-9)
    reason += (
        f"\n- Prominence: S1={s1_prominence:.3f}, S2={s2_prominence:.3f}, "
        f"S2/S1={actual_ratio:.2f} (Expected max {expected_max_ratio:.2f} at {bpm:.0f} BPM)"
    )

    # --- 3. Apply contractility logic ---
    if actual_ratio > expected_max_ratio:
        # S2 is too prominent for this BPM → penalize
        violation = (actual_ratio / (expected_max_ratio + 1e-9)) - 1.0
        penalty_strength = params.get("contractility_penalty_strength", 0.4)
        penalty = violation * penalty_strength
        confidence = max(0.0, base_confidence - penalty)
        reason += (
            f"\n Contractility Penalty: -{penalty:.2f} "
            f"(S2 too prominent for BPM; prominence ratio {actual_ratio:.2f} > expected {expected_max_ratio:.2f}) -> {confidence:.2f}"
        )
    elif actual_ratio < expected_max_ratio * 0.5:
        # S1 is much more prominent than expected → mild boost
        boost = params.get("contractility_boost_amount", 0.15)
        confidence = min(1.0, base_confidence + boost)
        reason += (
            f"\n Contractility Boost: +{boost:.2f} "
            f"(S1 >> S2 at {bpm:.0f} BPM; prominence ratio {actual_ratio:.2f}) -> {confidence:.2f}"
        )
    else:
        # Within expected range → leave confidence unchanged
        confidence = base_confidence
        reason += (
            f"\n Contractility Neutral: prominence ratio {actual_ratio:.2f} within expected range "
            f"for {bpm:.0f} BPM, confidence unchanged"
        )

    return confidence, reason


def _apply_other_pairing_adjustments(
    confidence: float,
    s1_idx: int,
    s2_idx: int,
    audio_envelope: np.ndarray,
    dynamic_noise_floor: pd.Series,
    long_term_bpm: float,
    pairing_ratio: float,
    params: Dict,
    sample_rate: int,
    peak_bpm_time_sec: Optional[float],
    recovery_end_time_sec: Optional[float],
    beat_count: int,
) -> Tuple[float, str]:
    """
    Applies non-contractility adjustments:
    - Stability based on recent pairing success, with recovery-phase awareness.
    - (Keeps the door open for future non-prominence heuristics.)

    All S1/S2 prominence-based contractility logic is handled separately.
    """
    reason = ""

    # --- Stability adjustment ---
    if beat_count >= 5:
        floor = params.get("stability_confidence_floor", 0.85)
        ceiling = params.get("stability_confidence_ceiling", 1.10)
        
        # During recovery, S2 may have been physiologically absent during high BPM,
        # causing pairing_ratio to drop artificially. To avoid penalizing valid S1-S2
        # pairs when S2 re-emerges, we use a higher (more forgiving) floor.
        current_time_sec = s1_idx / sample_rate
        if (peak_bpm_time_sec is not None and 
            recovery_end_time_sec is not None and
            peak_bpm_time_sec <= current_time_sec <= recovery_end_time_sec):
            # Override floor with recovery-specific value (default 1.0 = no penalty)
            recovery_floor = params.get("recovery_phase_stability_floor", 0.95)
            original_floor = floor
            floor = max(floor, recovery_floor)
            reason += f"\n- Recovery Phase Adjust: floor {original_floor:.2f} → {floor:.2f} (peak at {peak_bpm_time_sec:.1f}s)"

        stability_factor = np.interp(pairing_ratio, [0.0, 1.0], [floor, ceiling])
        confidence *= stability_factor
        reason += f"\n- Stability Adjust: x{stability_factor:.2f} (Pairing Ratio: {pairing_ratio:.0%}, Floor: {floor:.2f}) → {confidence:.2f}"

    return max(0.0, min(1.0, confidence)), reason


def calculate_lone_s1_confidence(current_peak_idx: int, last_s1_idx: int, long_term_bpm: float, audio_envelope: np.ndarray,
                                 dynamic_noise_floor: pd.Series, sample_rate: int, params: Dict) -> Tuple[float, List[str]]:
    """
    Calculates a confidence score for a Lone S1 candidate based on a weighted average of
    its rhythmic timing and its amplitude consistency with the previous beat, and returns
    human-readable detail lines explaining the calculation.
    """
    # --- 1. Calculate Rhythmic Fit Score ---
    expected_rr_sec = calculate_bpm_intervals(long_term_bpm, params)["rr_interval"]
    actual_rr_sec = (current_peak_idx - last_s1_idx) / sample_rate
    rhythm_deviation_pct = abs(actual_rr_sec - expected_rr_sec) / expected_rr_sec

    rhythm_score = np.interp(
        rhythm_deviation_pct,
        [0.0, 0.15, 0.40, 0.60],  # Hardcoded rhythm deviation points
        [1.0, 0.8, 0.4, 0.0]      # Hardcoded rhythm confidence curve
    )
    rhythm_reason = (
        f"Rhythm Fit {rhythm_score:.2f}: interval {actual_rr_sec:.3f}s vs expected {expected_rr_sec:.3f}s "
        f"(deviation {rhythm_deviation_pct:.0%}; map 0/15/30/50% -> 1.00/0.80/0.40/0.00)"
    )

    # --- 2. Calculate Amplitude Fit Score ---
    last_s1_strength = max(0, audio_envelope[last_s1_idx] - dynamic_noise_floor.iloc[last_s1_idx])
    current_peak_strength = max(0, audio_envelope[current_peak_idx] - dynamic_noise_floor.iloc[current_peak_idx])
    amplitude_ratio = current_peak_strength / (last_s1_strength + 1e-9)

    amplitude_score = np.interp(
        amplitude_ratio,
        [0.0, 0.4, 0.7, 1.0],      # Hardcoded amplitude ratio points
        [0.0, 0.4, 0.7, 1.0]       # Hardcoded amplitude confidence curve
    )
    amplitude_reason = (
        f"Amplitude Fit {amplitude_score:.2f}: strength ratio {amplitude_ratio:.2f}x "
        f"(map 0/0.4/0.7/1.0 -> 0/0.4/0.8/1.0)"
    )

    # --- 3. Combine Scores with Weights ---
    rhythm_weight = params.get('lone_s1_rhythm_weight', 0.65)
    amplitude_weight = params.get('lone_s1_amplitude_weight', 0.35)
    final_confidence = (rhythm_score * rhythm_weight) + (amplitude_score * amplitude_weight)

    reason_lines = [
        rhythm_reason,
        amplitude_reason,
        (
            f"Weighted Score: (Rhythm {rhythm_score:.2f} x {rhythm_weight:.2f}) + "
            f"(Amplitude {amplitude_score:.2f} x {amplitude_weight:.2f}) = {final_confidence:.3f}"
        ),
    ]
    return final_confidence, reason_lines

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
                    log_debug(f"  - SUCCESS: Re-labeling S1/S2 pair at {candidate_s1 / sample_rate:.2f}s.")
                    corrections_made += 1
                    peaks_to_add.add(candidate_s1)

                    original_reason_s1 = corrected_debug_info.get(candidate_s1)
                    corrected_debug_info[candidate_s1] = {
                        "peak_type": PeakType.S1_CORRECTED_GAP.value,
                        "sections": [
                            {"type": "original", "original_debug": original_reason_s1},
                        ],
                    }

                    original_reason_s2 = corrected_debug_info.get(candidate_s2)
                    corrected_debug_info[candidate_s2] = {
                        "peak_type": PeakType.S2_CORRECTED_GAP.value,
                        "sections": [
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


def detect_trapezoid_discontinuities(smoothed_bpm: pd.Series, bpm_times_sec: np.ndarray, params: Dict) -> List[Dict]:
    """
    The human eye can easily identify errors in the BPM/time graph so I implemented this function to allow 
    the script to identify them automatically.
    Detects trapezoid-shaped discontinuities in the average BPM series that are
    characteristic of a brief extra-beat artifact:
      - A very fast rise
      - A sustained (possibly slightly sloped) plateau
      - A very fast fall that returns to baseline

    I plan on pairing this with other logic to detect PVCs or issues with the algorithm's labeling but I haven't implemented it yet.

    It assumes `smoothed_bpm` contains the average BPM values and `bpm_times_sec`
    contains the corresponding time stamps in seconds.
    """
    if (
        smoothed_bpm is None
        or smoothed_bpm.empty
        or bpm_times_sec is None
        or len(bpm_times_sec) != len(smoothed_bpm)
    ):
        return []

    # Build working DataFrame equivalent to the CSV used in detectTrapezoid.py
    df = pd.DataFrame(
        {
            "Time (s)": bpm_times_sec.astype(float),
            "Average BPM": smoothed_bpm.to_numpy(dtype=float),
        }
    ).dropna(subset=["Time (s)", "Average BPM"])

    if len(df) < 4:
        return []

    # Calculate differences and instantaneous rate of BPM change
    df["Δt"] = df["Time (s)"].diff()
    df["ΔBPM"] = df["Average BPM"].diff()
    df["Rate"] = df["ΔBPM"] / df["Δt"]

    # --- CONFIGURATION (now driven by params, defaults mirror detectTrapezoid.py) ---
    RATE_THRESHOLD = params.get("trapezoid_rate_threshold", 7.0)                 # BPM/s
    MAX_EDGE_DURATION = params.get("trapezoid_max_edge_duration_sec", 1.5)       # seconds
    MIN_PLATEAU_DURATION = params.get("trapezoid_min_plateau_duration_sec", 1.5) # seconds
    MAX_PLATEAU_DURATION = params.get("trapezoid_max_plateau_duration_sec", 15.0)# seconds
    BASELINE_TOLERANCE = params.get("trapezoid_baseline_tolerance_bpm", 5.0)     # BPM
    MIN_JUMP = params.get("trapezoid_min_jump_bpm", 6.0)                         # BPM
    MIN_FALL_DELTA = params.get("trapezoid_min_fall_delta_bpm", 5.0)             # BPM

    # Step 1: Identify edge intervals (second point of each edge)
    df["is_rise"] = (df["Rate"] > RATE_THRESHOLD) & (df["Δt"] < MAX_EDGE_DURATION)
    df["is_fall"] = (df["Rate"] < -RATE_THRESHOLD) & (df["Δt"] < MAX_EDGE_DURATION)

    rise_indices = df[df["is_rise"]].index.tolist()
    fall_indices = df[df["is_fall"]].index.tolist()

    trapezoids: List[Dict] = []

    # Step 2: Match edges into trapezoids
    for rise_idx in rise_indices:
        # Need at least one sample before the rise edge for t1 / baseline
        if rise_idx <= 0:
            continue

        rise_time = float(df.loc[rise_idx, "Time (s)"])

        for fall_idx in list(fall_indices):
            # Need at least one sample before fall edge for t3
            if fall_idx <= 0:
                continue

            fall_time = float(df.loc[fall_idx, "Time (s)"])

            # Timing constraints: plateau must be long enough but not absurdly long
            plateau_duration = fall_time - rise_time
            if not (MIN_PLATEAU_DURATION <= plateau_duration <= MAX_PLATEAU_DURATION):
                continue

            # --- Validate plateau (region strictly between rise and fall) ---
            plateau_mask = (df["Time (s)"] > rise_time) & (df["Time (s)"] < fall_time)
            plateau_df = df[plateau_mask]
            if plateau_df.empty:
                continue

            # Allow sloped plateaus: median absolute rate should be modest
            if plateau_df["Rate"].abs().median() > RATE_THRESHOLD / 3.0:
                continue

            # --- Validate baseline recovery ---
            # Baseline before: up to 3 points before the rise edge
            before_start_idx = max(0, rise_idx - 3)
            before_end_idx = rise_idx - 1
            if before_end_idx < before_start_idx:
                continue
            baseline_before = float(
                df.loc[before_start_idx:before_end_idx, "Average BPM"].mean()
            )

            # Baseline after: up to 3 points starting at fall edge
            after_end_idx = min(fall_idx + 2, df.index[-1])
            baseline_after = float(
                df.loc[fall_idx:after_end_idx, "Average BPM"].mean()
            )

            if abs(baseline_after - baseline_before) > BASELINE_TOLERANCE:
                continue

            # --- Calculate the four key timestamps ---
            t1 = float(df.loc[rise_idx - 1, "Time (s)"])
            t2 = rise_time
            t3 = float(df.loc[fall_idx - 1, "Time (s)"])
            t4 = fall_time

            # Validate edge intervals are brief
            if (t2 - t1) > MAX_EDGE_DURATION or (t4 - t3) > MAX_EDGE_DURATION:
                continue

            # Enforce a minimum BPM change across the fall edge itself.
            # If the fall barely changes BPM, don't treat it as a trapezoid artifact.
            fall_start_bpm = float(df.loc[fall_idx - 1, "Average BPM"])
            fall_end_bpm = float(df.loc[fall_idx, "Average BPM"])
            if abs(fall_start_bpm - fall_end_bpm) < MIN_FALL_DELTA:
                continue

            # Calculate jump from baseline to plateau median
            plateau_median = float(plateau_df["Average BPM"].median())
            jump_size = plateau_median - baseline_before
            if jump_size < MIN_JUMP:
                continue

            plateau_slope = float(
                plateau_df["Average BPM"].iloc[-1] - plateau_df["Average BPM"].iloc[0]
            )

            # Store both timestamps and BPM values for debugging/plotting
            trap = {
                "t_start_rise": t1,
                "t_end_rise": t2,
                "t_start_fall": t3,
                "t_end_fall": t4,
                "bpm_start_rise": float(df.loc[rise_idx - 1, "Average BPM"]),
                "bpm_end_rise": float(df.loc[rise_idx, "Average BPM"]),
                "bpm_start_fall": fall_start_bpm,
                "bpm_end_fall": fall_end_bpm,
                "baseline_before": baseline_before,
                "plateau_median": plateau_median,
                "plateau_slope": plateau_slope,
                "jump_size": jump_size,
                "baseline_after": baseline_after,
                "baseline_diff": baseline_after - baseline_before,
                "plateau_duration": plateau_duration,
                "plateau_points": int(len(plateau_df)),
            }
            trapezoids.append(trap)

            # Remove used fall index so it cannot be reused by another rise
            fall_indices.remove(fall_idx)
            break

    if trapezoids:
        logging.info(f"Detected {len(trapezoids)} trapezoid HR artifacts (sudden plateau jumps):")
        for i, trap in enumerate(trapezoids, 1):
            logging.info(
                "  Trapezoid #%d: "
                "Rise %.3fs (%.1f BPM) → %.3fs (%.1f BPM); "
                "Plateau %.3fs → %.3fs (Δ%.3fs, %d pts); "
                "Fall %.3fs (%.1f BPM) → %.3fs (%.1f BPM)",
                i,
                trap["t_start_rise"],
                trap["bpm_start_rise"],
                trap["t_end_rise"],
                trap["bpm_end_rise"],
                trap["t_end_rise"],
                trap["t_start_fall"],
                trap["plateau_duration"],
                trap["plateau_points"],
                trap["t_start_fall"],
                trap["bpm_start_fall"],
                trap["t_end_fall"],
                trap["bpm_end_fall"],
            )
    else:
        logging.info("No trapezoid-like HR artifacts detected in average BPM series.")

    return trapezoids

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
    metrics['trapezoids'] = detect_trapezoid_discontinuities(metrics['smoothed_bpm'], metrics['bpm_times'], params)
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


def _load_manual_labels_csv(audio_file_path: str) -> Optional[Dict[str, str]]:
    """
    Looks for a '*_manually_Labeled_peaks.csv' file next to the analyzed audio file
    and loads it into a mapping: rounded time_sec ('%.3f') -> canonical label ('S1'/'S2'/'Noise').

    The CSV format matches the export from 'interactive_plot.js':
        time_sec,base_label,manual_label,x_plot_sec,y_plot
    """
    base_dir = os.path.dirname(audio_file_path) or "."
    base_name = os.path.basename(audio_file_path)
    csv_name = f"{base_name}_manually_Labeled_peaks.csv"
    csv_path = os.path.join(base_dir, csv_name)

    if not os.path.exists(csv_path):
        return None

    labels_by_time: Dict[str, str] = {}

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                logging.warning(f"Manual labels CSV is empty or missing header: {csv_path}")
                return None

            lower = [h.strip().lower() for h in header]
            try:
                time_idx = lower.index("time_sec")
            except ValueError:
                logging.warning(f"Manual labels CSV missing 'time_sec' column: {csv_path}")
                return None

            manual_idx = lower.index("manual_label") if "manual_label" in lower else -1
            base_idx = lower.index("base_label") if "base_label" in lower else -1

            if manual_idx == -1 and base_idx == -1:
                logging.warning(
                    f"Manual labels CSV must contain 'manual_label' or 'base_label' column: {csv_path}"
                )
                return None

            for row in reader:
                if len(row) <= time_idx:
                    continue
                raw_t = row[time_idx]
                try:
                    t = float(raw_t)
                except (TypeError, ValueError):
                    continue

                manual_label = ""
                base_label = ""
                if manual_idx != -1 and len(row) > manual_idx:
                    manual_label = (row[manual_idx] or "").strip()
                if base_idx != -1 and len(row) > base_idx:
                    base_label = (row[base_idx] or "").strip()

                # Prefer manual_label if present; otherwise fall back to base_label.
                chosen = manual_label or base_label
                if not chosen:
                    continue

                # Normalize to the same coarse label space we use for predictions.
                # The interactive tool uses exactly 'S1', 'S2', or 'Noise', but we
                # keep this tolerant in case of minor variations.
                val = chosen.strip()
                if val.startswith("S1"):
                    norm = "S1"
                elif val.startswith("S2"):
                    norm = "S2"
                elif val.lower().startswith("noise"):
                    norm = "Noise"
                else:
                    # Ignore unknown labels rather than guessing.
                    continue

                key = f"{t:.3f}"
                labels_by_time[key] = norm

    except Exception as e:
        logging.error(f"Failed to read manual labels CSV '{csv_path}': {e}")
        return None

    if not labels_by_time:
        logging.info(f"Manual labels CSV found but contained no usable label rows: {csv_path}")
        return None

    logging.info(
        f"Loaded {len(labels_by_time)} manual peak labels from '{csv_name}' for validation."
    )
    return labels_by_time


def _springer_states_to_peaks(assigned_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    From Springer assigned_states (1=S1, 2=systole, 3=S2, 4=diastole), compute segment
    midpoints for S1 and S2. Returns (s1_indices, s2_indices) as 1d arrays of sample indices.
    """
    assigned_states = np.asarray(assigned_states).flatten()
    s1_indices = []
    s2_indices = []
    n = len(assigned_states)
    i = 0
    while i < n:
        s = int(assigned_states[i])
        start = i
        while i < n and int(assigned_states[i]) == s:
            i += 1
        end = i
        mid = start + (end - start) // 2
        if s == 1:
            s1_indices.append(mid)
        elif s == 3:
            s2_indices.append(mid)
    return np.array(s1_indices, dtype=np.int64), np.array(s2_indices, dtype=np.int64)


def _springer_states_to_segments(
    assigned_states: np.ndarray, sample_rate: float
) -> List[Dict[str, Any]]:
    """
    From Springer assigned_states (1=S1, 2=systole, 3=S2, 4=diastole), build a list of
    segments for visualization: each {"s": start_sec, "e": end_sec, "state": 1|2|3|4}.
    """
    assigned_states = np.asarray(assigned_states).flatten()
    segments = []
    n = len(assigned_states)
    i = 0
    while i < n:
        state = int(assigned_states[i])
        start = i
        while i < n and int(assigned_states[i]) == state:
            i += 1
        end = i
        start_sec = start / float(sample_rate)
        end_sec = end / float(sample_rate)
        segments.append({"s": round(start_sec, 4), "e": round(end_sec, 4), "state": state})
    return segments


def _build_predicted_labels_for_validation(
    analysis_data: Dict, sample_rate: int
) -> Dict[str, str]:
    """
    Builds a mapping of rounded time_sec ('%.3f') -> coarse label ('S1'/'S2'/'Noise'/'Unknown')
    from the analysis debug info.
    """
    debug_info = analysis_data.get("beat_debug_info", {})
    labels_by_time: Dict[str, str] = {}

    if not debug_info:
        return labels_by_time

    for peak_idx, entry in debug_info.items():
        try:
            t = float(peak_idx) / float(sample_rate)
        except Exception:
            continue
        key = f"{t:.3f}"
        label = _simple_label_from_debug(entry)
        # In the rare case of duplicated times at this rounding, we let the
        # last one win; this mirrors how the JS importer behaves.
        labels_by_time[key] = label

    return labels_by_time


def _append_validation_results_row(
    regression_log_path: Optional[str],
    audio_file_path: str,
    manual_labels: Dict[str, str],
    predicted_labels: Dict[str, str],
) -> None:
    """
    Compares manual vs predicted labels and logs a per-file summary.

    For any file with discrepancies, it also logs each mismatched time
    with the algorithm's label and the correct manual label.
    """
    if not manual_labels:
        return

    all_truth_keys = set(manual_labels.keys())
    all_pred_keys = set(predicted_labels.keys())

    matched_keys = all_truth_keys & all_pred_keys
    missing_keys = all_truth_keys - all_pred_keys   # manual label exists, prediction missing
    extra_keys = all_pred_keys - all_truth_keys     # prediction exists, no manual label

    correct = 0
    mismatched = 0
    for k in matched_keys:
        if manual_labels[k] == predicted_labels.get(k):
            correct += 1
        else:
            mismatched += 1

    manual_count = len(all_truth_keys)
    predicted_count = len(all_pred_keys)
    missing = len(missing_keys)
    extra = len(extra_keys)
    total_errors = mismatched + missing + extra

    audio_name = os.path.basename(audio_file_path)

    # --- Console logging summary ---
    if total_errors == 0:
        logging.info(
            "Manual label validation for '%s': all %d peaks matched.",
            audio_name,
            manual_count,
        )
    else:
        logging.info(
            "Manual label validation for '%s': manual=%d, predicted=%d, matched=%d, "
            "correct=%d, mismatched=%d, missing=%d, extra=%d, total_errors=%d",
            audio_name,
            manual_count,
            predicted_count,
            len(matched_keys),
            correct,
            mismatched,
            missing,
            extra,
            total_errors,
        )

    # --- Optional regression-testing markdown log ---
    log_file = None
    if regression_log_path:
        try:
            log_file = open(regression_log_path, "a", encoding="utf-8")
            log_file.write(f"## {audio_name}\n\n")
            log_file.write(f"- **Manual peaks**: {manual_count}\n")
            log_file.write(f"- **Predicted peaks**: {predicted_count}\n")
            log_file.write(f"- **Matched peaks**: {len(matched_keys)}\n")
            log_file.write(f"- **Correct matches**: {correct}\n")
            log_file.write(f"- **Label mismatches**: {mismatched}\n")
            log_file.write(f"- **Missing detections**: {missing}\n")
            log_file.write(f"- **Extra detections**: {extra}\n")
            log_file.write(f"- **Total errors**: {total_errors}\n\n")

            if total_errors == 0:
                log_file.write("All peaks matched between algorithm and manual labels.\n\n")
        except Exception as e:
            logging.error(
                "Failed to append validation summary to regression log '%s': %s",
                regression_log_path,
                e,
            )
            log_file = None

    # Detailed per-peak differences: algorithm vs correct label.
    # 1) Label mismatches where both sides have a label.
    if mismatched > 0:
        logging.info("  Label mismatches for '%s':", audio_name)
        if log_file:
            log_file.write("### Label mismatches\n")
        for k in sorted(matched_keys):
            true_label = manual_labels[k]
            pred_label = predicted_labels.get(k, "Unknown")
            if true_label == pred_label:
                continue
            try:
                t = float(k)
            except ValueError:
                t = k
            formatted_t = f"{t:.3f}" if isinstance(t, (float, int)) else str(t)
            logging.info(
                "    t=%s s  manual=%s  predicted=%s",
                formatted_t,
                true_label,
                pred_label,
            )
            if log_file:
                log_file.write(
                    f"- t={formatted_t} s — **manual**: {true_label}, **predicted**: {pred_label}\n"
                )
        if log_file:
            log_file.write("\n")

    # 2) Manual peaks that had no corresponding prediction.
    if missing > 0:
        logging.info("  Missing detections for '%s' (manual label but no predicted peak):", audio_name)
        if log_file:
            log_file.write("### Missing detections (manual label but no predicted peak)\n")
        for k in sorted(missing_keys):
            true_label = manual_labels[k]
            try:
                t = float(k)
            except ValueError:
                t = k
            formatted_t = f"{t:.3f}" if isinstance(t, (float, int)) else str(t)
            logging.info(
                "    t=%s s  manual=%s  predicted=<none>",
                formatted_t,
                true_label,
            )
            if log_file:
                log_file.write(
                    f"- t={formatted_t} s — **manual**: {true_label}, **predicted**: <none>\n"
                )
        if log_file:
            log_file.write("\n")

    # 3) Extra predictions that have no manual label.
    if extra > 0:
        logging.info("  Extra detections for '%s' (predicted peak but no manual label):", audio_name)
        if log_file:
            log_file.write("### Extra detections (predicted peak but no manual label)\n")
        for k in sorted(extra_keys):
            pred_label = predicted_labels.get(k, "Unknown")
            try:
                t = float(k)
            except ValueError:
                t = k
            formatted_t = f"{t:.3f}" if isinstance(t, (float, int)) else str(t)
            logging.info(
                "    t=%s s  manual=<none>  predicted=%s",
                formatted_t,
                pred_label,
            )
            if log_file:
                log_file.write(
                    f"- t={formatted_t} s — **manual**: <none>, **predicted**: {pred_label}\n"
                )
        if log_file:
            log_file.write("\n")

    if log_file:
        try:
            log_file.flush()
            log_file.close()
        except Exception:
            pass


class _NoisyAlgorithmLogFilter(logging.Filter):
    """
    Filters out very chatty INFO-level messages that make benchmarking hard.
    WARNING/ERROR always pass through.
    """

    # Substrings that identify "noisy" algorithm-detail logs.
    _NOISY_SUBSTRINGS = (
        "KICK-START:",
        "CASCADE RESET:",
        "Trapezoid #",
        "LOOKAHEAD ",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True

        try:
            msg = record.getMessage()
        except Exception:
            return True

        return not any(s in msg for s in self._NOISY_SUBSTRINGS)


def analyze_wav_file(wav_file_path: str, params: Dict, start_bpm_hint: Optional[float], original_file_path: str, output_directory: str, output_options: Optional[Dict] = None):
    """Main analysis pipeline that orchestrates the refactored classes."""
    # Honor optional verbose logging flag from params to control how noisy the console is.
    # When disabled, we keep stage-level INFO logs but suppress very chatty algorithm-detail INFO logs.
    verbose_logging = bool(params.get("verbose_console_logging", True))
    root_logger = logging.getLogger()
    active_filters = []

    if not verbose_logging:
        filt = _NoisyAlgorithmLogFilter()
        for handler in root_logger.handlers:
            handler.addFilter(filt)
            active_filters.append((handler, filt))

    start_time = time.time()
    logging.info(f"--- Processing file: {os.path.basename(original_file_path)} ---")

    segmenter = (params.get("segmenter") or "default").strip().lower()
    if segmenter == "springer":
        # Springer path (paper-aligned): raw/resampled PCG at Springer Fs, no our preprocessing.
        # All filtering (25-400 Hz, spike removal, envelopes) is done inside the Springer pipeline.
        model_path = (params.get("springer_model_path") or "").strip()
        if not model_path or not os.path.isfile(model_path):
            logging.error(
                "Springer segmenter selected but no valid model file. Set springer_model_path to a .npz file."
            )
            return None
        try:
            from springer_hsmm import run_springer_segmentation_algorithm
            from springer_hsmm.model_io import load_springer_model
            from springer_hsmm.options import default_springer_hsmm_options
        except ImportError as e:
            logging.error("Springer HSMM package not available: %s", e)
            return None
        options = default_springer_hsmm_options()
        springer_fs = int(options["audio_Fs"])
        logging.info("--- Springer HSMM segmentation (paper pipeline at %d Hz) ---", springer_fs)
        audio_data, sample_rate = load_audio_mono(wav_file_path, springer_fs)
        model = load_springer_model(model_path)
        result = run_springer_segmentation_algorithm(
            audio_data,
            float(sample_rate),
            model["B_matrix"],
            model["pi_vector"],
            model["total_obs_distribution"],
            options=options,
            return_debug=True,
            return_viz_data=True,
        )
        assigned_states, audio_envelope = result[0], result[1]
        springer_debug = result[2] if len(result) > 2 else None
        springer_pipeline_viz = result[3] if len(result) > 3 else None
        s1_indices, s2_indices = _springer_states_to_peaks(assigned_states)
        final_peaks = np.asarray(s1_indices, dtype=np.int64)
        all_raw_peaks = np.unique(np.concatenate([np.asarray(s1_indices), np.asarray(s2_indices)])) if (len(s1_indices) or len(s2_indices)) else np.array([], dtype=np.int64)
        beat_debug = {}
        for idx in s1_indices:
            beat_debug[int(idx)] = {"peak_type": "S1 (Springer)", "sections": []}
        for idx in s2_indices:
            beat_debug[int(idx)] = {"peak_type": "S2 (Springer)", "sections": []}
        springer_segments = _springer_states_to_segments(assigned_states, float(sample_rate))
        if springer_debug is not None:
            springer_debug["num_segments"] = len(springer_segments)
            # Segment duration stats per state (ms) to check if S1/S2 lengths are plausible
            from collections import defaultdict
            dur_by_state = defaultdict(list)
            for seg in springer_segments:
                dur_sec = seg["e"] - seg["s"]
                dur_by_state[seg["state"]].append(dur_sec)
            segment_duration_stats = {}
            for st in (1, 2, 3, 4):
                durs = dur_by_state.get(st, [])
                if durs:
                    arr = np.array(durs)
                    segment_duration_stats[st] = {
                        "mean_sec": float(np.mean(arr)),
                        "std_sec": float(np.std(arr)) if len(arr) > 1 else 0.0,
                        "count": len(durs),
                    }
                else:
                    segment_duration_stats[st] = {"mean_sec": None, "std_sec": None, "count": 0}
            springer_debug["segment_duration_stats"] = segment_duration_stats
            # First N segments for visual check
            first_n = 20
            first_segments = [
                {"s": seg["s"], "e": seg["e"], "state": seg["state"], "duration_sec": round(seg["e"] - seg["s"], 4)}
                for seg in springer_segments[:first_n]
            ]
            springer_debug["first_segments"] = first_segments
            # Envelope-vs-Springer alignment: do Springer S1/S2 midpoints line up with envelope peaks?
            peak_indices, _ = find_peaks(
                audio_envelope,
                distance=max(1, int(0.15 * sample_rate)),
                prominence=np.percentile(audio_envelope, 75) * 0.1 if audio_envelope.size else 0.01,
            )
            envelope_peak_times_sec = (peak_indices / float(sample_rate)).tolist()[:20]
            s1_midpoints = [(seg["s"] + seg["e"]) / 2.0 for seg in springer_segments if seg["state"] == 1][:20]
            s2_midpoints = [(seg["s"] + seg["e"]) / 2.0 for seg in springer_segments if seg["state"] == 3][:20]
            def offset_ms_to_nearest_peak(midpoint_sec, peak_times_sec):
                if not peak_times_sec:
                    return None
                peak_arr = np.array(peak_times_sec)
                return float((midpoint_sec - peak_arr[np.argmin(np.abs(peak_arr - midpoint_sec))]) * 1000)
            s1_offsets_ms = [offset_ms_to_nearest_peak(m, envelope_peak_times_sec) for m in s1_midpoints]
            s2_offsets_ms = [offset_ms_to_nearest_peak(m, envelope_peak_times_sec) for m in s2_midpoints]
            springer_debug["envelope_peak_times_sec"] = envelope_peak_times_sec[:15]
            springer_debug["s1_midpoint_times_sec"] = s1_midpoints[:15]
            springer_debug["s2_midpoint_times_sec"] = s2_midpoints[:15]
            springer_debug["s1_to_nearest_peak_offset_ms"] = s1_offsets_ms
            springer_debug["s2_to_nearest_peak_offset_ms"] = s2_offsets_ms
            springer_debug["mean_abs_offset_s1_ms"] = float(np.mean(np.abs(s1_offsets_ms))) if s1_offsets_ms else None
            springer_debug["mean_abs_offset_s2_ms"] = float(np.mean(np.abs(s2_offsets_ms))) if s2_offsets_ms else None
        analysis_data = {
            "segmenter": "springer",
            "springer_s1_indices": np.asarray(s1_indices),
            "springer_s2_indices": np.asarray(s2_indices),
            "springer_segments": springer_segments,
            "springer_debug": springer_debug,
            "springer_pipeline_viz": springer_pipeline_viz,
            "beat_debug_info": beat_debug,
            "trough_indices": np.array([], dtype=np.int64),
            "dynamic_noise_floor_series": None,
        }
    else:
        # Default path: Stage 1 (preprocess, noise floor, preliminary pass) then classifier.
        # STAGE 1: Initialization
        audio_envelope, sample_rate = preprocess_audio(wav_file_path, params, output_directory, output_options)
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

    # OPTIONAL: Validation against manually labeled peaks (if a CSV exists next to the WAV).
    # This lets you batch-run a dataset and get an objective error count per file
    # without changing the main analysis workflow or outputs.
    try:
        manual_labels = _load_manual_labels_csv(original_file_path)
        if manual_labels:
            predicted_labels = _build_predicted_labels_for_validation(
                analysis_data, sample_rate
            )
            regression_log_path = None
            if output_options is not None:
                regression_log_path = output_options.get("regression_log_path")
            _append_validation_results_row(
                regression_log_path, original_file_path, manual_labels, predicted_labels
            )
        else:
            logging.info(
                "No manual labels CSV found for '%s'; skipping validation for this file.",
                os.path.basename(original_file_path),
            )
    except Exception as e:
        logging.error(
            "Manual label validation step failed for '%s': %s",
            os.path.basename(original_file_path),
            e,
        )

    # Set default output options if none provided
    if output_options is None:
        output_options = {
            'html': True,
            'png': False,
            'csv': True,
            'summary': True,
            'debug': True,
            'filtered_wav': True,
            'bpm_text': False,
            'spectrogram': True,
        }

    plotly_figure = None
    
    # Generate plot outputs if requested (HTML/PNG/CSV share the same figure generation)
    needs_plot_outputs = any([
        output_options.get('html', True),
        output_options.get('png', False),
        output_options.get('csv', True),
    ])

    if needs_plot_outputs:
        plotter = Plotter(
            original_file_path,
            params,
            sample_rate,
            output_directory,
            source_audio_path=wav_file_path,
            peak_type_helper=_get_peak_type_from_debug,
            format_debug_entry_func=format_debug_entry,
            peak_type_cls=PeakType,
        )
        plotly_figure = plotter.plot_and_save(audio_envelope, all_raw_peaks, analysis_data, final_metrics, output_options)
    else:
        logging.info("Skipping all plot outputs (HTML/PNG/CSV) as requested.")

    # Generate other outputs if requested
    needs_reporter = any([
        output_options.get('summary', True),
        output_options.get('debug', True),
        output_options.get('bpm_text', False),
    ])

    if needs_reporter:
        reporter = ReportGenerator(original_file_path, output_directory)
        
        if output_options.get('summary', True):
            reporter.save_analysis_summary(final_metrics)
        else:
            logging.info("Skipping summary generation as requested.")
            
        if output_options.get('debug', True):
            reporter.create_chronological_log(audio_envelope, sample_rate, all_raw_peaks, analysis_data, final_metrics)
        else:
            logging.info("Skipping debug log generation as requested.")
            
        if output_options.get('bpm_text', False):
            reporter.save_bpm_time_txt(final_metrics.get('smoothed_bpm'), final_metrics.get('bpm_times'))
        else:
            logging.info("Skipping BPM text export as requested.")
    else:
        logging.info("Skipping all report generation as requested.")

    duration = time.time() - start_time
    logging.info(f"--- Analysis stage finished in {duration:.2f} seconds (post-conversion). ---")

    # Remove filters so this setting is scoped to the analysis call.
    for handler, filt in active_filters:
        try:
            handler.removeFilter(filt)
        except Exception:
            pass

    return plotly_figure
