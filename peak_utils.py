# peak_utils.py
# Shared peak types, debug-entry formatters, and per-peak prominence calculations.
# Consumed by bpm_analysis (engine), plotting, and reporting.
# This module has no dependency on any other project module.

from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

# Piecewise-linear lookup tables shared by confidence scoring and hover rendering.
_RHYTHM_DEVIATION_XPOINTS: List[float] = [0.0, 0.15, 0.40, 0.60]
_RHYTHM_SCORE_YPOINTS: List[float]     = [1.0, 0.8,  0.4,  0.0]

_AMPLITUDE_RATIO_XPOINTS: List[float] = [0.0, 0.4, 0.7, 1.0]
_AMPLITUDE_SCORE_YPOINTS: List[float] = [0.0, 0.4, 0.7, 1.0]


class PeakType(Enum):
    """Enumeration for classifying heartbeat peaks."""
    S1_PAIRED = "S1 (Paired)"
    S2_PAIRED = "S2 (Paired)"
    LONE_S1_VALIDATED = "Lone S1"
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

        elif sec_type == "confidence_trace":
            steps = sec.get("steps", [])
            if steps:
                lines.append("- Confidence trace:")
                for s in steps:
                    step_name = s.get("step", "?")
                    detail = s.get("detail", "")
                    result = s.get("result")
                    result_str = f"{result:.2f}" if result is not None else "?"
                    if detail:
                        lines.append(f"    - {step_name}: {detail} → {result_str}")
                    else:
                        lines.append(f"    - {step_name}: → {result_str}")

        elif sec_type == "correction_reason":
            msg = sec.get("text")
            if msg:
                lines.append(f"- Correction: {msg}")

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

    if _is_noise_debug(entry):
        return "Noise"
    if _is_s1_paired_debug(entry) or _is_lone_s1_debug(entry):
        return "S1"
    if PeakType.is_s2(pt):
        return "S2"
    return "Unknown"


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
