# Cascade Reset and Kick-Start Logic Archive
## 1. Kick-Start Mechanism
**Purpose:** Recovery when the algorithm is stuck in "Lone S1 only" mode. If pairing ratio is low and recent Lone S1s are each followed by a Noise peak, temporarily override the pairing ratio to encourage pairing on the next peak.
**Config (removed from `config.py`):**
```python
# --- 4.5. Kick-Start Mechanism to Recover from Pairing Failure ---
"kickstart_check_threshold": 0.3,           # Only run the check if pairing_ratio is BELOW this value.
"kickstart_override_ratio": 0.60,           # The temporary pairing ratio to use if kick-start is triggered.
"kickstart_history_beats": 4,               # Look-back window: how many recent beats to inspect for the pattern.
"kickstart_min_lone_s1s": 3,                # How many of those beats must be Lone S1 candidates.
"kickstart_min_noise_matches": 3,           # How many of those Lone S1s must be immediately followed by a Noise peak.
```
**Classifier — `_kickstart_check` (removed):**
```python
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
```
**Integration:** The main loop called `kickstart_msg = self._kickstart_check(pairing_ratio)` and passed `kickstart_msg` into `_process_peak_pair`. When non-None, the message was prepended to debug sections (pair, skip-one, lone peak). The `pairing_ratio_override` field on `AnalysisState` was set by kick-start (consumption of the override was not fully wired in the codebase).

---
## 2. Cascade Reset Mechanism
**Purpose:** After several consecutive Lone S1 rhythm rejections, force the current peak to be accepted as a Lone S1 and reset the rejection counter ("cascade reset" safety mechanism).
**Config:** Used `cascade_reset_trigger_count` (default 3). Not in the same config block as kick-start; was read via `params.get("cascade_reset_trigger_count", 3)` in classifier.
**Classifier — cascade branch in `_classify_lone_peak` (removed):**
```python
# Inside _classify_lone_peak, when is_valid is False:
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
```
**State (confidence_engine.AnalysisState):**
- `consecutive_rr_rejections: int = 0` — count of consecutive Lone S1 rhythm rejections (removed).
- `pairing_ratio_override: Optional[float] = None` — set by kick-start (removed).
**PeakType (peak_utils):**
- `LONE_S1_CASCADE = "Lone S1 (Corrected by Cascade Reset)"` (removed).
**Other usages:** `consecutive_rr_rejections` was reset to 0 whenever a peak was paired or accepted as validated Lone S1 (in `_process_peak_pair` and in the validated branch of `_classify_lone_peak`).
---
## 3. Pipeline log filter (pipeline.py)
Substrings that were filtered from verbose logs:
- `"KICK-START:"`
- `"CASCADE RESET:"`
---
## 4. peak_utils — debug section type "kickstart"
In `format_debug_entry`, a section with `sec_type == "kickstart"` was rendered as:
```python
elif sec_type == "kickstart":
    msg = sec.get("text") or sec.get("message")
    if msg:
        lines.append(f"- {msg}")
```