# config.py
# Default parameters and output toggles for the analysis pipeline.
# Values are tuned for typical PCG recordings from consumer hardware.
# See Documentation.md "Parameter Tuning Rationale" for reasoning behind specific values.

import logging

DEFAULT_PARAMS = {
    # =================================================================================
    # 1. General & Preprocessing Settings
    # Controls the initial loading and filtering of the audio.
    # =================================================================================
    "downsample_factor": 300,     # Factor to reduce sample rate. Higher = faster processing, less detail.
    "save_filtered_wav": True,    # If True, saves *_filtered_debug.wav and *_filtered_inverse_debug.wav when output_options.filtered_wav is True.

    # Main preprocessing: target sample rate and bandpass (single wide band before envelope); typical PCG range for S1/S2.
    "preprocess_target_sample_rate": 600,   # Resample to this Hz for analysis; lower = faster, less detail.
    "preprocess_bandpass_low_hz": 30.0,     # increasing this reduces the amplitude of S1.
    "preprocess_bandpass_high_hz": 290.0,
    "preprocess_bandpass_order": 2,       # Butterworth order; higher order not yet validated for this pipeline.

    "enable_hum_removal": True,           # Detect and notch narrow low-frequency hums if present
    "hum_psd_window_sec": 4.0,            # PSD window length (seconds) for hum detection
    "hum_min_freq_hz": 40.0,              # Min frequency of narrow-band hum to consider
    "hum_max_freq_hz": 100.0,             # Max frequency of narrow-band hum to consider
    "hum_min_prominence_db": 8.0,         # Minimum prominence (dB) above local median to trigger notch
    "hum_min_prominence_over_second_db": 3.0,  # Gap over next peak before trusting detection
    "hum_notch_q": 35.0,                  # Q factor, Higher = narrower notch (try 35-40 for sharp hums)
    # Skip hum detection/notch when duration exceeds this (minutes). Saves cost on very long files; set None to always attempt hum removal when enabled.
    "hum_removal_skip_if_longer_than_min": 30.0,

    "envelope_smooth_window_ms": 40,      # Rolling window (ms) for smoothing Hilbert envelope after abs(analytic). Matches common PCG practice (e.g. 50 ms).

    # Inverse-band (HF noise envelope): load + FIR taper + Hilbert at this Hz, then resample to preprocess_target_sample_rate.
    # Much faster / lower RAM than full native (~44.1 kHz) on long files. Must exceed ~2× the HF taper top; clamped in code.
    # Set to None or 0 to load at the file's native sample rate (slowest, highest CPU/RAM).
    "inverse_band_working_sample_rate": 4000,
    # FIR magnitude taper for the HF-only path: stopband below low_hz, ramp low_hz→high_hz, passband above high_hz (linear firls).
    "inverse_band_taper_low_hz": 300.0,
    "inverse_band_taper_high_hz": 600.0,

    # HF noise strip (inverse-band envelope): gate + merge / min-duration / pad.
    # Higher noise_segment_gate_quantile or noise_segment_gate_min_amplitude → fewer samples exceed the gate → fewer noisy regions (noise strip + pass3_noise_unreliable_windows when pass3_calculate_noisy_regions).
    "noise_segment_gate_quantile": 0.90,    # File-local quantile of HF envelope; only at/above this level can count as noisy (was 0.85 hardcoded).
    "noise_segment_gate_min_amplitude": 0.03,  # Absolute floor on envelope; raise with quantile to be stricter (was 0.02 hardcoded).
    "noise_segment_merge_gap_ms": 500.0,    # Merge if gap < this before expand_ms and again after (expanded time).
    "noise_segment_min_duration_ms": 20.0,  # Drop shorter noisy blips after merge.
    "noise_segment_expand_ms": 100.0,       # Pad each segment start/end by this (quantile gate tends to clip early/late).
    # If merged+expanded HF-noise intervals cover more than this fraction of the file, discard all segments (unreliable gate: clipping, overload, low Nyquist).
    "noise_segment_max_coverage": 0.40,

    # =================================================================================
    # 2. Signal Feature Detection
    # Governs the initial identification of peaks and troughs in the audio envelope.
    # =================================================================================
    "min_peak_distance_sec": 0.1,        # I Adjusted This✔ Minimum time allowed between any two raw peaks.
    "peak_prominence_quantile": 0.50,    # Min prominence = this quantile of envelope. Higher reduces false peaks (e.g. Hilbert ripple).
    "trough_prominence_quantile": 0.3,   # How much a dip must stand out to be considered a 'trough'.

    # Peak position refinement: shift each raw peak to super-Gaussian-weighted center-of-mass (~100 ms window).
    "peak_refine_window_ms": 150,        # Window (ms) around each peak for CoM; ~100 ms covers typical S1 extent.
    "peak_refine_max_shift_ms": 40,      # Cap shift so noisy envelope cannot pull peak more than this (ms).
    "peak_refine_super_gaussian_n": 5,   # Super-Gaussian exponent (flat top); higher = more flat top.

    # =================================================================================
    # 3. Noise Estimation & Rejection
    # Rules for calculating the dynamic noise floor and vetoing noisy peaks.
    # =================================================================================
    # --- 3.1. Dynamic Noise Floor ---
    "noise_floor_quantile": 0.25,        # Quantile of troughs used to calculate the noise floor. lower = more sensitive to noise.
    "noise_window_sec": 2.5,               # I Adjusted This✔ Rolling window in seconds. smaller means more sensitive to noise.
    "trough_rejection_multiplier": 6.0, # I Adjusted This✔ A trough N-times higher than the draft noise floor is rejected.
    # I wanted to keep this value high to be conservative

    # --- 3.2. Peak Noise Vetoing ---
    "noise_confidence_threshold": 0.6,  # A peak is rejected if its calculated "noise confidence" exceeds this.
    "trough_veto_multiplier": 2.1,      # Vetoes a small peak if the next peak is N-times larger.
    "trough_noise_multiplier": 3.0,     # Marks a peak as noisy if its preceding trough is N-times the noise floor.
    "strong_peak_override_ratio": 6.0,  # A peak N-times the noise floor will bypass noise-rejection rules.

    # =================================================================================
    # 4. S1/S2 Pairing & Confidence Engine
    # The core logic for identifying S1→S2 (systole) pairs based on timing and physiology.
    # =================================================================================
    # --- 4.1. Core Pairing Rules ---
    "pairing_confidence_threshold": 0.50,      # Confidence score required to classify two peaks as an S1→S2 (systole) pair.
    "pass1_pairing_confidence_threshold": 0.7, # Pass 1 only: min S1→S2 (systole) pairing confidence for anchor beats (overrides pairing_confidence_threshold for that run).
    "s1_s2_interval_cap_sec": 0.4,             # The absolute maximum time (seconds) allowed between S1 and S2.
    "min_s1_s2_interval_sec": 0.10,            # Absolute minimum (100ms)
    "min_s1_s2_interval_rr_fraction": 0.23,    # Or 23% of total RR interval
    # Diastole (S2→next S1) plausibility bounds — used by calculate_bpm_intervals and Pass 3 correction.
    "min_diastole_nominal_frac": 0.35,         # Diastole can be this fraction of its nominal before flagged as too short. Lower → more tolerant of compressed diastoles (conservative). Higher → flag earlier (aggressive).
    "max_diastole_nominal_frac": 2.0,          # Diastole longer than this × nominal → considered a gap (used by dropout detection). Lower → flag gaps earlier.
    "min_diastole_sec": 0.08,                  # Absolute floor (seconds) for diastole regardless of BPM. Prevents near-zero diastoles at very high heart rates.
    # S1 and S2 acoustic event duration bounds (BPM-independent physiological constants).
    # These define how long the audible heart sound event itself lasts, not the intervals between sounds.
    # Used by Pass 3 to gate corrections that would squeeze a state into an impossibly short window.
    "s1_min_sec": 0.030,     # Shortest plausible S1 sound duration (10ms absolute floor).
    "s1_nominal_sec": 0.080, # Typical S1 sound duration (~80ms).
    "s1_max_sec": 0.120,     # Longest plausible S1 sound (beyond this it blurs into systole).
    "s2_min_sec": 0.030,     # Shortest plausible S2 sound duration (10ms absolute floor).
    "s2_nominal_sec": 0.080, # S2 is generally shorter and softer than S1 (~80ms typical).
    "s2_max_sec": 0.120,     # Longest plausible S2 sound.
    # BPM-dependent expected systole (S1→S2) (Weissler: https://www.desmos.com/calculator/ebqshptip0)
    "s1_s2_expected_weissler_ref_et_ms": 320, # Reference ejection time (ms) at ref_bpm.
    "s1_s2_expected_weissler_ref_bpm": 60,    # BPM at which ref_et_ms is defined.
    "s1_s2_expected_weissler_slope_ms_per_bpm": 1.26,  # ET decrease (ms) per BPM.
    "noise_prominence_threshold": 0.35,   # Peaks below this ratio are "suspect noise"
    "enable_lookahead_skipping": True,    # Enable/disable lookahead skipping

    # --- 4.2. Amplitude-Based Confidence Model ---
    "deviation_smoothing_factor": 0.05,   # Smoothing applied to the peak-to-peak amplitude deviation series.

    # --- 4.3. Physiology-Based Confidence Adjustment ---
    "stability_history_window": 20,         # Number of recent beats used to determine rhythm stability.
    "stability_confidence_floor": 0.7,      # I Adjusted This✔ At 0% pairing success, confidence is multiplied by this.
    "stability_confidence_ceiling": 1.3,    # I Adjusted This✔ At 100% pairing success, confidence is multiplied by this (e.g., a 10% boost).
    "recovery_phase_stability_floor": 0.90,  # Disable stability penalty during recovery (0% pairing → factor = 1.0)
    "s1_s2_boost_ratio": 1.2,               # S1 strength must be > (S2 strength * this value) to get a confidence boost.
    "boost_amount_min": 0.10,               # Additive confidence boost for a "good" pair in an unstable section.
    "boost_amount_max": 0.35,               # Additive confidence boost for a "good" pair in a stable section.
    "penalty_amount_min": 0.10,             # Subtractive confidence penalty for a "bad" pair in a stable section.
    "penalty_amount_max": 0.30,             # Subtractive confidence penalty for a "bad" pair in an unstable section.
    "forward_look_drop_threshold": 0.4,     # If next peak < 60% of S2, it's suspicious
    "forward_look_max_penalty": 0.4,        # Max penalty for this scenario
    "pairing_rr_penalty_max": 0.25,         # Multiplicative penalty for RR mismatch vs 60/BPM when evaluating an S1→S2 (systole) pair.
    # Contractility: S1/S2 prominence ratio. Expected from history (past N pairs) or BPM power-curve fallback.
    "contractility_expected_use_history": True,   # If True, expected S1/S2 = mean of last N accepted pairs; else BPM power curve.
    "contractility_expected_history_count": 8,   # Number of past S1/S2 ratios to average.
    "contractility_expected_history_min": 1,      # Min history length before using average (else BPM fallback).
    "contractility_pair_rate_window_sec": 5.0,    # Pair rate in this window blends history vs BPM: 100% pairs → use history; lower → blend toward BPM.
    # BPM fallback: power curve expected_ratio = low + (high - low) * ((BPM - bpm_min) / (bpm_max - bpm_min)) ** exponent.
    "contractility_bpm_min": 60,                 # BPM at which ratio = low_ratio.
    "contractility_bpm_max": 200,                # BPM at which ratio = high_ratio.
    "contractility_low_ratio": 0.9,              # Expected S1/S2 at bpm_min (rest).
    "contractility_high_ratio": 3.5,             # Expected S1/S2 at bpm_max (high exertion).
    "contractility_power_exponent": 0.6,         # <1: steep rise at low BPM then flatter (contractility kicks in early).
    # Asymmetric deviation-based curve: L2=(1-r_low), L1=(1-a_low), R1=(1+a_high), R2=(1+r_high) × expected.
    "contractility_zero_crossing_low": 0.3,       # Left zero-crossing: L1 = expected × (1 - this).
    "contractility_zero_crossing_high": 0.4,      # Right zero-crossing: R1 = expected × (1 + this).
    "contractility_penalty_ramp_fraction_low": 1.3,   # Left ramp end: L2 = expected × (1 - this); penalty max at L2.
    "contractility_penalty_ramp_fraction_high": 2.5,  # Right ramp end: R2 = expected × (1 + this); penalty max at R2.
    "contractility_boost_max": 0.2,              # Max multiplicative boost at expected: confidence *= (1 + boost).
    "contractility_penalty_max": 0.5,             # Max multiplicative penalty when far outside band.
    "recovery_phase_duration_sec": 120,      # Duration (seconds) of the high-contractility state after peak BPM.
    "recovery_phase_min_peak_bpm": 110,      # Only enable recovery-phase adjust if pass 1 peak BPM >= this (avoids activating when BPM stays low).

    # --- 4.4. V-Shaped Interval: boost near expected, penalty outside ---
    # Linear boost from 0 at expected±zero_crossing to max at expected; linear penalty outside that band.
    "interval_v_penalty_max": 0.75,              # Max penalty (multiplicative) at ramp ends.
    "interval_v_boost_max": 0.6,                # Max boost at expected: confidence *= (1 + boost). 0 at zero-crossing boundaries.
    "interval_zero_crossing_fraction": 0.2,      # Fraction of expected where effect crosses zero: boost zone [expected*(1±this)].
    "interval_v_short_ramp_end_fraction": 0.4,  # Left: below this fraction of expected → hard reject; ramp from here up to left zero-crossing.
    "interval_v_long_ramp_end_fraction": 2.0,   # Right: ramp from right zero-crossing to this × expected → full penalty.
    "interval_v_long_reject_fraction": 2.5,     # Right: above this × expected → hard reject.
    # Expected systole (S1→S2) from past pairs (when enabled, overrides BPM-based expected for the V-shape)
    "s1_s2_expected_use_history": True,         # If True, expected = mean of last N accepted systole (S1→S2) intervals; else BPM-based.
    "s1_s2_expected_history_count": 10,        # Number of past systole (S1→S2) intervals to average.
    "s1_s2_expected_history_min": 1,           # Minimum history length before using average (else fallback to BPM).

    # =================================================================================
    # 5. Rhythm Plausibility & Validation
    # Rules for the algorithm's long-term BPM belief and beat-to-beat timing checks.
    # =================================================================================
    # --- 5.1. Long-Term BPM Belief ---
    "min_bpm": 40,                          # Absolute minimum BPM the algorithm will consider valid.
    "max_bpm": 240,                         # Absolute maximum BPM the algorithm will consider valid.
    "bpm_belief_learning_rate": 0.05,       # EMA weight for each new beat; lower = smoother but slower to track changes.
    "bpm_belief_max_change_per_beat": 3.0,  # Speed limiter: max BPM shift allowed per beat (scaled by interval length).

    # --- 5.2. Beat-to-Beat Validation ---
    "rr_interval_max_decrease_pct": 0.45, # A new R-R interval can't be more than 45% shorter than the previous one.
    "rr_interval_max_increase_pct": 0.70, # A new R-R interval can't be more than 70% longer than the previous one.
    "lone_s1_min_strength_ratio": 0.29,   # I Adjusted This✔ A Lone S1 candidate's strength must be at least this fraction of the previous S1's.
    "lone_s1_forward_check_pct": 0.44,    # I Adjusted This✔ A Lone S1 is rejected if the next peak is too close, implying a BPM spike.
    "lone_s1_forward_penalty_factor": 0.52,  # I Adjusted This✔ Multiplier applied when forward check suspects the peak is actually an S2.

    # --- 5.3. Lone S1 Gradient Confidence Engine ---
    "lone_s1_confidence_threshold": 0.50, # Final combined score needed to be accepted as a Lone S1.
    "lone_s1_rhythm_weight": 0.65,         # The weight given to the rhythmic timing score (0.0 to 1.0).
    "lone_s1_amplitude_weight": 0.35,      # The weight given to the amplitude consistency score.
    # k consecutive noise raw peaks before current + span ≈ (k+1)×RR → score span/(k+1) vs RR.
    "lone_s1_missed_beat_tolerance_frac": 0.22,  # |span − m×RR| / (m×RR) must be ≤ this (m = k+1).
    "lone_s1_forward_s1_vs_s2_min_ratio": 1.69,  # Forward check: current peak must be at least this × the next peak's amplitude to be treated as S1 rather than S2. Tuned empirically.

    # =================================================================================
    # 6. Pass 3 — Dense state timeline from Pass 2 (spectral S2 / Pass A–C / emissions removed; see pass3 archived logic.md)
    # =================================================================================

    # --- 6.1 Noise repair (global modifier after the initial timeline exists) ---
    "pass3_calculate_noisy_regions": True,  # HF strip: prefer pass3 windows from noise_event_segments. Sensitivity is noise_segment_gate_quantile / noise_segment_gate_min_amplitude (not this flag alone).
    # If enabled and HF-noise windows exist: clear state labels inside those spans and rebuild the full S1→systole→S2→diastole sequence.
    "pass3_enable_noise_repair": True,

    # --- 6.2 Insert missing states in large gaps (state-level) ---
    "pass3_calculate_large_gaps": True, # Just for display, you need to also enable pass3_enable_gap_state_insert or pass3_enable_noise_repair
    "pass3_enable_gap_state_insert": True,  # If True, long single-state spans can be cleared and rebuilt like noise repair (full-segment candidate + quiet trim).

    # --- 6.2.1 Large-gap peak recovery + anchor snapping ---
    # Reruns a more sensitive peak detector inside Pass 3 large-gap windows, then shifts
    # rebuilt S1/S2 segment boundaries to align with those recovered peaks (fill first, then shift).
    "pass3_enable_gap_snap_to_peaks": False,
    "pass3_gap_recovery_peak_prominence_quantile_insensitive": 0.75,  # Higher = fewer peaks, more likely real S1/S2.
    "pass3_gap_recovery_peak_prominence_quantile_sensitive": 0.5,    # Lower = more peaks; used as an "anything at all here?" scan.
    "pass3_gap_recovery_height_scale": 0.85,              # Multiply dynamic noise-floor threshold (if available).
    "pass3_gap_snap_window_ms": 100.0,                     # Search radius (ms) around each synthetic S1/S2 center when snapping to a recovered peak.

    # --- 6.2.2 Large diastole — Pass 2-style peak labeling then state fill (before noise / gap insert) ---
    # When a diastole segment exceeds the threshold (wall time), mark it unknown, run the
    # insensitive gap peak detector, classify those peaks like Pass 2 (S1/S2/noise), then paint states.
    "pass3_enable_peaks_labeling_in_large_gaps": True,
    "pass3_peaks_labeling_in_large_gaps_min_sec": 6.0,

    # --- 6.3 Final state timeline — envelope boundary paint ---
    "pass3_state_s1_window_ms": 120.0,          # Ceiling (ms) on how far transient edge detection may extend around each S1 peak.
    "pass3_state_s2_window_ms": 120.0,          # Same for S2.
    "pass3_state_edge_alpha": 0.03,             # Transient edge threshold as a fraction of weighted peak height (lower → wider S1/S2 regions).
    "pass3_state_edge_n_exp": 4.0,              # Super-Gaussian exponent for edge weighting (higher → harder cap at window edge). Keep ≤ peak_refine_super_gaussian_n.

    # =================================================================================
    # 7. Output, HRV & Reporting
    # Controls for final calculations, reports, and plots
    # =================================================================================
    "output_smoothing_window_sec": 3,        # Time window (seconds) for smoothing the final BPM curve for display (Gaussian σ ≈ window/3; lower = less smoothing).
    "hrv_window_size_beats": 40,             # Sliding window size (in beats) for HRV calculation.
    "hrv_step_size_beats": 5,                # How many beats the HRV window moves in each step.
    "enable_hrv_frequency_domain": True,     # If True, compute Lomb-Scargle LF/HF and optional global VLF/LF/HF.
    "hrv_global_min_duration_sec": 300.0,    # Only compute global spectrum when recording duration >= this (5 min).
    "plot_amplitude_scale_factor": 250.0,    # Adjusts the default y-axis range of the signal amplitude plot.
    # In plotting.py: avoid dashed lines (dash=...) for line traces--they cause noticeable lag.
    "plot_downsample_factor": 4,            # Downsample only large traces: Bandpass / Noise Removed / Noise Envelope, Dynamic Noise Floor (keep 1 of every N points). Does NOT apply to Average S1/S2 contractility, BPM, HRV, or markers.
    "pass1_bpm_outlier_window_sec": 10.0,   # Half-window (seconds) for pass 1 BPM outlier removal: keep point if within median ± k*MAD in [t-window, t+window].
    "pass1_bpm_outlier_mad_k": 2.5,         # Number of MADs. Lower = more aggressive outlier removal.
    "pass1_bpm_global_outlier_mad_k": 5.0,    # After local pass: global median ± k*MAD. Higher = less sensitive. Set <= 0 to disable this pass.
    "pass2_instant_bpm_outlier_window_sec": 8.0,  # Half-window (seconds) for pass 2/3 instantaneous BPM: local MAD outlier removal.
    "pass2_instant_bpm_outlier_mad_k": 8,       # Local MADs for pass 2/3 instant BPM. Lower = more aggressive.
    "pass1_bpm_gaussian_frac": 0.02,        # Used to derive Gaussian smoothing sigma for pass 1 BPM curve (smaller = tighter smoothing).
    "s1_s2_outlier_window_sec": 10.0,         # Half-window (seconds) for systole (S1→S2) MAD outlier removal. Increase → less aggressive (more global context). Decrease → more aggressive (more local).
    "s1_s2_outlier_mad_k": 2,              # MAD threshold multiplier for systole (S1→S2) outlier removal. Increase → less aggressive (keeps more points). Decrease → more aggressive (flags more as outliers).
    "s1_s2_global_outlier_mad_k": 5.0,     # After local pass: global median ± k*MAD on interval (s). Higher = less sensitive. <=0 disables.
    # Hard clamps (seconds) on measured interval durations before MAD/outliers — very wide defaults = conservative (only insane values dropped).
    "systole_duration_clamp_min_sec": 0.02,   # S1→S2 / systole segment duration floor (below → dropped).
    "systole_duration_clamp_max_sec": 3.0,    # Ceiling; above → dropped before smoothing/outliers.
    "diastole_duration_clamp_min_sec": 0.02,  # S2→next S1 / diastole segment floor.
    "diastole_duration_clamp_max_sec": 3.0,  # Ceiling for absurd gaps/mislabels.
    "systole_gaussian_frac": 0.05,          # Used to derive Gaussian smoothing sigma for systole curve (smaller = tighter smoothing).
    "contractility_average_window_sec": 1.0, # Time to average S1/S2 contractility plot: Used in: long-term (contractility vs BPM), short-term (S1 vs inhale/exhale)

    # --- 7.1. Long Plot Optimization ---
    # When enabled, very long recordings can skip detailed debug traces in the HTML plot
    # to keep file sizes manageable. Shorter recordings are always shown in full detail.
    "optimize_long_plots": False,                 # Long recordings: lighter traces (envelope/peaks debug, S1/S2/noise scores, systole/diastole overlay series). Does not override the >60 min plot-output cutoff in pipeline.
    "long_plot_duration_threshold_sec": 600.0,   # Duration threshold (seconds) to treat a file as "long" (default: 10 minutes).

    # --- 7.1.1. FFT Profiles (S1/S2 frequency spectra from raw and preprocessed audio) ---
    "enable_fft_profiles": True,                 # If True, generate separate HTML with S1/S2 FFT profiles.
    "fft_window_ms": 120.0,                      # Time window (ms) centered on each peak for FFT.
    "fft_max_peaks_per_type": 200,               # Max S1 and S2 peaks (each) for FFT; selected by highest pairing confidence. Lone S1s excluded.
    "fft_aggregate_sr": 32000,                   # Sample rate for multi-file FFT aggregation (common grid). Per-file uses native sr.
    "fft_neutral_band_low_hz": 10000.0,           # Neutral band low (Hz) for S2→S1 alignment (force same level in this band).
    "fft_neutral_band_high_hz": 14000.0,          # Neutral band high (Hz) for S2→S1 alignment.
    "fft_separation_low_hz": 10.0,                # Low bound (Hz) for S1 vs S2 frequency separation vector (algorithm use).
    "fft_separation_high_hz": 15000.0,            # High bound (Hz) for S1 vs S2 frequency separation vector.

    # =================================================================================
    # 8. Pass 4 — Viterbi Holistic Decoder  (pass3_emissions generation removed; restore from pass3 archived logic.md if needed)
    # =================================================================================
    "enable_pass4": False,                          # Guard: off until implementation matures. Set True to run Viterbi after Pass 3.
    "pass4_transition_self_loop_weight": 0.85,      # Higher → decoder prefers longer state durations (more inertia). Lower → allows faster state transitions.
    "pass4_emission_weight": 0.7,                   # Balance between spectral emissions (1.0) and BPM-prior only (0.0). Tune when emission quality is uncertain.

}

# Single source of truth for pipeline output toggles. GUI and analyze_wav_file use this;
# add new options here only (GUI builds checkboxes and get_output_options from these keys).
DEFAULT_OUTPUT_OPTIONS = {
    "html": True,
    # When False (default), generated HTML opens with S1/S2 beat hover tooltips off; user can enable via toolbar.
    "html_s1_s2_hover_on_by_default": False,
    # When True, embed a small script in the HTML file instead of copying interactive_plot.js (no audio/spectrogram/label JS).
    "html_inline_interactive_script": False,
    "png": False,
    "csv": False,
    "summary": False,
    "debug": True,
    "filtered_wav": False,
    # When True, converted/copied/split working WAVs are written under the output folder; when False, a temp dir is used.
    "working_wav_in_output": False,
    "spectrogram": False,
    "fft_profiles": True,
    "regression_log": False,
}

# Keys already warned about — each unknown key is logged once per process, not once per file.
_warned_unknown_param_keys: set = set()

def validate_params(params: dict) -> None:
    """Warn once per process for any key in params not present in DEFAULT_PARAMS.

    Catches typos and stale keys that would otherwise silently fall back to
    their hardcoded defaults in params.get("key", default) calls.
    Call this at the start of analyze_wav_file.
    """
    known = set(DEFAULT_PARAMS)
    new_unknown = set(params) - known - _warned_unknown_param_keys
    for key in sorted(new_unknown):
        logging.warning("Unknown param key %r — not in DEFAULT_PARAMS (typo or stale key?)", key)
    _warned_unknown_param_keys.update(new_unknown)