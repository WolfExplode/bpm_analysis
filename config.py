# config.py
# I manually tuned the parameters in this file to result in the fewest errors across my input data
# It's important to note that as my codebase evolves and I implement more efficient/better methods. many of these "bandaid fixes"
# will be outdated and the parameter tuning will not need to be so specific. 
# some of these params are too specific which indicates that the code is not as robust as it could be.
# although I have tried to avoid it, adjusting some of the params to reduce errors for one input file may cause another input file to have more errors.
# It feels like I have to balance the params to "fine tune" the code, which feels bad?
# but this may also be unavoidable due to the nature of the data. 


DEFAULT_PARAMS = {
    # =================================================================================
    # 1. General & Preprocessing Settings
    # Controls the initial loading and filtering of the audio.
    # =================================================================================
    "segmenter": "springer",       # "default" = peak-based classifier; "springer" = Springer HSMM segmentation
    "springer_model_path": r"C:\Users\WXP\Documents\GitHub\bpm_analysis\springer_hsmm\springer_model.npz",    # Path to .npz Springer model; required when segmenter is "springer"
    "target_sample_rate": 850,    # Resample audio to this rate (Hz). lower sample rate for optimization reasons
    # only frequencies below target_sample_rate/2 can be represented; higher frequencies are lost or aliased.
    "save_filtered_wav": True,    # If True, saves a .wav file of the filtered audio for debugging.

    # pyPCG denoising, I never managed to get this to work well, so I'm not using it for now. I should probably remove this
    'denoising_method': None,  # None, 'wavelet_auto', 'emd_savgol' emd_savgol is more agressive
    'wt_family': 'db8',
    'wt_level': 7,
    'emd_window': 40,
    'emd_poly': 2,
    'wavelet_threshold': None,            # None for auto, or float (0.0-1.0)
    "enable_hum_removal": True,           # Detect and notch narrow low-frequency hums if present
    "hum_psd_window_sec": 4.0,            # PSD window length (seconds) for hum detection
    "hum_min_freq_hz": 40.0,              # Min frequency of narrow-band hum to consider
    "hum_max_freq_hz": 100.0,             # Max frequency of narrow-band hum to consider
    "hum_min_prominence_db": 8.0,         # Minimum prominence (dB) above local median to trigger notch
    "hum_min_prominence_over_second_db": 3.0,  # Gap over next peak before trusting detection
    "hum_notch_q": 35.0,                  # Q factor, Higher = narrower notch (try 35-40 for sharp hums)

    # =================================================================================
    # 2. Signal Feature Detection
    # Governs the initial identification of peaks and troughs in the audio envelope.
    # =================================================================================
    "bandpass_lowcut_hz": 15,             # Pre-filter bandpass low cutoff (Hz) for S1/S2. Typical PCG range.
    "bandpass_highcut_hz": 250,          # Pre-filter bandpass high cutoff (Hz). Must be < sample_rate/2.
    "envelope_method": "hilbert",    # "hilbert" or "homomorphic" for A/B testing envelope extraction.
    "min_peak_distance_sec": 0.1,        # I Adjusted This✔ Minimum time allowed between any two raw peaks.
    "peak_prominence_quantile": 0.45,    # Min prominence = this quantile of envelope. Higher reduces false peaks.
    "trough_prominence_quantile": 0.1,   # How much a dip must stand out to be considered a 'trough'.
    "homomorphic_cutoff_hz": 25.0,       # Low-pass cutoff (Hz). Higher = less smoothing, sharper peaks (8 = very smooth, 25–40 = sharper).
    "homomorphic_filter_order": 2,       # Butterworth order for homomorphic envelope low-pass.
    "homomorphic_medfilt_kernel": 0,     # Median filter kernel size (0 = off). 3 = light spike removal.

    # =================================================================================
    # 3. Noise Estimation & Rejection
    # Rules for calculating the dynamic noise floor and vetoing noisy peaks.
    # =================================================================================
    # --- 3.1. Dynamic Noise Floor ---
    "noise_floor_quantile": 0.20,        # Quantile of troughs used to calculate the noise floor. (0.2 = 20th percentile).
    "noise_window_sec": 4,               # I Adjusted This✔ Rolling window in seconds. smaller means more sensitive to noise.
    "trough_rejection_multiplier": 10.0, # I Adjusted This✔ A trough N-times higher than the draft noise floor is rejected.
    # I wanted to keep this value high to be conservative

    # --- 3.2. Peak Noise Vetoing ---
    "noise_confidence_threshold": 0.6,  # A peak is rejected if its calculated "noise confidence" exceeds this.
    "trough_veto_multiplier": 2.1,      # Vetoes a small peak if the next peak is N-times larger.
    "trough_noise_multiplier": 3.0,     # Marks a peak as noisy if its preceding trough is N-times the noise floor.
    "strong_peak_override_ratio": 6.0,  # A peak N-times the noise floor will bypass noise-rejection rules.

    # =================================================================================
    # 4. S1/S2 Pairing & Confidence Engine
    # The core logic for identifying S1-S2 pairs based on timing and physiology.
    # =================================================================================
    # --- 4.1. Core Pairing Rules ---
    "pairing_confidence_threshold": 0.50, # Confidence score required to classify two peaks as an S1-S2 pair.
    "s1_s2_interval_cap_sec": 0.4,        # The absolute maximum time (seconds) allowed between S1 and S2.
    "s1_s2_interval_rr_fraction": 0.7,    # The S1-S2 interval cannot be longer than this fraction of the R-R interval.
    'min_s1_s2_interval_sec': 0.10,           # Absolute minimum (100ms)
    'min_s1_s2_interval_rr_fraction': 0.23,   # Or 23% of total RR interval
    'noise_prominence_threshold': 0.35,   # Peaks below this ratio are "suspect noise" 
    'enable_lookahead_skipping': True,    # Enable/disable lookahead skipping

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
    "s2_s1_ratio_low_bpm": 1.6,             # I adjusted this✔ to 1.6 because at low BPM, allows S2 to be up to 1.6x S1 strength before penalty.
    "s2_s1_ratio_high_bpm": 1.2,            # I adjusted this✔ At high BPM, expects S2 to be no more than 1.2x S1 strength.
    # We should test the contractility model with different values for these params to determine the best values for the dataset I have on hand.
    "contractility_bpm_low": 120.0,         # Below this BPM, the 'low BPM' confidence model is used.
    "contractility_bpm_high": 140.0,        # Above this BPM, the 'high BPM' confidence model is used.
    "contractility_penalty_strength": 0.35,  # Strength of the contractility penalty.
    "recovery_phase_duration_sec": 120,     # Duration (seconds) of the high-contractility state after peak BPM.

    # --- 4.4. Interval-Based Confidence Penalty ---
    "interval_penalty_start_factor": 1.0,     # Penalty begins when interval > (max_interval * this value).
    "interval_penalty_full_factor": 1.4,      # Penalty is at max when interval > (max_interval * this value).
    "interval_max_penalty": 0.75,             # Max confidence points to subtract for a long interval.

    # --- 4.5. Kick-Start Mechanism to Recover from Pairing Failure ---
    "kickstart_check_threshold": 0.3,           # Only run the check if pairing_ratio is BELOW this value.
    "kickstart_override_ratio": 0.60,           # The temporary pairing ratio to use if kick-start is triggered.

    # =================================================================================
    # 5. Rhythm Plausibility & Validation
    # Rules for the algorithm's long-term BPM belief and beat-to-beat timing checks.
    # =================================================================================
    # --- 5.1. Long-Term BPM Belief ---
    "min_bpm": 40,                          # Absolute minimum BPM the algorithm will consider valid.
    "max_bpm": 240,                         # Absolute maximum BPM the algorithm will consider valid.

    # --- 5.2. Beat-to-Beat Validation ---
    "rr_interval_max_decrease_pct": 0.45, # A new R-R interval can't be more than 45% shorter than the previous one.
    "rr_interval_max_increase_pct": 0.70, # A new R-R interval can't be more than 70% longer than the previous one.
    "lone_s1_min_strength_ratio": 0.29,   # I Adjusted This✔ A Lone S1 candidate's strength must be at least this fraction of the previous S1's.
    "lone_s1_forward_check_pct": 0.44,    # I Adjusted This✔ A Lone S1 is rejected if the next peak is too close, implying a BPM spike.
    "lone_s1_forward_penalty_factor": 0.52,  # I Adjusted This✔ Multiplier applied when forward check suspects the peak is actually an S2.

    # --- 5.4. Lone S1 Gradient Confidence Engine ---
    "lone_s1_confidence_threshold": 0.50, # Final combined score needed to be accepted as a Lone S1.
    "lone_s1_rhythm_weight": 0.65,         # The weight given to the rhythmic timing score (0.0 to 1.0).
    "lone_s1_amplitude_weight": 0.35,      # The weight given to the amplitude consistency score.

    # =================================================================================
    # 6. Post-Processing Correction Pass
    # Final analysis pass to identify and fix rhythmic discontinuities.
    # =================================================================================
    "enable_correction_pass": False,
    "rr_correction_threshold_pct": 0.40,      # An R-R interval shorter than (Median R-R * this_value) is a "discontinuity".
    "rr_correction_long_interval_pct": 1.70,  # An R-R interval longer than (Median R-R * this_value) is a "gap".
    "penalty_waiver_strength_ratio": 4.0,     # Required signal-to-noise ratio for an S1 to be used in a correction.
    "penalty_waiver_max_s2_s1_ratio": 2.5,    # Safety rail: S2/S1 amp ratio must be below this to allow a correction.

    # =================================================================================
    # 7. Output, HRV & Reporting
    # Controls for final calculations, reports, and plots.
    # =================================================================================
    "output_smoothing_window_sec": 5,        # Time window (seconds) for smoothing the final BPM curve for display.
    "hrv_window_size_beats": 40,             # Sliding window size (in beats) for HRV calculation.
    "hrv_step_size_beats": 5,                # How many beats the HRV window moves in each step.
    "plot_amplitude_scale_factor": 250.0,    # Adjusts the default y-axis range of the signal amplitude plot.
    "plot_downsample_factor": 2,             # The factor for downsampling plot traces (e.g., 5 = keep 1 of every 5 points).

    # --- 7.2. Long Plot Optimization ---
    # When enabled, very long recordings can skip detailed debug traces in the HTML plot
    # to keep file sizes manageable. Shorter recordings are always shown in full detail.
    "optimize_long_plots": False,                # Whether to enable long-plot optimization.
    "long_plot_duration_threshold_sec": 600.0,   # Duration threshold (seconds) to treat a file as "long" (default: 10 minutes).

    # --- 7.1. Trapezoid Artifact Detection ---
    # These control detection of brief, trapezoid-shaped BPM jumps that are often extra-beat artifacts.
    "trapezoid_rate_threshold": 7.0,            # BPM/s: physiologically implausible rate for rise/fall edges.
    "trapezoid_max_edge_duration_sec": 2,     # Maximum duration (seconds) for the rise/fall edges.
    "trapezoid_min_plateau_duration_sec": 1.5,  # Minimum plateau length between rise and fall.
    "trapezoid_max_plateau_duration_sec": 15.0, # Maximum plateau length between rise and fall.
    "trapezoid_baseline_tolerance_bpm": 5.0,    # Allowed BPM difference between pre- and post-artifact baseline.
    "trapezoid_min_jump_bpm": 3.0,              # Minimum BPM jump from baseline to plateau median.
    "trapezoid_min_fall_delta_bpm": 3.0,        # Minimum absolute BPM drop across the fall edge (start vs end of fall).
}