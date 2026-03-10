# audio_preprocessing.py
# Full audio preprocessing pipeline: format conversion, channel splitting, signal conditioning,
# envelope extraction, and noise-floor estimation.
# Consumed by bpm_analysis (main pipeline) and gui (conversion helpers).
import os
import logging
from typing import Dict, Optional, Tuple, List

from config import DEFAULT_OUTPUT_OPTIONS

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, sosfiltfilt, welch, iirnotch, find_peaks, hilbert
import librosa

try:
    from pydub import AudioSegment
except ImportError:
    logging.warning("Pydub library not found. Install with 'pip install pydub'.")
    AudioSegment = None


def _detect_and_remove_stationary_hum(
    audio_data: np.ndarray, sample_rate: int, params: Dict
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Detect a strong, stationary, narrow-band hum and remove it with a notch filter.

    The detection is intentionally conservative so that most recordings (without a
    clear hum) are left untouched.

    The reason I implemented this is to remove low frequency vibration noise, IFYKYK...
    
    Returns
    -------
    filtered_audio : np.ndarray
        The (possibly) hum-filtered signal.
    hum_freq_hz : Optional[float]
        Detected hum frequency in Hz, or None if nothing was removed.
    """
    if audio_data.size == 0:
        return audio_data, None

    if not params.get("enable_hum_removal", True):
        return audio_data, None

    try:
        # Use a relatively long window for a stable PSD estimate
        window_sec = float(params.get("hum_psd_window_sec", 4.0))
        nperseg = int(sample_rate * window_sec)
        nperseg = max(256, min(len(audio_data), nperseg))
        freqs, psd = welch(audio_data, fs=sample_rate, nperseg=nperseg)
    except Exception as e:
        logging.warning(f"Hum detection skipped (PSD computation failed): {e}")
        return audio_data, None

    # Restrict search to a low-frequency band where hums typically live
    fmin = float(params.get("hum_min_freq_hz", 30.0))
    fmax = float(params.get("hum_max_freq_hz", 120.0))
    band_mask = (freqs >= fmin) & (freqs <= fmax)

    if not np.any(band_mask):
        return audio_data, None

    freqs_band = freqs[band_mask]
    psd_band = psd[band_mask]

    if freqs_band.size < 3:
        return audio_data, None

    # Work in dB relative to the median so we look for a clearly dominant peak
    psd_db = 10.0 * np.log10(psd_band + 1e-12)
    median_db = float(np.median(psd_db))
    psd_db_rel = psd_db - median_db

    min_prom_db = float(params.get("hum_min_prominence_db", 10.0))

    try:
        peak_indices, properties = find_peaks(psd_db_rel, prominence=min_prom_db)
    except Exception as e:
        logging.warning(f"Hum detection skipped (peak finding failed): {e}")
        return audio_data, None

    if peak_indices.size == 0:
        logging.info(
            "Hum removal: no strong narrow-band peak detected in %.1f-%.1f Hz.", fmin, fmax
        )
        return audio_data, None

    prominences = properties.get("prominences", None)
    if prominences is None or len(prominences) == 0:
        return audio_data, None

    best_idx_in_peaks = int(np.argmax(prominences))
    best_prom = float(prominences[best_idx_in_peaks])

    # Optional extra check: ensure the strongest peak clearly stands out from the rest
    if len(prominences) > 1:
        # Second-strongest prominence
        second_best = float(np.partition(prominences, -2)[-2])
    else:
        second_best = 0.0

    min_gap_db = float(params.get("hum_min_prominence_over_second_db", 3.0))
    if second_best > 0.0 and (best_prom - second_best) < min_gap_db:
        logging.info(
            "Hum removal: strongest peak not clearly dominant (Δ%.1f dB). Skipping.",
            best_prom - second_best,
        )
        return audio_data, None

    hum_freq_hz = float(freqs_band[peak_indices[best_idx_in_peaks]])

    # Sanity check on frequency
    if hum_freq_hz <= 0.0 or hum_freq_hz >= (sample_rate / 2.0):
        return audio_data, None

    q = float(params.get("hum_notch_q", 30.0))

    try:
        # Normalized frequency (0-1) for iirnotch
        w0 = hum_freq_hz / (sample_rate / 2.0)
        b, a = iirnotch(w0, Q=q)
        filtered = filtfilt(b, a, audio_data)
        logging.info(
            "Hum removal: applied narrow notch at %.2f Hz (Q=%.1f).", hum_freq_hz, q
        )
        return filtered, hum_freq_hz
    except Exception as e:
        logging.warning(
            "Hum removal failed when applying notch at %.2f Hz: %s", hum_freq_hz, e
        )
        return audio_data, None


def apply_bandpass_only(audio: np.ndarray, sample_rate: int, params: Dict) -> np.ndarray:
    """
    Apply only bandpass filtering (no hum removal). Used for FFT profiles so
    preprocessed traces reflect spectral shape within the band of interest.
    Returns filtered audio at the same sample rate.
    """
    if audio.size == 0:
        return audio
    lowcut = float(params.get("preprocess_bandpass_low_hz", 20.0))
    highcut = float(params.get("preprocess_bandpass_high_hz", 220.0))
    order = int(params.get("preprocess_bandpass_order", 2))
    nyquist = 0.5 * sample_rate
    low, high = lowcut / nyquist, highcut / nyquist
    if high >= 1.0:
        return audio
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, audio)


def apply_signal_preprocessing(
    audio: np.ndarray, sample_rate: int, params: Dict
) -> np.ndarray:
    """
    Apply hum removal and bandpass to audio. Used for FFT profiles (preprocessed traces).
    Returns filtered audio at the same sample rate.
    """
    if audio.size == 0:
        return audio
    filtered, _ = _detect_and_remove_stationary_hum(audio, sample_rate, params)
    lowcut = float(params.get("preprocess_bandpass_low_hz", 20.0))
    highcut = float(params.get("preprocess_bandpass_high_hz", 220.0))
    order = int(params.get("preprocess_bandpass_order", 2))
    nyquist = 0.5 * sample_rate
    low, high = lowcut / nyquist, highcut / nyquist
    if high >= 1.0:
        return filtered
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, filtered)


def convert_to_wav(file_path: str, target_path: str) -> bool:
    """Converts a given audio file to WAV format."""
    if not AudioSegment:
        raise ImportError("Pydub/FFmpeg is required for audio conversion.")

    logging.info(f"Converting {os.path.basename(file_path)} to WAV format...")
    try:
        sound = AudioSegment.from_file(file_path)
        # Preserve original channel layout; downstream logic may choose to split channels.
        sound.export(target_path, format="wav")
        return True
    except Exception as e:
        logging.error(f"Could not convert file {file_path}. Error: {e}")
        return False


def split_wav_to_mono_channels(file_path: str, output_directory: str) -> List[str]:
    """
    For a possibly multi-channel WAV file, export one mono WAV per channel.

    Returns a list of file paths to the mono channel WAVs. If the input
    is already mono or splitting fails, the original file_path is returned
    as the only element.
    """
    if not AudioSegment:
        logging.warning("Pydub not available; cannot split channels. Using original file only.")
        return [file_path]

    try:
        sound = AudioSegment.from_file(file_path)
    except Exception as e:
        logging.warning(f"Failed to open WAV for channel splitting ({file_path}): {e}")
        return [file_path]

    if sound.channels <= 1:
        return [file_path]

    mono_segments = sound.split_to_mono()
    base_name = os.path.basename(os.path.splitext(file_path)[0])

    channel_paths: List[str] = []
    for idx, seg in enumerate(mono_segments):
        ch_idx = idx + 1
        out_path = os.path.join(output_directory, f"{base_name}_ch{ch_idx}.wav")
        try:
            seg.export(out_path, format="wav")
            channel_paths.append(out_path)
        except Exception as e:
            logging.warning(f"Failed to export channel {ch_idx} for {file_path}: {e}")

    # Fallback: if export failed for all channels, keep original file
    if not channel_paths:
        return [file_path]

    logging.info(
        "Split %s into %d mono channel file(s): %s",
        os.path.basename(file_path),
        len(channel_paths),
        ", ".join(os.path.basename(p) for p in channel_paths),
    )
    return channel_paths


def _calculate_dynamic_noise_floor(
    audio_envelope: np.ndarray, sample_rate: int, params: Dict
) -> Tuple[pd.Series, np.ndarray]:
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
    draft_noise_floor = draft_noise_floor.bfill().ffill()

    # --- STEP 3: Sanitize the trough list ---
    # Remove any troughs too far above the noise floor.
    sanitized_trough_indices = []
    rejection_multiplier = params.get('trough_rejection_multiplier', 4.0)
    for trough_idx in all_trough_indices:
        trough_amp = audio_envelope[trough_idx]
        floor_at_trough = draft_noise_floor.iloc[trough_idx]
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
        logging.warning("Not enough sanitized troughs remaining. Using non-sanitized floor as fallback.")
        dynamic_noise_floor = draft_noise_floor

    if dynamic_noise_floor.isnull().all():
        fallback_val = np.quantile(audio_envelope, 0.1)
        dynamic_noise_floor = pd.Series(fallback_val, index=np.arange(len(audio_envelope)))

    return dynamic_noise_floor, np.array(sanitized_trough_indices)


def preprocess_audio(
    file_path: str, params: Dict, output_directory: str, output_options: Optional[Dict] = None
) -> Tuple[np.ndarray, int, Optional[Dict[str, np.ndarray]], pd.Series, np.ndarray]:
    if output_options is None:
        output_options = DEFAULT_OUTPUT_OPTIONS.copy()

    save_debug_file = params["save_filtered_wav"] and output_options.get("filtered_wav", True)
    target_sample_rate = int(params.get("preprocess_target_sample_rate", 500))

    try:
        # Preserve historical behavior: simple mono mix of all channels.
        audio_downsampled, new_sample_rate = librosa.load(file_path, sr=target_sample_rate, mono=True)
    except Exception as e:
        logging.error(f"Librosa failed to load file: {e}")
        raise

    # Optional adaptive hum removal (e.g., ~50-70 Hz mains / equipment hum)
    audio_downsampled, detected_hum = _detect_and_remove_stationary_hum(
        audio_downsampled, new_sample_rate, params
    )
    if detected_hum is not None:
        logging.info("Detected and removed stationary hum at ~%.2f Hz.", detected_hum)

    # Bandpass for S1/S2 detection: typical PCG range where first and second heart sounds have most energy.
    lowcut = float(params.get("preprocess_bandpass_low_hz", 20.0))
    highcut = float(params.get("preprocess_bandpass_high_hz", 220.0))
    order = int(params.get("preprocess_bandpass_order", 2))
    nyquist = 0.5 * new_sample_rate
    low, high = lowcut / nyquist, highcut / nyquist

    if high >= 1.0:
        raise ValueError(f"Cannot create a {highcut}Hz filter. The sample rate of {new_sample_rate}Hz is too low.")

    sos = butter(order, [low, high], btype="band", output="sos")
    audio_filtered = sosfiltfilt(sos, audio_downsampled)

    if save_debug_file:
        base_name = os.path.basename(os.path.splitext(file_path)[0])
        debug_path = os.path.join(output_directory, f"{base_name}_filtered_debug.wav")

        # Resample to a browser-friendly sample rate for HTML5 audio playback.
        # Very low sample rates (e.g. 500 Hz) can cause some browsers to report
        # "Audio format not supported", even though the WAV file is valid.
        # Increased debug_sample_rate to 10k to avoid Empty filters detected in mel frequency basis warnings.
        debug_sample_rate = 10000
        try:
            peak = float(np.max(np.abs(audio_filtered))) if audio_filtered.size else 0.0
            if peak > 0:
                norm = audio_filtered / peak
            else:
                norm = audio_filtered

            # Upsample for playback while preserving duration
            debug_audio = librosa.resample(
                norm, orig_sr=new_sample_rate, target_sr=debug_sample_rate
            )
            normalized_audio = np.int16(
                np.clip(debug_audio, -1.0, 1.0) * 32767
            )
            wavfile.write(debug_path, debug_sample_rate, normalized_audio)
            logging.info(
                "Saved filtered audio WAV debug file (%s, %d Hz, int16) for HTML playback.",
                debug_path,
                debug_sample_rate,
            )
        except Exception as e:
            logging.error(f"Failed to write filtered debug WAV file {debug_path}: {e}")
    elif params["save_filtered_wav"] and not output_options.get("filtered_wav", True):
        logging.info("Skipping filtered audio WAV generation as requested.")

    # Hilbert envelope: magnitude of analytic signal for a sharper, more symmetric envelope
    # than abs + rolling mean, which helps peak timing stability (e.g. for HRV).
    analytic = hilbert(audio_filtered)
    envelope_raw = np.abs(analytic).astype(np.float64)
    # Smoothing to reduce ripple (e.g. between S1 and S2); window in ms from config (default 50 ms).
    smooth_ms = params.get("envelope_smooth_window_ms", 50)
    smooth_window = max(1, int(smooth_ms * new_sample_rate / 1000))
    audio_envelope = pd.Series(envelope_raw).rolling(
        window=smooth_window, min_periods=1, center=True
    ).mean().values

    noise_floor, trough_indices = _calculate_dynamic_noise_floor(audio_envelope, new_sample_rate, params)
    return audio_envelope, new_sample_rate, noise_floor, trough_indices
