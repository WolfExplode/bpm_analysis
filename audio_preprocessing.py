# audio_preprocessing.py
# Full audio preprocessing pipeline: format conversion, channel splitting, signal conditioning,
# envelope extraction, and noise-floor estimation.
# Consumed by bpm_analysis (main pipeline) and gui (conversion helpers).
import os
import logging
from typing import Dict, Optional, Tuple, List

from config import DEFAULT_OUTPUT_OPTIONS, output_stem_from_path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, firls, sosfiltfilt, welch, iirnotch, find_peaks, hilbert
import librosa

try:
    from pydub import AudioSegment
except ImportError:
    logging.warning("Pydub library not found. Install with 'pip install pydub'.")
    AudioSegment = None

# Noise-envelope path only: linear magnitude taper (FIR) — full stop below low edge, ramp to high edge, pass above.
_NOISE_ENVELOPE_TAPER_LOW_HZ = 300.0
_NOISE_ENVELOPE_TAPER_HIGH_HZ = 600.0


def _write_peak_normalized_debug_wav(
    out_path: str,
    signal: np.ndarray,
    orig_sr: int,
    target_sr: int = 10000,
) -> None:
    """Peak-normalize, resample for HTML playback, write int16 mono WAV."""
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    norm = signal / peak if peak > 0 else signal
    debug_audio = librosa.resample(norm, orig_sr=orig_sr, target_sr=target_sr)
    normalized_audio = np.int16(np.clip(debug_audio, -1.0, 1.0) * 32767)
    wavfile.write(out_path, target_sr, normalized_audio)


def _write_peak_normalized_wav_native_rate(out_path: str, signal: np.ndarray, sr: int) -> None:
    """Peak-normalize and write int16 mono WAV at the given sample rate (no resampling)."""
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    norm = signal / peak if peak > 0 else signal
    normalized_audio = np.int16(np.clip(norm, -1.0, 1.0) * 32767)
    wavfile.write(out_path, int(sr), normalized_audio)


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
    base_name = output_stem_from_path(file_path)

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


def _dense_troughs_linear_interpolate(
    n: int, trough_indices: np.ndarray, trough_amplitudes: np.ndarray
) -> np.ndarray:
    """
    Match pandas: sparse troughs on a RangeIndex, then linear interpolate (including
    flat extension past the last trough, same as Series.reindex(...).interpolate()).
    """
    dense = np.full(n, np.nan, dtype=np.float64)
    dense[np.asarray(trough_indices, dtype=np.intp)] = np.asarray(
        trough_amplitudes, dtype=np.float64
    )
    return pd.Series(dense).interpolate(method="linear").to_numpy()


def _rolling_quantile_center_bfill_ffill(
    y: np.ndarray, window: int, quantile_val: float, min_periods: int = 3
) -> np.ndarray:
    """Same as Series.rolling(center=True).quantile().bfill().ffill() on contiguous data."""
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    s = pd.Series(y, copy=False)
    rolled = s.rolling(window=window, min_periods=min_periods, center=True).quantile(quantile_val)
    return rolled.bfill().ffill().to_numpy()


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

    n = len(audio_envelope)
    noise_window_samples = int(params['noise_window_sec'] * sample_rate)
    quantile_val = params['noise_floor_quantile']

    # --- STEP 2: Draft noise floor from ALL troughs (dense interpolate + rolling quantile) ---
    dense_troughs_draft = _dense_troughs_linear_interpolate(
        n, all_trough_indices, audio_envelope[all_trough_indices]
    )
    draft_floor_arr = _rolling_quantile_center_bfill_ffill(
        dense_troughs_draft, noise_window_samples, quantile_val, min_periods=3
    )

    # --- STEP 3: Sanitize troughs (vectorized; same rule as per-index loop + pd.isna check) ---
    rejection_multiplier = params.get('trough_rejection_multiplier', 4.0)
    floor_at = draft_floor_arr[all_trough_indices]
    trough_amps = audio_envelope[all_trough_indices]
    keep = np.isfinite(floor_at) & (trough_amps <= rejection_multiplier * floor_at)
    sanitized_trough_indices = all_trough_indices[keep]

    logging.info(
        f"Trough Sanitization: Kept {len(sanitized_trough_indices)} of {len(all_trough_indices)} initial troughs."
    )

    # --- STEP 4: Final noise floor from sanitized troughs ---
    if len(sanitized_trough_indices) > 2:
        dense_troughs_final = _dense_troughs_linear_interpolate(
            n, sanitized_trough_indices, audio_envelope[sanitized_trough_indices]
        )
        final_floor_arr = _rolling_quantile_center_bfill_ffill(
            dense_troughs_final, noise_window_samples, quantile_val, min_periods=3
        )
        dynamic_noise_floor = pd.Series(final_floor_arr, index=np.arange(n))
    else:
        logging.warning("Not enough sanitized troughs remaining. Using non-sanitized floor as fallback.")
        dynamic_noise_floor = pd.Series(draft_floor_arr, index=np.arange(n))

    if dynamic_noise_floor.isnull().all():
        fallback_val = np.quantile(audio_envelope, 0.1)
        dynamic_noise_floor = pd.Series(fallback_val, index=np.arange(len(audio_envelope)))

    return dynamic_noise_floor, np.asarray(sanitized_trough_indices, dtype=np.intp)


def preprocess_audio(
    file_path: str, params: Dict, output_directory: str, output_options: Optional[Dict] = None
) -> Tuple[np.ndarray, int, pd.Series, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
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

    # Out-of-band path: native sample rate so high-pass content is not capped by analysis Nyquist.
    inverse_band_envelope: Optional[np.ndarray] = None
    audio_inverse_hp_native: Optional[np.ndarray] = None
    native_sr_int: Optional[int] = None
    # Noise envelope FIR taper band (for logging / debug); not tied to bandpass highcut.
    noise_envelope_hp_cut_hz: Optional[float] = None
    smooth_ms = float(params.get("envelope_smooth_window_ms", 50))

    if highcut <= 0.0:
        logging.warning("Inverse-band (noise envelope) skipped: preprocess_bandpass_high_hz must be > 0.")
    else:
        try:
            audio_native, native_sr = librosa.load(file_path, sr=None, mono=True)
            native_sr_int = int(round(float(native_sr)))
            audio_native = np.asarray(audio_native, dtype=np.float64)
        except Exception as e:
            logging.warning("Could not load native-rate audio for inverse-band path: %s", e)
            audio_native = np.array([], dtype=np.float64)
            native_sr_int = None

        if native_sr_int is not None and audio_native.size > 0:
            nyquist_native = 0.5 * float(native_sr_int)
            taper_lo = float(_NOISE_ENVELOPE_TAPER_LOW_HZ)
            taper_hi = float(_NOISE_ENVELOPE_TAPER_HIGH_HZ)
            if taper_hi >= nyquist_native:
                taper_hi = max(taper_lo + 20.0, nyquist_native * 0.999)
            if taper_lo >= taper_hi - 5.0:
                logging.warning(
                    "Inverse-band (noise taper) skipped: Nyquist %.1f Hz too low for %.0f–%.0f Hz taper.",
                    nyquist_native,
                    _NOISE_ENVELOPE_TAPER_LOW_HZ,
                    _NOISE_ENVELOPE_TAPER_HIGH_HZ,
                )
            else:
                noise_envelope_hp_cut_hz = taper_hi
                try:
                    # Piecewise-linear FIR: 0 → ramp → 1 (firls), zero-phase via filtfilt.
                    bands = [0.0, taper_lo, taper_lo, taper_hi, taper_hi, nyquist_native]
                    desired = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
                    width_hz = max(taper_hi - taper_lo, 50.0)
                    n_est = int(3.5 * float(native_sr_int) / width_hz)
                    numtaps = max(51, min(801, n_est))
                    if numtaps % 2 == 0:
                        numtaps += 1
                    fir_b = firls(numtaps, bands, desired, fs=float(native_sr_int))
                    audio_inverse_hp_native = filtfilt(fir_b, 1.0, audio_native)
                except Exception as e:
                    logging.warning("Inverse-band FIR taper (native rate) failed: %s", e)
                    audio_inverse_hp_native = None

                if audio_inverse_hp_native is not None and audio_inverse_hp_native.size > 0:
                    analytic_inv = hilbert(audio_inverse_hp_native)
                    envelope_inv_raw = np.abs(analytic_inv).astype(np.float64)
                    smooth_window_nat = max(1, int(smooth_ms * native_sr_int / 1000))
                    inv_smooth_nat = pd.Series(envelope_inv_raw).rolling(
                        window=smooth_window_nat, min_periods=1, center=True
                    ).mean().values
                    try:
                        inverse_band_envelope = librosa.resample(
                            inv_smooth_nat.astype(np.float64),
                            orig_sr=native_sr_int,
                            target_sr=new_sample_rate,
                        )
                        n_expect = len(audio_downsampled)
                        if inverse_band_envelope.size > n_expect:
                            inverse_band_envelope = inverse_band_envelope[:n_expect].copy()
                        elif inverse_band_envelope.size < n_expect:
                            pad = n_expect - inverse_band_envelope.size
                            inverse_band_envelope = np.pad(
                                inverse_band_envelope, (0, pad), mode="edge"
                            )
                    except Exception as e:
                        logging.warning("Could not resample inverse-band envelope to analysis rate: %s", e)
                        inverse_band_envelope = None

    low, high = lowcut / nyquist, highcut / nyquist

    if high >= 1.0:
        raise ValueError(f"Cannot create a {highcut}Hz filter. The sample rate of {new_sample_rate}Hz is too low.")

    sos = butter(order, [low, high], btype="band", output="sos")
    audio_filtered = sosfiltfilt(sos, audio_downsampled)

    if save_debug_file:
        base_name = output_stem_from_path(file_path)
        debug_path = os.path.join(output_directory, f"{base_name}_filtered_debug.wav")
        debug_sample_rate = 10000
        try:
            _write_peak_normalized_debug_wav(
                debug_path, audio_filtered, new_sample_rate, debug_sample_rate
            )
            logging.info(
                "Saved filtered audio WAV debug file (%s, %d Hz, int16) for HTML playback.",
                debug_path,
                debug_sample_rate,
            )
        except Exception as e:
            logging.error(f"Failed to write filtered debug WAV file {debug_path}: {e}")

        if audio_inverse_hp_native is not None and audio_inverse_hp_native.size > 0 and native_sr_int:
            inv_path = os.path.join(output_directory, f"{base_name}_filtered_inverse_debug.wav")
            try:
                _write_peak_normalized_wav_native_rate(
                    inv_path, audio_inverse_hp_native, native_sr_int
                )
                logging.info(
                    "Saved inverse-band (FIR taper %.0f–%.0f Hz → 1) debug WAV (%s, native %d Hz, int16).",
                    _NOISE_ENVELOPE_TAPER_LOW_HZ,
                    _NOISE_ENVELOPE_TAPER_HIGH_HZ,
                    inv_path,
                    native_sr_int,
                )
            except Exception as e:
                logging.error(f"Failed to write inverse-band debug WAV file {inv_path}: {e}")
    elif params["save_filtered_wav"] and not output_options.get("filtered_wav", True):
        logging.info("Skipping filtered audio WAV generation as requested.")

    # Hilbert envelope: magnitude of analytic signal for a sharper, more symmetric envelope
    # than abs + rolling mean, which helps peak timing stability (e.g. for HRV).
    analytic = hilbert(audio_filtered)
    envelope_raw = np.abs(analytic).astype(np.float64)
    # Smoothing to reduce ripple (e.g. between S1 and S2); window in ms from config (default 50 ms).
    smooth_window = max(1, int(smooth_ms * new_sample_rate / 1000))
    audio_envelope = pd.Series(envelope_raw).rolling(
        window=smooth_window, min_periods=1, center=True
    ).mean().values

    noise_removed_envelope: Optional[np.ndarray] = None
    if inverse_band_envelope is not None and len(inverse_band_envelope) == len(audio_envelope):
        noise_removed_envelope = np.maximum(
            0.0,
            audio_envelope.astype(np.float64) - inverse_band_envelope.astype(np.float64),
        )

    envelope_for_algorithm = (
        noise_removed_envelope
        if noise_removed_envelope is not None
        else audio_envelope
    )
    noise_floor, trough_indices = _calculate_dynamic_noise_floor(
        envelope_for_algorithm, new_sample_rate, params
    )
    return (
        audio_envelope,
        new_sample_rate,
        noise_floor,
        trough_indices,
        inverse_band_envelope,
        noise_removed_envelope,
    )
