# audio_preprocessing.py
# Full audio preprocessing pipeline: format conversion, channel splitting, signal conditioning,
# envelope extraction, and noise-floor estimation.
# Consumed by bpm_analysis (main pipeline) and gui (conversion helpers).
import gc
import os
import logging
import time
from typing import Dict, Optional, Tuple, List

from config import DEFAULT_OUTPUT_OPTIONS
from file_io import output_stem_from_path

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

CHANNEL_MODE_MIXED = "mixed"
CHANNEL_MODE_LEFT = "left"
CHANNEL_MODE_RIGHT = "right"
CHANNEL_MODE_ALL = "all"
CHANNEL_MODES = (CHANNEL_MODE_MIXED, CHANNEL_MODE_LEFT, CHANNEL_MODE_RIGHT, CHANNEL_MODE_ALL)
MAX_STEREO_CHANNEL_COUNT = 2


class TooManyAudioChannelsError(ValueError):
    """Raised when channel-specific processing is requested on >2-channel audio."""


def normalize_channel_mode(mode: str) -> str:
    m = (mode or CHANNEL_MODE_MIXED).strip().lower()
    if m not in CHANNEL_MODES:
        raise ValueError(f"Invalid channel_mode {mode!r}; expected one of {CHANNEL_MODES}.")
    return m


def get_audio_channel_count(file_path: str) -> int:
    """Return channel count (1 for mono). Falls back to 1 if pydub cannot open the file."""
    if not AudioSegment:
        return 1
    try:
        return int(AudioSegment.from_file(file_path).channels)
    except Exception as e:
        logging.warning("Could not read channel count for %s: %s", file_path, e)
        return 1


def _export_mono_channel(
    segment,
    file_path: str,
    output_directory: str,
    channel_index: int,
) -> str:
    """Export one 0-based mono channel; filename uses 1-based _chN suffix."""
    ch_num = channel_index + 1
    base_name = output_stem_from_path(file_path)
    out_path = os.path.join(output_directory, f"{base_name}_ch{ch_num}.wav")
    segment.export(out_path, format="wav")
    return out_path

def _log_preprocess_elapsed(label: str, t_start: float) -> float:
    """Log wall time since t_start (perf_counter); return a fresh start time for the next step."""
    dt = time.perf_counter() - t_start
    logging.info("Preprocess timing — %s: %.3f s", label, dt)
    return time.perf_counter()


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

    skip_over = params.get("hum_removal_skip_if_longer_than_min")
    if skip_over is not None:
        try:
            limit_min = float(skip_over)
        except (TypeError, ValueError):
            limit_min = 0.0
        if limit_min > 0.0:
            dur_min = (float(audio_data.size) / float(sample_rate)) / 60.0
            if dur_min > limit_min:
                logging.info(
                    "Hum removal skipped: duration %.1f min exceeds %.1f min limit.",
                    dur_min,
                    limit_min,
                )
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
    For a stereo WAV file, export one mono WAV per channel.

    Returns a list of file paths to the mono channel WAVs. If the input
    is already mono or splitting fails, the original file_path is returned
    as the only element. Raises TooManyAudioChannelsError when channel count > 2.
    """
    return resolve_wav_for_channel_mode(file_path, output_directory, CHANNEL_MODE_ALL)


def resolve_wav_for_channel_mode(
    file_path: str,
    output_directory: str,
    channel_mode: str = CHANNEL_MODE_MIXED,
) -> List[str]:
    """
    Select working WAV path(s) for analysis based on channel_mode.

    mixed — use file as-is (librosa downmixes at load time).
    left / right — export one stereo channel (ch1 = left, ch2 = right).
    all — export each stereo channel as separate mono WAVs.

    Mono inputs are returned unchanged for left/right/all. More than two channels
    raises TooManyAudioChannelsError.
    """
    mode = normalize_channel_mode(channel_mode)
    if mode == CHANNEL_MODE_MIXED:
        return [file_path]

    if not AudioSegment:
        logging.warning("Pydub not available; cannot select channels. Using original file only.")
        return [file_path]

    try:
        sound = AudioSegment.from_file(file_path)
    except Exception as e:
        logging.warning("Failed to open WAV for channel selection (%s): %s", file_path, e)
        return [file_path]

    n_channels = sound.channels
    if n_channels <= 1:
        return [file_path]

    if n_channels > MAX_STEREO_CHANNEL_COUNT:
        raise TooManyAudioChannelsError(
            f"'{os.path.basename(file_path)}' has {n_channels} audio channels; "
            "left/right/separate-channel analysis supports stereo (2 channels) only."
        )

    mono_segments = sound.split_to_mono()
    if mode == CHANNEL_MODE_ALL:
        channel_paths: List[str] = []
        for idx, seg in enumerate(mono_segments):
            try:
                channel_paths.append(_export_mono_channel(seg, file_path, output_directory, idx))
            except Exception as e:
                logging.warning("Failed to export channel %d for %s: %s", idx + 1, file_path, e)
        if not channel_paths:
            return [file_path]
        logging.info(
            "Split %s into %d mono channel file(s): %s",
            os.path.basename(file_path),
            len(channel_paths),
            ", ".join(os.path.basename(p) for p in channel_paths),
        )
        return channel_paths

    channel_index = 0 if mode == CHANNEL_MODE_LEFT else 1
    side_label = "left" if mode == CHANNEL_MODE_LEFT else "right"
    try:
        out_path = _export_mono_channel(
            mono_segments[channel_index], file_path, output_directory, channel_index
        )
    except Exception as e:
        logging.warning("Failed to export %s channel for %s: %s", side_label, file_path, e)
        return [file_path]

    logging.info(
        "Using %s channel only from %s: %s",
        side_label,
        os.path.basename(file_path),
        os.path.basename(out_path),
    )
    return [out_path]


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
    t_nf0 = time.perf_counter()
    min_peak_dist_samples = int(params['min_peak_distance_sec'] * sample_rate)
    trough_prom_thresh = np.quantile(audio_envelope, params['trough_prominence_quantile'])

    # --- STEP 1: Find all potential troughs initially ---
    t_step = time.perf_counter()
    all_trough_indices, _ = find_peaks(-audio_envelope, distance=min_peak_dist_samples, prominence=trough_prom_thresh)
    logging.info(
        "Preprocess timing — noise_floor (1) find_peaks: %.3f s (%d trough candidates)",
        time.perf_counter() - t_step,
        len(all_trough_indices),
    )

    # If we don't have enough troughs to begin with, fall back to a simple static floor.
    if len(all_trough_indices) < 5:
        logging.warning("Not enough troughs found for sanitization. Using a static noise floor.")
        fallback_value = np.quantile(audio_envelope, params['noise_floor_quantile'])
        dynamic_noise_floor = pd.Series(fallback_value, index=np.arange(len(audio_envelope)))
        logging.info(
            "Preprocess timing — noise_floor total: %.3f s (static fallback)",
            time.perf_counter() - t_nf0,
        )
        return dynamic_noise_floor, all_trough_indices

    n = len(audio_envelope)
    noise_window_samples = int(params['noise_window_sec'] * sample_rate)
    quantile_val = params['noise_floor_quantile']

    # --- STEP 2: Draft noise floor from ALL troughs (dense interpolate + rolling quantile) ---
    t_step = time.perf_counter()
    dense_troughs_draft = _dense_troughs_linear_interpolate(
        n, all_trough_indices, audio_envelope[all_trough_indices]
    )
    logging.info(
        "Preprocess timing — noise_floor (2a) dense interpolate (draft): %.3f s",
        time.perf_counter() - t_step,
    )
    t_step = time.perf_counter()
    draft_floor_arr = _rolling_quantile_center_bfill_ffill(
        dense_troughs_draft, noise_window_samples, quantile_val, min_periods=3
    )
    logging.info(
        "Preprocess timing — noise_floor (2b) rolling quantile draft floor: %.3f s (window=%d samples)",
        time.perf_counter() - t_step,
        noise_window_samples,
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
        t_step = time.perf_counter()
        dense_troughs_final = _dense_troughs_linear_interpolate(
            n, sanitized_trough_indices, audio_envelope[sanitized_trough_indices]
        )
        logging.info(
            "Preprocess timing — noise_floor (3a) dense interpolate (final): %.3f s",
            time.perf_counter() - t_step,
        )
        t_step = time.perf_counter()
        final_floor_arr = _rolling_quantile_center_bfill_ffill(
            dense_troughs_final, noise_window_samples, quantile_val, min_periods=3
        )
        logging.info(
            "Preprocess timing — noise_floor (3b) rolling quantile final floor: %.3f s",
            time.perf_counter() - t_step,
        )
        dynamic_noise_floor = pd.Series(final_floor_arr, index=np.arange(n))
    else:
        logging.warning("Not enough sanitized troughs remaining. Using non-sanitized floor as fallback.")
        dynamic_noise_floor = pd.Series(draft_floor_arr, index=np.arange(n))

    if dynamic_noise_floor.isnull().all():
        fallback_val = np.quantile(audio_envelope, 0.1)
        dynamic_noise_floor = pd.Series(fallback_val, index=np.arange(len(audio_envelope)))

    logging.info(
        "Preprocess timing — noise_floor total: %.3f s",
        time.perf_counter() - t_nf0,
    )
    return dynamic_noise_floor, np.asarray(sanitized_trough_indices, dtype=np.intp)


def preprocess_audio(
    file_path: str, params: Dict, output_directory: str, output_options: Optional[Dict] = None
) -> Tuple[np.ndarray, int, pd.Series, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if output_options is None:
        output_options = DEFAULT_OUTPUT_OPTIONS.copy()

    save_debug_file = params["save_filtered_wav"] and output_options.get("filtered_wav", True)
    target_sample_rate = int(params.get("preprocess_target_sample_rate", 500))

    t_preprocess = time.perf_counter()
    t_step = time.perf_counter()
    try:
        # Preserve historical behavior: simple mono mix of all channels.
        audio_downsampled, new_sample_rate = librosa.load(file_path, sr=target_sample_rate, mono=True)
    except Exception as e:
        logging.error(f"Librosa failed to load file: {e}")
        raise
    _dur_min = (float(audio_downsampled.size) / float(new_sample_rate)) / 60.0 if new_sample_rate else 0.0
    t_step = _log_preprocess_elapsed(
        f"librosa.load @ analysis rate ({new_sample_rate} Hz), {audio_downsampled.size} samples (~{_dur_min:.1f} min)",
        t_step,
    )

    # Optional adaptive hum removal (e.g., ~50-70 Hz mains / equipment hum)
    audio_downsampled, detected_hum = _detect_and_remove_stationary_hum(
        audio_downsampled, new_sample_rate, params
    )
    t_step = _log_preprocess_elapsed("hum detection / notch (if any)", t_step)
    if detected_hum is not None:
        logging.info("Detected and removed stationary hum at ~%.2f Hz.", detected_hum)

    # Bandpass for S1/S2 detection: typical PCG range where first and second heart sounds have most energy.
    lowcut = float(params.get("preprocess_bandpass_low_hz", 20.0))
    highcut = float(params.get("preprocess_bandpass_high_hz", 220.0))
    order = int(params.get("preprocess_bandpass_order", 2))
    nyquist = 0.5 * new_sample_rate

    # Out-of-band path: high-pass taper + Hilbert at inverse_band_working_sample_rate (or native if disabled in params).
    inverse_band_envelope: Optional[np.ndarray] = None
    audio_inverse_hp_native: Optional[np.ndarray] = None
    native_sr_int: Optional[int] = None  # effective rate for inverse-band FIR/Hilbert (working or native)
    smooth_ms = float(params.get("envelope_smooth_window_ms", 50))
    taper_lo_cfg = float(params.get("inverse_band_taper_low_hz", 300.0))
    taper_hi_cfg = float(params.get("inverse_band_taper_high_hz", 600.0))

    if highcut <= 0.0:
        logging.warning("Inverse-band (noise envelope) skipped: preprocess_bandpass_high_hz must be > 0.")
        t_step = _log_preprocess_elapsed("inverse-band path (skipped: highcut <= 0)", t_step)
    else:
        t_inv_overall = time.perf_counter()
        try:
            t_native_load = time.perf_counter()
            inv_sr_param = params.get("inverse_band_working_sample_rate")
            min_inv_sr = int(np.ceil(2.5 * taper_hi_cfg))
            if inv_sr_param is None or float(inv_sr_param) <= 0.0:
                audio_native, native_sr = librosa.load(file_path, sr=None, mono=True)
                native_sr_int = int(round(float(native_sr)))
                load_label = "native"
            else:
                inv_sr = int(max(float(inv_sr_param), float(min_inv_sr)))
                audio_native, native_sr = librosa.load(file_path, sr=inv_sr, mono=True)
                native_sr_int = int(round(float(native_sr)))
                load_label = f"inverse-band working ({native_sr_int} Hz, min {min_inv_sr})"
            audio_native = np.asarray(audio_native, dtype=np.float64)
            _nat_min = (float(audio_native.size) / float(native_sr_int)) / 60.0 if native_sr_int else 0.0
            _log_preprocess_elapsed(
                f"inverse-band: librosa.load @ {load_label}, {audio_native.size} samples (~{_nat_min:.1f} min)",
                t_native_load,
            )
        except Exception as e:
            logging.warning("Could not load audio for inverse-band path: %s", e)
            audio_native = np.array([], dtype=np.float64)
            native_sr_int = None

        if native_sr_int is not None and audio_native.size > 0:
            nyquist_native = 0.5 * float(native_sr_int)
            taper_lo = float(taper_lo_cfg)
            taper_hi = float(taper_hi_cfg)
            if taper_hi >= nyquist_native:
                taper_hi = max(taper_lo + 20.0, nyquist_native * 0.999)
            if taper_lo >= taper_hi - 5.0:
                logging.warning(
                    "Inverse-band (noise taper) skipped: Nyquist %.1f Hz too low for %.0f–%.0f Hz taper.",
                    nyquist_native,
                    taper_lo_cfg,
                    taper_hi_cfg,
                )
                logging.info(
                    "Preprocess timing — inverse-band: skip point reached at +%.3f s in inverse block",
                    time.perf_counter() - t_inv_overall,
                )
                del audio_native
            else:
                try:
                    # Piecewise-linear FIR: 0 → ramp → 1 (firls), zero-phase via filtfilt.
                    t_fir = time.perf_counter()
                    bands = [0.0, taper_lo, taper_lo, taper_hi, taper_hi, nyquist_native]
                    desired = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
                    width_hz = max(taper_hi - taper_lo, 50.0)
                    n_est = int(3.5 * float(native_sr_int) / width_hz)
                    numtaps = max(51, min(801, n_est))
                    if numtaps % 2 == 0:
                        numtaps += 1
                    fir_b = firls(numtaps, bands, desired, fs=float(native_sr_int))
                    logging.info(
                        "Preprocess timing — inverse-band: FIR design (firls), numtaps=%d: %.3f s",
                        numtaps,
                        time.perf_counter() - t_fir,
                    )
                    t_fb = time.perf_counter()
                    _native_len = audio_native.size
                    audio_inverse_hp_native = filtfilt(fir_b, 1.0, audio_native)
                    del audio_native
                    _log_preprocess_elapsed(
                        f"inverse-band: filtfilt (working len={_native_len})",
                        t_fb,
                    )
                except Exception as e:
                    logging.warning("Inverse-band FIR taper (working rate) failed: %s", e)
                    audio_inverse_hp_native = None
                    del audio_native

                if audio_inverse_hp_native is not None and audio_inverse_hp_native.size > 0:
                    t_h = time.perf_counter()
                    analytic_inv = hilbert(audio_inverse_hp_native)
                    envelope_inv_raw = np.abs(analytic_inv).astype(np.float64)
                    del analytic_inv
                    _log_preprocess_elapsed("inverse-band: Hilbert + abs (working rate)", t_h)
                    smooth_window_nat = max(1, int(smooth_ms * native_sr_int / 1000))
                    t_r = time.perf_counter()
                    inv_smooth_nat = pd.Series(envelope_inv_raw).rolling(
                        window=smooth_window_nat, min_periods=1, center=True
                    ).mean().values
                    del envelope_inv_raw
                    _log_preprocess_elapsed(
                        f"inverse-band: rolling mean @ working rate (window={smooth_window_nat} samples)",
                        t_r,
                    )
                    try:
                        t_rs = time.perf_counter()
                        inverse_band_envelope = librosa.resample(
                            inv_smooth_nat.astype(np.float64),
                            orig_sr=native_sr_int,
                            target_sr=new_sample_rate,
                        )
                        _log_preprocess_elapsed(
                            f"inverse-band: resample envelope {native_sr_int} Hz → {new_sample_rate} Hz",
                            t_rs,
                        )
                        n_expect = len(audio_downsampled)
                        if inverse_band_envelope.size > n_expect:
                            inverse_band_envelope = inverse_band_envelope[:n_expect].copy()
                        elif inverse_band_envelope.size < n_expect:
                            pad = n_expect - inverse_band_envelope.size
                            inverse_band_envelope = np.pad(
                                inverse_band_envelope, (0, pad), mode="edge"
                            )
                        del inv_smooth_nat
                    except Exception as e:
                        logging.warning("Could not resample inverse-band envelope to analysis rate: %s", e)
                        inverse_band_envelope = None
                        del inv_smooth_nat
        # Native-rate HF buffer: free now unless debug WAV still needs it.
        if not save_debug_file and audio_inverse_hp_native is not None:
            del audio_inverse_hp_native
            audio_inverse_hp_native = None
        if not save_debug_file:
            gc.collect()
        t_step = _log_preprocess_elapsed("inverse-band section (overall)", t_inv_overall)

    low, high = lowcut / nyquist, highcut / nyquist

    if high >= 1.0:
        raise ValueError(f"Cannot create a {highcut}Hz filter. The sample rate of {new_sample_rate}Hz is too low.")

    t_bp = time.perf_counter()
    sos = butter(order, [low, high], btype="band", output="sos")
    audio_filtered = sosfiltfilt(sos, audio_downsampled)
    t_step = _log_preprocess_elapsed(
        f"bandpass sosfiltfilt ({lowcut:.1f}–{highcut:.1f} Hz, order {order}, len={len(audio_downsampled)})",
        t_bp,
    )

    if save_debug_file:
        t_dbg = time.perf_counter()
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
                    "Saved inverse-band (FIR taper %.0f–%.0f Hz → 1) debug WAV (%s, %d Hz, int16).",
                    taper_lo_cfg,
                    taper_hi_cfg,
                    inv_path,
                    native_sr_int,
                )
            except Exception as e:
                logging.error(f"Failed to write inverse-band debug WAV file {inv_path}: {e}")
        if audio_inverse_hp_native is not None:
            del audio_inverse_hp_native
            audio_inverse_hp_native = None
        gc.collect()
        t_step = _log_preprocess_elapsed("optional debug WAV writes (filtered / inverse)", t_dbg)
    elif params["save_filtered_wav"] and not output_options.get("filtered_wav", True):
        logging.info("Skipping filtered audio WAV generation as requested.")

    # Hilbert envelope: magnitude of analytic signal for a sharper, more symmetric envelope
    # than abs + rolling mean, which helps peak timing stability (e.g. for HRV).
    t_main = time.perf_counter()
    analytic = hilbert(audio_filtered)
    envelope_raw = np.abs(analytic).astype(np.float64)
    del analytic
    t_main = _log_preprocess_elapsed("main path: Hilbert + abs @ analysis rate", t_main)
    # Smoothing to reduce ripple (e.g. between S1 and S2); window in ms from config (default 50 ms).
    smooth_window = max(1, int(smooth_ms * new_sample_rate / 1000))
    t_roll = time.perf_counter()
    audio_envelope = pd.Series(envelope_raw).rolling(
        window=smooth_window, min_periods=1, center=True
    ).mean().values
    t_roll = _log_preprocess_elapsed(
        f"main path: rolling mean on envelope (window={smooth_window} samples)",
        t_roll,
    )

    noise_removed_envelope: Optional[np.ndarray] = None
    if inverse_band_envelope is not None and len(inverse_band_envelope) == len(audio_envelope):
        t_sub = time.perf_counter()
        noise_removed_envelope = np.maximum(
            0.0,
            audio_envelope.astype(np.float64) - inverse_band_envelope.astype(np.float64),
        )
        _log_preprocess_elapsed("noise_removed envelope (bandpass − inverse-band)", t_sub)

    envelope_for_algorithm = (
        noise_removed_envelope
        if noise_removed_envelope is not None
        else audio_envelope
    )
    noise_floor, trough_indices = _calculate_dynamic_noise_floor(
        envelope_for_algorithm, new_sample_rate, params
    )
    logging.info(
        "Preprocess timing — total (preprocess_audio): %.3f s",
        time.perf_counter() - t_preprocess,
    )
    return (
        audio_envelope,
        new_sample_rate,
        noise_floor,
        trough_indices,
        inverse_band_envelope,
        noise_removed_envelope,
    )
