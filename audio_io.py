import os
import logging
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, welch, iirnotch, find_peaks, hilbert, medfilt
import librosa

try:
    from pydub import AudioSegment
except ImportError:
    logging.warning("Pydub library not found. Install with 'pip install pydub'.")
    AudioSegment = None

try:
    import pyPCG
    import pyPCG.preprocessing as preproc
    PYPCG_AVAILABLE = True
except ImportError:
    logging.warning("pyPCG-toolbox not installed. Install with: pip install pyPCG-toolbox")
    PYPCG_AVAILABLE = False
    preproc = None


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

        if nperseg > len(audio_data):
            nperseg = len(audio_data)

        freqs, psd = welch(audio_data, fs=sample_rate, nperseg=nperseg)
    except Exception as e:
        logging.warning("Hum detection skipped (PSD computation failed): %s", e)
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
        logging.warning("Hum detection skipped (peak finding failed): %s", e)
        return audio_data, None

    if peak_indices.size == 0:
        logging.info(
            "Hum removal: no strong narrow-band peak detected in %.1f–%.1f Hz.", fmin, fmax
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
        # Normalized frequency (0–1) for iirnotch
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
        logging.warning("Failed to open WAV for channel splitting (%s): %s", file_path, e)
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
            logging.warning("Failed to export channel %d for %s: %s", ch_idx, file_path, e)

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


def _apply_pypcg_denoising(audio_data: np.ndarray, sample_rate: int, params: Dict) -> np.ndarray:
    """Apply pyPCG denoising with tunable strength."""
    if params.get("denoising_method") is None or not PYPCG_AVAILABLE:
        return audio_data

    try:
        sig = pyPCG.pcg_signal(audio_data, sample_rate)
        method = params["denoising_method"]

        if method.startswith("wavelet"):
            import pywt

            wt_family = params.get("wt_family", "coif4")
            wt_level = params.get("wt_level", 5)

            coeffs = pywt.wavedec(sig.data, wt_family, level=wt_level)

            if method == "wavelet_auto":
                thresholded_coeffs = []
                for i, coeff in enumerate(coeffs):
                    if i == 0:
                        thresholded_coeffs.append(coeff)
                        continue

                    mad = np.median(np.abs(coeff - np.median(coeff)))
                    sigma = mad / 0.6745
                    tau = sigma * np.sqrt(2) * params.get("wavelet_threshold_multiplier", 1.0)

                    thresholded_coeffs.append(pywt.threshold(coeff, tau, mode="soft"))
            else:  # wavelet_manual
                manual_th = params.get("wavelet_threshold", 0.2)
                thresholded_coeffs = [
                    coeff if i == 0 else pywt.threshold(coeff, manual_th * np.max(np.abs(coeff)))
                    for i, coeff in enumerate(coeffs)
                ]

            denoised_data = pywt.waverec(thresholded_coeffs, wt_family)
            logging.info(
                "✓ Applied wavelet denoising (family=%s, level=%s, multiplier=%s)",
                wt_family,
                wt_level,
                params.get("wavelet_threshold_multiplier", 1.0),
            )
            return denoised_data

        elif method == "emd_savgol":
            denoised = preproc.emd_denoise_savgol(
                sig, window=params.get("emd_window", 10), poly=params.get("emd_poly", 3)
            )
            logging.info(
                "✓ Applied EMD denoising (window=%s, poly=%s)",
                params.get("emd_window"),
                params.get("emd_poly"),
            )
            return denoised.data

        else:
            logging.warning("Unknown denoising method: %s", method)
            return audio_data

    except ImportError as e:
        logging.error("❌ pyPCG-toolbox not installed. Install with: pip install pyPCG-toolbox")
        logging.error("Import error: %s", e)
        return audio_data
    except Exception as e:
        logging.error("❌ Denoising failed: %s", e)
        return audio_data


def homomorphic_envelope(
    audio_signal: np.ndarray,
    sample_rate: int,
    lowpass_cutoff: float = 25.0,
    filter_order: int = 2,
    medfilt_kernel: int = 0,
) -> np.ndarray:
    """
    Compute homomorphic envelope (Springer et al. style) for PCG.

    Rectify -> log -> low-pass (extract envelope) -> exp -> optional median filter.
    Higher lowpass_cutoff = less smoothing, sharper peaks.

    Parameters
    ----------
    audio_signal : np.ndarray
        Pre-filtered audio (e.g. bandpass for S1/S2).
    sample_rate : int
        Sampling frequency (Hz).
    lowpass_cutoff : float
        Low-pass cutoff (Hz). Higher = sharper envelope (e.g. 25–40).
    filter_order : int
        Butterworth low-pass order.
    medfilt_kernel : int
        Median filter kernel size; 0 = disabled. 3 = light spike removal.

    Returns
    -------
    envelope : np.ndarray
        Amplitude envelope (float64).
    """
    rectified = np.abs(audio_signal).astype(np.float64)
    epsilon = 1e-10
    log_rectified = np.log(rectified + epsilon)

    nyquist = sample_rate / 2.0
    normalized_cutoff = min(lowpass_cutoff / nyquist, 0.99)
    b, a = butter(filter_order, normalized_cutoff, btype="low")
    log_envelope = filtfilt(b, a, log_rectified)

    envelope = np.exp(log_envelope).astype(np.float64)
    if medfilt_kernel >= 3 and medfilt_kernel % 2 == 1:
        envelope = medfilt(envelope, kernel_size=medfilt_kernel).astype(np.float64)
    return envelope


def preprocess_audio(
    file_path: str, params: Dict, output_directory: str, output_options: Optional[Dict] = None
) -> Tuple[np.ndarray, int]:
    if output_options is None:
        output_options = {
            "html": True,
            "csv": True,
            "summary": True,
            "debug": True,
            "settings": True,
            "filtered_wav": True,
        }

    save_debug_file = params["save_filtered_wav"] and output_options.get("filtered_wav", True)
    target_sample_rate = int(params.get("target_sample_rate", 850))

    try:
        # Preserve historical behavior: simple mono mix of all channels.
        audio_downsampled, new_sample_rate = librosa.load(file_path, sr=target_sample_rate, mono=True)
    except Exception as e:
        logging.error("Librosa failed to load file: %s", e)
        raise

    audio_downsampled = _apply_pypcg_denoising(audio_downsampled, new_sample_rate, params)

    # Optional adaptive hum removal (e.g., ~50–70 Hz mains / equipment hum)
    audio_downsampled, detected_hum = _detect_and_remove_stationary_hum(
        audio_downsampled, new_sample_rate, params
    )
    if detected_hum is not None:
        logging.info("Detected and removed stationary hum at ~%.2f Hz.", detected_hum)

    # Bandpass for S1/S2 detection (configurable; typical PCG range ~15–250 Hz).
    lowcut = int(params.get("bandpass_lowcut_hz", 15))
    highcut = int(params.get("bandpass_highcut_hz", 250))
    nyquist = 0.5 * new_sample_rate
    low, high = lowcut / nyquist, highcut / nyquist

    if high >= 1.0:
        raise ValueError(f"Cannot create a {highcut}Hz filter. The sample rate of {new_sample_rate}Hz is too low.")

    b, a = butter(2, [low, high], btype="band")  # 2nd order; higher order not yet validated for this pipeline.
    audio_filtered = filtfilt(b, a, audio_downsampled)

    if save_debug_file:
        base_name = os.path.basename(os.path.splitext(file_path)[0])
        debug_path = os.path.join(output_directory, f"{base_name}_filtered_debug.wav")

        # Resample to a browser‑friendly sample rate for HTML5 audio playback.
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
            logging.error("Failed to write filtered debug WAV file %s: %s", debug_path, e)
    elif params["save_filtered_wav"] and not output_options.get("filtered_wav", True):
        logging.info("Skipping filtered audio WAV generation as requested.")

    envelope_method = (params.get("envelope_method") or "homomorphic").strip().lower()
    if envelope_method == "hilbert":
        # Hilbert envelope: magnitude of analytic signal; light smoothing to reduce ripple.
        analytic = hilbert(audio_filtered)
        envelope_raw = np.abs(analytic).astype(np.float64)
        smooth_window = max(1, new_sample_rate // 80)
        audio_envelope = pd.Series(envelope_raw).rolling(
            window=smooth_window, min_periods=1, center=True
        ).mean().values
    else:
        # Homomorphic envelope (Springer et al. style); default for A/B testing.
        audio_envelope = homomorphic_envelope(
            audio_filtered,
            new_sample_rate,
            lowpass_cutoff=float(params.get("homomorphic_cutoff_hz", 25.0)),
            filter_order=int(params.get("homomorphic_filter_order", 2)),
            medfilt_kernel=int(params.get("homomorphic_medfilt_kernel", 0)),
        )

    return audio_envelope, new_sample_rate


def load_audio_mono(file_path: str, sample_rate: int) -> Tuple[np.ndarray, int]:
    """
    Load audio as mono at the given sample rate with no filtering (no denoise, hum removal, or bandpass).
    Used for the Springer pipeline so the algorithm receives raw/resampled PCG and does its own 25-400 Hz bandpass.
    Returns (audio, sample_rate).
    """
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    except Exception as e:
        logging.error("Librosa failed to load file: %s", e)
        raise
    return audio, sr


def get_filtered_audio(file_path: str, params: Dict) -> Tuple[np.ndarray, int]:
    """
    Load and filter audio (denoise, hum removal, bandpass) without computing envelope.
    Used when segmenter is Springer so the same waveform is passed to the HSMM.
    Returns (filtered_audio, sample_rate); length matches what preprocess_audio would use.
    """
    target_sample_rate = int(params.get("target_sample_rate", 850))
    try:
        audio_downsampled, new_sample_rate = librosa.load(file_path, sr=target_sample_rate, mono=True)
    except Exception as e:
        logging.error("Librosa failed to load file: %s", e)
        raise
    audio_downsampled = _apply_pypcg_denoising(audio_downsampled, new_sample_rate, params)
    audio_downsampled, detected_hum = _detect_and_remove_stationary_hum(
        audio_downsampled, new_sample_rate, params
    )
    if detected_hum is not None:
        logging.info("Detected and removed stationary hum at ~%.2f Hz.", detected_hum)
    lowcut = int(params.get("bandpass_lowcut_hz", 15))
    highcut = int(params.get("bandpass_highcut_hz", 250))
    nyquist = 0.5 * new_sample_rate
    low, high = lowcut / nyquist, highcut / nyquist
    if high >= 1.0:
        raise ValueError(
            f"Cannot create a {highcut}Hz filter. The sample rate of {new_sample_rate}Hz is too low."
        )
    b, a = butter(2, [low, high], btype="band")
    audio_filtered = filtfilt(b, a, audio_downsampled)
    return audio_filtered, new_sample_rate

