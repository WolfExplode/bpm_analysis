# audio_preprocessing.py
# Engine stage 1: signal conditioning, envelope extraction, and noise-floor estimation.
# Pure computation — format conversion, channel splitting and debug WAV writing live in
# audio_io.py (outside the engine). Consumed by engine.py and fft_profiles.
import gc
import logging
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, firls, sosfiltfilt, welch, iirnotch, find_peaks, hilbert
import librosa
from .config import param

# (kind, signal, sample_rate) -> None; see preprocess_audio.
DebugAudioSink = Callable[[str, np.ndarray, int], None]

def _log_preprocess_elapsed(label: str, t_start: float) -> float:
    """Log wall time since t_start (perf_counter); return a fresh start time for the next step."""
    dt = time.perf_counter() - t_start
    logging.info("Preprocess timing — %s: %.3f s", label, dt)
    return time.perf_counter()


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

    if not param(params, "enable_hum_removal"):
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
        window_sec = float(param(params, "hum_psd_window_sec"))
        nperseg = int(sample_rate * window_sec)
        nperseg = max(256, min(len(audio_data), nperseg))
        freqs, psd = welch(audio_data, fs=sample_rate, nperseg=nperseg)
    except Exception as e:
        logging.warning("Hum detection skipped (PSD computation failed): %s", e)
        return audio_data, None

    # Restrict search to a low-frequency band where hums typically live
    fmin = float(param(params, "hum_min_freq_hz"))
    fmax = float(param(params, "hum_max_freq_hz"))
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

    min_prom_db = float(param(params, "hum_min_prominence_db"))

    try:
        peak_indices, properties = find_peaks(psd_db_rel, prominence=min_prom_db)
    except Exception as e:
        logging.warning("Hum detection skipped (peak finding failed): %s", e)
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

    min_gap_db = float(param(params, "hum_min_prominence_over_second_db"))
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

    q = float(param(params, "hum_notch_q"))

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
    lowcut = float(param(params, "preprocess_bandpass_low_hz"))
    highcut = float(param(params, "preprocess_bandpass_high_hz"))
    order = int(param(params, "preprocess_bandpass_order"))
    nyquist = 0.5 * sample_rate
    low, high = lowcut / nyquist, highcut / nyquist
    if high >= 1.0:
        return audio
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, audio)


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


def _centered_moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Bit-exact, O(n) replacement for
    pd.Series(x).rolling(window, min_periods=1, center=True).mean().values.

    Pandas centres with the window split (w//2 before, (w-1)//2 after); edges
    average the available points (min_periods=1). A cumulative-sum gives the same
    values far faster than the pandas rolling machinery.
    """
    x = np.asarray(x, dtype=np.float64)
    if window <= 1:
        return x.copy()
    n = x.shape[0]
    c = np.empty(n + 1, dtype=np.float64)
    c[0] = 0.0
    np.cumsum(x, out=c[1:])
    idx = np.arange(n)
    lo = np.maximum(0, idx - window // 2)
    hi = np.minimum(n, idx + (window - 1) // 2 + 1)
    return (c[hi] - c[lo]) / (hi - lo)


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
    rejection_multiplier = param(params, "trough_rejection_multiplier")
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
    file_path: str,
    params: Dict,
    debug_audio_sink: Optional[DebugAudioSink] = None,
) -> Tuple[np.ndarray, int, pd.Series, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load, condition and envelope the recording; never writes files.

    debug_audio_sink, when given, receives intermediate signals for debug playback as
    (kind, signal, sample_rate) with kind "filtered" (bandpass @ analysis rate) or
    "filtered_inverse" (inverse-band HF @ working native rate). The caller decides
    whether and where to persist them.
    """
    save_debug_file = debug_audio_sink is not None
    target_sample_rate = int(param(params, "preprocess_target_sample_rate"))
    analysis_start_sec = max(0.0, float(param(params, "analysis_start_sec")))

    t_preprocess = time.perf_counter()
    t_step = time.perf_counter()
    try:
        # Preserve historical behavior: simple mono mix of all channels.
        audio_downsampled, new_sample_rate = librosa.load(file_path, sr=target_sample_rate, mono=True)
    except Exception as e:
        logging.error("Librosa failed to load file: %s", e)
        raise

    if analysis_start_sec > 0.0:
        skip_n = int(round(analysis_start_sec * new_sample_rate))
        skip_n = min(skip_n, max(0, audio_downsampled.size - 1))
        if skip_n > 0:
            audio_downsampled = audio_downsampled[skip_n:]
            logging.info(
                "Skipping first %.2f s of recording (analysis_start_sec): dropped %d samples @ %d Hz.",
                analysis_start_sec, skip_n, new_sample_rate,
            )
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
    lowcut = float(param(params, "preprocess_bandpass_low_hz"))
    highcut = float(param(params, "preprocess_bandpass_high_hz"))
    order = int(param(params, "preprocess_bandpass_order"))
    nyquist = 0.5 * new_sample_rate

    # Out-of-band path: high-pass taper + Hilbert at inverse_band_working_sample_rate (or native if disabled in params).
    inverse_band_envelope: Optional[np.ndarray] = None
    audio_inverse_hp_native: Optional[np.ndarray] = None
    native_sr_int: Optional[int] = None  # effective rate for inverse-band FIR/Hilbert (working or native)
    smooth_ms = float(param(params, "envelope_smooth_window_ms"))
    taper_lo_cfg = float(param(params, "inverse_band_taper_low_hz"))
    taper_hi_cfg = float(param(params, "inverse_band_taper_high_hz"))

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
            if analysis_start_sec > 0.0 and native_sr_int:
                skip_n_native = int(round(analysis_start_sec * native_sr_int))
                skip_n_native = min(skip_n_native, max(0, audio_native.size - 1))
                if skip_n_native > 0:
                    audio_native = audio_native[skip_n_native:]
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
                    inv_smooth_nat = _centered_moving_average(envelope_inv_raw, smooth_window_nat)
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
        # The del statements above already drop the large HF buffers; a forced
        # full gc.collect() here cost ~28% of per-file runtime in batch profiling
        # for no benefit. Left opt-in for memory-constrained single-file runs.
        if not save_debug_file and bool(param(params, "preprocess_force_gc")):
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
        debug_audio_sink("filtered", audio_filtered, new_sample_rate)
        if audio_inverse_hp_native is not None and audio_inverse_hp_native.size > 0 and native_sr_int:
            debug_audio_sink("filtered_inverse", audio_inverse_hp_native, native_sr_int)
        if audio_inverse_hp_native is not None:
            del audio_inverse_hp_native
            audio_inverse_hp_native = None
        gc.collect()
        t_step = _log_preprocess_elapsed("optional debug WAV writes (filtered / inverse)", t_dbg)

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
    audio_envelope = _centered_moving_average(envelope_raw, smooth_window)
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
