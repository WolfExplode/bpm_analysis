"""
Heart rate and systolic time interval from PCG via autocorrelation.

Translated from getHeartRateSchmidt.m
"""

import numpy as np
from scipy.signal import fftconvolve

from springer_hsmm.signal import (
    butterworth_high_pass_filter,
    butterworth_low_pass_filter,
    homomorphic_envelope_with_hilbert,
    schmidt_spike_removal,
)


def get_heart_rate_schmidt(
    audio_data: np.ndarray,
    fs: float,
    *,
    pre_homomorphic_env: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Derive heart rate (BPM) and systolic time interval (seconds) from PCG.

    Uses bandpass 25-400 Hz, spike removal, homomorphic envelope,
    then autocorrelation: HR = peak in 0.5-2 s; systole = peak in 0.2 s to RR/2.

    If pre_homomorphic_env is provided (e.g. from springer_shared_preprocess),
    filtering and homomorphic envelope are skipped to avoid duplicate work.

    Returns
    -------
    heart_rate : float
        Beats per minute.
    systolic_time_interval : float
        Systolic duration in seconds.
    """
    if pre_homomorphic_env is not None:
        homomorphic_env = np.asarray(pre_homomorphic_env, dtype=np.float64).flatten()
    else:
        audio_data = np.asarray(audio_data, dtype=np.float64).flatten()
        audio_data = butterworth_low_pass_filter(audio_data, 2, 400, fs)
        audio_data = butterworth_high_pass_filter(audio_data, 2, 25, fs)
        audio_data = schmidt_spike_removal(audio_data, fs)
        homomorphic_env = homomorphic_envelope_with_hilbert(audio_data, fs)

    y = homomorphic_env - np.mean(homomorphic_env)
    n = len(y)
    # FFT-based autocorrelation O(n log n) instead of np.correlate O(n^2) for long signals
    c = fftconvolve(y, y[::-1], mode="full")[n - 1 :]
    c = c / (c[0] + 1e-12)  # normalize (coeff)

    min_index = int(0.5 * fs)
    max_index = int(2 * fs)
    max_index = min(max_index, len(c) - 1)
    if min_index >= max_index:
        heart_rate = 60.0
        systolic_time_interval = 0.3
        return heart_rate, systolic_time_interval

    segment = c[min_index : max_index + 1]
    index = int(np.argmax(segment))
    true_index = min_index + index
    heart_rate = 60.0 / (true_index / fs)

    max_sys_duration = int(round(((60 / heart_rate) * fs) / 2))
    min_sys_duration = int(round(0.2 * fs))
    max_sys_duration = min(max_sys_duration, len(c) - 1)
    if min_sys_duration >= max_sys_duration:
        systolic_time_interval = 0.3
        return heart_rate, systolic_time_interval
    segment_sys = c[min_sys_duration : max_sys_duration + 1]
    pos = int(np.argmax(segment_sys))
    sample_idx = min_sys_duration + pos
    systolic_time_interval = sample_idx / fs

    return heart_rate, systolic_time_interval
