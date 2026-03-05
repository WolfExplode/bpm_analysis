"""
Signal preprocessing: filters, normalise, spike removal, envelopes.

Translated from: normalise_signal.m, butterworth_*_filter.m,
schmidt_spike_removal.m, Homomorphic_Envelope_with_Hilbert.m, Hilbert_Envelope.m
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def normalise_signal(signal: np.ndarray) -> np.ndarray:
    """
    Subtract mean and divide by std (z-score normalisation).

    Translated from normalise_signal.m.
    """
    signal = np.asarray(signal, dtype=np.float64)
    mean_of_signal = np.mean(signal)
    std_of_signal = np.std(signal)
    if std_of_signal <= 0:
        return signal - mean_of_signal
    return (signal - mean_of_signal) / std_of_signal


def butterworth_high_pass_filter(
    original_signal: np.ndarray,
    order: int,
    cutoff: float,
    sampling_frequency: float,
) -> np.ndarray:
    """
    Zero-phase high-pass Butterworth filter (forward-backward filtfilt).

    Translated from butterworth_high_pass_filter.m.
    """
    nyq = sampling_frequency / 2.0
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high")
    return filtfilt(b, a, original_signal.astype(np.float64))


def butterworth_low_pass_filter(
    original_signal: np.ndarray,
    order: int,
    cutoff: float,
    sampling_frequency: float,
) -> np.ndarray:
    """
    Zero-phase low-pass Butterworth filter (forward-backward filtfilt).

    Translated from butterworth_low_pass_filter.m.
    """
    nyq = sampling_frequency / 2.0
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low")
    return filtfilt(b, a, original_signal.astype(np.float64))


def schmidt_spike_removal(original_signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Remove spikes: 500 ms windows, MAA > 3*median(MAA) replaced by zeros
    between zero-crossings. Loop until no such window remains.

    Translated from schmidt_spike_removal.m.
    """
    original_signal = np.asarray(original_signal, dtype=np.float64).flatten()
    windowsize = int(round(fs / 2))
    n = len(original_signal)
    trailingsamples = n % windowsize
    if trailingsamples > 0:
        usable = n - trailingsamples
        sampleframes = original_signal[:usable].reshape(windowsize, -1, order="F")
    else:
        sampleframes = original_signal.reshape(windowsize, -1, order="F")

    MAAs = np.max(np.abs(sampleframes), axis=0)
    median_maa = np.median(MAAs)

    while np.any(MAAs > 3 * median_maa):
        window_num = int(np.argmax(MAAs))
        window_signal = sampleframes[:, window_num]
        spike_position = int(np.argmax(np.abs(window_signal)))

        # Zero crossings: sign change
        sgn = np.sign(window_signal)
        diff_sgn = np.diff(sgn)
        zero_crossings = np.zeros(len(window_signal), dtype=bool)
        zero_crossings[:-1] = np.abs(diff_sgn) > 1

        # Last zero crossing before spike
        before = np.where(zero_crossings[: spike_position + 1])[0]
        spike_start = int(before[-1]) if len(before) > 0 else 0

        # First zero crossing after spike
        zero_crossings_after = zero_crossings.copy()
        zero_crossings_after[: spike_position + 1] = False
        after = np.where(zero_crossings_after)[0]
        spike_end = int(after[0]) if len(after) > 0 else windowsize - 1
        spike_end = min(spike_end, windowsize - 1)

        sampleframes[spike_start : spike_end + 1, window_num] = 0.0001
        MAAs = np.max(np.abs(sampleframes), axis=0)
        median_maa = np.median(MAAs)

    if trailingsamples > 0:
        despiked = sampleframes.flatten(order="F")
        despiked = np.concatenate([despiked, original_signal[usable:]])
    else:
        despiked = sampleframes.flatten(order="F")

    return despiked


def homomorphic_envelope_with_hilbert(
    input_signal: np.ndarray,
    sampling_frequency: float,
    lpf_frequency: float = 8.0,
) -> np.ndarray:
    """
    Homomorphic envelope: hilbert -> abs -> log -> LPF -> exp.
    Uses 1st-order Butterworth LPF at lpf_frequency (default 8 Hz).
    First sample set to second to remove spurious spike.

    Translated from Homomorphic_Envelope_with_Hilbert.m.
    """
    input_signal = np.asarray(input_signal, dtype=np.float64).flatten()
    analytic = hilbert(input_signal)
    log_env = np.log(np.abs(analytic) + 1e-10)
    nyq = sampling_frequency / 2.0
    normal_cutoff = min(lpf_frequency / nyq, 0.99)
    b, a = butter(1, normal_cutoff, btype="low")
    homomorphic_envelope = np.exp(filtfilt(b, a, log_env))
    homomorphic_envelope[0] = homomorphic_envelope[1]
    return homomorphic_envelope


def hilbert_envelope(input_signal: np.ndarray) -> np.ndarray:
    """
    Hilbert envelope: magnitude of analytic signal.

    Translated from Hilbert_Envelope.m.
    """
    input_signal = np.asarray(input_signal, dtype=np.float64).flatten()
    return np.abs(hilbert(input_signal))
