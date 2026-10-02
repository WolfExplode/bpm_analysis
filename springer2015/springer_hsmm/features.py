"""
PCG feature extraction: PSD, DWT, getSpringerPCGFeatures.

Translated from get_PSD_feature_Springer_HMM.m, getDWT.m, getSpringerPCGFeatures.m
"""

import numpy as np
from scipy.signal import spectrogram, resample

from springer_hsmm.options import default_springer_hsmm_options
from springer_hsmm.signal import (
    butterworth_high_pass_filter,
    butterworth_low_pass_filter,
    hilbert_envelope,
    homomorphic_envelope_with_hilbert,
    normalise_signal,
    schmidt_spike_removal,
)


def springer_shared_preprocess(audio_data: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Bandpass (25-400 Hz), spike removal, and homomorphic envelope.
    Returns (filtered_audio, homomorphic_env). Used once and passed to features and HR.
    """
    audio_data = np.asarray(audio_data, dtype=np.float64).flatten()
    audio_data = butterworth_low_pass_filter(audio_data, 2, 400, fs)
    audio_data = butterworth_high_pass_filter(audio_data, 2, 25, fs)
    audio_data = schmidt_spike_removal(audio_data, fs)
    homomorphic_env = homomorphic_envelope_with_hilbert(audio_data, fs)
    return audio_data, homomorphic_env


def get_psd_feature_springer_hmm(
    data: np.ndarray,
    sampling_frequency: float,
    frequency_limit_low: float = 40.0,
    frequency_limit_high: float = 60.0,
) -> np.ndarray:
    """
    PSD feature: mean power over frequency band [low, high] Hz per time frame.

    Translated from get_PSD_feature_Springer_HMM.m.
    MATLAB spectrogram: window Fs/40, noverlap Fs/80, nfft Fs/2.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    nperseg = int(sampling_frequency / 40)
    noverlap = int(round(sampling_frequency / 80))
    nfft = int(round(sampling_frequency / 2))
    nfft = max(nfft, nperseg)
    freqs, times, P = spectrogram(
        data, fs=sampling_frequency, nperseg=nperseg, noverlap=noverlap, nfft=nfft
    )
    low_idx = int(np.argmin(np.abs(freqs - frequency_limit_low)))
    high_idx = int(np.argmin(np.abs(freqs - frequency_limit_high)))
    high_idx = min(high_idx + 1, len(freqs))
    psd = np.mean(P[low_idx:high_idx, :], axis=0)
    return psd


def get_dwt(
    x: np.ndarray, level: int = 3, wavelet: str = "rbio3.9"
) -> tuple[np.ndarray, np.ndarray]:
    """
    DWT at level N; return detail and approximation at level N expanded to len(x).

    Matches getDWT.m: expand coefficients by repeating each 2^level times (no
    synthesis filter), then wkeep to len(x). Avoids group delay from upcoef.
    """
    try:
        import pywt
    except ImportError:
        raise ImportError("PyWavelets (pywt) is required for wavelet feature. pip install PyWavelets")

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    coeffs = pywt.wavedec(x, wavelet, level=level)
    # coeffs = [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    cD_n = np.asarray(coeffs[1]).flatten()
    cA_n = np.asarray(coeffs[0]).flatten()

    def expand_keep(d: np.ndarray, lev: int, length: int) -> np.ndarray:
        # Repeat each coefficient 2^lev times (MATLAB d(ones(1,2^k),:) then wkeep1(..., len))
        # MATLAB wkeep1(X, len) with two args extracts the central len elements, not the first.
        expanded = np.repeat(d, 2**lev)
        if len(expanded) >= length:
            start = (len(expanded) - length) // 2
            out = expanded[start : start + length]
        else:
            pad_total = length - len(expanded)
            left = pad_total // 2
            right = pad_total - left
            out = np.pad(expanded, (left, right), mode="edge")
        # MATLAB: zero out negligible values
        out[np.abs(out) < np.sqrt(np.finfo(float).eps)] = 0.0
        return out

    cD_expanded = expand_keep(cD_n, level, n)
    cA_expanded = expand_keep(cA_n, level, n)
    return cD_expanded, cA_expanded


def get_springer_pcg_features(
    audio_data: np.ndarray,
    fs: float,
    options: dict | None = None,
    *,
    pre_filtered: np.ndarray | None = None,
    pre_homomorphic: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """
    Extract PCG features: homomorphic envelope, Hilbert envelope, PSD, optional wavelet.
    All downsampled to audio_segmentation_Fs (50 Hz) and normalised.

    If pre_filtered and pre_homomorphic are provided (e.g. from springer_shared_preprocess),
    bandpass, spike removal, and homomorphic envelope are skipped to avoid duplicate work.

    Returns
    -------
    PCG_Features : np.ndarray
        Shape (T, n_features). T = duration in seconds * 50.
    features_fs : int
        Feature sampling rate (50 Hz).
    """
    options = options or default_springer_hsmm_options()
    features_fs = options["audio_segmentation_Fs"]
    include_wavelet = options.get("include_wavelet_feature", True)

    if pre_filtered is not None and pre_homomorphic is not None:
        audio_data = np.asarray(pre_filtered, dtype=np.float64).flatten()
        homomorphic_env = np.asarray(pre_homomorphic, dtype=np.float64).flatten()
    else:
        audio_data = np.asarray(audio_data, dtype=np.float64).flatten()
        audio_data = butterworth_low_pass_filter(audio_data, 2, 400, fs)
        audio_data = butterworth_high_pass_filter(audio_data, 2, 25, fs)
        audio_data = schmidt_spike_removal(audio_data, fs)
        homomorphic_env = homomorphic_envelope_with_hilbert(audio_data, fs)

    n_feat = int(len(homomorphic_env) * features_fs / fs)
    downsampled_homomorphic = resample(homomorphic_env, n_feat)
    downsampled_homomorphic = normalise_signal(downsampled_homomorphic)

    # Hilbert envelope
    hilbert_env = hilbert_envelope(audio_data)
    downsampled_hilbert = resample(hilbert_env, n_feat)
    downsampled_hilbert = normalise_signal(downsampled_hilbert)

    # PSD feature 40-60 Hz
    psd = get_psd_feature_springer_hmm(audio_data, fs, 40, 60)
    psd = resample(psd, n_feat)
    psd = normalise_signal(psd)

    list_features = [
        downsampled_homomorphic.reshape(-1, 1),
        downsampled_hilbert.reshape(-1, 1),
        psd.reshape(-1, 1),
    ]

    if include_wavelet:
        if len(audio_data) < fs * 1.025:
            pad = int(round(0.025 * fs))
            audio_data = np.concatenate([audio_data, np.zeros(pad)])
        cD_level3, _ = get_dwt(audio_data, level=3, wavelet="rbio3.9")
        wavelet_feat = np.abs(cD_level3[: len(homomorphic_env)])
        downsampled_wavelet = resample(wavelet_feat, n_feat)
        downsampled_wavelet = normalise_signal(downsampled_wavelet).reshape(-1, 1)
        list_features.append(downsampled_wavelet)

    PCG_Features = np.hstack(list_features)
    return PCG_Features, features_fs
