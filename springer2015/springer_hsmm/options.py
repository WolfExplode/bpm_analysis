"""
Default options for the Springer HSMM segmentation algorithm.

Translated from default_Springer_HSMM_options.m
"""

from typing import Any


def default_springer_hsmm_options() -> dict[str, Any]:
    """
    Return default options for the Springer segmentation algorithm.

    Returns
    -------
    dict
        audio_Fs : int
            Sampling frequency of the audio (Hz). Default 1000.
        audio_segmentation_Fs : int
            Downsampled feature rate (Hz). Default 50.
        use_mex : bool
            Whether to use MEX for Viterbi (Python port uses False).
        include_wavelet_feature : bool
            Whether to include wavelet feature in PCG features.
        segmentation_tolerance : float
            Tolerance for S1/S2 localization (seconds). Default 0.1.
    """
    return {
        "audio_Fs": 1000,
        "audio_segmentation_Fs": 50,
        "use_mex": False,  # No MEX in Python; use pure Python Viterbi
        "include_wavelet_feature": True,
        "segmentation_tolerance": 0.1,
    }
