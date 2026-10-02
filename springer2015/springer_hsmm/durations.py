"""
State duration distributions for the HSMM (S1, systole, S2, diastole).

Translated from get_duration_distributions.m
"""

from typing import Any

from springer_hsmm.options import default_springer_hsmm_options


def get_duration_distributions(
    heart_rate: float,
    systolic_time: float,
    options: dict | None = None,
) -> tuple[list[tuple[float, float]], int, int, int, int, int, int, int, int]:
    """
    Gaussian duration params (mean, variance in samples at feature Fs) and min/max.

    State 1 = S1, 2 = systole, 3 = S2, 4 = diastole.
    Durations are in samples at audio_segmentation_Fs (50 Hz).

    Returns
    -------
    d_distributions : list of (mean, var) for each state
    min_S1, max_S1, min_S2, max_S2, min_systole, max_systole, min_diastole, max_diastole : int
    """
    options = options or default_springer_hsmm_options()
    feat_fs = options["audio_segmentation_Fs"]

    mean_S1 = round(0.122 * feat_fs)
    std_S1 = round(0.022 * feat_fs)
    mean_S2 = round(0.094 * feat_fs)
    std_S2 = round(0.022 * feat_fs)

    mean_systole = round(systolic_time * feat_fs) - mean_S1
    std_systole = (25 / 1000) * feat_fs

    rr_sec = 60.0 / max(heart_rate, 1e-6)
    mean_diastole = (rr_sec - systolic_time - 0.094) * feat_fs
    std_diastole = 0.07 * mean_diastole + (6 / 1000) * feat_fs

    d_distributions: list[tuple[float, float]] = [
        (float(mean_S1), float(std_S1**2)),
        (float(mean_systole), float(std_systole**2)),
        (float(mean_S2), float(std_S2**2)),
        (float(mean_diastole), float(std_diastole**2)),
    ]

    min_systole = int(mean_systole - 3 * (std_systole + std_S1))
    max_systole = int(mean_systole + 3 * (std_systole + std_S1))
    min_diastole = int(mean_diastole - 3 * std_diastole)
    max_diastole = int(mean_diastole + 3 * std_diastole)

    min_S1 = int(mean_S1 - 3 * std_S1)
    if min_S1 < feat_fs / 50:
        min_S1 = int(feat_fs / 50)
    min_S2 = int(mean_S2 - 3 * std_S2)
    if min_S2 < feat_fs / 50:
        min_S2 = int(feat_fs / 50)
    max_S1 = int(mean_S1 + 3 * std_S1)
    max_S2 = int(mean_S2 + 3 * std_S2)

    return (
        d_distributions,
        min_S1,
        max_S1,
        min_S2,
        max_S2,
        min_systole,
        max_systole,
        min_diastole,
        max_diastole,
    )
