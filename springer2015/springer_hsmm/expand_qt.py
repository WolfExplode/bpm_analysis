"""
Expand state sequence from feature rate to original audio length/sample rate.

Translated from expand_qt.m
"""

import numpy as np


def expand_qt(
    original_qt: np.ndarray,
    old_fs: float,
    new_fs: float,
    new_length: int,
) -> np.ndarray:
    """
    Map state sequence at old_fs to new_length samples at new_fs.
    Segment boundaries found by diff(original_qt); each segment gets the state at its midpoint.
    """
    original_qt = np.asarray(original_qt).flatten()
    expanded_qt = np.zeros(new_length, dtype=original_qt.dtype)

    changes = np.where(np.diff(original_qt) != 0)[0] + 1
    indices_of_changes = np.concatenate([changes, [len(original_qt)]])

    start_index = 0
    for end_index in indices_of_changes:
        mid_point = start_index + (end_index - start_index) // 2
        value_at_mid_point = original_qt[mid_point]

        expanded_start = int(round((start_index / old_fs) * new_fs))
        expanded_end = int(round((end_index / old_fs) * new_fs))
        expanded_start = max(0, expanded_start)
        expanded_end = min(new_length, expanded_end)
        if expanded_start < expanded_end:
            expanded_qt[expanded_start:expanded_end] = value_at_mid_point
        start_index = end_index

    return expanded_qt
