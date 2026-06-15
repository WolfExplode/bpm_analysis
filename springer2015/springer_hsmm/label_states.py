"""
Assign state labels (1=S1, 2=systole, 3=S2, 4=diastole) to each frame from R-peak and end-T positions.

Translated from labelPCGStates.m.
Expects envelope at feature rate and s1_positions, s2_positions as indices in that envelope (0-based).
"""

import numpy as np


def label_pcg_states(
    envelope: np.ndarray,
    s1_positions: np.ndarray,
    s2_positions: np.ndarray,
    sampling_frequency: float,
) -> np.ndarray:
    """
    Assign state 1-4 to each sample of the envelope.

    State 1 = S1, 2 = systole, 3 = S2, 4 = diastole.
    s1_positions, s2_positions are indices into envelope (0-based).
    """
    envelope = np.asarray(envelope).flatten()
    n = len(envelope)
    states = np.zeros(n, dtype=np.int32)

    mean_S1 = 0.122 * sampling_frequency
    std_S1 = 0.022 * sampling_frequency
    mean_S2 = 0.092 * sampling_frequency
    std_S2 = 0.022 * sampling_frequency

    for i in range(len(s1_positions)):
        s1 = int(s1_positions[i])
        upper = min(n, int(round(s1 + mean_S1)))
        lo = max(0, s1)
        states[lo:upper] = 1

    for i in range(len(s2_positions)):
        s2 = int(s2_positions[i])
        lower_bound = max(0, s2 - int(np.floor(mean_S2 + std_S2)))
        upper_bound = min(n, int(np.ceil(s2 + np.floor(mean_S2 + std_S2))))
        search_window = envelope[lower_bound:upper_bound].copy()
        mask = states[lower_bound:upper_bound] != 1
        search_window[~mask] = -np.inf
        idx_in_window = int(np.argmax(search_window))
        S2_index = min(n - 1, lower_bound + idx_in_window)

        lo3 = max(0, int(np.ceil(S2_index - (mean_S2 / 2))))
        hi3 = min(n, int(np.ceil(S2_index + (mean_S2 / 2))))
        states[lo3:hi3] = 3

        # Diastole from end of S2 to next S1
        start_diastole = min(n, int(np.ceil(S2_index + (mean_S2 + 0 * std_S2) / 2)))
        diffs = s1_positions.astype(float) - s2
        diffs[diffs <= 0] = np.inf
        if np.all(np.isinf(diffs)):
            end_pos = n
        else:
            end_pos = int(s1_positions[np.argmin(diffs)]) - 1
            end_pos = min(end_pos, n)
        if start_diastole < end_pos:
            states[start_diastole:end_pos] = 4

    # First section before first definite state
    nonzero = np.where(states != 0)[0]
    if len(nonzero) > 0:
        first_def = nonzero[0]
        if first_def > 0:
            if states[first_def] == 1:
                states[:first_def] = 4
            elif states[first_def] == 3:
                states[:first_def] = 2
        last_def = nonzero[-1]
        if last_def < n - 1 or last_def == n - 1:
            if states[last_def] == 1:
                states[last_def:] = 2
            elif states[last_def] == 3:
                states[last_def:] = 4

    states[states == 0] = 2
    return states
