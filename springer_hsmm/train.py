"""
Training: label frames from annotations, then fit logistic regression B_matrix and total_obs.

Translated from trainBandPiMatricesSpringer.m, trainSpringerSegmentationAlgorithm.m
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from springer_hsmm.features import get_springer_pcg_features
from springer_hsmm.label_states import label_pcg_states
from springer_hsmm.options import default_springer_hsmm_options


def train_band_pi_matrices_springer(
    state_observation_values: list[list[np.ndarray]],
) -> tuple[list[np.ndarray], np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    state_observation_values: list of 4 lists (one per state); each inner list has
    arrays of shape (n_samples, n_features) from each recording.
    Returns B_matrix (list of 4 arrays: each is (n_features+1,) for coef and intercept),
    pi_vector (4,), total_obs_distribution (mean, cov).
    """
    n_states = 4
    pi_vector = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)

    statei_values: list[list[np.ndarray]] = [[] for _ in range(n_states)]
    for rec in state_observation_values:
        for s in range(n_states):
            statei_values[s].append(rec[s])

    all_per_state = [np.vstack(x) if len(x) > 0 else np.zeros((0, 0)) for x in statei_values]
    n_features = all_per_state[0].shape[1] if all_per_state[0].size > 0 else 0
    for a in all_per_state:
        if a.size > 0 and a.shape[1] != n_features:
            n_features = a.shape[1]
            break

    total_observation_sequence = np.vstack([x for x in all_per_state if x.size > 0])
    if total_observation_sequence.size == 0:
        total_obs_mean = np.zeros(n_features)
        total_obs_cov = np.eye(n_features)
    else:
        total_obs_mean = np.mean(total_observation_sequence, axis=0)
        total_obs_cov = np.cov(total_observation_sequence, rowvar=False)
        if total_obs_cov.ndim == 0:
            total_obs_cov = np.array([[total_obs_cov]])
        elif total_obs_cov.shape == (n_features,):
            total_obs_cov = np.diag(np.atleast_1d(total_obs_cov))

    B_matrix: list[np.ndarray] = []
    for state in range(n_states):
        length_of_state = all_per_state[state].shape[0] if all_per_state[state].size > 0 else 0
        length_per_other = (
            length_of_state // (n_states - 1)
            if length_of_state > 0
            else 0
        )
        min_other = min(
            (all_per_state[o].shape[0] for o in range(n_states) if o != state),
            default=0,
        )
        if min_other < length_per_other:
            length_per_other = min_other

        training_X_list = []
        training_y_list = []
        rng = np.random.default_rng()
        for other_state in range(n_states):
            arr = all_per_state[other_state]
            if arr.size == 0:
                continue
            n_samp = arr.shape[0]
            if other_state == state:
                n_select = length_per_other * (n_states - 1)
                n_select = min(n_select, n_samp)
                indices = rng.choice(n_samp, size=n_select, replace=False)
                training_X_list.append(arr[indices])
                training_y_list.append(np.ones(n_select, dtype=np.int32))  # class 1 = this state
            else:
                n_select = min(length_per_other, n_samp)
                indices = rng.choice(n_samp, size=n_select, replace=False)
                training_X_list.append(arr[indices])
                training_y_list.append(np.zeros(n_select, dtype=np.int32))  # class 0 = rest

        if not training_X_list:
            B_matrix.append(np.zeros(n_features + 1))
            continue
        X = np.vstack(training_X_list)
        y = np.concatenate(training_y_list)
        if np.all(y == y[0]) or len(np.unique(y)) < 2:
            B_matrix.append(np.zeros(n_features + 1))
            continue
        clf = LogisticRegression(
            random_state=0,
            max_iter=1000,
            solver="lbfgs",
        )
        clf.fit(X, y)
        # Store [intercept, coef] so we can apply as dot([1, x], B)
        B = np.concatenate([[clf.intercept_[0]], clf.coef_[0]])
        B_matrix.append(B)

    return B_matrix, pi_vector, (total_obs_mean, total_obs_cov)


def train_springer_segmentation_algorithm(
    pcg_list: list[np.ndarray],
    annotations_list: list[tuple[np.ndarray, np.ndarray]],
    fs: float,
    options: dict | None = None,
) -> tuple[list[np.ndarray], np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Train on list of PCG recordings and per-recording (s1_positions, s2_positions) at audio Fs.
    Converts annotation indices to feature-rate indices, then labels frames and fits B/pi/total_obs.
    """
    options = options or default_springer_hsmm_options()
    feat_fs = options["audio_segmentation_Fs"]

    state_observation_values: list[list[np.ndarray]] = []
    for pcg_audio, (s1_audio, s2_audio) in zip(pcg_list, annotations_list):
        pcg_audio = np.asarray(pcg_audio).flatten()
        s1_audio = np.asarray(s1_audio).flatten()
        s2_audio = np.asarray(s2_audio).flatten()

        PCG_Features, features_fs = get_springer_pcg_features(pcg_audio, fs, options)
        envelope = PCG_Features[:, 0]
        T = len(envelope)
        s1_feat = np.round(s1_audio * feat_fs / fs).astype(int)
        s2_feat = np.round(s2_audio * feat_fs / fs).astype(int)
        s1_feat = np.clip(s1_feat, 0, T - 1)
        s2_feat = np.clip(s2_feat, 0, T - 1)

        PCG_states = label_pcg_states(envelope, s1_feat, s2_feat, float(features_fs))

        rec_states: list[np.ndarray] = []
        for state_i in range(4):
            mask = PCG_states == (state_i + 1)
            rec_states.append(PCG_Features[mask])
        state_observation_values.append(rec_states)

    return train_band_pi_matrices_springer(state_observation_values)
