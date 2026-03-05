"""
Duration-dependent Viterbi decoding for the 4-state HSMM.

Translated from viterbiDecodePCG_Springer.m
Supports time-varying heart rate: pass heart_rate as np.ndarray (length T) for per-frame HR.
"""

import numpy as np
from scipy.stats import multivariate_normal

from springer_hsmm.durations import get_duration_distributions

# Minimum HR (BPM) used when computing max_duration and when clamping time-varying HR
_HR_MIN_BPM = 20.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _observation_probs(
    observation_sequence: np.ndarray,
    B_matrix: list[np.ndarray],
    pi_vector: np.ndarray,
    total_obs_mean: np.ndarray,
    total_obs_cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Emission P(obs|state) from logistic P(state|obs) and Bayes.
    Also returns posteriors P(state|obs) for debugging (should be high for correct state at each time).
    """
    T, n_feat = observation_sequence.shape
    N = 4
    obs_probs = np.zeros((T, N))
    posteriors = np.zeros((T, N))  # P(state n | obs_t) from LR
    for n in range(N):
        B = B_matrix[n]
        if B is None or len(B) == 0:
            obs_probs[:, n] = 1.0 / N
            posteriors[:, n] = 1.0 / N
            continue
        intercept = B[0]
        coef = B[1:]
        if len(coef) != n_feat:
            coef = np.resize(coef, n_feat)
        logit = observation_sequence @ coef + intercept
        pihat = _sigmoid(logit)  # P(state n | obs)
        posteriors[:, n] = pihat
        for t in range(T):
            try:
                Po = multivariate_normal.pdf(
                    observation_sequence[t], mean=total_obs_mean, cov=total_obs_cov
                )
            except Exception:
                Po = 1e-10
            Po = max(Po, 1e-300)
            obs_probs[t, n] = (pihat[t] * Po) / max(pi_vector[n], 1e-10)
    return obs_probs, posteriors


def _build_duration_probs_for_hr(
    heart_rate: float,
    systolic_time: float,
    N: int,
    max_duration_D: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (duration_probs (N, max_duration_D+1), duration_sum (N,)) for one HR value."""
    hr = max(float(heart_rate), 1e-6)
    (
        d_distributions,
        min_S1,
        max_S1,
        min_S2,
        max_S2,
        min_systole,
        max_systole,
        min_diastole,
        max_diastole,
    ) = get_duration_distributions(hr, systolic_time)

    duration_probs = np.zeros((N, max_duration_D + 1))
    for state_j in range(N):
        mean_d, var_d = d_distributions[state_j]
        std_d = np.sqrt(max(var_d, 1e-10))
        for d in range(1, max_duration_D + 1):
            if state_j == 0:
                if d < min_S1 or d > max_S1:
                    duration_probs[state_j, d] = np.finfo(float).tiny
                else:
                    duration_probs[state_j, d] = np.exp(
                        -0.5 * ((d - mean_d) ** 2) / var_d
                    ) / (std_d * np.sqrt(2 * np.pi))
            elif state_j == 1:
                if d < min_systole or d > max_systole:
                    duration_probs[state_j, d] = np.finfo(float).tiny
                else:
                    duration_probs[state_j, d] = np.exp(
                        -0.5 * ((d - mean_d) ** 2) / var_d
                    ) / (std_d * np.sqrt(2 * np.pi))
            elif state_j == 2:
                if d < min_S2 or d > max_S2:
                    duration_probs[state_j, d] = np.finfo(float).tiny
                else:
                    duration_probs[state_j, d] = np.exp(
                        -0.5 * ((d - mean_d) ** 2) / var_d
                    ) / (std_d * np.sqrt(2 * np.pi))
            else:
                if d < min_diastole or d > max_diastole:
                    duration_probs[state_j, d] = np.finfo(float).tiny
                else:
                    duration_probs[state_j, d] = np.exp(
                        -0.5 * ((d - mean_d) ** 2) / var_d
                    ) / (std_d * np.sqrt(2 * np.pi))
    duration_sum = np.sum(duration_probs, axis=1, keepdims=True)
    duration_sum = np.maximum(duration_sum, 1e-300)
    return duration_probs, duration_sum.squeeze(axis=1)


def viterbi_decode_pcg_springer(
    observation_sequence: np.ndarray,
    pi_vector: np.ndarray,
    B_matrix: list[np.ndarray],
    total_obs_distribution: tuple[np.ndarray, np.ndarray],
    heart_rate: float | np.ndarray,
    systolic_time: float,
    Fs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extended Viterbi for duration-dependent HSMM. Returns delta, psi, qt, observation_probs, posteriors.
    qt is the state sequence 1..4 for frames 0..T-1.
    observation_probs has shape (T, 4): per-frame emission b_j(O_t) = P(O_t|state j).
    posteriors has shape (T, 4): P(state j|O_t) from LR, for debug (high when that state is happening).

    heart_rate : float or np.ndarray
        If float, single global HR (BPM) for the whole recording.
        If np.ndarray of length T (number of feature frames), per-frame HR for time-varying duration model.
    """
    total_obs_mean, total_obs_cov = total_obs_distribution
    if total_obs_cov.ndim == 1:
        total_obs_cov = np.diag(np.atleast_1d(total_obs_cov))
    T, _ = observation_sequence.shape
    N = 4

    hr_is_array = isinstance(heart_rate, np.ndarray) and heart_rate.ndim >= 1
    if hr_is_array:
        hr_flat = np.asarray(heart_rate).flatten()
        if hr_flat.size != T:
            hr_is_array = False
            heart_rate = float(np.clip(np.nanmean(hr_flat), _HR_MIN_BPM, 300.0))
        else:
            hr_at_frame = np.clip(hr_flat.astype(np.float64), _HR_MIN_BPM, 300.0)
            hr_min = float(np.nanmin(hr_at_frame))
            heart_rate_scalar = max(hr_min, _HR_MIN_BPM)

    if not hr_is_array:
        heart_rate_scalar = max(float(heart_rate), _HR_MIN_BPM)

    max_duration_D = int(round((60.0 / heart_rate_scalar) * Fs))
    max_duration_D = max(max_duration_D, 10)

    observation_probs, posteriors = _observation_probs(
        observation_sequence, B_matrix, pi_vector, total_obs_mean, total_obs_cov
    )
    observation_probs = np.clip(observation_probs, 1e-300, None)

    if hr_is_array:
        duration_probs_by_frame = np.zeros((T, N, max_duration_D + 1), dtype=np.float64)
        duration_sum_by_frame = np.zeros((T, N), dtype=np.float64)
        for i in range(T):
            dp, ds = _build_duration_probs_for_hr(
                hr_at_frame[i], systolic_time, N, max_duration_D
            )
            duration_probs_by_frame[i] = dp
            duration_sum_by_frame[i] = ds
    else:
        duration_probs, duration_sum_1d = _build_duration_probs_for_hr(
            heart_rate_scalar, systolic_time, N, max_duration_D
        )
        duration_sum = np.maximum(duration_sum_1d, 1e-300)

    size_delta = T + max_duration_D - 1
    delta = np.full((size_delta, N), -np.inf)
    psi = np.zeros((size_delta, N), dtype=np.int32)
    psi_duration = np.zeros((size_delta, N), dtype=np.int32)

    delta[0, :] = np.log(pi_vector + 1e-300) + np.log(observation_probs[0, :])
    psi[0, :] = -1

    a_prev = [3, 0, 1, 2]  # previous state (0-based) for state j

    for t in range(1, size_delta):
        for j in range(N):
            for d in range(1, max_duration_D + 1):
                start_t = t - d
                if start_t < 0:
                    start_t = 0
                if start_t > T - 1:
                    start_t = T - 1
                end_t = min(t, T - 1)
                if start_t > end_t:
                    continue

                prev_j = a_prev[j]
                max_delta = delta[start_t, prev_j]

                prod_prob = np.prod(observation_probs[start_t : end_t + 1, j])
                if prod_prob <= 0:
                    prod_prob = np.finfo(float).tiny
                emission_probs = np.log(prod_prob)

                if hr_is_array:
                    center = max(0, min(T - 1, t - d // 2))
                    dur_prob = (
                        duration_probs_by_frame[center, j, d]
                        / duration_sum_by_frame[center, j]
                    )
                else:
                    dur_prob = duration_probs[j, d] / duration_sum[j]
                if dur_prob <= 0:
                    dur_prob = np.finfo(float).tiny
                delta_temp = max_delta + emission_probs + np.log(dur_prob)

                if delta_temp > delta[t, j]:
                    delta[t, j] = delta_temp
                    psi[t, j] = prev_j
                    psi_duration[t, j] = d

    temp_delta = delta[T:]
    if temp_delta.size == 0:
        qt = np.ones(T, dtype=np.int32)
        return delta, psi, qt, observation_probs, posteriors
    idx_flat = np.argmax(temp_delta)
    row_in_temp = idx_flat // N
    pos = T + row_in_temp
    state = idx_flat % N

    qt = np.zeros(T, dtype=np.int32)
    offset = pos
    count = 0
    while offset >= 0 and state >= 0 and count <= 1000:
        dur = psi_duration[offset, state]
        onset = offset - dur + 1
        start_qt = max(0, onset)
        end_qt = min(offset + 1, T)
        if start_qt < end_qt:
            qt[start_qt:end_qt] = state + 1
        if onset <= 1:
            if onset < 1:
                onset = 0
            break
        state = psi[offset, state]
        offset = onset - 1
        count += 1

    return delta, psi, qt, observation_probs, posteriors
