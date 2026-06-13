"""Viterbi decoder core + observation mapping (Pass 4 building blocks)."""
import numpy as np

import viterbi
from viterbi import (
    N_STATES,
    STATE_S1,
    STATE_SYSTOLE,
    STATE_S2,
    STATE_DIASTOLE,
)


def test_run_viterbi_recovers_obvious_state_sequence():
    # 3 timesteps, 2 states, no transition constraints.
    # Emissions strongly prefer state 0, then 1, then 0.
    eps = np.log(1e-6)
    strong = np.log(1.0)
    log_obs = np.array([
        [strong, eps],
        [eps, strong],
        [strong, eps],
    ])
    log_trans = np.zeros((2, 2))  # all transitions equally allowed (log 1 = 0)
    log_prior = np.zeros(2)
    path = viterbi.run_viterbi(log_obs, log_trans, log_prior)
    assert list(path) == [0, 1, 0]


def test_run_viterbi_respects_forbidden_transitions():
    # State 1 can never be entered (-inf into it); decoder must stay in state 0.
    eps = np.log(1e-6)
    strong = np.log(1.0)
    log_obs = np.array([
        [eps, strong],   # emission wants state 1...
        [eps, strong],
    ])
    log_trans = np.array([
        [0.0, float("-inf")],   # 0 -> 1 forbidden
        [0.0, float("-inf")],   # 1 -> 1 forbidden
    ])
    log_prior = np.array([0.0, float("-inf")])  # must start in state 0
    path = viterbi.run_viterbi(log_obs, log_trans, log_prior)
    assert list(path) == [0, 0]


def test_build_4state_log_obs_rows_normalized():
    emissions = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
    ])
    params = {"pass4_emission_weight": 0.7}
    log_obs = viterbi._build_4state_log_obs(emissions, params)
    assert log_obs.shape == (2, N_STATES)
    # exp(log_obs) rows sum to 1
    probs = np.exp(log_obs)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    # systole and diastole both map to P_Noise column -> equal masses
    np.testing.assert_allclose(log_obs[:, STATE_SYSTOLE], log_obs[:, STATE_DIASTOLE])


def test_build_transition_matrix_topology():
    params = dict(__import__("config").DEFAULT_PARAMS)
    T = viterbi.build_transition_matrix(bpm=80.0, sample_rate=600, params=params)
    assert T.shape == (N_STATES, N_STATES)
    # Allowed forward + self transitions are finite; everything else is -inf.
    allowed = {
        (STATE_S1, STATE_S1), (STATE_S1, STATE_SYSTOLE),
        (STATE_SYSTOLE, STATE_SYSTOLE), (STATE_SYSTOLE, STATE_S2),
        (STATE_S2, STATE_S2), (STATE_S2, STATE_DIASTOLE),
        (STATE_DIASTOLE, STATE_DIASTOLE), (STATE_DIASTOLE, STATE_S1),
    }
    for i in range(N_STATES):
        for j in range(N_STATES):
            if (i, j) in allowed:
                assert np.isfinite(T[i, j]), (i, j)
            else:
                assert T[i, j] == float("-inf"), (i, j)
    # Each row is a valid distribution: exp sums to ~1.
    np.testing.assert_allclose(np.exp(T).sum(axis=1), 1.0, atol=1e-9)
