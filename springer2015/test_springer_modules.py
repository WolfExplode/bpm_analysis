"""Unit tests for springer2015/springer_hsmm modules."""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from springer_hsmm.signal import (
    butterworth_high_pass_filter,
    butterworth_low_pass_filter,
    hilbert_envelope,
    homomorphic_envelope_with_hilbert,
    normalise_signal,
    schmidt_spike_removal,
)
from springer_hsmm.features import get_dwt, get_psd_feature_springer_hmm, get_springer_pcg_features
from springer_hsmm.durations import get_duration_distributions
from springer_hsmm.expand_qt import expand_qt
from springer_hsmm.viterbi import viterbi_decode_pcg_springer
from springer_hsmm.options import default_springer_hsmm_options


FS = 1000
SINE_1HZ = np.sin(2 * np.pi * 1.0 * np.arange(FS * 5) / FS)  # 5 s, 1 Hz
WHITE = np.random.default_rng(42).standard_normal(FS * 5)  # immutable after init


# ---------------------------------------------------------------------------
# signal.py
# ---------------------------------------------------------------------------

def test_normalise_signal_zero_mean_unit_std():
    out = normalise_signal(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    np.testing.assert_allclose(out.mean(), 0.0, atol=1e-10)
    np.testing.assert_allclose(out.std(), 1.0, atol=1e-10)


def test_normalise_signal_constant_returns_zero():
    out = normalise_signal(np.ones(100))
    np.testing.assert_array_equal(out, np.zeros(100))


def test_butterworth_lp_removes_high_freq():
    t = np.arange(FS * 2) / FS
    low = np.sin(2 * np.pi * 10 * t)
    high = np.sin(2 * np.pi * 400 * t)
    out = butterworth_low_pass_filter(low + high, 2, 100, FS)
    # Skip first/last 50 samples — filtfilt has edge transients there
    np.testing.assert_allclose(out[50:-50], low[50:-50], atol=0.05)


def test_butterworth_hp_removes_dc():
    sig = np.ones(FS * 2) + 0.1 * np.sin(2 * np.pi * 200 * np.arange(FS * 2) / FS)
    out = butterworth_high_pass_filter(sig, 2, 25, FS)
    assert abs(out.mean()) < 0.05


def test_butterworth_filters_preserve_length():
    sig = WHITE[:500]
    assert len(butterworth_low_pass_filter(sig, 2, 400, FS)) == 500
    assert len(butterworth_high_pass_filter(sig, 2, 25, FS)) == 500


def test_schmidt_spike_removal_zeroes_spike():
    # Non-zero background required: median_maa=0 on an all-zero signal → infinite loop
    t = np.arange(FS * 2) / FS
    sig = np.sin(2 * np.pi * 80 * t) * 0.1
    sig[300] = 100.0  # spike >> 3 * median window amplitude
    out = schmidt_spike_removal(sig.copy(), FS)
    assert abs(out[300]) < 1.0


def test_schmidt_spike_removal_preserves_length():
    assert len(schmidt_spike_removal(WHITE, FS)) == len(WHITE)


def test_homomorphic_envelope_positive_same_length():
    out = homomorphic_envelope_with_hilbert(SINE_1HZ, FS)
    assert len(out) == len(SINE_1HZ)
    assert np.all(out > 0)


def test_hilbert_envelope_positive_same_length():
    out = hilbert_envelope(SINE_1HZ)
    assert len(out) == len(SINE_1HZ)
    assert np.all(out >= 0)


# ---------------------------------------------------------------------------
# features.py
# ---------------------------------------------------------------------------

def test_get_dwt_output_length_matches_input():
    sig = WHITE[:FS]
    cD, cA = get_dwt(sig, level=3)
    assert len(cD) == len(sig)
    assert len(cA) == len(sig)


def test_get_psd_feature_shape_and_positive():
    psd = get_psd_feature_springer_hmm(WHITE, FS, 40, 60)
    assert psd.ndim == 1
    assert len(psd) > 0
    assert np.all(psd >= 0)


def test_get_springer_pcg_features_shape():
    opts = default_springer_hsmm_options()
    feat_fs = opts["audio_segmentation_Fs"]  # 50
    duration_sec = 3.0
    sig = np.sin(2 * np.pi * 80 * np.arange(int(FS * duration_sec)) / FS)
    features, out_fs = get_springer_pcg_features(sig, FS, opts)
    assert out_fs == feat_fs
    expected_T = int(len(sig) * feat_fs / FS)
    assert abs(features.shape[0] - expected_T) <= 2  # allow ±2 frames rounding
    assert features.shape[1] == 4  # hom + hilbert + psd + wavelet


def test_get_springer_pcg_features_no_wavelet():
    opts = {**default_springer_hsmm_options(), "include_wavelet_feature": False}
    sig = WHITE[:FS * 2]
    features, _ = get_springer_pcg_features(sig, FS, opts)
    assert features.shape[1] == 3


# ---------------------------------------------------------------------------
# durations.py
# ---------------------------------------------------------------------------

def test_duration_distributions_sensible_values():
    dists, min_S1, max_S1, min_S2, max_S2, min_sys, max_sys, min_dia, max_dia = \
        get_duration_distributions(75.0, 0.3)
    mean_S1, var_S1 = dists[0]
    mean_S2, var_S2 = dists[2]
    assert min_S1 < mean_S1 < max_S1
    assert min_S2 < mean_S2 < max_S2
    assert min_sys < dists[1][0] < max_sys
    assert min_dia < dists[3][0] < max_dia


def test_duration_distributions_higher_hr_shorter_diastole():
    _, *_, min_dia_60, max_dia_60 = get_duration_distributions(60.0, 0.3)
    _, *_, min_dia_100, max_dia_100 = get_duration_distributions(100.0, 0.3)
    # Higher HR → shorter RR → shorter diastole
    assert max_dia_100 < max_dia_60


# ---------------------------------------------------------------------------
# expand_qt.py
# ---------------------------------------------------------------------------

def test_expand_qt_identity_same_fs():
    qt = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32)
    out = expand_qt(qt, old_fs=50.0, new_fs=50.0, new_length=len(qt))
    assert len(out) == len(qt)
    # All four states should be present
    assert set(np.unique(out)).issuperset({1, 2, 3, 4})


def test_expand_qt_output_length():
    qt = np.array([1, 2, 3, 4] * 5, dtype=np.int32)
    new_len = 200
    out = expand_qt(qt, old_fs=50.0, new_fs=1000.0, new_length=new_len)
    assert len(out) == new_len


def test_expand_qt_state_values_in_range():
    qt = np.array([1, 1, 2, 3, 4, 4] * 10, dtype=np.int32)
    out = expand_qt(qt, old_fs=50.0, new_fs=500.0, new_length=600)
    assert set(np.unique(out[out > 0])).issubset({1, 2, 3, 4})


# ---------------------------------------------------------------------------
# viterbi.py
# ---------------------------------------------------------------------------

def _make_dummy_model(n_features=4):
    """Minimal B_matrix and total_obs_distribution for a test run."""
    rng = np.random.default_rng(7)
    B_matrix = [rng.standard_normal(n_features + 1) for _ in range(4)]
    pi_vector = np.array([0.25, 0.25, 0.25, 0.25])
    total_obs_mean = np.zeros(n_features)
    total_obs_cov = np.eye(n_features)
    return B_matrix, pi_vector, (total_obs_mean, total_obs_cov)


def test_viterbi_decode_output_length_and_values():
    T = 100
    n_features = 4
    obs = np.random.default_rng(11).standard_normal((T, n_features))
    B_matrix, pi_vector, total_obs_dist = _make_dummy_model(n_features)
    _, _, qt, obs_probs, posteriors = viterbi_decode_pcg_springer(
        obs, pi_vector, B_matrix, total_obs_dist,
        heart_rate=75.0, systolic_time=0.3, Fs=50.0,
    )
    assert len(qt) == T
    assert set(np.unique(qt)).issubset({0, 1, 2, 3, 4})  # 0 = unset, 1-4 = states
    assert obs_probs.shape == (T, 4)
    assert posteriors.shape == (T, 4)


def test_viterbi_decode_all_states_assigned():
    T = 300
    n_features = 4
    rng = np.random.default_rng(99)
    obs = rng.standard_normal((T, n_features))
    B_matrix, pi_vector, total_obs_dist = _make_dummy_model(n_features)
    _, _, qt, _, _ = viterbi_decode_pcg_springer(
        obs, pi_vector, B_matrix, total_obs_dist,
        heart_rate=75.0, systolic_time=0.3, Fs=50.0,
    )
    states_found = set(np.unique(qt[qt > 0]))
    assert len(states_found) >= 2, f"expected multiple states, got {states_found}"


def test_viterbi_cyclic_state_order():
    """States must follow cyclic order 1→2→3→4→1 (no backwards jumps)."""
    T = 200
    n_features = 4
    obs = np.random.default_rng(13).standard_normal((T, n_features))
    B_matrix, pi_vector, total_obs_dist = _make_dummy_model(n_features)
    _, _, qt, _, _ = viterbi_decode_pcg_springer(
        obs, pi_vector, B_matrix, total_obs_dist,
        heart_rate=75.0, systolic_time=0.3, Fs=50.0,
    )
    valid = qt[qt > 0]
    for a, b in zip(valid[:-1], valid[1:]):
        if a != b:
            # Valid transitions: 1→2, 2→3, 3→4, 4→1
            assert (b - a) % 4 == 1, f"illegal transition {a}→{b}"
