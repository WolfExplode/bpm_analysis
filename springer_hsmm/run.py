"""
Run Springer segmentation: features -> HR -> Viterbi -> expand to audio length.

Translated from runSpringerSegmentationAlgorithm.m
"""

import logging
import time

import numpy as np
from scipy.signal import resample

from springer_hsmm.expand_qt import expand_qt
from springer_hsmm.features import get_springer_pcg_features, springer_shared_preprocess
from springer_hsmm.heart_rate import get_heart_rate_schmidt
from springer_hsmm.options import default_springer_hsmm_options
from springer_hsmm.viterbi import viterbi_decode_pcg_springer

logger = logging.getLogger(__name__)


def run_springer_segmentation_algorithm(
    audio_data: np.ndarray,
    fs: float,
    B_matrix: list[np.ndarray],
    pi_vector: np.ndarray,
    total_observation_distribution: tuple[np.ndarray, np.ndarray],
    options: dict | None = None,
    *,
    return_debug: bool = False,
    return_viz_data: bool = False,
):
    """
    Assign state (1=S1, 2=systole, 3=S2, 4=diastole) to each sample of the audio.
    All preprocessing (25-400 Hz bandpass, spike removal, envelopes) is done inside this pipeline.

    Returns
    -------
    assigned_states : np.ndarray
        Shape (len(audio_data),), dtype int, values 1-4.
    homomorphic_envelope : np.ndarray
        Shape (len(audio_data),). Envelope from the pipeline for display (e.g. plotting).
    debug_info : dict, optional
        If return_debug is True, a dict with heart_rate, systolic_time_interval, features_shape,
        qt_state_counts, assigned_states_state_counts, num_segments, feature_mean, feature_std.
    """
    options = options or default_springer_hsmm_options()
    audio_data = np.asarray(audio_data).flatten()
    n_samples = len(audio_data)
    duration_sec = n_samples / float(fs)
    logger.info(
        "Springer segmentation: audio length=%d samples (%.2f s at %.0f Hz)",
        n_samples, duration_sec, fs,
    )

    # Do bandpass, spike removal, and homomorphic envelope once; reuse for features and HR.
    t0 = time.perf_counter()
    filtered_audio, homomorphic_env = springer_shared_preprocess(audio_data, float(fs))
    logger.info("Springer step 'shared_preprocess' finished in %.3f s", time.perf_counter() - t0)

    t0 = time.perf_counter()
    PCG_Features, features_fs = get_springer_pcg_features(
        audio_data, fs, options,
        pre_filtered=filtered_audio,
        pre_homomorphic=homomorphic_env,
    )
    logger.info(
        "Springer step 'get_springer_pcg_features' finished in %.3f s (features shape %s)",
        time.perf_counter() - t0, getattr(PCG_Features, "shape", None),
    )

    t0 = time.perf_counter()
    heart_rate, systolic_time_interval = get_heart_rate_schmidt(
        audio_data, fs, pre_homomorphic_env=homomorphic_env
    )
    logger.info(
        "Springer step 'get_heart_rate_schmidt' finished in %.3f s (HR=%.1f BPM)",
        time.perf_counter() - t0, heart_rate,
    )

    t0 = time.perf_counter()
    _, _, qt, observation_probs, posteriors = viterbi_decode_pcg_springer(
        PCG_Features,
        pi_vector,
        B_matrix,
        total_observation_distribution,
        heart_rate,
        systolic_time_interval,
        float(features_fs),
    )
    logger.info(
        "Springer step 'viterbi_decode_pcg_springer' finished in %.3f s",
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    assigned_states = expand_qt(qt, float(features_fs), float(fs), len(audio_data))
    logger.info("Springer step 'expand_qt' finished in %.3f s", time.perf_counter() - t0)

    if not return_debug:
        return assigned_states, homomorphic_env

    qt_counts = {}
    for s, c in zip(*np.unique(qt, return_counts=True)):
        qt_counts[int(s)] = int(c)
    assigned_counts = {}
    for s, c in zip(*np.unique(assigned_states, return_counts=True)):
        assigned_counts[int(s)] = int(c)
    # Per-state feature means (do the emissions differ by state?)
    feature_mean_by_state = {}
    for state in (1, 2, 3, 4):
        mask = qt == state
        if np.any(mask):
            feature_mean_by_state[state] = np.mean(PCG_Features[mask], axis=0).tolist()
        else:
            feature_mean_by_state[state] = [float("nan")] * PCG_Features.shape[1]
    debug_info = {
        "heart_rate": heart_rate,
        "systolic_time_interval": systolic_time_interval,
        "features_shape": PCG_Features.shape,
        "qt_state_counts": qt_counts,
        "assigned_states_state_counts": assigned_counts,
        "feature_mean": np.mean(PCG_Features, axis=0).tolist(),
        "feature_std": np.std(PCG_Features, axis=0).tolist(),
        "feature_mean_by_state": feature_mean_by_state,
    }
    if not return_viz_data:
        return assigned_states, homomorphic_env, debug_info
    # Visualization: what downstream steps see (for pipeline debug display).
    n_feat = int(len(homomorphic_env) * 50 / float(fs))
    n_feat = max(1, min(n_feat, len(homomorphic_env)))
    envelope_50hz = resample(homomorphic_env, n_feat)
    time_50hz = np.arange(n_feat) / 50.0
    viz_data = {
        "filtered_audio": np.asarray(filtered_audio, dtype=np.float64),
        "homomorphic_env": np.asarray(homomorphic_env, dtype=np.float64),
        "envelope_50hz": np.asarray(envelope_50hz, dtype=np.float64),
        "time_50hz": np.asarray(time_50hz, dtype=np.float64),
        "features_50hz": np.asarray(PCG_Features, dtype=np.float64),
        "emissions_50hz": np.asarray(observation_probs, dtype=np.float64),
        "posteriors_50hz": np.asarray(posteriors, dtype=np.float64),
        "qt_50hz": np.asarray(qt, dtype=np.int32),
        "sample_rate": float(fs),
    }
    return assigned_states, homomorphic_env, debug_info, viz_data
