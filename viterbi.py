"""
viterbi.py — Pass 4: holistic Viterbi sequence decoder.

4-state left-to-right Bakis HMM:
  State 0: S1       (first heart sound transient)
  State 1: systole  (S1→S2 interval)
  State 2: S2       (second heart sound transient)
  State 3: diastole (S2→S1 interval; longest phase)

Transition topology (no reverse transitions):
  S1 → S1  (self-loop while in S1 event)
  S1 → systole
  systole → systole
  systole → S2
  S2 → S2
  S2 → diastole
  diastole → diastole
  diastole → S1  (next cardiac cycle)

Self-loop probabilities are derived from expected state durations so that
the decoder implicitly models duration without an explicit duration model.

Emissions come from analysis_data["pass3_emissions"] (shape T×3):
  column 0 → P(S1|t)
  column 1 → P(S2|t)
  column 2 → P(Noise|t)

Columns 1 and 3 (systole and diastole) both map to P(Noise|t) since those
are transition/silence phases; see _build_4state_log_obs().

Entry point: run_pass4_viterbi()
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

from confidence_engine import calculate_bpm_intervals
from config import param


# ─────────────────────────────────────────────────────────────────────────────
# State indices
# ─────────────────────────────────────────────────────────────────────────────

STATE_S1       = 0
STATE_SYSTOLE  = 1
STATE_S2       = 2
STATE_DIASTOLE = 3
N_STATES       = 4


# ─────────────────────────────────────────────────────────────────────────────
# Transition matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_transition_matrix(bpm: float, sample_rate: int, params: Dict) -> np.ndarray:
    """
    Build a 4×4 left-to-right transition matrix for the cardiac cycle HMM.

    Self-loop probability for each state is derived from the expected state
    duration (in samples): p_self = 1 - 1/duration_samples.  The complement
    probability is placed on the single allowed forward transition.

    Returns log-probability matrix (4×4) in natural log.  -inf for
    disallowed transitions.
    """
    ivs = calculate_bpm_intervals(bpm, params)
    sl_weight = float(param(params, "pass4_transition_self_loop_weight"))

    s1_dur_samp  = max(2, int(round(float(ivs.get("s1_nominal",    0.040)) * sample_rate)))
    sys_dur_samp = max(2, int(round(float(ivs.get("s1_s2_nominal", 0.300)) * sample_rate)))
    s2_dur_samp  = max(2, int(round(float(ivs.get("s2_nominal",    0.030)) * sample_rate)))
    dia_dur_samp = max(2, int(round(float(ivs.get("s2_s1_nominal", 0.400)) * sample_rate)))

    def _self_loop(dur: int) -> float:
        """p_self proportional to expected duration, clamped by sl_weight.

        dur >= 2 (enforced above) so raw = 1 - 1/dur >= 0.5; the old
        `raw / max(raw, 1e-10) * raw` was a no-op that reduced to `* raw`.
        """
        raw = 1.0 - 1.0 / dur
        return float(np.clip(sl_weight * raw, 0.50, 0.999))

    p_s1_self   = _self_loop(s1_dur_samp)
    p_sys_self  = _self_loop(sys_dur_samp)
    p_s2_self   = _self_loop(s2_dur_samp)
    p_dia_self  = _self_loop(dia_dur_samp)

    _neg_inf = float("-inf")
    T = np.full((N_STATES, N_STATES), _neg_inf, dtype=np.float64)

    # S1 row
    T[STATE_S1,       STATE_S1]       = np.log(p_s1_self)
    T[STATE_S1,       STATE_SYSTOLE]  = np.log(1.0 - p_s1_self)
    # systole row
    T[STATE_SYSTOLE,  STATE_SYSTOLE]  = np.log(p_sys_self)
    T[STATE_SYSTOLE,  STATE_S2]       = np.log(1.0 - p_sys_self)
    # S2 row
    T[STATE_S2,       STATE_S2]       = np.log(p_s2_self)
    T[STATE_S2,       STATE_DIASTOLE] = np.log(1.0 - p_s2_self)
    # diastole row
    T[STATE_DIASTOLE, STATE_DIASTOLE] = np.log(p_dia_self)
    T[STATE_DIASTOLE, STATE_S1]       = np.log(1.0 - p_dia_self)

    return T


# ─────────────────────────────────────────────────────────────────────────────
# Observation mapping
# ─────────────────────────────────────────────────────────────────────────────

def _build_4state_log_obs(emissions: np.ndarray, params: Dict) -> np.ndarray:
    """
    Map 3-column emissions [P_S1, P_S2, P_Noise] to 4-state log-obs matrix.

    State–emission mapping:
      S1       → emissions[:, 0]   (P_S1)
      systole  → emissions[:, 2]   (P_Noise)
      S2       → emissions[:, 1]   (P_S2)
      diastole → emissions[:, 2]   (P_Noise)

    The emission_weight param blends spectral evidence with a uniform prior.
    """
    T = len(emissions)
    w = float(np.clip(param(params, "pass4_emission_weight"), 0.0, 1.0))
    uniform = 0.25  # uniform over 4 states → 0.25 each

    obs = np.zeros((T, N_STATES), dtype=np.float64)
    obs[:, STATE_S1]       = w * emissions[:, 0] + (1.0 - w) * uniform
    obs[:, STATE_SYSTOLE]  = w * emissions[:, 2] + (1.0 - w) * uniform
    obs[:, STATE_S2]       = w * emissions[:, 1] + (1.0 - w) * uniform
    obs[:, STATE_DIASTOLE] = w * emissions[:, 2] + (1.0 - w) * uniform

    # Normalize rows so they sum to 1, then take log.
    row_sums = obs.sum(axis=1, keepdims=True) + 1e-30
    log_obs  = np.log(obs / row_sums)
    return log_obs


# ─────────────────────────────────────────────────────────────────────────────
# Core Viterbi decoder
# ─────────────────────────────────────────────────────────────────────────────

def run_viterbi(
    log_obs: np.ndarray,        # shape (T, N_STATES)  log emission probs
    log_trans: np.ndarray,      # shape (N_STATES, N_STATES)
    log_prior: np.ndarray,      # shape (N_STATES,)
) -> np.ndarray:                # shape (T,) best state sequence
    """
    Standard log-domain Viterbi algorithm.

    All inputs are expected to be in natural-log probability space.
    -inf entries in log_trans correctly prevent disallowed transitions.
    """
    T, K = log_obs.shape
    viterbi  = np.full((T, K), float("-inf"), dtype=np.float64)
    backptr  = np.zeros((T, K), dtype=np.int32)

    viterbi[0] = log_prior + log_obs[0]

    for t in range(1, T):
        for s in range(K):
            trans_scores = viterbi[t - 1] + log_trans[:, s]
            best_prev    = int(np.argmax(trans_scores))
            viterbi[t, s] = trans_scores[best_prev] + log_obs[t, s]
            backptr[t, s]  = best_prev

    # Backtrack
    path = np.zeros(T, dtype=np.int32)
    path[T - 1] = int(np.argmax(viterbi[T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]

    return path


# ─────────────────────────────────────────────────────────────────────────────
# S1 peak extraction from Viterbi path
# ─────────────────────────────────────────────────────────────────────────────

def _extract_s1_peaks_from_path(
    path: np.ndarray,
    audio_envelope: np.ndarray,
    sample_rate: int,
    params: Dict,
) -> np.ndarray:
    """
    Find each contiguous run of S1 state in path and pick the sample with the
    highest audio_envelope value within that run as the S1 peak position.
    """
    n = len(path)
    peaks: List[int] = []
    i = 0
    while i < n:
        if path[i] != STATE_S1:
            i += 1
            continue
        j = i
        while j < n and path[j] == STATE_S1:
            j += 1
        # Run is [i, j); pick argmax amplitude within that run.
        seg = audio_envelope[i:j]
        if len(seg) > 0:
            peaks.append(i + int(np.argmax(seg)))
        i = j
    return np.asarray(peaks, dtype=np.int64) if peaks else np.empty(0, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_pass4_viterbi(
    s1_peaks_in: np.ndarray,
    analysis_data: Dict,
    audio_envelope: np.ndarray,
    sample_rate: int,
    params: Dict,
) -> Tuple[np.ndarray, Dict]:
    """
    Run the Pass 4 Viterbi holistic decoder.

    Requires analysis_data["pass3_emissions"] (see archived emissions generator in pass3 archived logic.md).
    Falls back to returning s1_peaks_in unchanged if emissions are unavailable.

    Stores analysis_data["pass4_state_sequence"] (int32 array, length = n_samples)
    and analysis_data["pass4_s1_peaks"] (int64 array).

    Returns (refined_s1_peaks, updated_analysis_data).
    """
    emissions = analysis_data.get("pass3_emissions")
    if emissions is None or not isinstance(emissions, np.ndarray) or emissions.ndim != 2:
        logging.warning(
            "Pass 4 Viterbi: pass3_emissions not available. Returning Pass 3 peaks unchanged."
        )
        return s1_peaks_in, analysis_data

    T = len(emissions)
    if T == 0:
        return s1_peaks_in, analysis_data

    # Build BPM estimate from Pass 3 peaks for the transition matrix.
    bpm = 80.0
    try:
        rr_arr = np.diff(s1_peaks_in.astype(np.float64)) / float(sample_rate)
        rr_arr = rr_arr[np.isfinite(rr_arr) & (rr_arr > 0)]
        if len(rr_arr) > 0:
            fb = float(60.0 / np.median(rr_arr))
            if np.isfinite(fb) and fb > 0:
                bpm = fb
    except Exception:
        pass

    log_trans = build_transition_matrix(bpm, sample_rate, params)

    # Prior: start in diastole (most of the time between beats) or uniform.
    log_prior = np.log(np.array([0.10, 0.05, 0.05, 0.80], dtype=np.float64))

    log_obs = _build_4state_log_obs(emissions, params)

    logging.info(
        "Pass 4 Viterbi: decoding %d samples at BPM=%.1f...", T, bpm,
    )
    path = run_viterbi(log_obs, log_trans, log_prior)

    s1_peaks_out = _extract_s1_peaks_from_path(path, audio_envelope, sample_rate, params)

    if len(s1_peaks_out) < 2:
        logging.warning(
            "Pass 4 Viterbi: only %d S1 peaks extracted — keeping Pass 3 result.",
            len(s1_peaks_out),
        )
        s1_peaks_out = s1_peaks_in

    analysis_data["pass4_state_sequence"] = path.astype(np.int32)
    analysis_data["pass4_s1_peaks"]       = s1_peaks_out

    logging.info(
        "Pass 4 Viterbi: extracted %d S1 peaks from path.", len(s1_peaks_out),
    )
    return s1_peaks_out, analysis_data
