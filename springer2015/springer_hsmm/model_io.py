"""
Save and load Springer model (B_matrix, pi_vector, total_obs_distribution).
"""

from typing import Any

import numpy as np


def save_springer_model(
    path: str,
    B_matrix: list[np.ndarray],
    pi_vector: np.ndarray,
    total_obs_distribution: tuple[np.ndarray, np.ndarray],
) -> None:
    """
    Save model to .npz. B_matrix is list of 4 arrays (each coef+intercept).
    total_obs_distribution is (mean, cov).
    """
    total_obs_mean, total_obs_cov = total_obs_distribution
    # npz doesn't support list of arrays directly; save as list of same-length arrays or pad
    max_len = max(len(B) for B in B_matrix)
    B_stack = np.array([np.pad(B, (0, max_len - len(B))) if len(B) < max_len else B for B in B_matrix])
    np.savez(
        path,
        B_matrix=B_stack,
        pi_vector=pi_vector,
        total_obs_mean=total_obs_mean,
        total_obs_cov=total_obs_cov,
    )


def load_springer_model(path: str) -> dict[str, Any]:
    """
    Load .npz and return dict with B_matrix (list of 4 arrays), pi_vector,
    total_obs_distribution (mean, cov). Trims zero-padding from B_matrix rows.
    """
    data = np.load(path, allow_pickle=False)
    B_stack = data["B_matrix"]
    n_features_plus_one = B_stack.shape[1]
    B_matrix = [B_stack[i] for i in range(B_stack.shape[0])]
    pi_vector = data["pi_vector"]
    total_obs_mean = data["total_obs_mean"]
    total_obs_cov = data["total_obs_cov"]
    if total_obs_cov.ndim == 0:
        total_obs_cov = np.array([[total_obs_cov]])
    return {
        "B_matrix": B_matrix,
        "pi_vector": pi_vector,
        "total_obs_distribution": (total_obs_mean, total_obs_cov),
    }
