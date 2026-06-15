"""Convert the paper's pretrained MATLAB Springer model to the port's .npz format.

Reads the three .mat files shipped with the original release:
  Springer_B_matrix.mat              -> 4 logistic-regression coef vectors (intercept + 4 features)
  Springer_pi_vector.mat             -> initial state distribution (4,)
  Springer_total_obs_distribution.mat-> (mean (4,), cov (4,4)) over pooled training obs

and writes springer_pretrained.npz via springer_hsmm.model_io.save_springer_model.

This is the exact model behind the ~97.4% CirCor reference. Features are 4-D, so the
segmentation must run with include_wavelet_feature=True (the port's default).
"""

import os

import numpy as np
import scipy.io as sio

from springer_hsmm.model_io import save_springer_model

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAT_DIR = os.path.join(
    _HERE,
    "logistic-regression-hsmm-based-heart-sound-segmentation-1.0",
    "cristhian_potes-204",
)
_OUT = os.path.join(_HERE, "springer_pretrained.npz")


def main() -> None:
    b_raw = sio.loadmat(os.path.join(_MAT_DIR, "Springer_B_matrix.mat"))["Springer_B_matrix"]
    pi_raw = sio.loadmat(os.path.join(_MAT_DIR, "Springer_pi_vector.mat"))["Springer_pi_vector"]
    tod_raw = sio.loadmat(
        os.path.join(_MAT_DIR, "Springer_total_obs_distribution.mat")
    )["Springer_total_obs_distribution"]

    # B: (1,4) object cells, each (5,1) = [intercept, c1, c2, c3, c4]. Flatten to (5,).
    B_matrix = [np.asarray(b_raw[0, i]).ravel().astype(np.float64) for i in range(4)]
    assert all(len(b) == 5 for b in B_matrix), [len(b) for b in B_matrix]

    pi_vector = np.asarray(pi_raw).ravel().astype(np.float64)
    assert pi_vector.shape == (4,), pi_vector.shape

    total_obs_mean = np.asarray(tod_raw[0, 0]).ravel().astype(np.float64)
    total_obs_cov = np.asarray(tod_raw[1, 0]).astype(np.float64)
    assert total_obs_mean.shape == (4,), total_obs_mean.shape
    assert total_obs_cov.shape == (4, 4), total_obs_cov.shape

    save_springer_model(_OUT, B_matrix, pi_vector, (total_obs_mean, total_obs_cov))
    print(f"Wrote {_OUT}")
    print(f"  B_matrix: 4 x {[len(b) for b in B_matrix]}")
    print(f"  pi_vector: {pi_vector}")
    print(f"  total_obs_mean: {total_obs_mean}")


if __name__ == "__main__":
    main()
