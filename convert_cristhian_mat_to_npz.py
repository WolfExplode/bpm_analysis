"""
Convert cristhian_potes' Springer .mat model files to a single .npz for use with our pipeline.

Place the three .mat files in the cristhian_potes-204 folder (or pass a directory):
  - Springer_B_matrix.mat
  - Springer_pi_vector.mat
  - Springer_total_obs_distribution.mat

Then run from repo root:
  python convert_cristhian_mat_to_npz.py
  python convert_cristhian_mat_to_npz.py path/to/folder

Output: cristhian_potes_model.npz (in the same directory as the .mat files, or cwd).
"""

import argparse
import os
import sys

import numpy as np
from scipy.io import loadmat


def _first_data_key(mat_dict: dict, *candidates: str) -> str:
    """Return the first key that looks like data (not __header__ etc.)."""
    skip = {"__header__", "__version__", "__globals__"}
    for c in candidates:
        if c in mat_dict and c not in skip:
            return c
    for k in mat_dict:
        if not k.startswith("__"):
            return k
    raise KeyError("No data variable found in .mat file")


def load_B_matrix(path: str) -> list[np.ndarray]:
    """Load Springer_B_matrix.mat; return list of 4 arrays [intercept, coef...]."""
    data = loadmat(path, struct_as_record=False, squeeze_me=True)
    key = _first_data_key(data, "B_matrix", "Springer_B_matrix")
    raw = data[key]
    # MATLAB: cell(1,4) -> may be (1,4) object array or (4,) depending on squeeze
    if raw.ndim == 2:
        rows = [raw[i, j] for i in range(raw.shape[0]) for j in range(raw.shape[1])]
    else:
        rows = np.atleast_1d(raw).flatten().tolist()
    B_matrix = []
    for i, row in enumerate(rows[:4]):
        arr = np.asarray(row).flatten().astype(np.float64)
        B_matrix.append(arr)
    if len(B_matrix) < 4:
        raise ValueError(f"B_matrix has {len(B_matrix)} rows, expected 4")
    return B_matrix[:4]


def load_pi_vector(path: str) -> np.ndarray:
    """Load Springer_pi_vector.mat; return 1D array of length 4."""
    data = loadmat(path, struct_as_record=False, squeeze_me=True)
    key = _first_data_key(data, "pi_vector", "Springer_pi_vector")
    pi = np.asarray(data[key]).flatten().astype(np.float64)
    if pi.size != 4:
        raise ValueError(f"pi_vector has size {pi.size}, expected 4")
    return pi[:4]


def load_total_obs_distribution(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load Springer_total_obs_distribution.mat; return (mean, cov)."""
    data = loadmat(path, struct_as_record=False, squeeze_me=False)
    key = _first_data_key(data, "total_obs_distribution", "Springer_total_obs_distribution")
    raw = data[key]
    # MATLAB cell(2,1): {1}=mean, {2}=cov
    if raw.ndim >= 1:
        elems = raw.flatten()
    else:
        elems = [raw]
    mean_arr = np.asarray(elems[0]).flatten().astype(np.float64)
    cov_arr = np.asarray(elems[1]).astype(np.float64)
    if cov_arr.ndim == 1:
        cov_arr = np.diag(np.atleast_1d(cov_arr))
    return mean_arr, cov_arr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert cristhian_potes Springer .mat model files to cristhian_potes_model.npz",
    )
    parser.add_argument(
        "mat_dir",
        nargs="?",
        default=None,
        help="Directory containing the three .mat files (default: logistic-regression-hsmm-based-heart-sound-segmentation-1.0/cristhian_potes-204)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="cristhian_potes_model.npz",
        help="Output .npz path (default: cristhian_potes_model.npz in mat_dir)",
    )
    args = parser.parse_args()

    if args.mat_dir is None:
        base = os.path.dirname(os.path.abspath(__file__))
        mat_dir = os.path.join(
            base,
            "logistic-regression-hsmm-based-heart-sound-segmentation-1.0",
            "cristhian_potes-204",
        )
    else:
        mat_dir = os.path.abspath(args.mat_dir)

    if not os.path.isdir(mat_dir):
        print(f"Error: directory not found: {mat_dir}", file=sys.stderr)
        sys.exit(1)

    names = (
        "Springer_B_matrix.mat",
        "Springer_pi_vector.mat",
        "Springer_total_obs_distribution.mat",
    )
    paths = [os.path.join(mat_dir, n) for n in names]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        print("Error: missing .mat files:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        print("Place the three .mat files in the directory and run again.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading from {mat_dir} ...")
    B_matrix = load_B_matrix(paths[0])
    pi_vector = load_pi_vector(paths[1])
    total_obs_mean, total_obs_cov = load_total_obs_distribution(paths[2])

    # Same layout as save_springer_model in model_io.py
    max_len = max(len(B) for B in B_matrix)
    B_stack = np.array(
        [
            np.pad(B, (0, max_len - len(B))) if len(B) < max_len else B
            for B in B_matrix
        ]
    )

    if os.path.isabs(args.output) or os.path.dirname(args.output):
        out_path = args.output
    else:
        out_path = os.path.join(mat_dir, args.output)

    np.savez(
        out_path,
        B_matrix=B_stack,
        pi_vector=pi_vector,
        total_obs_mean=total_obs_mean,
        total_obs_cov=total_obs_cov,
    )
    print(f"Saved: {out_path}")
    print("Use this path as springer_model_path in config or the UI to use cristhian_potes' model.")


if __name__ == "__main__":
    main()
