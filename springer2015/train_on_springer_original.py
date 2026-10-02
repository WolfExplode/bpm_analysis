"""Reproduce the pretrained Springer model using the original example_data.mat.

Validates our training pipeline: if weights match cristhian_potes_model.npz closely,
the pipeline is correct. Differences expected due to random sampling in
train_band_pi_matrices_springer, but sign patterns and magnitudes should align.

Usage:
    python springer2015/train_on_springer_original.py [--out FILE] [--seed N]
"""

import argparse
import os
import sys

import numpy as np
import scipy.io as sio

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from springer_hsmm.features import get_springer_pcg_features  # noqa: E402
from springer_hsmm.label_states import label_pcg_states  # noqa: E402
from springer_hsmm.model_io import load_springer_model, save_springer_model  # noqa: E402
from springer_hsmm.options import default_springer_hsmm_options  # noqa: E402
from springer_hsmm.train import train_band_pi_matrices_springer  # noqa: E402

EXAMPLE_DATA = os.path.join(
    _HERE,
    "Logistic Regression-HSMM-based Heart Sound Segmentation",
    "example_data.mat",
)
PRETRAINED = os.path.join(_HERE, "cristhian_potes_model.npz")
DEFAULT_OUT = os.path.join(_HERE, "springer_reproduced.npz")

AUDIO_FS = 1000.0  # example_data is recorded at 1000 Hz


def _print_model_summary(label, B_matrix, pi_vector, total_obs):
    mean, cov = total_obs
    print(f"\n{label} B_matrix (rows=states, cols=[intercept,f1,f2,f3,f4]):")
    for i, row in enumerate(B_matrix):
        print(f"  state {i+1}: {row}")
    print(f"  total_obs_mean:     {mean}")
    print(f"  total_obs_cov diag: {np.diag(cov)}")


def _compare_models(B_rep, B_pre):
    """Report per-state cosine similarity between reproduced and pretrained B_matrix."""
    print("\n=== Cosine similarity: reproduced vs pretrained ===")
    for i, (br, bp) in enumerate(zip(B_rep, B_pre)):
        norm = np.linalg.norm(br) * np.linalg.norm(bp)
        cos = float(np.dot(br, bp) / norm) if norm > 0 else 0.0
        print(f"  state {i+1}: cosine={cos:+.4f}  (|cos|=1.0 is perfect alignment)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce Springer pretrained model from example_data.mat.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for logistic regression sampling (original used MATLAB default)")
    args = ap.parse_args()

    if not os.path.isfile(EXAMPLE_DATA):
        print(f"Missing {EXAMPLE_DATA}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {EXAMPLE_DATA} ...")
    raw = sio.loadmat(EXAMPLE_DATA, simplify_cells=True)
    ed = raw["example_data"]
    audio_list = ed["example_audio_data"]   # (792,) object array of float64 vectors
    ann = ed["example_annotations"]         # (792, 2) — col0=S1 samples, col1=S2 samples

    n_recs = len(audio_list)
    print(f"  {n_recs} recordings at {AUDIO_FS} Hz")

    opts = default_springer_hsmm_options()
    feat_fs = opts["audio_segmentation_Fs"]  # 50 Hz

    state_obs: list[list[np.ndarray]] = []
    skipped = 0

    for i in range(n_recs):
        audio = np.asarray(audio_list[i], dtype=np.float64).flatten()
        s1_samp = np.asarray(ann[i, 0], dtype=np.float64).flatten()
        s2_samp = np.asarray(ann[i, 1], dtype=np.float64).flatten()

        if len(s1_samp) == 0 or len(s2_samp) == 0:
            skipped += 1
            continue

        try:
            PCG_Features, _ = get_springer_pcg_features(audio, AUDIO_FS, opts)
        except Exception as exc:
            print(f"  [{i+1}/{n_recs}] SKIP (feature error: {exc})")
            skipped += 1
            continue

        envelope = PCG_Features[:, 0]
        T = len(envelope)
        # Annotations in example_data.mat are already at feature rate (50 Hz),
        # matching how MATLAB's trainSpringerSegmentationAlgorithm calls labelPCGStates.
        s1_feat = np.clip(s1_samp.astype(int), 0, T - 1)
        s2_feat = np.clip(s2_samp.astype(int), 0, T - 1)

        PCG_states = label_pcg_states(envelope, s1_feat, s2_feat, float(feat_fs))
        state_obs.append([PCG_Features[PCG_states == (si + 1)] for si in range(4)])

        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{n_recs} ({skipped} skipped)")

    print(f"\nFitting on {len(state_obs)} recordings ({skipped} skipped)...")

    # Fix seed for reproducibility (MATLAB used a fixed state internally)
    np.random.seed(args.seed)
    B_matrix, pi_vector, total_obs = train_band_pi_matrices_springer(state_obs)

    save_springer_model(args.out, B_matrix, pi_vector, total_obs)
    print(f"Model written to {args.out}")

    _print_model_summary("Reproduced", B_matrix, pi_vector, total_obs)

    if os.path.isfile(PRETRAINED):
        pm = load_springer_model(PRETRAINED)
        _print_model_summary("Pretrained", pm["B_matrix"], pm["pi_vector"],
                              pm["total_obs_distribution"])
        _compare_models(B_matrix, pm["B_matrix"])


if __name__ == "__main__":
    main()
