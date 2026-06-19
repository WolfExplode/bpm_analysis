"""Retrain Springer HSMM on PhysioNet 2016 hand-corrected annotations.

Usage:
    python springer2015/train_on_physionet2016.py [--root DIR] [--out FILE] [--n N] [--seed S]

Output defaults to springer2015/springer_pn2016_trained.npz.
Pass to compare_circor.py via --model to evaluate.
"""

import argparse
import glob
import os
import random
import sys

import numpy as np
import scipy.io as sio
import soundfile as sf

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

DEFAULT_ROOT = (
    r"G:\HB other\PCG Datasets\PhysioNet Challenge2016Heart sound classification"
)
DEFAULT_OUT = os.path.join(_HERE, "springer_pn2016_trained.npz")

SETS = ("a", "b", "c", "d", "e", "f")


def _parse_state_name(raw) -> str:
    """Extract plain state name from mat object cell (may arrive as \"['S1']\" etc.)."""
    s = str(raw).strip()
    return s.strip("[]'\"")


def load_annotation(mat_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse StateAns .mat -> (s1_samples, s2_samples) center positions as float arrays.
    Annotation format: (N, 2) object array of (sample_index, state_name) transitions.
    """
    mat = sio.loadmat(mat_path)
    sa = mat["state_ans"]
    transitions: list[tuple[int, str]] = []
    for i in range(sa.shape[0]):
        idx = int(sa[i, 0].flat[0])
        name = _parse_state_name(sa[i, 1].flat[0])
        transitions.append((idx, name))

    s1, s2 = [], []
    for j, (idx, name) in enumerate(transitions):
        nxt = transitions[j + 1][0] if j + 1 < len(transitions) else idx
        center = (idx + nxt) / 2.0
        if name == "S1":
            s1.append(center)
        elif name == "S2":
            s2.append(center)

    return np.array(s1, dtype=np.float64), np.array(s2, dtype=np.float64)


def collect_pairs(root: str) -> list[tuple[str, str]]:
    """Return (wav_path, mat_path) pairs where both files exist."""
    pairs = []
    ann_root = os.path.join(root, "annotations", "hand_corrected")
    for s in SETS:
        wav_dir = os.path.join(root, f"training-{s}")
        mat_dir = os.path.join(ann_root, f"training-{s}_StateAns")
        if not os.path.isdir(wav_dir) or not os.path.isdir(mat_dir):
            continue
        for mat_path in sorted(glob.glob(os.path.join(mat_dir, "*.mat"))):
            rec_id = os.path.basename(mat_path).replace("_StateAns.mat", "")
            wav_path = os.path.join(wav_dir, rec_id + ".wav")
            if os.path.isfile(wav_path):
                pairs.append((wav_path, mat_path))
    return pairs


def _print_model_summary(label, B_matrix, pi_vector, total_obs):
    mean, cov = total_obs
    print(f"{label} B_matrix (rows=states, cols=[intercept,f1,f2,f3,f4]):")
    for i, row in enumerate(B_matrix):
        print(f"  state {i+1}: {row}")
    print(f"  total_obs_mean:     {mean}")
    print(f"  total_obs_cov diag: {np.diag(cov)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrain Springer HSMM on PhysioNet 2016.")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=0, help="Max recordings (0=all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = collect_pairs(args.root)
    if not pairs:
        print(f"No paired WAV+mat found under {args.root}", file=sys.stderr)
        sys.exit(1)

    if args.n and args.n < len(pairs):
        random.Random(args.seed).shuffle(pairs)
        pairs = pairs[: args.n]

    print(f"Training on {len(pairs)} recordings...")

    opts = default_springer_hsmm_options()
    feat_fs = opts["audio_segmentation_Fs"]

    state_obs: list[list[np.ndarray]] = []
    skipped = 0

    for i, (wav_path, mat_path) in enumerate(pairs, 1):
        name = os.path.splitext(os.path.basename(wav_path))[0]
        try:
            s1_samp, s2_samp = load_annotation(mat_path)
        except Exception as exc:
            print(f"  [{i}/{len(pairs)}] SKIP {name} (annotation error: {exc})")
            skipped += 1
            continue

        if len(s1_samp) == 0 or len(s2_samp) == 0:
            print(f"  [{i}/{len(pairs)}] SKIP {name} (no S1 or S2)")
            skipped += 1
            continue

        try:
            audio, rec_fs = sf.read(wav_path)
        except Exception as exc:
            print(f"  [{i}/{len(pairs)}] SKIP {name} (read error: {exc})")
            skipped += 1
            continue

        audio = np.asarray(audio, dtype=np.float64).flatten()
        try:
            PCG_Features, _ = get_springer_pcg_features(audio, float(rec_fs), opts)
        except Exception as exc:
            print(f"  [{i}/{len(pairs)}] SKIP {name} (feature error: {exc})")
            skipped += 1
            continue

        envelope = PCG_Features[:, 0]
        T = len(envelope)
        # Annotation sample indices are at rec_fs; convert to feat_fs
        s1_feat = np.clip(np.round(s1_samp * feat_fs / rec_fs).astype(int), 0, T - 1)
        s2_feat = np.clip(np.round(s2_samp * feat_fs / rec_fs).astype(int), 0, T - 1)

        PCG_states = label_pcg_states(envelope, s1_feat, s2_feat, float(feat_fs))
        state_obs.append([PCG_Features[PCG_states == (si + 1)] for si in range(4)])

        if i % 100 == 0:
            print(f"  processed {i}/{len(pairs)} ({skipped} skipped)")

    if not state_obs:
        print("No valid recordings after filtering.", file=sys.stderr)
        sys.exit(1)

    print(f"\nFitting on {len(state_obs)} recordings ({skipped} skipped)...")
    B_matrix, pi_vector, total_obs = train_band_pi_matrices_springer(state_obs)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_springer_model(args.out, B_matrix, pi_vector, total_obs)
    print(f"\nModel written to {args.out}")
    _print_model_summary("Trained (PN2016)", B_matrix, pi_vector, total_obs)

    pretrained = os.path.join(_HERE, "cristhian_potes_model.npz")
    if os.path.isfile(pretrained):
        pm = load_springer_model(pretrained)
        print()
        _print_model_summary("Pretrained", pm["B_matrix"], pm["pi_vector"],
                              pm["total_obs_distribution"])


if __name__ == "__main__":
    main()
