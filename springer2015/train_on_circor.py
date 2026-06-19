"""Retrain the Springer HSMM on CirCor training data.

Reads WAV + TSV pairs, converts ground-truth spans directly to per-frame state
labels at feat_fs (bypassing label_pcg_states, which expects ECG-onset positions).

Usage:
    python springer2015/train_on_circor.py [--root DIR] [--out FILE] [--n N] [--seed S]

Output defaults to springer2015/springer_circor_trained.npz.
Pass that path to compare_circor.py via --model to evaluate it.
"""

import argparse
import glob
import os
import random
import sys

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from springer_hsmm.features import get_springer_pcg_features  # noqa: E402
from springer_hsmm.model_io import save_springer_model  # noqa: E402
from springer_hsmm.options import default_springer_hsmm_options  # noqa: E402
from springer_hsmm.train import train_band_pi_matrices_springer  # noqa: E402

DEFAULT_ROOT = (
    r"G:\HB other\PCG Datasets"
    r"\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data"
)
DEFAULT_OUT = os.path.join(_HERE, "springer_circor_trained.npz")

CODE_TO_STATE = {1: "S1", 2: "systole", 3: "S2", 4: "diastole"}


def load_tsv_spans(path: str) -> list[tuple[float, float, int]]:
    spans = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                start, end, code = float(parts[0]), float(parts[1]), int(float(parts[2]))
            except ValueError:
                continue
            if end > start and code in CODE_TO_STATE:
                spans.append((start, end, code))
    return spans


def spans_to_frame_labels(spans: list[tuple[float, float, int]], feat_fs: float, T: int) -> np.ndarray:
    """Convert TSV spans directly to per-frame state labels at feat_fs.

    CirCor spans already encode all 4 states; no need to go through
    label_pcg_states (which expects ECG R-peak onsets, not span boundaries).
    Unlabeled frames default to diastole (4).
    """
    states = np.full(T, 4, dtype=np.int32)
    for start_sec, end_sec, code in spans:
        f0 = max(0, min(T, int(round(start_sec * feat_fs))))
        f1 = max(0, min(T, int(round(end_sec * feat_fs))))
        if f0 < f1:
            states[f0:f1] = code
    return states


def collect_recordings(root: str) -> list[tuple[str, str]]:
    found = []
    for tsv in sorted(glob.glob(os.path.join(root, "*.tsv"))):
        wav = tsv[:-4] + ".wav"
        if os.path.isfile(wav):
            found.append((wav, tsv))
    return found


def _print_model_summary(label, B_matrix, pi_vector, total_obs):
    mean, cov = total_obs
    print(f"{label} B_matrix (rows=states, cols=[intercept,f1,f2,f3,f4]):")
    for i, row in enumerate(B_matrix):
        print(f"  state {i+1}: {row}")
    print(f"  total_obs_mean:     {mean}")
    print(f"  total_obs_cov diag: {np.diag(cov)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrain Springer HSMM on CirCor.")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="CirCor training_data dir")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output .npz path")
    ap.add_argument("--n", type=int, default=0,
                    help="Max recordings to use (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = collect_recordings(args.root)
    if not recs:
        print(f"No recordings found under {args.root}", file=sys.stderr)
        sys.exit(1)

    if args.n and args.n < len(recs):
        random.Random(args.seed).shuffle(recs)
        recs = recs[: args.n]

    print(f"Training on {len(recs)} recordings...")

    opts = default_springer_hsmm_options()
    feat_fs = opts["audio_segmentation_Fs"]
    state_obs: list[list[np.ndarray]] = []
    skipped = 0

    for i, (wav, tsv) in enumerate(recs, 1):
        name = os.path.splitext(os.path.basename(wav))[0]
        spans = load_tsv_spans(tsv)
        has_s1 = any(c == 1 for _, _, c in spans)
        has_s2 = any(c == 3 for _, _, c in spans)
        if not has_s1 or not has_s2:
            print(f"  [{i}/{len(recs)}] SKIP {name} (no S1 or S2 labels)")
            skipped += 1
            continue

        try:
            audio, rec_fs = sf.read(wav)
        except Exception as exc:
            print(f"  [{i}/{len(recs)}] ERROR reading {name}: {exc}")
            skipped += 1
            continue

        audio = np.asarray(audio, dtype=np.float64).flatten()
        try:
            PCG_Features, _ = get_springer_pcg_features(audio, float(rec_fs), opts)
        except Exception as exc:
            print(f"  [{i}/{len(recs)}] SKIP {name} (feature error: {exc})")
            skipped += 1
            continue

        T = PCG_Features.shape[0]
        PCG_states = spans_to_frame_labels(spans, feat_fs, T)
        state_obs.append([PCG_Features[PCG_states == (si + 1)] for si in range(4)])

        if i % 100 == 0:
            print(f"  processed {i}/{len(recs)} ({skipped} skipped)")

    if not state_obs:
        print("No valid recordings after filtering.", file=sys.stderr)
        sys.exit(1)

    print(f"\nFitting on {len(state_obs)} recordings ({skipped} skipped)...")

    B_matrix, pi_vector, total_obs = train_band_pi_matrices_springer(state_obs)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_springer_model(args.out, B_matrix, pi_vector, total_obs)
    print(f"\nModel written to {args.out}")
    _print_model_summary("Trained", B_matrix, pi_vector, total_obs)

    pretrained = os.path.join(_HERE, "cristhian_potes_model.npz")
    if os.path.isfile(pretrained):
        from springer_hsmm.model_io import load_springer_model
        pm = load_springer_model(pretrained)
        print()
        _print_model_summary("Pretrained", pm["B_matrix"], pm["pi_vector"], pm["total_obs_distribution"])


if __name__ == "__main__":
    main()
