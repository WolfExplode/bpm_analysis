"""
Example script: load example_data.mat, train (or load model), run segmentation, optionally plot.

Run from repo root with:
  python -m springer_hsmm.run_example [--mat PATH] [--model PATH] [--save-model PATH] [--plot]
"""

import argparse
import os
import sys

import numpy as np

from springer_hsmm.example_data import load_example_data
from springer_hsmm.model_io import load_springer_model, save_springer_model
from springer_hsmm.options import default_springer_hsmm_options
from springer_hsmm.run import run_springer_segmentation_algorithm
from springer_hsmm.train import train_springer_segmentation_algorithm


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Springer HSMM segmentation (train and/or segment).")
    parser.add_argument(
        "--mat",
        default=None,
        help="Path to example_data.mat (default: lookup in logistic-regression-hsmm-based-heart-sound-segmentation-1.0)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to saved .npz model (if set, skip training and use this model)",
    )
    parser.add_argument(
        "--save-model",
        default=None,
        help="After training, save model to this path",
    )
    parser.add_argument(
        "--train-indices",
        default="1,47,361,402,572",
        help="Comma-separated indices (0-based) of recordings to use for training",
    )
    parser.add_argument(
        "--test-index",
        type=int,
        default=0,
        help="Index (0-based) of recording to segment after training",
    )
    parser.add_argument("--plot", action="store_true", help="Show matplotlib plot of states vs time")
    args = parser.parse_args()

    options = default_springer_hsmm_options()

    mat_path = args.mat
    if not mat_path:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mat_path = os.path.join(
            base,
            "logistic-regression-hsmm-based-heart-sound-segmentation-1.0",
            "example_data.mat",
        )
    if not os.path.isfile(mat_path):
        print(f"example_data.mat not found at {mat_path}. Use --mat PATH.", file=sys.stderr)
        sys.exit(1)
    audio_list, annotations_list, fs = load_example_data(mat_path)

    if args.model and os.path.isfile(args.model):
        model = load_springer_model(args.model)
        B_matrix = model["B_matrix"]
        pi_vector = model["pi_vector"]
        total_obs = model["total_obs_distribution"]
        print(f"Loaded model from {args.model}")
    else:
        train_indices = [int(x.strip()) for x in args.train_indices.split(",")]
        train_indices = [i for i in train_indices if 0 <= i < len(audio_list)]
        if not train_indices:
            train_indices = list(range(min(5, len(audio_list))))
        train_audio = [audio_list[i] for i in train_indices]
        train_annot = [annotations_list[i] for i in train_indices]
        print(f"Training on {len(train_audio)} recordings (indices {train_indices})...")
        B_matrix, pi_vector, total_obs = train_springer_segmentation_algorithm(
            train_audio, train_annot, fs, options
        )
        if args.save_model:
            save_springer_model(args.save_model, B_matrix, pi_vector, total_obs)
            print(f"Saved model to {args.save_model}")

    test_idx = min(args.test_index, len(audio_list) - 1)
    audio_data = audio_list[test_idx]

    assigned_states, _ = run_springer_segmentation_algorithm(
        audio_data, fs, B_matrix, pi_vector, total_obs, options
    )
    print(f"Segmentation done: {len(assigned_states)} samples, states 1-4 (S1/systole/S2/diastole).")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            t = np.arange(len(assigned_states)) / fs
            plt.figure()
            plt.plot(t, assigned_states, "r-", alpha=0.8)
            plt.xlabel("Time (s)")
            plt.ylabel("State (1=S1, 2=systole, 3=S2, 4=diastole)")
            plt.title("Springer HSMM assigned states")
            plt.show()
        except Exception as e:
            print(f"Plot failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
