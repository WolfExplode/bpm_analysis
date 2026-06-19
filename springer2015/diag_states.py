"""Quick diagnostic: run Springer on a few real CirCor recordings and report state distribution."""
import glob
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from springer_hsmm.model_io import load_springer_model
from springer_hsmm.options import default_springer_hsmm_options
from springer_hsmm.run import run_springer_segmentation_algorithm

NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cristhian_potes_model.npz")
ROOT = r"G:\HB other\PCG Datasets\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data"

if not os.path.exists(NPZ):
    print(f"Missing {NPZ}"); sys.exit(1)

model = load_springer_model(NPZ)
opts = default_springer_hsmm_options()

wavs = sorted(glob.glob(os.path.join(ROOT, "*.wav")))[:5]
if not wavs:
    print(f"No WAVs under {ROOT}"); sys.exit(0)

NAMES = {1: "S1", 2: "sys", 3: "S2", 4: "dia"}

for w in wavs:
    audio, fs = sf.read(w)
    audio = np.asarray(audio, dtype=np.float64).flatten()
    states, env, dbg = run_springer_segmentation_algorithm(
        audio, fs, model["B_matrix"], model["pi_vector"],
        model["total_obs_distribution"], opts, return_debug=True,
    )
    qt = dbg["qt_state_counts"]
    total = sum(qt.values())
    pct = {NAMES.get(k, k): f"{100*v/total:.1f}%" for k, v in sorted(qt.items())}
    print(f"{os.path.basename(w):18s}  HR={dbg['heart_rate']:.0f}bpm  {pct}")
