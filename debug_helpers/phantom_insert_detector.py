"""
Find *phantom* cardiac cycles: state segments the Pass-3 gap logic rebuilt
(rebuild_source in gap_insert / gap_label_pass3 / noise_repair) over a span that
has no actual acoustic heartbeat under it.

The failure this targets: in irregular recordings with a dropped beat, the gap
filler packs a full S1->systole->S2->diastole cycle into the silent pause even
though no heart sound is there. Symptom in the strip: an orange "rebuilt" band
sitting over a flat envelope with no peak marker.

For every contiguous rebuilt run it reports, per run:
  * time span and which rebuild_source painted it
  * how many *real* classified peaks (peak_classifications) fall inside
  * how many *sensitive* acoustic peaks the detector finds inside
  * envelope energy inside vs. the file's typical S1/S2 peak energy
  * the scale factor the filler used (flagged when it exceeded +/-30%)

A run is flagged PHANTOM when it has zero real classified peaks AND its peak
envelope energy is well below the file's genuine-beat energy.

Usage (from repo root):
    python debug_helpers/phantom_insert_detector.py "inputs/.../file.wav"
    python debug_helpers/phantom_insert_detector.py "inputs/.../file.wav" --energy-frac 0.35
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

from config import param  # noqa: E402
from peak_utils import PeakType  # noqa: E402
from correction import _detect_sensitive_peaks_in_large_gap_windows  # noqa: E402
from debug_helpers._common import (  # noqa: E402
    env_sample_rate, params, reconfigure_stdio, run_pipeline,
)

_REBUILD = ("gap_insert", "gap_label_pass3", "noise_repair")


def _run(wav, run_params):
    data = run_pipeline(wav, run_params)
    if data is None:
        return None, 0.0
    labels = data.get("pass3_state_labels")
    n = 0 if labels is None else len(labels)
    sr = env_sample_rate(wav, n) or 0.0
    return data, sr


def _envelope(data):
    """Best-effort: the envelope the strips/peaks are computed against."""
    # Mirror pipeline.algorithm_envelope: noise_removed_envelope, else bandpass.
    for k in ("noise_removed_envelope", "bandpass_envelope"):
        v = data.get(k)
        if v is not None and hasattr(v, "__len__") and len(v):
            return np.asarray(v, dtype=np.float64), k
    return None, None


def _runs(boundaries):
    """Group consecutive rebuilt segments into runs (one phantom cycle = one run)."""
    runs, cur = [], None
    for seg in boundaries:
        a0, a1, name = int(seg[0]), int(seg[1]), seg[2]
        meta = seg[3] if len(seg) > 3 and isinstance(seg[3], dict) else {}
        src = meta.get("rebuild_source")
        if src in _REBUILD:
            if cur and cur["src"] == src and a0 == cur["hi"]:
                cur["hi"] = a1
                cur["segs"].append((a0, a1, name))
            else:
                if cur:
                    runs.append(cur)
                cur = {"lo": a0, "hi": a1, "src": src, "segs": [(a0, a1, name)]}
        else:
            if cur:
                runs.append(cur)
                cur = None
    if cur:
        runs.append(cur)
    return runs


def _real_peaks(data, lo, hi):
    pc = data.get("peak_classifications") or {}
    out = []
    for idx, info in pc.items():
        i = int(idx)
        if lo <= i < hi:
            pt = info.get("peak_type", "") if isinstance(info, dict) else str(info)
            out.append((i, pt))
    return out


def _genuine_peak_energy(data, env):
    """Median envelope value at real S1/S2 peak samples = the 'a beat looks like this' level."""
    pc = data.get("peak_classifications") or {}
    vals = []
    for idx, info in pc.items():
        i = int(idx)
        pt = info.get("peak_type", "") if isinstance(info, dict) else str(info)
        if (PeakType.is_s1(pt) or PeakType.is_s2(pt)) and 0 <= i < len(env):
            vals.append(float(env[i]))
    return float(np.median(vals)) if vals else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("--energy-frac", type=float, default=0.40,
                    help="A run's peak energy below this fraction of genuine-beat "
                         "energy counts as 'no acoustic beat' (default 0.40).")
    ns = ap.parse_args(argv)

    reconfigure_stdio()

    run_params = params()
    data, sr = _run(ns.wav, run_params)
    if data is None or not sr:
        print("Could not determine sample rate.", file=sys.stderr)
        return 2

    env, env_key = _envelope(data)
    boundaries = data.get("pass3_state_boundaries") or []
    runs = _runs(boundaries)

    print(f"# {os.path.basename(ns.wav)}  (envelope sr ~{sr:.1f} Hz, env_key={env_key})")
    print(f"# {len(runs)} rebuilt run(s) total\n")

    if env is None:
        print("WARN: no envelope in analysis_data; energy test skipped.", file=sys.stderr)

    genuine = _genuine_peak_energy(data, env) if env is not None else float("nan")
    if env is not None:
        print(f"# genuine-beat peak energy (median S1/S2 envelope) = {genuine:.4g}")
        print(f"# phantom energy threshold = {ns.energy_frac:.2f} x genuine = "
              f"{ns.energy_frac * genuine:.4g}\n")

    sample_rate_hz = int(round(sr))
    nfloor = data.get("pass3_dynamic_noise_floor_series")
    q_sens = float(param(run_params, "pass3_gap_recovery_peak_prominence_quantile_sensitive"))

    n_phantom = 0
    for r in runs:
        lo, hi, src = r["lo"], r["hi"], r["src"]
        dur = (hi - lo) / sr
        real = _real_peaks(data, lo, hi)

        try:
            sens = _detect_sensitive_peaks_in_large_gap_windows(
                env if env is not None else np.zeros(1), sample_rate_hz,
                [{"start_sample": lo, "end_sample": hi}], run_params,
                prominence_quantile=q_sens,
                dynamic_noise_floor_series=nfloor,
            )
            n_sens = 0 if sens is None else int(len(sens))
        except Exception:
            n_sens = -1

        if env is not None and hi <= len(env) and hi > lo:
            seg_peak = float(np.max(env[lo:hi]))
            efrac = seg_peak / genuine if genuine and np.isfinite(genuine) else float("nan")
        else:
            seg_peak, efrac = float("nan"), float("nan")

        # Primary failure: the filler painted a full cycle the detector never
        # classified as a beat (no real S1/S2 peak under it). Energy only grades
        # how empty it is: 'silent' (dead flat) vs 'faint' (sub-beat ripple).
        phantom = len(real) == 0
        if phantom:
            n_phantom += 1
        if not phantom:
            tag = "ok"
        elif np.isfinite(efrac) and efrac < ns.energy_frac:
            tag = "PHANTOM/silent"
        else:
            tag = "PHANTOM/faint"

        print(f"[{tag:14}] {lo/sr:6.2f}-{hi/sr:6.2f}s  ({dur:.2f}s)  src={src}")
        print(f"          real_peaks={len(real)}  sensitive_peaks={n_sens}  "
              f"energy={seg_peak:.4g} ({efrac*100:.0f}% of genuine)")
        if real:
            for i, pt in sorted(real):
                print(f"            real peak @{i/sr:.2f}s  '{pt}'")
        print()

    print(f"# RESULT: {n_phantom} phantom insert(s) flagged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
