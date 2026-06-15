"""
Explain *why* each wide diastole candidate became QUIET (no insert) vs a GAP
(phantom-cycle insert) — by replaying the exact two-gate test the Pass-3 gap
detector uses.

Mechanism (see correction._detect_sensitive_peaks_in_large_gap_windows and
_pass3_find_gap_windows):

  For a wide candidate segment the sensitive peak detector runs over the segment.
  A local maximum is accepted as a "peak" only if it clears BOTH gates:
     (1) PROMINENCE gate:  prominence >= quantile(segment_envelope, q_sens)
     (2) HEIGHT gate:      envelope_height >= dynamic_noise_floor * height_scale

  0 accepted peaks  -> trigger = quiet_entire_gap_region   (CORRECT: no insert)
  >=1 accepted peak -> trim quiet prefix, treat rest as GAP -> phantom insert

So the entire quiet-vs-gap fork is decided by whether ONE bump in the silent
pause clears the noise floor (and the relative prominence quantile). This tool
prints, for every quiet and gap window the pipeline emitted, the strongest few
bumps with their prominence/height and a PASS/FAIL on each gate — making the
firing (or non-firing) bump visible.

Usage (from repo root):
    python debug_helpers/gap_decision_audit.py "inputs/.../file.wav"
    python debug_helpers/gap_decision_audit.py "inputs/.../file.wav" --top 5
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import find_peaks, peak_prominences  # noqa: E402

from config import param  # noqa: E402
from debug_helpers._common import (  # noqa: E402
    env_sample_rate, params, reconfigure_stdio, run_pipeline,
)


def _run(wav, run_params):
    data = run_pipeline(wav, run_params)
    if data is None:
        return None, 0.0
    labels = data.get("pass3_state_labels")
    n = 0 if labels is None else len(labels)
    sr = env_sample_rate(wav, n) or 0.0
    return data, sr


def _env(data):
    for k in ("noise_removed_envelope", "bandpass_envelope"):
        v = data.get(k)
        if v is not None and len(v):
            return np.asarray(v, dtype=np.float64), k
    return None, None


def _floor(data, n):
    s = data.get("dynamic_noise_floor_series")
    if isinstance(s, pd.Series):
        a = s.to_numpy(dtype=np.float64, copy=False)
        if len(a) == n:
            return a
    return None


def _gate_report(env, floor, lo, hi, q, height_scale, min_dist, sr, top):
    """Replay the detector's two gates over env[lo:hi]; return text lines."""
    seg = env[lo:hi]
    if seg.size < 4:
        return ["    (segment too short)"], 0
    prom_thresh = float(np.quantile(seg, q))
    if not np.isfinite(prom_thresh) or prom_thresh < 0:
        prom_thresh = 0.0

    # All local maxima ignoring gates, so we can show what *almost* fired too.
    cand, _ = find_peaks(seg, distance=min_dist)
    if cand.size == 0:
        return [f"    no local maxima.  prom_thresh(q={q})={prom_thresh:.4g}"], 0
    proms = peak_prominences(seg, cand)[0]

    # The accepted set = exactly what the detector would emit (both gates).
    height = floor[lo:hi] * height_scale if floor is not None else None
    acc, _ = find_peaks(seg, height=height, prominence=prom_thresh, distance=min_dist)
    n_acc = int(len(acc))

    order = np.argsort(proms)[::-1]
    lines = [f"    prom_thresh(q={q}) = {prom_thresh:.4g}    "
             f"height_gate = noise_floor x {height_scale}"]
    for j in order[:top]:
        i = int(cand[j])
        gi = lo + i
        h = float(seg[i])
        prom = float(proms[j])
        fl = float(height[i]) if height is not None else 0.0
        p_ok = prom >= prom_thresh
        h_ok = (h >= fl) if height is not None else True
        verdict = "ACCEPT" if (p_ok and h_ok) else "reject"
        why = []
        if not p_ok:
            why.append(f"prom {prom:.4g}<{prom_thresh:.4g}")
        if not h_ok:
            why.append(f"height {h:.4g}<floor {fl:.4g}")
        tail = "" if (p_ok and h_ok) else "  <- " + ", ".join(why)
        lines.append(
            f"      @{gi/sr:6.2f}s h={h:.4g} prom={prom:.4g} floor={fl:.4g} "
            f"[{'P' if p_ok else '.'}{'H' if h_ok else '.'}] {verdict}{tail}"
        )
    return lines, n_acc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("--top", type=int, default=4, help="Strongest N bumps to show per window.")
    ns = ap.parse_args(argv)

    reconfigure_stdio()

    run_params = params()
    data, sr = _run(ns.wav, run_params)
    if data is None or not sr:
        print("Could not determine sample rate.", file=sys.stderr)
        return 2

    env, ekey = _env(data)
    if env is None:
        print("No envelope in analysis_data.", file=sys.stderr)
        return 2
    floor = _floor(data, len(env))
    q = float(param(run_params, "pass3_gap_recovery_peak_prominence_quantile_sensitive"))
    hscale = float(param(run_params, "pass3_gap_recovery_height_scale"))
    min_dist = max(1, int(float(param(run_params, "min_peak_distance_sec")) * sr))

    quiet = data.get("pass3_gap_quiet_windows_samples") or []
    gaps = data.get("pass3_large_gap_windows_samples") or []

    print(f"# {os.path.basename(ns.wav)}  (sr~{sr:.1f}Hz env={ekey} "
          f"floor={'yes' if floor is not None else 'MISSING'})")
    print(f"# gates: prominence>=quantile(seg,{q})  AND  height>=floor*{hscale}\n")

    print(f"## QUIET windows ({len(quiet)}) — fork outcome per candidate\n")
    for w in sorted(quiet, key=lambda x: int(x["start_sample"])):
        lo, hi = int(w["start_sample"]), int(w["end_sample"])
        trig = w.get("trigger", "?")
        st = w.get("gap_region_candidate_state", "?")
        lines, n_acc = _gate_report(env, floor, lo, hi, q, hscale, min_dist, sr, ns.top)
        kind = "CORRECT no-insert" if trig == "quiet_entire_gap_region" else "-> trimmed, GAP follows"
        print(f"[{trig:24}] {lo/sr:6.2f}-{hi/sr:6.2f}s  {st}  ({kind})")
        print(f"          accepted_peaks_in_window={n_acc}")
        for ln in lines:
            print(ln)
        print()

    print(f"## GAP windows ({len(gaps)}) — these got a phantom cycle inserted\n")
    print("# NB: the detector ran over the *source segment* (= quiet prefix + gap),\n"
          "#     so the firing bump and prom_thresh are reported over that wider span.\n")
    for w in sorted(gaps, key=lambda x: int(x["start_sample"])):
        # Faithful span: the original segment the detector tested, not the trimmed gap.
        slo = int(w.get("source_seg_start", w["start_sample"]))
        shi = int(w.get("source_seg_end", w["end_sample"]))
        glo, ghi = int(w["start_sample"]), int(w["end_sample"])
        nsp = w.get("n_sensitive_peaks", "?")
        bpm = w.get("bpm_at_mid", float("nan"))
        lines, n_acc = _gate_report(env, floor, slo, shi, q, hscale, min_dist, sr, ns.top)
        print(f"[GAP/insert            ] source {slo/sr:6.2f}-{shi/sr:6.2f}s  "
              f"trimmed_gap {glo/sr:.2f}-{ghi/sr:.2f}s  "
              f"n_sensitive_peaks={nsp} bpm_at_mid={bpm:.0f}")
        print(f"          accepted_peaks_in_source={n_acc}  (>=1 => becomes a gap)")
        for ln in lines:
            print(ln)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
