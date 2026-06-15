#!/usr/bin/env python3
"""
Benchmark runner for BPM analysis.

Usage:
    python run_benchmark.py [input_dir]

Default input_dir: inputs/Difficulty 3

For each WAV file with a _manual_state_sequence.csv, runs the full analysis
pipeline and compares predicted S1 segments (pass3_state_boundaries) against
manual ground truth. Reports error counts per file and totals.

Error types:
    phase_flip  S1/S2 labels swapped for a stretch; counted once at the trigger point.
    miss        Manual S1 with no predicted S1 or S2 within tolerance.
    extra       Predicted S1 with no corresponding manual S1.

JSON summary written to benchmark_result.json alongside this script.
"""

import sys
import os
import csv
import json
import logging
import tempfile
import glob
from typing import List, Tuple, Dict, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))  # project root for config, pipeline, etc.

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from config import DEFAULT_PARAMS
from pipeline import analyze_wav_file

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOLERANCE_SEC = 0.10   # max center-time distance for an S1-to-S1 match
FLIP_N = 2             # consecutive wrong-phase matches before declaring a flip
FLIP_RECOVERY_N = 2    # consecutive correct matches needed to exit a flip region

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg_center(start: float, end: float) -> float:
    return (start + end) / 2.0


def _load_manual_state_sequence(csv_path: str) -> List[Tuple[float, float, str]]:
    segments = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            state = (row.get("state") or "").strip()
            if not state:
                continue
            try:
                start = float(row["start_sec"])
                end = float(row["end_sec"])
            except (KeyError, ValueError):
                continue
            segments.append((start, end, state))
    return segments


def _extract_start_bpm(wav_path: str) -> Optional[float]:
    """Parse [start,min-maxbpm] or Xbpm from filename. Returns float or None."""
    import re
    name = os.path.basename(wav_path)
    m = re.search(r'\[(\d+),\d+-\d+bpm\]', name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r'(\d+)bpm', name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _find_closest_unused(
    target: float,
    centers: List[float],
    used: List[bool],
    tolerance: float,
) -> Optional[int]:
    best_idx = None
    best_dist = float("inf")
    for i, c in enumerate(centers):
        if used[i]:
            continue
        d = abs(c - target)
        if d <= tolerance and d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def _find_closest(target: float, centers: List[float], tolerance: float) -> Optional[int]:
    best_idx = None
    best_dist = float("inf")
    for i, c in enumerate(centers):
        d = abs(c - target)
        if d <= tolerance and d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


# ---------------------------------------------------------------------------
# Per-segment context metrics
# ---------------------------------------------------------------------------

def _compute_segment_metrics(
    start_sec: float,
    end_sec: float,
    analysis_data: Dict,
    sample_rate: int,
) -> Dict:
    """Envelope + BPM metrics over [start_sec, end_sec]."""
    envelope = analysis_data.get("noise_removed_envelope")
    if envelope is None:
        envelope = analysis_data.get("bandpass_envelope")
    if envelope is None:
        return {}

    sr = float(sample_rate)
    start_s = max(0, int(start_sec * sr))
    end_s = min(len(envelope), int(end_sec * sr))
    if end_s <= start_s:
        return {}

    seg = np.asarray(envelope[start_s:end_s], dtype=np.float64)
    duration = (end_s - start_s) / sr
    auc = float(np.trapezoid(seg)) / sr
    env_peak = float(np.max(seg))
    env_mean = float(np.mean(seg))

    center_s = (start_s + end_s) // 2
    center_sec_val = center_s / sr

    nf = None
    noise_series = analysis_data.get("dynamic_noise_floor_series")
    if noise_series is not None:
        try:
            idx_arr = np.asarray(noise_series.index, dtype=np.float64)
            val_arr = np.asarray(noise_series.values, dtype=np.float64)
            nf = float(np.interp(center_s, idx_arr, val_arr))
        except Exception:
            pass

    bpm = None
    bpm_times = analysis_data.get("pass2_lt_bpm_times")
    bpm_vals = analysis_data.get("pass2_lt_bpm")
    if bpm_times is not None and bpm_vals is not None and len(bpm_times) > 1:
        bpm = float(np.interp(center_sec_val, np.asarray(bpm_times), np.asarray(bpm_vals)))

    result: Dict = {
        "duration_sec": round(duration, 4),
        "envelope_auc": round(auc, 6),
        "envelope_peak": round(env_peak, 4),
        "envelope_mean": round(env_mean, 4),
    }
    if nf is not None:
        result["noise_floor"] = round(nf, 4)
        if env_peak > 0:
            result["snr"] = round(env_peak / nf, 3) if nf > 0 else None
    if bpm is not None:
        result["bpm_at_time"] = round(bpm, 1)
    return result


def _get_peak_label_scores_in_window(
    start_sec: float,
    end_sec: float,
    analysis_data: Dict,
    sample_rate: int,
) -> Dict:
    """Return label_scores of the highest-amplitude detected peak in [start_sec, end_sec]."""
    peak_classifications = analysis_data.get("peak_classifications") or {}
    if not peak_classifications:
        return {}
    envelope = analysis_data.get("noise_removed_envelope")
    if envelope is None:
        envelope = analysis_data.get("bandpass_envelope")

    sr = float(sample_rate)
    start_s = int(start_sec * sr)
    end_s = int(end_sec * sr)

    best_amp = -1.0
    best_scores: Optional[Dict] = None
    for idx, entry in peak_classifications.items():
        try:
            s = int(idx)
        except (TypeError, ValueError):
            continue
        if not (start_s <= s < end_s):
            continue
        amp = float(envelope[s]) if envelope is not None and s < len(envelope) else 0.0
        if amp > best_amp:
            best_amp = amp
            best_scores = entry.get("label_scores", {})

    if not best_scores:
        return {}
    return {
        "label_score_s1": round(float(best_scores.get("S1", 0.0)), 4),
        "label_score_s2": round(float(best_scores.get("S2", 0.0)), 4),
        "label_score_noise": round(float(best_scores.get("noise", 0.0)), 4),
    }


def _segment_metrics_entry(
    offset: int,
    state: str,
    start_sec: float,
    end_sec: float,
    analysis_data: Dict,
    sample_rate: int,
) -> Dict:
    entry: Dict = {
        "offset": offset,
        "state": state,
        "start_sec": round(start_sec, 4),
        "end_sec": round(end_sec, 4),
    }
    entry.update(_compute_segment_metrics(start_sec, end_sec, analysis_data, sample_rate))
    entry.update(_get_peak_label_scores_in_window(start_sec, end_sec, analysis_data, sample_rate))
    return entry


def _get_neighbors(
    segments: List[Tuple[float, float, str]],
    center_sec: float,
    n: int,
    analysis_data: Dict,
    sample_rate: int,
) -> List[Dict]:
    """Return n segments before and after the segment containing center_sec, with metrics."""
    if not segments:
        return []

    center_idx = None
    for i, (s, e, _) in enumerate(segments):
        if s <= center_sec <= e:
            center_idx = i
            break
    if center_idx is None:
        dists = [min(abs(center_sec - s), abs(center_sec - e)) for s, e, _ in segments]
        center_idx = int(np.argmin(dists))

    result = []
    for offset in range(-n, n + 1):
        if offset == 0:
            continue
        idx = center_idx + offset
        if 0 <= idx < len(segments):
            seg_s, seg_e, seg_state = segments[idx]
            result.append(_segment_metrics_entry(offset, seg_state, seg_s, seg_e, analysis_data, sample_rate))
    return result


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def compare_file(
    manual_segments: List[Tuple[float, float, str]],
    pred_boundaries: List[Tuple],
    sample_rate: int,
    analysis_data: Optional[Dict] = None,
) -> Dict:
    """
    Compare predicted state boundaries against manual ground truth.
    Evaluates S1 states only.

    pred_boundaries: list of (start_sample, end_sample, state_name, metadata_dict)
    Returns a dict with error breakdown and per-error details.
    """
    sr = float(sample_rate)

    # Full sorted segment lists (all states) for neighbor context
    manual_all: List[Tuple[float, float, str]] = sorted(manual_segments, key=lambda x: x[0])
    pred_all: List[Tuple[float, float, str]] = sorted(
        [(b[0] / sr, b[1] / sr, str(b[2])) for b in pred_boundaries],
        key=lambda x: x[0],
    )

    # S1/S2 only for matching
    manual_s1 = [(s, e) for s, e, st in manual_all if st == "S1"]
    pred_s1 = [(s, e) for s, e, st in pred_all if st.lower() == "s1"]
    pred_s2 = [(s, e) for s, e, st in pred_all if st.lower() == "s2"]

    manual_centers = [_seg_center(s, e) for s, e in manual_s1]
    pred_s1_centers = [_seg_center(s, e) for s, e in pred_s1]
    pred_s2_centers = [_seg_center(s, e) for s, e in pred_s2]

    pred_s1_used = [False] * len(pred_s1)

    # Classify each manual S1: 'match' | 'flip_cand' | 'miss'
    match_types: List[str] = []
    matched_pred_s1_idx: List[Optional[int]] = []

    for mc in manual_centers:
        j = _find_closest_unused(mc, pred_s1_centers, pred_s1_used, TOLERANCE_SEC)
        if j is not None:
            pred_s1_used[j] = True
            match_types.append("match")
            matched_pred_s1_idx.append(j)
            continue
        k = _find_closest(mc, pred_s2_centers, TOLERANCE_SEC)
        if k is not None:
            match_types.append("flip_cand")
            matched_pred_s1_idx.append(None)
            continue
        match_types.append("miss")
        matched_pred_s1_idx.append(None)

    # Scan for flip regions; collect errors with segment bounds
    errors: List[Dict] = []
    flip_time_regions: List[Tuple[float, float]] = []

    in_flip = False
    flip_start_time: Optional[float] = None
    flip_consecutive = 0
    correct_consecutive = 0

    for i, mtype in enumerate(match_types):
        mc = manual_centers[i]
        ms, me = manual_s1[i]

        if mtype == "match":
            flip_consecutive = 0
            correct_consecutive += 1
            if in_flip and correct_consecutive >= FLIP_RECOVERY_N:
                flip_time_regions.append((flip_start_time, mc))
                in_flip = False
                flip_start_time = None
                correct_consecutive = 0

        elif mtype == "flip_cand":
            correct_consecutive = 0
            flip_consecutive += 1
            if not in_flip and flip_consecutive >= FLIP_N:
                start_i = i - FLIP_N + 1
                flip_start_time = manual_centers[start_i]
                fs, fe = manual_s1[start_i]
                errors.append({
                    "time_sec": round(flip_start_time, 3),
                    "start_sec": round(fs, 4),
                    "end_sec": round(fe, 4),
                    "type": "phase_flip",
                    "detail": "S1/S2 labels inverted starting here (upstream labeling error)",
                })
                in_flip = True

        elif mtype == "miss":
            flip_consecutive = 0
            correct_consecutive = 0
            if not in_flip:
                errors.append({
                    "time_sec": round(mc, 3),
                    "start_sec": round(ms, 4),
                    "end_sec": round(me, 4),
                    "type": "miss",
                    "detail": "manual S1 has no predicted S1 or S2 within tolerance",
                })

    if in_flip and manual_centers:
        flip_time_regions.append((flip_start_time, manual_centers[-1]))

    # Extra errors: unmatched predicted S1s outside flip regions
    for j, (ps, pe) in enumerate(pred_s1):
        if pred_s1_used[j]:
            continue
        pc = _seg_center(ps, pe)
        if any(ft[0] - TOLERANCE_SEC <= pc <= ft[1] + TOLERANCE_SEC for ft in flip_time_regions):
            continue
        errors.append({
            "time_sec": round(pc, 3),
            "start_sec": round(ps, 4),
            "end_sec": round(pe, 4),
            "type": "extra",
            "detail": "predicted S1 with no nearby manual S1",
        })

    errors.sort(key=lambda e: e["time_sec"])

    # Enrich errors with context metrics
    if analysis_data is not None:
        for err in errors:
            s_sec = err["start_sec"]
            e_sec = err["end_sec"]
            center_sec = (s_sec + e_sec) / 2.0

            # Error segment metrics + label scores
            err.update(_compute_segment_metrics(s_sec, e_sec, analysis_data, sample_rate))
            err.update(_get_peak_label_scores_in_window(s_sec, e_sec, analysis_data, sample_rate))

            # N=2 neighbors from predicted full sequence and manual full sequence
            err["predicted_context"] = _get_neighbors(pred_all, center_sec, 2, analysis_data, sample_rate)
            err["manual_context"] = _get_neighbors(manual_all, center_sec, 2, analysis_data, sample_rate)

    return {
        "manual_s1_count": len(manual_s1),
        "predicted_s1_count": len(pred_s1),
        "matched": sum(1 for t in match_types if t == "match"),
        "flip_errors": sum(1 for e in errors if e["type"] == "phase_flip"),
        "miss_errors": sum(1 for e in errors if e["type"] == "miss"),
        "extra_errors": sum(1 for e in errors if e["type"] == "extra"),
        "total_errors": len(errors),
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _collect_labeled_wav_files(root: str) -> List[Tuple[str, str, str]]:
    """
    Walk root recursively. Return (subdir_label, wav_path, csv_path) for every
    WAV that has a matching _manual_state_se*.csv beside it.
    """
    found = []
    for dirpath, _dirs, files in os.walk(root):
        subdir = os.path.relpath(dirpath, root)
        for fname in sorted(files):
            if not fname.lower().endswith(".wav"):
                continue
            wav_path = os.path.join(dirpath, fname)
            matches = glob.glob(glob.escape(wav_path) + "_manual_state_se*.csv")
            if matches:
                found.append((subdir, wav_path, matches[0]))
    return found


def _print_subtotal(label: str, n_files: int, n_s1: int, n_err: int, n_flip: int, n_miss: int, n_extra: int) -> None:
    rate = n_err / n_s1 * 100 if n_s1 else 0.0
    print(
        f"  {label}: files={n_files}  s1={n_s1}  "
        f"errors={n_err} ({rate:.1f}%)  "
        f"flip={n_flip} miss={n_miss} extra={n_extra}"
    )


def run_benchmark(input_dir: str) -> None:
    labeled = _collect_labeled_wav_files(input_dir)
    if not labeled:
        print("No labeled WAV files found.")
        return

    params = {
        **DEFAULT_PARAMS,
        "save_filtered_wav": False,
        "enable_fft_profiles": False,
    }
    # A/B a single param, e.g. BENCH_PARAM_OVERRIDES="pass3_interval_phase_relabel=false".
    for item in (s for s in os.environ.get("BENCH_PARAM_OVERRIDES", "").split(",") if s.strip()):
        k, _, v = item.partition("=")
        k, v = k.strip(), v.strip()
        if not k:
            continue
        if v.lower() in ("true", "false"):
            params[k] = v.lower() == "true"
        else:
            try:
                params[k] = float(v)
            except ValueError:
                params[k] = v

    output_options = {
        "html": False,
        "png": False,
        "csv": False,
        "summary": False,
        "debug": False,
        "filtered_wav": False,
        "spectrogram": False,
        "fft_profiles": False,
        "output_all_passes": False,
        "working_wav_in_output": False,
    }

    sample_rate = int(params.get("preprocess_target_sample_rate", 600))

    total_errors = total_flip = total_miss = total_extra = total_manual_s1 = 0
    file_results = []

    # Group by subdir for subtotals
    current_subdir = None
    sub_errors = sub_flip = sub_miss = sub_extra = sub_s1 = sub_files = 0

    sep = "=" * 68

    for subdir, wav_path, csv_path in labeled:
        if subdir != current_subdir:
            if current_subdir is not None:
                print()
                _print_subtotal(current_subdir, sub_files, sub_s1, sub_errors, sub_flip, sub_miss, sub_extra)
                print(sep)
            current_subdir = subdir
            sub_errors = sub_flip = sub_miss = sub_extra = sub_s1 = sub_files = 0
            print(f"\n{sep}")
            print(f"  {subdir}")
            print(sep)

        wav_name = os.path.basename(wav_path)
        manual_segments = _load_manual_state_sequence(csv_path)
        if not manual_segments:
            print(f"  SKIP {wav_name}  (empty labels)")
            continue

        start_bpm = _extract_start_bpm(wav_path)
        print(f"  {wav_name}")

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                _, _, _, analysis_data = analyze_wav_file(
                    wav_path,
                    params,
                    start_bpm,
                    original_file_path=wav_path,
                    output_directory=tmpdir,
                    output_options=output_options,
                    collect_fft_for_aggregate=False,
                )
            except Exception as exc:
                print(f"    PIPELINE ERROR: {exc}")
                continue

        if analysis_data is None:
            print("    PIPELINE ERROR: returned no data (too few peaks?)")
            continue

        pred_boundaries = analysis_data.get("pass3_state_boundaries") or []
        result = compare_file(manual_segments, pred_boundaries, sample_rate, analysis_data)

        n_s1 = result["manual_s1_count"]
        n_err = result["total_errors"]
        rate = n_err / n_s1 * 100 if n_s1 else 0.0

        total_manual_s1 += n_s1
        total_errors += n_err
        total_flip += result["flip_errors"]
        total_miss += result["miss_errors"]
        total_extra += result["extra_errors"]
        sub_s1 += n_s1
        sub_errors += n_err
        sub_flip += result["flip_errors"]
        sub_miss += result["miss_errors"]
        sub_extra += result["extra_errors"]
        sub_files += 1

        file_results.append({"subdir": subdir, "file": wav_name, **result})

        print(
            f"    s1={n_s1}  pred={result['predicted_s1_count']}  matched={result['matched']}  "
            f"errors={n_err} ({rate:.1f}%)  "
            f"flip={result['flip_errors']} miss={result['miss_errors']} extra={result['extra_errors']}"
        )
        for err in result["errors"]:
            snr_str = f"  snr={err['snr']:.2f}" if err.get("snr") is not None else ""
            peak_str = f"  peak={err['envelope_peak']:.3f}" if err.get("envelope_peak") is not None else ""
            bpm_str = f"  bpm={err['bpm_at_time']:.0f}" if err.get("bpm_at_time") is not None else ""
            s1_str = f"  s1_score={err['label_score_s1']:.2f}" if err.get("label_score_s1") is not None else ""
            print(f"      t={err['time_sec']:.3f}s  [{err['type']}]{snr_str}{peak_str}{bpm_str}{s1_str}")

    # Final subtotal for last subdir
    if current_subdir is not None:
        print()
        _print_subtotal(current_subdir, sub_files, sub_s1, sub_errors, sub_flip, sub_miss, sub_extra)

    # Grand total
    total_rate = total_errors / total_manual_s1 * 100 if total_manual_s1 else 0.0
    print(f"\n{sep}")
    print("GRAND TOTAL")
    print(f"  files          : {len(file_results)}")
    print(f"  manual S1      : {total_manual_s1}")
    print(f"  total errors   : {total_errors}  ({total_rate:.1f}%)")
    print(f"    phase_flip   : {total_flip}")
    print(f"    miss         : {total_miss}")
    print(f"    extra        : {total_extra}")
    print(sep)

    # JSON for programmatic consumption
    grand_total_rate = total_errors / total_manual_s1 * 100 if total_manual_s1 else 0.0
    summary = {
        "input_dir": input_dir,
        "tolerance_sec": TOLERANCE_SEC,
        "files": len(file_results),
        "total_manual_s1": total_manual_s1,
        "total_errors": total_errors,
        "error_rate_pct": round(grand_total_rate, 2),
        "flip_errors": total_flip,
        "miss_errors": total_miss,
        "extra_errors": total_extra,
        "per_file": [
            {k: v for k, v in r.items() if k != "errors"}
            for r in file_results
        ],
        "errors_by_file": [
            {"subdir": r["subdir"], "file": r["file"], "errors": r["errors"]}
            for r in file_results
            if r["errors"]
        ],
    }

    import datetime
    _script_dir = os.path.dirname(os.path.abspath(__file__))

    json_path = os.path.join(_script_dir, "benchmark_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON → {json_path}")

    # Lightweight summary — one row per file, no per-error detail.
    # Used by compare_fixes.py and other tooling; stable when new files are added.
    small = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "input_dir": input_dir,
        "tolerance_sec": TOLERANCE_SEC,
        "files": len(file_results),
        "total_manual_s1": total_manual_s1,
        "total_errors": total_errors,
        "error_rate_pct": round(grand_total_rate, 2),
        "flip_errors": total_flip,
        "miss_errors": total_miss,
        "extra_errors": total_extra,
        "per_file": [
            {
                "file": os.path.basename(r["file"]),
                "subdir": r["subdir"],
                "manual_s1": r["manual_s1_count"],
                "errors": r["total_errors"],
                "extra": r["extra_errors"],
                "miss": r["miss_errors"],
                "flip": r["flip_errors"],
                "error_rate_pct": round(
                    r["total_errors"] / r["manual_s1_count"] * 100
                    if r["manual_s1_count"] else 0.0, 1
                ),
            }
            for r in file_results
        ],
    }
    summary_path = os.path.join(_script_dir, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(small, f, indent=2)
    print(f"Summary → {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        input_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "inputs"
        )

    if not os.path.isdir(input_dir):
        print(f"Error: not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Benchmark: {input_dir}\n")
    run_benchmark(input_dir)
