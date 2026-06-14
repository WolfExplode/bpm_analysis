"""
Correlate the cardiac-state strip, the noise/quiet strip, and the peak labels
for one recording around a chosen time — to explain *why* a state-sequence
violation happens (e.g. an S2 band where the peaks were classified Noise).

It reads exactly what the two HTML strips read:
  * cardiac strip   -> analysis_data["pass3_state_boundaries"]
  * noise strip     -> pass3_noise_unreliable_windows_samples
                       pass3_large_gap_windows_samples
                       pass3_gap_quiet_windows_samples
  * peak markers    -> analysis_data["peak_classifications"]  (peak_type per peak)

Usage (from repo root):
    python debug_helpers/inspect_region.py "inputs/.../file.wav"            # auto: each violation
    python debug_helpers/inspect_region.py "inputs/.../file.wav" --at 5.65  # a time window
    python debug_helpers/inspect_region.py "inputs/.../file.wav" --from 5.0 --to 6.2

For each window it prints, time-ordered, every peak (with its peak_type and
whether that is S1 / S2 / Noise), every cardiac-state segment (with its anchor
metadata), and which noise/quiet/gap windows cover the region.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import soundfile as sf  # noqa: E402

from config import DEFAULT_PARAMS  # noqa: E402
from peak_utils import PeakType  # noqa: E402
from pipeline import analyze_wav_file  # noqa: E402
from debug_helpers.scan_sequence import _bpm_hint_from_name  # noqa: E402
from debug_helpers.state_sequence_detector import find_sequence_violations  # noqa: E402

_OO = {k: False for k in (
    "html", "png", "csv", "summary", "debug", "filtered_wav",
    "spectrogram", "fft_profiles", "output_all_passes", "working_wav_in_output",
)}


def _params():
    return {**DEFAULT_PARAMS, "save_filtered_wav": False, "enable_fft_profiles": False}


def _peak_kind(peak_type: str) -> str:
    if PeakType.is_s1(peak_type):
        return "S1"
    if PeakType.is_s2(peak_type):
        return "S2"
    if peak_type.strip().startswith("Noise"):
        return "NOISE"
    return "other"


def _noise_windows(data):
    """Return list of (lo, hi, kind) from the three noise/quiet/gap window keys."""
    out = []
    specs = [
        ("noise", "pass3_noise_unreliable_windows_samples"),
        ("large_gap", "pass3_large_gap_windows_samples"),
        ("quiet", "pass3_gap_quiet_windows_samples"),
    ]
    for kind, key in specs:
        for w in (data.get(key) or []):
            try:
                if isinstance(w, dict):
                    lo, hi = int(w["start_sample"]), int(w["end_sample"])
                else:
                    lo, hi = int(w[0]), int(w[1])
            except (TypeError, ValueError, KeyError, IndexError):
                continue
            out.append((lo, hi, kind))
    return out


def _run(wav, params):
    with tempfile.TemporaryDirectory() as tmp:
        _, _, _, data = analyze_wav_file(
            wav, params, _bpm_hint_from_name(wav),
            original_file_path=wav, output_directory=tmp,
            output_options=_OO, collect_fft_for_aggregate=False,
        )
    info = sf.info(wav)
    _labels = data.get("pass3_state_labels")
    n = 0 if _labels is None else len(_labels)
    sr = n / (info.frames / float(info.samplerate)) if info.frames else 0.0
    return data, sr


def _print_window(data, sr, lo, hi, header):
    pad = int(0.05 * sr)
    lo_p, hi_p = lo - pad, hi + pad
    print(header)

    # Noise/quiet/gap windows covering the region.
    covering = [w for w in _noise_windows(data) if w[1] > lo_p and w[0] < hi_p]
    if covering:
        for a, b, kind in sorted(covering):
            print(f"    noise-strip: {kind:9} [{a}:{b}] ({a/sr:.2f}-{b/sr:.2f}s)")
    else:
        print("    noise-strip: (no noise/quiet/gap window here)")

    # Time-ordered merge of peaks and state segments.
    pc = data.get("peak_classifications") or {}
    rows = []
    for idx, info in pc.items():
        i = int(idx)
        if lo_p <= i < hi_p:
            pt = info.get("peak_type", "") if isinstance(info, dict) else str(info)
            rows.append((i, "PEAK", f"{_peak_kind(pt):5} '{pt}'"))
    for seg in (data.get("pass3_state_boundaries") or []):
        a0, a1 = int(seg[0]), int(seg[1])
        if a1 > lo_p and a0 < hi_p:
            meta = seg[3] if len(seg) > 3 and isinstance(seg[3], dict) else {}
            m = {k: meta[k] for k in ("s1", "s2", "s1_next", "rebuild_source") if k in meta}
            rows.append((a0, "STATE", f"{seg[2]:9} [{a0}:{a1}] {m}"))
    for t, tag, text in sorted(rows, key=lambda r: (r[0], r[1])):
        print(f"    {t/sr:6.2f}s {t:7} {tag:5} {text}")
    print(flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wav", help="WAV file to inspect.")
    ap.add_argument("--at", type=float, help="Center time (sec); window is +/- --pad seconds.")
    ap.add_argument("--from", dest="t0", type=float, help="Window start (sec).")
    ap.add_argument("--to", dest="t1", type=float, help="Window end (sec).")
    ap.add_argument("--pad", type=float, default=0.6, help="Half-width for --at (sec).")
    ns = ap.parse_args(argv)

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    data, sr = _run(ns.wav, _params())
    if not sr:
        print("Could not determine sample rate.", file=sys.stderr)
        return 2
    print(f"# {os.path.basename(ns.wav)}  (envelope sr ~{sr:.1f} Hz)\n")

    if ns.t0 is not None and ns.t1 is not None:
        _print_window(data, sr, int(ns.t0 * sr), int(ns.t1 * sr),
                      f"=== window {ns.t0:.2f}-{ns.t1:.2f}s ===")
    elif ns.at is not None:
        _print_window(data, sr, int((ns.at - ns.pad) * sr), int((ns.at + ns.pad) * sr),
                      f"=== around {ns.at:.2f}s ===")
    else:
        vios = find_sequence_violations(data.get("pass3_state_boundaries") or [], sample_rate=sr)
        if not vios:
            print("No state-sequence violations in this file.")
            return 0
        print(f"{len(vios)} violation(s); showing each:\n")
        for v in vios:
            c = v["cur_lo"]
            _print_window(data, sr, c - int(ns.pad * sr), c + int(ns.pad * sr),
                          f"=== violation @{v.get('at_sec', c/sr):.2f}s  "
                          f"{v['prev_state']}->{v['cur_state']} [{v['kind']}] ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
