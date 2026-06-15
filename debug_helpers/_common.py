"""
Shared plumbing for the debug_helpers tools.

Every scanner/auditor here runs the same pipeline the GUI runs, on the same
input WAVs, with output artifacts suppressed — then feeds the result to a pure
detector. The boilerplate for *that* (params, output options, the
tempdir + ``analyze_wav_file`` call, envelope sample-rate recovery, WAV
collection, UTF-8 stdout, and the CPU-bound process pool) used to be copied into
every file. It lives here once.

Pure detectors (``*_detector.py``) deliberately do **not** import this — they
have no pipeline dependency so they stay unit-testable.
"""
from __future__ import annotations

import glob
import logging
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import soundfile as sf  # noqa: E402

from config import DEFAULT_PARAMS  # noqa: E402
from pipeline import analyze_wav_file  # noqa: E402

# Suppress every output artifact: these tools only want analysis_data.
OUTPUT_OPTIONS = {
    "html": False, "png": False, "csv": False, "summary": False, "debug": False,
    "filtered_wav": False, "spectrogram": False, "fft_profiles": False,
    "output_all_passes": False, "working_wav_in_output": False,
}


def params(**overrides):
    """DEFAULT_PARAMS with artifact writes off, plus any caller overrides."""
    p = {**DEFAULT_PARAMS, "save_filtered_wav": False, "enable_fft_profiles": False}
    p.update(overrides)
    return p


def reconfigure_stdio():
    """Force UTF-8 on stdout/stderr so CJK/emoji filenames survive a redirect
    to a cp1252 file on Windows (otherwise the first such print dies)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def bpm_hint_from_name(path):
    """Parse a leading starting BPM from a "[107,71-108bpm]" style name, else
    None. Mirrors the GUI/batch default (bpm_from_filename) so results match
    real runs."""
    m = re.search(r"\[(\d+(?:\.\d+)?)\s*,", os.path.basename(path))
    return float(m.group(1)) if m else None


def env_sample_rate(wav_path, n_samples):
    """Envelope sample rate = analysed samples / wall-clock duration of the WAV.
    Returns None if the WAV can't be read or has zero duration."""
    try:
        info = sf.info(wav_path)
        dur = info.frames / float(info.samplerate)
        return (n_samples / dur) if dur > 0 else None
    except Exception:  # noqa: BLE001
        return None


def collect_wavs(paths):
    """Expand a mix of WAV files and directories into a sorted WAV list."""
    wavs = []
    for p in paths:
        if os.path.isfile(p) and p.lower().endswith(".wav"):
            wavs.append(p)
        elif os.path.isdir(p):
            wavs.extend(sorted(glob.glob(os.path.join(p, "**", "*.wav"), recursive=True)))
    return wavs


def run_pipeline(wav_path, run_params, bpm_hint=None):
    """Run analyze_wav_file with artifacts off; return analysis_data or None.

    Pipeline exceptions are logged and swallowed (returns None) so a batch scan
    survives a single bad file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _, _, _, data = analyze_wav_file(
                wav_path, run_params, bpm_hint,
                original_file_path=wav_path,
                output_directory=tmp,
                output_options=OUTPUT_OPTIONS,
                collect_fft_for_aggregate=False,
            )
        except Exception as exc:  # noqa: BLE001
            logging.error("pipeline error on %s: %s", os.path.basename(wav_path), exc)
            return None
    return data or None


# ---------------------------------------------------------------------------
# Process pool. analyze_wav_file is CPU-bound, so processes (not threads) give
# real speedup. scan_fn must be a module-level function (picklable for Windows
# spawn); params ship to each worker once via the initializer.
# ---------------------------------------------------------------------------

_WORKER = {}


def _pool_init(scan_fn, run_params):
    _WORKER["fn"] = scan_fn
    _WORKER["params"] = run_params
    reconfigure_stdio()
    logging.getLogger().setLevel(logging.ERROR)


def _pool_worker(wav):
    return wav, _WORKER["fn"](wav, _WORKER["params"])


def parallel_scan(wavs, scan_fn, run_params, jobs):
    """Yield ``(wav, scan_fn(wav, params))`` for every WAV, serial or pooled.

    ``jobs <= 1`` runs in-process (good for debugging); otherwise a
    ProcessPoolExecutor with ``jobs`` workers. Results stream back in
    submission order.
    """
    jobs = max(1, int(jobs))
    if jobs == 1:
        for wav in wavs:
            yield wav, scan_fn(wav, run_params)
    else:
        with ProcessPoolExecutor(
            max_workers=jobs, initializer=_pool_init, initargs=(scan_fn, run_params)
        ) as ex:
            yield from ex.map(_pool_worker, wavs)


def default_jobs():
    return max(1, (os.cpu_count() or 2) - 1)
