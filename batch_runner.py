"""
Shared batch orchestration for GUI and CLI: dedupe inputs, resolve working WAV,
run analyze_wav_file per file/channel, optional parallel workers, aggregate FFT.
"""
from __future__ import annotations

import copy
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from audio_preprocessing import (
    CHANNEL_MODE_ALL,
    CHANNEL_MODE_MIXED,
    convert_to_wav,
    normalize_channel_mode,
    resolve_wav_for_channel_mode,
)
from file_io import find_companion_wav, normalize_output_filename_stem
from console_logging import configure_analysis_console_logging
from config import param

_EXT_PREFERENCE = {
    ".wav": 2,
    ".flac": 1,
    ".mp3": 1,
    ".m4a": 1,
    ".ogg": 1,
    ".mp4": 1,
    ".mkv": 1,
    ".mov": 1,
}


def extract_start_bpm_from_filename(file_path: str) -> Optional[float]:
    """
    Detect starting BPM from the file name (last rightmost match per pattern).
    Patterns: '[120,60-150bpm]', '120,60-150bpm', '90to132bpm', '150bpm' (case-insensitive).
    """
    base = os.path.basename(file_path)
    comma_matches = list(
        re.finditer(r"(\d+)\s*,\s*(\d+)\s*-\s*(\d+)\s*bpm", base, flags=re.IGNORECASE)
    )
    if comma_matches:
        m = comma_matches[-1]
        try:
            start_bpm = float(m.group(1))
            logging.info(
                "Using starting BPM %.1f from file name tag %s,%s-%sbpm in '%s'.",
                start_bpm,
                m.group(1),
                m.group(2),
                m.group(3),
                base,
            )
            return start_bpm
        except (TypeError, ValueError):
            pass

    to_matches = list(re.finditer(r"(\d+)\s*to\s*(\d+)\s*bpm", base, flags=re.IGNORECASE))
    if to_matches:
        m = to_matches[-1]
        try:
            start_bpm = float(m.group(1))
            end_bpm = float(m.group(2))
            logging.info(
                "Using starting BPM %.1f from file name range (%.1f–%.1f) in '%s'.",
                start_bpm,
                start_bpm,
                end_bpm,
                base,
            )
            return start_bpm
        except (TypeError, ValueError):
            pass

    simple_matches = list(re.finditer(r"(\d+)\s*bpm", base, flags=re.IGNORECASE))
    if not simple_matches:
        return None
    m = simple_matches[-1]
    try:
        bpm_val = float(m.group(1))
        logging.info("Using BPM %s from file name for '%s'.", bpm_val, base)
        return bpm_val
    except (TypeError, ValueError):
        return None


def dedupe_input_files(paths: List[str]) -> List[str]:
    """Prefer WAV when both compressed and WAV share the same base name (emoji/space tolerant)."""
    deduped: Dict[str, Tuple[int, str]] = {}
    for path in paths:
        base_name_only = os.path.splitext(os.path.basename(path))[0]
        key = normalize_output_filename_stem(base_name_only).casefold()
        ext = os.path.splitext(path)[1].lower()
        score = _EXT_PREFERENCE.get(ext, 0)

        if key not in deduped:
            deduped[key] = (score, path)
        else:
            existing_score, _ = deduped[key]
            if score > existing_score:
                deduped[key] = (score, path)
    return [entry[1] for entry in deduped.values()]


def resolve_working_wav(
    file_path: str,
    wav_io_dir: str,
    output_dir: str,
    output_options: Dict,
) -> str:
    """
    Return path to mono/mixed working WAV for analysis (reuse, copy, or convert).
    """
    base_name, ext = os.path.splitext(file_path)
    ext_lower = ext.lower()

    if ext_lower != ".wav":
        source_stem = os.path.basename(base_name)
        source_dir = os.path.dirname(file_path)
        output_stem = normalize_output_filename_stem(source_stem)
        candidate_wav = os.path.join(wav_io_dir, output_stem + ".wav")
        same_dir_wav = find_companion_wav(source_stem, source_dir, wav_io_dir)

        if same_dir_wav:
            if os.path.abspath(os.path.dirname(same_dir_wav)) == os.path.abspath(output_dir):
                wav_path = same_dir_wav
            elif not output_options.get("working_wav_in_output", False):
                wav_path = same_dir_wav
            else:
                wav_path = candidate_wav
                shutil.copy(same_dir_wav, wav_path)
            logging.info(
                "Reusing existing WAV '%s' for '%s' instead of converting.",
                os.path.basename(same_dir_wav),
                os.path.basename(file_path),
            )
        elif os.path.exists(candidate_wav):
            wav_path = candidate_wav
            logging.info(
                "Reusing existing WAV '%s' in working directory for '%s' instead of converting.",
                os.path.basename(candidate_wav),
                os.path.basename(file_path),
            )
        else:
            wav_path = candidate_wav
            if not convert_to_wav(file_path, wav_path):
                raise RuntimeError("File conversion failed.")
    else:
        input_dir = os.path.dirname(file_path)
        if os.path.abspath(output_dir) == os.path.abspath(input_dir):
            wav_path = file_path
        else:
            orig_base = os.path.basename(file_path)
            o_stem, o_ext = os.path.splitext(orig_base)
            out_wav_name = normalize_output_filename_stem(o_stem) + o_ext
            wav_path = os.path.join(wav_io_dir, out_wav_name)
            shutil.copy(file_path, wav_path)

    return wav_path


@dataclass
class BatchJob:
    job_index: int
    file_path: str
    output_dir: str
    params: Dict[str, Any]
    output_options: Dict[str, Any]
    global_bpm_hint: Optional[float]
    bpm_from_filename: bool
    channel_mode: str
    collect_fft_for_aggregate: bool


@dataclass
class BatchJobResult:
    job_index: int
    success: bool
    basename: str
    error: Optional[str] = None
    fft_data_list: List[Any] = field(default_factory=list)
    bpm_rename_info: Optional[Dict[str, Any]] = None
    analyze_had_multiple_channels: bool = False
    output_dir: str = ""


@dataclass(frozen=True)
class BatchParallelSummary:
    """Outcome of ``run_batch_parallel`` (single shared code path for GUI and CLI)."""

    all_ok: bool
    error_basenames: List[str]
    deduped_input_paths: List[str]
    results: List[BatchJobResult]


def run_single_input_file(
    file_path: str,
    output_dir: str,
    output_options: Dict[str, Any],
    params: Dict[str, Any],
    *,
    global_bpm_hint: Optional[float] = None,
    bpm_from_filename: bool = False,
    channel_mode: str = CHANNEL_MODE_MIXED,
    collect_fft_for_aggregate: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    conversion_status_callback: Optional[Callable[[str], None]] = None,
    batch_status_prefix: Optional[str] = None,
) -> BatchJobResult:
    """
    One original input: WAV resolve, optional channel split, analyze_wav_file per channel.
    """
    basename = os.path.basename(file_path)
    working_tmp: Optional[str] = None
    file_start_time = time.time()
    fft_data_list: List[Any] = []
    bpm_rename_info_for_file: Optional[Dict[str, Any]] = None
    multi_channel = False

    try:
        os.makedirs(output_dir, exist_ok=True)

        if output_options.get("working_wav_in_output", False):
            wav_io_dir = output_dir
        else:
            working_tmp = tempfile.mkdtemp(prefix="bpm_working_")
            wav_io_dir = working_tmp

        if conversion_status_callback is not None and os.path.splitext(file_path)[1].lower() != ".wav":
            conversion_status_callback(f"Converting {basename}...")

        wav_path = resolve_working_wav(file_path, wav_io_dir, output_dir, output_options)

        mode = normalize_channel_mode(channel_mode)
        wav_files_to_analyze = [wav_path]
        if mode != CHANNEL_MODE_MIXED:
            if conversion_status_callback is not None:
                if mode == CHANNEL_MODE_ALL:
                    conversion_status_callback(f"{basename}: Splitting stereo into mono channels...")
                else:
                    conversion_status_callback(
                        f"{basename}: Extracting {mode} channel..."
                    )
            wav_files_to_analyze = resolve_wav_for_channel_mode(wav_path, wav_io_dir, mode)
        multi_channel = len(wav_files_to_analyze) > 1

        if global_bpm_hint is not None:
            file_start_bpm_hint = global_bpm_hint
        elif bpm_from_filename:
            file_start_bpm_hint = extract_start_bpm_from_filename(file_path)
        else:
            file_start_bpm_hint = None

        from pipeline import analyze_wav_file

        for ch_idx, wav_for_analysis in enumerate(wav_files_to_analyze, start=1):
            if batch_status_prefix is not None:
                ch_suffix = f" (CH{ch_idx})" if len(wav_files_to_analyze) > 1 else ""
                prefix = batch_status_prefix + ch_suffix

                def _make_pc(pref: str) -> Callable[[str], None]:
                    def _cb(detail: str) -> None:
                        if progress_callback is not None:
                            progress_callback(f"{pref}: {detail}")

                    return _cb

                channel_progress = _make_pc(prefix)
            else:
                channel_progress = progress_callback

            _figure, fft_data, bpm_rename_info, _ = analyze_wav_file(
                wav_for_analysis,
                params,
                file_start_bpm_hint,
                original_file_path=file_path,
                output_directory=output_dir,
                output_options=output_options,
                collect_fft_for_aggregate=collect_fft_for_aggregate,
                progress_callback=channel_progress,
            )
            if fft_data is not None:
                fft_data_list.append(fft_data)
            if bpm_rename_info_for_file is None and bpm_rename_info is not None:
                bpm_rename_info_for_file = bpm_rename_info

        duration = time.time() - file_start_time
        logging.info(
            "=== Total processing time for '%s': %.2f seconds (including conversion & analysis). ===",
            basename,
            duration,
        )
        return BatchJobResult(
            job_index=-1,
            success=True,
            basename=basename,
            fft_data_list=fft_data_list,
            bpm_rename_info=bpm_rename_info_for_file,
            analyze_had_multiple_channels=multi_channel,
            output_dir=os.path.abspath(output_dir),
        )
    except Exception as e:
        logging.error("Error processing '%s': %s", basename, e)
        return BatchJobResult(
            job_index=-1,
            success=False,
            basename=basename,
            error=str(e),
            fft_data_list=fft_data_list,
            bpm_rename_info=bpm_rename_info_for_file,
            analyze_had_multiple_channels=multi_channel,
            output_dir=os.path.abspath(output_dir),
        )
    finally:
        if working_tmp:
            shutil.rmtree(working_tmp, ignore_errors=True)


def _ensure_pool_process_logging(params: Dict[str, Any]) -> None:
    """
    ProcessPoolExecutor workers are fresh processes without the parent's logging setup;
    without a StreamHandler, only WARNING+ reaches the console (lastResort).
    """
    root = logging.getLogger()
    if root.handlers:
        return
    configure_analysis_console_logging(
        general_debug=bool(param(params, "general_console_logging")),
        quiet=False,
        stream=sys.stdout,
    )


def _process_one_job(job: BatchJob) -> BatchJobResult:
    """Top-level worker entry for ProcessPoolExecutor (Windows spawn)."""
    _ensure_pool_process_logging(job.params)
    result = run_single_input_file(
        job.file_path,
        job.output_dir,
        job.output_options,
        job.params,
        global_bpm_hint=job.global_bpm_hint,
        bpm_from_filename=job.bpm_from_filename,
        channel_mode=job.channel_mode,
        collect_fft_for_aggregate=job.collect_fft_for_aggregate,
        progress_callback=None,
        conversion_status_callback=None,
    )
    result.job_index = job.job_index
    return result


def run_batch_parallel(
    input_paths: List[str],
    params: Dict[str, Any],
    output_options: Dict[str, Any],
    *,
    base_output_dir: str,
    output_next_to_input: bool,
    max_workers: int = 1,
    global_bpm_hint: Optional[float] = None,
    bpm_from_filename: bool = False,
    channel_mode: str = CHANNEL_MODE_MIXED,
    sequential_progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    sequential_conversion_callback: Optional[Callable[[int, int, str, str], None]] = None,
) -> BatchParallelSummary:
    """
    Dedupe inputs, run each file (optionally in parallel), aggregate FFT when >= 2 inputs.

    When ``max_workers == 1``, optional ``sequential_*_callback`` receive
    (job_index, total_jobs, file_path, message) for UI updates. Ignored when ``max_workers > 1``.
    """
    if max_workers < 1:
        max_workers = 1

    opts = copy.deepcopy(output_options)
    if opts.get("png") and max_workers > 1:
        logging.warning(
            "PNG export uses Kaleido/Chromium per process; high --jobs with PNG enabled can use large RAM. "
            "Consider reducing --jobs or disabling PNG for huge batches."
        )

    params_run = copy.deepcopy(params)
    deduped = dedupe_input_files(input_paths)
    total_files = len(deduped)
    collect_fft_for_aggregate = total_files >= 2

    jobs: List[BatchJob] = []
    for job_index, file_path in enumerate(deduped):
        if output_next_to_input:
            output_dir = os.path.dirname(file_path) or base_output_dir
        else:
            output_dir = base_output_dir
        os.makedirs(output_dir, exist_ok=True)

        job_opts = copy.deepcopy(opts)
        jobs.append(
            BatchJob(
                job_index=job_index,
                file_path=file_path,
                output_dir=os.path.abspath(output_dir),
                params=copy.deepcopy(params_run),
                output_options=job_opts,
                global_bpm_hint=global_bpm_hint,
                bpm_from_filename=bpm_from_filename,
                channel_mode=channel_mode,
                collect_fft_for_aggregate=collect_fft_for_aggregate,
            )
        )

    results: List[Optional[BatchJobResult]] = [None] * len(jobs)
    errors: List[str] = []

    if max_workers == 1:
        for job in jobs:
            i = job.job_index
            total = len(jobs)
            fp = job.file_path
            use_callbacks = (
                sequential_progress_callback is not None
                or sequential_conversion_callback is not None
            )
            if not use_callbacks:
                r = _process_one_job(job)
            else:
                prefix = f"({i + 1}/{total}) {os.path.basename(fp)}"

                def _make_conv(ji: int, jt: int, jfp: str) -> Callable[[str], None]:
                    def _conv(msg: str) -> None:
                        if sequential_conversion_callback is not None:
                            sequential_conversion_callback(ji, jt, jfp, msg)

                    return _conv

                def _make_prog(ji: int, jt: int, jfp: str) -> Callable[[str], None]:
                    def _prog(detail: str) -> None:
                        if sequential_progress_callback is not None:
                            sequential_progress_callback(ji, jt, jfp, detail)

                    return _prog

                r = run_single_input_file(
                    job.file_path,
                    job.output_dir,
                    job.output_options,
                    job.params,
                    global_bpm_hint=job.global_bpm_hint,
                    bpm_from_filename=job.bpm_from_filename,
                    channel_mode=job.channel_mode,
                    collect_fft_for_aggregate=job.collect_fft_for_aggregate,
                    progress_callback=_make_prog(i, total, fp) if sequential_progress_callback else None,
                    conversion_status_callback=_make_conv(i, total, fp)
                    if sequential_conversion_callback
                    else None,
                    batch_status_prefix=prefix,
                )
            r.job_index = job.job_index
            results[job.job_index] = r
            if not r.success:
                errors.append(r.basename)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {executor.submit(_process_one_job, job): job for job in jobs}
            for fut in as_completed(future_to_job):
                job = future_to_job[fut]
                try:
                    r = fut.result()
                    results[job.job_index] = r
                    if not r.success:
                        errors.append(r.basename)
                except Exception as e:
                    logging.exception("Worker failed for %s", job.file_path)
                    results[job.job_index] = BatchJobResult(
                        job_index=job.job_index,
                        success=False,
                        basename=os.path.basename(job.file_path),
                        error=str(e),
                    )
                    errors.append(os.path.basename(job.file_path))

    fft_results_for_aggregate: List[Any] = []
    for r in sorted((x for x in results if x is not None), key=lambda x: x.job_index):
        if r.success:
            fft_results_for_aggregate.extend(r.fft_data_list)

    if len(fft_results_for_aggregate) >= 2:
        try:
            from fft_profiles import aggregate_fft_profiles, save_aggregate_fft_profiles_html

            freqs, agg_r1, agg_r2, agg_b1, agg_b2 = aggregate_fft_profiles(
                fft_results_for_aggregate, params_run
            )
            aggregate_path = os.path.join(base_output_dir, "fft_profiles_aggregate.html")
            save_aggregate_fft_profiles_html(
                freqs, agg_r1, agg_r2, agg_b1, agg_b2, aggregate_path, params_run
            )
        except Exception as e:
            logging.warning("Aggregate FFT profiles failed: %s", e)

    all_ok = len(errors) == 0
    ordered_results: List[BatchJobResult] = []
    for i in range(len(jobs)):
        r = results[i]
        if r is None:
            raise RuntimeError(f"Missing batch result for job index {i}")
        ordered_results.append(r)
    return BatchParallelSummary(
        all_ok=all_ok,
        error_basenames=errors,
        deduped_input_paths=list(deduped),
        results=ordered_results,
    )
