"""
Headless batch analysis CLI with optional parallel workers (--jobs).

Output toggles and most options default from ui_settings.json (same as the GUI) when
you do not pass the corresponding CLI flag; any flag you pass overrides that field.

Usage (from repo root):
  python batch_cli.py --jobs 2 path/to/a.wav path/to/b.mp3
  python batch_cli.py --output-dir out --png --no-html *.wav

See python batch_cli.py --help
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List

from batch_runner import run_batch_parallel
from config import DEFAULT_PARAMS
from console_logging import configure_analysis_console_logging
from ui_settings_loader import batch_cli_defaults_from_ui_settings, load_ui_settings_json


def _meta_required_outputs_ok(opts: Dict[str, Any]) -> bool:
    meta = frozenset({
        "html_s1_s2_hover_on_by_default",
        "html_inline_interactive_script",
        "working_wav_in_output",
    })
    required = {k: v for k, v in opts.items() if k not in meta}
    return any(required.values())


def main(argv: List[str] | None = None) -> int:
    _SUP = argparse.SUPPRESS
    parser = argparse.ArgumentParser(
        description=(
            "Batch heart-sound BPM analysis (headless). Same pipeline as the GUI. "
            "Defaults match ui_settings.json when that file exists (except where noted); "
            "each CLI flag you pass overrides only that setting."
        ),
    )
    parser.add_argument("inputs", nargs="+", metavar="PATH", help="Audio file(s) to analyze.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=_SUP,
        metavar="N",
        help="Parallel processes (default: from ui_settings.json key cli_batch_jobs, else 1).",
    )
    parser.add_argument(
        "--output-dir",
        default=_SUP,
        metavar="DIR",
        help="Output base directory (default: ui_settings.json cli_output_dir, else processed_files).",
    )
    parser.add_argument(
        "--output-next-to-input",
        dest="output_next_to_input",
        action="store_const",
        const=True,
        default=_SUP,
        help="Write outputs beside each input file (overrides ui_settings.json output_to_input_dir).",
    )
    parser.add_argument(
        "--no-output-next-to-input",
        dest="output_next_to_input",
        action="store_const",
        const=False,
        default=_SUP,
        help="Write outputs under --output-dir (overrides ui_settings.json output_to_input_dir).",
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=_SUP,
        metavar="FLOAT",
        help="Global starting BPM hint (default: starting_bpm from ui_settings.json if set).",
    )
    parser.add_argument(
        "--bpm-from-filename",
        dest="bpm_from_filename",
        action="store_const",
        const=True,
        default=_SUP,
        help="Parse starting BPM from each file name (default: ui_settings.json bpm_from_filename).",
    )
    parser.add_argument(
        "--no-bpm-from-filename",
        dest="bpm_from_filename",
        action="store_const",
        const=False,
        default=_SUP,
        help="Do not parse BPM from file names.",
    )
    parser.add_argument(
        "--all-channels",
        dest="all_channels",
        action="store_const",
        const=True,
        default=_SUP,
        help="Analyze each audio channel separately (default: ui_settings.json process_all_channels).",
    )
    parser.add_argument(
        "--no-all-channels",
        dest="all_channels",
        action="store_const",
        const=False,
        default=_SUP,
        help="Analyze mixed channel only.",
    )
    parser.add_argument(
        "--optimize-long-plots",
        dest="optimize_long_plots",
        action="store_const",
        const=True,
        default=_SUP,
        help="Enable long-plot optimizations (default: ui_settings.json optimize_long_plots).",
    )
    parser.add_argument(
        "--no-optimize-long-plots",
        dest="optimize_long_plots",
        action="store_const",
        const=False,
        default=_SUP,
        help="Disable long-plot optimizations.",
    )
    parser.add_argument(
        "--algorithm-verbose",
        dest="algorithm_verbose",
        action="store_const",
        const=True,
        default=_SUP,
        help="Verbose algorithm INFO logs (default: ui_settings.json algorithm_console_logging).",
    )
    parser.add_argument(
        "--no-algorithm-verbose",
        dest="algorithm_verbose",
        action="store_const",
        const=False,
        default=_SUP,
        help="Reduce algorithm-detail console logging.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Root log level WARNING (ignores ui_settings.json general_console_logging).",
    )

    og = parser.add_argument_group(
        "Outputs (default: each flag from ui_settings.json / GUI checkboxes; pass a flag to override)"
    )
    og.add_argument("--html", dest="html_ex", action="store_const", const=True, default=_SUP, help="Enable HTML BPM plot.")
    og.add_argument("--no-html", dest="html_ex", action="store_const", const=False, default=_SUP, help="Disable HTML BPM plot.")
    og.add_argument("--png", dest="png_ex", action="store_const", const=True, default=_SUP, help="Enable plot PNG (Kaleido).")
    og.add_argument("--no-png", dest="png_ex", action="store_const", const=False, default=_SUP, help="Disable plot PNG.")
    og.add_argument("--csv", dest="csv_ex", action="store_const", const=True, default=_SUP, help="Enable CSV export.")
    og.add_argument("--no-csv", dest="csv_ex", action="store_const", const=False, default=_SUP, help="Disable CSV export.")
    og.add_argument("--summary", dest="summary_ex", action="store_const", const=True, default=_SUP, help="Enable summary report.")
    og.add_argument("--no-summary", dest="summary_ex", action="store_const", const=False, default=_SUP, help="Disable summary report.")
    og.add_argument("--debug", dest="debug_ex", action="store_const", const=True, default=_SUP, help="Enable debug markdown report.")
    og.add_argument("--no-debug", dest="debug_ex", action="store_const", const=False, default=_SUP, help="Disable debug report.")
    og.add_argument("--filtered-wav", dest="filtered_wav_ex", action="store_const", const=True, default=_SUP, help="Save filtered WAV.")
    og.add_argument(
        "--no-filtered-wav",
        dest="filtered_wav_ex",
        action="store_const",
        const=False,
        default=_SUP,
        help="Skip filtered WAV.",
    )
    og.add_argument(
        "--working-wav-in-output",
        dest="working_wav_ex",
        action="store_const",
        const=True,
        default=_SUP,
        help="Keep converted working WAV in output folder.",
    )
    og.add_argument(
        "--no-working-wav-in-output",
        dest="working_wav_ex",
        action="store_const",
        const=False,
        default=_SUP,
        help="Use a temp dir for working WAV.",
    )
    og.add_argument("--spectrogram", dest="spectrogram_ex", action="store_const", const=True, default=_SUP, help="Enable spectrogram PNG.")
    og.add_argument(
        "--no-spectrogram",
        dest="spectrogram_ex",
        action="store_const",
        const=False,
        default=_SUP,
        help="Disable spectrogram.",
    )
    og.add_argument("--fft-profiles", dest="fft_profiles_ex", action="store_const", const=True, default=_SUP, help="Enable FFT profile HTML.")
    og.add_argument(
        "--no-fft-profiles",
        dest="fft_profiles_ex",
        action="store_const",
        const=False,
        default=_SUP,
        help="Disable FFT profiles.",
    )
    og.add_argument("--regression-log", dest="regression_log_ex", action="store_const", const=True, default=_SUP, help="Append regression log.")
    og.add_argument(
        "--no-regression-log",
        dest="regression_log_ex",
        action="store_const",
        const=False,
        default=_SUP,
        help="Disable regression log.",
    )

    ag = parser.add_argument_group("HTML extras (defaults from ui_settings.json when not overridden)")
    ag.add_argument(
        "--output-all-passes",
        dest="output_all_passes_ex",
        action="store_const",
        const=True,
        default=_SUP,
        help="Generate intermediate pass outputs.",
    )
    ag.add_argument(
        "--no-output-all-passes",
        dest="output_all_passes_ex",
        action="store_const",
        const=False,
        default=_SUP,
        help="Only final pass outputs where applicable.",
    )
    ag.add_argument(
        "--html-s1-s2-hover-on",
        dest="html_s1_s2_hover_ex",
        action="store_const",
        const=True,
        default=_SUP,
        help="HTML: S1/S2 hover tooltips on by default.",
    )
    ag.add_argument(
        "--no-html-s1-s2-hover-on",
        dest="html_s1_s2_hover_ex",
        action="store_const",
        const=False,
        default=_SUP,
        help="HTML: S1/S2 hover tooltips off by default.",
    )
    ag.add_argument(
        "--html-inline-script",
        dest="html_inline_script_ex",
        action="store_const",
        const=True,
        default=_SUP,
        help="HTML: inline interactive script instead of external JS.",
    )
    ag.add_argument(
        "--no-html-inline-script",
        dest="html_inline_script_ex",
        action="store_const",
        const=False,
        default=_SUP,
        help="HTML: use external interactive script.",
    )

    ns = parser.parse_args(argv)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    settings_path = os.path.join(os.getcwd(), "ui_settings.json")
    ui_raw = load_ui_settings_json(settings_path)
    merged = batch_cli_defaults_from_ui_settings(ui_raw)

    configure_analysis_console_logging(
        general_debug=bool(merged["general_console_logging"]),
        quiet=bool(ns.quiet),
        stream=sys.stdout,
    )
    if ui_raw is not None:
        logging.info("Loaded defaults from %s", settings_path)

    inputs: List[str] = []
    for raw in ns.inputs:
        p = os.path.normpath(raw)
        if not os.path.isfile(p):
            logging.error("Not a file (skipping): %s", raw)
            continue
        inputs.append(p)
    if not inputs:
        logging.error("No valid input files.")
        return 2

    jobs = merged["jobs"] if not hasattr(ns, "jobs") else max(1, int(ns.jobs))
    out_dir = merged["output_dir"] if not hasattr(ns, "output_dir") else str(ns.output_dir)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    output_next_to_input = merged["output_next_to_input"]
    if hasattr(ns, "output_next_to_input"):
        output_next_to_input = bool(ns.output_next_to_input)

    global_bpm_hint = merged["global_bpm_hint"]
    if hasattr(ns, "bpm"):
        global_bpm_hint = float(ns.bpm)

    bpm_from_filename = bool(merged["bpm_from_filename"])
    if hasattr(ns, "bpm_from_filename"):
        bpm_from_filename = bool(ns.bpm_from_filename)
    if global_bpm_hint is not None:
        bpm_from_filename = False

    process_all_channels = merged["process_all_channels"]
    if hasattr(ns, "all_channels"):
        process_all_channels = bool(ns.all_channels)

    optimize_long = merged["optimize_long_plots"]
    if hasattr(ns, "optimize_long_plots"):
        optimize_long = bool(ns.optimize_long_plots)

    algorithm_verbose = merged["algorithm_console_logging"]
    if hasattr(ns, "algorithm_verbose"):
        algorithm_verbose = bool(ns.algorithm_verbose)

    opts = dict(merged["output_options"])
    overrides = [
        ("html", "html_ex"),
        ("png", "png_ex"),
        ("csv", "csv_ex"),
        ("summary", "summary_ex"),
        ("debug", "debug_ex"),
        ("filtered_wav", "filtered_wav_ex"),
        ("working_wav_in_output", "working_wav_ex"),
        ("spectrogram", "spectrogram_ex"),
        ("fft_profiles", "fft_profiles_ex"),
        ("regression_log", "regression_log_ex"),
    ]
    for opt_key, ns_attr in overrides:
        if hasattr(ns, ns_attr):
            opts[opt_key] = bool(getattr(ns, ns_attr))
    if hasattr(ns, "output_all_passes_ex"):
        opts["output_all_passes"] = bool(ns.output_all_passes_ex)
    if hasattr(ns, "html_s1_s2_hover_ex"):
        opts["html_s1_s2_hover_on_by_default"] = bool(ns.html_s1_s2_hover_ex)
    if hasattr(ns, "html_inline_script_ex"):
        opts["html_inline_interactive_script"] = bool(ns.html_inline_script_ex)

    if not _meta_required_outputs_ok(opts):
        logging.error(
            "At least one output type must be enabled (html, png, csv, summary, debug, "
            "filtered_wav, spectrogram, fft_profiles, or regression_log). "
            "Enable one in the GUI or pass e.g. --html on the command line.",
        )
        return 2

    if opts.get("regression_log") and jobs > 1:
        logging.warning("Regression log is disabled when --jobs > 1.")

    params = DEFAULT_PARAMS.copy()
    if optimize_long:
        params["optimize_long_plots"] = True
    params["algorithm_console_logging"] = algorithm_verbose

    if opts.get("regression_log") and jobs <= 1:
        from time_utils import timestamp_str

        reg_path = os.path.join(out_dir, "regression_testing_output_log.md")
        try:
            with open(reg_path, "w", encoding="utf-8") as log_file:
                log_file.write("# Regression Testing Output Log\n")
                log_file.write(f"*Generated on: {timestamp_str()}*\n\n")
        except OSError as e:
            logging.error("Failed to initialize regression log: %s", e)
        else:
            opts["regression_log_path"] = reg_path

    all_ok, errors = run_batch_parallel(
        inputs,
        params,
        opts,
        base_output_dir=out_dir,
        output_next_to_input=output_next_to_input,
        max_workers=jobs,
        global_bpm_hint=global_bpm_hint,
        bpm_from_filename=bpm_from_filename,
        process_all_channels=process_all_channels,
    )

    if not all_ok:
        logging.error("Batch finished with errors: %s", ", ".join(errors))
        return 1
    logging.info("Batch finished successfully (%d file(s)).", len(inputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
