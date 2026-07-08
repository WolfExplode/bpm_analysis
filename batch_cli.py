"""
Headless batch analysis CLI with optional parallel workers (--jobs).

Output toggles and most options default from ui_settings.json (same as the GUI) when
you do not pass the corresponding CLI flag; any flag you pass overrides that field.

Usage (from repo root):
  python batch_cli.py --jobs 2 path/to/a.wav path/to/b.mp3
  python batch_cli.py --output-dir out --png --no-html *.wav
  python batch_cli.py --rename-input-with-bpm *.wav   # or enable in ui_settings.json

See python batch_cli.py --help
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List

from batch_runner import run_batch_parallel
from bpm_input_rename import rename_analysis_outputs_after_input_bpm_rename, try_rename_input_with_bpm_annotation
from config import DEFAULT_PARAMS, ui_settings_path
from console_logging import configure_analysis_console_logging
from audio_preprocessing import CHANNEL_MODE_ALL, CHANNEL_MODE_MIXED, normalize_channel_mode
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
        help="Analyze each stereo channel separately (same as --channel all).",
    )
    parser.add_argument(
        "--no-all-channels",
        dest="all_channels",
        action="store_const",
        const=False,
        default=_SUP,
        help="Use mixed mono downmix (same as --channel mixed).",
    )
    parser.add_argument(
        "--channel",
        choices=("mixed", "left", "right", "all"),
        default=_SUP,
        help="Channel selection: mixed (default), left, right, or all (default: ui_settings.json channel_mode).",
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
        "--springer",
        dest="use_springer_algorithm",
        action="store_const",
        const=True,
        default=_SUP,
        help="Use the Springer 2015 HSMM algorithm instead of the native multi-pass pipeline "
        "(default: ui_settings.json use_springer_algorithm).",
    )
    parser.add_argument(
        "--no-springer",
        dest="use_springer_algorithm",
        action="store_const",
        const=False,
        default=_SUP,
        help="Use the native multi-pass pipeline (default behavior).",
    )
    parser.add_argument(
        "--springer-model",
        dest="springer_model",
        default=_SUP,
        help="Springer model .npz file name under springer2015/ (default: ui_settings.json springer_model).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Root log level WARNING (ignores ui_settings.json general_console_logging).",
    )
    parser.add_argument(
        "--rename-input-with-bpm",
        dest="rename_input_with_bpm",
        action="store_const",
        const=True,
        default=_SUP,
        help="After success, rename each input with BPM tag (default: ui_settings.json rename_input_with_bpm).",
    )
    parser.add_argument(
        "--no-rename-input-with-bpm",
        dest="rename_input_with_bpm",
        action="store_const",
        const=False,
        default=_SUP,
        help="Do not rename input files after analysis.",
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
    og.add_argument(
        "--filtered-wav",
        dest="filtered_wav_ex",
        action="store_const",
        const=True,
        default=_SUP,
        help="Save bandpass and out-of-band (inverse) debug WAVs.",
    )
    og.add_argument(
        "--no-filtered-wav",
        dest="filtered_wav_ex",
        action="store_const",
        const=False,
        default=_SUP,
        help="Skip bandpass / inverse debug WAVs.",
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
    settings_path = ui_settings_path()
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

    channel_mode = normalize_channel_mode(merged["channel_mode"])
    if hasattr(ns, "channel") and ns.channel is not _SUP:
        channel_mode = normalize_channel_mode(ns.channel)
    elif hasattr(ns, "all_channels") and ns.all_channels is not _SUP:
        channel_mode = CHANNEL_MODE_ALL if ns.all_channels else CHANNEL_MODE_MIXED

    optimize_long = merged["optimize_long_plots"]
    if hasattr(ns, "optimize_long_plots"):
        optimize_long = bool(ns.optimize_long_plots)

    algorithm_verbose = merged["algorithm_console_logging"]
    if hasattr(ns, "algorithm_verbose"):
        algorithm_verbose = bool(ns.algorithm_verbose)

    rename_input_with_bpm = bool(merged["rename_input_with_bpm"])
    if hasattr(ns, "rename_input_with_bpm"):
        rename_input_with_bpm = bool(ns.rename_input_with_bpm)

    use_springer_algorithm = bool(merged["use_springer_algorithm"])
    if hasattr(ns, "use_springer_algorithm"):
        use_springer_algorithm = bool(ns.use_springer_algorithm)

    springer_model = merged["springer_model"]
    if hasattr(ns, "springer_model"):
        springer_model = ns.springer_model

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
            "filtered_wav, spectrogram, or fft_profiles). "
            "Enable one in the GUI or pass e.g. --html on the command line.",
        )
        return 2

    params = DEFAULT_PARAMS.copy()
    if optimize_long:
        params["optimize_long_plots"] = True
    params["algorithm_console_logging"] = algorithm_verbose
    params["general_console_logging"] = bool(merged["general_console_logging"])
    params["use_springer_algorithm"] = use_springer_algorithm
    params["springer_model"] = springer_model

    summary = run_batch_parallel(
        inputs,
        params,
        opts,
        base_output_dir=out_dir,
        output_next_to_input=output_next_to_input,
        max_workers=jobs,
        global_bpm_hint=global_bpm_hint,
        bpm_from_filename=bpm_from_filename,
        channel_mode=channel_mode,
    )

    if rename_input_with_bpm:
        for fp, res in zip(summary.deduped_input_paths, summary.results):
            if not res.success:
                continue
            info = res.bpm_rename_info
            if info is None:
                continue
            if res.analyze_had_multiple_channels:
                logging.warning(
                    "BPM rename skipped for '%s': multi-channel analysis.",
                    os.path.basename(fp),
                )
                continue
            new_path = try_rename_input_with_bpm_annotation(fp, info)
            if new_path:
                rename_analysis_outputs_after_input_bpm_rename(fp, new_path, res.output_dir)

    if not summary.all_ok:
        logging.error("Batch finished with errors: %s", ", ".join(summary.error_basenames))
        return 1
    logging.info("Batch finished successfully (%d file(s)).", len(summary.deduped_input_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
