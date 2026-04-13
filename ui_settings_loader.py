"""
Load ui_settings.json for headless tools (batch_cli) without importing the GUI.

Keep UI_SETTINGS_OUTPUT_OPTION_KEYS in sync with gui.OUTPUT_FILE_OPTIONS keys (order + names).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from config import DEFAULT_OUTPUT_OPTIONS

# Must match gui.OUTPUT_FILE_OPTIONS (first element of each tuple).
UI_SETTINGS_OUTPUT_OPTION_KEYS = (
    "html",
    "fft_profiles",
    "png",
    "csv",
    "spectrogram",
    "summary",
    "filtered_wav",
    "working_wav_in_output",
    "debug",
    "regression_log",
)


def migrate_ui_settings_keys(settings: Dict[str, Any]) -> None:
    """In-place migration for older ui_settings.json (same rules as gui.load_ui_settings)."""
    if "algorithm_console_logging" not in settings and "verbose_console_logging" in settings:
        settings["algorithm_console_logging"] = settings["verbose_console_logging"]
    if "general_console_logging" not in settings and "general_debug_logging" in settings:
        settings["general_console_logging"] = settings["general_debug_logging"]


def load_ui_settings_json(settings_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(settings_path):
        return None
    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        migrate_ui_settings_keys(data)
        return data
    except (OSError, json.JSONDecodeError) as e:
        logging.warning("Could not load UI settings from %s: %s", settings_path, e)
        return None


def parse_starting_bpm_from_settings(settings: Dict[str, Any]) -> Optional[float]:
    s = (settings.get("starting_bpm") or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def batch_cli_defaults_from_ui_settings(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Defaults for batch_cli when CLI flags are omitted (mirrors GUI batch behavior).

    Optional / shared JSON keys:
    - cli_batch_jobs: int (default 1; GUI "Parallel batch jobs" uses the same key as batch_cli --jobs)
    - cli_output_dir: str (default 'processed_files')
    """
    s = settings or {}
    opts = DEFAULT_OUTPUT_OPTIONS.copy()
    for key in UI_SETTINGS_OUTPUT_OPTION_KEYS:
        sk = "output_" + key
        if sk in s:
            opts[key] = bool(s[sk])
    if "output_all_passes" in s:
        opts["output_all_passes"] = bool(s["output_all_passes"])
    if "html_s1_s2_hover_on_by_default" in s:
        opts["html_s1_s2_hover_on_by_default"] = bool(s["html_s1_s2_hover_on_by_default"])
    if "html_inline_interactive_script" in s:
        opts["html_inline_interactive_script"] = bool(s["html_inline_interactive_script"])

    try:
        jobs = int(s.get("cli_batch_jobs", 1))
    except (TypeError, ValueError):
        jobs = 1
    jobs = max(1, jobs)

    out_dir = s.get("cli_output_dir")
    if out_dir is None or not str(out_dir).strip():
        out_dir = "processed_files"
    else:
        out_dir = str(out_dir).strip()

    return {
        "output_options": opts,
        "jobs": jobs,
        "output_dir": out_dir,
        "output_next_to_input": bool(s.get("output_to_input_dir", False)),
        "global_bpm_hint": parse_starting_bpm_from_settings(s),
        "bpm_from_filename": bool(s.get("bpm_from_filename", False)),
        "process_all_channels": bool(s.get("process_all_channels", False)),
        "optimize_long_plots": bool(s.get("optimize_long_plots", False)),
        "algorithm_console_logging": bool(s.get("algorithm_console_logging", True)),
        "general_console_logging": bool(s.get("general_console_logging", False)),
    }
