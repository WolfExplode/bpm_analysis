# app_settings.py
# Application-level defaults that sit around the analysis engine: which output
# artifacts to write, GUI checkbox defaults, and where ui_settings.json lives.
# Engine parameters (DEFAULT_PARAMS, param, validate_params) stay in config.py.

import os
import sys

# Settings filename. Prefixed so it is identifiable when written to %APPDATA% by the .exe build.
UI_SETTINGS_FILENAME = "BPM_Analyzer_ui_settings.json"


def ui_settings_path():
    """Resolve the ui_settings file location.

    Frozen (.exe via PyInstaller): %APPDATA%\\BPM_Analyzer_ui_settings.json so the
    settings persist next to the user, not in the temp extraction dir.
    From source: cwd/BPM_Analyzer_ui_settings.json (dev workflow, project root).
    """
    if getattr(sys, "frozen", False):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, UI_SETTINGS_FILENAME)
    return os.path.join(os.getcwd(), UI_SETTINGS_FILENAME)


# Single source of truth for pipeline output toggles. GUI and analyze_wav_file use this;
# add new options here only (GUI builds checkboxes and get_output_options from these keys).
DEFAULT_OUTPUT_OPTIONS = {
    "html": True,
    # When False (default), generated HTML opens with S1/S2 beat hover tooltips off; user can enable via toolbar.
    "html_s1_s2_hover_on_by_default": False,
    # When True, embed a small script in the HTML file instead of copying interactive_plot.js (no audio/spectrogram/label JS).
    "html_inline_interactive_script": False,
    "png": True,
    "csv": False,
    "summary": False,
    "debug": False,
    "filtered_wav": False,
    # When True, converted/copied/split working WAVs are written under the output folder; when False, a temp dir is used.
    "working_wav_in_output": False,
    "spectrogram": False,
    "fft_profiles": False,
}

# Single source of truth for GUI checkbox/spinbox defaults. gui.py reads these;
# ui_settings.json overrides on subsequent launches.
DEFAULT_UI_SETTINGS = {
    "bpm_from_filename": True,
    "rename_input_with_bpm": False,
    "channel_mode": "mixed",          # matches CHANNEL_MODE_MIXED in audio_io
    "output_to_input_dir": True,
    "output_all_passes": False,
    "algorithm_console_logging": False,
    "general_console_logging": False,
    "use_springer_algorithm": False,
    "auto_switch_algorithm": False,
    "optimize_long_plots": False,
    "cli_batch_jobs": 1,
    "auto_close_when_done": False,
    "analysis_start_sec": 0.0,
}
