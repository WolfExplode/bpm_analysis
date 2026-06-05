# gui.py

import os
import queue
import threading
import tkinter as tk
import json
import subprocess
import platform
import logging
import time
import datetime
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from config import DEFAULT_PARAMS, DEFAULT_OUTPUT_OPTIONS
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Optional, Tuple
from time_utils import timestamp_str
from ui_settings_loader import migrate_ui_settings_keys
from audio_preprocessing import (
    CHANNEL_MODE_ALL,
    CHANNEL_MODE_LEFT,
    CHANNEL_MODE_MIXED,
    CHANNEL_MODE_RIGHT,
    normalize_channel_mode,
)
from console_logging import configure_analysis_console_logging
from bpm_input_rename import rename_analysis_outputs_after_input_bpm_rename, try_rename_input_with_bpm_annotation

class UIMessageType(Enum):
    STATUS = auto()
    ANALYSIS_COMPLETE = auto()
    ERROR = auto()

@dataclass
class UIMessage:
    type: UIMessageType
    data: Any = None

# Order and labels for output file checkboxes (keys must match DEFAULT_OUTPUT_OPTIONS in config).
# First 8 in 2-column grid, 9th full width. Default values come from config only.
OUTPUT_FILE_OPTIONS = (
    ("html", "Heart Rate Graph (.html)"),
    ("fft_profiles", "S1/S2 FFT Profiles (.html)"),
    ("png", "Plot PNG (.png)"),
    ("csv", "bpm/time Data (.csv)"),
    ("spectrogram", "Spectrogram (.png)"), 
    ("summary", "Summary Report (.md)"),
    ("filtered_wav", "Bandpass + out-of-band debug WAVs (.wav)"),
    ("working_wav_in_output", "Converted / working WAV in output folder"),
    ("debug", "Debug Report (.md)"),
    ("regression_log", "Regression testing output log (.md)"),
)

CHANNEL_MODE_OPTIONS = (
    (CHANNEL_MODE_MIXED, "Mixed (mono downmix)"),
    (CHANNEL_MODE_ALL, "Each channel separately (stereo \u2192 CH1 & CH2)"),
    (CHANNEL_MODE_LEFT, "Left channel only"),
    (CHANNEL_MODE_RIGHT, "Right channel only"),
)
_CHANNEL_MODE_LABEL_BY_VALUE = {v: label for v, label in CHANNEL_MODE_OPTIONS}
_CHANNEL_MODE_VALUE_BY_LABEL = {label: v for v, label in CHANNEL_MODE_OPTIONS}

class BPMApp:
    # Minimum window size when auto-sized or resized by user
    MIN_WIDTH = 420
    MIN_HEIGHT = 380

    def __init__(self, root, initial_files=None):
        self.root = root
        self.root.title("Heartbeat BPM Analyzer")
        self.root.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self.style = ttkb.Style(theme='minty')
        self.current_files = []
        self.params = DEFAULT_PARAMS.copy()
        self.log_queue = queue.Queue()
        self.settings_file = os.path.join(os.getcwd(), "ui_settings.json")
        self._loading_settings = True  # Prevent saving during initialization
        self._analysis_running = False
        self._general_console_log_filters: list[tuple[logging.Handler, logging.Filter]] = []
        self._dnd_available = self._init_drag_drop(root)
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close_request)
        self.load_ui_settings()
        self._loading_settings = False  # Re-enable saving after load
        self.root.after(100, self.process_log_queue)
        self.root.after(150, self._fit_window_to_content)
        cli_files = []
        if initial_files:
            for raw in initial_files:
                p = os.path.normpath(raw)
                if os.path.isfile(p):
                    cli_files.append(p)
        if cli_files:
            self._apply_selected_files(cli_files)
        else:
            self._find_initial_audio_file()

    def create_widgets(self):
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame = self.main_frame

        # File selection (drag-and-drop onto this frame when tkinterdnd2 is available)
        file_frame = ttk.LabelFrame(main_frame, text="Audio File(s)", padding=10)
        file_frame.grid(row=0, column=0, sticky="ew", pady=5)
        hint = (
            "No files selected — drag audio files here or click Browse"
            if self._dnd_available
            else "No files selected"
        )
        self.file_label = ttk.Label(file_frame, text=hint, wraplength=450)
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        browse_btn = ttk.Button(file_frame, text="Browse", command=self.select_file, bootstyle=INFO)
        browse_btn.pack(side=tk.RIGHT, padx=5)
        if self._dnd_available:
            self._register_file_drop_target(file_frame)

        # Parameters
        param_frame = ttk.LabelFrame(main_frame, text="Analysis Parameters", padding=10)
        param_frame.grid(row=1, column=0, sticky="ew", pady=5)
        ttk.Label(param_frame, text="Starting BPM (optional):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.bpm_entry = ttk.Entry(param_frame)
        self.bpm_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        # Save settings when BPM entry changes
        self.bpm_entry.bind('<KeyRelease>', lambda e: self.save_ui_settings())
        self.bpm_entry.bind('<FocusOut>', lambda e: self.save_ui_settings())

        self.bpm_from_filename = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            param_frame,
            text="Read starting BPM from file name (e.g. 120,60-150bpm \u2192 120; 90to132bpm \u2192 90; 150bpm)",
            variable=self.bpm_from_filename,
            command=self.save_ui_settings,
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))
        self.bpm_from_filename.trace("w", lambda *args: self.save_ui_settings())

        self.rename_input_with_bpm = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            param_frame,
            text="Rename input file with BPM tag: [start,min-maxbpm] (uses analysis; not for multi-channel)",
            variable=self.rename_input_with_bpm,
            command=self.save_ui_settings,
        ).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))
        self.rename_input_with_bpm.trace("w", lambda *args: self.save_ui_settings())

        # Channel handling option
        self.channel_mode = tk.StringVar(value=_CHANNEL_MODE_LABEL_BY_VALUE[CHANNEL_MODE_MIXED])
        ttk.Label(param_frame, text="Audio channels:").grid(row=3, column=0, sticky=tk.W, pady=(4, 0))
        channel_combo = ttk.Combobox(
            param_frame,
            textvariable=self.channel_mode,
            values=[label for _, label in CHANNEL_MODE_OPTIONS],
            state="readonly",
            width=42,
        )
        channel_combo.grid(row=3, column=1, sticky=tk.W, pady=(4, 0), padx=(6, 0))
        channel_combo.bind("<<ComboboxSelected>>", lambda e: self.save_ui_settings())

        # Output file options (defaults from config only)
        for opt_key, _ in OUTPUT_FILE_OPTIONS:
            setattr(self, "output_" + opt_key, tk.BooleanVar(value=DEFAULT_OUTPUT_OPTIONS.get(opt_key, False)))
        self.optimize_long_plots = tk.BooleanVar(value=False)
        self.output_to_input_dir = tk.BooleanVar(value=False)
        self.output_all_passes = tk.BooleanVar(value=True)
        self.algorithm_console_logging = tk.BooleanVar(value=True)
        self.general_console_logging = tk.BooleanVar(value=False)
        self.html_s1_s2_hover_on_by_default = tk.BooleanVar(
            value=DEFAULT_OUTPUT_OPTIONS.get("html_s1_s2_hover_on_by_default", False)
        )
        self.html_inline_interactive_script = tk.BooleanVar(
            value=DEFAULT_OUTPUT_OPTIONS.get("html_inline_interactive_script", False)
        )

        # Output files section — left column then right column (top to bottom each)
        output_frame = ttk.LabelFrame(main_frame, text="Output Files", padding="10")
        output_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

        n_opts = len(OUTPUT_FILE_OPTIONS)
        half = (n_opts + 1) // 2

        def on_output_change(*args):
            self.save_ui_settings()

        for i in range(half):
            for col, idx in enumerate([i, i + half]):
                if idx >= n_opts:
                    break
                opt_key, label = OUTPUT_FILE_OPTIONS[idx]
                var = getattr(self, "output_" + opt_key)
                ttk.Checkbutton(
                    output_frame, text=label, variable=var, command=self.save_ui_settings
                ).grid(row=i, column=col, sticky="w", padx=(0, 20))
                var.trace("w", on_output_change)

        # Output location
        ttk.Checkbutton(
            output_frame,
            text="Save outputs to same directory as input files",
            variable=self.output_to_input_dir,
            command=self.save_ui_settings,
        ).grid(row=half, column=0, columnspan=2, sticky="w", padx=(0, 20), pady=(8, 0))

        ttk.Checkbutton(
            output_frame,
            text="HTML: start with S1/S2 hover tooltips on (toolbar \u201cS1/S2 Hover info\u201d)",
            variable=self.html_s1_s2_hover_on_by_default,
            command=self.save_ui_settings,
        ).grid(row=half + 1, column=0, columnspan=2, sticky="w", padx=(0, 20), pady=(4, 0))

        self.html_s1_s2_hover_on_by_default.trace("w", lambda *args: self.save_ui_settings())

        ttk.Checkbutton(
            output_frame,
            text="HTML: embed minimal script only (no interactive_plot.js; chart + hover + legend filter only)",
            variable=self.html_inline_interactive_script,
            command=self.save_ui_settings,
        ).grid(row=half + 2, column=0, columnspan=2, sticky="w", padx=(0, 20), pady=(2, 0))

        self.html_inline_interactive_script.trace("w", lambda *args: self.save_ui_settings())

        # Select All/None buttons
        btn_frame_output = ttk.Frame(output_frame)
        btn_frame_output.grid(row=half + 3, column=0, columnspan=2, pady=(10, 0))
        ttk.Button(btn_frame_output, text="Select All", command=self.select_all_outputs,
                  bootstyle=SECONDARY).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(btn_frame_output, text="Select None", command=self.select_none_outputs,
                  bootstyle=SECONDARY).grid(row=0, column=1)

        self.output_to_input_dir.trace("w", lambda *args: self.save_ui_settings())

        # Debugging options
        debug_frame = ttk.LabelFrame(main_frame, text="Debugging Options", padding="10")
        debug_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)

        ttk.Checkbutton(
            debug_frame,
            text="Algorithm console logging (debug for algorithm details)",
            variable=self.algorithm_console_logging,
            command=self.save_ui_settings,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=(0, 20))

        ttk.Checkbutton(
            debug_frame,
            text="General console logging (debug for timing and other debug strings; silences noisy libraries)",
            variable=self.general_console_logging,
            command=self.save_ui_settings,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=(0, 20), pady=(2, 0))

        ttk.Checkbutton(
            debug_frame,
            text="Optimize long HTML plots over 10 min by hiding detailed debug traces to reduce file size",
            variable=self.optimize_long_plots,
            command=self.save_ui_settings,
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=(0, 20), pady=(2, 0))

        ttk.Checkbutton(
            debug_frame,
            text="Output all passes (pass 1, pass 2, pass 3)",
            variable=self.output_all_passes,
            command=self.save_ui_settings,
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=(0, 20), pady=(2, 0))

        self.cli_batch_jobs = tk.IntVar(value=1)
        jobs_row = ttk.Frame(debug_frame)
        jobs_row.grid(row=4, column=0, columnspan=2, sticky="w", padx=(0, 20), pady=(2, 0))
        ttk.Label(jobs_row, text="Parallel batch jobs (1 = sequential):").pack(side=tk.LEFT)
        _jobs_spin = ttk.Spinbox(
            jobs_row,
            from_=1,
            to=32,
            width=5,
            textvariable=self.cli_batch_jobs,
            command=self.save_ui_settings,
        )
        _jobs_spin.pack(side=tk.LEFT, padx=(8, 0))
        _jobs_spin.bind("<FocusOut>", lambda e: self.save_ui_settings())
        self.cli_batch_jobs.trace("w", lambda *args: self.save_ui_settings())

        self.auto_close_when_done = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            debug_frame,
            text="Auto-close when batch finishes successfully (no per-file errors)",
            variable=self.auto_close_when_done,
            command=self.save_ui_settings,
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=(0, 20), pady=(2, 0))
        self.auto_close_when_done.trace("w", lambda *args: self.save_ui_settings())

        self.algorithm_console_logging.trace("w", lambda *args: self.save_ui_settings())
        self.general_console_logging.trace("w", lambda *args: self.save_ui_settings())
        self.optimize_long_plots.trace("w", lambda *args: self.save_ui_settings())
        self.output_all_passes.trace("w", lambda *args: self.save_ui_settings())

        # Action Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=4, column=0, sticky="ew", pady=20)
        self.analyze_btn = ttk.Button(btn_frame, text="Analyze", command=self.start_analysis_thread, bootstyle=SUCCESS, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.RIGHT, padx=5)
        self.open_html_btn = ttk.Button(btn_frame, text="Open Last HTML Report", command=self.open_last_html, bootstyle=INFO)
        self.open_html_btn.pack(side=tk.RIGHT, padx=5)

        # Status Bar
        self.status_var = tk.StringVar(value="Select one or more audio files to begin.")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.grid(row=5, column=0, sticky="ew", pady=(10, 0))

        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        param_frame.columnconfigure(1, weight=1)
        output_frame.columnconfigure(0, weight=1)
        output_frame.columnconfigure(1, weight=1)
        debug_frame.columnconfigure(0, weight=1)

    def _fit_window_to_content(self):
        """Resize the window to fit the current content (all visible elements)."""
        self.root.update_idletasks()
        req_w = self.main_frame.winfo_reqwidth()
        req_h = self.main_frame.winfo_reqheight()
        # Add margin for window decorations and padding
        margin_w = 50
        margin_h = 50
        w = max(self.MIN_WIDTH, req_w + margin_w)
        h = max(self.MIN_HEIGHT, req_h + margin_h)
        self.root.geometry(f"{w}x{h}")

    def process_log_queue(self):
        try:
            while not self.log_queue.empty():
                msg: UIMessage = self.log_queue.get(0)

                if msg.type == UIMessageType.STATUS:
                    self.status_var.set(msg.data)
                elif msg.type == UIMessageType.ANALYSIS_COMPLETE:
                    batch_ok = True
                    if isinstance(msg.data, tuple) and len(msg.data) == 2:
                        final_message, batch_ok = msg.data[0], bool(msg.data[1])
                    else:
                        final_message = msg.data if msg.data else "Analysis complete!"
                    self.status_var.set(final_message)
                    self.analyze_btn.config(state=tk.NORMAL)
                    if self.auto_close_when_done.get() and batch_ok:
                        self.root.after(150, self.root.destroy)
                elif msg.type == UIMessageType.ERROR:
                     self.status_var.set("An error occurred. Check logs and messagebox.")
                     messagebox.showerror("Analysis Error", msg.data)
        finally:
            self.root.after(100, self.process_log_queue)

    def _init_drag_drop(self, root):
        """Load tkdnd on the root window so child widgets can accept file drops."""
        try:
            from tkinterdnd2 import TkinterDnD
            TkinterDnD._require(root)
            return True
        except ImportError:
            logging.warning("tkinterdnd2 not installed; drag-and-drop disabled.")
            return False
        except RuntimeError as e:
            logging.warning("Drag-and-drop unavailable: %s", e)
            return False

    def _register_file_drop_target(self, widget):
        from tkinterdnd2 import DND_FILES

        widget.drop_target_register(DND_FILES)
        widget.dnd_bind("<<Drop>>", self._on_files_dropped)

    def _on_files_dropped(self, event):
        from tkinterdnd2 import COPY as DND_COPY

        try:
            paths = self.root.tk.splitlist(event.data)
        except tk.TclError:
            return DND_COPY
        files = []
        for p in paths:
            norm = os.path.normpath(p)
            if os.path.isfile(norm):
                files.append(norm)
        if files:
            self._apply_selected_files(files)
        return DND_COPY

    def _apply_selected_files(self, paths):
        """Apply a non-empty list of file paths (same behavior as Browse after selection)."""
        if not paths:
            return
        self.current_files = list(paths)
        self.file_label.config(text=f"{len(self.current_files)} files selected")
        self.analyze_btn.config(state=tk.NORMAL)
        self.save_ui_settings()
        if len(self.current_files) == 1:
            self._update_status("Ready to analyze the selected file.")
        else:
            self.bpm_entry.delete(0, tk.END)
            self._update_status(f"Ready to analyze {len(self.current_files)} files.")

    def select_file(self):
        filetypes = [('Audio files', '*.wav *.mp3 *.m4a *.flac *.ogg *.mp4 *.mkv *.mov'), ('All files', '*.*')]
        filenames = filedialog.askopenfilename(
            title="Select one or more audio files",
            filetypes=filetypes,
            multiple=True
        )
        if filenames:
            self._apply_selected_files(list(filenames))

    def _find_initial_audio_file(self):
        """
        Automatically finds all supported audio files in the current directory
        and loads them into the application. If only one file is found, it
        attempts to load its corresponding analysis settings.
        Only runs if no files were already loaded from saved settings.
        """
        # Skip auto-detection if files were already loaded from saved settings
        if self.current_files:
            return
            
        supported = ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.mp4', '.mkv', '.mov')
        found_files = []
        try:
            # Find all supported files in the script's directory
            for filename in os.listdir(os.getcwd()):
                if filename.lower().endswith(supported):
                    full_path = os.path.join(os.getcwd(), filename)
                    found_files.append(full_path)

            if found_files:
                self.current_files = found_files

                # Update the GUI to show what was loaded
                label_text = f"{len(self.current_files)} files loaded"
                self.file_label.config(text=label_text)
                self.analyze_btn.config(state=tk.NORMAL)
                
                # Save the auto-detected files to settings
                self.save_ui_settings()

                self._update_status(f"Auto-loaded {len(self.current_files)} files from the current directory.")

        except Exception as e:
            # Fails silently if it can't read the directory
            pass

    def _update_status(self, message):
        """Safely update the status bar from any thread."""
        self.root.after(0, lambda: self.status_var.set(message))

    def _schedule_rename_warning(self, reason: str, basename: str) -> None:
        """Show a non-blocking dialog from the worker thread (tk main thread)."""

        def show():
            messagebox.showwarning(
                "Input file not renamed",
                f"{reason}\n\nFile: {basename}",
            )

        self.root.after(0, show)

    def _apply_batch_input_bpm_renames(self, ops: List[Tuple[str, Any]]) -> None:
        """Apply BPM filename renames on the main thread after a batch (sequential or parallel)."""
        if not self.rename_input_with_bpm.get():
            return
        for file_path, result in ops:
            if not getattr(result, "success", False):
                continue
            info = getattr(result, "bpm_rename_info", None)
            if info is None:
                continue
            if getattr(result, "analyze_had_multiple_channels", False):
                self._schedule_rename_warning(
                    "Multi-channel analysis: renaming the input file is skipped.",
                    os.path.basename(file_path),
                )
                continue
            new_abs = self._try_rename_input_with_bpm_annotation(file_path, info)
            if new_abs:
                rename_analysis_outputs_after_input_bpm_rename(
                    file_path, new_abs, getattr(result, "output_dir", "") or ""
                )
                self._sync_file_list_after_rename(file_path, new_abs)

    def _sync_file_list_after_rename(self, old_path: str, new_path: str) -> None:
        old_abs = os.path.abspath(old_path)
        for i, p in enumerate(self.current_files):
            if os.path.abspath(p) == old_abs:
                self.current_files[i] = new_path
                break
        self.save_ui_settings()

    def _try_rename_input_with_bpm_annotation(self, file_path: str, info: dict) -> Optional[str]:
        """Rename file_path to append/replace trailing start,min-maxbpm tag. Returns new absolute path, or None."""
        return try_rename_input_with_bpm_annotation(
            file_path,
            info,
            warn=lambda reason, base: self._schedule_rename_warning(reason, base),
        )

    def _extract_bpm_from_filename(self, file_path: str):
        """
        Try to detect a starting BPM from the file name if the user did not enter one.

        Uses the last (rightmost) match in the base name for each pattern type, in order:
        '120,60-150bpm' → 120; '90to132bpm' → 90; '150bpm' → 150 (case-insensitive).
        """
        from batch_runner import extract_start_bpm_from_filename

        return extract_start_bpm_from_filename(file_path)

    _SETTINGS_VAR_KEYS = (
        ('algorithm_console_logging', 'general_console_logging', 'bpm_from_filename', 'rename_input_with_bpm')
        + tuple('output_' + k for k, _ in OUTPUT_FILE_OPTIONS)
        + (
            'optimize_long_plots',
            'output_to_input_dir',
            'output_all_passes',
            'auto_close_when_done',
            'html_s1_s2_hover_on_by_default',
            'html_inline_interactive_script',
            'cli_batch_jobs',
        )
    )

    def _apply_general_console_logging(self, enabled: bool) -> None:
        """
        Control whether DEBUG-level logs are emitted to the console.
        Algorithm-detail verbosity is controlled separately by params["algorithm_console_logging"].

        With root at DEBUG, some dependencies (Numba JIT, pydub/ffmpeg, asyncio, Kaleido's
        choreographer/CDP layer) would flood the console; those loggers stay at WARNING while
        this mode is on (see console_logging.configure_analysis_console_logging).
        """
        configure_analysis_console_logging(
            general_debug=bool(enabled),
            tracked_filters=self._general_console_log_filters,
        )

    def save_ui_settings(self):
        """Save current UI settings to a JSON file."""
        if self._loading_settings:
            return
        try:
            settings = {k: getattr(self, k).get() for k in self._SETTINGS_VAR_KEYS}
            label = self.channel_mode.get()
            settings["channel_mode"] = _CHANNEL_MODE_VALUE_BY_LABEL.get(label, CHANNEL_MODE_MIXED)
            settings['starting_bpm'] = self.bpm_entry.get().strip()
            settings['last_files'] = self.current_files if self.current_files else []
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            logging.warning(f"Could not save UI settings: {e}")

    def load_ui_settings(self):
        """Load UI settings from a JSON file if it exists."""
        if not os.path.exists(self.settings_file):
            return
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            migrate_ui_settings_keys(settings)
            if settings.get('starting_bpm'):
                self.bpm_entry.delete(0, tk.END)
                self.bpm_entry.insert(0, settings['starting_bpm'])
            if "channel_mode" in settings:
                mode = normalize_channel_mode(settings["channel_mode"])
                self.channel_mode.set(_CHANNEL_MODE_LABEL_BY_VALUE.get(mode, _CHANNEL_MODE_LABEL_BY_VALUE[CHANNEL_MODE_MIXED]))
            for k in self._SETTINGS_VAR_KEYS:
                if k in settings:
                    getattr(self, k).set(settings[k])
            try:
                j = int(self.cli_batch_jobs.get())
                self.cli_batch_jobs.set(max(1, min(j, 32)))
            except (tk.TclError, ValueError, TypeError):
                self.cli_batch_jobs.set(1)
            if settings.get('last_files'):
                existing = [p for p in settings['last_files'] if os.path.exists(p)]
                if existing:
                    self.current_files = existing
                    self.file_label.config(text=f"{len(self.current_files)} files loaded from previous session")
                    self.analyze_btn.config(state=tk.NORMAL)
                    self._update_status(f"Loaded {len(self.current_files)} files from previous session.")
        except Exception as e:
            logging.warning(f"Could not load UI settings: {e}")
        finally:
            self._loading_settings = False

    def open_last_html(self):
        """Find and open the most recently generated HTML report file."""
        # Default output folder plus each input file's folder (when "save next to input" was used).
        dirs_to_scan = set()
        processed_dir = os.path.join(os.getcwd(), "processed_files")
        if os.path.isdir(processed_dir):
            dirs_to_scan.add(os.path.abspath(processed_dir))
        for fp in self.current_files:
            parent = os.path.dirname(os.path.abspath(fp))
            if os.path.isdir(parent):
                dirs_to_scan.add(parent)

        if not dirs_to_scan:
            messagebox.showwarning(
                "No Reports",
                "No output folders found (processed_files or folders of selected inputs). Run an analysis first.",
            )
            return

        # BPM plot or FFT profile HTML; pick newest mtime across all scanned dirs.
        html_suffixes = (
            "_pass1.html",
            "_pass2.html",
            "_pass3.html",
            "_bpm_plot.html",
            "_fft_profiles.html",
        )
        html_files = []
        read_errors = []
        for output_dir in dirs_to_scan:
            try:
                for filename in os.listdir(output_dir):
                    if filename.endswith(".html") and any(
                        filename.endswith(suffix) for suffix in html_suffixes
                    ):
                        file_path = os.path.join(output_dir, filename)
                        mtime = os.path.getmtime(file_path)
                        html_files.append((mtime, file_path, filename))
            except Exception as e:
                read_errors.append((output_dir, e))

        if read_errors and not html_files:
            messagebox.showerror(
                "Error",
                "Could not read output folder(s): "
                + "; ".join(f"{d}: {err}" for d, err in read_errors),
            )
            return

        if not html_files:
            messagebox.showwarning(
                "No Reports",
                "No HTML reports found (BPM plot or FFT profiles) in processed_files or next to your selected inputs. Run an analysis first.",
            )
            return
        
        # Sort by modification time (most recent first)
        html_files.sort(reverse=True)
        most_recent_file = html_files[0][1]
        
        # Open the file with the system's default application and close the application
        try:
            if platform.system() == 'Windows':
                os.startfile(most_recent_file)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', most_recent_file])
            else:  # Linux and others
                subprocess.run(['xdg-open', most_recent_file])
            self._update_status(f"Opened: {os.path.basename(most_recent_file)}")
            # Close the application after opening the HTML file
            self.root.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Could not open HTML file: {e}")

    def select_all_outputs(self):
        """Select all output file options."""
        for opt_key, _ in OUTPUT_FILE_OPTIONS:
            getattr(self, "output_" + opt_key).set(True)

    def select_none_outputs(self):
        """Deselect all output file options."""
        for opt_key, _ in OUTPUT_FILE_OPTIONS:
            getattr(self, "output_" + opt_key).set(False)

    def get_output_options(self):
        """Get the current output file selection as a dictionary (keys match config.DEFAULT_OUTPUT_OPTIONS)."""
        opts = {opt_key: getattr(self, "output_" + opt_key).get() for opt_key, _ in OUTPUT_FILE_OPTIONS}
        opts["output_all_passes"] = self.output_all_passes.get()
        opts["html_s1_s2_hover_on_by_default"] = self.html_s1_s2_hover_on_by_default.get()
        opts["html_inline_interactive_script"] = self.html_inline_interactive_script.get()
        return opts

    def start_analysis_thread(self):
        """Starts the analysis in a new thread."""
        if not self.current_files:
            messagebox.showerror("Error", "No files selected")
            return

        # Check if at least one output option is selected
        output_options = self.get_output_options()
        _meta_output_keys = frozenset({
            "html_s1_s2_hover_on_by_default",
            "html_inline_interactive_script",
            "working_wav_in_output",
        })
        required_outputs = {k: v for k, v in output_options.items() if k not in _meta_output_keys}
        if not any(required_outputs.values()):
            messagebox.showerror("Error", "Please select at least one output file type to generate.")
            return

        # Save settings before starting analysis
        self.save_ui_settings()

        self.analyze_btn.config(state=tk.DISABLED)
        try:
            j = int(self.cli_batch_jobs.get())
        except (tk.TclError, ValueError, TypeError):
            j = 1
        j = max(1, min(j, 32))
        if j > 1:
            self._update_status(
                f"Starting batch: {len(self.current_files)} file(s), {j} parallel workers..."
            )
        else:
            self._update_status(f"Starting batch analysis of {len(self.current_files)} files...")

        self._analysis_running = True
        analysis_thread = threading.Thread(target=self._run_analysis_in_background)
        analysis_thread.daemon = True
        analysis_thread.start()

    def _on_close_request(self):
        """Quit immediately during analysis (terminate workers, exit process); otherwise close normally."""
        if self._analysis_running:
            try:
                self.save_ui_settings()
            except Exception:
                pass
            try:
                import multiprocessing

                for p in multiprocessing.active_children():
                    try:
                        p.terminate()
                    except Exception:
                        pass
            except Exception:
                pass
            os._exit(0)
        self.root.destroy()

    def _run_analysis_in_background(self):
        try:
            from batch_runner import run_batch_parallel

            bpm_override_input = self.bpm_entry.get().strip()
            bpm_override_hint = float(bpm_override_input) if bpm_override_input else None
            global_start_bpm_hint = bpm_override_hint
            bpm_from_filename = self.bpm_from_filename.get()
            rename_after = self.rename_input_with_bpm.get()

            base_output_dir = os.path.join(os.getcwd(), "processed_files")
            os.makedirs(base_output_dir, exist_ok=True)

            label = self.channel_mode.get()
            channel_mode = _CHANNEL_MODE_VALUE_BY_LABEL.get(label, CHANNEL_MODE_MIXED)
            optimize_long_plots = self.optimize_long_plots.get()
            algorithm_console_logging = self.algorithm_console_logging.get()
            general_console_logging = self.general_console_logging.get()

            self._apply_general_console_logging(bool(general_console_logging))

            try:
                max_workers = int(self.cli_batch_jobs.get())
            except (tk.TclError, ValueError, TypeError):
                max_workers = 1
            max_workers = max(1, min(max_workers, 32))

            regression_log_path = None
            if self.output_regression_log.get() and max_workers <= 1:
                regression_log_path = os.path.join(base_output_dir, "regression_testing_output_log.md")
                try:
                    with open(regression_log_path, "w", encoding="utf-8") as log_file:
                        log_file.write("# Regression Testing Output Log\n")
                        log_file.write(f"*Generated on: {timestamp_str()}*\n\n")
                except Exception as e:
                    logging.error("Failed to initialize regression testing output log: %s", e)
                    regression_log_path = None

            self.params["optimize_long_plots"] = bool(optimize_long_plots)
            self.params["algorithm_console_logging"] = bool(algorithm_console_logging)
            self.params["general_console_logging"] = bool(general_console_logging)

            output_options = self.get_output_options()
            if regression_log_path and max_workers <= 1:
                output_options["regression_log_path"] = regression_log_path
            elif self.output_regression_log.get() and max_workers > 1:
                self.log_queue.put(
                    UIMessage(
                        UIMessageType.STATUS,
                        "Regression log disabled for parallel batch (use Parallel jobs = 1 to enable).",
                    )
                )

            if max_workers > 1:
                self.log_queue.put(
                    UIMessage(
                        UIMessageType.STATUS,
                        f"Running {len(self.current_files)} input path(s) with {max_workers} workers "
                        "(per-file live status only when Parallel jobs = 1).",
                    )
                )

            def _seq_conv(job_i: int, total: int, _file_path: str, msg: str) -> None:
                self.log_queue.put(
                    UIMessage(UIMessageType.STATUS, f"({job_i + 1}/{total}) {msg}")
                )

            def _seq_prog(job_i: int, total: int, file_path: str, detail: str) -> None:
                bn = os.path.basename(file_path)
                self.log_queue.put(
                    UIMessage(UIMessageType.STATUS, f"({job_i + 1}/{total}) {bn}: {detail}")
                )

            summary = run_batch_parallel(
                self.current_files,
                self.params,
                output_options,
                base_output_dir=base_output_dir,
                output_next_to_input=bool(self.output_to_input_dir.get()),
                max_workers=max_workers,
                global_bpm_hint=global_start_bpm_hint,
                bpm_from_filename=bpm_from_filename,
                channel_mode=channel_mode,
                sequential_progress_callback=_seq_prog if max_workers == 1 else None,
                sequential_conversion_callback=_seq_conv if max_workers == 1 else None,
            )

            deduped = summary.deduped_input_paths
            total_files = len(deduped)
            errors = list(summary.error_basenames)
            files_processed = sum(1 for r in summary.results if r.success)

            for _fp, result in zip(deduped, summary.results):
                if result.success:
                    continue
                error_info = f"Error processing '{result.basename}':\n{result.error or 'Unknown error'}"
                self.log_queue.put(UIMessage(UIMessageType.ERROR, error_info))

            if rename_after:
                rename_ops = list(zip(deduped, summary.results))
                self.root.after(0, lambda ops=rename_ops: self._apply_batch_input_bpm_renames(ops))

            if not errors:
                completion_message = f"Successfully processed all {total_files} file(s)."
            else:
                completion_message = (
                    f"Batch finished. Completed {files_processed}/{total_files}. "
                    f"Errors in: {', '.join(errors)}"
                )

            self.log_queue.put(
                UIMessage(UIMessageType.ANALYSIS_COMPLETE, (completion_message, not errors))
            )

        except Exception as e:
            error_info = f"A critical error occurred during batch setup:\n{str(e)}"
            self.log_queue.put(UIMessage(UIMessageType.ERROR, error_info))
            self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL))
        finally:
            self._analysis_running = False
