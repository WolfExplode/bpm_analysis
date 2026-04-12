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
import re
import datetime
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from config import DEFAULT_PARAMS, DEFAULT_OUTPUT_OPTIONS, strip_output_filename_emojis
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any
from time_utils import timestamp_str

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
    ("filtered_wav", "Filtered Audio (.wav)"),
    ("debug", "Debug Report (.md)"),
    ("regression_log", "Regression testing output log (.md)"),
)


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
        self._dnd_available = self._init_drag_drop(root)
        self.create_widgets()
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

        # Channel handling option
        self.process_all_channels = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            param_frame,
            text="Analyze each audio channel separately (stereo \u2192 CH1 & CH2 outputs)",
            variable=self.process_all_channels,
            command=self.save_ui_settings,
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))

        # Output file options (defaults from config only)
        for opt_key, _ in OUTPUT_FILE_OPTIONS:
            setattr(self, "output_" + opt_key, tk.BooleanVar(value=DEFAULT_OUTPUT_OPTIONS.get(opt_key, False)))
        self.optimize_long_plots = tk.BooleanVar(value=False)
        self.output_to_input_dir = tk.BooleanVar(value=False)
        self.output_all_passes = tk.BooleanVar(value=True)
        self.verbose_console_logging = tk.BooleanVar(value=True)
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
            text="Save outputs next to input files (instead of 'processed_files')",
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
            text="Verbose console logging (detailed algorithm messages)",
            variable=self.verbose_console_logging,
            command=self.save_ui_settings,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=(0, 20))

        ttk.Checkbutton(
            debug_frame,
            text="Optimize long HTML plots over 10 min by hiding detailed debug traces to reduce file size",
            variable=self.optimize_long_plots,
            command=self.save_ui_settings,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=(0, 20), pady=(2, 0))

        ttk.Checkbutton(
            debug_frame,
            text="Output all passes (pass 1, pass 2, pass 3)",
            variable=self.output_all_passes,
            command=self.save_ui_settings,
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=(0, 20), pady=(2, 0))

        self.verbose_console_logging.trace("w", lambda *args: self.save_ui_settings())
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
                    final_message = msg.data if msg.data else "Analysis complete!"
                    self.status_var.set(final_message)
                    self.analyze_btn.config(state=tk.NORMAL)
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

    def _extract_bpm_from_filename(self, file_path: str):
        """
        Try to detect a starting BPM from the file name if the user did not enter one.

        Looks for patterns like '120bpm' or '120 bpm' (case-insensitive) in the base file name.
        Returns a float BPM value if found, otherwise None.
        """
        base = os.path.basename(file_path)
        # Case-insensitive search for "<number> [optional space] bpm"
        match = re.search(r"(\d+)\s*bpm", base, flags=re.IGNORECASE)
        if not match:
            return None
        try:
            bpm_val = float(match.group(1))
            logging.info(f"Using BPM {bpm_val} from file name for '{base}'.")
            return bpm_val
        except (TypeError, ValueError):
            return None

    _SETTINGS_VAR_KEYS = (
        ('process_all_channels', 'verbose_console_logging')
        + tuple('output_' + k for k, _ in OUTPUT_FILE_OPTIONS)
        + (
            'optimize_long_plots',
            'output_to_input_dir',
            'output_all_passes',
            'html_s1_s2_hover_on_by_default',
            'html_inline_interactive_script',
        )
    )

    def save_ui_settings(self):
        """Save current UI settings to a JSON file."""
        if self._loading_settings:
            return
        try:
            settings = {k: getattr(self, k).get() for k in self._SETTINGS_VAR_KEYS}
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
            if settings.get('starting_bpm'):
                self.bpm_entry.delete(0, tk.END)
                self.bpm_entry.insert(0, settings['starting_bpm'])
            for k in self._SETTINGS_VAR_KEYS:
                if k in settings:
                    getattr(self, k).set(settings[k])
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
        html_suffixes = ("_pass1.html", "_pass2.html", "_pass3.html", "_fft_profiles.html")
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
        _meta_html_keys = frozenset({"html_s1_s2_hover_on_by_default", "html_inline_interactive_script"})
        required_outputs = {k: v for k, v in output_options.items() if k not in _meta_html_keys}
        if not any(required_outputs.values()):
            messagebox.showerror("Error", "Please select at least one output file type to generate.")
            return

        # Save settings before starting analysis
        self.save_ui_settings()

        self.analyze_btn.config(state=tk.DISABLED)
        self._update_status(f"Starting batch analysis of {len(self.current_files)} files...")

        analysis_thread = threading.Thread(target=self._run_analysis_in_background)
        analysis_thread.daemon = True
        analysis_thread.start()

    def _run_analysis_in_background(self):
        try:
            from pipeline import analyze_wav_file
            from audio_preprocessing import convert_to_wav, split_wav_to_mono_channels
            from fft_profiles import aggregate_fft_profiles, save_aggregate_fft_profiles_html
            import shutil

            # Check for a global BPM value to override all individual settings.
            bpm_override_input = self.bpm_entry.get().strip()
            bpm_override_hint = float(bpm_override_input) if bpm_override_input else None
            # If the user entered a value, use it for the whole batch.
            # If left blank, we will try to infer BPM from each file name instead.
            global_start_bpm_hint = bpm_override_hint

            base_output_dir = os.path.join(os.getcwd(), "processed_files")
            os.makedirs(base_output_dir, exist_ok=True)

            # Initialize regression testing output log if requested.
            regression_log_path = None
            if self.output_regression_log.get():
                regression_log_path = os.path.join(base_output_dir, "regression_testing_output_log.md")
                try:
                    with open(regression_log_path, "w", encoding="utf-8") as log_file:
                        log_file.write("# Regression Testing Output Log\n")
                        log_file.write(f"*Generated on: {timestamp_str()}*\n\n")
                except Exception as e:
                    logging.error(f"Failed to initialize regression testing output log: {e}")
                    regression_log_path = None

            # Read batch-wide options once
            process_all_channels = self.process_all_channels.get()
            optimize_long_plots = self.optimize_long_plots.get()
            verbose_console_logging = self.verbose_console_logging.get()

            # Deduplicate inputs by base filename so we don't process both 'name.wav' and 'name.mp4'.
            deduped_files = {}
            # Prefer WAV when both a compressed file and a WAV with the same base name exist.
            ext_preference = {
                '.wav': 2,
                '.flac': 1,
                '.mp3': 1,
                '.m4a': 1,
                '.ogg': 1,
                '.mp4': 1,
                '.mkv': 1,
                '.mov': 1,
            }
            for path in self.current_files:
                base_name_only = os.path.splitext(os.path.basename(path))[0]
                key = base_name_only.lower()
                ext = os.path.splitext(path)[1].lower()
                score = ext_preference.get(ext, 0)

                if key not in deduped_files:
                    deduped_files[key] = (score, path)
                else:
                    existing_score, _ = deduped_files[key]
                    # Replace only if the new file type is preferred (e.g., WAV over others).
                    if score > existing_score:
                        deduped_files[key] = (score, path)

            input_files = [entry[1] for entry in deduped_files.values()]

            total_files = len(input_files)
            files_processed = 0
            errors = []
            fft_results_for_aggregate = []
            collect_fft_for_aggregate = total_files >= 2

            # --- BATCH PROCESSING LOOP ---
            for i, file_path in enumerate(input_files):
                try:
                    file_start_time = time.time()

                    self.log_queue.put(UIMessage(UIMessageType.STATUS,
                                                 f"({i + 1}/{total_files}) Processing: {os.path.basename(file_path)}"))

                    # Decide where outputs for this file should go
                    if self.output_to_input_dir.get():
                        output_dir = os.path.dirname(file_path) or base_output_dir
                    else:
                        output_dir = base_output_dir

                    os.makedirs(output_dir, exist_ok=True)

                    base_name, ext = os.path.splitext(file_path)
                    ext_lower = ext.lower()

                    if ext_lower != '.wav':
                        source_stem = os.path.basename(base_name)
                        source_dir = os.path.dirname(file_path)
                        same_dir_wav = os.path.join(source_dir, source_stem + ".wav")
                        output_stem = strip_output_filename_emojis(source_stem)
                        output_dir_wav = os.path.join(output_dir, output_stem + ".wav")

                        if os.path.exists(same_dir_wav):
                            # Reuse an existing WAV next to the input file, copying to output_dir if needed.
                            if os.path.abspath(os.path.dirname(same_dir_wav)) == os.path.abspath(output_dir):
                                wav_path = same_dir_wav
                            else:
                                wav_path = output_dir_wav
                                shutil.copy(same_dir_wav, wav_path)
                            logging.info(
                                "Reusing existing WAV '%s' for '%s' instead of converting.",
                                os.path.basename(same_dir_wav),
                                os.path.basename(file_path),
                            )
                        elif os.path.exists(output_dir_wav):
                            # Reuse an existing WAV already in the output directory.
                            wav_path = output_dir_wav
                            logging.info(
                                "Reusing existing WAV '%s' in output directory for '%s' instead of converting.",
                                os.path.basename(output_dir_wav),
                                os.path.basename(file_path),
                            )
                        else:
                            wav_path = output_dir_wav
                            self.log_queue.put(UIMessage(UIMessageType.STATUS,
                                                         f"({i + 1}/{total_files}) Converting {os.path.basename(file_path)}..."))
                            if not convert_to_wav(file_path, wav_path):
                                raise Exception("File conversion failed.")
                    else:
                        # If the output directory is the same as the input directory, reuse the original WAV
                        input_dir = os.path.dirname(file_path)
                        if os.path.abspath(output_dir) == os.path.abspath(input_dir):
                            wav_path = file_path
                        else:
                            orig_base = os.path.basename(file_path)
                            o_stem, o_ext = os.path.splitext(orig_base)
                            out_wav_name = strip_output_filename_emojis(o_stem) + o_ext
                            wav_path = os.path.join(output_dir, out_wav_name)
                            shutil.copy(file_path, wav_path)

                    # Decide which WAV(s) to analyze: either the single mixed file, or
                    # one per channel if requested.
                    wav_files_to_analyze = [wav_path]
                    if process_all_channels:
                        self.log_queue.put(
                            UIMessage(
                                UIMessageType.STATUS,
                                f"({i + 1}/{total_files}) {os.path.basename(file_path)}: Splitting stereo into mono channels...",
                            )
                        )
                        wav_files_to_analyze = split_wav_to_mono_channels(wav_path, output_dir)

                    # Pass the file-specific start_bpm_hint and output options to the analysis function.
                    output_options = self.get_output_options()
                    if regression_log_path:
                        # Share the regression log path with the analysis pipeline so that
                        # per-file validation results can be appended in one central log.
                        output_options["regression_log_path"] = regression_log_path

                    # Determine starting BPM hint for this original input file.
                    if global_start_bpm_hint is not None:
                        file_start_bpm_hint = global_start_bpm_hint
                    else:
                        file_start_bpm_hint = self._extract_bpm_from_filename(file_path)

                    # Ensure plotting logic sees the long-plot optimization preference
                    if optimize_long_plots:
                        self.params["optimize_long_plots"] = True
                    # Ensure verbose logging preference is visible to the analysis pipeline
                    self.params["verbose_console_logging"] = bool(verbose_console_logging)

                    for ch_idx, wav_for_analysis in enumerate(wav_files_to_analyze, start=1):
                        if len(wav_files_to_analyze) > 1:
                            status_suffix = f" (CH{ch_idx})"
                        else:
                            status_suffix = ""

                        status_prefix = f"({i + 1}/{total_files}) {os.path.basename(file_path)}{status_suffix}"

                        def _make_progress_cb(prefix):
                            def _cb(detail: str):
                                self.log_queue.put(
                                    UIMessage(UIMessageType.STATUS, f"{prefix}: {detail}")
                                )

                            return _cb

                        progress_cb = _make_progress_cb(status_prefix)

                        _figure, fft_data = analyze_wav_file(
                            wav_for_analysis,
                            self.params,
                            file_start_bpm_hint,
                            original_file_path=file_path,
                            output_directory=output_dir,
                            output_options=output_options,
                            collect_fft_for_aggregate=collect_fft_for_aggregate,
                            progress_callback=progress_cb,
                        )
                        if fft_data is not None:
                            fft_results_for_aggregate.append(fft_data)
                    files_processed += 1

                    # Log total wall-clock time for this original input file (including conversion, splitting, and analysis).
                    duration = time.time() - file_start_time
                    logging.info(
                        "=== Total processing time for '%s': %.2f seconds (including conversion & analysis). ===",
                        os.path.basename(file_path),
                        duration,
                    )

                except Exception as e:
                    # Inner try-except block to handle errors for a single file
                    error_info = f"Error processing '{os.path.basename(file_path)}':\n{str(e)}"
                    self.log_queue.put(UIMessage(UIMessageType.ERROR, error_info))
                    errors.append(os.path.basename(file_path))

            # --- AGGREGATE FFT (when 2+ files were analyzed with FFT) ---
            if len(fft_results_for_aggregate) >= 2:
                try:
                    freqs, agg_r1, agg_r2, agg_b1, agg_b2 = aggregate_fft_profiles(fft_results_for_aggregate, self.params)
                    aggregate_path = os.path.join(base_output_dir, "fft_profiles_aggregate.html")
                    save_aggregate_fft_profiles_html(
                        freqs, agg_r1, agg_r2, agg_b1, agg_b2, aggregate_path, self.params
                    )
                except Exception as e:
                    logging.warning("Aggregate FFT profiles failed: %s", e)

            # --- POST-LOOP COMPLETION MESSAGE ---
            if not errors:
                completion_message = f"Successfully processed all {total_files} files."
            else:
                completion_message = f"Batch finished. Processed {files_processed}/{total_files}. Errors in: {', '.join(errors)}"

            self.log_queue.put(UIMessage(UIMessageType.ANALYSIS_COMPLETE, completion_message))

        except Exception as e:
            # Outer try-except block for critical errors (e.g., imports)
            error_info = f"A critical error occurred during batch setup:\n{str(e)}"
            self.log_queue.put(UIMessage(UIMessageType.ERROR, error_info))
            self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL))
