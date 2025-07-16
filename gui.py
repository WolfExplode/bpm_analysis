# gui.py

import os
import queue
import threading
import tkinter as tk
import json
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from config import DEFAULT_PARAMS
from dataclasses import dataclass
from enum import Enum, auto

class UIMessageType(Enum):
    STATUS = auto()
    ANALYSIS_COMPLETE = auto()
    ERROR = auto()

@dataclass
class UIMessage:
    type: UIMessageType
    data: any = None

class BPMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Heartbeat BPM Analyzer (Batch Mode)")
        self.root.geometry("550x350")
        self.style = ttkb.Style(theme='minty')
        self.current_files = []
        self.params = DEFAULT_PARAMS.copy() # Assumes DEFAULT_PARAMS is imported
        self.log_queue = queue.Queue()
        self.create_widgets()
        self.root.after(100, self.process_log_queue)
        self._find_initial_audio_file()
        
        # Check if output folder exists and enable button if it does
        output_dir = os.path.join(os.getcwd(), "processed_files")
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            self.open_folder_btn.config(state=tk.NORMAL)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="Audio File(s)", padding=10)
        file_frame.pack(fill=tk.X, pady=5)
        self.file_label = ttk.Label(file_frame, text="No files selected", wraplength=450)
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        browse_btn = ttk.Button(file_frame, text="Browse", command=self.select_file, bootstyle=INFO)
        browse_btn.pack(side=tk.RIGHT, padx=5)

        # Parameters
        param_frame = ttk.LabelFrame(main_frame, text="Analysis Parameters", padding=10)
        param_frame.pack(fill=tk.X, pady=5)
        ttk.Label(param_frame, text="Starting BPM (optional):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.bpm_entry = ttk.Entry(param_frame)
        self.bpm_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)

        # Action Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=20)
        self.analyze_btn = ttk.Button(btn_frame, text="Analyze", command=self.start_analysis_thread, bootstyle=SUCCESS, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.RIGHT, padx=5)
        self.open_folder_btn = ttk.Button(btn_frame, text="Open Output Folder", command=self.open_output_folder, bootstyle=INFO, state=tk.DISABLED)
        self.open_folder_btn.pack(side=tk.RIGHT, padx=5)

        # Status Bar
        self.status_var = tk.StringVar(value="Select one or more audio files to begin.")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

        param_frame.columnconfigure(1, weight=1)

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
                    self.open_folder_btn.config(state=tk.NORMAL)
                elif msg.type == UIMessageType.ERROR:
                     self.status_var.set("An error occurred. Check logs and messagebox.")
                     messagebox.showerror("Analysis Error", msg.data)
        finally:
            self.root.after(100, self.process_log_queue)

    def select_file(self):
        filetypes = [('Audio files', '*.wav *.mp3 *.m4a *.flac *.ogg *.mp4 *.mkv *.mov'), ('All files', '*.*')]
        filenames = filedialog.askopenfilename(
            title="Select one or more audio files",
            filetypes=filetypes,
            multiple=True
        )
        if filenames:
            self.current_files = list(filenames)
            label_text = f"{len(self.current_files)} files selected"
            self.file_label.config(text=label_text)
            self.analyze_btn.config(state=tk.NORMAL)

            # If only one file is chosen, try to load its settings.
            if len(self.current_files) == 1:
                self._load_settings_for_file(self.current_files[0])
            else:
                # If multiple files are selected, clear the entry to avoid confusion.
                # The user must enter a value to be used for the whole batch.
                self.bpm_entry.delete(0, tk.END)
                self._update_status(f"Ready to analyze {len(self.current_files)} files.")

    def _find_initial_audio_file(self):
        """
        Automatically finds all supported audio files in the current directory
        and loads them into the application. If only one file is found, it
        attempts to load its corresponding analysis settings.
        """
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

                # If only one file was auto-detected, try to load its settings
                if len(self.current_files) == 1:
                    self._load_settings_for_file(self.current_files[0])
                else:
                    # Otherwise, set a general status for batch processing
                    self._update_status(f"Auto-loaded {len(self.current_files)} files from the current directory.")

        except Exception as e:
            # Fails silently if it can't read the directory
            pass

    def _load_settings_for_file(self, file_path: str):
        """Checks for and loads 'start_bpm_hint' from a corresponding JSON file."""
        output_dir = os.path.join(os.getcwd(), "processed_files")
        base_name, _ = os.path.splitext(os.path.basename(file_path))
        settings_path = os.path.join(output_dir, f"{base_name}_Analysis_Settings.json")

        self.bpm_entry.delete(0, tk.END)

        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)

                if settings.get('start_bpm_hint') is not None:
                    bpm_value = settings['start_bpm_hint']
                    self.bpm_entry.insert(0, str(bpm_value))
                    self._update_status(f"Loaded BPM hint ({bpm_value}) from settings file.")
                else:
                    self._update_status(f"Found settings file, but no BPM hint inside.")
            except Exception as e:
                self._update_status(f"Error reading settings file for {os.path.basename(file_path)}.")
                print(f"ERROR: Could not parse {settings_path}. Details: {e}")
        else:
            self._update_status(f"Ready to analyze. No previous settings file found.")

    def open_output_folder(self):
        """Opens the output folder where analysis results are stored."""
        output_dir = os.path.join(os.getcwd(), "processed_files")
        if os.path.exists(output_dir):
            # Use the appropriate command for Windows
            os.startfile(output_dir)
            self._update_status(f"Opened output folder: {output_dir}")
        else:
            messagebox.showerror("Error", "Output folder does not exist")
            
    def _update_status(self, message):
        """Safely update the status bar from any thread."""
        self.root.after(0, lambda: self.status_var.set(message))

    def start_analysis_thread(self):
        """Starts the analysis in a new thread."""
        if not self.current_files:
            messagebox.showerror("Error", "No files selected")
            return

        self.analyze_btn.config(state=tk.DISABLED)
        self._update_status(f"Starting batch analysis of {len(self.current_files)} files...")

        analysis_thread = threading.Thread(target=self._run_analysis_in_background)
        analysis_thread.daemon = True
        analysis_thread.start()

    def _run_analysis_in_background(self):
        try:
            from bpm_analysis import analyze_wav_file, convert_to_wav
            import shutil

            # Check for a global BPM value to override all individual settings.
            bpm_override_input = self.bpm_entry.get().strip()
            bpm_override_hint = float(bpm_override_input) if bpm_override_input else None

            output_dir = os.path.join(os.getcwd(), "processed_files")
            os.makedirs(output_dir, exist_ok=True)

            total_files = len(self.current_files)
            files_processed = 0
            errors = []

            # --- BATCH PROCESSING LOOP ---
            for i, file_path in enumerate(self.current_files):
                try:
                    self.log_queue.put(UIMessage(UIMessageType.STATUS,
                                                 f"({i + 1}/{total_files}) Processing: {os.path.basename(file_path)}"))

                    # --- START: Per-File Settings Logic ---
                    # The BPM hint to be used for the current file.
                    start_bpm_hint = None
                    if bpm_override_hint is not None:
                        # Use the global override if the user entered a value.
                        start_bpm_hint = bpm_override_hint
                    else:
                        # Otherwise, try to load settings for this specific file.
                        base_name_for_settings, _ = os.path.splitext(os.path.basename(file_path))
                        settings_path = os.path.join(output_dir, f"{base_name_for_settings}_Analysis_Settings.json")
                        if os.path.exists(settings_path):
                            try:
                                with open(settings_path, 'r', encoding='utf-8') as f:
                                    settings = json.load(f)
                                if settings.get('start_bpm_hint') is not None:
                                    start_bpm_hint = float(settings['start_bpm_hint'])
                            except Exception:
                                # If file is corrupt or unreadable, just proceed without the hint.
                                pass
                    # --- END: Per-File Settings Logic ---

                    base_name, ext = os.path.splitext(file_path)
                    wav_path = os.path.join(output_dir, f"{os.path.basename(base_name)}.wav")

                    if ext.lower() != '.wav':
                        self.log_queue.put(UIMessage(UIMessageType.STATUS,
                                                     f"({i + 1}/{total_files}) Converting {os.path.basename(file_path)}..."))
                        if not convert_to_wav(file_path, wav_path):
                            raise Exception("File conversion failed.")
                    else:
                        shutil.copy(file_path, wav_path)

                    self.log_queue.put(
                        UIMessage(UIMessageType.STATUS, f"({i + 1}/{total_files}) Analyzing heartbeat..."))

                    # Pass the file-specific start_bpm_hint to the analysis function.
                    analyze_wav_file(wav_path, self.params, start_bpm_hint, original_file_path=file_path,
                                     output_directory=output_dir)
                    files_processed += 1

                except Exception as e:
                    # Inner try-except block to handle errors for a single file
                    error_info = f"Error processing '{os.path.basename(file_path)}':\n{str(e)}"
                    self.log_queue.put(UIMessage(UIMessageType.ERROR, error_info))
                    errors.append(os.path.basename(file_path))

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