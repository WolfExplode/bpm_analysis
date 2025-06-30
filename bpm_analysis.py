#====== Version 0.2.0 ======
import os
import warnings
import csv
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

try:
    from pydub import AudioSegment
except ImportError:
    print("Pydub library not found. Please install it with 'pip install pydub'")
    print("You also need FFmpeg installed and in your system's PATH for audio conversion.")
    AudioSegment = None

def convert_to_wav(file_path, target_path):
    """
    Converts any audio file supported by FFmpeg to a mono WAV file.
    """
    if not AudioSegment:
        raise ImportError("Pydub/FFmpeg is required for audio conversion.")

    print(f"Converting {os.path.basename(file_path)} to WAV format...")
    try:
        sound = AudioSegment.from_file(file_path)
        sound = sound.set_channels(1) # Convert to mono
        sound.export(target_path, format="wav")
        return True
    except Exception as e:
        print(f"Could not convert file {file_path}. Error: {e}")
        return False


def preprocess_audio(file_path, downsample_factor=50, bandpass_freqs=(20, 150), save_debug_file=False):
    """
    Loads, filters, and preprocesses the audio file.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample_rate, audio_data = wavfile.read(file_path)

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # --- 1. Band-pass filter to isolate heart sounds ---
    lowcut, highcut = bandpass_freqs
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(2, [low, high], btype='band')
    audio_filtered = filtfilt(b, a, audio_data)

    if save_debug_file:
        debug_path = f"{os.path.splitext(file_path)[0]}_filtered_debug.wav"
        normalized_audio = np.int16(audio_filtered / np.max(np.abs(audio_filtered)) * 32767)
        wavfile.write(debug_path, sample_rate, normalized_audio)
        print(f"Saved filtered debug audio to {debug_path}")

    # --- 2. Downsample for performance ---
    new_sample_rate = sample_rate // downsample_factor
    audio_downsampled = audio_filtered[::downsample_factor]

    # --- 3. Calculate the envelope of the signal ---
    audio_abs = np.abs(audio_downsampled)
    window_size = new_sample_rate // 10
    audio_envelope = pd.Series(audio_abs).rolling(window=window_size, min_periods=1, center=True).mean().values

    return audio_envelope, new_sample_rate

def find_heartbeat_peaks(audio_envelope, sample_rate, min_bpm=40, max_bpm=220, start_bpm_hint=None):
    """
    Finds heartbeats by dynamically identifying S1-S2 patterns.
    This version incorporates user hints for starting BPM.
    Returns the final filtered peaks and the initial raw peaks for debugging.
    """
    min_peak_distance_samples = int((60.0 / 240.0) * sample_rate)
    prominence_threshold = np.quantile(audio_envelope, 0.6)
    height_threshold = np.mean(audio_envelope) * 0.6

    all_peaks, _ = find_peaks(
        audio_envelope,
        distance=min_peak_distance_samples,
        prominence=prominence_threshold,
        height=height_threshold
    )

    if len(all_peaks) < 3:
        return all_peaks, all_peaks

    peak_times_sec = all_peaks / sample_rate
    intervals_sec = np.diff(peak_times_sec)

    s1_s2_max_interval_sec = 0.35 # Default fallback

    # --- Use user-provided BPM to guide S1-S2 interval detection ---
    if start_bpm_hint and start_bpm_hint > 0:
        print(f"Using user hint: Starting BPM = {start_bpm_hint}")
        expected_beat_interval = 60.0 / start_bpm_hint
        # The systolic interval (S1-S2) is typically 30-45% of the total cycle.
        # We'll use 50% for a safe upper margin.
        s1_s2_max_interval_sec = expected_beat_interval * 0.50
    # --- Improved automatic detection when no hint is given ---
    elif len(intervals_sec) > 5:
        print("Using improved automatic detection for S1-S2 interval.")
        # Estimate the full beat cycle by summing adjacent intervals (systolic + diastolic)
        # This is more robust to variations than just the median of all intervals.
        beat_cycle_estimates = intervals_sec[:-1] + intervals_sec[1:]
        robust_beat_interval = np.median(beat_cycle_estimates)
        dynamic_threshold = robust_beat_interval * 0.45
        s1_s2_max_interval_sec = np.clip(dynamic_threshold, 0.15, 0.4) # Clamp to physiological range

    print(f"Set max S1-S2 interval threshold to {s1_s2_max_interval_sec:.2f} seconds.")

    true_beat_indices = []
    i = 0
    while i < len(intervals_sec):
        # If the interval is short, it's likely an S1-S2 pair
        if intervals_sec[i] <= s1_s2_max_interval_sec:
            peak1_idx, peak2_idx = all_peaks[i], all_peaks[i + 1]
            # Pick the louder of the two peaks (S1 or S2) as the primary beat
            true_beat_indices.append(peak1_idx if audio_envelope[peak1_idx] >= audio_envelope[peak2_idx] else peak2_idx)
            i += 2 # Skip the next peak as it's part of the pair
        else:
            # This interval is too long to be S1-S2, so the peak is a standalone beat.
            true_beat_indices.append(all_peaks[i])
            i += 1

    # Add the very last detected peak if it wasn't processed
    if i == len(intervals_sec):
        true_beat_indices.append(all_peaks[-1])

    # Remove duplicates that might arise from edge cases
    true_beat_indices = np.array(list(dict.fromkeys(true_beat_indices)))

    # Final filtering based on min/max BPM rules
    if len(true_beat_indices) > 1:
        final_peak_times = true_beat_indices / sample_rate
        final_intervals = np.diff(final_peak_times)
        min_interval, max_interval = 60.0 / max_bpm, 60.0 / min_bpm

        # Start with the first valid peak
        filtered_peaks = [true_beat_indices[0]]
        for j in range(len(final_intervals)):
            # Add the next peak only if the resulting interval is within the plausible BPM range
            if min_interval <= final_intervals[j] <= max_interval:
                filtered_peaks.append(true_beat_indices[j + 1])
        return np.array(filtered_peaks), all_peaks

    return true_beat_indices, all_peaks


def calculate_bpm_series(peaks, sample_rate, smoothing_window_sec=5):
    """
    Calculates the BPM over time from the detected peaks and smooths it.
    """
    if len(peaks) < 2:
        return pd.Series(dtype=np.float64), np.array([])

    peak_times = peaks / sample_rate
    time_diffs = np.diff(peak_times)
    instant_bpm = 60.0 / time_diffs
    bpm_series = pd.Series(instant_bpm, index=peak_times[1:])

    avg_heart_rate = np.median(instant_bpm)
    if avg_heart_rate > 0:
        beats_in_window = int(np.ceil((smoothing_window_sec / 60) * avg_heart_rate))
        beats_in_window = max(2, beats_in_window)
        smoothed_bpm = bpm_series.rolling(window=beats_in_window, min_periods=1, center=True).mean()
    else:
        smoothed_bpm = pd.Series(dtype=np.float64)

    time_points = (peak_times[:-1] + peak_times[1:]) / 2
    return smoothed_bpm, time_points

def plot_results(audio_envelope, peaks, all_raw_peaks, smoothed_bpm, bpm_times, sample_rate, file_name):
    """
    Creates an interactive plot of the results, including a hidden raw peaks trace for debugging.
    """
    time_axis = np.arange(len(audio_envelope)) / sample_rate
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=time_axis, y=audio_envelope, name="Audio Envelope", line=dict(color="#47a5c4")), secondary_y=False)

    if len(all_raw_peaks) > 0:
        fig.add_trace(
            go.Scatter(
                x=all_raw_peaks / sample_rate,
                y=audio_envelope[all_raw_peaks],
                mode='markers',
                name='All Detected Peaks (Raw)',
                marker=dict(color='grey', symbol='x', size=6),
                visible='legendonly' # This hides the trace by default
            ),
            secondary_y=False
        )

    fig.add_trace(go.Scatter(x=peaks / sample_rate, y=audio_envelope[peaks], mode='markers', name='Detected Heartbeats', marker=dict(color='#e36f6f', size=8)), secondary_y=False)

    if not smoothed_bpm.empty:
        fig.add_trace(go.Scatter(x=bpm_times, y=smoothed_bpm, name="Smoothed BPM", line=dict(color="#4a4a4a", width=3)), secondary_y=True)

    if not smoothed_bpm.empty:
        max_bpm_val, min_bpm_val, avg_bpm_val = smoothed_bpm.max(), smoothed_bpm.min(), smoothed_bpm.mean()
        max_bpm_time, min_bpm_time = smoothed_bpm.idxmax(), smoothed_bpm.idxmin()
        fig.add_annotation(x=max_bpm_time, y=max_bpm_val, text=f"Max BPM: {max_bpm_val:.1f}", showarrow=True, arrowhead=1, ax=20, ay=-40, font=dict(color="#e36f6f"), yref="y2")
        fig.add_annotation(x=min_bpm_time, y=min_bpm_val, text=f"Min BPM: {min_bpm_val:.1f}", showarrow=True, arrowhead=1, ax=20, ay=40, font=dict(color="#a3d194"), yref="y2")
        fig.add_annotation(x=bpm_times[-1] if bpm_times.size > 0 else 0, y=avg_bpm_val, text=f"Avg BPM: {avg_bpm_val:.1f}", showarrow=False, xanchor="right", yanchor="top", font=dict(color="#4a4a4a"), yref="y2")

    max_time_sec = time_axis[-1] if len(time_axis) > 0 else 1
    tick_interval_sec = max(30, np.ceil(max_time_sec / 10))
    tick_vals_sec = np.arange(0, max_time_sec + tick_interval_sec, tick_interval_sec)

    # Generate combined tick text: "seconds (minutes:seconds)"
    tick_text_combined = []
    for s in tick_vals_sec:
        minutes = int(s // 60)
        remaining_seconds = int(s % 60)
        tick_text_combined.append(f"{int(s)}s ({minutes:02d}:{remaining_seconds:02d})")

    fig.update_layout(
        title_text=f"Heartbeat Analysis - {os.path.basename(file_name)}",
        dragmode='pan',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=100, b=100),  # Add top and bottom margin for the axes

        # Define the single x-axis for combined seconds and minutes display
        xaxis=dict(
            title_text="Time", # Simplified title
            tickvals=tick_vals_sec,
            ticktext=tick_text_combined,
        )
    )

    max_amplitude = np.max(audio_envelope) if len(audio_envelope) > 0 else 1000
    fig.update_yaxes(title_text="Signal Amplitude", secondary_y=False, range=[-0.1 * max_amplitude, 8.0 * max_amplitude])
    fig.update_yaxes(title_text="Beats Per Minute (BPM)", secondary_y=True, range=[max(0, smoothed_bpm.min(skipna=True)-10) if not smoothed_bpm.empty else 0, smoothed_bpm.max(skipna=True)+10 if not smoothed_bpm.empty else 260])

    output_html_path = f"{os.path.splitext(file_name)[0]}_bpm_plot.html"
    fig.write_html(output_html_path, config={'scrollZoom': True})
    print(f"Interactive plot saved to {output_html_path}")

def save_bpm_to_csv(bpm_series, time_points, output_path):
    """Saves the calculated BPM data to a CSV file."""
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Time (s)", "BPM"])
        if not bpm_series.empty:
            for t, bpm in zip(time_points, bpm_series):
                if not np.isnan(bpm):
                    writer.writerow([f"{t:.2f}", f"{bpm:.1f}"])

def analyze_wav_file(wav_file_path, params, start_bpm_hint):
    """Runs the full analysis pipeline on a single WAV file."""
    file_name_no_ext = os.path.splitext(wav_file_path)[0]
    print(f"Processing file: {os.path.basename(wav_file_path)}...")

    audio_envelope, sample_rate = preprocess_audio(wav_file_path, params['downsample_factor'], params['bandpass_freqs'], params['save_debug_wav'])

    # Pass the user hint to the peak finding function
    peaks, all_raw_peaks = find_heartbeat_peaks(
        audio_envelope, sample_rate, params['min_bpm'], params['max_bpm'],
        start_bpm_hint=start_bpm_hint
    )
    print(f"Detected {len(peaks)} true heartbeats from {len(all_raw_peaks)} raw peaks.")

    if len(peaks) < 2:
        print("Not enough peaks detected to calculate BPM.")
        plot_results(audio_envelope, peaks, all_raw_peaks, pd.Series(dtype=np.float64), np.array([]), sample_rate, wav_file_path)
        return

    smoothed_bpm, bpm_times = calculate_bpm_series(peaks, sample_rate, params['smoothing_window_sec'])

    output_csv_path = f"{file_name_no_ext}_bpm_analysis.csv"
    save_bpm_to_csv(smoothed_bpm, bpm_times, output_csv_path)
    print(f"BPM data saved to {output_csv_path}")

    plot_results(audio_envelope, peaks, all_raw_peaks, smoothed_bpm, bpm_times, sample_rate, wav_file_path)

class BPMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Heartbeat BPM Analyzer")
        self.root.geometry("500x300") # Adjusted geometry due to removed elements
        self.style = ttkb.Style(theme='minty')  #Choose from available themes like 'minty', 'litera', 'pulse', etc.

        self.current_file = None
        self.params = {
            "downsample_factor": 100,
            "bandpass_freqs": (20, 150),
            "min_bpm": 40,
            "max_bpm": 220,
            "smoothing_window_sec": 5,
            "save_debug_wav": True
        }

        self.create_widgets()
        self._find_initial_audio_file() # Call this after widgets are created

    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="Audio File", padding=10)
        file_frame.pack(fill=tk.X, pady=5)

        self.file_label = ttk.Label(file_frame, text="No file selected", wraplength=400)
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        browse_btn = ttk.Button(file_frame, text="Browse", command=self.select_file, bootstyle=INFO)
        browse_btn.pack(side=tk.RIGHT, padx=5)

        # Analysis parameters
        param_frame = ttk.LabelFrame(main_frame, text="Analysis Parameters", padding=10)
        param_frame.pack(fill=tk.X, pady=5)

        # BPM hint
        ttk.Label(param_frame, text="Starting BPM (optional):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.bpm_entry = ttk.Entry(param_frame)
        self.bpm_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)

        # Action buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        self.analyze_btn = ttk.Button(btn_frame, text="Analyze", command=self.analyze, bootstyle=SUCCESS, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.RIGHT, padx=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, pady=(10, 0))

        # Configure grid weights
        param_frame.columnconfigure(1, weight=1)

    def select_file(self):
        filetypes = [
            ('Audio files', '*.wav *.mp3 *.m4a *.flac *.ogg *.mp4 *.mkv'),
            ('All files', '*.*')
        ]

        filename = filedialog.askopenfilename(title="Select audio file", filetypes=filetypes)
        if filename:
            self.current_file = filename
            self.file_label.config(text=os.path.basename(filename))
            self.analyze_btn.config(state=tk.NORMAL)
            self.status_var.set("Ready to analyze")

    def _find_initial_audio_file(self):
        """Attempts to auto-populate with the first valid audio file in the current working directory."""
        supported_extensions = ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.mp4')

        # Use os.getcwd() to get the current working directory, which is more reliable in interactive environments.
        script_dir = os.getcwd()

        for filename in os.listdir(script_dir):
            file_path = os.path.join(script_dir, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(supported_extensions):
                self.current_file = file_path
                self.file_label.config(text=os.path.basename(file_path))
                self.analyze_btn.config(state=tk.NORMAL)
                self.status_var.set(f"Auto-loaded: {os.path.basename(file_path)}")
                return # Found one, so stop

        # If no file was found
        self.file_label.config(text="No file selected")
        self.analyze_btn.config(state=tk.DISABLED)
        self.status_var.set("No audio file found in current directory. Please browse.")

    def analyze(self):
        if not self.current_file:
            messagebox.showerror("Error", "No file selected")
            return

        try:
            self.status_var.set("Analyzing...")
            self.root.update()

            # Get user inputs
            bpm_input = self.bpm_entry.get().strip()
            start_bpm_hint = float(bpm_input) if bpm_input else None

            # Create converted directory if needed
            converted_dir = "converted_wavs"
            if not os.path.exists(converted_dir):
                os.makedirs(converted_dir)

            # Process the file
            base_name, ext = os.path.splitext(self.current_file)
            wav_path = os.path.join(converted_dir, f"{os.path.basename(base_name)}.wav")

            if ext.lower() != '.wav':
                if not AudioSegment:
                    messagebox.showerror("Error", "Pydub/FFmpeg is required for audio conversion of non-WAV files. Please install them.")
                    self.status_var.set("Conversion failed: Missing Pydub/FFmpeg")
                    return
                if not convert_to_wav(self.current_file, wav_path):
                    self.status_var.set("Conversion failed")
                    return
            else:
                import shutil
                shutil.copy(self.current_file, wav_path)

            analyze_wav_file(wav_path, self.params, start_bpm_hint)
            self.status_var.set("Analysis complete - check console for results")

        except Exception as e:
            self.status_var.set("Error during analysis")
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            traceback.print_exc()

def main():
    """Main function to launch the GUI."""
    root = ttkb.Window()
    app = BPMApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()