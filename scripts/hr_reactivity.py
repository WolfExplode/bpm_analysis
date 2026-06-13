#!/usr/bin/env python3
"""
Smartwatch BPM/Time analyzer: detects exercise periods and computes HRA & HRR.

Inputs:
 - CSV with two columns: time_seconds, bpm  (header optional; will attempt to parse)
 - Age (years)
 - Resting HR (bpm)

Outputs:
 - Plotly interactive graph with detected exercise windows, start/peak/end markers
 - Per-period metrics table and summary
 - Option to export results to CSV

Tuning parameters available in the GUI:
 - inc_threshold: bpm/sec (threshold for considering HR increasing)
 - dec_threshold: bpm/sec (negative threshold for considering HR decreasing)
 - min_hill_duration: seconds (minimum duration to consider a hill)
 - sustain_seconds: seconds (how long a slope condition must hold to count)
 - allowed_gap: seconds (allow small dips/gaps inside a hill)
 - derivative_window: seconds (window for computing derivative smoothing; 1 disables smoothing)
"""

# =============================================================================
# CONFIGURATION SECTION - Adjust these values to tweak the analysis
# =============================================================================

# Default user inputs
DEFAULT_AGE = 16
DEFAULT_RESTING_HR = 57

# Exercise detection parameters
DEFAULT_INC_THRESHOLD = 0.6  # bpm/sec (threshold for considering HR increasing)
DEFAULT_DEC_THRESHOLD = -0.6  # bpm/sec (negative threshold for considering HR decreasing)
DEFAULT_MIN_HILL_DURATION = 30  # seconds (minimum duration to consider a hill)
DEFAULT_SUSTAIN_SECONDS = 10  # seconds (how long a slope condition must hold to count)
DEFAULT_ALLOWED_GAP = 5  # seconds (allow small dips/gaps inside a hill)
DEFAULT_DERIVATIVE_WINDOW = 2  # seconds (window for computing derivative smoothing)

# Advanced detection parameters (used in improved algorithm)
DEFAULT_HR_START_THRESHOLD_PCT = 0.15  # 15% of HR reserve above resting
DEFAULT_HR_DROP_THRESHOLD_PCT = 0.10  # 10% of HR reserve drop from peak
DEFAULT_BASE_INC_RATE = 0.3  # Base rate in bpm/sec for exercise detection
DEFAULT_HR_RESERVE_INC_FACTOR = 0.008  # 0.8% of HR reserve per second
DEFAULT_BASE_DEC_RATE = -0.2  # Base recovery rate in bpm/sec
DEFAULT_HR_RESERVE_DEC_FACTOR = -0.005  # 0.5% of HR reserve per second

# Plotting parameters
DEFAULT_PLOT_Y_RANGE_BPM = [50, 200]  # BPM range for primary y-axis
DEFAULT_PLOT_Y_RANGE_DERIVATIVE = [-50, 50]  # Derivative range for secondary y-axis

# Test data generation parameters
DEFAULT_TEST_DURATION_MINUTES = 10
DEFAULT_TEST_SAMPLE_RATE = 1.0  # samples per second

# =============================================================================

import os
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import ttkbootstrap as tb
from ttkbootstrap.constants import *

from scipy.interpolate import interp1d

from time_utils import timestamp_str, seconds_to_datetime


# ---------- Core analysis functions ----------

def load_csv(filepath):
    """Load CSV and attempt to parse columns (time_seconds, bpm)."""
    df = pd.read_csv(filepath)
    # permissive column handling
    cols = [c.lower().strip() for c in df.columns]
    if 'time' in cols and 'bpm' in cols:
        pass
    elif 'time_seconds' in cols and 'bpm' in cols:
        pass
    elif len(cols) >= 2:
        # assume first two are time and bpm
        df = df.iloc[:, :2]
        df.columns = ['time_seconds', 'bpm']
    else:
        raise ValueError("CSV must have at least two columns (time_seconds, bpm)")
    # normalize column names
    colmap = {c: c for c in df.columns}
    lower_map = {c.lower(): c for c in df.columns}
    if 'time' in lower_map:
        colmap[lower_map['time']] = 'time_seconds'
    elif 'time_seconds' in lower_map:
        colmap[lower_map['time_seconds']] = 'time_seconds'
    if 'bpm' in lower_map:
        colmap[lower_map['bpm']] = 'bpm'
    df = df.rename(columns=colmap)
    df = df[['time_seconds', 'bpm']]
    df = df.dropna().sort_values('time_seconds').reset_index(drop=True)
    return df

def compute_derivative(df, derivative_window=3):
    """
    Compute approximate dbpm/dt (bpm per second).
    derivative_window: number of samples window (in seconds) to compute rolling slope smoothing.
                       If <=1, compute simple adjacent difference.
    Returns df with 'dbpm_dt'.
    """
    t = df['time_seconds'].to_numpy()
    hr = df['bpm'].to_numpy()
    # compute instantaneous slope between neighbors
    dt = np.diff(t)
    dh = np.diff(hr)
    # avoid division by zero
    slopes = np.concatenate(([0.0], dh / np.where(dt == 0, 1e-6, dt)))
    df = df.copy()
    df['dbpm_dt_raw'] = slopes
    if derivative_window is None or derivative_window <= 1:
        df['dbpm_dt'] = df['dbpm_dt_raw']
    else:
        # smoothing: simple moving average over approximate seconds window
        # convert seconds-window to number of samples by nearest integer average sample spacing
        avg_dt = np.median(dt) if len(dt) > 0 else 1.0
        window_samples = max(1, int(round(derivative_window / avg_dt)))
        # Light Gaussian smoothing (closer to core pipeline smoothing than boxcar mean)
        try:
            from scipy.ndimage import gaussian_filter1d
            # window_samples ≈ 6σ → σ ≈ window/6
            sigma = max(1.0, float(window_samples) / 6.0)
            df['dbpm_dt'] = gaussian_filter1d(df['dbpm_dt_raw'].to_numpy(dtype=float), sigma=sigma, mode="nearest")
        except Exception:
            df['dbpm_dt'] = df['dbpm_dt_raw'].rolling(window=window_samples, center=True, min_periods=1).mean()
    return df

def merge_bool_runs(times, mask, allowed_gap=5.0):
    """
    Given boolean mask aligned to times array, merge runs allowing small gaps up to allowed_gap seconds.
    Returns list of (start_idx, end_idx) inclusive indices for each merged True run.
    """
    runs = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        # start a run
        start = i
        j = i
        while j < n:
            if mask[j]:
                j += 1
                continue
            # found a gap; check gap size until next True
            gap_start = j
            k = j
            while k < n and not mask[k]:
                k += 1
            gap_end = k
            if k >= n:
                # end of array
                j = k
                break
            gap_duration = times[gap_end] - times[gap_start]
            if gap_duration <= allowed_gap:
                # merge across the gap
                j = gap_end  # continue from next True
                continue
            else:
                # stop run at last True before gap
                break
        end = max(start, j-1)
        runs.append((start, end))
        i = j
    return runs

def find_sustained_condition(df_times, df_values, idx, direction='backward', threshold=0.5, sustain_seconds=10.0):
    """
    Starting at idx, move backward or forward until we find sustained period where value < threshold (if direction backward)
    or > threshold depending on sign. We use this helper to find where slope becomes non-positive for a sustained window.
    Returns the index where the sustained condition begins/ends. If not found, returns the array boundary.
    This function is generic; use with dbpm_dt and threshold signs.
    """
    n = len(df_times)
    if direction == 'backward':
        i = idx
        while i >= 0:
            # window from (i - w + 1) to i; find duration of consecutive indices where condition holds
            # We'll check windows of sustain_seconds: find earliest j <= i where time[i] - time[j] >= sustain_seconds
            j = i
            while j >= 0 and (df_times[i] - df_times[j]) < sustain_seconds:
                j -= 1
            j += 1
            # now window j..i is at least sustain_seconds long (or from 0 to i)
            window_vals = df_values[j:i+1]
            # check condition: for 'backward' we consider condition satisfied if all vals are below threshold
            if np.all(window_vals <= threshold):
                return j
            i -= 1
        return 0
    else:
        i = idx
        while i < n:
            j = i
            while j < n and (df_times[j] - df_times[i]) < sustain_seconds:
                j += 1
            j -= 1
            window_vals = df_values[i:j+1]
            if np.all(window_vals <= threshold):
                return j
            i += 1
        return n-1

def find_inflection_point(hr_values, times):
    """
    Find the inflection point in HR data where the curve changes from concave to convex
    (or vice versa). This helps identify the true start/end of exercise periods.
    Returns the index of the inflection point, or None if not found.
    """
    if len(hr_values) < 5:
        return None

    # Calculate second derivative (acceleration)
    if len(times) > 0:
        dt = np.diff(times)
        if np.any(dt == 0):
            return None
        first_deriv = np.diff(hr_values) / dt
        second_deriv = np.diff(first_deriv) / dt[1:]
    else:
        # Fallback to simple differences if no time data
        first_deriv = np.diff(hr_values)
        second_deriv = np.diff(first_deriv)

    if len(second_deriv) < 3:
        return None

    # Find where second derivative changes sign (inflection point)
    sign_changes = np.where(np.diff(np.sign(second_deriv)) != 0)[0]

    if len(sign_changes) == 0:
        return None

    # Return the first significant inflection point
    # Add 2 to account for the double differentiation
    return sign_changes[0] + 2

def detect_exercise_periods(df,
                            resting_hr,
                            max_hr,
                            inc_threshold=None,
                            dec_threshold=None,
                            min_hill_duration=30.0,
                            sustain_seconds=10.0,
                            allowed_gap=5.0,
                            derivative_window=3.0):
    """
    Detect exercise 'hills' where HR is rising.
    Returns list of dicts with start_idx, peak_idx, end_idx, start_time, peak_time, end_time, start_hr, peak_hr, end_hr.
    Method summary:
     - compute dbpm/dt; determine is_increasing where slope > inc_threshold
     - merge runs allowing small gaps
     - keep runs with duration >= min_hill_duration
     - for each run, find peak index (max hr), then walk backwards from peak to find t_start using either:
         - HR <= resting + 0.1 * HR_reserve  OR
         - derivative <= inc_threshold for sustain_seconds
       Walk forward from peak to find t_end similarly based on dec_threshold sustained
    """
    df = df.copy().reset_index(drop=True)
    times = df['time_seconds'].to_numpy()
    hr = df['bpm'].to_numpy()
    df = compute_derivative(df, derivative_window=derivative_window)
    slopes = df['dbpm_dt'].to_numpy()

    hr_reserve = max_hr - resting_hr

    # Improved default thresholds based on physiological principles
    if inc_threshold is None:
        # More sophisticated threshold: base rate + percentage of HR reserve
        inc_threshold = DEFAULT_BASE_INC_RATE + (DEFAULT_HR_RESERVE_INC_FACTOR * hr_reserve)
    if dec_threshold is None:
        # Recovery threshold should be more aggressive than exercise threshold
        dec_threshold = DEFAULT_BASE_DEC_RATE + (DEFAULT_HR_RESERVE_DEC_FACTOR * hr_reserve)

    is_increasing = slopes > inc_threshold

    # merge runs permitting tiny gaps
    runs = merge_bool_runs(times, is_increasing, allowed_gap=allowed_gap)

    periods = []
    for (sidx, eidx) in runs:
        start_time = times[sidx]
        end_time = times[eidx]
        run_duration = end_time - start_time
        if run_duration < min_hill_duration:
            continue
        # find internal peak index (global maximum hp inside run)
        local_hr = hr[sidx:eidx+1]
        if len(local_hr) == 0:
            continue
        local_peak_rel = int(np.argmax(local_hr))
        peak_idx = sidx + local_peak_rel
        peak_time = times[peak_idx]
        peak_hr = hr[peak_idx]

        # Improved exercise start detection
        # Look for the point where HR starts to rise significantly from baseline
        hr_start_threshold = resting_hr + DEFAULT_HR_START_THRESHOLD_PCT * hr_reserve

        # Method 1: Find where HR crosses the threshold
        hr_below_threshold = np.where(hr[:peak_idx+1] <= hr_start_threshold)[0]
        hr_start_candidate = hr_below_threshold[-1] if len(hr_below_threshold) > 0 else sidx

        # Method 2: Find where derivative becomes positive and sustained
        derivative_start_candidate = find_sustained_condition(times, slopes, peak_idx, direction='backward',
                                                             threshold=inc_threshold, sustain_seconds=sustain_seconds)

        # Method 3: Find the actual inflection point (where second derivative changes)
        inflection_candidate = find_inflection_point(hr[:peak_idx+1], times[:peak_idx+1])
        if inflection_candidate is not None:
            inflection_candidate = sidx + inflection_candidate

        # Choose the most conservative (earliest) start point
        cand_start_idx = max(sidx, min(hr_start_candidate, derivative_start_candidate,
                                      inflection_candidate if inflection_candidate is not None else peak_idx))
        cand_start_idx = min(cand_start_idx, peak_idx)

        # Improved exercise end detection
        # Method 1: Find where derivative becomes negative and sustained
        forward_idx = find_sustained_condition(times, -slopes, peak_idx, direction='forward',
                                               threshold=-dec_threshold, sustain_seconds=sustain_seconds)

        # Method 2: Find where HR drops significantly from peak
        hr_drop_threshold = peak_hr - DEFAULT_HR_DROP_THRESHOLD_PCT * hr_reserve
        hr_drop_idx_rel = np.where(hr[peak_idx:] <= hr_drop_threshold)[0]
        hr_drop_candidate = peak_idx + hr_drop_idx_rel[0] if len(hr_drop_idx_rel) > 0 else eidx

        # Method 3: Find recovery inflection point
        recovery_inflection = find_inflection_point(hr[peak_idx:], times[peak_idx:])
        recovery_inflection_candidate = peak_idx + recovery_inflection if recovery_inflection is not None else eidx

        # Choose the most conservative (earliest) end point
        cand_end_idx = min(eidx, forward_idx, hr_drop_candidate, recovery_inflection_candidate)

        # make sure indices sensible
        if cand_end_idx <= cand_start_idx:
            # if no sensible end found, use run end
            cand_end_idx = eidx

        periods.append({
            'run_start_idx': sidx,
            'run_end_idx': eidx,
            'start_idx': int(cand_start_idx),
            'peak_idx': int(peak_idx),
            'end_idx': int(cand_end_idx),
            'start_time': float(times[cand_start_idx]),
            'peak_time': float(peak_time),
            'end_time': float(times[cand_end_idx]),
            'start_hr': float(hr[cand_start_idx]),
            'peak_hr': float(peak_hr),
            'end_hr': float(hr[cand_end_idx]),
            'inc_threshold': float(inc_threshold),
            'dec_threshold': float(dec_threshold)
        })

    return periods, df

def compute_hra_hrr_for_period(df, period, hr_reserve, t90_pct=0.9):
    """
    For a given detected period (dict), compute:
     - HRA_peak = (peak_hr - start_hr) / (t_peak - t_start)
     - HRA_90 = (0.9*peak_hr - start_hr) / (t_90 - t_start)  [if t_90 exists within period]
     - HRA_50 = (0.5*peak_hr - start_hr) / (t_50 - t_start)  [if t_50 exists within period]
     - HRR_60 = peak_hr - hr_at_1min_after_end (if data exists, interpolate)
     - HRR_30 = peak_hr - hr_at_30sec_after_end (if data exists, interpolate)
     - HRR_120 = peak_hr - hr_at_2min_after_end (if data exists, interpolate)
    Returns dict with metrics.
    """
    times = df['time_seconds'].to_numpy()
    hr = df['bpm'].to_numpy()

    start_idx = period['start_idx']
    peak_idx = period['peak_idx']
    end_idx = period['end_idx']

    t_start = times[start_idx]
    t_peak = times[peak_idx]
    t_end = times[end_idx]
    hr_start = hr[start_idx]
    hr_peak = hr[peak_idx]

    # HRA_peak
    delta_t_peak = max(1e-6, t_peak - t_start)
    hra_peak = (hr_peak - hr_start) / delta_t_peak

    # HRA_90 - find when HR reaches 90% of peak
    target90 = hr_peak * t90_pct
    idx90_rel = np.where(hr[start_idx:peak_idx+1] >= target90)[0]
    if len(idx90_rel) > 0:
        idx90 = start_idx + idx90_rel[0]
        t90 = times[idx90]
        delta_t_90 = max(1e-6, t90 - t_start)
        hra_90 = (hr[idx90] - hr_start) / delta_t_90
    else:
        hra_90 = None

    # HRA_50 - find when HR reaches 50% of peak (useful for very steep climbs)
    target50 = hr_start + 0.5 * (hr_peak - hr_start)
    idx50_rel = np.where(hr[start_idx:peak_idx+1] >= target50)[0]
    if len(idx50_rel) > 0:
        idx50 = start_idx + idx50_rel[0]
        t50 = times[idx50]
        delta_t_50 = max(1e-6, t50 - t_start)
        hra_50 = (hr[idx50] - hr_start) / delta_t_50
    else:
        hra_50 = None

    # Multiple HRR calculations
    def calculate_hrr(interval_seconds):
        t_after = t_end + interval_seconds
        if t_after <= times[-1]:
            f = interp1d(times, hr, bounds_error=False, fill_value='extrapolate')
            hr_at_after = float(f(t_after))
            return hr_peak - hr_at_after
        else:
            # Use last available sample within reasonable range
            mask = (times >= t_end) & (times <= t_end + interval_seconds + 30)
            if mask.any():
                last_hr = float(hr[mask][-1])
                return hr_peak - last_hr
            return None

    hrr_30 = calculate_hrr(30)
    hrr_60 = calculate_hrr(60)
    hrr_120 = calculate_hrr(120)

    # Calculate additional metrics
    exercise_duration = t_end - t_start
    peak_duration = t_peak - t_start
    recovery_duration = t_end - t_peak

    # Calculate HR reserve utilization
    hr_reserve_used = hr_peak - hr_start
    hr_reserve_utilization = hr_reserve_used / hr_reserve if hr_reserve > 0 else 0

    return {
        'hra_peak_bpm_per_s': hra_peak,
        'hra_90_bpm_per_s': hra_90,
        'hra_50_bpm_per_s': hra_50,
        'hr_peak': float(hr_peak),
        'hr_start': float(hr_start),
        't_start': float(t_start),
        't_peak': float(t_peak),
        't_end': float(t_end),
        'hr_end': float(period['end_hr']),
        'hrr_30_bpm': hrr_30,
        'hrr_60_bpm': hrr_60,
        'hrr_120_bpm': hrr_120,
        'exercise_duration_sec': float(exercise_duration),
        'peak_duration_sec': float(peak_duration),
        'recovery_duration_sec': float(recovery_duration),
        'hr_reserve_used_bpm': float(hr_reserve_used),
        'hr_reserve_utilization_pct': float(hr_reserve_utilization * 100)
    }

# ---------- GUI and plotting ----------

class HRAnalyzerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("HR Reactivity Analyzer")
        self.style = tb.Style(theme="superhero")
        self.frame = tb.Frame(master, padding=12)
        self.frame.pack(fill=BOTH, expand=True)

        # inputs
        self.filepath_var = tk.StringVar()
        self.age_var = tk.IntVar(value=DEFAULT_AGE)
        self.rest_hr_var = tk.IntVar(value=DEFAULT_RESTING_HR)

        # tuning params
        self.inc_threshold_var = tk.DoubleVar(value=DEFAULT_INC_THRESHOLD)
        self.dec_threshold_var = tk.DoubleVar(value=DEFAULT_DEC_THRESHOLD)
        self.min_hill_sec_var = tk.IntVar(value=DEFAULT_MIN_HILL_DURATION)
        self.sustain_seconds_var = tk.IntVar(value=DEFAULT_SUSTAIN_SECONDS)
        self.allowed_gap_var = tk.IntVar(value=DEFAULT_ALLOWED_GAP)
        self.derivative_window_var = tk.IntVar(value=DEFAULT_DERIVATIVE_WINDOW)

        self.create_widgets()

        self.df = None
        self.detected_periods = None
        self.derived_df = None
        self.metrics = None

    def create_widgets(self):
        row = 0
        tb.Label(self.frame, text="CSV file (time_seconds,bpm):").grid(column=0, row=row, sticky=W)
        tb.Entry(self.frame, textvariable=self.filepath_var, width=62).grid(column=1, row=row, columnspan=3, sticky=W)
        tb.Button(self.frame, text="Browse", bootstyle="info-outline", command=self.browse_file).grid(column=4, row=row, sticky=W)
        row += 1

        tb.Label(self.frame, text="Age (years):").grid(column=0, row=row, sticky=W)
        tb.Entry(self.frame, textvariable=self.age_var, width=8).grid(column=1, row=row, sticky=W)
        tb.Label(self.frame, text="Resting HR (bpm):").grid(column=2, row=row, sticky=W)
        tb.Entry(self.frame, textvariable=self.rest_hr_var, width=8).grid(column=3, row=row, sticky=W)
        row += 1

        # tuning
        tb.Separator(self.frame).grid(column=0, row=row, columnspan=5, sticky="we", pady=6)
        row += 1
        tb.Label(self.frame, text="inc_threshold (bpm/s):").grid(column=0, row=row, sticky=W)
        tb.Entry(self.frame, textvariable=self.inc_threshold_var, width=8).grid(column=1, row=row, sticky=W)
        tb.Label(self.frame, text="dec_threshold (bpm/s):").grid(column=2, row=row, sticky=W)
        tb.Entry(self.frame, textvariable=self.dec_threshold_var, width=8).grid(column=3, row=row, sticky=W)
        row += 1
        tb.Label(self.frame, text="min_hill_duration (s):").grid(column=0, row=row, sticky=W)
        tb.Entry(self.frame, textvariable=self.min_hill_sec_var, width=8).grid(column=1, row=row, sticky=W)
        tb.Label(self.frame, text="sustain_seconds (s):").grid(column=2, row=row, sticky=W)
        tb.Entry(self.frame, textvariable=self.sustain_seconds_var, width=8).grid(column=3, row=row, sticky=W)
        row += 1
        tb.Label(self.frame, text="allowed_gap (s):").grid(column=0, row=row, sticky=W)
        tb.Entry(self.frame, textvariable=self.allowed_gap_var, width=8).grid(column=1, row=row, sticky=W)
        tb.Label(self.frame, text="derivative_window (s):").grid(column=2, row=row, sticky=W)
        tb.Entry(self.frame, textvariable=self.derivative_window_var, width=8).grid(column=3, row=row, sticky=W)
        row += 1

        # action buttons
        tb.Button(self.frame, text="Run Analysis", bootstyle="success", command=self.run_analysis).grid(column=0, row=row, pady=8, sticky=W)
        tb.Button(self.frame, text="Export Results CSV", bootstyle="primary", command=self.export_results).grid(column=1, row=row, pady=8, sticky=W)
        tb.Button(self.frame, text="Save Plot (HTML)", bootstyle="link", command=self.save_plot_html).grid(column=2, row=row, pady=8, sticky=W)
        tb.Button(self.frame, text="Generate Test Data", bootstyle="info", command=self.generate_test_data).grid(column=3, row=row, pady=8, sticky=W)
        row += 1

        # output text area
        self.output_text = tb.Text(self.frame, height=12, wrap='word')
        self.output_text.grid(column=0, row=row, columnspan=5, sticky="nsew", pady=(6,0))
        self.frame.rowconfigure(row, weight=1)

    def browse_file(self):
        fp = filedialog.askopenfilename(filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if fp:
            self.filepath_var.set(fp)

    def run_analysis(self):
        fp = self.filepath_var.get().strip()
        if not fp or not os.path.exists(fp):
            messagebox.showerror("File not found", "Please select a valid CSV file.")
            return
        try:
            self.df = load_csv(fp)
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load CSV: {e}")
            return

        age = int(self.age_var.get())
        resting_hr = float(self.rest_hr_var.get())
        max_hr = 220 - age
        hr_reserve = max_hr - resting_hr

        inc_threshold = float(self.inc_threshold_var.get())
        dec_threshold = float(self.dec_threshold_var.get())
        min_hill = float(self.min_hill_sec_var.get())
        sustain = float(self.sustain_seconds_var.get())
        allowed_gap = float(self.allowed_gap_var.get())
        derivative_window = float(self.derivative_window_var.get())

        periods, derived_df = detect_exercise_periods(self.df,
                                                      resting_hr=resting_hr,
                                                      max_hr=max_hr,
                                                      inc_threshold=inc_threshold,
                                                      dec_threshold=dec_threshold,
                                                      min_hill_duration=min_hill,
                                                      sustain_seconds=sustain,
                                                      allowed_gap=allowed_gap,
                                                      derivative_window=derivative_window)
        self.detected_periods = periods
        self.derived_df = derived_df

        metrics_list = []
        for p in periods:
            m = compute_hra_hrr_for_period(self.df, p, hr_reserve)
            metrics_list.append(m)

        # create result DataFrame
        metrics_df = pd.DataFrame(metrics_list)
        self.metrics = metrics_df

        # display summary
        self.show_summary(age, resting_hr, max_hr, hr_reserve, metrics_df)

        # plot
        fig = self.make_plot(self.df, derived_df, periods, metrics_df)
        # open in browser with scrollZoom enabled
        plot_config = {'scrollZoom': True}
        pio.show(fig, config=plot_config)

    def show_summary(self, age, resting_hr, max_hr, hr_reserve, metrics_df):
        out_lines = []
        out_lines.append(f"Analysis run: {datetime.datetime.now().isoformat()}")
        out_lines.append(f"Age: {age}  Resting HR: {resting_hr}  MaxHR (220-age): {max_hr:.1f}  HRreserve: {hr_reserve:.1f}")
        if metrics_df is None or len(metrics_df) == 0:
            out_lines.append("No exercise periods detected with current parameters.")
        else:
            out_lines.append(f"Detected {len(metrics_df)} exercise period(s).")
            # show per-period
            for i, row in metrics_df.iterrows():
                hra_p = row['hra_peak_bpm_per_s']
                hra90 = row['hra_90_bpm_per_s']
                hra50 = row.get('hra_50_bpm_per_s', None)
                hrr30 = row.get('hrr_30_bpm', None)
                hrr60 = row['hrr_60_bpm']
                hrr120 = row.get('hrr_120_bpm', None)
                duration = row.get('exercise_duration_sec', None)
                hr_util = row.get('hr_reserve_utilization_pct', None)

                out_lines.append(f"Period {i+1}:")
                out_lines.append(f"  HRA_peak = {hra_p:.3f} bpm/s")
                if hra90 is not None:
                    out_lines.append(f"  HRA_90 = {hra90:.3f} bpm/s")
                if hra50 is not None:
                    out_lines.append(f"  HRA_50 = {hra50:.3f} bpm/s")
                if hrr30 is not None:
                    out_lines.append(f"  HRR_30 = {hrr30:.1f} bpm")
                if hrr60 is not None:
                    out_lines.append(f"  HRR_60 = {hrr60:.1f} bpm")
                if hrr120 is not None:
                    out_lines.append(f"  HRR_120 = {hrr120:.1f} bpm")
                if duration is not None:
                    out_lines.append(f"  Duration = {duration:.1f}s")
                if hr_util is not None:
                    out_lines.append(f"  HR Reserve Used = {hr_util:.1f}%")
                out_lines.append("")

            # summary stats
            numeric = metrics_df.select_dtypes(include=[np.number])
            out_lines.append("Summary Statistics:")
            key_metrics = ['hra_peak_bpm_per_s', 'hra_90_bpm_per_s', 'hra_50_bpm_per_s',
                          'hrr_30_bpm', 'hrr_60_bpm', 'hrr_120_bpm', 'exercise_duration_sec',
                          'hr_reserve_utilization_pct']
            for col in key_metrics:
                if col in numeric:
                    vals = numeric[col].dropna()
                    if len(vals) > 0:
                        out_lines.append(f" {col}: mean={vals.mean():.3f}, max={vals.max():.3f}, min={vals.min():.3f}")
        self.output_text.delete('1.0', tk.END)
        self.output_text.insert(tk.END, "\n".join(out_lines))

    def make_plot(self, df, derived_df, periods, metrics_df):
        times = df['time_seconds']
        hr = df['bpm']

        # Create figure with subplots for secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Convert time to datetime for consistent formatting with bpm_analysis.py
        time_axis_dt = pd.to_datetime([seconds_to_datetime(t) for t in times])

        # Add main BPM trace with styling matching bpm_analysis.py
        fig.add_trace(go.Scatter(
            x=time_axis_dt,
            y=hr,
            mode='lines',
            name='BPM',
            line=dict(color="#4a4a4a", width=3)
        ), secondary_y=True)

        # Add slopes as secondary axis with matching styling
        if 'dbpm_dt' in derived_df.columns:
            slope_times_dt = pd.to_datetime([seconds_to_datetime(t) for t in derived_df['time_seconds']])
            fig.add_trace(go.Scatter(
                x=slope_times_dt,
                y=derived_df['dbpm_dt'],
                mode='lines',
                name='dbpm/dt',
                line=dict(color='cyan', width=2, dash='dot')
            ), secondary_y=False)

        # Add shaded windows and markers with improved styling
        for i, p in enumerate(periods):
            s = p['start_time']
            e = p['end_time']
            pk = p['peak_time']

            # Convert times to datetime for consistency
            s_dt = seconds_to_datetime(s)
            e_dt = seconds_to_datetime(e)
            pk_dt = seconds_to_datetime(pk)

            # Add shaded rectangle with better styling
            fig.add_vrect(x0=s_dt, x1=e_dt, fillcolor="rgba(255, 182, 193, 0.3)", opacity=0.3, layer="below", line_width=0)

            # Add markers with improved styling and hover templates
            fig.add_trace(go.Scatter(
                x=[s_dt],
                y=[p['start_hr']],
                mode='markers',
                marker=dict(symbol='circle', size=8, color='#2ca02c'),
                name=f'P{i+1} Start',
                hovertemplate=f"<b>Period {i+1} Start</b><br>Time: {s:.1f}s<br>HR: {p['start_hr']:.1f} BPM<extra></extra>"
            ), secondary_y=True)

            fig.add_trace(go.Scatter(
                x=[pk_dt],
                y=[p['peak_hr']],
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='#e36f6f'),
                name=f'P{i+1} Peak',
                hovertemplate=f"<b>Period {i+1} Peak</b><br>Time: {pk:.1f}s<br>HR: {p['peak_hr']:.1f} BPM<extra></extra>"
            ), secondary_y=True)

            fig.add_trace(go.Scatter(
                x=[e_dt],
                y=[p['end_hr']],
                mode='markers',
                marker=dict(symbol='x', size=10, color='#ff7f0e'),
                name=f'P{i+1} End',
                hovertemplate=f"<b>Period {i+1} End</b><br>Time: {e:.1f}s<br>HR: {p['end_hr']:.1f} BPM<extra></extra>"
            ), secondary_y=True)

            # Add annotations with improved styling
            m = None
            if metrics_df is not None and i < len(metrics_df):
                m = metrics_df.iloc[i]
            if m is not None:
                hra_peak = m['hra_peak_bpm_per_s']
                hrr = m['hrr_60_bpm']
                annotation = f"<b>Period {i+1}</b><br>HRA_peak: {hra_peak:.3f} bpm/s<br>HRR60: {('NA' if pd.isna(hrr) else f'{hrr:.1f} bpm')}"
                fig.add_annotation(
                    x=pk_dt,
                    y=p['peak_hr'],
                    text=annotation,
                    showarrow=True,
                    arrowhead=1,
                    yshift=10,
                    font=dict(color="#e36f6f"),
                    yref="y2"
                )

        # Configure layout to match bpm_analysis.py style
        plot_title = f"HR Reactivity Analysis - {timestamp_str()}"

        fig.update_layout(
            template="plotly_dark",
            title_text=plot_title,
            dragmode='pan',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=140, b=100),
            hovermode='x unified'
        )

        # Configure X-axis with datetime formatting like bpm_analysis.py
        tick_positions_sec = np.linspace(0, times.iloc[-1], num=10)
        tickvals = [seconds_to_datetime(s) for s in tick_positions_sec]
        ticktext = [f"{int(s // 60):02d}:{int(s % 60):02d} ({s:.2f})" for s in tick_positions_sec]

        fig.update_xaxes(
            title_text="Time",
            tickvals=tickvals,
            ticktext=ticktext,
            hoverformat='%M:%S.%L'
        )

        # Configure Y-axes to match bpm_analysis.py
        fig.update_yaxes(title_text="BPM / HRV", secondary_y=True, range=DEFAULT_PLOT_Y_RANGE_BPM)
        fig.update_yaxes(title_text="dbpm/dt", secondary_y=False, range=DEFAULT_PLOT_Y_RANGE_DERIVATIVE)

        # Add min/max BPM annotations similar to bpm_analysis.py
        if len(hr) > 0:
            max_bpm_val = hr.max()
            min_bpm_val = hr.min()
            max_bpm_time = times.iloc[hr.idxmax()]
            min_bpm_time = times.iloc[hr.idxmin()]

            # Convert to datetime for consistency
            max_bpm_time_dt = seconds_to_datetime(max_bpm_time)
            min_bpm_time_dt = seconds_to_datetime(min_bpm_time)

            # Add annotation for the maximum BPM
            fig.add_annotation(
                x=max_bpm_time_dt,
                y=max_bpm_val,
                text=f"Max: {max_bpm_val:.1f} BPM",
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=-40,
                font=dict(color="#e36f6f"),
                yref="y2"
            )

            # Add annotation for the minimum BPM
            fig.add_annotation(
                x=min_bpm_time_dt,
                y=min_bpm_val,
                text=f"Min: {min_bpm_val:.1f} BPM",
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=40,
                font=dict(color="#a3d194"),
                yref="y2"
            )

        # Add summary annotation box similar to bpm_analysis.py
        if metrics_df is not None and not metrics_df.empty:
            annotation_text = "<b>HR Reactivity Analysis Summary</b><br>"
            annotation_text += f"Detected {len(periods)} exercise period(s)<br>"

            # Add summary statistics
            if 'hra_peak_bpm_per_s' in metrics_df.columns:
                hra_vals = metrics_df['hra_peak_bpm_per_s'].dropna()
                if len(hra_vals) > 0:
                    annotation_text += f"Avg HRA: {hra_vals.mean():.3f} bpm/s<br>"
                    annotation_text += f"Max HRA: {hra_vals.max():.3f} bpm/s<br>"

            if 'hrr_60_bpm' in metrics_df.columns:
                hrr_vals = metrics_df['hrr_60_bpm'].dropna()
                if len(hrr_vals) > 0:
                    annotation_text += f"Avg HRR: {hrr_vals.mean():.1f} BPM<br>"
                    annotation_text += f"Max HRR: {hrr_vals.max():.1f} BPM<br>"

            if 'hr_reserve_utilization_pct' in metrics_df.columns:
                util_vals = metrics_df['hr_reserve_utilization_pct'].dropna()
                if len(util_vals) > 0:
                    annotation_text += f"Avg HR Reserve Used: {util_vals.mean():.1f}%<br>"
                    annotation_text += f"Max HR Reserve Used: {util_vals.max():.1f}%"

            fig.add_annotation(
                text=annotation_text,
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0.02,
                y=0.98,
                bordercolor='black',
                borderwidth=1,
                bgcolor='rgba(255, 253, 231, 0.4)'
            )

        return fig

    def export_results(self):
        if self.metrics is None or self.detected_periods is None:
            messagebox.showinfo("No results", "Please run analysis first.")
            return
        # build a combined DataFrame
        per = []
        for i, p in enumerate(self.detected_periods):
            m = self.metrics.iloc[i] if i < len(self.metrics) else {}
            row = dict(
                period_index=i+1,
                start_time=p['start_time'],
                peak_time=p['peak_time'],
                end_time=p['end_time'],
                start_hr=p['start_hr'],
                peak_hr=p['peak_hr'],
                end_hr=p['end_hr'],
            )
            # Include all available metrics
            metric_keys = ['hra_peak_bpm_per_s', 'hra_90_bpm_per_s', 'hra_50_bpm_per_s',
                          'hrr_30_bpm', 'hrr_60_bpm', 'hrr_120_bpm', 'exercise_duration_sec',
                          'peak_duration_sec', 'recovery_duration_sec', 'hr_reserve_used_bpm',
                          'hr_reserve_utilization_pct']
            row.update({k: (m[k] if k in m else None) for k in metric_keys})
            per.append(row)
        outdf = pd.DataFrame(per)
        fp = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV','*.csv')])
        if not fp:
            return
        outdf.to_csv(fp, index=False)
        messagebox.showinfo("Exported", f"Results exported to {fp}")

    def save_plot_html(self):
        if self.df is None or self.detected_periods is None:
            messagebox.showinfo("No plot", "Run analysis and open the plot, then save.")
            return
        fig = self.make_plot(self.df, self.derived_df, self.detected_periods, self.metrics)
        fp = filedialog.asksaveasfilename(defaultextension='.html', filetypes=[('HTML','*.html')])
        if not fp:
            return
        # Save with scrollZoom enabled, matching bpm_analysis.py behavior
        plot_config = {'scrollZoom': True, 'toImageButtonOptions': {'filename': 'HR_Reactivity_Analysis', 'format': 'png', 'scale': 2}}
        pio.write_html(fig, file=fp, config=plot_config, auto_open=False)
        messagebox.showinfo("Saved", f"Plot saved to {fp}")

    def generate_test_data(self):
        """Generate sample HR data for testing the analysis."""
        try:
            # Get current age and resting HR from GUI
            age = int(self.age_var.get())
            resting_hr = float(self.rest_hr_var.get())
            max_hr = 220 - age

            # Generate test data
            df = create_sample_hr_data(resting_hr=resting_hr, max_hr=max_hr)

            # Save to file
            fp = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV','*.csv')],
                                            initialvalue="sample_hr_data.csv")
            if fp:
                df.to_csv(fp, index=False)
                self.filepath_var.set(fp)
                messagebox.showinfo("Test Data Generated", f"Sample HR data saved to {fp}\nYou can now run the analysis on this test data.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate test data: {e}")

# ---------- Test Data Generation ----------

def create_sample_hr_data(duration_minutes=DEFAULT_TEST_DURATION_MINUTES, resting_hr=70, max_hr=180, sample_rate=DEFAULT_TEST_SAMPLE_RATE):
    """
    Create sample heart rate data for testing the analysis.
    Simulates a typical exercise session with warm-up, exercise, and recovery.
    """
    duration_seconds = duration_minutes * 60
    times = np.arange(0, duration_seconds, 1/sample_rate)

    # Create a realistic HR profile
    hr_data = np.zeros_like(times)

    # Resting phase (first 2 minutes)
    rest_duration = 120
    rest_mask = times <= rest_duration
    hr_data[rest_mask] = resting_hr + np.random.normal(0, 2, np.sum(rest_mask))

    # Warm-up phase (2-4 minutes)
    warmup_start = 120
    warmup_end = 240
    warmup_mask = (times > warmup_start) & (times <= warmup_end)
    warmup_progress = (times[warmup_mask] - warmup_start) / (warmup_end - warmup_start)
    target_hr_warmup = resting_hr + 0.3 * (max_hr - resting_hr)
    hr_data[warmup_mask] = resting_hr + warmup_progress * (target_hr_warmup - resting_hr) + np.random.normal(0, 3, np.sum(warmup_mask))

    # Exercise phase (4-8 minutes) - steep climb
    exercise_start = 240
    exercise_peak = 480
    exercise_mask = (times > exercise_start) & (times <= exercise_peak)
    exercise_progress = (times[exercise_mask] - exercise_start) / (exercise_peak - exercise_start)
    # Create a steep climb with some variation
    hr_data[exercise_mask] = target_hr_warmup + exercise_progress * (max_hr - target_hr_warmup) + np.random.normal(0, 5, np.sum(exercise_mask))

    # Peak exercise (8-9 minutes) - sustained high HR
    peak_start = 480
    peak_end = 540
    peak_mask = (times > peak_start) & (times <= peak_end)
    hr_data[peak_mask] = max_hr + np.random.normal(0, 3, np.sum(peak_mask))

    # Recovery phase (9-10 minutes) - steep decline
    recovery_start = 540
    recovery_end = 600
    recovery_mask = (times > recovery_start) & (times <= recovery_end)
    recovery_progress = (times[recovery_mask] - recovery_start) / (recovery_end - recovery_start)
    recovery_target = resting_hr + 0.2 * (max_hr - resting_hr)
    hr_data[recovery_mask] = max_hr - recovery_progress * (max_hr - recovery_target) + np.random.normal(0, 4, np.sum(recovery_mask))

    # Smooth the data slightly
    from scipy.ndimage import gaussian_filter1d
    hr_data = gaussian_filter1d(hr_data, sigma=2)

    # Create DataFrame
    df = pd.DataFrame({
        'time_seconds': times,
        'bpm': hr_data
    })

    return df

def create_test_csv(filename="sample_hr_data.csv"):
    """Create a sample CSV file for testing."""
    df = create_sample_hr_data()
    df.to_csv(filename, index=False)
    print(f"Sample HR data saved to {filename}")
    return filename

# ---------- Run GUI ----------

def main():
    root = tb.Window(themename="superhero")
    # The app wires itself into `root`'s widget tree, so `root` keeps it alive.
    HRAnalyzerApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
