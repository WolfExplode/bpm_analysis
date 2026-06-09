# Trapezoid Artifact Detection Archive
Archived from active codebase before removal.
## `hrv.py` — `detect_trapezoid_discontinuities(...)` (archived)
```python
def detect_trapezoid_discontinuities(smoothed_bpm: pd.Series, bpm_times_sec: np.ndarray, params: Dict) -> List[Dict]:
    """
    Detects trapezoid-shaped discontinuities in the average BPM series that are
    characteristic of a brief extra-beat artifact:
      - A very fast rise
      - A sustained (possibly slightly sloped) plateau
      - A very fast fall that returns to baseline
    Detection only -- results are not yet used to correct labels. See
    Documentation.md "Trapezoid Artifacts" for design rationale and future plans.
    """
    if (
        smoothed_bpm is None
        or smoothed_bpm.empty
        or bpm_times_sec is None
        or len(bpm_times_sec) != len(smoothed_bpm)
    ):
        return []
    # Build working DataFrame equivalent to the CSV used in detectTrapezoid.py
    df = pd.DataFrame(
        {
            "Time (s)": bpm_times_sec.astype(float),
            "Average BPM": smoothed_bpm.to_numpy(dtype=float),
        }
    ).dropna(subset=["Time (s)", "Average BPM"])
    if len(df) < 4:
        return []
    # Calculate differences and instantaneous rate of BPM change
    df["Δt"] = df["Time (s)"].diff()
    df["ΔBPM"] = df["Average BPM"].diff()
    df["Rate"] = df["ΔBPM"] / df["Δt"]
    # --- CONFIGURATION (now driven by params, defaults mirror detectTrapezoid.py) ---
    RATE_THRESHOLD = params.get("trapezoid_rate_threshold", 7.0)                 # BPM/s
    MAX_EDGE_DURATION = params.get("trapezoid_max_edge_duration_sec", 1.5)       # seconds
    MIN_PLATEAU_DURATION = params.get("trapezoid_min_plateau_duration_sec", 1.5) # seconds
    MAX_PLATEAU_DURATION = params.get("trapezoid_max_plateau_duration_sec", 15.0)# seconds
    BASELINE_TOLERANCE = params.get("trapezoid_baseline_tolerance_bpm", 5.0)     # BPM
    MIN_JUMP = params.get("trapezoid_min_jump_bpm", 6.0)                         # BPM
    MIN_FALL_DELTA = params.get("trapezoid_min_fall_delta_bpm", 5.0)             # BPM
    # Step 1: Identify edge intervals (second point of each edge)
    df["is_rise"] = (df["Rate"] > RATE_THRESHOLD) & (df["Δt"] < MAX_EDGE_DURATION)
    df["is_fall"] = (df["Rate"] < -RATE_THRESHOLD) & (df["Δt"] < MAX_EDGE_DURATION)
    rise_indices = df[df["is_rise"]].index.tolist()
    fall_indices = df[df["is_fall"]].index.tolist()
    trapezoids: List[Dict] = []
    # Step 2: Match edges into trapezoids
    for rise_idx in rise_indices:
        # Need at least one sample before the rise edge for t1 / baseline
        if rise_idx <= 0:
            continue
        rise_time = float(df.loc[rise_idx, "Time (s)"])
        for fall_idx in list(fall_indices):
            # Need at least one sample before fall edge for t3
            if fall_idx <= 0:
                continue
            fall_time = float(df.loc[fall_idx, "Time (s)"])
            # Timing constraints: plateau must be long enough but not absurdly long
            plateau_duration = fall_time - rise_time
            if not (MIN_PLATEAU_DURATION <= plateau_duration <= MAX_PLATEAU_DURATION):
                continue
            # --- Validate plateau (region strictly between rise and fall) ---
            plateau_mask = (df["Time (s)"] > rise_time) & (df["Time (s)"] < fall_time)
            plateau_df = df[plateau_mask]
            if plateau_df.empty:
                continue
            # Allow sloped plateaus: median absolute rate should be modest
            if plateau_df["Rate"].abs().median() > RATE_THRESHOLD / 3.0:
                continue
            # --- Validate baseline recovery ---
            # Baseline before: up to 3 points before the rise edge
            before_start_idx = max(0, rise_idx - 3)
            before_end_idx = rise_idx - 1
            if before_end_idx < before_start_idx:
                continue
            baseline_before = float(
                df.loc[before_start_idx:before_end_idx, "Average BPM"].mean()
            )
            # Baseline after: up to 3 points starting at fall edge
            after_end_idx = min(fall_idx + 2, df.index[-1])
            baseline_after = float(
                df.loc[fall_idx:after_end_idx, "Average BPM"].mean()
            )
            if abs(baseline_after - baseline_before) > BASELINE_TOLERANCE:
                continue
            # --- Calculate the four key timestamps ---
            t1 = float(df.loc[rise_idx - 1, "Time (s)"])
            t2 = rise_time
            t3 = float(df.loc[fall_idx - 1, "Time (s)"])
            t4 = fall_time
            # Validate edge intervals are brief
            if (t2 - t1) > MAX_EDGE_DURATION or (t4 - t3) > MAX_EDGE_DURATION:
                continue
            # Enforce a minimum BPM change across the fall edge itself.
            # If the fall barely changes BPM, don't treat it as a trapezoid artifact.
            fall_start_bpm = float(df.loc[fall_idx - 1, "Average BPM"])
            fall_end_bpm = float(df.loc[fall_idx, "Average BPM"])
            if abs(fall_start_bpm - fall_end_bpm) < MIN_FALL_DELTA:
                continue
            # Calculate jump from baseline to plateau median
            plateau_median = float(plateau_df["Average BPM"].median())
            jump_size = plateau_median - baseline_before
            if jump_size < MIN_JUMP:
                continue
            plateau_slope = float(
                plateau_df["Average BPM"].iloc[-1] - plateau_df["Average BPM"].iloc[0]
            )
            # Store both timestamps and BPM values for debugging/plotting
            trap = {
                "t_start_rise": t1,
                "t_end_rise": t2,
                "t_start_fall": t3,
                "t_end_fall": t4,
                "bpm_start_rise": float(df.loc[rise_idx - 1, "Average BPM"]),
                "bpm_end_rise": float(df.loc[rise_idx, "Average BPM"]),
                "bpm_start_fall": fall_start_bpm,
                "bpm_end_fall": fall_end_bpm,
                "baseline_before": baseline_before,
                "plateau_median": plateau_median,
                "plateau_slope": plateau_slope,
                "jump_size": jump_size,
                "baseline_after": baseline_after,
                "baseline_diff": baseline_after - baseline_before,
                "plateau_duration": plateau_duration,
                "plateau_points": int(len(plateau_df)),
            }
            trapezoids.append(trap)
            # Remove used fall index so it cannot be reused by another rise
            fall_indices.remove(fall_idx)
            break
    if trapezoids:
        logging.info(f"Detected {len(trapezoids)} trapezoid HR artifacts (sudden plateau jumps):")
        for i, trap in enumerate(trapezoids, 1):
            logging.info(
                "  Trapezoid #%d: "
                "Rise %.3fs (%.1f BPM) → %.3fs (%.1f BPM); "
                "Plateau %.3fs → %.3fs (Δ%.3fs, %d pts); "
                "Fall %.3fs (%.1f BPM) → %.3fs (%.1f BPM)",
                i,
                trap["t_start_rise"],
                trap["bpm_start_rise"],
                trap["t_end_rise"],
                trap["bpm_end_rise"],
                trap["t_end_rise"],
                trap["t_start_fall"],
                trap["plateau_duration"],
                trap["plateau_points"],
                trap["t_start_fall"],
                trap["bpm_start_fall"],
                trap["t_end_fall"],
                trap["bpm_end_fall"],
            )
    else:
        logging.info("No trapezoid-like HR artifacts detected in average BPM series.")
    return trapezoids
```
## `plotting.py` — `_add_trapezoid_shapes(...)` (archived)
```python
def _add_trapezoid_shapes(self, trapezoids: Optional[List[Dict]]):
    """Draws trapezoid outlines and markers for detected HR artifacts."""
    if not trapezoids:
        return
    for idx, trap in enumerate(trapezoids, start=1):
        event_sequence = [
            ("Start of rise", trap["t_start_rise"], trap["bpm_start_rise"]),
            ("End of rise", trap["t_end_rise"], trap["bpm_end_rise"]),
            ("Start of fall", trap["t_start_fall"], trap["bpm_start_fall"]),
            ("End of fall", trap["t_end_fall"], trap["bpm_end_fall"]),
        ]
        x_times = [seconds_to_datetime(t) for _, t, _ in event_sequence]
        y_values = [bpm for _, _, bpm in event_sequence]
        customdata = [
            f"<b>{label}</b><br>{t:.3f}s<br>{bpm:.1f} BPM" for label, t, bpm in event_sequence
        ]
        self.fig.add_trace(
            go.Scatter(
                x=x_times,
                y=y_values,
                mode="lines+markers",
                name="Trapezoid Artifacts",
                marker=dict(symbol="circle-open", size=8, color="#ffd166"),
                line=dict(color="#ffd166", width=2),
                customdata=customdata,
                hovertemplate="%{customdata}<extra></extra>",
                legendgroup="Trapezoid Artifacts",
                showlegend=(idx == 1),
            ),
            secondary_y=True,
        )
```