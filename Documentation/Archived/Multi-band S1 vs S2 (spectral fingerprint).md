# Archived: Multi-band S1 vs S2 (spectral fingerprint)
This document archives the Multi-band S1 vs S2 logic removed from the codebase so it can be reimplemented later. The feature used two bandpass envelopes (S1-band and S2-band) and Gaussian-weighted energy at each peak to adjust S1–S2 pairing confidence.

---
## 1. Config (`config.py`)
```python
    # --- Multi-band S1 vs S2 (spectral fingerprint) ---
    "enable_multiband_s1_s2": True,      # Use S1-band vs S2-band energy to adjust pairing confidence.
    "s1_band_low_hz": 25.0,             # S1 typical range 20-60 Hz.
    "s1_band_high_hz": 60.0,
    "s2_band_low_hz": 200.0,             # S2 typical range 60-200 Hz.
    "s2_band_high_hz": 290.0,
    "multiband_boost_max": 0.1,         # Max confidence boost when band energies support S1-S2.
    "multiband_penalty_max": 0.1,      # Max confidence penalty when bands suggest wrong order.
    "multiband_peak_window_ms": 130.0,   # Time window (ms) centered on each peak; covers whole beat. Converted to samples using sample rate.
    "multiband_gaussian_sigma_ms": 25.0, # Gaussian sigma (ms) for weighting; typically window/4 so weight falls off by edges. Used for Gaussian-weighted sum of band energy.
```
---
## 2. Audio preprocessing: band envelope (`audio_preprocessing.py`)
Helper to compute one band envelope:
```python
def _compute_band_envelope(
    audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float, smooth_window: int
) -> np.ndarray:
    """Bandpass filter, Hilbert envelope, and rolling mean for one frequency band."""
    nyquist = 0.5 * sample_rate
    low = max(1e-6, low_hz / nyquist)
    high = min(1.0 - 1e-6, high_hz / nyquist)
    if low >= high:
        return np.zeros_like(audio, dtype=np.float64)
    sos = butter(2, [low, high], btype="band", output="sos")
    filtered = sosfiltfilt(sos, audio)
    analytic = hilbert(filtered)
    envelope_raw = np.abs(analytic).astype(np.float64)
    envelope = pd.Series(envelope_raw).rolling(
        window=smooth_window, min_periods=1, center=True
    ).mean().values
    return envelope
```
In `preprocess_audio`, after building `audio_envelope`, the multiband block (return value was `audio_envelope, new_sample_rate, band_envelopes, noise_floor, trough_indices`):
```python
    # Multi-band S1 vs S2: separate envelopes for S1-band (e.g. 20-60 Hz) and S2-band (e.g. 60-200 Hz).
    band_envelopes = None
    if params.get("enable_multiband_s1_s2", True):
        s1_low = float(params.get("s1_band_low_hz", 20.0))
        s1_high = float(params.get("s1_band_high_hz", 60.0))
        s2_low = float(params.get("s2_band_low_hz", 60.0))
        s2_high = float(params.get("s2_band_high_hz", 200.0))
        band_envelopes = {
            "s1_band": _compute_band_envelope(
                audio_filtered, new_sample_rate, s1_low, s1_high, smooth_window
            ),
            "s2_band": _compute_band_envelope(
                audio_filtered, new_sample_rate, s2_low, s2_high, smooth_window
            ),
        }
    noise_floor, trough_indices = _calculate_dynamic_noise_floor(...)
    return audio_envelope, new_sample_rate, band_envelopes, noise_floor, trough_indices
```
---
## 3. Confidence engine (`confidence_engine.py`)
**Constructor:** `PairingEngine.__init__` took `band_envelopes: Optional[Dict[str, np.ndarray]] = None` and set `self.band_envelopes = band_envelopes`.
**Static helper:**
```python
    @staticmethod
    def _gaussian_weighted_energy(
        band_arr: np.ndarray, peak_idx: int, half: int, sigma_samp: float, n: int
    ) -> float:
        """Gaussian-windowed energy of a band envelope centred on a peak sample."""
        lo = max(0, peak_idx - half)
        hi = min(n, peak_idx + half + 1)
        if hi <= lo:
            return 0.0
        offsets = np.arange(lo, hi, dtype=np.float64) - float(peak_idx)
        weights = np.exp(-0.5 * (offsets / sigma_samp) ** 2)
        weights /= weights.sum()
        return float(np.sum(weights * band_arr[lo:hi]))
```
**In `attempt_pair`, after contractility/prominence adjustment:**
```python
        # --- Multi-band S1 vs S2: spectral fingerprint adjustment ---
        if self.params.get("enable_multiband_s1_s2", True) and self.band_envelopes is not None:
            s1_band = self.band_envelopes.get("s1_band")
            s2_band = self.band_envelopes.get("s2_band")
            if s1_band is not None and s2_band is not None:
                n = len(self.audio_envelope)
                window_ms = float(self.params.get("multiband_peak_window_ms", 100.0))
                sigma_ms = float(self.params.get("multiband_gaussian_sigma_ms", 25.0))
                window_samples = max(1, int(round(window_ms * 0.001 * self.sample_rate)))
                if window_samples % 2 == 0:
                    window_samples += 1
                half = min((window_samples - 1) // 2, n // 2)
                sigma_samp = max(1e-6, sigma_ms * 0.001 * self.sample_rate)
                e_s1_at_first  = self._gaussian_weighted_energy(s1_band, s1_candidate_idx, half, sigma_samp, n)
                e_s2_at_first  = self._gaussian_weighted_energy(s2_band, s1_candidate_idx, half, sigma_samp, n)
                e_s1_at_second = self._gaussian_weighted_energy(s1_band, s2_candidate_idx, half, sigma_samp, n)
                e_s2_at_second = self._gaussian_weighted_energy(s2_band, s2_candidate_idx, half, sigma_samp, n)
                # For correct S1-S2: first peak should have more S1-band, second more S2-band.
                # consistency > 1 means bands support this pair; < 1 means bands suggest wrong order.
                consistency = (e_s1_at_first * e_s2_at_second) / (e_s2_at_first * e_s1_at_second + 1e-9)
                mb_boost_max   = self.params.get("multiband_boost_max", 0.12)
                mb_penalty_max = self.params.get("multiband_penalty_max", 0.15)
                if consistency >= 1.2:
                    delta = min(mb_boost_max, (consistency - 1.0) * 0.5)
                    confidence = min(1.0, confidence + delta)
                    steps.append({"step": "Multiband", "detail": f"bands support pair (ratio {consistency:.2f}) → +{delta:.2f}", "result": confidence})
                elif consistency <= 0.85:
                    delta = min(mb_penalty_max, (1.0 - consistency) * 0.5)
                    confidence = max(0.0, confidence - delta)
                    steps.append({"step": "Multiband", "detail": f"bands oppose pair (ratio {consistency:.2f}) → -{delta:.2f}", "result": confidence})
                else:
                    steps.append({"step": "Multiband", "detail": f"bands neutral (ratio {consistency:.2f}) → no change", "result": confidence})
```
---
## 4. Pipeline (`pipeline.py`)
- `_run_pass1` had parameter `band_envelopes: Optional[Dict[str, np.ndarray]] = None` and passed it into `PeakClassifier(..., band_envelopes)`.
- Main flow: `audio_envelope, sample_rate, band_envelopes, noise_floor, troughs = preprocess_audio(...)` then passed `band_envelopes` into `_run_pass1` and into `PeakClassifier(..., band_envelopes)`.
- After `classifier.classify_peaks()`:
```python
    # Attach band envelopes to analysis_data for plotting (S1/S2 band debug traces)
    if band_envelopes is not None:
        analysis_data["s1_band"] = band_envelopes.get("s1_band")
        analysis_data["s2_band"] = band_envelopes.get("s2_band")
```
---
## 5. Classifier (`classifier.py`)
`PeakClassifier.__init__` had `band_envelopes: Optional[Dict[str, np.ndarray]] = None` and passed it to `PairingEngine`:
```python
        self.pairing_engine = PairingEngine(
            audio_envelope, sample_rate, params, peak_bpm_time_sec, recovery_end_time_sec,
            band_envelopes=band_envelopes,
        )
```
---
## 6. Plotting (`plotting.py`)
- Import: `from confidence_engine import PairingEngine` (used only for `PairingEngine._gaussian_weighted_energy` in this block).
- In the method that adds envelope + noise floor + S1/S2 band traces (e.g. `_add_envelope_traces` or equivalent), the following block added S1/S2 band energy traces and “proportion at peaks” scatter:
```python
        # S1/S2 band energy (continuous) and proportion at peaks (what the algorithm uses).
        s1_band = analysis_data.get("s1_band")
        s2_band = analysis_data.get("s2_band")
        s1_low = self.params.get("s1_band_low_hz", 20)
        s1_high = self.params.get("s1_band_high_hz", 60)
        s2_low = self.params.get("s2_band_low_hz", 60)
        s2_high = self.params.get("s2_band_high_hz", 200)
        if (
            s1_band is not None
            and s2_band is not None
            and len(s1_band) == len(audio_envelope)
            and len(s2_band) == len(audio_envelope)
        ):
            # Continuous band energy traces (raw envelope in each band; may appear temporally smeared).
            plot_s1_band = s1_band[::factor] if factor > 1 and len(s1_band) >= factor else s1_band
            plot_s2_band = s2_band[::factor] if factor > 1 and len(s2_band) >= factor else s2_band
            self.fig.add_trace(
                go.Scatter(
                    x=plot_time_axis_dt,
                    y=plot_s1_band,
                    name=f"S1 band energy ({s1_low:.0f}-{s1_high:.0f} Hz)",
                    line=dict(color="darkorange", width=1.2),
                    hovertemplate="S1 band: %{y:.4f}<extra></extra>",
                    visible="legendonly",
                ),
                secondary_y=False,
            )
            self.fig.add_trace(
                go.Scatter(
                    x=plot_time_axis_dt,
                    y=plot_s2_band,
                    name=f"S2 band energy ({s2_low:.0f}-{s2_high:.0f} Hz)",
                    line=dict(color="purple", width=1.2),
                    hovertemplate="S2 band: %{y:.4f}<extra></extra>",
                    visible="legendonly",
                ),
                secondary_y=False,
            )
        if (
            s1_band is not None
            and s2_band is not None
            and len(s1_band) == len(audio_envelope)
            and len(s2_band) == len(audio_envelope)
            and all_raw_peaks is not None
            and len(all_raw_peaks) > 0
        ):
            eps = 1e-9
            scale = float(np.max(plot_envelope)) if len(plot_envelope) > 0 else 1.0
            if scale < 1e-9:
                scale = 1.0
            peak_indices = np.asarray(all_raw_peaks)
            in_bounds = (peak_indices >= 0) & (peak_indices < len(s1_band))
            peak_indices = peak_indices[in_bounds]
            if len(peak_indices) > 0:
                # Use same Gaussian-windowed band energy as multiband pairing in confidence_engine.
                n = len(audio_envelope)
                window_ms = float(self.params.get("multiband_peak_window_ms", 100.0))
                sigma_ms = float(self.params.get("multiband_gaussian_sigma_ms", 25.0))
                window_samples = max(1, int(round(window_ms * 0.001 * self.sample_rate)))
                if window_samples % 2 == 0:
                    window_samples += 1
                half = min((window_samples - 1) // 2, n // 2)
                sigma_samp = max(1e-6, sigma_ms * 0.001 * self.sample_rate)
                s1_proportion_w = []
                s2_proportion_w = []
                for peak_idx in peak_indices:
                    e_s1 = PairingEngine._gaussian_weighted_energy(
                        s1_band, int(peak_idx), half, sigma_samp, n
                    )
                    e_s2 = PairingEngine._gaussian_weighted_energy(
                        s2_band, int(peak_idx), half, sigma_samp, n
                    )
                    total_w = e_s1 + e_s2 + eps
                    s1_proportion_w.append(e_s1 / total_w)
                    s2_proportion_w.append(e_s2 / total_w)
                s1_proportion_w = np.array(s1_proportion_w)
                s2_proportion_w = np.array(s2_proportion_w)
                s1_at_peaks = s1_proportion_w * scale
                s2_at_peaks = s2_proportion_w * scale
                peak_times_sec = peak_indices.astype(float) / self.sample_rate
                peak_times_dt = pd.to_datetime([seconds_to_datetime(float(t)) for t in peak_times_sec])
                self.fig.add_trace(
                    go.Scatter(
                        x=peak_times_dt,
                        y=s1_at_peaks,
                        mode="markers",
                        name=f"S1 proportion at peaks ({s1_low:.0f}-{s1_high:.0f} Hz)",
                        marker=dict(color="darkorange", size=6, symbol="triangle-up"),
                        hovertemplate="S1 proportion: %{customdata:.3f}<extra></extra>",
                        customdata=s1_proportion_w,
                        visible="legendonly",
                    ),
                    secondary_y=False,
                )
                self.fig.add_trace(
                    go.Scatter(
                        x=peak_times_dt,
                        y=s2_at_peaks,
                        mode="markers",
                        name=f"S2 proportion at peaks ({s2_low:.0f}-{s2_high:.0f} Hz)",
                        marker=dict(color="purple", size=6, symbol="triangle-down"),
                        hovertemplate="S2 proportion: %{customdata:.3f}<extra></extra>",
                        customdata=s2_proportion_w,
                        visible="legendonly",
                    ),
                    secondary_y=False,
                )
```
---
## Summary
- **Config:** Multiband flags and band/boost/penalty/window/sigma parameters.
- **Preprocessing:** `_compute_band_envelope` and building `band_envelopes` dict; return value included `band_envelopes`.
- **Confidence:** `PairingEngine` stored `band_envelopes`, used `_gaussian_weighted_energy`, and applied multiband boost/penalty in `attempt_pair` via the consistency ratio.
- **Pipeline:** Unpacked `band_envelopes` from `preprocess_audio`, passed it to pass1 and classifier, and wrote `s1_band`/`s2_band` into `analysis_data` when present.
- **Classifier:** Passed `band_envelopes` into `PairingEngine`.
- **Plotting:** S1/S2 band continuous traces and Gaussian-weighted proportion-at-peaks scatter using `PairingEngine._gaussian_weighted_energy`.
To reimplement: restore config keys, restore `_compute_band_envelope` and band_envelope construction in preprocessing (and return `band_envelopes` again), restore `band_envelopes` and multiband step in `PairingEngine`, restore parameter threading in pipeline and classifier, and restore the plotting block (and `PairingEngine` import if needed).
