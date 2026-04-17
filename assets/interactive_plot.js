// assets/interactive_plot.js
// JavaScript logic for the interactive BPM analysis HTML output.
// This script expects a global configuration object:
//   window.BPM_ANALYZER_CONFIG = {
//       totalDuration: number,
//       spectrogramSources: { original: string, filtered: string },
//       spectrogramAvailable: { original: boolean, filtered: boolean },
//       audioSources: { original: string, filtered: string },
//       audioLabels: { original: string, filtered: string },
//       htmlS1S2HoverOnByDefault: boolean  // optional; default false = hover off until user toggles toolbar
//   };

(function () {
  const cfg = window.BPM_ANALYZER_CONFIG || {};
  const TOTAL_DURATION = cfg.totalDuration || 0;
  const EPOCH = new Date(0);
  const SPECTROGRAM_SOURCES = cfg.spectrogramSources || {};
  const SPECTROGRAM_AVAILABLE = cfg.spectrogramAvailable || {};
  const AUDIO_SOURCES = cfg.audioSources || {};
  const AUDIO_LABELS = cfg.audioLabels || {};
  const ANALYSIS_SUMMARY = typeof cfg.analysisSummary === "string" ? cfg.analysisSummary : "";
  const beatHoverDefaultOn = cfg.htmlS1S2HoverOnByDefault === true;
  const PASS3_SEGMENTS_AFTER = Array.isArray(cfg.pass3SegmentsAfter)
    ? cfg.pass3SegmentsAfter
    : Array.isArray(cfg.pass3Segments)
      ? cfg.pass3Segments
      : [];
  const PASS3_SEGMENTS_BEFORE = Array.isArray(cfg.pass3SegmentsBefore) ? cfg.pass3SegmentsBefore : [];
  const PASS3_SEGMENTS_DEFAULT_VIEW =
    typeof cfg.pass3SegmentsDefaultView === "string" ? cfg.pass3SegmentsDefaultView : "after";
  let pass3SegmentsActive = PASS3_SEGMENTS_DEFAULT_VIEW === "before" ? PASS3_SEGMENTS_BEFORE : PASS3_SEGMENTS_AFTER;

  // DOM Elements
  const audio = document.getElementById("audio-player");
  const playBtn = document.getElementById("play-btn");
  const stopBtn = document.getElementById("stop-btn");
  const syncBtn = document.getElementById("sync-btn");
  const spectrogramBtn = document.getElementById("spectrogram-btn");
  const spectrogramOpacity = document.getElementById("spectrogram-opacity");
  const spectrogramContainer = document.getElementById("spectrogram-container");
  const spectrogramImage = document.getElementById("spectrogram-image");
  const volumeSlider = document.getElementById("volume-slider");
  const currentTimeEl = document.getElementById("current-time");
  const timelineScrubber = document.getElementById("timeline-scrubber");
  const timelineProgress = document.getElementById("timeline-progress");
  const timelinePlayhead = document.getElementById("timeline-playhead");
  const timelineTicks = document.getElementById("timeline-ticks");
  const chartPlayhead = document.getElementById("chart-playhead");
  const chartContainer = document.getElementById("chart-container");
  const audioFileNameEl = document.getElementById("audio-file-name");
  const audioSourceSelect = document.getElementById("audio-source-select");
  const stateStripCanvas = document.getElementById("state-strip-plot");
  const stateStripTooltip = document.getElementById("state-strip-tooltip");
  const axisGridButtons = document.querySelectorAll("[data-grid-axis]");
  const labelTypeSelect = document.getElementById("label-type-select");
  const applyLabelBtn = document.getElementById("apply-label-btn");
  const flipLabelsRightBtn = document.getElementById("flip-labels-right-btn");
  const downloadLabelsBtn = document.getElementById("download-labels-btn");
  const importLabelsBtn = document.getElementById("import-labels-btn");
  const importLabelsInput = document.getElementById("import-labels-input");
  const analysisSummaryBtn = document.getElementById("analysis-summary-btn");
  const analysisSummaryOverlay = document.getElementById("analysis-summary-overlay");
  const analysisSummaryText = document.getElementById("analysis-summary-text");
  const analysisSummaryClose = document.getElementById("analysis-summary-close");
  const pass3StateViewSelect = document.getElementById("pass3-state-view-select");

  const DEFAULT_AUDIO_KEY = "original";
  let currentAudioKey = DEFAULT_AUDIO_KEY;
  if (audioFileNameEl) {
    audioFileNameEl.dataset.defaultName = audioFileNameEl.textContent || "";
  }

  function hasPlaybackAudio() {
    if (!AUDIO_SOURCES || typeof AUDIO_SOURCES !== "object") return false;
    const orig = AUDIO_SOURCES.original;
    const filt = AUDIO_SOURCES.filtered;
    return (
      (typeof orig === "string" && orig.trim() !== "") ||
      (typeof filt === "string" && filt.trim() !== "")
    );
  }

  if (playBtn && !hasPlaybackAudio()) {
    playBtn.title = "No WAV file available for playback";
  }

  let isPlaying = false;
  let isSynced = true;
  let isSpectrogramVisible = false;
  const BEAT_HOVER_TRACES = ["S1 Beats", "S2 Beats", "Noise/Rejected"];
  let beatHoverEnabled = beatHoverDefaultOn;
  let plotlyGraphDiv = null;
  let xAxisRange = null;
  let fullXAxisRange = null; // Store the full x-axis range for spectrogram positioning
  // Editable peak labels (one entry per plotted peak)
  let editablePeaks = [];
  // Overlay Plotly traces for manual labels
  const manualLabelTraceIndices = { S1: null, S2: null, Noise: null };

  /** Get numeric value at index from Plotly/array-like y data (handles _inputArray, bdata, etc.). */
  function getNumericFromArrayLike(yContainer, index) {
    if (!yContainer || typeof index !== "number" || index < 0) return null;
    const tryAt = (src) => {
      if (!src || typeof src.length !== "number" || src.length <= index) return null;
      const v = src[index];
      const num = typeof v === "number" ? v : parseFloat(v);
      return Number.isFinite(num) ? num : null;
    };
    return (
      tryAt(yContainer) ||
      tryAt(yContainer._inputArray) ||
      tryAt(yContainer.bdata) ||
      tryAt(yContainer.data) ||
      tryAt(yContainer.values) ||
      null
    );
  }

  const logAudioSource = () => {
    if (!audio) return;
    console.log("🔊 Audio source path:", audio.src);
    console.log("📁 Expected audio file location relative to HTML:", audio.src);
  };

  const updateSpectrogramSourceForCurrentAudio = () => {
    if (!spectrogramImage) return;
    const src = SPECTROGRAM_SOURCES[currentAudioKey];
    if (src) {
      spectrogramImage.src = src;
    }
  };

  const updateAudioSource = (key, resumePlayback = false) => {
    if (!audio) return;
    const candidateKey = key && AUDIO_SOURCES[key] ? key : DEFAULT_AUDIO_KEY;
    const src = AUDIO_SOURCES[candidateKey];

    if (!src) {
      console.warn("🔇 Audio source unavailable for", key);
      return;
    }

    currentAudioKey = candidateKey;
    audio.src = src;
    if (audioFileNameEl) {
      audioFileNameEl.textContent =
        AUDIO_LABELS[candidateKey] || audioFileNameEl.dataset.defaultName || "";
      audioFileNameEl.title = audioFileNameEl.textContent;
    }
    if (audioSourceSelect) {
      audioSourceSelect.value = candidateKey;
    }
    audio.load();
    logAudioSource();
    console.log(
      "🔁 Switched audio to",
      AUDIO_LABELS[candidateKey] || candidateKey,
      src
    );

    // If spectrogram is visible, update it to match the current audio source
    if (isSpectrogramVisible) {
      if (SPECTROGRAM_AVAILABLE[currentAudioKey] && SPECTROGRAM_SOURCES[currentAudioKey]) {
        updateSpectrogramSourceForCurrentAudio();
        updateSpectrogramPosition();
      } else if (spectrogramImage && spectrogramBtn) {
        // Hide spectrogram if not available for this source
        spectrogramImage.classList.add("hidden");
        spectrogramBtn.classList.remove("active");
        isSpectrogramVisible = false;
        console.warn(
          "No spectrogram available for audio source:",
          currentAudioKey
        );
      }
    }
    if (resumePlayback && isPlaying) {
      audio.play().catch((e) => console.log("Audio play error:", e));
    }
  };

  if (audioSourceSelect) {
    audioSourceSelect.addEventListener("change", (event) => {
      updateAudioSource(event.target.value, isPlaying);
    });
  }

  updateAudioSource(audioSourceSelect ? audioSourceSelect.value : DEFAULT_AUDIO_KEY);

  // Format time as MM:SS.mmm (seconds)
  function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 1000);
    return `${String(mins).padStart(2, "0")}:${String(secs).padStart(
      2,
      "0"
    )}.${String(ms).padStart(3, "0")} (${seconds.toFixed(2)}s)`;
  }

  // Convert seconds to datetime (epoch + seconds)
  function secondsToDatetime(seconds) {
    return new Date(EPOCH.getTime() + seconds * 1000);
  }

  // Get x-axis position for a given time
  function getXPositionForTime(seconds) {
    if (!plotlyGraphDiv || !xAxisRange) return null;

    const datetime = secondsToDatetime(seconds);
    const xMin = new Date(xAxisRange[0]).getTime();
    const xMax = new Date(xAxisRange[1]).getTime();
    const xTime = datetime.getTime();

    const plotArea = plotlyGraphDiv._fullLayout;
    if (!plotArea) return null;

    const xaxis = plotArea.xaxis;
    if (!xaxis) return null;

    const plotLeft = xaxis._offset;
    const plotWidth = xaxis._length;

    const ratio = (xTime - xMin) / (xMax - xMin);
    return plotLeft + ratio * plotWidth;
  }

  // Initialize timeline ticks
  function initTimelineTicks() {
    if (!timelineTicks) return;
    timelineTicks.innerHTML = "";
    const numMajorTicks = 10;
    const numMinorTicks = 50;

    // Major ticks with labels
    for (let i = 0; i <= numMajorTicks; i++) {
      const percent = (i / numMajorTicks) * 100;
      const time = (i / numMajorTicks) * TOTAL_DURATION;

      const tick = document.createElement("div");
      tick.className = "timeline-tick major";
      tick.style.left = percent + "%";
      timelineTicks.appendChild(tick);

      const label = document.createElement("div");
      label.className = "tick-label";
      label.style.left = percent + "%";
      label.textContent = `${Math.floor(time / 60)}:${String(
        Math.floor(time % 60)
      ).padStart(2, "0")}`;
      timelineTicks.appendChild(label);
    }

    // Minor ticks
    for (let i = 0; i < numMinorTicks; i++) {
      if (i % (numMinorTicks / numMajorTicks) === 0) continue;
      const percent = (i / numMinorTicks) * 100;

      const tick = document.createElement("div");
      tick.className = "timeline-tick minor";
      tick.style.left = percent + "%";
      timelineTicks.appendChild(tick);
    }
  }

  let pass3StripRaf = 0;
  function scheduleDrawPass3StateStrip() {
    if (pass3StripRaf) return;
    pass3StripRaf = window.requestAnimationFrame(() => {
      pass3StripRaf = 0;
      drawPass3StateStrip();
    });
  }

  function drawPass3StateStrip() {
    if (!stateStripCanvas || !pass3SegmentsActive || pass3SegmentsActive.length === 0) return;
    if (!plotlyGraphDiv || !plotlyGraphDiv._fullLayout) return;
    const fl = plotlyGraphDiv._fullLayout;
    const xaxis = fl.xaxis;
    const yaxis = fl.yaxis;
    if (!xaxis || !yaxis) return;
    if (!xAxisRange || xAxisRange.length < 2) return;

    // Determine visible x range in ms since epoch.
    const x0ms = new Date(xAxisRange[0]).getTime();
    const x1ms = new Date(xAxisRange[1]).getTime();
    if (!Number.isFinite(x0ms) || !Number.isFinite(x1ms) || x1ms <= x0ms) return;

    // Plot area geometry (px) relative to plotlyGraphDiv.
    const left = Math.floor(xaxis._offset || 0);
    const width = Math.floor(xaxis._length || 0);
    const topPlot = Math.floor(yaxis._offset || 0);
    const heightPlot = Math.floor(yaxis._length || 0);
    if (width <= 1 || heightPlot <= 1) return;

    // Place strip slightly above the x-axis tick labels area, inside plot area.
    const stripHeight = 10;
    const top = topPlot + heightPlot - stripHeight - 1;

    stateStripCanvas.style.left = `${left}px`;
    stateStripCanvas.style.top = `${top}px`;
    stateStripCanvas.style.width = `${width}px`;
    stateStripCanvas.style.height = `${stripHeight}px`;
    stateStripCanvas.width = Math.max(1, width);
    stateStripCanvas.height = Math.max(1, stripHeight);

    const ctx = stateStripCanvas.getContext("2d");
    if (!ctx) return;

    const colors = {
      S1: "#e36f6f",
      systole: "#666666",
      S2: "#f0a030",
      diastole: "#999999",
    };

    ctx.clearRect(0, 0, width, stripHeight);
    // Soft background so strip is visible even over spectrogram.
    ctx.fillStyle = "rgba(0,0,0,0.25)";
    ctx.fillRect(0, 0, width, stripHeight);

    const span = x1ms - x0ms;
    for (const seg of pass3SegmentsActive) {
      if (!seg) continue;
      const startSec = typeof seg.start === "number" ? seg.start : parseFloat(seg.start);
      const endSec = typeof seg.end === "number" ? seg.end : parseFloat(seg.end);
      const state = typeof seg.state === "string" ? seg.state : String(seg.state || "");
      if (!Number.isFinite(startSec) || !Number.isFinite(endSec) || endSec <= startSec) continue;

      const s0 = secondsToDatetime(startSec).getTime();
      const s1 = secondsToDatetime(endSec).getTime();
      if (s1 <= x0ms || s0 >= x1ms) continue;
      const cl0 = Math.max(x0ms, s0);
      const cl1 = Math.min(x1ms, s1);
      const px0 = Math.floor(((cl0 - x0ms) / span) * width);
      const px1 = Math.ceil(((cl1 - x0ms) / span) * width);
      const x = Math.max(0, Math.min(width, px0));
      const xEnd = Math.max(0, Math.min(width, px1));
      if (xEnd <= x) continue;
      ctx.fillStyle = colors[state] || "#555555";
      ctx.fillRect(x, 0, xEnd - x, stripHeight);
    }
  }

  // -------------------------------------------------------------------------
  // State-strip hover tooltip
  // -------------------------------------------------------------------------
  function _segmentAtTime(tSec) {
    if (!pass3SegmentsActive || pass3SegmentsActive.length === 0) return null;
    // Binary search for the segment containing tSec.
    let lo = 0, hi = pass3SegmentsActive.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const seg = pass3SegmentsActive[mid];
      if (tSec < seg.start) { hi = mid - 1; }
      else if (tSec >= seg.end) { lo = mid + 1; }
      else { return seg; }
    }
    return null;
  }

  function _formatTooltip(seg) {
    if (!seg) return null;
    const r = seg.reasoning;
    const stateName = seg.state;
    const labelMap = { S1: "S1", systole: "Systole", S2: "S2", diastole: "Diastole" };
    const label = labelMap[stateName] || stateName;
    const tStart = typeof seg.start === "number" ? seg.start.toFixed(2) : "?";
    let lines = [`${label} @ ${tStart}s`];
    if (r) {
      lines.push(`Expected: ${r.expected_ms}ms  •  Measured: ${r.measured_ms}ms`);
      if (r.notes && r.notes.length > 0) {
        for (const note of r.notes) {
          lines.push(note);
        }
      }
    }
    return lines.join("\n");
  }

  function initStateStripHover() {
    if (!stateStripCanvas || !stateStripTooltip) return;

    stateStripCanvas.addEventListener("mousemove", (e) => {
      if (!xAxisRange || xAxisRange.length < 2) return;
      const rect = stateStripCanvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const w = rect.width || 1;
      const x0ms = new Date(xAxisRange[0]).getTime();
      const x1ms = new Date(xAxisRange[1]).getTime();
      const tSec = (x0ms + (mouseX / w) * (x1ms - x0ms)) / 1000;
      const seg = _segmentAtTime(tSec);
      const text = _formatTooltip(seg);
      if (!text) {
        stateStripTooltip.classList.add("hidden");
        return;
      }
      stateStripTooltip.textContent = text;
      stateStripTooltip.classList.remove("hidden");
      // Position tooltip near cursor, keeping it on screen.
      const ttW = 340, ttH = 90;
      let tx = e.clientX + 14;
      let ty = e.clientY - 10;
      if (tx + ttW > window.innerWidth)  tx = e.clientX - ttW - 14;
      if (ty + ttH > window.innerHeight) ty = e.clientY - ttH - 4;
      stateStripTooltip.style.left = `${Math.max(0, tx)}px`;
      stateStripTooltip.style.top  = `${Math.max(0, ty)}px`;
    });

    stateStripCanvas.addEventListener("mouseleave", () => {
      stateStripTooltip.classList.add("hidden");
    });
  }

  function initPass3StateViewSelector() {
    if (!pass3StateViewSelect) return;
    pass3StateViewSelect.innerHTML = "";
    const opts = [];
    if (PASS3_SEGMENTS_BEFORE && PASS3_SEGMENTS_BEFORE.length > 0) {
      opts.push({ value: "before", label: "Before correction" });
    }
    if (PASS3_SEGMENTS_AFTER && PASS3_SEGMENTS_AFTER.length > 0) {
      opts.push({ value: "after", label: "After correction" });
    }
    if (opts.length <= 1) {
      pass3StateViewSelect.style.display = "none";
      return;
    }
    pass3StateViewSelect.style.display = "";
    for (const o of opts) {
      const el = document.createElement("option");
      el.value = o.value;
      el.textContent = o.label;
      pass3StateViewSelect.appendChild(el);
    }
    const initial =
      PASS3_SEGMENTS_DEFAULT_VIEW === "before" && PASS3_SEGMENTS_BEFORE.length > 0
        ? "before"
        : "after";
    pass3StateViewSelect.value = initial;
    pass3SegmentsActive = initial === "before" ? PASS3_SEGMENTS_BEFORE : PASS3_SEGMENTS_AFTER;
    pass3StateViewSelect.addEventListener("change", (ev) => {
      const v = ev && ev.target ? ev.target.value : "after";
      pass3SegmentsActive = v === "before" ? PASS3_SEGMENTS_BEFORE : PASS3_SEGMENTS_AFTER;
      scheduleDrawPass3StateStrip();
    });
  }

  // Update playhead positions
  function updatePlayhead(currentTime) {
    const percent = TOTAL_DURATION > 0 ? (currentTime / TOTAL_DURATION) * 100 : 0;

    // Update timeline
    if (timelineProgress) {
      timelineProgress.style.width = percent + "%";
    }
    if (timelinePlayhead) {
      timelinePlayhead.style.left = percent + "%";
    }

    // Update time display
    if (currentTimeEl) {
      currentTimeEl.textContent = formatTime(currentTime);
    }

    // Update chart playhead if synced
    if (isSynced && plotlyGraphDiv && chartPlayhead) {
      const xPos = getXPositionForTime(currentTime);
      if (xPos !== null) {
        chartPlayhead.style.display = "block";
        chartPlayhead.style.left = xPos + "px";
      } else {
        chartPlayhead.style.display = "none";
      }
    }
  }

  // Seek to position
  function seekTo(seconds) {
    if (!audio) return;
    audio.currentTime = Math.max(0, Math.min(seconds, TOTAL_DURATION));
    updatePlayhead(audio.currentTime);
  }

  // Play/Pause toggle
  function togglePlay() {
    if (!audio) return;
    if (!hasPlaybackAudio()) return; // no WAV for playback; warning is on play button tooltip
    if (isPlaying) {
      audio.pause();
      if (playBtn) {
        playBtn.textContent = "▶ Play";
        playBtn.classList.remove("active");
      }
    } else {
      audio.play().catch((e) => console.log("Audio play error:", e));
      if (playBtn) {
        playBtn.textContent = "⏸ Pause";
        playBtn.classList.add("active");
      }
    }
    isPlaying = !isPlaying;
  }

  // Stop playback
  function stopPlayback() {
    if (!audio) return;
    audio.pause();
    audio.currentTime = 0;
    isPlaying = false;
    if (playBtn) {
      playBtn.textContent = "▶ Play";
      playBtn.classList.remove("active");
    }
    updatePlayhead(0);
  }

  // Toggle sync
  function toggleSync() {
    isSynced = !isSynced;
    if (syncBtn) {
      syncBtn.classList.toggle("active", isSynced);
    }
    if (!isSynced && chartPlayhead) {
      chartPlayhead.style.display = "none";
    } else if (audio) {
      updatePlayhead(audio.currentTime);
    }
  }

  // Toggle spectrogram visibility
  function toggleSpectrogram() {
    if (!spectrogramImage || !spectrogramBtn) return;
    if (!SPECTROGRAM_AVAILABLE[currentAudioKey]) {
      alert("Spectrogram not available for this audio source.");
      return;
    }
    isSpectrogramVisible = !isSpectrogramVisible;
    spectrogramBtn.classList.toggle("active", isSpectrogramVisible);
    spectrogramImage.classList.toggle("hidden", !isSpectrogramVisible);
    if (isSpectrogramVisible) {
      spectrogramImage.style.opacity = spectrogramOpacity ? spectrogramOpacity.value : "0.4";
      updateSpectrogramSourceForCurrentAudio();
      updateSpectrogramPosition();
    } else {
      spectrogramImage.style.removeProperty("opacity");
    }
  }

  const pendingAxisGridUpdates = {};

  function getAxisShowGrid(axisKey) {
    if (!axisKey) {
      return true;
    }
    if (Object.prototype.hasOwnProperty.call(pendingAxisGridUpdates, axisKey)) {
      return pendingAxisGridUpdates[axisKey];
    }
    if (plotlyGraphDiv && plotlyGraphDiv._fullLayout) {
      const axisLayout = plotlyGraphDiv._fullLayout[axisKey];
      if (axisLayout && typeof axisLayout.showgrid === "boolean") {
        return axisLayout.showgrid;
      }
    }
    return true;
  }

  function refreshAxisGridButtons() {
    if (!axisGridButtons || axisGridButtons.length === 0) {
      return;
    }
    axisGridButtons.forEach((button) => {
      const axisKey = button.dataset.gridAxis;
      const showGrid = getAxisShowGrid(axisKey);
      button.classList.toggle("active", showGrid);
    });
  }

  function applyAxisGridState(axisKey, showGrid) {
    if (!axisKey) {
      return;
    }
    if (!plotlyGraphDiv) {
      pendingAxisGridUpdates[axisKey] = showGrid;
      return;
    }
    const layoutKey = axisKey + ".showgrid";
    const updates = {};
    updates[layoutKey] = showGrid;
    Plotly.relayout(plotlyGraphDiv, updates).then(() => {
      refreshAxisGridButtons();
    });
  }

  function toggleAxisGrid(event) {
    const button = event.currentTarget;
    const axisKey = button && button.dataset ? button.dataset.gridAxis : null;
    if (!axisKey) {
      return;
    }
    const nextState = !getAxisShowGrid(axisKey);
    button.classList.toggle("active", nextState);
    applyAxisGridState(axisKey, nextState);
  }

  function applyBeatHoverToPlot() {
    if (!plotlyGraphDiv) return;
    const indices = BEAT_HOVER_TRACES.map((name) => findTraceIndexByName(name)).filter((i) => i !== null);
    if (!indices.length) return;
    if (!beatHoverEnabled) {
      indices.forEach((i) => {
        if (!plotlyGraphDiv.data[i]._origHoverTemplate) {
          plotlyGraphDiv.data[i]._origHoverTemplate =
            plotlyGraphDiv.data[i].hovertemplate || "%{customdata}<extra></extra>";
        }
      });
    }
    const hoverinfo = beatHoverEnabled ? "all" : "skip";
    const hovertemplate = beatHoverEnabled
      ? indices.map((i) => plotlyGraphDiv.data[i]._origHoverTemplate || "%{customdata}<extra></extra>")
      : indices.map(() => null);
    Plotly.restyle(plotlyGraphDiv, { hoverinfo, hovertemplate }, indices).then(() => {
      const btn = document.getElementById("hover-toggle-btn");
      if (btn) btn.classList.toggle("active", beatHoverEnabled);
    });
  }

  function toggleBeatHover() {
    if (!plotlyGraphDiv) return;
    beatHoverEnabled = !beatHoverEnabled;
    applyBeatHoverToPlot();
  }

  function flushPendingAxisGridUpdates() {
    if (!plotlyGraphDiv) {
      return;
    }
    const updates = {};
    let hasUpdates = false;
    for (const axisKey in pendingAxisGridUpdates) {
      if (!Object.prototype.hasOwnProperty.call(pendingAxisGridUpdates, axisKey)) {
        continue;
      }
      updates[axisKey + ".showgrid"] = pendingAxisGridUpdates[axisKey];
      delete pendingAxisGridUpdates[axisKey];
      hasUpdates = true;
    }
    if (hasUpdates) {
      Plotly.relayout(plotlyGraphDiv, updates).then(() => {
        refreshAxisGridButtons();
      });
    }
  }

  // Update spectrogram opacity (only when spectrogram is toggled on; slider must not make it visible when off)
  function updateSpectrogramOpacity(value) {
    if (!spectrogramImage) return;
    if (!isSpectrogramVisible) {
      spectrogramImage.style.removeProperty("opacity");
      return;
    }
    spectrogramImage.style.opacity = value;
  }

  // Update spectrogram position and scale based on current view
  function updateSpectrogramPosition() {
    if (
      !plotlyGraphDiv ||
      !isSpectrogramVisible ||
      !SPECTROGRAM_AVAILABLE[currentAudioKey] ||
      !xAxisRange ||
      !spectrogramContainer ||
      !spectrogramImage
    )
      return;

    const plotArea = plotlyGraphDiv._fullLayout;
    if (!plotArea) return;

    const xaxis = plotArea.xaxis;
    const yaxis = plotArea.yaxis;
    if (!xaxis || !yaxis) return;

    // Get plot area dimensions
    const plotLeft = xaxis._offset;
    const plotWidth = xaxis._length;
    const plotTop = yaxis._offset;
    const plotHeight = yaxis._length;

    // Get current view range
    const viewXMin = new Date(xAxisRange[0]).getTime();
    const viewXMax = new Date(xAxisRange[1]).getTime();

    // Get full data range (0 to total duration)
    const fullXMin = EPOCH.getTime();
    const fullXMax = EPOCH.getTime() + TOTAL_DURATION * 1000;

    // Calculate what portion of the full data is visible
    const visibleStartRatio = (viewXMin - fullXMin) / (fullXMax - fullXMin || 1);
    const visibleEndRatio = (viewXMax - fullXMin) / (fullXMax - fullXMin || 1);
    const visibleRatio = visibleEndRatio - visibleStartRatio || 1;

    // Calculate spectrogram dimensions
    const spectrogramFullWidth = plotWidth / visibleRatio;
    const spectrogramLeft = plotLeft - visibleStartRatio * spectrogramFullWidth;

    // Position the spectrogram container to match plot area
    spectrogramContainer.style.left = plotLeft + "px";
    spectrogramContainer.style.top = plotTop + "px";
    spectrogramContainer.style.width = plotWidth + "px";
    spectrogramContainer.style.height = plotHeight + "px";

    // Position the spectrogram image
    spectrogramImage.style.left = spectrogramLeft - plotLeft + "px";
    spectrogramImage.style.width = spectrogramFullWidth + "px";
    spectrogramImage.style.height = plotHeight + "px";
    spectrogramImage.style.top = "0px";
  }

  // Event Listeners
  if (playBtn) playBtn.addEventListener("click", togglePlay);
  if (stopBtn) stopBtn.addEventListener("click", stopPlayback);
  if (syncBtn) syncBtn.addEventListener("click", toggleSync);
  if (spectrogramBtn) spectrogramBtn.addEventListener("click", toggleSpectrogram);

  if (spectrogramOpacity) {
    spectrogramOpacity.addEventListener("input", (e) => {
      updateSpectrogramOpacity(parseFloat(e.target.value));
    });
  }

  function openAnalysisSummaryModal() {
    if (analysisSummaryText) {
      analysisSummaryText.value = ANALYSIS_SUMMARY || "No summary data available.";
    }
    if (analysisSummaryOverlay) {
      analysisSummaryOverlay.classList.add("visible");
      analysisSummaryOverlay.setAttribute("aria-hidden", "false");
    }
  }

  function closeAnalysisSummaryModal() {
    if (analysisSummaryOverlay) {
      analysisSummaryOverlay.classList.remove("visible");
      analysisSummaryOverlay.setAttribute("aria-hidden", "true");
    }
  }

  if (analysisSummaryBtn) {
    analysisSummaryBtn.addEventListener("click", openAnalysisSummaryModal);
  }
  if (analysisSummaryClose) {
    analysisSummaryClose.addEventListener("click", closeAnalysisSummaryModal);
  }
  if (analysisSummaryOverlay) {
    analysisSummaryOverlay.addEventListener("click", (e) => {
      if (e.target === analysisSummaryOverlay) closeAnalysisSummaryModal();
    });
  }

  if (volumeSlider && audio) {
    volumeSlider.addEventListener("input", (e) => {
      audio.volume = parseFloat(e.target.value);
    });
  }

  if (axisGridButtons && axisGridButtons.length > 0) {
    axisGridButtons.forEach((button) => {
      button.addEventListener("click", toggleAxisGrid);
    });
  }

  // --- Legend category filter (Debug vs Analysis Data) ---
  const LEGEND_DEBUG_NAMES = new Set([
    "Audio Envelope",
    "Dynamic Noise Floor",
    "Troughs",
    "S1 Beats",
    "S2 Beats",
    "Noise/Rejected",
    "BPM Trend (Belief)",
    "Manual S1",
    "Manual S2",
    "Manual Noise",
  ]);

  // Traces that appear in both Debug and Analysis Data views
  const LEGEND_IN_BOTH_NAMES = new Set(["Average BPM"]);

  // In Analysis Data view, y-axis range is set to (max amplitude of visible analysis traces) * this factor.
  const ANALYSIS_VIEW_Y_RANGE_MULTIPLIER = 5;

  let legendCategoryInitialState = null;
  let signalAxisRangeDefault = null;

  /** Returns max Y value from traces that are visible in Analysis Data view and use the primary y-axis. */
  function getMaxYFromAnalysisTraces() {
    if (!plotlyGraphDiv || !plotlyGraphDiv.data) return 0;
    let maxY = 0;
    for (let ti = 0; ti < plotlyGraphDiv.data.length; ti++) {
      const tr = plotlyGraphDiv.data[ti];
      const name = (tr.name || "").trim();
      const isDebug = LEGEND_DEBUG_NAMES.has(name);
      const inBoth = LEGEND_IN_BOTH_NAMES.has(name);
      if (isDebug && !inBoth) continue;
      if (tr.yaxis && tr.yaxis !== "y") continue;
      const ySrc = tr.y || tr._inputArray || tr.bdata || tr.data || tr.values;
      const len = ySrc && typeof ySrc.length === "number" ? ySrc.length : 0;
      for (let i = 0; i < len; i++) {
        const v = getNumericFromArrayLike(ySrc, i);
        if (v !== null && Number.isFinite(v)) maxY = Math.max(maxY, v);
      }
    }
    return maxY;
  }

  function snapshotLegendCategoryDefaults() {
    if (!plotlyGraphDiv || !plotlyGraphDiv.data || legendCategoryInitialState) return;
    legendCategoryInitialState = plotlyGraphDiv.data.map((tr) => ({
      visible: tr.visible === undefined ? true : tr.visible,
      showlegend: tr.showlegend !== false,
    }));
    if (
      plotlyGraphDiv._fullLayout &&
      plotlyGraphDiv._fullLayout.yaxis &&
      signalAxisRangeDefault === null
    ) {
      const r = plotlyGraphDiv._fullLayout.yaxis.range;
      if (r && Array.isArray(r) && r.length === 2) {
        signalAxisRangeDefault = [Number(r[0]), Number(r[1])];
      }
    }
  }

  function getDefaultForTrace(index) {
    if (legendCategoryInitialState && index < legendCategoryInitialState.length) {
      return legendCategoryInitialState[index];
    }
    return { visible: true, showlegend: true };
  }

  function applyLegendCategoryFilter(value) {
    if (!plotlyGraphDiv || !plotlyGraphDiv.data) return;
    const data = plotlyGraphDiv.data;
    const visibility = [];
    const showlegend = [];
    for (let i = 0; i < data.length; i++) {
      const name = (data[i].name || "").trim();
      const isDebug = LEGEND_DEBUG_NAMES.has(name);
      const defaultState = getDefaultForTrace(i);
      if (value === "all") {
        visibility.push(defaultState.visible);
        showlegend.push(defaultState.showlegend);
      } else if (value === "debug") {
        const show = isDebug || LEGEND_IN_BOTH_NAMES.has(name);
        if (show) {
          visibility.push(defaultState.visible);
          showlegend.push(defaultState.showlegend);
        } else {
          visibility.push(false);
          showlegend.push(false);
        }
      } else {
        const show = !isDebug || LEGEND_IN_BOTH_NAMES.has(name);
        if (show) {
          visibility.push(defaultState.visible);
          showlegend.push(defaultState.showlegend);
        } else {
          visibility.push(false);
          showlegend.push(false);
        }
      }
    }
    Plotly.restyle(plotlyGraphDiv, { visible: visibility, showlegend: showlegend });

    // In Analysis Data view, set signal (y) axis to 10× max amplitude of visible analysis traces; restore default otherwise
    if (plotlyGraphDiv._fullLayout && plotlyGraphDiv._fullLayout.yaxis) {
      let range;
      if (value === "analysis") {
        const maxY = getMaxYFromAnalysisTraces();
        range = [0, Math.max(maxY * ANALYSIS_VIEW_Y_RANGE_MULTIPLIER, 1)];
      } else {
        range = signalAxisRangeDefault;
      }
      if (range && Array.isArray(range) && range.length === 2) {
        Plotly.relayout(plotlyGraphDiv, { "yaxis.range": range });
      }
    }
  }

  // --- Labeling helpers ---

  // Find the index of a trace by its exact name.
  function findTraceIndexByName(targetName) {
    if (!plotlyGraphDiv || !plotlyGraphDiv.data) return null;
    for (let i = 0; i < plotlyGraphDiv.data.length; i++) {
      const tr = plotlyGraphDiv.data[i];
      if (!tr) continue;
      const name = tr.name || "";
      if (name === targetName) return i;
    }
    return null;
  }

  // Sample the "Audio Envelope" trace at an arbitrary time (in seconds).
  // Returns { xVal, yVal } or null if the envelope trace is missing.
  function getEnvelopePointAtTime(timeSec) {
    if (!plotlyGraphDiv || !plotlyGraphDiv.data) return null;
    const idx = findTraceIndexByName("Audio Envelope");
    if (idx === null) return null;

    const tr = plotlyGraphDiv.data[idx];
    if (!tr || !tr.x || !tr.y) return null;

    let bestI = -1;
    let bestDt = Infinity;

    for (let i = 0; i < tr.x.length; i++) {
      const xVal = tr.x[i];
      if (!xVal) continue;

      let tSec = null;
      if (xVal instanceof Date) {
        const ms = xVal.getTime();
        if (Number.isFinite(ms)) {
          tSec = (ms - EPOCH.getTime()) / 1000;
        }
      } else {
        const d = new Date(xVal);
        const ms = d.getTime();
        if (Number.isFinite(ms)) {
          tSec = (ms - EPOCH.getTime()) / 1000;
        }
      }
      if (tSec === null || !Number.isFinite(tSec)) continue;

      const dt = Math.abs(tSec - timeSec);
      if (dt < bestDt) {
        bestDt = dt;
        bestI = i;
      }
    }

    if (bestI < 0) return null;

    let yVal = null;
    try {
      const candidate = tr.y[bestI];
      const num =
        typeof candidate === "number" ? candidate : parseFloat(candidate);
      if (!Number.isFinite(num)) return null;
      yVal = num;
    } catch (e) {
      return null;
    }

    return { xVal: tr.x[bestI], yVal };
  }

  /**
   * Smoothed BPM for this pass at timeSec (nearest sample on the pass BPM curve).
   * Trace names match plotting.py: Pass 3 / Pass 2 / pass 1 preliminary / final "Average BPM".
   */
  function getBpmSmoothedAtTime(timeSec) {
    if (!plotlyGraphDiv || !plotlyGraphDiv.data || !Number.isFinite(timeSec)) {
      return null;
    }
    const candidateNames = [
      "BPM (Pass 3)",
      "BPM (Pass 2)",
      "BPM (pass 1)",
      "Average BPM",
    ];
    for (let ni = 0; ni < candidateNames.length; ni++) {
      const idx = findTraceIndexByName(candidateNames[ni]);
      if (idx === null) continue;
      const tr = plotlyGraphDiv.data[idx];
      if (!tr || !tr.x || !tr.y || !tr.x.length) continue;

      let bestI = -1;
      let bestDt = Infinity;
      for (let i = 0; i < tr.x.length; i++) {
        const xVal = tr.x[i];
        if (!xVal) continue;

        let tSec = null;
        if (xVal instanceof Date) {
          const ms = xVal.getTime();
          if (Number.isFinite(ms)) {
            tSec = (ms - EPOCH.getTime()) / 1000;
          }
        } else {
          const d = new Date(xVal);
          const ms = d.getTime();
          if (Number.isFinite(ms)) {
            tSec = (ms - EPOCH.getTime()) / 1000;
          }
        }
        if (tSec === null || !Number.isFinite(tSec)) continue;

        const dt = Math.abs(tSec - timeSec);
        if (dt < bestDt) {
          bestDt = dt;
          bestI = i;
        }
      }

      if (bestI < 0) continue;
      const yVal = getNumericFromArrayLike(tr.y, bestI);
      if (yVal !== null && Number.isFinite(yVal)) {
        return yVal;
      }
    }
    return null;
  }

  function buildEditablePeaks() {
    editablePeaks = [];
    if (!plotlyGraphDiv || !plotlyGraphDiv.data) return;

    const baseTraceIndices = new Set();

    plotlyGraphDiv.data.forEach((trace, traceIndex) => {
      const name = trace.name || "";
      let baseLabel = null;
      if (name === "S1 Beats") baseLabel = "S1";
      else if (name === "S2 Beats") baseLabel = "S2";
      else if (name === "Noise/Rejected") baseLabel = "Noise";

      if (!baseLabel || !trace.x || !trace.x.length) return;
      baseTraceIndices.add(traceIndex);

      for (let i = 0; i < trace.x.length; i++) {
        const xVal = trace.x[i];
        if (!xVal) continue;
        const tSec = (new Date(xVal).getTime() - EPOCH.getTime()) / 1000;

        // Capture the y-value at this peak so we can reliably round-trip it in CSV exports.
        let yVal = null;
        try {
          if (trace && trace.y) {
            yVal = getNumericFromArrayLike(trace.y, i);
          }
        } catch (e) {
          // ignore extraction errors
        }

        editablePeaks.push({
          timeSec: tSec,
          traceIndex,
          pointIndex: i,
          baseLabel,
          manualLabel: baseLabel,
          yVal,
        });
      }
    });

    // Dim the original classifier markers slightly so manual overlays stand out.
    if (baseTraceIndices.size > 0) {
      Plotly.restyle(
        plotlyGraphDiv,
        { opacity: 0.90 },
        Array.from(baseTraceIndices)
      );
    }
    refreshManualLabelTraces();
  }

  function ensureManualLabelTraces() {
    if (!plotlyGraphDiv) return;
    // If traces already exist, nothing to do.
    if (
      manualLabelTraceIndices.S1 !== null &&
      manualLabelTraceIndices.S2 !== null &&
      manualLabelTraceIndices.Noise !== null
    ) {
      return;
    }

    const baseIndex = plotlyGraphDiv.data.length;
    const commonOpts = {
      mode: "markers",
      yaxis: "y",
      showlegend: true,
      visible: "legendonly",
      hovertemplate:
        "Manual %{customdata[0]}<br>Time: %{x|%M:%S.%L}<br>Base: %{customdata[1]}<extra></extra>",
      customdata: [],
    };

    const traces = [
      {
        name: "Manual S1",
        marker: { color: "#ff4d4d", size: 10, symbol: "circle-open" },
        x: [],
        y: [],
        ...commonOpts,
      },
      {
        name: "Manual S2",
        marker: { color: "#ffa94d", size: 9, symbol: "circle-open" },
        x: [],
        y: [],
        ...commonOpts,
      },
      {
        name: "Manual Noise",
        marker: { color: "#bbbbbb", size: 8, symbol: "x" },
        x: [],
        y: [],
        ...commonOpts,
      },
    ];

    Plotly.addTraces(plotlyGraphDiv, traces);
    manualLabelTraceIndices.S1 = baseIndex;
    manualLabelTraceIndices.S2 = baseIndex + 1;
    manualLabelTraceIndices.Noise = baseIndex + 2;
  }

  function refreshManualLabelTraces() {
    if (!plotlyGraphDiv || !editablePeaks.length) return;
    ensureManualLabelTraces();

    const s1X = [];
    const s1Y = [];
    const s1Custom = [];

    const s2X = [];
    const s2Y = [];
    const s2Custom = [];

    const noiseX = [];
    const noiseY = [];
    const noiseCustom = [];
    const unknownLabels = new Set();

    const canonicalLabel = (s) =>
      String(s || "")
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9_]/g, "");

    editablePeaks.forEach((p) => {
      const tDate = secondsToDatetime(p.timeSec);
      let yVal = null;

      // 1) Prefer stored yVal if available
      if (typeof p.yVal === "number" && Number.isFinite(p.yVal)) {
        yVal = p.yVal;
      } else {
        // 2) Try to read from the original base trace
        const baseTrace =
          plotlyGraphDiv &&
          p.traceIndex != null &&
          plotlyGraphDiv.data &&
          plotlyGraphDiv.data[p.traceIndex]
            ? plotlyGraphDiv.data[p.traceIndex]
            : null;
        if (baseTrace && baseTrace.y) {
          yVal = getNumericFromArrayLike(baseTrace.y, p.pointIndex) ?? yVal;
        }

        // 3) Fallback: sample from the Audio Envelope trace at this time
        if (yVal === null || typeof yVal === "undefined") {
          const envPt = getEnvelopePointAtTime(p.timeSec);
          if (envPt && typeof envPt.yVal === "number") {
            yVal = envPt.yVal;
          }
        }

        // Cache back onto the peak object for future exports/updates.
        if (typeof yVal === "number" && Number.isFinite(yVal)) {
          p.yVal = yVal;
        }
      }
      if (yVal === null || typeof yVal === "undefined") return;

      const entry = [p.manualLabel, p.baseLabel];
      const lblCanon = canonicalLabel(p.manualLabel);

      if (lblCanon === "s1") {
        s1X.push(tDate);
        s1Y.push(yVal);
        s1Custom.push(entry);
      } else if (lblCanon === "s2") {
        s2X.push(tDate);
        s2Y.push(yVal);
        s2Custom.push(entry);
      } else if (
        lblCanon === "noise" ||
        lblCanon === "noiserejected" ||
        lblCanon === "rejected" ||
        lblCanon === "artifact" ||
        lblCanon === "artifacts"
      ) {
        noiseX.push(tDate);
        noiseY.push(yVal);
        noiseCustom.push(entry);
      } else {
        if (p.manualLabel) {
          unknownLabels.add(p.manualLabel);
        }
      }
    });

    Plotly.restyle(
      plotlyGraphDiv,
      { x: [s1X], y: [s1Y], customdata: [s1Custom] },
      manualLabelTraceIndices.S1
    );
    Plotly.restyle(
      plotlyGraphDiv,
      { x: [s2X], y: [s2Y], customdata: [s2Custom] },
      manualLabelTraceIndices.S2
    );
    Plotly.restyle(
      plotlyGraphDiv,
      { x: [noiseX], y: [noiseY], customdata: [noiseCustom] },
      manualLabelTraceIndices.Noise
    );
  }

  function findNearestEditablePeak(timeSec) {
    if (!editablePeaks.length) return null;
    let best = null;
    let bestDt = Infinity;
    editablePeaks.forEach((p) => {
      const dt = Math.abs(p.timeSec - timeSec);
      if (dt < bestDt) {
        bestDt = dt;
        best = p;
      }
    });
    if (!best) return null;
    return { peak: best, delta: bestDt };
  }

  function applyLabelToNearestPeak() {
    if (!audio) return;
    if (!editablePeaks.length) {
      alert("No peaks available to relabel in this plot.");
      return;
    }
    const targetTime = audio.currentTime;
    const result = findNearestEditablePeak(targetTime);
    if (!result) {
      alert("Could not find any peaks to relabel.");
      return;
    }
    const { peak, delta } = result;
    const desiredLabel =
      (labelTypeSelect && labelTypeSelect.value) ? labelTypeSelect.value : "S1";

    // Optional sanity threshold: 0.4s from playhead
    if (delta > 0.4) {
      if (
        !window.confirm(
          `Nearest peak is ${delta.toFixed(
            3
          )}s away from playhead at t=${targetTime.toFixed(
            3
          )}s. Apply label ${desiredLabel} anyway?`
        )
      ) {
        return;
      }
    }

    const prev = peak.manualLabel;
    peak.manualLabel = desiredLabel;
    console.log(
      `Relabeled peak at t=${peak.timeSec.toFixed(
        3
      )}s from ${prev} → ${desiredLabel}`
    );
    refreshManualLabelTraces();
  }

  // Flip all S1/S2 labels to the right of the current playhead.
  function flipLabelsRightOfPlayhead() {
    if (!audio) return;
    if (!editablePeaks.length) {
      alert("No peaks available to flip in this plot.");
      return;
    }

    const cutoffTime = audio.currentTime;
    let flippedCount = 0;

    editablePeaks.forEach((p) => {
      if (!Number.isFinite(p.timeSec)) return;
      if (p.timeSec < cutoffTime) return;

      const manual = p.manualLabel || p.baseLabel;
      if (manual === "S1") {
        p.manualLabel = "S2";
        flippedCount++;
      } else if (manual === "S2") {
        p.manualLabel = "S1";
        flippedCount++;
      }
    });

    if (flippedCount > 0) {
      refreshManualLabelTraces();
    } else {
      alert("No S1/S2 peaks to flip to the right of the playhead.");
    }
  }

  function downloadLabelsCsv() {
    if (!editablePeaks.length) {
      alert("No peak labels available to export.");
      return;
    }

    // Debug-friendly export: include both logical time and the actual x/y used for plotting.
    // x_plot_sec is just time_sec; y_plot is the envelope/sample value used when drawing.
    // bpm_smoothed is the pass smoothed BPM curve at that time (nearest point on the plot).
    const header =
      "time_sec,base_label,manual_label,x_plot_sec,y_plot,bpm_smoothed\n";
    const sorted = [...editablePeaks].sort(
      (a, b) => a.timeSec - b.timeSec
    );
    let missingYCount = 0;
    const lines = sorted.map((p, idx) => {
      // time_sec / x_plot_sec
      const t = Number.isFinite(p.timeSec) ? p.timeSec : NaN;

      // y_plot: prefer stored yVal; otherwise, derive from base trace if possible.
      let yPlot = null;
      if (typeof p.yVal === "number") {
        yPlot = p.yVal;
      } else if (
        plotlyGraphDiv &&
        p.traceIndex != null &&
        p.pointIndex != null &&
        plotlyGraphDiv.data &&
        plotlyGraphDiv.data[p.traceIndex] &&
        plotlyGraphDiv.data[p.traceIndex].y
      ) {
        const baseTrace = plotlyGraphDiv.data[p.traceIndex];
        yPlot = getNumericFromArrayLike(baseTrace.y, p.pointIndex) ?? yPlot;
      } else {
        // Fallback for robustness: sample from Audio Envelope at this time.
        try {
          const envPt = getEnvelopePointAtTime(p.timeSec);
          if (envPt && typeof envPt.yVal === "number") {
            yPlot = envPt.yVal;
          }
        } catch (e) {
          // ignore export-time envelope lookup errors
        }
      }

      const safeY = Number.isFinite(yPlot) ? yPlot : "";

      if (!Number.isFinite(yPlot)) missingYCount++;

      const bpmSmoothed = Number.isFinite(t) ? getBpmSmoothedAtTime(t) : null;
      const safeBpm =
        bpmSmoothed !== null && Number.isFinite(bpmSmoothed)
          ? bpmSmoothed.toFixed(3)
          : "";

      return [
        Number.isFinite(t) ? t.toFixed(3) : "",
        p.baseLabel ?? "",
        p.manualLabel ?? "",
        Number.isFinite(t) ? t.toFixed(3) : "",
        safeY,
        safeBpm,
      ].join(",");
    });
    const csvContent = header + lines.join("\n");

    const blob = new Blob([csvContent], {
      type: "text/csv;charset=utf-8;",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const baseName =
      (audioFileNameEl && audioFileNameEl.dataset && audioFileNameEl.dataset.defaultName) ||
      "labels";
    link.href = url;
    link.download = `${baseName}_manually_Labeled_peaks.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  // Import labels from a CSV that was previously exported by this tool.
  // This implementation is intentionally simple and deterministic:
  // - It matches peaks by rounded time_sec (3 decimal places).
  // - It updates manual_label (and base_label, if present) on existing peaks.
  // - It does NOT create new peaks; rows without a matching time are ignored.
  function applyImportedLabelsCsv(csvText) {
    if (!csvText) {
      alert("No CSV content available to import labels.");
      return;
    }
    if (!editablePeaks.length) {
      alert("No peaks available to relabel in this plot.");
      return;
    }

    const lines = csvText.split(/\r?\n/).filter((l) => l.trim().length > 0);
    if (lines.length <= 1) {
      alert("CSV appears to be empty or missing data rows.");
      return;
    }

    const headerCells = lines[0].split(",");
    const lower = headerCells.map((h) => h.trim().toLowerCase());
    const timeIdx = lower.indexOf("time_sec");
    const manualIdx = lower.indexOf("manual_label");
    const baseIdx = lower.indexOf("base_label");

    if (timeIdx === -1 || (manualIdx === -1 && baseIdx === -1)) {
      alert(
        'CSV must contain at least "time_sec" and "manual_label" or "base_label" columns.'
      );
      return;
    }

    // Build a lookup table from rounded time_sec -> { manual, base }.
    const importedByTime = new Map();
    for (let i = 1; i < lines.length; i++) {
      const row = lines[i];
      const cells = row.split(",");
      if (cells.length <= timeIdx) continue;

      const tRaw = cells[timeIdx];
      const t = parseFloat(tRaw);
      if (!Number.isFinite(t)) continue;
      const key = t.toFixed(3);

      const manualCell =
        manualIdx !== -1 && cells.length > manualIdx ? cells[manualIdx] : "";
      const baseCell =
        baseIdx !== -1 && cells.length > baseIdx ? cells[baseIdx] : "";

      const manual = (manualCell || "").trim();
      const base = (baseCell || "").trim();

      if (!manual && !base) continue;

      importedByTime.set(key, { manual, base });
    }

    if (!importedByTime.size) {
      alert("No usable label rows found in CSV.");
      return;
    }

    let updatedCount = 0;

    editablePeaks.forEach((p) => {
      if (!Number.isFinite(p.timeSec)) return;
      const key = p.timeSec.toFixed(3);
      const rec = importedByTime.get(key);
      if (!rec) return;

      const { manual, base } = rec;
      if (!manual && !base) return;

      if (manual) {
        p.manualLabel = manual;
      }
      if (base) {
        p.baseLabel = base;
      }
      updatedCount++;
    });

    if (updatedCount > 0) {
      refreshManualLabelTraces();
      // Ensure manual trace legends are visible and base S1/S2/Noise traces go to legendonly.
      const showManualTraces = Object.values(manualLabelTraceIndices).filter(
        (idx) => typeof idx === "number" && idx >= 0
      );
      if (showManualTraces.length && plotlyGraphDiv) {
        Plotly.restyle(plotlyGraphDiv, { visible: true }, showManualTraces);
      }

      const hideNames = ["S1 Beats", "S2 Beats", "Noise/Rejected"];
      const hideIndices = hideNames
        .map((name) => findTraceIndexByName(name))
        .filter((idx) => typeof idx === "number" && idx >= 0);
      if (hideIndices.length && plotlyGraphDiv) {
        Plotly.restyle(plotlyGraphDiv, { visible: "legendonly" }, hideIndices);
      }
    } else {
      alert(
        "No labels from CSV could be matched to existing peaks by time_sec. Check that the file was exported from this viewer."
      );
    }
  }

  if (applyLabelBtn) {
    applyLabelBtn.addEventListener("click", applyLabelToNearestPeak);
  }
  if (flipLabelsRightBtn) {
    flipLabelsRightBtn.addEventListener("click", flipLabelsRightOfPlayhead);
  }
  if (downloadLabelsBtn) {
    downloadLabelsBtn.addEventListener("click", downloadLabelsCsv);
  }
  if (importLabelsBtn && importLabelsInput) {
    // Use a hidden file input so Import CSV is only processed on explicit click.
    importLabelsBtn.addEventListener("click", () => {
      if (!plotlyGraphDiv || !editablePeaks.length) {
        alert(
          "Plot not ready or no peaks available yet. Wait for the chart to finish loading before importing labels."
        );
        return;
      }
      importLabelsInput.value = "";
      importLabelsInput.click();
    });

    importLabelsInput.addEventListener("change", (event) => {
      const file =
        event && event.target && event.target.files && event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e && e.target && e.target.result ? e.target.result : "";
        applyImportedLabelsCsv(text);
      };
      reader.readAsText(file);
    });
  }

  // Timeline scrubber click/drag
  let isDragging = false;

  function handleTimelineInteraction(e) {
    if (!timelineScrubber) return;
    const rect = timelineScrubber.getBoundingClientRect();
    const percent = Math.max(
      0,
      Math.min(1, (e.clientX - rect.left) / (rect.width || 1))
    );
    seekTo(percent * TOTAL_DURATION);
  }

  if (timelineScrubber) {
    timelineScrubber.addEventListener("mousedown", (e) => {
      isDragging = true;
      handleTimelineInteraction(e);
    });
  }

  document.addEventListener("mousemove", (e) => {
    if (isDragging) {
      handleTimelineInteraction(e);
    }
  });

  document.addEventListener("mouseup", () => {
    isDragging = false;
  });

  // Audio error handling
  if (audio) {
    audio.addEventListener("error", function () {
      let error_msg = "Unknown error";
      switch (audio.error && audio.error.code) {
        case 1:
          error_msg = "Audio loading aborted";
          break;
        case 2:
          error_msg = "Network error - file not found or inaccessible";
          break;
        case 3:
          error_msg = "Audio decoding error - file may be corrupted";
          break;
        case 4:
          error_msg = "Audio format not supported";
          break;
      }

      console.error("❌ Audio Error:", error_msg, "Code:", audio.error && audio.error.code);
    });

    // Debug: log audio load status
    audio.addEventListener("canplaythrough", function () {
      console.log("✅ Audio file loaded successfully and can play through");
    });

    audio.addEventListener("loadstart", function () {
      console.log("🔄 Starting to load audio...");
    });

    // Audio time update
    audio.addEventListener("timeupdate", () => {
      updatePlayhead(audio.currentTime);
    });

    audio.addEventListener("ended", () => {
      isPlaying = false;
      if (playBtn) {
        playBtn.textContent = "▶ Play";
        playBtn.classList.remove("active");
      }
    });
  }

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    // Don't trigger if typing in an input or textarea
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;

    if (e.code === "Escape" && analysisSummaryOverlay && analysisSummaryOverlay.classList.contains("visible")) {
      closeAnalysisSummaryModal();
      e.preventDefault();
      return;
    }

    switch (e.code) {
      case "Space":
        e.preventDefault();
        togglePlay();
        break;
      case "KeyS":
        stopPlayback();
        break;
      case "ArrowLeft":
        e.preventDefault();
        seekTo(audio ? audio.currentTime - 5 : 0);
        break;
      case "ArrowRight":
        e.preventDefault();
        seekTo(audio ? audio.currentTime + 5 : 0);
        break;
      case "Home":
        e.preventDefault();
        seekTo(0);
        break;
      case "End":
        e.preventDefault();
        seekTo(TOTAL_DURATION);
        break;
      case "KeyG":
        toggleSpectrogram();
        break;
      case "Digit1":
        // Label nearest peak as S1
        e.preventDefault();
        if (labelTypeSelect) {
          labelTypeSelect.value = "S1";
        }
        applyLabelToNearestPeak();
        break;
      case "Digit2":
        // Label nearest peak as S2
        e.preventDefault();
        if (labelTypeSelect) {
          labelTypeSelect.value = "S2";
        }
        applyLabelToNearestPeak();
        break;
      case "Digit3":
        // Label nearest peak as Noise
        e.preventDefault();
        if (labelTypeSelect) {
          labelTypeSelect.value = "Noise";
        }
        applyLabelToNearestPeak();
        break;
      case "KeyA":
        // A/B compare state strip: A = "before" correction
        if (pass3StateViewSelect && pass3StateViewSelect.style.display !== "none") {
          if (PASS3_SEGMENTS_BEFORE && PASS3_SEGMENTS_BEFORE.length > 0) {
            e.preventDefault();
            pass3StateViewSelect.value = "before";
            pass3SegmentsActive = PASS3_SEGMENTS_BEFORE;
            scheduleDrawPass3StateStrip();
          }
        }
        break;
      case "KeyD":
        // A/B compare state strip: D = "after" correction
        if (pass3StateViewSelect && pass3StateViewSelect.style.display !== "none") {
          if (PASS3_SEGMENTS_AFTER && PASS3_SEGMENTS_AFTER.length > 0) {
            e.preventDefault();
            pass3StateViewSelect.value = "after";
            pass3SegmentsActive = PASS3_SEGMENTS_AFTER;
            scheduleDrawPass3StateStrip();
          }
        }
        break;
    }
  });

  // Initialize Plotly integration after chart loads
  function initPlotlyIntegration() {
    const graphDivs = document.querySelectorAll(".plotly-graph-div");
    if (graphDivs.length > 0) {
      plotlyGraphDiv = graphDivs[0];
      refreshAxisGridButtons();
      flushPendingAxisGridUpdates();
      buildEditablePeaks();

      const legendCategoryFilter = document.getElementById("legend-category-filter");
      if (legendCategoryFilter) {
        legendCategoryFilter.addEventListener("change", function () {
          applyLegendCategoryFilter(this.value);
        });
      }
      const hoverToggleBtn = document.getElementById("hover-toggle-btn");
      if (hoverToggleBtn) {
        hoverToggleBtn.classList.toggle("active", beatHoverEnabled);
        hoverToggleBtn.addEventListener("click", toggleBeatHover);
      }

      applyBeatHoverToPlot();

      function updateAxisRange() {
        if (plotlyGraphDiv._fullLayout && plotlyGraphDiv._fullLayout.xaxis) {
          xAxisRange = plotlyGraphDiv._fullLayout.xaxis.range;
          if (!fullXAxisRange && xAxisRange) {
            fullXAxisRange = [...xAxisRange];
          }
        }
      }

      updateAxisRange();

      plotlyGraphDiv.on("plotly_relayout", function () {
        updateAxisRange();
        if (audio) {
          updatePlayhead(audio.currentTime);
        }
        updateSpectrogramPosition();
        refreshAxisGridButtons();
        scheduleDrawPass3StateStrip();
      });

      plotlyGraphDiv.on("plotly_afterplot", function () {
        snapshotLegendCategoryDefaults();
        updateAxisRange();
        updateSpectrogramPosition();
        refreshAxisGridButtons();
        scheduleDrawPass3StateStrip();
      });

      window.addEventListener("resize", () => {
        updateAxisRange();
        if (audio) {
          updatePlayhead(audio.currentTime);
        }
        updateSpectrogramPosition();
        scheduleDrawPass3StateStrip();
        Plotly.Plots.resize(plotlyGraphDiv);
      });

      plotlyGraphDiv.on("plotly_click", function (data) {
        if (data.points && data.points.length > 0) {
          const point = data.points[0];
          if (point.x) {
            const clickTime = new Date(point.x);
            const seconds = (clickTime.getTime() - EPOCH.getTime()) / 1000;
            seekTo(seconds);
          }
        }
      });

      setTimeout(updateSpectrogramPosition, 100);
    } else {
      setTimeout(initPlotlyIntegration, 100);
    }
  }

  // Initialize spectrogram controls based on availability
  function initSpectrogramControls() {
    if (!spectrogramBtn || !spectrogramOpacity) return;
    const anySpectrogramAvailable =
      SPECTROGRAM_AVAILABLE.original || SPECTROGRAM_AVAILABLE.filtered;
    if (!anySpectrogramAvailable) {
      spectrogramBtn.style.opacity = "0.5";
      spectrogramBtn.style.cursor = "not-allowed";
      spectrogramOpacity.disabled = true;
      spectrogramOpacity.style.opacity = "0.5";
    }
  }

  // Initialize
  initTimelineTicks();
  initSpectrogramControls();
  initPass3StateViewSelector();
  initStateStripHover();
  setTimeout(initPlotlyIntegration, 500);

  // DEBUG: Check for audio file presence relative to HTML
  const debugAudioPath =
    AUDIO_SOURCES[currentAudioKey] || AUDIO_SOURCES[DEFAULT_AUDIO_KEY] || "";
  if (debugAudioPath) {
    console.log("📂 Checking for audio file in same directory...", debugAudioPath);
    fetch("./" + decodeURIComponent(debugAudioPath), { method: "HEAD" })
      .then((response) => {
        if (response.ok) {
          console.log("✅ Audio file found at expected location!");
        } else {
          console.error("❌ Audio file NOT found at expected location");
        }
      })
      .catch((err) => {
        console.error("❌ Cannot access audio file:", err);
        console.log(
          "💡 If you're using file:// protocol, try running a local server instead:"
        );
        console.log("   python -m http.server 8000");
      });
  } else {
    console.warn("⚠️ No audio file specified for HEAD check.");
  }
})();


