// assets/interactive_plot.js
// JavaScript logic for the interactive BPM analysis HTML output.
// This script expects a global configuration object:
//   window.BPM_ANALYZER_CONFIG = {
//       totalDuration: number,
//       spectrogramSources: { original: string, filtered: string, filtered_inverse?: string },
//       spectrogramAvailable: { original: boolean, filtered: boolean, filtered_inverse?: boolean },
//       audioSources: { original: string, filtered: string, filtered_inverse?: string },
//       audioLabels: { original: string, filtered: string, filtered_inverse?: string },
//       htmlS1S2HoverOnByDefault: boolean,   // optional; default false
//       bpmIntervalParams: { s1_nominal_sec, s2_nominal_sec, weissler_ref_et_ms, ... }
//   };

(function () {
  const cfg = window.BPM_ANALYZER_CONFIG || {};
  const TOTAL_DURATION = cfg.totalDuration || 0;
  // Match Python's naive datetime epoch (1970-01-01 00:00:00 local time),
  // not UTC epoch millis, so strip overlays/tooltips align with x-axis values.
  const EPOCH = new Date(1970, 0, 1, 0, 0, 0, 0);
  const SPECTROGRAM_SOURCES = cfg.spectrogramSources || {};
  const SPECTROGRAM_AVAILABLE = cfg.spectrogramAvailable || {};
  const AUDIO_SOURCES = cfg.audioSources || {};
  const AUDIO_LABELS = cfg.audioLabels || {};
  const ANALYSIS_SUMMARY = typeof cfg.analysisSummary === "string" ? cfg.analysisSummary : "";
  const beatHoverDefaultOn = cfg.htmlS1S2HoverOnByDefault === true;
  const BPM_INTERVAL_PARAMS = cfg.bpmIntervalParams || {};
  const PASS3_SEGMENTS_AFTER = Array.isArray(cfg.pass3SegmentsAfter)
    ? cfg.pass3SegmentsAfter
    : Array.isArray(cfg.pass3Segments)
      ? cfg.pass3Segments
      : [];
  const PASS3_SEGMENTS_BEFORE = Array.isArray(cfg.pass3SegmentsBefore) ? cfg.pass3SegmentsBefore : [];
  const PASS3_SEGMENTS_DEFAULT_VIEW =
    typeof cfg.pass3SegmentsDefaultView === "string" ? cfg.pass3SegmentsDefaultView : "after";
  let pass3SegmentsActive = PASS3_SEGMENTS_DEFAULT_VIEW === "before" ? PASS3_SEGMENTS_BEFORE : PASS3_SEGMENTS_AFTER;

  function pass3SegmentsToManual(segments) {
    const validStates = new Set(["S1", "S2", "systole", "diastole", "noisy"]);
    const out = [];
    for (const seg of segments || []) {
      if (!seg) continue;
      const startSec = typeof seg.start === "number" ? seg.start : parseFloat(seg.start);
      const endSec = typeof seg.end === "number" ? seg.end : parseFloat(seg.end);
      const state = typeof seg.state === "string" ? seg.state : String(seg.state || "");
      if (!Number.isFinite(startSec) || !Number.isFinite(endSec) || endSec <= startSec) continue;
      if (state === "unknown" || state === "gap") continue;
      if (!validStates.has(state)) continue;
      out.push({ start_sec: startSec, end_sec: endSec, state, source: "auto", bpm_at_mid: null });
    }
    return out.sort((a, b) => a.start_sec - b.start_sec);
  }

  const _initialPass3ForManual =
    PASS3_SEGMENTS_DEFAULT_VIEW === "before" && PASS3_SEGMENTS_BEFORE.length > 0
      ? PASS3_SEGMENTS_BEFORE
      : PASS3_SEGMENTS_AFTER;

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
  const cardiacStateStripCanvas = document.getElementById("cardiac-state-strip-plot");
  const cardiacStateStripTooltip = document.getElementById("cardiac-state-strip-tooltip");
  const noiseStateStripCanvas = document.getElementById("noise-state-strip-plot");
  const noiseStateStripTooltip = document.getElementById("noise-state-strip-tooltip");
  const manualStateStripCanvas = document.getElementById("manual-state-strip-plot");
  const manualStateStripTooltip = document.getElementById("manual-state-strip-tooltip");
  const _rawNoiseSegs = Array.isArray(cfg.noiseEventSegments) ? cfg.noiseEventSegments : [];
  const _rawGapSegs = Array.isArray(cfg.pass3LargeGapSegments) ? cfg.pass3LargeGapSegments : [];
  const _rawGapQuietSegs = Array.isArray(cfg.pass3GapQuietSegments) ? cfg.pass3GapQuietSegments : [];
  const NOISE_EVENT_SEGMENTS = _rawNoiseSegs
    .slice()
    .sort((a, b) => {
      const s0 = typeof a.start === "number" ? a.start : parseFloat(a.start);
      const s1 = typeof b.start === "number" ? b.start : parseFloat(b.start);
      return s0 - s1;
    });
  const PASS3_LARGE_GAP_SEGMENTS = _rawGapSegs
    .slice()
    .sort((a, b) => {
      const s0 = typeof a.start === "number" ? a.start : parseFloat(a.start);
      const s1 = typeof b.start === "number" ? b.start : parseFloat(b.start);
      return s0 - s1;
    });
  const PASS3_GAP_QUIET_SEGMENTS = _rawGapQuietSegs
    .slice()
    .sort((a, b) => {
      const s0 = typeof a.start === "number" ? a.start : parseFloat(a.start);
      const s1 = typeof b.start === "number" ? b.start : parseFloat(b.start);
      return s0 - s1;
    });
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
  const regenerateStatesBtn = document.getElementById("regenerate-states-btn");

  const DEFAULT_AUDIO_KEY = "original";
  let currentAudioKey = DEFAULT_AUDIO_KEY;
  if (audioFileNameEl) {
    audioFileNameEl.dataset.defaultName = audioFileNameEl.textContent || "";
  }

  function hasPlaybackAudio() {
    if (!AUDIO_SOURCES || typeof AUDIO_SOURCES !== "object") return false;
    const orig = AUDIO_SOURCES.original;
    const filt = AUDIO_SOURCES.filtered;
    const inv = AUDIO_SOURCES.filtered_inverse;
    return (
      (typeof orig === "string" && orig.trim() !== "") ||
      (typeof filt === "string" && filt.trim() !== "") ||
      (typeof inv === "string" && inv.trim() !== "")
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
  let fullXAxisRange = null;
  // Manual state segments: [{start_sec, end_sec, state, source, bpm_at_mid}, ...]
  // Seeded from Pass 3 algorithm labels (source "auto"); user edits use source "manual".
  //
  // _autoBaseline is an immutable reference — never mutated, never cloned.
  // Undo/redo stacks store commands (not snapshots), so each entry is O(1) or O(local).
  // Fills (systole/diastole) are always re-derived and never stored in commands.
  const _autoBaseline = pass3SegmentsToManual(_initialPass3ForManual);
  let manualStateSegments = _autoBaseline.map((s) => ({ ...s }));
  let manualStripEdited = false;
  // Each stack entry: { undo: cmd, redo: cmd }
  // Commands: "place" | "unplace" | "remove" | "restore" | "snapshot"
  const manualStateUndoStack = [];
  const manualStateRedoStack = [];
  const MANUAL_STATE_UNDO_MAX = 50;

  // Rebuild manualStateSegments from the auto baseline + an array of manual edits.
  // Manual segments punch holes in the auto baseline; regenerated fills are re-derived after.
  function _restoreFromSnapshot(snapshot) {
    const autoSegs = _autoBaseline
      .filter((a) => !snapshot.some((m) => m.end_sec > a.start_sec && m.start_sec < a.end_sec))
      .map((s) => ({ ...s }));
    manualStateSegments = [...autoSegs, ...snapshot].sort((a, b) => a.start_sec - b.start_sec);
    rebuildRegenGaps();
  }

  // Re-add auto baseline segments in [start, end] not covered by any current manual segment.
  // Called after undoing a place to restore auto segments that were displaced.
  function _restoreAutoInRange(start, end) {
    const manual = manualStateSegments.filter((s) => s.source === "manual");
    const toAdd = _autoBaseline
      .filter((a) => a.end_sec > start && a.start_sec < end &&
        !manual.some((m) => m.end_sec > a.start_sec && m.start_sec < a.end_sec))
      .map((s) => ({ ...s }));
    if (toAdd.length > 0) {
      manualStateSegments = [...manualStateSegments, ...toAdd].sort((a, b) => a.start_sec - b.start_sec);
    }
  }

  // Execute a single undo/redo command against manualStateSegments.
  // Commands:
  //   "place"    — clear [editStart,editEnd], add placed, local regen   (redo of place)
  //   "unplace"  — remove placed, restore displaced+auto, local regen   (undo of place)
  //   "remove"   — remove segment, local regen                          (redo of remove)
  //   "restore"  — add segment back, local regen                        (undo of remove)
  //   "snapshot" — full snapshot restore + full regen (flip, import)
  function _applyCmd(cmd) {
    switch (cmd.type) {
      case "place":
        manualStateSegments = manualStateSegments.filter((s) => s.end_sec <= cmd.editStart || s.start_sec >= cmd.editEnd);
        manualStateSegments.push({ ...cmd.placed });
        manualStateSegments.sort((a, b) => a.start_sec - b.start_sec);
        if (cmd.regenNeeded) rebuildRegenGapsLocal(cmd.editStart, cmd.editEnd);
        break;
      case "unplace":
        manualStateSegments = manualStateSegments.filter((s) =>
          !(s.source === "manual" &&
            Math.abs(s.start_sec - cmd.placed.start_sec) < 1e-9 &&
            Math.abs(s.end_sec   - cmd.placed.end_sec)   < 1e-9 &&
            s.state === cmd.placed.state)
        );
        manualStateSegments.push(...cmd.displaced.map((s) => ({ ...s })));
        manualStateSegments.sort((a, b) => a.start_sec - b.start_sec);
        _restoreAutoInRange(cmd.editStart, cmd.editEnd);
        if (cmd.regenNeeded) rebuildRegenGapsLocal(cmd.editStart, cmd.editEnd);
        break;
      case "remove":
        manualStateSegments = manualStateSegments.filter((s) =>
          !(s.source === "manual" &&
            Math.abs(s.start_sec - cmd.segment.start_sec) < 1e-9 &&
            Math.abs(s.end_sec   - cmd.segment.end_sec)   < 1e-9 &&
            s.state === cmd.segment.state)
        );
        if (cmd.regenNeeded) rebuildRegenGapsLocal(cmd.editStart, cmd.editEnd);
        break;
      case "restore":
        manualStateSegments = [...manualStateSegments, { ...cmd.segment }].sort((a, b) => a.start_sec - b.start_sec);
        if (cmd.regenNeeded) rebuildRegenGapsLocal(cmd.editStart, cmd.editEnd);
        break;
      case "snapshot":
        _restoreFromSnapshot(cmd.snapshot);
        if (!cmd.snapshot.length) manualStripEdited = false;
        break;
    }
  }

  // Record an edit as a {undo, redo} command pair. Clears the redo stack.
  function _recordEdit(undoCmd, redoCmd) {
    manualStateUndoStack.push({ undo: undoCmd, redo: redoCmd });
    if (manualStateUndoStack.length > MANUAL_STATE_UNDO_MAX) manualStateUndoStack.shift();
    manualStateRedoStack.length = 0;
  }

  // Legacy: used by flip and regenerate (snapshot-based, rare operations).
  function _snapshotManualEdits() {
    return manualStateSegments
      .filter((s) => s.source === "manual" && s.state !== "systole" && s.state !== "diastole")
      .map((s) => ({ ...s }));
  }

  function undoManualState() {
    if (!manualStateUndoStack.length) return;
    const { undo, redo } = manualStateUndoStack.pop();
    manualStateRedoStack.push({ undo, redo });
    if (manualStateRedoStack.length > MANUAL_STATE_UNDO_MAX) manualStateRedoStack.shift();
    _applyCmd(undo);
    if (!manualStateUndoStack.length) manualStripEdited = false;
    scheduleDrawPass3StateStrip();
    console.log(`Undo: ${undo.type}`);
  }

  function redoManualState() {
    if (!manualStateRedoStack.length) return;
    const { undo, redo } = manualStateRedoStack.pop();
    manualStateUndoStack.push({ undo, redo });
    if (manualStateUndoStack.length > MANUAL_STATE_UNDO_MAX) manualStateUndoStack.shift();
    _applyCmd(redo);
    manualStripEdited = true;
    scheduleDrawPass3StateStrip();
    console.log(`Redo: ${redo.type}`);
  }

  function revealManualStateStrip() {
    manualStripEdited = true;
  }

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

  // Inverse of getXPositionForTime: chart-container px -> seconds.
  function getTimeForXPosition(px) {
    if (!plotlyGraphDiv || !plotlyGraphDiv._fullLayout || !xAxisRange) return null;
    const xaxis = plotlyGraphDiv._fullLayout.xaxis;
    if (!xaxis) return null;
    const plotLeft = xaxis._offset;
    const plotWidth = xaxis._length;
    if (!plotWidth) return null;
    const ratio = (px - plotLeft) / plotWidth;
    const xMin = new Date(xAxisRange[0]).getTime();
    const xMax = new Date(xAxisRange[1]).getTime();
    const xTime = xMin + ratio * (xMax - xMin);
    return (xTime - EPOCH.getTime()) / 1000;
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
  const STATE_STRIP_HEIGHT = 10;
  const STATE_STRIP_GAP = 2;
  const CARDIAC_STATE_COLORS = {
    S1: "#e36f6f",
    systole: "#666666",
    S2: "#f0a030",
    diastole: "#999999",
    noisy: "#c0392b",
  };
  const MANUAL_S1_S2_SATURATED = { S1: "#e36f6f", S2: "#f0a030" };
  const MANUAL_S1_S2_FADED = { S1: "#633838", S2: "#735028" };

  /** Shared layout: manual row at plot bottom, cardiac directly above, noise above cardiac. */
  function getStateStripLayout() {
    if (!plotlyGraphDiv || !plotlyGraphDiv._fullLayout) return null;
    const fl = plotlyGraphDiv._fullLayout;
    const xaxis = fl.xaxis;
    const yaxis = fl.yaxis;
    if (!xaxis || !yaxis) return null;
    if (!xAxisRange || xAxisRange.length < 2) return null;

    const left = Math.floor(xaxis._offset || 0);
    const width = Math.floor(xaxis._length || 0);
    const topPlot = Math.floor(yaxis._offset || 0);
    const heightPlot = Math.floor(yaxis._length || 0);
    if (width <= 1 || heightPlot <= 1) return null;

    const manualTop = topPlot + heightPlot - STATE_STRIP_HEIGHT - 1;
    const cardiacTop = manualTop - STATE_STRIP_GAP - STATE_STRIP_HEIGHT;
    const noiseTop = cardiacTop - STATE_STRIP_GAP - STATE_STRIP_HEIGHT;

    return { left, width, manualTop, cardiacTop, noiseTop };
  }

  function scheduleDrawPass3StateStrip() {
    if (pass3StripRaf) return;
    pass3StripRaf = window.requestAnimationFrame(() => {
      pass3StripRaf = 0;
      drawNoiseStateStrip();
      drawPass3CardiacStateStrip();
      drawManualStateStrip();
    });
  }

  function drawPass3CardiacStateStrip() {
    if (!cardiacStateStripCanvas || !pass3SegmentsActive || pass3SegmentsActive.length === 0) return;
    const layout = getStateStripLayout();
    if (!layout) return;

    const x0ms = new Date(xAxisRange[0]).getTime();
    const x1ms = new Date(xAxisRange[1]).getTime();
    if (!Number.isFinite(x0ms) || !Number.isFinite(x1ms) || x1ms <= x0ms) return;

    const { left, width, cardiacTop: top } = layout;
    const stripHeight = STATE_STRIP_HEIGHT;

    cardiacStateStripCanvas.style.left = `${left}px`;
    cardiacStateStripCanvas.style.top = `${top}px`;
    cardiacStateStripCanvas.style.width = `${width}px`;
    cardiacStateStripCanvas.style.height = `${stripHeight}px`;
    cardiacStateStripCanvas.style.display = "";
    cardiacStateStripCanvas.width = Math.max(1, width);
    cardiacStateStripCanvas.height = Math.max(1, stripHeight);

    const ctx = cardiacStateStripCanvas.getContext("2d");
    if (!ctx) return;

    const colors = CARDIAC_STATE_COLORS;

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
      // Gaps: Python clips segments so nothing is drawn over HF-noise times; unknown is unused in segments.
      if (state === "unknown" || state === "gap") continue;
      ctx.fillStyle = colors[state] || "#555555";
      ctx.fillRect(x, 0, xEnd - x, stripHeight);
    }
  }

  function drawManualStateStrip() {
    if (!manualStateStripCanvas) return;
    if (!manualStripEdited || manualStateSegments.length === 0) {
      manualStateStripCanvas.style.display = "none";
      manualStateStripCanvas.width = 1;
      manualStateStripCanvas.height = 1;
      return;
    }
    const layout = getStateStripLayout();
    if (!layout) return;

    const x0ms = new Date(xAxisRange[0]).getTime();
    const x1ms = new Date(xAxisRange[1]).getTime();
    if (!Number.isFinite(x0ms) || !Number.isFinite(x1ms) || x1ms <= x0ms) return;

    const { left, width, manualTop: top } = layout;
    const stripHeight = STATE_STRIP_HEIGHT;

    manualStateStripCanvas.style.left = `${left}px`;
    manualStateStripCanvas.style.top = `${top}px`;
    manualStateStripCanvas.style.width = `${width}px`;
    manualStateStripCanvas.style.height = `${stripHeight}px`;
    manualStateStripCanvas.style.display = "";
    manualStateStripCanvas.width = Math.max(1, width);
    manualStateStripCanvas.height = Math.max(1, stripHeight);

    const ctx = manualStateStripCanvas.getContext("2d");
    if (!ctx) return;

    function colorForManualSegment(seg) {
      if (seg.state === "systole" || seg.state === "diastole") {
        return CARDIAC_STATE_COLORS[seg.state];
      }
      const beatPalette = seg.source === "manual" ? MANUAL_S1_S2_SATURATED : MANUAL_S1_S2_FADED;
      return beatPalette[seg.state] || CARDIAC_STATE_COLORS[seg.state] || "#555555";
    }

    ctx.clearRect(0, 0, width, stripHeight);
    ctx.fillStyle = "rgba(0,0,0,0.25)";
    ctx.fillRect(0, 0, width, stripHeight);

    const span = x1ms - x0ms;
    for (const seg of manualStateSegments) {
      const s0 = secondsToDatetime(seg.start_sec).getTime();
      const s1 = secondsToDatetime(seg.end_sec).getTime();
      if (s1 <= x0ms || s0 >= x1ms) continue;
      const cl0 = Math.max(x0ms, s0);
      const cl1 = Math.min(x1ms, s1);
      const px0 = Math.floor(((cl0 - x0ms) / span) * width);
      const px1 = Math.ceil(((cl1 - x0ms) / span) * width);
      const x = Math.max(0, Math.min(width, px0));
      const xEnd = Math.max(0, Math.min(width, px1));
      if (xEnd <= x) continue;
      ctx.fillStyle = colorForManualSegment(seg);
      ctx.fillRect(x, 0, xEnd - x, stripHeight);
    }
  }

  function drawNoiseStateStrip() {
    if (!noiseStateStripCanvas || !plotlyGraphDiv || !plotlyGraphDiv._fullLayout) return;
    if (
      (!NOISE_EVENT_SEGMENTS || NOISE_EVENT_SEGMENTS.length === 0) &&
      (!PASS3_LARGE_GAP_SEGMENTS || PASS3_LARGE_GAP_SEGMENTS.length === 0) &&
      (!PASS3_GAP_QUIET_SEGMENTS || PASS3_GAP_QUIET_SEGMENTS.length === 0)
    ) {
      noiseStateStripCanvas.style.display = "none";
      noiseStateStripCanvas.width = 1;
      noiseStateStripCanvas.height = 1;
      return;
    }
    const layout = getStateStripLayout();
    if (!layout) return;

    const x0ms = new Date(xAxisRange[0]).getTime();
    const x1ms = new Date(xAxisRange[1]).getTime();
    if (!Number.isFinite(x0ms) || !Number.isFinite(x1ms) || x1ms <= x0ms) return;

    const { left, width, noiseTop: top } = layout;
    const stripHeight = STATE_STRIP_HEIGHT;

    noiseStateStripCanvas.style.left = `${left}px`;
    noiseStateStripCanvas.style.top = `${top}px`;
    noiseStateStripCanvas.style.width = `${width}px`;
    noiseStateStripCanvas.style.height = `${stripHeight}px`;
    noiseStateStripCanvas.style.display = "";
    noiseStateStripCanvas.width = Math.max(1, width);
    noiseStateStripCanvas.height = Math.max(1, stripHeight);

    const ctx = noiseStateStripCanvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, width, stripHeight);
    ctx.fillStyle = "rgba(0,0,0,0.25)";
    ctx.fillRect(0, 0, width, stripHeight);

    const span = x1ms - x0ms;
    const noiseColor = "#c0392b";
    const gapColor = "#f39c12";
    const gapQuietColor = "#7f8c8d";
    for (const seg of NOISE_EVENT_SEGMENTS) {
      if (!seg) continue;
      const startSec = typeof seg.start === "number" ? seg.start : parseFloat(seg.start);
      const endSec = typeof seg.end === "number" ? seg.end : parseFloat(seg.end);
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
      ctx.fillStyle = noiseColor;
      ctx.fillRect(x, 0, xEnd - x, stripHeight);
    }

    for (const seg of PASS3_LARGE_GAP_SEGMENTS) {
      if (!seg) continue;
      const startSec = typeof seg.start === "number" ? seg.start : parseFloat(seg.start);
      const endSec = typeof seg.end === "number" ? seg.end : parseFloat(seg.end);
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
      ctx.fillStyle = gapColor;
      ctx.fillRect(x, 0, xEnd - x, stripHeight);
    }

    for (const seg of PASS3_GAP_QUIET_SEGMENTS) {
      if (!seg) continue;
      const startSec = typeof seg.start === "number" ? seg.start : parseFloat(seg.start);
      const endSec = typeof seg.end === "number" ? seg.end : parseFloat(seg.end);
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
      ctx.fillStyle = gapQuietColor;
      ctx.fillRect(x, 0, xEnd - x, stripHeight);
    }
  }

  // -------------------------------------------------------------------------
  // Cardiac / noise strip hover tooltips
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
    const labelMap = { S1: "S1", systole: "Systole", S2: "S2", diastole: "Diastole", noisy: "Noisy" };
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

  function _noiseSegmentAtTime(tSec) {
    if (!NOISE_EVENT_SEGMENTS || NOISE_EVENT_SEGMENTS.length === 0) return null;
    let lo = 0;
    let hi = NOISE_EVENT_SEGMENTS.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const seg = NOISE_EVENT_SEGMENTS[mid];
      const s0 = typeof seg.start === "number" ? seg.start : parseFloat(seg.start);
      const s1 = typeof seg.end === "number" ? seg.end : parseFloat(seg.end);
      if (tSec < s0) hi = mid - 1;
      else if (tSec >= s1) lo = mid + 1;
      else return seg;
    }
    return null;
  }

  function _gapSegmentAtTime(tSec) {
    if (!PASS3_LARGE_GAP_SEGMENTS || PASS3_LARGE_GAP_SEGMENTS.length === 0) return null;
    let lo = 0;
    let hi = PASS3_LARGE_GAP_SEGMENTS.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const seg = PASS3_LARGE_GAP_SEGMENTS[mid];
      if (tSec < seg.start) { hi = mid - 1; }
      else if (tSec >= seg.end) { lo = mid + 1; }
      else { return seg; }
    }
    return null;
  }

  function _gapQuietSegmentAtTime(tSec) {
    if (!PASS3_GAP_QUIET_SEGMENTS || PASS3_GAP_QUIET_SEGMENTS.length === 0) return null;
    let lo = 0;
    let hi = PASS3_GAP_QUIET_SEGMENTS.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const seg = PASS3_GAP_QUIET_SEGMENTS[mid];
      if (tSec < seg.start) { hi = mid - 1; }
      else if (tSec >= seg.end) { lo = mid + 1; }
      else { return seg; }
    }
    return null;
  }

  function _formatNoiseTooltip(seg) {
    if (!seg) return null;
    const t0 = typeof seg.start === "number" ? seg.start.toFixed(2) : "?";
    const t1 = typeof seg.end === "number" ? seg.end.toFixed(2) : "?";
    const dur =
      seg.duration_ms !== undefined && seg.duration_ms !== null
        ? String(seg.duration_ms)
        : "";
    const peak = seg.peak !== undefined && seg.peak !== null ? String(seg.peak) : "";
    const mean = seg.mean !== undefined && seg.mean !== null ? String(seg.mean) : "";
    const lines = [`Noisy segment (HF envelope)`, `${t0}s – ${t1}s`];
    if (dur) lines.push(`Duration: ${dur} ms`);
    if (peak) lines.push(`Peak noise env: ${peak}`);
    if (mean) lines.push(`Mean noise env: ${mean}`);
    if (seg.threshold !== undefined && seg.threshold !== null)
      lines.push(`HF env (quantile): ≥ ${seg.threshold}`);
    if (
      seg.min_amplitude_gate !== undefined &&
      seg.min_amplitude_gate !== null
    )
      lines.push(`HF env (floor): > ${seg.min_amplitude_gate}`);
    return lines.join("\n");
  }

  /** Position fixed strip tooltips above the cursor (bottom edge near clientY − gap), using measured size. */
  function _placeStripTooltipAboveCursor(el, clientX, clientY) {
    const gap = 10;
    const margin = 8;
    el.classList.remove("hidden");
    void el.offsetHeight;
    const r = el.getBoundingClientRect();
    let tx = clientX + 14;
    if (tx + r.width > window.innerWidth - margin) tx = clientX - r.width - 14;
    tx = Math.max(margin, Math.min(tx, window.innerWidth - r.width - margin));
    const tyIdeal = clientY - r.height - gap;
    const tyMax = Math.max(margin, window.innerHeight - r.height - margin);
    const ty = Math.max(margin, Math.min(tyIdeal, tyMax));
    el.style.left = `${tx}px`;
    el.style.top = `${ty}px`;
  }

  function _formatGapTooltip(seg) {
    if (!seg) return null;
    const t0 = typeof seg.start === "number" ? seg.start.toFixed(2) : "?";
    const t1 = typeof seg.end === "number" ? seg.end.toFixed(2) : "?";
    const lines = [`Gap region`, `${t0}s – ${t1}s`];
    if (seg.gap_region_candidate_state) lines.push(`Gap region candidate: ${seg.gap_region_candidate_state}`);
    else if (seg.source_state) lines.push(`Source state: ${seg.source_state}`);
    if (seg.bpm_at_mid !== undefined && seg.bpm_at_mid !== null) lines.push(`BPM@mid: ${Number(seg.bpm_at_mid).toFixed(1)}`);
    if (seg.cycle0_samples !== undefined && seg.cycle0_samples !== null)
      lines.push(`Nominal cycle: ${seg.cycle0_samples} samples`);
    if (seg.segment_samples !== undefined && seg.segment_samples !== null)
      lines.push(`Segment: ${seg.segment_samples} samples`);
    return lines.join("\n");
  }

  function _formatGapQuietTooltip(seg) {
    if (!seg) return null;
    const t0 = typeof seg.start === "number" ? seg.start.toFixed(2) : "?";
    const t1 = typeof seg.end === "number" ? seg.end.toFixed(2) : "?";
    const lines = [`Quiet (trimmed from gap region)`, `${t0}s – ${t1}s`];
    if (seg.gap_region_candidate_state) lines.push(`Gap region candidate: ${seg.gap_region_candidate_state}`);
    if (seg.trigger) lines.push(`Trigger: ${seg.trigger}`);
    return lines.join("\n");
  }

  function initCardiacAndNoiseStripHovers() {
    if (cardiacStateStripCanvas && cardiacStateStripTooltip) {
      cardiacStateStripCanvas.addEventListener("mousemove", (e) => {
        if (!xAxisRange || xAxisRange.length < 2) return;
        const rect = cardiacStateStripCanvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const w = rect.width || 1;
        const x0ms = new Date(xAxisRange[0]).getTime();
        const x1ms = new Date(xAxisRange[1]).getTime();
        const tSec = (x0ms + (mouseX / w) * (x1ms - x0ms) - EPOCH.getTime()) / 1000;
        const seg = _segmentAtTime(tSec);
        const text = _formatTooltip(seg);
        if (!text) {
          cardiacStateStripTooltip.classList.add("hidden");
          return;
        }
        cardiacStateStripTooltip.textContent = text;
        _placeStripTooltipAboveCursor(cardiacStateStripTooltip, e.clientX, e.clientY);
      });
      cardiacStateStripCanvas.addEventListener("mouseleave", () => {
        cardiacStateStripTooltip.classList.add("hidden");
      });
    }

    if (noiseStateStripCanvas && noiseStateStripTooltip) {
      noiseStateStripCanvas.addEventListener("mousemove", (e) => {
        if (!xAxisRange || xAxisRange.length < 2) return;
        const rect = noiseStateStripCanvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const w = rect.width || 1;
        const x0ms = new Date(xAxisRange[0]).getTime();
        const x1ms = new Date(xAxisRange[1]).getTime();
        const tSec = (x0ms + (mouseX / w) * (x1ms - x0ms) - EPOCH.getTime()) / 1000;
        const gapSeg = _gapSegmentAtTime(tSec);
        const quietSeg = _gapQuietSegmentAtTime(tSec);
        const noiseSeg = _noiseSegmentAtTime(tSec);
        const text = _formatGapTooltip(gapSeg) || _formatGapQuietTooltip(quietSeg) || _formatNoiseTooltip(noiseSeg);
        if (!text) {
          noiseStateStripTooltip.classList.add("hidden");
          return;
        }
        noiseStateStripTooltip.textContent = text;
        _placeStripTooltipAboveCursor(noiseStateStripTooltip, e.clientX, e.clientY);
      });
      noiseStateStripCanvas.addEventListener("mouseleave", () => {
        noiseStateStripTooltip.classList.add("hidden");
      });
    }

    if (manualStateStripCanvas && manualStateStripTooltip) {
      manualStateStripCanvas.addEventListener("mousemove", (e) => {
        if (!xAxisRange || xAxisRange.length < 2) return;
        const rect = manualStateStripCanvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const w = rect.width || 1;
        const x0ms = new Date(xAxisRange[0]).getTime();
        const x1ms = new Date(xAxisRange[1]).getTime();
        const tSec = (x0ms + (mouseX / w) * (x1ms - x0ms) - EPOCH.getTime()) / 1000;
        const seg = manualStateSegments.find((s) => tSec >= s.start_sec && tSec < s.end_sec) || null;
        if (!seg) {
          manualStateStripTooltip.classList.add("hidden");
          return;
        }
        const durMs = Math.round((seg.end_sec - seg.start_sec) * 1000);
        const lines = [
          `${seg.state}`,
          `${seg.start_sec.toFixed(3)}s → ${seg.end_sec.toFixed(3)}s  (${durMs}ms)`,
          `Source: ${seg.source}`,
        ];
        const bpm = seg.bpm_at_mid ?? getBpmSmoothedAtTime((seg.start_sec + seg.end_sec) / 2);
        if (bpm && Number.isFinite(bpm)) lines.push(`BPM: ${bpm.toFixed(1)}`);
        manualStateStripTooltip.textContent = lines.join("\n");
        _placeStripTooltipAboveCursor(manualStateStripTooltip, e.clientX, e.clientY);
      });
      manualStateStripCanvas.addEventListener("mouseleave", () => {
        manualStateStripTooltip.classList.add("hidden");
      });
    }
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
    // Re-evaluate loop gate: seeking right of the region disengages it.
    loopEngaged = loopActive && audio.currentTime < loopEndSec;
    updatePlayhead(audio.currentTime);
  }

  // ----- Loop region (FL Studio / Edison style) -----
  // Shift+drag on the chart selects a region; playback loops within it.
  // Shift+click (no drag) clears the selection.
  let loopActive = false;
  let loopEngaged = false; // false = play through without wrapping (playhead started right of region)
  let loopStartSec = 0;
  let loopEndSec = 0;
  let loopFillEl = null;
  let loopStartMarkerEl = null;
  let loopEndMarkerEl = null;

  function ensureLoopRegionEls() {
    if (loopFillEl || !chartContainer) return;
    loopFillEl = document.createElement("div");
    loopFillEl.className = "loop-region-fill";
    loopStartMarkerEl = document.createElement("div");
    loopStartMarkerEl.className = "loop-region-marker";
    loopEndMarkerEl = document.createElement("div");
    loopEndMarkerEl.className = "loop-region-marker";
    chartContainer.appendChild(loopFillEl);
    chartContainer.appendChild(loopStartMarkerEl);
    chartContainer.appendChild(loopEndMarkerEl);
  }

  function positionLoopRegion() {
    if (!loopFillEl) return;
    const hide = !loopActive || !plotlyGraphDiv || !plotlyGraphDiv._fullLayout;
    const xStart = hide ? null : getXPositionForTime(loopStartSec);
    const xEnd = hide ? null : getXPositionForTime(loopEndSec);
    if (xStart === null || xEnd === null) {
      loopFillEl.style.display = "none";
      loopStartMarkerEl.style.display = "none";
      loopEndMarkerEl.style.display = "none";
      return;
    }
    const left = Math.min(xStart, xEnd);
    const right = Math.max(xStart, xEnd);
    loopFillEl.style.display = "block";
    loopFillEl.style.left = left + "px";
    loopFillEl.style.width = right - left + "px";
    loopStartMarkerEl.style.display = "block";
    loopStartMarkerEl.style.left = left + "px";
    loopEndMarkerEl.style.display = "block";
    loopEndMarkerEl.style.left = right + "px";
  }

  function setLoopRegion(aSec, bSec) {
    loopStartSec = Math.max(0, Math.min(aSec, bSec));
    loopEndSec = Math.min(TOTAL_DURATION, Math.max(aSec, bSec));
    loopActive = loopEndSec - loopStartSec > 0.02;
    loopEngaged = loopActive && !!audio && audio.currentTime < loopEndSec;
    positionLoopRegion();
  }

  function clearLoopRegion() {
    loopActive = false;
    loopEngaged = false;
    positionLoopRegion();
  }

  // rAF watcher: tighter loop boundary than 4 Hz timeupdate. Self-stops on pause.
  function loopWatch() {
    if (!isPlaying) return;
    if (loopActive && loopEngaged && audio && audio.currentTime >= loopEndSec) {
      audio.currentTime = loopStartSec;
    }
    requestAnimationFrame(loopWatch);
  }

  function setupLoopRegionDrag() {
    if (!plotlyGraphDiv || !chartContainer) return;
    let dragging = false;
    let dragStartSec = 0;
    let dragStartX = 0;
    let moved = false;

    function clientXToSec(clientX) {
      const rect = chartContainer.getBoundingClientRect();
      const sec = getTimeForXPosition(clientX - rect.left);
      if (sec === null) return null;
      return Math.max(0, Math.min(TOTAL_DURATION, sec));
    }

    function onMove(ev) {
      if (!dragging) return;
      if (Math.abs(ev.clientX - dragStartX) > 3) moved = true;
      const cur = clientXToSec(ev.clientX);
      if (cur !== null) setLoopRegion(dragStartSec, cur);
    }
    function onUp() {
      if (!dragging) return;
      dragging = false;
      document.removeEventListener("mousemove", onMove, true);
      document.removeEventListener("mouseup", onUp, true);
      if (!moved) clearLoopRegion(); // shift-click clears
    }

    // Capture phase on the graph div: fires before Plotly's drag-layer
    // handlers, so stopPropagation blocks the built-in shift-zoom/select.
    plotlyGraphDiv.addEventListener(
      "mousedown",
      function (e) {
        if (!e.shiftKey || e.button !== 0) return;
        const sec = clientXToSec(e.clientX);
        if (sec === null) return;
        e.preventDefault();
        e.stopPropagation();
        dragging = true;
        moved = false;
        dragStartSec = sec;
        dragStartX = e.clientX;
        document.addEventListener("mousemove", onMove, true);
        document.addEventListener("mouseup", onUp, true);
      },
      true
    );
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
      if (loopActive) {
        if (audio.currentTime < loopEndSec) {
          // Left of or inside region: snap to start, loop normally.
          audio.currentTime = loopStartSec;
          updatePlayhead(audio.currentTime);
          loopEngaged = true;
        } else {
          // Right of region: snap back to loop start.
          audio.currentTime = loopStartSec;
          updatePlayhead(audio.currentTime);
          loopEngaged = true;
        }
      }
      audio.play().catch((e) => console.log("Audio play error:", e));
      if (playBtn) {
        playBtn.textContent = "⏸ Pause";
        playBtn.classList.add("active");
      }
      requestAnimationFrame(loopWatch);
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
  // Audience per trace comes from the Python TRACE_AUDIENCE registry (injected via config),
  // so categorization lives in one place. Unknown/unlisted names default to "debug" so a new
  // trace never silently leaks into the end-user Analysis Data view.
  const TRACE_AUDIENCE = cfg.traceAudience || {};
  function audienceOf(name) {
    return TRACE_AUDIENCE[name] || "debug";
  }
  function isDebugName(name) {
    return audienceOf(name) === "debug";
  }
  function isBothName(name) {
    return audienceOf(name) === "both";
  }

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
      const isDebug = isDebugName(name);
      const inBoth = isBothName(name);
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
      const isDebug = isDebugName(name);
      const defaultState = getDefaultForTrace(i);
      if (value === "all") {
        visibility.push(defaultState.visible);
        showlegend.push(defaultState.showlegend);
      } else if (value === "debug") {
        const show = isDebug || isBothName(name);
        if (show) {
          visibility.push(defaultState.visible);
          showlegend.push(defaultState.showlegend);
        } else {
          visibility.push(false);
          showlegend.push(false);
        }
      } else {
        const show = !isDebug || isBothName(name);
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

  // Apply the configured default view ("debug" on final pass) once, after defaults are snapshotted.
  let defaultLegendViewApplied = false;
  function applyDefaultLegendViewOnce() {
    if (defaultLegendViewApplied || !legendCategoryInitialState) return;
    const sel = document.getElementById("legend-category-filter");
    if (!sel) return; // no selector (non-final pass) -> leave every trace as-is
    const view = cfg.defaultLegendView || "all";
    defaultLegendViewApplied = true;
    sel.value = view;
    applyLegendCategoryFilter(view);
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

  // ─── State labeling ────────────────────────────────────────────────────────

  // Port of confidence_engine.calculate_bpm_intervals (nominal S1/S2/systole/diastole durations).
  function calcPhaseDurations(bpm) {
    const p = BPM_INTERVAL_PARAMS;
    bpm = Math.max(typeof bpm === "number" && Number.isFinite(bpm) ? bpm : 70, 1e-6);
    const rr = 60.0 / bpm;
    const refEtMs = typeof p.weissler_ref_et_ms === "number" ? p.weissler_ref_et_ms : 300;
    const refBpm  = typeof p.weissler_ref_bpm === "number" ? p.weissler_ref_bpm : 60;
    const slope   = typeof p.weissler_slope_ms_per_bpm === "number" ? p.weissler_slope_ms_per_bpm : 1.0;
    const minAbs  = typeof p.min_s1_s2_interval_sec === "number" ? p.min_s1_s2_interval_sec : 0.15;
    const capAbs  = typeof p.s1_s2_interval_cap_sec === "number" ? p.s1_s2_interval_cap_sec : 0.4;
    const sys = Math.min(capAbs, Math.max(minAbs, (refEtMs - slope * (bpm - refBpm)) / 1000));
    return {
      s1:       typeof p.s1_nominal_sec === "number" ? p.s1_nominal_sec : 0.080,
      s2:       typeof p.s2_nominal_sec === "number" ? p.s2_nominal_sec : 0.080,
      systole:  sys,
      diastole: Math.max(0, rr - sys),
      rr,
    };
  }

  // Place a manual state centered on the current playhead.
  // Removes any existing segment (manual or regenerated) that overlaps the new span.
  function applyManualState(state) {
    if (!audio) return;
    const tCenter = audio.currentTime;
    const bpm = getBpmSmoothedAtTime(tCenter) || 70;
    const durations = calcPhaseDurations(bpm);
    const dur = state === "S1" ? durations.s1 : state === "S2" ? durations.s2 : 0.5;
    const half = dur / 2;
    const start = Math.max(0, tCenter - half);
    const end   = Math.min(TOTAL_DURATION > 0 ? TOTAL_DURATION : 1e9, tCenter + half);

    const regenNeeded = state === "S1" || state === "S2" ||
      manualStateSegments.some((s) => (s.state === "S1" || s.state === "S2") && s.end_sec > start && s.start_sec < end);

    // Capture displaced manual segments before clearing the range (for undo).
    const displaced = manualStateSegments.filter(
      (seg) => seg.source === "manual" && seg.end_sec > start && seg.start_sec < end
    ).map((s) => ({ ...s }));

    const placed = { start_sec: start, end_sec: end, state, source: "manual", bpm_at_mid: bpm };
    _recordEdit(
      { type: "unplace", placed, displaced, regenNeeded, editStart: start, editEnd: end },
      { type: "place",   placed, regenNeeded, editStart: start, editEnd: end }
    );

    manualStateSegments = manualStateSegments.filter(
      (seg) => seg.end_sec <= start || seg.start_sec >= end
    );
    manualStateSegments.push(placed);
    manualStateSegments.sort((a, b) => a.start_sec - b.start_sec);
    finishManualStateEdit(`Manual ${state} placed at t=${tCenter.toFixed(3)}s [${start.toFixed(3)}, ${end.toFixed(3)}]`, regenNeeded, start, end);
  }

  // Remove whichever state segment contains the current playhead.
  function removeStateAtPlayhead() {
    if (!audio) return;
    const t = audio.currentTime;
    const seg = manualStateSegments.find((s) => t >= s.start_sec && t < s.end_sec) || null;
    if (!seg) return;

    const regenNeeded = seg.state === "S1" || seg.state === "S2";
    const storedSeg = { ...seg };
    _recordEdit(
      { type: "restore", segment: storedSeg, regenNeeded, editStart: seg.start_sec, editEnd: seg.end_sec },
      { type: "remove",  segment: storedSeg, regenNeeded, editStart: seg.start_sec, editEnd: seg.end_sec }
    );
    manualStateSegments = manualStateSegments.filter((s) => s !== seg);
    finishManualStateEdit(`Removed ${seg.state} at t=${t.toFixed(3)}s [${seg.start_sec.toFixed(3)}, ${seg.end_sec.toFixed(3)}]`, regenNeeded, seg.start_sec, seg.end_sec);
  }

  // Flip S1↔S2 for all anchor segments whose center is >= playhead.
  // Clears regenerated segments right of playhead; gaps are refilled automatically.
  function flipManualStatesRight() {
    if (!audio) return;
    const cutoff = audio.currentTime;
    let flippedCount = 0;

    const snapshotBefore = _snapshotManualEdits();

    // Build new array — use spread instead of in-place mutation to avoid corrupting
    // any shared object references from _autoBaseline.
    manualStateSegments = manualStateSegments
      .filter((seg) => {
        const center = (seg.start_sec + seg.end_sec) / 2;
        return center < cutoff || seg.source !== "regenerated";
      })
      .map((seg) => {
        const center = (seg.start_sec + seg.end_sec) / 2;
        if (center < cutoff || (seg.state !== "S1" && seg.state !== "S2")) return seg;
        flippedCount++;
        return { ...seg, state: seg.state === "S1" ? "S2" : "S1", source: "manual" };
      });

    if (flippedCount === 0) {
      alert("No S1/S2 states to flip to the right of the playhead.");
      return;
    }

    const snapshotAfter = _snapshotManualEdits();
    _recordEdit(
      { type: "snapshot", snapshot: snapshotBefore },
      { type: "snapshot", snapshot: snapshotAfter }
    );
    finishManualStateEdit(`Flipped ${flippedCount} S1/S2 state(s) right of t=${cutoff.toFixed(3)}s`, true, cutoff, Infinity);
  }

  function getRegenAnchors() {
    return manualStateSegments
      .filter((s) => (s.state === "S1" || s.state === "S2") && (s.source === "manual" || s.source === "auto"))
      .sort((a, b) => a.start_sec - b.start_sec);
  }

  // Full file-wide gap rebuild. Used by flip-right, undo/redo restore, and the Regenerate button.
  // S1→S2 gap → systole; S2→S1 gap → diastole. Other pairings left empty.
  function rebuildRegenGaps() {
    const anchors = getRegenAnchors();
    if (anchors.length < 2) {
      manualStateSegments = manualStateSegments.filter((s) => s.source !== "regenerated");
      return { ok: false, anchorCount: anchors.length, fillCount: 0 };
    }

    const newSegs = _buildFills(anchors);

    // Systole/diastole are always re-derived — drop them all before rebuilding.
    const other = manualStateSegments.filter((s) =>
      s.state !== "S1" && s.state !== "S2" &&
      s.state !== "systole" && s.state !== "diastole"
    );
    manualStateSegments = [...anchors, ...newSegs, ...other].sort((a, b) => a.start_sec - b.start_sec);
    return { ok: true, anchorCount: anchors.length, fillCount: newSegs.length };
  }

  // Local gap rebuild around a single edit — only touches the window between the nearest
  // S1/S2 anchors bracketing [editStart, editEnd]. O(local) instead of O(file).
  function rebuildRegenGapsLocal(editStart, editEnd) {
    const allAnchors = getRegenAnchors(); // sorted
    if (allAnchors.length < 2) {
      manualStateSegments = manualStateSegments.filter((s) => s.source !== "regenerated");
      return;
    }

    // Expand window outward to the nearest anchors on each side of the edit.
    let winStart = editStart, winEnd = editEnd;
    for (let i = allAnchors.length - 1; i >= 0; i--) {
      if (allAnchors[i].end_sec <= editStart) { winStart = allAnchors[i].start_sec; break; }
    }
    for (let i = 0; i < allAnchors.length; i++) {
      if (allAnchors[i].start_sec >= editEnd) { winEnd = allAnchors[i].end_sec; break; }
    }

    // Remove all fills inside the window — systole/diastole are always re-derived.
    manualStateSegments = manualStateSegments.filter((s) => {
      const inWin = s.end_sec > winStart && s.start_sec < winEnd;
      if (!inWin) return true;
      if (s.state === "systole" || s.state === "diastole") return false;
      return true;
    });

    // Rebuild fills only for anchors inside the window.
    const winAnchors = allAnchors.filter((a) => a.end_sec > winStart && a.start_sec < winEnd);
    const newSegs = _buildFills(winAnchors);
    if (newSegs.length > 0) {
      manualStateSegments = [...manualStateSegments, ...newSegs].sort((a, b) => a.start_sec - b.start_sec);
    }
  }

  function _buildFills(anchors) {
    const segs = [];
    for (let i = 0; i < anchors.length - 1; i++) {
      const a = anchors[i], b = anchors[i + 1];
      const gapStart = a.end_sec, gapEnd = b.start_sec;
      if (gapEnd <= gapStart) continue;
      let fillState = null;
      if (a.state === "S1" && b.state === "S2") fillState = "systole";
      else if (a.state === "S2" && b.state === "S1") fillState = "diastole";
      if (!fillState) continue;
      const tMid = (gapStart + gapEnd) / 2;
      segs.push({ start_sec: gapStart, end_sec: gapEnd, state: fillState, source: "regenerated", bpm_at_mid: getBpmSmoothedAtTime(tMid) || 70 });
    }
    return segs;
  }

  function applyAutoRegenAfterEdit() {
    if (!manualStripEdited) return;
    const result = rebuildRegenGaps();
    if (result.ok) {
      console.log(`Auto-regenerated ${result.fillCount} gap segment(s) between ${result.anchorCount} anchors.`);
    }
  }

  // editStart/editEnd: when provided, uses local regen (fast); otherwise falls back to
  // full file-wide regen (used by flip-right and other bulk operations).
  function finishManualStateEdit(logMsg, regenNeeded = true, editStart = null, editEnd = null) {
    revealManualStateStrip();
    if (regenNeeded) {
      if (editStart !== null && editEnd !== null) {
        rebuildRegenGapsLocal(editStart, editEnd);
      } else {
        applyAutoRegenAfterEdit();
      }
    }
    scheduleDrawPass3StateStrip();
    if (logMsg) console.log(logMsg);
  }

  // Regenerate button: same rebuild as auto-regen, with undo.
  // Auto-generated labels are truth until the first user edit; after that,
  // remaining auto S1/S2 anchors still count alongside manual placements.
  function regenerateGaps() {
    if (!manualStripEdited) {
      if (!manualStateSegments.length) {
        alert("No state labels available to regenerate.");
        return;
      }
      revealManualStateStrip();
      scheduleDrawPass3StateStrip();
      console.log("Auto-generated labels shown unchanged (truth until first edit).");
      return;
    }

    const anchors = getRegenAnchors();
    if (anchors.length < 2) {
      alert("Need at least 2 S1/S2 anchors to regenerate gaps.");
      return;
    }

    const result = rebuildRegenGaps();
    revealManualStateStrip();
    scheduleDrawPass3StateStrip();
    console.log(`Regenerated ${result.fillCount} gap segment(s) between ${result.anchorCount} anchors.`);
  }

  function downloadStateCsv() {
    if (!manualStateSegments.length) {
      alert("No manual states to export.");
      return;
    }

    // Pre-build a sorted BPM lookup so CSV export is O(N log N) not O(N*M).
    const bpmLookup = (() => {
      if (!plotlyGraphDiv || !plotlyGraphDiv.data) return null;
      const candidateNames = ["BPM (Pass 3)", "BPM (Pass 2)", "BPM (pass 1)", "Average BPM"];
      for (const name of candidateNames) {
        const idx = findTraceIndexByName(name);
        if (idx === null) continue;
        const tr = plotlyGraphDiv.data[idx];
        if (!tr || !tr.x || !tr.y || !tr.x.length) continue;
        const pairs = [];
        const epochMs = EPOCH.getTime();
        for (let i = 0; i < tr.x.length; i++) {
          const xVal = tr.x[i];
          if (!xVal) continue;
          const ms = xVal instanceof Date ? xVal.getTime() : new Date(xVal).getTime();
          if (!Number.isFinite(ms)) continue;
          const tSec = (ms - epochMs) / 1000;
          const yVal = getNumericFromArrayLike(tr.y, i);
          if (Number.isFinite(tSec) && yVal !== null && Number.isFinite(yVal)) {
            pairs.push([tSec, yVal]);
          }
        }
        if (!pairs.length) continue;
        pairs.sort((a, b) => a[0] - b[0]);
        return pairs;
      }
      return null;
    })();

    const bpmAtTimeFast = bpmLookup
      ? (timeSec) => {
          // Binary search nearest point.
          let lo = 0, hi = bpmLookup.length - 1;
          while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (bpmLookup[mid][0] < timeSec) lo = mid + 1; else hi = mid;
          }
          // Check lo and lo-1 for nearest.
          let best = bpmLookup[lo][1];
          if (lo > 0 && Math.abs(bpmLookup[lo - 1][0] - timeSec) < Math.abs(bpmLookup[lo][0] - timeSec)) {
            best = bpmLookup[lo - 1][1];
          }
          return Number.isFinite(best) ? best : null;
        }
      : getBpmSmoothedAtTime;

    const header = "start_sec,end_sec,state,source,bpm_at_mid\n";
    const stateRows = [...manualStateSegments]
      .sort((a, b) => a.start_sec - b.start_sec)
      .map((seg) => {
        const bpm = Number.isFinite(seg.bpm_at_mid)
          ? seg.bpm_at_mid
          : bpmAtTimeFast((seg.start_sec + seg.end_sec) / 2);
        return [
          seg.start_sec.toFixed(3),
          seg.end_sec.toFixed(3),
          seg.state,
          seg.source,
          Number.isFinite(bpm) ? bpm.toFixed(3) : "",
        ].join(",");
      });

    const csvContent = header + stateRows.join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url  = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const baseName = (audioFileNameEl && audioFileNameEl.dataset && audioFileNameEl.dataset.defaultName) || "analysis";
    link.href = url;
    link.download = `${baseName}_manual_state_sequence.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  const MANUAL_BPM_TRACE_NAME = "BPM (Manual)";

  // Gaussian kernel regression matching hrv.py:_gaussian_kernel_smooth.
  // O(n²) — fine for typical S1 counts (~few hundred beats).
  function _gaussianKernelSmooth(tEvals, tData, yData, sigma) {
    if (!tData.length || tData.length !== yData.length) return tEvals.map(() => NaN);
    if (sigma <= 1e-9) {
      // Nearest-neighbor fallback.
      return tEvals.map((t) => {
        let bestI = 0, bestDt = Infinity;
        for (let i = 0; i < tData.length; i++) {
          const dt = Math.abs(tData[i] - t);
          if (dt < bestDt) { bestDt = dt; bestI = i; }
        }
        return yData[bestI];
      });
    }
    const mean = yData.reduce((a, b) => a + b, 0) / yData.length;
    return tEvals.map((t) => {
      let wsum = 0, wbpm = 0;
      for (let i = 0; i < tData.length; i++) {
        const d = (tData[i] - t) / sigma;
        const w = Math.exp(-0.5 * d * d);
        wsum += w;
        wbpm += w * yData[i];
      }
      return wsum > 1e-12 ? wbpm / wsum : mean;
    });
  }

  function computeBpmFromManualSegs() {
    const s1s = manualStateSegments
      .filter((s) => s.state === "S1")
      .sort((a, b) => a.start_sec - b.start_sec);
    if (s1s.length < 2) return null;

    const tData = [], bpmRaw = [];
    for (let i = 0; i < s1s.length - 1; i++) {
      const t0 = (s1s[i].start_sec + s1s[i].end_sec) / 2;
      const t1 = (s1s[i + 1].start_sec + s1s[i + 1].end_sec) / 2;
      const interval = t1 - t0;
      if (interval <= 0) continue;
      tData.push(t1); // BPM timestamped at second beat — matches Python convention
      bpmRaw.push(60 / interval);
    }
    if (!tData.length) return null;

    const smoothingWindowSec = BPM_INTERVAL_PARAMS.output_smoothing_window_sec ?? 3;
    const sigma = Math.max(0.05, smoothingWindowSec / 3.0);
    const smoothed = _gaussianKernelSmooth(tData, tData, bpmRaw, sigma);

    const epochMs = EPOCH.getTime();
    return {
      x: tData.map((t) => new Date(epochMs + t * 1000)),
      y: smoothed,
    };
  }

  function updateManualBpmTrace() {
    if (!plotlyGraphDiv) return;
    const bpmData = computeBpmFromManualSegs();
    const existingIdx = findTraceIndexByName(MANUAL_BPM_TRACE_NAME);
    if (!bpmData) {
      if (existingIdx !== null) Plotly.deleteTraces(plotlyGraphDiv, existingIdx);
      return;
    }
    if (existingIdx !== null) {
      Plotly.restyle(plotlyGraphDiv, { x: [bpmData.x], y: [bpmData.y] }, existingIdx);
    } else {
      TRACE_AUDIENCE[MANUAL_BPM_TRACE_NAME] = "both";
      Plotly.addTraces(plotlyGraphDiv, {
        x: bpmData.x,
        y: bpmData.y,
        name: MANUAL_BPM_TRACE_NAME,
        type: "scatter",
        mode: "lines+markers",
        line: { color: "#00e5a0", width: 2 },
        marker: { color: "#00e5a0", size: 5 },
        yaxis: "y2",
        hovertemplate: "%{y:.1f} BPM<extra>BPM (Manual)</extra>",
      });
    }
  }

  // Import replaces the entire manual state timeline.
  function importStateCsv(csvText) {
    if (!csvText) {
      alert("No CSV content to import.");
      return;
    }
    const lines = csvText.split(/\r?\n/).filter((l) => l.trim().length > 0);
    if (lines.length <= 1) {
      alert("CSV appears to be empty or missing data rows.");
      return;
    }
    const headerCells = lines[0].split(",");
    const lower = headerCells.map((h) => h.trim().toLowerCase());
    const iStart  = lower.indexOf("start_sec");
    const iEnd    = lower.indexOf("end_sec");
    const iState  = lower.indexOf("state");
    const iSource = lower.indexOf("source");
    const iBpm    = lower.indexOf("bpm_at_mid");

    if (iStart === -1 || iEnd === -1 || iState === -1) {
      alert('CSV must contain "start_sec", "end_sec", and "state" columns.');
      return;
    }

    const validStates = new Set(["S1", "S2", "systole", "diastole", "noisy"]);
    const imported = [];
    for (let i = 1; i < lines.length; i++) {
      const cells = lines[i].split(",");
      const start = parseFloat(cells[iStart]);
      const end   = parseFloat(cells[iEnd]);
      const state = (cells[iState] || "").trim();
      if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) continue;
      if (!validStates.has(state)) continue;
      // Import is a clean slate: all segments become "manual" regardless of their original
      // source. Fills (systole/diastole) will be re-derived by regen; S1/S2/noisy become
      // user-controlled anchors that survive undo snapshots.
      const source = "manual";
      const bpm    = iBpm !== -1 ? parseFloat(cells[iBpm]) : NaN;
      imported.push({ start_sec: start, end_sec: end, state, source, bpm_at_mid: Number.isFinite(bpm) ? bpm : null });
    }

    if (!imported.length) {
      alert("No valid state rows found in CSV.");
      return;
    }

    const snapshotBefore = _snapshotManualEdits();
    manualStateSegments = imported.sort((a, b) => a.start_sec - b.start_sec);
    finishManualStateEdit(`Imported ${manualStateSegments.length} state segment(s) from CSV.`, false);
    updateManualBpmTrace();
    const snapshotAfter = _snapshotManualEdits();
    _recordEdit(
      { type: "snapshot", snapshot: snapshotBefore },
      { type: "snapshot", snapshot: snapshotAfter }
    );
  }

  function downloadBpmCsv() {
    const s1s = manualStateSegments
      .filter((s) => s.state === "S1")
      .sort((a, b) => a.start_sec - b.start_sec);
    if (!s1s.length) {
      alert("No S1 segments available to export.");
      return;
    }

    // BPM lookup: prefer Manual trace, then original analysis curves.
    const bpmCandidates = [MANUAL_BPM_TRACE_NAME, "BPM (Pass 3)", "BPM (Pass 2)", "BPM (pass 1)", "Average BPM"];
    const bpmLookup = (() => {
      if (!plotlyGraphDiv || !plotlyGraphDiv.data) return null;
      const epochMs = EPOCH.getTime();
      for (const name of bpmCandidates) {
        const idx = findTraceIndexByName(name);
        if (idx === null) continue;
        const tr = plotlyGraphDiv.data[idx];
        if (!tr || !tr.x || !tr.y || !tr.x.length) continue;
        const pairs = [];
        for (let i = 0; i < tr.x.length; i++) {
          const xVal = tr.x[i];
          if (!xVal) continue;
          const ms = xVal instanceof Date ? xVal.getTime() : new Date(xVal).getTime();
          if (!Number.isFinite(ms)) continue;
          const tSec = (ms - epochMs) / 1000;
          const yVal = getNumericFromArrayLike(tr.y, i);
          if (Number.isFinite(tSec) && yVal !== null && Number.isFinite(yVal)) pairs.push([tSec, yVal]);
        }
        if (!pairs.length) continue;
        pairs.sort((a, b) => a[0] - b[0]);
        return pairs;
      }
      return null;
    })();

    const bpmAtTime = bpmLookup
      ? (timeSec) => {
          let lo = 0, hi = bpmLookup.length - 1;
          while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (bpmLookup[mid][0] < timeSec) lo = mid + 1; else hi = mid;
          }
          let best = bpmLookup[lo][1];
          if (lo > 0 && Math.abs(bpmLookup[lo - 1][0] - timeSec) < Math.abs(bpmLookup[lo][0] - timeSec)) {
            best = bpmLookup[lo - 1][1];
          }
          return Number.isFinite(best) ? best : null;
        }
      : getBpmSmoothedAtTime;

    const header = "start_sec,end_sec,state,bpm\n";
    const rows = s1s.map((seg) => {
      const tMid = (seg.start_sec + seg.end_sec) / 2;
      const bpm = bpmAtTime(tMid);
      return [
        seg.start_sec.toFixed(3),
        seg.end_sec.toFixed(3),
        seg.state,
        Number.isFinite(bpm) ? bpm.toFixed(3) : "",
      ].join(",");
    });

    const csvContent = header + rows.join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url  = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const baseName = (audioFileNameEl && audioFileNameEl.dataset && audioFileNameEl.dataset.defaultName) || "analysis";
    link.href = url;
    link.download = `${baseName}_bpm.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  if (applyLabelBtn) {
    applyLabelBtn.addEventListener("click", () => {
      const state = (labelTypeSelect && labelTypeSelect.value) || "S1";
      applyManualState(state);
    });
  }
  if (flipLabelsRightBtn) {
    flipLabelsRightBtn.addEventListener("click", flipManualStatesRight);
  }
  if (regenerateStatesBtn) {
    regenerateStatesBtn.addEventListener("click", regenerateGaps);
  }
  if (downloadLabelsBtn) {
    downloadLabelsBtn.addEventListener("click", downloadStateCsv);
  }
  const downloadBpmBtn = document.getElementById("download-bpm-btn");
  if (downloadBpmBtn) {
    downloadBpmBtn.addEventListener("click", downloadBpmCsv);
  }
  if (importLabelsBtn && importLabelsInput) {
    importLabelsBtn.addEventListener("click", () => {
      importLabelsInput.value = "";
      importLabelsInput.click();
    });
    importLabelsInput.addEventListener("change", (event) => {
      const file = event && event.target && event.target.files && event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e && e.target && e.target.result ? e.target.result : "";
        importStateCsv(text);
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
      if (loopActive && loopEngaged && audio.currentTime >= loopEndSec) {
        audio.currentTime = loopStartSec;
      }
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
    // Don't trigger if typing in a form control
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT")) return;

    if (e.code === "Escape" && analysisSummaryOverlay && analysisSummaryOverlay.classList.contains("visible")) {
      closeAnalysisSummaryModal();
      e.preventDefault();
      return;
    }

    if ((e.ctrlKey || e.metaKey) && e.code === "KeyZ" && e.shiftKey) {
      e.preventDefault();
      redoManualState();
      return;
    }

    if ((e.ctrlKey || e.metaKey) && e.code === "KeyZ" && !e.shiftKey) {
      e.preventDefault();
      undoManualState();
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
        e.preventDefault();
        if (labelTypeSelect) labelTypeSelect.value = "S1";
        applyManualState("S1");
        break;
      case "Digit2":
        e.preventDefault();
        if (labelTypeSelect) labelTypeSelect.value = "S2";
        applyManualState("S2");
        break;
      case "Digit3":
        e.preventDefault();
        if (labelTypeSelect) labelTypeSelect.value = "noisy";
        applyManualState("noisy");
        break;
      case "Backspace":
      case "Delete":
      case "KeyX":
        e.preventDefault();
        removeStateAtPlayhead();
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
      ensureLoopRegionEls();
      setupLoopRegionDrag();

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
        positionLoopRegion();
      });

      plotlyGraphDiv.on("plotly_afterplot", function () {
        snapshotLegendCategoryDefaults();
        applyDefaultLegendViewOnce();
        updateAxisRange();
        updateSpectrogramPosition();
        refreshAxisGridButtons();
        scheduleDrawPass3StateStrip();
        positionLoopRegion();
      });

      window.addEventListener("resize", () => {
        updateAxisRange();
        if (audio) {
          updatePlayhead(audio.currentTime);
        }
        updateSpectrogramPosition();
        scheduleDrawPass3StateStrip();
        Plotly.Plots.resize(plotlyGraphDiv);
        positionLoopRegion();
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
      SPECTROGRAM_AVAILABLE.original ||
      SPECTROGRAM_AVAILABLE.filtered ||
      SPECTROGRAM_AVAILABLE.filtered_inverse;
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
  initCardiacAndNoiseStripHovers();
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


