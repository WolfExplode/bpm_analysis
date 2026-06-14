// Inlined into HTML when "embed minimal script" is enabled (no separate .js file beside the HTML).
// Keeps: legend category filter, S1/S2 hover toggle. Does not: audio sync, spectrogram, labeling, keyboard shortcuts.

(function () {
  const cfg = window.BPM_ANALYZER_CONFIG || {};
  // Match Python's naive datetime epoch (1970-01-01 00:00:00 local time),
  // not UTC epoch millis, so interactions align with x-axis values.
  const EPOCH = new Date(1970, 0, 1, 0, 0, 0, 0);
  let plotlyGraphDiv = null;
  const BEAT_HOVER_TRACES = ["S1 Beats", "S2 Beats", "Noise/Rejected"];
  let beatHoverEnabled = cfg.htmlS1S2HoverOnByDefault === true;

  function getNumericFromArrayLike(yContainer, index) {
    if (!yContainer || typeof index !== "number" || index < 0) return null;
    const tryAt = function (src) {
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

  // Trace audience comes from the Python TRACE_AUDIENCE registry (injected via config).
  // Unknown/unlisted names default to "debug" so new traces never leak into Analysis Data.
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
  const ANALYSIS_VIEW_Y_RANGE_MULTIPLIER = 5;
  let legendCategoryInitialState = null;
  let signalAxisRangeDefault = null;

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
    legendCategoryInitialState = plotlyGraphDiv.data.map(function (tr) {
      return {
        visible: tr.visible === undefined ? true : tr.visible,
        showlegend: tr.showlegend !== false,
      };
    });
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

  function findTraceIndexByName(targetName) {
    if (!plotlyGraphDiv || !plotlyGraphDiv.data) return null;
    for (let i = 0; i < plotlyGraphDiv.data.length; i++) {
      const tr = plotlyGraphDiv.data[i];
      if (!tr) continue;
      if ((tr.name || "") === targetName) return i;
    }
    return null;
  }

  function applyBeatHoverToPlot() {
    if (!plotlyGraphDiv) return;
    const indices = BEAT_HOVER_TRACES.map(function (n) {
      return findTraceIndexByName(n);
    }).filter(function (i) {
      return i !== null;
    });
    if (!indices.length) return;
    if (!beatHoverEnabled) {
      indices.forEach(function (i) {
        if (!plotlyGraphDiv.data[i]._origHoverTemplate) {
          plotlyGraphDiv.data[i]._origHoverTemplate =
            plotlyGraphDiv.data[i].hovertemplate || "%{customdata}<extra></extra>";
        }
      });
    }
    const hoverinfo = beatHoverEnabled ? "all" : "skip";
    const hovertemplate = beatHoverEnabled
      ? indices.map(function (i) {
          return (
            plotlyGraphDiv.data[i]._origHoverTemplate || "%{customdata}<extra></extra>"
          );
        })
      : indices.map(function () {
          return null;
        });
    Plotly.restyle(plotlyGraphDiv, { hoverinfo: hoverinfo, hovertemplate: hovertemplate }, indices).then(
      function () {
        const btn = document.getElementById("hover-toggle-btn");
        if (btn) btn.classList.toggle("active", beatHoverEnabled);
      }
    );
  }

  function toggleBeatHover() {
    if (!plotlyGraphDiv) return;
    beatHoverEnabled = !beatHoverEnabled;
    applyBeatHoverToPlot();
  }

  function initPlotlyIntegration() {
    const graphDivs = document.querySelectorAll(".plotly-graph-div");
    if (graphDivs.length > 0) {
      plotlyGraphDiv = graphDivs[0];
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
      plotlyGraphDiv.on("plotly_afterplot", function () {
        snapshotLegendCategoryDefaults();
        applyDefaultLegendViewOnce();
      });
      window.addEventListener("resize", function () {
        Plotly.Plots.resize(plotlyGraphDiv);
      });
    } else {
      setTimeout(initPlotlyIntegration, 100);
    }
  }

  setTimeout(initPlotlyIntegration, 300);
})();
