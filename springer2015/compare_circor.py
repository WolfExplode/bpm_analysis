"""Standalone Springer-vs-your-pipeline comparison on CirCor recordings.

NOT integrated into the main codebase: this script only *imports* it (read-only,
the same way benchmarking/adapters/circor.py does) to obtain your pipeline's
segmentation. It produces, per recording, an interactive Plotly HTML in the same
visual style as plotting.py (Plotly, secondary-y, legend filter, shaded state
bands over the homomorphic envelope) showing three segmentations stacked:

    Ground truth  (CirCor .tsv)
    Springer      (ported pretrained HSMM, springer_pretrained.npz)
    Yours         (analyze_wav_file -> pass3_state_boundaries)

plus a table of per-file Se / PPV / F1 (S1 detection vs GT, reusing
benchmarking/bench_scoring) for Springer and for your pipeline.

Usage:
    python springer2015/compare_circor.py [--root DIR] [--n 6] [--seed 0] [--out DIR]
"""

import argparse
import glob
import os
import random
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # repo root (the main codebase)
_BENCH = os.path.join(_ROOT, "benchmarking")
for _p in (_HERE, _ROOT, _BENCH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plotly.graph_objects as go  # noqa: E402

# --- Springer port (this dir) ---
from springer_hsmm.model_io import load_springer_model  # noqa: E402
from springer_hsmm.options import default_springer_hsmm_options  # noqa: E402
from springer_hsmm.run import run_springer_segmentation_algorithm  # noqa: E402

# --- read-only borrows from the main codebase / benchmark harness ---
from bench_scoring import (  # noqa: E402
    DEFAULT_TOLERANCES_SEC,
    Span,
    derive_metrics,
    filter_to_windows,
    s1_centers,
    s2_centers,
    score_file,
)
from config import DEFAULT_PARAMS, param  # noqa: E402
from pipeline import analyze_wav_file  # noqa: E402

DEFAULT_ROOT = (
    r"G:\HB other\PCG Datasets"
    r"\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data"
)
MODEL_NPZ = os.path.join(_HERE, "springer_pretrained.npz")

CODE_TO_STATE = {1: "S1", 2: "systole", 3: "S2", 4: "diastole"}
STATE_COLOR = {
    "S1": "#d62728",       # red
    "systole": "#ff7f0e",  # orange
    "S2": "#1f77b4",       # blue
    "diastole": "#2ca02c", # green
}
# Springer's decoded states lead the CirCor S1 labels by a near-constant ~120 ms
# (verified across recordings: raw F1@60ms ~ 0%, but a single +120 ms shift lifts
# clean files to 65-100%). The envelope/preprocessing and the pretrained model are
# correctly aligned; the offset is systematic in the HSMM decode stage and matches
# the reference MATLAB, i.e. it is an S1-onset *convention* difference vs CirCor,
# not random error. We report BOTH raw and this fixed-offset-aligned score so the
# comparison against your pipeline is fair. One global constant, not a per-file fit.
ALIGN_OFFSET_SEC = 0.12

SPR_ALIGNED_LABEL = f"Springer +{int(ALIGN_OFFSET_SEC*1000)}ms"

# Project theme (mirrors assets/template.html)
THEME = {
    "bg": "#111", "panel": "#1e1e2e", "panel2": "#151520", "accent": "#00d4ff",
    "btn": "#2a2a35", "border": "#444", "text": "#e0e0e0", "muted": "#aaa",
}

_OUTPUT_OPTIONS = {
    "html": False, "png": False, "csv": False, "summary": False, "debug": False,
    "filtered_wav": False, "spectrogram": False, "fft_profiles": False,
    "output_all_passes": False, "working_wav_in_output": False,
}


# --------------------------------------------------------------------------
# Ground truth (.tsv)  — same parsing rules as benchmarking/adapters/circor.py
# --------------------------------------------------------------------------

def load_tsv(path: str) -> List[Span]:
    spans: List[Span] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                start, end, code = float(parts[0]), float(parts[1]), int(float(parts[2]))
            except ValueError:
                continue
            state = CODE_TO_STATE.get(code)
            if state is None or end <= start:
                continue
            spans.append((start, end, state))
    return spans


def labeled_windows(spans: List[Span]) -> List[Tuple[float, float]]:
    if not spans:
        return []
    ordered = sorted((s, e) for s, e, _ in spans)
    windows: List[Tuple[float, float]] = []
    cur_s, cur_e = ordered[0]
    for s, e in ordered[1:]:
        if s <= cur_e + 1e-6:
            cur_e = max(cur_e, e)
        else:
            windows.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    windows.append((cur_s, cur_e))
    return windows


def collect_recordings(root: str) -> List[Tuple[str, str]]:
    found = []
    for tsv in sorted(glob.glob(os.path.join(root, "*.tsv"))):
        wav = tsv[:-4] + ".wav"
        if os.path.isfile(wav):
            found.append((wav, tsv))
    return found


# --------------------------------------------------------------------------
# Predicted segmentations -> spans (seconds)
# --------------------------------------------------------------------------

def states_to_spans(states: np.ndarray, fs: float) -> List[Span]:
    """Per-sample int states (1..4, 0=ignore) -> contiguous (start,end,name) spans."""
    spans: List[Span] = []
    states = np.asarray(states).astype(int)
    n = len(states)
    i = 0
    while i < n:
        s = states[i]
        j = i
        while j < n and states[j] == s:
            j += 1
        name = CODE_TO_STATE.get(int(s))
        if name is not None:
            spans.append((i / fs, j / fs, name))
        i = j
    return spans


def springer_spans(wav_path: str, model: Dict, opts: Dict) -> Tuple[List[Span], np.ndarray, float]:
    audio, fs = sf.read(wav_path)
    audio = np.asarray(audio, dtype=np.float64).flatten()
    states, env = run_springer_segmentation_algorithm(
        audio, fs, model["B_matrix"], model["pi_vector"],
        model["total_obs_distribution"], opts,
    )
    env = np.asarray(env, dtype=np.float64)
    rng = np.ptp(env)
    env_norm = (env - env.min()) / rng if rng > 0 else np.zeros_like(env)
    return states_to_spans(states, fs), env_norm, float(fs)


def yours_spans(wav_path: str, params: Dict) -> List[Span]:
    sr = float(int(param(params, "preprocess_target_sample_rate") or 600))
    with tempfile.TemporaryDirectory() as tmp:
        _, _, _, data = analyze_wav_file(
            wav_path, params, None,
            original_file_path=wav_path,
            output_directory=tmp,
            output_options=_OUTPUT_OPTIONS,
            collect_fft_for_aggregate=False,
        )
    if not data:
        return []
    boundaries = data.get("pass3_state_boundaries") or []
    return [(b[0] / sr, b[1] / sr, str(b[2])) for b in boundaries]


# --------------------------------------------------------------------------
# Metrics  (S1 detection vs GT, micro within labeled windows)
# --------------------------------------------------------------------------

def shift_spans(spans: List[Span], offset: float) -> List[Span]:
    return [(s + offset, e + offset, st) for (s, e, st) in spans]


def f1_vs_gt(gt: List[Span], pred: List[Span], offset: float = 0.0) -> Dict[float, Dict[str, float]]:
    manual_s1 = s1_centers(gt)
    windows = labeled_windows(gt)
    ps1 = filter_to_windows([c + offset for c in s1_centers(pred)], windows)
    ps2 = filter_to_windows([c + offset for c in s2_centers(pred)], windows)
    counts = score_file(manual_s1, ps1, ps2, DEFAULT_TOLERANCES_SEC)
    return {tol: {**counts[tol], **derive_metrics(counts[tol])} for tol in DEFAULT_TOLERANCES_SEC}


# --------------------------------------------------------------------------
# Plot — same visual language as the project's HTML (dark theme, chart toolbar,
# "Show:" filter), formatted as a categorical state-timeline so row labels never
# clip. Each pipeline is one horizontal-bar trace (segments colored by state),
# toggleable via the toolbar select. Metrics render as a styled HTML table.
# --------------------------------------------------------------------------

PIPELINE_ROWS = ["Ground truth", "Springer", SPR_ALIGNED_LABEL, "Yours"]

STRIP_H = 16        # px per pipeline row
STRIP_GAP = 3       # px between rows


def _env_xy(env_norm, fs):
    t = np.arange(len(env_norm)) / fs
    step = max(1, len(t) // 6000)
    return t[::step], np.asarray(env_norm, dtype=np.float64)[::step]


def _build_figure(env_norm, fs):
    """Envelope-only Plotly figure. State strips rendered as HTML canvas overlays."""
    fig = go.Figure()
    ex, en = _env_xy(env_norm, fs)
    fig.add_trace(go.Scatter(
        x=ex, y=en, mode="lines",
        line=dict(color=THEME["accent"], width=1.1), name="envelope",
        showlegend=False, hovertemplate="%{x:.2f}s<extra>envelope</extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["bg"],
        font=dict(family="Segoe UI, Roboto, sans-serif", color=THEME["text"], size=12),
        margin=dict(l=150, r=24, t=16, b=46), height=640,
        dragmode="pan", autosize=True, hovermode="x",
        xaxis=dict(
            title_text="Time (s)", showgrid=True, gridcolor="#333",
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="#888", spikethickness=1, spikedash="solid",
        ),
        yaxis=dict(
            title_text="envelope", showgrid=False, zeroline=False, fixedrange=False,
        ),
    )
    return fig


def build_html(
    name: str, env_norm: np.ndarray, fs: float,
    gt: List[Span], spr: List[Span], yours: List[Span],
    metrics: List[Tuple[str, Dict]], out_path: str,
) -> None:
    import json as _json
    spr_aligned = shift_spans(spr, ALIGN_OFFSET_SEC)
    pipeline_list = [
        ("Ground truth", gt),
        ("Springer", spr),
        (SPR_ALIGNED_LABEL, spr_aligned),
        ("Yours", yours),
    ]
    fig = _build_figure(env_norm, fs)
    plot_div = fig.to_html(
        full_html=False, include_plotlyjs="cdn", div_id="cmp-plot",
        config={"displaylogo": False, "responsive": True, "scrollZoom": True,
                "displayModeBar": True},
    )
    pipelines_js = _json.dumps([
        {"label": label,
         "segs": [{"start": s, "end": e, "state": st} for s, e, st in spans]}
        for label, spans in pipeline_list
    ])
    html = _wrap_html(name, plot_div, metrics, pipelines_js)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def _metrics_table_html(metrics: List[Tuple[str, Dict]]) -> str:
    tols = list(DEFAULT_TOLERANCES_SEC)
    heads = "".join(
        f"<th>{m}@{int(t*1000)}ms</th>" for m in ("Se", "PPV", "F1") for t in tols
    )
    body = ""
    for nm, m in metrics:
        cells = "".join(
            f"<td>{m[t][k]*100:.1f}%</td>"
            for k in ("se", "ppv", "f1") for t in tols
        )
        hi = " class='hi'" if "Yours" not in nm and "raw" not in nm else ""
        body += f"<tr{hi}><td class='lbl'>{nm}</td>{cells}</tr>"
    return f"<table class='metrics'><thead><tr><th>Pipeline</th>{heads}</tr></thead><tbody>{body}</tbody></table>"


def _swatches_html() -> str:
    return "".join(
        f"<span class='sw'><i style='background:{c}'></i>{s}</span>"
        for s, c in STATE_COLOR.items()
    )


def _wrap_html(name, plot_div, metrics, pipelines_js) -> str:
    th = THEME
    table = _metrics_table_html(metrics)
    swatches = _swatches_html()
    n = len(PIPELINE_ROWS)
    # Canvas + label elements (one per pipeline row)
    canvas_html = "".join(
        f'<canvas id="strip-c{i}" style="position:absolute;pointer-events:none;z-index:10"></canvas>'
        f'<div id="strip-l{i}" style="position:absolute;z-index:10;display:flex;align-items:center;'
        f'justify-content:flex-end;padding-right:8px;font-size:11px;color:{th["muted"]};'
        f'pointer-events:none;white-space:nowrap;overflow:hidden"></div>'
        for i in range(n)
    )
    al = int(ALIGN_OFFSET_SEC * 1000)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>{name} — Springer vs Yours</title>
<style>
  *{{box-sizing:border-box}}
  body{{margin:0;background:{th['bg']};color:{th['text']};
       font-family:'Segoe UI',Roboto,sans-serif;font-size:13px}}
  #toolbar{{display:flex;align-items:center;gap:10px;padding:6px 12px;
       background:rgba(40,40,50,.6);border-bottom:1px solid #333;flex-shrink:0}}
  #toolbar .title{{font-weight:600;color:{th['text']}}}
  #toolbar .file{{color:{th['muted']};font-family:Consolas,monospace;font-size:12px}}
  #toolbar .note{{color:{th['accent']};font-size:11px}}
  #toolbar .spacer{{margin-left:auto}}
  #toolbar label{{color:{th['muted']};font-size:12px}}
  select{{padding:3px 8px;background:{th['btn']};color:#ddd;border:1px solid {th['border']};
       border-radius:3px;font-size:12px;cursor:pointer}}
  select:hover{{border-color:#666}}
  .swatches{{display:flex;gap:14px;padding:5px 12px;border-bottom:1px solid #222;color:{th['muted']};flex-shrink:0}}
  .sw{{display:flex;align-items:center;gap:5px;font-size:12px}}
  .sw i{{width:12px;height:12px;border-radius:2px;display:inline-block}}
  #plot-wrapper{{position:relative;flex:1;min-height:0}}
  #cmp-plot{{width:100%;height:100%}}
  #cmp-plot .plotly,#cmp-plot .plot-container{{height:100%!important}}
  .panel{{padding:10px 12px;flex-shrink:0}}
  table.metrics{{border-collapse:collapse;font-size:12px;min-width:560px}}
  table.metrics th,table.metrics td{{padding:4px 10px;text-align:right;border-bottom:1px solid #2a2a35}}
  table.metrics th{{color:{th['muted']};font-weight:600}}
  table.metrics td.lbl,table.metrics th:first-child{{text-align:left}}
  table.metrics tr.hi td{{color:{th['accent']}}}
  .cap{{color:{th['muted']};margin:2px 0 10px;max-width:760px;line-height:1.5}}
  html,body{{height:100%;overflow:hidden}}
  body{{display:flex;flex-direction:column}}
</style></head><body>
<div id="toolbar">
  <span class="title">Springer HSMM vs your pipeline</span>
  <span class="file">{name}</span>
  <span class="note">Springer leads CirCor S1 by ~{al}ms (see "+{al}ms" row)</span>
  <span class="spacer"></span>
  <label for="show">Show:</label>
  <select id="show">
    <option value="all">All rows</option>
    <option value="gt_spr">GT + Springer +{al}ms</option>
    <option value="gt_yours">GT + Yours</option>
    <option value="spr_yours">Springer +{al}ms + Yours</option>
    <option value="gt_only">Ground truth only</option>
  </select>
  <span class="note">scroll/box-zoom to zoom envelope y-axis</span>
</div>
<div class="swatches">{swatches}</div>
<div id="plot-wrapper">
  {plot_div}
  {canvas_html}
</div>
<div class="panel">
  <div class="cap">S1 detection vs ground truth (micro within labeled windows).
  Raw Springer ~0 due to constant onset offset; +{al}ms row is same segmentation, calibrated.</div>
  {table}
</div>
<script>
var PIPELINES = {pipelines_js};
var STATE_COLORS = {{"S1":"{STATE_COLOR['S1']}","systole":"{STATE_COLOR['systole']}","S2":"{STATE_COLOR['S2']}","diastole":"{STATE_COLOR['diastole']}"}};
var SPR_ALIGNED = "{SPR_ALIGNED_LABEL}";
var STRIP_H = {STRIP_H};
var STRIP_GAP = {STRIP_GAP};
var visibleSet = new Set(PIPELINES.map(function(p){{return p.label;}}));

var VIEWS = {{
  all: PIPELINES.map(function(p){{return p.label;}}),
  gt_spr: ["Ground truth", SPR_ALIGNED],
  gt_yours: ["Ground truth", "Yours"],
  spr_yours: [SPR_ALIGNED, "Yours"],
  gt_only: ["Ground truth"]
}};

function drawStrips() {{
  var gd = document.getElementById("cmp-plot");
  if (!gd || !gd._fullLayout) return;
  var fl = gd._fullLayout;
  var xa = fl.xaxis, ya = fl.yaxis;
  if (!xa || !ya || !xa._length || !ya._length) return;

  var left = xa._offset || 0;
  var pxW  = xa._length;
  var plotBottom = (ya._offset || 0) + (ya._length || 0);
  var x0 = xa.range[0], x1 = xa.range[1];
  var span = x1 - x0;
  if (span <= 0) return;

  var n = PIPELINES.length;
  for (var i = 0; i < n; i++) {{
    var pipe = PIPELINES[i];
    var visible = visibleSet.has(pipe.label);
    var top = plotBottom - (n - i) * (STRIP_H + STRIP_GAP);

    var canvas = document.getElementById("strip-c" + i);
    var lbl    = document.getElementById("strip-l" + i);

    if (!visible) {{
      canvas.style.display = "none";
      lbl.style.display = "none";
      continue;
    }}

    canvas.style.left   = left + "px";
    canvas.style.top    = top + "px";
    canvas.style.width  = pxW + "px";
    canvas.style.height = STRIP_H + "px";
    canvas.style.display = "";
    canvas.width  = Math.max(1, Math.round(pxW));
    canvas.height = STRIP_H;

    lbl.style.left   = "0px";
    lbl.style.top    = top + "px";
    lbl.style.width  = left + "px";
    lbl.style.height = STRIP_H + "px";
    lbl.style.display = "";
    lbl.textContent = pipe.label;

    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, STRIP_H);
    ctx.fillStyle = "rgba(0,0,0,0.30)";
    ctx.fillRect(0, 0, canvas.width, STRIP_H);

    for (var j = 0; j < pipe.segs.length; j++) {{
      var seg = pipe.segs[j];
      if (seg.end <= x0 || seg.start >= x1) continue;
      var cl0 = Math.max(x0, seg.start);
      var cl1 = Math.min(x1, seg.end);
      var px0 = Math.floor(((cl0 - x0) / span) * pxW);
      var px1 = Math.ceil(((cl1 - x0) / span) * pxW);
      if (px1 <= px0) continue;
      ctx.fillStyle = STATE_COLORS[seg.state] || "#555";
      ctx.globalAlpha = 0.82;
      ctx.fillRect(px0, 0, px1 - px0, STRIP_H);
      ctx.globalAlpha = 1.0;
    }}
  }}
}}

document.getElementById("show").addEventListener("change", function(e) {{
  var v = e.target.value;
  visibleSet = new Set(VIEWS[v] || PIPELINES.map(function(p){{return p.label;}}));
  drawStrips();
}});

var gd = document.getElementById("cmp-plot");
gd.on("plotly_relayout", drawStrips);
gd.on("plotly_afterplot", drawStrips);
window.addEventListener("resize", function(){{ setTimeout(drawStrips, 60); }});
setTimeout(drawStrips, 300);
</script>
</body></html>"""


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def _pick_files_ui() -> Tuple[Optional[List[str]], Optional[str]]:
    """Open tkinter dialogs to select WAV files + output dir. Returns (wavs, out_dir) or (None, None)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None, None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    wavs = filedialog.askopenfilenames(
        title="Select WAV files to compare",
        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        initialdir=DEFAULT_ROOT if os.path.isdir(DEFAULT_ROOT) else os.path.expanduser("~"),
    )
    if not wavs:
        root.destroy()
        return None, None
    out_dir = filedialog.askdirectory(
        title="Output directory (Cancel = default compare_out/)",
        initialdir=os.path.join(_HERE, "compare_out"),
    )
    root.destroy()
    return list(wavs), out_dir or os.path.join(_HERE, "compare_out")


def _wavs_to_recs(wav_paths: List[str]) -> List[Tuple[str, str]]:
    recs = []
    for w in wav_paths:
        t = os.path.splitext(w)[0] + ".tsv"
        if os.path.isfile(t):
            recs.append((w, t))
        else:
            print(f"WARNING: no .tsv alongside {os.path.basename(w)}, skipping")
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description="Springer vs your pipeline on CirCor.")
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="CirCor training_data dir (random sample)")
    ap.add_argument("--n", type=int, default=6, help="files to sample from --root")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(_HERE, "compare_out"))
    ap.add_argument("--files", nargs="+", metavar="WAV",
                    help="specific WAV paths to process (skips --root/--n/--seed)")
    ap.add_argument("--ui", action="store_true",
                    help="open file picker (default when run with no arguments)")
    args = ap.parse_args()

    if not os.path.isfile(MODEL_NPZ):
        print(f"Missing {MODEL_NPZ}. Run: python springer2015/port_pretrained_model.py", file=sys.stderr)
        sys.exit(1)

    # Resolve recordings list
    if args.files:
        recs = _wavs_to_recs(args.files)
        if not recs:
            print("No valid WAV+TSV pairs in --files.", file=sys.stderr); sys.exit(1)
    elif args.ui or len(sys.argv) == 1:
        wavs, picked_out = _pick_files_ui()
        if not wavs:
            print("No files selected."); sys.exit(0)
        recs = _wavs_to_recs(wavs)
        if not recs:
            print("No valid WAV+TSV pairs selected.", file=sys.stderr); sys.exit(1)
        args.out = picked_out
    else:
        recs = collect_recordings(args.root)
        if not recs:
            print(f"No CirCor recordings under {args.root}", file=sys.stderr); sys.exit(1)
        random.Random(args.seed).shuffle(recs)
        recs = recs[: args.n]

    os.makedirs(args.out, exist_ok=True)

    model = load_springer_model(MODEL_NPZ)
    opts = default_springer_hsmm_options()
    params = {**DEFAULT_PARAMS, "save_filtered_wav": False, "enable_fft_profiles": False}

    spr_lbl = f"Springer +{int(ALIGN_OFFSET_SEC*1000)}ms"
    who_keys = ["Springer (raw)", spr_lbl, "Yours"]
    agg = {w: {t: {"tp": 0, "fn": 0, "fp": 0} for t in DEFAULT_TOLERANCES_SEC} for w in who_keys}

    for i, (wav, tsv) in enumerate(recs, 1):
        name = os.path.splitext(os.path.basename(wav))[0]
        gt = load_tsv(tsv)
        if not s1_centers(gt):
            print(f"[{i}/{len(recs)}] SKIP {name} (no S1 labels)")
            continue
        try:
            spr, env_norm, fs = springer_spans(wav, model, opts)
            yrs = yours_spans(wav, params)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(recs)}] ERROR {name}: {exc}")
            continue
        m_spr = f1_vs_gt(gt, spr)
        m_spr_a = f1_vs_gt(gt, spr, ALIGN_OFFSET_SEC)
        m_yrs = f1_vs_gt(gt, yrs)
        per = [("Springer (raw)", m_spr), (spr_lbl, m_spr_a), ("Yours", m_yrs)]
        for w, m in per:
            for tol in DEFAULT_TOLERANCES_SEC:
                for k in ("tp", "fn", "fp"):
                    agg[w][tol][k] += m[tol][k]
        out_path = os.path.join(args.out, f"{name}_compare.html")
        build_html(name, env_norm, fs, gt, spr, yrs, per, out_path)
        t6 = 0.06
        print(f"[{i}/{len(recs)}] {name:14s} "
              f"Springer raw={m_spr[t6]['f1']*100:5.1f}%  "
              f"+{int(ALIGN_OFFSET_SEC*1000)}ms={m_spr_a[t6]['f1']*100:5.1f}%  "
              f"Yours={m_yrs[t6]['f1']*100:5.1f}%  (F1@60ms)  -> {os.path.basename(out_path)}")

    print("\n=== aggregate over rendered files (micro, S1 detection vs GT) ===")
    for who in who_keys:
        for tol in DEFAULT_TOLERANCES_SEC:
            m = derive_metrics(agg[who][tol])
            print(f"  {who:16s} tol={int(tol*1000)}ms  "
                  f"Se={m['se']*100:5.1f}%  PPV={m['ppv']*100:5.1f}%  F1={m['f1']*100:5.1f}%")
    print(f"\nHTML written to {args.out}")


if __name__ == "__main__":
    main()
