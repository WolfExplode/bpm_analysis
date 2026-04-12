# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files

# Collect Plotly data files including validators
plotly_datas = collect_data_files("plotly")

# Kaleido is used by Plotly to export static images (e.g., PNG).
kaleido_datas = collect_data_files("kaleido")

# Collect ttkbootstrap themes and other package data so the themed UI renders correctly
ttk_datas = collect_data_files("ttkbootstrap")

# Bundle local JS assets needed at runtime (e.g., interactive Plotly controls)
extra_datas = [
    (os.path.join("assets", "interactive_plot.js"), os.path.join("assets")),
    (os.path.join("assets", "html_inline_minimal.js"), os.path.join("assets")),
]

a = Analysis(
    ["main.pyw"],
    pathex=[],
    binaries=[],
    datas=plotly_datas + kaleido_datas + ttk_datas + extra_datas,
    hiddenimports=[
        # Core third‑party libs used across the project
        "ttkbootstrap",
        "pandas",
        "scipy",
        "numpy",
        "plotly",
        "plotly.validators",
        "plotly.graph_objects",
        "plotly.express",
        "kaleido",
        "kaleido.scopes",
        "kaleido.scopes.plotly",
        "pydub",
        "librosa",
        "matplotlib",
        "matplotlib.pyplot",

        # Project modules that may be imported indirectly
        "gui",
        "config",
        "classifier",
        "confidence_engine",
        "hrv",
        "correction",
        "pipeline",
        "audio_preprocessing",
        "plotting",
        "reporting",
        "validation",
        "peak_utils",
        "time_utils",
        "heartbeat_labeler",
        "hr_reactivity",

        # Optional / dynamic imports
        "PIL",
        "PIL._tkinter_finder",
        "pyPCG",
        "pyPCG.preprocessing",
        "pywt",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="BPM_Analyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)