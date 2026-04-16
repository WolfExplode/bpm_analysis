# -*- mode: python ; coding: utf-8 -*-
import os
import importlib.util
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

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
    (os.path.join("assets", "template.html"), os.path.join("assets")),
]

def _optional_hiddenimports(mod_names):
    """Return only modules that are importable in the build env."""
    out = []
    for name in mod_names:
        if importlib.util.find_spec(name) is not None:
            out.append(name)
    return out

# SciPy uses internal array_api_compat shims (e.g. scipy._lib.array_api_compat.numpy.fft)
# that PyInstaller can miss, leading to runtime ModuleNotFoundError.
scipy_array_api_hiddenimports = collect_submodules("scipy._lib.array_api_compat")

a = Analysis(
    ["main.py"],
    pathex=[os.path.abspath(".")],
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

        # Modules used by the pipeline that sometimes get missed by static analysis
        "fft_profiles",
        "viterbi",
        "emissions",
        "ui_settings_loader",
        "console_logging",
    ]
    + _optional_hiddenimports(
        [
            # Optional / dynamic imports
            "PIL",
            "PIL._tkinter_finder",
            "pyPCG",
            "pyPCG.preprocessing",
            "pywt",
        ]
    )
    + scipy_array_api_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Prevent PyInstaller from pulling in large ML stacks that are present
        # in the environment but not used by this project (can cause recursion/stack overflows).
        "tensorflow",
        "tensorflow-plugins",
        "keras",
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "jax",
        "jaxlib",
        "openvino",
        "pyside6",
        "PySide6",
        "pyqt5",
        "PyQt5",
    ],
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