"""Engine isolation: engine.py and everything it imports stay free of UI/output code.

The engine (see engine.py docstring) is pure computation. GUI, CLI, plotting,
reports, settings files and audio-file conversion live outside it and depend on
it — never the other way round. If this fails, move the new dependency out of the
engine (or pass the data in / hand it out through engine.run_analysis's hooks)
rather than widening ENGINE_MODULES.
"""
import ast
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Repo-root modules that make up the engine.
ENGINE_MODULES = {
    "engine",
    "config",
    "audio_preprocessing",
    "noise_segments",
    "classifier",
    "confidence_engine",
    "peak_utils",
    "peak_label_scores",
    "correction",
    "phase_decision",
    "hrv",
    "viterbi",
    "time_utils",
    "analysis_data_schema",
}

# Third-party packages that only the UI / output layers may use.
FORBIDDEN_THIRD_PARTY = {"tkinter", "ttkbootstrap", "tkinterdnd2", "plotly", "kaleido", "matplotlib", "pydub"}


def _root_modules():
    return {f[:-3] for f in os.listdir(ROOT) if f.endswith(".py")}


def _imported_top_level_names(path):
    """Top-level module names of every import in the file, including function-local ones."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_engine_modules_exist():
    missing = ENGINE_MODULES - _root_modules()
    assert not missing, f"ENGINE_MODULES lists modules that no longer exist: {sorted(missing)}"


def test_engine_imports_only_engine_modules():
    root_modules = _root_modules()
    violations = []
    for mod in sorted(ENGINE_MODULES):
        for name in sorted(_imported_top_level_names(os.path.join(ROOT, f"{mod}.py"))):
            if name in root_modules and name not in ENGINE_MODULES:
                violations.append(f"{mod}.py imports non-engine module '{name}'")
            elif name in FORBIDDEN_THIRD_PARTY:
                violations.append(f"{mod}.py imports UI/output package '{name}'")
    assert not violations, "Engine boundary violated:\n  " + "\n  ".join(violations)


def test_importing_engine_loads_no_ui_or_output_modules():
    # Runtime check in a fresh interpreter: catches transitive pulls the AST scan can't see.
    watched = sorted(FORBIDDEN_THIRD_PARTY | (_root_modules() - ENGINE_MODULES))
    code = (
        "import sys, engine; "
        f"print(','.join(m for m in {watched!r} if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300, check=True
    )
    loaded = [m for m in out.stdout.strip().split(",") if m]
    assert not loaded, f"Importing engine loaded UI/output modules: {loaded}"
