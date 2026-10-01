"""Engine isolation: pcg/engine stays free of UI/output code.

The engine is pure computation. The app, CLI, file formats and audio-file handling
live outside it and depend on it — never the other way round. If this fails, move
the new dependency out of the engine (or pass the data in / hand it out through
run_analysis's hooks) rather than importing it.
"""
import ast
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_DIR = os.path.join(ROOT, "pcg", "engine")

# Third-party packages that only the UI / output layers may use.
FORBIDDEN_THIRD_PARTY = {
    "tkinter", "ttkbootstrap", "tkinterdnd2", "plotly", "kaleido", "matplotlib", "pydub",
    "PySide6", "pyqtgraph", "sounddevice",
}


def _engine_files():
    return [os.path.join(ENGINE_DIR, f) for f in sorted(os.listdir(ENGINE_DIR)) if f.endswith(".py")]


def _imports(path):
    """(level, module) of every import in the file, including function-local ones."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield 0, alias.name
        elif isinstance(node, ast.ImportFrom):
            yield node.level, node.module or ""


def test_engine_imports_only_engine_and_third_party():
    root_modules = {f[:-3] for f in os.listdir(ROOT) if f.endswith(".py")} | {"pcg", "benchmarking", "debug_helpers"}
    violations = []
    for path in _engine_files():
        name = os.path.basename(path)
        for level, module in _imports(path):
            top = module.split(".")[0]
            if level > 1:
                violations.append(f"{name} imports from outside the engine package (level {level})")
            elif level == 0 and top in root_modules:
                violations.append(f"{name} imports project module '{module}' (use a relative import)")
            elif level == 0 and top in FORBIDDEN_THIRD_PARTY:
                violations.append(f"{name} imports UI/output package '{top}'")
    assert not violations, "Engine boundary violated:\n  " + "\n  ".join(violations)


def test_importing_engine_loads_no_ui_or_output_modules():
    # Runtime check in a fresh interpreter: catches transitive pulls the AST scan can't see.
    code = (
        "import sys, pcg.engine; "
        "print(','.join(m for m in sys.modules "
        f"if m.split('.')[0] in {sorted(FORBIDDEN_THIRD_PARTY)!r} "
        "or (m.startswith('pcg.') and not m.startswith('pcg.engine'))))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300, check=True
    )
    loaded = [m for m in out.stdout.strip().split(",") if m]
    assert not loaded, f"Importing the engine loaded UI/output modules: {loaded}"
