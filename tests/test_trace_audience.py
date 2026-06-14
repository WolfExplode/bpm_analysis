"""Guard: keep plotting.TRACE_AUDIENCE as the single source of truth for the HTML
"Show:" category filter (Analysis Data vs Debug).

The interactive HTML decides which traces an end user sees from the injected
TRACE_AUDIENCE map; an unregistered trace silently falls back to "debug" in the JS.
That fail-safe protects the end-user view, but a trace meant for Analysis Data that
is never registered would silently go missing. These tests catch that at build time:

  * every audience value is one of the three valid buckets,
  * every static ``name="..."`` trace literal in plotting.py is registered,
  * the dynamically-named (f-string) traces are enumerated and registered.
"""
import ast
import os

import plotting
from plotting import TRACE_AUDIENCE, _DYNAMIC_TRACE_NAMES

VALID_AUDIENCES = {"analysis", "debug", "both"}

_PLOTTING_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plotting.py"
)


def _static_trace_name_literals():
    """Every string literal passed as a ``name=`` keyword argument in plotting.py.

    Covers the go.Scatter traces (the common path for adding a trace). Variable and
    f-string ``name=`` values are not constants and are intentionally skipped here;
    the f-string names are guarded separately via _DYNAMIC_TRACE_NAMES.
    """
    src = open(_PLOTTING_PATH, encoding="utf-8").read()
    tree = ast.parse(src, _PLOTTING_PATH)
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "name":
                continue
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                names.add(kw.value.value)
    return names


def test_all_audience_values_valid():
    bad = {name: aud for name, aud in TRACE_AUDIENCE.items() if aud not in VALID_AUDIENCES}
    assert not bad, f"TRACE_AUDIENCE values must be one of {VALID_AUDIENCES}: {bad}"


def test_every_static_trace_name_is_registered():
    literals = _static_trace_name_literals()
    missing = sorted(n for n in literals if n not in TRACE_AUDIENCE)
    assert not missing, (
        "Plot trace(s) not in plotting.TRACE_AUDIENCE; register each as "
        "'analysis', 'debug', or 'both' so the HTML category filter knows where "
        "it belongs:\n  " + "\n  ".join(missing)
    )


def test_dynamic_trace_names_are_registered():
    missing = sorted(n for n in _DYNAMIC_TRACE_NAMES if n not in TRACE_AUDIENCE)
    assert not missing, f"Dynamic trace names missing from TRACE_AUDIENCE: {missing}"
