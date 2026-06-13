"""Guard: keep config as the single source of truth for parameter defaults.

Fails if any module reads a DEFAULT_PARAMS key via `<params>.get(key, <literal>)`
instead of `param(params, key)`. A hardcoded literal fallback silently drifts
from config.DEFAULT_PARAMS when the default is retuned, so a partial params dict
would behave differently from a full one. Use `config.param(...)` instead.

This locks in the rollout so the smell cannot creep back in.
"""
import ast
import glob
import os

import config

# Receivers that are the canonical params dict (or `params or {}`). Other dicts
# that happen to share a key name (e.g. the ui_settings dict `s`) are exempt.
PARAMS_RECEIVERS = {"params", "self.params", "pc"}

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _offending_sites():
    keys = set(config.DEFAULT_PARAMS)
    hits = []
    for path in glob.glob(os.path.join(_ROOT, "*.py")):
        if os.path.basename(path) == "config.py":
            continue
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src, path)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get" and len(node.args) == 2):
                key_node = node.args[0]
                if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
                    continue
                if key_node.value not in keys:
                    continue
                # Literal default? (expression defaults are intentional, skip.)
                try:
                    ast.literal_eval(node.args[1])
                except Exception:
                    continue
                recv = ast.get_source_segment(src, node.func.value)
                if recv in PARAMS_RECEIVERS:
                    hits.append(f"{os.path.basename(path)}:{node.lineno}  "
                                f"{recv}.get(\"{key_node.value}\", ...)  -> use param({recv}, \"{key_node.value}\")")
    return hits


def test_no_literal_default_drift_for_config_keys():
    sites = _offending_sites()
    assert not sites, (
        "Found params.get(key, <literal>) for DEFAULT_PARAMS keys; replace with "
        "config.param(...) to keep config as the single source of truth:\n  "
        + "\n  ".join(sites)
    )
