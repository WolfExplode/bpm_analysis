"""Guard the design language: pcg/app/theme.py is the only place colours are defined, and its
trace-styling rules keep every trace in a lane distinguishable."""
import os
import re

from pcg.app import theme
from pcg.app.theme import TraceMeta
from pcg.engine import traces as tr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HEX = re.compile(r"""["']#[0-9a-fA-F]{3,8}["']""")


def test_no_colour_literals_outside_theme():
    files = [os.path.join(_ROOT, "pcg", "engine", "traces.py")]
    app = os.path.join(_ROOT, "pcg", "app")
    files += [os.path.join(app, f) for f in os.listdir(app) if f.endswith(".py") and f != "theme.py"]
    hits = []
    for path in files:
        for n, line in enumerate(open(path, encoding="utf-8"), 1):
            if _HEX.search(line):
                hits.append(f"{os.path.relpath(path, _ROOT)}:{n}: {line.strip()}")
    assert not hits, "colour literals belong in pcg/app/theme.py:\n" + "\n".join(hits)


def test_every_engine_role_has_a_colour():
    assert set(tr.ROLES) <= set(theme.ROLE_COLORS)


def test_role_beats_stage_colour():
    s = theme.assign_styles([TraceMeta("S1 score", "scores", tr.KIND_LINE, theme.PASS_2, tr.ROLE_S1)])
    assert s["S1 score"].color == theme.S1


def test_debug_traces_take_their_stage_colour():
    s = theme.assign_styles([TraceMeta("BPM (Pass 1)", "bpm", tr.KIND_LINE, theme.PASS_1)])
    assert s["BPM (Pass 1)"].color == theme.STAGE_COLORS[theme.PASS_1]


def test_lookalikes_in_one_lane_get_distinct_dashes_and_markers():
    metas = [
        TraceMeta("a", "bpm", tr.KIND_LINE, theme.PASS_1),
        TraceMeta("b", "bpm", tr.KIND_LINE, theme.PASS_1),
        TraceMeta("c", "bpm", tr.KIND_LINE, theme.PASS_1, estimate=True),
        TraceMeta("p", "bpm", tr.KIND_POINTS, theme.PASS_1),
        TraceMeta("q", "bpm", tr.KIND_POINTS, theme.PASS_1),
        TraceMeta("other lane", "signal", tr.KIND_LINE, theme.PASS_1),
    ]
    s = theme.assign_styles(metas)
    assert (s["a"].dash, s["b"].dash, s["c"].dash) == (theme.DASH_SOLID, theme.DASH_DOT, theme.DASH_DASH)
    assert s["p"].symbol != s["q"].symbol
    assert s["other lane"].dash == theme.DASH_SOLID


def test_unknown_groups_sort_between_pass_3_and_result():
    order = theme.stage_order(["Result", "My experiment", "Pass 1"])
    assert order.index("Pass 3") < order.index("My experiment") < order.index("Result")


def test_legend_label_drops_the_pass():
    from pcg.app.workspace import legend_label

    assert legend_label("BPM (Pass 1)") == "BPM"
    assert legend_label("Instant BPM (Pass 1, outliers removed)") == "Instant BPM (outliers removed)"
    assert legend_label("Measured systole curve (final)") == "Measured systole curve (final)"
