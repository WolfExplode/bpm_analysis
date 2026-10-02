"""The app's design language: colour tokens, trace styling, fonts and the Qt stylesheet.

Every colour the app paints comes from this module (tests/test_theme.py enforces it). The rules:

- Chrome is neutral grey. Its only colour is ACCENT, for selection, focus and the one primary
  action on a screen.
- A data colour means one thing everywhere: S1 (and the systole it starts) is coral, S2 (and
  diastole) cyan, noise and Noisy spans grey, issues (Defects, Disagreements) magenta, results lime.
- Debug traces take the colour of the stage that computed them (Preprocessing, Pass 1-3), so a
  pass reads as one hue in every lane. Estimates (beliefs, priors, expectations) are dashed.
- Numbers, times and tooltips use the monospace face.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from pcg.engine import traces as tr

# ─────────────────────────────────────────────────────────────────────────────
# Tokens
# ─────────────────────────────────────────────────────────────────────────────

# Surfaces, darkest first: plots sit lowest, controls highest.
SURFACE_0 = "#0f1113"   # plot canvas
SURFACE_1 = "#14171a"   # lane legends, tables, lists
SURFACE_2 = "#181b1f"   # window, header, panels
SURFACE_3 = "#1f2328"   # controls, menus, tooltips
LINE = "#23282d"        # dividers
BORDER = "#2c3137"
BORDER_STRONG = "#3b4249"

TEXT = "#e6e8ea"
TEXT_2 = "#a2aab2"      # supporting text, idle buttons
TEXT_3 = "#6f777f"      # labels, axis ticks, hints
TEXT_4 = "#4c535a"      # disabled, hidden traces

ACCENT = "#5b9cf5"
ACCENT_HOVER = "#78aff7"
ON_ACCENT = "#0b1220"

# Data
S1 = "#f08a64"
S2 = "#4fc1d6"
NOISE = "#8a929a"
ISSUE = "#e05fd0"
METRIC = "#c7e36b"
ANNOTATION = "#e6e8ea"
ENVELOPE = "#8796a4"       # the envelope's outline; its body is ENVELOPE_FILL
ENVELOPE_FILL = "#46525d"
PLAYHEAD = "#ffffff"
# The exported BPM chart (no S1/S2 marks there, so the waveform can be cyan without clashing).
CHART_BPM = METRIC
CHART_WAVE = "#3f9fc2"

# Stages, in the order the engine computes them.
PREPROCESSING, PASS_1, PASS_2, PASS_3, RESULT, ANNOTATION_GROUP = (
    "Preprocessing", "Pass 1", "Pass 2", "Pass 3", "Result", "Annotation")
DEBUG_STAGES = (PREPROCESSING, PASS_1, PASS_2, PASS_3)
ALWAYS_LISTED = (RESULT, ANNOTATION_GROUP)
STAGE_COLORS = {
    PREPROCESSING: "#c4a882",
    PASS_1: "#9d8cf2",
    PASS_2: "#e3b450",
    PASS_3: "#6fc79a",
    RESULT: METRIC,
    ANNOTATION_GROUP: ANNOTATION,
}
STAGE_TAGS = {PREPROCESSING: "pre", PASS_1: "P1", PASS_2: "P2", PASS_3: "P3", RESULT: "result",
              ANNOTATION_GROUP: "ann"}
# Groups a debug trace may invent (emit_trace) get one of these, by name.
EXTRA_STAGE_COLORS = ("#d98bb5", "#8fb3e0", "#d4cf7a")

# Semantic roles (set by the engine on a Trace) override the stage colour.
ROLE_ANNOTATION = "annotation"  # workspace-only: series rebuilt from the Annotation
ROLE_COLORS = {
    tr.ROLE_ENVELOPE: ENVELOPE,
    tr.ROLE_S1: S1,
    tr.ROLE_S2: S2,
    tr.ROLE_NOISE: NOISE,
    tr.ROLE_SYSTOLE: S1,
    tr.ROLE_DIASTOLE: S2,
    tr.ROLE_NEUTRAL: "#c3c9cf",
    ROLE_ANNOTATION: ANNOTATION,
}

# Status tones: (text, tint behind it)
TONES = {
    "info": ("#9cc2f7", "#1c2a3d"),
    "success": ("#8fd6a4", "#1a3123"),
    "warning": ("#e8c27a", "#3a2e14"),
    "danger": ("#f08a8a", "#3d1d1d"),
    "issue": ("#f0a6e6", "#3a1f37"),
    "muted": (TEXT_3, SURFACE_3),
}

UI_FONT = "Segoe UI"
UI_SIZE = 9
MONO_FONTS = ["Cascadia Mono", "Consolas", "Courier New"]

BAND_ALPHA = 45          # debug windows drawn across a lane
FILL_ALPHA = 255         # the envelope body (ENVELOPE_FILL is already dark)
DIM_OPACITY = 0.15       # other traces while one legend entry is hovered


def blend(fg: str, bg: str, t: float) -> str:
    """fg at opacity t over bg, as an opaque hex colour (state rows rasterise opaque colours)."""
    a, b = QtGui.QColor(fg), QtGui.QColor(bg)
    mix = [round(x * t + y * (1 - t)) for x, y in zip((a.red(), a.green(), a.blue()), (b.red(), b.green(), b.blue()))]
    return "#{:02x}{:02x}{:02x}".format(*mix)


# States lane palettes. A beat is a bright sound followed by a faint tint of its phase.
STATE_FILLS = {
    "S1": S1,
    "S2": S2,
    "systole": blend(S1, SURFACE_0, 0.22),
    "diastole": blend(S2, SURFACE_0, 0.17),
    "unknown": TEXT_4,
    "noisy": blend(NOISE, SURFACE_0, 0.55),
}
# Annotation spans: hand-made at full strength, ones still as the algorithm seeded them faded.
ANNOTATION_FILLS = {
    "S1|hand": S1, "S2|hand": S2, "noisy|hand": STATE_FILLS["noisy"],
    "S1|algorithm": blend(S1, SURFACE_0, 0.5), "S2|algorithm": blend(S2, SURFACE_0, 0.5),
    "noisy|algorithm": blend(NOISE, SURFACE_0, 0.35),
}
DISAGREEMENT_FILLS = {"missed": ISSUE, "swapped": blend(ISSUE, TEXT, 0.45), "extra": blend(ISSUE, SURFACE_0, 0.5)}
DEFECT_FILLS = {"defect": ISSUE}


# ─────────────────────────────────────────────────────────────────────────────
# Trace styles
# ─────────────────────────────────────────────────────────────────────────────

DASH_SOLID, DASH_DASH, DASH_DOT, DASH_DASHDOT = "solid", "dash", "dot", "dashdot"
_LINE_VARIANTS = {False: (DASH_SOLID, DASH_DOT, DASH_DASHDOT), True: (DASH_DASH, DASH_DASHDOT, DASH_DOT)}
_SYMBOLS = ("o", "t", "s", "d", "x")


@dataclass(frozen=True)
class TraceMeta:
    """What styling and the legends need to know about a trace (engine traces and the workspace's own)."""
    name: str
    lane: str
    kind: str
    group: str
    role: str = ""
    estimate: bool = False
    visible: bool = False  # shown by default

    @classmethod
    def of(cls, t: tr.Trace) -> "TraceMeta":
        return cls(t.name, t.lane, t.kind, t.group, t.role, t.estimate, t.visible)


@dataclass(frozen=True)
class TraceStyle:
    color: str
    dash: str = DASH_SOLID
    width: float = 1.0
    symbol: str = "o"
    hollow: bool = False   # points drawn as outlines
    fill: bool = False     # filled down to zero (the envelope)


def stage_color(group: str) -> str:
    if group in STAGE_COLORS:
        return STAGE_COLORS[group]
    return EXTRA_STAGE_COLORS[sum(map(ord, group)) % len(EXTRA_STAGE_COLORS)]


def stage_order(groups: Iterable[str]) -> List[str]:
    """Stages in pipeline order; groups the engine invents go after Pass 3, before Result."""
    known = list(DEBUG_STAGES)
    extra = sorted(g for g in set(groups) if g not in STAGE_COLORS)
    return known + extra + list(ALWAYS_LISTED)


def assign_styles(metas: Iterable[TraceMeta]) -> Dict[str, TraceStyle]:
    """A style per trace. Traces that would look identical in one lane (same colour, kind and
    dashing) get the next dash pattern or marker in catalog order, so every entry stays distinct."""
    used: Dict[Tuple[str, str, str, bool], int] = {}
    out: Dict[str, TraceStyle] = {}
    for m in metas:
        color = ROLE_COLORS.get(m.role) or stage_color(m.group)
        key = (m.lane, color, m.kind, m.estimate)
        n = used.get(key, 0)
        used[key] = n + 1
        if m.kind == tr.KIND_POINTS:
            out[m.name] = TraceStyle(color, symbol=_SYMBOLS[n % len(_SYMBOLS)], hollow=m.role == tr.ROLE_NOISE)
        elif m.kind == tr.KIND_LINE:
            variants = _LINE_VARIANTS[m.estimate]
            out[m.name] = TraceStyle(color, dash=variants[n % len(variants)],
                                     width=1.6 if m.group == RESULT and not m.estimate else 1.0,
                                     fill=m.role == tr.ROLE_ENVELOPE)
        else:
            out[m.name] = TraceStyle(color)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Qt helpers
# ─────────────────────────────────────────────────────────────────────────────

_QT_DASH = {
    DASH_SOLID: QtCore.Qt.PenStyle.SolidLine, DASH_DASH: QtCore.Qt.PenStyle.DashLine,
    DASH_DOT: QtCore.Qt.PenStyle.DotLine, DASH_DASHDOT: QtCore.Qt.PenStyle.DashDotLine,
}


def qcolor(c: str, alpha: int = 255) -> QtGui.QColor:
    col = QtGui.QColor(c)
    col.setAlpha(alpha)
    return col


def pen(style: TraceStyle, alpha: int = 255) -> QtGui.QPen:
    p = pg.mkPen(qcolor(style.color, alpha), width=style.width)
    p.setStyle(_QT_DASH[style.dash])
    p.setCosmetic(True)
    return p


def mono_font(size: float = UI_SIZE) -> QtGui.QFont:
    f = QtGui.QFont()
    f.setFamilies(MONO_FONTS)
    f.setPointSizeF(size)
    f.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    return f


def set_prop(w: QtWidgets.QWidget, name: str, value) -> None:
    """Set a dynamic property the stylesheet keys on, and re-apply the stylesheet."""
    w.setProperty(name, value)
    w.style().unpolish(w)
    w.style().polish(w)


def badge(text: str = "", tone: str = "info") -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text)
    lab.setProperty("badge", tone)
    return lab


def style_axis(axis: pg.AxisItem, mono: bool = True) -> None:
    axis.setPen(pg.mkPen(BORDER_STRONG))
    axis.setTextPen(pg.mkPen(TEXT_3))
    axis.setTickFont(mono_font(8) if mono else QtGui.QFont(UI_FONT, 8))


def _stylesheet() -> str:
    tones = "\n".join(
        f'QLabel[badge="{k}"] {{ color: {fg}; background: {bg}; border-radius: 4px; padding: 1px 7px; }}'
        for k, (fg, bg) in TONES.items())
    accent_tint = blend(ACCENT, SURFACE_2, 0.16)
    accent_edge = blend(ACCENT, SURFACE_2, 0.45)
    return f"""
QMainWindow, QDialog {{ background: {SURFACE_2}; }}
QToolTip {{ background: {SURFACE_3}; color: {TEXT}; border: 1px solid {BORDER}; padding: 6px;
            font-family: "{MONO_FONTS[0]}", "{MONO_FONTS[1]}"; font-size: 8.5pt; }}
QMenuBar {{ background: {SURFACE_1}; color: {TEXT_2}; border-bottom: 1px solid {LINE}; }}
QMenuBar::item {{ padding: 4px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {SURFACE_3}; color: {TEXT}; }}
QMenu {{ background: {SURFACE_3}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 5px 22px 5px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background: {accent_tint}; }}
QMenu::separator {{ height: 1px; background: {LINE}; margin: 4px 6px; }}

QTabWidget::pane {{ border: none; border-top: 1px solid {LINE}; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{ background: transparent; color: {TEXT_3}; padding: 6px 14px; border: none;
                border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {TEXT_2}; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}

QPushButton {{ background: transparent; color: {TEXT_2}; border: 1px solid {BORDER}; border-radius: 5px;
               padding: 4px 10px; }}
QPushButton:hover {{ background: {SURFACE_3}; color: {TEXT}; border-color: {BORDER_STRONG}; }}
QPushButton:pressed {{ background: {LINE}; }}
QPushButton:checked {{ background: {accent_tint}; color: {TEXT}; border-color: {accent_edge}; }}
QPushButton:disabled {{ color: {TEXT_4}; border-color: {LINE}; }}
QPushButton[primary="true"] {{ background: {ACCENT}; color: {ON_ACCENT}; border-color: {ACCENT}; font-weight: 600;
                               padding: 4px 16px; }}
QPushButton[primary="true"]:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QToolButton {{ background: transparent; color: {TEXT_3}; border: none; border-radius: 4px; padding: 0 4px; }}
QToolButton:hover {{ color: {TEXT}; background: {SURFACE_3}; }}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {SURFACE_3}; color: {TEXT}; border: 1px solid {SURFACE_3};
    border-radius: 4px; padding: 3px 6px; selection-background-color: {accent_tint}; }}
QComboBox:hover, QSpinBox:hover, QLineEdit:hover {{ border-color: {BORDER_STRONG}; }}
QComboBox:focus, QSpinBox:focus, QLineEdit:focus {{ border-color: {accent_edge}; }}
QComboBox QAbstractItemView {{ background: {SURFACE_3}; color: {TEXT}; border: 1px solid {BORDER};
    selection-background-color: {accent_tint}; outline: 0; }}
QLineEdit[invalid="true"] {{ color: {TONES['danger'][0]}; border-color: {TONES['danger'][0]}; }}
QCheckBox {{ color: {TEXT_2}; spacing: 7px; }}
QCheckBox:hover {{ color: {TEXT}; }}
QCheckBox::indicator {{ width: 12px; height: 12px; border: 1px solid {BORDER_STRONG}; border-radius: 3px;
                        background: {SURFACE_3}; }}
QCheckBox::indicator:hover {{ border-color: {TEXT_3}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QTableView, QListView, QTreeView, QTextBrowser {{ background: {SURFACE_1}; color: {TEXT}; border: none;
    gridline-color: {LINE}; selection-background-color: {accent_tint}; selection-color: {TEXT}; outline: 0; }}
QTableView::item {{ padding: 0 8px; border-bottom: 1px solid {LINE}; }}
QListView::item {{ padding: 2px 6px; }}
QListView::item:hover, QTableView::item:hover {{ background: {SURFACE_2}; }}
QListView::item:selected, QTableView::item:selected {{ background: {accent_tint}; color: {TEXT}; }}
QHeaderView::section {{ background: {SURFACE_1}; color: {TEXT_3}; border: none; border-bottom: 1px solid {LINE};
    padding: 5px 8px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle {{ background: {BORDER}; border-radius: 3px; margin: 2px; }}
QScrollBar::handle:vertical {{ min-height: 24px; }}
QScrollBar::handle:horizontal {{ min-width: 24px; }}
QScrollBar::handle:hover {{ background: {BORDER_STRONG}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QSplitter::handle {{ background: {LINE}; }}
QSplitter::handle:hover {{ background: {ACCENT}; }}

QFrame#dropLine {{ background: {ACCENT}; }}
QFrame#header {{ background: {SURFACE_2}; border-bottom: 1px solid {LINE}; }}
QFrame#laneLegend {{ background: {SURFACE_1}; border-right: 1px solid {LINE}; }}
QFrame#overviewLegend {{ background: {SURFACE_1}; border-right: 1px solid {LINE}; border-bottom: 1px solid {LINE}; }}
QFrame#laneBar {{ background: {SURFACE_1}; border-top: 1px solid {LINE}; }}
QFrame#sidePanel {{ background: {SURFACE_1}; border-left: 1px solid {LINE}; }}
QLabel#title {{ color: {TEXT}; font-weight: 600; font-size: 10pt; }}
QLabel#laneTitle {{ color: {TEXT}; font-weight: 600; }}
QLabel[tone="muted"] {{ color: {TEXT_3}; }}
QLabel[tone="secondary"] {{ color: {TEXT_2}; }}
QLabel[tone="accent"] {{ color: {ACCENT}; }}
QFrame#segmented {{ background: {SURFACE_3}; border-radius: 6px; }}
QPushButton[segment="true"] {{ background: transparent; color: {TEXT_2}; border: none; border-radius: 4px;
                               padding: 4px 12px; }}
QPushButton[segment="true"]:hover {{ color: {TEXT}; background: {SURFACE_2}; }}
QPushButton[segment="true"]:checked {{ background: {blend(ACCENT, SURFACE_3, 0.28)}; color: {TEXT}; }}
QPushButton[segment="true"]:disabled {{ color: {TEXT_4}; background: transparent; }}
QFrame#footer {{ background: {SURFACE_1}; border-top: 1px solid {LINE}; }}
QPushButton[chip="lane"] {{ border: 1px dashed {BORDER_STRONG}; border-radius: 10px; padding: 1px 10px; color: {TEXT_2}; }}
QPushButton[chip="lane"]:hover {{ border-style: solid; color: {TEXT}; }}
{tones}
"""


def apply(app: QtWidgets.QApplication) -> None:
    """Install the palette, stylesheet, fonts and PyQtGraph defaults."""
    app.setStyle("Fusion")
    app.setFont(QtGui.QFont(UI_FONT, UI_SIZE))
    pal = QtGui.QPalette()
    for role, color in (
        (QtGui.QPalette.ColorRole.Window, SURFACE_2), (QtGui.QPalette.ColorRole.WindowText, TEXT),
        (QtGui.QPalette.ColorRole.Base, SURFACE_1), (QtGui.QPalette.ColorRole.AlternateBase, SURFACE_2),
        (QtGui.QPalette.ColorRole.Text, TEXT), (QtGui.QPalette.ColorRole.Button, SURFACE_3),
        (QtGui.QPalette.ColorRole.ButtonText, TEXT_2), (QtGui.QPalette.ColorRole.BrightText, TEXT),
        (QtGui.QPalette.ColorRole.Highlight, blend(ACCENT, SURFACE_2, 0.35)),
        (QtGui.QPalette.ColorRole.HighlightedText, TEXT), (QtGui.QPalette.ColorRole.Link, ACCENT),
        (QtGui.QPalette.ColorRole.ToolTipBase, SURFACE_3), (QtGui.QPalette.ColorRole.ToolTipText, TEXT),
        (QtGui.QPalette.ColorRole.PlaceholderText, TEXT_4), (QtGui.QPalette.ColorRole.Light, BORDER_STRONG),
        (QtGui.QPalette.ColorRole.Midlight, BORDER), (QtGui.QPalette.ColorRole.Mid, LINE),
        (QtGui.QPalette.ColorRole.Dark, SURFACE_1), (QtGui.QPalette.ColorRole.Shadow, SURFACE_0),
    ):
        pal.setColor(role, QtGui.QColor(color))
    for role in (QtGui.QPalette.ColorRole.WindowText, QtGui.QPalette.ColorRole.Text,
                 QtGui.QPalette.ColorRole.ButtonText):
        pal.setColor(QtGui.QPalette.ColorGroup.Disabled, role, QtGui.QColor(TEXT_4))
    app.setPalette(pal)
    app.setStyleSheet(_stylesheet())
    pg.setConfigOptions(antialias=False, background=SURFACE_0, foreground=TEXT_3)


def tone_colors(tone: Optional[str]) -> Tuple[str, str]:
    return TONES.get(tone or "muted", TONES["muted"])
