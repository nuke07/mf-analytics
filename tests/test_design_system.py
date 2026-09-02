"""
The cyan terminal design system, tested.

Three classes of regression this guards against:

  1. Palette drift. The colours used to live in four places — constants.py,
     visualizations/_theme.py, ui.py's card helpers, and config.toml — and
     nothing failed when you changed one and missed the rest. Now everything
     derives from utils/theme.py, and config.toml (which Streamlit reads before
     Python runs, so it MUST repeat three values) is asserted against it.

  2. Emoji creeping back. The sweep removed 198 decorative glyphs. A new page
     header with a chart emoji in front of it would undo the pass silently.

  3. Semantic colour being spent on decoration. In a returns tool green means
     up and red means down. If the accent, or a chart series, drifts into
     either family, a brand colour starts reading as a verdict on the number
     beside it.

Run:  python test_design_system.py
"""
import glob
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
# ROOT is the repo root, one level up from tests/. Every path in this file
# hangs off it, so it must not be the test's own directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def relative_luminance(rgb):
    def ch(c):
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    la, lb = relative_luminance(hex_to_rgb(a)), relative_luminance(hex_to_rgb(b))
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def hue(h):
    """Hue in degrees, 0-360."""
    r, g, b = (c / 255 for c in hex_to_rgb(h))
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    if d == 0:
        return None
    if mx == r:
        return (60 * ((g - b) / d)) % 360
    if mx == g:
        return (60 * ((b - r) / d) + 120) % 360
    return (60 * ((r - g) / d) + 240) % 360


import utils.theme as T

# ── 1. One source of truth ─────────────────────────────────────────────────
print("\n[TOKENS] Colour has exactly one definition")

from utils.constants import CHART_COLORS, QUARTILE_COLORS
from visualizations import _theme as vt

check("constants.CHART_COLORS is theme.CHART_SERIES",
      CHART_COLORS is T.CHART_SERIES)
check("constants.QUARTILE_COLORS is theme.QUARTILE",
      QUARTILE_COLORS is T.QUARTILE)
check("chart module draws its series from theme",
      vt.COLORS is T.CHART_SERIES)
check("chart up/down match the semantic tokens",
      vt.UP_COLOR == T.UP and vt.DOWN_COLOR == T.DOWN)
check("chart axis ticks use the tabular mono face",
      "Mono" in getattr(vt, "TICK_FAMILY", ""), vt.TICK_FAMILY)

# config.toml is read by Streamlit before Python runs, so it duplicates three
# tokens by necessity. That duplication is exactly what drifts.
cfg = open(os.path.join(ROOT, ".streamlit", "config.toml"), encoding="utf-8").read()


def toml_val(key):
    m = re.search(rf'^{key}\s*=\s*"([^"]+)"', cfg, re.M)
    return m.group(1) if m else None


for key, token, name in (("primaryColor", T.ACCENT, "ACCENT"),
                         ("backgroundColor", T.GROUND, "GROUND"),
                         ("secondaryBackgroundColor", T.PANEL, "PANEL"),
                         ("textColor", T.INK, "INK")):
    check(f"config.toml {key} matches theme.{name}",
          (toml_val(key) or "").upper() == token.upper(),
          f"{toml_val(key)} vs {token}")

# No page may hardcode a hex colour — that is how the old palette survived
# three previous attempts to change it.
page_srcs = {os.path.basename(p): open(p, encoding="utf-8").read()
             for p in sorted(glob.glob(os.path.join(ROOT, "pages", "*.py")))
             + [os.path.join(ROOT, "app.py")]}


def code_only(src):
    src = re.sub(r'("""|\'\'\')(?:.|\n)*?\1', "", src)
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))


HEX = re.compile(r"#[0-9A-Fa-f]{6}\b")
hardcoded = {f: sorted(set(HEX.findall(code_only(s))))
             for f, s in page_srcs.items() if HEX.search(code_only(s))}
check("no page hardcodes a hex colour", not hardcoded, f"{hardcoded}")

# ── 2. The Material palette is gone ────────────────────────────────────────
print("\n[PALETTE] The consumer palette is gone")

MATERIAL = ["#2196F3", "#4CAF50", "#F44336", "#FF9800", "#9C27B0", "#00BCD4"]
tree = []
for d in ("pages", "utils", "visualizations", "analytics", "data"):
    tree += sorted(glob.glob(os.path.join(ROOT, d, "*.py")))
tree.append(os.path.join(ROOT, "app.py"))

left = []
for f in tree:
    body = code_only(open(f, encoding="utf-8").read())
    for m in MATERIAL:
        if m.lower() in body.lower():
            left.append(f"{os.path.basename(f)}:{m}")
check("no Material Design hex survives in live code", not left, f"{left[:4]}")

# Hex is not the only spelling. Chart fills need translucency, so a dozen
# Material colours were written as rgba() literals and sailed through a
# hex-only scan — the distribution bars on the home page were still Google
# blue after the palette had supposedly been replaced. theme.rgba() is the
# supported way to get a token at partial opacity.
MATERIAL_RGB = [(33, 150, 243), (76, 175, 80), (244, 67, 54),
                (255, 152, 0), (156, 39, 176), (0, 188, 212)]
rgba_left = []
for f in tree:
    body = code_only(open(f, encoding="utf-8").read())
    for r, g, b in MATERIAL_RGB:
        if re.search(rf"rgba\(\s*{r}\s*,\s*{g}\s*,\s*{b}\s*,", body):
            rgba_left.append(f"{os.path.basename(f)}:rgba({r},{g},{b})")
check("no Material Design rgba() survives either", not rgba_left, f"{rgba_left[:4]}")
check("theme exposes an rgba() helper so fills can use tokens",
      callable(getattr(T, "rgba", None))
      and T.rgba("#00C2D1", 0.5) == "rgba(0,194,209,0.5)",
      T.rgba(T.ACCENT, 0.5))
check("the ground is true black", T.GROUND == "#000000", T.GROUND)
check("the accent is indigo", T.ACCENT.upper() == "#5E7CE8", T.ACCENT)
check("ember is the attention colour", T.EMBER.upper() == "#FF9E2C", T.EMBER)

# INDIGO means "where you are" and must never appear in a chart, or a selected
# row starts looking like a series. EMBER is deliberately allowed in both: it
# means "the subject under examination" whether that is a headline KPI or the
# fund's own line, so sharing it is consistency rather than collision.
check("indigo is not a chart series colour",
      T.ACCENT not in T.CHART_SERIES,
      "selection colour must not appear in data")
_near_indigo = [c for c in T.CHART_SERIES
                if hue(c) is not None and abs(hue(c) - hue(T.ACCENT)) < 22]
check("no series sits near the selection indigo", not _near_indigo, f"{_near_indigo}")
check("ember leads the series palette, by design",
      T.CHART_SERIES[0].upper() == T.EMBER.upper())
check("a second colour exists for attention",
      hasattr(T, "AMBER") and T.AMBER != T.ACCENT, getattr(T, "AMBER", None))
check("there is a named primary DATA colour, separate from the accent",
      getattr(T, "DATA_PRIMARY", None) == T.CHART_SERIES[0]
      and T.DATA_PRIMARY != T.ACCENT, getattr(T, "DATA_PRIMARY", None))

# No chart may draw a data mark in the interface accent. This is what actually
# stopped one colour owning the screen: the home-page bars, the NAV line, the
# scatter default and the Monte Carlo fan were all painted in the accent, so
# every chart in the app came out the same hue as the active tab.
_viz = sorted(glob.glob(os.path.join(ROOT, "visualizations", "*.py")))
_viz.append(os.path.join(ROOT, "app.py"))
_accent_in_data = []
for _f in _viz:
    for _i, _l in enumerate(open(_f, encoding="utf-8"), 1):
        if _l.lstrip().startswith("#") or "activecolor" in _l:
            continue                      # activecolor is chrome, not a data mark
        if "T.ACCENT" in _l:
            _accent_in_data.append(f"{os.path.basename(_f)}:{_i}")
check("no chart paints a data mark in the selection indigo",
      not _accent_in_data, f"{_accent_in_data[:4]}")

# Chart chrome: the terminal layout.
check("the plot area is the page, not a lifted panel", vt.BG_PLOT == T.GROUND, vt.BG_PLOT)

# Six charts painted their own plot background instead of taking it from the
# theme, so they kept a grey rectangle after the ground went black. Any chart
# that sets its own must set it to the ground.
_lifted = []
for _f in _viz + sorted(glob.glob(os.path.join(ROOT, "pages", "*.py"))):
    # _theme.py is the definition site — it uses BG_PLOT, which the check
    # directly above already asserts equals the ground.
    if os.path.basename(_f) == "_theme.py":
        continue
    for _i, _l in enumerate(open(_f, encoding="utf-8"), 1):
        if _l.lstrip().startswith("#") or "plot_bgcolor" not in _l:
            continue
        if "GROUND" not in _l and "BG_PLOT" not in _l:
            _lifted.append(f"{os.path.basename(_f)}:{_i}")
check("no chart paints its own lifted plot background", not _lifted, f"{_lifted[:4]}")
_lay = vt.base_layout(title="t", y_title="%")
check("the value axis is on the right", _lay.yaxis.side == "right")
check("the legend sits above the plot, not inside it", _lay.legend.y > 1, _lay.legend.y)
check("the x axis has ticks, not a lattice", _lay.xaxis.showgrid is False)
check("right margin leaves room for last-value badges", _lay.margin.r >= 70, _lay.margin.r)

# Last-value badges, including the collision case that made two indices ending
# 4 tenths of a percent apart print one label on top of the other.
import plotly.graph_objects as _go
_f = _go.Figure()
_f.add_scatter(y=[1, 2, 209.3], line=dict(color=T.CHART_SERIES[0]))
_f.add_scatter(y=[1, 2, 205.2], line=dict(color=T.CHART_SERIES[1]))
_f.add_scatter(y=[None, None, None])          # must not raise
vt.last_value_badges(_f)
_ann = _f.layout.annotations
check("a badge is placed per numeric series", len(_ann) == 2, f"{len(_ann)}")
check("crowded badges are pushed apart",
      abs(_ann[0].y - _ann[1].y) > 1.0, f"{abs(_ann[0].y - _ann[1].y):.1f}")
check("the badge still prints the TRUE value, not the nudged one",
      any("209.3" in a.text for a in _ann),
      "position may move; the number may not")
check("badge text is dark on the series colour",
      all(a.font.color == "#000000" for a in _ann))

_wired = [f for f in ("nav_chart", "alpha_charts", "rolling_returns")
          if "last_value_badges(fig)" in
          open(os.path.join(ROOT, "visualizations", f + ".py"), encoding="utf-8").read()]
check("the main line charts carry badges", len(_wired) == 3, f"{_wired}")

def saturation(h):
    r, g, b = (c / 255 for c in hex_to_rgb(h))
    mx, mn = max(r, g, b), min(r, g, b)
    if mx == mn:
        return 0.0
    l = (mx + mn) / 2
    return (mx - mn) / (2 - mx - mn) if l > 0.5 else (mx - mn) / (mx + mn)

# On true black there is nothing for a saturated hue to buzz against, so the
# marks are allowed their saturation back — that was the "pale" complaint.
check("series carry real saturation on black",
      min(saturation(c) for c in T.CHART_SERIES[:6]) > 0.45,
      f"min {min(saturation(c) for c in T.CHART_SERIES[:6]):.0%}")
check("the benchmark has one identity: white line, slate fill",
      T.BENCHMARK == "#FFFFFF" and T.BENCHMARK_FILL != "#FFFFFF")

# Categorical hues are assigned in fixed order and NEVER cycled. Cycling handed
# two different funds on one chart the identical colour with nothing to
# separate them; past the last slot the neutral is returned instead.
from visualizations._theme import get_color as _gc
check("series colours do not cycle past the palette",
      _gc(T.MAX_SERIES) == T.NEUTRAL and _gc(T.MAX_SERIES + 9) == T.NEUTRAL,
      f"slot {T.MAX_SERIES} -> {_gc(T.MAX_SERIES)}")
check("every real slot is distinct",
      len(set(T.CHART_SERIES)) == T.MAX_SERIES)
check("the palette covers the widest comparison the app allows",
      T.MAX_SERIES >= 8, f"{T.MAX_SERIES} slots vs 8 portfolio funds")

# ── 3. Semantic colour is not spent on decoration ──────────────────────────
print("\n[SEMANTICS] Green means up, red means down, and nothing else")

h_acc = hue(T.ACCENT)
check("the accent is not in the green or red families",
      not (75 <= h_acc <= 165) and not (h_acc >= 330 or h_acc <= 20),
      f"accent hue {h_acc:.0f}°")
check("up and down are distinguishable from each other",
      abs(hue(T.UP) - hue(T.DOWN)) > 60,
      f"up {hue(T.UP):.0f}° vs down {hue(T.DOWN):.0f}°")

# A series colour that lands on the semantic hues would make one fund's line
# read as "the good one".
clashes = []
for c in T.CHART_SERIES:
    hc = hue(c)
    if hc is None:
        continue
    for sem, label in ((T.UP, "UP"), (T.DOWN, "DOWN")):
        if abs(hc - hue(sem)) < 12:
            clashes.append(f"{c}~{label}")
check("no chart series collides with a semantic colour", not clashes, f"{clashes}")
check("semantics are desaturated from the Material originals",
      T.UP != "#4CAF50" and T.DOWN != "#F44336")

# ── 4. Contrast ────────────────────────────────────────────────────────────
print("\n[CONTRAST] Legible on the dark ground")

# Floors are far above the WCAG minimums on purpose. Computer Modern's
# hairlines need luminance that a sans-serif does not: at 6:1 the metadata line
# under the fund name was hard to read, and that is well inside "AA compliant".
# Lower than the Computer Modern era, and deliberately so: those floors existed
# to compensate for a face that only rendered ~63% of its ink at small sizes.
# A grotesque renders what it is given, so hierarchy is legible again.
for name, col, floor in (("ink", T.INK, 15.0), ("ink_dim", T.INK_DIM, 10.0),
                         ("ink_faint", T.INK_FAINT, 6.5),
                         ("accent", T.ACCENT, 4.5), ("up", T.UP, 3.0),
                         ("down", T.DOWN, 3.0)):
    r = contrast(col, T.GROUND)
    check(f"{name} clears {floor}:1 on the page ground", r >= floor, f"{r:.1f}:1")

# Micro-labels are small and grey by design, but must stay above the 3:1 floor
# for large/secondary text rather than disappearing into the panel.
r = contrast(T.INK_FAINT, T.PANEL)
check("the dimmest text is still bright on a panel", r >= 6.0, f"{r:.1f}:1")

# ── 5. Type ────────────────────────────────────────────────────────────────
print("\n[TYPE] Figures are tabular")

css = T._css()
check("the stylesheet sets tabular figures", "tabular-nums" in css)
check("metric values are monospaced",
      "stMetricValue" in css and "--mf-mono" in css)
check("dataframe cells are monospaced", "stDataFrame" in css)
# Micro-labels are set in a DRAWN small-caps face, so text-transform is
# deliberately absent: forcing uppercase first would throw away the
# caps-and-small-caps letterforms and give flat all-caps instead.
def rule_for(sel):
    """The declaration block of the first rule mentioning `sel`.

    Slicing a fixed number of characters after the selector was fragile: a long
    explanatory comment inside the rule pushed the declaration past the window
    and the check failed on correct CSS.
    """
    i = css.index(sel)
    j = css.index("{", i)
    return css[j:css.index("}", j)]

_lbl = rule_for("stMetricLabel")
check("metric labels are uppercase mono micro-labels",
      "--mf-mono" in _lbl and "text-transform: uppercase" in _lbl
      and "letter-spacing" in _lbl)
# Fonts are bundled and served locally, not fetched from a CDN — the app runs
# offline and a typeface that silently falls back is worse than one present.
check("no font is fetched from a CDN", "googleapis" not in css and "http" not in css)
check("the faces are registered for Streamlit to serve",
      "fontFaces" in cfg and "Archivo-400.woff2" in cfg
      and "PlexMono-400.woff2" in cfg)
check("no Latin Modern reference survives",
      "LMRoman" not in cfg and "Latin Modern" not in cfg)
check("static serving is on, or the font URLs 404",
      "enableStaticServing = true" in cfg)
for _f in ("Archivo-400", "Archivo-500", "Archivo-600", "Archivo-700",
           "PlexMono-400", "PlexMono-500", "PlexMono-600"):
    check(f"{_f}.woff2 is present in static/fonts",
          os.path.exists(os.path.join(ROOT, "static", "fonts", f"{_f}.woff2")))
check("the roles point at the bundled faces",
      'font                     = "Archivo' in cfg
      and 'codeFont                 = "IBM Plex Mono' in cfg)
check("corners are squared off", "border-radius: 2px" in css)
check("the accent styles the active tab",
      'aria-selected="true"' in css and "--mf-accent" in css)
check("focus is visible for keyboard users", "focus-visible" in css)

# ── 6. Emoji stay gone ─────────────────────────────────────────────────────
print("\n[EMOJI] 198 decorative glyphs removed, and they stay removed")

KEEP = set("₹✓✗▲▼■─↓→←↑−≤≥≈·—…’‘“”")


def decorative(ch):
    o = ord(ch)
    if ch in KEEP or o < 0x2000:
        return False
    return (0x1F000 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF
            or 0x2B00 <= o <= 0x2BFF
            or o in (0xFE0E, 0xFE0F, 0x200D, 0x2139, 0x2194, 0x21A9,
                     0x23F1, 0x2696, 0x2699))


offenders = []
for f in tree:
    for i, line in enumerate(open(f, encoding="utf-8"), 1):
        # page_icon sets the BROWSER TAB icon, outside the app's own surface.
        if line.lstrip().startswith("#") or "page_icon" in line:
            continue
        hits = [c for c in line if decorative(c)]
        if hits:
            offenders.append(f"{os.path.basename(f)}:{i} {''.join(hits)}")
check("no decorative glyph remains in a user-facing string",
      not offenders, f"{offenders[:4]}")

# The semantic ones must NOT have been deleted along with the rest.
check("the coverage marks survived as text glyphs",
      T.MARK_YES == "✓" and T.MARK_NO == "·",
      f"{T.MARK_YES!r} / {T.MARK_NO!r}")
dq = page_srcs["7_Data_Quality.py"]
check("the coverage matrix still marks yes/no",
      "MARK_YES" in dq and "MARK_NO" in dq)
pa = page_srcs["4_Portfolio_Analytics.py"]
check("Portfolio A/B keep their identity colour via swatch()",
      "swatch(" in pa and "SLOT_COLORS[0]" in pa)
fa = page_srcs["5_Factor_Attribution.py"]
check("the regime legend still names bull/sideways/bear",
      all(w in fa for w in ("Bull", "Sideways", "Bear")))

# ── 7. The sweep did not corrupt code ──────────────────────────────────────
print("\n[SWEEP SAFETY] The three things the first attempt broke")

check("category.replace(' ', '_') was not emptied",
      "replace('','_')" not in dq and "replace('', '_')" not in dq)
check("the sidebar markdown list kept its bullets",
      'f"- {COVERAGE_LABELS' in dq)
check("no st.* call was left with a dangling comma",
      not any(re.search(r"\(\s*\n\s*,", code_only(s)) for s in page_srcs.values()))

import py_compile
import tempfile
cdir = tempfile.mkdtemp()
broken = []
for f in tree:
    try:
        py_compile.compile(f, doraise=True,
                           cfile=os.path.join(cdir, os.path.basename(f) + "c"))
    except Exception as e:
        broken.append(f"{os.path.basename(f)}: {e}")
check("every module still compiles", not broken, f"{broken[:2]}")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
