"""
utils/theme.py
==============
The single source of truth for colour and type across the whole application.

Before this module the palette lived in four places that had quietly drifted:
Material Design hex codes in utils/constants.CHART_COLORS, a near-copy in
visualizations/_theme.py, a third set inlined in utils/ui.py's card helpers,
and a fourth in .streamlit/config.toml. Changing "the blue" meant finding all
four, and nothing failed if you missed one.

Everything now derives from the tokens below. config.toml still has to repeat
three of them because Streamlit reads it before Python runs — verify_fixes.py
asserts those stay in step.

DESIGN NOTES
------------
Accent (cyan) does exactly three jobs and no others:
    1. mark the selected row in a table
    2. rail the panel that currently has focus
    3. highlight the single figure a page is about
It is never used to make something look nice, because an accent that appears
everywhere stops meaning anything.

UP and DOWN are reserved. In a returns tool green means up and red means down,
permanently, so no other element may use a green or red. This is why the
accent is cyan and not, say, a pleasant teal-green: a brand colour from the
semantic families gets read as a verdict on the number beside it.

Both semantic colours are deliberately desaturated from the Material values
they replace (#4CAF50 -> #3FB68B, #F44336 -> #E5484D). At full saturation a
200-row ranking table vibrates.
"""

from typing import Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# GROUND AND SURFACES
# ─────────────────────────────────────────────────────────────────────────────
# Neutrals carry a slight blue bias rather than being pure grey — pure grey
# beside a cyan accent reads as muddy.

GROUND      = "#000000"   # the page. Actual black, not near-black.
PANEL       = "#0A0B0E"   # barely lifted — separates a table header from its rows
PANEL_HI    = "#101219"   # inputs, hovered rows
RULE        = "#1E2230"   # visible hairline
RULE_SOFT   = "#14171F"   # row separators

INK         = "#E6EAF2"   # primary text and figures
INK_DIM     = "#B4BFCF"   # captions, secondary prose
INK_FAINT   = "#8C99AC"   # micro-labels, units

# ─────────────────────────────────────────────────────────────────────────────
# TWO COLOURS DRIVE ATTENTION
# ─────────────────────────────────────────────────────────────────────────────
# On a lifted ground the palette had to stay muted or the hues buzzed against
# each other, which read as washed out. True black gives them nothing to buzz
# against, so saturation comes back up.
#
#   INDIGO — WHERE YOU ARE. Active page, selected row, focus ring, slider.
#            Chrome only, and the one hard rule below: it must never appear in
#            the series palette, or a selected row starts looking like a line.
#
#   EMBER  — WHAT MATTERS. The headline figure on a page, warnings, AND the
#            primary data series. Unlike indigo it is ALLOWED in charts,
#            because it means the same thing in both places: the subject under
#            examination. Ember on a KPI and ember on the fund's line are one
#            idea, not a collision.

ACCENT      = "#5E7CE8"   # indigo — selection and focus. Never on data.
ACCENT_DIM  = "#3F58BF"   # hover / pressed
ACCENT_WASH = "rgba(94,124,232,0.14)"   # selected-row fill

AMBER       = "#FF9E2C"   # ember — the subject: headline figure, primary series
EMBER       = AMBER       # the name used in the design templates
WARN        = AMBER

UP          = "#00D68F"   # gains. Reserved — never a series colour.
DOWN        = "#FF4D5E"   # losses. Reserved.
NEUTRAL     = "#6E7A8C"   # reference lines, zero, N/A

# The benchmark keeps ONE identity across every chart: white where it is drawn
# as a line, slate where it needs an area. White bars over black read as grey
# mush, which is the only reason the second value exists.
BENCHMARK      = "#FFFFFF"
BENCHMARK_FILL = "#8C99AC"

# ─────────────────────────────────────────────────────────────────────────────
# TYPE
# ─────────────────────────────────────────────────────────────────────────────
# Archivo for words, IBM Plex Mono for figures. Both bundled in static/fonts
# and served locally — see static/fonts/README.md.
#
# Archivo is a neutral grotesque in the Swiss/International line, which is the
# family Bloomberg's own identity sits in: tight apertures, low stroke contrast,
# no personality of its own. That last part is the point — a UI face should get
# out of the way of the numbers.
#
# It replaces Latin Modern (Computer Modern), which was beautiful and wrong for
# this: a print face whose hairlines measured only ~63% ink coverage at caption
# size on a dark screen, so small text could not be made properly legible at any
# colour. A grotesque has thick, even stems and simply does not have that
# problem — which is why the text tokens above could go back to a real
# hierarchy instead of all sitting at near-white.
#
# Figures stay monospaced: Plex Mono's digits are all 600 units, so decimal
# points line up down a ranking column.
#
# Neither family ships a Greek subset, so the beta and sigma in labels like
# "Market beta" fall back — hence the explicitly Greek-capable fallbacks below.

FONT_TEXT = ('Archivo, "Segoe UI", system-ui, -apple-system, '
             '"Helvetica Neue", "DejaVu Sans", sans-serif')
FONT_MONO = ('"IBM Plex Mono", ui-monospace, SFMono-Regular, '
             'Menlo, Consolas, "DejaVu Sans Mono", monospace')

# Kept as aliases so existing call sites keep working.
FONT_SANS = FONT_TEXT
FONT_CAPS = FONT_TEXT     # micro-labels are CSS uppercase now, not a caps face

# Plotly wants a plain family string, not a CSS stack with quotes.
PLOTLY_MONO = "IBM Plex Mono, monospace"
PLOTLY_SANS = "Archivo, sans-serif"
PLOTLY_TEXT = PLOTLY_SANS

# ─────────────────────────────────────────────────────────────────────────────
# SERIES PALETTE
# ─────────────────────────────────────────────────────────────────────────────
# Ordered for distinguishability at the front, where most charts stop. UP and
# DOWN are excluded on purpose: a fund's line must never accidentally be the
# colour that means "gain" elsewhere on the same screen.

# Blue is RESERVED FOR CHROME. Indigo marks selection, so no series may sit
# near it — the ice blue in slot 6 is deliberately much lighter and cooler than
# the accent, and there is no mid-blue in the list at all.
#
# Assigned in fixed order and never cycled. UP and DOWN are excluded so a fund's
# line is never accidentally the colour that means "gain" elsewhere on screen.
#
# Checked with a colour-vision validator against a black surface rather than by
# eye: worst adjacent pair separates at deltaE 8.6 under deuteranopia and 15.9
# for normal vision, and all six clear 3:1 contrast. The validator also wants
# marks inside a mid-lightness band and these sit above it — dimming them to
# comply collapsed colour-blind separation from 8.6 to 4.9, so the brightness
# stays and the deviation is deliberate.
CHART_SERIES: List[str] = [
    "#FF9E2C",   # ember      — the subject
    "#1FBDD1",   # cyan
    "#E8479E",   # magenta
    "#A97BE0",   # violet
    "#C8CF45",   # lime
    "#7FD4FF",   # ice
    "#7FE05C",   # spring
    "#D98FE8",   # orchid
]

# EIGHT, and that is the ceiling — not an arbitrary stopping point.
#
# With indigo reserved for selection and green and red reserved for direction,
# the hue circle genuinely does not hold more: of fourteen further candidates
# tested, twelve landed within 20 degrees of a reserved colour or within 16 of
# a slot already taken. A ninth series is therefore not a generated hue — it
# folds into NEUTRAL, and the chart should be faceted or the list trimmed
# instead. series_colour() enforces that rather than cycling, because cycling
# silently gives two different funds the same colour on one chart.
MAX_SERIES = len(CHART_SERIES)


def series_colour(index: int) -> str:
    """Colour for the nth series. Beyond the palette, grey — never a repeat."""
    return CHART_SERIES[index] if 0 <= index < MAX_SERIES else NEUTRAL

# Quartiles are a ranked scale, so they read as a ramp rather than as twelve
# unrelated hues. Q1 borrows the UP green and Q4 the DOWN red because here the
# ranking genuinely is better/worse — the one place the semantics apply to
# something that is not a return.
QUARTILE: Dict[str, str] = {
    "Q1":  UP,
    "Q2":  "#7FD6B4",
    "Q3":  "#FF9E2C",
    "Q4":  DOWN,
    "N/A": NEUTRAL,
}

# Portfolio / fund slot identity. Replaces the coloured-circle emoji that used
# to distinguish Portfolio A from Portfolio B.
# The primary DATA colour. Charts reach for this, never for ACCENT: a NAV line
# and the active tab must not be the same hue, or the accent stops meaning
# "where you are" and just becomes the house colour again.
DATA_PRIMARY = CHART_SERIES[0]

SLOTS: List[str] = ["#FF9E2C", "#1FBDD1", "#E8479E", "#A97BE0",
                    "#C8CF45", "#7FD4FF", "#FFD166", "#5FD9A8"]

# Market-regime colours. Bull/bear legitimately map to up/down.
REGIME: Dict[str, str] = {
    "Bull":     UP,
    "Sideways": WARN,
    "Bear":     DOWN,
}

# ─────────────────────────────────────────────────────────────────────────────
# TEXT MARKS
# ─────────────────────────────────────────────────────────────────────────────
# Replacements for the emoji that carried real meaning. These are text glyphs,
# not emoji: they inherit colour, align on the text baseline, and render the
# same on every platform instead of turning into a colour picture.

MARK_YES  = "✓"     # check
MARK_NO   = "·"     # middle dot — absence, without shouting
MARK_UP   = "▲"     # up triangle
MARK_DOWN = "▼"     # down triangle
MARK_DL   = "↓"     # download arrow


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLESHEET
# ─────────────────────────────────────────────────────────────────────────────

def _css() -> str:
    """The stylesheet injected once per page render."""
    return f"""
<style>
/* Archivo + IBM Plex Mono, served from ./static (server.enableStaticServing).
   No CDN: the app runs locally and must not depend on the network for its
   typeface. 120 KB total, browser-cached. */
@font-face {{ font-family:'Archivo'; font-weight:400; font-style:normal;
  font-display:swap; src:url('app/static/fonts/Archivo-400.woff2') format('woff2'); }}
@font-face {{ font-family:'Archivo'; font-weight:500; font-style:normal;
  font-display:swap; src:url('app/static/fonts/Archivo-500.woff2') format('woff2'); }}
@font-face {{ font-family:'Archivo'; font-weight:600; font-style:normal;
  font-display:swap; src:url('app/static/fonts/Archivo-600.woff2') format('woff2'); }}
@font-face {{ font-family:'Archivo'; font-weight:700; font-style:normal;
  font-display:swap; src:url('app/static/fonts/Archivo-700.woff2') format('woff2'); }}
@font-face {{ font-family:'IBM Plex Mono'; font-weight:400; font-style:normal;
  font-display:swap; src:url('app/static/fonts/PlexMono-400.woff2') format('woff2'); }}
@font-face {{ font-family:'IBM Plex Mono'; font-weight:500; font-style:normal;
  font-display:swap; src:url('app/static/fonts/PlexMono-500.woff2') format('woff2'); }}
@font-face {{ font-family:'IBM Plex Mono'; font-weight:600; font-style:normal;
  font-display:swap; src:url('app/static/fonts/PlexMono-600.woff2') format('woff2'); }}

:root {{
  --mf-ground:{GROUND};   --mf-panel:{PANEL};      --mf-panel-hi:{PANEL_HI};
  --mf-rule:{RULE};       --mf-rule-soft:{RULE_SOFT};
  --mf-ink:{INK};         --mf-ink-dim:{INK_DIM};  --mf-ink-faint:{INK_FAINT};
  --mf-accent:{ACCENT};   --mf-up:{UP};            --mf-down:{DOWN};
  --mf-accent-wash:{ACCENT_WASH};
  --mf-mono:{FONT_MONO};  --mf-sans:{FONT_TEXT};
  --mf-amber:{AMBER};
}}

html, body, [class*="css"] {{ font-family: var(--mf-sans); }}

/* ── Density ───────────────────────────────────────────────────────────
   Streamlit ships generous padding suited to a consumer form. A terminal
   wants roughly a third more on screen. */
.block-container {{ padding-top:2.2rem; padding-bottom:3rem; max-width:1600px; }}
[data-testid="stVerticalBlock"] {{ gap:0.65rem; }}
hr {{ margin:0.9rem 0; border-color:var(--mf-rule); }}

/* ── Figures ───────────────────────────────────────────────────────────
   The core change: every number monospaced with tabular figures so that
   decimal points align down a column. */
[data-testid="stMetricValue"] {{
  font-family: var(--mf-mono) !important;
  font-variant-numeric: tabular-nums;
  font-size: 1.3rem !important;
  font-weight: 500 !important;
  letter-spacing: -0.01em;
  color: var(--mf-ink);
}}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {{
  /* Uppercase mono micro-labels. Archivo would work too, but the mono keeps
     the label and the figure beneath it on the same rhythm, and a grotesque
     has even enough stems that CSS uppercase looks intentional rather than
     stretched — which is not true of a fine-stroked print face. */
  font-family: var(--mf-mono) !important;
  font-size: 0.64rem !important;
  letter-spacing: 0.11em;
  text-transform: uppercase;
  color: var(--mf-ink-faint) !important;
  font-weight: 500 !important;
}}
[data-testid="stMetricDelta"] {{
  font-family: var(--mf-mono) !important;
  font-variant-numeric: tabular-nums;
  font-size: 0.76rem !important;
}}
[data-testid="stMetric"] {{
  padding: 0.5rem 0.8rem 0.55rem;
  border-left: 1px solid var(--mf-rule-soft);
  border-bottom: 1px solid var(--mf-rule);
  background: transparent;
}}
/* The first metric in a row is the one the page is about: ember rail, ember
   figure. Every other metric stays in plain ink, so "the headline" is a
   position AND a colour rather than a guess. */
[data-testid="stHorizontalBlock"] > div:first-child [data-testid="stMetric"] {{
  border-left: 2px solid {AMBER};
}}
[data-testid="stHorizontalBlock"] > div:first-child [data-testid="stMetricValue"] {{
  color: {AMBER};
}}

/* Tables: monospace figures, hairline rules, square corners. */
[data-testid="stDataFrame"] {{ font-family: var(--mf-mono); }}
[data-testid="stDataFrame"] div[role="gridcell"] {{
  font-variant-numeric: tabular-nums;
  font-size: 0.8rem;
}}
[data-testid="stDataFrame"] div[role="columnheader"] {{
  font-family: var(--mf-mono);
  font-size: 0.66rem;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--mf-ink-faint);
}}
[data-testid="stDataFrame"] > div {{ border-radius:0 !important; border-color:var(--mf-rule) !important; }}
[data-testid="stDataFrame"] div[role="row"]:hover {{ background: var(--mf-accent-wash); }}

/* ── Chrome ────────────────────────────────────────────────────────────
   Square everything. Rounded corners on a data panel read as a consumer
   card; a terminal separates with rules, not with boxes. */
.stButton > button,
.stDownloadButton > button,
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] input,
[data-baseweb="select"] > div,
[data-baseweb="input"] {{
  border-radius: 2px !important;
}}
.stButton > button, .stDownloadButton > button {{
  font-family: var(--mf-mono);
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  border: 1px solid var(--mf-rule);
  background: var(--mf-panel);
  color: var(--mf-ink);
  transition: border-color .12s ease, color .12s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  border-color: var(--mf-accent);
  color: var(--mf-accent);
}}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {{
  outline: 2px solid var(--mf-accent);
  outline-offset: 1px;
}}

/* Tabs read as a terminal's view switcher: a rule with an accent underline
   on the active view, rather than pill-shaped buttons. */
.stTabs [data-baseweb="tab-list"] {{
  gap: 0; border-bottom: 1px solid var(--mf-rule); background: transparent;
}}
.stTabs [data-baseweb="tab"] {{
  font-family: var(--mf-mono);
  font-size: 0.72rem;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  padding: 0.45rem 0.95rem;
  border-radius: 0;
  color: var(--mf-ink-faint);
}}
.stTabs [aria-selected="true"] {{
  color: var(--mf-accent) !important;
  border-bottom: 2px solid var(--mf-accent);
  background: transparent !important;
}}

/* Section headings: small, uppercase, mono — hierarchy by case and weight
   rather than by an emoji sitting in front of the words. */
h1 {{ font-size:1.42rem !important; font-weight:600 !important; letter-spacing:-0.02em; }}
h2 {{ font-size:1.05rem !important; font-weight:600 !important; letter-spacing:-0.01em; }}
h3 {{
  font-family: var(--mf-mono) !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--mf-ink-dim) !important;
  font-weight: 600 !important;
  margin-bottom: 0.3rem !important;
}}

/* Sidebar */
/* The sidebar was a grey slab beside a dark page — the single dullest thing
   on screen. It is now the same black as the page, separated by a rule, and
   the only colour in it is the indigo bar on the page you are actually on. */
[data-testid="stSidebar"] {{
  background: var(--mf-ground);
  border-right: 1px solid var(--mf-rule);
}}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
  border-left: 2px solid transparent;
  border-radius: 0;
  padding-left: 10px;
}}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"],
[data-testid="stSidebar"] [data-testid="stSidebarNav"] li > div[class*="active"] a {{
  background: var(--mf-accent-wash) !important;
  border-left-color: var(--mf-accent);
  color: var(--mf-ink) !important;
  font-weight: 500;
}}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] {{
  background: var(--mf-panel-hi) !important;
  border-color: var(--mf-rule) !important;
}}
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
  font-family: var(--mf-mono);
  font-size: 0.64rem;
  text-transform: uppercase;
  /* Tighter tracking than the main-pane labels on purpose: the sidebar
     column is ~250px, and 0.1em on an uppercase label wrapped
     "RISK-FREE RATE (%)" onto two lines with "(%)" stranded. */
  letter-spacing: 0.03em;
  color: var(--mf-ink-faint);
  white-space: nowrap;
}}

/* Alerts: a left rule instead of a filled rounded block. */
[data-testid="stAlert"] {{
  border-radius: 0 !important;
  border: none;
  border-left: 2px solid var(--mf-accent);
  background: var(--mf-panel);
}}
/* Warnings and errors are the app asking for attention, so they take ember
   rather than the selection colour. */
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]),
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {{
  border-left-color: {AMBER};
}}

/* Progress + slider pick up the accent. */
[data-testid="stProgress"] > div > div > div {{ background: var(--mf-accent); }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background: var(--mf-accent); }}

/* Captions.
   Streamlit styles the caption's inner <p> directly, so a rule on the
   container alone loses the cascade — the tokens were lifted to near-white and
   the metadata line under the fund name stayed grey anyway. Hence the explicit
   descendants and !important. */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] span,
[data-testid="stCaptionContainer"] li {{
  color: var(--mf-ink-dim) !important;
  /* No size or weight compensation needed any more: a grotesque has even,
     thick stems and renders at full ink at this size, which the print face
     it replaced could not. */
  font-size: 0.82rem;
}}
/* Small print elsewhere (help text, markdown <small>) follows the same floor. */
small, .stMarkdown small {{ color: var(--mf-ink-dim); }}

/* Inline code — scheme codes, metric keys. Dimming these made an identifier
   you might want to copy the least readable thing on the page. */
code, .stMarkdown code, [data-testid="stMarkdownContainer"] code {{
  color: var(--mf-ink) !important;
  background: var(--mf-panel-hi);
  border: 1px solid var(--mf-rule-soft);
  border-radius: 2px;
  padding: 0 4px;
}}
</style>
"""


def inject_css() -> None:
    """
    Install the stylesheet. Idempotent per script run: Streamlit re-executes
    the page top to bottom on every interaction, so this is called again on
    each rerun, but a second identical <style> block is harmless and the
    session-state guard keeps the DOM clean within a single run.
    """
    import streamlit as st
    st.markdown(_css(), unsafe_allow_html=True)


def rgba(hex_color: str, alpha: float) -> str:
    """
    A token colour at partial opacity, as a CSS/Plotly rgba() string.

    Chart fills and hover backgrounds need translucency, and before this the
    only way to write one was an rgba() literal — which is how a dozen
    Material Design colours survived a hex-only search and shipped in the
    charts after the palette had supposedly been replaced.
    """
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def num(value: str, color: str = INK, size: str = "1rem") -> str:
    """Inline HTML for a figure set in the tabular mono face."""
    return (f'<span style="font-family:{FONT_MONO};font-variant-numeric:tabular-nums;'
            f'color:{color};font-size:{size}">{value}</span>')


def label(text: str, color: str = INK_FAINT) -> str:
    """Inline HTML for an uppercase micro-label."""
    return (f'<span style="font-family:{FONT_MONO};font-size:0.63rem;'
            f'letter-spacing:0.13em;text-transform:uppercase;color:{color}">{text}</span>')
