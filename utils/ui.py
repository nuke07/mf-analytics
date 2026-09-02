"""
utils/ui.py
===========
Shared Streamlit UI building blocks.

Every page used to hand-roll its own sidebar. The risk-free-rate control was
copy-pasted six times, the TRI staleness block five times (carrying a broken
indent faithfully into every copy), and the pieces had drifted: page 9
defaulted the RF rate to 6.5 while every other page used 7.0, page 6 had no
TRI block and a hand-rolled refresh button that lost its confirmation
message, and the plan-type radio had three different help strings.

Defining each control once here means a change lands everywhere, and the
divergence cannot silently reappear.

Usage — compose in the order the page needs:

    from utils.ui import (
        sidebar_header, category_selector, plan_selector,
        rf_control, tri_status, sidebar_footer, kpi,
    )

    with st.sidebar:
        sidebar_header()
        category  = category_selector()
        plan_type = plan_selector()
        # ... page-specific widgets (fund pickers) go here ...
        rf_pct, rf_rate = rf_control()
        render_refresh_button()
        tri_status()
        sidebar_footer()
"""

from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

from utils.constants import (
    # APP_TITLE / APP_ICON are no longer imported here: the sidebar wordmark
    # is set in the mono face by sidebar_header() rather than composed from a
    # title string and a chart emoji.
    APP_SUBTITLE, APP_VERSION,
    CATEGORIES, DEFAULT_RISK_FREE_RATE, PLAN_TYPES, METRIC_HELP,
)
from utils.formatters import fmt_pct, fmt_ratio, fmt_days
from utils.session import render_refresh_button   # re-exported for convenience
from utils import theme as T

# Slider bounds for the risk-free rate, in percent.
RF_MIN, RF_MAX, RF_STEP = 4.0, 9.0, 0.1

# The RF rate is held in TWO session-state entries, and the split is load-bearing.
#
# RF_KEY is the slider's widget key. Binding the widget to a key (rather than
# passing `value=`) is what makes the −/+ buttons work: a keyless slider
# re-reads `value` only on its first render, so once the user had dragged it,
# every button increment was silently discarded.
#
# RF_STORE is a PLAIN key that no widget owns. Streamlit re-registers widgets
# when you navigate between pages in a multipage app, and a keyed slider whose
# identity is re-derived initialises from `value` — which rf_control does not
# pass — and therefore falls back to min_value. The symptom was that every page
# showed 7.0% while Rankings showed 4.0%, silently computing Sharpe, Sortino
# and every alpha at the wrong risk-free rate. Session state was correct at the
# time; the WIDGET discarded it.
#
# So the value of record lives in RF_STORE, which nothing can reset, and the
# widget key is re-seeded from it at the top of every run.
# RF_KEY is a PREFIX, not the key itself: each page gets its own slider entry,
# `rf_rate_<page id>`. A single shared widget key across pages is what caused
# the bug — Streamlit re-registers widgets on navigation, and the re-registered
# slider initialised from min_value instead of the value already in session
# state, so Rankings quietly ran at 4.0% while every other page showed 7.0%.
RF_KEY   = "rf_rate"
# The value of record. A plain key, owned by no widget, so nothing resets it.
RF_STORE = "rf_rate_value"
# Which page rendered the control last, so we know when the user has ARRIVED on
# a page (re-seed the slider from the store) versus merely triggered a rerun on
# the page they were already on (leave the slider alone, or a drag is undone).
RF_LAST_PAGE = "rf_last_page"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

def sidebar_header(show_subtitle: bool = False) -> None:
    """
    App identity at the top of the sidebar, and the one place the global
    stylesheet is installed.

    Every page calls this, which is why the CSS hook lives here rather than in
    a separate setup function that a new page could forget to call. Streamlit
    re-runs the whole script on each interaction, so this fires every render;
    that is intended and cheap.
    """
    T.inject_css()

    # Wordmark, not a logo: the app title set in the mono face with the
    # accent on the first word. This replaces a chart emoji standing in for
    # a brand.
    st.markdown(
        f'<div style="font-family:{T.FONT_MONO};font-size:0.82rem;'
        f'letter-spacing:0.14em;text-transform:uppercase;line-height:1.35;'
        f'margin-bottom:2px">'
        f'<span style="color:{T.ACCENT};font-weight:600">MF</span>'
        f'<span style="color:{T.INK}"> ANALYTICS</span></div>'
        f'<div style="font-family:{T.FONT_MONO};font-size:0.6rem;'
        f'letter-spacing:0.1em;color:{T.INK_FAINT}">'
        f'QUANTITATIVE · INDIA</div>',
        unsafe_allow_html=True,
    )
    if show_subtitle:
        st.caption(APP_SUBTITLE)
    st.divider()


def category_selector(label: str = "Category") -> str:
    """
    Category dropdown, persisted across pages.

    Guards against a stale session value that is no longer in CATEGORIES,
    which would otherwise raise ValueError out of CATEGORIES.index().
    """
    previous = st.session_state.get("selected_category", CATEGORIES[0])
    index    = CATEGORIES.index(previous) if previous in CATEGORIES else 0

    category = st.selectbox(
        label, CATEGORIES, index=index,
        help="Funds are ranked and compared within a single category only.",
    )
    st.session_state["selected_category"] = category
    return category


def plan_selector() -> str:
    """Direct/Regular radio, persisted across pages."""
    previous = st.session_state.get("plan_type", PLAN_TYPES[0])
    index    = PLAN_TYPES.index(previous) if previous in PLAN_TYPES else 0

    plan_type = st.radio(
        "Plan Universe", PLAN_TYPES,
        index=index, horizontal=True,
        help="Direct: no distributor commission. Regular: distributor-advised. "
             "Never mix both — Direct plans look better purely because of the "
             "fee difference, not manager skill.",
    )
    st.session_state["plan_type"] = plan_type
    return plan_type


def fund_selector(
    fund_names: List[str],
    label: str = "Select Fund",
) -> Optional[str]:
    """
    Single-fund dropdown that remembers the selection across pages, so
    navigating from Fund Analytics to Predictive Analytics keeps your fund.
    """
    if not fund_names:
        return None
    previous = st.session_state.get("selected_fund", fund_names[0])
    index    = fund_names.index(previous) if previous in fund_names else 0

    selected = st.selectbox(label, fund_names, index=index,
                            help="Type to filter the list.")
    st.session_state["selected_fund"] = selected
    return selected


def rf_control() -> Tuple[float, float]:
    """
    Risk-free rate slider with fine −/+ nudge buttons.

    Returns:
        (rf_pct, rf_rate) — e.g. (7.0, 0.07)

    The slider is bound to RF_KEY rather than given a `value=`. That is the
    part that matters: a keyless slider only honours `value` on its first
    render, so once the user had dragged it, the −/+ buttons wrote to session
    state and were then immediately overwritten by the widget's own stale
    state. Binding to the key means the buttons mutate the widget's value
    directly and the rerun picks it up.
    """
    default_pct = round(DEFAULT_RISK_FREE_RATE * 100, 1)
    page = _page_id()
    key  = rf_key_for(page)

    # Seed this page's slider from the store when the user ARRIVES here.
    # Doing it on every run would undo a drag: Streamlit writes the dragged
    # value into the widget key and reruns, and an unconditional re-seed would
    # immediately overwrite it with the pre-drag number.
    arriving = st.session_state.get(RF_LAST_PAGE) != page
    if arriving or key not in st.session_state:
        stored = float(st.session_state.get(RF_STORE, default_pct))
        st.session_state[key] = round(min(RF_MAX, max(RF_MIN, stored)), 1)
    st.session_state[RF_LAST_PAGE] = page

    col_slider, col_down, col_up = st.columns([4, 1, 1])

    col_slider.slider(
        "Risk-Free Rate %",
        min_value=RF_MIN, max_value=RF_MAX, step=RF_STEP,
        key=key,
        help="Used for Sharpe, Sortino and every alpha calculation. "
             "Roughly the 10-year government bond yield.",
    )

    # The nudge buttons MUST mutate the slider's value from inside an on_click
    # callback, not from the button's return value.
    #
    # Streamlit forbids assigning to a widget's key once that widget has been
    # instantiated in the current run — and the buttons render after the
    # slider, so doing the update inline raises:
    #     StreamlitAPIException: st.session_state.rf_rate cannot be modified
    #     after the widget with key rf_rate is instantiated
    #
    # Callbacks run BEFORE the script re-executes, which is the one point
    # where writing to a widget key is allowed. It also removes the need for
    # an explicit st.rerun(): Streamlit reruns automatically after a callback.
    col_down.button("−", key=f"rf_down_{page}", help="Decrease by 0.1%",
                    on_click=_nudge_rf, args=(-RF_STEP, key))
    col_up.button("+", key=f"rf_up_{page}", help="Increase by 0.1%",
                  on_click=_nudge_rf, args=(RF_STEP, key))

    # Whatever the user just did to the slider becomes the new value of record,
    # which is what the next page will be seeded from.
    rf_pct = float(st.session_state[key])
    st.session_state[RF_STORE] = rf_pct
    return rf_pct, rf_pct / 100.0


def rf_key_for(page_id: str) -> str:
    """The RF slider's session-state key on a given page."""
    return f"{RF_KEY}_{page_id}"


def _nudge_rf(delta: float, key: str) -> None:
    """
    Shift the risk-free rate by `delta`, clamped to the slider's range.

    Writes BOTH this page's widget key and the durable store. The callback runs
    before the script re-executes, which is the one point at which writing to a
    widget key is allowed; updating the store as well is what carries the change
    to the next page the user visits.
    """
    current = float(st.session_state.get(
        key, st.session_state.get(RF_STORE, DEFAULT_RISK_FREE_RATE * 100)))
    new = round(min(RF_MAX, max(RF_MIN, current + delta)), 1)
    st.session_state[key]      = new
    st.session_state[RF_STORE] = new


# Indices shown in the staleness panel. Keep to the benchmarks users actually
# see; the full TRI set is on the Data Quality page.
_TRI_PANEL = [
    ("NIFTY 500",          "Nifty 500"),
    ("NIFTY 100",          "Nifty 100"),
    ("NIFTY MIDCAP 150",   "Midcap 150"),
    ("NIFTY SMALLCAP 250", "Smallcap 250"),
    ("NIFTY 50",           "Nifty 50"),
]


def tri_status() -> None:
    """
    Benchmark data freshness panel.

    Shows the last available date per index, warns when a series is stale,
    and says plainly when a category has fallen back to an index-fund proxy
    rather than true TRI — previously an unexplained 'proxy' label.
    """
    from data.tri_loader import (
        get_tri_nav, get_tri_staleness_warning, is_tri_available,
    )

    st.divider()
    st.markdown("**Benchmark Data**")

    for index_name, label in _TRI_PANEL:
        if not is_tri_available(index_name):
            st.caption(
                f"{label}: index-fund proxy",
                help="True TRI data is unavailable for this index, so an "
                     "index-fund NAV is standing in. Slightly less accurate — "
                     "the fund's own expense ratio is baked in.",
            )
            continue

        nav = get_tri_nav(index_name)
        last_date = nav.index[-1].strftime("%d %b %Y") if nav is not None else "?"

        if get_tri_staleness_warning(index_name):
            st.warning(f"{label}: {last_date}")
        else:
            st.caption(f"✓ {label}: {last_date}")


def sidebar_footer() -> None:
    """Data provenance and version, at the foot of the sidebar."""
    st.divider()
    st.caption(
        f"NAV from AMFI via mfapi.in · updated daily after 8 PM IST  \n"
        f"Benchmarks: NSE Total Return Indices  \n"
        f"v{APP_VERSION}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────

def kpi(
    col,
    label: str,
    value,
    kind: str = "ratio",
    help: Optional[str] = None,
    metric_key: Optional[str] = None,
    delta=None,
    delta_color: str = "normal",
) -> None:
    """
    One KPI card, formatted and NaN-guarded.

    Replaces four near-identical local helpers on Fund Analytics alone
    (_kpi/_akpi/_mkpi/_fkpi, differing only in a default flag) plus raw
    col.metric() calls elsewhere that had no NaN guard and rendered "nan%".

    Args:
        col:        Streamlit column (or st itself)
        label:      Display label
        value:      Numeric value, or None/NaN
        kind:       "pct" | "ratio" | "days" | "num" | "raw"
        help:       Tooltip. If omitted and metric_key is given, the tooltip
                    is looked up in METRIC_HELP.
        metric_key: Engine metric key, used for the METRIC_HELP lookup.
        delta:      Optional st.metric delta.
        delta_color: "normal" tints the delta green/red by sign. Pass "off"
                    when the delta is context rather than a change — a
                    confidence interval printed under a Sharpe ratio is not
                    good news or bad news, and colouring it green would say
                    it was.
    """
    tooltip = help or (METRIC_HELP.get(metric_key) if metric_key else None)

    if value is None or (isinstance(value, float) and np.isnan(value)):
        col.metric(label, "N/A", help=tooltip)
        return

    if kind == "pct":
        text = fmt_pct(value)
    elif kind == "days":
        text = fmt_days(value)
    elif kind == "num":
        text = f"{value:.2f}%"
    elif kind == "count":
        # A count is not a percentage. sharpe_n_obs went in as kind="num" and
        # rendered 5305 observations as "5305.00%".
        text = f"{int(value):,}"
    elif kind == "raw":
        text = str(value)
    else:
        text = fmt_ratio(value)

    col.metric(label, text, delta=delta, delta_color=delta_color, help=tooltip)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL
# ─────────────────────────────────────────────────────────────────────────────

def _page_id() -> str:
    """
    Stable per-page suffix for widget keys.

    Streamlit requires unique keys within a page, and each page renders its
    own sidebar, so the RF buttons need a page-scoped key. Derived from the
    running script's path.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx is not None and getattr(ctx, "main_script_path", None):
            return str(ctx.page_script_hash or ctx.main_script_path)
    except Exception:
        pass
    return "default"


# ─────────────────────────────────────────────────────────────────────────────
# FUND SLOT PICKER
# ─────────────────────────────────────────────────────────────────────────────

# Slot identity colours. Previously a Material set defined here; now the shared
# palette, so a fund's colour in the picker matches its line on every chart.
SLOT_COLORS = T.SLOTS


def fund_slot_row(
    slot_idx:  int,
    all_cat:   dict,
    prefix:    str,
    color:     str = T.ACCENT,
    with_weight: bool = False,
    ratios:    tuple = (0.4, 1.9, 3.85, 1.0),
) -> Optional[dict]:
    """
    One category → fund (→ weight) selection row.

    Portfolio Analytics and Factor Attribution each had their own hand-built
    version of this, with different column ratios (so the dropdowns were
    visibly different widths), different placeholder labels ("c"/"f" vs
    "cat"/"fund"), and neither participating in the `selected_fund` session
    convention — so picking a fund on Fund Analytics and navigating here
    started you from empty slots.

    Returns {name, code, category, weight} or None when the slot is empty.
    """
    if with_weight:
        c_label, c_cat, c_fund, c_weight = st.columns(list(ratios))
    else:
        c_label, c_cat, c_fund = st.columns(list(ratios)[:3])
        c_weight = None

    c_label.markdown(
        f"<div style='padding-top:8px;font-weight:700;color:{color}'>"
        f"{slot_idx}</div>",
        unsafe_allow_html=True,
    )

    cat_key = f"{prefix}_cat_{slot_idx}"
    # Seed slot 1 from whatever fund the user was last looking at, so moving
    # between pages carries the selection rather than resetting it.
    if cat_key not in st.session_state and slot_idx == 1:
        st.session_state[cat_key] = st.session_state.get(
            "selected_category", "—")

    cat = c_cat.selectbox(
        f"Category for slot {slot_idx}", ["—"] + CATEGORIES,
        key=cat_key, label_visibility="collapsed",
    )

    if cat == "—":
        c_fund.selectbox(
            f"Fund for slot {slot_idx}", ["—"],
            key=f"{prefix}_fund_{slot_idx}",
            disabled=True, label_visibility="collapsed",
        )
        if c_weight is not None:
            c_weight.number_input(
                f"Weight for slot {slot_idx}", 0.0, 100.0, 0.0, step=0.5,
                key=f"{prefix}_w_{slot_idx}",
                disabled=True, label_visibility="collapsed",
            )
        return None

    fund_list = all_cat.get(cat, [])
    fund_opts = [f["name"] for f in fund_list]
    fund_map  = {f["name"]: f["code"] for f in fund_list}

    fund_key_name = f"{prefix}_fund_{slot_idx}"
    if fund_key_name not in st.session_state and slot_idx == 1:
        remembered = st.session_state.get("selected_fund")
        if remembered in fund_opts:
            st.session_state[fund_key_name] = remembered

    fund_sel = c_fund.selectbox(
        f"Fund for slot {slot_idx}", ["—"] + fund_opts,
        key=fund_key_name, label_visibility="collapsed",
        help="Type to filter.",
    )

    weight = 0.0
    if c_weight is not None:
        weight = c_weight.number_input(
            f"Weight for slot {slot_idx}", 0.0, 100.0, 0.0, step=0.5,
            key=f"{prefix}_w_{slot_idx}",
            disabled=(fund_sel == "—"), label_visibility="collapsed",
        )

    if fund_sel == "—":
        return None
    return {"name": fund_sel, "code": fund_map[fund_sel],
            "category": cat, "weight": weight}


def duplicate_funds(slots: list) -> list:
    """
    Names appearing in more than one slot.

    Portfolio Analytics checked for this; Factor Attribution did not, so you
    could pick the same fund three times and get three identical columns of
    betas presented as a comparison.
    """
    seen, dupes = set(), []
    for s in slots:
        if not s:
            continue
        if s["name"] in seen and s["name"] not in dupes:
            dupes.append(s["name"])
        seen.add(s["name"])
    return dupes


def stale_result_notice(changed: bool, what: str = "selection") -> None:
    """
    Explain why on-screen results no longer match the controls.

    Portfolio Analytics and Factor Attribution both used to DELETE the cached
    result the instant any control changed, then st.stop() with no message —
    blanking the page and discarding up to 80 seconds of NAV loading because
    the user nudged a dropdown. Keeping the result and saying it is stale is
    strictly better than silently throwing it away.
    """
    if changed:
        st.info(
            f"The {what} has changed since these results were computed — "
            "the figures below are from the previous run. Click **Run** to "
            "refresh them.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

# Plotly's default toolbar carries the Plotly logo, a "Produced with Plotly"
# link, and a row of buttons that do not apply to any chart here — lasso and
# box select do nothing useful on a NAV line, and autoscale duplicates the
# reset. Passing this on every chart is what keeps the toolbars identical
# across all 35 charts in the app instead of depending on which call site
# remembered to configure it.
CHART_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "lasso2d", "select2d", "autoScale2d",
        "hoverClosestCartesian", "hoverCompareCartesian",
    ],
    "toImageButtonOptions": {"format": "png", "scale": 2},
    # "hover", not True. Forcing the toolbar always-visible parked it in the
    # top-right corner directly on top of the chart titles, which got longer
    # and wider when the app moved to a serif face. On hover it is there when
    # wanted and out of the way otherwise.
    "displayModeBar": "hover",
    "scrollZoom": False,
}


def chart(fig, container=None, **kwargs) -> None:
    """
    Render a Plotly figure with the platform's standard chart chrome.

    Use this instead of st.plotly_chart() everywhere. It exists so the
    toolbar configuration lives in one place: before it, all 35 chart call
    sites used bare st.plotly_chart() with no config at all, which meant the
    Plotly logo and a set of inapplicable toolbar buttons appeared on every
    chart in the app.

    Args:
        fig:       A plotly Figure.
        container: Streamlit container/column to draw into. Defaults to st.
        **kwargs:  Passed through to st.plotly_chart (e.g. key=).
    """
    if fig is None:
        return
    target = container if container is not None else st
    # width="stretch" is the current spelling; use_container_width is
    # deprecated in Streamlit 1.62 and slated for removal.
    kwargs.setdefault("width", "stretch")
    target.plotly_chart(fig, config=CHART_CONFIG, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# KPI ROWS
# ─────────────────────────────────────────────────────────────────────────────

def kpi_row(specs: List[dict], per_row: int = 5) -> None:
    """
    Render KPI cards in rows of equal width.

    Streamlit sizes st.columns(n) to fill the container, so a block written
    as columns(5) followed by columns(4) renders a row of five narrow cards
    above a row of four wide ones, and the card edges do not line up. The
    factor block on Fund Analytics did exactly that with its nine loadings.

    This chunks the specs into rows of `per_row` and pads the final row with
    empty columns, so every card in the block is the same width and the grid
    aligns vertically.

    Each spec is a dict of kpi() keyword arguments plus "label" and "value":

        kpi_row([
            {"label": "6F Alpha (Ann.)", "value": m.get("alpha_6f"), "kind": "pct"},
            {"label": "Market Beta",     "value": m.get("beta_market_6f")},
        ], per_row=5)
    """
    if not specs:
        return
    for start in range(0, len(specs), per_row):
        chunk = specs[start:start + per_row]
        cols  = st.columns(per_row)
        for col, spec in zip(cols, chunk):
            args = dict(spec)
            kpi(col, args.pop("label"), args.pop("value"), **args)
        # Remaining columns in a short final row stay empty on purpose —
        # that is what keeps the card widths equal to the rows above.


# ─────────────────────────────────────────────────────────────────────────────
# CARDS
# ─────────────────────────────────────────────────────────────────────────────

# One card style for the whole app. app.py had two inline variants and the
# Factor Attribution page three more, each with its own radius, padding and
# border alpha, so panels that were meant to read as the same kind of object
# looked subtly different depending on which page you were on.
CARD_BG     = T.PANEL
CARD_BORDER = T.RULE
CARD_RADIUS = "2px"        # square: a rounded data panel reads as a consumer card
CARD_PAD    = "12px 14px"

ACCENT      = T.ACCENT
MUTED       = T.INK_DIM

# Tones carry meaning, so they are kept rather than flattened into the neutral
# card: the regime legend on Factor Attribution reads bull/sideways/bear by
# colour. What they share with every other card is geometry.
CARD_TONES = {
    "neutral": (T.PANEL,                    T.RULE),
    "accent":  ("rgba(0,194,209,0.06)",     "rgba(0,194,209,0.28)"),
    "up":      ("rgba(63,182,139,0.10)",    "rgba(63,182,139,0.30)"),
    "flat":    ("rgba(232,163,61,0.10)",    "rgba(232,163,61,0.30)"),
    "down":    ("rgba(229,72,77,0.10)",     "rgba(229,72,77,0.30)"),
}


def card(
    body: str,
    container=None,
    tone: str = "neutral",
    min_height: Optional[int] = None,
    center: bool = False,
) -> None:
    """
    Render one panel in the shared card style.

    Args:
        body:       Inner HTML. Build it with card_stat()/card_title() rather
                    than hand-writing colours, so the type scale stays shared.
        container:  Streamlit container/column. Defaults to st.
        tone:       One of CARD_TONES. "accent" for counts and highlights;
                    up/flat/down for the regime legend's semantic colours.
        min_height: Fixed minimum height, for cards laid out in a grid where
                    ragged bottoms would otherwise show.
        center:     Centre the text (used by the compact legend chips).
    """
    target = container if container is not None else st
    bg, border = CARD_TONES.get(tone, CARD_TONES["neutral"])
    mh = f"min-height:{min_height}px;" if min_height else ""
    ta = "text-align:center;" if center else ""
    target.markdown(
        f'<div style="background:{bg};border:1px solid {border};'
        f'border-radius:{CARD_RADIUS};padding:{CARD_PAD};'
        f'margin-bottom:10px;{mh}{ta}">{body}</div>',
        unsafe_allow_html=True,
    )


def card_title(text: str, icon: str = "") -> str:
    """
    Heading line for use inside card().

    The `icon` parameter is retained so the signature does not break, but it is
    ignored: section headings are now carried by case, size and weight rather
    than by a picture in front of the words.
    """
    return (
        f'<div style="font-family:{T.FONT_MONO};font-size:0.66rem;'
        f'letter-spacing:0.13em;text-transform:uppercase;color:{T.INK_DIM};'
        f'margin-bottom:7px">{text}</div>'
    )


def card_stat(value: str, caption: str = "", color: str = None) -> str:
    """
    A figure with its unit, for use inside card().

    The unit sits on the SAME baseline as the number rather than under it.
    Stacked, a count card read as three separate lines ("Large Cap" / "6" /
    "funds") and stood twice as tall as it needed to.
    """
    color = color or T.INK
    cap = (f'<span style="font-family:{T.FONT_MONO};font-size:0.62rem;'
           f'letter-spacing:0.1em;text-transform:uppercase;color:{T.INK_FAINT}">'
           f'{caption}</span>') if caption else ""
    return (
        f'<div style="display:flex;align-items:baseline;gap:6px">'
        f'<span style="font-family:{T.FONT_MONO};font-variant-numeric:tabular-nums;'
        f'font-size:1.4rem;font-weight:500;color:{color};letter-spacing:-0.01em">'
        f'{value}</span>{cap}</div>'
    )


def card_body(text: str) -> str:
    """Descriptive paragraph, for use inside card()."""
    return (
        f'<div style="font-size:0.80rem;color:{T.INK_DIM};line-height:1.5">'
        f'{text}</div>'
    )


def swatch(label: str, color: str, weight: int = 600) -> str:
    """
    A named series with its identity colour, as inline HTML.

    Replaces the coloured-circle emoji that distinguished Portfolio A from
    Portfolio B. The point of those was never decoration — they tied a heading
    to a line on the chart below it — so the colour is kept and only the
    pictograph is dropped. Drawing it from SLOT_COLORS means the heading and
    the chart line cannot drift apart.
    """
    return (
        f'<span style="color:{color};font-size:0.78em;vertical-align:0.08em">■</span>'
        f'<span style="font-weight:{weight};margin-left:7px">{label}</span>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_button(
    df,
    filename: str,
    label: str = f"{T.MARK_DL} CSV",
    container=None,
    key: Optional[str] = None,
) -> None:
    """
    Standard CSV export control.

    Every table the app computes should be exportable — a user who has waited
    for a factor regression should not have to retype the numbers to use them.
    Factor Attribution and Predictive Analytics had no export at all before
    this; Rankings and Data Quality each rolled their own button with a
    different label.

    Silently does nothing for an empty or missing frame, so call sites do not
    each need their own guard.
    """
    if df is None or getattr(df, "empty", True):
        return
    target = container if container is not None else st

    # Keep the index only when it carries information. Several tables here are
    # built with .set_index("Factor") or .set_index("Regime"), and exporting
    # those with index=False would hand the user a column of numbers with no
    # row labels. A default RangeIndex is just 0..n and is dropped.
    index = getattr(df, "index", None)
    keep_index = bool(getattr(index, "name", None)) or (
        index is not None and not isinstance(index, pd.RangeIndex)
    )

    target.download_button(
        label,
        data      = df.to_csv(index=keep_index).encode("utf-8"),
        file_name = filename,
        mime      = "text/csv",
        key       = key or f"dl_{filename}_{_page_id()}",
        help      = "Export this table as a CSV file.",
    )
