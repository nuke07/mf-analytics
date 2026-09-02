"""
app.py
======
MF Quantitative Analytics Platform — Home Page

Entry point. Run with:  streamlit run app.py

This IS the dashboard. The pages in pages/ are added to the sidebar
by Streamlit's multi-page app system.

Phase D change: 1_Dashboard.py and 2_Category_Explorer.py removed.
  - Dashboard content now lives here (it already did — duplication resolved)
  - Category Explorer replaced by Quartile View tab inside Rankings page
"""

import streamlit as st
import plotly.graph_objects as go

from data.fund_loader     import get_all_schemes
from data.category_mapper import get_category_fund_counts
from utils.constants      import (
    APP_TITLE, APP_ICON, APP_SUBTITLE, CATEGORIES,
)
from utils.ui          import (
    sidebar_header, plan_selector, rf_control, tri_status,
    sidebar_footer, render_refresh_button,
    chart,
    card, card_title, card_stat, card_body,
)
from utils import theme as T

st.set_page_config(
    page_title            = f"{APP_TITLE}",
    page_icon             = APP_ICON,
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # The subtitle belongs to the hero, not the sidebar; showing it in both
    # put the same sentence on screen twice.
    sidebar_header()

    rf_pct, rf_rate = rf_control()

    plan_type = plan_selector()

    st.divider()
    render_refresh_button()

    tri_status()

    sidebar_footer()

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title(APP_TITLE)
st.markdown(
    f"<p style='font-size:1.05em; color:{T.INK_DIM};'>{APP_SUBTITLE}</p>",
    unsafe_allow_html=True,
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD SCHEME DATA
# ─────────────────────────────────────────────────────────────────────────────
plan_type = st.session_state.get("plan_type", "Direct")

with st.spinner("Loading scheme registry…"):
    all_schemes = get_all_schemes()

if not all_schemes:
    st.error(
        "Unable to load mutual fund data.\n\n"
        "**Steps to fix:**\n"
        "1. Open Anaconda Prompt\n"
        "2. Run `python debug_connection.py`\n"
        "3. Follow the instructions in the output"
    )
    st.stop()

counts       = get_category_fund_counts(all_schemes)
total_growth = sum(counts.values())

# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total AMFI Schemes",   f"{len(all_schemes):,}")
k2.metric("Growth Funds Tracked", f"{total_growth:,}")
k3.metric("Categories Supported", f"{len(CATEGORIES)}")
k4.metric("Active Risk-Free Rate",f"{rf_pct:.1f}%")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY CARDS + BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")

# The per-category pictographs are gone: a building emoji beside "Large Cap"
# told the reader nothing the words did not already say, and twelve of them
# in a grid read as a consumer app. The card carries the name and the count.

with left:
    st.subheader("Fund Counts by Category")
    st.caption(f"**{plan_type} plans** · Growth only · ETFs, FoFs, Dividend/IDCW excluded.")
    card_cols = st.columns(3)
    for i, cat in enumerate(CATEGORIES):
        n    = counts.get(cat, 0)
        card(
            card_title(cat) + card_stat(str(n), "funds"),
            container = card_cols[i % 3],
            tone      = "accent",
            min_height= 74,
        )

with right:
    st.subheader("Distribution")
    sorted_cats = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    cats, ns    = zip(*sorted_cats) if sorted_cats else ([], [])
    max_n       = max(ns) if ns else 1

    fig = go.Figure(go.Bar(
        x=ns, y=cats, orientation="h",
        marker=dict(
            color=[T.rgba(T.DATA_PRIMARY, round(0.35 + 0.65 * (v / max_n), 2)) for v in ns],
            line=dict(color=T.rgba(T.DATA_PRIMARY, 0.7), width=1),
        ),
        text=[str(v) for v in ns], textposition="outside",
        hovertemplate="%{y}: %{x} funds<extra></extra>",
    ))
    fig.update_layout(
        height=420, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=T.GROUND,
        font=dict(color=T.INK, family=T.PLOTLY_MONO, size=11),
        margin=dict(l=130, r=55, t=20, b=30),
        xaxis=dict(gridcolor=T.rgba(T.INK, 0.07), title="Number of Funds"),
        yaxis=dict(autorange="reversed"), showlegend=False,
    )
    chart(fig)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION GUIDE
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Navigation Guide")

# All seven pages, grouped the way the sidebar now orders them. This list
# used to name only four, so Portfolio Analytics, Predictive Analytics and
# Factor Attribution were undiscoverable from the home page.
nav_items = [
    ("1 · Fund Analytics",
     "Deep dive on a single fund — every metric, charts, alpha against both "
     "its category benchmark and the broad market, and the 6-factor model."),
    ("2 · Fund Comparison",
     "Two to five funds side by side: trailing returns rebased to 0%, "
     "drawdowns, rolling-return distributions, and a metrics table."),
    ("3 · Rankings",
     "Category-wide league tables across six groups — returns, risk, "
     "rolling & hit rate, alpha & factors, vs broad market, and quartiles. "
     "CSV export on every table."),
    ("4 · Portfolio Analytics",
     "Build up to two portfolios of eight funds each and compare them under "
     "identical rebalancing and period settings."),
    ("5 · Factor Attribution",
     "Where a fund's return actually came from: market, size, value, "
     "momentum, quality and low-volatility exposure — and what is left as "
     "alpha, including how the tilts shift across bull and bear regimes."),
    ("6 · Predictive Analytics",
     "Risk forecasting, not return prediction. GARCH volatility, "
     "block-bootstrap Monte Carlo, and drawdown-at-risk."),
    ("7 · Data Quality",
     "NAV history length, gaps, and per-metric coverage — worth checking "
     "before trusting a category's rankings."),
]

_nav_cols = st.columns(4)
for _i, (title, desc) in enumerate(nav_items):
    col = _nav_cols[_i % 4]
    card(
        card_title(title) + card_body(desc),
        container  = col,
        min_height = 168,
    )

st.divider()
st.info(
    "**This platform provides institutional-style quantitative analytics only.**  "
    "It does not provide investment recommendations, ratings, or advice.  "
    "All rankings and metrics are computed within a single category — "
    "cross-category comparisons are not supported by design.",
)
