"""
pages/4_Fund_Comparison.py
===========================
Fund Comparison — Trailing Returns style (Value Research).

All funds start at 0% at the common start of the selected period.
Period selector: 1M / 3M / 6M / 1Y / 3Y / 5Y / All
"""

import streamlit as st
import pandas as pd
import numpy as np

from data.fund_loader      import get_all_categorized_schemes, get_nav_history
from analytics.engine      import compute_fund_metrics
from visualizations.nav_chart import plot_trailing_returns
from visualizations        import plot_drawdown, plot_rolling_timeseries, plot_rolling_distribution
from utils.formatters      import fmt_pct, fmt_ratio, fmt_days
from utils.session         import fund_key as _fund_key
from utils.ui          import (
    sidebar_header, category_selector, plan_selector, rf_control,
    tri_status, render_refresh_button,
    chart,
)

st.set_page_config(page_title="Fund Comparison — MF Analytics", page_icon="⚖️", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    sidebar_header()

    category = category_selector()

    # The plan radio must be rendered BEFORE the fund universe is loaded.
    # It used to sit below, so the loader read the previous run's plan_type:
    # switching Direct→Regular showed "Regular" in the header while every
    # fund listed and every NAV loaded was still Direct, until the user
    # happened to interact with something else.
    plan_type = plan_selector()

    with st.spinner("Loading fund list…"):
        all_cat   = get_all_categorized_schemes(plan_type=plan_type)
        fund_list = all_cat.get(category, [])

    if not fund_list:
        st.warning("No funds found."); st.stop()

    fund_names = [f["name"] for f in fund_list]
    fund_codes = {f["name"]: f["code"] for f in fund_list}

    selected_funds = st.multiselect(
        "Select Funds (2–5)", fund_names,
        default=fund_names[:2], max_selections=5,
    )

    st.divider()
    rf_pct, rf_rate = rf_control()

    render_refresh_button()

    tri_status()

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Fund Comparison")

if len(selected_funds) < 2:
    st.info("Select at least **2 funds** from the sidebar to begin."); st.stop()

st.subheader(f"Comparing {len(selected_funds)} {plan_type} funds — {category}")
st.caption(
    "Each fund starts at **0%** at the common start of the selected period. "
    "Differences show true relative performance."
)
st.divider()

# ── Load NAV + Compute metrics ────────────────────────────────────────────────
nav_dict:    dict = {}
from data.benchmark_loader import (
    get_benchmark_nav, get_benchmark_info, get_market_nav, MARKET_DISPLAY_NAME,
)

bm_info    = get_benchmark_info(category)
bm_nav_df  = None
mkt_nav_df = get_market_nav()   # second yardstick: the broad market
if bm_info["available"]:
    bm_nav_df = get_benchmark_nav(category)
    if bm_nav_df is not None:
        from data.nav_processor import process_nav
        bm_nav = process_nav(bm_nav_df)
        if bm_nav is not None:
            nav_dict[bm_info["display_name"]] = bm_nav
dd_dict:     dict = {}
roll1y_dict: dict = {}
roll3y_dict: dict = {}
all_metrics: dict = {}

bar = st.progress(0, text="Loading fund data…")
for i, name in enumerate(selected_funds):
    bar.progress((i + 1) / len(selected_funds), text=f"Loading: {name[:55]}…")
    code = fund_codes[name]
    ck   = _fund_key(code, rf_pct, category)

    if ck not in st.session_state:
        nav_df  = get_nav_history(code)
        # Pass the benchmark through. It was already loaded above for the
        # chart overlay but never handed to the engine, so every
        # benchmark-relative row in the comparison table below (Jensen's
        # Alpha, Capture Ratio) was permanently "N/A".
        metrics = compute_fund_metrics(
            nav_df,
            rf_rate          = rf_rate,
            fund_name        = name,
            benchmark_nav_df = bm_nav_df if bm_info.get("available") else None,
            benchmark_name   = bm_info.get("display_name"),
            market_nav_df    = mkt_nav_df,
            market_name      = MARKET_DISPLAY_NAME,
        )
        st.session_state[ck] = metrics
    else:
        metrics = st.session_state[ck]

    all_metrics[name] = metrics
    if metrics.get("is_valid"):
        nav_dict[name]    = metrics.get("nav")
        dd_dict[name]     = metrics.get("drawdown_series")
        roll1y_dict[name] = metrics.get("_series_1y")
        roll3y_dict[name] = metrics.get("_series_3y")

bar.empty()

valid_count = sum(1 for m in all_metrics.values() if m.get("is_valid"))
if valid_count == 0:
    st.error("None of the selected funds have valid NAV data."); st.stop()

# ── Period Selector + Trailing Returns Chart ───────────────────────────────────
period = st.radio(
    "Select Period",
    options    = ["1M", "3M", "6M", "1Y", "3Y", "5Y", "All"],
    index      = 3,
    horizontal = True,
    key        = "comparison_period",
)

chart(
    plot_trailing_returns(
        nav_dict,
        period_label = period,
        title = (
            f"Trailing Returns ({period}) — "
            f"{len(selected_funds)} {plan_type} funds, {category}"
        ),
        height = 500,
    ),
)

st.divider()

# ── Drawdown + 1Y Rolling ──────────────────────────────────────────────────────
r1, r2 = st.columns(2, gap="medium")
with r1:
    st.subheader("Drawdown Comparison")
    if dd_dict:
        chart(
            plot_drawdown(dd_dict, title="Drawdown Comparison"),
        )

with r2:
    st.subheader("1-Year Rolling Returns")
    valid_1y = {k: v for k, v in roll1y_dict.items() if v is not None}
    if valid_1y:
        chart(
            plot_rolling_timeseries(valid_1y, "1-Year"),
        )
    else:
        st.caption("Insufficient history for rolling returns.")

# ── Rolling distribution ───────────────────────────────────────────────────────
valid_1y = {k: v for k, v in roll1y_dict.items() if v is not None}
valid_3y = {k: v for k, v in roll3y_dict.items() if v is not None}

if valid_1y or valid_3y:
    rd1, rd2 = st.columns(2, gap="medium")
    with rd1:
        if valid_1y:
            chart(
                plot_rolling_distribution(valid_1y, "1-Year"),
            )
    with rd2:
        if valid_3y:
            chart(
                plot_rolling_distribution(valid_3y, "3-Year"),
            )

st.divider()

# ── Side-by-side Metrics Table ─────────────────────────────────────────────────
st.subheader("Side-by-Side Metrics")
st.caption(f"Risk-free rate: {rf_pct:.1f}%")

COMPARE = [
    ("cagr_1y",               "1Y CAGR",                "pct"),
    ("cagr_3y",               "3Y CAGR",                "pct"),
    ("cagr_5y",               "5Y CAGR",                "pct"),
    ("cagr_inception",        "Since Inception CAGR",   "pct"),
    ("annualized_volatility", "Annualized Volatility",  "pct"),
    ("max_drawdown",          "Max Drawdown",           "pct"),
    ("sharpe",                "Sharpe Ratio",           "ratio"),
    ("sortino",               "Sortino Ratio",          "ratio"),
    ("calmar",                "Calmar Ratio",           "ratio"),
    ("median_rolling_1y",     "Median 1Y Rolling",      "pct"),
    ("worst_rolling_1y",      "Worst 1Y Rolling Return","pct"),
    ("median_rolling_3y",     "Median 3Y Rolling",      "pct"),
    ("win_rate",              "Monthly Win Rate",       "pct"),
    ("capture_ratio",         "Capture Ratio",          "ratio"),
    ("jensens_alpha",         "Jensen's Alpha (vs Category)", "pct"),
    ("jensens_alpha_mkt",     "Jensen's Alpha (vs Market)",   "pct"),
    ("beta_mkt",              "Beta (vs Market)",             "ratio"),
    # active_bet_score removed in Phase D (Active Share proxies were
    # structurally broken for Indian markets) but the row was left behind
    # here, rendering "N/A" for every fund and exporting that way to CSV.
    ("momentum_1m",           "1M Return",              "pct"),
    ("momentum_3m",           "3M Return",              "pct"),
    ("momentum_6m",           "6M Return",              "pct"),
]

def _fmt(val, kind):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    try:
        v = float(val)
        if kind == "pct":   return fmt_pct(v)
        if kind == "ratio": return fmt_ratio(v)
        if kind == "days":  return fmt_days(v)
    except Exception:
        return "N/A"
    return str(val)

rows = []
for key, label, kind in COMPARE:
    row = {"Metric": label}
    for name in selected_funds:
        m = all_metrics.get(name, {})
        row[name[:28]] = _fmt(m.get(key), kind)
    rows.append(row)

cdf = pd.DataFrame(rows).set_index("Metric")
st.dataframe(cdf, width="stretch")

csv = cdf.reset_index().to_csv(index=False).encode("utf-8")
st.download_button(
    "↓Download Comparison (CSV)",
    data=csv,
    file_name=f"{category.replace(' ','_')}_comparison.csv",
    mime="text/csv",
)
