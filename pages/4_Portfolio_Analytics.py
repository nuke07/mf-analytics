"""
pages/7_Portfolio_Analytics.py
================================
Portfolio Analytics

Build up to two weighted multi-fund portfolios, compare them head-to-head,
and analyse both against Nifty 500.

Portfolio A — required (existing behaviour)
Portfolio B — optional; included in comparison when fully configured

Shared controls: rebalancing frequency, analysis period.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

from data.fund_loader        import (
    get_all_categorized_schemes, load_navs_parallel,
)
from data.nav_processor      import process_nav
from data.benchmark_loader   import get_benchmark_nav, get_benchmark_info
from analytics.performance   import calc_all_cagr
from analytics.volatility    import calc_all_volatility
from analytics.risk          import calc_all_risk
from analytics.risk_adjusted import calc_all_risk_adjusted
from analytics.alpha         import calc_all_alpha
from analytics.consistency   import calc_all_consistency
from visualizations._theme          import base_layout, get_color
from visualizations.drawdown_chart  import plot_drawdown
from visualizations.nav_chart       import plot_trailing_returns
from visualizations.rolling_returns import plot_rolling_combined
from utils.constants  import TRADING_DAYS_PER_YEAR, MAR
from utils.formatters import fmt_pct, fmt_ratio
from utils.ui          import (
    sidebar_header, plan_selector, rf_control, tri_status,
    kpi, render_refresh_button, fund_slot_row, stale_result_notice, SLOT_COLORS,
    swatch,
    chart,
)
from utils import theme as T

st.set_page_config(
    page_title = "Portfolio Analytics — MF Analytics",
    page_icon  = "💼",
    layout     = "wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    sidebar_header()

    plan_type = plan_selector()

    st.divider()
    rf_pct, rf_rate = rf_control()

    render_refresh_button()
    tri_status()

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _construct_portfolio(nav_dict, weights_frac, rebalance_freq):
    """
    Build portfolio returns and NAV from individual fund NAVs.
    Returns (port_nav, port_returns, fund_returns_df) or (None, None, None).
    """
    funds = list(weights_frac.keys())
    ret_dict        = {f: nav_dict[f].pct_change().dropna() for f in funds}
    fund_returns_df = pd.DataFrame(ret_dict).dropna()

    if len(fund_returns_df) < 63:
        return None, None, None

    dates          = fund_returns_df.index
    returns_matrix = fund_returns_df[funds].values
    w_target       = np.array([weights_frac[f] for f in funds])
    current_w      = w_target.copy()
    portfolio_rets = np.zeros(len(dates))

    for t in range(len(dates)):
        if t > 0 and rebalance_freq != "static":
            prev, curr = dates[t - 1], dates[t]
            do_rebalance = (
                (rebalance_freq == "monthly"   and curr.month != prev.month) or
                (rebalance_freq == "quarterly" and (curr.month-1)//3 != (prev.month-1)//3) or
                (rebalance_freq == "annual"    and curr.year != prev.year)
            )
            if do_rebalance:
                current_w = w_target.copy()

        portfolio_rets[t] = float(np.dot(current_w, returns_matrix[t]))
        new_values = current_w * (1.0 + returns_matrix[t])
        total      = new_values.sum()
        if total > 0:
            current_w = new_values / total

    port_returns = pd.Series(portfolio_rets, index=dates, name="Portfolio")
    port_nav     = (1.0 + port_returns).cumprod() * 100.0
    return port_nav, port_returns, fund_returns_df


def _apply_period(series, period_label, custom_start=None, custom_end=None):
    if custom_start is not None and custom_end is not None:
        s, e   = pd.Timestamp(custom_start), pd.Timestamp(custom_end)
        sliced = series[(series.index >= s) & (series.index <= e)]
        return sliced if len(sliced) >= 5 else series
    if period_label == "All":
        return series
    months = {"1Y": 12, "3Y": 36, "5Y": 60}.get(period_label, 0)
    if not months:
        return series
    cutoff = series.index[-1] - pd.DateOffset(months=months)
    sliced = series[series.index >= cutoff]
    return sliced if len(sliced) >= 5 else series


def _plot_correlation_heatmap(fund_returns_df, funds, title="Fund Pairwise Return Correlations"):
    corr   = fund_returns_df[funds].corr().round(3)
    labels = [n[:30] + "…" if len(n) > 30 else n for n in funds]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale="RdYlGn", zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        colorbar=dict(title="ρ", tickformat=".2f"),
        hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(base_layout(title=title, height=max(380, 80 * len(funds))))
    fig.update_layout(xaxis=dict(tickangle=-30))
    return fig


def _plot_rolling_volatility(port_ret_a, bm_returns, bm_name, port_ret_b=None, window=63):
    """Rolling annualised volatility — Portfolio A, optional Portfolio B, Benchmark."""
    def _vol_series(ret):
        return (ret.rolling(window).std().dropna() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)

    fig = go.Figure()
    vol_a = _vol_series(port_ret_a)
    fig.add_trace(go.Scatter(
        x=vol_a.index, y=vol_a.values, name="Portfolio A",
        line=dict(color=get_color(0), width=2),
        hovertemplate="%{x|%d %b %Y}: %{y:.2f}%<extra>Portfolio A</extra>",
    ))
    if port_ret_b is not None and len(port_ret_b) > window:
        vol_b = _vol_series(port_ret_b)
        fig.add_trace(go.Scatter(
            x=vol_b.index, y=vol_b.values, name="Portfolio B",
            line=dict(color=get_color(2), width=2, dash="dash"),
            hovertemplate="%{x|%d %b %Y}: %{y:.2f}%<extra>Portfolio B</extra>",
        ))
    if bm_returns is not None and len(bm_returns) > window:
        common = port_ret_a.index.intersection(bm_returns.index)
        vol_bm = _vol_series(bm_returns.reindex(common))
        fig.add_trace(go.Scatter(
            x=vol_bm.index, y=vol_bm.values, name=bm_name,
            line=dict(color=get_color(1), width=1.5, dash="dot"),
            hovertemplate=f"%{{x|%d %b %Y}}: %{{y:.2f}}%<extra>{bm_name}</extra>",
        ))
    fig.update_layout(base_layout(title=f"Rolling {window}-Day Annualised Volatility", height=380))
    fig.update_layout(yaxis=dict(ticksuffix="%"))
    return fig


def _contribution_charts(funds, weights_frac, fund_returns_df, port_ret, label="Portfolio"):
    ret_contribs = []
    for f in funds:
        fnav  = (1 + fund_returns_df[f]).cumprod() * 100
        fcagr = calc_all_cagr(fnav).get("cagr_inception") or 0.0
        ret_contribs.append(weights_frac[f] * fcagr * 100)

    cov_ann      = fund_returns_df[funds].cov() * TRADING_DAYS_PER_YEAR
    w_arr        = np.array([weights_frac[f] for f in funds])
    port_var     = float(w_arr @ cov_ann.values @ w_arr)
    port_vol_ann = np.sqrt(port_var) if port_var > 0 else 1e-9
    marginal     = cov_ann.values @ w_arr
    risk_contribs = (w_arr * marginal / port_vol_ann * 100).tolist()

    colors      = [get_color(i) for i in range(len(funds))]
    short_names = [n[:32] + "…" if len(n) > 32 else n for n in funds]

    def _hbar(x_vals, title, x_title):
        f = go.Figure(go.Bar(
            y=short_names, x=x_vals, orientation="h",
            marker_color=colors,
            text=[f"{v:.2f}%" for v in x_vals], textposition="outside",
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        ))
        f.update_layout(base_layout(title=title, height=max(300, 55 * len(funds))))
        f.update_layout(xaxis=dict(title=x_title, ticksuffix="%"),
                        margin=dict(l=20, r=60, t=50, b=30))
        return f

    return (
        _hbar(ret_contribs,  f"Return Contribution — {label}", "% Return"),
        _hbar(risk_contribs, f"Risk Contribution — {label}",   "% Volatility"),
    )


def _row_metrics(nav_s, ret_s, label, rf):
    def _f(v): return "N/A" if (v is None or (isinstance(v,float) and np.isnan(v))) else fmt_pct(v)
    def _r(v): return "N/A" if (v is None or (isinstance(v,float) and np.isnan(v))) else fmt_ratio(v)
    if len(nav_s) < 30:
        return {"Name": label, "1Y CAGR":"N/A","3Y CAGR":"N/A","Incep. CAGR":"N/A",
                "Ann. Vol":"N/A","Max DD":"N/A","Sharpe":"N/A","Sortino":"N/A","Calmar":"N/A"}
    perf = calc_all_cagr(nav_s)
    risk = calc_all_risk(nav_s)
    vol  = calc_all_volatility(ret_s, mar=MAR)
    radj = calc_all_risk_adjusted(
        returns=ret_s,
        cagr_for_calmar=perf.get("cagr_3y") or perf.get("cagr_inception"),
        max_drawdown=risk.get("max_drawdown"), rf_rate=rf,
    )
    return {
        "Name": label,
        "1Y CAGR":     _f(perf.get("cagr_1y")),
        "3Y CAGR":     _f(perf.get("cagr_3y")),
        "Incep. CAGR": _f(perf.get("cagr_inception")),
        "Ann. Vol":    _f(vol.get("annualized_volatility")),
        "Max DD":      _f(risk.get("max_drawdown")),
        "Sharpe":      _r(radj.get("sharpe")),
        "Sortino":     _r(radj.get("sortino")),
        "Calmar":      _r(radj.get("calmar")),
    }


def _fund_slot_builder(prefix, all_cat, label_color=None):
    label_color = label_color or SLOT_COLORS[0]
    """
    Render 8 fund-selection rows for one portfolio.
    prefix: "pf"  for Portfolio A
            "pfb" for Portfolio B
    Returns list of selected slot dicts.
    """
    slots = []
    RATIOS = (0.4, 1.9, 3.85, 1.0)
    header = st.columns(list(RATIOS))
    for col, txt in zip(header, ["#", "Category", "Fund", "Weight %"]):
        col.markdown(
            f"<div style='color:{T.INK_DIM};font-size:0.78em;padding-bottom:2px'>{txt}</div>",
            unsafe_allow_html=True,
        )
    # Shared row renderer — Factor Attribution uses the same one, so the two
    # pages no longer drift on column widths, placeholder labels, or whether
    # the slot remembers what you picked elsewhere in the app.
    for i in range(1, 9):
        slot = fund_slot_row(
            i, all_cat, prefix=prefix,
            color=SLOT_COLORS[(i - 1) % len(SLOT_COLORS)],
            with_weight=True, ratios=RATIOS,
        )
        if slot:
            slots.append(slot)
    return slots


def _validate_slots(slots, label):
    """Return (weights_ok, total_weight, status message)."""
    names        = [s["name"] for s in slots]
    has_dupes    = len(names) != len(set(names))
    total_weight = sum(s["weight"] for s in slots)
    n            = len(slots)
    ok = (abs(total_weight - 100.0) < 0.01) and n >= 2 and not has_dupes
    return ok, total_weight, has_dupes


def _show_weight_status(slots, label, col_ratio=(1, 3)):
    ok, total, has_dupes = _validate_slots(slots, label)
    wt_col, status_col  = st.columns(col_ratio, gap="medium")
    wt_col.metric(f"{label} Total", f"{total:.1f}%")
    n = len(slots)
    if n == 0:
        status_col.info(f"Select at least 2 funds for {label}.")
    elif has_dupes:
        status_col.error(f"{label}: duplicate fund detected.")
    elif not ok:
        rem = 100.0 - total
        status_col.error(
            f"{label} weights sum to **{total:.1f}%** — need **100%**.  "
            f"({'Add' if rem > 0 else 'Remove'} **{abs(rem):.1f}%**)"
        )
    else:
        status_col.success(f"✓ {label}: {n} funds · 100%")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("Portfolio Analytics")
st.caption(
    "Build up to two portfolios, compare them head-to-head, and analyse "
    "performance vs Nifty 500. Portfolio B is optional — leave it empty for "
    "single-portfolio mode."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — PORTFOLIO BUILDER
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading fund universe…"):
    all_cat = get_all_categorized_schemes(plan_type=plan_type)

# ── Portfolio A ────────────────────────────────────────────────────────────
st.markdown(
    f"<h4 style='color:{SLOT_COLORS[0]};margin-bottom:4px'>Portfolio A — My Portfolio</h4>",
    unsafe_allow_html=True,
)
selected_slots_a = _fund_slot_builder("pf", all_cat)
weights_ok_a     = _show_weight_status(selected_slots_a, "Portfolio A")

st.divider()

# ── Portfolio B ────────────────────────────────────────────────────────────
st.markdown(
    f"<h4 style='color:{SLOT_COLORS[1]};margin-bottom:4px'>Portfolio B — Their Portfolio "
    f"<span style='font-size:0.75em;color:{T.INK_FAINT}'>(optional)</span></h4>",
    unsafe_allow_html=True,
)
selected_slots_b = _fund_slot_builder("pfb", all_cat)

# Portfolio B is active only if the user has filled any slots
has_b_slots  = len(selected_slots_b) > 0
weights_ok_b = False
if has_b_slots:
    weights_ok_b = _show_weight_status(selected_slots_b, "Portfolio B")
else:
    st.caption("Portfolio B is empty — running in single-portfolio mode.")

st.divider()

# ── Shared Settings ─────────────────────────────────────────────────────────
set_l, set_r = st.columns(2, gap="large")

with set_l:
    st.markdown("**Rebalancing** *(applied to both portfolios)*")
    rebalance_choice = st.radio(
        "Rebalancing frequency",
        ["Static (no rebalancing)", "Monthly", "Quarterly", "Annual"],
        index=2, label_visibility="collapsed",
    )
    rebalance_freq  = rebalance_choice.split(" ")[0].lower()
    rebalance_label = rebalance_choice

with set_r:
    st.markdown("**Analysis Period**")
    period_label = st.radio(
        "Period", ["1Y", "3Y", "5Y", "All"],
        index=3, horizontal=True, label_visibility="collapsed",
    )
    use_custom = st.toggle("Use custom date range")
    custom_start, custom_end = None, None
    if use_custom:
        d1, d2 = st.columns(2)
        custom_start = d1.date_input("From", value=date(2018, 1, 1), key="pf_cstart")
        custom_end   = d2.date_input("To",   value=date.today(),     key="pf_cend")
        if custom_start >= custom_end:
            st.error("'From' date must be before 'To' date.")
            custom_start, custom_end = None, None

st.divider()

# Portfolio B must be either empty OR valid — not partially filled
b_blocking = has_b_slots and not weights_ok_b
run_disabled = not weights_ok_a or b_blocking
if b_blocking:
    st.warning("Complete Portfolio B weights (or clear all slots) before running.")

run_btn = st.button(
    "Run Portfolio Analysis",
    type="primary", width="stretch",
    disabled=run_disabled,
)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

# Cache key covers both portfolios
pf_sig = (
    str(sorted([(s["name"], s["weight"]) for s in selected_slots_a])) +
    str(sorted([(s["name"], s["weight"]) for s in selected_slots_b])) +
    rebalance_freq
)
# Keep results when the builder changes; flag them stale rather than
# discarding an 80-second load because a dropdown moved.
_pf_stale = ("_pf_result" in st.session_state
             and st.session_state.get("_pf_sig") != pf_sig)

def _load_and_build(slots, label):
    """Load NAVs and construct portfolio. Returns result dict or None."""
    raw_navs = {}
    prog = st.progress(0, text=f"Loading {label} NAVs…")

    # Concurrent fetch. These are independent HTTP round-trips, so loading
    # them one at a time made the wait the sum of the latencies rather than
    # the max — 8 funds at 2-5s each was 16-40s of mostly-idle waiting per
    # portfolio, doubled when Portfolio B was in use.
    def _on_done(done, total, code):
        prog.progress(done / total,
                      text=f"{label} · {done} of {total} loaded")

    nav_frames = load_navs_parallel(
        [slot["code"] for slot in slots], progress_cb=_on_done,
    )

    for slot in slots:
        nav = process_nav(nav_frames.get(slot["code"]))
        if nav is not None:
            raw_navs[slot["name"]] = nav

    prog.empty()

    missing = [s["name"] for s in slots if s["name"] not in raw_navs]
    if missing:
        st.warning(f"{label}: NAV unavailable for {', '.join(missing)}")

    valid_slots = [s for s in slots if s["name"] in raw_navs]
    if len(valid_slots) < 2:
        st.error(f"{label}: need at least 2 funds with valid NAV data.")
        return None

    funds_in    = [s["name"] for s in valid_slots]
    weights_pct = {s["name"]: s["weight"] for s in valid_slots}
    w_sum       = sum(weights_pct.values())
    weights_frac= {k: v/w_sum for k, v in weights_pct.items()}

    with st.spinner(f"Constructing {label} NAV…"):
        port_nav, port_returns, fund_returns_df = _construct_portfolio(
            raw_navs, weights_frac, rebalance_freq,
        )

    if port_nav is None:
        st.error(f"{label}: insufficient overlapping NAV history.")
        return None

    return {
        "port_nav":        port_nav,
        "port_returns":    port_returns,
        "fund_returns_df": fund_returns_df,
        "funds":           funds_in,
        "weights_frac":    weights_frac,
        "weights_pct":     weights_pct,
        "raw_navs":        raw_navs,
        "slots":           valid_slots,
    }

if run_btn and weights_ok_a:
    st.session_state["_pf_sig"] = pf_sig

    # ── Build Portfolio A ──────────────────────────────────────────────────
    result_a = _load_and_build(selected_slots_a, "Portfolio A")
    if result_a is None:
        st.stop()

    # ── Build Portfolio B (if provided) ────────────────────────────────────
    result_b = None
    if has_b_slots and weights_ok_b:
        result_b = _load_and_build(selected_slots_b, "Portfolio B")

    # ── Load Nifty 500 benchmark ───────────────────────────────────────────
    with st.spinner("Loading Nifty 500 benchmark…"):
        bm_nav_df  = get_benchmark_nav("Flexi Cap")
        bm_info    = get_benchmark_info("Flexi Cap")
        bm_nav_raw = process_nav(bm_nav_df) if bm_nav_df is not None else None

    bm_nav_aligned = None
    bm_returns_full = None
    if bm_nav_raw is not None:
        # Align to Portfolio A's date range (primary portfolio)
        common_idx = result_a["port_returns"].index.intersection(bm_nav_raw.index)
        if len(common_idx) > 30:
            bm_nav_aligned  = bm_nav_raw.reindex(common_idx)
            bm_returns_full = bm_nav_aligned.pct_change().dropna()

    st.session_state["_pf_result"] = {
        "a":               result_a,
        "b":               result_b,
        "has_b":           result_b is not None,
        "bm_nav":          bm_nav_aligned,
        "bm_returns":      bm_returns_full,
        "bm_name":         bm_info.get("display_name", "Nifty 500 TRI"),
        "rebalance_label": rebalance_label,
    }
    st.success(
        f"Portfolio A ready · "
        f"{'Portfolio B ready · ' if result_b else ''}"
        f"{'Benchmark loaded' if bm_nav_aligned is not None else 'Benchmark unavailable'}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — RESULTS
# ─────────────────────────────────────────────────────────────────────────────

if "_pf_result" not in st.session_state:
    st.info(
        "Build **Portfolio A** above — pick a category and fund for each "
        "holding, set weights totalling 100%, then click "
        "**Run Portfolio Analysis**.\n\n"
        "Portfolio B is optional: fill it in to compare two allocations "
        "side by side under identical rebalancing and period settings.",
    )
    st.stop()

stale_result_notice(_pf_stale, "portfolio or its settings")

res   = st.session_state["_pf_result"]
res_a = res["a"]
res_b = res["b"]
has_b = res["has_b"]

bm_nav      = res["bm_nav"]
bm_returns  = res["bm_returns"]
bm_name     = res["bm_name"]
rebalance_label = res["rebalance_label"]

# ── Period-slice Portfolio A ───────────────────────────────────────────────
port_nav_a      = _apply_period(res_a["port_nav"],     period_label, custom_start, custom_end)
port_ret_a      = _apply_period(res_a["port_returns"], period_label, custom_start, custom_end)
fund_returns_a  = res_a["fund_returns_df"]
funds_a         = res_a["funds"]
weights_frac_a  = res_a["weights_frac"]
weights_pct_a   = res_a["weights_pct"]

# ── Period-slice Portfolio B (if present) ─────────────────────────────────
port_nav_b, port_ret_b, fund_returns_b, funds_b, weights_frac_b, weights_pct_b = \
    (None,) * 6
if has_b:
    port_nav_b     = _apply_period(res_b["port_nav"],     period_label, custom_start, custom_end)
    port_ret_b     = _apply_period(res_b["port_returns"], period_label, custom_start, custom_end)
    fund_returns_b = res_b["fund_returns_df"]
    funds_b        = res_b["funds"]
    weights_frac_b = res_b["weights_frac"]
    weights_pct_b  = res_b["weights_pct"]

# ── Period-slice Benchmark ─────────────────────────────────────────────────
bm_nav_p = _apply_period(bm_nav,    period_label, custom_start, custom_end) if bm_nav    is not None else None
bm_ret_p = _apply_period(bm_returns,period_label, custom_start, custom_end) if bm_returns is not None else None

# ── Effective period display ───────────────────────────────────────────────
eff_start = port_nav_a.index[0].strftime("%d %b %Y")
eff_end   = port_nav_a.index[-1].strftime("%d %b %Y")
n_days    = len(port_nav_a)

st.divider()
st.caption(
    f"Analysis period: **{eff_start} → {eff_end}** "
    f"({n_days} trading days · {rebalance_label})"
    + (" · Comparison mode: A vs B" if has_b else "")
)

# ── Compute metrics — Portfolio A ──────────────────────────────────────────
rf     = rf_rate   # from rf_control() in the sidebar — single source
perf_a = calc_all_cagr(port_nav_a)
risk_a = calc_all_risk(port_nav_a)
vol_a  = calc_all_volatility(port_ret_a, mar=MAR)
radj_a = calc_all_risk_adjusted(
    returns=port_ret_a,
    cagr_for_calmar=perf_a.get("cagr_3y") or perf_a.get("cagr_inception"),
    max_drawdown=risk_a.get("max_drawdown"), rf_rate=rf,
)

# ── Compute metrics — Portfolio B ──────────────────────────────────────────
perf_b = risk_b = vol_b = radj_b = None
if has_b:
    perf_b = calc_all_cagr(port_nav_b)
    risk_b = calc_all_risk(port_nav_b)
    vol_b  = calc_all_volatility(port_ret_b, mar=MAR)
    radj_b = calc_all_risk_adjusted(
        returns=port_ret_b,
        cagr_for_calmar=perf_b.get("cagr_3y") or perf_b.get("cagr_inception"),
        max_drawdown=risk_b.get("max_drawdown"), rf_rate=rf,
    )

# ── Alpha vs benchmark ─────────────────────────────────────────────────────
def _compute_alpha(port_ret_p, bm_ret_p, rf):
    if bm_ret_p is None or len(bm_ret_p) <= 60:
        return None
    common = port_ret_p.index.intersection(bm_ret_p.index)
    if len(common) <= 60:
        return None
    return calc_all_alpha(
        port_ret_p.reindex(common).dropna(),
        bm_ret_p.reindex(common).dropna(),
        rf,
    )

alpha_a = _compute_alpha(port_ret_a, bm_ret_p, rf)
alpha_b = _compute_alpha(port_ret_b, bm_ret_p, rf) if has_b else None

# ─────────────────────────────────────────────────────────────────────────────
# RESULT TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Risk",
    "Fund Breakdown",
    "Full Comparison",
])

# ── TAB 1: OVERVIEW ──────────────────────────────────────────────────────────
with tab1:
    if has_b:
        # ── Side-by-side KPI comparison ───────────────────────────────────
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown(swatch("Portfolio A", SLOT_COLORS[0]), unsafe_allow_html=True)
            k1,k2,k3 = st.columns(3)
            kpi(k1,"Incep. CAGR",perf_a.get("cagr_inception"), kind='pct')
            kpi(k2,"3Y CAGR",    perf_a.get("cagr_3y"), kind='pct')
            kpi(k3,"1Y CAGR",    perf_a.get("cagr_1y"), kind='pct')
            k4,k5,k6 = st.columns(3)
            kpi(k4,"Sharpe",  radj_a.get("sharpe"), kind='ratio')
            kpi(k5,"Sortino", radj_a.get("sortino"), kind='ratio')
            kpi(k6,"Ann. Vol",vol_a.get("annualized_volatility"), kind='pct')
        with col_b:
            st.markdown(swatch("Portfolio B", SLOT_COLORS[1]), unsafe_allow_html=True)
            k1,k2,k3 = st.columns(3)
            kpi(k1,"Incep. CAGR",perf_b.get("cagr_inception"), kind='pct')
            kpi(k2,"3Y CAGR",    perf_b.get("cagr_3y"), kind='pct')
            kpi(k3,"1Y CAGR",    perf_b.get("cagr_1y"), kind='pct')
            k4,k5,k6 = st.columns(3)
            kpi(k4,"Sharpe",  radj_b.get("sharpe"), kind='ratio')
            kpi(k5,"Sortino", radj_b.get("sortino"), kind='ratio')
            kpi(k6,"Ann. Vol",vol_b.get("annualized_volatility"), kind='pct')
    else:
        st.subheader("Portfolio A Performance")
        k1,k2,k3,k4,k5,k6 = st.columns(6)
        kpi(k1,"Incep. CAGR",perf_a.get("cagr_inception"), kind='pct')
        kpi(k2,"3Y CAGR",    perf_a.get("cagr_3y"), kind='pct')
        kpi(k3,"1Y CAGR",    perf_a.get("cagr_1y"), kind='pct')
        kpi(k4,"Sharpe",     radj_a.get("sharpe"), kind='ratio')
        kpi(k5,"Sortino",    radj_a.get("sortino"), kind='ratio')
        kpi(k6,"Ann. Vol",   vol_a.get("annualized_volatility"), kind='pct')

    st.divider()

    # ── Trailing returns chart ─────────────────────────────────────────────
    title_suffix = f"Portfolio A vs Portfolio B vs {bm_name}" if has_b else f"Portfolio A vs {bm_name}"
    st.subheader(f"{title_suffix}")
    chart_period = st.radio(
        "Chart period", ["1M","3M","6M","1Y","3Y","5Y","All"],
        index=3, horizontal=True, key="pf_chart_period",
    )
    nav_chart = {"Portfolio A": res_a["port_nav"]}
    if has_b:
        nav_chart["Portfolio B"] = res_b["port_nav"]
    if bm_nav is not None:
        nav_chart[bm_name] = bm_nav
    chart(
        plot_trailing_returns(nav_chart, period_label=chart_period, height=480),
    )

    # ── Alpha vs benchmark ─────────────────────────────────────────────────
    if alpha_a or alpha_b:
        st.divider()
        st.subheader(f"Alpha vs {bm_name}")
        if has_b:
            col_a, col_b = st.columns(2, gap="large")
            with col_a:
                st.markdown(swatch("Portfolio A", SLOT_COLORS[0]), unsafe_allow_html=True)
                if alpha_a:
                    a1,a2,a3 = st.columns(3)
                    kpi(a1,"Jensen's Alpha",alpha_a.get("jensens_alpha"), kind='pct')
                    kpi(a2,"Beta",          alpha_a.get("beta"), kind='ratio')
                    kpi(a3,"Info Ratio",    alpha_a.get("information_ratio"), kind='ratio')
                    a4,a5,a6 = st.columns(3)
                    kpi(a4,"Tracking Error",alpha_a.get("tracking_error"), kind='pct')
                    kpi(a5,"Up Capture",    alpha_a.get("up_capture"), kind='ratio')
                    kpi(a6,"Down Capture",  alpha_a.get("down_capture"), kind='ratio')
                    t = alpha_a.get("alpha_tstat")
                    if t is not None:
                        if abs(t) >= 2.0: st.success(f"Significant (|t|={t:.2f})")
                        else:             st.info(f"Not significant (|t|={t:.2f})")
            with col_b:
                st.markdown(swatch("Portfolio B", SLOT_COLORS[1]), unsafe_allow_html=True)
                if alpha_b:
                    a1,a2,a3 = st.columns(3)
                    kpi(a1,"Jensen's Alpha",alpha_b.get("jensens_alpha"), kind='pct')
                    kpi(a2,"Beta",          alpha_b.get("beta"), kind='ratio')
                    kpi(a3,"Info Ratio",    alpha_b.get("information_ratio"), kind='ratio')
                    a4,a5,a6 = st.columns(3)
                    kpi(a4,"Tracking Error",alpha_b.get("tracking_error"), kind='pct')
                    kpi(a5,"Up Capture",    alpha_b.get("up_capture"), kind='ratio')
                    kpi(a6,"Down Capture",  alpha_b.get("down_capture"), kind='ratio')
                    t = alpha_b.get("alpha_tstat")
                    if t is not None:
                        if abs(t) >= 2.0: st.success(f"Significant (|t|={t:.2f})")
                        else:             st.info(f"Not significant (|t|={t:.2f})")
        else:
            if alpha_a:
                a1,a2,a3,a4,a5,a6 = st.columns(6)
                kpi(a1,"Jensen's Alpha",alpha_a.get("jensens_alpha"), kind='pct')
                kpi(a2,"Beta",          alpha_a.get("beta"), kind='ratio')
                kpi(a3,"Info Ratio",    alpha_a.get("information_ratio"), kind='ratio')
                kpi(a4,"Tracking Error",alpha_a.get("tracking_error"), kind='pct')
                kpi(a5,"Up Capture",    alpha_a.get("up_capture"), kind='ratio')
                kpi(a6,"Down Capture",  alpha_a.get("down_capture"), kind='ratio')
                t = alpha_a.get("alpha_tstat")
                if t is not None:
                    if abs(t) >= 2.0: st.success(f"Significant (|t|={t:.2f})")
                    else:             st.info(f"Not significant (|t|={t:.2f})")
    elif bm_nav is None:
        st.warning("Nifty 500 benchmark not available — alpha metrics cannot be computed.")

# ── TAB 2: RISK ───────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Risk Analysis")

    if has_b:
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown(swatch("Portfolio A", SLOT_COLORS[0]), unsafe_allow_html=True)
            r1,r2,r3,r4 = st.columns(4)
            kpi(r1,"Max DD",      risk_a.get("max_drawdown"), kind='pct')
            kpi(r2,"Avg DD",      risk_a.get("avg_drawdown"), kind='pct')
            kpi(r3,"Ann. Vol",    vol_a.get("annualized_volatility"), kind='pct')
            kpi(r4,"Sharpe",      radj_a.get("sharpe"), kind='ratio')
        with col_b:
            st.markdown(swatch("Portfolio B", SLOT_COLORS[1]), unsafe_allow_html=True)
            r1,r2,r3,r4 = st.columns(4)
            kpi(r1,"Max DD",      risk_b.get("max_drawdown"), kind='pct')
            kpi(r2,"Avg DD",      risk_b.get("avg_drawdown"), kind='pct')
            kpi(r3,"Ann. Vol",    vol_b.get("annualized_volatility"), kind='pct')
            kpi(r4,"Sharpe",      radj_b.get("sharpe"), kind='ratio')
    else:
        r1,r2,r3,r4 = st.columns(4)
        kpi(r1,"Max Drawdown",  risk_a.get("max_drawdown"), kind='pct')
        kpi(r2,"Avg Drawdown",  risk_a.get("avg_drawdown"), kind='pct')
        kpi(r3,"DD Duration",   risk_a.get("drawdown_duration"), kind='days')
        kpi(r4,"Calmar Ratio",  radj_a.get("calmar"), kind='ratio')
        r5,r6,r7,r8 = st.columns(4)
        kpi(r5,"Ann. Volatility",vol_a.get("annualized_volatility"), kind='pct')
        kpi(r6,"Downside Vol",   vol_a.get("downside_volatility"), kind='pct')
        kpi(r7,"Sortino",        radj_a.get("sortino"), kind='ratio')
        r8.metric("Rebalancing",  rebalance_label.split(" ")[0])

    st.divider()

    # ── Drawdown chart ─────────────────────────────────────────────────────
    dd_dict = {}
    dd_a = risk_a.get("drawdown_series")
    if dd_a is not None:
        dd_dict["Portfolio A"] = _apply_period(dd_a, period_label, custom_start, custom_end)
    if has_b:
        dd_b = risk_b.get("drawdown_series")
        if dd_b is not None:
            dd_dict["Portfolio B"] = _apply_period(dd_b, period_label, custom_start, custom_end)
    if dd_dict:
        chart(plot_drawdown(dd_dict))

    st.divider()

    # ── Rolling volatility ─────────────────────────────────────────────────
    st.subheader("Rolling Annualised Volatility (63-Day Window)")
    chart(
        _plot_rolling_volatility(port_ret_a, bm_ret_p, bm_name,
                                  port_ret_b=port_ret_b if has_b else None),
    )

    st.divider()

    # ── Rolling 1Y returns ─────────────────────────────────────────────────
    st.subheader("1-Year Rolling Returns")
    roll_dict = {}
    cons_a = calc_all_consistency(port_nav_a)
    s1y_a  = cons_a.get("_series_1y")
    if s1y_a is not None:
        roll_dict["Portfolio A"] = s1y_a
    if has_b:
        cons_b = calc_all_consistency(port_nav_b)
        s1y_b  = cons_b.get("_series_1y")
        if s1y_b is not None:
            roll_dict["Portfolio B"] = s1y_b

    if roll_dict:
        chart(
            plot_rolling_combined(roll_dict, window_label="1-Year", height=480),
        )
    else:
        st.info("Rolling 1Y returns require at least 2 years in the selected period.")

# ── TAB 3: FUND BREAKDOWN ─────────────────────────────────────────────────────
with tab3:
    if has_b:
        bd_tab_a, bd_tab_b = st.tabs(["Portfolio A", "Portfolio B"])
    else:
        bd_tab_a = st.container()
        bd_tab_b = None

    def _render_breakdown(container, pf_label, slots, funds, weights_frac,
                          weights_pct, fund_returns_df, port_ret_p):
        with container:
            st.subheader(f"Fund Breakdown — {pf_label}")
            w_rows = [
                {"Fund": s["name"], "Category": s["category"],
                 "Weight": f"{s['weight']:.1f}%"}
                for s in slots
            ]
            st.dataframe(pd.DataFrame(w_rows), width="stretch", hide_index=True)

            st.divider()

            # Correlation heatmap
            st.subheader("Fund Return Correlations")
            frd_idx = fund_returns_df.index.intersection(port_ret_p.index)
            frd_p   = fund_returns_df.reindex(frd_idx).dropna()
            if len(frd_p) > 30 and len(funds) >= 2:
                chart(
                    _plot_correlation_heatmap(frd_p, funds, f"Correlations — {pf_label}"),
                )
            else:
                st.info("Insufficient overlapping data for correlation matrix.")

            st.divider()

            # Contribution charts
            st.subheader("Return & Risk Contribution")
            if len(frd_p) > 30:
                fig_ret, fig_risk = _contribution_charts(
                    funds, weights_frac, frd_p, port_ret_p, label=pf_label,
                )
                cr, ck = st.columns(2, gap="large")
                with cr: chart(fig_ret)
                with ck: chart(fig_risk)
            else:
                st.info("Insufficient data for contribution analysis.")

    _render_breakdown(
        bd_tab_a, "Portfolio A",
        res_a["slots"], funds_a, weights_frac_a, weights_pct_a,
        fund_returns_a, port_ret_a,
    )
    if has_b and bd_tab_b is not None:
        _render_breakdown(
            bd_tab_b, "Portfolio B",
            res_b["slots"], funds_b, weights_frac_b, weights_pct_b,
            fund_returns_b, port_ret_b,
        )

# ── TAB 4: FULL COMPARISON ───────────────────────────────────────────────────
with tab4:
    st.subheader("Full Comparison")
    st.caption(
        f"Period: **{eff_start} → {eff_end}**  ·  Rebalancing: **{rebalance_label}**  ·  "
        "All metrics computed on the same date range for fair comparison."
    )

    ORDERED_COLS = [
        "Weight","1Y CAGR","3Y CAGR","Incep. CAGR",
        "Ann. Vol","Max DD","Sharpe","Sortino","Calmar",
    ]

    rows = []

    # Portfolio A
    row_a         = _row_metrics(port_nav_a, port_ret_a, "Portfolio A", rf)
    row_a["Weight"] = "100%"
    rows.append(row_a)

    # Portfolio B
    if has_b:
        row_b           = _row_metrics(port_nav_b, port_ret_b, "Portfolio B", rf)
        row_b["Weight"] = "100%"
        rows.append(row_b)

    # Benchmark
    if bm_nav_p is not None and bm_ret_p is not None and len(bm_nav_p) > 30:
        bm_row          = _row_metrics(bm_nav_p, bm_ret_p, f"{bm_name}", rf)
        bm_row["Weight"]= "—"
        rows.append(bm_row)

    # Separator + Individual funds — Portfolio A
    frd_idx_a = fund_returns_a.index.intersection(port_ret_a.index)
    for fname in funds_a:
        fret = fund_returns_a[fname].reindex(frd_idx_a).dropna()
        fnav = (1 + fret).cumprod() * 100
        if len(fnav) > 30:
            short    = fname[:46] + "…" if len(fname) > 46 else fname
            fund_row = _row_metrics(fnav, fret, f"  A · {short}", rf)
            fund_row["Weight"] = f"{weights_pct_a[fname]:.1f}%"
            rows.append(fund_row)

    # Individual funds — Portfolio B
    if has_b:
        frd_idx_b = fund_returns_b.index.intersection(port_ret_b.index)
        for fname in funds_b:
            fret = fund_returns_b[fname].reindex(frd_idx_b).dropna()
            fnav = (1 + fret).cumprod() * 100
            if len(fnav) > 30:
                short    = fname[:46] + "…" if len(fname) > 46 else fname
                fund_row = _row_metrics(fnav, fret, f"  B · {short}", rf)
                fund_row["Weight"] = f"{weights_pct_b[fname]:.1f}%"
                rows.append(fund_row)

    if rows:
        compare_df = pd.DataFrame(rows).set_index("Name")[ORDERED_COLS]
        st.dataframe(compare_df, width="stretch")
        csv = compare_df.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(
            "↓Download Comparison (CSV)",
            data=csv,
            file_name="portfolio_comparison.csv",
            mime="text/csv",
            key="dl_pf_compare",
        )
    else:
        st.warning("No data available for comparison table.")
