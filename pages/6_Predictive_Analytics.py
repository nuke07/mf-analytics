"""
pages/8_Predictive_Analytics.py
================================
Predictive Analytics — Risk Forecasting & Scenario Analysis

Applies to a single fund selected in the sidebar.

Three tabs:
  Volatility Forecast  — GARCH(1,1): conditional vol, 30/60/90 day
                            forecast, VaR, CVaR, model parameters
  Monte Carlo          — Block bootstrap: fan chart, probability of
                            shortfall, terminal return distribution
  Drawdown Risk        — Derived from Monte Carlo paths: max drawdown
                            distribution, Drawdown at Risk, exceed probs

Note (Phase E): Market Regimes tab (HMM) removed — hmmlearn has Python
  version compatibility issues that risk deployment stability. Will be
  revisited when Nifty 500 TRI data is available.

IMPORTANT FRAMING:
  All outputs are scenario analysis or statistical risk estimation.
  This page does NOT forecast future returns. Expected returns are
  set to the fund's own historical mean — not a prediction.
  Volatility forecasts from GARCH are empirically meaningful.
  Monte Carlo shows the range of outcomes consistent with history.

Data required: Fund daily NAV (already available in platform).
No external data required for any tab.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data.fund_loader      import get_all_categorized_schemes, get_nav_history
from data.nav_processor    import process_nav, compute_daily_returns
from analytics.garch_model import get_garch_summary
from analytics.monte_carlo import run_monte_carlo
from visualizations._theme import base_layout, get_color
from utils.constants  import TRADING_DAYS_PER_YEAR
from utils.formatters import fmt_pct
from utils.ui          import (
    sidebar_header, category_selector, plan_selector, rf_control,
    tri_status, render_refresh_button,
    chart,
    export_button,
)
from utils import theme as T

st.set_page_config(
    page_title = "Predictive Analytics — MF Analytics",
    page_icon  = "🔮",
    layout     = "wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    sidebar_header()

    category = category_selector()

    plan_type = plan_selector()

    with st.spinner("Loading funds…"):
        all_cat   = get_all_categorized_schemes(plan_type=plan_type)
        fund_list = all_cat.get(category, [])

    if not fund_list:
        st.warning("No funds found."); st.stop()

    fund_names = [f["name"] for f in fund_list]
    fund_codes = {f["name"]: f["code"] for f in fund_list}
    prev       = st.session_state.get("selected_fund", fund_names[0])
    sel_name   = st.selectbox("Fund", fund_names,
                               index=fund_names.index(prev) if prev in fund_names else 0)
    st.session_state["selected_fund"] = sel_name
    sel_code = fund_codes[sel_name]

    st.divider()

    rf_pct, rf_rate = rf_control()

    render_refresh_button()
    tri_status()

# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS  (page-local — these charts are not reused elsewhere)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_garch_vol(garch: dict, fund_name: str) -> go.Figure:
    """GARCH conditional volatility history + forward forecast points."""
    cond_vol   = garch["cond_vol"]
    forecasts  = garch["forecasts"]
    hist_avg   = garch["historical_avg_vol"]
    current_v  = garch["current_ann_vol"]

    # Show last 2 years of history
    display_vol = cond_vol.iloc[-min(504, len(cond_vol)):]
    last_date   = display_vol.index[-1]

    fig = go.Figure()

    # Historical conditional vol
    fig.add_trace(go.Scatter(
        x=display_vol.index, y=display_vol.values * 100,
        mode="lines", name="Conditional Vol (Ann.)",
        line=dict(color=get_color(0), width=1.5),
        hovertemplate="%{x|%d %b %Y}: %{y:.2f}%<extra>Conditional Vol</extra>",
    ))

    # Historical average line
    fig.add_hline(
        y=hist_avg * 100, line_dash="dash",
        line_color="rgba(255,255,255,0.4)",
        annotation_text=f"Hist. Avg {hist_avg*100:.1f}%",
        annotation_position="bottom right",
    )

    # Forecast points at +30, +60, +90 days
    if forecasts:
        forecast_dates  = [last_date + pd.tseries.offsets.BDay(h) for h in sorted(forecasts)]
        forecast_vols   = [forecasts[h] * 100 for h in sorted(forecasts)]
        forecast_labels = [f"+{h}d: {forecasts[h]*100:.1f}%" for h in sorted(forecasts)]

        # Dashed connector from current to first forecast
        fig.add_trace(go.Scatter(
            x=[last_date] + forecast_dates,
            y=[current_v * 100] + forecast_vols,
            mode="lines+markers",
            name="Forecast",
            line=dict(color=get_color(2), width=2, dash="dot"),
            marker=dict(size=10, symbol="diamond", color=get_color(2)),
            text=["Current"] + forecast_labels,
            hovertemplate="%{text}<extra>Forecast</extra>",
        ))

    fig.update_layout(base_layout(
        title=f"GARCH Conditional Volatility — {fund_name[:50]}",
        height=420,
    ))
    fig.update_layout(
        yaxis=dict(title="Annualised Volatility (%)", ticksuffix="%"),
        xaxis=dict(title=""),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def _plot_mc_fan(mc: dict, fund_name: str) -> go.Figure:
    """Monte Carlo fan chart — percentile bands of simulated NAV paths."""
    nav_pcts     = mc["nav_percentiles"]
    horizon_days = mc["horizon_days"]
    initial_nav  = mc["initial_nav"]

    # Future business dates starting from today
    future_dates = pd.bdate_range(
        start=pd.Timestamp.today() + pd.tseries.offsets.BDay(1),
        periods=horizon_days,
    )

    fig = go.Figure()

    # Filled bands (add upper first, then lower with fill='tonexty')
    BANDS = [
        (5,  95, 0.06, "P5–P95"),
        (10, 90, 0.10, "P10–P90"),
        (25, 75, 0.18, "P25–P75 (IQR)"),
    ]
    for lower_p, upper_p, opacity, label in BANDS:
        fig.add_trace(go.Scatter(
            x=future_dates, y=nav_pcts[upper_p],
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=nav_pcts[lower_p],
            mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor=T.rgba(T.DATA_PRIMARY, opacity),
            name=label,
            hoverinfo="skip",
        ))

    # Median line
    fig.add_trace(go.Scatter(
        x=future_dates, y=nav_pcts[50],
        mode="lines", name="Median (P50)",
        line=dict(color=T.DATA_PRIMARY, width=2.5),
        hovertemplate="%{x|%b %Y}: NAV %{y:.1f}<extra>Median</extra>",
    ))

    # P10 and P90 as labelled boundaries
    for p, clr, nm in [(10, T.DOWN, "P10"), (90, T.UP, "P90")]:
        fig.add_trace(go.Scatter(
            x=future_dates, y=nav_pcts[p],
            mode="lines", name=nm,
            line=dict(color=clr, width=1, dash="dot"),
            hovertemplate=f"%{{x|%b %Y}}: NAV %{{y:.1f}}<extra>{nm}</extra>",
        ))

    # Break-even line
    fig.add_hline(
        y=initial_nav, line_dash="dash",
        line_color="rgba(255,255,255,0.5)",
        annotation_text="Starting NAV (100)",
        annotation_position="bottom left",
    )

    years = mc["horizon_years"]
    fig.update_layout(base_layout(
        title=f"Monte Carlo Simulation — {years:.0f}-Year Horizon  ({mc['n_sims']:,} paths)",
        height=480,
    ))
    fig.update_layout(
        yaxis=dict(title="Simulated NAV (start = 100)"),
        xaxis=dict(title=""),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def _plot_terminal_distribution(mc: dict) -> go.Figure:
    """Histogram of simulated terminal total returns."""
    terminal_rets = mc["terminal_stats"]["terminal_returns"] * 100  # in %
    var_95_pct    = mc["terminal_stats"]["var_cvar"][0.95]["var"] * 100

    # Color bins: positive = green, negative = red
    pos_rets = terminal_rets[terminal_rets >= 0]
    neg_rets = terminal_rets[terminal_rets <  0]

    fig = go.Figure()
    if len(neg_rets) > 0:
        fig.add_trace(go.Histogram(
            x=neg_rets, nbinsx=60,
            marker_color=T.rgba(T.DOWN, 0.7), name="Loss",
            hovertemplate="Return %{x:.1f}%: %{y} paths<extra>Loss</extra>",
        ))
    if len(pos_rets) > 0:
        fig.add_trace(go.Histogram(
            x=pos_rets, nbinsx=80,
            marker_color=T.rgba(T.UP, 0.7), name="Gain",
            hovertemplate="Return %{x:.1f}%: %{y} paths<extra>Gain</extra>",
        ))

    # VaR line
    fig.add_vline(
        x=-var_95_pct, line_dash="dash", line_color=T.WARN,
        annotation_text=f"VaR 95% = {var_95_pct:.1f}%",
        annotation_position="top right",
    )

    years = mc["horizon_years"]
    fig.update_layout(base_layout(
        title=f"Terminal Total Return Distribution ({years:.0f}Y horizon)",
        height=360,
    ))
    fig.update_layout(
        xaxis=dict(title="Total Return (%)", ticksuffix="%"),
        yaxis=dict(title="Number of Paths"),
        barmode="overlay",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def _plot_dd_distribution(mc: dict) -> go.Figure:
    """Histogram of maximum drawdown across all simulated paths."""
    max_dds      = mc["terminal_stats"]["max_drawdowns"] * 100   # in %, negative
    # drawdown_at_risk_95 is a POSITIVE severity magnitude; the x-axis of this
    # histogram is negative drawdowns, so negate it to place the line.
    dar_95_pct   = -mc["terminal_stats"]["drawdown_at_risk_95"] * 100

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=max_dds, nbinsx=80,
        marker_color=T.rgba(T.DOWN, 0.65),
        name="Max Drawdown",
        hovertemplate="Max DD %{x:.1f}%: %{y} paths<extra></extra>",
    ))

    # Drawdown at Risk line (95th percentile of drawdown = worst 5% of outcomes)
    fig.add_vline(
        x=dar_95_pct, line_dash="dash", line_color=T.WARN,
        annotation_text=f"Drawdown at Risk (95%) = {abs(dar_95_pct):.1f}%",
        annotation_position="top left",
    )

    # Median drawdown
    median_dd = float(np.median(max_dds))
    fig.add_vline(
        x=median_dd, line_dash="dot", line_color="rgba(255,255,255,0.5)",
        annotation_text=f"Median DD = {abs(median_dd):.1f}%",
        annotation_position="top right",
    )

    years = mc["horizon_years"]
    fig.update_layout(base_layout(
        title=f"Maximum Drawdown Distribution ({years:.0f}Y horizon, {mc['n_sims']:,} paths)",
        height=380,
    ))
    fig.update_layout(
        xaxis=dict(title="Maximum Drawdown (%)", ticksuffix="%"),
        yaxis=dict(title="Number of Paths"),
    )
    return fig



# ─────────────────────────────────────────────────────────────────────────────
# HEADER + DISCLAIMER
# ─────────────────────────────────────────────────────────────────────────────
st.title("Predictive Analytics")
st.caption(f"Single-fund risk forecasting and scenario analysis · {sel_name[:70]}")

st.warning(
    "**This page contains scenario analysis and statistical risk estimation — not return forecasts.**  "
    "GARCH forecasts future *volatility* (empirically valid). Monte Carlo shows the *range of outcomes* "
    "consistent with the historical return distribution. Neither tool predicts the direction or magnitude "
    "of future returns. Past distributions may not reflect future market conditions.",
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
s1, s2 = st.columns(2, gap="large")

with s1:
    st.markdown("**Simulation Horizon**")
    horizon_label = st.radio(
        "Horizon", ["1 Year", "3 Years", "5 Years"],
        index=1, horizontal=True, label_visibility="collapsed",
    )
    horizon_years = {"1 Year": 1.0, "3 Years": 3.0, "5 Years": 5.0}[horizon_label]

with s2:
    st.markdown("**Monte Carlo Paths**")
    n_sims_label = st.radio(
        "Paths", ["1,000 (fast)", "5,000", "10,000 (recommended)"],
        index=2, horizontal=True, label_visibility="collapsed",
    )
    n_sims = {"1,000 (fast)": 1_000, "5,000": 5_000, "10,000 (recommended)": 10_000}[n_sims_label]

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# RUN BUTTON + COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
run_key = f"pred_{sel_code}_{rf_pct}_{horizon_years}_{n_sims}"

# Invalidate cache if settings changed
for k in list(st.session_state.keys()):
    if k.startswith("pred_") and k != run_key:
        if st.session_state.get("_pred_active_key") == k:
            st.session_state.pop(k, None)

run_btn = st.button(
    "Run Analysis",
    type="primary", width="stretch",
)

if run_btn or st.session_state.get(run_key):

    if not st.session_state.get(run_key):
        st.session_state["_pred_active_key"] = run_key
        prog = st.progress(0, text="Loading NAV history…")

        # Step 1: Load NAV
        nav_df  = get_nav_history(sel_code)
        nav     = process_nav(nav_df)

        if nav is None:
            st.error("Could not load NAV data for this fund."); st.stop()

        returns = compute_daily_returns(nav)
        if len(returns.dropna()) < 252:
            st.error("Insufficient NAV history (minimum 1 year required)."); st.stop()

        # Step 2: GARCH
        prog.progress(20, text="Fitting GARCH(1,1) volatility model…")
        garch_res = get_garch_summary(returns, rf_rate=rf_rate)

        # Step 3: Monte Carlo
        prog.progress(55, text=f"Running {n_sims:,} Monte Carlo simulations…")
        mc_res = run_monte_carlo(
            returns,
            horizon_years = horizon_years,
            n_sims        = n_sims,
            initial_nav   = 100.0,
            block_size    = 21,
        )

        prog.progress(100, text="Complete.")
        prog.empty()

        st.session_state[run_key] = {
            "nav":     nav,
            "returns": returns,
            "garch":   garch_res,
            "mc":      mc_res,
        }

    # ── Load from cache ───────────────────────────────────────────────────
    cached = st.session_state[run_key]
    nav     = cached["nav"]
    returns = cached["returns"]
    garch   = cached["garch"]
    mc      = cached["mc"]

    n_days  = len(returns.dropna())
    hist_yrs= n_days / TRADING_DAYS_PER_YEAR
    st.caption(
        f"Analysis based on **{n_days:,}** trading days "
        f"({hist_yrs:.1f} years · "
        f"{returns.dropna().index[0].strftime('%d %b %Y')} → "
        f"{returns.dropna().index[-1].strftime('%d %b %Y')})"
    )

    # ─────────────────────────────────────────────────────────────────────
    # TABS
    # ─────────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "Volatility Forecast",
        "Monte Carlo",
        "Drawdown Risk",
    ])

    # ─────────────────────────────────────────────────────────────────────
    # TAB 1: GARCH VOLATILITY
    # ─────────────────────────────────────────────────────────────────────
    with tab1:
        if not garch["is_valid"]:
            st.error(f"GARCH model could not be fitted: {garch.get('error', 'Unknown error')}")
        else:
            st.subheader("GARCH(1,1) Volatility Forecast")
            st.caption(
                "Volatility clustering — high-vol days tend to follow high-vol days — is one of "
                "the most robustly documented patterns in financial returns. GARCH(1,1) captures "
                "this structure and provides genuinely informative short-horizon volatility forecasts."
            )

            # ── KPI row ───────────────────────────────────────────────────
            k1, k2, k3, k4, k5 = st.columns(5)
            regime_color = {"High": "", "Normal": "", "Low": ""}.get(
                garch["current_vol_regime"], ""
            )
            k1.metric(
                "Current Vol (Ann.)",
                f"{garch['current_ann_vol']*100:.2f}%",
                delta=f"{regime_color} {garch['current_vol_regime']} regime",
                delta_color="off",
            )
            if garch["forecasts"]:
                k2.metric("30-Day Forecast", f"{garch['forecasts'][30]*100:.2f}%")
                k3.metric("60-Day Forecast", f"{garch['forecasts'][60]*100:.2f}%")
                k4.metric("90-Day Forecast", f"{garch['forecasts'][90]*100:.2f}%")
            k5.metric("Historical Avg", f"{garch['historical_avg_vol']*100:.2f}%")

            st.divider()

            # ── Conditional volatility chart ──────────────────────────────
            chart(_plot_garch_vol(garch, sel_name))

            st.divider()

            # ── VaR / CVaR ────────────────────────────────────────────────
            st.subheader("1-Day Value at Risk & Expected Shortfall")
            st.caption(
                "VaR: maximum expected daily loss at the given confidence level.  "
                "CVaR (Expected Shortfall): average loss in the worst (1-CL)% of days.  "
                "Based on current GARCH conditional volatility assuming normal residuals."
            )

            if garch["var_cvar"]:
                vc = garch["var_cvar"]
                v1, v2, v3, v4 = st.columns(4)
                v1.metric("VaR 95%",  fmt_pct(vc[0.95]["var"]),  help="Max 1-day loss at 95% confidence")
                v2.metric("CVaR 95%", fmt_pct(vc[0.95]["cvar"]), help="Avg 1-day loss in worst 5% of days")
                v3.metric("VaR 99%",  fmt_pct(vc[0.99]["var"]),  help="Max 1-day loss at 99% confidence")
                v4.metric("CVaR 99%", fmt_pct(vc[0.99]["cvar"]), help="Avg 1-day loss in worst 1% of days")

            st.divider()

            # ── Model parameters ──────────────────────────────────────────
            st.subheader("Model Parameters")
            pers = garch["persistence"]
            hl   = garch["half_life_days"]
            p1, p2, p3, p4, p5 = st.columns(5)
            p1.metric("ω (Baseline)",  f"{garch['omega']:.6f}"  if garch["omega"]  is not None else "N/A")
            p2.metric("α (Shock)",     f"{garch['alpha']:.4f}"  if garch["alpha"]  is not None else "N/A",
                      help="Sensitivity to last period's shock")
            p3.metric("β (Persistence)",f"{garch['beta']:.4f}" if garch["beta"]   is not None else "N/A",
                      help="Persistence of past variance")
            p4.metric("α + β",         f"{pers:.4f}"            if pers is not None else "N/A",
                      help="Near 1 = long-memory volatility")
            p5.metric("Half-Life",     f"{hl:.0f} days"         if hl  is not None else "N/A",
                      help="Days for a vol shock to decay to half its initial size")

            if pers is not None:
                if pers > 0.97:
                    st.info(
                        f"High persistence (α+β = {pers:.4f}): volatility shocks decay very slowly. "
                        "Current elevated or suppressed volatility is likely to persist for weeks.",
                    )
                elif pers < 0.85:
                    st.info(
                        f"Low persistence (α+β = {pers:.4f}): volatility reverts quickly to average. "
                        "Current conditions are unlikely to persist beyond a few days.",
                    )

    # ─────────────────────────────────────────────────────────────────────
    # TAB 2: MONTE CARLO
    # ─────────────────────────────────────────────────────────────────────
    with tab2:
        if not mc["is_valid"]:
            st.error(f"Monte Carlo simulation failed: {mc.get('error', 'Unknown error')}")
        else:
            ts = mc["terminal_stats"]

            st.subheader(f"Monte Carlo Scenario Analysis — {horizon_label} Horizon")
            st.caption(
                f"**{mc['n_sims']:,} paths** generated by block bootstrap (21-day blocks) resampling "
                f"from the fund's own historical return distribution. "
                "No normality assumption. Fat tails and skewness are preserved exactly as observed."
            )

            # ── KPI row ───────────────────────────────────────────────────
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric(
                "Median Ann. Return",
                fmt_pct(ts["median_ann_return"]),
                help="50th percentile annualised return across all paths",
            )
            k2.metric(
                "Mean Ann. Return",
                fmt_pct(ts["mean_ann_return"]),
                help="Average annualised return across all paths",
            )
            k3.metric(
                "Probability of Loss",
                f"{ts['prob_loss']*100:.1f}%",
                help="% of paths where terminal NAV < starting NAV",
            )
            if 0.95 in ts["var_cvar"]:
                k4.metric(
                    f"VaR 95% ({horizon_label})",
                    fmt_pct(ts["var_cvar"][0.95]["var"]),
                    help="Total loss exceeded in 5% of worst outcomes",
                )
                k5.metric(
                    f"CVaR 95% ({horizon_label})",
                    fmt_pct(ts["var_cvar"][0.95]["cvar"]),
                    help="Average total loss in worst 5% of outcomes",
                )

            st.divider()

            # ── Fan chart ─────────────────────────────────────────────────
            chart(_plot_mc_fan(mc, sel_name))

            # ── Terminal return distribution ───────────────────────────────
            st.subheader("Terminal Return Distribution")
            col_hist, col_table = st.columns([2, 1], gap="large")

            with col_hist:
                chart(_plot_terminal_distribution(mc))

            with col_table:
                st.markdown(f"**Outcome Percentiles ({horizon_label})**")
                pct_data = ts["return_percentiles"]
                pct_df = pd.DataFrame([
                    {"Percentile": f"P{p}", "Total Return": f"{v:.1f}%"}
                    for p, v in sorted(pct_data.items())
                ])
                st.dataframe(pct_df, width="stretch", hide_index=True)
                export_button(pct_df, "outcome_percentiles.csv",
                              label="↓ Percentiles CSV", key="dl_pa_pct")

                st.markdown(f"**Shortfall Probabilities**")
                vc = ts["var_cvar"]
                sf_rows = []
                for cl, vals in sorted(vc.items(), reverse=True):
                    sf_rows.append({
                        "Confidence": f"{int(cl*100)}%",
                        "VaR":  fmt_pct(vals["var"]),
                        "CVaR": fmt_pct(vals["cvar"]),
                    })
                _sf_df = pd.DataFrame(sf_rows)
                st.dataframe(_sf_df, width="stretch", hide_index=True)
                export_button(_sf_df, "shortfall_var_cvar.csv",
                              label="↓ VaR / CVaR CSV", key="dl_pa_sf")

    # ─────────────────────────────────────────────────────────────────────
    # TAB 3: DRAWDOWN RISK
    # ─────────────────────────────────────────────────────────────────────
    with tab3:
        if not mc["is_valid"]:
            st.error("Drawdown Risk requires Monte Carlo — see Tab 2 for error.")
        else:
            ts = mc["terminal_stats"]

            st.subheader(f"Drawdown Risk Analysis — {horizon_label} Horizon")
            st.caption(
                "Derived from Monte Carlo simulated paths. For each of the "
                f"{mc['n_sims']:,} paths, the maximum drawdown over the full "
                f"{horizon_label} horizon is computed. The distribution below "
                "shows how severe a drawdown could be under historical conditions."
            )

            # ── KPI row ───────────────────────────────────────────────────
            # dd_pcts are POSITIVE severity magnitudes (see monte_carlo.py):
            # percentile p = "p% of paths had a drawdown no worse than this",
            # so severity increases left to right across this row.
            dd_pcts = ts["max_dd_percentiles"]
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Median Max DD",     f"{dd_pcts[50]:.1f}%",
                      help="Half of all simulated paths saw a drawdown worse than this")
            k2.metric("P75 Max DD",        f"{dd_pcts[75]:.1f}%",
                      help="1 path in 4 saw a drawdown worse than this")
            k3.metric("P90 Max DD",        f"{dd_pcts[90]:.1f}%",
                      help="1 path in 10 saw a drawdown worse than this")
            k4.metric("Drawdown at Risk",  f"{ts['drawdown_at_risk_95']*100:.1f}%",
                      help="Only 5% of simulated paths saw a drawdown worse than this")
            k5.metric("P99 Max DD",        f"{dd_pcts[99]:.1f}%",
                      help="The severe tail — 1 path in 100 was worse than this")

            st.divider()

            # ── Distribution chart ────────────────────────────────────────
            chart(_plot_dd_distribution(mc))

            st.divider()

            # ── Exceedance probability table ──────────────────────────────
            col_prob, col_note = st.columns([1, 1], gap="large")

            with col_prob:
                st.markdown(f"**Probability of Max Drawdown Exceeding Threshold ({horizon_label})**")
                dd_probs = ts["dd_exceed_probs"]
                prob_rows = [
                    {
                        "Drawdown Threshold": f">{int(thr*100)}%",
                        "Probability":        f"{prob*100:.1f}%",
                        "1-in-N":             f"1 in {1/prob:.0f}" if prob > 0 else "Rare",
                    }
                    for thr, prob in sorted(dd_probs.items())
                ]
                _prob_df = pd.DataFrame(prob_rows)
                st.dataframe(_prob_df, width="stretch", hide_index=True)
                export_button(_prob_df, "drawdown_probabilities.csv",
                              label="↓ Drawdown probs CSV", key="dl_pa_dd")

            with col_note:
                st.markdown("**How to read this table**")
                st.markdown(
                    "Each row answers: *'If I hold this fund for "
                    f"{horizon_label}, what is the probability that I experience "
                    "a peak-to-trough loss greater than X% at some point?'*  \n\n"
                    "**Drawdown at Risk (95%)** means: in 95% of historical scenarios, "
                    "the maximum drawdown over this horizon did not exceed that level. "
                    "Only 5% of simulated paths experienced something worse.  \n\n"
                    "Probabilities are based on **historical return patterns only**. "
                    "Future crises may be more severe than any in the training window."
                )

else:
    st.info(
        "Configure settings above and click **Run Analysis** to compute "
        "GARCH volatility forecasts, Monte Carlo scenario analysis, and drawdown risk.",
    )