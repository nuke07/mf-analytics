"""
pages/3_Fund_Analytics.py
==========================
Fund Analytics — Deep Dive
"""

import streamlit as st
import pandas as pd
import numpy as np

from data.fund_loader        import get_nav_history, get_all_categorized_schemes
from data.benchmark_loader   import (
    get_benchmark_nav, get_benchmark_info, get_market_nav,
    MARKET_DISPLAY_NAME, is_market_same_as_category,
)
from data.factor_loader      import (
    get_factor_returns_6f, get_factor_availability_6f, FACTOR_DISPLAY_NAMES,
)
from analytics.engine        import compute_fund_metrics
from analytics.factor_model  import FACTOR_6F_NAMES
from visualizations.nav_chart        import plot_single_nav, plot_trailing_returns
from visualizations.drawdown_chart   import plot_drawdown
from visualizations.rolling_returns  import plot_rolling_combined
from visualizations.alpha_charts     import (
    plot_fund_vs_benchmark, plot_rolling_alpha,
)
from visualizations.momentum_charts  import (
    plot_bull_bear_alpha, plot_alpha_persistence_timeline,
)
from visualizations.factor_charts    import plot_rolling_alpha_factors
from utils.constants  import (
    COVERAGE_LABELS,
)
from utils.formatters import fmt_pct, fmt_ratio, fmt_days, fmt_nav, fmt_date
from utils.validators import build_quality_report
from utils.session    import (
    fund_key as _fund_key,
)
from utils.ui          import (
    sidebar_header, category_selector, plan_selector, rf_control,
    tri_status, kpi, render_refresh_button,
    chart,
    kpi_row,
)

st.set_page_config(page_title="Fund Analytics — MF Analytics", page_icon="📋", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
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

    prev = st.session_state.get("selected_fund", fund_names[0])
    idx  = fund_names.index(prev) if prev in fund_names else 0
    selected_name = st.selectbox("Select Fund", fund_names, index=idx)
    st.session_state["selected_fund"] = selected_name
    selected_code = fund_codes[selected_name]

    st.divider()
    rf_pct, rf_rate = rf_control()

    render_refresh_button()
    tri_status()
        

# ── Load + Compute ─────────────────────────────────────────────────────────────
st.title("Fund Analytics")

# Compute the COMPLETE metric set once — base + alpha + factor model.
#
# This page previously ran the engine three times: once here without a
# benchmark (so every alpha and factor metric came back None), and again
# inside the Alpha and Factor tabs with the benchmark supplied. The All
# Metrics tab read the first, benchmark-less result and therefore showed
# "N/A" for 18 metrics that the other two tabs were displaying with real
# values, for the same fund, on the same screen.
#
# Computing the richest set up front fixes that and drops two redundant
# NAV fetches per fund.
ck = _fund_key(selected_code, rf_pct, category)
if ck not in st.session_state:
    with st.spinner(f"Loading NAV for {selected_name[:60]}…"):
        nav_df = get_nav_history(selected_code)

    with st.spinner("Loading benchmark and factor data…"):
        bm_info_base = get_benchmark_info(category)
        bm_nav_base  = get_benchmark_nav(category) if bm_info_base.get("available") else None
        try:
            factor_df_base, _, _ = get_factor_returns_6f(rf_rate=rf_rate)
        except Exception:
            factor_df_base = None
        # Second yardstick: the broad market, alongside the SEBI category
        # benchmark. For Flexi/Multi/ELSS/Value/Contra/Focused these are the
        # same index, which the Alpha tab says out loud rather than showing
        # two identical columns as if they were independent readings.
        mkt_nav_base = get_market_nav()

    with st.spinner("Computing metrics…"):
        metrics = compute_fund_metrics(
            nav_df,
            rf_rate           = rf_rate,
            fund_name         = selected_name,
            benchmark_nav_df  = bm_nav_base,
            benchmark_name    = bm_info_base.get("display_name"),
            factor_returns_df = factor_df_base,
            market_nav_df     = mkt_nav_base,
            market_name       = MARKET_DISPLAY_NAME,
        )
    st.session_state[ck] = metrics
else:
    metrics = st.session_state[ck]

if not metrics.get("is_valid"):
    st.error(f"Could not compute metrics for **{selected_name}**.")
    for w in metrics.get("warnings", []): st.warning(w)
    st.stop()

for w in metrics.get("warnings", []): st.warning(w)
summary = metrics.get("summary", {})

# ── Header ────────────────────────────────────────────────────────────────────
st.subheader(selected_name)
st.caption(
    f"Universe: **{plan_type} plans** | Category: **{category}** | "
    f"Scheme Code: `{selected_code}` | "
    f"Inception: {fmt_date(summary.get('start_date'))} | "
    f"History: {summary.get('history_years', 'N/A')} years | "
    f"Latest NAV: {fmt_nav(summary.get('current_nav'))} "
    f"({fmt_date(summary.get('end_date'))})"
)
st.divider()

# ── KPI cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

kpi(k1, "3Y CAGR", metrics.get("cagr_3y"), kind='pct', metric_key="cagr_3y")
kpi(k2, "1Y CAGR", metrics.get("cagr_1y"), kind='pct', metric_key="cagr_1y")
kpi(k3, "Ann. Volatility", metrics.get("annualized_volatility"), kind='pct', metric_key="annualized_volatility")

# Sharpe belongs in the headline row, and it belongs there WITH its interval.
# The confidence band first shipped buried in the All Metrics tab inside a
# collapsed expander, which is the same as not shipping it: the number people
# actually act on is the one on the front page, and a bare 0.32 invites a
# precision the estimate does not have. The band is the delta line, uncoloured,
# because an interval is context and not good or bad news.
_ci_lo = metrics.get("sharpe_ci_low")
_ci_hi = metrics.get("sharpe_ci_high")
kpi(k4, "Sharpe Ratio", metrics.get("sharpe"), kind='ratio', metric_key="sharpe",
    delta=(f"95% CI  {_ci_lo:+.2f} to {_ci_hi:+.2f}"
           if _ci_lo is not None and _ci_hi is not None else None),
    delta_color="off")

if _ci_lo is not None and _ci_lo < 0 <= metrics.get("sharpe", 0):
    st.caption(
        f"The Sharpe interval includes zero — on {metrics.get('sharpe_n_obs'):,} "
        "observations this fund has not reliably beaten cash. Two funds whose "
        "intervals overlap cannot be ranked against each other."
    )
st.divider()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_charts, tab_alpha, tab_factor, tab_metrics, tab_quality = st.tabs([
    "Charts",
    "Alpha Analytics",
    "Factor Model",
    "All Metrics",
    "Data Quality",
])

# ── TAB 1: CHARTS ──────────────────────────────────────────────────────────────
with tab_charts:
    nav = metrics.get("nav")
    dd  = metrics.get("drawdown_series")
    s1  = metrics.get("_series_1y")
    s3  = metrics.get("_series_3y")

    r1l, r1r = st.columns(2, gap="medium")
    with r1l:
        if nav is not None:
            chart(plot_single_nav(nav, selected_name))
        else:
            st.warning("NAV chart not available.")
    with r1r:
        if dd is not None:
            chart(plot_drawdown({selected_name: dd}))
        else:
            st.warning("Drawdown chart not available.")

    if s1 is not None:
        chart(
            plot_rolling_combined({selected_name: s1}, window_label="1-Year", height=650),
        )
    else:
        st.info("1-Year rolling returns require at least 2 years of NAV history.")

    if s3 is not None:
        chart(
            plot_rolling_combined({selected_name: s3}, window_label="3-Year", height=650),
        )


# ── TAB 2: ALPHA ANALYTICS ─────────────────────────────────────────────────────
with tab_alpha:
    st.subheader("Alpha Analytics")

    bm_info = get_benchmark_info(category)
    st.info(
        f"**Benchmark:** {bm_info['display_name']}  |  "
        f"**Proxy used:** {bm_info['scheme_name'][:70]}  |  "
        f"**Available:** {'✓' if bm_info['available'] else 'Not found'}",
    )

    if not bm_info["available"]:
        st.warning("No benchmark index fund found. Check connectivity.")
    else:
        # Already computed once at page load, with the benchmark supplied.
        full_metrics = metrics

        # ── Two yardsticks, category first ────────────────────────────────────
        _same_bm = is_market_same_as_category(category)

        st.markdown(f"**vs Category Benchmark — {bm_info['display_name']}**")
        st.caption(
            "The SEBI-mandated comparison: did this fund beat the yardstick "
            "its peers are measured against?"
        )
        a1, a2, a3, a4, a5 = st.columns(5)
        kpi(a1, "Jensen's Alpha",    full_metrics.get("jensens_alpha"), kind='pct',
              help="Return above what the fund's beta to this benchmark would predict.")
        kpi(a2, "Alpha t-Stat",      full_metrics.get("alpha_tstat"),
              help="|t| ≥ 2 suggests the alpha is unlikely to be chance.")
        kpi(a3, "Information Ratio", full_metrics.get("information_ratio"),
              help="Excess return per unit of tracking error.")
        kpi(a4, "Capture Ratio",     full_metrics.get("capture_ratio"),
              help="Up-capture ÷ down-capture. Above 1 is favourable.")
        kpi(a5, "Beta",              full_metrics.get("beta"),
              help="Sensitivity to this benchmark. 1.0 = moves one-for-one.")

        st.markdown(f"**vs Broad Market — {MARKET_DISPLAY_NAME}**")
        if _same_bm:
            st.caption(
                f"For **{category}**, the category benchmark *is* the broad "
                "market under SEBI, so these two rows are the same measurement "
                "— not two independent readings."
            )
        else:
            st.caption(
                "The same fund measured against simply owning the market. A "
                f"fund can beat {bm_info['display_name']} and still lag "
                f"{MARKET_DISPLAY_NAME}, or the reverse — both are real."
            )
        m1, m2, m3, m4, m5 = st.columns(5)
        kpi(m1, "Jensen's Alpha",    full_metrics.get("jensens_alpha_mkt"), kind='pct')
        kpi(m2, "Alpha t-Stat", full_metrics.get("alpha_tstat_mkt"), kind='ratio')
        kpi(m3, "Information Ratio", full_metrics.get("information_ratio_mkt"), kind='ratio')
        kpi(m4, "Capture Ratio", full_metrics.get("capture_ratio_mkt"), kind='ratio')
        kpi(m5, "Beta", full_metrics.get("beta_mkt"), kind='ratio')

        st.divider()

        # ── Fund vs Benchmark — period selector ───────────────────────────────
        st.subheader("Fund vs Benchmark — Trailing Returns")
        period_alpha = st.radio(
            "Period",
            options    = ["1M", "3M", "6M", "1Y", "3Y", "5Y", "All"],
            index      = 3,
            horizontal = True,
            key        = "alpha_period",
        )

        bm_nav_obj   = full_metrics.get("_benchmark_nav")
        fund_nav_obj = full_metrics.get("nav")
        _market_nav_series = full_metrics.get("_market_nav")

        if fund_nav_obj is not None and bm_nav_obj is not None:
            chart(
                plot_fund_vs_benchmark(
                    fund_nav_obj, bm_nav_obj,
                    selected_name, bm_info["display_name"],
                    period_label=period_alpha,
                    height=460,
                    # Third line only where it adds something: for the
                    # categories benchmarked to Nifty 500 it would trace the
                    # category line exactly.
                    market_nav  = None if _same_bm else _market_nav_series,
                    market_name = MARKET_DISPLAY_NAME,
                ),
            )
        elif fund_nav_obj is not None:
            chart(
                plot_trailing_returns(
                    {selected_name: fund_nav_obj},
                    period_label=period_alpha,
                ),
            )

        # Rolling alpha
        roll_alpha = full_metrics.get("_rolling_alpha")
        if roll_alpha is not None:
            chart(
                plot_rolling_alpha({selected_name: roll_alpha}, "1-Year"),
            )

        sig = full_metrics.get("alpha_tstat")
        if sig is not None:
            if abs(sig) >= 2.0:
                st.success(f"Alpha is **statistically significant** (|t| = {sig:.2f} ≥ 2.0) — manager skill likely real.")
            else:
                st.warning(f"Alpha is **not statistically significant** (|t| = {sig:.2f} < 2.0) — may be noise.")

        st.divider()

        # ── Momentum & Persistence ────────────────────────────────────────────
        st.subheader("Return Momentum")
        m1, m2, m3 = st.columns(3)
        kpi(m1, "1M Return", full_metrics.get("momentum_1m"),
            kind='pct', metric_key="momentum_1m")
        kpi(m2, "3M Return", full_metrics.get("momentum_3m"),
            kind='pct', metric_key="momentum_3m")
        kpi(m3, "6M Return", full_metrics.get("momentum_6m"),
            kind='pct', metric_key="momentum_6m")
        st.caption(
            "Trailing returns, not annualised. A 12-month figure is omitted "
            "because it is arithmetically identical to the 1Y CAGR shown above."
        )

        st.divider()
        st.subheader("Alpha Persistence")
        p1, p2, p3, p4 = st.columns(4)
        kpi(p1, "Persistence Score", full_metrics.get("alpha_persistence"),
            kind='pct', metric_key="alpha_persistence")
        kpi(p2, "Bull Alpha", full_metrics.get("bull_alpha"),
            kind='pct', metric_key="bull_alpha")
        kpi(p3, "Bear Alpha", full_metrics.get("bear_alpha"),
            kind='pct', metric_key="bear_alpha")
        kpi(p4, "Regime Ratio",      full_metrics.get("alpha_regime_ratio"), kind='ratio')

        roll_alpha_obj = full_metrics.get("_rolling_alpha")
        if roll_alpha_obj is not None:
            chart(
                plot_alpha_persistence_timeline(roll_alpha_obj, selected_name),
            )
        if full_metrics.get("bull_alpha") is not None:
            chart(
                plot_bull_bear_alpha({selected_name: full_metrics}),
            )


# ── TAB 3: FACTOR MODEL ────────────────────────────────────────────────────────
with tab_factor:
    st.subheader("Factor Model (6-Factor)")

    avail     = get_factor_availability_6f()
    n_factors = sum(bool(v) for v in avail.values())

    st.info(
        f"**Model:** {n_factors}-Factor  |  " +
        "  |  ".join([
            f"{'✓' if avail.get(f) else '·'} {FACTOR_DISPLAY_NAMES.get(f, f)}"
            for f in FACTOR_6F_NAMES
        ]),
    )

    if n_factors < 6:
        st.warning(
            "The 6-factor model needs all six factors. Quality (QMJ) and "
            "Low-Vol (BAB) are TRI-only — no index-fund proxy exists for "
            "them — so a missing TRI file disables the model entirely. "
            "Check data/tri/ for NIFTY_200_QUALITY_30_TRI.csv and "
            "NIFTY_100_LOW_VOLATILITY_30_TRI.csv."
        )
    else:
        # Already computed once at page load, with factor returns supplied.
        # This previously used its own cache key — a raw f-string that
        # skipped ANALYTICS_VERSION and was not cleared by the Refresh
        # button, so factor results could silently outlive a version bump.
        factor_metrics = metrics

        # Nine cards laid out in rows of five. Written as columns(5) followed
        # by columns(4), this rendered four WIDE cards beneath five narrow
        # ones — Streamlit stretches whatever count it is given to the full
        # container width, so the card edges never lined up. kpi_row() pads
        # the short row instead, keeping every card the same size.
        kpi_row([
            {"label": "6F Alpha (Ann.)",  "value": factor_metrics.get("alpha_6f"),
             "kind": "pct", "metric_key": "alpha_6f"},
            {"label": "6F t-Stat",        "value": factor_metrics.get("alpha_6f_tstat"),
             "metric_key": "alpha_6f_tstat"},
            {"label": "6F R²",            "value": factor_metrics.get("r_squared_6f"),
             "metric_key": "r_squared_6f"},
            {"label": "Market β",         "value": factor_metrics.get("beta_market_6f"),
             "metric_key": "beta_market_6f"},
            {"label": "Size β (SMB)",     "value": factor_metrics.get("beta_smb"),
             "metric_key": "beta_smb"},
            {"label": "Value β (HML)",    "value": factor_metrics.get("beta_hml"),
             "metric_key": "beta_hml"},
            {"label": "Momentum β (WML)", "value": factor_metrics.get("beta_wml"),
             "metric_key": "beta_wml"},
            {"label": "Quality β (QMJ)",  "value": factor_metrics.get("beta_qmj"),
             "metric_key": "beta_qmj"},
            {"label": "Low-Vol β (BAB)",  "value": factor_metrics.get("beta_bab"),
             "metric_key": "beta_bab"},
        ], per_row=5)

        st.caption(
            "Betas are raw (conventional) loadings: 1.0 on Market means the "
            "fund moves one-for-one with Nifty 500. Alpha is what remains "
            "after all six factor exposures are paid for — the part not "
            "explained by simply tilting toward size, value, momentum, "
            "quality or low volatility."
        )

        roll_6f = factor_metrics.get("_rolling_alpha_6f")
        if roll_6f is not None:
            chart(
                plot_rolling_alpha_factors({selected_name: roll_6f}),
            )

        tstat = factor_metrics.get("alpha_6f_tstat")
        if tstat is not None:
            if abs(tstat) >= 2.0:
                st.success(
                    f"6-Factor Alpha is **statistically significant** "
                    f"(|t| = {tstat:.2f}) over this sample — the excess return "
                    "is unlikely to be explained by factor tilts alone."
                )
            else:
                st.warning(
                    f"6-Factor Alpha is **not statistically significant** "
                    f"(|t| = {tstat:.2f}) — the fund's excess return is "
                    "consistent with its factor exposures rather than skill."
                )


# ── TAB 4: ALL METRICS ─────────────────────────────────────────────────────────
with tab_metrics:
    st.caption("All quantitative metrics computed for this fund.")

    SECTIONS = {
        "Performance": [
            ("cagr_1y",        "1-Year CAGR",         "pct"),
            ("cagr_3y",        "3-Year CAGR",         "pct"),
            ("cagr_5y",        "5-Year CAGR",         "pct"),
            ("cagr_inception", "Since Inception CAGR","pct"),
        ],
        "Volatility": [
            ("annualized_volatility", "Annualized Volatility", "pct"),
            ("downside_volatility",   "Downside Volatility",   "pct"),
        ],
        "Risk": [
            ("max_drawdown",      "Maximum Drawdown",       "pct"),
            ("avg_drawdown",      "Average Drawdown",       "pct"),
            ("drawdown_duration", "Max Drawdown Duration",  "days"),
        ],
        "Risk-Adjusted": [
            ("sharpe",  "Sharpe Ratio",  "ratio"),
            ("sortino", "Sortino Ratio", "ratio"),
            ("calmar",  "Calmar Ratio",  "ratio"),
            # A Sharpe ratio is an estimate. On three years of data its standard
            # error is near 0.58 — wide enough that a headline 1.00 is not
            # reliably different from 0.50. Jensen's alpha has shipped with a
            # t-statistic since day one; the metric the rankings actually sort
            # on shipped bare. See analytics/uncertainty.py.
            ("sharpe_ci_low",  "Sharpe — 95% Low",  "ratio"),
            ("sharpe_ci_high", "Sharpe — 95% High", "ratio"),
            ("sharpe_se",      "Sharpe Std Error",  "ratio"),
            ("sharpe_n_obs",   "Observations",      "count"),
            ("sharpe_acf_inflation", "Serial-Correlation Factor", "ratio"),
        ],
        "Alpha (vs Benchmark)": [
            ("excess_return",    "Excess Return (Ann.)",    "pct"),
            ("beta",             "Beta",                    "ratio"),
            ("r_squared",        "R-Squared",               "ratio"),
            ("tracking_error",   "Tracking Error",          "pct"),
            ("information_ratio","Information Ratio",        "ratio"),
            ("jensens_alpha",    "Jensen's Alpha (Ann.)",   "pct"),
            ("alpha_tstat",      "Alpha t-Statistic",       "ratio"),
            ("up_capture",       "Up-Capture Ratio",        "num"),
            ("down_capture",     "Down-Capture Ratio",      "num"),
            ("capture_ratio",    "Capture Ratio",           "ratio"),
        ],
        "Momentum": [
            ("momentum_1m",   "1M Return",            "pct"),
            ("momentum_3m",   "3M Return",            "pct"),
            ("momentum_6m",   "6M Return",            "pct"),
            ("alpha_momentum","Alpha Momentum (12M)", "pct"),
        ],
        "Alpha Persistence": [
            ("alpha_persistence",     "Alpha Persistence Score", "pct"),
            ("bull_alpha",            "Bull Market Alpha",       "pct"),
            ("bear_alpha",            "Bear Market Alpha",       "pct"),
            ("alpha_regime_ratio",    "Alpha Regime Ratio",      "ratio"),
            ("drawdown_recovery_rate","Drawdown Recovery (days)","days"),
        ],
        "Factor Model (6F)": [
            ("alpha_6f",        "6-Factor Alpha (Ann.)",   "pct"),
            ("alpha_6f_tstat",  "6-Factor t-Stat",         "ratio"),
            ("r_squared_6f",    "6-Factor R-Squared",      "ratio"),
            ("beta_market_6f",  "Market Beta",             "ratio"),
            ("beta_smb",        "Size Loading (SMB)",      "ratio"),
            ("beta_hml",        "Value Loading (HML)",     "ratio"),
            ("beta_wml",        "Momentum Loading (WML)",  "ratio"),
            ("beta_qmj",        "Quality Loading (QMJ)",   "ratio"),
            ("beta_bab",        "Low-Vol Loading (BAB)",   "ratio"),
            ("contrib_alpha",   "Pure Alpha Contribution", "pct"),
        ],
        "vs Broad Market (Nifty 500 TRI)": [
            ("jensens_alpha_mkt",     "Jensen's Alpha (Ann.)", "pct"),
            ("alpha_tstat_mkt",       "Alpha t-Stat",          "ratio"),
            ("beta_mkt",              "Beta",                  "ratio"),
            ("r_squared_mkt",         "R-Squared",             "ratio"),
            ("tracking_error_mkt",    "Tracking Error",        "pct"),
            ("information_ratio_mkt", "Information Ratio",     "ratio"),
            ("up_capture_mkt",        "Up-Capture (%)",        "num"),
            ("down_capture_mkt",      "Down-Capture (%)",      "num"),
            ("capture_ratio_mkt",     "Capture Ratio",         "ratio"),
        ],
        "Consistency (1Y Rolling)": [
            ("median_rolling_1y", "Median 1Y Rolling", "pct"),
            ("std_rolling_1y",    "Std Dev 1Y Rolling","pct"),
            ("best_rolling_1y",   "Best 1Y Rolling",   "pct"),
            ("worst_rolling_1y",  "Worst 1Y Rolling",  "pct"),
        ],
        "Stability": [
            ("positive_freq", "Positive Day Frequency", "pct"),
            ("win_rate",      "Monthly Win Rate",        "pct"),
        ],
    }

    def _fmt(val, kind):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        if kind == "pct":   return fmt_pct(val)
        if kind == "ratio": return fmt_ratio(val)
        if kind == "days":  return fmt_days(val)
        if kind == "num":   return f"{val:.2f}%"
        # A count is not a percentage — sharpe_n_obs went in as "num" and
        # rendered 5,305 observations as "5305.00%".
        if kind == "count": return f"{int(val):,}"
        return str(val)

    for section_title, metric_list in SECTIONS.items():
        with st.expander(section_title, expanded=False):
            rows = [
                {"Metric": label, "Value": _fmt(metrics.get(key), kind)}
                for key, label, kind in metric_list
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ── TAB 5: DATA QUALITY ────────────────────────────────────────────────────────
with tab_quality:
    nav    = metrics.get("nav")
    report = build_quality_report(selected_name, nav)

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("History",     f"{report.get('history_years', 0)} yrs")
    q2.metric("Data Points", f"{report.get('data_points', 0):,}")
    _miss = report.get("missing_pct")
    q3.metric("Missing %",
              "N/A" if _miss is None or (isinstance(_miss, float) and np.isnan(_miss))
              else f"{_miss:.1f}%")
    q4.metric("Start Date",  fmt_date(report.get("start_date")))

    for w in report.get("warnings", []): st.warning(w)

    st.subheader("Metric Coverage")
    coverage = report.get("coverage", {})
    cov_rows = [
        {
            # COVERAGE_LABELS, not METRIC_LABELS — the coverage map is keyed
            # on MIN_DAYS names ("1y_cagr"), not engine names ("cagr_1y"),
            # so METRIC_LABELS missed 8 of 18 and printed the raw key.
            "Metric":    COVERAGE_LABELS.get(k, k),
            "Available": "✓ Yes" if v else "No (insufficient history)",
        }
        for k, v in coverage.items()
    ]
    yes_n = sum(1 for r in cov_rows if "Yes" in r["Available"])
    st.caption(f"{yes_n} of {len(cov_rows)} metrics available.")
    st.dataframe(pd.DataFrame(cov_rows), width="stretch", hide_index=True)
