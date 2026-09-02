"""
pages/5_Rankings.py
====================
Rankings — Category-wise Ranking Tables

Six tabs, driven by the RANK_SPEC declaration in the body:
  1. Returns            — CAGR (1Y/3Y/5Y/inception) + 1M/3M/6M trailing
  2. Risk               — Sharpe/Sortino/Calmar, volatility, drawdown
  3. Rolling & Hit Rate — median rolling returns, % positive, win rate
  4. Alpha & Factors    — category-relative alpha + 6-factor loadings
  5. vs Broad Market    — the same alpha family vs Nifty 500 TRI
  6. Quartile View      — heatmap + scatter + full metrics table

Was 11 tabs. Consistency / Stability / Persistence were three synonyms, and
two pairs shared an emoji so the strip could not be read at a glance. Every
table is now one tuple in RANK_SPEC rather than a hand-written block.
"""

import streamlit as st
import pandas as pd
import numpy as np

from data.fund_loader      import (
    get_all_categorized_schemes, load_navs_parallel,
)
from analytics.engine      import compute_category_metrics, compute_category_quartiles
from utils.formatters      import fmt_pct, fmt_ratio, fmt_days, style_quartile
from utils.session         import (
    rankings_done_key, category_full_df_key, category_fund_metrics_key,
)
from visualizations.alpha_charts    import plot_capture_scatter
from visualizations.momentum_charts import (
    plot_bull_bear_alpha, plot_momentum_heatmap,
)
from visualizations.scatter_plots   import plot_risk_return_scatter, plot_vol_cagr_scatter
from visualizations.heatmaps        import plot_quartile_heatmap
from utils.ui          import (
    sidebar_header, category_selector, plan_selector, rf_control,
    tri_status, render_refresh_button,
    chart,
)

st.set_page_config(page_title="Rankings — MF Analytics", page_icon="🏆", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    sidebar_header()

    category = category_selector()

    _top_n_choice = st.radio(
        "Funds per table", ["10", "20", "All"],
        horizontal=True,
        help="How many rows each ranking table shows. 'All' can be a very "
             "long page in a large category.",
    )
    top_n = 9999 if _top_n_choice == "All" else int(_top_n_choice)

    plan_type = plan_selector()

    st.divider()
    rf_pct, rf_rate = rf_control()

    render_refresh_button()
    tri_status()

# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"Rankings — {category}")
st.caption(
    f"**{plan_type} plans only.** "
    f"Funds ranked within **{category}** only. "
    "Rankings are not comparable across categories."
)
st.divider()

# ── Load Fund List ─────────────────────────────────────────────────────────────
with st.spinner("Loading fund list…"):
    all_cat   = get_all_categorized_schemes(plan_type=plan_type)
    fund_list = all_cat.get(category, [])

if not fund_list:
    st.warning("No funds found for this category."); st.stop()

# ── Analytics Trigger ─────────────────────────────────────────────────────────
st.info(
    f"Rankings require computing metrics for all **{len(fund_list)} funds** in {category}. "
    f"First run takes ~{len(fund_list)*3//60 + 1}–{len(fund_list)*5//60 + 2} minutes. "
    "Results are cached for 1 hour.",
)

analytics_key = rankings_done_key(category)
run_btn = st.button(
    f"Compute Rankings for {category}  ({len(fund_list)} funds)",
    type="primary", width="stretch",
)

if run_btn or st.session_state.get(analytics_key):

    if not st.session_state.get(analytics_key):
        # Concurrent fetch — a 40-fund category was 40 serial round-trips.
        progress = st.progress(0, text="Loading NAVs…")

        def _on_done(done, total, code):
            progress.progress(done / total,
                              text=f"Fetching NAVs — {done} of {total}")

        _frames = load_navs_parallel(
            [f["code"] for f in fund_list], progress_cb=_on_done,
        )
        nav_dict = {f["name"]: _frames.get(f["code"]) for f in fund_list}
        progress.empty()

        with st.spinner("Computing metrics + alpha + factor model for all funds…"):
            from data.benchmark_loader import (
                get_benchmark_nav, get_benchmark_info,
                get_market_nav, MARKET_DISPLAY_NAME,
            )
            from data.factor_loader    import get_factor_returns_6f
            bm_info   = get_benchmark_info(category)
            bm_nav_df = get_benchmark_nav(category) if bm_info["available"] else None
            factor_df, _, _ = get_factor_returns_6f(rf_rate=rf_rate)
            mkt_nav_df = get_market_nav()

            fund_metrics = compute_category_metrics(
                nav_dict,
                rf_rate           = rf_rate,
                benchmark_nav_df  = bm_nav_df,
                benchmark_name    = bm_info["display_name"],
                factor_returns_df = factor_df,
                market_nav_df     = mkt_nav_df,
                market_name       = MARKET_DISPLAY_NAME,
            )
            full_df = compute_category_quartiles(fund_metrics)

        st.session_state[category_full_df_key(category)]      = full_df
        st.session_state[category_fund_metrics_key(category)] = fund_metrics
        st.session_state[analytics_key]                        = True
        st.success(f"Rankings ready for {len(fund_metrics)} funds!")

    full_df      = st.session_state.get(category_full_df_key(category),      pd.DataFrame())
    fund_metrics = st.session_state.get(category_fund_metrics_key(category), {})

    if full_df.empty:
        st.warning("No data available for rankings."); st.stop()

    valid_n = sum(1 for m in fund_metrics.values() if m.get("is_valid"))
    st.caption(f"Rankings computed from {valid_n} of {len(fund_metrics)} funds with sufficient data.")

    # ── Helper: render one ranking table ──────────────────────────────────────
    def _fmt(val, kind):
        if val is None:
            return "N/A"
        try:
            v = float(val)
            if np.isnan(v): return "N/A"
            if kind == "pct":   return fmt_pct(v)
            if kind == "ratio": return fmt_ratio(v)
            if kind == "days":  return fmt_days(v)
            if kind == "num":   return f"{v:.2f}%"
        except Exception:
            return "N/A"
        return str(val)

    def _ranking_table(metric_key, label, kind, ascending=False, key_suffix=""):
        """
        Render a ranked table for `metric_key`.
        key_suffix must be unique whenever the same metric_key appears more
        than once on the page (best vs worst, or repeated across tabs).
        """
        if metric_key not in full_df.columns:
            st.caption(f"_{label} — insufficient data_"); return

        col = pd.to_numeric(full_df[metric_key], errors="coerce").dropna()
        if col.empty:
            st.caption(f"_{label} — no valid values_"); return

        sorted_df = full_df.sort_values(metric_key, ascending=ascending).head(top_n)
        q_col     = f"{metric_key}_quartile"

        rows = []
        for rank, (fund_name, row) in enumerate(sorted_df.iterrows(), start=1):
            val = row.get(metric_key)
            q   = row.get(q_col, "N/A") if q_col in sorted_df.columns else "N/A"
            rows.append({
                "Rank": rank, "Fund": fund_name,
                label: _fmt(val, kind), "Quartile": str(q),
            })

        df_out = pd.DataFrame(rows)
        st.dataframe(
            df_out.style.map(style_quartile, subset=["Quartile"]),
            width="stretch", hide_index=True,
            height=min(450, 42 + 36 * len(df_out)),
        )
        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"↓Download {label} Ranking (CSV)",
            data=csv,
            file_name=f"{category.replace(' ','_')}_{metric_key}_ranking.csv",
            mime="text/csv",
            key=f"dl_{metric_key}{key_suffix}",
        )

    # ── TAB SPECIFICATION ─────────────────────────────────────────────────────
    #
    # Every ranking table on this page is one entry here. Previously the same
    # markdown-header + _ranking_table() pair was written out 43 times across
    # 250 lines, which is how two tabs ended up rendering the SAME table:
    # momentum_3m and momentum_6m appeared under both "Absolute Returns" and
    # "Momentum", and the key_suffix machinery existed purely to stop the two
    # copies colliding on Streamlit widget keys.
    #
    # As tuples the duplication is visible at a glance, and adding a metric is
    # one line rather than a copy-paste.
    #
    # Tuple: (metric_key, column label, format, ascending, heading)
    #   ascending=True  → smallest first (for "lower is better" metrics)
    #
    # 11 tabs collapsed to 6. The old split was over-partitioned: Consistency,
    # Stability and Persistence are three English synonyms, and two pairs of
    # tabs shared an emoji so the strip was unreadable.

    RANK_SPEC = {
        "Returns": {
            "chart": lambda: plot_momentum_heatmap(full_df),
            "caption": (
                "Point-in-time and annualised returns. Note that a 12-month "
                "trailing return and a 1Y CAGR are arithmetically the same "
                "number, so only 1Y CAGR is shown."
            ),
            "groups": [
                [("cagr_1y",        "1Y CAGR",       "pct",  False, "Top — 1Y CAGR"),
                 ("cagr_3y",        "3Y CAGR",       "pct",  False, "Top — 3Y CAGR")],
                [("cagr_5y",        "5Y CAGR",       "pct",  False, "Top — 5Y CAGR"),
                 ("cagr_inception", "Inception CAGR","pct",  False, "Top — Since Inception")],
                [("momentum_1m",    "1M Return",     "pct",  False, "Top — 1 Month"),
                 ("momentum_3m",    "3M Return",     "pct",  False, "Top — 3 Months")],
                [("momentum_6m",    "6M Return",     "pct",  False, "Top — 6 Months"),
                 ("momentum_6m",    "6M Return",     "pct",  True,  "Bottom — 6 Months")],
            ],
        },
        "Risk": {
            "caption": (
                "Volatility and drawdown, plus the three risk-adjusted ratios. "
                "**Sharpe** divides excess return by total volatility; "
                "**Sortino** penalises only downside moves, so it reads higher "
                "for funds whose swings are mostly upward; **Calmar** divides "
                "by the worst drawdown instead of volatility, which rewards "
                "funds that avoid deep falls rather than merely steady ones. "
                "They usually agree — where they disagree is the interesting case."
            ),
            "groups": [
                [("sharpe",  "Sharpe",  "ratio", False, "Top — Sharpe Ratio"),
                 ("sortino", "Sortino", "ratio", False, "Top — Sortino Ratio"),
                 ("calmar",  "Calmar",  "ratio", False, "Top — Calmar Ratio")],
                [("annualized_volatility", "Ann. Vol", "pct", True,
                  "Lowest — Annualised Volatility"),
                 ("downside_volatility",   "Downside Vol", "pct", True,
                  "Lowest — Downside Volatility"),
                 ("max_drawdown", "Max DD", "pct", False,
                  "Smallest — Max Drawdown")],
            ],
        },
        "Rolling & Hit Rate": {
            "caption": (
                "How consistent the outcome has been, and how often the fund "
                "was positive. Median rolling return is shown rather than the "
                "average — for a skewed return distribution the median is the "
                "more honest central estimate."
            ),
            "groups": [
                [("median_rolling_1y", "Median 1Y", "pct", False,
                  "Top — Median 1Y Rolling"),
                 ("median_rolling_3y", "Median 3Y", "pct", False,
                  "Top — Median 3Y Rolling")],
                [("worst_rolling_1y",  "Worst 1Y",  "pct", False,
                  "Best of the worst — 1Y Rolling"),
                 ("std_rolling_1y",    "Std 1Y",    "pct", True,
                  "Least timing-dependent — 1Y Rolling")],
                [("pct_positive_rolling_1y", "% Positive 1Y", "pct", False,
                  "% of positive 1Y windows"),
                 ("pct_positive_rolling_3y", "% Positive 3Y", "pct", False,
                  "% of positive 3Y windows")],
                [("win_rate",      "Win Rate",  "pct", False, "Top — Monthly Win Rate"),
                 ("positive_freq", "Pos. Days", "pct", False, "Top — Positive Day Frequency")],
            ],
        },
        "Alpha & Factors": {
            "chart": lambda: plot_capture_scatter(full_df, category),
            "chart2": lambda: plot_bull_bear_alpha(fund_metrics),
            "caption": (
                "Everything measured against the fund's SEBI category "
                "benchmark, plus the 6-factor decomposition. Jensen's Alpha "
                "asks whether the fund beat its benchmark given its beta; "
                "6-Factor Alpha asks the harder question — whether anything "
                "is left once size, value, momentum, quality and low-volatility "
                "tilts are also paid for."
            ),
            "gate": "jensens_alpha",
            "groups": [
                [("jensens_alpha",     "Jensen's Alpha", "pct",   False, "Top — Jensen's Alpha"),
                 ("information_ratio", "Info Ratio",     "ratio", False, "Top — Information Ratio"),
                 ("capture_ratio",     "Capture",        "ratio", False, "Top — Capture Ratio")],
                [("down_capture",      "Down-Capture %", "num",   True,  "Lowest — Down-Capture"),
                 ("alpha_persistence", "Persistence",    "pct",   False, "Top — Alpha Persistence"),
                 ("bear_alpha",        "Bear Alpha",     "pct",   False, "Top — Bear-Market Alpha")],
                [("alpha_6f",          "6F Alpha",       "pct",   False, "Top — 6-Factor Alpha"),
                 ("alpha_6f_tstat",    "Alpha t-Stat",   "ratio", False, "Most significant alpha"),
                 ("r_squared_6f",      "6F R²",          "ratio", False, "Highest — 6-Factor R²")],
                [("beta_smb", "SMB β", "ratio", False, "Size tilt (SMB)"),
                 ("beta_hml", "HML β", "ratio", False, "Value tilt (HML)"),
                 ("beta_wml", "WML β", "ratio", False, "Momentum tilt (WML)")],
                [("beta_qmj",        "QMJ β",    "ratio", False, "Quality tilt (QMJ)"),
                 ("beta_bab",        "BAB β",    "ratio", False, "Low-Vol tilt (BAB)"),
                 ("beta_market_6f",  "Market β", "ratio", False, "Market beta")],
            ],
        },
        "vs Broad Market": {
            "caption": (
                "The same funds measured against Nifty 500 TRI instead of "
                "their category benchmark — did the fund beat simply owning "
                "the market? Where a fund's rank differs from the Alpha tab, "
                "its category benchmark is flattering or penalising it. For "
                "Flexi Cap, Multi Cap, ELSS, Value, Contra and Focused the two "
                "benchmarks are the same index, so the tables will match."
            ),
            "gate": "jensens_alpha_mkt",
            "groups": [
                [("jensens_alpha_mkt",     "Alpha vs Mkt", "pct",   False,
                  "Top — Jensen's Alpha vs Market"),
                 ("information_ratio_mkt", "IR vs Mkt",    "ratio", False,
                  "Top — Information Ratio vs Market")],
                [("capture_ratio_mkt",     "Capture vs Mkt", "ratio", False,
                  "Top — Capture Ratio vs Market"),
                 ("beta_mkt",              "Beta vs Mkt",    "ratio", False,
                  "Highest — Market Beta")],
            ],
        },
    }

    def _render_spec(spec):
        """Render one tab from its specification."""
        if spec.get("caption"):
            st.caption(spec["caption"])

        # Charts first — they give the shape of the category before the
        # reader works through the ranked tables.
        for chart_key in ("chart", "chart2"):
            builder = spec.get(chart_key)
            if builder is None:
                continue
            try:
                chart(builder())
            except Exception as exc:
                st.caption(f"_Chart unavailable: {exc}_")

        gate = spec.get("gate")
        if gate and not (gate in full_df.columns and full_df[gate].notna().any()):
            st.info(
                "These rankings need benchmark-relative metrics, which were "
                "not available when this category was computed. Click "
                "**Compute Rankings** again to refresh.",
            )
            return

        for group in spec["groups"]:
            st.divider()
            cols = st.columns(len(group), gap="large")
            for col, (key, label, kind, asc, heading) in zip(cols, group):
                with col:
                    st.markdown(f"**{heading}**")
                    _ranking_table(
                        key, label, kind, ascending=asc,
                        key_suffix=f"_{heading[:12]}",
                    )

    # ── TABS ──────────────────────────────────────────────────────────────────
    _tab_names = list(RANK_SPEC.keys()) + ["Quartile View"]
    _tabs = st.tabs(_tab_names)

    for _tab, _name in zip(_tabs[:-1], _tab_names[:-1]):
        with _tab:
            st.subheader(_name)
            _render_spec(RANK_SPEC[_name])

    tab_quartile = _tabs[-1]

    # ── Tab 11: Quartile View ─────────────────────────────────────────────────
    # Replaces pages/2_Category_Explorer.py (removed in Phase D).
    # Shows all funds × all metrics with Q1–Q4 colour coding, plus
    # scatter plots and a downloadable key-metrics table.
    with tab_quartile:
        st.subheader("Quartile View — All Funds · All Metrics")
        st.caption(
            "**Q1** = Best 25% in category (green)  |  "
            "**Q2** = Next 25%  |  "
            "**Q3** = Next 25%  |  "
            "**Q4** = Worst 25% (red)  |  "
            "**N/A** = Insufficient history for this metric"
        )

        # ── Scatter plots ─────────────────────────────────────────────────────
        sc1, sc2 = st.columns(2, gap="medium")
        with sc1:
            try:
                chart(
                    plot_risk_return_scatter(full_df),
                )
            except Exception as e:
                st.caption(f"_Risk-Return scatter unavailable: {e}_")
        with sc2:
            try:
                chart(
                    plot_vol_cagr_scatter(full_df),
                )
            except Exception as e:
                st.caption(f"_Vol-CAGR scatter unavailable: {e}_")

        st.divider()

        # ── Quartile heatmap — the primary "show all quartiles" view ──────────
        st.subheader("Q1–Q4 Heatmap — Every Fund, Every Metric")
        try:
            heatmap_height = max(420, 100 + 38 * len(full_df))
            chart(
                plot_quartile_heatmap(full_df, height=heatmap_height),
            )
        except Exception as e:
            st.warning(f"Quartile heatmap unavailable: {e}")

        st.divider()

        # ── Key metrics table with CSV download ───────────────────────────────
        st.subheader("Key Metrics Table")
        st.caption("Summary of the most actionable metrics for each fund in the category.")

        _KEY_COLS = {
            "cagr_1y":               "1Y CAGR",
            "cagr_3y":               "3Y CAGR",
            "cagr_5y":               "5Y CAGR",
            "annualized_volatility": "Ann. Vol",
            "max_drawdown":          "Max DD",
            "sharpe":                "Sharpe",
            "sortino":               "Sortino",
            "jensens_alpha":         "Alpha",
            "capture_ratio":         "Capture Ratio",
            "win_rate":              "Win Rate",
        }

        rows = []
        for fund_name, m in fund_metrics.items():
            if not m.get("is_valid"):
                continue
            row = {"Fund": fund_name}
            for key, col_label in _KEY_COLS.items():
                val = m.get(key)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    row[col_label] = "N/A"
                elif key in {"sharpe", "sortino", "capture_ratio"}:
                    row[col_label] = f"{val:.3f}"
                else:
                    row[col_label] = fmt_pct(val)
            rows.append(row)

        if rows:
            table_df = pd.DataFrame(rows).set_index("Fund")
            st.dataframe(
                table_df,
                width="stretch",
                height=min(600, 42 + 35 * len(table_df)),
            )
            csv = table_df.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button(
                "↓Download Key Metrics (CSV)",
                data=csv,
                file_name=f"{category.replace(' ','_')}_key_metrics.csv",
                mime="text/csv",
                key="dl_quartile_view_metrics",
            )
        else:
            st.warning("No valid fund data to display.")
