"""
pages/6_Data_Quality.py
========================
Data Quality — Coverage Report

Shows which funds have sufficient NAV history to compute each metric.
Helps users understand WHY certain metrics show N/A for specific funds.

Two views:
  1. Category Summary  — how many funds in the category have each metric
  2. Fund-Level Report — per-fund coverage for a selected category

No analytics computation happens here — only NAV history is loaded
(one API call per fund) to check date ranges.
"""

import streamlit as st
import pandas as pd
import numpy as np

from data.fund_loader   import get_all_categorized_schemes, get_nav_history
from data.nav_processor import process_nav
from utils.constants    import (
    MIN_DAYS, COVERAGE_LABELS,
)
from utils.session      import dq_scan_key, dq_reports_key
from utils.validators   import build_quality_report
from utils.formatters   import fmt_date
from utils.ui          import (
    sidebar_header, category_selector, plan_selector, tri_status,
    render_refresh_button,
    export_button,
)
from utils import theme as T

st.set_page_config(
    page_title = "Data Quality — MF Analytics",
    page_icon  = "🔬",
    layout     = "wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    sidebar_header()

    category = category_selector()

    plan_type = plan_selector()

    st.divider()
    render_refresh_button()
    tri_status()

    st.divider()
    st.markdown("**Minimum History Required**")
    # Derived from MIN_DAYS so this can never drift from the thresholds the
    # coverage check actually applies. It used to be a hardcoded list.
    st.markdown("\n".join(
        f"- {COVERAGE_LABELS.get(k, k)}: {v/365:.0f}y" if v >= 365
        else f"- {COVERAGE_LABELS.get(k, k)}: {v}d"
        for k, v in sorted(MIN_DAYS.items(), key=lambda kv: -kv[1])
    ))

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("Data Quality Report")
st.caption(
    "Shows NAV history coverage for each fund — which metrics can be computed "
    "and which require longer history."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD FUND LIST
# ─────────────────────────────────────────────────────────────────────────────

plan_type = st.session_state.get("plan_type", "Direct")

with st.spinner(f"Loading {category} fund list…"):
    all_cat   = get_all_categorized_schemes(plan_type=plan_type)
    fund_list = all_cat.get(category, [])

if not fund_list:
    st.warning("No funds found for this category.")
    st.stop()

st.subheader(f"{category} — {len(fund_list)} {plan_type} Funds")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD NAV SUMMARIES (lighter than full analytics)
# ─────────────────────────────────────────────────────────────────────────────

scan_key = dq_scan_key(category)

run_scan = st.button(
    f"Scan NAV History for {len(fund_list)} Funds",
    type="primary", width="stretch",
)

if run_scan or st.session_state.get(scan_key):

    if not st.session_state.get(scan_key):
        reports = {}
        progress = st.progress(0, text="Scanning NAV history…")

        for i, fund in enumerate(fund_list):
            progress.progress(
                (i + 1) / len(fund_list),
                text=f"Scanning: {fund['name'][:55]} ({i+1}/{len(fund_list)})",
            )
            nav_df = get_nav_history(fund["code"])
            nav    = process_nav(nav_df) if nav_df is not None else None
            report = build_quality_report(fund["name"], nav)
            reports[fund["name"]] = report

        progress.empty()
        st.session_state[scan_key]             = True
        st.session_state[dq_reports_key(category)] = reports
        st.success(f"Scanned {len(reports)} funds.")

    reports = st.session_state.get(dq_reports_key(category), {})

    if not reports:
        st.warning("No report data available.")
        st.stop()

    # ─────────────────────────────────────────────────────────────────────────
    # KPI SUMMARY ROW
    # ─────────────────────────────────────────────────────────────────────────

    total         = len(reports)
    has_3y        = sum(1 for r in reports.values() if r.get("coverage", {}).get("3y_cagr"))
    has_5y        = sum(1 for r in reports.values() if r.get("coverage", {}).get("5y_cagr"))
    has_rolling3y = sum(1 for r in reports.values() if r.get("coverage", {}).get("rolling_3y"))
    avg_history   = np.mean([r.get("history_years", 0) for r in reports.values() if r.get("history_years", 0) > 0])

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Funds Scanned",    total)
    k2.metric("Have 3Y+ History",       f"{has_3y} / {total}")
    k3.metric("Have 5Y+ History",       f"{has_5y} / {total}")
    k4.metric("Have 4Y+ (3Y Rolling)",  f"{has_rolling3y} / {total}")
    k5.metric("Avg History (years)",    f"{avg_history:.1f}")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # TABS
    # ─────────────────────────────────────────────────────────────────────────

    tab1, tab2, tab3 = st.tabs([
        "Fund History Summary",
        "Metric Coverage Matrix",
        "Warnings",
    ])

    # ── TAB 1: Fund History Summary ───────────────────────────────────────────
    with tab1:
        st.subheader("NAV History per Fund")
        st.caption("Sorted by history length — longest history at top.")

        rows = []
        for fund_name, r in reports.items():
            years  = r.get("history_years", 0)
            points = r.get("data_points", 0)
            miss   = r.get("missing_pct", 0)
            start  = fmt_date(r.get("start_date"))
            end    = fmt_date(r.get("end_date"))

            # History badge
            if years >= 10:   badge = "10Y+"
            elif years >= 7:  badge = "7Y+"
            elif years >= 5:  badge = "5Y+"
            elif years >= 3:  badge = "3Y+"
            elif years >= 1:  badge = "1Y+"
            else:             badge = "<1Y"

            rows.append({
                "Fund Name":      fund_name,
                "History":        badge,
                "Years":          round(years, 1),
                "Data Points":    f"{points:,}",
                "Missing %":      f"{miss:.1f}%",
                "Start Date":     start,
                "End Date":       end,
            })

        summary_df = (
            pd.DataFrame(rows)
            .sort_values("Years", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(summary_df, width="stretch", hide_index=True,
                     height=min(600, 42 + 35 * len(summary_df)))

        csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "↓Download Summary (CSV)",
            data=csv,
            file_name=f"{category.replace(' ','_')}_data_quality.csv",
            mime="text/csv",
        )

    # ── TAB 2: Metric Coverage Matrix ────────────────────────────────────────
    with tab2:
        st.subheader("Metric Coverage Matrix")
        st.caption("✓ = sufficient history  |  · = insufficient history for this metric")

        # The grid is a curated subset: coverage is tracked for all 18 metrics
        # in COVERAGE_LABELS, but an 18-column ✅/❌ table beside a full fund
        # name is unreadable. The complete list is shown underneath instead.
        KEY_METRICS = [
            "1y_cagr", "3y_cagr", "5y_cagr", "inception_cagr",
            "sharpe", "sortino", "calmar",
            "rolling_1y", "rolling_3y",
            "volatility", "max_drawdown",
        ]
        # Shared vocabulary — see utils/constants.COVERAGE_LABELS. This page
        # used to keep its own private dict, which is how "Max Drawdown" here
        # and "Max DD" on Fund Analytics drifted apart.
        METRIC_DISPLAY = COVERAGE_LABELS

        matrix_rows = {}
        for fund_name, r in reports.items():
            cov = r.get("coverage", {})
            matrix_rows[fund_name] = {
                METRIC_DISPLAY[m]: T.MARK_YES if cov.get(m, False) else T.MARK_NO
                for m in KEY_METRICS
            }

        matrix_df = (
            pd.DataFrame.from_dict(matrix_rows, orient="index")
            .reset_index()
            .rename(columns={"index": "Fund Name"})
        )

        # Sort: most coverage at top.
        #
        # These column names MUST come from KEY_METRICS, which is what built
        # the frame. Taking METRIC_DISPLAY.values() instead asks for all 18
        # labels from a frame that only has 11 columns, and pandas raises
        # KeyError on the 7 that were never added — the page crashed outright
        # for any category with funds in it.
        metric_cols = [METRIC_DISPLAY[m] for m in KEY_METRICS]
        matrix_df["_score"] = matrix_df[metric_cols].apply(
            lambda row: (row == "✓").sum(), axis=1
        )
        matrix_df = matrix_df.sort_values("_score", ascending=False).drop(columns="_score")

        st.dataframe(matrix_df, width="stretch", hide_index=True,
                     height=min(600, 42 + 35 * len(matrix_df)))
        export_button(matrix_df, f"{category.replace(' ', '_')}_coverage_matrix.csv",
                      label="↓ Coverage matrix CSV", key="dl_dq_matrix")

        # Coverage summary per metric — the FULL set, including the seven the
        # grid above leaves out. The caption says so, because a metric present
        # in one table and absent from the other otherwise reads as a bug.
        st.markdown("**Coverage % per Metric:**")
        st.caption(
            f"All {len(METRIC_DISPLAY)} tracked metrics — the grid above shows "
            f"the {len(KEY_METRICS)} most used."
        )
        cov_summary = {}
        for m, label in METRIC_DISPLAY.items():
            n_have = sum(
                1 for r in reports.values()
                if r.get("coverage", {}).get(m, False)
            )
            cov_summary[label] = f"{n_have}/{total}  ({n_have/total*100:.0f}%)"

        cov_df = pd.DataFrame(
            [{"Metric": k, "Funds with Sufficient Data": v}
             for k, v in cov_summary.items()]
        )
        st.dataframe(cov_df, width="stretch", hide_index=True)
        export_button(cov_df, f"{category.replace(' ', '_')}_coverage_summary.csv",
                      label="↓ Coverage summary CSV", key="dl_dq_cov")

    # ── TAB 3: Warnings ───────────────────────────────────────────────────────
    with tab3:
        st.subheader("Data Warnings")
        st.caption("Funds with data quality issues flagged during scanning.")

        warned_funds = {
            name: r["warnings"]
            for name, r in reports.items()
            if r.get("warnings")
        }

        if not warned_funds:
            st.success("No data quality warnings found for any fund in this category.")
        else:
            st.info(f"{len(warned_funds)} of {total} funds have warnings.")
            for fund_name, warnings in warned_funds.items():
                with st.expander(f"{fund_name}"):
                    for w in warnings:
                        st.write(f"• {w}")
