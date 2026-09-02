"""
visualizations/alpha_charts.py
================================
Alpha generation charts — benchmark-relative visualizations.

plot_fund_vs_benchmark: Updated to show % returns from period start (not rebased 100).
"""

import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional
from visualizations._theme import (
    base_layout, empty_figure, get_color,
    UP_COLOR, DOWN_COLOR, NEUTRAL_COLOR,
)
from utils import theme as T
from visualizations._theme import last_value_badges


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — FUND vs BENCHMARK  (% return from period start)
# ─────────────────────────────────────────────────────────────────────────────

PERIOD_MAP = {
    "1M": 1, "3M": 3, "6M": 6,
    "1Y": 12, "3Y": 36, "5Y": 60, "All": None,
}


def plot_fund_vs_benchmark(
    fund_nav:       pd.Series,
    benchmark_nav:  pd.Series,
    fund_name:      str,
    benchmark_name: str,
    period_label:   str = "All",
    height:         int = 440,
    market_nav:     Optional[pd.Series] = None,
    market_name:    str = "",
) -> go.Figure:
    """
    Fund vs its category benchmark, optionally with the broad market as a
    third reference line.

    All lines are rebased to 0% at the start of the selected period, so
    vertical distance reads directly as relative performance. The shaded
    area is fund-minus-category-benchmark — the SEBI comparison stays the
    visual anchor; the market is context, drawn as a muted dashed line.

    Args:
        fund_nav:       Clean NAV series for the fund
        benchmark_nav:  Clean NAV series for the category benchmark
        fund_name:      Fund display name
        benchmark_name: Category benchmark display name (e.g. "Nifty 100 TRI")
        period_label:   "1M","3M","6M","1Y","3Y","5Y","All"
        height:         Chart height in pixels
        market_nav:     Optional broad-market NAV series (Nifty 500 TRI).
                        Pass None when the category benchmark IS the market
                        (Flexi/Multi/ELSS/Value/Contra/Focused) — plotting it
                        twice would just draw the same line over itself.
        market_name:    Market display name

    Returns:
        go.Figure — all lines start at 0%, shaded gap vs category benchmark.
    """
    if fund_nav is None or benchmark_nav is None:
        return empty_figure("Fund or benchmark NAV not available")

    # Common date range
    common = fund_nav.index.intersection(benchmark_nav.index)
    if len(common) < 10:
        return empty_figure("Insufficient overlapping history")

    f = fund_nav.reindex(common)
    b = benchmark_nav.reindex(common)

    # Slice to period
    period_months = PERIOD_MAP.get(period_label)
    if period_months is not None:
        end   = f.index[-1]
        start = end - pd.DateOffset(months=period_months)
        f = f[f.index >= start]
        b = b[b.index >= start]
        if len(f) < 5:
            f = fund_nav.reindex(common)
            b = benchmark_nav.reindex(common)

    # Common start after slicing
    cs = max(f.index[0], b.index[0])
    f  = f[f.index >= cs]
    b  = b[b.index >= cs]

    # % return from common start
    f_pct = (f / f.iloc[0] - 1) * 100
    b_pct = (b / b.iloc[0] - 1) * 100

    # Broad market, rebased onto the window the fund/benchmark pair already
    # settled on. Reindexing (rather than re-intersecting) means a gap in the
    # market series can never shorten the primary comparison.
    m_pct = None
    if market_nav is not None and market_name != benchmark_name:
        m = market_nav.reindex(f_pct.index).dropna()
        if len(m) >= 5:
            m_pct = (m / m.iloc[0] - 1) * 100

    fig = go.Figure()

    # Benchmark line
    fig.add_trace(go.Scatter(
        x=b_pct.index, y=b_pct.values,
        name=benchmark_name,
        mode="lines",
        line=dict(color=T.WARN, width=1.8, dash="dot"),
        hovertemplate=(
            f"<b>{benchmark_name}</b><br>"
            "Date: %{x|%d %b %Y}<br>"
            "Return: %{y:+.2f}%<extra></extra>"
        ),
    ))

    # Fund line — filled to benchmark
    fig.add_trace(go.Scatter(
        x=f_pct.index, y=f_pct.values,
        name=fund_name,
        mode="lines",
        line=dict(color=T.DATA_PRIMARY, width=2),
        fill="tonexty",
        fillcolor=T.rgba(T.DATA_PRIMARY, 0.08),
        hovertemplate=(
            f"<b>{fund_name}</b><br>"
            "Date: %{x|%d %b %Y}<br>"
            "Return: %{y:+.2f}%<extra></extra>"
        ),
    ))

    # Broad market — added LAST so the fund's fill="tonexty" still anchors to
    # the category benchmark trace immediately before it. Muted and dashed:
    # this is context, not the primary comparison.
    if m_pct is not None:
        fig.add_trace(go.Scatter(
            x=m_pct.index, y=m_pct.values,
            name=market_name,
            mode="lines",
            line=dict(color=NEUTRAL_COLOR, width=1.5, dash="dash"),
            hovertemplate=(
                f"<b>{market_name}</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "Return: %{y:+.2f}%<extra></extra>"
            ),
        ))

    # Zero baseline
    fig.add_hline(
        y=0, line_dash="dot",
        line_color="rgba(255,255,255,0.30)", line_width=1.2,
        annotation_text="0%", annotation_position="right",
        annotation_font_size=10,
        annotation_font_color="rgba(200,200,200,0.6)",
    )

    # Outperformance annotation — one line per yardstick, so "+0.7%" is never
    # ambiguous about which benchmark it refers to.
    def _tag(value: float, label: str) -> str:
        col  = UP_COLOR if value >= 0 else DOWN_COLOR
        sign = "+" if value >= 0 else ""
        return f"<span style='color:{col}'><b>{sign}{value:.1f}% vs {label}</b></span>"

    final_diff = float(f_pct.iloc[-1] - b_pct.reindex(f_pct.index).iloc[-1])
    lines = [_tag(final_diff, "category")] if m_pct is not None else \
            [_tag(final_diff, "benchmark")]

    if m_pct is not None:
        final_mkt = float(f_pct.reindex(m_pct.index).iloc[-1] - m_pct.iloc[-1])
        lines.append(_tag(final_mkt, "market"))

    fig.add_annotation(
        text="<br>".join(lines),
        xref="paper", yref="paper", x=0.99, y=0.99,
        showarrow=False, xanchor="right", align="right",
        font=dict(size=12),
        bgcolor=T.rgba(T.PANEL_HI, 0.85), borderpad=4,
    )

    fig.update_layout(base_layout(
        title=f"Fund vs Benchmark — {period_label}",
        x_title="Date",
        y_title=f"Return from {cs.strftime('%d %b %Y')} (%)",
        height=height, hovermode="x unified",
    ))
    fig.update_yaxes(ticksuffix="%", zeroline=True,
                     zerolinecolor="rgba(255,255,255,0.30)", zerolinewidth=1.2)

    # Stamp each line's final value on the right axis in its own colour.
    last_value_badges(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — ROLLING ALPHA
# ─────────────────────────────────────────────────────────────────────────────

def plot_rolling_alpha(
    rolling_alpha_dict: Dict[str, Optional[pd.Series]],
    window_label:       str = "1-Year",
    height:             int = 400,
) -> go.Figure:
    valid = {k: v for k, v in rolling_alpha_dict.items()
             if v is not None and len(v) > 0}
    if not valid:
        return empty_figure("Rolling alpha requires 2+ years of overlapping history")

    fig = go.Figure()
    for i, (name, series) in enumerate(valid.items()):
        pct = (series * 100).dropna()
        fig.add_trace(go.Scatter(
            x=pct.index, y=pct.values, name=name,
            mode="lines", line=dict(color=get_color(i), width=1.8),
            hovertemplate=(
                f"<b>{name}</b><br>Date: %{{x|%d %b %Y}}<br>"
                f"{window_label} Rolling Alpha: %{{y:.2f}}%<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line_dash="dash",
                  line_color=T.rgba(T.WARN, 0.6), line_width=1.5,
                  annotation_text="0% Alpha", annotation_position="right",
                  annotation_font_color=NEUTRAL_COLOR, annotation_font_size=10)

    fig.update_layout(base_layout(
        title=f"{window_label} Rolling Jensen's Alpha",
        x_title="Date", y_title="Annualized Alpha (%)",
        height=height, hovermode="x unified",
    ))
    fig.update_yaxes(ticksuffix="%")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — CAPTURE RATIO SCATTER
# ─────────────────────────────────────────────────────────────────────────────

def plot_capture_scatter(
    full_df:  pd.DataFrame,
    category: str,
    height:   int = 500,
) -> go.Figure:
    required = ["up_capture", "down_capture"]
    for col in required:
        if col not in full_df.columns:
            return empty_figure("Capture ratio data not available")

    plot_df = full_df[required].dropna()
    if plot_df.empty:
        return empty_figure("No capture ratio data")

    x_vals = plot_df["down_capture"]
    y_vals = plot_df["up_capture"]
    names  = [n[:38] + "…" if len(n) > 38 else n for n in plot_df.index]

    from utils.constants import QUARTILE_COLORS
    cap_q = "capture_ratio_quartile"
    colors = [
        QUARTILE_COLORS.get(str(full_df.loc[n, cap_q]), T.DATA_PRIMARY)
        if cap_q in full_df.columns and n in full_df.index else T.DATA_PRIMARY
        for n in plot_df.index
    ]

    cap_ratios = [
        full_df.loc[n, "capture_ratio"] if "capture_ratio" in full_df.columns
        and n in full_df.index else None
        for n in plot_df.index
    ]

    hover = [
        f"<b>{n}</b><br>Up-Capture: {float(y_vals.iloc[i]):.1f}%<br>"
        f"Down-Capture: {float(x_vals.iloc[i]):.1f}%<br>"
        f"Capture Ratio: {f'{cap_ratios[i]:.3f}' if cap_ratios[i] else 'N/A'}"
        "<extra></extra>"
        for i, n in enumerate(plot_df.index)
    ]

    fig = go.Figure()

    x_mid, y_mid = 100.0, 100.0
    x_min = max(float(x_vals.min()) * 0.92, 40)
    x_max = float(x_vals.max()) * 1.08
    y_min = max(float(y_vals.min()) * 0.92, 40)
    y_max = float(y_vals.max()) * 1.08

    qs = dict(type="rect", xref="x", yref="y", line=dict(width=0))
    fig.add_shape(**qs, x0=x_min, x1=x_mid, y0=y_mid, y1=y_max, fillcolor=T.rgba(T.UP, 0.06))
    fig.add_shape(**qs, x0=x_mid, x1=x_max, y0=y_min, y1=y_mid, fillcolor=T.rgba(T.DOWN, 0.06))
    fig.add_shape(**qs, x0=x_min, x1=x_mid, y0=y_min, y1=y_mid, fillcolor="rgba(255,255,255,0.02)")
    fig.add_shape(**qs, x0=x_mid, x1=x_max, y0=y_mid, y1=y_max, fillcolor="rgba(255,255,255,0.02)")

    fig.add_hline(y=100, line_dash="dot", line_color="rgba(255,255,255,0.15)", line_width=1)
    fig.add_vline(x=100, line_dash="dot", line_color="rgba(255,255,255,0.15)", line_width=1)

    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode="markers+text",
        marker=dict(size=11, color=colors,
                    line=dict(color="rgba(255,255,255,0.3)", width=1), opacity=0.88),
        text=names, textposition="top center",
        textfont=dict(size=8, color="#C0C0C0"),
        hovertemplate=hover, showlegend=False,
    ))

    _ql = dict(showarrow=False, font=dict(size=8, color="rgba(200,200,200,0.25)"))
    fig.add_annotation(text="IDEAL ✓", x=x_min*1.02, y=y_max*0.98, xanchor="left", **_ql)
    fig.add_annotation(text="WORST ✗", x=x_max*0.98, y=y_min*1.02, xanchor="right", **_ql)

    fig.update_layout(base_layout(
        title=f"Capture Ratio Map — {category}",
        x_title="Down-Capture (%) — Lower = better →",
        y_title="Up-Capture (%) — Higher = better ↑",
        height=height, hovermode="closest",
    ))
    fig.update_xaxes(ticksuffix="%")
    fig.update_yaxes(ticksuffix="%")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — ALPHA COMPARISON BAR
# ─────────────────────────────────────────────────────────────────────────────


