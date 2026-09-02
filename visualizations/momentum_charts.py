"""
visualizations/momentum_charts.py
===================================
Phase B visualization charts.

Chart 1 — Momentum Bar Chart
    Side-by-side 3M / 6M / 12M momentum bars for one or more funds.
    Immediately shows which funds have strong recent momentum.

Chart 2 — Bull / Bear Alpha Chart
    Grouped bars showing bull alpha vs bear alpha per fund.
    The ideal fund has positive bars in BOTH regimes.

Chart 3 — Alpha Persistence Timeline
    Area chart of the rolling alpha series coloured green when positive,
    red when negative — visual persistence check at a glance.

Chart 4 — Momentum Heatmap
    Category-wide grid: funds × momentum periods, colour-coded by strength.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional
from visualizations._theme import (
    base_layout, empty_figure, get_color,
    UP_COLOR, DOWN_COLOR,
)
from utils import theme as T


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — MOMENTUM BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_momentum_bars(
    fund_metrics_dict: Dict[str, Dict],
    height:            int = 400,
) -> go.Figure:
    """
    Grouped bar chart showing 3M / 6M / 12M momentum for each fund.

    Green bars = positive momentum (fund rose over the period).
    Red bars   = negative momentum (fund fell over the period).

    Args:
        fund_metrics_dict: {fund_name: metrics_dict}
                           Must contain momentum_3m, momentum_6m and cagr_1y.
                           The 12-month rung reads cagr_1y: a 12-month
                           trailing return and a 1Y CAGR are the same
                           number, so momentum_12m was retired.
        height:            Chart height in pixels.

    Returns:
        go.Figure with grouped momentum bars.
    """
    valid = {
        k: v for k, v in fund_metrics_dict.items()
        if v.get("is_valid") and any(
            v.get(m) is not None
            for m in ["momentum_3m", "momentum_6m", "cagr_1y"]
        )
    }

    if not valid:
        return empty_figure("No momentum data available")

    WINDOWS = [
        ("momentum_3m",  "3 Month"),
        ("momentum_6m",  "6 Month"),
        ("cagr_1y",      "1 Year"),
    ]

    fig = go.Figure()
    names = [n[:35] + "…" if len(n) > 35 else n for n in valid.keys()]

    for i, (key, label) in enumerate(WINDOWS):
        values = [
            (m.get(key) or 0) * 100
            for m in valid.values()
        ]
        bar_colors = [UP_COLOR if v >= 0 else DOWN_COLOR for v in values]

        fig.add_trace(go.Bar(
            name          = label,
            x             = names,
            y             = values,
            marker_color  = [get_color(i)] * len(values),
            opacity       = 0.82,
            hovertemplate = (
                "<b>%{x}</b><br>"
                f"{label} Momentum: %{{y:.2f}}%"
                "<extra></extra>"
            ),
        ))

    fig.add_hline(
        y=0, line_dash="dot",
        line_color="rgba(255,255,255,0.2)", line_width=1,
    )

    fig.update_layout(
        base_layout(
            title     = "Return Momentum — 3M / 6M / 12M",
            x_title   = "Fund",
            y_title   = "Return (%)",
            height    = height,
            hovermode = "x unified",
        ),
        barmode = "group",
    )
    fig.update_yaxes(ticksuffix="%")
    fig.update_xaxes(tickangle=-25)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — BULL / BEAR ALPHA
# ─────────────────────────────────────────────────────────────────────────────

def plot_bull_bear_alpha(
    fund_metrics_dict: Dict[str, Dict],
    height:            int = 420,
) -> go.Figure:
    """
    Grouped bar chart showing bull alpha and bear alpha side by side per fund.

    Ideal profile: both bars green (positive in both regimes).
    Dangerous profile: green bull bar, red bear bar (only looks good in rallies).

    Args:
        fund_metrics_dict: {fund_name: metrics_dict}
                           Must contain bull_alpha and bear_alpha.
        height:            Chart height.

    Returns:
        go.Figure with grouped bull/bear alpha bars.
    """
    valid = {
        k: v for k, v in fund_metrics_dict.items()
        if v.get("is_valid") and (
            v.get("bull_alpha") is not None or
            v.get("bear_alpha") is not None
        )
    }

    if not valid:
        return empty_figure("Bull/Bear alpha data not available")

    names      = [n[:35] + "…" if len(n) > 35 else n for n in valid.keys()]
    bull_vals  = [(v.get("bull_alpha") or 0) * 100 for v in valid.values()]
    bear_vals  = [(v.get("bear_alpha") or 0) * 100 for v in valid.values()]

    fig = go.Figure()

    # Bull alpha bars
    fig.add_trace(go.Bar(
        name          = "Bull Market Alpha",
        x             = names,
        y             = bull_vals,
        marker_color  = T.rgba(T.DATA_PRIMARY, 0.80),
        hovertemplate = "<b>%{x}</b><br>Bull Alpha: %{y:.2f}%<extra></extra>",
    ))

    # Bear alpha bars
    fig.add_trace(go.Bar(
        name          = "Bear Market Alpha",
        x             = names,
        y             = bear_vals,
        marker_color  = T.rgba(T.WARN, 0.80),
        hovertemplate = "<b>%{x}</b><br>Bear Alpha: %{y:.2f}%<extra></extra>",
    ))

    # Zero line
    fig.add_hline(
        y=0, line_dash="dot",
        line_color="rgba(255,255,255,0.25)", line_width=1.2,
    )

    fig.update_layout(
        base_layout(
            title     = "Bull vs Bear Market Alpha",
            x_title   = "Fund",
            y_title   = "Annualized Alpha (%)",
            height    = height,
            hovermode = "x unified",
        ),
        barmode = "group",
    )
    fig.update_yaxes(ticksuffix="%")
    fig.update_xaxes(tickangle=-25)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — ALPHA PERSISTENCE TIMELINE
# ─────────────────────────────────────────────────────────────────────────────

def plot_alpha_persistence_timeline(
    rolling_alpha: Optional[pd.Series],
    fund_name:     str = "",
    height:        int = 360,
) -> go.Figure:
    """
    Area chart of the rolling alpha series — green when positive, red when negative.

    Shows visually whether the fund's alpha is persistent (mostly green)
    or sporadic (alternating red and green patches).

    Args:
        rolling_alpha: pd.Series of annualized rolling alpha values
        fund_name:     Fund display name for the chart title
        height:        Chart height.

    Returns:
        go.Figure with colour-split area chart.
    """
    if rolling_alpha is None or len(rolling_alpha) == 0:
        return empty_figure("Rolling alpha data not available (requires 2+ years of benchmark history)")

    pct = (rolling_alpha * 100).dropna()
    if len(pct) < 10:
        return empty_figure("Insufficient rolling alpha data points")

    fig = go.Figure()

    # Positive alpha area — green
    pos = pct.clip(lower=0)
    fig.add_trace(go.Scatter(
        x=pct.index, y=pos.values,
        name="Positive Alpha",
        mode="lines",
        line=dict(color=UP_COLOR, width=0.5),
        fill="tozeroy",
        fillcolor=T.rgba(T.UP, 0.25),
        hovertemplate="Date: %{x|%d %b %Y}<br>Alpha: %{y:.2f}%<extra></extra>",
    ))

    # Negative alpha area — red
    neg = pct.clip(upper=0)
    fig.add_trace(go.Scatter(
        x=pct.index, y=neg.values,
        name="Negative Alpha",
        mode="lines",
        line=dict(color=DOWN_COLOR, width=0.5),
        fill="tozeroy",
        fillcolor=T.rgba(T.DOWN, 0.25),
        hovertemplate="Date: %{x|%d %b %Y}<br>Alpha: %{y:.2f}%<extra></extra>",
    ))

    # Full line on top for clarity
    fig.add_trace(go.Scatter(
        x=pct.index, y=pct.values,
        name="1Y Rolling Alpha",
        mode="lines",
        line=dict(color="#E0E0E0", width=1.5),
        hovertemplate="Date: %{x|%d %b %Y}<br>Rolling Alpha: %{y:.2f}%<extra></extra>",
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash",
                  line_color=T.rgba(T.WARN, 0.5), line_width=1.2)

    # Persistence annotation
    pct_positive = float((pct > 0).mean() * 100)
    fig.add_annotation(
        text=f"<b>Alpha positive {pct_positive:.0f}% of the time</b>",
        xref="paper", yref="paper",
        x=0.01, y=0.97,
        showarrow=False,
        font=dict(size=11, color=UP_COLOR if pct_positive >= 50 else DOWN_COLOR),
        bgcolor=T.rgba(T.PANEL_HI, 0.85),
        borderpad=4,
    )

    title = f"Rolling Alpha Persistence — {fund_name}" if fund_name else "Rolling Alpha Persistence"
    fig.update_layout(
        base_layout(
            title=title,
            x_title="Date",
            y_title="1-Year Rolling Alpha (%)",
            height=height,
            hovermode="x unified",
        )
    )
    fig.update_yaxes(ticksuffix="%")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — MOMENTUM HEATMAP (category-wide)
# ─────────────────────────────────────────────────────────────────────────────

def plot_momentum_heatmap(
    full_df: pd.DataFrame,
    height:  int = 500,
) -> go.Figure:
    """
    Heatmap showing momentum strength across all funds in a category.

    Rows = funds, Columns = 3M / 6M / 12M / Alpha Momentum / Momentum Sharpe
    Colour = green (strong positive) → red (negative)

    Args:
        full_df: Full metrics DataFrame with momentum columns.
        height:  Chart height.

    Returns:
        go.Figure with annotated momentum heatmap.
    """
    MOMENTUM_COLS = {
        "momentum_3m":     "3M Return",
        "momentum_6m":     "6M Return",
        "cagr_1y":         "1Y Return",
        "alpha_momentum":  "Alpha Mom.",
    }

    available = {k: v for k, v in MOMENTUM_COLS.items() if k in full_df.columns}
    if not available:
        return empty_figure("No momentum data in metrics table")

    sub = full_df[list(available.keys())].dropna(how="all")
    if sub.empty:
        return empty_figure("No momentum data available")

    # Normalise each column 0→1 for colour (higher = greener)
    z_norm = pd.DataFrame(index=sub.index, columns=list(available.keys()), dtype=float)
    z_text = pd.DataFrame(index=sub.index, columns=list(available.keys()), dtype=str)

    for col in available:
        col_data = pd.to_numeric(sub[col], errors="coerce")
        mn, mx = col_data.min(), col_data.max()
        rng = mx - mn
        for idx in sub.index:
            raw = col_data.get(idx)
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                z_norm.loc[idx, col] = np.nan
                z_text.loc[idx, col] = "N/A"
            else:
                z_norm.loc[idx, col] = (raw - mn) / rng if rng != 0 else 0.5
                # Format display
                if col == "alpha_momentum":
                    z_text.loc[idx, col] = f"{raw:.2f}"
                else:
                    z_text.loc[idx, col] = f"{raw*100:.1f}%"

    x_labels = list(available.values())
    y_labels  = [n[:35] + "…" if len(n) > 35 else n for n in sub.index]

    colorscale = [
        [0.00, "#b71c1c"], [0.30, "#e53935"],
        [0.45, T.INK_FAINT], [0.55, T.INK_FAINT],
        [0.70, "#66BB6A"], [1.00, "#2E7D32"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z_norm.values.astype(float),
        x=x_labels, y=y_labels,
        text=z_text.values,
        texttemplate="%{text}",
        textfont=dict(size=9, color="white"),
        colorscale=colorscale,
        showscale=False,
        zmin=0, zmax=1,
        xgap=2, ygap=2,
        hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Momentum Heatmap", x=0.01,
                   font=dict(size=14, color="#E0E0E0")),
        height=max(height, 80 + 35 * len(sub)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=T.GROUND,
        font=dict(color="#E0E0E0", size=10),
        margin=dict(l=220, r=20, t=55, b=80),
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
    )

    return fig
