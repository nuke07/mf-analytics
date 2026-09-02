"""
visualizations/scatter_plots.py
================================
Scatter Plot Charts — Charts 7 & 8 of 8.

Chart 7: Risk vs Return Scatter
    X-axis = Annualized Volatility (risk)
    Y-axis = 3-Year CAGR (return)

    The ideal fund is TOP-LEFT: high return, low risk.
    The worst fund is BOTTOM-RIGHT: low return, high risk.

    Quadrant dividers at median volatility and median return
    divide the chart into four zones.

Chart 8: Volatility vs CAGR Scatter (1-Year focus)
    Same structure as Chart 7 but uses 1-Year CAGR on Y-axis.
    Useful for recent performance comparison.

Both charts:
    - Each fund is a labelled data point
    - Hover shows all key metrics for that fund
    - Quartile colouring (Q1=green, Q4=red)
    - A reference "efficient frontier" diagonal is optionally shown

Usage:
    from visualizations.scatter_plots import plot_risk_return_scatter
    from utils.ui import chart

    chart(plot_risk_return_scatter(full_df))

    Render through utils.ui.chart() rather than st.plotly_chart(): it applies
    the shared toolbar config, and use_container_width is deprecated.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional
from utils.constants import QUARTILE_COLORS
from visualizations._theme import (
    base_layout, empty_figure,
)
from utils import theme as T


# ─────────────────────────────────────────────────────────────────────────────
# CHART 7 — RISK vs RETURN (3Y CAGR vs Volatility)
# ─────────────────────────────────────────────────────────────────────────────

def plot_risk_return_scatter(
    full_df: pd.DataFrame,
    title:   Optional[str] = None,
    height:  int = 520,
) -> go.Figure:
    """
    Risk vs Return scatter — Annualized Volatility (X) vs 3Y CAGR (Y).

    Points are coloured by their Sharpe Ratio quartile:
        Q1 = Green (best risk-adjusted performance)
        Q4 = Red   (worst risk-adjusted performance)

    Args:
        full_df: Full metrics + quartile DataFrame (from compute_category_quartiles)
                 Must have columns: annualized_volatility, cagr_3y,
                 and optionally sharpe_quartile.
        title:   Optional chart title.
        height:  Chart height in pixels.

    Returns:
        go.Figure with labelled scatter plot and quadrant dividers.
    """
    required = ["annualized_volatility", "cagr_3y"]
    for col in required:
        if col not in full_df.columns:
            return empty_figure(f"Column '{col}' not found in data")

    plot_df = full_df[required + ["sharpe_quartile"]].copy() \
        if "sharpe_quartile" in full_df.columns \
        else full_df[required].copy()

    plot_df = plot_df.dropna(subset=required)
    if plot_df.empty:
        return empty_figure("No funds have both Volatility and 3Y CAGR data")

    x_vals = plot_df["annualized_volatility"] * 100   # → percentage
    y_vals = plot_df["cagr_3y"] * 100
    names  = [_truncate(n, 40) for n in plot_df.index]

    # Colour by Sharpe quartile if available
    quartile_col = "sharpe_quartile" if "sharpe_quartile" in plot_df.columns else None
    colors = _get_quartile_colors(plot_df, quartile_col)

    # Hover text: show all key metrics
    hover_texts = _build_hover_text(full_df, plot_df.index)

    fig = go.Figure()

    # ── Quadrant dividers ─────────────────────────────────────────────────────
    x_mid = float(x_vals.median())
    y_mid = float(y_vals.median())
    _add_quadrants(fig, x_mid, y_mid, x_vals, y_vals)

    # ── Data points ───────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x             = x_vals,
            y             = y_vals,
            mode          = "markers+text",
            marker        = dict(
                size         = 11,
                color        = colors,
                line         = dict(color="rgba(255,255,255,0.3)", width=1),
                opacity      = 0.88,
            ),
            text          = names,
            textposition  = "top center",
            textfont      = dict(size=8, color="#C0C0C0"),
            hovertemplate = hover_texts,
            name          = "",
            showlegend    = False,
        )
    )

    # ── Quartile colour legend ─────────────────────────────────────────────────
    if quartile_col:
        _add_quartile_scatter_legend(fig)

    fig.update_layout(
        base_layout(
            title     = title or "Risk vs Return — Volatility vs 3Y CAGR",
            x_title   = "Annualized Volatility (%) — Lower is safer →",
            y_title   = "3-Year CAGR (%) — Higher is better ↑",
            height    = height,
            hovermode = "closest",
            legend    = True,
        )
    )
    fig.update_xaxes(ticksuffix="%")
    fig.update_yaxes(ticksuffix="%")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 8 — VOLATILITY vs 1Y CAGR
# ─────────────────────────────────────────────────────────────────────────────

def plot_vol_cagr_scatter(
    full_df: pd.DataFrame,
    title:   Optional[str] = None,
    height:  int = 520,
) -> go.Figure:
    """
    Volatility vs 1-Year CAGR scatter — recent performance focus.

    Same visual design as plot_risk_return_scatter() but uses
    1-Year CAGR on Y-axis instead of 3-Year.

    Args:
        full_df: Full metrics + quartile DataFrame.
        title:   Optional chart title.
        height:  Chart height.

    Returns:
        go.Figure
    """
    required = ["annualized_volatility", "cagr_1y"]
    for col in required:
        if col not in full_df.columns:
            return empty_figure(f"Column '{col}' not found in data")

    plot_df = full_df[required].copy().dropna(subset=required)
    if plot_df.empty:
        return empty_figure("No funds have both Volatility and 1Y CAGR data")

    x_vals = plot_df["annualized_volatility"] * 100
    y_vals = plot_df["cagr_1y"] * 100
    names  = [_truncate(n, 40) for n in plot_df.index]

    quartile_col = "cagr_1y_quartile" if "cagr_1y_quartile" in full_df.columns else None
    colors = _get_quartile_colors(plot_df, quartile_col, full_df)

    hover_texts = _build_hover_text(full_df, plot_df.index)

    fig = go.Figure()

    x_mid = float(x_vals.median())
    y_mid = float(y_vals.median())
    _add_quadrants(fig, x_mid, y_mid, x_vals, y_vals)

    fig.add_trace(
        go.Scatter(
            x             = x_vals,
            y             = y_vals,
            mode          = "markers+text",
            marker        = dict(
                size    = 11,
                color   = colors,
                line    = dict(color="rgba(255,255,255,0.3)", width=1),
                opacity = 0.88,
            ),
            text          = names,
            textposition  = "top center",
            textfont      = dict(size=8, color="#C0C0C0"),
            hovertemplate = hover_texts,
            name          = "",
            showlegend    = False,
        )
    )

    if quartile_col:
        _add_quartile_scatter_legend(fig)

    fig.update_layout(
        base_layout(
            title     = title or "Volatility vs 1-Year CAGR",
            x_title   = "Annualized Volatility (%) — Lower is safer →",
            y_title   = "1-Year CAGR (%) — Higher is better ↑",
            height    = height,
            hovermode = "closest",
        )
    )
    fig.update_xaxes(ticksuffix="%")
    fig.update_yaxes(ticksuffix="%")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# GENERIC SCATTER (configurable axes — used by future pages)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_PCT_METRICS = {
    "cagr_1y", "cagr_3y", "cagr_5y", "cagr_inception",
    "annualized_volatility", "downside_volatility",
    "max_drawdown", "avg_drawdown",
    "median_rolling_1y", "median_rolling_3y",
    "positive_freq", "win_rate",
    "pct_positive_rolling_1y", "pct_positive_rolling_3y",
}


def _truncate(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[:max_len - 3] + "..."


def _get_quartile_colors(
    plot_df:      pd.DataFrame,
    quartile_col: Optional[str],
    source_df:    Optional[pd.DataFrame] = None,
) -> list:
    """Map quartile labels to colours for scatter markers."""
    src = source_df if source_df is not None else plot_df
    default = T.DATA_PRIMARY

    if quartile_col is None or quartile_col not in src.columns:
        return [default] * len(plot_df)

    return [
        QUARTILE_COLORS.get(str(src.loc[idx, quartile_col]), default)
        if idx in src.index else default
        for idx in plot_df.index
    ]


def _build_hover_text(full_df: pd.DataFrame, index) -> list:
    """Build rich hover text for each fund showing key metrics."""
    texts = []
    for fund in index:
        if fund not in full_df.index:
            texts.append(f"<b>{_truncate(str(fund), 40)}</b><extra></extra>")
            continue

        row = full_df.loc[fund]

        def _fmt(key, pct=False, ratio=False):
            val = row.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "N/A"
            if pct:
                return f"{val*100:.2f}%"
            if ratio:
                return f"{val:.3f}"
            return str(val)

        text = (
            f"<b>{_truncate(str(fund), 40)}</b><br>"
            f"3Y CAGR:    {_fmt('cagr_3y', pct=True)}<br>"
            f"Volatility: {_fmt('annualized_volatility', pct=True)}<br>"
            f"Max DD:     {_fmt('max_drawdown', pct=True)}<br>"
            f"Sharpe:     {_fmt('sharpe', ratio=True)}<br>"
            f"Sortino:    {_fmt('sortino', ratio=True)}<br>"
            f"Win Rate:   {_fmt('win_rate', pct=True)}"
            "<extra></extra>"
        )
        texts.append(text)

    return texts


def _add_quadrants(
    fig:   go.Figure,
    x_mid: float,
    y_mid: float,
    x_vals: pd.Series,
    y_vals: pd.Series,
) -> None:
    """Add median-split quadrant dividers and labels."""
    x_min = float(x_vals.min()) * 0.85
    x_max = float(x_vals.max()) * 1.15
    y_min = float(y_vals.min()) * (1.15 if y_vals.min() < 0 else 0.85)
    y_max = float(y_vals.max()) * 1.15

    # Vertical divider
    fig.add_shape(
        type="line", x0=x_mid, x1=x_mid, y0=y_min, y1=y_max,
        line=dict(color="rgba(255,255,255,0.15)", dash="dot", width=1),
    )
    # Horizontal divider
    fig.add_shape(
        type="line", x0=x_min, x1=x_max, y0=y_mid, y1=y_mid,
        line=dict(color="rgba(255,255,255,0.15)", dash="dot", width=1),
    )

    # Quadrant labels
    _ql = dict(showarrow=False, font=dict(size=8, color="rgba(200,200,200,0.3)"))
    fig.add_annotation(text="LOW RISK / HIGH RETURN",  x=x_min*1.05, y=y_max*0.97, xanchor="left",  **_ql)
    fig.add_annotation(text="HIGH RISK / HIGH RETURN", x=x_max*0.99, y=y_max*0.97, xanchor="right", **_ql)
    fig.add_annotation(text="LOW RISK / LOW RETURN",   x=x_min*1.05, y=y_min*0.97, xanchor="left",  **_ql)
    fig.add_annotation(text="HIGH RISK / LOW RETURN",  x=x_max*0.99, y=y_min*0.97, xanchor="right", **_ql)


def _add_quartile_scatter_legend(fig: go.Figure) -> None:
    """Add dummy traces for quartile colour legend."""
    for label, color in [
        ("Q1 — Best Sharpe",    QUARTILE_COLORS["Q1"]),
        ("Q2",                  QUARTILE_COLORS["Q2"]),
        ("Q3",                  QUARTILE_COLORS["Q3"]),
        ("Q4 — Worst Sharpe",   QUARTILE_COLORS["Q4"]),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=9, color=color),
                name=label,
                showlegend=True,
            )
        )
