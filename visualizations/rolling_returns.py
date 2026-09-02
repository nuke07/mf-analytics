"""
visualizations/rolling_returns.py
==================================
Rolling Return Charts — Charts 3 & 4 of 8.

Chart 3: Rolling Return Time Series
    Line chart showing how the rolling N-year CAGR changes over time.
    Each point answers: "What was the annualized return if I invested
    exactly N years before this date?"

    A flat, high line = consistently good fund.
    A wildly oscillating line = inconsistent fund.

Chart 4: Rolling Return Distribution
    Histogram showing the distribution of all rolling return values.
    Answers: "How often does this fund deliver positive returns?"

    A distribution skewed right of 0% = fund rarely has negative periods.
    A wide distribution = high variance in outcomes (inconsistent).

Usage:
    from visualizations.rolling_returns import (
        plot_rolling_timeseries,
        plot_rolling_distribution,
    )

    fig1 = plot_rolling_timeseries({"Fund A": series_1y}, window_label="1-Year")
    fig2 = plot_rolling_distribution({"Fund A": series_1y}, window_label="1-Year")
"""

import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional
from visualizations._theme import (
    base_layout, empty_figure, get_color,
    UP_COLOR, DOWN_COLOR,
)
from utils import theme as T
from visualizations._theme import last_value_badges


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — ROLLING RETURN TIME SERIES
# ─────────────────────────────────────────────────────────────────────────────

def plot_rolling_timeseries(
    rolling_dict:  Dict[str, Optional[pd.Series]],
    window_label:  str = "1-Year",
    title:         Optional[str] = None,
    height:        int = 400,
) -> go.Figure:
    """
    Rolling return time series — how the rolling CAGR evolves over time.

    Args:
        rolling_dict:  {fund_name: rolling_return_series}
                       Series of annualized rolling returns (output of
                       compute_rolling_returns in nav_processor).
        window_label:  Human-readable window label e.g. "1-Year" or "3-Year".
        title:         Optional title override.
        height:        Chart height in pixels.

    Returns:
        go.Figure with one line per fund + zero reference line.
    """
    valid = {
        k: v for k, v in rolling_dict.items()
        if v is not None and len(v) > 0
    }

    if not valid:
        return empty_figure(f"Insufficient data for {window_label} rolling returns")

    fig = go.Figure()

    for i, (name, series) in enumerate(valid.items()):
        pct = series * 100    # Convert to percentage for display
        color = get_color(i)

        fig.add_trace(
            go.Scatter(
                x             = pct.index,
                y             = pct.values,
                name          = name,
                mode          = "lines",
                line          = dict(color=color, width=1.8),
                hovertemplate = (
                    f"<b>{name}</b><br>"
                    "Date: %{x|%d %b %Y}<br>"
                    f"{window_label} Rolling CAGR: %{{y:.2f}}%"
                    "<extra></extra>"
                ),
            )
        )

    # Zero reference — negative rolling returns are below this
    fig.add_hline(
        y           = 0,
        line_dash   = "dash",
        line_color  = T.rgba(T.DOWN, 0.5),
        line_width  = 1.2,
        annotation_text       = "0%",
        annotation_position   = "right",
        annotation_font_size  = 10,
        annotation_font_color = DOWN_COLOR,
    )

    if title is None:
        title = f"{window_label} Rolling Return (Annualized CAGR)"

    fig.update_layout(
        base_layout(
            title     = title,
            x_title   = "Date (end of rolling window)",
            y_title   = f"{window_label} Rolling CAGR (%)",
            height    = height,
            hovermode = "x unified",
        )
    )
    fig.update_yaxes(ticksuffix="%")

    # Stamp each line's final value on the right axis in its own colour.
    last_value_badges(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — ROLLING RETURN DISTRIBUTION (HISTOGRAM)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rolling_distribution(
    rolling_dict:  Dict[str, Optional[pd.Series]],
    window_label:  str = "1-Year",
    title:         Optional[str] = None,
    height:        int = 400,
    nbins:         int = 40,
) -> go.Figure:
    """
    Histogram showing the distribution of rolling return values.

    Green bars = positive rolling periods (fund made money over the window).
    Red bars   = negative rolling periods (fund lost money).

    Overlapping histograms are used for multi-fund comparison.
    For single fund, split green/red coloring is applied.

    Args:
        rolling_dict:  {fund_name: rolling_return_series}
        window_label:  e.g. "1-Year" or "3-Year"
        title:         Optional title override.
        height:        Chart height.
        nbins:         Number of histogram bins.

    Returns:
        go.Figure with histogram overlays + vertical zero line.
    """
    valid = {
        k: v for k, v in rolling_dict.items()
        if v is not None and len(v) > 0
    }

    if not valid:
        return empty_figure(f"Insufficient data for {window_label} distribution")

    fig = go.Figure()
    single_fund = len(valid) == 1

    for i, (name, series) in enumerate(valid.items()):
        pct = (series * 100).dropna()
        color = get_color(i)

        if single_fund:
            # Split into positive/negative for colour-coded bars
            positive = pct[pct >= 0]
            negative = pct[pct < 0]

            # Shared bin settings for alignment
            all_min = float(pct.min()) - 1
            all_max = float(pct.max()) + 1
            bin_size = (all_max - all_min) / nbins

            fig.add_trace(
                go.Histogram(
                    x         = positive,
                    name      = "Positive Periods",
                    xbins     = dict(start=0, end=all_max, size=bin_size),
                    marker    = dict(color=UP_COLOR, opacity=0.75, line=dict(width=0.5, color="rgba(0,0,0,0.2)")),
                    hovertemplate = "Return: %{x:.1f}%<br>Count: %{y}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Histogram(
                    x         = negative,
                    name      = "Negative Periods",
                    xbins     = dict(start=all_min, end=0, size=bin_size),
                    marker    = dict(color=DOWN_COLOR, opacity=0.75, line=dict(width=0.5, color="rgba(0,0,0,0.2)")),
                    hovertemplate = "Return: %{x:.1f}%<br>Count: %{y}<extra></extra>",
                )
            )

            # Stat annotations
            pct_positive = (pct >= 0).sum() / len(pct) * 100
            fig.add_annotation(
                text      = (
                    f"<b>Positive periods: {pct_positive:.1f}%</b><br>"
                    f"Avg: {pct.mean():.2f}%  |  "
                    f"Worst: {pct.min():.2f}%  |  "
                    f"Best: {pct.max():.2f}%"
                ),
                xref      = "paper", yref = "paper",
                x = 0.02, y = 0.97,
                showarrow = False,
                align     = "left",
                bgcolor   = T.rgba(T.PANEL_HI, 0.85),
                bordercolor = "rgba(255,255,255,0.1)",
                font      = dict(size=10, color="#E0E0E0"),
            )

        else:
            # Multiple funds: semi-transparent overlapping histograms
            fig.add_trace(
                go.Histogram(
                    x         = pct,
                    name      = name,
                    nbinsx    = nbins,
                    marker    = dict(
                        color   = color,
                        opacity = 0.55,
                        line    = dict(width=0.5, color="rgba(0,0,0,0.3)"),
                    ),
                    hovertemplate = f"<b>{name}</b><br>Return: %{{x:.1f}}%<br>Count: %{{y}}<extra></extra>",
                )
            )

    # Zero reference line
    fig.add_vline(
        x           = 0,
        line_dash   = "dash",
        line_color  = T.rgba(T.DOWN, 0.6),
        line_width  = 1.5,
        annotation_text       = "0%",
        annotation_position   = "top",
        annotation_font_size  = 10,
        annotation_font_color = DOWN_COLOR,
    )

    if not single_fund:
        fig.update_layout(barmode="overlay")

    if title is None:
        title = f"{window_label} Rolling Return Distribution"

    fig.update_layout(
        base_layout(
            title   = title,
            x_title = f"{window_label} Rolling CAGR (%)",
            y_title = "Number of Periods",
            height  = height,
        )
    )
    fig.update_xaxes(ticksuffix="%")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED VIEW (both charts stacked — for Fund Analytics page)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rolling_combined(
    rolling_dict:  Dict[str, Optional[pd.Series]],
    window_label:  str = "1-Year",
    height:        int = 750,
) -> go.Figure:
    """
    Vertically stacked layout: rolling time series on top, distribution below.
    Used on the Fund Analytics page for a compact single-fund view.

    Args:
        rolling_dict:  {fund_name: rolling_return_series}
        window_label:  "1-Year" or "3-Year"
        height:        Total figure height (split 55/45 between subplots)

    Returns:
        go.Figure with two vertically stacked subplots
    """
    from plotly.subplots import make_subplots

    valid = {k: v for k, v in rolling_dict.items() if v is not None and len(v) > 0}
    if not valid:
        return empty_figure(f"Insufficient data for {window_label} rolling analysis")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.10,
        subplot_titles=(
            f"{window_label} Rolling Return (Annualized CAGR)",
            f"{window_label} Rolling Return Distribution",
        ),
        row_heights=[0.55, 0.45],
    )

    for i, (name, series) in enumerate(valid.items()):
        pct = (series * 100).dropna()
        color = get_color(i)

        # Row 1: Time series
        fig.add_trace(
            go.Scatter(
                x=pct.index, y=pct.values, name=name,
                mode="lines", line=dict(color=color, width=1.8),
                hovertemplate=f"<b>{name}</b><br>Date: %{{x|%d %b %Y}}<br>CAGR: %{{y:.2f}}%<extra></extra>",
            ),
            row=1, col=1,
        )

        # Row 2: Histogram
        fig.add_trace(
            go.Histogram(
                x=pct, name=name,
                nbinsx=35,
                marker=dict(color=color, opacity=0.7),
                showlegend=False,
                hovertemplate=f"<b>{name}</b><br>Return: %{{x:.1f}}%<br>Count: %{{y}}<extra></extra>",
            ),
            row=2, col=1,
        )

    # Zero lines
    fig.add_hline(y=0, line_dash="dash", line_color=T.rgba(T.DOWN, 0.5), line_width=1.2, row=1, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color=T.rgba(T.DOWN, 0.5), line_width=1.2, row=2, col=1)

    fig.update_layout(
        height          = height,
        paper_bgcolor   = "rgba(0,0,0,0)",
        plot_bgcolor    = T.GROUND,
        font            = dict(color="#E0E0E0", family="sans-serif"),
        showlegend      = len(valid) > 1,
        hovermode       = "x unified",
    )
    fig.update_yaxes(ticksuffix="%", gridcolor="rgba(255,255,255,0.08)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")

    return fig
