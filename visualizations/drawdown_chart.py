"""
visualizations/drawdown_chart.py
=================================
Drawdown Chart — Chart 2 of 8.

Shows how far a fund has fallen from its all-time high at each point in time.
The drawdown series is always ≤ 0.

Visual design:
    - Red filled area below zero → immediately communicates "loss"
    - The deeper the red, the more severe the drawdown
    - Zero line is clearly visible (the fund's all-time high line)
    - Multi-fund: overlapping lines, no fill (to avoid occlusion)

Annotations (single fund mode only):
    - Top 3 worst drawdown events are annotated with their magnitude

Usage:
    from visualizations.drawdown_chart import plot_drawdown

    # Single fund (with fill and annotations)
    fig = plot_drawdown({"Axis Bluechip": dd_series})

    # Comparison (lines only)
    fig = plot_drawdown({"Fund A": dd_a, "Fund B": dd_b})
"""

import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional, List
from visualizations._theme import (
    base_layout, empty_figure, get_color,
    DOWN_COLOR, NEUTRAL_COLOR, ZERO_LINE_COLOR,
)
from utils import theme as T


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHART FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_drawdown(
    drawdown_dict: Dict[str, Optional[pd.Series]],
    title:         Optional[str] = None,
    height:        int = 380,
    annotate:      bool = True,
) -> go.Figure:
    """
    Plot the drawdown series for one or more funds.

    Args:
        drawdown_dict: {fund_name: drawdown_series}
                       drawdown_series = output of calc_drawdown_series()
                       Values are always ≤ 0 (percentage below peak).
        title:         Optional chart title override.
        height:        Chart height in pixels.
        annotate:      If True (single fund only), annotate the 3 worst
                       drawdown troughs with their magnitude labels.

    Returns:
        go.Figure ready for st.plotly_chart()
    """
    valid = {
        k: v for k, v in drawdown_dict.items()
        if v is not None and len(v) > 0
    }

    if not valid:
        return empty_figure("No drawdown data available")

    fig = go.Figure()
    single_fund = len(valid) == 1

    for i, (name, dd) in enumerate(valid.items()):
        # Convert to percentage for display
        dd_pct = dd * 100
        color = get_color(i) if not single_fund else DOWN_COLOR

        if single_fund:
            # Filled red area for single fund — maximum visual impact
            fig.add_trace(
                go.Scatter(
                    x             = dd.index,
                    y             = dd_pct,
                    name          = name,
                    mode          = "lines",
                    line          = dict(color=DOWN_COLOR, width=1.5),
                    fill          = "tozeroy",
                    fillcolor     = T.rgba(T.DOWN, 0.20),
                    hovertemplate = (
                        "Date: %{x|%d %b %Y}<br>"
                        "Drawdown: %{y:.2f}%"
                        "<extra></extra>"
                    ),
                )
            )
            # Add annotations for worst periods
            if annotate:
                _annotate_worst_drawdowns(fig, dd, dd_pct, n=3)

        else:
            # Multi-fund: lines only, no fill (would obscure each other)
            fig.add_trace(
                go.Scatter(
                    x             = dd.index,
                    y             = dd_pct,
                    name          = name,
                    mode          = "lines",
                    line          = dict(color=color, width=1.8),
                    hovertemplate = (
                        f"<b>{name}</b><br>"
                        "Date: %{x|%d %b %Y}<br>"
                        "Drawdown: %{y:.2f}%"
                        "<extra></extra>"
                    ),
                )
            )

    # Zero reference line — represents the fund's all-time high
    all_dates = pd.concat(list(valid.values())).index
    fig.add_hline(
        y           = 0,
        line_dash   = "dot",
        line_color  = ZERO_LINE_COLOR,
        line_width  = 1,
        annotation_text      = "All-Time High",
        annotation_position  = "top right",
        annotation_font_size = 10,
        annotation_font_color = NEUTRAL_COLOR,
    )

    if title is None:
        if single_fund:
            title = f"Drawdown — {list(valid.keys())[0]}"
        else:
            title = "Drawdown Comparison"

    fig.update_layout(
        base_layout(
            title     = title,
            x_title   = "Date",
            y_title   = "Drawdown (%)",
            height    = height,
            hovermode = "x unified",
        )
    )

    # Y-axis: always show negative values only, add % suffix
    fig.update_yaxes(ticksuffix="%", tickformat=".1f")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_worst_drawdowns(
    fig:    go.Figure,
    dd:     pd.Series,
    dd_pct: pd.Series,
    n:      int = 3,
) -> None:
    """
    Add text annotations at the trough of the N worst drawdown events.
    Modifies fig in place.

    Args:
        fig:    Plotly figure to annotate
        dd:     Drawdown series (fractions, ≤ 0)
        dd_pct: Drawdown series in percentages (for display)
        n:      Number of worst events to annotate
    """
    # Find the n worst trough points (most negative values)
    # Use a rolling window to avoid annotating points too close together
    try:
        worst_idx = dd_pct.nsmallest(n * 50).index   # Oversample to find distinct troughs

        # Cluster nearby points — keep only the local minimum in each cluster
        MIN_SEPARATION_DAYS = 90
        annotated: List[pd.Timestamp] = []
        for date in worst_idx:
            # Check if this point is far enough from already-annotated points
            if all(abs((date - d).days) > MIN_SEPARATION_DAYS for d in annotated):
                annotated.append(date)
            if len(annotated) >= n:
                break

        for date in annotated:
            val = dd_pct[date]
            fig.add_annotation(
                x          = date,
                y          = val,
                text       = f"<b>{val:.1f}%</b>",
                showarrow  = True,
                arrowhead  = 2,
                arrowcolor = DOWN_COLOR,
                arrowwidth = 1.5,
                ax         = 0,
                ay         = 30,   # Push label upward (positive = up in screen coords)
                font       = dict(color=DOWN_COLOR, size=10),
                bgcolor    = T.rgba(T.PANEL_HI, 0.85),
                borderpad  = 3,
            )
    except Exception:
        pass   # Never crash due to annotation failure


# ─────────────────────────────────────────────────────────────────────────────
# UNDERWATER CHART (alternative view)
# ─────────────────────────────────────────────────────────────────────────────


