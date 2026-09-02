"""
visualizations/heatmaps.py
===========================
Heatmap Charts — Charts 5 & 6 of 8.

Chart 5: Metric Comparison Heatmap
    Shows raw metric values for multiple funds across multiple metrics.
    Each cell is colour-coded by value (green=good, red=bad).
    Useful for spotting patterns: which fund is consistently green?

Chart 6: Quartile Heatmap
    Shows Q1/Q2/Q3/Q4 labels instead of raw values.
    Standardised across metrics — Q1 always means "top 25% in category".
    At a glance you can see which fund dominates its category.

Usage:
    from visualizations.heatmaps import plot_metric_heatmap, plot_quartile_heatmap

    fig5 = plot_metric_heatmap(metrics_df, selected_metrics)
    fig6 = plot_quartile_heatmap(full_df, selected_metrics)
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Optional
from utils.constants import LOWER_IS_BETTER, METRIC_LABELS, QUARTILE_COLORS
from visualizations._theme import empty_figure
from utils import theme as T


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT METRICS FOR HEATMAPS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_HEATMAP_METRICS: List[str] = [
    "cagr_1y", "cagr_3y", "cagr_5y",
    "annualized_volatility", "max_drawdown",
    "sharpe", "sortino", "calmar",
    "median_rolling_1y", "win_rate",
]


# ─────────────────────────────────────────────────────────────────────────────
# CHART 5 — METRIC COMPARISON HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_heatmap(
    metrics_df:       pd.DataFrame,
    selected_metrics: Optional[List[str]] = None,
    title:            Optional[str] = None,
    height:           int = 500,
) -> go.Figure:
    """
    Heatmap of raw metric values — funds vs metrics, colour-coded by value.

    Each metric column is independently normalised to [0,1] so metrics on
    different scales (% CAGR vs ratio) are visually comparable.
    For LOWER_IS_BETTER metrics the scale is inverted (low=green, high=red).

    Args:
        metrics_df:       DataFrame — rows=funds, columns=metric_keys
                          (output of engine.compute_category_metrics / build_metrics_dataframe)
        selected_metrics: List of metric keys to display.
                          Defaults to DEFAULT_HEATMAP_METRICS.
        title:            Optional chart title.
        height:           Chart height in pixels.

    Returns:
        go.Figure with annotated heatmap.
    """
    if metrics_df is None or metrics_df.empty:
        return empty_figure("No metrics data available")

    metrics = selected_metrics or DEFAULT_HEATMAP_METRICS
    available = [m for m in metrics if m in metrics_df.columns]

    if not available:
        return empty_figure("Selected metrics not found in data")

    sub = metrics_df[available].copy()

    # Drop funds with ALL missing values
    sub = sub.dropna(how="all")
    if sub.empty:
        return empty_figure("All funds have insufficient data")

    # Normalise each column to [0,1] independently (for colour scale)
    z_norm   = pd.DataFrame(index=sub.index, columns=available, dtype=float)
    z_text   = pd.DataFrame(index=sub.index, columns=available, dtype=str)

    for col in available:
        col_data = pd.to_numeric(sub[col], errors="coerce")
        col_min, col_max = col_data.min(), col_data.max()
        rng = col_max - col_min

        for idx in sub.index:
            raw = col_data[idx]
            if pd.isna(raw) or rng == 0:
                z_norm.loc[idx, col]  = np.nan
                z_text.loc[idx, col]  = "N/A"
            else:
                # Normalise: 0 = worst, 1 = best
                norm = (raw - col_min) / rng
                if col in LOWER_IS_BETTER:
                    norm = 1 - norm   # Invert: low raw value → high norm score
                z_norm.loc[idx, col] = norm

                # Format display text
                if col in ["drawdown_duration", "max_consec_positive", "max_consec_negative"]:
                    z_text.loc[idx, col] = f"{int(raw)}d"
                elif abs(raw) < 10:
                    z_text.loc[idx, col] = f"{raw:.3f}"
                else:
                    z_text.loc[idx, col] = f"{raw*100:.1f}%"

    # Axis labels
    x_labels = [METRIC_LABELS.get(m, m) for m in available]
    y_labels  = [_truncate(name, 35) for name in sub.index]

    fig = go.Figure(
        go.Heatmap(
            z              = z_norm.values.astype(float),
            x              = x_labels,
            y              = y_labels,
            text           = z_text.values,
            texttemplate   = "%{text}",
            textfont       = dict(size=9, color="white"),
            colorscale     = _green_red_scale(),
            showscale      = False,
            zmin           = 0,
            zmax           = 1,
            hovertemplate  = (
                "<b>%{y}</b><br>"
                "Metric: %{x}<br>"
                "Value: %{text}"
                "<extra></extra>"
            ),
            xgap           = 2,
            ygap           = 2,
        )
    )

    fig.update_layout(
        title         = dict(
            text  = title or "Metric Comparison Heatmap",
            x     = 0.01,
            font  = dict(size=14, color="#E0E0E0"),
        ),
        height        = max(height, 80 + 35 * len(sub)),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = T.GROUND,
        font          = dict(color="#E0E0E0", size=10),
        margin        = dict(l=220, r=20, t=55, b=120),
        xaxis         = dict(
            tickangle = -35,
            tickfont  = dict(size=9),
            side      = "bottom",
        ),
        yaxis         = dict(
            tickfont  = dict(size=9),
            autorange = "reversed",   # Top fund at top
        ),
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 6 — QUARTILE HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_quartile_heatmap(
    full_df:          pd.DataFrame,
    selected_metrics: Optional[List[str]] = None,
    title:            Optional[str] = None,
    height:           int = 500,
) -> go.Figure:
    """
    Heatmap showing Q1–Q4 quartile labels per fund per metric.

    Each cell is coloured:
        Q1 = Green    (top 25% in category)
        Q2 = Light green
        Q3 = Orange
        Q4 = Red      (bottom 25%)
        N/A = Grey

    Args:
        full_df:          Output of engine.compute_category_quartiles()
                          Contains '{metric}_quartile' columns.
        selected_metrics: List of base metric keys (not quartile keys).
                          Defaults to DEFAULT_HEATMAP_METRICS.
        title:            Optional title.
        height:           Chart height.

    Returns:
        go.Figure with colour-coded quartile heatmap.
    """
    if full_df is None or full_df.empty:
        return empty_figure("No quartile data available")

    metrics = selected_metrics or DEFAULT_HEATMAP_METRICS

    # Filter to metrics that have quartile columns in the dataframe
    quartile_cols = [f"{m}_quartile" for m in metrics if f"{m}_quartile" in full_df.columns]
    available_metrics = [m for m in metrics if f"{m}_quartile" in full_df.columns]

    if not quartile_cols:
        return empty_figure("No quartile columns found — run category analysis first")

    sub = full_df[quartile_cols].copy()
    sub = sub.dropna(how="all")
    if sub.empty:
        return empty_figure("No quartile data after filtering")

    # Map quartile labels to numeric values for colour scale
    Q_NUM = {"Q1": 4, "Q2": 3, "Q3": 2, "Q4": 1, "N/A": 0}

    z_num  = sub.map(lambda v: Q_NUM.get(str(v), 0))
    z_text = sub.fillna("N/A")

    x_labels  = [METRIC_LABELS.get(m, m) for m in available_metrics]
    y_labels  = [_truncate(name, 35) for name in sub.index]

    # Custom discrete colour scale
    colorscale = [
        [0.00, "rgba(80,80,80,0.5)"],   # 0 = N/A
        [0.20, "rgba(80,80,80,0.5)"],
        [0.21, QUARTILE_COLORS["Q4"]],   # 1 = Q4 red
        [0.40, QUARTILE_COLORS["Q4"]],
        [0.41, QUARTILE_COLORS["Q3"]],   # 2 = Q3 orange
        [0.60, QUARTILE_COLORS["Q3"]],
        [0.61, QUARTILE_COLORS["Q2"]],   # 3 = Q2 light green
        [0.80, QUARTILE_COLORS["Q2"]],
        [0.81, QUARTILE_COLORS["Q1"]],   # 4 = Q1 green
        [1.00, QUARTILE_COLORS["Q1"]],
    ]

    fig = go.Figure(
        go.Heatmap(
            z              = z_num.values,
            x              = x_labels,
            y              = y_labels,
            text           = z_text.values,
            texttemplate   = "%{text}",
            textfont       = dict(size=10, color="white", family="sans-serif"),
            colorscale     = colorscale,
            showscale      = False,
            zmin           = 0,
            zmax           = 4,
            hovertemplate  = (
                "<b>%{y}</b><br>"
                "Metric: %{x}<br>"
                "Quartile: %{text}"
                "<extra></extra>"
            ),
            xgap           = 3,
            ygap           = 3,
        )
    )

    # Add a colour legend as shapes (since showscale=False)
    _add_quartile_legend(fig)

    fig.update_layout(
        title         = dict(
            text  = title or "Quartile Rankings Heatmap",
            x     = 0.01,
            font  = dict(size=14, color="#E0E0E0"),
        ),
        height        = max(height, 100 + 38 * len(sub)),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = T.GROUND,
        font          = dict(color="#E0E0E0", size=10),
        margin        = dict(l=220, r=20, t=75, b=120),
        xaxis         = dict(
            tickangle = -35,
            tickfont  = dict(size=9),
        ),
        yaxis         = dict(
            tickfont  = dict(size=9),
            autorange = "reversed",
        ),
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _truncate(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[:max_len - 3] + "..."


def _green_red_scale():
    """Custom green→yellow→red colorscale where 1=green (best), 0=red (worst)."""
    return [
        [0.00, "#b71c1c"],   # Deep red — worst
        [0.25, "#e53935"],
        [0.45, T.INK_FAINT],   # neutral midpoint
        [0.55, T.INK_FAINT],
        [0.75, "#66BB6A"],
        [1.00, "#2E7D32"],   # Deep green — best
    ]


def _add_quartile_legend(fig: go.Figure) -> None:
    """Add Q1/Q2/Q3/Q4/N/A legend annotations in the top-right corner."""
    items = [
        ("Q1 — Top 25%",   QUARTILE_COLORS["Q1"]),
        ("Q2 — Next 25%",  QUARTILE_COLORS["Q2"]),
        ("Q3 — Next 25%",  QUARTILE_COLORS["Q3"]),
        ("Q4 — Bottom 25%",QUARTILE_COLORS["Q4"]),
        ("N/A — No Data",  "#9E9E9E"),
    ]
    y_start = 1.12
    for label, color in items:
        fig.add_annotation(
            text      = f"<span style='color:{color}'>■</span> {label}",
            xref      = "paper", yref = "paper",
            x         = 0.99,
            y         = y_start,
            showarrow = False,
            align     = "right",
            xanchor   = "right",
            font      = dict(size=10, color="#E0E0E0"),
        )
        y_start -= 0.04
