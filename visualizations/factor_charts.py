"""
visualizations/factor_charts.py
================================
Phase C — Fama-French factor model charts.

Chart 1 — Factor Loading Bar Chart
    Shows β_mkt, β_smb, β_hml, β_wml for one or more funds.
    The zero line separates positive/negative tilts.

Chart 2 — Factor Contribution Chart
    Stacked bar showing how much each factor contributed to the fund's
    total annualized return vs how much came from pure alpha.

Chart 3 — Rolling 4-Factor Alpha
    Same as rolling Jensen's alpha but controlling for all 4 factors.
    Purer measure of skill persistence.

Chart 4 — Factor Exposure Heatmap
    Category-wide: funds × factor loadings, colour-coded.
    Instantly shows which funds have similar factor profiles.
"""

import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional
from visualizations._theme import (
    base_layout, empty_figure, get_color,
    DOWN_COLOR,
)
from utils import theme as T


# Factor display config



# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — FACTOR LOADING BAR CHART
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — FACTOR CONTRIBUTION (RETURN ATTRIBUTION)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — ROLLING 4-FACTOR ALPHA
# ─────────────────────────────────────────────────────────────────────────────

def plot_rolling_alpha_factors(
    rolling_dict: Dict[str, Optional[pd.Series]],
    height:       int = 400,
) -> go.Figure:
    """
    Rolling factor-model alpha over time.

    Factor-count agnostic — the caller supplies whatever rolling alpha
    series its model produced. Named for the 4-factor model originally,
    now fed the 6-factor series.

    Args:
        rolling_dict: {fund_name: rolling_alpha_series}
        height:       Chart height.

    Returns:
        go.Figure with rolling 4-factor alpha line.
    """
    valid = {k: v for k, v in rolling_dict.items()
             if v is not None and len(v) > 0}

    if not valid:
        return empty_figure(
            "Rolling 4-Factor Alpha requires 2+ years of "
            "overlapping fund and factor history"
        )

    fig = go.Figure()

    for i, (name, series) in enumerate(valid.items()):
        pct   = (series * 100).dropna()
        color = get_color(i)

        fig.add_trace(go.Scatter(
            x=pct.index, y=pct.values,
            name=name, mode="lines",
            line=dict(color=color, width=1.8),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "4F Rolling Alpha: %{y:.2f}%"
                "<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line_dash="dash",
                  line_color=T.rgba(T.DOWN, 0.5), line_width=1.5,
                  annotation_text="0% True Alpha",
                  annotation_position="right",
                  annotation_font_color=DOWN_COLOR,
                  annotation_font_size=10)

    fig.update_layout(
        base_layout(
            title     = "Rolling 1-Year 4-Factor Alpha (True Alpha after Factor Adjustment)",
            x_title   = "Date",
            y_title   = "Annualized 4-Factor Alpha (%)",
            height    = height,
            hovermode = "x unified",
        )
    )
    fig.update_yaxes(ticksuffix="%")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — FACTOR EXPOSURE HEATMAP
# ─────────────────────────────────────────────────────────────────────────────


