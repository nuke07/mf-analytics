"""
visualizations/_theme.py
========================
Shared Plotly theme and layout helpers used by all chart modules.

All charts in the platform use this theme so they are visually consistent
with each other and with the Streamlit dark config.toml.

Importing:
    from visualizations._theme import apply_theme, COLORS, GRID_COLOR
"""

import plotly.graph_objects as go
from typing import Optional, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────────────────────

from utils import theme as _t

COLORS          = _t.CHART_SERIES

BG_PAPER        = "rgba(0,0,0,0)"       # transparent — inherits the page ground
# The plot area is the PAGE, not a panel. Sitting it on a lifted panel drew a
# grey rectangle behind every chart and cost the marks contrast; on a terminal
# the chart is a window onto black.
BG_PLOT         = _t.GROUND
GRID_COLOR      = "#161A22"   # there when looked for, gone when reading a shape
ZERO_LINE_COLOR = "#39404F"   # the ONE emphasised rule
FONT_COLOR      = _t.INK
FONT_FAMILY     = _t.PLOTLY_SANS
# Axis ticks are numbers, so they get the tabular mono face like every other
# number in the app. Without this the chart axes were the last place in the
# UI where digits did not line up.
TICK_FAMILY     = _t.PLOTLY_MONO

UP_COLOR        = _t.UP
DOWN_COLOR      = _t.DOWN
NEUTRAL_COLOR   = _t.NEUTRAL


# ─────────────────────────────────────────────────────────────────────────────
# BASE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def base_layout(
    title:       Optional[str] = None,
    x_title:     Optional[str] = None,
    y_title:     Optional[str] = None,
    height:      int = 420,
    legend:      bool = True,
    hovermode:   str = "x unified",
    **extra,
) -> go.Layout:
    """
    Return a go.Layout object with the platform's standard dark theme applied.

    Args:
        title:     Chart title string
        x_title:   X-axis label
        y_title:   Y-axis label
        height:    Chart height in pixels
        legend:    Whether to show the legend
        hovermode: Plotly hovermode ('x unified', 'closest', False)
        **extra:   Any additional go.Layout kwargs

    Returns:
        go.Layout with dark theme pre-applied.
    """
    layout_dict: Dict[str, Any] = dict(
        height          = height,
        paper_bgcolor   = BG_PAPER,
        plot_bgcolor    = BG_PLOT,
        hovermode       = hovermode,
        margin          = dict(l=18, r=78, t=58 if title else 40, b=46),
        font            = dict(color=FONT_COLOR, family=FONT_FAMILY, size=12),
        showlegend      = legend,
        # Legend ABOVE the plot, never inside it. Floating inside, it landed on
        # the chart title and on the data — every comparison page had a legend
        # box sitting across its own heading. y > 1 puts it in the margin.
        legend          = dict(
            bgcolor      = "rgba(0,0,0,0)",
            bordercolor  = "rgba(0,0,0,0)",
            borderwidth  = 0,
            font         = dict(size=10, family=TICK_FAMILY, color=_t.INK_DIM),
            orientation  = "h",
            yanchor      = "bottom",
            y            = 1.04,
            xanchor      = "left",
            x            = 0,
            itemsizing   = "constant",
        ),
        xaxis = dict(
            gridcolor       = GRID_COLOR,
            gridwidth       = 1,
            zerolinecolor   = ZERO_LINE_COLOR,
            showgrid        = False,     # time needs ticks, not a lattice
            title           = dict(text=x_title or "", font=dict(size=11)),
            tickfont        = dict(size=10, family=TICK_FAMILY),
        ),
        yaxis = dict(
            gridcolor       = GRID_COLOR,
            gridwidth       = 1,
            zerolinecolor   = ZERO_LINE_COLOR,
            zerolinewidth   = 1,
            showgrid        = True,
            # The value axis sits on the RIGHT — where a terminal puts it, and
            # where the eye already is when it reaches a series' latest point.
            side            = "right",
            title           = dict(text=y_title or "", font=dict(size=11)),
            tickfont        = dict(size=10, family=TICK_FAMILY),
        ),
    )

    if title:
        layout_dict["title"] = dict(
            text    = title,
            x       = 0.01,
            xanchor = "left",
            font    = dict(size=11, color=_t.INK_DIM, family=TICK_FAMILY),
            y       = 0.985, yanchor = "top",
        )

    layout_dict.update(extra)
    return go.Layout(**layout_dict)


def empty_figure(message: str = "Insufficient data") -> go.Figure:
    """
    Return a blank Plotly figure with a centred message.
    Used when a chart cannot be rendered due to missing data.
    """
    fig = go.Figure()
    fig.add_annotation(
        text      = f"<b>{message}</b>",
        xref      = "paper", yref = "paper",
        x = 0.5,  y = 0.5,
        showarrow = False,
        font      = dict(size=14, color=NEUTRAL_COLOR),
    )
    fig.update_layout(
        paper_bgcolor = BG_PAPER,
        plot_bgcolor  = BG_PLOT,
        xaxis         = dict(visible=False),
        yaxis         = dict(visible=False),
        height        = 350,
    )
    return fig


def get_color(index: int) -> str:
    """
    Colour for the nth series.

    This used to cycle with `index % len(COLORS)`, which meant that past the end
    of the palette two different funds on one chart were handed the identical
    colour with nothing to tell them apart. Past the last slot it now returns
    the neutral grey instead — visibly "not one of the eight", which is a
    legible outcome rather than a silent collision.
    """
    return _t.series_colour(index)


def last_value_badges(fig, fmt="{:+.1f}%", min_gap_frac=0.055):
    """
    Stamp each trace's final value on the right-hand axis, in that trace's own
    colour — the one Bloomberg idea most worth copying.

    Identity travels with the number instead of living in a legend across the
    panel, so you never trace a line back to a swatch to work out which series
    ended where.

    Badges that land on top of each other are pushed apart. Two indices ending
    four-tenths of a percent apart (Smallcap +209.3 against Midcap +205.2)
    otherwise print one label directly over the other.

    Safe to call on any figure: traces with no numeric y, or fewer than two
    points, are skipped rather than raising.
    """
    import numpy as _np

    picks = []
    for tr in fig.data:
        y = getattr(tr, "y", None)
        if y is None or len(y) == 0:
            continue
        arr = _np.asarray(y, dtype="float64")
        arr = arr[_np.isfinite(arr)]
        if arr.size == 0:
            continue
        colour = (getattr(getattr(tr, "line", None), "color", None)
                  or getattr(getattr(tr, "marker", None), "color", None))
        if not isinstance(colour, str):
            continue
        picks.append([float(arr[-1]), colour])

    if not picks:
        return fig

    lo = min(v for v, _ in picks)
    hi = max(v for v, _ in picks)
    span = (hi - lo) or 1.0
    gap = span * min_gap_frac

    # Place from the bottom up, pushing any badge that crowds the one below it.
    picks.sort(key=lambda p: p[0])
    placed = []
    for value, colour in picks:
        y = value
        if placed and y < placed[-1] + gap:
            y = placed[-1] + gap
        placed.append(y)
        fig.add_annotation(
            xref="paper", x=1.005, xanchor="left",
            y=y, yanchor="middle", showarrow=False,
            text=f"<b>{fmt.format(value)}</b>",
            font=dict(family=TICK_FAMILY, size=10, color="#000000"),
            bgcolor=colour, borderpad=3,
        )
    return fig
