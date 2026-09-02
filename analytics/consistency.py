"""
analytics/consistency.py
========================
Rolling return statistics — measures how consistently a fund delivers returns
across different time periods, not just at a single snapshot.

Why rolling returns matter:
    A fund with a great 3-year CAGR may have delivered almost all of that
    return in one exceptional year. Rolling returns reveal whether the fund
    delivers good returns consistently across ALL 3-year windows,
    not just the particular one ending today.

Rolling Return (annualized):
    At each date t, compute the annualized CAGR over the window [t-W, t].
    This gives one data point per day, forming a distribution.

    From this distribution we compute:
        - Average rolling return (central tendency)
        - Median rolling return (robust central tendency)
        - Std dev of rolling returns (consistency — lower is more consistent)
        - Best rolling return (upside potential)
        - Worst rolling return (downside risk)

Windows computed: 1-year (252 trading days) and 3-year (756 trading days).
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from data.nav_processor import compute_rolling_returns
from utils.validators import has_sufficient_data


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING RETURN STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def calc_rolling_stats(
    nav: Optional[pd.Series],
    window_years: float,
) -> Dict[str, Optional[object]]:
    """
    Compute statistical summary of the rolling return distribution.

    Args:
        nav:          Clean daily NAV series (DatetimeIndex, ascending)
        window_years: Rolling window length in years (1.0 or 3.0)

    Returns:
        Dict with keys:
            avg     → mean annualized rolling return (float or None)
            median  → median rolling return (float or None)
            std     → std dev of rolling returns (float or None)
            best    → maximum (best) rolling return (float or None)
            worst   → minimum (worst) rolling return (float or None)
            count   → number of rolling periods computed (int)
            series  → full pd.Series of rolling returns (for charts)

    The 'series' key is included so visualizations can access the full
    distribution without recomputing rolling returns separately.
    """
    empty = {
        "avg":    None,
        "median": None,
        "std":    None,
        "best":   None,
        "worst":  None,
        "count":  0,
        "series": None,
    }

    if nav is None:
        return empty

    rolling = compute_rolling_returns(nav, window_years=window_years)
    if rolling is None or len(rolling) == 0:
        return empty

    # Remove any residual NaN/inf (should be clean from compute_rolling_returns)
    rolling = rolling.replace([np.inf, -np.inf], np.nan).dropna()

    if len(rolling) < 10:
        return empty

    return {
        "avg":    float(rolling.mean()),
        "median": float(rolling.median()),
        "std":    float(rolling.std(ddof=1)),
        "best":   float(rolling.max()),
        "worst":  float(rolling.min()),
        "count":  int(len(rolling)),
        "series": rolling,   # pd.Series — consumed by visualizations
    }


# ─────────────────────────────────────────────────────────────────────────────
# NAMED WINDOW HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def calc_rolling_1y_stats(nav: Optional[pd.Series]) -> Dict[str, Optional[object]]:
    """
    1-Year rolling return statistics.
    Requires minimum 2 years of NAV history (so rolling windows exist).
    """
    if not has_sufficient_data(nav, "rolling_1y"):
        return {
            "avg": None, "median": None, "std": None,
            "best": None, "worst": None, "count": 0, "series": None,
        }
    return calc_rolling_stats(nav, window_years=1.0)


def calc_rolling_3y_stats(nav: Optional[pd.Series]) -> Dict[str, Optional[object]]:
    """
    3-Year rolling return statistics.
    Requires minimum 4 years of NAV history.
    """
    if not has_sufficient_data(nav, "rolling_3y"):
        return {
            "avg": None, "median": None, "std": None,
            "best": None, "worst": None, "count": 0, "series": None,
        }
    return calc_rolling_stats(nav, window_years=3.0)


# ─────────────────────────────────────────────────────────────────────────────
# BATCH COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def calc_all_consistency(nav: Optional[pd.Series]) -> Dict[str, Optional[float]]:
    """
    Compute all 10 consistency metrics (5 per rolling window) in one call.

    Also returns the rolling return series for chart use under
    '_series_1y' and '_series_3y' keys (prefixed with underscore
    to distinguish from scalar metrics in the engine output).

    Args:
        nav: Clean daily NAV series

    Returns:
        Dict with metric keys (scalars) and series keys (for charts):
            median_rolling_1y, std_rolling_1y,
            best_rolling_1y, worst_rolling_1y  (same pattern for _3y)
            _series_1y, _series_3y  (pd.Series — used by visualizations)
    """
    stats_1y = calc_rolling_1y_stats(nav)
    stats_3y = calc_rolling_3y_stats(nav)

    return {
        # ── 1-Year Rolling ────────────────────────────────────────────────
        "median_rolling_1y": stats_1y["median"],
        "std_rolling_1y":    stats_1y["std"],
        "best_rolling_1y":   stats_1y["best"],
        "worst_rolling_1y":  stats_1y["worst"],

        # ── 3-Year Rolling ────────────────────────────────────────────────
        "median_rolling_3y": stats_3y["median"],
        "std_rolling_3y":    stats_3y["std"],
        "best_rolling_3y":   stats_3y["best"],
        "worst_rolling_3y":  stats_3y["worst"],

        # ── Series (for charts — NOT scalar metrics) ───────────────────────
        "_series_1y": stats_1y["series"],
        "_series_3y": stats_3y["series"],
    }
