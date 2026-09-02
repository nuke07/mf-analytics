"""
analytics/stability.py
======================
Stability metrics — how often a fund generates positive vs negative returns.

These are frequency-based metrics that complement the magnitude-based
metrics like volatility and drawdown. A fund can have low volatility
but still be negative more often than positive.

Positive Return Frequency:
    Fraction of trading days where daily return > 0.
    Formula: count(r_t > 0) / total_trading_days

    A fund with 55%+ positive days is broadly stable.
    Most large-cap equity funds are around 52–56%.

Negative Return Frequency:
    Fraction of trading days where daily return < 0.
    Formula: count(r_t < 0) / total_trading_days

    Note: negative_freq was retired — it is the arithmetic complement of
    positive_freq (up to flat days) and could never rank funds differently.

Win Rate (Monthly):
    Fraction of calendar months where the month-end NAV > month-start NAV.
    Monthly data is used (instead of daily) to smooth out noise and give a
    more meaningful picture of consistent performance.

    Formula: count(monthly_return > 0) / total_months

    Win Rate ≥ 60% is considered strong.
    Win Rate < 45% indicates the fund is frequently negative month-to-month.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict


# ─────────────────────────────────────────────────────────────────────────────
# POSITIVE RETURN FREQUENCY
# ─────────────────────────────────────────────────────────────────────────────

def calc_positive_freq(returns: Optional[pd.Series]) -> Optional[float]:
    """
    Fraction of trading days with a strictly positive return.

    Args:
        returns: Daily simple return series

    Returns:
        Float between 0 and 1, or None if insufficient data.
        e.g. 0.534 means the fund was positive on 53.4% of trading days.
    """
    if returns is None or len(returns) < 10:
        return None

    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) == 0:
        return None

    return float((clean > 0).sum() / len(clean))


# ─────────────────────────────────────────────────────────────────────────────
# NEGATIVE RETURN FREQUENCY
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# WIN RATE (MONTHLY)
# ─────────────────────────────────────────────────────────────────────────────

def calc_win_rate(monthly_returns: Optional[pd.Series]) -> Optional[float]:
    """
    Fraction of calendar months with a positive total return.

    Uses month-end to month-end returns so each month is weighted equally
    regardless of the number of trading days it contains.

    Args:
        monthly_returns: Monthly simple return series
                         (from compute_monthly_returns in nav_processor)

    Returns:
        Float between 0 and 1, or None if fewer than 3 months available.
        e.g. 0.65 means the fund was positive in 65% of months.
    """
    if monthly_returns is None or len(monthly_returns) < 3:
        return None

    clean = monthly_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3:
        return None

    return float((clean > 0).sum() / len(clean))


# ─────────────────────────────────────────────────────────────────────────────
# BATCH COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def calc_all_stability(
    returns: Optional[pd.Series],
    monthly_returns: Optional[pd.Series],
) -> Dict[str, Optional[float]]:
    """
    Compute all three stability metrics in one call.

    Args:
        returns:         Daily simple return series
        monthly_returns: Monthly simple return series

    Returns:
        Dict with keys: positive_freq, win_rate
    """
    return {
        "positive_freq": calc_positive_freq(returns),
        "win_rate":      calc_win_rate(monthly_returns),
    }
