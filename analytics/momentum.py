"""
analytics/momentum.py
======================
Momentum metrics — 1M / 3M / 6M / 12M point-in-time returns,
alpha momentum, and risk-adjusted momentum (Momentum Sharpe).
"""

import pandas as pd
from typing import Optional, Dict
from utils.constants import DEFAULT_RISK_FREE_RATE


def _momentum_for_months(nav: Optional[pd.Series], months: int) -> Optional[float]:
    """Simple return from N calendar months ago to most recent NAV date."""
    if nav is None or len(nav) < 20:
        return None
    end_date   = nav.index[-1]
    start_date = end_date - pd.DateOffset(months=months)
    past_nav   = nav[nav.index <= start_date]
    if len(past_nav) == 0:
        return None
    start_val = float(past_nav.iloc[-1])
    end_val   = float(nav.iloc[-1])
    if start_val <= 0:
        return None
    return float((end_val / start_val) - 1)


def calc_momentum_1m(nav: Optional[pd.Series]) -> Optional[float]:
    """1-Month return — used for short-term absolute return rankings."""
    return _momentum_for_months(nav, months=1)


def calc_momentum_3m(nav: Optional[pd.Series]) -> Optional[float]:
    """3-Month return."""
    return _momentum_for_months(nav, months=3)


def calc_momentum_6m(nav: Optional[pd.Series]) -> Optional[float]:
    """6-Month return."""
    return _momentum_for_months(nav, months=6)



def calc_alpha_momentum(
    fund_returns:      Optional[pd.Series],
    benchmark_returns: Optional[pd.Series],
    rf_rate:           float = DEFAULT_RISK_FREE_RATE,
    months:            int = 12,
) -> Optional[float]:
    """
    Alpha Momentum — Jensen's alpha over the last N months only.
    Removes market component from momentum signal.
    """
    if fund_returns is None or benchmark_returns is None:
        return None
    end_date   = min(fund_returns.index[-1], benchmark_returns.index[-1])
    start_date = end_date - pd.DateOffset(months=months)
    f = fund_returns[(fund_returns.index >= start_date) & (fund_returns.index <= end_date)]
    b = benchmark_returns[(benchmark_returns.index >= start_date) & (benchmark_returns.index <= end_date)]
    if len(f) < 30 or len(b) < 30:
        return None
    from analytics.alpha import calc_jensens_alpha
    return calc_jensens_alpha(f, b, rf_rate=rf_rate)



def calc_all_momentum(
    nav:               Optional[pd.Series],
    fund_returns:      Optional[pd.Series] = None,
    benchmark_returns: Optional[pd.Series] = None,
    rf_rate:           float = DEFAULT_RISK_FREE_RATE,
) -> Dict[str, Optional[float]]:
    """Compute all momentum metrics in one call."""
    return {
        "momentum_1m":     calc_momentum_1m(nav),
        "momentum_3m":     calc_momentum_3m(nav),
        "momentum_6m":     calc_momentum_6m(nav),
        "alpha_momentum":  calc_alpha_momentum(
            fund_returns, benchmark_returns, rf_rate
        ) if fund_returns is not None and benchmark_returns is not None else None,
    }
