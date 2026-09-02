"""
analytics/performance.py
========================
Performance metric calculations — CAGR for fixed periods and since inception.

Formula:
    CAGR = (End NAV / Start NAV) ^ (1 / actual_years) - 1

Design notes:
  - We use ACTUAL elapsed years from the sliced NAV dates, not the nominal
    window. This avoids over/understating returns when the exact anniversary
    date falls on a non-trading day.
  - All functions return None (not NaN, not 0) when data is insufficient.
    The engine and UI treat None as "metric not available".
  - Vectorized where possible; no Python loops.
"""

import pandas as pd
from typing import Optional, Dict
from data.nav_processor import slice_nav_for_years
from utils.constants import MIN_DAYS
from utils.validators import has_sufficient_data


# ─────────────────────────────────────────────────────────────────────────────
# CORE CAGR ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _cagr_from_endpoints(
    start_nav: float,
    end_nav: float,
    actual_years: float,
) -> Optional[float]:
    """
    Compute CAGR given start NAV, end NAV, and holding period in years.
    Returns None for any invalid input.
    """
    if start_nav <= 0 or end_nav <= 0 or actual_years < 0.1:
        return None
    try:
        return float((end_nav / start_nav) ** (1.0 / actual_years) - 1.0)
    except (ZeroDivisionError, ValueError, OverflowError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL CAGR FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def calc_cagr(nav: Optional[pd.Series], years: float) -> Optional[float]:
    """
    CAGR over the most recent N years.

    Slices the NAV to the last `years` years, then computes CAGR between
    the first and last available points in that slice.

    Args:
        nav:   Clean daily NAV series (DatetimeIndex, ascending)
        years: Lookback window in years (e.g. 1.0, 3.0, 5.0)

    Returns:
        CAGR as a decimal fraction (e.g. 0.1523 = 15.23%), or None.
    """
    if nav is None or len(nav) < 2:
        return None

    sliced = slice_nav_for_years(nav, years)
    if sliced is None or len(sliced) < 10:
        return None

    actual_years = (sliced.index[-1] - sliced.index[0]).days / 365.25
    return _cagr_from_endpoints(
        start_nav=float(sliced.iloc[0]),
        end_nav=float(sliced.iloc[-1]),
        actual_years=actual_years,
    )


def calc_cagr_1y(nav: Optional[pd.Series]) -> Optional[float]:
    """1-Year CAGR. Requires at least 365 calendar days of history."""
    if not has_sufficient_data(nav, "1y_cagr"):
        return None
    return calc_cagr(nav, years=1.0)


def calc_cagr_3y(nav: Optional[pd.Series]) -> Optional[float]:
    """3-Year CAGR. Requires at least 3 * 365 calendar days of history."""
    if not has_sufficient_data(nav, "3y_cagr"):
        return None
    return calc_cagr(nav, years=3.0)


def calc_cagr_5y(nav: Optional[pd.Series]) -> Optional[float]:
    """5-Year CAGR. Requires at least 5 * 365 calendar days of history."""
    if not has_sufficient_data(nav, "5y_cagr"):
        return None
    return calc_cagr(nav, years=5.0)


def calc_cagr_inception(nav: Optional[pd.Series]) -> Optional[float]:
    """
    Since-Inception CAGR — from the very first NAV point to the most recent.

    This is the most robust CAGR figure since it does not depend on a
    specific lookback window.

    Args:
        nav: Full clean NAV series from fund inception

    Returns:
        Annualized CAGR from inception, or None.
    """
    if nav is None or len(nav) < 10:
        return None

    total_days = (nav.index[-1] - nav.index[0]).days
    if total_days < MIN_DAYS["inception_cagr"]:
        return None

    actual_years = total_days / 365.25
    return _cagr_from_endpoints(
        start_nav=float(nav.iloc[0]),
        end_nav=float(nav.iloc[-1]),
        actual_years=actual_years,
    )


# ─────────────────────────────────────────────────────────────────────────────
# BATCH COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def calc_all_cagr(nav: Optional[pd.Series]) -> Dict[str, Optional[float]]:
    """
    Compute all four CAGR metrics for a fund in a single call.

    Args:
        nav: Clean daily NAV series

    Returns:
        Dict with keys: cagr_1y, cagr_3y, cagr_5y, cagr_inception
        Each value is a float or None.

    Example:
        >>> metrics = calc_all_cagr(nav_series)
        >>> metrics['cagr_3y']   # e.g. 0.1523 (15.23%)
    """
    return {
        "cagr_1y":        calc_cagr_1y(nav),
        "cagr_3y":        calc_cagr_3y(nav),
        "cagr_5y":        calc_cagr_5y(nav),
        "cagr_inception": calc_cagr_inception(nav),
    }
