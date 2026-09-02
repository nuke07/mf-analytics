"""
validators.py
=============
Data validation and sufficiency checks for the MF Analytics Platform.

Every analytics function should call these validators BEFORE computing
anything — this prevents division-by-zero, empty-series errors, and
misleading results from funds with insufficient history.

Functions return (is_valid: bool, warnings: List[str]) tuples so the
Streamlit UI can display contextual warnings without crashing.
"""

import pandas as pd
from typing import List, Tuple, Dict, Optional
from utils.constants import MIN_DAYS


# ─────────────────────────────────────────────────────────────────────────────
# NAV SERIES VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def check_nav_series(nav: Optional[pd.Series]) -> Tuple[bool, List[str]]:
    """
    Validate a NAV series for basic usability.

    Checks:
      - Series exists and is non-empty
      - Has a DatetimeIndex
      - Has at least 30 data points
      - Reports NaN percentage as a warning

    Args:
        nav: NAV series indexed by date

    Returns:
        (is_valid, list_of_warning_strings)
    """
    warnings: List[str] = []

    if nav is None:
        return False, ["No NAV data returned from API."]

    if not isinstance(nav, pd.Series):
        return False, ["NAV data is not a valid Series object."]

    if len(nav) == 0:
        return False, ["NAV series is empty."]

    if len(nav) < 30:
        return False, [
            f"Insufficient data: only {len(nav)} data points available. "
            f"Minimum 30 required for any calculation."
        ]

    # Check for DatetimeIndex (required for time-based slicing)
    if not isinstance(nav.index, pd.DatetimeIndex):
        return False, ["NAV index is not a DatetimeIndex — cannot perform date-based operations."]

    # Count and report NaN values (informational warning, not a blocker)
    nan_count = int(nav.isna().sum())
    if nan_count > 0:
        pct = nan_count / len(nav) * 100
        if pct > 20:
            warnings.append(
                f"High missing data: {nan_count} of {len(nav)} NAV values are NaN ({pct:.1f}%). "
                f"Results may be unreliable."
            )
        else:
            warnings.append(
                f"{nan_count} missing NAV values ({pct:.1f}%) were forward-filled."
            )

    return True, warnings


def has_sufficient_data(nav: Optional[pd.Series], metric: str) -> bool:
    """
    Check whether a NAV series has enough calendar history to compute a metric.

    Args:
        nav:    Clean NAV series indexed by DatetimeIndex
        metric: Key from MIN_DAYS (e.g. 'sharpe', '3y_cagr')

    Returns:
        True if history length ≥ required minimum
    """
    if nav is None or len(nav) == 0:
        return False

    if not isinstance(nav.index, pd.DatetimeIndex):
        return False

    if len(nav) < 2:
        return False

    required_days = MIN_DAYS.get(metric, 365)
    actual_days = (nav.index[-1] - nav.index[0]).days

    return actual_days >= required_days


def get_data_coverage(nav: Optional[pd.Series]) -> Dict[str, bool]:
    """
    Return a full coverage map — which metrics CAN be calculated for this fund.

    Args:
        nav: Clean NAV series

    Returns:
        Dict[metric_key, can_compute: bool]

    Example:
        {
          '1y_cagr': True,
          '3y_cagr': True,
          '5y_cagr': False,   ← fund not old enough
          'rolling_3y': False,
          ...
        }
    """
    return {
        metric: has_sufficient_data(nav, metric)
        for metric in MIN_DAYS.keys()
    }


def get_history_years(nav: Optional[pd.Series]) -> float:
    """
    Return the number of years of history in a NAV series.

    Args:
        nav: Clean NAV series indexed by DatetimeIndex

    Returns:
        Float years (e.g. 7.3), or 0.0 if invalid
    """
    if nav is None or len(nav) < 2:
        return 0.0
    if not isinstance(nav.index, pd.DatetimeIndex):
        return 0.0
    days = (nav.index[-1] - nav.index[0]).days
    return round(days / 365.25, 1)


# ─────────────────────────────────────────────────────────────────────────────
# RETURNS SERIES VALIDATION
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY-LEVEL VALIDATION
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE DATA QUALITY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def build_quality_report(
    fund_name: str,
    nav: Optional[pd.Series],
) -> Dict:
    """
    Build a complete data quality report for a single fund.
    Used on the Data Quality page.

    Returns a dict with:
      - fund_name
      - history_years
      - data_points
      - missing_pct
      - start_date    (pd.Timestamp or None)
      - end_date      (pd.Timestamp or None)
      - coverage (per-metric bool map)
      - warnings (list of strings)

    start_date and end_date are the first and last dates of the NAV series.
    Both callers (the Data Quality page and the Fund Analytics quality tab)
    have always read these keys; they were simply never returned, so those
    columns rendered "N/A" for every fund and exported that way to CSV.
    """
    all_warnings: List[str] = []

    nav_valid, nav_warnings = check_nav_series(nav)
    all_warnings.extend(nav_warnings)

    if not nav_valid or nav is None:
        return {
            "fund_name": fund_name,
            "history_years": 0.0,
            "data_points": 0,
            "missing_pct": 100.0,
            "start_date": None,
            "end_date": None,
            "coverage": {m: False for m in MIN_DAYS},
            "warnings": all_warnings,
        }

    # Compute data density
    total_calendar_days = (nav.index[-1] - nav.index[0]).days
    missing_pct = 0.0
    if total_calendar_days > 0:
        # We expect roughly 252/365 * total_days data points
        expected_points = total_calendar_days * (252 / 365)
        missing_pct = max(0.0, (1 - len(nav) / expected_points) * 100)

    return {
        "fund_name": fund_name,
        "history_years": get_history_years(nav),
        "data_points": len(nav),
        "missing_pct": round(missing_pct, 1),
        "start_date": nav.index[0],
        "end_date": nav.index[-1],
        "coverage": get_data_coverage(nav),
        "warnings": all_warnings,
    }
