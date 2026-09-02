"""
formatters.py
=============
Display formatting utilities for the MF Analytics Platform.

All functions are pure — they take a raw numeric value and return a
formatted string safe for display in Streamlit tables and charts.
They NEVER raise exceptions — N/A is returned for any bad input.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union


# ─────────────────────────────────────────────────────────────────────────────
# CORE FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────

def fmt_pct(value: Optional[float], decimals: int = 2) -> str:
    """
    Format a float (0.15 → '15.00%').
    Handles None, NaN, and inf gracefully.

    Args:
        value:    Raw float fraction (e.g. 0.15 for 15%)
        decimals: Decimal places in output

    Returns:
        Formatted string like '15.23%' or 'N/A'
    """
    if value is None:
        return "N/A"
    try:
        if np.isnan(value) or np.isinf(value):
            return "N/A"
        return f"{value * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def fmt_num(value: Optional[float], decimals: int = 2) -> str:
    """
    Format a plain float number.

    Args:
        value:    Raw float
        decimals: Decimal places

    Returns:
        Formatted string like '3.14' or 'N/A'
    """
    if value is None:
        return "N/A"
    try:
        if np.isnan(value) or np.isinf(value):
            return "N/A"
        return f"{value:.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def fmt_ratio(value: Optional[float], decimals: int = 3) -> str:
    """
    Format a financial ratio (Sharpe, Sortino, Calmar, etc.).
    Uses 3 decimal places by default for precision.

    Args:
        value:    Raw float ratio
        decimals: Decimal places (default 3)

    Returns:
        Formatted string like '1.234' or 'N/A'
    """
    return fmt_num(value, decimals)


def fmt_nav(value: Optional[float]) -> str:
    """
    Format a NAV value with Indian Rupee symbol and 4 decimal places.
    e.g. 45.2381 → '₹45.2381'

    Args:
        value: NAV as float

    Returns:
        Formatted string like '₹45.2381' or 'N/A'
    """
    if value is None:
        return "N/A"
    try:
        if np.isnan(value) or np.isinf(value):
            return "N/A"
        return f"₹{value:,.4f}"
    except (TypeError, ValueError):
        return "N/A"


def fmt_days(days: Optional[Union[int, float]]) -> str:
    """
    Format a number of calendar days into a human-readable duration string.
    e.g. 400 → '1y 1m', 45 → '1m 15d', 10 → '10d'

    Args:
        days: Number of calendar days

    Returns:
        Human-readable string or 'N/A'
    """
    if days is None:
        return "N/A"
    try:
        if np.isnan(days) or np.isinf(days):
            return "N/A"
        days = int(days)
        if days <= 0:
            return "0d"
        if days < 30:
            return f"{days}d"
        elif days < 365:
            months = days // 30
            remaining_days = days % 30
            return f"{months}m {remaining_days}d"
        else:
            years = days // 365
            remaining = days % 365
            months = remaining // 30
            return f"{years}y {months}m"
    except (TypeError, ValueError):
        return "N/A"


def fmt_date(dt) -> str:
    """
    Format a date or Timestamp to 'DD-Mon-YYYY'.
    e.g. 2020-03-15 → '15-Mar-2020'

    Args:
        dt: datetime, Timestamp, or date string

    Returns:
        Formatted date string or 'N/A'
    """
    try:
        if dt is None or (isinstance(dt, float) and np.isnan(dt)):
            return "N/A"
        return pd.Timestamp(dt).strftime("%d-%b-%Y")
    except Exception:
        return "N/A"



# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT STYLING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def style_quartile(val: str) -> str:
    """
    Return a CSS style string for a quartile badge in a Streamlit dataframe.
    Used with df.style.applymap(style_quartile).

    Args:
        val: One of 'Q1', 'Q2', 'Q3', 'Q4', 'N/A'

    Returns:
        CSS string for background + text color
    """
    styles = {
        "Q1": "background-color: #1b5e20; color: #a5d6a7; font-weight: bold",
        "Q2": "background-color: #33691e; color: #c5e1a5; font-weight: bold",
        "Q3": "background-color: #e65100; color: #ffe0b2; font-weight: bold",
        "Q4": "background-color: #b71c1c; color: #ffcdd2; font-weight: bold",
        "N/A": "background-color: #263238; color: #78909c",
    }
    return styles.get(str(val), "")



# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE DISPLAY BUILDERS
# ─────────────────────────────────────────────────────────────────────────────


