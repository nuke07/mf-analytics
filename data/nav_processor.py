"""
nav_processor.py
================
NAV series processing pipeline — the numerical foundation of all analytics.

This module sits between the raw mftool data (fund_loader.py) and the
analytics engine (analytics/). It is responsible for:

  1. Cleaning raw NAV DataFrames into analysis-ready Series
  2. Computing return series (simple and log)
  3. Slicing NAV for specific time periods
  4. Aligning multiple funds to a common date range
  5. Providing summary statistics about the series

All functions are pure (no side effects, no API calls).
All functions handle None/empty inputs gracefully.
Vectorized numpy/pandas operations are used throughout.
"""

import pandas as pd
import numpy as np
from typing import Optional
from utils.constants import TRADING_DAYS_PER_YEAR


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: CLEAN RAW NAV DATA
# ─────────────────────────────────────────────────────────────────────────────

def process_nav(raw_df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    """
    Convert a raw NAV DataFrame (from fund_loader) into a clean NAV Series.

    Processing pipeline:
      1. Extract 'nav' column and ensure float dtype
      2. Ensure DatetimeIndex, sort ascending
      3. Remove duplicate dates (keep last = most recent price)
      4. Remove zero / negative NAVs (data errors)
      5. Resample to daily frequency (fills exchange-closure gaps)
      6. Forward-fill gaps up to 5 consecutive days
         (handles weekends, holidays without inventing data)
      7. Drop any remaining NaN
      8. Return None if < 10 valid data points remain

    Args:
        raw_df: DataFrame with DatetimeIndex and 'nav' float column
                (as returned by fund_loader.get_nav_history)

    Returns:
        Clean pd.Series of NAV values (float64) indexed by DatetimeIndex,
        or None if data is unusable.
    """
    if raw_df is None or raw_df.empty:
        return None

    if 'nav' not in raw_df.columns:
        return None

    nav: pd.Series = raw_df['nav'].copy()

    # Ensure DatetimeIndex
    if not isinstance(nav.index, pd.DatetimeIndex):
        try:
            nav.index = pd.to_datetime(nav.index)
        except Exception:
            return None

    # Sort ascending (oldest → newest)
    nav = nav.sort_index()

    # Remove duplicate dates — keep last (closing NAV of the day)
    nav = nav[~nav.index.duplicated(keep='last')]

    # Ensure float dtype
    nav = pd.to_numeric(nav, errors='coerce')

    # Remove zero or negative values (NAV corrections / data errors)
    nav = nav[nav > 0]

    # Resample to calendar-daily frequency, then forward-fill up to 5 days.
    # This fills weekends and public holidays (Indian market closes on ~113 days/year).
    # limit=5 prevents filling over long data gaps (e.g. fund suspension).
    nav = nav.resample('D').last()
    nav = nav.ffill(limit=5)
    nav = nav.dropna()

    if len(nav) < 10:
        return None

    return nav.astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: COMPUTE RETURN SERIES
# ─────────────────────────────────────────────────────────────────────────────

def compute_daily_returns(nav: Optional[pd.Series]) -> Optional[pd.Series]:
    """
    Compute daily simple returns from a NAV series.

    Formula: r_t = (NAV_t / NAV_{t-1}) - 1

    Simple returns are used for:
      - Win rate / frequency calculations
      - Rolling return computations
      - Sharpe / Sortino denominators (industry convention)

    Extreme values (|return| > 50%) are capped — these indicate
    NAV correction errors, not real fund performance.

    Args:
        nav: Clean NAV series (from process_nav)

    Returns:
        Series of daily simple returns (float64), or None if invalid
    """
    if nav is None or len(nav) < 2:
        return None

    returns = nav.pct_change()
    returns = returns.iloc[1:]    # Drop first NaN from pct_change
    returns = returns.dropna()

    # Cap extreme returns — 50% daily gain/loss is physically impossible
    # for an open-ended mutual fund (no leverage, daily NAV)
    returns = returns.clip(lower=-0.50, upper=0.50)

    if len(returns) == 0:
        return None

    return returns


def compute_log_returns(nav: Optional[pd.Series]) -> Optional[pd.Series]:
    """
    Compute daily log returns from a NAV series.

    Formula: r_t = ln(NAV_t / NAV_{t-1})

    Log returns are used for:
      - Skewness and kurtosis (distribution analytics)
      - Volatility estimation (log returns are approximately normal)
      - Long-horizon compounding calculations

    Args:
        nav: Clean NAV series (from process_nav)

    Returns:
        Series of daily log returns (float64), or None if invalid
    """
    if nav is None or len(nav) < 2:
        return None

    # np.log(nav / nav.shift(1)) is equivalent to np.log(nav).diff()
    # We use the ratio form to avoid log(0) issues
    shifted = nav.shift(1)
    ratio = nav / shifted
    ratio = ratio.iloc[1:]     # Drop first NaN
    ratio = ratio[ratio > 0]   # Safety check before log

    log_returns = np.log(ratio)
    log_returns = log_returns.dropna()
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()

    if len(log_returns) == 0:
        return None

    return log_returns


def compute_monthly_returns(nav: Optional[pd.Series]) -> Optional[pd.Series]:
    """
    Compute monthly simple returns by resampling NAV to month-end.

    Used for win rate calculation (monthly is less noisy than daily).

    Args:
        nav: Clean daily NAV series

    Returns:
        Series of monthly simple returns, or None if insufficient data
    """
    if nav is None or len(nav) < 60:    # Need at least 2 months
        return None

    monthly_nav = nav.resample('ME').last().dropna()

    if len(monthly_nav) < 2:
        return None

    monthly_returns = monthly_nav.pct_change().iloc[1:].dropna()
    return monthly_returns


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: TIME PERIOD SLICING
# ─────────────────────────────────────────────────────────────────────────────

def slice_nav_for_years(
    nav: Optional[pd.Series],
    years: float,
) -> Optional[pd.Series]:
    """
    Return the most recent N years of NAV data.

    Args:
        nav:   Full NAV series
        years: Number of years to look back from the most recent date

    Returns:
        Sliced NAV series or None if insufficient history
    """
    if nav is None or len(nav) == 0:
        return None

    end_date = nav.index[-1]
    start_date = end_date - pd.DateOffset(years=years)

    sliced = nav[nav.index >= start_date]

    # Need at least 10 points to be useful
    if len(sliced) < 10:
        return None

    return sliced




# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: MULTI-FUND ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def align_nav_series(
    nav_dict: dict,
) -> dict:
    """
    Align multiple NAV series to their common overlapping date range.

    Used in Fund Comparison — ensures all funds are evaluated over the
    SAME time period so comparisons are apples-to-apples.

    Args:
        nav_dict: {fund_name_or_code: pd.Series} mapping

    Returns:
        {fund_name_or_code: aligned_pd.Series} — all series have
        identical DatetimeIndex spanning their common date range.
        Funds with no overlap are excluded.
    """
    if not nav_dict:
        return {}

    # Filter out None series
    valid = {k: v for k, v in nav_dict.items() if v is not None and len(v) > 0}

    if len(valid) == 0:
        return {}

    if len(valid) == 1:
        return valid

    # Find common date range (intersection of all series)
    common_idx = valid[list(valid.keys())[0]].index
    for series in valid.values():
        common_idx = common_idx.intersection(series.index)

    if len(common_idx) < 30:
        # Not enough common trading days — return each series unaligned
        # (callers should check this)
        return valid

    # Reindex each series to the common index (forward-fill any gaps)
    aligned = {}
    for key, series in valid.items():
        aligned[key] = series.reindex(common_idx).ffill()

    return aligned


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: SERIES SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def get_series_summary(
    nav: Optional[pd.Series],
    fund_name: str = "",
) -> dict:
    """
    Compute a quick summary of a NAV series — used on the Data Quality page.

    Args:
        nav:       Clean NAV series
        fund_name: Display name (just passed through)

    Returns:
        Dict with metadata about the series:
          - fund_name, start_date, end_date
          - history_years, data_points
          - current_nav, inception_nav, total_return
          - missing_day_pct (how many calendar days lack a NAV)
    """
    if nav is None or len(nav) == 0:
        return {
            "fund_name": fund_name,
            "start_date": None,
            "end_date": None,
            "history_years": 0.0,
            "data_points": 0,
            "current_nav": None,
            "inception_nav": None,
            "total_return": None,
            "missing_day_pct": 100.0,
        }

    start = nav.index[0]
    end = nav.index[-1]
    total_calendar_days = max((end - start).days, 1)

    # Expected trading days ≈ total_calendar_days * (252/365)
    expected_trading_days = total_calendar_days * (TRADING_DAYS_PER_YEAR / 365)
    missing_pct = max(0.0, (1 - len(nav) / expected_trading_days) * 100)

    return {
        "fund_name": fund_name,
        "start_date": start,
        "end_date": end,
        "history_years": round(total_calendar_days / 365.25, 1),
        "data_points": len(nav),
        "current_nav": float(nav.iloc[-1]),
        "inception_nav": float(nav.iloc[0]),
        "total_return": float((nav.iloc[-1] / nav.iloc[0]) - 1),
        "missing_day_pct": round(missing_pct, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: ROLLING WINDOWS
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling_returns(
    nav: Optional[pd.Series],
    window_years: float,
) -> Optional[pd.Series]:
    """
    Compute annualized rolling returns for a given window.

    At each date t, the rolling return is the annualized CAGR of the fund
    over the window ending at t.

    Formula: r_t = (NAV_t / NAV_{t - window}) ^ (1 / window_years) - 1

    This is computed by taking a rolling product of (1 + daily_return)
    over the window, then annualizing.

    Args:
        nav:          Clean daily NAV series
        window_years: Rolling window size in years (e.g. 1.0 or 3.0)

    Returns:
        Series of annualized rolling returns (one per day),
        or None if insufficient history
    """
    if nav is None or len(nav) == 0:
        return None

    # Window in trading days
    window_days = int(window_years * TRADING_DAYS_PER_YEAR)

    if len(nav) < window_days + 30:   # Need enough history beyond the window
        return None

    daily_returns = compute_daily_returns(nav)
    if daily_returns is None:
        return None

    # Rolling product of (1 + daily_return) over window
    # = cumulative return over the window
    gross_returns = (1 + daily_returns).rolling(window=window_days, min_periods=window_days)
    rolling_gross = gross_returns.apply(np.prod, raw=True)

    # Annualize: (cumulative_gross) ^ (252 / window_days) - 1
    annualized = rolling_gross ** (TRADING_DAYS_PER_YEAR / window_days) - 1

    # Drop leading NaN (from the rolling window warmup period)
    annualized = annualized.dropna()

    if len(annualized) < 10:
        return None

    return annualized
