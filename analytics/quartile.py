"""
analytics/quartile.py
=====================
Quartile assignment and category-level ranking tables.

Quartile System:
    Q1 = Best 25%   (top performers)
    Q2 = Next 25%
    Q3 = Next 25%
    Q4 = Worst 25%  (bottom performers)

    For most metrics (Sharpe, CAGR, Win Rate): higher = better → Q1 is highest.
    For risk metrics (Volatility, Drawdown):   lower  = better → Q1 is lowest.

    Determined by LOWER_IS_BETTER in utils/constants.py.
"""

import pandas as pd
from typing import Dict, List
from utils.constants import LOWER_IS_BETTER


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE METRIC QUARTILE ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def assign_quartile_labels(
    series: pd.Series,
    lower_is_better: bool = False,
) -> pd.Series:
    result = pd.Series("N/A", index=series.index, dtype=str)
    valid_mask = series.notna()
    valid = series[valid_mask]

    if len(valid) == 0:
        return result

    if len(valid) < 4:
        ranks = valid.rank(ascending=not lower_is_better, method="average")
        n = len(valid)
        for idx, rank in ranks.items():
            pct = rank / n
            if pct <= 0.25:   result[idx] = "Q1"
            elif pct <= 0.50: result[idx] = "Q2"
            elif pct <= 0.75: result[idx] = "Q3"
            else:             result[idx] = "Q4"
        return result

    q25 = float(valid.quantile(0.25))
    q50 = float(valid.quantile(0.50))
    q75 = float(valid.quantile(0.75))

    def _label(val: float) -> str:
        if lower_is_better:
            if val <= q25:   return "Q1"
            elif val <= q50: return "Q2"
            elif val <= q75: return "Q3"
            else:            return "Q4"
        else:
            if val >= q75:   return "Q1"
            elif val >= q50: return "Q2"
            elif val >= q25: return "Q3"
            else:            return "Q4"

    for idx, val in valid.items():
        result[idx] = _label(val)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# QUARTILE METRICS LIST
# Active Share Proxies removed (Phase D) — they systematically mislabel
# Indian funds as closet indexers due to structural market concentration.
# ─────────────────────────────────────────────────────────────────────────────

QUARTILE_METRICS: List[str] = [
    # Performance
    "cagr_1y", "cagr_3y", "cagr_5y", "cagr_inception",
    # Volatility
    "annualized_volatility", "downside_volatility",
    # Risk
    "max_drawdown", "avg_drawdown", "drawdown_duration",
    # Risk-Adjusted
    "sharpe", "sortino", "calmar",
    # Consistency
    "median_rolling_1y", "std_rolling_1y",
    "best_rolling_1y", "worst_rolling_1y", "median_rolling_3y", "std_rolling_3y",
    "best_rolling_3y", "worst_rolling_3y",
    # Distribution
    "skewness", "kurtosis",
    # Stability
    "positive_freq", "win_rate",
    # Persistence (rolling periods)
    "pct_positive_rolling_1y", "pct_positive_rolling_3y",
    # Alpha Generation (Phase A)
    "excess_return", "beta", "r_squared", "tracking_error",
    "information_ratio", "jensens_alpha", "alpha_tstat",
    "up_capture", "down_capture", "capture_ratio",
    # Momentum (Phase B)
    "momentum_1m", "momentum_3m", "momentum_6m",
    "alpha_momentum",
    # Alpha Persistence & Regime (Phase B)
    "alpha_persistence", "bull_alpha", "bear_alpha",
    "alpha_regime_ratio", "drawdown_recovery_rate",
    # Factor Model (6F)
    "alpha_6f", "alpha_6f_tstat", "r_squared_6f", "beta_market_6f",
    "beta_smb", "beta_hml", "beta_wml", "beta_qmj", "beta_bab",
    "contrib_market", "contrib_smb", "contrib_hml",
    "contrib_wml", "contrib_qmj", "contrib_bab", "contrib_alpha",
    # Dual benchmarking — alpha family vs the broad market
    "excess_return_mkt", "beta_mkt", "r_squared_mkt", "tracking_error_mkt",
    "information_ratio_mkt", "jensens_alpha_mkt", "alpha_tstat_mkt",
    "up_capture_mkt", "down_capture_mkt", "capture_ratio_mkt",
]


def build_metrics_dataframe(fund_metrics: Dict[str, Dict]) -> pd.DataFrame:
    rows = {}
    for fund_name, metrics in fund_metrics.items():
        rows[fund_name] = {
            key: metrics.get(key)
            for key in QUARTILE_METRICS
            if not key.startswith("_")
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def add_quartile_columns(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()
    for col in QUARTILE_METRICS:
        if col not in df.columns:
            continue
        is_lower_better = col in LOWER_IS_BETTER
        quartile_col    = f"{col}_quartile"
        numeric_series  = pd.to_numeric(df[col], errors="coerce")
        df[quartile_col] = assign_quartile_labels(
            numeric_series, lower_is_better=is_lower_better,
        )
    return df


def build_full_quartile_table(fund_metrics: Dict[str, Dict]) -> pd.DataFrame:
    metrics_df = build_metrics_dataframe(fund_metrics)
    return add_quartile_columns(metrics_df)



