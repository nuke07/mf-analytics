"""
utils/constants.py — all app-wide configuration.

Categories, keyword maps, plan/option filters, minimum-history thresholds,
metric labels, colours, and ANALYTICS_VERSION (bump it whenever metric values
or their shape change, to force cache invalidation).
"""
from typing import Dict, List

APP_TITLE: str    = "MF Quantitative Analytics"
APP_ICON: str     = ""
APP_SUBTITLE: str = "Institutional-Grade Mutual Fund Analysis · India"
APP_VERSION: str  = "1.0.0"

# phase_g2: five redundant metrics retired (see RETIRED METRICS below) and
# Rankings rebuilt from 11 tabs to 6 off a declarative spec. Cached phase_g1
# results still carry the removed keys, so they must be discarded.
ANALYTICS_VERSION: str = "phase_g2"

# The single source of truth for the RF default. This used to be 0.065 while
# every page hardcoded 7.0 in its slider (and page 9 used 6.5), so the
# constant disagreed with the UI and was referenced by nothing. utils/ui.py
# now derives the slider default from this value — change it here only.
DEFAULT_RISK_FREE_RATE: float = 0.07
TRADING_DAYS_PER_YEAR: int   = 252
MAR: float = 0.0

CATEGORIES: List[str] = [
    "Large Cap","Mid Cap","Small Cap","Flexi Cap","Multi Cap","ELSS",
    "Value","Contra","Focused","Aggressive Hybrid","Balanced Advantage","Index Funds",
]

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Large Cap":  ["large cap","bluechip","blue chip","large-cap","largecap"],
    "Mid Cap":    ["mid cap","midcap","mid-cap"],
    "Small Cap":  ["small cap","smallcap","small-cap"],
    "Flexi Cap":  ["flexi cap","flexicap","flexi-cap","flexible cap"],
    "Multi Cap":  ["multi cap","multicap","multi-cap"],
    "ELSS":       ["elss","long term equity","tax saver","taxsaver","tax saving","tax relief"],
    "Value":      ["value discovery","value fund"," value "],
    "Contra":     ["contra"],
    "Focused":    ["focused","focus fund","focussed","focus 25","focus 30"],
    "Aggressive Hybrid": ["aggressive hybrid","hybrid equity","equity hybrid","equity & debt","equity and debt"],
    "Balanced Advantage": ["balanced advantage","dynamic asset allocation","baf","dynamic equity"],
    "Index Funds": ["index fund","nifty 50 ","nifty next 50","nifty 100 ","sensex fund","nifty midcap 150","nifty smallcap"],
}
INDEX_EXCLUSIONS: List[str] = ["etf","exchange traded"]

PLAN_TYPES: List[str]      = ["Direct","Regular"]
DIRECT_KEYWORDS: List[str] = ["direct"]
REGULAR_KEYWORDS: List[str]= ["regular"]

PREFERRED_OPTIONS: List[str] = ["growth"]
EXCLUDED_PLAN_KEYWORDS: List[str] = [
    "idcw","dividend","bonus","weekly","monthly dividend",
    "quarterly dividend","annual dividend","payout","reinvestment","segregated",
]
EXCLUDED_STRUCTURE_KEYWORDS: List[str] = [
    "etf","exchange traded","fund of fund"," fof ","interval fund",
    "fixed maturity","fmp","close ended","liquid fund",
    "overnight fund","arbitrage","gilt","debt fund",
]

MIN_DAYS: Dict[str, int] = {
    "1y_cagr":365,"3y_cagr":365*3,"5y_cagr":365*5,"inception_cagr":30,
    "volatility":30,"downside_volatility":30,"max_drawdown":30,
    "avg_drawdown":30,"drawdown_duration":30,"sharpe":252,"sortino":252,
    "calmar":252,"rolling_1y":365*2,"rolling_3y":365*4,
    "skewness":30,"kurtosis":30,"win_rate":30,"streaks":30,
}

# Display labels for the MIN_DAYS coverage keys.
#
# METRIC_LABELS is keyed on ENGINE metric names ("cagr_1y"); MIN_DAYS is keyed
# on COVERAGE names ("1y_cagr"). Looking coverage keys up in METRIC_LABELS
# silently misses 8 of 18 and falls back to the raw key, so the coverage table
# used to mix "Sharpe Ratio" with "1y_cagr". Every MIN_DAYS key must appear
# here — the round-trip test in the docstring below is worth keeping in mind
# when adding a new entry to MIN_DAYS.
COVERAGE_LABELS: Dict[str, str] = {
    "1y_cagr":             "1Y CAGR",
    "3y_cagr":             "3Y CAGR",
    "5y_cagr":             "5Y CAGR",
    "inception_cagr":      "Since Inception CAGR",
    "volatility":          "Annualized Volatility",
    "downside_volatility": "Downside Volatility",
    "max_drawdown":        "Max Drawdown",
    "avg_drawdown":        "Avg Drawdown",
    "drawdown_duration":   "Drawdown Duration",
    "sharpe":              "Sharpe Ratio",
    "sortino":             "Sortino Ratio",
    "calmar":              "Calmar Ratio",
    "rolling_1y":          "1Y Rolling Returns",
    "rolling_3y":          "3Y Rolling Returns",
    "skewness":            "Skewness",
    "kurtosis":            "Kurtosis",
    "win_rate":            "Win Rate",
    "streaks":             "Win/Loss Streaks",
}

# Palettes live in utils/theme.py — the single source of truth for colour.
# These names are kept because a dozen call sites import them from here, but
# they are now aliases rather than a second, drifting definition.
from utils.theme import CHART_SERIES as _CHART_SERIES, QUARTILE as _QUARTILE

CHART_COLORS: List[str] = _CHART_SERIES
QUARTILE_COLORS: Dict[str, str] = _QUARTILE


# ─────────────────────────────────────────────────────────────────────────────
# RETIRED METRICS (phase_g2)
#
# Removed after an empirical redundancy check (765 pseudo-funds built from the
# 13 NSE TRI series, run through this repo's own metric functions —
# redundancy_probe.py). Each was near-perfectly rank-correlated with a metric
# that remains, so it could never produce a different ranking:
#
#   momentum_12m    rho = +0.9996 with cagr_1y; median difference 2 basis
#                   points. Annualising over a ~1.0-year window is a no-op, so
#                   "12M Momentum" and "1Y CAGR" were the same number ranked
#                   in two different tabs.
#   momentum_sharpe rho = +0.9735 with cagr_1y, and +0.9732 with the momentum
#                   it is derived from.
#   negative_freq   rho = -0.9997 with positive_freq — its arithmetic
#                   complement, up to flat days.
#   avg_rolling_1y  Kept median_rolling_1y / _3y instead: the median is the
#   avg_rolling_3y  more honest central estimate for a skewed return
#                   distribution, and showing both invited false precision.
#
# DELIBERATELY KEPT, against the first draft of this list:
#   calmar          rho = +0.919 with Sortino, but it is the only ratio
#                   measured against DRAWDOWN rather than volatility. That is
#                   a real conceptual distinction, and the correlation was
#                   measured on index series whose drawdowns are more alike
#                   than a real fund panel's would be. Sharpe/Sortino/Calmar
#                   now sit in one tab with guidance on when they diverge,
#                   which was the actual problem.
# ─────────────────────────────────────────────────────────────────────────────

LOWER_IS_BETTER: List[str] = [
    "annualized_volatility","downside_volatility","max_drawdown","avg_drawdown",
    "drawdown_duration","kurtosis","std_rolling_1y","std_rolling_3y",
    "worst_rolling_1y","worst_rolling_3y","down_capture","drawdown_recovery_rate",
    # Dual-benchmark counterparts of the above
    "down_capture_mkt","tracking_error_mkt",
]

METRIC_LABELS: Dict[str, str] = {
    "cagr_1y":"1Y CAGR","cagr_3y":"3Y CAGR","cagr_5y":"5Y CAGR",
    "cagr_inception":"Since Inception CAGR",
    "annualized_volatility":"Annualized Volatility","downside_volatility":"Downside Volatility",
    "max_drawdown":"Max Drawdown","avg_drawdown":"Avg Drawdown",
    "drawdown_duration":"Drawdown Duration (days)",
    "sharpe":"Sharpe Ratio","sortino":"Sortino Ratio","calmar":"Calmar Ratio",
    "sharpe_se":"Sharpe Std Error","sharpe_ci_low":"Sharpe 95% Low",
    "sharpe_ci_high":"Sharpe 95% High","sharpe_n_obs":"Sharpe Observations",
    "sharpe_acf_inflation":"Serial-Correlation Factor",
    "median_rolling_1y":"Median 1Y Rolling Return",
    "std_rolling_1y":"Std Dev 1Y Rolling Return","best_rolling_1y":"Best 1Y Rolling Return",
    "worst_rolling_1y":"Worst 1Y Rolling Return","median_rolling_3y":"Median 3Y Rolling Return",
    "std_rolling_3y":"Std Dev 3Y Rolling Return","best_rolling_3y":"Best 3Y Rolling Return",
    "worst_rolling_3y":"Worst 3Y Rolling Return",
    "skewness":"Skewness","kurtosis":"Kurtosis (Excess)",
    "positive_freq":"Positive Day Frequency",
    "win_rate":"Win Rate (Monthly)",
    "pct_positive_rolling_1y":"% Positive 1Y Rolling Periods",
    "pct_positive_rolling_3y":"% Positive 3Y Rolling Periods",
    "max_consec_positive":"Max Consecutive Positive Days",
    "max_consec_negative":"Max Consecutive Negative Days",
    "excess_return":"Excess Return (Ann.)","beta":"Beta","r_squared":"R-Squared",
    "tracking_error":"Tracking Error","information_ratio":"Information Ratio",
    "jensens_alpha":"Jensen's Alpha (Ann.)","alpha_tstat":"Alpha t-Statistic",
    "up_capture":"Up-Capture Ratio (%)","down_capture":"Down-Capture Ratio (%)",
    "capture_ratio":"Capture Ratio",
    "momentum_1m":"1M Return","momentum_3m":"3M Momentum","momentum_6m":"6M Momentum","alpha_momentum":"Alpha Momentum (12M)",
    "alpha_persistence":"Alpha Persistence Score","bull_alpha":"Bull Market Alpha",
    "bear_alpha":"Bear Market Alpha","alpha_regime_ratio":"Alpha Regime Ratio",
    "drawdown_recovery_rate":"Drawdown Recovery (days)",
    # 6-Factor model (replaced the 4F model). These are RAW betas — the
    # conventional reading, where Market 1.0 means the fund moves one-for-one
    # with Nifty 500. The Factor Attribution page shows a separate,
    # dimensionless standardised set for comparing factors against each other.
    "alpha_6f":"6-Factor Alpha (Ann.)","alpha_6f_tstat":"6-Factor Alpha t-Stat",
    "r_squared_6f":"6-Factor R-Squared",
    "beta_market_6f":"Market Beta (6F)","beta_smb":"Size Loading (SMB)",
    "beta_hml":"Value Loading (HML)","beta_wml":"Momentum Loading (WML)",
    "beta_qmj":"Quality Loading (QMJ)","beta_bab":"Low-Vol Loading (BAB)",
    "contrib_market":"Market Contribution (%)",
    "contrib_smb":"Size Contribution (%)","contrib_hml":"Value Contribution (%)",
    "contrib_wml":"Momentum Contribution (%)","contrib_qmj":"Quality Contribution (%)",
    "contrib_bab":"Low-Vol Contribution (%)",
    "contrib_alpha":"Pure Alpha Contribution (%)",
    # Dual benchmarking — the same alpha family measured against the broad
    # market (Nifty 500 TRI) instead of the SEBI category benchmark.
    "excess_return_mkt":"Excess Return vs Market (Ann.)",
    "beta_mkt":"Beta vs Market","r_squared_mkt":"R-Squared vs Market",
    "tracking_error_mkt":"Tracking Error vs Market",
    "information_ratio_mkt":"Information Ratio vs Market",
    "jensens_alpha_mkt":"Jensen's Alpha vs Market (Ann.)",
    "alpha_tstat_mkt":"Alpha t-Stat vs Market",
    "up_capture_mkt":"Up-Capture vs Market (%)",
    "down_capture_mkt":"Down-Capture vs Market (%)",
    "capture_ratio_mkt":"Capture Ratio vs Market",
}


# ─────────────────────────────────────────────────────────────────────────────
# METRIC HELP TEXT
#
# One-line plain-English explanations, surfaced as `help=` tooltips wherever a
# metric is displayed. The audit found that not one st.metric on Fund
# Analytics, Fund Comparison, Data Quality or Portfolio Analytics passed any
# help text — "Capture Ratio", "Alpha t-Stat", "QMJ" and the rest appeared
# with no explanation anywhere in the app.
#
# Keyed on engine metric names, so utils.ui.kpi(metric_key=...) can look them
# up automatically. Written for someone who knows funds but not econometrics:
# say what the number means and what a good value looks like.
# ─────────────────────────────────────────────────────────────────────────────

METRIC_HELP: Dict[str, str] = {
    # Performance
    "cagr_1y": "Annualised return over the last 1 year.",
    "cagr_3y": "Annualised return over the last 3 years.",
    "cagr_5y": "Annualised return over the last 5 years.",
    "cagr_inception": "Annualised return over the fund's full available history.",

    # Volatility & risk
    "annualized_volatility":
        "How much returns swing around, annualised. Higher means a bumpier ride, "
        "not necessarily worse returns.",
    "downside_volatility":
        "Volatility counting only losses. Ignores upside swings, which most "
        "investors don't mind.",
    "max_drawdown":
        "Worst peak-to-trough fall the fund has actually suffered. The number "
        "to ask yourself whether you could have held through.",
    "avg_drawdown": "Average size of all peak-to-trough falls.",
    "drawdown_duration":
        "Longest run of days spent below a previous peak — how long you would "
        "have waited to get back to even.",

    # Risk-adjusted
    "sharpe":
        "Return above the risk-free rate per unit of total volatility. "
        "Above 1 is good; compare only within a category.",
    "sortino":
        "Like Sharpe, but penalises only downside volatility. Higher than "
        "Sharpe means the fund's swings are mostly upward.",
    "calmar":
        "Return per unit of worst drawdown. Rewards funds that avoid deep falls.",

    # How much of the Sharpe is the sample rather than the fund
    "sharpe_se":
        "Standard error of the Sharpe ratio. Roughly 0.58 on three years of "
        "daily data, so a Sharpe of 1.00 is not reliably different from 0.5.",
    "sharpe_ci_low":
        "Lower end of the 95% confidence interval for the Sharpe ratio. If it "
        "sits below zero, the fund has not reliably beaten cash on this sample.",
    "sharpe_ci_high":
        "Upper end of the 95% confidence interval for the Sharpe ratio. Two "
        "funds whose intervals overlap cannot be ranked against each other.",
    "sharpe_n_obs":
        "Daily observations behind the Sharpe ratio. The interval narrows with "
        "the square root of this, so five years is only ~30% tighter than three.",
    "sharpe_acf_inflation":
        "How much serial correlation in the NAV widens the interval. 1.0 means "
        "none; above 1 usually means illiquid holdings priced with a lag, which "
        "flatters the Sharpe ratio and makes it less certain than it looks.",

    # Consistency & stability
    "median_rolling_1y":
        "Middle return across every 1-year window — less distorted by one "
        "exceptional year than the average.",
    "worst_rolling_1y": "Worst 1-year stretch the fund has been through.",
    "best_rolling_1y": "Best 1-year stretch the fund has been through.",
    "std_rolling_1y":
        "How much the 1-year return varies depending on when you invested. "
        "Lower means the outcome depended less on timing.",
    "pct_positive_rolling_1y":
        "Share of 1-year windows that ended positive.",
    "pct_positive_rolling_3y":
        "Share of 3-year windows that ended positive.",
    "win_rate": "Share of calendar months with a positive return.",
    "positive_freq": "Share of trading days with a positive return.",
    "skewness":
        "Asymmetry of returns. Positive means occasional large gains; "
        "negative means occasional large losses.",
    "kurtosis":
        "Fat-tailedness. Higher means extreme moves happen more often than a "
        "normal distribution would predict.",

    # Alpha vs benchmark
    "jensens_alpha":
        "Return above what the fund's benchmark exposure alone would predict. "
        "The classic measure of manager value-add.",
    "alpha_tstat":
        "Statistical confidence in that alpha. |t| ≥ 2 means it is unlikely to "
        "be luck over this sample.",
    "beta":
        "Sensitivity to the benchmark. 1.0 moves one-for-one; above 1 "
        "amplifies both gains and losses.",
    "r_squared":
        "Share of the fund's movement explained by the benchmark. Very high "
        "values on an active fund suggest closet indexing.",
    "tracking_error":
        "How far the fund's returns stray from the benchmark. Low means it "
        "hugs the index.",
    "information_ratio":
        "Excess return per unit of tracking error — reward for the active risk "
        "taken. Above 0.5 is respectable.",
    "up_capture":
        "Share of the benchmark's gains captured in up markets. Above 100% is good.",
    "down_capture":
        "Share of the benchmark's losses suffered in down markets. Below 100% "
        "is good — lower is better.",
    "capture_ratio":
        "Up-capture divided by down-capture. Above 1 means the fund keeps more "
        "of the upside than it takes of the downside.",
    "excess_return": "Annualised return above the benchmark.",

    # Momentum & persistence
    "momentum_3m": "Total return over the last 3 months.",
    "momentum_6m": "Total return over the last 6 months.",
    "alpha_momentum": "Recent trend in the fund's alpha over the last 12 months.",
    "alpha_persistence":
        "How consistently the fund has generated positive alpha, rather than "
        "producing it all in one lucky stretch.",
    "bull_alpha": "Alpha generated during rising markets.",
    "bear_alpha": "Alpha generated during falling markets.",
    "alpha_regime_ratio":
        "Bull alpha relative to bear alpha. Shows whether the manager adds "
        "value mainly in rallies or in declines.",
    "drawdown_recovery_rate":
        "Typical number of days taken to recover from a drawdown.",

    # 6-factor model
    "alpha_6f":
        "Return left over once market, size, value, momentum, quality and "
        "low-volatility exposures are all accounted for. The closest this data "
        "gets to genuine skill.",
    "alpha_6f_tstat":
        "Statistical confidence in the 6-factor alpha. |t| ≥ 2 means it is "
        "unlikely to be chance.",
    "r_squared_6f":
        "Share of the fund's returns explained by the six factors together.",
    "beta_market_6f":
        "Sensitivity to the broad market (Nifty 500). 1.0 moves one-for-one.",
    "beta_smb":
        "Size tilt. Positive means the fund leans toward smaller companies "
        "than the market.",
    "beta_hml":
        "Value tilt. Positive means a lean toward cheaper, value-style stocks.",
    "beta_wml":
        "Momentum tilt. Positive means a lean toward recent winners.",
    "beta_qmj":
        "Quality tilt (Quality-Minus-Junk). Positive means a lean toward "
        "profitable, stable, low-debt companies.",
    "beta_bab":
        "Low-volatility tilt (Betting-Against-Beta). Positive means a lean "
        "toward steadier, lower-beta stocks.",
    "contrib_alpha":
        "The share of return attributable to alpha rather than factor exposure.",

    # Dual benchmarking
    "jensens_alpha_mkt":
        "Alpha measured against the broad market (Nifty 500 TRI) instead of "
        "the SEBI category benchmark — did the fund beat simply owning the market?",
    "beta_mkt": "Sensitivity to the broad market rather than the category benchmark.",
    "information_ratio_mkt": "Information ratio measured against the broad market.",
    "capture_ratio_mkt": "Capture ratio measured against the broad market.",
    "alpha_tstat_mkt": "Statistical confidence in the market-relative alpha.",
    "r_squared_mkt": "Share of the fund's movement explained by the broad market.",
    "tracking_error_mkt": "How far the fund strays from the broad market.",
    "up_capture_mkt": "Share of the market's gains captured in up markets.",
    "down_capture_mkt": "Share of the market's losses suffered in down markets.",
    "excess_return_mkt": "Annualised return above the broad market.",
}

# Remaining engine keys — added so every metric the app can display carries a
# tooltip. utils.ui.kpi() looks these up by metric_key.
METRIC_HELP.update({
    "median_rolling_3y": "Middle return across every 3-year window.",
    "std_rolling_3y":
        "How much the 3-year return varies with entry timing. Lower is steadier.",
    "best_rolling_3y":   "Best 3-year stretch the fund has been through.",
    "worst_rolling_3y":  "Worst 3-year stretch the fund has been through.",
    "max_consec_positive": "Longest run of consecutive positive days.",
    "max_consec_negative": "Longest run of consecutive negative days.",
    "momentum_1m":       "Total return over the last 1 month.",
    "contrib_market":
        "Portion of annual return attributable to market exposure.",
    "contrib_smb":  "Portion of annual return attributable to the size tilt.",
    "contrib_hml":  "Portion of annual return attributable to the value tilt.",
    "contrib_wml":  "Portion of annual return attributable to the momentum tilt.",
    "contrib_qmj":  "Portion of annual return attributable to the quality tilt.",
    "contrib_bab":  "Portion of annual return attributable to the low-volatility tilt.",
})
