"""
analytics/uncertainty.py
========================
How much of a Sharpe ratio is signal, and how much is the sample.

Why this module exists
----------------------
The app was already careful about one number and silent about another. Jensen's
alpha ships with a standard error and a t-statistic, and utils/constants.py even
tells the user that "|t| >= 2 means it is unlikely to be luck". Sharpe, Sortino
and Calmar — the metrics the Rankings page actually sorts on — shipped as bare
point estimates to two decimal places, with nothing to say how much of the gap
between rank 3 and rank 7 was real.

It usually is not. A Sharpe ratio estimated on three years of daily data carries
a standard error near 0.58. A fund showing 1.00 has a 95% interval that includes
zero. Sorting forty such funds and presenting the order as information overstates
what the data supports.

The estimator
-------------
Lo (2002), "The Statistics of Sharpe Ratios", Financial Analysts Journal 58(4),
36-52. For IID returns the asymptotic standard error of a Sharpe ratio estimated
from n observations is

    SE(SR) = sqrt( (1 + SR^2 / 2) / n )                                    (Lo eq. 9)

with SR and SE both on the sampling frequency's own scale. Annualising the ratio
by sqrt(periods_per_year) scales the standard error by the same factor, so the
interval annualises with the estimate.

Fund NAV returns are not IID. Stale pricing and illiquid holdings induce positive
serial correlation, which makes the naive annualised Sharpe too flattering and its
IID standard error too small. Lo's section on non-IID returns gives the correction:
the annualisation factor is not sqrt(q) but

    eta(q) = q / sqrt( q + 2 * sum_{k=1}^{q-1} (q - k) * rho_k )           (Lo eq. 19)

and the sampling variance inflates by the same serial-correlation structure. Both
are implemented here; `adjust_autocorrelation=True` is the default because for
Indian funds the correction is not decorative.

What this is not
----------------
It is not a significance test on skill. A Sharpe interval that excludes zero says
the fund beat cash by more than sampling noise, not that the manager is good — see
claude/PREDICTABILITY_RESEARCH.md for why those are different claims.

KNOWN DEFECT IN THE POINT ESTIMATE THIS MODULE DECORATES
--------------------------------------------------------
The Sharpe ratio these intervals wrap is currently understated by roughly 44%,
and the cause is upstream of this file.

data/nav_processor.process_nav() resamples NAV onto a CALENDAR-day grid and
forward-fills, so a 21-year series becomes 7,812 rows covering 5,306 trading
days. analytics.risk_adjusted.calc_sharpe() then annualises that series by
sqrt(252) and subtracts rf/252 on every one of the 365 days a year, which
over-charges the risk-free rate by 1.47x. Measured against the same indices
computed on their published trading days:

    Nifty 100            app 0.283   correct 0.501   ratio 0.566
    Nifty Midcap 150     app 0.318   correct 0.545   ratio 0.583
    Nifty Smallcap 250   app 0.269   correct 0.481   ratio 0.560

Sortino carries the same defect for the same reason. The consistency of the
ratio across indices is the signature of a systematic scaling error, not noise.
Because the factor is nearly constant it mostly preserves RANK, which is why it
has gone unnoticed, but it does not preserve rank exactly — funds with different
NAV reporting gaps get different amounts of padding — and every absolute figure
shown to a user is wrong. The app's own help text says "above 1 is good", which
almost nothing can reach at a 44% discount.

Not fixed here on purpose: process_nav and compute_daily_returns have 36 call
sites, and correcting the grid moves every Sharpe, Sortino, Calmar, volatility
and quartile in the product. That is a deliberate change with its own test pass,
not a side effect of adding a confidence interval. What this module does do is
refuse to compound it — see effective_observations().
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.constants import DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR

# Two-sided normal quantiles. The Sharpe estimator is asymptotically normal;
# at the sample sizes here (hundreds of daily observations) the normal
# approximation is the one Lo derives and is what the literature reports.
_Z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}

# Below this many observations the asymptotics are not worth quoting. Roughly
# one year of daily data.
MIN_OBS_FOR_INTERVAL = 250

# Lags used for the serial-correlation correction. One trading month: long
# enough for the stale-pricing effect in illiquid holdings, short enough that
# the sample autocorrelations still carry information rather than noise.
DEFAULT_ACF_LAGS = 21

# A standard deviation at or below this is treated as zero. Testing `sd == 0.0`
# is not enough — a constant series returns a float like 1e-19 rather than an
# exact zero, and dividing by that produced a Sharpe ratio of 1.06e+17.
_MIN_SD = 1e-12


def _z(confidence: float) -> float:
    if confidence in _Z:
        return _Z[confidence]
    raise ValueError(f"confidence must be one of {sorted(_Z)}, got {confidence}")


def _clean_returns(returns: Optional[pd.Series]) -> Optional[pd.Series]:
    if returns is None or len(returns) == 0:
        return None
    s = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    return s if len(s) >= 2 else None


def autocorrelation_inflation(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    max_lag: Optional[int] = None,
) -> float:
    """
    The variance inflation factor implied by serial correlation in `returns`.

    Returns  q / eta(q)^2  =  ( q + 2 * sum (q-k) rho_k ) / q, i.e. the factor by
    which the true sampling variance of the annualised Sharpe exceeds the IID
    figure. A value of 1.0 means the IID assumption holds; above 1.0 means the
    naive interval is too narrow and the naive annualised Sharpe too high.

    Estimated as a Newey-West HAC variance factor with Bartlett weights:

        VIF = 1 + 2 * sum_{k=1..m} (1 - k/(m+1)) * rho_k

    with m from the standard Newey-West rule, m = floor(4 * (n/100)^(2/9)).

    Why not Lo's eq. 19 literally. That formula weights rho_k by (q - k), which
    for daily data means multiplying each sample autocorrelation by roughly 240
    and summing 251 of them. A sample rho_k has a standard deviation near
    1/sqrt(n), so the weighting amplifies noise rather than signal: fed pure IID
    returns, the literal form returned 1.006, 1.077 and 1.403 on three draws of
    the same process, and ranked AR(1) with phi=0.05 BELOW phi=0.0. It is the
    right expression for annualising the point estimate and the wrong one for
    the variance of that estimate, which is a HAC problem and has a HAC answer.

    Bartlett weights are bounded by 1 and taper to zero, so the estimator is
    consistent and its sampling noise is roughly a quarter of the naive form's.
    Each rho_k still carries the usual -1/n small-sample bias correction.

    Clamped to [0.25, 4.0] as a last guard against a pathological series.
    """
    s = _clean_returns(returns)
    if s is None:
        return 1.0

    n = len(s)
    if max_lag is not None:
        m = int(max_lag)
    else:
        m = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    m = int(min(m, periods_per_year - 1, max(1, n // 10)))
    if m < 1:
        return 1.0

    x = s.to_numpy(dtype=float)
    x = x - x.mean()
    denom = float(x @ x)
    if denom <= 0:
        return 1.0

    acc = 0.0
    for k in range(1, m + 1):
        rho_k = float(x[:-k] @ x[k:]) / denom + 1.0 / n   # small-sample bias
        acc += (1.0 - k / (m + 1.0)) * rho_k

    factor = 1.0 + 2.0 * acc
    if not np.isfinite(factor):
        return 1.0
    return float(min(4.0, max(0.25, factor)))


def effective_observations(returns: Optional[pd.Series]) -> int:
    """
    How many of these returns are actual observations.

    data/nav_processor.process_nav() resamples NAV to a CALENDAR-day grid and
    forward-fills weekends and holidays, so about 32% of the "daily returns"
    reaching this module are exactly zero because the market was shut, not
    because the fund stood still. A 21-year series arrives with 7,812 rows over
    5,306 real trading days.

    Counting the padded rows would divide the sampling variance by 1.47x too
    much and hand back an interval roughly 21% narrower than the data supports —
    a confidence interval that is itself overconfident. Exact zeros are counted
    out instead. A genuinely flat trading day is rare enough in an equity NAV
    quoted to four decimals that the error runs the safe way.

    NOTE: the same padding also distorts the Sharpe POINT estimate, which this
    module does not touch — see the note at the top of this file.
    """
    s = _clean_returns(returns)
    if s is None:
        return 0
    n_nonzero = int((s.abs() > 1e-12).sum())
    # If a series really is mostly flat (some liquid/overnight debt funds),
    # fall back to the full length rather than reporting almost no data.
    return n_nonzero if n_nonzero >= 0.25 * len(s) else int(len(s))


def sharpe_standard_error(
    sharpe_annual: float,
    n_obs: int,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    inflation: float = 1.0,
) -> Optional[float]:
    """
    Standard error of an ANNUALISED Sharpe ratio (Lo 2002, eq. 9).

    The formula is defined on the sampling frequency, so the annualised ratio is
    converted back down, the IID standard error taken there, and the result
    scaled up again — which is why `periods_per_year` appears twice and does not
    cancel.

        SE(SR_ann) = sqrt(ppy) * sqrt( (1 + SR_period^2 / 2) / n ) * sqrt(inflation)

    `inflation` is the serial-correlation variance factor from
    autocorrelation_inflation(); leave it at 1.0 for the IID figure.
    """
    if sharpe_annual is None or n_obs is None or n_obs < 2:
        return None
    if not np.isfinite(sharpe_annual):
        return None

    ppy = float(periods_per_year)
    sr_period = float(sharpe_annual) / np.sqrt(ppy)
    var_period = (1.0 + 0.5 * sr_period ** 2) / float(n_obs)
    se = np.sqrt(ppy) * np.sqrt(var_period) * np.sqrt(max(inflation, 0.0))
    return float(se) if np.isfinite(se) else None


def sharpe_interval(
    returns: Optional[pd.Series],
    rf_rate: float = DEFAULT_RISK_FREE_RATE,
    confidence: float = 0.95,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    adjust_autocorrelation: bool = True,
) -> Dict[str, Optional[float]]:
    """
    Sharpe ratio with its confidence interval, from the same inputs as
    analytics.risk_adjusted.calc_sharpe so the two can never disagree.

    Returns a dict with:
        sharpe        annualised point estimate (matches calc_sharpe)
        sharpe_se     standard error of that estimate
        sharpe_ci_low / sharpe_ci_high
        sharpe_n_obs  observations the estimate rests on
        sharpe_acf_inflation  variance inflation from serial correlation

    Every value is None when the series is too short to say anything honest.
    """
    empty: Dict[str, Optional[float]] = {
        "sharpe_se": None, "sharpe_ci_low": None, "sharpe_ci_high": None,
        "sharpe_n_obs": None, "sharpe_acf_inflation": None,
    }

    s = _clean_returns(returns)
    if s is None:
        return {"sharpe": None, **empty}

    excess = s - (rf_rate / float(periods_per_year))
    sd = float(excess.std(ddof=1))
    if not np.isfinite(sd) or sd <= _MIN_SD:
        return {"sharpe": None, **empty}

    sharpe = float(excess.mean() / sd) * np.sqrt(periods_per_year)
    if not np.isfinite(sharpe):
        return {"sharpe": None, **empty}

    n = effective_observations(s)
    if n < MIN_OBS_FOR_INTERVAL:
        # The point estimate is still reported — the app has always shown it —
        # but an interval from under a year of data would be theatre.
        return {"sharpe": sharpe, **empty, "sharpe_n_obs": n}

    inflation = (autocorrelation_inflation(excess, periods_per_year)
                 if adjust_autocorrelation else 1.0)
    se = sharpe_standard_error(sharpe, n, periods_per_year, inflation)
    if se is None:
        return {"sharpe": sharpe, **empty, "sharpe_n_obs": n}

    z = _z(confidence)
    return {
        "sharpe": sharpe,
        "sharpe_se": se,
        "sharpe_ci_low": sharpe - z * se,
        "sharpe_ci_high": sharpe + z * se,
        "sharpe_n_obs": n,
        "sharpe_acf_inflation": inflation,
    }


def difference_is_significant(
    sharpe_a: float, se_a: float,
    sharpe_b: float, se_b: float,
    correlation: float = 0.0,
    confidence: float = 0.95,
) -> bool:
    """
    Can these two Sharpe ratios be told apart?

        Var(A - B) = Var(A) + Var(B) - 2 * corr * SE(A) * SE(B)

    `correlation` is the correlation between the two funds' RETURN series, used
    as a proxy for the correlation of their Sharpe estimators. It defaults to 0,
    which is the conservative choice and deliberately so: funds inside one SEBI
    category are highly correlated, so the true variance of the difference is
    SMALLER than the independent formula gives. Passing 0 therefore overstates
    the uncertainty and declares fewer pairs distinguishable than the data
    strictly allows — a ranking tool should err that way, not the other.

    Pass the measured correlation when both series are to hand and the tighter,
    correct interval is wanted.
    """
    if None in (sharpe_a, se_a, sharpe_b, se_b):
        return False
    var = se_a ** 2 + se_b ** 2 - 2.0 * correlation * se_a * se_b
    if var <= 0:
        return False
    return abs(sharpe_a - sharpe_b) > _z(confidence) * np.sqrt(var)


def indistinguishable_bands(
    items: Sequence[Tuple[str, Optional[float], Optional[float]]],
    correlation: float = 0.0,
    confidence: float = 0.95,
) -> List[Optional[int]]:
    """
    Group a ranked list into tiers that cannot be told apart statistically.

    `items` is (name, sharpe, standard_error), in the order the table displays —
    best first. Returns a band number per item, 1 for the top tier, counting up.
    Items with no estimate get None.

    A new band opens at the first fund the current band's LEADER is significantly
    better than. Comparing against the leader rather than the previous fund stops
    a long chain of individually-insignificant steps from silently walking the
    whole table into one band.

    The point is not to hide the ordering. It is to show that ranks 1 through 9
    are frequently one tier, so a user does not switch funds over a gap that is
    noise.
    """
    bands: List[Optional[int]] = []
    band = 1
    leader: Optional[Tuple[float, float]] = None

    for _name, sharpe, se in items:
        if sharpe is None or se is None:
            bands.append(None)
            continue
        if leader is None:
            leader = (sharpe, se)
            bands.append(band)
            continue
        if difference_is_significant(leader[0], leader[1], sharpe, se,
                                     correlation=correlation,
                                     confidence=confidence):
            band += 1
            leader = (sharpe, se)
        bands.append(band)

    return bands


def band_frame(
    df: pd.DataFrame,
    name_col: str = "Fund Name",
    sharpe_col: str = "sharpe",
    se_col: str = "sharpe_se",
    correlation: float = 0.0,
    confidence: float = 0.95,
) -> pd.Series:
    """
    indistinguishable_bands() over a ranking DataFrame, returning a Series
    aligned to its index. The frame must already be sorted the way it is shown.
    """
    if df is None or df.empty or sharpe_col not in df or se_col not in df:
        return pd.Series(dtype="object")
    items = list(zip(
        df[name_col] if name_col in df else df.index,
        df[sharpe_col], df[se_col],
    ))
    return pd.Series(indistinguishable_bands(items, correlation, confidence),
                     index=df.index)
