"""
analytics/monte_carlo.py
========================
Block bootstrap Monte Carlo simulation for mutual fund NAV paths.

Why bootstrap instead of parametric simulation?
  Parametric Monte Carlo assumes normally distributed returns. Indian mutual
  fund returns exhibit significant skewness and fat tails (kurtosis > 3),
  which normal distribution underestimates. Bootstrap resamples from the
  actual empirical distribution — fat tails, skewness, and all — without
  making any distributional assumption.

Why block bootstrap instead of IID bootstrap?
  Daily returns are not independent. Volatility clusters (high-vol days
  follow high-vol days). IID bootstrap destroys this structure. Block
  bootstrap draws contiguous blocks (default: 21 trading days = 1 month),
  preserving short-term autocorrelation and volatility clustering.

Risk of overfitting: None.
  This is pure resampling — no model parameters are estimated or fitted.
  The simulation reflects exactly what the historical distribution looked like.

Limitation:
  The past distribution may not reflect future conditions. The simulation
  cannot predict structural breaks, regime changes, or events outside the
  historical window.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from utils.constants import TRADING_DAYS_PER_YEAR


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL: Path generation
# ─────────────────────────────────────────────────────────────────────────────

def _block_bootstrap(
    returns:      np.ndarray,
    horizon_days: int,
    n_sims:       int,
    block_size:   int,
    rng:          np.random.Generator,
) -> Optional[np.ndarray]:
    """
    Generate (n_sims, horizon_days) array of simulated daily returns
    via block bootstrap.

    Algorithm:
        For each simulation:
          1. Determine number of blocks needed to cover horizon_days
          2. Draw random block start positions uniformly
          3. Concatenate blocks, trim to exactly horizon_days
    """
    n_obs     = len(returns)
    max_start = n_obs - block_size

    if max_start < 1:
        return None

    n_blocks = int(np.ceil(horizon_days / block_size))
    paths    = np.empty((n_sims, horizon_days), dtype=np.float64)

    for sim in range(n_sims):
        starts = rng.integers(0, max_start, size=n_blocks)
        drawn  = np.concatenate([returns[s : s + block_size] for s in starts])
        paths[sim] = drawn[:horizon_days]

    return paths


def _nav_from_returns(paths: np.ndarray, initial_nav: float) -> np.ndarray:
    """
    Convert return paths → NAV paths.
    nav[t] = initial_nav × ∏(1 + r_i) for i = 1..t
    """
    return np.cumprod(1.0 + paths, axis=1) * initial_nav


def _max_drawdown_vectorized(nav_paths: np.ndarray) -> np.ndarray:
    """
    Vectorised maximum drawdown for all simulated paths.

    Returns array of shape (n_sims,) with negative values
    (e.g. -0.30 = 30% drawdown).
    """
    running_max = np.maximum.accumulate(nav_paths, axis=1)
    drawdowns   = (nav_paths - running_max) / running_max
    return drawdowns.min(axis=1)   # most negative per path


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL: Statistics
# ─────────────────────────────────────────────────────────────────────────────

def _path_percentiles(
    nav_paths:   np.ndarray,
    percentiles: list,
) -> Dict:
    """Percentile bands across simulated NAV paths (shape: n_sims × horizon)."""
    return {p: np.percentile(nav_paths, p, axis=0) for p in percentiles}


def _terminal_statistics(
    nav_paths:    np.ndarray,
    initial_nav:  float,
    horizon_years:float,
    cl_levels:    list,
) -> Dict:
    """
    Summary statistics at the terminal horizon across all simulated paths.

    All returns expressed as decimals (e.g. 0.08 = 8%).
    VaR and CVaR expressed as positive decimals (loss magnitudes).
    """
    terminal_navs    = nav_paths[:, -1]
    terminal_returns = terminal_navs / initial_nav - 1.0
    ann_returns      = (terminal_navs / initial_nav) ** (1.0 / horizon_years) - 1.0

    # Probability of loss (terminal NAV < starting NAV)
    prob_loss = float((terminal_navs < initial_nav).mean())

    # VaR and CVaR on terminal total return
    var_cvar = {}
    for cl in cl_levels:
        alpha       = 1.0 - cl
        var_return  = float(np.percentile(terminal_returns, alpha * 100))
        tail_mask   = terminal_returns <= var_return
        cvar_return = float(terminal_returns[tail_mask].mean()) if tail_mask.any() else var_return
        var_cvar[cl] = {
            "var":  -min(var_return,  0.0),   # positive loss magnitude
            "cvar": -min(cvar_return, 0.0),
        }

    # Terminal return percentiles
    return_pcts = {
        p: float(np.percentile(terminal_returns * 100, p))
        for p in [5, 10, 25, 50, 75, 90, 95]
    }

    # ── Drawdown statistics ───────────────────────────────────────────────
    # max_drawdowns are NEGATIVE (e.g. -0.32 = a 32% peak-to-trough fall).
    #
    # Percentiles taken directly on negative values run backwards against
    # intuition: a HIGHER percentile of a negative series is the MILDER
    # outcome. Reporting "P90 max drawdown" off np.percentile(dd, 90) would
    # therefore understate the tail — the severe tail of a negative series
    # sits at percentile 5, not 95.
    #
    # To remove that trap entirely we convert to SEVERITY MAGNITUDES first
    # (positive, larger = worse) and take percentiles on those. Percentile p
    # now means: "p% of simulated paths had a drawdown no worse than this."
    # Callers must render these as-is — no abs(), no sign flip.
    max_drawdowns = _max_drawdown_vectorized(nav_paths)   # negative decimals
    dd_severity   = -max_drawdowns                        # positive decimals

    dd_pcts = {
        p: float(np.percentile(dd_severity * 100, p))
        for p in [5, 10, 25, 50, 75, 90, 95, 99]
    }

    # Drawdown at Risk (95%): the severity that only 5% of paths exceed.
    dar_95 = float(np.percentile(dd_severity, 95))        # positive magnitude

    # Probability of exceeding drawdown thresholds
    dd_probs = {}
    for threshold in [0.10, 0.20, 0.30, 0.40, 0.50]:
        dd_probs[threshold] = float((max_drawdowns < -threshold).mean())

    return {
        "mean_ann_return":      float(ann_returns.mean()),
        "median_ann_return":    float(np.median(ann_returns)),
        "prob_loss":            prob_loss,
        "var_cvar":             var_cvar,
        "return_percentiles":   return_pcts,
        # Positive severity magnitudes in %. Percentile p = "p% of paths had a
        # drawdown no worse than this". Render directly; do NOT call abs().
        "max_dd_percentiles":   dd_pcts,
        # Positive severity magnitude (decimal). Exceeded by 5% of paths.
        "drawdown_at_risk_95":  dar_95,
        "dd_exceed_probs":      dd_probs,
        "max_drawdowns":        max_drawdowns,   # negative decimals, for histogram
        "terminal_returns":     terminal_returns, # full array for histogram
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(
    returns:       pd.Series,
    horizon_years: float = 3.0,
    n_sims:        int   = 10_000,
    initial_nav:   float = 100.0,
    block_size:    int   = 21,
    seed:          int   = 42,
    cl_levels:     list  = [0.90, 0.95, 0.99],
    percentiles:   list  = [5, 10, 25, 50, 75, 90, 95],
) -> Dict:
    """
    Full block-bootstrap Monte Carlo simulation pipeline.

    Args:
        returns:       Daily simple returns (decimal, e.g. 0.01 = 1%)
        horizon_years: Simulation horizon in years
        n_sims:        Number of simulated paths (10,000 recommended)
        initial_nav:   Starting NAV (100 for relative analysis)
        block_size:    Block length in days (21 = 1 month, preserves clustering)
        seed:          Random seed for reproducibility
        cl_levels:     Confidence levels for VaR/CVaR
        percentiles:   Percentiles to compute for fan chart

    Returns dict with keys:
        is_valid         — bool
        nav_percentiles  — {percentile: array of shape (horizon_days,)}
        nav_mean         — array of shape (horizon_days,)
        terminal_stats   — dict with return distribution and drawdown stats
        horizon_days     — int
        horizon_years    — float
        n_sims           — int
        initial_nav      — float
    """
    empty = {"is_valid": False, "error": ""}

    clean = returns.dropna()
    if len(clean) < 252:
        empty["error"] = "Minimum 252 trading days required."
        return empty

    if horizon_years <= 0:
        empty["error"] = "Horizon must be positive."
        return empty

    horizon_days = int(horizon_years * TRADING_DAYS_PER_YEAR)
    r_arr        = clean.values
    rng          = np.random.default_rng(seed)

    paths = _block_bootstrap(r_arr, horizon_days, n_sims, block_size, rng)
    if paths is None:
        empty["error"] = "Insufficient history for block bootstrap."
        return empty

    nav_paths      = _nav_from_returns(paths, initial_nav)
    nav_pcts       = _path_percentiles(nav_paths, percentiles)
    terminal_stats = _terminal_statistics(nav_paths, initial_nav, horizon_years, cl_levels)

    return {
        "is_valid":        True,
        "nav_percentiles": nav_pcts,
        "nav_mean":        nav_paths.mean(axis=0),
        "terminal_stats":  terminal_stats,
        "horizon_days":    horizon_days,
        "horizon_years":   horizon_years,
        "n_sims":          n_sims,
        "initial_nav":     initial_nav,
    }
