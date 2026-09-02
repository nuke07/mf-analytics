"""
analytics/factor_model.py
==========================
Factor model analytics for mutual fund return attribution.

One model, six factors (Market, SMB, HML, WML, QMJ, BAB):
    calc_all_factor_model_6f()     <- entry point called by engine.py
    calc_factor_model_6f()         <- full 6F OLS, standardised and raw betas
    calc_rolling_factor_betas()    <- rolling window betas per factor
    calc_regime_betas()            <- betas split by market regime

The 4-factor Carhart tier (calc_factor_model / calc_all_factor_model) was
deleted once every caller had moved to 6F; nothing in the app regressed on
four factors any more. The one survivor with a 4F name,
_calc_rolling_alpha_4f, is factor-count agnostic and is still what produces
the rolling-alpha series for the 6F model.

Standardisation note (6F model only):
    Factors are pre-scaled to zero mean, unit variance using full-sample
    statistics before OLS. Standardised betas are directly comparable:
    a QMJ beta of 0.4 and SMB beta of 0.2 unambiguously means a stronger
    Quality tilt than Size tilt regardless of raw factor volatilities.

    Raw betas (unstandardised) are also computed and stored separately
    for return attribution: contribution_k = raw_beta_k × mean(factor_k) × 252
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from utils.constants import TRADING_DAYS_PER_YEAR, DEFAULT_RISK_FREE_RATE


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL OLS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _ols_with_stats(
    Y: np.ndarray,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    OLS regression with standard errors and R².

    Returns:
        betas:   coefficient array [intercept, b1, b2, ...]
        t_stats: t-statistic per coefficient
        r2:      coefficient of determination
    """
    n, k     = X.shape
    XtX      = X.T @ X
    betas    = np.linalg.lstsq(X, Y, rcond=None)[0]
    Y_hat    = X @ betas
    residuals= Y - Y_hat
    ss_res   = float(residuals @ residuals)
    ss_tot   = float(((Y - Y.mean()) ** 2).sum())
    r2       = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    dof      = max(n - k, 1)
    sigma_sq = ss_res / dof
    try:
        var_betas = sigma_sq * np.linalg.inv(XtX)
        se        = np.sqrt(np.maximum(np.diag(var_betas), 0.0))
    except np.linalg.LinAlgError:
        se = np.full_like(betas, np.nan)
    t_stats = np.where(se > 0, betas / se, np.nan)
    return betas, t_stats, r2


# ─────────────────────────────────────────────────────────────────────────────
# 4-FACTOR MODEL  (used by engine.py — interface unchanged)
# ─────────────────────────────────────────────────────────────────────────────


def _calc_rolling_alpha_4f(
    fund_returns:       pd.Series,
    factor_df:          pd.DataFrame,
    factors_available:  list,
    rf_rate:            float,
    window:             int = TRADING_DAYS_PER_YEAR,
) -> Optional[pd.Series]:
    try:
        rf_daily = rf_rate / TRADING_DAYS_PER_YEAR
        common   = fund_returns.dropna().index.intersection(
            factor_df.dropna(how="any").index
        )
        excess = (fund_returns.reindex(common) - rf_daily)
        fdata  = factor_df[factors_available].reindex(common)

        if len(common) < window + 30:
            return None

        alphas, dates = [], []
        for end in range(window, len(common)):
            sl = slice(end - window, end)
            ef = excess.iloc[sl].values
            xf = fdata.iloc[sl].values
            X  = np.column_stack([np.ones(window), xf])
            valid = (~np.isnan(ef)) & (~np.isnan(xf).any(axis=1))
            if valid.sum() < window * 0.8:
                continue
            try:
                b, _, _ = _ols_with_stats(ef[valid], X[valid])
                if np.isfinite(b[0]):
                    alphas.append(b[0] * TRADING_DAYS_PER_YEAR)
                    dates.append(common[end])
            except Exception:
                continue

        return pd.Series(alphas, index=dates) if alphas else None
    except Exception:
        return None



# ─────────────────────────────────────────────────────────────────────────────
# 6-FACTOR MODEL  (used by Factor Attribution page)
# ─────────────────────────────────────────────────────────────────────────────

FACTOR_6F_NAMES = ["market", "smb", "hml", "wml", "qmj", "bab"]

_EMPTY_6F_ENGINE: Dict = {
    "alpha_6f": None, "alpha_6f_tstat": None, "r_squared_6f": None,
    "beta_market_6f": None, "beta_smb": None, "beta_hml": None,
    "beta_wml": None, "beta_qmj": None, "beta_bab": None,
    "contrib_market": None, "contrib_smb": None, "contrib_hml": None,
    "contrib_wml": None, "contrib_qmj": None, "contrib_bab": None,
    "contrib_alpha": None,
    "factors_used": [], "_rolling_alpha_6f": None,
}


def calc_all_factor_model_6f(
    fund_returns: Optional[pd.Series],
    factor_df:    Optional[pd.DataFrame],
    rf_rate:      float = DEFAULT_RISK_FREE_RATE,
) -> Dict:
    """
    Entry point called by engine.py — the 6-factor model in engine key shape.

    This replaced the 4F entry point at the 6F migration; the 4F functions
    have since been deleted, so this is the only factor model in the app.

    Key mapping:
      - beta_{factor}      → the RAW beta, so that "market beta 0.95" carries
        its conventional meaning in rankings and the All Metrics list. Raw
        betas are comparable ACROSS FUNDS because every fund is regressed on
        the same factor series. The standardised betas — comparable across
        DIFFERENT factors — stay on the Factor Attribution page.
      - contrib_{factor}   → computed from the RAW beta, since a return
        attribution has to be in return units:
        contribution_k = raw_beta_k × mean(factor_k) × 252
      - alpha_6f           → the RAW intercept, annualised. See the long note
        in calc_factor_model_6f explaining why it must not come from the
        standardised fit.

    Identity worth remembering: alpha_6f + Σ contrib_{factor} equals the
    fund's realised annualised excess return over rf.
    """
    if fund_returns is None or factor_df is None or factor_df.empty:
        return dict(_EMPTY_6F_ENGINE)

    core = calc_factor_model_6f(fund_returns, factor_df, rf_rate)
    if core.get("alpha_6f") is None:
        return dict(_EMPTY_6F_ENGINE)

    result: Dict = {
        "alpha_6f":       core.get("alpha_6f"),
        "alpha_6f_tstat": core.get("alpha_6f_tstat"),
        "r_squared_6f":   core.get("r_squared_6f"),
        "contrib_alpha":  core.get("alpha_6f"),
        "factors_used":   list(FACTOR_6F_NAMES),
    }

    for fname in FACTOR_6F_NAMES:
        key = "beta_market_6f" if fname == "market" else f"beta_{fname}"
        # RAW betas for the engine, not standardised ones. Rankings tables and
        # the All Metrics list are read with the conventional interpretation —
        # "market beta 0.95", "SMB loading 0.60" — and a raw beta is directly
        # comparable across funds because every fund is regressed on the same
        # factor series. The dimensionless standardised betas are for
        # comparing DIFFERENT factors against each other, which is what the
        # Factor Attribution page's grid does; they stay there.
        result[key]                 = core.get(f"beta_{fname}_raw")
        result[f"contrib_{fname}"]  = core.get(f"contrib_{fname}")

    # Rolling alpha, on RAW factors — the helper is factor-count agnostic.
    available = [f for f in FACTOR_6F_NAMES if f in factor_df.columns]
    result["_rolling_alpha_6f"] = _calc_rolling_alpha_4f(
        fund_returns, factor_df, available, rf_rate
    )
    return result


def calc_factor_model_6f(
    fund_returns: pd.Series,
    factor_df:   pd.DataFrame,
    rf_rate:     float = DEFAULT_RISK_FREE_RATE,
) -> Dict:
    """
    6-Factor OLS regression with both standardised and raw betas.

    Standardised betas: factor_df pre-scaled to zero mean, unit variance.
        → Directly comparable across factors and funds.
        → A beta of 0.5 means 0.5 standard deviation tilt in that factor.

    Raw betas: OLS on original (unstandardised) factor returns.
        → Used for return contribution calculation.
        → contribution_k = raw_beta_k × mean(factor_k) × 252

    Args:
        fund_returns: Daily simple returns (decimal)
        factor_df:    DataFrame with columns market/smb/hml/wml/qmj/bab
        rf_rate:      Annual risk-free rate

    Returns dict with keys:
        alpha_6f, alpha_6f_tstat
        r_squared_6f
        beta_{factor}_std     → standardised beta per factor
        beta_{factor}_raw     → raw OLS beta per factor
        tstat_{factor}        → t-statistic per factor
        contrib_{factor}      → annualised return contribution per factor (%)
        contrib_alpha_6f      → alpha contribution
        effective_start       → first common date (pd.Timestamp)
        effective_end         → last common date (pd.Timestamp)
        n_obs                 → number of observations used
    """
    empty = {
        "alpha_6f": None, "alpha_6f_tstat": None, "r_squared_6f": None,
        "n_obs": 0, "effective_start": None, "effective_end": None,
        **{f"beta_{f}_std": None for f in FACTOR_6F_NAMES},
        **{f"beta_{f}_raw": None for f in FACTOR_6F_NAMES},
        **{f"tstat_{f}": None   for f in FACTOR_6F_NAMES},
        **{f"contrib_{f}": None for f in FACTOR_6F_NAMES},
        "contrib_alpha_6f": None,
    }

    try:
        required = [f for f in FACTOR_6F_NAMES if f in factor_df.columns]
        if len(required) != 6:
            return empty

        rf_daily = rf_rate / TRADING_DAYS_PER_YEAR
        common   = fund_returns.dropna().index.intersection(
            factor_df[required].dropna(how="any").index
        )
        if len(common) < 252:
            return empty

        Y       = (fund_returns.reindex(common) - rf_daily).values
        F_raw   = factor_df[required].reindex(common)
        F_means = F_raw.mean()
        F_stds  = F_raw.std()
        F_std   = (F_raw - F_means) / F_stds

        X_std = np.column_stack([np.ones(len(common)), F_std.values])
        X_raw = np.column_stack([np.ones(len(common)), F_raw.values])

        b_std, t_std, r2   = _ols_with_stats(Y, X_std)
        b_raw, t_raw, _    = _ols_with_stats(Y, X_raw)

        # ── Standardised betas must divide by the DEPENDENT variable's SD too ─
        #
        # Standardising only the regressors leaves b_std_k = b_raw_k · σ_k,
        # which still carries the units of Y — a daily return. Those come out
        # around 0.01, so the documented reading ("a loading of 1.0 means one
        # full standard-deviation tilt") could never be true of them.
        #
        # A proper standardised (beta) coefficient divides by σ_Y as well:
        #
        #     b_beta_k = b_raw_k · σ_k / σ_Y
        #
        # That is dimensionless and reads as "a one-SD move in this factor
        # moves the fund b_beta_k standard deviations" — comparable across
        # factors of different volatility AND across funds.
        sigma_Y = float(np.std(Y, ddof=1))
        if sigma_Y > 0:
            b_std = np.asarray(b_std, dtype=float).copy()
            b_std[1:] = b_std[1:] / sigma_Y

        # ── Alpha MUST come from the RAW fit, never the standardised one ─────
        #
        # Standardising the regressors DOES move the intercept. Centering each
        # factor by its mean shifts the intercept by the sum of the factor risk
        # premia the fund is exposed to:
        #
        #     a_std = a_raw + Σ_k ( b_raw_k · mean(factor_k) )
        #
        # With every regressor centred, the OLS intercept collapses to simply
        # mean(Y) — the fund's average excess return. So b_std[0] is not alpha
        # at all: it is total excess return over the risk-free rate, and it
        # reported every fund as a huge alpha generator. The falsifying case:
        # Nifty 500 IS the market factor, so its alpha must be zero, yet this
        # returned +8.45% at R² = 1.000 (its own excess return).
        #
        # The standardised BETAS remain correct and are still what we want for
        # cross-factor comparability — only the intercept was wrong.
        # R² is unaffected: standardisation is an affine reparameterisation.
        alpha_ann = float(b_raw[0] * TRADING_DAYS_PER_YEAR)

        result = {
            "alpha_6f":       alpha_ann,
            "alpha_6f_tstat": float(t_raw[0]),
            "r_squared_6f":   float(r2),
            "n_obs":          len(common),
            "effective_start":common[0],
            "effective_end":  common[-1],
            "contrib_alpha_6f": alpha_ann,
        }

        for i, fname in enumerate(required):
            result[f"beta_{fname}_std"] = float(b_std[i + 1])
            result[f"beta_{fname}_raw"] = float(b_raw[i + 1])
            result[f"tstat_{fname}"]    = float(t_std[i + 1])
            # Return contribution = raw_beta × mean_factor_return × 252
            result[f"contrib_{fname}"]  = float(
                b_raw[i + 1] * F_means[fname] * TRADING_DAYS_PER_YEAR
            )

        return result

    except Exception as exc:
        import logging
        logging.getLogger(__name__).error(f"6F model failed: {exc}")
        return empty


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING FACTOR BETAS  (for Tab 2 of Factor Attribution page)
# ─────────────────────────────────────────────────────────────────────────────

def calc_rolling_factor_betas(
    fund_returns: pd.Series,
    factor_df:   pd.DataFrame,
    rf_rate:     float = DEFAULT_RISK_FREE_RATE,
    window_days: int   = TRADING_DAYS_PER_YEAR,
) -> Optional[pd.DataFrame]:
    """
    Compute rolling standardised factor betas for all 6 factors.

    Standardisation is done ONCE on the full sample, then applied to
    each window — ensuring beta=1.0 means the same across all time points.

    Args:
        fund_returns: Daily returns (decimal)
        factor_df:    6-column factor DataFrame
        rf_rate:      Annual risk-free rate
        window_days:  Rolling window length (63 / 126 / 252)

    Returns:
        pd.DataFrame(index=dates, columns=factor_names)
        or None if insufficient data.
    """
    try:
        required = [f for f in FACTOR_6F_NAMES if f in factor_df.columns]
        if len(required) < 1:
            return None

        rf_daily = rf_rate / TRADING_DAYS_PER_YEAR
        common   = fund_returns.dropna().index.intersection(
            factor_df[required].dropna(how="any").index
        )
        if len(common) < window_days + 30:
            return None

        excess = (fund_returns.reindex(common) - rf_daily)

        # Standardise ONCE on full sample
        F_raw  = factor_df[required].reindex(common)
        F_means= F_raw.mean()
        F_stds = F_raw.std()
        F_std  = (F_raw - F_means) / F_stds

        # Divide by the FULL-SAMPLE σ of the dependent variable, matching
        # calc_factor_model_6f, so these betas are dimensionless and share one
        # fixed scale across every window. Using each window's own σ_Y would
        # make the line move when the fund's volatility changes even if its
        # factor exposure did not — the opposite of what this chart is for.
        sigma_Y = float(excess.std(ddof=1))
        if not sigma_Y or not np.isfinite(sigma_Y) or sigma_Y <= 0:
            sigma_Y = 1.0

        rolling_betas = []
        rolling_dates = []

        for end in range(window_days, len(common)):
            sl = slice(end - window_days, end)
            ef = excess.iloc[sl].values
            xf = F_std.iloc[sl].values

            valid = (~np.isnan(ef)) & (~np.isnan(xf).any(axis=1))
            if valid.sum() < window_days * 0.8:
                continue

            try:
                X = np.column_stack([np.ones(valid.sum()), xf[valid]])
                b, _, _ = _ols_with_stats(ef[valid], X)
                if np.all(np.isfinite(b[1:])):
                    rolling_betas.append(b[1:] / sigma_Y)
                    rolling_dates.append(common[end])
            except Exception:
                continue

        if len(rolling_betas) < 10:
            return None

        return pd.DataFrame(
            rolling_betas,
            index  = rolling_dates,
            columns= required,
        )

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# REGIME-CONDITIONAL BETAS  (for Tab 4 of Factor Attribution page)
# ─────────────────────────────────────────────────────────────────────────────

def calc_regime_betas(
    fund_returns: pd.Series,
    factor_df:   pd.DataFrame,
    rf_rate:     float = DEFAULT_RISK_FREE_RATE,
    min_obs:     int   = 60,
) -> Dict:
    """
    Estimate standardised factor betas separately for each market regime.

    Regime classification from Nifty 500 returns (market factor + rf):
        Bull:     rolling 252-day annualised return > 10%
        Sideways: 0% – 10%
        Bear:     < 0%

    Args:
        fund_returns: Daily returns
        factor_df:    6-column factor DataFrame
        rf_rate:      Annual risk-free rate
        min_obs:      Minimum days per regime for reliable estimation

    Returns dict:
        {regime_name: {factor_name: standardised_beta, ...}}
        Regime is None if fewer than min_obs days available.
    """
    result = {"Bull": None, "Sideways": None, "Bear": None,
              "regime_counts": {}, "regime_dates": {}}
    try:
        required = [f for f in FACTOR_6F_NAMES if f in factor_df.columns]
        if len(required) < 1 or "market" not in required:
            return result

        rf_daily = rf_rate / TRADING_DAYS_PER_YEAR
        common   = fund_returns.dropna().index.intersection(
            factor_df[required].dropna(how="any").index
        )
        if len(common) < 252 + min_obs:
            return result

        # Nifty500 total return = market_factor + rf_daily
        nifty500_ret = factor_df["market"].reindex(common) + rf_daily

        # 252-day rolling annualised return
        rolling_ann  = nifty500_ret.rolling(252).mean() * TRADING_DAYS_PER_YEAR

        # Classify regimes
        regimes = pd.cut(
            rolling_ann,
            bins   = [-np.inf, 0.0, 0.10, np.inf],
            labels = ["Bear", "Sideways", "Bull"],
        )

        excess = (fund_returns.reindex(common) - rf_daily)
        F_raw  = factor_df[required].reindex(common)
        F_means= F_raw.mean()
        F_stds = F_raw.std()
        sigma_Y_full = float(excess.std(ddof=1))
        if not sigma_Y_full or not np.isfinite(sigma_Y_full) or sigma_Y_full <= 0:
            sigma_Y_full = 1.0
        F_std  = (F_raw - F_means) / F_stds

        for regime_name in ["Bull", "Sideways", "Bear"]:
            mask = (regimes == regime_name) & regimes.notna()
            n    = int(mask.sum())
            result["regime_counts"][regime_name] = n
            result["regime_dates"][regime_name]  = (
                common[mask][0], common[mask][-1]
            ) if n > 0 else None

            if n < min_obs:
                result[regime_name] = None
                continue

            ef = excess[mask].values
            xf = F_std[mask].values
            X      = np.column_stack([np.ones(n), xf])
            X_raw  = np.column_stack([np.ones(n), F_raw[mask].values])

            try:
                b, t, r2       = _ols_with_stats(ef, X)
                b_raw, t_raw, _= _ols_with_stats(ef, X_raw)
                # Same full-sample σ_Y scaling as calc_factor_model_6f, so
                # regime betas sit on the same dimensionless scale as the
                # full-period ones and can be read side by side.
                result[regime_name] = {
                    fname: {"beta": float(b[i+1] / sigma_Y_full),
                            "tstat": float(t[i+1])}
                    for i, fname in enumerate(required)
                }
                # Alpha from the RAW fit — see the note in calc_factor_model_6f.
                # Standardising shifts the intercept by the factor risk premia
                # the fund is exposed to, which turns "alpha" into total excess
                # return. Note the standardisation here uses FULL-SAMPLE means,
                # so within a regime the centred regressors do not have zero
                # mean and the bias is Σ b_k · mean_full(factor_k) — still
                # wrong, just less obviously so.
                result[regime_name]["alpha"]       = float(b_raw[0] * TRADING_DAYS_PER_YEAR)
                result[regime_name]["alpha_tstat"] = float(t_raw[0])
                result[regime_name]["r2"]    = float(r2)
                result[regime_name]["n_obs"] = n
            except Exception:
                result[regime_name] = None

    except Exception:
        pass

    return result
