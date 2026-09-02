"""
data/factor_loader.py
======================
Constructs factor return series for Fama-French style models.

Two public functions:
    get_factor_returns_6f(rf_rate)     → 4-factor model (used by engine.py)
    get_factor_returns_6f(rf_rate)  → 6-factor model (used by Factor Attribution page)

Factor definitions:
    Market  =  Nifty500 TRI return  −  rf_daily
    SMB     =  Nifty Smallcap250 TRI  −  Nifty100 TRI
    HML     =  Nifty500 Value50 TRI   −  Nifty500 TRI
    WML     =  Nifty200 Momentum30 TRI −  Nifty500 TRI
    QMJ     =  Nifty200 Quality30 TRI  −  Nifty500 TRI   [6F only]
    BAB     =  Nifty100 LowVol30 TRI   −  Nifty100 TRI   [6F only]

Data strategy (TRI-first):
    1. Try true TRI from data/tri/ via tri_loader.get_tri_nav()
    2. Fall back to Direct Growth index fund NAV from mftool
    QMJ and BAB have no index fund proxies — TRI only.

6F model requires all 6 factors. If any TRI file is missing for
QMJ or BAB, get_factor_returns_6f() returns None.
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, List, Tuple
import logging
from utils import theme as T

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY NAMES
# ─────────────────────────────────────────────────────────────────────────────

FACTOR_DISPLAY_NAMES: Dict[str, str] = {
    "market": "Market (Mkt-Rf)",
    "smb":    "Size (SMB)",
    "hml":    "Value (HML)",
    "wml":    "Momentum (WML)",
    "qmj":    "Quality (QMJ)",
    "bab":    "Low Vol (BAB)",
}

# One colour per factor, drawn from the shared series palette so a factor's
# colour in the attribution chart is the same colour it has everywhere else.
# Deliberately no green or red here: HML being green would read as "value is
# the good factor" rather than as a neutral identity.
FACTOR_COLORS: Dict[str, str] = {
    "market": T.CHART_SERIES[0],   # cyan
    "smb":    T.CHART_SERIES[1],   # ember
    "hml":    T.CHART_SERIES[2],   # violet
    "wml":    T.CHART_SERIES[3],   # steel
    "qmj":    T.CHART_SERIES[4],   # magenta
    "bab":    T.CHART_SERIES[7],   # slate
    "alpha":  T.INK,               # alpha is the residual, not a factor
}

# ─────────────────────────────────────────────────────────────────────────────
# PROXY SEARCH KEYWORDS (fallback for 4F factors)
# ─────────────────────────────────────────────────────────────────────────────

FACTOR_PROXY_KEYWORDS: Dict[str, List[str]] = {
    "market": [
        "nifty 500 index fund", "nifty500 index fund", "nifty 500 ",
    ],
    "small": [
        "nifty smallcap 250 index fund", "smallcap 250 index fund", "nifty smallcap 250",
    ],
    "large": [
        "nifty 100 index fund", "nifty100 index fund", "nifty 100 ",
    ],
    "value": [
        "nifty 500 value 50", "value 50 index", "nifty value 50", "nifty500 value",
    ],
    "momentum": [
        "nifty 200 momentum 30", "momentum 30 index", "nifty200 momentum", "nifty 200 momentum",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL: TRI LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _load_tri_nav_series(index_name: str) -> Optional[pd.Series]:
    """
    Load a TRI series as a clean pd.Series via tri_loader → process_nav.
    Returns None if the CSV is unavailable or fails to parse.
    """
    try:
        from data.tri_loader import get_tri_nav
        from data.nav_processor import process_nav
        nav_df = get_tri_nav(index_name)
        if nav_df is None:
            return None
        return process_nav(nav_df)
    except Exception as exc:
        logger.warning(f"TRI load failed for '{index_name}': {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL: PROXY LOADING (fallback)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _find_proxy_scheme(factor_role: str) -> Optional[Dict]:
    keywords = FACTOR_PROXY_KEYWORDS.get(factor_role, [])
    if not keywords:
        return None
    from data.fund_loader import get_all_schemes
    all_schemes = get_all_schemes()
    if not all_schemes:
        return None
    exclusions = ["etf", "exchange traded", "idcw", "dividend", "regular"]
    candidates = {
        code: name for code, name in all_schemes.items()
        if "direct" in name.lower()
        and "growth" in name.lower()
        and not any(e in name.lower() for e in exclusions)
    }
    for kw in keywords:
        for code, name in candidates.items():
            if kw in name.lower():
                return {"code": code, "name": name}
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def _load_proxy_nav(factor_role: str) -> Optional[pd.Series]:
    scheme = _find_proxy_scheme(factor_role)
    if scheme is None:
        return None
    from data.fund_loader import get_nav_history
    from data.nav_processor import process_nav
    nav_df = get_nav_history(scheme["code"])
    if nav_df is None:
        return None
    return process_nav(nav_df)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL: TRI-FIRST COMPONENT LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _load_component(tri_index: str, proxy_role: Optional[str] = None) -> Optional[pd.Series]:
    """
    Load a factor component NAV series.
    Tries TRI first; if unavailable and proxy_role given, falls back to proxy.
    """
    nav = _load_tri_nav_series(tri_index)
    if nav is not None:
        return nav
    if proxy_role:
        logger.info(f"TRI unavailable for {tri_index}, using proxy ({proxy_role})")
        return _load_proxy_nav(proxy_role)
    return None


def _compute_spread(
    long_nav:  pd.Series,
    short_nav: pd.Series,
    min_common: int = 60,
) -> Optional[pd.Series]:
    """
    Compute long_return − short_return over common dates.
    Returns None if insufficient common history.
    """
    from data.nav_processor import compute_daily_returns
    long_ret  = compute_daily_returns(long_nav)
    short_ret = compute_daily_returns(short_nav)
    common = long_ret.index.intersection(short_ret.index)
    if len(common) < min_common:
        return None
    return long_ret.reindex(common) - short_ret.reindex(common)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: 4-FACTOR MODEL (used by engine.py — unchanged interface)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: 6-FACTOR MODEL (used by Factor Attribution page)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_factor_returns_6f(
    rf_rate: float = 0.065,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Optional[str]], Optional[str]]:
    """
    Construct daily factor returns for the 6-factor model.
    ALL 6 factors required — returns None if any factor is unavailable.

    QMJ and BAB are TRI-only (no index fund proxy exists).

    Returns:
        (factor_df, source_names, error_message)
        factor_df columns: market, smb, hml, wml, qmj, bab
        error_message: None on success, string describing failure
    """
    from data.nav_processor import compute_daily_returns

    rf_daily = rf_rate / 252

    # ── Load all components ──────────────────────────────────────────────────
    market_nav   = _load_component("NIFTY 500",                  proxy_role="market")
    small_nav    = _load_component("NIFTY SMALLCAP 250",         proxy_role="small")
    large_nav    = _load_component("NIFTY 100",                  proxy_role="large")
    value_nav    = _load_component("NIFTY500 VALUE 50",          proxy_role="value")
    momentum_nav = _load_component("NIFTY200 MOMENTUM 30",       proxy_role="momentum")
    quality_nav  = _load_tri_nav_series("NIFTY200 QUALITY 30")   # TRI only
    lowvol_nav   = _load_tri_nav_series("NIFTY100 LOW VOLATILITY 30")  # TRI only

    # ── Validate all 6 are available ─────────────────────────────────────────
    missing = []
    if market_nav   is None: missing.append("Market (NIFTY 500)")
    if small_nav    is None: missing.append("Small (NIFTY SMALLCAP 250)")
    if large_nav    is None: missing.append("Large (NIFTY 100)")
    if value_nav    is None: missing.append("Value (NIFTY500 VALUE 50)")
    if momentum_nav is None: missing.append("Momentum (NIFTY200 MOMENTUM 30)")
    if quality_nav  is None: missing.append("Quality (NIFTY200 QUALITY 30) — TRI required")
    if lowvol_nav   is None: missing.append("Low Vol (NIFTY100 LOW VOLATILITY 30) — TRI required")

    if missing:
        err = (
            "6-Factor model requires all 6 factor series. Missing: "
            + ", ".join(missing)
            + ". Run python -m scripts.update_indices to refresh TRI data."
        )
        return None, {}, err

    # ── Build factor series ───────────────────────────────────────────────────
    mkt_ret = compute_daily_returns(market_nav)

    smb = _compute_spread(small_nav,    large_nav)
    hml = _compute_spread(value_nav,    market_nav)
    wml = _compute_spread(momentum_nav, market_nav)
    qmj = _compute_spread(quality_nav,  market_nav)
    bab = _compute_spread(lowvol_nav,   large_nav)   # BAB vs Nifty100 universe

    spread_missing = []
    if smb is None: spread_missing.append("SMB")
    if hml is None: spread_missing.append("HML")
    if wml is None: spread_missing.append("WML")
    if qmj is None: spread_missing.append("QMJ")
    if bab is None: spread_missing.append("BAB")

    if spread_missing:
        err = f"Insufficient common history to compute: {', '.join(spread_missing)}"
        return None, {}, err

    factor_df = pd.DataFrame({
        "market": mkt_ret - rf_daily,
        "smb":    smb,
        "hml":    hml,
        "wml":    wml,
        "qmj":    qmj,
        "bab":    bab,
    }).dropna()

    if len(factor_df) < 252:
        return None, {}, "Less than 252 common trading days across all 6 factors."

    source_names = {
        "market": "NIFTY 500 TRI" if _load_tri_nav_series("NIFTY 500") is not None else "proxy",
        "smb":    "NIFTY SMALLCAP 250 TRI − NIFTY 100 TRI",
        "hml":    "NIFTY500 VALUE 50 TRI − NIFTY 500 TRI",
        "wml":    "NIFTY200 MOMENTUM 30 TRI − NIFTY 500 TRI",
        "qmj":    "NIFTY200 QUALITY 30 TRI − NIFTY 500 TRI",
        "bab":    "NIFTY100 LOW VOLATILITY 30 TRI − NIFTY 100 TRI",
    }

    logger.info(f"6F returns: {list(factor_df.columns)}, {len(factor_df)} days, "
                f"{factor_df.index[0].strftime('%d %b %Y')} → {factor_df.index[-1].strftime('%d %b %Y')}")

    return factor_df, source_names, None


# ─────────────────────────────────────────────────────────────────────────────
# AVAILABILITY CHECKS (used by UI)
# ─────────────────────────────────────────────────────────────────────────────


def get_factor_availability_6f() -> Dict[str, bool]:
    """Which 6F factors are available. Used in Factor Attribution page."""
    factor_df, _, err = get_factor_returns_6f()
    if factor_df is None:
        return {f: False for f in ["market", "smb", "hml", "wml", "qmj", "bab"]}
    return {f: f in factor_df.columns for f in ["market", "smb", "hml", "wml", "qmj", "bab"]}
