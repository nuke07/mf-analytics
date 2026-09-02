"""
data/benchmark_loader.py
=========================
Benchmark data layer for alpha generation features.

Strategy (Phase F — TRI Integration):
    TRI-FIRST: When a validated TRI CSV exists in data/tri/, use it.
    FALLBACK:  If the CSV is missing or unreadable, fall back to the
               existing index fund NAV proxy (original behaviour).

    This makes the upgrade backward-compatible. The rest of the platform
    (engine, pages, visualizations) is completely unaware of the change —
    get_benchmark_nav() still returns a DataFrame with 'nav' column or None.

TRI → Category mapping:
    Large Cap         → NIFTY 100         (data/tri/NIFTY_100_TRI.csv)
    Mid Cap           → NIFTY MIDCAP 150  (data/tri/NIFTY_MIDCAP_150_TRI.csv)
    Small Cap         → NIFTY SMALLCAP 250(data/tri/NIFTY_SMALLCAP_250_TRI.csv)
    Flexi Cap         → NIFTY 500         (data/tri/NIFTY_500_TRI.csv)
    Multi Cap         → NIFTY 500
    ELSS              → NIFTY 500
    Value             → NIFTY 500
    Contra            → NIFTY 500
    Focused           → NIFTY 500
    Aggressive Hybrid → NIFTY 50          (data/tri/NIFTY_50_TRI.csv)
    Balanced Advantage→ NIFTY 50
    Index Funds       → None (each fund is its own benchmark)

Proxy fallback (original behaviour — unchanged):
    Uses keyword search against mftool scheme names to find a Direct Growth
    index fund that tracks the appropriate benchmark. Tracking error vs true
    TRI is typically < 0.2% per year — acceptable but inferior to true TRI.
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TRI INDEX MAPPING  (new in Phase F)
# ─────────────────────────────────────────────────────────────────────────────

# Maps each MF category to its true benchmark TRI index name.
# Keys must exactly match keys in indices/config/index_metadata.py INDEX_METADATA.
CATEGORY_TO_TRI_INDEX: Dict[str, Optional[str]] = {
    "Large Cap":          "NIFTY 100",
    "Mid Cap":            "NIFTY MIDCAP 150",
    "Small Cap":          "NIFTY SMALLCAP 250",
    "Flexi Cap":          "NIFTY 500",
    "Multi Cap":          "NIFTY 500",
    "ELSS":               "NIFTY 500",
    "Value":              "NIFTY 500",
    "Contra":             "NIFTY 500",
    "Focused":            "NIFTY 500",
    "Aggressive Hybrid":  "NIFTY 50",
    "Balanced Advantage": "NIFTY 50",
    "Index Funds":        None,          # No single benchmark
}


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK CONFIGURATION  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_DISPLAY_NAMES: Dict[str, str] = {
    "Large Cap":          "Nifty 100 TRI",
    "Mid Cap":            "Nifty Midcap 150 TRI",
    "Small Cap":          "Nifty Smallcap 250 TRI",
    "Flexi Cap":          "Nifty 500 TRI",
    "Multi Cap":          "Nifty 500 TRI",
    "ELSS":               "Nifty 500 TRI",
    "Value":              "Nifty 500 TRI",
    "Contra":             "Nifty 500 TRI",
    "Focused":            "Nifty 500 TRI",
    "Aggressive Hybrid":  "Nifty 50 TRI",
    "Balanced Advantage": "Nifty 50 TRI",
    "Index Funds":        "N/A — each fund tracks its own index",
}

BENCHMARK_SEARCH_KEYWORDS: Dict[str, List[str]] = {
    "Large Cap": [
        "nifty 100 index fund",
        "nifty100 index fund",
        "nifty 100 index",
    ],
    "Mid Cap": [
        "nifty midcap 150 index fund",
        "midcap 150 index fund",
        "nifty midcap 150",
    ],
    "Small Cap": [
        "nifty smallcap 250 index fund",
        "smallcap 250 index fund",
        "nifty smallcap 250",
    ],
    "Flexi Cap": [
        "nifty 500 index fund",
        "nifty500 index fund",
        "nifty 500 ",
    ],
    "Multi Cap": [
        "nifty 500 index fund",
        "nifty 500 ",
    ],
    "ELSS": [
        "nifty 500 index fund",
        "nifty 500 ",
    ],
    "Value": [
        "nifty 500 index fund",
        "nifty 500 ",
    ],
    "Contra": [
        "nifty 500 index fund",
        "nifty 500 ",
    ],
    "Focused": [
        "nifty 500 index fund",
        "nifty 500 ",
    ],
    "Aggressive Hybrid": [
        "nifty 50 index fund",
        "nifty50 index fund",
        "nifty 50 ",
    ],
    "Balanced Advantage": [
        "nifty 50 index fund",
        "nifty 50 ",
    ],
    "Index Funds": [],
}


# ─────────────────────────────────────────────────────────────────────────────
# PROXY SCHEME FINDER  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def find_benchmark_scheme(category: str) -> Optional[Dict]:
    """
    Find the best matching Direct Growth index fund for a category's benchmark.
    Used as fallback when TRI CSV is not available.
    """
    keywords = BENCHMARK_SEARCH_KEYWORDS.get(category, [])
    if not keywords:
        return None

    from data.fund_loader import get_all_schemes

    all_schemes = get_all_schemes()
    if not all_schemes:
        logger.error("Cannot find benchmark scheme — scheme list is empty")
        return None

    exclusions = ["etf", "exchange traded", "idcw", "dividend", "regular"]
    candidates = {
        code: name
        for code, name in all_schemes.items()
        if "direct" in name.lower()
        and "growth" in name.lower()
        and not any(excl in name.lower() for excl in exclusions)
    }

    for keyword in keywords:
        for code, name in candidates.items():
            if keyword in name.lower():
                logger.info(f"Benchmark proxy for {category}: {name} ({code})")
                return {"code": code, "name": name}

    logger.warning(f"No benchmark scheme found for {category}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK NAV  (TRI-first in Phase F)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_benchmark_nav(category: str) -> Optional[pd.DataFrame]:
    """
    Fetch the benchmark NAV/TRI series for a category.

    Step 1 — True TRI (Phase F):
        Look up CATEGORY_TO_TRI_INDEX for the category.
        If a CSV exists in data/tri/, load and return it.
        Return format: DataFrame(DatetimeIndex, 'nav' column) — identical
        to get_nav_history() so process_nav() works without any change.

    Step 2 — Proxy fallback (original behaviour):
        If TRI CSV is missing or unreadable, find a Direct Growth index fund
        via keyword search and fetch its NAV from mftool.

    Args:
        category: One of the 12 supported category strings.

    Returns:
        DataFrame with DatetimeIndex and 'nav' float column, or None.
    """
    # ── Step 1: Try true TRI ──────────────────────────────────────────────
    tri_index = CATEGORY_TO_TRI_INDEX.get(category)

    if tri_index is not None:
        try:
            from data.tri_loader import get_tri_nav
            tri_nav = get_tri_nav(tri_index)
            if tri_nav is not None and not tri_nav.empty:
                logger.info(
                    f"Benchmark for {category}: using true TRI ({tri_index})"
                )
                return tri_nav
        except Exception as exc:
            logger.warning(
                f"TRI load failed for {category} ({tri_index}): {exc}. "
                "Falling back to index fund proxy."
            )

    # ── Step 2: Proxy fallback ────────────────────────────────────────────
    benchmark = find_benchmark_scheme(category)
    if benchmark is None:
        return None

    from data.fund_loader import get_nav_history

    nav_df = get_nav_history(benchmark["code"])
    if nav_df is None:
        logger.warning(
            f"Could not fetch proxy benchmark NAV for {category} "
            f"(scheme: {benchmark.get('name')})"
        )
    return nav_df


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK INFO  (extended in Phase F)
# ─────────────────────────────────────────────────────────────────────────────

def get_benchmark_info(category: str) -> Dict:
    """
    Return display-ready benchmark information for a category.

    Phase F additions:
        'data_source'      → "tri" if true TRI CSV is used, "proxy" otherwise
        'staleness_warning'→ warning string if TRI data is > 7 days old, else None

    Existing keys (unchanged — all callers remain compatible):
        'display_name'  → e.g. 'Nifty 100 TRI'
        'scheme_name'   → index fund name (proxy) or 'True TRI — niftyindices.com'
        'scheme_code'   → scheme code (proxy) or None (TRI)
        'available'     → True if benchmark can be loaded
    """
    display   = BENCHMARK_DISPLAY_NAMES.get(category, "N/A")
    tri_index = CATEGORY_TO_TRI_INDEX.get(category)

    # ── Check TRI availability ────────────────────────────────────────────
    if tri_index is not None:
        try:
            from data.tri_loader import get_tri_nav, get_tri_staleness_warning, is_tri_available

            if is_tri_available(tri_index):
                tri_nav = get_tri_nav(tri_index)
                if tri_nav is not None and not tri_nav.empty:
                    staleness = get_tri_staleness_warning(tri_index)
                    return {
                        "display_name":     display,
                        "scheme_name":      "True TRI — niftyindices.com",
                        "scheme_code":      None,
                        "available":        True,
                        "data_source":      "tri",
                        "staleness_warning": staleness,
                    }
        except Exception as exc:
            logger.warning(f"TRI info check failed for {category}: {exc}")

    # ── Proxy fallback info ───────────────────────────────────────────────
    scheme = find_benchmark_scheme(category)
    return {
        "display_name":     display,
        "scheme_name":      scheme["name"] if scheme else "Not found",
        "scheme_code":      scheme["code"] if scheme else None,
        "available":        scheme is not None,
        "data_source":      "proxy",
        "staleness_warning": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BULK AVAILABILITY CHECK  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# BROAD MARKET  (dual benchmarking)
# ─────────────────────────────────────────────────────────────────────────────

MARKET_INDEX_NAME   = "NIFTY 500"
MARKET_DISPLAY_NAME = "Nifty 500 TRI"


def get_market_nav() -> Optional[pd.DataFrame]:
    """
    NAV series for the broad market — the SECOND yardstick shown alongside
    each fund's SEBI category benchmark.

    Nifty 500 covers roughly 95% of Indian listed market capitalisation,
    which makes it the closest available stand-in for "just owning the
    market". A mid-cap fund can beat Nifty Midcap 150 while lagging Nifty
    500, or the reverse; both readings are real and the pair is the point.

    Returns DataFrame(DatetimeIndex, 'nav'), or None.
    """
    try:
        from data.tri_loader import get_tri_nav
        nav = get_tri_nav(MARKET_INDEX_NAME)
        if nav is not None and not nav.empty:
            return nav
    except Exception as e:
        logger.warning(f"Market benchmark ({MARKET_INDEX_NAME}) unavailable: {e}")
    return None


def is_market_same_as_category(category: str) -> bool:
    """
    True when a category's own benchmark IS the broad market index.

    Applies to Flexi Cap, Multi Cap, ELSS, Value, Contra and Focused — all
    benchmarked to Nifty 500 under SEBI. For those the two sets of numbers
    coincide by construction, and the UI must say so rather than presenting
    them as two independent readings.
    """
    return CATEGORY_TO_TRI_INDEX.get(category) == MARKET_INDEX_NAME
