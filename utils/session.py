"""
utils/session.py
================
Centralised session state key management.

All analytics cache keys include ANALYTICS_VERSION — when new metrics are
added and the version is bumped in constants.py, every cached result is
automatically invalidated on the next page load without the user needing
to do anything.

Usage in pages:
    from utils.session import (
        fund_key, category_key, alpha_key,
        clear_analytics_cache, render_refresh_button,
    )
"""

import streamlit as st
from utils.constants import ANALYTICS_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# KEY BUILDERS
# Each key includes ANALYTICS_VERSION so bumping the version auto-invalidates.
# ─────────────────────────────────────────────────────────────────────────────

def fund_key(scheme_code: str, rf_pct: float, category: str = "") -> str:
    """
    Session state key for a single-fund metrics dict.

    The category is part of the key because it determines which benchmark
    (and therefore which alpha, beta and factor numbers) the metrics were
    computed against. Omitting it would let a fund's Large Cap metrics be
    served after the user switched the category selector to Mid Cap.
    """
    cat = f"_{category}" if category else ""
    return f"fund_metrics_{scheme_code}_{rf_pct}{cat}_{ANALYTICS_VERSION}"


def alpha_key(scheme_code: str, rf_pct: float, category: str) -> str:
    """Session state key for alpha+Phase B metrics of a single fund."""
    return f"alpha_{scheme_code}_{rf_pct}_{category}_{ANALYTICS_VERSION}"



def category_full_df_key(category: str) -> str:
    """Session state key for the full category metrics+quartile DataFrame."""
    return f"full_df_{category}_{ANALYTICS_VERSION}"


def category_fund_metrics_key(category: str) -> str:
    """Session state key for the {fund_name: metrics_dict} category dict."""
    return f"fund_metrics_{category}_{ANALYTICS_VERSION}"


def rankings_done_key(category: str) -> str:
    """Session state key for 'have rankings been computed' flag."""
    return f"rankings_done_{category}_{ANALYTICS_VERSION}"


def dq_scan_key(category: str) -> str:
    """Session state key for data quality scan results."""
    return f"dq_scan_{category}_{ANALYTICS_VERSION}"


def dq_reports_key(category: str) -> str:
    """Session state key for data quality reports dict."""
    return f"dq_reports_{category}_{ANALYTICS_VERSION}"


# ─────────────────────────────────────────────────────────────────────────────
# CACHE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def clear_analytics_cache() -> int:
    """
    Clear ALL analytics-related session state keys.

    Called when the user clicks the Refresh NAV Data button.
    Also clears st.cache_data (the mftool / NAV API cache).

    Returns:
        Number of session state keys cleared.
    """
    # "factor_" is listed for safety: page 3 used to cache 4-factor results
    # under a raw f-string key that skipped ANALYTICS_VERSION and was never
    # cleared here. That key is gone, but a user's existing session may
    # still be carrying one.
    ANALYTICS_PREFIXES = (
        "fund_metrics_", "full_df_", "analytics_done_",
        "rankings_done_", "alpha_", "factor_", "dq_scan_", "dq_reports_",
    )
    keys_to_remove = [
        k for k in list(st.session_state.keys())
        if any(k.startswith(p) for p in ANALYTICS_PREFIXES)
    ]
    for key in keys_to_remove:
        del st.session_state[key]

    # Also clear the Streamlit function cache (NAV API calls)
    st.cache_data.clear()

    return len(keys_to_remove)


def render_refresh_button(location=None) -> None:
    """
    Render the standard Refresh NAV Data button.
    When clicked, clears both API cache and analytics session state.

    Args:
        location: Optional Streamlit container (defaults to current context).
    """
    ctx = location or st

    if ctx.button("Refresh NAV Data", width="stretch",
                  help="Clears cached NAV data and all computed analytics. "
                       "Everything will be recomputed fresh."):
        n_cleared = clear_analytics_cache()
        # Stash the message rather than calling st.success() here: the
        # st.rerun() below tears the page down before a toast written now
        # would ever paint, so the user saw nothing happen.
        st.session_state["_refresh_notice"] = n_cleared
        st.rerun()

    # Rendered on the run *after* the click, so it survives long enough to read.
    if "_refresh_notice" in st.session_state:
        n = st.session_state.pop("_refresh_notice")
        ctx.success(f"✓ Cache cleared — {n} analytics results removed.")
