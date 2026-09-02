"""
data/tri_loader.py
==================
Integration bridge between the indice_loader CSV files and mf-analytics.

This is the ONLY file in the platform that knows about the TRI CSV format.
Everything above this layer (benchmark_loader, engine, pages) receives a
standard pd.DataFrame with DatetimeIndex and 'nav' column — identical to
the output of get_nav_history() — and has no awareness of the source.

CSV format (confirmed from niftyindices.com):
    Columns : Date (YYYY-MM-DD), TotalReturnsIndex (float)
    Encoding: UTF-8, no BOM
    Base val: varies by index (e.g. 1000.0 for Nifty 500, 1256.38 for Nifty 50)
    Note    : Absolute base value is irrelevant — platform uses pct_change() only

Return contract:
    get_tri_nav() returns pd.DataFrame with:
        - DatetimeIndex named "Date"
        - Single column named "nav" (float)
    This is identical to get_nav_history() — process_nav() works without modification.

Staleness policy:
    If the last date in a CSV is more than TRI_STALENESS_DAYS calendar days ago,
    get_tri_staleness_warning() returns a warning string.
    This is NOT a hard error — stale TRI data is still usable.
    Run scripts/update_indices.py and push to GitHub to refresh.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from indices.config.index_metadata import INDEX_METADATA

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# tri_loader.py lives in data/ — one level below mf_analytics root
_TRI_DATA_DIR = Path(__file__).resolve().parent / "tri"


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Number of calendar days before a staleness warning is raised.
# 7 days covers weekends + one market holiday before alerting.
TRI_STALENESS_DAYS = 7


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_csv_path(index_name: str) -> Optional[Path]:
    """
    Resolve the CSV file path for a given index name.
    Returns None if the index is not in the registry.
    """
    metadata = INDEX_METADATA.get(index_name)
    if metadata is None:
        return None
    return _TRI_DATA_DIR / metadata["filename"]


def _load_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Load a TRI CSV and return a DataFrame with DatetimeIndex and 'nav' column.

    Transformation:
        Date (YYYY-MM-DD string) → DatetimeIndex
        TotalReturnsIndex (float) → 'nav' column

    Returns None on any read or parse failure.
    """
    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"])

        if "Date" not in df.columns or "TotalReturnsIndex" not in df.columns:
            logger.error(
                f"TRI CSV missing expected columns in {csv_path.name}. "
                f"Found: {list(df.columns)}"
            )
            return None

        df["TotalReturnsIndex"] = pd.to_numeric(
            df["TotalReturnsIndex"], errors="coerce"
        )
        df = df.dropna(subset=["Date", "TotalReturnsIndex"])
        df = df.sort_values("Date").set_index("Date")
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        # Rename to 'nav' to match get_nav_history() contract
        nav_df = df[["TotalReturnsIndex"]].rename(
            columns={"TotalReturnsIndex": "nav"}
        )

        return nav_df

    except Exception as exc:
        logger.error(f"Failed to load TRI CSV {csv_path}: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_tri_nav(index_name: str) -> Optional[pd.DataFrame]:
    """
    Load a TRI series from the local CSV store.

    Args:
        index_name: Registry key exactly as defined in INDEX_METADATA,
                    e.g. "NIFTY 500", "NIFTY 100", "NIFTY MIDCAP 150"

    Returns:
        pd.DataFrame with:
            - DatetimeIndex (daily trading dates)
            - 'nav' column (float, TRI level values)
        or None if the CSV is missing or cannot be read.

    Notes:
        - Return format is identical to data.fund_loader.get_nav_history()
        - Cached for 1 hour (CSVs only change on push; cache avoids re-reads)
        - Falls back to None silently — benchmark_loader handles the fallback
    """
    csv_path = _get_csv_path(index_name)

    if csv_path is None:
        logger.warning(f"Index '{index_name}' not found in INDEX_METADATA registry.")
        return None

    if not csv_path.exists():
        logger.info(
            f"TRI CSV not found for '{index_name}' at {csv_path}. "
            "Will fall back to index fund NAV proxy."
        )
        return None

    nav_df = _load_csv(csv_path)

    if nav_df is None or nav_df.empty:
        logger.warning(f"TRI CSV for '{index_name}' loaded empty or failed to parse.")
        return None

    logger.info(
        f"TRI loaded: {index_name} — "
        f"{len(nav_df):,} rows, "
        f"{nav_df.index[0].strftime('%d %b %Y')} → "
        f"{nav_df.index[-1].strftime('%d %b %Y')}"
    )
    return nav_df


def get_tri_staleness_warning(index_name: str) -> Optional[str]:
    """
    Check whether the TRI data for an index is stale.

    Not cached — needs a fresh timestamp comparison on every call.

    Args:
        index_name: Registry key (same as get_tri_nav)

    Returns:
        Warning string if data is more than TRI_STALENESS_DAYS old,
        None if data is fresh or could not be loaded.
    """
    nav_df = get_tri_nav(index_name)
    if nav_df is None or nav_df.empty:
        return None

    last_date = nav_df.index[-1]
    today     = pd.Timestamp.today().normalize()
    delta     = (today - last_date).days

    if delta > TRI_STALENESS_DAYS:
        return (
            f"{index_name} TRI data is **{delta} days old** "
            f"(last update: {last_date.strftime('%d %b %Y')}). "
            "Run `python -m scripts.update_indices` locally and push to refresh."
        )

    return None


def is_tri_available(index_name: str) -> bool:
    """
    Quick check: does a valid TRI CSV exist for this index?
    Used by benchmark_loader to decide whether to use TRI or fall back to proxy.
    """
    csv_path = _get_csv_path(index_name)
    return csv_path is not None and csv_path.exists()
