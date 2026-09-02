"""
fund_loader.py
==============
The ONLY file that talks to mftool and the AMFI/mfapi data sources.

mftool v3.3 API changes (BREAKING from v2.x):
  - get_available_schemes(amc_name)  → now requires AMC name parameter
  - get_scheme_codes()               → use this for ALL schemes
  - get_scheme_historical_nav(code, as_Dataframe=True)  → still works
  - history(code)                    → new method (uses yfinance codes)

Fallback strategy:
  If mftool's get_scheme_codes() fails (AMFI URL blocked), we fetch
  directly from mfapi.in/mf using requests. This is the same backend
  that mftool uses for individual scheme details and NAV history.

Caching:
  All functions use @st.cache_data(ttl=3600) — 1 hour cache.
  Errors are NEVER silently swallowed — they are printed AND returned
  as informative error strings so the UI can display them.
"""

import streamlit as st
import pandas as pd
import requests
from mftool import Mftool
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

# Single shared mftool instance — created LAZILY.
#
# Mftool.__init__ calls get_scheme_codes() (mftool.py:64), which hits AMFI
# over the network. Building it at module scope meant that when AMFI was
# unreachable, `import data.fund_loader` itself raised — taking down every
# page before the triple-fallback in get_all_schemes() could run. The
# fallback chain only works if importing this module never touches the
# network.
_mf = None


def _get_mftool():
    """Return the shared Mftool instance, constructing it on first use."""
    global _mf
    if _mf is None:
        _mf = Mftool()
    return _mf

# Direct API URL (mftool's backend — used as fallback)
_MFAPI_ALL_URL    = "https://api.mfapi.in/mf"
_MFAPI_SCHEME_URL = "https://api.mfapi.in/mf/{code}"
# amfiindia.com now 302-redirects here. requests follows redirects, but
# pointing straight at the current host saves a hop and one failure mode.
_AMFI_NAV_URL     = "https://portal.amfiindia.com/spages/NAVAll.txt"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-IN,en;q=0.9",
}


# ─────────────────────────────────────────────────────────────────────────────
# SCHEME REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# SCHEME NAME CONTRACT
#
# Everything downstream (category_mapper.filter_preferred_plans,
# filter_direct_plans, filter_regular_plans) identifies a fund's plan and
# option by SUBSTRING MATCH on the scheme name. That requires names in the
# composite form:
#
#     "Axis ELSS Tax Saver Fund - Direct Plan - Growth Option"
#
# mfapi.in still returns exactly that. AMFI's NAVAll.txt used to as well —
# but its schema changed. It is now 8 fields with Plan and Option split into
# their own columns, and field 3 holding only the bare fund name:
#
#   OLD (6 fields):
#     Scheme Code;ISIN Payout;ISIN Reinvest;Scheme Name;NAV;Date
#     → field 3 = "Axis Banking & PSU Debt Fund - Direct Plan - Growth Option"
#
#   NEW (8 fields):
#     Scheme Code;ISIN Payout;ISIN Reinvest;Scheme Name;Plan;Option;NAV;Date
#     → field 3 = "Axis Banking & PSU Debt Fund"
#     → field 4 = "Direct Plan"
#     → field 5 = "Growth Option"
#
# Under the new schema no name contains "growth" or "direct", so the growth
# filter discarded essentially the entire universe and the app reported a
# handful of funds. We therefore RECOMPOSE the legacy form on parse, and
# validate every source before trusting it.
# ─────────────────────────────────────────────────────────────────────────────

def _is_numeric(value: str) -> bool:
    """True if the field parses as a number (i.e. it is a NAV, not a plan)."""
    try:
        float(value.strip())
        return True
    except (ValueError, AttributeError):
        return False


def _compose_scheme_name(name: str, plan: str = "", option: str = "") -> str:
    """Rebuild the legacy 'Name - Plan - Option' form from split columns."""
    parts = [p for p in (name.strip(), plan.strip(), option.strip()) if p and p != "-"]
    return " - ".join(parts)


def _validate_scheme_names(schemes: Dict[str, str], source: str) -> Dict[str, str]:
    """
    Reject a source whose names lack plan/option information.

    Without this guard a source that silently changes shape (as AMFI just
    did) still "succeeds" — it returns thousands of rows — and the breakage
    only surfaces much later as an inexplicably tiny fund universe. Failing
    loudly here lets get_all_schemes() fall through to a source that still
    honours the contract.
    """
    if not schemes:
        raise ValueError(f"{source} returned no schemes.")

    sample = list(schemes.values())[:4000]
    with_option = sum(
        1 for n in sample
        if "growth" in n.lower() or "idcw" in n.lower() or "dividend" in n.lower()
    )
    with_plan = sum(
        1 for n in sample
        if "direct" in n.lower() or "regular" in n.lower()
    )

    # Healthy AMFI/mfapi data is overwhelmingly plan- and option-suffixed.
    # 20% is far below any plausible real value and far above zero noise.
    if with_option / len(sample) < 0.20 or with_plan / len(sample) < 0.20:
        raise ValueError(
            f"{source} returned {len(schemes):,} schemes but only "
            f"{with_option} of {len(sample)} sampled names carry an option "
            f"(Growth/IDCW) and {with_plan} carry a plan (Direct/Regular). "
            "The source has probably split Plan/Option into separate fields. "
            "Names must be in 'Fund Name - Plan - Option' form."
        )
    return schemes


def _fetch_schemes_via_mftool() -> Dict[str, str]:
    """
    Fetch all schemes using mftool's get_scheme_codes().

    NOTE: mftool 3.3 reads AMFI's field 3 directly (mftool.py:95,
    `scheme_info[scheme[0]] = scheme[3]`) with no awareness of the new
    Plan/Option columns, so on the current AMFI schema it returns BARE fund
    names. _validate_scheme_names() catches that and forces a fallback.
    """
    codes = _get_mftool().get_scheme_codes(as_json=False)
    if not codes:
        raise ValueError("mftool.get_scheme_codes() returned empty — AMFI URL may be blocked.")
    return _validate_scheme_names(dict(codes), "mftool.get_scheme_codes()")


def _fetch_schemes_via_amfi_direct() -> Dict[str, str]:
    """
    Fetch the scheme list straight from AMFI NAVAll.txt.

    Handles BOTH schemas:
      - 8+ fields → Plan (4) and Option (5) are separate; recompose the name
      - 6-7 fields → legacy layout, field 3 already carries plan and option

    Returns {code: composite_name}, or raises on failure.
    """
    r = requests.get(_AMFI_NAV_URL, headers=_HEADERS, timeout=20, allow_redirects=True)
    r.raise_for_status()

    schemes: Dict[str, str] = {}
    for line in r.text.splitlines():
        if ";" not in line:
            continue                      # section headers and AMC names
        parts = line.split(";")
        if len(parts) < 4:
            continue

        code = parts[0].strip()
        if not code.isdigit():
            continue                      # skips the column-header row

        name = parts[3].strip()
        if not name:
            continue

        # Discriminate on field count: legacy is exactly 6, current is 8.
        # Testing `>= 6` would misread a legacy row and append the NAV and
        # date to the fund name. The numeric check on field 4 is a second
        # guard — in the legacy layout that position holds the NAV.
        if len(parts) >= 8 and not _is_numeric(parts[4]):
            # Current schema: Plan and Option in their own columns.
            schemes[code] = _compose_scheme_name(name, parts[4], parts[5])
        else:
            # Legacy schema: field 3 already holds the composite name.
            schemes[code] = name

    if not schemes:
        raise ValueError("AMFI NAVAll.txt parsed but no schemes found.")
    return _validate_scheme_names(schemes, "AMFI NAVAll.txt")


def _fetch_schemes_via_mfapi_direct() -> Dict[str, str]:
    """
    Second fallback: fetch full scheme list from mfapi.in/mf.
    Returns {code: name} or raises on failure.
    """
    r = requests.get(_MFAPI_ALL_URL, headers=_HEADERS, timeout=20)
    r.raise_for_status()

    data = r.json()   # List of {schemeCode, schemeName}
    schemes = {
        str(item["schemeCode"]): item["schemeName"]
        for item in data
        if "schemeCode" in item and "schemeName" in item
    }
    if not schemes:
        raise ValueError("mfapi.in/mf returned empty list.")
    # mfapi.in still publishes composite names ("... - Direct Plan - Growth
    # Option"), but validate anyway so a future change there fails loudly too.
    return _validate_scheme_names(schemes, "mfapi.in/mf")


@st.cache_data(ttl=3600, show_spinner=False)
def get_all_schemes() -> Dict[str, str]:
    """
    Fetch all mutual fund scheme codes and names.
    Tries three sources in order:
      1. mftool.get_scheme_codes()   (uses AMFI NAVAll.txt internally)
      2. Direct AMFI NAVAll.txt fetch
      3. Direct mfapi.in/mf fetch

    Returns:
        {scheme_code: scheme_name} dict, or {} if all sources fail.
        Never raises — errors are logged with full detail.
    """
    errors: List[str] = []

    # Order matters. Our own AMFI parser goes FIRST because it is the only
    # one that understands the current NAVAll.txt schema. mftool is kept as a
    # fallback for networks where the direct fetch is blocked but its bundled
    # session succeeds — its output is validated, so if it returns bare names
    # we fall straight through instead of trusting them.
    for label, fetch in (
        ("AMFI NAVAll.txt",           _fetch_schemes_via_amfi_direct),
        ("mfapi.in/mf",               _fetch_schemes_via_mfapi_direct),
        ("mftool.get_scheme_codes()", _fetch_schemes_via_mftool),
    ):
        try:
            schemes = fetch()
            logger.info(f"[fund_loader] {label}: {len(schemes):,} schemes loaded")
            st.session_state["scheme_source"] = label
            return schemes
        except Exception as e:
            msg = f"{label} failed: {type(e).__name__}: {e}"
            errors.append(msg)
            logger.warning(f"[fund_loader] {msg}")

    # ── All failed ───────────────────────────────────────────────────────────
    logger.error(f"[fund_loader] ALL sources failed:\n" + "\n".join(errors))
    # Store errors so the UI can display them
    st.session_state["scheme_load_errors"] = errors
    return {}



# ─────────────────────────────────────────────────────────────────────────────
# NAV HISTORY
# ─────────────────────────────────────────────────────────────────────────────

def _parse_mfapi_nav_response(data: dict) -> Optional[pd.DataFrame]:
    """
    Parse the raw mfapi.in JSON response into a clean NAV DataFrame.

    mfapi.in data format:
        data: [{"date": "31-05-2024", "nav": "45.2381"}, ...]
        (newest first — we reverse to oldest first)

    Returns DataFrame with:
        - DatetimeIndex (ascending, named 'date')
        - 'nav' column as float64
    """
    nav_list = data.get("data", [])
    if not nav_list:
        return None

    df = pd.DataFrame(nav_list)

    if "date" not in df.columns or "nav" not in df.columns:
        return None

    # Try multiple date formats (mfapi.in uses DD-MM-YYYY but some sources differ)
    for fmt in ["%d-%m-%Y", "%d-%b-%Y", "%Y-%m-%d"]:
        try:
            df["date"] = pd.to_datetime(df["date"], format=fmt, errors="raise")
            break
        except (ValueError, TypeError):
            continue
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date", "nav"])
    df = df[df["nav"] > 0]
    df = df.sort_values("date").set_index("date")
    df = df[~df.index.duplicated(keep="last")]
    df = df[["nav"]]

    return df if not df.empty else None


@st.cache_data(ttl=3600, show_spinner=False)
def get_nav_history(scheme_code: str) -> Optional[pd.DataFrame]:
    """
    Fetch complete NAV history for a scheme.

    Returns DataFrame with:
        - DatetimeIndex ascending (named 'date')
        - 'nav' column as float64

    Tries two sources:
        1. mftool.get_scheme_historical_nav()
        2. Direct mfapi.in request

    Returns None if both fail.
    """
    scheme_code = str(scheme_code)

    # ── Attempt 1: mftool ────────────────────────────────────────────────────
    try:
        raw = _get_mftool().get_scheme_historical_nav(scheme_code, as_Dataframe=True)

        if raw is not None and isinstance(raw, pd.DataFrame) and not raw.empty:
            df = raw.copy()

            # mftool 3.3 returns: index='date'(str), columns=['nav', 'dayChange']
            if df.index.name == "date":
                df = df.reset_index()

            df.columns = [c.lower().strip() for c in df.columns]

            if "date" in df.columns and "nav" in df.columns:
                for fmt in ["%d-%m-%Y", "%d-%b-%Y", "%Y-%m-%d"]:
                    try:
                        df["date"] = pd.to_datetime(df["date"], format=fmt, errors="raise")
                        break
                    except (ValueError, TypeError):
                        continue
                else:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")

                df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
                df = df.dropna(subset=["date", "nav"])
                df = df[df["nav"] > 0]
                df = df.sort_values("date").set_index("date")
                df = df[~df.index.duplicated(keep="last")]
                df = df[["nav"]]

                if not df.empty:
                    return df

    except Exception as e:
        logger.warning(f"[fund_loader] mftool NAV({scheme_code}) failed: {type(e).__name__}: {e}")

    # ── Attempt 2: Direct mfapi.in request ──────────────────────────────────
    try:
        url = _MFAPI_SCHEME_URL.format(code=scheme_code)
        r = requests.get(url, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        return _parse_mfapi_nav_response(data)

    except Exception as e:
        logger.error(f"[fund_loader] Direct NAV({scheme_code}) also failed: {type(e).__name__}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY-FILTERED SCHEME LISTS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_all_categorized_schemes(plan_type: str = "Direct") -> Dict[str, List[Dict]]:
    """
    Load all schemes and group by category in one cached call.

    Args:
        plan_type: "Direct" or "Regular" — filters to one universe only.
                   Cached separately per plan_type so switching is instant.

    Returns:
        {category_name: [{code, name}, ...]} for all 12 categories.
    """
    from data.category_mapper import get_category_for_scheme, filter_by_plan_type
    from utils.constants import CATEGORIES

    all_schemes = get_all_schemes()
    if not all_schemes:
        return {cat: [] for cat in CATEGORIES}

    # filter_by_plan_type handles both the growth filter AND the direct/regular split
    filtered = filter_by_plan_type(all_schemes, plan_type=plan_type)
    result: Dict[str, List[Dict]] = {cat: [] for cat in CATEGORIES}

    for code, name in filtered.items():
        category = get_category_for_scheme(name)
        if category and category in result:
            result[category].append({"code": code, "name": name})

    for cat in result:
        result[cat] = sorted(result[cat], key=lambda x: x["name"])

    return result


@st.cache_data(ttl=3600, show_spinner=False)
def get_schemes_for_category(category: str, plan_type: str = "Direct") -> List[Dict]:
    """
    Return [{code, name}] for all Growth-plan funds in a single category.

    Args:
        category:  One of the 12 category strings.
        plan_type: "Direct" or "Regular".
    """
    all_cat = get_all_categorized_schemes(plan_type=plan_type)
    return all_cat.get(category, [])


# ─────────────────────────────────────────────────────────────────────────────
# BATCH NAV LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_navs_for_funds(
    fund_list: List[Dict],
    progress_callback=None,
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Load NAV history for a list of funds with optional progress reporting.

    Args:
        fund_list:         [{code, name}, ...]
        progress_callback: Optional callable(current, total, fund_name)

    Returns:
        {scheme_code: DataFrame or None}
    """
    result: Dict[str, Optional[pd.DataFrame]] = {}

    for i, fund in enumerate(fund_list):
        code = fund["code"]
        if progress_callback:
            try:
                progress_callback(i, len(fund_list), fund["name"])
            except Exception:
                pass
        result[code] = get_nav_history(code)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL NAV LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_navs_parallel(
    codes,
    max_workers: int = 6,
    progress_cb  = None,
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Fetch NAV history for many schemes concurrently.

    Every page loaded NAVs in a plain for-loop, so the wait was the SUM of the
    round-trips: Portfolio Analytics with two full portfolios meant 16 serial
    fetches at 2–5s each, up to ~80 seconds of mostly-idle waiting. These are
    independent I/O-bound HTTP calls, so they overlap perfectly.

    get_nav_history is wrapped in @st.cache_data, and Streamlit's cache reads
    the *script run context* of the calling thread. Worker threads do not
    inherit it, which produces "missing ScriptRunContext" warnings and makes
    cache behaviour unreliable — so the caller's context is explicitly
    attached to each worker.

    Args:
        codes:       Iterable of scheme codes. Duplicates are fetched once.
        max_workers: Concurrency. Kept modest — mfapi.in is a free community
                     API and there is no reason to hammer it.
        progress_cb: Optional callable(done:int, total:int, code:str) invoked
                     as each fetch completes, for progress bars.

    Returns:
        {scheme_code: DataFrame or None}. A failed fetch yields None rather
        than raising, matching get_nav_history's own contract.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    unique = list(dict.fromkeys(str(c) for c in codes))
    if not unique:
        return {}

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
        ctx = get_script_run_ctx()
    except Exception:
        ctx = None

    def _fetch(code):
        try:
            return code, get_nav_history(code)
        except Exception as e:
            logger.warning(f"[fund_loader] NAV fetch failed for {code}: {e}")
            return code, None

    results: Dict[str, Optional[pd.DataFrame]] = {}
    workers = max(1, min(max_workers, len(unique)))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch, c) for c in unique]

        # Attach the Streamlit context to the pool's threads. Must happen
        # after submit(), since the pool creates threads lazily.
        if ctx is not None:
            for t in getattr(pool, "_threads", ()):
                try:
                    add_script_run_ctx(t, ctx)
                except Exception:
                    pass

        for i, fut in enumerate(as_completed(futures), start=1):
            code, nav = fut.result()
            results[code] = nav
            if progress_cb is not None:
                try:
                    progress_cb(i, len(unique), code)
                except Exception:
                    pass

    return results
