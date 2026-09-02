"""
category_mapper.py
==================
Maps mftool scheme names to our 12 standardized fund categories.

Design decisions:
  - Uses keyword matching on lowercase scheme names (no extra API calls)
  - Priority order matters — more specific categories are checked first
    to avoid false positives (e.g. "Focused" before "Large Cap")
  - Index Funds are identified first and ETFs are excluded explicitly
  - filter_preferred_plans() removes Dividend, IDCW, ETF, FoF variants
    so only Growth open-ended funds remain for analysis

To extend: add new keywords to CATEGORY_KEYWORDS in constants.py.
"""

from typing import Optional, Dict, List
from utils.constants import (
    CATEGORY_KEYWORDS,
    EXCLUDED_PLAN_KEYWORDS,
    EXCLUDED_STRUCTURE_KEYWORDS,
    INDEX_EXCLUSIONS,
    PREFERRED_OPTIONS,
    DIRECT_KEYWORDS,
    REGULAR_KEYWORDS,
    CATEGORIES,
)


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def get_category_for_scheme(scheme_name: str) -> Optional[str]:
    """
    Detect the category of a mutual fund from its scheme name.

    Algorithm:
      1. Lowercase the name for case-insensitive matching
      2. Check Index Funds first (with ETF exclusion)
      3. Check remaining 11 categories in priority order

    Priority order is important:
      - "Contra" before "Large Cap" (some contra funds say "Large Cap Contra")
      - "Focused" before "Large Cap" (Focused 25 funds may have "large cap" in name)
      - "ELSS" before "Multi Cap" etc.
      - "Balanced Advantage" before "Aggressive Hybrid"

    Args:
        scheme_name: Full scheme name string from mftool

    Returns:
        Category string (one of CATEGORIES) or None if unmatched
    """
    if not scheme_name or not isinstance(scheme_name, str):
        return None

    name_lower = scheme_name.lower()

    # ── Step 1: Index Funds (check first, then exclude ETFs) ─────────────────
    has_index_keyword = any(kw in name_lower for kw in CATEGORY_KEYWORDS["Index Funds"])
    is_etf = any(excl in name_lower for excl in INDEX_EXCLUSIONS)

    if has_index_keyword and not is_etf:
        return "Index Funds"

    # ── Step 2: All other categories in priority order ────────────────────────
    # DO NOT include "Index Funds" here — already handled above
    PRIORITY_ORDER: List[str] = [
        "Contra",               # Most specific — before Value/Large Cap
        "Focused",              # "Focused 25" before "Large Cap"
        "ELSS",                 # Tax saver — before Multi Cap
        "Balanced Advantage",   # Before Aggressive Hybrid
        "Aggressive Hybrid",    # Before Multi Cap
        "Multi Cap",            # Before Flexi Cap
        "Flexi Cap",            # Before Large Cap (some flexi names include "large")
        "Small Cap",
        "Mid Cap",
        "Large Cap",
        "Value",                # Last — "Value" keyword is short and can false-match
    ]

    for category in PRIORITY_ORDER:
        keywords = CATEGORY_KEYWORDS.get(category, [])
        for keyword in keywords:
            if keyword in name_lower:
                return category

    return None   # Unrecognized — will be excluded from all categories


# ─────────────────────────────────────────────────────────────────────────────
# PLAN FILTERING
# ─────────────────────────────────────────────────────────────────────────────

def filter_preferred_plans(all_schemes: Dict[str, str]) -> Dict[str, str]:
    """
    Filter the full scheme dict to Growth option only, removing:
      - Dividend / IDCW / Bonus options
      - ETFs (exchange-traded funds)
      - Fund of Funds
      - Fixed Maturity Plans, Interval Funds, etc.

    A scheme MUST contain a PREFERRED_OPTIONS keyword ('growth') to pass.
    It MUST NOT contain any EXCLUDED_PLAN_KEYWORDS or EXCLUDED_STRUCTURE_KEYWORDS.

    Args:
        all_schemes: {scheme_code: scheme_name} from mftool

    Returns:
        Filtered {scheme_code: scheme_name} dict
    """
    filtered: Dict[str, str] = {}

    for code, name in all_schemes.items():
        if not name or not isinstance(name, str):
            continue

        name_lower = name.lower()

        # ── MUST be a growth option ───────────────────────────────────────────
        if not any(opt in name_lower for opt in PREFERRED_OPTIONS):
            continue

        # ── MUST NOT be a dividend/IDCW/bonus option ─────────────────────────
        if any(excl in name_lower for excl in EXCLUDED_PLAN_KEYWORDS):
            continue

        # ── MUST NOT be a structural exclusion (ETF, FoF, etc.) ──────────────
        # Exception: the "debt fund" keyword is there to drop pure debt
        # schemes, but Aggressive Hybrid funds are literally named
        # "... Equity & Debt Fund" (ICICI Prudential Equity & Debt Fund,
        # SBI Equity Hybrid Fund's peers, etc). Excluding them on that
        # substring wiped out the whole Aggressive Hybrid category.
        hits = [excl for excl in EXCLUDED_STRUCTURE_KEYWORDS if excl in name_lower]
        if hits:
            is_equity_hybrid = any(
                kw in name_lower for kw in CATEGORY_KEYWORDS["Aggressive Hybrid"]
            )
            if not (is_equity_hybrid and hits == ["debt fund"]):
                continue

        filtered[code] = name

    return filtered


def filter_direct_plans(schemes: Dict[str, str]) -> Dict[str, str]:
    """
    From an already-filtered Growth scheme dict, keep only Direct plans.

    Args:
        schemes: Pre-filtered {code: name} dict (Growth only)

    Returns:
        {code: name} with only Direct Plan schemes
    """
    return {
        code: name
        for code, name in schemes.items()
        if any(k in name.lower() for k in DIRECT_KEYWORDS)
    }


def filter_regular_plans(schemes: Dict[str, str]) -> Dict[str, str]:
    """
    From an already-filtered Growth scheme dict, keep only Regular plans.

    Note: Some older schemes don't have "Regular" in their name — they were
    named before the Direct/Regular distinction existed. This function
    keeps schemes that either explicitly say "regular" OR have neither
    "direct" nor "regular" in their name.

    Args:
        schemes: Pre-filtered {code: name} dict (Growth only)

    Returns:
        {code: name} with Regular Plan schemes
    """
    result = {}
    for code, name in schemes.items():
        name_lower = name.lower()
        if any(k in name_lower for k in REGULAR_KEYWORDS):
            result[code] = name
        elif not any(k in name_lower for k in DIRECT_KEYWORDS):
            # Older scheme without plan label — include as regular
            result[code] = name
    return result


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED PLAN TYPE FILTER
# ─────────────────────────────────────────────────────────────────────────────

def filter_by_plan_type(
    all_schemes: Dict[str, str],
    plan_type: str = "Direct",
) -> Dict[str, str]:
    """
    Single entry point for plan-type filtering.

    Pipeline:
        1. filter_preferred_plans()  — remove Dividend/IDCW/ETF/FoF
        2. filter_direct_plans()     — if plan_type == "Direct"
           OR filter_regular_plans() — if plan_type == "Regular"

    Args:
        all_schemes: Raw {code: name} dict from mftool (all 40,000+ schemes)
        plan_type:   "Direct" or "Regular" (from sidebar session state)

    Returns:
        Filtered {code: name} dict for the chosen universe.
    """
    # Step 1: remove Dividend, IDCW, ETF, FoF regardless of plan type
    growth_only = filter_preferred_plans(all_schemes)

    # Step 2: split by plan type
    if plan_type == "Direct":
        return filter_direct_plans(growth_only)
    elif plan_type == "Regular":
        return filter_regular_plans(growth_only)
    else:
        # Fallback — should not happen, but return all growth plans
        return growth_only


# ─────────────────────────────────────────────────────────────────────────────
# NAME CLEANING
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def get_category_fund_counts(all_schemes: Dict[str, str]) -> Dict[str, int]:
    """
    Count how many Growth-plan funds exist per category.
    Used on the Dashboard page for the overview table.

    Args:
        all_schemes: Full {code: name} dict from mftool

    Returns:
        Dict {category: count}
    """
    preferred = filter_preferred_plans(all_schemes)
    counts: Dict[str, int] = {cat: 0 for cat in CATEGORIES}

    for name in preferred.values():
        category = get_category_for_scheme(name)
        if category and category in counts:
            counts[category] += 1

    return counts
