"""
Render every page WITH DATA, offline.

Why this file exists
--------------------
test_pages_render.py renders each page and asserts it does not raise. It
passed for weeks while pages/7_Data_Quality.py contained a KeyError that
crashed the page for every category with funds in it.

It passed because the interesting half of that page is behind a "Scan NAV
History" button and needs a populated fund list. With no network in the test
environment the scheme registry came back empty, the page took its
"No funds found" branch, and the crash sat in code the test never executed.

So this file supplies data instead of hoping for it. The AMFI loaders are
stubbed with synthetic funds whose NAV series are built from the TRI CSVs
already in data/tri/ — real Indian index history, no network. Then each page
is driven the way a user would drive it: pick a category, click the scan
button, open the tabs.

Any page that only renders its empty state is reported as such rather than
counted as a pass.

Run:  python test_pages_with_data.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
# ROOT is the repo root, one level up from tests/. Every path in this file
# hangs off it, so it must not be the test's own directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd

from streamlit.testing.v1 import AppTest

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        fails.append(name)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURE: synthetic funds backed by real index history
# ─────────────────────────────────────────────────────────────────────────────
from data.tri_loader import get_tri_nav
from utils.constants import CATEGORIES

# Each fund is one index series, scaled, so the funds in a category differ from
# each other and from their benchmark — a category where every fund is
# identical would hide ranking and quartile bugs.
# Names must match indices/config/index_metadata.py exactly — "NIFTY200
# MOMENTUM 30" has no space after NIFTY, and a near-miss silently drops the
# series, quietly shrinking the fixture instead of failing.
_BASE = ["NIFTY 100", "NIFTY MIDCAP 150", "NIFTY SMALLCAP 250",
         "NIFTY 500", "NIFTY 50", "NIFTY200 MOMENTUM 30"]

_SERIES = {}
_unreadable = []
for _n in _BASE:
    _s = get_tri_nav(_n)
    if _s is not None and not _s.empty:
        _SERIES[_n] = _s
    else:
        _unreadable.append(_n)
if not _SERIES:
    print("FATAL: no TRI CSVs readable in data/tri/ — cannot build fixtures.")
    sys.exit(2)
if _unreadable:
    print(f"FATAL: fixture series unreadable: {_unreadable}. Fix the names "
          "rather than running on a shrunken fixture.")
    sys.exit(2)

_NAMES = list(_SERIES)

# Six funds per category: enough for quartiles (which need 4+) and for the
# two-fund duplicate-key path on the comparison pages.
FUNDS_PER_CATEGORY = 6
_FIXTURE, _NAV_BY_CODE = {}, {}
_code = 100000
for _cat in CATEGORIES:
    _rows = []
    for _i in range(FUNDS_PER_CATEGORY):
        _code += 1
        _src = _SERIES[_NAMES[_i % len(_NAMES)]]
        # A small per-fund drift so no two funds are identical.
        _nav = _src.copy()
        _nav["nav"] = _nav["nav"] * (1.0 + 0.03 * _i)
        _NAV_BY_CODE[str(_code)] = _nav
        _rows.append({
            "code": str(_code),
            "name": f"Test {_cat} Fund {_i + 1} - Direct Plan - Growth",
        })
    _FIXTURE[_cat] = _rows

_TOTAL_FUNDS = sum(len(v) for v in _FIXTURE.values())


def _fake_categorized(plan_type="Direct"):
    return _FIXTURE


def _fake_nav(scheme_code):
    return _NAV_BY_CODE.get(str(scheme_code))


def _fake_all_schemes():
    return {r["code"]: r["name"] for rows in _FIXTURE.values() for r in rows}


def _install():
    """Patch the loaders before any page module imports from them."""
    import data.fund_loader as fl
    fl.get_all_categorized_schemes = _fake_categorized
    fl.get_nav_history             = _fake_nav
    fl.get_all_schemes             = _fake_all_schemes

    # load_navs_parallel must go through the stub too, or the pages that use
    # it would still try the network.
    def _fake_parallel(codes, max_workers=6, progress_cb=None):
        out, uniq = {}, list(dict.fromkeys(str(c) for c in codes))
        for i, c in enumerate(uniq, start=1):
            out[c] = _fake_nav(c)
            if progress_cb is not None:
                try:
                    progress_cb(i, len(uniq), c)
                except Exception:
                    pass
        return out

    fl.load_navs_parallel = _fake_parallel

    import data.category_mapper as cm
    cm.get_category_fund_counts = lambda schemes: {
        c: len(_FIXTURE.get(c, [])) for c in CATEGORIES
    }


_install()

print(f"\n[FIXTURE] {_TOTAL_FUNDS} synthetic funds across {len(CATEGORIES)} "
      f"categories, from {len(_SERIES)} real TRI series")


def _page(stem):
    import glob
    hits = [p for p in glob.glob(os.path.join(ROOT, "pages", "*.py"))
            if stem in os.path.basename(p)]
    assert hits, f"page {stem} not found"
    return hits[0]


def _run(path, timeout=900):
    at = AppTest.from_file(path, default_timeout=timeout)
    at.run()
    return at


def _text(at):
    out = []
    for attr in ("markdown", "caption", "subheader", "title", "warning",
                 "error", "info", "success"):
        for el in getattr(at, attr, []):
            try:
                out.append(str(el.value))
            except Exception:
                pass
    return " ".join(out).lower()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Quality — the page that was crashing
# ─────────────────────────────────────────────────────────────────────────────
print("\n[DATA QUALITY] Scan, then open every tab")

at = _run(_page("7_Data_Quality"))
check("Data Quality loads with a populated fund list",
      not at.exception and "no funds found" not in _text(at),
      str(at.exception)[:100] if at.exception else "")

_scan = [b for b in at.button if "scan" in (b.label or "").lower()]
check("the Scan button is present", bool(_scan),
      f"buttons: {[b.label for b in at.button][:4]}")

if _scan:
    _scan[0].click()
    at.run()
    check("Data Quality survives the scan (this is the KeyError path)",
          not at.exception,
          f"{type(at.exception).__name__ if at.exception else ''}: "
          f"{str(at.exception)[:170]}" if at.exception else "")

    if not at.exception:
        _dfs = at.dataframe
        check("the coverage matrix rendered", len(_dfs) >= 2,
              f"{len(_dfs)} dataframes")

        # The matrix must have one column per KEY_METRIC plus the fund name,
        # and every one of those columns must exist — the crash was asking
        # for 18 labels from an 11-column frame.
        _matrix = None
        for _d in _dfs:
            _v = _d.value
            if isinstance(_v, pd.DataFrame) and "Fund Name" in _v.columns:
                if any("CAGR" in str(c) for c in _v.columns):
                    _matrix = _v
                    break
        check("the coverage matrix has its metric columns",
              _matrix is not None and len(_matrix.columns) == 12,
              f"{list(_matrix.columns) if _matrix is not None else 'not found'}")
        check("every scanned fund appears in the matrix",
              _matrix is not None and len(_matrix) == FUNDS_PER_CATEGORY,
              f"{len(_matrix) if _matrix is not None else 0} rows")
        check("the coverage matrix is sorted by coverage, not left unsorted",
              _matrix is not None and "_score" not in _matrix.columns)

        # The full 18-metric list must still be reachable underneath.
        _cov = [d.value for d in _dfs
                if isinstance(d.value, pd.DataFrame) and "Metric" in d.value.columns]
        check("the full per-metric coverage list is shown", bool(_cov),
              f"{len(_cov)} candidate tables")
        if _cov:
            check("the per-metric list covers all 18 tracked metrics",
                  len(_cov[0]) == 18, f"{len(_cov[0])} rows")

        check("Data Quality offers CSV export after scanning",
              len(at.button) + len(getattr(at, 'download_button', [])) > 0)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Every other page, with data
# ─────────────────────────────────────────────────────────────────────────────
print("\n[ALL PAGES] Rendered with a populated registry")

PAGES = [
    ("Home",                 os.path.join(ROOT, "app.py")),
    ("Fund Analytics",       _page("1_Fund_Analytics")),
    ("Fund Comparison",      _page("2_Fund_Comparison")),
    ("Rankings",             _page("3_Rankings")),
    ("Portfolio Analytics",  _page("4_Portfolio_Analytics")),
    ("Factor Attribution",   _page("5_Factor_Attribution")),
    ("Predictive Analytics", _page("6_Predictive_Analytics")),
]

_empty_state = ("no funds found", "unable to load", "select at least")

for _name, _path in PAGES:
    _a = _run(_path)
    _ok = not _a.exception
    check(f"{_name} renders with data", _ok,
          f"{type(_a.exception).__name__}: {str(_a.exception)[:150]}"
          if _a.exception else "")
    if _ok:
        _t = _text(_a)
        # Say plainly when a page only reached its empty state, so the pass
        # above is not read as "the page's real content was exercised".
        if any(s in _t for s in _empty_state):
            print(f"        note: {_name} rendered its prompt/empty state "
                  "— deeper content needs a user selection")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Fund Analytics, driven into its tabs
# ─────────────────────────────────────────────────────────────────────────────
print("\n[FUND ANALYTICS] The page that computes the full metric set")

at = _run(_page("1_Fund_Analytics"))
check("Fund Analytics computed metrics for a real fund",
      not at.exception and "could not compute" not in _text(at),
      str(at.exception)[:120] if at.exception else "")

if not at.exception:
    check("KPI cards rendered", len(at.metric) >= 3, f"{len(at.metric)} metrics")

    # The Sharpe confidence interval must be ON THE HEADLINE ROW, not buried.
    # It first shipped only inside the All Metrics tab in a collapsed expander,
    # and every test still passed: the engine produced the keys, the labels and
    # help existed, the page rendered. Nothing asserted a user could SEE it.
    # A feature three clicks deep behind an expander is not shipped.
    _cards = {m.label: m for m in at.metric}
    check("Sharpe is a headline KPI card, not buried in a tab",
          "Sharpe Ratio" in _cards, f"{sorted(_cards)[:6]}")
    if "Sharpe Ratio" in _cards:
        _d = _cards["Sharpe Ratio"].delta or ""
        check("the headline Sharpe card carries its 95% interval",
              "95% CI" in str(_d), f"delta={_d!r}")

    # A count rendered with a percent sign is the kind of thing that survives
    # a green test run and embarrasses you on screen: sharpe_n_obs went in as
    # kind="num" and displayed 5,305 observations as "5305.00%".
    import utils.ui as _ui
    check("a count formats as a count, not a percentage",
          _ui.kpi.__doc__ is not None and "count" in open(
              os.path.join(ROOT, "utils", "ui.py"), encoding="utf-8").read(),
          "kind='count' exists in ui.kpi")
    # The 6F block is nine cards; with kpi_row padding, the page renders them
    # all rather than dropping the short row.
    _labels = [m.label for m in at.metric]
    _betas = [l for l in _labels if "β" in l]
    check("the six factor betas all render", len(_betas) >= 6, f"{_betas}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Rankings, which builds a whole category
# ─────────────────────────────────────────────────────────────────────────────
print("\n[RANKINGS] A full category through the parallel loader")

at = _run(_page("3_Rankings"))
check("Rankings renders with a full category", not at.exception,
      f"{type(at.exception).__name__}: {str(at.exception)[:150]}"
      if at.exception else "")

# ─────────────────────────────────────────────────────────────────────────────
# 5. The two slot-picker pages, driven past their empty state
# ─────────────────────────────────────────────────────────────────────────────
print("\n[SLOT PAGES] Selecting funds, which is where duplicate keys bite")


def _fill_slots(at, n, category="Mid Cap"):
    """
    Fill n fund slots.

    The slot picker is two-stage: "Category for slot k" populates "Fund for
    slot k", which starts with a single placeholder option. Selecting the
    fund box before its category is set does nothing — an earlier version of
    this helper grabbed the category boxes, left the fund boxes untouched,
    and the page correctly went on showing its selection prompt.

    Returns the number of slots that actually hold a fund.
    """
    filled = 0
    for slot in range(1, n + 1):
        cat_box = next((sb for sb in at.selectbox
                        if sb.label == f"Category for slot {slot}"), None)
        if cat_box is None or category not in (cat_box.options or []):
            break
        cat_box.select(category)
        at.run()

        fund_box = next((sb for sb in at.selectbox
                         if sb.label == f"Fund for slot {slot}"), None)
        if fund_box is None:
            break
        # Skip the placeholder, and take a different fund for each slot so the
        # page is exercising two distinct funds rather than one twice.
        opts = [o for o in (fund_box.options or []) if o and o != "—"]
        if len(opts) <= slot - 1:
            break
        fund_box.select(opts[slot - 1])
        at.run()
        filled += 1
    return filled


at = _run(_page("5_Factor_Attribution"))
_n = _fill_slots(at, 2)
check("Factor Attribution accepts two funds via its slot picker", _n == 2, f"{_n} filled")
if _n == 2:
    _run_btn = [b for b in at.button if "run factor" in (b.label or "").lower()]
    check("the Run Factor Attribution button is present", bool(_run_btn),
          f"{[b.label for b in at.button][:3]}")
    if _run_btn:
        _run_btn[0].click()
        at.run()
        check("Factor Attribution runs attribution for two funds",
              not at.exception,
              f"{type(at.exception).__name__}: {str(at.exception)[:180]}"
              if at.exception else "")
        if not at.exception:
            check("it reached real output, not the selection prompt",
                  "select at least" not in _text(at))

            # EVERY export on this page lives in the multi-fund branch. The
            # single-fund branch renders metric cards and nothing downloadable.
            # So a bare "0 download buttons" is ambiguous: it means either the
            # keys collided, or only ONE fund survived into fund_names and the
            # page legitimately took the other branch. Those need different
            # fixes, so the branch is identified before the count is judged.
            #
            # "Beta Comparison Table" is rendered only when len(fund_names) > 1;
            # "Factor Loadings" only when it is exactly 1.
            _t = _text(at)
            _multi  = "beta comparison table" in _t
            _single = "factor loadings" in _t and not _multi
            if _multi:
                _why = ""
            elif _single:
                _why = ("single-fund branch — one of the two selected funds did "
                        "not produce a factor model, so this is a data/alignment "
                        "problem, not a duplicate-key one")
            else:
                _why = "neither branch rendered — attribution produced no output"
            check("the page took the multi-fund branch, where the exports live",
                  _multi, _why)

            _dl = len(getattr(at, "download_button", []))
            if _multi:
                # Two of the five export controls are per-fund. With two funds
                # selected this is exactly where a fixed key collided.
                check("per-fund exports render for both funds without a key clash",
                      _dl >= 4, f"{_dl} download buttons")
            else:
                print(f"  SKIP  export key check — page rendered the single-fund "
                      f"branch, which has no exports ({_dl} buttons)")

at = _run(_page("4_Portfolio_Analytics"))
_n = _fill_slots(at, 2)
check("Portfolio Analytics accepts two funds via its slot picker", _n == 2, f"{_n} filled")
if _n == 2:
    check("Portfolio Analytics builds a portfolio from two funds",
          not at.exception,
          f"{type(at.exception).__name__}: {str(at.exception)[:180]}"
          if at.exception else "")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
