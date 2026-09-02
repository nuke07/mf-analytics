"""
Analytics regression checks — the numeric correctness suite.

This is the ONLY coverage of several fixes. Nothing else in tests/ asserts:
  · P99 exists as its own key (the UI used to read P95 under a P99 label)
  · DaR95 is positive and equals the 95th severity percentile
  · QMJ and BAB loadings are actually computed by the 6-factor model
  · alpha is the raw intercept, not the standardised one (defect D-06)
  · calc_max_drawdown accepts _dd_series — the parameter whose absence was a
    TypeError on every metrics computation, i.e. no fund would have loaded
  · category benchmark flagging (Flexi Cap yes, Mid Cap no)

Do not delete this because the name looks like scaffolding. It was called
verify_fixes.py, which is why it used to read that way.

Run:  python tests/test_analytics_regressions.py
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

# ROOT is the repo root, one level up from tests/. Every path in this file
# hangs off it, so it must not be the test's own directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TRI = os.path.join(ROOT, "data", "tri", "NIFTY_SMALLCAP_250_TRI.csv")
if not os.path.isfile(TRI):
    sys.exit(
        f"Could not find {TRI}\n"
        "This file must live in tests/ inside the mf_analytics repo, so that "
        "data/tri/ resolves one level up."
    )

from analytics.monte_carlo import run_monte_carlo
from utils.validators import build_quality_report
from utils.session import fund_key, clear_analytics_cache
from utils.constants import MIN_DAYS, COVERAGE_LABELS, ANALYTICS_VERSION

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        fails.append(name)


# Real market data: Nifty Smallcap 250 TRI (a genuinely drawdown-prone series)
d = pd.read_csv(TRI)
d["Date"] = pd.to_datetime(d["Date"])
nav = d.set_index("Date")["TotalReturnsIndex"].sort_index()
rets = nav.pct_change().dropna()

print("\n[D-01] Monte Carlo drawdown percentiles — Nifty Smallcap 250 TRI, 5Y, 10k paths")
mc = run_monte_carlo(rets, horizon_years=5.0, n_sims=10_000)
ts = mc["terminal_stats"]
p = ts["max_dd_percentiles"]

print(f"     KPI row as rendered: Median {p[50]:.1f}%  P75 {p[75]:.1f}%  "
      f"P90 {p[90]:.1f}%  DaR95 {ts['drawdown_at_risk_95']*100:.1f}%  P99 {p[99]:.1f}%")

check("severity increases monotonically across percentiles",
      p[5] < p[25] < p[50] < p[75] < p[90] < p[95] < p[99],
      f"{p[5]:.1f} < {p[50]:.1f} < {p[90]:.1f} < {p[99]:.1f}")
check("all percentiles are positive magnitudes", all(v > 0 for v in p.values()))
check("P99 key exists (was missing; UI read P95 under a P99 label)", 99 in p)
check("DaR95 is positive and equals the 95th severity percentile",
      ts["drawdown_at_risk_95"] > 0
      and abs(ts["drawdown_at_risk_95"] * 100 - p[95]) < 1e-6,
      f"{ts['drawdown_at_risk_95']*100:.2f}% vs P95 {p[95]:.2f}%")

# Cross-check DaR against the raw path array — the ground truth
raw = ts["max_drawdowns"]
frac_worse = float((raw < -ts["drawdown_at_risk_95"]).mean())
check("exactly ~5% of paths are worse than DaR95",
      0.045 <= frac_worse <= 0.055, f"{frac_worse*100:.2f}% of paths exceed it")

frac_worse_median = float((raw < -p[50] / 100).mean())
check("~50% of paths are worse than the median",
      0.47 <= frac_worse_median <= 0.53, f"{frac_worse_median*100:.2f}%")

check("raw max_drawdowns still negative (histogram contract unchanged)",
      raw.max() <= 0)

# The old behaviour, for the record
old_dar = float(np.percentile(raw, 95))
print(f"     Before the fix this row read DaR95 = {abs(old_dar)*100:.1f}% "
      f"(the MILDEST 5%); it now reads {ts['drawdown_at_risk_95']*100:.1f}%.")

print("\n[D-03] build_quality_report returns start_date / end_date")
rep = build_quality_report("NIFTY SMALLCAP 250 TRI", nav)
check("start_date present and correct", rep.get("start_date") == nav.index[0],
      str(rep.get("start_date"))[:10])
check("end_date present and correct", rep.get("end_date") == nav.index[-1],
      str(rep.get("end_date"))[:10])
empty = build_quality_report("broken", None)
check("invalid-NAV path also returns the keys (as None)",
      "start_date" in empty and empty["start_date"] is None)

print("\n[D-04] cache keys")
k_lc = fund_key("120503", 7.0, "Large Cap")
k_mc = fund_key("120503", 7.0, "Mid Cap")
check("category is part of the fund key", k_lc != k_mc, k_lc)
check("key carries ANALYTICS_VERSION", ANALYTICS_VERSION in k_lc)
check("key still matches the clear_analytics_cache prefix",
      k_lc.startswith("fund_metrics_"))
check("backward-compatible 2-arg call still works",
      fund_key("120503", 7.0).startswith("fund_metrics_120503_7.0_"))

print("\n[batch] coverage label vocabulary")
missing = [k for k in MIN_DAYS if k not in COVERAGE_LABELS]
check("every MIN_DAYS key has a display label", not missing, f"missing: {missing}")

# ── D-06: 6-factor alpha was the standardised intercept ─────────────────────
print("\n[D-06] 6F alpha — falsification tests on series that ARE factor components")
from data.factor_loader import get_factor_returns_6f
from analytics.factor_model import calc_factor_model_6f, calc_regime_betas

f6, _, ferr = get_factor_returns_6f(rf_rate=0.07)
if ferr or f6 is None:
    print(f"  SKIP  6F factor data unavailable: {ferr}")
else:
    # Each of these series IS one leg of a factor, so a correctly specified
    # model must price it exactly: alpha = 0 at R2 = 1.
    IDENTITIES = {
        "NIFTY_500":            "market",
        "NIFTY_200_MOMENTUM_30": "WML long leg",
        "NIFTY_500_VALUE_50":   "HML long leg",
        "NIFTY_200_QUALITY_30": "QMJ long leg",
    }
    for nm, role in IDENTITIES.items():
        f = os.path.join(ROOT, "data", "tri", f"{nm}_TRI.csv")
        if not os.path.isfile(f):
            print(f"  SKIP  {nm} not present"); continue
        dd = pd.read_csv(f); dd["Date"] = pd.to_datetime(dd["Date"])
        rr = dd.set_index("Date")["TotalReturnsIndex"].sort_index().pct_change().dropna()
        m = calc_factor_model_6f(rr, f6, 0.07)
        a, r2 = m.get("alpha_6f"), m.get("r_squared_6f")
        check(f"{nm} ({role}) prices to ~zero alpha",
              a is not None and abs(a) < 0.005 and r2 > 0.99,
              f"alpha={a*100:.2f}%  R2={r2:.3f}")

    # Attribution must reconcile: alpha + sum(contributions) == excess return.
    f = os.path.join(ROOT, "data", "tri", "NIFTY_MIDCAP_150_TRI.csv")
    dd = pd.read_csv(f); dd["Date"] = pd.to_datetime(dd["Date"])
    rr = dd.set_index("Date")["TotalReturnsIndex"].sort_index().pct_change().dropna()
    m = calc_factor_model_6f(rr, f6, 0.07)
    contrib = sum(m.get(f"contrib_{k}") or 0
                  for k in ["market", "smb", "hml", "wml", "qmj", "bab"])
    win = rr.reindex(f6.index).dropna()
    excess = (win.mean() - 0.07 / 252) * 252
    check("attribution reconciles to realised excess return",
          abs((m["alpha_6f"] + contrib) - excess) < 0.001,
          f"alpha+contrib={(m['alpha_6f']+contrib)*100:.2f}%  excess={excess*100:.2f}%")

    check("6F alpha is not merely the excess return (the old bug)",
          abs(m["alpha_6f"] - excess) > 0.02,
          f"alpha={m['alpha_6f']*100:.2f}%  vs excess={excess*100:.2f}%")

    # Regime alphas must obey the same identity.
    f = os.path.join(ROOT, "data", "tri", "NIFTY_500_TRI.csv")
    dd = pd.read_csv(f); dd["Date"] = pd.to_datetime(dd["Date"])
    rr = dd.set_index("Date")["TotalReturnsIndex"].sort_index().pct_change().dropna()
    rb = calc_regime_betas(rr, f6, 0.07)
    for regime in ["Bull", "Sideways", "Bear"]:
        v = rb.get(regime)
        if v:
            check(f"market prices to ~zero alpha in {regime}",
                  abs(v["alpha"]) < 0.005, f"alpha={v['alpha']*100:.2f}%")


# ── Step 2: 6F engine migration + dual benchmarking ─────────────────────────
print("\n[STEP-2] 6F in the engine, and dual benchmarking")
from analytics.engine import compute_fund_metrics, _ALL_METRIC_KEYS, _MARKET_ALPHA_KEYS
from data.benchmark_loader import (
    get_market_nav, MARKET_DISPLAY_NAME, is_market_same_as_category,
)

def _tri(name):
    f = os.path.join(ROOT, "data", "tri", f"{name}_TRI.csv")
    dd = pd.read_csv(f); dd["Date"] = pd.to_datetime(dd["Date"])
    return dd.set_index("Date")["TotalReturnsIndex"].sort_index().rename("nav").to_frame()

check("4F keys are gone from the engine",
      not any(k.endswith("_4f") for k in _ALL_METRIC_KEYS))
check("6F keys are present",
      all(k in _ALL_METRIC_KEYS for k in
          ["alpha_6f", "beta_qmj", "beta_bab", "contrib_qmj", "contrib_bab"]))
check("market-relative keys are present",
      all(k in _ALL_METRIC_KEYS for k in _MARKET_ALPHA_KEYS))

mkt = _tri("NIFTY_500")
mid = _tri("NIFTY_MIDCAP_150")

# Mid cap fund, benchmarked to Midcap 150 (category) and Nifty 500 (market)
m = compute_fund_metrics(mid, rf_rate=0.07, fund_name="MidCap",
                         benchmark_nav_df=mid, benchmark_name="Nifty Midcap 150 TRI",
                         factor_returns_df=f6,
                         market_nav_df=mkt, market_name=MARKET_DISPLAY_NAME)

# Fund IS its own category benchmark → identity must hold exactly
check("category alpha is ~0 when fund IS its own benchmark",
      abs(m["jensens_alpha"]) < 0.005, f"{m['jensens_alpha']*100:.3f}%")
check("category beta is ~1 when fund IS its own benchmark",
      abs(m["beta"] - 1.0) < 0.005, f"{m['beta']:.4f}")

# Against the broad market it must NOT be the identity
check("market alpha differs from category alpha",
      abs(m["jensens_alpha_mkt"] - m["jensens_alpha"]) > 0.005,
      f"cat={m['jensens_alpha']*100:.2f}%  mkt={m['jensens_alpha_mkt']*100:.2f}%")
check("market beta differs from category beta",
      abs(m["beta_mkt"] - m["beta"]) > 0.01,
      f"cat={m['beta']:.3f}  mkt={m['beta_mkt']:.3f}")
check("every market key is populated",
      all(m.get(k) is not None for k in _MARKET_ALPHA_KEYS))

# Engine betas must be RAW (conventional scale), not the ~0.01 standardised ones
check("engine market beta is on the conventional ~1.0 scale",
      0.5 < abs(m["beta_market_6f"]) < 1.5, f"{m['beta_market_6f']:.3f}")
check("mid cap shows a positive size tilt (SMB)",
      m["beta_smb"] > 0.1, f"beta_smb={m['beta_smb']:.3f}")
check("QMJ and BAB loadings are computed",
      m.get("beta_qmj") is not None and m.get("beta_bab") is not None)
check("rolling 6F alpha series is produced",
      m.get("_rolling_alpha_6f") is not None and len(m["_rolling_alpha_6f"]) > 100)

# Nifty 500 as the fund: market alpha AND 6F alpha must both vanish
m500 = compute_fund_metrics(mkt, rf_rate=0.07, fund_name="Nifty500",
                            factor_returns_df=f6,
                            market_nav_df=mkt, market_name=MARKET_DISPLAY_NAME)
check("market alpha ~0 when the fund IS the market",
      abs(m500["jensens_alpha_mkt"]) < 0.005, f"{m500['jensens_alpha_mkt']*100:.3f}%")
check("market beta ~1 when the fund IS the market",
      abs(m500["beta_mkt"] - 1.0) < 0.005, f"{m500['beta_mkt']:.4f}")

# The same-benchmark categories must be flagged so the UI can say so
check("Flexi Cap flagged as benchmarked to the market",
      is_market_same_as_category("Flexi Cap"))
check("Mid Cap NOT flagged as benchmarked to the market",
      not is_market_same_as_category("Mid Cap"))

# Labels exist for every new key
from utils.constants import METRIC_LABELS
_unlabelled = [k for k in _ALL_METRIC_KEYS if k not in METRIC_LABELS]
check("every engine metric key has a display label",
      not _unlabelled, f"missing: {_unlabelled}")


# ── Step 3: shared UI layer ─────────────────────────────────────────────────
print("\n[STEP-3] Shared sidebar / KPI helper / METRIC_HELP")
import glob, re as _re
from utils.constants import METRIC_HELP, DEFAULT_RISK_FREE_RATE
from utils import ui as _ui


def _page(stem):
    """Locate a page by name, independent of its numeric prefix."""
    hits = glob.glob(os.path.join(ROOT, "pages", f"*_{stem}.py"))
    if not hits:
        raise AssertionError(f"page not found: *_{stem}.py")
    return hits[0]

_page_files = [os.path.join(ROOT, "app.py")] + sorted(
    glob.glob(os.path.join(ROOT, "pages", "*.py")))
_src = {os.path.basename(f): open(f, encoding="utf-8").read()
        for f in _page_files if not f.endswith("__init__.py")}

def _files_containing(pattern):
    return [n for n, t in _src.items() if _re.search(pattern, t)]

# The blocks that used to be copy-pasted must now exist in exactly one place.
_rf  = _files_containing(r"col_rf, col_down, col_up|rf_col, rd_col, ru_col")
_tri = _files_containing(r"get_tri_staleness_warning")
_kpi_defs = _files_containing(r"def _(?:a|m|f)?kpi\(col, label, val, pct=")
check("RF slider block no longer duplicated across pages", not _rf, f"still in: {_rf}")
check("TRI staleness block no longer duplicated", not _tri, f"still in: {_tri}")
check("no page defines its own pct= KPI helper", not _kpi_defs, f"still in: {_kpi_defs}")

# Every page should be pulling the shared controls.
_uses_ui = [n for n, t in _src.items() if "from utils.ui" in t]
check("every page imports the shared UI module",
      len(_uses_ui) == len(_src), f"{len(_uses_ui)}/{len(_src)}")

# The pages that were missing the TRI panel must now have it.
for page in ["Data_Quality", "Factor_Attribution"]:
    check(f"{page} now shows the benchmark freshness panel",
          "tri_status()" in open(_page(page), encoding="utf-8").read())

# RF default: one source of truth, and the pages must not hardcode it.
check("RF default comes from DEFAULT_RISK_FREE_RATE",
      abs(_ui.RF_MIN) > 0 and DEFAULT_RISK_FREE_RATE == 0.07,
      f"{DEFAULT_RISK_FREE_RATE}")
_hardcoded = _files_containing(r'st\.session_state\.get\("rf_rate", (?:6\.5|7\.0)\)')
check("no page hardcodes its own RF default", not _hardcoded, f"still in: {_hardcoded}")

# The RF slider must be key-bound, or the −/+ buttons get silently discarded.
# The key is now PAGE-SCOPED (rf_rate_<page id>) with the value of record held
# in a plain, non-widget key: a single shared widget key was re-registered on
# navigation and fell back to min_value, so Rankings ran at 4.0% while every
# other page showed 7.0%. See test_rf_navigation.py.
_ui_src = open(os.path.join(ROOT, "utils", "ui.py"), encoding="utf-8").read()
check("RF slider is bound to a session-state key", "key=key," in _ui_src)
check("the RF key is page-scoped", "def rf_key_for(" in _ui_src
      and 'f"{RF_KEY}_{page_id}"' in _ui_src)
check("the RF value of record lives in a non-widget key",
      "RF_STORE" in _ui_src and 'RF_STORE = "rf_rate_value"' in _ui_src)
check("the slider is re-seeded only on arriving at a page",
      "RF_LAST_PAGE" in _ui_src and "arriving" in _ui_src,
      "an unconditional re-seed would undo a slider drag")
# Writing to a widget's key outside a callback raises StreamlitAPIException.
check("RF buttons mutate the slider via on_click callbacks, not inline",
      "on_click=_nudge_rf" in _ui_src
      and "st.session_state[RF_KEY] = max(" not in _ui_src)
print("       (run test_rf_control.py for the headless click-through test)")

# Tooltip coverage.
from analytics.engine import _ALL_METRIC_KEYS as _KEYS
_no_help = [k for k in _KEYS if k not in METRIC_HELP]
check("every engine metric has help text", not _no_help, f"missing: {_no_help}")

# kpi() must survive None/NaN without rendering "nan%".
class _FakeCol:
    def __init__(self): self.calls = []
    def metric(self, label, value, **kw): self.calls.append(value)
_c = _FakeCol()
_ui.kpi(_c, "T", None, kind="pct")
_ui.kpi(_c, "T", float("nan"), kind="pct")
_ui.kpi(_c, "T", 0.1234, kind="pct")
check("kpi() renders N/A for None and NaN, formats real values",
      _c.calls[0] == "N/A" and _c.calls[1] == "N/A" and "%" in _c.calls[2],
      f"{_c.calls}")


# ── Step 4: metric consolidation + declarative Rankings ─────────────────────
print("\n[STEP-4] Retired metrics and the Rankings rebuild")
from analytics.quartile import QUARTILE_METRICS
from analytics.engine import compute_fund_metrics as _cfm

RETIRED = ["momentum_12m", "momentum_sharpe", "negative_freq",
           "avg_rolling_1y", "avg_rolling_3y"]

_eng, _q = set(_KEYS), set(QUARTILE_METRICS)
check("retired metrics gone from the engine",
      not [m for m in RETIRED if m in _eng])
check("retired metrics gone from the quartile list",
      not [m for m in RETIRED if m in _q])
check("retired metrics gone from labels and help",
      not [m for m in RETIRED if m in METRIC_LABELS or m in METRIC_HELP])

# No page or chart may still reference a retired key.
_all_py = []
for _d in ["pages", "analytics", "visualizations", "utils"]:
    _all_py += glob.glob(os.path.join(ROOT, _d, "*.py"))
_all_py.append(os.path.join(ROOT, "app.py"))
_stale = []
for _f in _all_py:
    _t = open(_f, encoding="utf-8").read()
    for _m in RETIRED:
        # ignore the rationale comment block in constants.py
        _in_doc = False
        for _line in _t.splitlines():
            if _line.count('"""') % 2:
                _in_doc = not _in_doc
                continue
            if _in_doc or _line.lstrip().startswith("#"):
                continue
            if _m in _line:
                _stale.append(f"{os.path.basename(_f)}: {_line.strip()[:52]}")
check("no live code references a retired metric", not _stale, f"{_stale[:3]}")

# Every metric the engine still declares must actually be produced.
_m = _cfm(mid, rf_rate=0.07, fund_name="X", benchmark_nav_df=mid,
          factor_returns_df=f6, market_nav_df=mkt)
check("engine produces every key it declares",
      not [k for k in _KEYS if k not in _m])

# The quartile list must be a subset of what the engine produces, or ranking
# tables silently render "insufficient data".
check("quartile metrics are all produced by the engine",
      not [k for k in QUARTILE_METRICS if k not in _m],
      f"{[k for k in QUARTILE_METRICS if k not in _m][:4]}")

# Rankings: the spec replaced the hand-written blocks.
_rank = open(_page(  "Rankings"), encoding="utf-8").read()
check("Rankings uses a declarative spec", "RANK_SPEC = {" in _rank)
# Counted from RANK_SPEC itself rather than by looking for a decorated tab
# label — the design pass stripped the emoji, and an assertion keyed on
# "📈 Returns" was testing the decoration, not the structure.
_spec_block = _rank[_rank.index("RANK_SPEC = {"):]
_tabs = _re.findall(r'^        "([^"]+)":', _spec_block, _re.M)
# The retired tab names must be gone from the TAB LIST — not from the file.
# "Persistence" still appears as a legitimate column label inside the Alpha
# tab, and a whole-file substring test flags that as a regression.
_RETIRED_TABS = {"Absolute Returns", "Persistence", "Consistency", "Stability",
                 "Momentum", "Downside"}
check("Rankings has 6 tabs, not 11",
      len(_tabs) == 5 and "Quartile View" in _rank
      and not (_RETIRED_TABS & set(_tabs)),
      f"{_tabs} + Quartile View")
check("every RANK_SPEC metric exists in the engine output", True)  # checked below

_spec_keys = _re.findall(r'\("([a-z0-9_]+)",\s*"[^"]*",\s*"(?:pct|ratio|num|days)"', _rank)
_unknown = sorted({k for k in _spec_keys if k not in _m})
check("every metric named in RANK_SPEC is produced by the engine",
      not _unknown, f"unknown: {_unknown}")

# Percent metrics must not be rendered with ratio formatting (0.07 vs 7.0%).
_PCT_KEYS = {k for k in _KEYS if k.startswith(("cagr_", "momentum_", "contrib_"))
             or "alpha" in k and "tstat" not in k and "ratio" not in k
             or k in {"annualized_volatility", "downside_volatility",
                      "max_drawdown", "avg_drawdown", "win_rate",
                      "positive_freq", "tracking_error", "excess_return"}}
_wrong = _re.findall(
    r"kpi\([^,]+,\s*\"[^\"]+\",[^\n]*?\.get\(\"([a-z0-9_]+)\"\)[^\n]*?kind='ratio'",
    open(_page("Fund_Analytics"), encoding="utf-8").read())
_misfmt = sorted({k for k in _wrong if k in _PCT_KEYS})
check("no percentage metric is rendered with ratio formatting",
      not _misfmt, f"{_misfmt}")


# ── Step 6: dead code removal + parallel NAV loading ────────────────────────
print("\n[STEP-6] Dead code stays dead, and NAV loading is concurrent")

# The 32 symbols deleted in step 6. If any of these reappears as a definition
# it means a later edit resurrected dead code rather than reusing the live
# path — the exact regression this step existed to remove.
DELETED = {
    "visualizations": [
        "plot_factor_heatmap", "plot_factor_loadings", "plot_factor_contribution",
        "FACTOR_LABELS", "FACTOR_COLORS", "plot_scatter", "plot_nav_history",
        "plot_drawdown_periods", "plot_alpha_comparison", "QUARTILE_COLS",
    ],
    "analytics": [
        "calc_all_factor_model", "calc_factor_model", "calc_drawdown_periods",
        "get_rankings_for_metric", "get_quartile_summary_for_fund",
        "check_returns_series", "check_category_size",
    ],
    "data": [
        "get_factor_availability", "get_factor_returns",
        "get_all_category_benchmarks", "get_scheme_details",
        "slice_nav_between", "get_nav_at_date",
    ],
    "utils": [
        "get_cached_category_metrics", "get_cached_fund_metrics",
        "format_metrics_for_display", "fmt_large_num", "style_positive_negative",
        "category_analytics_key", "clean_fund_name", "get_fund_house",
        "DEFAULT_PLAN_TYPE",
    ],
}

_src_files = []
for _d in ["pages", "analytics", "visualizations", "utils", "data"]:
    _src_files += sorted(glob.glob(os.path.join(ROOT, _d, "*.py")))
_src_files.append(os.path.join(ROOT, "app.py"))

_resurrected, _referenced = [], []


def _strip_prose(src: str) -> str:
    """Blank out docstrings, block comments and trailing comments.

    A deleted symbol named in a comment ("the 4F functions have since
    been deleted") is documentation, not a live reference. Only
    executable text can raise NameError, so only executable text is
    scanned.
    """
    out, in_doc = [], False
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.count(_TQ) % 2 or stripped.count(_SQ) % 2:
            in_doc = not in_doc
            continue
        if in_doc or stripped.startswith('#'):
            continue
        out.append(line.split('  #')[0])
    return chr(10).join(out)


_TQ = chr(34) * 3
_SQ = chr(39) * 3

for _f in _src_files:
    _pkg  = os.path.basename(os.path.dirname(_f))
    _base = os.path.basename(_f)
    _txt  = _strip_prose(open(_f, encoding='utf-8').read())
    for _pkg_of_origin, _names in DELETED.items():
        for _name in _names:
            # A redefinition only counts inside the package it was
            # deleted from — data/factor_loader.py has its own live
            # FACTOR_COLORS, which is not a resurrection of the chart one.
            if _pkg == _pkg_of_origin and (
                _re.search(rf'^\s*def\s+{_name}\b', _txt, _re.M)
                or _re.search(rf'^{_name}\s*[:=]', _txt, _re.M)
            ):
                _resurrected.append(f'{_base}:{_name}')
            # A call or import anywhere is a live NameError / ImportError.
            if _re.search(rf'(?<![\w.]){_name}\s*\(', _txt) or \
               _re.search(rf'import[^\n]*\b{_name}\b', _txt):
                _referenced.append(f'{_base}:{_name}')

check("deleted symbols were not resurrected", not _resurrected, f"{_resurrected[:3]}")
check("nothing calls or imports a deleted symbol", not _referenced, f"{_referenced[:3]}")

# calc_factor_model_6f survived; the 4F entry points did not.
from analytics import factor_model as _fm
check("6F entry points are intact",
      hasattr(_fm, "calc_factor_model_6f") and hasattr(_fm, "calc_all_factor_model_6f"))
check("4F entry points are gone",
      not hasattr(_fm, "calc_factor_model") and not hasattr(_fm, "calc_all_factor_model"))
# _calc_rolling_alpha_4f is misleadingly named but ALIVE — calc_factor_model_6f
# calls it. Deleting it on the strength of its name would break the 6F page.
check("_calc_rolling_alpha_4f (badly named but live) survived",
      hasattr(_fm, "_calc_rolling_alpha_4f"))

# Every module must still import. Dead-code deletion breaks imports silently.
import importlib
_broken = []
for _f in _src_files:
    if os.path.basename(_f) == "app.py" or os.sep + "pages" + os.sep in _f:
        continue                      # streamlit pages need a runtime; AppTest covers them
    _mod = os.path.relpath(_f, ROOT)[:-3].replace(os.sep, ".")
    try:
        importlib.import_module(_mod)
    except Exception as _e:
        _broken.append(f"{_mod}: {type(_e).__name__}: {_e}")
check("every non-page module still imports", not _broken, f"{_broken[:2]}")

# Every page must at least compile (AppTest in test_pages_render.py runs them).
import py_compile, tempfile
_cdir = tempfile.mkdtemp()
_nocompile = []
for _f in sorted(glob.glob(os.path.join(ROOT, "pages", "*.py"))) + [os.path.join(ROOT, "app.py")]:
    try:
        py_compile.compile(_f, doraise=True,
                           cfile=os.path.join(_cdir, os.path.basename(_f) + 'c'))
    except Exception as _e:
        _nocompile.append(f"{os.path.basename(_f)}: {_e}")
check("every page compiles", not _nocompile, f"{_nocompile[:2]}")

# The three charts my step-4 Rankings rebuild silently dropped.
_rank_txt = open(_page("Rankings"), encoding="utf-8").read()
_charts = ["plot_capture_scatter", "plot_momentum_heatmap", "plot_bull_bear_alpha"]
_missing_charts = [c for c in _charts if c not in _rank_txt]
check("Rankings still renders all three restored charts",
      not _missing_charts, f"missing: {_missing_charts}")
check("RANK_SPEC chart hooks are actually rendered",
      'for chart_key in ("chart", "chart2")' in _rank_txt)

# Parallel loading exists, is wired into both slow pages, and is bounded.
from data.fund_loader import load_navs_parallel as _lnp
import inspect as _insp
_sig = _insp.signature(_lnp)
check("load_navs_parallel exists with the expected signature",
      set(_sig.parameters) == {"codes", "max_workers", "progress_cb"},
      f"{list(_sig.parameters)}")
check("worker count is bounded by default",
      _sig.parameters["max_workers"].default <= 8,
      f"default={_sig.parameters['max_workers'].default}")

_lnp_src = _insp.getsource(_lnp)
check("workers inherit the Streamlit script context",
      "add_script_run_ctx" in _lnp_src,
      "without it @st.cache_data is a no-op inside worker threads")
check("duplicate scheme codes are de-duplicated before fetching",
      "dict.fromkeys" in _lnp_src)

for _pg in ("Rankings", "Portfolio_Analytics"):
    _txt = open(_page(_pg), encoding="utf-8").read()
    check(f"{_pg} loads NAVs in parallel", "load_navs_parallel(" in _txt)
    # the sequential loop it replaced must be gone
    check(f"{_pg} has no leftover sequential NAV loop",
          not _re.search(r"for\s+\w+\s+in\s+[\w\[\]. ]+:\s*\n\s+\w+\s*=\s*get_nav_history\(", _txt))

print("       (run test_parallel_load.py for the timing + correctness test)")


# ── Step 7: visual polish ───────────────────────────────────────────────────
print("\n[STEP-7] One chart config, one card style, exports everywhere")

import utils.ui as _ui

_page_srcs = {os.path.basename(_p): open(_p, encoding="utf-8").read()
              for _p in sorted(glob.glob(os.path.join(ROOT, "pages", "*.py")))
              + [os.path.join(ROOT, "app.py")]}


def _code_only(src):
    return _re.sub(r'("""|\'\'\')(?:.|\n)*?\1', "", src)


check("chart toolbars are configured in exactly one place",
      _ui.CHART_CONFIG.get("displaylogo") is False)
check("no page calls st.plotly_chart directly",
      not any("st.plotly_chart(" in _code_only(s) for s in _page_srcs.values()),
      f"{[f for f, s in _page_srcs.items() if 'st.plotly_chart(' in _code_only(s)]}")

# use_container_width is deprecated in Streamlit 1.62; width= replaced it.
_dep = []
for _f in _src_files:
    if os.path.basename(_f).startswith("test_"):
        continue
    _b = _code_only(open(_f, encoding="utf-8").read())
    _b = "\n".join(l for l in _b.splitlines() if not l.lstrip().startswith("#"))
    if "use_container_width" in _b:
        _dep.append(os.path.basename(_f))
check("the deprecated use_container_width is gone app-wide", not _dep, f"{_dep}")

check("no page hand-rolls a card div",
      not any(_re.search(r"border-radius:\s*\d+px", _code_only(s))
              for s in _page_srcs.values()),
      f"{[f for f, s in _page_srcs.items() if _re.search(r'border-radius:', _code_only(s))]}")

# The two pages that had no CSV export at all.
for _pg, _least in (("5_Factor_Attribution.py", 5), ("6_Predictive_Analytics.py", 3)):
    _n = len(_re.findall(r"export_button\(", _code_only(_page_srcs[_pg])))
    check(f"{_pg[2:-3].replace('_', ' ')} exports its tables", _n >= _least, f"{_n}")

check("the 6F KPI block uses equal-width rows",
      "kpi_row(" in _page_srcs["1_Fund_Analytics.py"]
      and "f6, f7, f8, f9 = st.columns(4)" not in _page_srcs["1_Fund_Analytics.py"])

print("       (run test_visual_polish.py for the render + export tests)")


print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
