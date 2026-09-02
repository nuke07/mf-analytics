"""
The Sharpe confidence interval, tested against things with known answers.

Four classes of regression this guards against, three of which were live bugs
caught while writing the module:

  1. The standard error must match Lo (2002) eq. 9 and must not depend on
     whether the series is sampled daily or monthly.
  2. The serial-correlation factor must return ~1.0 on IID data and rise
     monotonically with AR(1) phi. The first implementation used Lo's eq. 19
     literally, which weights each sample autocorrelation by (q-k) — roughly
     240 for daily data — and returned 1.006, 1.077 and 1.403 on three draws of
     the SAME IID process. Bartlett weights fixed it.
  3. A zero-variance series must return None. Testing `sd == 0.0` let a constant
     series through as ~1e-19 and produced a Sharpe ratio of 1.06e+17.
  4. The interval must be computed on REAL observations. process_nav pads NAV
     onto a calendar grid, so ~32% of the returns reaching this module are
     exact zeros from closed markets; counting them narrows the interval by
     about 21% for no reason.

Run:  python tests/test_uncertainty.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
# ROOT is the repo root, one level up from tests/.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from analytics.risk_adjusted import calc_sharpe
from analytics.uncertainty import (
    autocorrelation_inflation,
    difference_is_significant,
    effective_observations,
    indistinguishable_bands,
    sharpe_interval,
    sharpe_standard_error,
)

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def ar1(phi, n=2520, seed=0, mu=0.0004, sd=0.01):
    r = np.random.default_rng(seed)
    e = r.normal(0, sd, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return pd.Series(x + mu)


# ── 1. The closed form ──────────────────────────────────────────────────────
print("\n[SE] Lo (2002) eq. 9")

# SE = sqrt(ppy) * sqrt((1 + (S/sqrt(ppy))^2 / 2) / n), computed independently.
for yrs in (1, 3, 5, 10):
    n = 252 * yrs
    s_ann = 1.0
    want = np.sqrt(252) * np.sqrt((1 + (s_ann / np.sqrt(252)) ** 2 / 2) / n)
    got = sharpe_standard_error(s_ann, n)
    check(f"{yrs}y daily SE matches the closed form",
          abs(got - want) < 1e-12, f"{got:.4f}")

se3 = sharpe_standard_error(1.0, 756)
check("a 3-year Sharpe of 1.00 has a 95% interval that includes zero",
      1.0 - 1.96 * se3 < 0, f"low end {1.0 - 1.96 * se3:+.3f}")

# The estimator is defined on the sampling frequency, so the same span of
# calendar time must give the same answer whether sampled daily or monthly.
for yrs in (3, 5, 10):
    d = sharpe_standard_error(1.0, 252 * yrs, 252)
    m = sharpe_standard_error(1.0, 12 * yrs, 12)
    check(f"{yrs}y: daily and monthly SE agree", abs(d - m) < 0.02,
          f"daily {d:.4f} vs monthly {m:.4f}")

check("SE falls with the square root of n",
      abs(sharpe_standard_error(1.0, 1000) / sharpe_standard_error(1.0, 4000) - 2.0) < 0.01)
check("SE rises with the Sharpe ratio",
      sharpe_standard_error(3.0, 1000) > sharpe_standard_error(0.5, 1000))
check("SE is None for a degenerate sample", sharpe_standard_error(1.0, 1) is None)

# ── 2. Serial correlation ───────────────────────────────────────────────────
print("\n[ACF] The factor must be ~1.0 on IID data and monotone in phi")

rng = np.random.default_rng(11)
iid_runs = [autocorrelation_inflation(pd.Series(rng.normal(0.0004, 0.01, 2520)))
            for _ in range(10)]
check("IID returns give a factor near 1.0",
      abs(np.mean(iid_runs) - 1.0) < 0.08,
      f"mean {np.mean(iid_runs):.3f}")
check("the IID estimate is not wildly noisy (the eq.19 failure)",
      np.std(iid_runs) < 0.12, f"sd {np.std(iid_runs):.3f}")
check("no IID draw lands absurdly high",
      max(iid_runs) < 1.30, f"max {max(iid_runs):.3f}")

means = []
for phi in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40):
    means.append(np.mean([autocorrelation_inflation(ar1(phi, seed=s))
                          for s in range(8)]))
check("the factor increases monotonically with AR(1) phi",
      all(a < b for a, b in zip(means, means[1:])),
      " ".join(f"{m:.2f}" for m in means))
# Theory: VIF -> (1+phi)/(1-phi). Bartlett truncation biases it slightly low.
for phi, got in zip((0.10, 0.20, 0.30), means[2:5]):
    want = (1 + phi) / (1 - phi)
    check(f"phi={phi} lands near the theoretical {want:.2f}",
          abs(got - want) < 0.25, f"{got:.3f}")

# ── 3. Guards ───────────────────────────────────────────────────────────────
print("\n[GUARDS] Degenerate input returns None, never a number")

for label, series in (
    ("a constant series", pd.Series([0.001] * 500)),
    ("all zeros", pd.Series([0.0] * 500)),
    ("float-noise-only variance",
     pd.Series(np.full(500, 0.001) + np.random.default_rng(0).normal(0, 1e-15, 500))),
):
    r = sharpe_interval(series)
    check(f"{label} yields no Sharpe at all", r["sharpe"] is None, f"{r['sharpe']}")

check("None input is handled", sharpe_interval(None)["sharpe"] is None)
check("an empty series is handled", sharpe_interval(pd.Series(dtype=float))["sharpe"] is None)

short = sharpe_interval(pd.Series(np.random.default_rng(0).normal(0.0004, 0.01, 100)))
check("under a year of data reports the estimate but no interval",
      short["sharpe"] is not None and short["sharpe_se"] is None)

nan_mix = pd.Series([0.01, np.nan, np.inf, -0.02] * 200)
check("NaN and inf are dropped rather than propagated",
      sharpe_interval(nan_mix)["sharpe"] is not None)

# ── 4. Effective observations ───────────────────────────────────────────────
print("\n[N] Padded calendar days are not observations")

padded = pd.Series([0.01, 0.0, 0.0, -0.005, 0.007, 0.0, 0.0] * 400)   # 2 of 7 real
check("exact zeros are excluded from the count",
      effective_observations(padded) == 400 * 3, f"{effective_observations(padded)}")
flat = pd.Series([0.0] * 900 + [0.01] * 100)
check("a genuinely flat series falls back to full length rather than ~0",
      effective_observations(flat) == 1000, f"{effective_observations(flat)}")

# ── 5. Agreement with the shipped Sharpe, on real data ──────────────────────
print("\n[REAL] The interval wraps the same number calc_sharpe reports")

TRI = os.path.join(ROOT, "data", "tri")
prev_inflation = None
for name in ("NIFTY_100", "NIFTY_MIDCAP_150", "NIFTY_SMALLCAP_250"):
    path = os.path.join(TRI, f"{name}_TRI.csv")
    if not os.path.isfile(path):
        print(f"  SKIP  {name} not present")
        continue
    d = pd.read_csv(path, parse_dates=["Date"])
    r = d.set_index("Date")["TotalReturnsIndex"].sort_index().pct_change().dropna()

    a = calc_sharpe(r, rf_rate=0.07)
    b = sharpe_interval(r, rf_rate=0.07)
    check(f"{name}: point estimate is identical to calc_sharpe",
          abs(a - b["sharpe"]) < 1e-12, f"{a:.6f} vs {b['sharpe']:.6f}")
    check(f"{name}: the interval brackets the estimate",
          b["sharpe_ci_low"] < b["sharpe"] < b["sharpe_ci_high"],
          f"[{b['sharpe_ci_low']:.3f}, {b['sharpe_ci_high']:.3f}]")
    check(f"{name}: n matches the published row count, not a padded one",
          abs(b["sharpe_n_obs"] - len(r)) <= 2,
          f"{b['sharpe_n_obs']} vs {len(r)}")

# Liquidity ordering: the less liquid the segment, the more stale pricing, the
# more serial correlation. If this ever inverts, the estimator has drifted.
infl = {}
for name in ("NIFTY_100", "NIFTY_MIDCAP_150", "NIFTY_SMALLCAP_250"):
    path = os.path.join(TRI, f"{name}_TRI.csv")
    if os.path.isfile(path):
        d = pd.read_csv(path, parse_dates=["Date"])
        r = d.set_index("Date")["TotalReturnsIndex"].sort_index().pct_change().dropna()
        infl[name] = sharpe_interval(r, rf_rate=0.07)["sharpe_acf_inflation"]
if len(infl) == 3:
    check("serial correlation rises as the cap segment gets less liquid",
          infl["NIFTY_100"] < infl["NIFTY_MIDCAP_150"] < infl["NIFTY_SMALLCAP_250"],
          " < ".join(f"{k.split('_')[-1]} {v:.2f}" for k, v in infl.items()))

# ── 6. Banding ──────────────────────────────────────────────────────────────
print("\n[BANDS] Funds that cannot be told apart share a band")

items = [("A", 0.95, 0.30), ("B", 0.90, 0.30), ("C", 0.84, 0.30),
         ("D", 0.40, 0.30), ("E", 0.05, 0.30)]
bands = indistinguishable_bands(items)
check("the top of a tight cluster is one band", bands[0] == bands[1] == bands[2],
      f"{bands}")
check("a clearly worse fund falls to a lower band", bands[-1] > bands[0], f"{bands}")
check("bands never decrease down a sorted table",
      all(a <= b for a, b in zip([x for x in bands if x], [x for x in bands if x][1:])),
      f"{bands}")

tight = indistinguishable_bands(items, correlation=0.85)
check("using the real peer correlation separates more funds",
      max(tight) >= max(bands), f"indep {max(bands)} vs corr {max(tight)}")

check("missing estimates get no band",
      indistinguishable_bands([("A", 1.0, 0.2), ("B", None, None)])[1] is None)
check("an empty list is handled", indistinguishable_bands([]) == [])

check("identical funds are never called different",
      not difference_is_significant(1.0, 0.3, 1.0, 0.3))
check("a huge gap is called different",
      difference_is_significant(3.0, 0.2, 0.1, 0.2))
check("independence is the conservative assumption",
      not difference_is_significant(1.0, 0.3, 0.2, 0.3)
      and difference_is_significant(1.0, 0.3, 0.2, 0.3, correlation=0.9),
      "correlated peers are separable where independent ones are not")

# ── 7. The engine carries it through ────────────────────────────────────────
print("\n[ENGINE] compute_fund_metrics exposes the interval")

from analytics.engine import compute_fund_metrics, _ALL_METRIC_KEYS
from utils.constants import METRIC_HELP, METRIC_LABELS

NEW = ["sharpe_se", "sharpe_ci_low", "sharpe_ci_high",
       "sharpe_n_obs", "sharpe_acf_inflation"]
check("every new key is declared by the engine",
      all(k in _ALL_METRIC_KEYS for k in NEW))
check("every new key has a display label",
      all(k in METRIC_LABELS for k in NEW),
      f"{[k for k in NEW if k not in METRIC_LABELS]}")
check("every new key has help text",
      all(k in METRIC_HELP for k in NEW),
      f"{[k for k in NEW if k not in METRIC_HELP]}")

path = os.path.join(TRI, "NIFTY_MIDCAP_150_TRI.csv")
if os.path.isfile(path):
    d = pd.read_csv(path, parse_dates=["Date"])
    nav = d.set_index("Date")["TotalReturnsIndex"].sort_index().rename("nav").to_frame()
    m = compute_fund_metrics(nav, rf_rate=0.07, fund_name="Mid")
    check("the engine produces every new key", all(m.get(k) is not None for k in NEW),
          f"{ {k: m.get(k) for k in NEW if m.get(k) is None} }")
    check("the engine's interval brackets the engine's Sharpe",
          m["sharpe_ci_low"] < m["sharpe"] < m["sharpe_ci_high"],
          f"{m['sharpe']:.3f} in [{m['sharpe_ci_low']:.3f}, {m['sharpe_ci_high']:.3f}]")
    # Adding a diagnostic must never move the number being diagnosed.
    check("adding the interval did not change the Sharpe ratio",
          abs(m["sharpe"] - calc_sharpe(m["returns"], rf_rate=0.07)) < 1e-12)

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
