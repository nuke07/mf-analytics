"""
Correctness and speed test for concurrent NAV loading.

Two things must hold, and only one of them is about speed:

  1. The parallel loader must return EXACTLY what a sequential loop would —
     same codes, same frames, same handling of failures. A faster loader that
     quietly drops a fund is worse than a slow one.
  2. It must actually overlap the I/O. A ThreadPoolExecutor around a call
     that holds a lock, or that Streamlit serialises, would look correct and
     buy nothing.

The real API is stubbed with a fixed artificial latency so the timing is
deterministic and does not depend on the network.

Run:  python test_parallel_load.py
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
# ROOT is the repo root, one level up from tests/. Every path in this file
# hangs off it, so it must not be the test's own directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd

import data.fund_loader as fl

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        fails.append(name)


LATENCY = 0.25            # seconds per simulated fetch
N_FUNDS = 12
CALLS = []


def _fake_nav(code):
    """Stand-in for the real HTTP fetch: sleeps, then returns a tiny frame."""
    CALLS.append(code)
    time.sleep(LATENCY)
    if code == "FAIL":
        raise RuntimeError("simulated upstream error")
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    return pd.DataFrame({"nav": range(50)}, index=idx)


fl.get_nav_history = _fake_nav
codes = [f"C{i}" for i in range(N_FUNDS)]

print(f"\n[PARALLEL] {N_FUNDS} fetches at {LATENCY}s each")

# ── Sequential baseline ─────────────────────────────────────────────────────
CALLS.clear()
t0 = time.perf_counter()
sequential = {c: _fake_nav(c) for c in codes}
seq_time = time.perf_counter() - t0
print(f"     sequential: {seq_time:.2f}s")

# ── Parallel ────────────────────────────────────────────────────────────────
CALLS.clear()
t0 = time.perf_counter()
parallel = fl.load_navs_parallel(codes, max_workers=6)
par_time = time.perf_counter() - t0
print(f"     parallel:   {par_time:.2f}s   ({seq_time / par_time:.1f}x faster)")

check("every code is returned", set(parallel) == set(codes),
      f"{len(parallel)} of {len(codes)}")
check("frames match the sequential result",
      all(parallel[c].equals(sequential[c]) for c in codes))
check("each code is fetched exactly once", len(CALLS) == N_FUNDS, f"{len(CALLS)} calls")
check("parallel is meaningfully faster", par_time < seq_time / 2,
      f"{seq_time:.2f}s -> {par_time:.2f}s")
check("wall time is near the theoretical floor",
      par_time < (N_FUNDS / 6) * LATENCY + 0.4,
      f"{par_time:.2f}s vs floor {(N_FUNDS/6)*LATENCY:.2f}s")

# ── Failure handling must match get_nav_history's contract ─────────────────
CALLS.clear()
mixed = fl.load_navs_parallel(["C0", "FAIL", "C1"])
check("a failed fetch yields None, not an exception",
      mixed.get("FAIL") is None and mixed.get("C0") is not None,
      f"{ {k: (v is None) for k, v in mixed.items()} }")
check("one bad fund does not lose the others", len(mixed) == 3)

# ── Duplicates and empties ──────────────────────────────────────────────────
CALLS.clear()
dup = fl.load_navs_parallel(["C0", "C0", "C1", "C0"])
check("duplicate codes are fetched once", len(CALLS) == 2, f"{len(CALLS)} calls")
check("duplicates collapse in the result", set(dup) == {"C0", "C1"})
check("empty input returns an empty dict", fl.load_navs_parallel([]) == {})

# ── Progress callback ───────────────────────────────────────────────────────
seen = []
fl.load_navs_parallel(codes[:4], progress_cb=lambda d, t, c: seen.append((d, t)))
check("progress callback fires once per fund, counting up",
      [d for d, _ in seen] == [1, 2, 3, 4] and all(t == 4 for _, t in seen),
      f"{seen}")

# A callback that raises must not take the whole load down with it.
def _bad_cb(d, t, c):
    raise ValueError("callback blew up")

try:
    ok = fl.load_navs_parallel(codes[:3], progress_cb=_bad_cb)
    check("a failing progress callback does not break the load", len(ok) == 3)
except Exception as e:
    check("a failing progress callback does not break the load", False, str(e))

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
