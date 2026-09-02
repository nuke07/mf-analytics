"""
Run every suite in tests/ and report one verdict.

    python tests/run_all.py            # everything
    python tests/run_all.py --fast     # skip the two browser suites

Each suite is a standalone script that exits non-zero on failure, so they run
as subprocesses rather than being imported — one suite blowing up cannot take
the runner down with it, and a hung browser test can be timed out.

Suite order is cheapest-first: the pure-analytics and static-source checks run
before anything that renders a page, so an obvious break fails in seconds
instead of after a browser launch.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (filename, needs_browser, timeout_seconds)
SUITES: list[tuple[str, bool, int]] = [
    ("test_analytics_regressions.py", False, 300),
    ("test_uncertainty.py",           False, 180),
    ("test_amfi_parse.py",            False, 120),
    ("test_design_system.py",         False, 180),
    ("test_visual_polish.py",         False, 180),
    ("test_parallel_load.py",         False, 180),
    ("test_rf_control.py",            False, 300),
    ("test_pages_render.py",          False, 600),
    ("test_pages_with_data.py",       False, 900),
    ("test_rf_navigation.py",         True,  600),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="skip suites that drive a real browser")
    args = ap.parse_args()

    results: list[tuple[str, str, float]] = []
    for name, needs_browser, timeout in SUITES:
        path = os.path.join(HERE, name)
        if not os.path.isfile(path):
            results.append((name, "MISSING", 0.0))
            continue
        if args.fast and needs_browser:
            results.append((name, "skipped", 0.0))
            continue

        print(f"\n{'─' * 72}\n▶ {name}\n{'─' * 72}", flush=True)
        t0 = time.time()
        try:
            proc = subprocess.run([sys.executable, path], cwd=ROOT,
                                  timeout=timeout)
            status = "PASS" if proc.returncode == 0 else f"FAIL ({proc.returncode})"
        except subprocess.TimeoutExpired:
            status = f"TIMEOUT ({timeout}s)"
        results.append((name, status, time.time() - t0))

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    for name, status, secs in results:
        print(f"  {name:<34} {status:<16} {secs:6.1f}s")

    bad = [n for n, s, _ in results if s not in ("PASS", "skipped")]
    print(f"\n  {len(results) - len(bad)}/{len(results)} ok")
    if bad:
        print("  failing: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
