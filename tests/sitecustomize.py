"""
Test-only loader stubs, activated by MF_STUB_DATA=1.

Python imports any importable module named `sitecustomize` at interpreter
startup. That is exactly what makes this work: a Streamlit subprocess we do not
control cannot be told to import a stub module, but it will pick this one up on
its own — which lets the browser tests run the real app against synthetic funds
built from the TRI CSVs in data/tri/, with no network and no AMFI.

That same mechanism is why this file lives in tests/ and NOT at the repo root.
At the root it would be auto-imported by every `python` and `streamlit` run in
this directory, and one stray MF_STUB_DATA=1 in someone's environment would
have the real app quietly computing returns on synthetic funds. Here it is
unreachable unless a test deliberately puts tests/ on PYTHONPATH — see
test_rf_navigation.py, currently the only caller.

Without the environment variable this file does nothing at all.
"""
import os

if os.environ.get("MF_STUB_DATA") == "1":
    def _install():
        import pandas as pd
        from data.tri_loader import get_tri_nav
        from utils.constants import CATEGORIES
        import data.fund_loader as fl
        import data.category_mapper as cm

        base = ["NIFTY 100", "NIFTY MIDCAP 150", "NIFTY SMALLCAP 250",
                "NIFTY 500", "NIFTY 50", "NIFTY200 MOMENTUM 30"]
        series = {n: s for n in base
                  if (s := get_tri_nav(n)) is not None and not s.empty}
        names = list(series)

        fixture, navs, code = {}, {}, 100000
        for cat in CATEGORIES:
            rows = []
            for i in range(6):
                code += 1
                nav = series[names[i % len(names)]].copy()
                nav["nav"] = nav["nav"] * (1.0 + 0.03 * i)
                navs[str(code)] = nav
                rows.append({"code": str(code),
                             "name": f"Test {cat} Fund {i+1} - Direct Plan - Growth"})
            fixture[cat] = rows

        fl.get_all_categorized_schemes = lambda plan_type="Direct": fixture
        fl.get_nav_history = lambda c: navs.get(str(c))
        fl.get_all_schemes = lambda: {r["code"]: r["name"]
                                      for rs in fixture.values() for r in rs}

        def _par(codes, max_workers=6, progress_cb=None):
            uniq = list(dict.fromkeys(str(c) for c in codes))
            out = {}
            for i, c in enumerate(uniq, 1):
                out[c] = navs.get(c)
                if progress_cb:
                    try: progress_cb(i, len(uniq), c)
                    except Exception: pass
            return out
        fl.load_navs_parallel = _par
        cm.get_category_fund_counts = lambda s: {c: len(fixture.get(c, []))
                                                 for c in CATEGORIES}

    try:
        _install()
    except Exception as exc:            # never break a real run
        print(f"[sitecustomize] stub install skipped: {exc}")
