"""
Headless render test for the Streamlit pages.

Static checks cannot catch a Streamlit runtime exception — the RF-button
regression proved that. This renders each page with the network-dependent
loaders stubbed, so import errors, widget-key collisions and layout mistakes
surface here rather than in the browser.

Run:  python test_pages_render.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
# ROOT is the repo root, one level up from tests/. Every path in this file
# hangs off it, so it must not be the test's own directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from streamlit.testing.v1 import AppTest

from utils.ui import RF_KEY

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        fails.append(name)


# Fund data needs AMFI; stub the registry so the UI paths run offline.
STUB = """
import sys
sys.path.insert(0, {root!r})
import pandas as pd
import data.fund_loader as fl

_CATS = ["Large Cap", "Mid Cap", "Small Cap", "Flexi Cap"]
_FUNDS = {{
    c: [{{"name": c + " Fund " + str(i), "code": str(100000 + i)}} for i in range(4)]
    for c in _CATS
}}
fl.get_all_categorized_schemes = lambda plan_type="Direct": _FUNDS
fl.get_all_schemes = lambda: {{str(100000 + i): "Fund " + str(i) +
                              " - Direct Plan - Growth" for i in range(20)}}
fl.get_nav_history = lambda code: None
""".format(root=ROOT)


def render(page):
    src = open(os.path.join(ROOT, page), encoding="utf-8").read()
    app = AppTest.from_string(STUB + "\n" + src, default_timeout=120)
    app.run()
    return app


PAGES = [
    ("pages/1_Fund_Analytics.py",      "Fund Analytics"),
    ("pages/2_Fund_Comparison.py",     "Fund Comparison"),
    ("pages/3_Rankings.py",            "Rankings"),
    ("pages/4_Portfolio_Analytics.py", "Portfolio Analytics"),
    ("pages/5_Factor_Attribution.py",  "Factor Attribution"),
    ("pages/6_Predictive_Analytics.py", "Predictive Analytics"),
    ("pages/7_Data_Quality.py",        "Data Quality"),
]

print("\n[PAGES] Every page renders without a Streamlit exception")
apps = {}
for page, label in PAGES:
    app = render(page)
    apps[label] = app
    exc = str(app.exception[0].value) if app.exception else ""
    check(f"{label} renders", not app.exception, exc[:100])

print("\n[EMPTY STATES] The two pages that used to blank out silently")
for label in ["Portfolio Analytics", "Factor Attribution"]:
    app = apps[label]
    if app.exception:
        check(f"{label} first-visit guidance", False, "page raised")
        continue
    text = " ".join(i.value for i in app.info).lower()
    check(f"{label} explains what to do on first visit",
          "click" in text and "run" in text,
          (text[:64] + "…") if text else "no st.info rendered")

print("\n[SLOT PICKER] Both pages use the shared row renderer")
for label, want in [("Portfolio Analytics", 8), ("Factor Attribution", 3)]:
    app = apps[label]
    if app.exception:
        check(f"{label} slot rows", False, "page raised")
        continue
    # Each empty slot renders a category + a (disabled) fund selectbox.
    n = len(app.selectbox)
    check(f"{label} renders {want} slot rows", n >= want * 2,
          f"{n} selectboxes")

print("\n[SIDEBAR] The shared controls appear where they belong")
# Data Quality reports NAV coverage, not risk-adjusted metrics, so it has no
# risk-free rate — correctly, and deliberately.
NEEDS_RF = {label for _, label in PAGES} - {"Data Quality"}
for _, label in PAGES:
    app = apps[label]
    if app.exception:
        continue
    # Matched against the label ui.rf_control() actually renders, imported
    # rather than duplicated here — the previous hardcoded "Risk-Free Rate (%)"
    # broke the moment the design pass shortened the label, reporting the
    # control as missing when it was present and working.
    # RF_KEY is now a prefix — each page owns rf_rate_<page id>.
    has_rf = any((s.key or "").startswith(RF_KEY) for s in app.sidebar.slider)
    if label in NEEDS_RF:
        check(f"{label} has the shared RF control", has_rf)
    else:
        check(f"{label} correctly has no RF control", not has_rf)

print("\n[ACCESSIBILITY] No widget is left with an empty label")
for _, label in PAGES:
    src = open(os.path.join(ROOT, dict((l, p) for p, l in PAGES)[label]),
               encoding="utf-8").read()
    check(f"{label} has no empty widget labels",
          'st.radio("",' not in src and 'st.selectbox("",' not in src
          and 'st.slider("",' not in src)

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
