"""
Headless UI test for the risk-free-rate control.

This exists because the RF control has now broken in two different ways that
no import-time or lint check could catch:

  1. A keyless slider ignored the −/+ buttons entirely: st.slider only honours
     `value=` on first render, so once the user dragged it, every button
     increment was silently discarded.
  2. Fixing that by binding the slider to a session-state key and assigning to
     that key inline raised StreamlitAPIException — Streamlit forbids writing
     to a widget's key after the widget has been instantiated in that run.

Both bugs only appear when a real user clicks the button. streamlit.testing
runs the script headlessly and simulates that click, so the behaviour is
actually exercised rather than reasoned about.

Run:  python test_rf_control.py
"""
import os
import sys

# ROOT is the repo root, one level up from tests/. Every path in this file
# hangs off it, so it must not be the test's own directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from streamlit.testing.v1 import AppTest

from utils.constants import DEFAULT_RISK_FREE_RATE

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        fails.append(name)


# A minimal page that uses nothing but the shared control, so a failure here
# points at the control and not at page-specific data loading.
HARNESS = """
import sys
sys.path.insert(0, %r)
import streamlit as st
from utils.ui import rf_control

with st.sidebar:
    rf_pct, rf_rate = rf_control()

st.write(f"RF_PCT={rf_pct}")
st.write(f"RF_RATE={rf_rate}")
""" % ROOT


def rf_value(app):
    for el in app.markdown:
        if el.value.startswith("RF_PCT="):
            return float(el.value.split("=")[1])
    raise AssertionError("RF_PCT not rendered")


print("\n[RF] Risk-free-rate control — headless interaction test")

app = AppTest.from_string(HARNESS, default_timeout=30)
app.run()

check("page renders without an exception",
      not app.exception, str(app.exception[0].value) if app.exception else "")

default_pct = round(DEFAULT_RISK_FREE_RATE * 100, 1)
check("slider starts at DEFAULT_RISK_FREE_RATE",
      rf_value(app) == default_pct, f"{rf_value(app)} vs {default_pct}")

# ── The regression that just broke in production ───────────────────────────
before = rf_value(app)
app.button[1].click().run()          # the "+" button
check("clicking + does not raise StreamlitAPIException",
      not app.exception, str(app.exception[0].value) if app.exception else "")
check("clicking + actually increases the rate",
      rf_value(app) == round(before + 0.1, 1), f"{before} -> {rf_value(app)}")

app.button[0].click().run()          # the "−" button
check("clicking − does not raise",
      not app.exception, str(app.exception[0].value) if app.exception else "")
check("clicking − returns to the starting value",
      rf_value(app) == before, f"back to {rf_value(app)}")

# ── The original bug: increments must survive a manual slider drag ─────────
app.sidebar.slider[0].set_value(8.0).run()
check("dragging the slider sets the value", rf_value(app) == 8.0, f"{rf_value(app)}")

app.button[1].click().run()
check("+ still works AFTER the slider has been dragged (the original bug)",
      rf_value(app) == 8.1, f"expected 8.1, got {rf_value(app)}")

# ── Clamping at both ends ──────────────────────────────────────────────────
app.sidebar.slider[0].set_value(9.0).run()
app.button[1].click().run()
check("+ clamps at the maximum", rf_value(app) == 9.0, f"{rf_value(app)}")

app.sidebar.slider[0].set_value(4.0).run()
app.button[0].click().run()
check("− clamps at the minimum", rf_value(app) == 4.0, f"{rf_value(app)}")

# ── The decimal returned to callers must match the percent shown ───────────
app.sidebar.slider[0].set_value(7.5).run()
rate = next(float(el.value.split("=")[1]) for el in app.markdown
            if el.value.startswith("RF_RATE="))
check("returned decimal matches the displayed percent",
      abs(rate - 0.075) < 1e-9, f"{rate}")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
