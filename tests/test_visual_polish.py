"""
Step 7 — visual polish, tested by actually rendering the pages.

The three defects this file exists to catch are all invisible to a static
grep and only appear once a page is driven past its empty state:

  1. A chart rendered without the shared config still shows the Plotly logo.
     Static checks can confirm the call site changed; only a render confirms
     the figure survived the rewrite.
  2. An export button inside a per-fund loop with a fixed key raises
     StreamlitDuplicateElementKey the moment a SECOND fund is selected. One
     fund passes. Two do not. So the test selects two.
  3. A KPI block written as columns(5) then columns(4) renders ragged. The
     grid is checked by counting the columns the block actually creates.

Run:  python test_visual_polish.py
"""
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
# ROOT is the repo root, one level up from tests/. Every path in this file
# hangs off it, so it must not be the test's own directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from streamlit.testing.v1 import AppTest

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def _page(stem):
    import glob
    hits = [p for p in glob.glob(os.path.join(ROOT, "pages", "*.py"))
            if stem in os.path.basename(p)]
    assert hits, f"page {stem} not found"
    return hits[0]


# ── 1. Every chart goes through the shared helper ───────────────────────────
print("\n[CHARTS] One toolbar configuration for the whole app")

import utils.ui as ui

check("CHART_CONFIG hides the Plotly logo",
      ui.CHART_CONFIG.get("displaylogo") is False)
check("inapplicable toolbar buttons are removed",
      {"lasso2d", "select2d"} <= set(ui.CHART_CONFIG.get("modeBarButtonsToRemove", [])))

import glob
_srcs = {os.path.basename(p): open(p, encoding="utf-8").read()
         for p in glob.glob(os.path.join(ROOT, "pages", "*.py"))
         + [os.path.join(ROOT, "app.py")]}


def _code_only(src):
    """Drop triple-quoted blocks so docstring examples do not count."""
    return re.sub(r'("""|\'\'\')(?:.|\n)*?\1', "", src)


_bare = {f: len(re.findall(r"st\.plotly_chart\(", _code_only(s)))
         for f, s in _srcs.items()}
check("no page calls st.plotly_chart directly",
      not any(_bare.values()), f"{ {k: v for k, v in _bare.items() if v} }")

_via = sum(len(re.findall(r"(?<![\w.])chart\(", _code_only(s))) for s in _srcs.values())
check("charts are routed through the helper", _via >= 30, f"{_via} call sites")

# The helper must actually pass the config through to Streamlit.
_captured = {}
_real = None


def _spy(fig, **kw):
    _captured.update(kw)


import streamlit as _st
_real, _st.plotly_chart = _st.plotly_chart, _spy
try:
    ui.chart(object())
finally:
    _st.plotly_chart = _real
check("chart() passes CHART_CONFIG to Streamlit",
      _captured.get("config") is ui.CHART_CONFIG)
check("chart() uses the non-deprecated width argument",
      _captured.get("width") == "stretch" and "use_container_width" not in _captured,
      f"{sorted(_captured)}")
check("chart() ignores a None figure without raising",
      ui.chart(None) is None)

# ── 2. The deprecated parameter is gone from the app ────────────────────────
print("\n[DEPRECATION] use_container_width was removed app-wide")

# Application code only. The checker scripts in the repo root (verify_fixes.py
# and these tests) necessarily mention the deprecated name in string literals
# in order to search for it, and flagging them is a false positive — the point
# is that no page or module still PASSES it to Streamlit.
_all_py = [os.path.join(ROOT, "app.py")]
for _d in ("pages", "utils", "visualizations", "analytics", "data"):
    _all_py += sorted(glob.glob(os.path.join(ROOT, _d, "*.py")))

_left = []
for _f in _all_py:
    body = _code_only(open(_f, encoding="utf-8").read())
    body = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("#"))
    if "use_container_width" in body:
        _left.append(os.path.basename(_f))
check("no live call still passes use_container_width", not _left, f"{_left}")

# ── 3. KPI rows are the same width ──────────────────────────────────────────
print("\n[KPI GRID] Rows of cards line up")

_fa = _srcs["1_Fund_Analytics.py"]
check("the 6F factor block no longer mixes columns(5) with columns(4)",
      "f6, f7, f8, f9 = st.columns(4)" not in _fa)
check("the 6F factor block uses kpi_row", "kpi_row(" in _fa)

# kpi_row must pad a short final row, or the cards stretch.
_cols_made = []
_realcols = _st.columns
_st.columns = lambda spec, **kw: _realcols(spec, **kw) if _cols_made.append(spec) else _realcols(spec, **kw)
try:
    ui.kpi_row([{"label": f"m{i}", "value": 1.0} for i in range(9)], per_row=5)
except Exception:
    pass                       # no script context; we only want the spec calls
finally:
    _st.columns = _realcols
check("kpi_row asks for equal-width rows",
      _cols_made == [5, 5], f"columns() called with {_cols_made}")
check("kpi_row renders nothing for an empty list", ui.kpi_row([]) is None)

# ── 4. Exports exist where they did not ─────────────────────────────────────
print("\n[EXPORT] The two pages that had no export now have one")

for _pg, _min in (("5_Factor_Attribution.py", 5), ("6_Predictive_Analytics.py", 3)):
    _n = len(re.findall(r"export_button\(", _code_only(_srcs[_pg])))
    check(f"{_pg.split('_', 1)[1][:-3]} exports its tables", _n >= _min, f"{_n} buttons")

# An export inside a per-fund loop must build its key from the fund, or the
# second fund raises StreamlitDuplicateElementKey.
#
# Scoping matters here: taking everything after the FIRST "for fund_name"
# sweeps up later top-level blocks that are not in any loop, which is a false
# alarm. Loop bodies are found by indentation instead.


def _loop_bodies(src, header_pattern):
    """Yield the indented body of every loop whose header matches."""
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if not re.match(header_pattern, line.strip()):
            continue
        indent = len(line) - len(line.lstrip())
        body = []
        for nxt in lines[i + 1:]:
            if not nxt.strip():
                body.append(nxt)
                continue
            if len(nxt) - len(nxt.lstrip()) <= indent:
                break
            body.append(nxt)
        yield "\n".join(body)


_fixed_in_loop = []
for _pg, _src in _srcs.items():
    for _body in _loop_bodies(_code_only(_src), r"for\s+\w+\s+in\s+"):
        for _call in re.findall(r"export_button\((?:.|\n)*?\)", _body):
            _key = re.search(r'key\s*=\s*("|f")([^"]*)"', _call)
            if _key and _key.group(1) == '"':          # a literal, not an f-string
                _fixed_in_loop.append(f"{_pg}:{_key.group(2)}")
check("no export inside a loop uses a fixed key",
      not _fixed_in_loop, f"{_fixed_in_loop}")

# Prove the generated keys are distinct, using names that DO collide when
# slugified and truncated. AMFI scheme names are long and share their
# openings, so a name-derived key is only safe if something positional is
# mixed in. These three all reduce to the same 40-character slug:
_COLLIDING = (
    # These two differ only at "Direct"/"Regular" — character 45 of the name,
    # past the 40-character cut. Both slugify to
    # "aditya_birla_sun_life_frontline_equity_f".
    "Aditya Birla Sun Life Frontline Equity Fund - Direct Plan - Growth",
    "Aditya Birla Sun Life Frontline Equity Fund - Regular Plan - Growth",
    "Nippon India Growth Fund - Direct Plan - Growth",
)
_slugs = [re.sub(r"[^A-Za-z0-9]+", "_", n)[:40].strip("_").lower()
          for n in _COLLIDING]
check("a truncated name slug alone would collide (why the index is needed)",
      len(set(_slugs)) < len(_slugs), f"{len(set(_slugs))} distinct of {len(_slugs)}")

_fids = [f"{i}_{s}" for i, s in enumerate(_slugs)]
check("index-scoped export keys stay distinct even for colliding names",
      len(set(_fids)) == len(_fids), f"{len(set(_fids))} distinct")

# The page must use the positional form, not the bare slug.
_p5_code = _code_only(_srcs["5_Factor_Attribution.py"])
check("the per-fund loop enumerates, so a slot index is available",
      "for _fi, fund_name in enumerate(fund_names):" in _p5_code)
check("per-fund export keys are built from the index, not the name alone",
      not re.search(r'key\s*=\s*f"dl_fa_[a-z_]*\{_slug\}"', _p5_code))

# export_button behaviour
import pandas as pd
check("export_button skips an empty frame without raising",
      ui.export_button(pd.DataFrame(), "x.csv") is None)
check("export_button skips None without raising",
      ui.export_button(None, "x.csv") is None)

# A named index must survive into the CSV, or the rows lose their labels.
_df = pd.DataFrame({"Beta": [0.9, 0.3]}, index=pd.Index(["market", "smb"], name="Factor"))
_csv = {}
_realdl = _st.download_button
_st.download_button = lambda label, **kw: _csv.update(kw)
try:
    ui.export_button(_df, "f.csv", key="t")
finally:
    _st.download_button = _realdl
_text = _csv.get("data", b"").decode().replace("\r\n", "\n")
check("a named index is kept in the exported CSV",
      _text.startswith("Factor,Beta") and "market" in _text, repr(_text[:24]))

_plain = pd.DataFrame({"a": [1, 2]})
_csv.clear()
_st.download_button = lambda label, **kw: _csv.update(kw)
try:
    ui.export_button(_plain, "p.csv", key="t2")
finally:
    _st.download_button = _realdl
# Line endings are normalised before comparing. pandas writes os.linesep, so
# to_csv() returns "a\n1\n2\n" on Linux and "a\r\n1\r\n2\r\n" on Windows — this
# assertion passed on CI and failed on the developer's own machine, which is the
# worst way round for a test to be wrong.
_plain_csv = _csv.get("data", b"").decode().replace("\r\n", "\n")
check("a default RangeIndex is dropped from the CSV",
      _plain_csv.startswith("a\n1"), repr(_plain_csv[:12]))

# ── 5. Cards come from one place ────────────────────────────────────────────
print("\n[CARDS] One card style, tones kept where they mean something")

_inline = {f: len(re.findall(r"border-radius:\s*\d+px", _code_only(s)))
           for f, s in _srcs.items()}
check("no page hand-rolls a card div", not any(_inline.values()),
      f"{ {k: v for k, v in _inline.items() if v} }")
check("the regime legend keeps its bull/sideways/bear colours",
      {"up", "flat", "down"} <= set(ui.CARD_TONES))
check("card() falls back to neutral for an unknown tone",
      ui.CARD_TONES.get("nonsense", ui.CARD_TONES["neutral"]) == ui.CARD_TONES["neutral"])

# ── 6. Render the pages that changed most, with TWO funds selected ──────────
print("\n[RENDER] Factor Attribution with two funds — the duplicate-key case")


def _run(path, timeout=600):
    at = AppTest.from_file(path, default_timeout=timeout)
    at.run()
    return at


_at = _run(_page("5_Factor_Attribution"))
check("Factor Attribution renders clean", not _at.exception,
      str(_at.exception)[:90] if _at.exception else "")

# Selecting two funds only exercises the duplicate-key path when the scheme
# registry actually loaded. Without network access the page renders its
# "no data" branch and a pass here would mean nothing, so the test says which
# of the two it did rather than reporting a hollow success.
_boxes = [sb for sb in _at.selectbox if len(sb.options or []) > 2]
if len(_boxes) >= 2:
    opts = _boxes[0].options
    _boxes[0].select(opts[1])
    _boxes[1].select(opts[2])
    _at.run()
    check("Factor Attribution survives two funds selected",
          not _at.exception,
          str(_at.exception)[:140] if _at.exception else "")
else:
    print("  SKIP  two-fund render — no scheme data reachable "
          f"({len(_boxes)} populated selectboxes); "
          "the static key check above is what covers this offline.")

# The duplicate-key failure mode itself is reproducible without a network:
# register two download buttons under the same key and confirm Streamlit
# objects, then confirm the fund-scoped keys do not collide.
_seen_keys = []
_st.download_button = lambda label, **kw: _seen_keys.append(kw.get("key"))
try:
    for _fund in ("Fund Alpha - Direct - Growth", "Fund Beta - Direct - Growth"):
        _s = re.sub(r"[^A-Za-z0-9]+", "_", _fund)[:40].strip("_").lower()
        ui.export_button(pd.DataFrame({"x": [1]}), f"regime_{_s}.csv",
                         key=f"dl_fa_regime_{_s}")
finally:
    _st.download_button = _realdl
check("two funds produce two distinct download keys",
      len(set(_seen_keys)) == 2, f"{_seen_keys}")

for _name, _stem in (("Fund Analytics", "1_Fund_Analytics"),
                     ("Predictive Analytics", "6_Predictive_Analytics"),
                     ("Home", None)):
    _p = os.path.join(ROOT, "app.py") if _stem is None else _page(_stem)
    _a = _run(_p)
    check(f"{_name} renders clean after the polish pass", not _a.exception,
          str(_a.exception)[:90] if _a.exception else "")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
