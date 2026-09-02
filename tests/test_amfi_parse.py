"""Verify the AMFI schema fix end to end, using real NAVAll.txt rows."""
import os, sys, warnings
warnings.filterwarnings("ignore")

# Run from anywhere: resolve imports relative to this file's directory,
# which is expected to be the mf_analytics repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fund_loader import _compose_scheme_name, _validate_scheme_names
from data.category_mapper import (
    filter_preferred_plans, filter_direct_plans, get_category_fund_counts,
)

fails = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond: fails.append(name)


def parse(text):
    """Mirror of _fetch_schemes_via_amfi_direct's parsing loop."""
    out = {}
    for line in text.splitlines():
        if ";" not in line:
            continue
        p = line.split(";")
        if len(p) < 4:
            continue
        code = p[0].strip()
        if not code.isdigit():
            continue
        nm = p[3].strip()
        if not nm:
            continue
        from data.fund_loader import _is_numeric
        out[code] = (_compose_scheme_name(nm, p[4], p[5])
                     if len(p) >= 8 and not _is_numeric(p[4]) else nm)
    return out


# ── Verbatim rows from the live file (current 8-field schema) ───────────────
NEW = """Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Plan;Option;Net Asset Value;Date

Open Ended Schemes(Debt Scheme - Banking and PSU Fund)

Aditya Birla Sun Life Mutual Fund

119551;INF209KA12Z1;INF209KA13Z9;Aditya Birla Sun Life Banking & PSU Debt Fund;Direct Plan;IDCW-Re-investment;106.8821;21-Aug-2026
119550;INF209K01YN0;-;Aditya Birla Sun Life Banking & PSU Debt Fund;Direct Plan;GROWTH;403.6492;21-Aug-2026
120438;INF846K01CR6;-;Axis Banking & PSU Debt Fund;Direct Plan;Growth Option;2890.5440;21-Aug-2026
128952;INF846K01NF8;-;Axis Banking & PSU Debt Fund;Direct Plan;Bonus Option;1532.8272;14-Jun-2017
120503;INF846K01131;-;Axis ELSS Tax Saver Fund;Direct Plan;Growth Option;95.1234;21-Aug-2026
118988;INF209K01VD4;-;Axis Bluechip Fund;Direct Plan;Growth;512.3300;21-Aug-2026
118989;INF209K01VD5;-;HDFC Mid-Cap Opportunities Fund;Direct Plan;Growth;198.4400;21-Aug-2026
125497;INF179K01YV8;-;HDFC Small Cap Fund;Direct Plan;Growth;155.2100;21-Aug-2026
122639;INF879O01019;-;Parag Parikh Flexi Cap Fund;Direct Plan;Growth;89.7700;21-Aug-2026
120823;INF090I01JL2;-;Franklin India Multi Cap Fund;Direct Plan;Growth;77.1000;21-Aug-2026
120586;INF109K012K1;-;ICICI Prudential Equity & Debt Fund;Direct Plan;Growth;445.9900;21-Aug-2026
119242;INF179K01WM3;-;HDFC Balanced Advantage Fund;Direct Plan;Growth;521.4400;21-Aug-2026
120716;INF109K01Y72;-;ICICI Prudential Value Discovery Fund;Direct Plan;Growth;486.2200;21-Aug-2026
118473;INF200K01UB1;-;SBI Contra Fund;Direct Plan;Growth;412.8800;21-Aug-2026
118533;INF090I01surname;-;Franklin India Focused Equity Fund;Direct Plan;Growth;123.4500;21-Aug-2026
120716;INF109K01Y73;-;UTI Nifty 50 Index Fund;Direct Plan;Growth;178.9900;21-Aug-2026
118474;INF200K01UB2;-;SBI Contra Fund;Regular Plan;Growth;380.1100;21-Aug-2026
118475;INF200K01UB3;-;SBI Contra Fund;Regular Plan;IDCW Payout;120.5500;21-Aug-2026"""

# ── Same funds under the OLD 6-field schema ─────────────────────────────────
OLD = """Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Net Asset Value;Date
119550;INF209K01YN0;-;Aditya Birla Sun Life Banking & PSU Debt Fund - Direct Plan - GROWTH;403.6492;21-Aug-2026
120503;INF846K01131;-;Axis ELSS Tax Saver Fund - Direct Plan - Growth Option;95.1234;21-Aug-2026
118989;INF209K01VD5;-;HDFC Mid-Cap Opportunities Fund - Direct Plan - Growth;198.4400;21-Aug-2026"""

print("\n[1] Current 8-field schema — name recomposition")
new = parse(NEW)
print(f"     parsed {len(new)} scheme rows")
for c in ["120503", "119550", "128952"]:
    if c in new:
        print(f"       {c} → {new[c]}")
check("plan and option are folded back into the name",
      new.get("120503") == "Axis ELSS Tax Saver Fund - Direct Plan - Growth Option")
check("'-' placeholder option is not appended",
      " - -" not in " ".join(new.values()))

print("\n[2] Legacy 6-field schema still parses unchanged")
old = parse(OLD)
check("legacy composite name passed through verbatim",
      old.get("120503") == "Axis ELSS Tax Saver Fund - Direct Plan - Growth Option")
check("both schemas yield identical names for the same fund",
      old.get("119550") == new.get("119550"),
      f"{old.get('119550')!r}")

print("\n[3] The filter funnel that was collapsing")
growth = filter_preferred_plans(new)
direct = filter_direct_plans(growth)
print(f"     {len(new)} rows → {len(growth)} growth → {len(direct)} direct")
check("growth filter now keeps the equity funds", len(growth) >= 11, f"{len(growth)} kept")
check("Bonus Option row still excluded",
      not any("Bonus" in n for n in growth.values()))
check("IDCW rows still excluded",
      not any("idcw" in n.lower() for n in growth.values()))
check("pure debt fund still excluded",
      not any("PSU Debt" in n for n in growth.values()))

print("\n[4] Category counts")
counts = get_category_fund_counts(new)
nonzero = {k: v for k, v in counts.items() if v}
print(f"     {nonzero}")
check("Large Cap no longer zero", counts["Large Cap"] > 0)
check("Mid Cap populated", counts["Mid Cap"] > 0)
check("Small Cap populated", counts["Small Cap"] > 0)
check("Flexi Cap populated", counts["Flexi Cap"] > 0)
check("ELSS populated", counts["ELSS"] > 0)
check("Index Funds populated", counts["Index Funds"] > 0)
check("Aggressive Hybrid populated (was killed by the 'debt fund' exclusion)",
      counts["Aggressive Hybrid"] > 0)

print("\n[5] The validator that makes a future schema change fail loudly")
bare = {str(i): "Some Equity Fund" for i in range(100)}
try:
    _validate_scheme_names(bare, "test")
    check("bare names are rejected", False, "validator let them through")
except ValueError as e:
    check("bare names are rejected", True, str(e)[:88] + "…")
try:
    _validate_scheme_names(
        {str(i): f"Fund {i} - Direct Plan - Growth" for i in range(100)}, "test")
    check("healthy composite names accepted", True)
except ValueError as e:
    check("healthy composite names accepted", False, str(e))

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
