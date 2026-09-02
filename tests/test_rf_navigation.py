"""
The risk-free rate must survive navigating between pages.

Why this test needs a real browser, unlike the other seven suites:

    AppTest renders ONE page per instance. It cannot navigate, so it cannot
    observe what happens to a widget when Streamlit re-registers it on the
    next page — and that is exactly where this broke. Every headless test
    passed while the running app showed 7.0% on six pages and 4.0% on
    Rankings, silently computing Sharpe, Sortino and every alpha there at a
    risk-free rate the user never chose.

    The cause: a single shared widget key across pages. On navigation the
    re-registered slider initialised from min_value instead of the value
    already sitting in session state. The fix gives each page its own slider
    key and keeps the value of record in a plain, non-widget key that nothing
    can reset — see rf_control() in utils/ui.py.

It was found by screenshotting the app and reading the number, which is the
only reason it was found at all.

Requires playwright + a chromium at /opt/pw-browsers/chromium.
Run:  python tests/test_rf_navigation.py
"""
import os, sys, subprocess, time, warnings, re
warnings.filterwarnings("ignore")
from playwright.sync_api import sync_playwright
# ROOT is the repo root, one level up from tests/. Every path in this file
# hangs off it, so it must not be the test's own directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

# tests/ goes on PYTHONPATH ahead of ROOT so the child interpreter auto-imports
# tests/sitecustomize.py and installs the synthetic-fund stubs.
#
# That file lives in tests/ rather than the repo root deliberately. Python
# imports any importable module named `sitecustomize` at interpreter startup,
# so at the root ANY `python` or `streamlit` run from this directory would load
# it — one stray MF_STUB_DATA=1 in the environment and the real app would be
# quietly serving synthetic funds. Here it is only ever reachable when a test
# puts tests/ on the path on purpose. ROOT stays on the path because the stub
# itself imports data.tri_loader and utils.constants.
PORT = 8911
env = dict(os.environ, PYTHONPATH=HERE + os.pathsep + ROOT, MF_STUB_DATA="1")
proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py",
                         "--server.port", str(PORT), "--server.headless", "true"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)


def wait_for_server(port, timeout=120):
    """Poll until Streamlit accepts a connection.

    This used to be time.sleep(15), tuned on one machine. A slower box boots
    Streamlit in more than fifteen seconds and the test then failed for reasons
    that had nothing to do with the risk-free rate.
    """
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"streamlit exited early with code {proc.returncode}")
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                time.sleep(2)          # accepted != finished rendering
                return
        except OSError:
            time.sleep(0.5)
    raise SystemExit(f"streamlit did not answer on port {port} within {timeout}s")


def launch_chromium(pw):
    """Use whatever chromium this machine actually has.

    MF_CHROMIUM wins if set. Otherwise a pinned path is used only when it
    exists — it points at the Linux container this test was written in, and on
    Windows playwright installs to %LOCALAPPDATA%\\ms-playwright instead. Falling
    through to playwright's own managed browser is what makes `playwright
    install chromium` sufficient on any platform.
    """
    override = os.environ.get("MF_CHROMIUM")
    if override:
        return pw.chromium.launch(executable_path=override)
    pinned = "/opt/pw-browsers/chromium"
    if os.path.exists(pinned):
        return pw.chromium.launch(executable_path=pinned)
    return pw.chromium.launch()


def rf(pg):
    txt = pg.locator('[data-testid="stSlider"]').first.inner_text()
    nums = re.findall(r"\d+\.\d+", txt)
    return nums[0] if nums else "?"


def settle(pg, timeout=45000):
    """Wait until the slider reading stops changing.

    Fixed wait_for_timeout(13000) calls were the other machine-speed
    assumption. Reading too early returns the PREVIOUS page's value, which is
    indistinguishable from the very bug this file exists to catch — a false
    FAIL that looks exactly like a true one. Polling until two consecutive
    reads agree is both safer and faster.
    """
    pg.wait_for_selector('[data-testid="stSlider"]', timeout=timeout)
    deadline = time.time() + timeout / 1000.0
    last = None
    while time.time() < deadline:
        cur = rf(pg)
        if cur != "?" and cur == last:
            return cur
        last = cur
        pg.wait_for_timeout(700)
    return last if last is not None else "?"


fails = []


def check(n, c, d=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"  — {d}" if d else ""))
    if not c:
        fails.append(n)


try:
    wait_for_server(PORT)
    with sync_playwright() as pw:
        b = launch_chromium(pw)
        pg = b.new_page(viewport={"width": 1680, "height": 1050})
        pg.goto(f"http://localhost:{PORT}/", wait_until="networkidle", timeout=90000)
        check("home starts at the default", settle(pg) == "7.00", rf(pg))
        for label in ["Rankings", "Fund Analytics", "Portfolio Analytics",
                      "Factor Attribution", "Predictive Analytics"]:
            pg.get_by_role("link", name=label).first.click()
            check(f"{label} keeps 7.00 across navigation", settle(pg) == "7.00", rf(pg))
        # change it, then navigate — the new value must travel
        pg.get_by_role("link", name="Rankings").first.click()
        settle(pg)
        # Confirm each click LANDED before sending the next. A fixed pause here
        # (2500ms originally, 1200ms after the speed-up) is a bet on how fast
        # Streamlit reruns; lose the bet and a click is swallowed, the rate
        # reads 7.40 instead of 7.50, and the test blames the feature rather
        # than its own timing.
        for i in range(5):
            before = rf(pg)
            pg.get_by_role("button", name="+").first.click()
            deadline = time.time() + 20
            while time.time() < deadline and rf(pg) == before:
                pg.wait_for_timeout(300)
            if rf(pg) == before:
                print(f"  note: click {i+1} did not register within 20s "
                      f"(still {before})")
        v = settle(pg)
        check("+ raises the rate on Rankings", v == "7.50", v)
        pg.get_by_role("link", name="Fund Analytics").first.click()
        check("the changed rate travels to another page", settle(pg) == "7.50", rf(pg))
        b.close()
finally:
    proc.terminate()
print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
