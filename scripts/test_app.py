"""Headless smoke test of the Streamlit app.

Uses Streamlit's own AppTest runner to execute ``app.py`` exactly as a browser
session would -- including the widget interactions that trigger each model --
and fails loudly on any uncaught exception.

Run:  python scripts/test_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

from src import config  # noqa: E402

passed = failed = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"   [PASS] {label}" + (f" — {detail}" if detail else ""))
    else:
        failed += 1
        print(f"   [FAIL] {label}" + (f" — {detail}" if detail else ""))


def exceptions(at: AppTest) -> list[str]:
    return [f"{e.value}" for e in at.exception]


print("=" * 72)
print("Streamlit app smoke test")
print("=" * 72)

t0 = time.time()
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
at.run()
print(f"   initial script run: {time.time() - t0:.1f}s")

check("app runs without exceptions", not at.exception, "; ".join(exceptions(at)))
if at.exception:
    print("\n".join(exceptions(at)))
    sys.exit(1)

check("title rendered", any("EMS Dispatch" in t.value for t in at.title))
check("all six tabs present", len(at.tabs) >= 6, f"{len(at.tabs)} tabs")
check("sidebar controls present", len(at.sidebar.slider) >= 4,
      f"{len(at.sidebar.slider)} sliders, {len(at.sidebar.selectbox)} selectboxes")
check("KPI metrics rendered", len(at.metric) >= 10, f"{len(at.metric)} metrics")
check("dataframes rendered", len(at.dataframe) >= 5, f"{len(at.dataframe)} tables")

# The AHP tab computes on every run, so its consistency metrics must be present.
metric_labels = [m.label for m in at.metric]
check("AHP consistency metrics shown",
      all(k in metric_labels for k in ("λmax", "CI", "RI", "CR")))

# --- presentation layer -------------------------------------------------
# The deck's title slide and methodology diagram are part of the deliverable,
# so they are checked like any other output rather than eyeballed.
html_blob = "".join(e.body for e in at.main if e.type == "html")
tab_labels = [t.label for t in at.main.get("tab")]

check("methodology flow diagram rendered",
      "<svg" in html_blob and "Seattle 911 feed" in html_blob)
check("flow diagram shows both optimisations",
      "Covering model" in html_blob and "Assignment model" in html_blob)
check("flow diagram labels the covering -> assignment linkage",
      "stations become ambulance bases" in html_blob)
check("team roster rendered with names and roll numbers",
      all(n in html_blob for n, _ in config.TEAM)
      and all(r in html_blob for _, r in config.TEAM),
      f"{len(config.TEAM)} members")
check("AHP member tabs are named after the team",
      all(n.split()[0] in tab_labels for n, _ in config.TEAM),
      ", ".join(n.split()[0] for n, _ in config.TEAM))

# The deck's narrative slides -- the four objectives and the tool list -- are
# stated in the Overview so the app can be walked through against the deck.
prose = ("".join(str(e.value) for e in at.markdown)
         + "".join(str(e.value) for e in at.caption))
OBJECTIVES = ("Interactive data exploration", "Optimal station placement",
              "Optimal dispatch", "Visual decision support")
TOOLS = ("Python + pandas", "PuLP", "Streamlit", "Folium / Plotly",
         "Haversine", "Data ingestion")
check("all four project objectives stated",
      all(o in prose for o in OBJECTIVES), f"{len(OBJECTIVES)} objectives")
check("project significance stated",
      "oldest and most" in prose and "real city with" in prose)
check("all six tools listed", all(t in prose for t in TOOLS), f"{len(TOOLS)} tools")

# --- drive the covering model ------------------------------------------
buttons = {b.label: b for b in at.button}
print(f"\n   buttons available: {list(buttons)}")

target = next((lbl for lbl in buttons if "set-covering" in lbl.lower()), None)
check("set-covering button present", target is not None, str(target))
if target:
    t0 = time.time()
    buttons[target].click().run()
    check("covering model runs without exceptions", not at.exception,
          "; ".join(exceptions(at)))
    print(f"   covering solve round-trip: {time.time() - t0:.1f}s")
    labels = [m.label for m in at.metric]
    check("covering results rendered",
          "Stations opened" in labels and "Demand covered" in labels)
    opened = next((m.value for m in at.metric if m.label == "Stations opened"), None)
    covered = next((m.value for m in at.metric if m.label == "Demand covered"), None)
    print(f"   -> {opened} stations opened, {covered} of demand covered")

# --- Mode B: maximal covering ------------------------------------------
mode_radio = next((r for r in at.radio if any("Mode A" in str(o) for o in r.options)), None)
check("covering mode radio present", mode_radio is not None)
if mode_radio:
    mode_b = next(o for o in mode_radio.options if "Mode B" in str(o))
    mode_radio.set_value(mode_b).run()
    check("switching to maximal covering renders", not at.exception,
          "; ".join(exceptions(at)))
    b = {btn.label: btn for btn in at.button}
    mb = next((lbl for lbl in b if "maximal-covering" in lbl.lower()), None)
    check("maximal-covering button present", mb is not None, str(mb))
    if mb:
        b[mb].click().run()
        check("maximal covering runs without exceptions", not at.exception,
              "; ".join(exceptions(at)))
        lbls = [m.label for m in at.metric]
        opened = next((m.value for m in at.metric if m.label == "Stations opened"), None)
        covered = next((m.value for m in at.metric if m.label == "Demand covered"), None)
        check("maximal covering results rendered", "Demand covered" in lbls)
        print(f"   -> Mode B: {opened} stations, {covered} of demand covered")
    # back to Mode A so later steps use the set-covering result
    mode_radio.set_value(next(o for o in mode_radio.options if "Mode A" in str(o))).run()

# --- threshold sweep ----------------------------------------------------
buttons = {b.label: b for b in at.button}
sweep = next((lbl for lbl in buttons if "sweep" in lbl.lower()), None)
if sweep:
    buttons[sweep].click().run()
    check("5/8/10-minute sweep runs without exceptions", not at.exception,
          "; ".join(exceptions(at)))

# --- benchmark vs real stations ----------------------------------------
buttons = {b.label: b for b in at.button}
bench = next((lbl for lbl in buttons if "benchmark" in lbl.lower()), None)
if bench:
    buttons[bench].click().run()
    check("benchmark vs real stations runs without exceptions", not at.exception,
          "; ".join(exceptions(at)))

# --- assignment model ---------------------------------------------------
buttons = {b.label: b for b in at.button}
asg_btn = next((lbl for lbl in buttons if "assignment" in lbl.lower()), None)
check("assignment button reachable after covering ran", asg_btn is not None,
      str(asg_btn))
if asg_btn:
    t0 = time.time()
    buttons[asg_btn].click().run()
    check("assignment model runs without exceptions", not at.exception,
          "; ".join(exceptions(at)))
    print(f"   assignment solve round-trip: {time.time() - t0:.1f}s")
    labels = [m.label for m in at.metric]
    check("dispatch results rendered", "Total travel time" in labels)
    tot = next((m.value for m in at.metric if m.label == "Total travel time"), None)
    served = next((m.value for m in at.metric if m.label == "Served / unserved"), None)
    print(f"   -> total travel time {tot}, served/unserved {served}")

# --- exercise the sidebar controls -------------------------------------
print("\n   exercising sidebar controls…")
at.sidebar.selectbox[0].set_value("Last 30 days").run()
check("switching study period works", not at.exception, "; ".join(exceptions(at)))

at.sidebar.toggle[0].set_value(True).run()
check("trauma-inclusion toggle works", not at.exception, "; ".join(exceptions(at)))

at.sidebar.slider[1].set_value(60.0).run()   # effective speed
check("changing the speed assumption works", not at.exception, "; ".join(exceptions(at)))

at.sidebar.select_slider[0].set_value(5).run()  # response-time limit
check("changing the response-time limit works", not at.exception,
      "; ".join(exceptions(at)))

at.sidebar.selectbox[0].set_value("Entire dataset").run()
check("entire-dataset period works", not at.exception, "; ".join(exceptions(at)))

print("\n" + "=" * 72)
print(f"   {passed} passed, {failed} failed")
print("=" * 72)
sys.exit(1 if failed else 0)
