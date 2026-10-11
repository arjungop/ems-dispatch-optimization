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
check("sidebar controls present", len(at.sidebar.slider) >= 4,
      f"{len(at.sidebar.slider)} sliders, {len(at.sidebar.selectbox)} selectboxes")
check("dataframes rendered", len(at.dataframe) >= 5, f"{len(at.dataframe)} tables")

# --- presentation layer -------------------------------------------------
# The app is one tab per stage of the pipeline, and the site-scoring stage
# nests one tab per team member filling the comparison table independently.
tab_labels = [t.label for t in at.main.get("tab")]
STAGES = ("Data", "Site scoring", "Station placement", "Dispatch")
check("every pipeline stage has a tab",
      all(any(s in lbl for lbl in tab_labels) for s in STAGES),
      ", ".join(STAGES))
check("one AHP tab per team member",
      all(f"Member {i + 1}" in tab_labels for i in range(len(config.TEAM))),
      f"{len(config.TEAM)} members")

# The section headings carry the page's structure, so they are checked like
# any other output rather than eyeballed.
headings = [s.value for s in at.subheader]
# Each heading is the narrative line for its section; the chart inside states
# the measure ("Top 15 ZIP zones by EMS incident count"), so the two no longer
# repeat each other and the headings are checked on their own wording.
# "Sensitivity analysis" rather than the older "Threshold sensitivity": the
# section now sweeps the station budget as well as the response-time limit.
SECTIONS = ("Where demand concentrates", "When load spikes", "Demand weights",
            "Judgement consistency", "Candidate sites", "Sensitivity analysis",
            "Robustness of the station count", "Benchmark vs. real stations")
check("every section heading rendered",
      all(any(s in h for h in headings) for s in SECTIONS),
      f"{len(headings)} headings")

metric_labels = [m.label for m in at.metric]
HEADLINE = ("EMS incidents", "Demand zones", "Avg per day", "Busiest zone",
            "Peak hour")
check("headline KPIs rendered", all(k in metric_labels for k in HEADLINE),
      f"{len(at.metric)} metrics")

# The AHP tab computes on every run, so its consistency metrics must be present.
check("AHP consistency metrics shown",
      all(k in metric_labels for k in ("λmax", "CI", "RI", "CR")))

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
mode_sel = next((r for r in at.segmented_control
                 if any("Mode A" in str(o) for o in r.options)), None)
check("covering mode selector present", mode_sel is not None)
if mode_sel:
    mode_b = next(o for o in mode_sel.options if "Mode B" in str(o))
    mode_sel.set_value(mode_b).run()
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
    mode_sel.set_value(next(o for o in mode_sel.options if "Mode A" in str(o))).run()

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

# --- regressions caught during the end-to-end audit --------------------
print("\n   regression checks…")
import re  # noqa: E402

# Reset to the default period before the consistency checks below.
at.sidebar.selectbox[0].set_value("Last 12 months").run()
at.sidebar.toggle[0].set_value(False).run()

# The header and the Data tab must agree on how many zones the models plan
# around. They once disagreed (39 vs 28) because one counted every ZIP in the
# feed and the other counted only the material ones.
agree = True
for floor in (0.0, 0.5, 1.0):
    next(sl for sl in at.sidebar.slider if "materiality" in sl.label).set_value(floor).run()
    hdr = next((c.value for c in at.caption if "demand zones" in str(c.value)), "")
    m = re.search(r"\*\*(\d+)\*\* demand zones", hdr)
    kpi = next((x.value for x in at.metric if x.label == "Demand zones"), None)
    if not m or str(kpi) != m.group(1):
        agree = False
check("header and Data tab agree on the zone count", agree,
      "checked at three materiality floors")

# Pulling the candidate pool to its floor must not break the budget slider:
# its default once sat above a max that shrinks with the pool.
next(sl for sl in at.sidebar.slider if "materiality" in sl.label).set_value(0.1).run()
pool = next((sl for sl in at.slider if "Candidate sites" in sl.label), None)
if pool:
    pool.set_value(int(pool.min)).run()
    check("smallest candidate pool renders", not at.exception, "; ".join(exceptions(at)))
    mode_r = next((r for r in at.segmented_control
                   if any("Mode A" in str(o) for o in r.options)), None)
    if mode_r:
        mode_r.set_value(next(o for o in mode_r.options if "Mode B" in str(o))).run()
        check("budget slider survives the smallest pool", not at.exception,
              "; ".join(exceptions(at)))
        mode_r = next(r for r in at.segmented_control
                      if any("Mode A" in str(o) for o in r.options))
        mode_r.set_value(next(o for o in mode_r.options if "Mode A" in str(o))).run()
    pool = next(sl for sl in at.slider if "Candidate sites" in sl.label)
    pool.set_value(int(pool.max)).run()

# Each stage ends on a map, and a Folium map that fails to build takes the
# whole script down -- so all three are required to be on the page at once.
maps = [e for e in at.main if e.type == "component_instance"]
check("all three maps render", len(maps) == 3,
      f"{len(maps)} of 3 — demand, coverage, dispatch")

print("\n" + "=" * 72)
print(f"   {passed} passed, {failed} failed")
print("=" * 72)
sys.exit(1 if failed else 0)
