"""Dynamic EMS Dispatch & Station Placement Optimization.

An interactive application that turns real Seattle 911 data into optimised
decisions about where ambulances wait and how they are dispatched.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src import ahp as ahp_mod
from src import assignment as asg
from src import config, covering, data_loader, stats, viz

st.set_page_config(
    page_title="Dynamic EMS Dispatch & Station Placement Optimization",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAP_H = 520


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading Seattle 911 incidents…")
def load_all():
    incidents = data_loader.load_incidents()
    stations = data_loader.load_existing_stations()
    return incidents, stations


@st.cache_data(show_spinner=False)
def load_zip_geojson():
    p = config.ZCTA_GEOJSON
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data(show_spinner="Filtering the selected period…")
def period_slice(start, end, include_trauma: bool, min_share: float):
    incidents, _ = load_all()
    sel = data_loader.filter_period(incidents, start, end, include_trauma)
    all_zones = data_loader.build_zones(sel, min_share) if len(sel) else pd.DataFrame()
    return sel, all_zones


def travel_kwargs() -> dict:
    return {
        "speed_kmh": st.session_state["speed"],
        "circuity": st.session_state["circuity"],
        "turnout_min": st.session_state["turnout"],
    }


def kpi(col, label: str, value: str, help_text: str | None = None):
    col.metric(label, value, help=help_text)


# ---------------------------------------------------------------------------
# Data availability gate
# ---------------------------------------------------------------------------

if not config.CALLS_PARQUET.exists():
    st.title("🚑 Dynamic EMS Dispatch & Station Placement Optimization")
    st.error("Processed data not found — the app needs the real Seattle dataset first.")
    st.code(
        "python scripts/download_data.py   # Seattle 911 (kzjm-xkqj), stations, ZIP polygons\n"
        "python scripts/preprocess.py      # clean, EMS-filter, tag ZIP zones",
        language="bash",
    )
    st.stop()

incidents_all, existing_stations = load_all()
zip_geojson = load_zip_geojson()

DATA_MIN = incidents_all["datetime"].min().date()
DATA_MAX = incidents_all["datetime"].max().date()


# ---------------------------------------------------------------------------
# Sidebar — the controls that drive every model
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Study period")
    preset = st.selectbox(
        "Quick range",
        ["Last 12 months", "Last 3 years", "Last 30 days", "Entire dataset", "Custom"],
        index=0,
        help="Load a chosen period of Seattle EMS data.",
    )
    offsets = {
        "Last 30 days": pd.DateOffset(days=30),
        "Last 12 months": pd.DateOffset(years=1),
        "Last 3 years": pd.DateOffset(years=3),
    }
    preset_start = (
        DATA_MIN if preset == "Entire dataset"
        else max(DATA_MIN, (pd.Timestamp(DATA_MAX) - offsets[preset]).date())
        if preset in offsets else DATA_MIN
    )

    if preset == "Custom":
        dates = st.date_input(
            "Date range",
            value=(max(DATA_MIN, (pd.Timestamp(DATA_MAX) - pd.DateOffset(years=1)).date()),
                   DATA_MAX),
            min_value=DATA_MIN, max_value=DATA_MAX,
        )
        if isinstance(dates, (tuple, list)) and len(dates) == 2:
            start_date, end_date = dates
        else:  # user is mid-selection, having picked only the start date
            start_date = dates[0] if isinstance(dates, (tuple, list)) else dates
            end_date = DATA_MAX
    else:
        start_date, end_date = preset_start, DATA_MAX
        st.caption(f"{start_date:%d %b %Y} → {end_date:%d %b %Y}")

    include_trauma = st.toggle(
        "Include trauma & rescue calls",
        value=False,
        help="Off = aid/medic dispatches only (the project's 'genuine EMS/aid calls'). "
             "On = also count collisions, rescues and violence, which generate patients "
             "but are typed by their cause.",
    )

    min_share_pct = st.slider(
        "Zone materiality floor (% of demand)", 0.0, 1.0,
        100 * data_loader.DEFAULT_MIN_DEMAND_SHARE, 0.05,
        help="ZIP zones below this share of period demand are excluded from the "
             "models. The feed carries a long tail of fringe ZIPs with a handful "
             "of mutual-aid calls a year; set covering would otherwise give each "
             "of them its own dedicated station.",
    )

    st.header("Response-time model")
    st.caption("Haversine straight-line distance → estimated road response time.")
    st.slider("Effective speed (km/h)", 20.0, 80.0, config.DEFAULT_SPEED_KMH, 2.5,
              key="speed", help="Average emergency-response speed on city streets.")
    st.slider("Road circuity factor", 1.0, 1.8, config.DEFAULT_CIRCUITY, 0.05,
              key="circuity", help="Streets are longer than the straight line between "
                                   "two points; 1.3 is the usual urban value.")
    st.slider("Turnout time (min)", 0.0, 4.0, config.DEFAULT_TURNOUT_MIN, 0.5,
              key="turnout", help="Time from alarm to wheels rolling, added to travel time.")

    threshold = st.select_slider(
        "Response-time limit (min)",
        options=[4, 5, 6, 7, 8, 9, 10, 12, 15],
        value=8,
        help="A zone counts as covered if a station can reach it within this limit.",
    )
    st.caption(
        f"Implied reach ≈ "
        f"{max(threshold - st.session_state['turnout'], 0) / 60 * st.session_state['speed'] / st.session_state['circuity']:.1f} km"
    )

sel, all_zones = period_slice(start_date, end_date, include_trauma, min_share_pct / 100.0)

if sel.empty:
    st.title("🚑 Dynamic EMS Dispatch & Station Placement Optimization")
    st.warning("No EMS incidents in the selected period. Widen the date range.")
    st.stop()

zones = data_loader.material_zones(all_zones)
dropped = all_zones.loc[~all_zones["material"]] if "material" in all_zones else pd.DataFrame()

if zones.empty:
    st.title("🚑 Dynamic EMS Dispatch & Station Placement Optimization")
    st.warning("The materiality floor excluded every zone. Lower it in the sidebar.")
    st.stop()

tk = travel_kwargs()

st.markdown(
    f"<div style='font-size:10.5px;font-weight:700;letter-spacing:0.09em;"
    f"opacity:0.55;margin-bottom:-6px'>{config.COURSE.upper()}</div>",
    unsafe_allow_html=True,
)
st.title("🚑 Dynamic EMS Dispatch & Station Placement Optimization")
st.caption(
    "Optimizing where ambulances wait and how they are dispatched, using real Seattle 911 data — "
    f"**{len(sel):,}** EMS incidents across **{zones.shape[0]}** demand zones, "
    f"{start_date:%d %b %Y} → {end_date:%d %b %Y}."
)

if len(dropped):
    st.caption(
        f"↳ {len(dropped)} fringe ZIP zone(s) below the {min_share_pct:.2f}% materiality "
        f"floor were excluded from the models "
        f"({int(dropped['demand'].sum()):,} incidents = "
        f"{100 * dropped['demand_share'].sum():.3f}% of period demand): "
        f"{', '.join(dropped['zip_code'].astype(str))}."
    )

tabs = st.tabs([
    "Overview",
    "1 · Data & Statistics",
    "2 · AHP Site Scoring",
    "3 · Covering Model",
    "4 · Dispatch Assignment",
    "Decision Map",
])


# ===========================================================================
# Overview
# ===========================================================================
with tabs[0]:
    st.subheader("The problem")
    c1, c2 = st.columns(2)
    c1.info("**Q1 — Where to wait?**  \nPlacing stations well means more of the city "
            "sits within a short response-time radius.")
    c2.info("**Q2 — Who to send?**  \nDuring simultaneous emergencies, smart "
            "ambulance-to-incident matching cuts everyone's wait.")
    st.caption(
        "**Why it matters** — emergency response is one of the oldest and most "
        "studied applications of operations research, and a domain where better "
        "models translate directly into lives. We apply it to a real city with "
        "real data."
    )

    st.subheader("What we deliver")
    o1, o2 = st.columns(2)
    o1.markdown(
        "**1 · Interactive data exploration**  \nLoad a chosen period of Seattle EMS "
        "data and generate statistics on demand — busiest zones, peak hours, demand "
        "patterns.  \n→ *Tab “1 · Data & Statistics”*\n\n"
        "**3 · Optimal dispatch**  \nUse the assignment model to match available "
        "ambulances to incidents and minimise total response time.  \n"
        "→ *Tab “4 · Dispatch Assignment”*"
    )
    o2.markdown(
        "**2 · Optimal station placement**  \nUse the covering model to determine "
        "where ambulance stations should be located across the city.  \n"
        "→ *Tab “3 · Covering Model”*\n\n"
        "**4 · Visual decision support**  \nPresent the statistics and optimisation "
        "results on an interactive city map.  \n→ *Tab “Decision Map”*"
    )

    st.subheader("The data")
    d1, d2, d3 = st.columns(3)
    kpi(d1, "Incident records", f"{len(incidents_all):,}",
        "Cleaned & de-duplicated from the Seattle Real-Time Fire 911 Calls feed.")
    kpi(d2, "Coverage", f"{DATA_MIN:%b %Y} – {DATA_MAX:%b %Y}")
    kpi(d3, "Real SFD stations", f"{len(existing_stations)}",
        "Used as the benchmark our optimised placement is compared against.")

    st.markdown(
        "**Seattle Real-Time Fire 911 Calls** — City of Seattle Open Data Portal "
        "(`data.seattle.gov`), dataset ID `kzjm-xkqj`. Public domain, open licence, "
        "updated every 5 minutes. This is the *demand* side of the project: it records "
        "where and when real emergencies occur across the city."
    )

    st.markdown("**What's inside the dataset**")
    st.dataframe(pd.DataFrame([
        ("address / latitude / longitude", "Exact incident location", "Mapping demand, distances"),
        ("type", "Kind of call (medical, aid, fire…)", "Filtering to EMS calls only"),
        ("datetime", "Date and time of the call", "Peak-hour stats, dispatch scenario"),
        ("zip code (computed)", "Which zone the incident is in", "Grouping demand into zones"),
        ("incident number", "Unique ID per call", "De-duplication"),
    ], columns=["Field", "Meaning", "Used for"]), hide_index=True, width="stretch")

    st.subheader("How everything connects")
    st.html(viz.flow_diagram())
    m1, m2, m3, m4 = st.columns(4)
    m1.success("**Step 1 — Statistics**  \nBusiest zones, peak timing, demand weights.")
    m2.success("**Step 2 — AHP**  \nRank candidate areas on three weighted criteria.")
    m3.success("**Step 3 — Covering (ILP)**  \nWhere stations should go.")
    m4.success("**Step 4 — Assignment (LP)**  \nWhich unit goes to which incident.")

    st.caption(
        "The link between the two optimisations is the core contribution: the station "
        "locations chosen by the covering model become the ambulance bases used by the "
        "assignment model."
    )

    st.subheader("Tools & tech stack")
    t = st.columns(3)
    t[0].markdown("**Python + pandas**  \nLoading data & computing statistics")
    t[1].markdown("**PuLP**  \nSolving the covering (ILP) & assignment (LP) models")
    t[2].markdown("**Streamlit**  \nThe interactive user interface")
    t = st.columns(3)
    t[0].markdown("**Folium / Plotly**  \nMapping incidents & stations")
    t[1].markdown("**Haversine**  \nEstimating travel distance / time")
    t[2].markdown("**Data ingestion**  \nExport CSV from portal / API into the app")

    st.html(viz.team_strip(config.COURSE, config.TEAM))


# ===========================================================================
# 1 · Data & Statistics
# ===========================================================================
with tabs[1]:
    st.subheader("Step 1 — Data cleaning & statistics")
    st.caption(
        "Filtered to genuine EMS/aid calls, grouped by ZIP-code zone. These counts "
        "also become each zone's demand weight for both models."
    )

    h = stats.headline(sel)
    k = st.columns(5)
    kpi(k[0], "EMS incidents", f"{h['incidents']:,}")
    kpi(k[1], "Demand zones", f"{h['zones']}")
    kpi(k[2], "Avg per day", f"{h['per_day']:.0f}")
    kpi(k[3], "Busiest zone", h["busiest_zone"])
    kpi(k[4], "Peak hour", h["peak_hour"])

    st.divider()
    zone_stats = stats.busiest_zones(sel)
    by_hour = stats.calls_by_hour(sel)
    by_wd = stats.calls_by_weekday(sel)

    st.markdown("#### Busiest zones — where demand concentrates")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(viz.chart_busiest_zones(zone_stats), width="stretch")
    c2.dataframe(
        zone_stats[["rank", "zip_code", "incidents", "share_pct", "cumulative_pct"]]
        .rename(columns={"zip_code": "ZIP", "share_pct": "share %",
                         "cumulative_pct": "cumulative %"}),
        hide_index=True, width="stretch", height=430,
    )

    st.markdown("#### Peak timing — when load spikes")
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.chart_calls_by_hour(by_hour), width="stretch")
    c2.plotly_chart(viz.chart_calls_by_weekday(by_wd), width="stretch")
    st.plotly_chart(viz.chart_heatmap(stats.hour_weekday_heatmap(sel)),
                    width="stretch")

    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.chart_call_types(stats.call_type_mix(sel)), width="stretch")
    with c2:
        st.plotly_chart(viz.chart_daily_trend(stats.daily_trend(sel)), width="stretch")
        st.markdown("**Demand weights fed into both optimisations**")
        st.dataframe(
            stats.demand_weights(zones)[["rank", "zip_code", "demand", "weight_pct"]]
            .rename(columns={"zip_code": "ZIP", "demand": "incidents",
                             "weight_pct": "weight %"}),
            hide_index=True, width="stretch", height=260,
        )

    st.divider()
    st.markdown("#### Demand on the map")
    st_folium(viz.map_demand(zones, sel, zip_geojson), height=MAP_H,
              width="stretch", returned_objects=[], key="map_demand")


# ===========================================================================
# 2 · AHP
# ===========================================================================
with tabs[2]:
    st.subheader("Step 2 — AHP multi-criteria site scoring")
    st.caption(
        "AHP combines competing criteria into one priority score per candidate area, "
        "so we can rank which zones most deserve a station."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**The three criteria**")
        for name in config.AHP_CRITERIA:
            st.markdown(f"- **{name}** — {config.AHP_CRITERIA_HELP[name]}")
    with c2:
        st.markdown("**Building the weights**")
        st.markdown(
            "1. **Pairwise comparison** — compare the criteria two at a time on the "
            "Saaty 1–9 scale (1 = equally important … 9 = extremely more important). "
            "For 3 criteria, just 3 comparisons.\n"
            "2. **Derive the weights** — the comparison matrix is converted into a "
            "weight for each criterion (all summing to 100%)."
        )

    st.divider()
    st.warning(
        "**The sliders below start on placeholder judgements, not the team's.** "
        "Each member should set their own comparisons in their tab. The criterion "
        "weights — and therefore the site ranking and the candidate pool the "
        "covering model chooses from — depend on these values."
    )

    n_members = st.number_input(
        "Team members filling the table independently", 1, 8, len(config.TEAM), 1,
        help="Each member fills the table independently; the judgements are then "
             "combined into one matrix.",
    )

    # The full Saaty scale: 1–9 plus the even intermediate values, with the
    # reciprocals that express the comparison the other way round.
    SAATY = [1.0 / v for v in range(9, 1, -1)] + [float(v) for v in range(1, 10)]
    LBL = {v: (f"1/{round(1 / v)}" if v < 1 else f"{v:g}") for v in SAATY}
    PAIRS = [(0, 1), (0, 2), (1, 2)]

    per_member = []
    # Named from the roster so each tab is the member who actually filled it;
    # extra tabs fall back to a number if the count is raised above the team.
    member_names = [
        config.TEAM[i][0] if i < len(config.TEAM) else f"Member {i + 1}"
        for i in range(int(n_members))
    ]
    # Tabs get the given name only — full names do not fit the tab strip.
    tab_labels = [n.split()[0] for n in member_names]
    member_tabs = st.tabs(tab_labels)
    for mi, mt in enumerate(member_tabs):
        with mt:
            comp = {}
            cols = st.columns(3)
            for pi, (i, j) in enumerate(PAIRS):
                a, b = config.AHP_CRITERIA[i], config.AHP_CRITERIA[j]
                default = ahp_mod.DEFAULT_COMPARISONS[(i, j)]
                val = cols[pi].select_slider(
                    f"**{a}** vs **{b}**", options=SAATY, value=default,
                    format_func=lambda v: LBL[v], key=f"ahp_{mi}_{i}_{j}",
                    help=f"How many times more important is {a} than {b}? "
                         f"Values below 1 mean {b} is the more important one.",
                )
                comp[(i, j)] = val
            per_member.append(comp)

    result = ahp_mod.run_ahp(per_member)

    st.divider()
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Aggregated comparison matrix**")
        st.caption("Members combined with the element-wise geometric mean, the only "
                   "average that keeps the matrix reciprocal.")
        st.dataframe(result.matrix_frame().round(3), width="stretch")
        st.plotly_chart(viz.chart_ahp_weights(result.weights_series),
                        width="stretch")
    with c2:
        st.markdown("**Consistency check — keeping the judgement logical**")
        d = result.diagnostics
        cc = st.columns(4)
        kpi(cc[0], "λmax", f"{d['lambda_max']:.4f}")
        kpi(cc[1], "CI", f"{d['CI']:.4f}", "(λmax − n) / (n − 1)")
        kpi(cc[2], "RI", f"{d['RI']:.2f}", "Saaty's Random Index for this matrix size")
        kpi(cc[3], "CR", f"{d['CR']:.4f}", "CI / RI — must stay below 0.10")

        if d["consistent"]:
            st.success(f"**CR = {d['CR']:.4f} < 0.10** — the judgements are consistent "
                       "and the weights are trustworthy.")
        else:
            st.error(f"**CR = {d['CR']:.4f} ≥ 0.10** — the judgements contradict each "
                     "other. Revise the comparisons before using these weights.")

        st.markdown("**Per-member consistency**")
        st.dataframe(pd.DataFrame([
            {"Member": member_names[p["member"] - 1], "λmax": round(p["lambda_max"], 4),
             "CI": round(p["CI"], 4), "CR": round(p["CR"], 4),
             "CR < 0.1": "yes" if p["consistent"] else "NO",
             **{c: round(w, 4) for c, w in zip(result.criteria, p["weights"])}}
            for p in result.per_member
        ]), hide_index=True, width="stretch")

    st.divider()
    st.markdown("#### Candidate sites ranked by AHP priority")
    scored = ahp_mod.score_sites(zones, existing_stations, result, threshold, **tk)
    st.session_state["ahp_result"] = result
    st.session_state["ahp_scored"] = scored

    c1, c2 = st.columns([3, 4])
    c1.plotly_chart(viz.chart_ahp_ranking(scored), width="stretch")
    c2.dataframe(
        scored[["rank", "zip_code", "Incident Density", "Equity Gap",
                "Coverage Potential", "zones_reached", "ahp_score"]]
        .rename(columns={"zip_code": "ZIP", "Incident Density": "incidents",
                         "Equity Gap": "km to nearest station",
                         "Coverage Potential": "demand reachable",
                         "zones_reached": "zones reachable",
                         "ahp_score": "AHP score"})
        .round(3),
        hide_index=True, width="stretch", height=430,
    )
    st.caption(
        "Criteria are max-normalised onto [0, 1] before weighting, because all three "
        "are benefit criteria on ratio scales. The resulting ranking feeds the covering "
        "model in the next step."
    )


# ===========================================================================
# 3 · Covering model
# ===========================================================================
with tabs[3]:
    st.subheader("Step 3 — Optimisation 1: Covering model (ILP)")
    st.caption(
        "Decide **where** to place stations. A zone is covered if a station can reach "
        f"it within the {threshold}-minute limit set in the sidebar."
    )

    # The AHP tab above runs earlier in the same script pass and always stores
    # its ranking, so this is a fallback rather than a gate.
    scored = st.session_state.get("ahp_scored")
    if scored is None:
        scored = ahp_mod.score_sites(
            zones, existing_stations, ahp_mod.run_ahp([ahp_mod.DEFAULT_COMPARISONS]),
            threshold, **tk)

    cand_pool = st.slider(
        "Candidate sites considered (top-N by AHP priority)",
        3, len(scored), len(scored), 1,
        help="AHP ranks the areas; the covering model then chooses among the "
             "highest-ranked ones.",
    )
    candidates = scored.head(cand_pool)[["zip_code", "latitude", "longitude"]].copy()
    cand_ahp = scored.head(cand_pool)["ahp_score"].to_numpy()

    mode = st.radio(
        "Mode",
        ["Mode A — Set Covering (minimum stations)",
         "Mode B — Maximal Covering (fixed budget)"],
        horizontal=True,
    )

    if mode.startswith("Mode A"):
        st.markdown(
            "Determines the **minimum number of stations** required such that every "
            "demand zone is served within the response time. Each candidate location is "
            "a binary decision, and the model guarantees complete coverage at the lowest "
            "possible facility count.  \n**Objective: minimise number of stations.**"
        )
        run = st.button("Solve set-covering model", type="primary")
        if run:
            with st.spinner("Solving ILP…"):
                st.session_state["cover"] = covering.solve_set_covering(
                    candidates, zones, threshold, cand_ahp, tk)
    else:
        st.markdown(
            "Fixes the number of stations at a **user-defined limit** and positions them "
            "to serve the greatest possible share of demand. This reflects real-world "
            "budget constraints, where full coverage may be infeasible.  \n"
            "**Objective: maximise demand covered.**"
        )
        p = st.slider("Station budget P", 1, min(30, len(candidates)), 8, 1)
        run = st.button("Solve maximal-covering model", type="primary")
        if run:
            with st.spinner("Solving ILP…"):
                st.session_state["cover"] = covering.solve_maximal_covering(
                    candidates, zones, threshold, p, tk)

    res = st.session_state.get("cover")
    if res is None:
        st.info("Choose a mode and run the model to see results.")
    else:
        st.divider()
        st.markdown(f"#### Result — {res.mode} @ {res.threshold_min:g} min")
        k = st.columns(5)
        kpi(k[0], "Stations opened", f"{res.n_stations}")
        kpi(k[1], "Demand covered", f"{res.demand_pct:.1f}%",
            f"{res.demand_covered:,.0f} of {res.demand_total:,.0f} incidents")
        kpi(k[2], "Zones covered", f"{res.zones_covered}/{res.zones_total}")
        kpi(k[3], "Solver status", res.status)
        kpi(k[4], "Solve time", f"{res.solve_seconds:.2f}s")

        if res.uncoverable:
            st.warning(
                f"{len(res.uncoverable)} zone(s) cannot be reached within "
                f"{res.threshold_min:g} min by *any* candidate site and were excluded "
                f"from the covering constraints: {', '.join(res.uncoverable)}. "
                "Raise the limit or the assumed speed to bring them in range."
            )

        c1, c2 = st.columns([2, 3])
        c1.markdown("**Stations opened**")
        c1.dataframe(res.chosen[["station", "zip_code", "latitude", "longitude"]]
                     .rename(columns={"zip_code": "zone"}).round(4),
                     hide_index=True, width="stretch", height=320)
        c2.markdown("**Per-zone outcome**")
        c2.dataframe(
            res.zone_table.assign(status=np.where(res.zone_table["covered"],
                                                  "covered", "NOT covered"))
            [["zip_code", "demand", "status", "nearest_station_zone", "response_min"]]
            .rename(columns={"zip_code": "ZIP", "demand": "incidents",
                             "nearest_station_zone": "served from",
                             "response_min": "response (min)"}),
            hide_index=True, width="stretch", height=320)

        st.markdown("#### Placement on the map")
        st_folium(viz.map_coverage(res, existing_stations, True, **tk),
                  height=MAP_H, width="stretch",
                  returned_objects=[], key="map_cover")

    st.divider()
    st.markdown("#### How does coverage change at 5 vs 8 vs 10 minutes?")
    if st.button("Run the 5 / 8 / 10-minute sweep"):
        with st.spinner("Re-solving the set-covering model at each limit…"):
            st.session_state["sweep"] = covering.threshold_sweep(
                candidates, zones, config.RESPONSE_THRESHOLDS_MIN, cand_ahp, tk)
    sweep = st.session_state.get("sweep")
    if sweep is not None:
        c1, c2, c3 = st.columns([2, 2, 3])
        c1.plotly_chart(viz.chart_threshold_stations(sweep), width="stretch")
        c2.plotly_chart(viz.chart_threshold_coverage(sweep), width="stretch")
        c3.dataframe(sweep.round(2), hide_index=True, width="stretch")
        st.caption("A tighter limit forces more stations; a looser one lets fewer "
                   "stations blanket the same city.")

    st.divider()
    st.markdown("#### How does our placement compare to Seattle's real stations?")
    if st.button("Benchmark against the 36 real SFD stations"):
        rows = []
        with st.spinner("Scoring both plans on identical assumptions…"):
            for t in config.RESPONSE_THRESHOLDS_MIN:
                ours = covering.solve_set_covering(candidates, zones, t, cand_ahp, tk)
                theirs = covering.evaluate_fixed_stations(existing_stations, zones, t, tk)
                rows.append({"plan": "Optimised placement", "threshold_min": t,
                             "stations": ours.n_stations,
                             "demand_covered_pct": ours.demand_pct,
                             "zones_covered": ours.zones_covered})
                rows.append({"plan": "Seattle's real stations", "threshold_min": t,
                             "stations": len(existing_stations),
                             "demand_covered_pct": theirs.demand_pct,
                             "zones_covered": theirs.zones_covered})
        st.session_state["bench"] = rows
    rows = st.session_state.get("bench")
    if rows:
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(viz.chart_comparison(rows), width="stretch")
        c2.dataframe(pd.DataFrame(rows).round(2), hide_index=True,
                     width="stretch")
        st.caption(
            "Both plans are scored with the same travel-time model, so the comparison "
            "is like-for-like. Our placement reaches comparable coverage from far fewer "
            "sites because it is optimised against demand rather than accumulated "
            "historically — the real network also serves fire suppression, which this "
            "EMS-only model does not price in."
        )


# ===========================================================================
# 4 · Dispatch assignment
# ===========================================================================
with tabs[4]:
    st.subheader("Step 4 — Optimisation 2: Assignment model (LP)")
    st.caption(
        "Decide **which** ambulance goes to **which** incident during a busy hour, so "
        "that total travel time across all responders is minimised."
    )

    res = st.session_state.get("cover")
    # An else-branch rather than st.stop(): every tab renders in the same script
    # pass, so stopping here would blank out the Decision Map tab below.
    if res is None or res.n_stations == 0:
        st.info("Run the covering model in **3 · Covering Model** first — its station "
                "locations are the ambulance bases used here. This linkage is what "
                "integrates both optimisations into one framework.")
    else:

        st.success(
            f"Ambulance bases taken from the covering model: **{res.n_stations} stations** "
            f"({res.mode} @ {res.threshold_min:g} min)."
        )

        busiest = stats.busiest_hours_table(sel, top_n=25)
        c1, c2 = st.columns([3, 2])
        with c1:
            options = list(busiest["hour_start"])
            chosen_hour = st.selectbox(
                "Peak hour of simultaneous emergencies",
                options,
                format_func=lambda t: (f"{pd.Timestamp(t):%a %d %b %Y, %H:%M} — "
                                       f"{int(busiest.loc[busiest['hour_start'] == t, 'incidents'].iloc[0])} incidents"),
                help="The busiest clock hours in the selected period.",
            )
            units = st.slider("Ready units per station", 1, 4, 1,
                              help="One ready unit stationed at each base, per the model's "
                                   "setup. Raise it if the hour has more calls than bases.")
        with c2:
            st.markdown("**Busiest hours in this period**")
            st.dataframe(busiest.head(10)[["rank", "hour_start", "weekday", "incidents"]],
                         hide_index=True, width="stretch", height=250)

        hour_inc = data_loader.incidents_in_hour(sel, chosen_hour)
        ambulances = asg.ambulances_from_stations(res.chosen, units)

        c1, c2, c3 = st.columns(3)
        kpi(c1, "Incidents this hour", f"{len(hour_inc)}")
        kpi(c2, "Ambulances available", f"{len(ambulances)}")
        kpi(c3, "Balance", "balanced" if len(hour_inc) == len(ambulances)
            else ("more incidents" if len(hour_inc) > len(ambulances) else "more units"))

        if st.button("Solve assignment model", type="primary"):
            with st.spinner("Solving LP…"):
                st.session_state["asg"] = asg.solve_assignment(ambulances, hour_inc, tk)
                st.session_state["asg_amb"] = ambulances

        a = st.session_state.get("asg")
        if a is None:
            st.info("Pick an hour and solve to see the dispatch plan.")
        else:
            st.divider()
            k = st.columns(5)
            kpi(k[0], "Total travel time", f"{a.total_time_min:.1f} min",
                "The quantity the LP minimises.")
            kpi(k[1], "Mean response", f"{a.mean_time_min:.1f} min")
            kpi(k[2], "Worst response", f"{a.max_time_min:.1f} min")
            kpi(k[3], "Served / unserved", f"{a.n_served} / {a.n_unserved}")
            kpi(k[4], "Idle units", f"{a.n_idle}")

            if a.dummy_kind == "none":
                st.caption("Equal counts — the assignment matrix was already square.")
            else:
                st.caption(
                    f"Unequal counts handled by padding the cost matrix with "
                    f"**{a.dummy_kind}** at zero cost — the transportation-problem link. "
                    + ("Incidents matched to a dummy row cannot be served this hour."
                       if a.dummy_kind == "dummy ambulances"
                       else "Units matched to a dummy column stay idle.")
                )

            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown("**Dispatch plan**")
                st.dataframe(
                    a.assignments[["incident_number", "time", "type", "zip_code", "address",
                                   "ambulance", "base_zone", "response_min"]]
                    .rename(columns={"incident_number": "incident", "zip_code": "ZIP",
                                     "base_zone": "from zone",
                                     "response_min": "response (min)"}),
                    hide_index=True, width="stretch", height=360)
            with c2:
                st.plotly_chart(viz.chart_dispatch_comparison(a.total_time_min,
                                                              a.greedy_total_min or 0.0),
                                width="stretch")
                if a.greedy_total_min:
                    st.metric("Improvement over nearest-unit-first dispatch",
                              f"{a.improvement_pct:.1f}%",
                              help="Greedy dispatch sends the nearest free unit call by "
                                   "call, which can strand a later incident with a very "
                                   "long drive. The LP optimises the hour as a whole.")

            st.plotly_chart(viz.chart_response_distribution(a.assignments, threshold),
                            width="stretch")

            with st.expander("Cost matrix (travel time in minutes)"):
                st.markdown("**Real ambulances × real incidents**")
                st.dataframe(a.cost_matrix, width="stretch")
                if a.dummy_kind != "none":
                    st.markdown(f"**Balanced matrix — padded with {a.dummy_kind}**")
                    st.dataframe(a.balanced_matrix, width="stretch")

            st.markdown("#### Dispatch on the map")
            st_folium(viz.map_dispatch(a, st.session_state["asg_amb"]), height=MAP_H,
                      width="stretch", returned_objects=[], key="map_dispatch")


# ===========================================================================
# Decision map
# ===========================================================================
with tabs[5]:
    st.subheader("Visual decision support")
    st.caption("The statistics and optimisation results on one interactive city map.")

    res = st.session_state.get("cover")
    layer = st.radio(
        "Layer",
        ["Demand by zone", "Optimised station placement", "Peak-hour dispatch"],
        horizontal=True,
    )

    if layer == "Demand by zone":
        st_folium(viz.map_demand(zones, sel, zip_geojson), height=600,
                  width="stretch", returned_objects=[], key="dm_demand")
        st.caption("Circle size and colour both encode EMS incident volume per ZIP "
                   "zone. Toggle the incident heat layer from the control top-right.")
    elif layer == "Optimised station placement":
        if res is None:
            st.info("Run the covering model in **3 · Covering Model** first.")
        else:
            st_folium(viz.map_coverage(res, existing_stations, True, **tk), height=600,
                      width="stretch", returned_objects=[], key="dm_cover")
            st.caption("Blue markers are the stations our model opens, with their "
                       f"{res.threshold_min:g}-minute reach. Green zones are covered, "
                       "red are not. Seattle's real stations can be switched on from "
                       "the layer control.")
    else:
        a = st.session_state.get("asg")
        if a is None:
            st.info("Solve the assignment model in **4 · Dispatch Assignment** first.")
        else:
            st_folium(viz.map_dispatch(a, st.session_state["asg_amb"]), height=600,
                      width="stretch", returned_objects=[], key="dm_disp")
            st.caption("Each line is one ambulance dispatched to one incident under the "
                       "optimal plan. Red incidents had no unit available this hour.")

st.divider()
st.caption(
    "Data: Seattle Real-Time Fire 911 Calls (`kzjm-xkqj`, data.seattle.gov, public "
    "domain) · Seattle fire stations (Seattle City GIS) · ZIP boundaries (US Census "
    "TIGERweb). Built with pandas, PuLP, Streamlit, Folium and Plotly."
)
