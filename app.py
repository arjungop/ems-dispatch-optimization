"""Dynamic EMS Dispatch & Station Placement Optimization.

An interactive application that turns real Seattle 911 data into optimised
decisions about where ambulances wait and how they are dispatched.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src import ahp as ahp_mod
from src import assignment as asg
from src import config, covering, data_loader, stats, viz

st.set_page_config(
    page_title="Dynamic EMS Dispatch & Station Placement Optimization",
    page_icon=":material/ambulance:",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAP_H = 480

# Streamlit's dataframe row pitch: the bundled grid draws a 32px row plus a 1px
# separator, and the header row occupies the same 33px. Heights are derived from
# it rather than guessed -- an arbitrary pixel height slices the last visible row
# in half, and a box taller than the frame leaves blank filler rows under the
# data; both read as an unfinished table.
ROW_PX = 33


# ---------------------------------------------------------------------------
# Table presentation
# ---------------------------------------------------------------------------
# Every figure in the app is formatted in one place, so a count is thousands
# separated everywhere it appears, a percentage always carries its sign, and a
# response time always reads in minutes to one decimal. Left to pandas, the
# same column renders as 11.9239 in one table and 11.92 in the next.

def tbl_height(n_rows: int, max_rows: int) -> int:
    """A dataframe height that ends exactly on a row boundary."""
    return ROW_PX * (min(int(n_rows), max_rows) + 1) + 2


def count_col(label: str, **kw):
    return st.column_config.NumberColumn(label, format="localized", **kw)


def pct_col(label: str, **kw):
    return st.column_config.NumberColumn(label, format="%.1f%%", **kw)


def min_col(label: str, **kw):
    return st.column_config.NumberColumn(label, format="%.1f min", **kw)


def km_col(label: str, **kw):
    return st.column_config.NumberColumn(label, format="%.2f km", **kw)


def rank_col(label: str = "#"):
    return st.column_config.NumberColumn(label, format="%d", width="small")


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


def kpi(col, label: str, value: str, icon: str | None = None, border: bool = True,
        help_text: str | None = None):
    # Metrics are their own cards, except where they already sit inside one --
    # a bordered card nested in a bordered card reads as clutter.
    col.metric(label, value, icon=icon, border=border, help=help_text)


def kpi_row():
    """A metric row that wraps rather than squeezes.

    ``st.columns(5)`` splits the page into five equal cells whatever the window
    is doing, and at laptop widths the cells are narrower than the content, so
    Streamlit ellipsises the tiles -- "EMS incidents / 79,152" renders as
    "EMS inci... / 79,...". A horizontal container keeps each tile at its
    natural size and moves the overflow onto a second line instead.
    """
    return st.container(horizontal=True, horizontal_alignment="left")


TITLE = ":material/ambulance: Dynamic EMS Dispatch & Station Placement Optimization"


# ---------------------------------------------------------------------------
# Data availability gate
# ---------------------------------------------------------------------------

if not config.CALLS_PARQUET.exists():
    st.title(TITLE)
    st.error(
        "Processed data not found — the app needs the real Seattle dataset first.",
        icon=":material/database_off:",
    )
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
    st.header(":material/calendar_month: Study period")
    preset = st.selectbox(
        "Quick range",
        ["Last 12 months", "Last 3 years", "Last 30 days", "Entire dataset", "Custom"],
        index=0,
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

    include_trauma = st.toggle("Include trauma & rescue calls", value=False)

    st.header(":material/speed: Response-time model")

    # The limit is the one number the models are actually steered by, so it
    # stays in the open; the calibration constants behind it are assumptions a
    # reader rarely changes, and are tucked away so the sidebar reads as two
    # decisions rather than seven knobs. The slot reserves the limit's position
    # above the expander even though its sliders must be created first.
    limit_slot = st.container()

    with st.expander("Model assumptions", icon=":material/tune:"):
        st.slider("Effective speed (km/h)", 20.0, 80.0, config.DEFAULT_SPEED_KMH, 2.5,
                  key="speed")
        st.slider("Road circuity factor", 1.0, 1.8, config.DEFAULT_CIRCUITY, 0.05,
                  key="circuity")
        st.slider("Turnout time (min)", 0.0, 4.0, config.DEFAULT_TURNOUT_MIN, 0.5,
                  key="turnout")
        min_share_pct = st.slider(
            "Zone materiality floor (% of demand)", 0.0, 1.0,
            100 * data_loader.DEFAULT_MIN_DEMAND_SHARE, 0.05,
        )

    with limit_slot:
        threshold = st.select_slider(
            "Response-time limit (min)",
            options=[4, 5, 6, 7, 8, 9, 10, 12, 15],
            value=8,
        )
        st.caption(
            f"Implied reach ≈ "
            f"{max(threshold - st.session_state['turnout'], 0) / 60 * st.session_state['speed'] / st.session_state['circuity']:.1f} km"
        )

    st.caption("Seattle 911 EMS data · every model solves locally.")

sel, all_zones = period_slice(start_date, end_date, include_trauma, min_share_pct / 100.0)

if sel.empty:
    st.title(TITLE)
    st.warning("No EMS incidents in the selected period. Widen the date range.",
               icon=":material/event_busy:")
    st.stop()

zones = data_loader.material_zones(all_zones)

if zones.empty:
    st.title(TITLE)
    st.warning("The materiality floor excluded every zone. Lower it in the sidebar.",
               icon=":material/filter_alt_off:")
    st.stop()

tk = travel_kwargs()

st.title(TITLE)
st.caption(
    f"**{len(sel):,}** EMS incidents across **{zones.shape[0]}** demand zones, "
    f"{start_date:%d %b %Y} → {end_date:%d %b %Y}."
)

tabs = st.tabs([
    ":material/bar_chart: Data",
    ":material/target: Site scoring",
    ":material/location_on: Station placement",
    ":material/local_shipping: Dispatch",
])


# ===========================================================================
# Data
# ===========================================================================
with tabs[0]:
    h = stats.headline(sel)
    # The hour tile carries only the start of the peak hour: the full
    # "13:00-14:00" span overflows a metric at this column width and is
    # silently ellipsised to "13:00-14:...", which reads as a rendering fault.
    peak_start = h["peak_hour"].split("-")[0]

    k = kpi_row()
    kpi(k, "EMS incidents", f"{h['incidents']:,}", icon=":material/ambulance:")
    kpi(k, "Demand zones", f"{zones.shape[0]}", icon=":material/map:")
    kpi(k, "Avg per day", f"{h['per_day']:,.0f}", icon=":material/calendar_today:")
    kpi(k, "Busiest zone", h["busiest_zone"], icon=":material/location_on:")
    kpi(k, "Peak hour", peak_start, icon=":material/schedule:",
        help_text=f"Busiest hour of the day: {h['peak_hour']}.")

    zone_stats = stats.busiest_zones(sel)
    by_hour = stats.calls_by_hour(sel)
    by_wd = stats.calls_by_weekday(sel)

    st.subheader("Where demand concentrates", divider="gray")
    # The ranked chart is sized by its 15 bars; the table beside it is pinned to
    # the same number of rows so the two cards end on the same line.
    c1, c2 = st.columns([3, 2], vertical_alignment="top")
    with c1:
        with st.container(border=True):
            st.plotly_chart(viz.chart_busiest_zones(zone_stats), width="stretch")
    with c2:
        with st.container(border=True):
            st.dataframe(
                zone_stats,
                hide_index=True, width="stretch",
                height=tbl_height(len(zone_stats), 15),
                column_order=["rank", "zip_code", "incidents", "share_pct",
                              "cumulative_pct"],
                column_config={
                    "rank": rank_col(),
                    "zip_code": st.column_config.TextColumn("ZIP", width="small"),
                    "incidents": count_col("Incidents"),
                    "share_pct": pct_col("Share"),
                    "cumulative_pct": pct_col("Cumulative"),
                },
            )

    st.subheader("When load spikes", divider="gray")
    c1, c2 = st.columns(2, vertical_alignment="top")
    with c1:
        with st.container(border=True):
            st.plotly_chart(viz.chart_calls_by_hour(by_hour), width="stretch")
    with c2:
        with st.container(border=True):
            st.plotly_chart(viz.chart_calls_by_weekday(by_wd), width="stretch")
    with st.expander("More detail — hour × weekday, call mix, daily trend",
                      icon=":material/query_stats:"):
        st.plotly_chart(viz.chart_heatmap(stats.hour_weekday_heatmap(sel)),
                        width="stretch")
        c1, c2 = st.columns(2, vertical_alignment="top")
        c1.plotly_chart(viz.chart_call_types(stats.call_type_mix(sel)), width="stretch")
        c2.plotly_chart(viz.chart_daily_trend(stats.daily_trend(sel)), width="stretch")

    # The weights and the map are the same fact twice -- the number each zone
    # carries into the models, and where that zone is -- so they share a row.
    # Four columns stretched across the full page width put a one-digit rank in
    # a 200px cell and read as a half-empty table.
    st.subheader("Demand weights fed into both optimisations", divider="gray")
    weights_tbl = stats.demand_weights(zones)
    c1, c2 = st.columns([2, 3], vertical_alignment="top")
    with c1:
        with st.container(border=True):
            st.markdown("**Weight per zone**")
            st.dataframe(
                weights_tbl, hide_index=True, width="stretch",
                height=tbl_height(len(weights_tbl), 15),
                column_order=["rank", "zip_code", "demand", "weight_pct"],
                column_config={
                    "rank": rank_col(),
                    "zip_code": st.column_config.TextColumn("ZIP", width="small"),
                    "demand": count_col("Incidents"),
                    "weight_pct": st.column_config.ProgressColumn(
                        "Weight", format="%.1f%%", min_value=0.0,
                        max_value=float(weights_tbl["weight_pct"].max()),
                    ),
                },
            )
    with c2:
        with st.container(border=True):
            st.markdown("**Demand across the city**")
            st.caption("Circle area is incident volume; shading fills the ZIP zone "
                       "it belongs to. Hover any zone for its exact count.")
            st_folium(viz.map_demand(zones, zip_geojson), height=MAP_H,
                      width="stretch", returned_objects=[], key="map_demand")


# ===========================================================================
# Site Scoring
# ===========================================================================
with tabs[1]:
    st.caption(
        "Each team member rates the three siting criteria against one another on "
        "Saaty's 1–9 scale. The ratings are aggregated, checked for logical "
        "consistency, and turned into the weights that rank every candidate zone."
    )

    # The full Saaty scale: 1–9 plus the even intermediate values, with the
    # reciprocals that express the comparison the other way round.
    SAATY = [1.0 / v for v in range(9, 1, -1)] + [float(v) for v in range(1, 10)]
    LBL = {v: (f"1/{round(1 / v)}" if v < 1 else f"{v:g}") for v in SAATY}
    PAIRS = [(0, 1), (0, 2), (1, 2)]

    with st.container(border=True):
        st.markdown("**Pairwise judgements**")
        # A number input left to its own devices stretches the full width of a
        # wide-layout page, which makes a one-digit control look like a text
        # field; it is held to a readable width here instead.
        n_members = st.columns([1, 3])[0].number_input(
            "Team members", 1, 8, len(config.TEAM), 1,
            help="Each member fills the comparison table independently; their "
                 "matrices are aggregated into one set of weights.",
        )

        per_member = []
        # Names stay internal (looking up each member's saved judgements) -- the
        # UI only ever shows the generic "Member N" label.
        member_names = [
            config.TEAM[i][0] if i < len(config.TEAM) else f"Member {i + 1}"
            for i in range(int(n_members))
        ]
        member_labels = [f"Member {i + 1}" for i in range(int(n_members))]
        member_tabs = st.tabs(member_labels)
        for mi, mt in enumerate(member_tabs):
            with mt:
                comp = {}
                cols = st.columns(3)
                for pi, (i, j) in enumerate(PAIRS):
                    a, b = config.AHP_CRITERIA[i], config.AHP_CRITERIA[j]
                    default = config.judgements_for(member_names[mi])[(i, j)]
                    val = cols[pi].select_slider(
                        f"**{a}** vs **{b}**", options=SAATY, value=default,
                        format_func=lambda v: LBL[v], key=f"ahp_{mi}_{i}_{j}",
                        help=(f"{a}: {config.AHP_CRITERIA_HELP[a]}  \n"
                              f"{b}: {config.AHP_CRITERIA_HELP[b]}  \n"
                              f"Above 1 favours {a}; below 1 favours {b}."),
                    )
                    comp[(i, j)] = val
                per_member.append(comp)

    result = ahp_mod.run_ahp(per_member)

    st.subheader("Judgement consistency", divider="gray")
    # Short headers: the criteria names are long enough that a 3x3 matrix of
    # them overflows its card and the last column is cut off mid-word.
    SHORT = {"Incident Density": "Density", "Equity Gap": "Equity",
             "Coverage Potential": "Coverage"}
    matrix = result.matrix_frame().round(3).rename(index=SHORT, columns=SHORT)

    c1, c2 = st.columns([3, 2], vertical_alignment="top")
    with c1:
        with st.container(border=True):
            st.markdown("**Aggregated comparison matrix**")
            st.dataframe(
                matrix, width="stretch", height=tbl_height(len(matrix), 3),
                column_config={c: st.column_config.NumberColumn(c, format="%.3f")
                               for c in matrix.columns},
            )
            st.plotly_chart(viz.chart_ahp_weights(result.weights_series),
                            width="stretch")
    with c2:
        with st.container(border=True):
            st.markdown("**Consistency check**")
            d = result.diagnostics
            cc = st.columns(2)
            kpi(cc[0], "λmax", f"{d['lambda_max']:.4f}", border=False,
                help_text="Principal eigenvalue of the aggregated matrix.")
            kpi(cc[1], "CI", f"{d['CI']:.4f}", border=False,
                help_text="Consistency Index: (λmax − n) / (n − 1).")
            cc = st.columns(2)
            kpi(cc[0], "RI", f"{d['RI']:.2f}", border=False,
                help_text="Saaty's Random Consistency Index for a matrix this size.")
            kpi(cc[1], "CR", f"{d['CR']:.4f}", border=False,
                help_text="Consistency Ratio: CI / RI. Must stay below 0.10.")

            if d["consistent"]:
                st.success(f"**CR = {d['CR']:.4f} < 0.10** — the judgements are consistent "
                           "and the weights are trustworthy.",
                           icon=":material/check_circle:")
            else:
                st.error(f"**CR = {d['CR']:.4f} ≥ 0.10** — the judgements contradict each "
                         "other. Revise the comparisons before using these weights.",
                         icon=":material/error:")

            with st.expander("Per-member consistency", icon=":material/group:"):
                per_member_tbl = pd.DataFrame([
                    {"Member": member_labels[p["member"] - 1],
                     "λmax": p["lambda_max"], "CI": p["CI"], "CR": p["CR"],
                     "Consistent": bool(p["consistent"]),
                     **{SHORT[c]: w for c, w in zip(result.criteria, p["weights"])}}
                    for p in result.per_member
                ])
                st.dataframe(
                    per_member_tbl, hide_index=True, width="stretch",
                    height=tbl_height(len(per_member_tbl), 8),
                    column_config={
                        "λmax": st.column_config.NumberColumn(format="%.4f"),
                        "CI": st.column_config.NumberColumn(format="%.4f"),
                        "CR": st.column_config.NumberColumn(format="%.4f"),
                        "Consistent": st.column_config.CheckboxColumn(
                            help="CR below 0.10."),
                        **{SHORT[c]: st.column_config.NumberColumn(
                            SHORT[c], format="percent")
                           for c in result.criteria},
                    },
                )

    st.subheader("Candidate sites ranked by priority", divider="gray")
    scored = ahp_mod.score_sites(zones, existing_stations, result, threshold, **tk)
    st.session_state["ahp_result"] = result
    st.session_state["ahp_scored"] = scored

    c1, c2 = st.columns([2, 3], vertical_alignment="top")
    with c1:
        with st.container(border=True):
            st.plotly_chart(viz.chart_ahp_ranking(scored), width="stretch")
    with c2:
        with st.container(border=True):
            st.markdown("**Criterion values behind each score**")
            st.dataframe(
                scored, hide_index=True, width="stretch",
                height=tbl_height(len(scored), 15),
                column_order=["rank", "zip_code", "Incident Density", "Equity Gap",
                              "Coverage Potential", "zones_reached", "ahp_score"],
                column_config={
                    "rank": rank_col(),
                    "zip_code": st.column_config.TextColumn("ZIP", width="small"),
                    "Incident Density": count_col(
                        "Incidents", help="Emergencies in the zone."),
                    "Equity Gap": km_col(
                        "Nearest station",
                        help="Distance to the closest station Seattle already runs."),
                    "Coverage Potential": count_col(
                        "Demand in reach",
                        help="Incidents reachable from here inside the limit."),
                    "zones_reached": count_col(
                        "Zones", width="small",
                        help="Demand zones reachable from here inside the limit."),
                    "ahp_score": st.column_config.NumberColumn(
                        "Priority", format="%.3f", width="small"),
                },
            )


# ===========================================================================
# Station Placement
# ===========================================================================
with tabs[2]:
    # The Site Scoring tab above runs earlier in the same script pass and
    # always stores its ranking, so this is a fallback rather than a gate.
    scored = st.session_state.get("ahp_scored")
    if scored is None:
        scored = ahp_mod.score_sites(
            zones, existing_stations,
            ahp_mod.run_ahp([config.judgements_for(n) for n, _ in config.TEAM]),
            threshold, **tk)

    # The three controls that define a run are grouped into one panel: loose on
    # the page, a full-width slider and a bare button read as leftover widgets
    # rather than as the model's inputs.
    with st.container(border=True):
        st.markdown("**Set up the run**")
        left, right = st.columns([2, 3], vertical_alignment="top")

        # A short period or a high materiality floor can leave very few zones,
        # and a slider whose min exceeds its max is an outright error -- so
        # below four sites there is nothing to choose and the whole pool is used.
        n_sites = len(scored)
        with left:
            if n_sites > 3:
                cand_pool = st.slider(
                    "Candidate sites considered (top-N by priority)",
                    3, n_sites, n_sites, 1,
                )
            else:
                cand_pool = n_sites
                st.caption(f"All {n_sites} candidate site(s) in this period are considered.")
        candidates = scored.head(cand_pool)[["zip_code", "latitude", "longitude"]].copy()
        cand_ahp = scored.head(cand_pool)["ahp_score"].to_numpy()

        with right:
            mode = st.segmented_control(
                "Objective",
                ["Mode A — minimum stations for full coverage",
                 "Mode B — best coverage for a fixed budget"],
                default="Mode A — minimum stations for full coverage",
                required=True,
            )

        if mode.startswith("Mode A"):
            st.caption(f"Opens the fewest stations that put every demand zone within "
                       f"{threshold:g} minutes.")
            run = st.button("Solve the set-covering model", type="primary",
                            icon=":material/play_arrow:")
            if run:
                with st.spinner("Solving…"):
                    st.session_state["cover"] = covering.solve_set_covering(
                        candidates, zones, threshold, cand_ahp, tk)
        else:
            # The budget cannot exceed the candidate pool, and the default has to
            # be clamped into that range -- a value above max_value is an error,
            # and the pool shrinks as soon as the slider above is pulled down.
            p_max = max(1, min(30, len(candidates)))
            if p_max > 1:
                p = st.columns([2, 3])[0].slider(
                    "Station budget P", 1, p_max, min(8, p_max), 1)
            else:
                p = 1
                st.caption("Only one candidate site available — the budget is fixed at 1.")
            st.caption(f"Places exactly P stations so the demand covered within "
                       f"{threshold:g} minutes is as large as possible.")
            run = st.button("Solve the maximal-covering model", type="primary",
                            icon=":material/play_arrow:")
            if run:
                with st.spinner("Solving…"):
                    st.session_state["cover"] = covering.solve_maximal_covering(
                        candidates, zones, threshold, p, tk)

    res = st.session_state.get("cover")
    if res is None:
        st.info("Choose an objective above and solve to see the placement.",
                icon=":material/arrow_upward:")
    else:
        st.subheader(f"Result — {res.mode} at {res.threshold_min:g} min", divider="gray")
        k = kpi_row()
        kpi(k, "Stations opened", f"{res.n_stations}", icon=":material/location_on:")
        kpi(k, "Demand covered", f"{res.demand_pct:.1f}%", icon=":material/pie_chart:")
        kpi(k, "Zones covered", f"{res.zones_covered}/{res.zones_total}",
            icon=":material/map:")
        kpi(k, "Solver status", res.status, icon=":material/check_circle:")
        kpi(k, "Solve time", f"{res.solve_seconds:.2f}s", icon=":material/timer:")

        if res.uncoverable:
            st.warning(
                f"{len(res.uncoverable)} zone(s) cannot be reached within "
                f"{res.threshold_min:g} min by *any* candidate site and were excluded "
                f"from the covering constraints: {', '.join(res.uncoverable)}. "
                "Raise the limit or the assumed speed to bring them in range.",
                icon=":material/warning:",
            )

        zone_table = res.zone_table.assign(
            status=np.where(res.zone_table["covered"], "Covered", "Not covered"))

        c1, c2 = st.columns([2, 3], vertical_alignment="top")
        with c1:
            with st.container(border=True):
                st.markdown("**Stations opened**")
                st.dataframe(
                    res.chosen, hide_index=True, width="stretch",
                    height=tbl_height(len(res.chosen), 10),
                    column_order=["station", "zip_code", "latitude", "longitude"],
                    column_config={
                        "station": st.column_config.TextColumn("Station", width="small"),
                        "zip_code": st.column_config.TextColumn("Zone", width="small"),
                        "latitude": st.column_config.NumberColumn("Lat", format="%.4f"),
                        "longitude": st.column_config.NumberColumn("Lon", format="%.4f"),
                    },
                )
        with c2:
            with st.container(border=True):
                st.markdown("**Per-zone outcome**")
                st.dataframe(
                    zone_table, hide_index=True, width="stretch",
                    height=tbl_height(len(zone_table), 10),
                    column_order=["zip_code", "demand", "status",
                                  "nearest_station_zone", "response_min"],
                    column_config={
                        "zip_code": st.column_config.TextColumn("ZIP", width="small"),
                        "demand": count_col("Incidents"),
                        "status": st.column_config.TextColumn("Status", width="small"),
                        "nearest_station_zone": st.column_config.TextColumn(
                            "Served from", width="small"),
                        "response_min": min_col("Response"),
                    },
                )

        with st.container(border=True):
            st.markdown("**Stations & coverage on the map**")
            st.caption(f"Shaded circles are the {res.threshold_min:g}-minute reach of "
                       "each opened station. Zone dots are green where covered, red "
                       "where not.")
            st_folium(viz.map_coverage(res, existing_stations, True, **tk), height=MAP_H,
                      width="stretch", returned_objects=[], key="map_cover")

    st.subheader("Threshold sensitivity", divider="gray")
    with st.container(border=True):
        st.markdown("**How the plan changes with the response-time limit**")
        st.caption("Re-solves the covering model at 5, 8 and 10 minutes on the same "
                   "candidate pool, so the cost of a tighter promise is explicit.")
        if st.button("Run the 5 / 8 / 10-minute sweep", icon=":material/query_stats:"):
            with st.spinner("Re-solving at each limit…"):
                st.session_state["sweep"] = covering.threshold_sweep(
                    candidates, zones, config.RESPONSE_THRESHOLDS_MIN, cand_ahp, tk)
        sweep = st.session_state.get("sweep")
        if sweep is not None:
            c1, c2, c3 = st.columns([2, 2, 3], vertical_alignment="top")
            c1.plotly_chart(viz.chart_threshold_stations(sweep), width="stretch")
            c2.plotly_chart(viz.chart_threshold_coverage(sweep), width="stretch")
            c3.dataframe(
                sweep, hide_index=True, width="stretch",
                height=tbl_height(len(sweep), 6),
                column_order=["threshold_min", "stations_required",
                              "demand_covered_pct", "zones_covered"],
                column_config={
                    "threshold_min": st.column_config.NumberColumn(
                        "Limit", format="%d min", width="small"),
                    "stations_required": count_col("Stations", width="small"),
                    "demand_covered_pct": pct_col("Demand covered"),
                    "zones_covered": count_col("Zones", width="small"),
                },
            )

    st.subheader("Benchmark vs. real stations", divider="gray")
    with st.container(border=True):
        st.markdown("**Optimised placement against Seattle's own**")
        st.caption(f"Scores this plan and the {len(existing_stations)} real SFD "
                   "stations on identical travel assumptions at each limit.")
        if st.button(f"Benchmark against the {len(existing_stations)} real SFD stations",
                     icon=":material/compare_arrows:"):
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
            bench = pd.DataFrame(rows)
            c1, c2 = st.columns([3, 2], vertical_alignment="top")
            c1.plotly_chart(viz.chart_comparison(rows), width="stretch")
            c2.dataframe(
                bench, hide_index=True, width="stretch",
                height=tbl_height(len(bench), 6),
                column_config={
                    "plan": st.column_config.TextColumn("Plan"),
                    "threshold_min": st.column_config.NumberColumn(
                        "Limit", format="%d min", width="small"),
                    "stations": count_col("Stations", width="small"),
                    "demand_covered_pct": pct_col("Demand covered"),
                    "zones_covered": count_col("Zones", width="small"),
                },
            )


# ===========================================================================
# Dispatch
# ===========================================================================
with tabs[3]:
    res = st.session_state.get("cover")
    if res is None or res.n_stations == 0:
        st.info("Solve the covering model in **Station placement** first — its "
                "stations become the ambulance bases this tab dispatches from.",
                icon=":material/arrow_back:")
    else:
        st.caption(
            f"Ambulance bases: **{res.n_stations} stations** "
            f"({res.mode} at {res.threshold_min:g} min)."
        )

        busiest = stats.busiest_hours_table(sel, top_n=25)

        c1, c2 = st.columns([3, 2], vertical_alignment="top")
        with c1:
            with st.container(border=True):
                st.markdown("**Choose the scenario**")
                options = list(busiest["hour_start"])
                chosen_hour = st.selectbox(
                    "Peak hour of simultaneous emergencies",
                    options,
                    format_func=lambda t: (f"{pd.Timestamp(t):%a %d %b %Y, %H:%M} — "
                                           f"{int(busiest.loc[busiest['hour_start'] == t, 'incidents'].iloc[0])} incidents"),
                )
                # The deck's setup is one ready unit per base, so that stays the
                # default -- but the range has to reach a balanced fleet, or the
                # model's own "each incident -> exactly one ambulance" is
                # unreachable for a busy hour and most calls go unserved. The
                # bound is a fixed 12 rather than one derived from the current
                # station count: a slider whose max moves is a different widget
                # each time, and Streamlit would silently reset it whenever the
                # plan changed.
                n_calls = int(busiest.loc[busiest["hour_start"] == chosen_hour,
                                          "incidents"].iloc[0])
                need = -(-n_calls // max(res.n_stations, 1))  # ceil division
                units = st.slider("Ready units per station", 1, 12, 1,
                                  key="units_per_station")
                if need > 1:
                    st.caption(
                        f"↳ {need} units per station ({need * res.n_stations} in total) "
                        "would field every call this hour — below that, the surplus "
                        "calls wait."
                    )
        with c2:
            with st.container(border=True):
                st.markdown("**Busiest hours in this period**")
                st.dataframe(
                    busiest, hide_index=True, width="stretch",
                    height=tbl_height(len(busiest), 6),
                    column_order=["rank", "hour_start", "weekday", "incidents"],
                    column_config={
                        "rank": rank_col(),
                        "hour_start": st.column_config.DatetimeColumn(
                            "Hour", format="DD MMM YYYY, HH:mm"),
                        "weekday": st.column_config.TextColumn("Day", width="small"),
                        "incidents": count_col("Incidents", width="small"),
                    },
                )

        hour_inc = data_loader.incidents_in_hour(sel, chosen_hour)
        ambulances = asg.ambulances_from_stations(res.chosen, units)
        gap = max(len(hour_inc) - len(ambulances), 0)

        k = kpi_row()
        kpi(k, "Incidents this hour", f"{len(hour_inc)}", icon=":material/ambulance:")
        kpi(k, "Ambulances available", f"{len(ambulances)}",
            icon=":material/local_shipping:")
        kpi(k, "Capacity gap", f"{gap}", icon=":material/balance:",
            help_text="Calls with no unit left to send, before the model is solved.")
        kpi(k, "Response-time limit", f"{threshold:g} min", icon=":material/speed:")

        if st.button("Solve the assignment model", type="primary",
                     icon=":material/route:"):
            with st.spinner("Solving…"):
                st.session_state["asg"] = asg.solve_assignment(ambulances, hour_inc, tk)
                st.session_state["asg_amb"] = ambulances

        a = st.session_state.get("asg")
        if a is not None:
            st.subheader("Dispatch result", divider="gray")
            k = kpi_row()
            kpi(k, "Total travel time", f"{a.total_time_min:.1f} min",
                icon=":material/timer:")
            kpi(k, "Mean response", f"{a.mean_time_min:.1f} min", icon=":material/speed:")
            kpi(k, "Worst response", f"{a.max_time_min:.1f} min",
                icon=":material/priority_high:")
            kpi(k, "Served / unserved", f"{a.n_served} / {a.n_unserved}",
                icon=":material/fact_check:")
            kpi(k, "Idle units", f"{a.n_idle}", icon=":material/directions_car:")

            # Eight columns of incident detail do not fit beside a chart: in a
            # two-thirds column the unit and response time -- the two the model
            # actually decides -- are the ones pushed off the right edge. The
            # plan gets the full width and the charts sit below it.
            # The model writes "UNSERVED (no unit available)" into the unit
            # field, which is the widest string in the table and repeats on
            # every unserved row; the blank response time beside it already
            # says the same thing.
            plan = a.assignments.assign(ambulance=np.where(
                a.assignments["served"], a.assignments["ambulance"], "Unserved"))

            with st.container(border=True):
                st.markdown("**Dispatch plan**")
                st.dataframe(
                    plan, hide_index=True, width="stretch",
                    height=tbl_height(len(plan), 10),
                    column_order=["incident_number", "time", "type", "zip_code",
                                  "address", "ambulance", "base_zone", "response_min"],
                    column_config={
                        "incident_number": st.column_config.TextColumn(
                            "Incident", width="small"),
                        "time": st.column_config.DatetimeColumn(
                            "Time", format="HH:mm", width="small"),
                        "type": st.column_config.TextColumn("Type"),
                        "zip_code": st.column_config.TextColumn("ZIP", width="small"),
                        "address": st.column_config.TextColumn("Address", width="medium"),
                        "ambulance": st.column_config.TextColumn("Unit"),
                        "base_zone": st.column_config.TextColumn(
                            "From zone", width="small"),
                        "response_min": min_col("Response", width="small"),
                    },
                )

            c1, c2 = st.columns([3, 2], vertical_alignment="top")
            with c1:
                with st.container(border=True):
                    st.plotly_chart(
                        viz.chart_response_distribution(a.assignments, threshold),
                        width="stretch")
            with c2:
                with st.container(border=True):
                    st.plotly_chart(
                        viz.chart_dispatch_comparison(a.benchmark_lp_min or 0.0,
                                                      a.greedy_total_min or 0.0),
                        width="stretch")
                    if a.greedy_total_min:
                        st.metric("Improvement over nearest-unit-first dispatch",
                                  f"{a.improvement_pct:.1f}%", border=False)

            with st.expander("Cost matrix (travel time in minutes)",
                              icon=":material/calculate:"):
                st.markdown("**Real ambulances × real incidents**")
                st.dataframe(a.cost_matrix, width="stretch")
                if a.dummy_kind != "none":
                    st.markdown(f"**Balanced matrix — padded with {a.dummy_kind}**")
                    st.dataframe(a.balanced_matrix, width="stretch")

            with st.container(border=True):
                st.markdown("**Dispatch plan on the map**")
                st.caption("Lines run from each ambulance base to the incident it was "
                           "sent to. Red dots are calls no unit could reach.")
                st_folium(viz.map_dispatch(a, st.session_state["asg_amb"]), height=MAP_H,
                          width="stretch", returned_objects=[], key="map_dispatch")
