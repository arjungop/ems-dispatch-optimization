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


def fitted_height(n_rows: int, px: int) -> int:
    """A table height that ends on a row boundary at or just under ``px``.

    For pairing a table against something whose height is a plain number -- a
    Folium map, which streamlit-folium takes in pixels -- rather than against
    a figure.
    """
    return tbl_height(n_rows, max(3, (px - 2) // ROW_PX - 1))


# The vertical space a bold `st.markdown` caption line takes inside a card.
# Where the table card carries one and the chart card does not -- the chart's
# own title lives inside the figure -- the table has that much less room if the
# two cards are to end together.
CARD_LABEL_PX = 38


def paired_height(n_rows: int, chart, label: bool = False) -> int:
    """A table height that lands on the same line as the chart beside it.

    Almost every section here is a chart in one card next to its table in
    another, and the pair only reads as a single row if both cards end
    together. Capping the table at an arbitrary row count does not do that:
    the 15-bar ranked chart is 500px tall while ``tbl_height(15, 15)`` is 530,
    and a 7-row result table beside a 10-row one leaves a 99px step. Deriving
    the cap from the figure's own height keeps the two within one row pitch of
    each other whatever the data does.
    """
    px = viz.chart_height(chart) - (CARD_LABEL_PX if label else 0)
    return fitted_height(n_rows, px)


def count_col(label: str, **kw):
    return st.column_config.NumberColumn(label, format="localized", **kw)


def pct_col(label: str, **kw):
    return st.column_config.NumberColumn(label, format="%.1f%%", **kw)


def min_col(label: str, **kw):
    return st.column_config.NumberColumn(label, format="%.1f min", **kw)


def km_col(label: str, **kw):
    return st.column_config.NumberColumn(label, format="%.2f km", **kw)


def fmt_seconds(s: float) -> str:
    """A solve time that stays a measurement at every scale.

    These models solve in single-digit milliseconds on a laptop, and a fixed
    ``%.2f`` renders that as "0.00s" -- which reads as a stopwatch that never
    started rather than as the headline that an exact ILP over the whole city
    is instant.
    """
    if s < 1.0:
        return f"{s * 1000:.0f} ms"
    return f"{s:.2f} s"


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


# A tile wide enough for the longest label the app puts on one ("Response-time
# limit", "Served / unserved") and short enough that five fit the content area
# at a 13-inch laptop width. Fixing it is what stops a wrapped tile from
# stretching -- see ``kpi_row``.
KPI_W = 196


def kpi(col, label: str, value: str, icon: str | None = None, border: bool = True,
        help_text: str | None = None, width: int | str = KPI_W):
    # Metrics are their own cards, except where they already sit inside one --
    # a bordered card nested in a bordered card reads as clutter.
    col.metric(label, value, icon=icon, border=border, help=help_text,
               width=width)


def kpi_row():
    """A metric row that wraps rather than squeezes.

    ``st.columns(5)`` splits the page into five equal cells whatever the window
    is doing, and at laptop widths the cells are narrower than the content, so
    Streamlit ellipsises the tiles -- "EMS incidents / 79,152" renders as
    "EMS inci... / 79,...". A horizontal container keeps each tile at its
    natural size and moves the overflow onto a second line instead.

    The tiles carry a fixed ``KPI_W`` rather than the default ``width="stretch"``
    because stretch is what makes wrapping look broken: below about 1400px the
    fifth tile drops to its own line and then expands to the full page width,
    so "Peak hour / 13:00" renders as a banner three times the size of the four
    tiles above it. At a fixed width the overflow row just starts a new, evenly
    sized rank.
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
    TOP_N = 15
    top_zones = zone_stats.head(TOP_N)
    # The cumulative share is the headline the column was there to deliver --
    # "the top 15 zones carry 80% of the city's demand" -- and as a column it
    # was the first thing to fall off the right edge of a narrow window. Said
    # once in the caption it is both safe and easier to read than 15 running
    # totals, and it buys the four remaining columns enough width to survive a
    # 13-inch laptop.
    top_share = float(top_zones["share_pct"].sum())
    fig_zones = viz.chart_busiest_zones(zone_stats, top_n=TOP_N)

    c1, c2 = st.columns([3, 2], vertical_alignment="top")
    with c1:
        with st.container(border=True):
            st.plotly_chart(fig_zones, width="stretch")
    with c2:
        with st.container(border=True):
            st.dataframe(
                top_zones,
                hide_index=True, width="stretch",
                height=paired_height(len(top_zones), fig_zones),
                # Three columns. In the two-fifths card this sits in -- 255px
                # at a 13-inch laptop width -- a fourth does not fit, and the
                # rank is the one worth losing: the rows are already in rank
                # order beside a chart that is itself a ranking.
                column_order=["zip_code", "incidents", "share_pct"],
                column_config={
                    "zip_code": st.column_config.TextColumn("ZIP", width="small"),
                    "incidents": count_col("Incidents"),
                    "share_pct": pct_col("Share"),
                },
            )
    st.caption(
        f"The {len(top_zones)} busiest ZIP zones carry **{top_share:.1f}%** of all "
        f"{len(sel):,} EMS incidents in this period — demand is concentrated enough "
        "that placing stations against it beats spreading them evenly."
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
    # Even weighting rather than [2, 3]: three columns including a progress bar
    # need about 300px and the two-fifths split left the table 255.
    c1, c2 = st.columns([1, 1], vertical_alignment="top")
    with c1:
        with st.container(border=True):
            st.markdown("**Weight per zone**")
            # Matched to the map's caption so both cards carry a header and one
            # caption line above their body and therefore end together.
            st.caption("Every zone enters the covering model weighted by its "
                       "incident count; the bar is that weight relative to the "
                       "busiest zone.")
            # The rank column is dropped rather than squeezed: the rows are
            # already in rank order, and at a laptop width the four columns
            # left the progress bar with no room to draw.
            st.dataframe(
                weights_tbl, hide_index=True, width="stretch",
                height=fitted_height(len(weights_tbl), MAP_H),
                column_order=["zip_code", "demand", "weight_pct"],
                column_config={
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

    # Saaty's verbal anchors. The slider renders its position as a bare number
    # and draws no labels at either end, so on screen a judgement is the digit
    # "3" on a track with no scale and no direction -- unreadable as the
    # intensity *and* the direction of a preference, which is the whole content
    # of an AHP comparison. Each slider states its reading in words underneath.
    SAATY_WORDS = {
        1: "equally important", 2: "slightly", 3: "moderately",
        4: "moderately to strongly", 5: "strongly",
        6: "strongly to very strongly", 7: "very strongly",
        8: "very strongly to extremely", 9: "extremely",
    }

    def saaty_phrase(v: float, a: str, b: str) -> str:
        """The slider's position, read back as the sentence it encodes."""
        if abs(v - 1.0) < 1e-9:
            return f"**{a}** and **{b}** matter equally"
        hi, lo, k = (a, b, v) if v > 1 else (b, a, 1.0 / v)
        return f"**{hi}** is {SAATY_WORDS[round(k)]} more important than **{lo}**"

    with st.container(border=True):
        st.markdown("**Pairwise judgements**")
        # A number input left to its own devices stretches the full width of a
        # wide-layout page, which makes a one-digit control look like a text
        # field; it is held to a readable width here instead.
        head_l, head_r = st.columns([1, 3], vertical_alignment="bottom")
        n_members = head_l.number_input(
            "Team members", 1, 8, len(config.TEAM), 1,
            help="Each member fills the comparison table independently; their "
                 "matrices are aggregated into one set of weights.",
        )
        head_r.caption(
            "**Saaty's 1–9 scale** —  1 equal · 3 moderate · 5 strong · "
            "7 very strong · 9 extreme, with 2, 4, 6, 8 in between. A reciprocal "
            "(1/3, 1/5 …) points the same intensity the other way."
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
                    cols[pi].caption(
                        f"1/9 ◄ {LBL[val]} ► 9 — {saaty_phrase(val, a, b)}")
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

            # Shown open rather than behind an expander. "Did every member's
            # own matrix pass CR < 0.10, or is one of them carrying the
            # aggregate?" is the first question asked of a group AHP, and the
            # aggregated CR above cannot answer it -- a geometric mean of four
            # inconsistent matrices can still come out consistent. It is also
            # what balances this card against the matrix card beside it.
            st.markdown("**Per-member consistency**")
            # Three columns, not six. The per-member criterion weights were
            # also here, and six columns in a two-fifths card cut the last one
            # off mid-header ("Cov…") with its values clipped beside it. The
            # question this table exists to answer is whether each member's own
            # matrix passes, so it keeps the CR and the verdict; the weights
            # that actually drive the model are the aggregated ones charted in
            # the card to the left.
            per_member_tbl = pd.DataFrame([
                {"Member": member_labels[p["member"] - 1],
                 "CR": p["CR"], "Consistent": bool(p["consistent"])}
                for p in result.per_member
            ])
            st.dataframe(
                per_member_tbl, hide_index=True, width="stretch",
                height=tbl_height(len(per_member_tbl), 8),
                column_config={
                    "Member": st.column_config.TextColumn(width="small"),
                    "CR": st.column_config.NumberColumn(
                        format="%.4f", width="small",
                        help="Consistency Ratio for this member's own matrix."),
                    "Consistent": st.column_config.CheckboxColumn(
                        "Passes", width="small",
                        help="CR below 0.10 — this member's matrix is consistent "
                             "on its own, not merely after aggregation."),
                },
            )

    st.subheader("Candidate sites ranked by priority", divider="gray")
    scored = ahp_mod.score_sites(zones, existing_stations, result, threshold, **tk)
    st.session_state["ahp_result"] = result
    st.session_state["ahp_scored"] = scored

    fig_rank = viz.chart_ahp_ranking(scored)
    c1, c2 = st.columns([2, 3], vertical_alignment="top")
    with c1:
        with st.container(border=True):
            st.plotly_chart(fig_rank, width="stretch")
    with c2:
        with st.container(border=True):
            st.markdown("**Criterion values behind each score**")
            # The rank column is gone: the rows are in rank order and the chart
            # beside them is a ranking, so it was a column of width spent
            # restating the row position -- width the three criterion values
            # need to stay on screen at a laptop size.
            st.dataframe(
                scored, hide_index=True, width="stretch",
                height=paired_height(len(scored), fig_rank, label=True),
                # Five columns, not six: "zones_reached" is a second reading of
                # Coverage Potential (the zones behind the demand already in
                # the column next to it), and at a laptop width it was the
                # column that pushed the priority score itself off the edge.
                column_order=["zip_code", "Incident Density", "Equity Gap",
                              "Coverage Potential", "ahp_score"],
                column_config={
                    "zip_code": st.column_config.TextColumn("ZIP", width="small"),
                    "Incident Density": count_col(
                        "Incidents", help="Emergencies in the zone."),
                    "Equity Gap": km_col(
                        "Nearest stn",
                        help="Distance to the closest station Seattle already runs."),
                    "Coverage Potential": count_col(
                        "In reach",
                        help="Incidents reachable from here inside the limit."),
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
        kpi(k, "Solve time", fmt_seconds(res.solve_seconds),
            icon=":material/timer:",
            help_text="Wall-clock time in the ILP solver for this run.")

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

        # Both result tables are given the same number of visible rows, set by
        # whichever has fewer. Sized independently they step against each other
        # -- seven opened stations beside ten zones left a 99px ledge between
        # the two cards -- and padding the shorter one to match would fill it
        # with blank filler rows instead, which reads as an unfinished table.
        pair_rows = max(5, min(len(res.chosen), len(zone_table), 10))
        pair_px = tbl_height(pair_rows, pair_rows)

        c1, c2 = st.columns([2, 3], vertical_alignment="top")
        with c1:
            with st.container(border=True):
                st.markdown("**Stations opened**")
                # The latitude and longitude that used to be here could not fit
                # this card -- it is 255px wide at a 13-inch laptop size, and a
                # readable coordinate pair needs 150 of that on its own. They
                # are also the least useful thing to put in it: the map
                # directly below plots every one of these stations. The column
                # that replaces them answers the question the table is actually
                # asked, which is what each opened station is for.
                workload = (zone_table[zone_table["covered"]]
                            .groupby("nearest_station_zone", observed=True)["demand"]
                            .sum())
                chosen = res.chosen.assign(serves=[
                    int(workload.get(z, 0)) for z in res.chosen["zip_code"]])
                st.dataframe(
                    chosen, hide_index=True, width="stretch", height=pair_px,
                    column_order=["station", "zip_code", "serves"],
                    column_config={
                        "station": st.column_config.TextColumn("Station", width="small"),
                        "zip_code": st.column_config.TextColumn("Zone", width="small"),
                        "serves": count_col(
                            "Serves",
                            help="Incidents in the zones this station is the "
                                 "nearest opened station for, inside the limit."),
                    },
                )
        with c2:
            with st.container(border=True):
                st.markdown("**Per-zone outcome**")
                st.dataframe(
                    zone_table, hide_index=True, width="stretch", height=pair_px,
                    column_order=["zip_code", "demand", "status",
                                  "nearest_station_zone", "response_min"],
                    column_config={
                        "zip_code": st.column_config.TextColumn("ZIP", width="small"),
                        "demand": count_col("Incidents"),
                        "status": st.column_config.TextColumn("Status", width="small"),
                        "nearest_station_zone": st.column_config.TextColumn(
                            "From", width="small",
                            help="The opened station this zone is served from."),
                        "response_min": min_col("Response", width="small"),
                    },
                )

        with st.container(border=True):
            st.markdown("**Stations & coverage on the map**")
            st.caption(f"Shaded circles are the {res.threshold_min:g}-minute reach of "
                       "each opened station. Zone dots are green where covered, red "
                       "where not.")
            st_folium(viz.map_coverage(res, existing_stations, True, **tk), height=MAP_H,
                      width="stretch", returned_objects=[], key="map_cover")

    st.subheader("Sensitivity analysis", divider="gray")
    with st.container(border=True):
        st.markdown("**What the plan costs at other promises and other budgets**")
        st.caption(
            f"Two sweeps over the same candidate pool: the covering model re-solved at "
            f"{', '.join(f'{t:g}' for t in config.RESPONSE_THRESHOLDS_MIN)} minutes, and "
            f"the maximal-covering model re-solved at every budget from one station "
            f"upwards at the current {threshold:g}-minute limit."
        )
        if st.button("Run both sweeps", icon=":material/query_stats:",
                     key="btn_sweep"):
            with st.spinner("Re-solving at each limit and each budget…"):
                st.session_state["sweep"] = covering.threshold_sweep(
                    candidates, zones, config.RESPONSE_THRESHOLDS_MIN, cand_ahp, tk)
                st.session_state["grid"] = covering.assumption_grid(
                    candidates, zones, threshold, cand_ahp,
                    turnout_min=st.session_state["turnout"])
                # Carried only as far as the budget that first reaches full
                # coverage, plus a couple of stations to show the curve has
                # flattened: every further P re-solves an ILP for a flat line.
                sc = st.session_state.get("cover")
                knee = sc.n_stations if sc is not None and sc.n_stations else 8
                st.session_state["curve"] = covering.marginal_coverage_curve(
                    candidates, zones, threshold,
                    min(len(candidates), max(4, knee + 3)), tk)
                st.session_state["curve_knee"] = knee

        sweep = st.session_state.get("sweep")
        curve = st.session_state.get("curve")
        if sweep is not None:
            c1, c2 = st.columns(2, vertical_alignment="top")
            with c1:
                st.plotly_chart(viz.chart_threshold_stations(sweep), width="stretch")
            with c2:
                # The coverage-per-limit chart that used to sit here drew three
                # bars at 100.0%, 100.0%, 100.0% -- a set-covering model reaches
                # full coverage at every limit by construction, so the chart
                # could only ever say so three times. Coverage against *budget*
                # is the question the bars were being asked to answer, and it
                # has a real shape: the diminishing return that justifies the
                # maximal-covering framing in the first place.
                if curve is not None:
                    st.plotly_chart(
                        viz.chart_marginal_curve(
                            curve, st.session_state.get("curve_knee")),
                        width="stretch")
            st.dataframe(
                sweep, hide_index=True, width="stretch",
                height=tbl_height(len(sweep), 6),
                column_order=["threshold_min", "stations_required",
                              "demand_covered_pct", "zones_covered"],
                column_config={
                    "threshold_min": st.column_config.NumberColumn(
                        "Response-time limit", format="%d min"),
                    "stations_required": count_col("Stations needed"),
                    "demand_covered_pct": pct_col("Demand covered"),
                    "zones_covered": count_col("Zones covered"),
                },
            )

    # ----------------------------------------------------------------------
    # How much of the answer is the data, and how much is the assumption
    # ----------------------------------------------------------------------
    # The response times here are computed, not measured -- the 911 feed
    # carries none -- so the station count is a function of two sliders as
    # much as of the city. Reporting it as a single number invites the only
    # question that can really wound this project ("isn't this just a
    # simulation?"); reporting the range answers it first.
    st.subheader("Robustness of the station count", divider="gray")
    with st.container(border=True):
        st.markdown("**The same city, 16 readings of the travel model**")
        st.caption(
            "Response times in this project are not observed — the Seattle feed records "
            "no response times — they are computed as *straight-line km × circuity ÷ "
            "speed + turnout*. The covering model is re-solved below at every "
            "combination of those two constants, so how much of the headline is the "
            "data and how much is the assumption is visible rather than implied."
        )
        grid = st.session_state.get("grid")
        if grid is None:
            st.info("Run the sweeps above to fill this in — the grid is solved alongside them.",
                    icon=":material/arrow_upward:")
        else:
            lo, hi = int(grid["stations_required"].min()), int(grid["stations_required"].max())
            worst_cover = float(grid["demand_covered_pct"].min())
            g1, g2 = st.columns([3, 2], vertical_alignment="top")
            with g1:
                st.plotly_chart(viz.chart_assumption_grid(grid), width="stretch")
            with g2:
                k = kpi_row()
                kpi(k, "Stations needed", f"{lo}–{hi}", icon=":material/location_on:",
                    help_text="Across all 16 travel assumptions, at the current limit.")
                kpi(k, "Worst coverage", f"{worst_cover:.1f}%",
                    icon=":material/shield:",
                    help_text="Lowest demand covered in any cell of the grid.")
                kpi(k, "Still under Seattle's", f"{len(existing_stations) - hi} fewer",
                    icon=":material/trending_down:",
                    help_text="Even the most pessimistic cell against the real "
                              f"{len(existing_stations)}-station network.")
            if hi < len(existing_stations):
                st.success(
                    f"**The count is assumption-dependent; the conclusion is not.** "
                    f"Full coverage needs **{lo} stations** under the optimistic reading "
                    f"and **{hi}** under the pessimistic one — a {hi / max(lo, 1):.1f}× "
                    f"spread, so *{lo}* should never be quoted on its own. In all "
                    f"{len(grid)} cells the optimised plan still reaches "
                    f"{worst_cover:.0f}% of demand from fewer than Seattle's "
                    f"{len(existing_stations)} stations.",
                    icon=":material/verified:",
                )
            else:
                st.warning(
                    f"Under the most pessimistic travel assumptions this plan needs "
                    f"**{hi}** stations, at or above Seattle's {len(existing_stations)}. "
                    "The placement advantage does not survive that reading.",
                    icon=":material/warning:",
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
            # Chart over table rather than beside it. Five columns in the
            # two-fifths column this used to sit in lost "Demand covered" to
            # an ellipsis and pushed "Zones covered" off the right edge
            # entirely -- on the one table in the app whose whole job is to
            # show that both plans cover the same demand.
            st.plotly_chart(viz.chart_comparison(rows), width="stretch")

            at_limit = bench[bench["threshold_min"] == threshold]
            if len(at_limit) == 2:
                ours = at_limit[at_limit["plan"] == "Optimised placement"].iloc[0]
                theirs = at_limit[at_limit["plan"] != "Optimised placement"].iloc[0]
                st.success(
                    f"At the {threshold:g}-minute limit the optimised plan covers "
                    f"**{ours['demand_covered_pct']:.1f}%** of demand from "
                    f"**{int(ours['stations'])} stations**, against "
                    f"**{theirs['demand_covered_pct']:.1f}%** from Seattle's "
                    f"**{int(theirs['stations'])}** — the same coverage from "
                    f"{int(theirs['stations']) / max(int(ours['stations']), 1):.1f}× "
                    "fewer sites, because placement follows demand rather than history.",
                    icon=":material/insights:",
                )
                # Stated here rather than left for a reviewer to raise. The
                # comparison is real but it is not like-for-like, and the
                # ratio above is the most quotable number in the app.
                st.caption(
                    f":material/warning: **Read this ratio with two caveats.** Seattle's "
                    f"{int(theirs['stations'])} are *fire* stations — they also carry "
                    "suppression, hazmat and technical rescue, missions this model does "
                    "not place for, so a purpose-built EMS network should need fewer "
                    "sites and the gap overstates any inefficiency on their part. And "
                    f"{int(ours['stations'])} is the count under the current travel "
                    "assumptions; see *Robustness of the station count* above for how "
                    "far it moves. The defensible claim is directional — demand-driven "
                    "placement needs fewer sites than historical placement — not the "
                    "specific multiple."
                )
            st.dataframe(
                bench, hide_index=True, width="stretch",
                height=tbl_height(len(bench), 6),
                column_order=["plan", "threshold_min", "stations",
                              "demand_covered_pct", "zones_covered"],
                column_config={
                    "plan": st.column_config.TextColumn("Plan", width="medium"),
                    "threshold_min": st.column_config.NumberColumn(
                        "Response-time limit", format="%d min"),
                    "stations": count_col("Stations"),
                    "demand_covered_pct": pct_col("Demand covered"),
                    "zones_covered": count_col("Zones covered"),
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
                # The fleet opens at the size that can actually field the hour,
                # not at one unit per base. One unit per base puts 7 ambulances
                # against a 30-call hour, and the assignment LP then answers a
                # question nobody asked: with no units left, 23 of the 30 calls
                # come back "Unserved", the dispatch plan is mostly blank, and
                # the response histogram has seven bars in it -- the model reads
                # as broken when it is merely starved. Worse, an LP free to
                # leave calls unserved minimises by serving the *cheapest* ones,
                # so the under-resourced run is also the least honest one (see
                # ``assignment.benchmark_vs_greedy``). Scarcity is still one
                # drag of the slider away, and the capacity-gap tile below
                # reports it either way.
                #
                # The bound is a fixed 12 rather than one derived from the
                # current station count: a slider whose max moves is a different
                # widget each time, and Streamlit would silently reset it
                # whenever the plan changed.
                n_calls = int(busiest.loc[busiest["hour_start"] == chosen_hour,
                                          "incidents"].iloc[0])
                need = -(-n_calls // max(res.n_stations, 1))  # ceil division
                units = st.slider("Ready units per station", 1, 12,
                                  min(12, max(1, need)), key="units_per_station")
                fleet = units * res.n_stations
                if fleet >= n_calls:
                    st.caption(
                        f"↳ {fleet} ambulances for {n_calls} calls — every call can "
                        "be fielded, so the LP is choosing *which* unit goes where."
                    )
                else:
                    st.caption(
                        f"↳ {fleet} ambulances for {n_calls} calls — "
                        f"{n_calls - fleet} will wait. {need} units per station "
                        f"({need * res.n_stations} in total) would field them all."
                    )
        with c2:
            with st.container(border=True):
                st.markdown("**Busiest hours in this period**")
                # The weekday had its own column, which at this width clipped
                # "Wednesday" to "Wednesda". It is one token inside the
                # timestamp beside it, so it moves into the format string.
                st.dataframe(
                    busiest, hide_index=True, width="stretch",
                    height=tbl_height(len(busiest), 5),
                    # The rank column goes for the same reason as everywhere
                    # else -- the rows are in rank order -- and here it is what
                    # the full timestamp needs in order to render whole.
                    column_order=["hour_start", "incidents"],
                    column_config={
                        "hour_start": st.column_config.DatetimeColumn(
                            "Hour", format="ddd DD MMM YYYY, HH:mm"),
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
                icon=":material/timer:",
                help_text="Summed over the calls this plan served. The benchmark "
                          "chart below uses a different, like-for-like workload — "
                          "see the note under it.")
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
            # An unserved row has no response time, and a NumberColumn renders
            # that missing value as the literal string "None" -- Python's
            # repr leaking into a dispatch sheet. Formatted here instead, so
            # the absence reads as an em dash.
            plan = a.assignments.assign(
                ambulance=np.where(a.assignments["served"],
                                   a.assignments["ambulance"], "Unserved"),
                response_txt=np.where(
                    a.assignments["served"],
                    a.assignments["response_min"].map(
                        lambda v: f"{v:.1f} min" if pd.notna(v) else "—"),
                    "—"),
            )

            with st.container(border=True):
                st.markdown("**Dispatch plan**")
                st.dataframe(
                    plan, hide_index=True, width="stretch",
                    height=tbl_height(len(plan), 10),
                    # Six columns at 150/75px apiece = 600px, inside the 700px
                    # this card gets at a 13-inch laptop width. Eight columns
                    # came to 750px and Streamlit answered by dropping the
                    # rightmost ones from the grid entirely -- silently, and
                    # starting with the response time, the single number the
                    # model exists to produce. The two that go are the ZIP
                    # (the address beside it already locates the call) and the
                    # call type (the LP minimises travel time regardless of it;
                    # the call mix is on the Data tab).
                    column_order=["incident_number", "time", "address",
                                  "ambulance", "base_zone", "response_txt"],
                    column_config={
                        "incident_number": st.column_config.TextColumn(
                            "Incident", width="medium"),
                        "time": st.column_config.DatetimeColumn(
                            "Time", format="HH:mm", width="small"),
                        "address": st.column_config.TextColumn("Address", width="medium"),
                        "ambulance": st.column_config.TextColumn("Unit", width="small"),
                        "base_zone": st.column_config.TextColumn(
                            "From", width="small",
                            help="Zone of the station the unit was sent from."),
                        "response_txt": st.column_config.TextColumn(
                            "Response", width="small", alignment="right"),
                    },
                )

            # The comparison card carries a chart, a metric and an explanatory
            # note; the distribution card carries only a chart. Left at the
            # same figure height the two cards ended 125px apart, so the taller
            # card's content sets the height of the shorter card's chart.
            c1, c2 = st.columns([3, 2], vertical_alignment="top")
            with c1:
                with st.container(border=True):
                    st.plotly_chart(
                        viz.chart_response_distribution(a.assignments, threshold,
                                                        height=430),
                        width="stretch")
            with c2:
                with st.container(border=True):
                    st.plotly_chart(
                        viz.chart_dispatch_comparison(a.benchmark_lp_min or 0.0,
                                                      a.greedy_total_min or 0.0,
                                                      a.benchmark_n_incidents),
                        width="stretch")
                    if a.greedy_total_min:
                        st.metric("Saved vs nearest-unit-first",
                                  f"{a.improvement_pct:.1f}%", border=False,
                                  width="stretch",
                                  help="Share of total fleet travel time the "
                                       "optimal matching removes, on the same "
                                       "calls as the greedy baseline.")
            # Without this the tab can show two different totals for the same
            # hour under the same words -- the tile above reports the plan's own
            # served calls, the chart the first k calls in arrival order -- and
            # a reader is left to assume one of them is wrong. Full width,
            # below both cards, so it does not unbalance the row it explains.
            st.caption(
                f"Both bars above are scored on the same workload: the first "
                f"**{a.benchmark_n_incidents}** calls of the hour in arrival order, "
                "which is what a dispatcher actually has to clear. Restricting both "
                "sides this way measures the quality of the matching rather than "
                "which calls each method chose to answer — so when the fleet is too "
                "small to field the whole hour, this total is a different quantity "
                "from the *Total travel time* tile above."
            )

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
