"""Render every figure the IEEE paper uses, as vector PDF.

The figures are drawn from the same models and the same colour tokens the
Streamlit app uses (``src/viz.py``), so the paper and the application read as
one system rather than two separate design efforts.

Colour policy, inherited from src/viz.py:
  * categorical identity uses a fixed slot order -- slot 1 is always "our
    optimised result", slot 2 always "the existing / benchmark result";
  * magnitude uses the single-hue light->dark blue ramp, never a rainbow;
  * text wears ink tokens, never the series colour.

Run:  python paper/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon as MplPolygon
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import ahp as ahp_mod  # noqa: E402
from src import assignment as asg  # noqa: E402
from src import config, covering, data_loader, stats  # noqa: E402

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

TK = dict(speed_kmh=config.DEFAULT_SPEED_KMH,
          circuity=config.DEFAULT_CIRCUITY,
          turnout_min=config.DEFAULT_TURNOUT_MIN)

# --- tokens, mirrored from src/viz.py --------------------------------------
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # categorical slots
SEQ_BLUE = ["#e8f1fb", "#c3dbf4", "#98c0ec", "#6aa2e0", "#3f85d3",
            "#2a78d6", "#1d5da6", "#133f73"]
INK_PRIMARY, INK_SECOND, INK_MUTED = "#0b0b0b", "#52514e", "#8a8985"
GRID, SURFACE = "#e4e3df", "#ffffff"
RAMP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

# IEEE column geometry.
COL, DBL = 3.45, 7.16

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK_SECOND,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_SECOND,
    "ytick.color": INK_SECOND,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def recede(ax, xgrid=False, ygrid=True):
    """Recessive axes: the data is the figure, the frame is scaffolding."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(length=2.5)


def save(fig, name):
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print(f"  {name}")


# ===========================================================================
# Rebuild the pipeline so every figure is drawn from the live models
# ===========================================================================
print("solving models…")
incidents = data_loader.load_incidents()
existing = data_loader.load_existing_stations()
end = incidents["datetime"].max().date()
start = (pd.Timestamp(end) - pd.DateOffset(years=1)).date()
sel = data_loader.filter_period(incidents, start, end, include_trauma=False)
all_zones = data_loader.build_zones(sel)
zones = data_loader.material_zones(all_zones)

ahp_res = ahp_mod.run_ahp([config.judgements_for(n) for n, _ in config.TEAM])
scored = ahp_mod.score_sites(zones, existing, ahp_res, 8.0, **TK)
cands = scored[["zip_code", "latitude", "longitude"]].copy()
ascore = scored["ahp_score"].to_numpy()

cov8 = covering.solve_set_covering(cands, zones, 8.0, ascore, TK)
sweep = covering.threshold_sweep(cands, zones, [5, 8, 10], ascore, TK)
curve = covering.marginal_coverage_curve(cands, zones, 8.0, 12, TK)
real_bench = {t: covering.evaluate_fixed_stations(existing, zones, t, TK)
              for t in (5, 8, 10)}

hour_start, _ = data_loader.peak_hour_window(sel)
peak = data_loader.incidents_in_hour(sel, hour_start)
disp = {}
FLEETS = (1, 2, 5)   # 5/station is what the app defaults to: the
                     # smallest fleet that can field the whole peak hour
for ups in FLEETS:
    amb = asg.ambulances_from_stations(cov8.chosen, units_per_station=ups)
    disp[ups] = asg.solve_assignment(amb, peak, TK)

print("drawing figures…")


# ===========================================================================
# Fig. 1 - pipeline architecture
# ===========================================================================
# Drawn rather than screenshotted so the contrast is controlled: every filled
# box pairs a dark ground with white text or a light ground with dark text.
# The earlier version of this diagram set white labels on a pale blue fill,
# which rendered as blank boxes in print.
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(DBL, 1.95))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

    DARK, LIGHT = "#1d2b3a", "#eaf1fd"

    def box(x, y, w, h, kicker, title, dark=True, edge=None):
        fc = DARK if dark else LIGHT
        ec = edge or (DARK if dark else S1)
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2",
            linewidth=0.9, facecolor=fc, edgecolor=ec))
        tc_k = "#9fb3c8" if dark else INK_MUTED
        tc_t = "#ffffff" if dark else INK_PRIMARY
        # matplotlib has no letter-spacing property, so the kicker is tracked
        # out by hand -- it is the one label that reads as an eyebrow rather
        # than as prose.
        ax.text(x + w / 2, y + h * 0.66, " ".join(kicker), ha="center",
                va="center", fontsize=5.6, color=tc_k, weight="bold")
        ax.text(x + w / 2, y + h * 0.30, title, ha="center", va="center",
                fontsize=7.4, color=tc_t, weight="bold")

    def arrow(x1, y1, x2, y2, color=INK_MUTED, style="-|>"):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=7,
            linewidth=0.9, color=color, shrinkA=0, shrinkB=0))

    box(1, 11, 15, 8, "DATA IN", "Seattle 911 feed")
    box(20, 11, 16, 8, "STEP 1", "Clean & zone")
    box(40, 11, 15, 8, "STEP 2", "AHP ranks sites")
    box(60, 19, 20, 8, "STEP 3 · ILP", "Covering model", dark=False)
    box(60, 3, 20, 8, "STEP 4 · LP", "Assignment model", dark=False)
    box(84, 11, 15, 8, "OUTPUT", "Decision maps")

    arrow(16.5, 15, 19.5, 15)
    arrow(36.5, 15, 39.5, 15)
    arrow(55.5, 15, 58, 15)
    arrow(58, 15, 58, 23); arrow(58, 23, 59.5, 23)
    # The linking arrow and its caption both have to sit inside the 60-80 band
    # the two model boxes occupy; pushed right they collide with the lines
    # routing both models into the output box.
    arrow(64.5, 18.3, 64.5, 11.9, color=S1)
    ax.text(66.0, 15.1, "stations become\nambulance bases", fontsize=5.6,
            color=S1, style="italic", ha="left", va="center")
    arrow(80.5, 23, 82, 23); arrow(82, 23, 82, 15); arrow(82, 15, 83.5, 15)
    arrow(80.5, 7, 82, 7); arrow(82, 7, 82, 15)
    save(fig, "fig1_pipeline.pdf")


# ===========================================================================
# Fig. 2 - demand concentration (two panels, never a dual axis)
# ===========================================================================
def fig_demand():
    bz = stats.busiest_zones(sel).head(12)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL, 3.5),
                                 gridspec_kw={"hspace": 0.55})

    # bz is already sorted largest-first, so reversing the y positions puts the
    # busiest zone at the top. Inverting the axis as well would flip it back.
    y = np.arange(len(bz))[::-1]
    a1.barh(y, bz["incidents"], height=0.62, color=S1, linewidth=0)
    a1.set_yticks(y); a1.set_yticklabels(bz["zip_code"].astype(str))
    a1.set_xlabel("EMS incidents in the study year")
    a1.set_title("(a) Demand by ZIP zone", loc="left", color=INK_PRIMARY)
    a1.set_xlim(0, bz["incidents"].max() * 1.18)
    for yi, v in zip(y, bz["incidents"]):
        a1.text(v + bz["incidents"].max() * 0.02, yi, f"{v:,}", va="center",
                fontsize=6.2, color=INK_SECOND)
    recede(a1, xgrid=True, ygrid=False)
    a1.set_xticks([])
    a1.spines["bottom"].set_visible(False)

    full = stats.busiest_zones(sel)
    a2.plot(np.arange(1, len(full) + 1), full["cumulative_pct"], color=S1,
            linewidth=1.6, solid_capstyle="round")
    a2.axhline(80, color=INK_MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    n80 = int((full["cumulative_pct"] < 80).sum()) + 1
    a2.plot([n80], [full["cumulative_pct"].iloc[n80 - 1]], "o", color=S1,
            markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.2)
    a2.annotate(f"{n80} zones carry 80%", (n80, full["cumulative_pct"].iloc[n80 - 1]),
                textcoords="offset points", xytext=(6, -9), fontsize=6.4,
                color=INK_SECOND)
    a2.set_xlabel("Zones, ranked by demand")
    a2.set_ylabel("Cumulative share (%)")
    a2.set_title("(b) Concentration of demand", loc="left", color=INK_PRIMARY)
    a2.set_ylim(0, 105); a2.set_xlim(1, len(full))
    recede(a2)
    save(fig, "fig2_demand.pdf")


# ===========================================================================
# Fig. 3 - temporal profile
# ===========================================================================
def fig_temporal():
    bh = stats.calls_by_hour(sel)
    bw = stats.calls_by_weekday(sel)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DBL, 1.85),
                                 gridspec_kw={"wspace": 0.22})

    peak_h = int(bh.loc[bh["incidents"].idxmax(), "hour"])
    colors = [S2 if int(h) == peak_h else S1 for h in bh["hour"]]
    a1.bar(bh["hour"], bh["incidents"], width=0.68, color=colors, linewidth=0)
    a1.set_xlabel("Hour of day"); a1.set_ylabel("EMS incidents")
    a1.set_title("(a) Calls by hour of day", loc="left", color=INK_PRIMARY)
    a1.set_xticks(range(0, 24, 3))
    a1.annotate(f"peak {peak_h:02d}:00", (peak_h, bh["incidents"].max()),
                textcoords="offset points", xytext=(4, 2), fontsize=6.4,
                color=S2, weight="bold")
    recede(a1)

    a2.bar(range(len(bw)), bw["incidents"], width=0.62, color=S1, linewidth=0)
    a2.set_xticks(range(len(bw)))
    a2.set_xticklabels([str(d)[:3] for d in bw.iloc[:, 0]])
    a2.set_xlabel("Day of week"); a2.set_ylabel("EMS incidents")
    a2.set_title("(b) Calls by weekday", loc="left", color=INK_PRIMARY)
    recede(a2)
    save(fig, "fig3_temporal.pdf")


# ===========================================================================
# Fig. 4 - optimised placement on the city
# ===========================================================================
def fig_map():
    gj = json.loads((ROOT / "data/raw/seattle_zcta.geojson").read_text())
    dem = dict(zip(zones["zip_code"].astype(str), zones["demand"]))
    vmax = max(dem.values())

    fig, ax = plt.subplots(figsize=(COL, 4.0))
    for feat in gj["features"]:
        z = str(feat["properties"].get("ZCTA5"))
        geom = feat["geometry"]
        polys = ([geom["coordinates"]] if geom["type"] == "Polygon"
                 else geom["coordinates"])
        for poly in polys:
            ring = np.array(poly[0])
            if z in dem:
                fc = RAMP(dem[z] / vmax * 0.92 + 0.08)
                ec, lw, a = "#ffffff", 0.5, 1.0
            else:
                fc, ec, lw, a = "#f4f4f1", "#ffffff", 0.4, 0.75
            ax.add_patch(MplPolygon(ring, closed=True, facecolor=fc,
                                    edgecolor=ec, linewidth=lw, alpha=a))

    # The choropleth already owns the blue ramp, so the markers must not also
    # be blue -- magnitude and identity would then share a hue and the opened
    # stations would disappear into the zones they sit on. The optimised sites
    # are drawn white-on-ink instead, which reads against every step of the
    # ramp: the dark outline carries on the pale zones, the white fill on the
    # dark ones.
    ax.scatter(existing["longitude"], existing["latitude"], s=13, marker="s",
               facecolor=S2, edgecolor="#ffffff", linewidth=0.5, zorder=4)
    ax.scatter(cov8.chosen["longitude"], cov8.chosen["latitude"], s=78,
               marker="*", facecolor="#ffffff", edgecolor=INK_PRIMARY,
               linewidth=0.9, zorder=5)

    ax.set_xlim(-122.44, -122.22); ax.set_ylim(47.48, 47.74)
    ax.set_aspect(1 / np.cos(np.radians(47.61)))
    ax.axis("off")
    # Below the map rather than inside it -- an inset legend sits on the
    # northern zones at this extent.
    ax.legend(handles=[
        Line2D([], [], marker="*", linestyle="none", markersize=10,
               markerfacecolor="#ffffff", markeredgecolor=INK_PRIMARY,
               label=f"Optimised ({cov8.n_stations})"),
        Line2D([], [], marker="s", linestyle="none", markersize=4.5,
               markerfacecolor=S2, markeredgecolor="#ffffff",
               label=f"Seattle Fire ({len(existing)})"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False,
        handletextpad=0.4, columnspacing=1.2, borderpad=0.0)

    sm = plt.cm.ScalarMappable(cmap=RAMP,
                               norm=plt.Normalize(vmin=0, vmax=vmax))
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01, shrink=0.52,
                      location="right")
    cb.set_label("EMS incidents per zone", fontsize=6.6, color=INK_SECOND)
    cb.ax.tick_params(labelsize=6, length=2, color=GRID)
    cb.outline.set_visible(False)
    save(fig, "fig4_map.pdf")


# ===========================================================================
# Fig. 5 - marginal coverage curve
# ===========================================================================
def fig_curve():
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    ax.plot(curve["stations"], curve["demand_covered_pct"], color=S1,
            linewidth=1.8, marker="o", markersize=3.4,
            markeredgecolor=SURFACE, markeredgewidth=0.9,
            solid_capstyle="round")
    p_full = int(curve.loc[curve["demand_covered_pct"] >= 99.999,
                           "stations"].min())
    ax.axvline(p_full, color=INK_MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.annotate(f"full coverage at P = {p_full}", (p_full, 52),
                textcoords="offset points", xytext=(5, 0), fontsize=6.4,
                color=INK_SECOND, va="center")
    ax.set_xlabel("Station budget $P$")
    ax.set_ylabel("Demand covered (%)")
    ax.set_ylim(40, 104)
    ax.set_xticks(curve["stations"][::2])
    recede(ax)
    save(fig, "fig5_curve.pdf")


# ===========================================================================
# Fig. 6 - threshold sweep vs the real network
# ===========================================================================
def fig_sweep():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DBL, 1.9),
                                 gridspec_kw={"wspace": 0.26})
    t = sweep["threshold_min"].astype(int)
    x = np.arange(len(t))

    a1.bar(x, sweep["stations_required"], width=0.5, color=S1, linewidth=0)
    for xi, v in zip(x, sweep["stations_required"]):
        a1.text(xi, v + 0.45, str(int(v)), ha="center", fontsize=7,
                color=INK_SECOND, weight="bold")
    a1.set_xticks(x); a1.set_xticklabels([f"{v} min" for v in t])
    a1.set_ylabel("Stations required")
    a1.set_xlabel("Response-time limit")
    a1.set_title("(a) Stations needed for full coverage", loc="left",
                 color=INK_PRIMARY)
    a1.set_ylim(0, max(sweep["stations_required"]) * 1.22)
    recede(a1)

    # Coverage differs between the two networks only in the second decimal, so
    # this panel has to zoom in -- and a bar chart may not. A truncated bar
    # baseline makes a 0.28-point gap look like a rout. Dots encode position
    # rather than length, so a non-zero axis is honest here; the connector
    # carries the size of the gap.
    ours = sweep["demand_covered_pct"].to_numpy()
    real = np.array([real_bench[int(v)].demand_pct for v in t])
    # The two series are offset onto their own half-rows. At 8 and 10 minutes
    # both networks reach exactly 100%, and drawn on a shared row the second
    # marker would sit precisely behind the first and read as missing.
    y = np.arange(len(t))[::-1]
    dy = 0.13
    for yi, o, r in zip(y, ours, real):
        a2.plot([min(o, r), max(o, r)], [yi + dy, yi - dy], color=GRID,
                linewidth=1.2, solid_capstyle="round", zorder=1)
    a2.scatter(real, y - dy, s=26, color=S2, zorder=3, edgecolor=SURFACE,
               linewidth=0.8, label=f"Seattle's {len(existing)} stations")
    a2.scatter(ours, y + dy, s=26, color=S1, zorder=3, edgecolor=SURFACE,
               linewidth=0.8, label="Optimised")
    # One label per row unless the two networks actually differ -- printing
    # "100.00%" twice on a row where both reach it is noise, and at this row
    # pitch the two labels collide.
    for yi, o, r in zip(y, ours, real):
        a2.text(o, yi + dy + 0.17, f"{o:.2f}%", ha="center", fontsize=6.2,
                color=INK_SECOND)
        if abs(o - r) > 5e-3:
            a2.text(r, yi - dy - 0.42, f"{r:.2f}%", ha="center", fontsize=6.2,
                    color=INK_SECOND)
    a2.set_yticks(y); a2.set_yticklabels([f"{v} min" for v in t])
    a2.set_xlabel("Demand covered (%)")
    a2.set_ylabel("Response-time limit")
    a2.set_title("(b) Coverage achieved", loc="left", color=INK_PRIMARY)
    a2.set_xlim(99.6, 100.12)
    a2.set_ylim(-0.75, len(t) - 0.25)
    a2.legend(frameon=False, loc="lower left", handlelength=0.8,
              handletextpad=0.3, borderpad=0.0, labelspacing=0.25)
    recede(a2, xgrid=True, ygrid=False)
    save(fig, "fig6_sweep.pdf")


# ===========================================================================
# Fig. 7 - dispatch: optimal vs greedy, like for like
# ===========================================================================
def fig_dispatch():
    fig, ax = plt.subplots(figsize=(COL, 2.05))
    labels, lp, gd = [], [], []
    for ups in FLEETS:
        r = disp[ups]
        labels.append(f"{r.n_ambulances} units\n({r.benchmark_n_incidents} calls)")
        lp.append(r.benchmark_lp_min)
        gd.append(r.greedy_total_min)

    # A hair of surface between the two fills, so the pair reads as two marks
    # rather than one two-tone block.
    x = np.arange(len(labels)); w = 0.32; off = w / 2 + 0.014
    ax.bar(x - off, lp, width=w, color=S1, linewidth=0,
           label="Assignment LP (optimal)")
    ax.bar(x + off, gd, width=w, color=S2, linewidth=0,
           label="Nearest-unit-first greedy")
    for xi, a, b in zip(x, lp, gd):
        ax.text(xi - off, a + 2, f"{a:.1f}", ha="center", fontsize=6.4,
                color=INK_SECOND)
        ax.text(xi + off, b + 2, f"{b:.1f}", ha="center", fontsize=6.4,
                color=INK_SECOND)
        ax.annotate(f"−{100 * (b - a) / b:.1f}%", (xi, max(a, b) + 13),
                    ha="center", fontsize=7, color=S1, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Total fleet travel time (min)")
    ax.set_ylim(0, max(gd) * 1.26)
    ax.legend(frameon=False, loc="upper left", handlelength=1.1,
              bbox_to_anchor=(0, 1.02))
    recede(ax)
    save(fig, "fig7_dispatch.pdf")


def fig_robustness():
    """Stations required across the travel assumptions.

    The one figure that is about the model's own uncertainty rather than its
    output. Response times here are computed, not observed, so the station
    count is a function of two constants; drawn as a grid, the reader can see
    at once how much of "seven stations" is Seattle and how much is the
    assumption. Single-hue ramp, every cell labelled, because the corner
    values are the ones that get quoted.
    """
    grid = covering.assumption_grid(cands, zones, 8.0, ascore)
    piv = grid.pivot(index="circuity", columns="speed_kmh",
                     values="stations_required")
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    im = ax.imshow(piv.to_numpy(), cmap=RAMP, aspect="auto")
    lo, hi = piv.to_numpy().min(), piv.to_numpy().max()
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = int(piv.to_numpy()[i, j])
            # White on the dark end of the ramp, ink on the light end, so the
            # label stays legible across the whole scale.
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=7.6,
                    color="#ffffff" if v > (lo + hi) / 2 else INK_PRIMARY)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([f"{r:g}" for r in piv.index])
    ax.set_xlabel("Effective speed (km/h)")
    ax.set_ylabel("Road circuity factor")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Stations for full coverage", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=0)
    cb.outline.set_visible(False)
    save(fig, "fig8_robustness.pdf")


for fn in (fig_pipeline, fig_demand, fig_temporal, fig_map, fig_curve,
           fig_sweep, fig_dispatch, fig_robustness):
    fn()

print(f"\n{len(list(OUT.glob('*.pdf')))} figures -> {OUT}")
