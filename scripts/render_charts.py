"""Render every chart against the real data into one static HTML page.

Used to eyeball the figures for label collisions, geometry and overflow --
the things a palette validator cannot check.

Run:  python scripts/render_charts.py [out.html]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import ahp as ahp_mod  # noqa: E402
from src import assignment as asg  # noqa: E402
from src import covering, data_loader, stats, viz  # noqa: E402

TK = dict(speed_kmh=45.0, circuity=1.30, turnout_min=1.0)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "charts_preview.html"

inc = data_loader.load_incidents()
existing = data_loader.load_existing_stations()
end = inc["datetime"].max().date()
start = (pd.Timestamp(end) - pd.DateOffset(years=1)).date()
sel = data_loader.filter_period(inc, start, end, False)
zones = data_loader.material_zones(data_loader.build_zones(sel))

ahp_res = ahp_mod.run_ahp([ahp_mod.DEFAULT_COMPARISONS])
scored = ahp_mod.score_sites(zones, existing, ahp_res, 8, **TK)
candidates = scored[["zip_code", "latitude", "longitude"]]
cov = covering.solve_set_covering(candidates, zones, 8, scored["ahp_score"].to_numpy(), TK)
sweep = covering.threshold_sweep(candidates, zones, [5, 8, 10],
                                 scored["ahp_score"].to_numpy(), TK)
curve = covering.marginal_coverage_curve(candidates, zones, 8, 10, TK)

bench = []
for t in [5, 8, 10]:
    ours = covering.solve_set_covering(candidates, zones, t, None, TK)
    theirs = covering.evaluate_fixed_stations(existing, zones, t, TK)
    bench.append({"plan": "Optimised placement", "threshold_min": t,
                  "demand_covered_pct": ours.demand_pct})
    bench.append({"plan": "Seattle's real stations", "threshold_min": t,
                  "demand_covered_pct": theirs.demand_pct})

peak_ts, _ = data_loader.peak_hour_window(sel)
hour_inc = data_loader.incidents_in_hour(sel, peak_ts)
amb = asg.ambulances_from_stations(cov.chosen, 2)
a = asg.solve_assignment(amb, hour_inc, TK)

figs = [
    ("Busiest zones", viz.chart_busiest_zones(stats.busiest_zones(sel))),
    ("Calls by hour", viz.chart_calls_by_hour(stats.calls_by_hour(sel))),
    ("Calls by weekday", viz.chart_calls_by_weekday(stats.calls_by_weekday(sel))),
    ("Weekday x hour heatmap", viz.chart_heatmap(stats.hour_weekday_heatmap(sel))),
    ("Daily trend", viz.chart_daily_trend(stats.daily_trend(sel))),
    ("Call type mix", viz.chart_call_types(stats.call_type_mix(sel))),
    ("AHP weights", viz.chart_ahp_weights(ahp_res.weights_series)),
    ("AHP site ranking", viz.chart_ahp_ranking(scored)),
    ("Stations per threshold", viz.chart_threshold_stations(sweep)),
    ("Coverage per threshold", viz.chart_threshold_coverage(sweep)),
    ("Marginal coverage curve", viz.chart_marginal_curve(curve)),
    ("Optimised vs real stations", viz.chart_comparison(bench)),
    ("Response distribution", viz.chart_response_distribution(a.assignments, 8)),
    ("LP vs greedy dispatch", viz.chart_dispatch_comparison(a.total_time_min,
                                                            a.greedy_total_min)),
]

parts = [
    "<!doctype html><meta charset='utf-8'>",
    "<title>EMS charts preview</title>",
    "<style>body{background:#fcfcfb;color:#0b0b0b;font:14px system-ui;margin:0;"
    "padding:24px}h2{font-size:14px;color:#52514e;margin:28px 0 6px;"
    "font-weight:600}.card{background:#fff;border:1px solid #e4e3df;"
    "border-radius:10px;padding:10px;margin-bottom:8px;overflow-x:auto}</style>",
    "<h1 style='font-size:20px'>Chart preview — real Seattle EMS data</h1>",
    f"<p style='color:#52514e'>Period {start} → {end}: {len(sel):,} EMS incidents, "
    f"{len(zones)} zones, {cov.n_stations} stations at 8 min.</p>",
]
for i, (name, fig) in enumerate(figs):
    parts.append(f"<h2>{i + 1}. {name}</h2><div class='card'>")
    parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
    parts.append("</div>")

OUT.write_text("\n".join(parts), encoding="utf-8")
print(f"wrote {len(figs)} charts -> {OUT}")
