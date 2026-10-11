"""End-to-end check of the whole pipeline against the real downloaded data.

Runs every step the app runs -- load, stats, AHP, both covering modes, the
threshold sweep, the benchmark against Seattle's real stations, and the
dispatch assignment -- and asserts the properties each model must satisfy.

Run:  python scripts/verify_pipeline.py
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import ahp as ahp_mod  # noqa: E402
from src import assignment as asg  # noqa: E402
from src import config, covering, data_loader, stats  # noqa: E402

TK = dict(speed_kmh=config.DEFAULT_SPEED_KMH,
          circuity=config.DEFAULT_CIRCUITY,
          turnout_min=config.DEFAULT_TURNOUT_MIN)

passed = failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"   [PASS] {label}" + (f" — {detail}" if detail else ""))
    else:
        failed += 1
        print(f"   [FAIL] {label}" + (f" — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


t_start = time.time()

# ---------------------------------------------------------------------------
section("STEP 1 — Data cleaning & statistics")
# ---------------------------------------------------------------------------
incidents = data_loader.load_incidents()
existing = data_loader.load_existing_stations()
print(f"   loaded {len(incidents):,} incidents, {len(existing)} real SFD stations")
print(f"   range  {incidents['datetime'].min()} -> {incidents['datetime'].max()}")

# Which dataset this checkout has decides what "complete" means. A full local
# checkout carries the whole 2.2M-row feed; a deployed one carries only the
# committed slice (see src/config.py). Asserting 2.1M either way turns a
# correct deployment into a red run, and dropping the assertion altogether
# would stop catching a locally truncated Parquet -- so the bound follows the
# file that was actually loaded.
ON_FULL_DATASET = config.CALLS_PARQUET == config.CALLS_PARQUET_FULL
if ON_FULL_DATASET:
    check("dataset is the full ~2.2M feed", len(incidents) > 2_100_000,
          f"{len(incidents):,} rows")
else:
    span_years = (incidents["datetime"].max() - incidents["datetime"].min()).days / 365.25
    check(f"deploy slice holds its {config.DEPLOY_YEARS} years",
          len(incidents) > 100_000 and span_years >= config.DEPLOY_YEARS - 0.1,
          f"{len(incidents):,} rows over {span_years:.1f} years")
check("incident numbers unique", incidents["incident_number"].is_unique)
check("no missing coordinates", incidents[["latitude", "longitude"]].notna().all().all())
check("all incidents have a ZIP zone", incidents["zip_code"].notna().all())
check("coordinates inside Seattle", bool(
    incidents["latitude"].between(47.40, 47.80).all()
    and incidents["longitude"].between(-122.50, -122.15).all()))
check("core EMS is a subset of the extended tier",
      int((incidents["is_ems_core"] & ~incidents["is_ems_trauma"]).sum()) == 0)

end = incidents["datetime"].max().date()
start = (pd.Timestamp(end) - pd.DateOffset(years=1)).date()
sel = data_loader.filter_period(incidents, start, end, include_trauma=False)
all_zones = data_loader.build_zones(sel)
zones = data_loader.material_zones(all_zones)
dropped = all_zones.loc[~all_zones["material"]]
print(f"\n   study period {start} -> {end}: {len(sel):,} EMS incidents")
print(f"   {len(all_zones)} ZIP zones, {len(zones)} above the materiality floor")
print(f"   excluded {len(dropped)} fringe zone(s) holding "
      f"{int(dropped['demand'].sum()):,} incidents "
      f"({100 * dropped['demand_share'].sum():.3f}% of demand): "
      f"{', '.join(dropped['zip_code'].astype(str))}")

check("period slice is non-empty", len(sel) > 0)
check("period slice respects the window", bool(
    sel["datetime"].min().date() >= start and sel["datetime"].min().date() <= end))
check("period slice is EMS-only", bool(sel["is_ems_core"].all()))
check("zone demand sums to the incident count",
      int(all_zones["demand"].sum()) == len(sel),
      f"{int(all_zones['demand'].sum()):,} == {len(sel):,}")
check("demand shares sum to 1", abs(float(all_zones["demand_share"].sum()) - 1.0) < 1e-9)
check("materiality floor keeps >99% of demand",
      float(zones["demand_share"].sum()) > 0.99,
      f"{100 * zones['demand_share'].sum():.3f}% retained")
check("every excluded zone is genuinely tiny",
      bool(dropped.empty or (dropped["demand"] < zones["demand"].min()).all()))

h = stats.headline(sel)
bz = stats.busiest_zones(sel)
print(f"   busiest zone {h['busiest_zone']}, peak hour {h['peak_hour']}, "
      f"{h['per_day']:.0f} EMS calls/day")
print("\n   top 5 zones by demand:")
for _, r in bz.head(5).iterrows():
    print(f"     {int(r['rank'])}. ZIP {r['zip_code']}  {int(r['incidents']):>7,} "
          f"({r['share_pct']:5.2f}%)")

check("busiest-zone counts sum to the total", int(bz["incidents"].sum()) == len(sel))
check("hourly histogram sums to the total",
      int(stats.calls_by_hour(sel)["incidents"].sum()) == len(sel))
check("weekday histogram sums to the total",
      int(stats.calls_by_weekday(sel)["incidents"].sum()) == len(sel))
check("heatmap sums to the total",
      int(stats.hour_weekday_heatmap(sel).to_numpy().sum()) == len(sel))

# ---------------------------------------------------------------------------
section("STEP 2 — AHP multi-criteria site scoring")
# ---------------------------------------------------------------------------
# The judgements under test are the ones the app actually uses, not a private
# fixture -- otherwise the suite could pass on numbers nobody ever sees.
members = [config.judgements_for(name) for name, _ in config.TEAM]
print(f"   judgements: {'the team\'s own' if config.JUDGEMENTS_ARE_REAL else 'PLACEHOLDER'}"
      f" ({len(members)} members)")
res_ahp = ahp_mod.run_ahp(members)
d = res_ahp.diagnostics
print("   weights: " + ", ".join(f"{c} {w * 100:.1f}%"
                                 for c, w in zip(res_ahp.criteria, res_ahp.weights)))
print(f"   lambda_max = {d['lambda_max']:.4f}   CI = {d['CI']:.4f}   "
      f"RI = {d['RI']}   CR = {d['CR']:.4f}")

check("weights sum to 1", abs(float(res_ahp.weights.sum()) - 1.0) < 1e-9)
check("weights are all positive", bool((res_ahp.weights > 0).all()))
check("lambda_max >= n (Saaty bound)", d["lambda_max"] >= len(res_ahp.criteria) - 1e-9)
check("CR < 0.10 — judgements are consistent", d["consistent"], f"CR = {d['CR']:.4f}")
check("aggregated matrix stays reciprocal",
      bool(np.allclose(res_ahp.matrix * res_ahp.matrix.T, 1.0)))
check("every member individually consistent",
      all(p["consistent"] for p in res_ahp.per_member))

scored = ahp_mod.score_sites(zones, existing, res_ahp, 8, **TK)
print("\n   top 5 candidate sites by AHP priority:")
for _, r in scored.head(5).iterrows():
    print(f"     {int(r['rank'])}. ZIP {r['zip_code']}  score {r['ahp_score']:.4f}  "
          f"({int(r['Incident Density']):,} calls, "
          f"{r['Equity Gap']:.2f} km to nearest station, "
          f"reaches {int(r['zones_reached'])} zones)")

check("every zone scored", len(scored) == len(zones))
check("AHP scores are in [0, 1]",
      bool(scored["ahp_score"].between(0, 1 + 1e-9).all()))
check("ranking is sorted descending",
      bool(scored["ahp_score"].is_monotonic_decreasing))
check("equity gap is non-negative", bool((scored["Equity Gap"] >= 0).all()))

# ---------------------------------------------------------------------------
section("STEP 3 — Optimisation 1: Covering model (ILP)")
# ---------------------------------------------------------------------------
candidates = scored[["zip_code", "latitude", "longitude"]].copy()
cand_ahp = scored["ahp_score"].to_numpy()

print(f"   solver in use: {config.solver_name()}")
print("   Mode A — Set Covering")
cov = covering.solve_set_covering(candidates, zones, 8, cand_ahp, TK)
print(f"     status {cov.status}: {cov.n_stations} stations cover "
      f"{cov.zones_covered}/{cov.zones_total} zones = {cov.demand_pct:.2f}% of demand "
      f"({cov.solve_seconds:.2f}s)")
print(f"     opened at ZIPs: {', '.join(cov.chosen['zip_code'].astype(str))}")

check("set covering solved to optimality", cov.status == "Optimal")
check("full coverage of every reachable zone",
      cov.zones_covered + len(cov.uncoverable) == cov.zones_total)
check("opens at least one station", cov.n_stations >= 1)
check("chosen sites come from the candidate pool",
      set(cov.chosen["zip_code"]).issubset(set(candidates["zip_code"])))

print("\n   Mode B — Maximal Covering")
prev = -1.0
mono = True
for p in [1, 3, 5, 8]:
    mc = covering.solve_maximal_covering(candidates, zones, 8, p, TK)
    print(f"     P={p:>2}: {mc.demand_pct:6.2f}% of demand, "
          f"{mc.zones_covered}/{mc.zones_total} zones ({mc.solve_seconds:.2f}s)")
    if mc.demand_pct < prev - 1e-6:
        mono = False
    prev = mc.demand_pct
    if p == 5:
        mc5 = mc

check("maximal covering non-decreasing in P", mono)
check("maximal covering opens exactly P", mc5.n_stations == 5)

mc_at_opt = covering.solve_maximal_covering(candidates, zones, 8, cov.n_stations, TK)
check("MCLP at P = set-cover optimum reaches full coverage",
      abs(mc_at_opt.demand_pct - cov.demand_pct) < 1e-6,
      f"{mc_at_opt.demand_pct:.4f}% vs {cov.demand_pct:.4f}%")

print("\n   Threshold sweep — 5 vs 8 vs 10 minutes")
sweep = covering.threshold_sweep(candidates, zones, [5, 8, 10], cand_ahp, TK)
for _, r in sweep.iterrows():
    print(f"     {int(r['threshold_min']):>2} min -> {int(r['stations_required'])} stations, "
          f"{r['demand_covered_pct']:.2f}% of demand, "
          f"{int(r['uncoverable_zones'])} unreachable zone(s)")

check("looser limit never needs more stations",
      bool((sweep.sort_values('threshold_min')['stations_required']
            .diff().dropna() <= 0).all()))
check("all sweep solves optimal", bool((sweep["status"] == "Optimal").all()))

print("\n   Benchmark vs Seattle's 36 real stations")
for t in [5, 8, 10]:
    ours = covering.solve_set_covering(candidates, zones, t, cand_ahp, TK)
    theirs = covering.evaluate_fixed_stations(existing, zones, t, TK)
    print(f"     {t:>2} min | ours: {ours.n_stations:>2} stations -> {ours.demand_pct:6.2f}% "
          f"| Seattle: {len(existing)} stations -> {theirs.demand_pct:6.2f}%")
    if t == 8:
        bench8 = (ours, theirs)

check("benchmark evaluates without error", bench8[1].zones_total == len(zones))
check("optimised plan uses fewer stations than Seattle's network",
      bench8[0].n_stations < len(existing),
      f"{bench8[0].n_stations} vs {len(existing)}")

# ---------------------------------------------------------------------------
section("STEP 4 — Optimisation 2: Assignment model (LP)")
# ---------------------------------------------------------------------------
peak_ts, peak_n = data_loader.peak_hour_window(sel)
print(f"   busiest hour in the period: {peak_ts:%a %d %b %Y %H:%M} with {peak_n} incidents")

hour_inc = data_loader.incidents_in_hour(sel, peak_ts)
check("peak hour extraction matches the count", len(hour_inc) == peak_n)

for units in (1, 2):
    amb = asg.ambulances_from_stations(cov.chosen, units)
    a = asg.solve_assignment(amb, hour_inc, TK)
    print(f"\n   {len(amb)} units ({units}/station) vs {len(hour_inc)} incidents "
          f"-> {a.dummy_kind}")
    print(f"     status {a.status}: total {a.total_time_min:.2f} min, "
          f"mean {a.mean_time_min:.2f} min, worst {a.max_time_min:.2f} min "
          f"({a.solve_seconds:.2f}s)")
    print(f"     served {a.n_served}, unserved {a.n_unserved}, idle {a.n_idle}")
    print(f"     like-for-like over the first {a.benchmark_n_incidents} calls: "
          f"optimal {a.benchmark_lp_min:.2f} min vs greedy {a.greedy_total_min:.2f} min "
          f"-> LP is {a.improvement_pct:.2f}% better")

    check(f"[{units}/station] LP solved to optimality", a.status == "Optimal")
    check(f"[{units}/station] every incident accounted for",
          a.n_served + a.n_unserved == a.n_incidents)
    served_units = a.assignments.loc[a.assignments["served"], "ambulance"]
    check(f"[{units}/station] each ambulance used at most once", served_units.is_unique)
    check(f"[{units}/station] serves min(units, incidents)",
          a.n_served == min(len(amb), len(hour_inc)))
    check(f"[{units}/station] LP is never worse than greedy",
          a.benchmark_lp_min <= a.greedy_total_min + 1e-9)
    check(f"[{units}/station] benchmark compares equal workloads",
          a.benchmark_n_incidents == min(len(amb), len(hour_inc)),
          f"{a.benchmark_n_incidents} calls on both sides")
    check(f"[{units}/station] all response times positive",
          bool((a.assignments.loc[a.assignments['served'], 'response_min'] > 0).all()))

    # Independent optimality cross-check against the Hungarian algorithm.
    try:
        from scipy.optimize import linear_sum_assignment
        bal, _, _, _ = asg.balance(asg.build_cost_matrix(amb, hour_inc, **TK))
        ri, ci = linear_sum_assignment(bal)
        check(f"[{units}/station] matches the Hungarian optimum",
              abs(a.total_time_min - bal[ri, ci].sum()) < 1e-6,
              f"{a.total_time_min:.6f} vs {bal[ri, ci].sum():.6f}")
    except ImportError:
        print("     (scipy not available — skipped Hungarian cross-check)")

# ---------------------------------------------------------------------------
section("STEP 5 — Optimality cross-checks against brute force")
# ---------------------------------------------------------------------------
# Everything above asserts the properties each model must satisfy. These go
# further and prove optimality outright, on instances small enough to enumerate
# exhaustively -- so a silent solver swap, or a drift in how a model is stated,
# cannot slip through looking plausible.
rng = np.random.default_rng(20260921)


def _synthetic(n: int) -> pd.DataFrame:
    """A small random city: n candidate sites that are also the demand zones."""
    return pd.DataFrame({
        "zip_code": [f"Z{i}" for i in range(n)],
        "latitude": 47.6 + rng.uniform(-0.10, 0.10, n),
        "longitude": -122.33 + rng.uniform(-0.10, 0.10, n),
        "demand": rng.integers(1, 500, n).astype(float),
    })


lscp_bad = mclp_bad = lscp_n = mclp_n = 0
for _ in range(25):
    zs = _synthetic(int(rng.integers(4, 8)))
    cs = zs[["zip_code", "latitude", "longitude"]].copy()
    thr = float(rng.choice([3.0, 4.0, 5.0, 6.0]))
    amat, _ = covering.coverage_matrix(cs, zs, thr, **TK)
    need = np.where(amat.any(axis=0))[0]
    if not len(need):
        continue
    sites = range(len(cs))

    # Mode A: the smallest subset of sites that covers every reachable zone.
    best_k = next(
        k for k in range(1, len(cs) + 1)
        if any(all(amat[list(combo), j].any() for j in need)
               for combo in itertools.combinations(sites, k))
    )
    lscp_n += 1
    if covering.solve_set_covering(cs, zs, thr, None, TK).n_stations != best_k:
        lscp_bad += 1

    # Mode B: the most demand any P sites can reach.
    p = int(rng.integers(1, len(cs) + 1))
    dem = zs["demand"].to_numpy(dtype=float)
    best_d = max(
        sum(dem[j] for j in range(len(zs)) if amat[list(combo), j].any())
        for combo in itertools.combinations(sites, p)
    )
    mclp_n += 1
    if abs(covering.solve_maximal_covering(cs, zs, thr, p, TK).demand_covered - best_d) > 1e-6:
        mclp_bad += 1

check("set covering is provably optimal", lscp_bad == 0,
      f"matched brute force on {lscp_n} fully-enumerated instances")
check("maximal covering is provably optimal", mclp_bad == 0,
      f"matched brute force on {mclp_n} fully-enumerated instances")

# AHP weights must be the principal eigenvector -- the largest-modulus one,
# which is what Perron-Frobenius guarantees exists for a positive matrix.
eig_bad = 0
for _ in range(200):
    k = int(rng.integers(3, 7))
    mat = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            v = float(rng.choice([1 / 9, 1 / 7, 1 / 5, 1 / 3, 1, 3, 5, 7, 9]))
            mat[i, j], mat[j, i] = v, 1.0 / v
    vals, vecs = np.linalg.eig(mat)
    ref = np.abs(vecs[:, int(np.argmax(np.abs(vals)))].real)
    if not np.allclose(ahp_mod.priority_vector(mat), ref / ref.sum(), atol=1e-8):
        eig_bad += 1
check("AHP weights are the principal eigenvector", eig_bad == 0,
      "matched the largest-modulus eigenvector on 200 random matrices")

# On an equal workload the optimal assignment can never lose to greedy.
lp_lost = 0
for _ in range(150):
    cm = rng.uniform(0.5, 30.0, size=(int(rng.integers(1, 7)), int(rng.integers(1, 7))))
    lp_t, gr_t, _ = asg.benchmark_vs_greedy(cm)
    if lp_t > gr_t + 1e-7:
        lp_lost += 1
check("optimal dispatch never loses to greedy", lp_lost == 0,
      "over 150 random cost matrices")

# ---------------------------------------------------------------------------
section("RESULT")
# ---------------------------------------------------------------------------
print(f"   {passed} passed, {failed} failed   ({time.time() - t_start:.0f}s)")
sys.exit(1 if failed else 0)
