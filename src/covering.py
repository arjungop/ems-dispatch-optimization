"""Step 3 -- Optimisation 1: the covering model (ILP).

Decides WHERE to place ambulance stations. A zone counts as covered if a
station can reach it within a user-set response-time limit.

    Mode A -- Set Covering (LSCP)
        Minimise the number of stations subject to every demand zone being
        covered. Answers "what is the smallest fleet of stations that leaves
        nobody outside the limit?"

    Mode B -- Maximal Covering (MCLP)
        Fix the station count at P and place them to cover the greatest
        possible share of demand. Answers the budget-constrained question.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pulp

from . import config, geo


@dataclass
class CoverageResult:
    mode: str
    threshold_min: float
    status: str
    chosen: pd.DataFrame                 # the opened stations
    zone_table: pd.DataFrame             # per-zone coverage detail
    n_stations: int
    demand_covered: float
    demand_total: float
    zones_covered: int
    zones_total: int
    uncoverable: list[str] = field(default_factory=list)
    solve_seconds: float = 0.0

    @property
    def demand_pct(self) -> float:
        return 100.0 * self.demand_covered / self.demand_total if self.demand_total else 0.0


def coverage_matrix(
    candidates: pd.DataFrame,
    zones: pd.DataFrame,
    threshold_min: float,
    speed_kmh: float = config.DEFAULT_SPEED_KMH,
    circuity: float = config.DEFAULT_CIRCUITY,
    turnout_min: float = config.DEFAULT_TURNOUT_MIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (candidates x zones) matrix ``a[i][j] = 1`` if i covers j.

    Also returns the underlying travel-time matrix so the app can report how
    far outside the limit the misses actually are.
    """
    t = geo.time_matrix_min(
        candidates, zones,
        speed_kmh=speed_kmh, circuity=circuity, turnout_min=turnout_min,
    )
    return (t <= threshold_min), t


def _zone_table(
    zones: pd.DataFrame,
    a: np.ndarray,
    times: np.ndarray,
    open_idx: list[int],
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Per-zone outcome: covered or not, and by which station, how fast."""
    if open_idx:
        sub_t = times[open_idx, :]
        nearest_pos = sub_t.argmin(axis=0)
        nearest_time = sub_t.min(axis=0)
        nearest_cand = [open_idx[p] for p in nearest_pos]
        covered = a[open_idx, :].any(axis=0)
        served_by = candidates["zip_code"].to_numpy()[nearest_cand]
    else:
        nearest_time = np.full(len(zones), np.inf)
        covered = np.zeros(len(zones), dtype=bool)
        served_by = np.array([None] * len(zones), dtype=object)

    return pd.DataFrame({
        "zip_code": zones["zip_code"].to_numpy(),
        "demand": zones["demand"].to_numpy(),
        "latitude": zones["latitude"].to_numpy(),
        "longitude": zones["longitude"].to_numpy(),
        "covered": covered,
        "nearest_station_zone": served_by,
        "response_min": np.round(nearest_time, 2),
    }).sort_values("demand", ascending=False).reset_index(drop=True)


def solve_set_covering(
    candidates: pd.DataFrame,
    zones: pd.DataFrame,
    threshold_min: float,
    ahp_scores: np.ndarray | None = None,
    travel_kwargs: dict | None = None,
    msg: bool = False,
) -> CoverageResult:
    """Mode A -- minimum stations such that every demand zone is covered.

        minimise    sum_i y_i        (+ tiny AHP tie-break)
        subject to  sum_{i in N(j)} y_i >= 1   for every zone j
                    y_i in {0, 1}

    Zones that no candidate can reach within the limit are structurally
    impossible to cover; they are reported separately and excluded from the
    constraints so the rest of the model still yields an answer.
    """
    travel_kwargs = travel_kwargs or {}
    a, times = coverage_matrix(candidates, zones, threshold_min, **travel_kwargs)

    reachable_zone = a.any(axis=0)
    uncoverable = zones.loc[~reachable_zone, "zip_code"].astype(str).tolist()

    n_c, n_z = a.shape
    prob = pulp.LpProblem("Set_Covering_Station_Placement", pulp.LpMinimize)
    y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n_c)]

    # Primary objective is the station count. A tiny AHP-derived term breaks
    # ties among equally-sized optimal solutions in favour of the sites the
    # multi-criteria ranking prefers; its weight is small enough (<1/n) that it
    # can never justify opening an extra station.
    obj = pulp.lpSum(y)
    if ahp_scores is not None and len(ahp_scores) == n_c:
        eps = 0.5 / max(n_c, 1)
        s = np.asarray(ahp_scores, dtype=float)
        rng = s.max() - s.min()
        pref = (s.max() - s) / rng if rng > 0 else np.zeros_like(s)
        obj += eps * pulp.lpSum(float(pref[i]) * y[i] for i in range(n_c))
    prob += obj

    for j in range(n_z):
        if not reachable_zone[j]:
            continue
        prob += pulp.lpSum(y[i] for i in range(n_c) if a[i, j]) >= 1, f"cover_{j}"

    solver = config.get_solver(msg=msg)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]

    open_idx = [i for i in range(n_c) if y[i].value() is not None and y[i].value() > 0.5]
    return _package(
        "Set Covering", threshold_min, status, open_idx, candidates, zones,
        a, times, uncoverable, prob.solutionTime,
    )


def solve_maximal_covering(
    candidates: pd.DataFrame,
    zones: pd.DataFrame,
    threshold_min: float,
    p: int,
    travel_kwargs: dict | None = None,
    msg: bool = False,
) -> CoverageResult:
    """Mode B -- with exactly P stations, cover the most demand possible.

        maximise    sum_j d_j * z_j
        subject to  z_j <= sum_{i in N(j)} y_i     for every zone j
                    sum_i y_i = P
                    y_i, z_j in {0, 1}

    ``d_j`` is the zone's incident count, so the objective maximises people
    reached rather than area or zone count.
    """
    travel_kwargs = travel_kwargs or {}
    a, times = coverage_matrix(candidates, zones, threshold_min, **travel_kwargs)
    d = zones["demand"].to_numpy(dtype=float)

    n_c, n_z = a.shape
    p = int(max(1, min(p, n_c)))

    prob = pulp.LpProblem("Maximal_Covering_Station_Placement", pulp.LpMaximize)
    y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n_c)]
    z = [pulp.LpVariable(f"z_{j}", cat="Binary") for j in range(n_z)]

    prob += pulp.lpSum(float(d[j]) * z[j] for j in range(n_z))
    for j in range(n_z):
        covering = [y[i] for i in range(n_c) if a[i, j]]
        prob += z[j] <= (pulp.lpSum(covering) if covering else 0), f"link_{j}"
    prob += pulp.lpSum(y) == p, "station_budget"

    prob.solve(config.get_solver(msg=msg))
    status = pulp.LpStatus[prob.status]

    open_idx = [i for i in range(n_c) if y[i].value() is not None and y[i].value() > 0.5]
    uncoverable = zones.loc[~a.any(axis=0), "zip_code"].astype(str).tolist()
    return _package(
        f"Maximal Covering (P={p})", threshold_min, status, open_idx,
        candidates, zones, a, times, uncoverable, prob.solutionTime,
    )


def _package(mode, threshold_min, status, open_idx, candidates, zones,
             a, times, uncoverable, solve_seconds) -> CoverageResult:
    zt = _zone_table(zones, a, times, open_idx, candidates)
    chosen = candidates.iloc[open_idx].copy().reset_index(drop=True)
    chosen.insert(0, "station", [f"S{i + 1}" for i in range(len(chosen))])

    d_total = float(zones["demand"].sum())
    d_cov = float(zt.loc[zt["covered"], "demand"].sum())
    return CoverageResult(
        mode=mode, threshold_min=threshold_min, status=status,
        chosen=chosen, zone_table=zt, n_stations=len(open_idx),
        demand_covered=d_cov, demand_total=d_total,
        zones_covered=int(zt["covered"].sum()), zones_total=len(zones),
        uncoverable=uncoverable, solve_seconds=float(solve_seconds or 0.0),
    )


# ---------------------------------------------------------------------------
# Analyses layered on top of the models
# ---------------------------------------------------------------------------

def threshold_sweep(
    candidates: pd.DataFrame,
    zones: pd.DataFrame,
    thresholds=None,
    ahp_scores: np.ndarray | None = None,
    travel_kwargs: dict | None = None,
) -> pd.DataFrame:
    """How does coverage change at 5 vs 8 vs 10 minutes?

    Re-solves the set-covering model at each limit. Tightening the limit forces
    more stations; loosening it lets fewer stations blanket the city.
    """
    thresholds = thresholds or config.RESPONSE_THRESHOLDS_MIN
    rows = []
    for t in thresholds:
        res = solve_set_covering(candidates, zones, t, ahp_scores, travel_kwargs)
        rows.append({
            "threshold_min": t,
            "stations_required": res.n_stations,
            "zones_covered": res.zones_covered,
            "zones_total": res.zones_total,
            "demand_covered_pct": res.demand_pct,
            "uncoverable_zones": len(res.uncoverable),
            "status": res.status,
        })
    return pd.DataFrame(rows)


def evaluate_fixed_stations(
    stations: pd.DataFrame,
    zones: pd.DataFrame,
    threshold_min: float,
    travel_kwargs: dict | None = None,
    label: str = "Existing stations",
) -> CoverageResult:
    """Score an already-decided set of stations on the same coverage measure.

    Used to benchmark our optimised placement against the 36 stations Seattle
    actually operates, on identical assumptions.
    """
    travel_kwargs = travel_kwargs or {}
    a, times = coverage_matrix(stations, zones, threshold_min, **travel_kwargs)
    open_idx = list(range(len(stations)))

    st = stations.copy()
    if "zip_code" not in st.columns:
        st["zip_code"] = st.get("station_id", pd.Series(range(len(st)))).astype(str)

    return _package(label, threshold_min, "Fixed", open_idx, st, zones,
                    a, times, [], 0.0)


def marginal_coverage_curve(
    candidates: pd.DataFrame,
    zones: pd.DataFrame,
    threshold_min: float,
    max_p: int,
    travel_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Demand covered as a function of the station budget P.

    Shows the diminishing return that justifies the maximal-covering framing:
    the first handful of stations buy most of the coverage.
    """
    rows = []
    for p in range(1, int(max_p) + 1):
        res = solve_maximal_covering(candidates, zones, threshold_min, p, travel_kwargs)
        rows.append({
            "stations": p,
            "demand_covered_pct": res.demand_pct,
            "zones_covered": res.zones_covered,
        })
    df = pd.DataFrame(rows)
    df["marginal_gain_pct"] = df["demand_covered_pct"].diff().fillna(
        df["demand_covered_pct"]
    )
    return df
