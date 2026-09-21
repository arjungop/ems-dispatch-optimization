"""Step 4 -- Optimisation 2: the assignment model (LP).

Decides WHICH ambulance goes to WHICH incident during a busy hour so that
total travel time across all responders is minimised.

    Incidents  = real calls from a chosen peak hour
    Ambulances = one ready unit stationed at each base
    Each incident -> exactly one ambulance
    Each ambulance -> at most one incident

The bases come straight from the covering model's chosen station set. That
linkage is what joins the two optimisations into one framework: where we put
the stations determines how fast we can dispatch from them.

Unequal counts are handled the transportation-problem way, by padding the cost
matrix with dummy rows/columns so the problem is balanced and always feasible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pulp

from . import config, geo


@dataclass
class AssignmentResult:
    status: str
    assignments: pd.DataFrame        # one row per incident
    cost_matrix: pd.DataFrame        # real ambulances x real incidents (minutes)
    balanced_matrix: pd.DataFrame    # square matrix incl. dummy rows/cols
    total_time_min: float
    mean_time_min: float
    max_time_min: float
    n_incidents: int
    n_ambulances: int
    n_served: int
    n_unserved: int
    n_idle: int
    dummy_kind: str                  # "none" | "dummy ambulances" | "dummy incidents"
    solve_seconds: float = 0.0
    greedy_total_min: float | None = None
    benchmark_lp_min: float | None = None  # optimal total over the benchmark calls
    benchmark_n_incidents: int = 0         # how many calls both sides had to clear

    @property
    def improvement_pct(self) -> float:
        """Routing advantage of optimal dispatch over nearest-unit-first greedy.

        Both sides are measured on the same calls -- see ``benchmark_vs_greedy``
        for why the comparison has to be restricted that way.
        """
        if not self.greedy_total_min or self.benchmark_lp_min is None:
            return 0.0
        return 100.0 * (self.greedy_total_min - self.benchmark_lp_min) / self.greedy_total_min


def build_cost_matrix(
    ambulances: pd.DataFrame,
    incidents: pd.DataFrame,
    speed_kmh: float = config.DEFAULT_SPEED_KMH,
    circuity: float = config.DEFAULT_CIRCUITY,
    turnout_min: float = config.DEFAULT_TURNOUT_MIN,
) -> np.ndarray:
    """Travel time in minutes from each ambulance base to each incident."""
    return geo.time_matrix_min(
        ambulances, incidents,
        speed_kmh=speed_kmh, circuity=circuity, turnout_min=turnout_min,
    )


def balance(cost: np.ndarray) -> tuple[np.ndarray, str, int, int]:
    """Pad a rectangular cost matrix into a square one with dummies.

    Fewer ambulances than incidents -> dummy ambulance rows absorb the calls
    that genuinely cannot be served this hour. More ambulances than incidents
    -> dummy incident columns absorb the units that stay idle. Dummy cells cost
    zero, so they never distort the real total travel time being minimised.
    """
    n_amb, n_inc = cost.shape
    if n_amb == n_inc:
        return cost.copy(), "none", 0, 0
    if n_amb < n_inc:
        pad = n_inc - n_amb
        return np.vstack([cost, np.zeros((pad, n_inc))]), "dummy ambulances", pad, 0
    pad = n_amb - n_inc
    return np.hstack([cost, np.zeros((n_amb, pad))]), "dummy incidents", 0, pad


def greedy_baseline(cost: np.ndarray) -> float:
    """Total time under 'send the nearest free unit, call by call'.

    This is what dispatch does without optimisation, and it is the benchmark
    the LP is meant to beat: locally-nearest choices can strand a later call
    with a very long drive.
    """
    n_amb, n_inc = cost.shape
    free = set(range(n_amb))
    total = 0.0
    for j in range(n_inc):  # incidents processed in arrival order
        if not free:
            break
        i = min(free, key=lambda r: cost[r, j])
        total += float(cost[i, j])
        free.discard(i)
    return total


def _optimal_total(cost: np.ndarray, msg: bool = False) -> float:
    """Optimal total travel time for this cost matrix, via the same balanced LP."""
    bal, _, _, _ = balance(cost)
    n = bal.shape[0]
    prob = pulp.LpProblem("assignment_benchmark", pulp.LpMinimize)
    x = {(i, j): pulp.LpVariable(f"b_{i}_{j}", lowBound=0, upBound=1, cat="Continuous")
         for i in range(n) for j in range(n)}
    prob += pulp.lpSum(float(bal[i, j]) * x[(i, j)] for i in range(n) for j in range(n))
    for j in range(n):
        prob += pulp.lpSum(x[(i, j)] for i in range(n)) == 1
    for i in range(n):
        prob += pulp.lpSum(x[(i, j)] for j in range(n)) <= 1
    prob.solve(config.get_solver(msg=msg))
    return float(pulp.value(prob.objective) or 0.0)


def benchmark_vs_greedy(cost: np.ndarray, msg: bool = False) -> tuple[float, float, int]:
    """Optimal vs greedy dispatch, measured over the same calls.

    A peak hour usually has more calls than ready units. Unserved calls cost
    nothing in the objective, so an unrestricted LP would quietly serve the
    *cheapest* calls and leave the far ones waiting, while greedy has to take
    calls as they arrive. Comparing those two totals would measure triage --
    which calls each method chose to answer -- rather than routing, and would
    overstate the LP's advantage several-fold.

    So both sides are given the same workload: the first ``k = min(n_amb,
    n_inc)`` calls in arrival order, which is what a dispatcher actually has to
    clear. What remains in the gap is purely the quality of the matching.

    Returns ``(optimal_total_min, greedy_total_min, k)``.
    """
    n_amb, n_inc = cost.shape
    k = min(n_amb, n_inc)
    sub = cost[:, :k]
    return _optimal_total(sub, msg), greedy_baseline(sub), k


def solve_assignment(
    ambulances: pd.DataFrame,
    incidents: pd.DataFrame,
    travel_kwargs: dict | None = None,
    msg: bool = False,
) -> AssignmentResult:
    """Minimise total travel time over a balanced assignment LP.

        minimise    sum_ij c_ij x_ij
        subject to  sum_i x_ij = 1     every incident gets exactly one unit
                    sum_j x_ij <= 1    every unit takes at most one incident
                    x_ij >= 0

    The constraint matrix is totally unimodular, so the LP relaxation is
    guaranteed to have an integral optimum -- no binary variables needed, which
    is why this is stated as an LP rather than an ILP.
    """
    travel_kwargs = travel_kwargs or {}
    if len(ambulances) == 0 or len(incidents) == 0:
        raise ValueError("assignment needs at least one ambulance and one incident")

    cost = build_cost_matrix(ambulances, incidents, **travel_kwargs)
    bal, kind, pad_rows, pad_cols = balance(cost)
    n = bal.shape[0]

    prob = pulp.LpProblem("EMS_Dispatch_Assignment", pulp.LpMinimize)
    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1, cat="Continuous")
        for i in range(n) for j in range(n)
    }
    prob += pulp.lpSum(float(bal[i, j]) * x[(i, j)] for i in range(n) for j in range(n))
    for j in range(n):
        prob += pulp.lpSum(x[(i, j)] for i in range(n)) == 1, f"incident_{j}"
    for i in range(n):
        prob += pulp.lpSum(x[(i, j)] for j in range(n)) <= 1, f"ambulance_{i}"

    prob.solve(config.get_solver(msg=msg))
    status = pulp.LpStatus[prob.status]

    n_amb, n_inc = cost.shape
    amb_label = ambulances.get("station", pd.Series([f"A{i+1}" for i in range(n_amb)]))
    amb_label = list(amb_label.astype(str)) if hasattr(amb_label, "astype") else list(amb_label)
    amb_zone = list(ambulances["zip_code"].astype(str)) if "zip_code" in ambulances else [""] * n_amb

    rows = []
    exact = []  # full-precision response times; the table below is rounded for display
    for j in range(n_inc):
        picked = next(
            (i for i in range(n) if x[(i, j)].value() and x[(i, j)].value() > 0.5), None
        )
        real = picked is not None and picked < n_amb
        if real:
            exact.append(float(cost[picked, j]))
        rows.append({
            "incident_number": incidents["incident_number"].iloc[j],
            "time": incidents["datetime"].iloc[j],
            "type": incidents["type"].iloc[j],
            "zip_code": incidents["zip_code"].iloc[j],
            "address": incidents["address"].iloc[j] if "address" in incidents else "",
            "latitude": incidents["latitude"].iloc[j],
            "longitude": incidents["longitude"].iloc[j],
            "ambulance": amb_label[picked] if real else "UNSERVED (no unit available)",
            "base_zone": amb_zone[picked] if real else "",
            "base_lat": ambulances["latitude"].iloc[picked] if real else np.nan,
            "base_lon": ambulances["longitude"].iloc[picked] if real else np.nan,
            "response_min": round(float(cost[picked, j]), 2) if real else np.nan,
            "served": bool(real),
        })
    assignments = pd.DataFrame(rows)

    served = assignments["served"]
    times = pd.Series(exact, dtype=float)
    n_idle = max(n_amb - int(served.sum()), 0)

    bench_lp, greedy_min, bench_k = benchmark_vs_greedy(cost, msg)

    labels_amb = amb_label + [f"dummy-amb-{k+1}" for k in range(pad_rows)]
    labels_inc = [str(v) for v in incidents["incident_number"]] + [
        f"dummy-inc-{k+1}" for k in range(pad_cols)
    ]

    return AssignmentResult(
        status=status,
        assignments=assignments,
        cost_matrix=pd.DataFrame(
            np.round(cost, 2), index=amb_label,
            columns=[str(v) for v in incidents["incident_number"]],
        ),
        balanced_matrix=pd.DataFrame(np.round(bal, 2), index=labels_amb, columns=labels_inc),
        total_time_min=float(times.sum()),
        mean_time_min=float(times.mean()) if len(times) else 0.0,
        max_time_min=float(times.max()) if len(times) else 0.0,
        n_incidents=n_inc,
        n_ambulances=n_amb,
        n_served=int(served.sum()),
        n_unserved=int((~served).sum()),
        n_idle=n_idle,
        dummy_kind=kind,
        solve_seconds=float(prob.solutionTime or 0.0),
        greedy_total_min=greedy_min,
        benchmark_lp_min=bench_lp,
        benchmark_n_incidents=bench_k,
    )


def ambulances_from_stations(stations: pd.DataFrame, units_per_station: int = 1) -> pd.DataFrame:
    """One ready unit stationed at each base, per the model's setup.

    ``units_per_station`` > 1 replicates each base so a busy hour with more
    calls than stations can still be served.
    """
    out = stations.copy().reset_index(drop=True)
    if "station" not in out.columns:
        out["station"] = [f"S{i + 1}" for i in range(len(out))]
    if "zip_code" not in out.columns:
        out["zip_code"] = out.get("station_id", out["station"]).astype(str)

    if units_per_station <= 1:
        out["unit"] = out["station"]
        return out

    frames = []
    for k in range(units_per_station):
        f = out.copy()
        f["unit"] = f["station"] + f"-{k + 1}"
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out["station"] = out["unit"]
    return out
