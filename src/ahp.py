"""Step 2 -- AHP multi-criteria site scoring.

Combines three competing criteria into one priority score per candidate zone:

    Incident Density    number of emergencies in the area
    Equity Gap          distance to the nearest existing station
    Coverage Potential  how many zones a station here could reach quickly

Weights come from Saaty pairwise comparisons on the 1-9 scale, checked for
logical consistency via lambda_max -> CI -> CR, with CR < 0.1 required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, geo


# ---------------------------------------------------------------------------
# Pairwise comparison matrix
# ---------------------------------------------------------------------------

def build_matrix(comparisons: dict[tuple[int, int], float], n: int) -> np.ndarray:
    """Assemble a reciprocal comparison matrix from the upper-triangle ratings.

    ``comparisons[(i, j)] = a`` means criterion *i* is judged *a* times as
    important as criterion *j* on the Saaty 1-9 scale (values < 1 express the
    reverse). The reciprocal half and the unit diagonal are filled in.
    """
    m = np.ones((n, n), dtype=float)
    for (i, j), val in comparisons.items():
        val = float(val)
        if val <= 0:
            raise ValueError(f"comparison ({i},{j}) must be positive, got {val}")
        m[i, j] = val
        m[j, i] = 1.0 / val
    return m


def priority_vector(matrix: np.ndarray) -> np.ndarray:
    """Principal right eigenvector of the comparison matrix, normalised to 1.

    This is Saaty's exact method (rather than the row-geometric-mean
    approximation), so the weights are consistent with the lambda_max used for
    the consistency ratio.
    """
    vals, vecs = np.linalg.eig(matrix)
    idx = int(np.argmax(vals.real))
    w = np.abs(vecs[:, idx].real)
    return w / w.sum()


def consistency(matrix: np.ndarray, weights: np.ndarray) -> dict:
    """lambda_max -> Consistency Index -> Consistency Ratio.

    CR < 0.10 means the human judgements do not meaningfully contradict each
    other and the derived weights can be trusted.
    """
    n = matrix.shape[0]
    lam_max = float((matrix @ weights / weights).mean())
    ci = (lam_max - n) / (n - 1) if n > 1 else 0.0
    ri = config.SAATY_RI.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    return {
        "n": n,
        "lambda_max": lam_max,
        "CI": ci,
        "RI": ri,
        "CR": cr,
        "consistent": bool(cr < config.CR_THRESHOLD),
    }


def aggregate_matrices(matrices: list[np.ndarray]) -> np.ndarray:
    """Combine each team member's independent matrix into one.

    Uses the element-wise geometric mean (Aggregation of Individual Judgements).
    The geometric mean is the correct average for ratio judgements: it is the
    only one that keeps the result reciprocal, so if the members' views of
    A-vs-B average to 3, their view of B-vs-A automatically averages to 1/3.
    """
    if not matrices:
        raise ValueError("no matrices to aggregate")
    stack = np.stack([np.asarray(m, dtype=float) for m in matrices])
    return np.exp(np.log(stack).mean(axis=0))


@dataclass
class AHPResult:
    criteria: list[str]
    matrix: np.ndarray
    weights: np.ndarray
    diagnostics: dict
    per_member: list[dict] = field(default_factory=list)

    @property
    def weights_series(self) -> pd.Series:
        return pd.Series(self.weights, index=self.criteria, name="weight")

    def weights_table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Criterion": self.criteria,
            "Weight": self.weights,
            "Weight %": 100.0 * self.weights,
        })

    def matrix_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.matrix, index=self.criteria, columns=self.criteria)


def run_ahp(
    comparisons_per_member: list[dict[tuple[int, int], float]],
    criteria: list[str] | None = None,
) -> AHPResult:
    """Full AHP: per-member matrices -> aggregate -> weights -> CR check."""
    criteria = criteria or config.AHP_CRITERIA
    n = len(criteria)

    matrices, per_member = [], []
    for k, comp in enumerate(comparisons_per_member):
        m = build_matrix(comp, n)
        w = priority_vector(m)
        diag = consistency(m, w)
        matrices.append(m)
        per_member.append({"member": k + 1, "matrix": m, "weights": w, **diag})

    agg = aggregate_matrices(matrices)
    weights = priority_vector(agg)
    diag = consistency(agg, weights)
    return AHPResult(criteria, agg, weights, diag, per_member)


# ---------------------------------------------------------------------------
# Criterion scores per candidate zone
# ---------------------------------------------------------------------------

def criteria_scores(
    zones: pd.DataFrame,
    existing_stations: pd.DataFrame,
    threshold_min: float,
    speed_kmh: float = config.DEFAULT_SPEED_KMH,
    circuity: float = config.DEFAULT_CIRCUITY,
    turnout_min: float = config.DEFAULT_TURNOUT_MIN,
) -> pd.DataFrame:
    """Raw values of the three AHP criteria for every candidate zone.

    All three are benefit criteria -- larger is a stronger case for a station.
    """
    kw = dict(speed_kmh=speed_kmh, circuity=circuity, turnout_min=turnout_min)

    # 1. Incident Density -- emergencies in the area.
    density = zones["demand"].to_numpy(dtype=float)

    # 2. Equity Gap -- km to the nearest station Seattle already operates.
    if existing_stations is not None and len(existing_stations):
        d_exist = geo.distance_matrix_km(zones, existing_stations)
        equity = d_exist.min(axis=1)
    else:
        equity = np.zeros(len(zones))

    # 3. Coverage Potential -- demand reachable from here inside the threshold.
    t_zz = geo.time_matrix_min(zones, zones, **kw)
    reachable = t_zz <= threshold_min
    coverage = reachable @ zones["demand"].to_numpy(dtype=float)
    zones_reached = reachable.sum(axis=1)

    return pd.DataFrame({
        "zip_code": zones["zip_code"].to_numpy(),
        "latitude": zones["latitude"].to_numpy(),
        "longitude": zones["longitude"].to_numpy(),
        "Incident Density": density,
        "Equity Gap": equity,
        "Coverage Potential": coverage,
        "zones_reached": zones_reached,
    })


def normalise(scores: pd.DataFrame, criteria: list[str]) -> pd.DataFrame:
    """Scale each criterion onto [0, 1] so the weights can combine them.

    Max-normalisation (value / column max) is used because every criterion is
    a benefit measured on a ratio scale from a true zero.
    """
    out = scores.copy()
    for c in criteria:
        col = out[c].to_numpy(dtype=float)
        hi = col.max()
        out[c + " (norm)"] = col / hi if hi > 0 else np.zeros_like(col)
    return out


def score_sites(
    zones: pd.DataFrame,
    existing_stations: pd.DataFrame,
    ahp: AHPResult,
    threshold_min: float,
    **travel_kwargs,
) -> pd.DataFrame:
    """Rank every candidate zone by its weighted AHP priority score."""
    raw = criteria_scores(zones, existing_stations, threshold_min, **travel_kwargs)
    norm = normalise(raw, ahp.criteria)

    weighted = np.zeros(len(norm))
    for c, w in zip(ahp.criteria, ahp.weights):
        weighted += w * norm[c + " (norm)"].to_numpy(dtype=float)
    norm["ahp_score"] = weighted

    norm = norm.sort_values("ahp_score", ascending=False).reset_index(drop=True)
    norm.insert(0, "rank", np.arange(1, len(norm) + 1))
    return norm


# PLACEHOLDER JUDGEMENTS -- not real team input.
#
# The project calls for each member to fill the pairwise table independently and
# for the results to be combined. These values are a defensible starting point
# only, so the app has something coherent to open with: density judged
# moderately-to-strongly more important than the equity gap, strongly more
# important than raw coverage potential, equity slightly ahead of coverage
# potential. They yield CR = 0.0032, comfortably inside the 0.1 limit.
#
# Replace them with the team's actual comparisons via the per-member tabs in
# "2 - AHP Site Scoring"; the criterion weights, and therefore the site ranking
# and the candidate pool handed to the covering model, all depend on them.
DEFAULT_COMPARISONS: dict[tuple[int, int], float] = {
    (0, 1): 3.0,
    (0, 2): 5.0,
    (1, 2): 2.0,
}
