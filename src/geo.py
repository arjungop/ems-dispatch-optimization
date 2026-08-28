"""Geographic helpers: Haversine distance, travel-time estimation, ZIP tagging."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Fully vectorised / broadcastable."""
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2)
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def distance_matrix_km(origins: pd.DataFrame, destinations: pd.DataFrame) -> np.ndarray:
    """(n_origins x n_destinations) Haversine matrix.

    Both frames need ``latitude`` / ``longitude`` columns.
    """
    o_lat = origins["latitude"].to_numpy(dtype=float)[:, None]
    o_lon = origins["longitude"].to_numpy(dtype=float)[:, None]
    d_lat = destinations["latitude"].to_numpy(dtype=float)[None, :]
    d_lon = destinations["longitude"].to_numpy(dtype=float)[None, :]
    return haversine_km(o_lat, o_lon, d_lat, d_lon)


def travel_time_min(
    distance_km,
    speed_kmh: float = config.DEFAULT_SPEED_KMH,
    circuity: float = config.DEFAULT_CIRCUITY,
    turnout_min: float = config.DEFAULT_TURNOUT_MIN,
):
    """Convert straight-line km into an estimated response time in minutes.

    ``distance_km`` is inflated by ``circuity`` to approximate street travel,
    converted at ``speed_kmh``, then ``turnout_min`` is added so the result is
    time from alarm to arrival rather than pure driving time.
    """
    return turnout_min + (np.asarray(distance_km, dtype=float) * circuity / speed_kmh) * 60.0


def time_matrix_min(origins, destinations, **kwargs) -> np.ndarray:
    """Travel-time matrix in minutes between two lat/lon frames."""
    return travel_time_min(distance_matrix_km(origins, destinations), **kwargs)


# ---------------------------------------------------------------------------
# ZIP-code zone assignment
# ---------------------------------------------------------------------------

def load_zip_polygons(path: Path | None = None):
    """Return ``(zip_codes, shapely_geometries)`` for the Seattle-area ZCTAs."""
    from shapely.geometry import shape

    path = path or config.ZCTA_GEOJSON
    gj = json.loads(Path(path).read_text(encoding="utf-8"))
    zips, geoms = [], []
    for feat in gj["features"]:
        code = feat["properties"].get("ZCTA5") or feat["properties"].get("GEOID")
        if code is None or feat.get("geometry") is None:
            continue
        zips.append(str(code))
        geoms.append(shape(feat["geometry"]))
    return zips, geoms


def assign_zip_codes(df: pd.DataFrame, path: Path | None = None) -> pd.Series:
    """Point-in-polygon tag every incident with the ZIP zone it falls in.

    An STRtree keeps this near-linear, which matters at 2.2M points. Incidents
    outside every polygon come back as NaN and are dropped by the loader.
    """
    from shapely import points as shapely_points
    from shapely import STRtree

    zips, geoms = load_zip_polygons(path)
    tree = STRtree(geoms)

    pts = shapely_points(
        df["longitude"].to_numpy(dtype=float),
        df["latitude"].to_numpy(dtype=float),
    )
    # `query` with predicate returns aligned (input_index, tree_index) pairs.
    src_idx, tree_idx = tree.query(pts, predicate="within")

    out = np.full(len(df), None, dtype=object)
    zip_arr = np.asarray(zips, dtype=object)
    # A point can only be "within" one ZCTA; last write wins on any boundary tie.
    out[src_idx] = zip_arr[tree_idx]
    return pd.Series(out, index=df.index, name="zip_code")
