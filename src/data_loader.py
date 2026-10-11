"""Step 1a -- ingestion and cleaning.

Turns the raw Seattle 911 feed into a tidy EMS-only incident table with a ZIP
zone attached to every row, cached to Parquet so the app starts instantly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import config, geo

# Seattle's land extent, used to discard obviously bad coordinates (the feed
# contains a small number of 0/0 and out-of-region rows).
LAT_RANGE = (47.40, 47.80)
LON_RANGE = (-122.50, -122.15)


def ems_mask(types: pd.Series, include_trauma: bool = False) -> pd.Series:
    """Boolean mask selecting genuine EMS/aid calls from the ``type`` column."""
    patterns = list(config.EMS_CORE_PATTERNS)
    if include_trauma:
        patterns += config.EMS_TRAUMA_PATTERNS
    rx = re.compile("|".join(patterns), flags=re.IGNORECASE)
    return types.fillna("").str.contains(rx, regex=True)


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Type-cast, de-duplicate and drop rows that cannot be mapped."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["type"] = df["type"].astype("string").str.strip()

    df = df.dropna(subset=["datetime", "latitude", "longitude", "type"])

    # De-duplication: the feed republishes rows as an incident is updated, so
    # the same incident number can appear several times.
    df = df.drop_duplicates(subset=["incident_number"], keep="first")

    in_box = (
        df["latitude"].between(*LAT_RANGE)
        & df["longitude"].between(*LON_RANGE)
    )
    return df.loc[in_box].reset_index(drop=True)


def build_processed(
    calls_csv: Path | None = None,
    out_path: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Full clean -> ZIP-tag -> Parquet pipeline. Run once after downloading."""
    calls_csv = calls_csv or config.CALLS_CSV
    # The canonical full-dataset path, never the resolved read path: on a
    # checkout that only has the committed deploy slice, `config.CALLS_PARQUET`
    # points *at* that slice, and regenerating would overwrite it.
    out_path = out_path or config.CALLS_PARQUET_FULL
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    log(f"Reading {calls_csv} ...")
    raw = pd.read_csv(
        calls_csv,
        dtype={"incident_number": "string", "type": "string", "address": "string"},
    )
    log(f"  raw rows                     : {len(raw):,}")

    df = clean_raw(raw)
    log(f"  after cleaning & de-dup      : {len(df):,}")

    # Tag EMS membership at both tiers so the app can switch without re-running
    # the expensive spatial join.
    df["is_ems_core"] = ems_mask(df["type"], include_trauma=False)
    df["is_ems_trauma"] = ems_mask(df["type"], include_trauma=True)
    log(f"  core EMS/aid calls           : {int(df['is_ems_core'].sum()):,}")
    log(f"  incl. trauma/rescue          : {int(df['is_ems_trauma'].sum()):,}")

    log("  assigning ZIP zones (point-in-polygon) ...")
    df["zip_code"] = geo.assign_zip_codes(df)
    before = len(df)
    df = df.dropna(subset=["zip_code"]).reset_index(drop=True)
    log(f"  inside a ZIP polygon         : {len(df):,} ({before - len(df):,} dropped)")

    df["zip_code"] = df["zip_code"].astype("string")
    df["hour"] = df["datetime"].dt.hour.astype("int16")
    df["weekday"] = df["datetime"].dt.dayofweek.astype("int8")
    df["date"] = df["datetime"].dt.date
    df["year"] = df["datetime"].dt.year.astype("int16")

    cols = [
        "incident_number", "type", "datetime", "year", "hour", "weekday",
        "address", "latitude", "longitude", "zip_code", "is_ems_core", "is_ems_trauma",
    ]
    df = df[cols]
    df.to_parquet(out_path, index=False)
    log(f"  wrote {len(df):,} rows -> {out_path} "
        f"({out_path.stat().st_size / 1e6:.1f} MB)")
    return df


def load_incidents(path: Path | None = None) -> pd.DataFrame:
    """Load the processed incident table."""
    path = path or config.CALLS_PARQUET
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run:\n"
            "  python scripts/download_data.py\n"
            "  python scripts/preprocess.py"
        )
    # The app holds this table for the whole session, so it is worth loading
    # lean. Two savings, together roughly halving it at 2.2M rows:
    #
    #   `date` duplicates `datetime` -- daily_trend derives its own from
    #   `datetime` -- and costs ~88 MB as object dtype, so it is never read.
    #
    #   `type` has 270 distinct values and `zip_code` 49, so holding them as
    #   categories rather than strings saves ~75 MB. Every groupby over them
    #   already passes observed=True, which is the categorical-aware form.
    import pyarrow.parquet as pq

    wanted = [c for c in pq.ParquetFile(path).schema_arrow.names if c != "date"]
    df = pd.read_parquet(path, columns=wanted)
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ("type", "zip_code"):
        if col in df.columns and not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")
    return df


def load_existing_stations(path: Path | None = None) -> pd.DataFrame:
    """Seattle's real operating fire stations (the benchmark for comparison)."""
    path = path or config.STATIONS_CSV
    df = pd.read_csv(path)
    return df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


def filter_period(
    df: pd.DataFrame,
    start,
    end,
    include_trauma: bool = False,
) -> pd.DataFrame:
    """Select EMS incidents inside an inclusive date window."""
    flag = "is_ems_trauma" if include_trauma else "is_ems_core"
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) + pd.Timedelta(days=1)  # inclusive of the end date
    mask = df[flag] & (df["datetime"] >= start) & (df["datetime"] < end)
    return df.loc[mask].reset_index(drop=True)


DEFAULT_MIN_DEMAND_SHARE = 0.001  # 0.1% of period demand


def build_zones(
    incidents: pd.DataFrame,
    min_demand_share: float = DEFAULT_MIN_DEMAND_SHARE,
) -> pd.DataFrame:
    """Aggregate incidents into demand zones -- one row per ZIP code.

    Each zone gets a demand-weighted centroid (the mean incident location),
    which is both the demand point and the candidate station site used by the
    covering model. Using the incident centroid rather than the polygon
    centroid puts the candidate where the calls actually are.

    ``material`` flags whether a zone is big enough to plan around. The feed
    contains a long tail of fringe ZIPs holding a handful of calls a year --
    mutual-aid runs and GPS jitter across the city boundary, in suburbs Seattle
    Fire does not primarily serve. Set covering treats a one-call zone exactly
    like a ten-thousand-call zone, so left in they would each demand their own
    dedicated station and dominate the answer. Zones below the share floor stay
    in the table (nothing is hidden) but are excluded from the models.
    """
    grp = incidents.groupby("zip_code", observed=True)
    zones = grp.agg(
        demand=("incident_number", "size"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
    ).reset_index()
    zones = zones.sort_values("demand", ascending=False).reset_index(drop=True)
    zones["demand_share"] = zones["demand"] / zones["demand"].sum()
    zones["material"] = zones["demand_share"] >= min_demand_share
    return zones


def material_zones(zones: pd.DataFrame) -> pd.DataFrame:
    """The zones the optimisation models actually plan around."""
    if "material" not in zones.columns:
        return zones.reset_index(drop=True)
    return zones.loc[zones["material"]].reset_index(drop=True)


def peak_hour_window(incidents: pd.DataFrame) -> tuple[pd.Timestamp, int]:
    """Find the single busiest calendar hour in the selected period.

    Returns ``(hour_start_timestamp, incident_count)`` -- the concrete
    "peak hour of simultaneous emergencies" the assignment model runs on.
    """
    if incidents.empty:
        raise ValueError("no incidents in the selected period")
    counts = incidents["datetime"].dt.floor("h").value_counts()
    ts = counts.idxmax()
    return pd.Timestamp(ts), int(counts.max())


def incidents_in_hour(incidents: pd.DataFrame, hour_start) -> pd.DataFrame:
    """All incidents that occurred within one clock hour."""
    hour_start = pd.Timestamp(hour_start)
    mask = (incidents["datetime"] >= hour_start) & (
        incidents["datetime"] < hour_start + pd.Timedelta(hours=1)
    )
    return incidents.loc[mask].sort_values("datetime").reset_index(drop=True)
