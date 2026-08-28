"""Step 1b -- on-demand statistics.

Answers the three questions the project promises the user can ask of any
chosen period: which zones are busiest, when load spikes, and how much demand
weight each zone carries into the optimisation models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def headline(incidents: pd.DataFrame) -> dict:
    """Top-level counters for the summary tiles."""
    if incidents.empty:
        return {"incidents": 0, "zones": 0, "days": 0, "per_day": 0.0,
                "busiest_zone": "-", "peak_hour": "-", "call_types": 0}

    days = max(incidents["datetime"].dt.normalize().nunique(), 1)
    by_zone = incidents["zip_code"].value_counts()
    by_hour = incidents["hour"].value_counts()
    peak_h = int(by_hour.idxmax())
    return {
        "incidents": len(incidents),
        "zones": int(incidents["zip_code"].nunique()),
        "days": days,
        "per_day": len(incidents) / days,
        "busiest_zone": str(by_zone.idxmax()),
        "peak_hour": f"{peak_h:02d}:00-{(peak_h + 1) % 24:02d}:00",
        "call_types": int(incidents["type"].nunique()),
    }


def busiest_zones(incidents: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    """Incident counts per ZIP zone -- where demand concentrates."""
    out = (
        incidents.groupby("zip_code", observed=True)
        .agg(incidents=("incident_number", "size"),
             latitude=("latitude", "mean"),
             longitude=("longitude", "mean"))
        .reset_index()
        .sort_values("incidents", ascending=False)
        .reset_index(drop=True)
    )
    total = out["incidents"].sum()
    out["share_pct"] = 100.0 * out["incidents"] / total if total else 0.0
    out["cumulative_pct"] = out["share_pct"].cumsum()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out.head(top_n) if top_n else out


def calls_by_hour(incidents: pd.DataFrame) -> pd.DataFrame:
    """Call volume by hour of day -- when load spikes."""
    counts = incidents["hour"].value_counts().reindex(range(24), fill_value=0)
    days = max(incidents["datetime"].dt.normalize().nunique(), 1)
    return pd.DataFrame({
        "hour": range(24),
        "incidents": counts.to_numpy(),
        "avg_per_day": counts.to_numpy() / days,
    })


def calls_by_weekday(incidents: pd.DataFrame) -> pd.DataFrame:
    """Call volume by day of week."""
    counts = incidents["weekday"].value_counts().reindex(range(7), fill_value=0)
    return pd.DataFrame({
        "weekday": WEEKDAY_NAMES,
        "weekday_num": range(7),
        "incidents": counts.to_numpy(),
    })


def hour_weekday_heatmap(incidents: pd.DataFrame) -> pd.DataFrame:
    """Weekday x hour matrix, for spotting the true load peaks."""
    pivot = (
        incidents.pivot_table(index="weekday", columns="hour",
                              values="incident_number", aggfunc="size")
        .reindex(index=range(7), columns=range(24))
        .fillna(0)
        .astype(int)
    )
    pivot.index = WEEKDAY_NAMES
    return pivot


def call_type_mix(incidents: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Breakdown of which EMS call types make up the demand."""
    counts = incidents["type"].value_counts()
    out = counts.head(top_n).rename_axis("type").reset_index(name="incidents")
    out["share_pct"] = 100.0 * out["incidents"] / counts.sum()
    return out


def daily_trend(incidents: pd.DataFrame) -> pd.DataFrame:
    """Daily incident counts across the selected period.

    The 911 feed is harvested live, so the final calendar day in the data is
    usually cut off partway through and its count is not comparable to a full
    day. It is flagged rather than dropped, so the chart can mark it honestly
    instead of showing an unexplained cliff.
    """
    s = incidents["datetime"].dt.normalize().value_counts().sort_index()
    out = pd.DataFrame({"date": s.index, "incidents": s.to_numpy()})
    out["partial"] = False
    if len(out):
        last = incidents.loc[incidents["datetime"] >= out["date"].iloc[-1], "datetime"]
        # A complete day runs to just before midnight; anything short is truncated.
        out.loc[out.index[-1], "partial"] = bool(last.max().hour < 23)
    return out


def demand_weights(zones: pd.DataFrame) -> pd.DataFrame:
    """Per-zone importance fed into both optimisations.

    The raw incident count is the weight the covering model maximises; the
    normalised share is what gets reported as "% of demand covered".
    """
    out = zones[["zip_code", "demand", "latitude", "longitude"]].copy()
    out["weight_pct"] = 100.0 * out["demand"] / out["demand"].sum()
    out = out.sort_values("demand", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def busiest_hours_table(incidents: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """The individual clock hours with the most simultaneous emergencies.

    These are the candidate scenarios for the dispatch/assignment model.
    """
    counts = incidents["datetime"].dt.floor("h").value_counts().head(top_n)
    out = counts.rename_axis("hour_start").reset_index(name="incidents")
    out["weekday"] = pd.to_datetime(out["hour_start"]).dt.day_name()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out
