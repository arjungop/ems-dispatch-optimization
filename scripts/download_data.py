"""
Data ingestion for the Dynamic EMS Dispatch & Station Placement project.

Pulls the three real-world sources the project depends on:

  1. Seattle Real-Time Fire 911 Calls  -- City of Seattle Open Data Portal,
     dataset id ``kzjm-xkqj`` (~2.2M incident records, public domain).
     This is the DEMAND side: where and when real emergencies occur.

  2. Seattle Fire Stations             -- Seattle City GIS ArcGIS FeatureServer.
     The 36 stations Seattle actually operates, used as the real-world
     benchmark our optimised placement is compared against.

  3. ZIP Code Tabulation Areas         -- US Census TIGERweb.
     Polygons used to compute the ``zip code`` zone of each incident, which is
     the zoning unit for demand aggregation in every downstream model.

Run:  python scripts/download_data.py [--limit N] [--force]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SOCRATA_DOMAIN = "data.seattle.gov"
DATASET_ID = "kzjm-xkqj"
PAGE_SIZE = 50_000

ARCGIS_SEATTLE = "https://services.arcgis.com/ZOyb2t4B0UYuYNYH/ArcGIS/rest/services"
TIGERWEB_ZCTA = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
    "PUMA_TAD_TAZ_UGA_ZCTA/MapServer/11"
)

# Bounding box generous enough to capture every ZCTA that touches Seattle.
SEATTLE_BBOX = {"xmin": -122.47, "ymin": 47.44, "xmax": -122.20, "ymax": 47.78}

CALLS_CSV = RAW / "seattle_911_calls.csv"
STATIONS_CSV = RAW / "seattle_fire_stations.csv"
ZCTA_GEOJSON = RAW / "seattle_zcta.geojson"

FIELDS = ["incident_number", "type", "datetime", "address", "latitude", "longitude"]

UA = {"User-Agent": "OR-Project-EMS-Dispatch/1.0 (academic use)"}


def fetch(url: str, timeout: int = 180, retries: int = 5) -> bytes:
    """GET with exponential backoff -- the portal rate-limits anonymous callers."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            last = exc
            wait = 2**attempt
            print(f"    retry {attempt + 1}/{retries} in {wait}s ({exc})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed after {retries} attempts: {url}") from last


def total_rows() -> int:
    url = f"https://{SOCRATA_DOMAIN}/resource/{DATASET_ID}.json?$select=count(*)%20as%20n"
    return int(json.loads(fetch(url))[0]["n"])


def download_calls(limit: int | None = None, force: bool = False) -> None:
    """Page the Socrata resource endpoint into a single flat CSV.

    Paging (rather than the one-shot ``rows.csv`` export) is used deliberately:
    the export endpoint frequently times out on datasets of this size, while
    ordered paging is resumable and gives progress feedback.
    """
    if CALLS_CSV.exists() and not force:
        print(f"[1/3] 911 calls already present: {CALLS_CSV}")
        return

    n_total = total_rows()
    target = min(n_total, limit) if limit else n_total
    print(f"[1/3] Seattle Real-Time Fire 911 Calls ({DATASET_ID})")
    print(f"      portal reports {n_total:,} records; downloading {target:,}")

    tmp = CALLS_CSV.with_suffix(".csv.part")
    written = 0
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        offset = 0
        while written < target:
            page = min(PAGE_SIZE, target - written)
            params = {
                "$select": ",".join(FIELDS),
                "$order": "datetime DESC,incident_number",  # stable page ordering
                "$limit": page,
                "$offset": offset,
            }
            url = (
                f"https://{SOCRATA_DOMAIN}/resource/{DATASET_ID}.csv?"
                + urllib.parse.urlencode(params)
            )
            body = fetch(url).decode("utf-8", "replace")
            rows = list(csv.DictReader(io.StringIO(body)))
            if not rows:
                print("      portal returned an empty page - stopping early")
                break
            writer.writerows(rows)
            written += len(rows)
            offset += len(rows)
            pct = 100.0 * written / target
            print(f"      {written:>9,} / {target:,}  ({pct:5.1f}%)", flush=True)

    tmp.replace(CALLS_CSV)
    size_mb = CALLS_CSV.stat().st_size / 1e6
    print(f"      saved {written:,} rows -> {CALLS_CSV} ({size_mb:.1f} MB)")


def download_stations(force: bool = False) -> None:
    """Seattle's 36 real fire stations (the benchmark for our placement)."""
    if STATIONS_CSV.exists() and not force:
        print(f"[2/3] Fire stations already present: {STATIONS_CSV}")
        return

    print("[2/3] Seattle Fire Stations (Seattle City GIS)")
    params = {
        "where": "1=1",
        "outFields": "STNID,ADDRESS",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "json",
    }
    url = f"{ARCGIS_SEATTLE}/Fire_Stations/FeatureServer/0/query?" + urllib.parse.urlencode(params)
    data = json.loads(fetch(url))

    with STATIONS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["station_id", "address", "latitude", "longitude"])
        for feat in data.get("features", []):
            attrs, geom = feat["attributes"], feat.get("geometry") or {}
            if geom.get("y") is None or geom.get("x") is None:
                continue
            writer.writerow(
                [attrs.get("STNID"), attrs.get("ADDRESS"), geom["y"], geom["x"]]
            )
    n = sum(1 for _ in STATIONS_CSV.open()) - 1
    print(f"      saved {n} stations -> {STATIONS_CSV}")


def download_zcta(force: bool = False) -> None:
    """ZIP polygons so each incident can be tagged with its zone."""
    if ZCTA_GEOJSON.exists() and not force:
        print(f"[3/3] ZIP boundaries already present: {ZCTA_GEOJSON}")
        return

    print("[3/3] ZIP Code Tabulation Areas (US Census TIGERweb)")
    params = {
        "where": "1=1",
        "geometry": json.dumps({**SEATTLE_BBOX, "spatialReference": {"wkid": 4326}}),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "ZCTA5,GEOID",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }
    url = f"{TIGERWEB_ZCTA}/query?" + urllib.parse.urlencode(params)
    gj = json.loads(fetch(url))
    ZCTA_GEOJSON.write_text(json.dumps(gj), encoding="utf-8")
    print(f"      saved {len(gj.get('features', []))} ZIP polygons -> {ZCTA_GEOJSON}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--limit", type=int, default=None,
        help="cap the number of 911 records (default: the full ~2.2M dataset)",
    )
    ap.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    started = time.time()
    download_stations(force=args.force)
    download_zcta(force=args.force)
    download_calls(limit=args.limit, force=args.force)
    print(f"\nDone in {time.time() - started:.0f}s. Raw data in {RAW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
