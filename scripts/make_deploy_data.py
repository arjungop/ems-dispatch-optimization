"""Build the committed dataset the hosted app runs on.

The full pipeline (`download_data.py` + `preprocess.py`) produces 2.2M calls
back to 2010: 61 MB of Parquet that peaks around 615 MB resident once pandas
has it in memory. Neither the 191 MB source CSV nor that Parquet is in the
repository, and a free Streamlit container has roughly 1 GB for the whole
process, so a deployment needs a smaller copy of the same table.

This writes it. The slice is the most recent ``config.DEPLOY_YEARS`` of the
*already cleaned* incident table -- same columns, same dtypes, same ZIP zones,
just fewer rows -- plus the two raw reference files that are small enough to
commit unchanged. Nothing is recomputed, so the deployed app cannot disagree
with the local one about any period they both cover.

Run:  python scripts/make_deploy_data.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

# Reference files small enough to ship as they are.
RAW_COPIES = ["seattle_fire_stations.csv", "seattle_zcta.geojson"]


def mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def main() -> int:
    full = config.CALLS_PARQUET_FULL
    if not full.exists():
        print(f"error: the full dataset is not here ({full}).\n"
              "       Run `python scripts/download_data.py` then "
              "`python scripts/preprocess.py` first.", file=sys.stderr)
        return 1

    config.DEPLOY.mkdir(parents=True, exist_ok=True)

    inc = pd.read_parquet(full)
    cutoff = inc["datetime"].max() - pd.DateOffset(years=config.DEPLOY_YEARS)
    sliced = inc[inc["datetime"] >= cutoff].reset_index(drop=True)

    out = config.DEPLOY / "ems_incidents.parquet"
    # zstd over the default snappy: a little slower to write, materially
    # smaller on disk, and read support ships with the pyarrow already pinned
    # in requirements.txt.
    sliced.to_parquet(out, compression="zstd", index=False)

    print(f"full   : {len(inc):>9,} rows  {mb(full):>6.1f} MB  "
          f"{inc['datetime'].min():%b %Y} -> {inc['datetime'].max():%b %Y}")
    print(f"deploy : {len(sliced):>9,} rows  {mb(out):>6.1f} MB  "
          f"{sliced['datetime'].min():%b %Y} -> {sliced['datetime'].max():%b %Y}")
    print(f"         in memory: {sliced.memory_usage(deep=True).sum() / 1e6:.0f} MB")

    missing = []
    for name in RAW_COPIES:
        src = config.RAW / name
        if not src.exists():
            missing.append(name)
            continue
        shutil.copy2(src, config.DEPLOY / name)
        print(f"copied : {name}  ({mb(config.DEPLOY / name):.2f} MB)")

    if missing:
        print(f"\nerror: missing raw reference file(s): {', '.join(missing)}.\n"
              "       Run `python scripts/download_data.py` first.",
              file=sys.stderr)
        return 1

    total = sum(mb(p) for p in config.DEPLOY.iterdir() if p.is_file())
    print(f"\n{config.DEPLOY.relative_to(config.ROOT)}/ is {total:.1f} MB — commit it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
