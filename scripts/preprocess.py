"""Clean the raw 911 feed and cache it as Parquet.

Run once after ``scripts/download_data.py``. The expensive part is the
point-in-polygon ZIP join over ~2.2M incidents; caching it means the Streamlit
app starts in seconds rather than minutes.

Run:  python scripts/preprocess.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_loader  # noqa: E402


def main() -> int:
    started = time.time()
    df = data_loader.build_processed()

    print("\n--- summary -------------------------------------------------")
    print(f"  incidents kept        : {len(df):,}")
    print(f"  date range            : {df['datetime'].min()}  ->  {df['datetime'].max()}")
    print(f"  ZIP zones             : {df['zip_code'].nunique()}")
    print(f"  distinct call types   : {df['type'].nunique()}")
    print(f"  core EMS/aid calls    : {int(df['is_ems_core'].sum()):,}")
    print(f"  incl. trauma/rescue   : {int(df['is_ems_trauma'].sum()):,}")
    print(f"\nDone in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
