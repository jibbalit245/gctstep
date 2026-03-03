"""
data/fetch_databento.py
Pulls CME GC continuous contract data from Databento into a local parquet cache.
Run this on Paperspace before training, or locally for incremental updates.

Usage:
    python data/fetch_databento.py                  # Full historical pull
    python data/fetch_databento.py --incremental    # Last 30 days only
"""

import argparse
import databento as db
import pandas as pd
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATABENTO_API_KEY, DATA, BASE_DIR

CACHE_DIR = BASE_DIR / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ohlcv(start: str, end: str = None, schema: str = "ohlcv-1h") -> pd.DataFrame:
    """
    Pull GC continuous contract OHLCV bars from Databento.
    Uses open-interest roll (GC.n.0) — best for tracking liquidity.
    """
    logger.info(f"Fetching {schema} data from {start} to {end or 'now'}")

    client = db.Historical(DATABENTO_API_KEY)

    data = client.timeseries.get_range(
        dataset=DATA["dataset"],        # GLBX.MDP3
        schema=schema,
        stype_in="continuous",
        symbols=[DATA["symbol"]],       # GC.n.0
        start=start,
        end=end,
    )

    df = data.to_df()

    # Rename to standard column names
    df = df.rename(columns={
        "ts_event": "timestamp",
        "open":  "open",
        "high":  "high",
        "low":   "low",
        "close": "close",
        "volume":"volume",
    })

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    df = df[["open", "high", "low", "close", "volume"]].dropna()

    logger.info(f"Fetched {len(df):,} bars ({df.index.min()} → {df.index.max()})")
    return df


def fetch_tick_data(start: str, end: str) -> pd.DataFrame:
    """
    Pull raw trade ticks for order flow imbalance features.
    WARNING: Tick data is large. Keep date ranges small.
    """
    logger.info(f"Fetching tick data {start} to {end}")

    client = db.Historical(DATABENTO_API_KEY)
    data = client.timeseries.get_range(
        dataset=DATA["dataset"],
        schema="trades",
        stype_in="continuous",
        symbols=[DATA["symbol"]],
        start=start,
        end=end,
    )

    df = data.to_df()
    df["timestamp"] = pd.to_datetime(df["ts_event"], utc=True)
    df = df.set_index("timestamp")
    return df[["price", "size", "side"]].dropna()


def main():
    parser = argparse.ArgumentParser(description="Fetch GC data from Databento")
    parser.add_argument("--incremental", action="store_true",
                        help="Only fetch last 30 days (faster, for updates)")
    parser.add_argument("--schema", default="ohlcv-1h",
                        choices=["ohlcv-1h", "ohlcv-5m", "ohlcv-1d"],
                        help="Bar resolution to fetch")
    args = parser.parse_args()

    cache_file = CACHE_DIR / f"gc_{args.schema.replace('-', '_')}.parquet"

    if args.incremental and cache_file.exists():
        # Load existing and find last date
        existing = pd.read_parquet(cache_file)
        start = (existing.index.max() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        logger.info(f"Incremental update from {start}")
    else:
        start = DATA["history_start"]
        existing = None
        logger.info(f"Full historical pull from {start}")

    new_data = fetch_ohlcv(start=start, schema=args.schema)

    if existing is not None:
        # Merge and deduplicate
        combined = pd.concat([existing, new_data])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
    else:
        combined = new_data

    combined.to_parquet(cache_file)
    logger.success(f"Saved {len(combined):,} bars to {cache_file}")

    # Also save daily bars for macro context
    if args.schema != "ohlcv-1d":
        daily_file = CACHE_DIR / "gc_ohlcv_1d.parquet"
        if not daily_file.exists():
            logger.info("Also pulling daily bars for macro context...")
            daily = fetch_ohlcv(start=DATA["history_start"], schema="ohlcv-1d")
            daily.to_parquet(daily_file)
            logger.success(f"Daily bars saved: {len(daily):,} rows")


if __name__ == "__main__":
    main()
