"""
data/fetch_alt_data.py
Fetches free alternative data used as ML features:
  - FRED: DXY, real yields (TIPS), VIX, Fed Funds Rate
  - CFTC: Commitments of Traders (COT) reports
  - Yahoo Finance: Silver (for gold/silver ratio), VIX

Usage:
    python data/fetch_alt_data.py
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FRED_API_KEY, DATA, BASE_DIR

CACHE_DIR = BASE_DIR / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

START = DATA["history_start"]


def fetch_fred_data() -> pd.DataFrame:
    """
    Pull macro indicators from FRED (free).
    These are the strongest predictors for gold direction.
    """
    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)

    series = {
        "dxy":        "DTWEXBGS",    # USD broad index (best DXY proxy from FRED)
        "tips_10y":   "DFII10",      # 10Y real yield (most important for gold)
        "breakeven":  "T10YIE",      # 10Y breakeven inflation
        "fed_funds":  "FEDFUNDS",    # Fed Funds effective rate
        "gvz":        "GVZCLS",      # CBOE Gold Volatility Index
        "credit_spread": "BAMLH0A0HYM2",  # HY credit spread (risk-off proxy)
    }

    dfs = {}
    for name, series_id in series.items():
        logger.info(f"Fetching FRED: {name} ({series_id})")
        try:
            s = fred.get_series(series_id, observation_start=START)
            dfs[name] = s
        except Exception as e:
            logger.warning(f"Failed to fetch {name}: {e}")

    macro = pd.DataFrame(dfs)
    macro.index = pd.to_datetime(macro.index, utc=True)
    macro = macro.sort_index().ffill()  # Forward-fill weekends/holidays
    macro.to_parquet(CACHE_DIR / "macro_fred.parquet")
    logger.success(f"FRED data saved: {len(macro):,} rows")
    return macro


def fetch_yfinance_data() -> pd.DataFrame:
    """
    Pull market data from Yahoo Finance.
    Free, no API key needed.
    """
    tickers = {
        "silver":    "SI=F",      # Silver futures (gold/silver ratio)
        "vix":       "^VIX",      # VIX — risk sentiment
        "oil":       "CL=F",      # Crude oil (inflationary pressure)
        "gld":       "GLD",       # Gold ETF (retail positioning proxy)
        "spy":       "SPY",       # S&P 500 (risk-on/off context)
        "tlt":       "TLT",       # 20Y Treasury ETF
    }

    dfs = {}
    for name, ticker in tickers.items():
        logger.info(f"Fetching Yahoo Finance: {name} ({ticker})")
        try:
            df = yf.download(ticker, start=START, auto_adjust=True, progress=False)
            dfs[name] = df["Close"]
        except Exception as e:
            logger.warning(f"Failed to fetch {name}: {e}")

    market = pd.DataFrame(dfs)
    market.index = pd.to_datetime(market.index, utc=True)
    market = market.sort_index().ffill()
    market.to_parquet(CACHE_DIR / "market_yfinance.parquet")
    logger.success(f"Yahoo Finance data saved: {len(market):,} rows")
    return market


def fetch_cot_data() -> pd.DataFrame:
    """
    Pull CFTC Commitments of Traders report.
    Gold code: 088691 (COMEX Gold 100 Troy Oz)
    Released every Friday at 3:30 PM ET for the prior Tuesday.
    """
    import cot_reports as cot

    logger.info("Fetching COT reports (this may take a minute)...")
    try:
        # Get legacy futures-only report
        df = cot.cot_all(cot_report_type="legacy_fut")

        # Filter for gold
        gold_code = "088691"
        gold_cot = df[df["CFTC_Commodity_Code"] == gold_code].copy()

        # Key columns we care about
        cols = {
            "Report_Date_as_YYYY-MM-DD": "date",
            "NonComm_Positions_Long_All":  "nc_long",
            "NonComm_Positions_Short_All": "nc_short",
            "Comm_Positions_Long_All":     "comm_long",
            "Comm_Positions_Short_All":    "comm_short",
            "Open_Interest_All":           "open_interest",
        }

        gold_cot = gold_cot.rename(columns=cols)[list(cols.values())]
        gold_cot["date"] = pd.to_datetime(gold_cot["date"], utc=True)
        gold_cot = gold_cot.set_index("date").sort_index()

        # Derived features
        gold_cot["nc_net"] = gold_cot["nc_long"] - gold_cot["nc_short"]
        gold_cot["comm_net"] = gold_cot["comm_long"] - gold_cot["comm_short"]
        gold_cot["nc_net_pct"] = gold_cot["nc_net"].rank(pct=True)  # 0-1 percentile

        gold_cot.to_parquet(CACHE_DIR / "cot_gold.parquet")
        logger.success(f"COT data saved: {len(gold_cot):,} weeks")
        return gold_cot

    except Exception as e:
        logger.error(f"COT fetch failed: {e}")
        logger.info("Returning empty COT dataframe — features will be NaN (handled in pipeline)")
        return pd.DataFrame()


def load_all_alt_data() -> pd.DataFrame:
    """
    Load all alt data from cache and merge into a single daily DataFrame.
    Call this from features.py to get macro features.
    """
    files = {
        "macro": CACHE_DIR / "macro_fred.parquet",
        "market": CACHE_DIR / "market_yfinance.parquet",
        "cot": CACHE_DIR / "cot_gold.parquet",
    }

    dfs = {}
    for name, path in files.items():
        if path.exists():
            dfs[name] = pd.read_parquet(path)
        else:
            logger.warning(f"{name} cache not found at {path} — run fetch_alt_data.py first")

    if not dfs:
        return pd.DataFrame()

    # Merge all on date index, forward-fill gaps (COT is weekly)
    combined = pd.concat(dfs.values(), axis=1)
    combined = combined.sort_index().ffill()
    return combined


if __name__ == "__main__":
    logger.info("Fetching all alternative data sources...")
    fetch_fred_data()
    fetch_yfinance_data()
    fetch_cot_data()
    logger.success("All alternative data cached to data/cache/")
