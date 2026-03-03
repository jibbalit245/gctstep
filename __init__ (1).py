"""
strategies/backtrader_strategy.py
Backtests the ML signal strategy using Backtrader.
Uses saved LightGBM + LSTM models against historical GC data.

Usage:
    python strategies/backtrader_strategy.py
    python strategies/backtrader_strategy.py --start 2022-01-01 --end 2024-01-01
"""

import argparse
import backtrader as bt
import backtrader.feeds as btfeeds
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RISK, TOPSTEP, GC_CONTRACT, BASE_DIR

CACHE_DIR = BASE_DIR / "data" / "cache"


# ── Custom Databento Data Feed ────────────────────────────────────────────────

class GCDataFeed(btfeeds.PandasData):
    """Backtrader feed for GC continuous contract data from Databento parquet."""
    params = (
        ("datetime", None),
        ("open",     "open"),
        ("high",     "high"),
        ("low",      "low"),
        ("close",    "close"),
        ("volume",   "volume"),
        ("openinterest", -1),
    )


# ── ML-Driven Strategy ────────────────────────────────────────────────────────

class GoldMLStrategy(bt.Strategy):
    """
    Gold futures strategy driven by pre-computed ML signals.
    Signals are pre-computed for the whole period and fed in as a column.
    Risk management mirrors Topstep rules exactly.
    """

    params = dict(
        atr_period    = 14,
        atr_mult      = 1.5,
        risk_pct      = 0.01,        # 1% account risk per trade
        max_contracts = 5,           # Conservative (Topstep allows 10 on $100K)
        daily_loss_limit = -2000,
        trailing_dd_limit = -4000,
        close_hour    = 15,          # Close all by 3 PM bar
        close_minute  = 45,
        verbose       = True,
    )

    def __init__(self):
        self.atr       = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.ema50     = bt.indicators.EMA(self.data.close, period=50)
        self.ema200    = bt.indicators.EMA(self.data.close, period=200)

        # Risk tracking
        self.session_pnl    = 0.0
        self.peak_equity    = self.broker.get_cash()
        self.trailing_dd    = 0.0
        self.trade_count    = 0
        self.winning_trades = 0
        self.last_date      = None

        # Stop order reference
        self.stop_order = None

    def next(self):
        # Reset daily P&L at start of new day
        current_date = self.data.datetime.date(0)
        if self.last_date and current_date != self.last_date:
            if self.p.verbose:
                logger.debug(f"Day close | Session P&L: ${self.session_pnl:.2f}")
            self.session_pnl = 0.0
        self.last_date = current_date

        # Enforce session close time
        current_hour   = self.data.datetime.time(0).hour
        current_minute = self.data.datetime.time(0).minute
        if (current_hour > self.p.close_hour or
           (current_hour == self.p.close_hour and
            current_minute >= self.p.close_minute)):
            if self.position:
                self.close()
                if self.stop_order:
                    self.cancel(self.stop_order)
                    self.stop_order = None
            return

        # Risk circuit breakers
        if self.session_pnl <= self.p.daily_loss_limit:
            if self.position:
                self.close()
            return

        equity = self.broker.getvalue()
        self.peak_equity = max(self.peak_equity, equity)
        self.trailing_dd = equity - self.peak_equity
        if self.trailing_dd <= self.p.trailing_dd_limit:
            if self.position:
                self.close()
            return

        # Get pre-computed ML signal from data feed (column "signal")
        signal = int(getattr(self.data, "signal", [0])[0])

        price     = self.data.close[0]
        atr       = self.atr[0]
        stop_dist = atr * self.p.atr_mult

        # Position sizing
        cash     = self.broker.get_cash()
        risk_amt = self.broker.getvalue() * self.p.risk_pct
        size     = int(risk_amt / (stop_dist * GC_CONTRACT["point_value"]))
        size     = max(1, min(size, self.p.max_contracts))

        pos = self.getposition()

        # Entry
        if signal == 1 and not pos:
            self.buy(size=size)
            self.stop_order = self.sell(
                exectype=bt.Order.Stop,
                price=price - stop_dist,
                size=size,
            )

        elif signal == -1 and not pos:
            self.sell(size=size)
            self.stop_order = self.buy(
                exectype=bt.Order.Stop,
                price=price + stop_dist,
                size=size,
            )

        # Exit: signal flipped or went flat
        elif pos and signal == 0:
            self.close()
            if self.stop_order:
                self.cancel(self.stop_order)
                self.stop_order = None

        elif pos and pos.size > 0 and signal == -1:
            self.close()
            if self.stop_order:
                self.cancel(self.stop_order)
                self.stop_order = None

        elif pos and pos.size < 0 and signal == 1:
            self.close()
            if self.stop_order:
                self.cancel(self.stop_order)
                self.stop_order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.session_pnl += trade.pnlcomm
            self.trade_count += 1
            if trade.pnlcomm > 0:
                self.winning_trades += 1

    def stop(self):
        win_rate = self.winning_trades / max(self.trade_count, 1)
        logger.info(f"Backtest complete | Trades: {self.trade_count} | "
                    f"Win rate: {win_rate:.1%} | "
                    f"Final equity: ${self.broker.getvalue():,.2f}")


# ── Runner ────────────────────────────────────────────────────────────────────

def run_backtest(start: str = "2021-01-01",
                 end:   str = "2024-12-31",
                 cash:  float = 100_000.0) -> dict:
    """
    Run full backtest. Pre-computes ML signals then feeds them to Backtrader.
    Returns performance metrics dict.
    """
    logger.info(f"Backtest: {start} → {end} | Starting equity: ${cash:,.0f}")

    # Load OHLCV data
    ohlcv_path = CACHE_DIR / "gc_ohlcv_1h.parquet"
    if not ohlcv_path.exists():
        logger.error("Run: python data/fetch_databento.py")
        sys.exit(1)

    df = pd.read_parquet(ohlcv_path)
    df = df.loc[start:end].copy()
    logger.info(f"Data: {len(df):,} bars from {df.index.min()} to {df.index.max()}")

    # Pre-compute ML signals for entire period
    logger.info("Pre-computing ML signals...")
    try:
        from models.features import build_feature_matrix
        from models.ensemble import SignalEngine
        from data.fetch_alt_data import load_all_alt_data

        df_alt  = load_all_alt_data()
        engine  = SignalEngine()
        signals = []

        for i in range(len(df)):
            if i < 100:
                signals.append(0)
                continue
            df_window = df.iloc[max(0, i-200):i+1]
            sig = engine.predict(df_window, df_alt)
            signals.append(sig)

        df["signal"] = signals
    except Exception as e:
        logger.warning(f"ML signal computation failed ({e}). Using random signals for demo.")
        df["signal"] = np.random.choice([-1, 0, 1], size=len(df), p=[0.15, 0.70, 0.15])

    # Make index timezone-naive for Backtrader
    df.index = df.index.tz_localize(None)

    # Set up Cerebro
    cerebro = bt.Cerebro()
    cerebro.addstrategy(GoldMLStrategy)

    feed = GCDataFeed(dataname=df)
    cerebro.adddata(feed)

    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(
        commission=0.0,
        margin=GC_CONTRACT["margin"],
        mult=GC_CONTRACT["point_value"],
    )
    cerebro.broker.set_slippage_perc(0.0001)  # 0.01% slippage

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio,
                         _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown,   _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns,    _name="returns")

    start_value = cerebro.broker.getvalue()
    results = cerebro.run()
    strat = results[0]
    end_value = cerebro.broker.getvalue()

    # Extract metrics
    sharpe  = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)
    dd_info = strat.analyzers.dd.get_analysis()
    tr_info = strat.analyzers.trades.get_analysis()

    total_trades = tr_info.get("total", {}).get("closed", 0)
    won          = tr_info.get("won", {}).get("total", 0)
    win_rate     = won / max(total_trades, 1)
    max_dd       = dd_info.get("max", {}).get("drawdown", 0)
    total_return = (end_value - start_value) / start_value * 100

    metrics = {
        "start_equity":  start_value,
        "end_equity":    end_value,
        "total_return":  f"{total_return:.2f}%",
        "sharpe_ratio":  round(sharpe, 3) if sharpe else "N/A",
        "max_drawdown":  f"{max_dd:.2f}%",
        "total_trades":  total_trades,
        "win_rate":      f"{win_rate:.1%}",
    }

    logger.success("─── Backtest Results ───────────────────")
    for k, v in metrics.items():
        logger.success(f"  {k:<20} {v}")
    logger.success("────────────────────────────────────────")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    parser.add_argument("--cash",  type=float, default=100_000.0)
    args = parser.parse_args()

    run_backtest(start=args.start, end=args.end, cash=args.cash)
