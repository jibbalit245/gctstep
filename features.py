"""
execution/topstep_bot.py
Live trading bot for GC gold futures via TopstepX / ProjectX API.
Runs on your LOCAL machine — Topstep prohibits VPS/cloud execution.

Usage:
    python execution/topstep_bot.py --mode demo    # Paper trading (default)
    python execution/topstep_bot.py --mode live    # Real money — be careful
"""

import asyncio
import argparse
import pandas as pd
import databento as db
from collections import deque
from datetime import datetime
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (TOPSTEP_USERNAME, TOPSTEP_API_KEY, TRADING_MODE,
                    PROJECTX, GC_CONTRACT, RISK, DATABENTO_API_KEY, DATA)
from models.ensemble import SignalEngine
from execution.risk_guard import TopstepRiskGuard, TradingHaltedError
from data.fetch_alt_data import load_all_alt_data

# Rolling bar buffer — keeps last N bars for feature computation
BAR_BUFFER_SIZE = 300  # Need at least seq_len (48) + warmup (50) + buffer


class GoldTradingBot:
    """
    Main bot loop:
      1. Subscribe to Databento live GC ticks
      2. Aggregate into 1H bars
      3. On each completed bar → run ML signal
      4. Execute via ProjectX API (respecting all Topstep rules)
    """

    def __init__(self, mode: str = "demo"):
        self.mode       = mode
        self.bar_buffer = deque(maxlen=BAR_BUFFER_SIZE)
        self.current_bar = None
        self.open_position = None         # Current position (contracts, direction)
        self.signal_engine = SignalEngine()
        self.risk_guard    = TopstepRiskGuard(account_balance=100_000)
        self.df_alt        = load_all_alt_data()
        self.client        = None          # ProjectX REST client
        self.account_id    = None
        self.contract_id   = None

        logger.info(f"Bot initialized | Mode: {self.mode.upper()}")
        if mode == "live":
            logger.warning("⚠️  LIVE MODE — real money will be traded")

    # ── Startup ───────────────────────────────────────────────────────────────

    async def connect(self):
        """Authenticate with ProjectX API and get account/contract IDs."""
        from project_x_py import ProjectX
        from project_x_py.models import ProjectXConfig

        config = ProjectXConfig(**PROJECTX)

        logger.info("Connecting to TopstepX / ProjectX API...")
        self.client = await ProjectX.from_credentials(
            username=TOPSTEP_USERNAME,
            api_key=TOPSTEP_API_KEY,
            config=config,
        ).__aenter__()

        await self.client.authenticate()

        accounts = await self.client.search_accounts()
        if not accounts:
            raise RuntimeError("No accounts found. Check your API credentials.")
        self.account_id = accounts[0].id
        logger.info(f"Account: {accounts[0].name} (ID: {self.account_id})")

        # Find GC front month contract
        contracts = await self.client.search_contracts(
            text=GC_CONTRACT["symbol_px"])
        if not contracts:
            raise RuntimeError("GC contract not found via ProjectX API.")
        self.contract_id = contracts[0].id
        logger.info(f"Contract: {contracts[0].name} (ID: {self.contract_id})")

    # ── Market Data ───────────────────────────────────────────────────────────

    async def start_data_feed(self):
        """Subscribe to Databento live GC tick feed and aggregate into bars."""
        logger.info("Starting Databento live feed for GC...")

        live_client = db.Live(DATABENTO_API_KEY)
        live_client.subscribe(
            dataset=DATA["dataset"],
            schema="trades",
            stype_in="continuous",
            symbols=[DATA["symbol"]],
        )

        current_hour = None
        bar_trades = []

        for record in live_client:
            ts = pd.Timestamp(record.ts_event, unit="ns", tz="UTC")
            hour = ts.floor("1H")

            if current_hour is None:
                current_hour = hour

            if hour != current_hour:
                # Bar completed — build OHLCV and process
                if bar_trades:
                    prices  = [t["price"] for t in bar_trades]
                    volumes = [t["size"]  for t in bar_trades]
                    bar = {
                        "timestamp": current_hour,
                        "open":   prices[0],
                        "high":   max(prices),
                        "low":    min(prices),
                        "close":  prices[-1],
                        "volume": sum(volumes),
                    }
                    self.bar_buffer.append(bar)
                    await self.on_bar_close(bar)

                current_hour = hour
                bar_trades = []

            bar_trades.append({"price": record.price / 1e9, "size": record.size})

    # ── Signal + Execution ────────────────────────────────────────────────────

    async def on_bar_close(self, bar: dict):
        """Called on every completed 1H bar. Core decision loop."""
        logger.debug(f"Bar closed: {bar['timestamp']} | "
                     f"O:{bar['open']:.2f} H:{bar['high']:.2f} "
                     f"L:{bar['low']:.2f} C:{bar['close']:.2f}")

        # Force close check
        if self.risk_guard.should_close_soon:
            logger.info("Approaching session close — flattening all positions")
            await self.close_position("Session close")
            return

        # Risk check
        try:
            self.risk_guard.check_all()
        except TradingHaltedError as e:
            logger.critical(f"Risk halt: {e}")
            await self.close_position("Risk halt")
            return

        # Need enough bars for the ML model
        if len(self.bar_buffer) < 100:
            logger.debug(f"Warming up: {len(self.bar_buffer)}/100 bars")
            return

        # Build DataFrame from buffer
        df = pd.DataFrame(list(self.bar_buffer)).set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)

        # Get ML signal
        signal_info = self.signal_engine.signal_strength(df, self.df_alt)
        signal = signal_info["signal"]

        logger.info(f"Signal: {signal:+d} | "
                    f"Prob: {signal_info['probability']:.3f} | "
                    f"Confidence: {signal_info['confidence']} | "
                    f"Risk: {self.risk_guard.status['session_pnl']:.2f}")

        # Execute
        await self.execute_signal(signal, bar["close"])

    async def execute_signal(self, signal: int, current_price: float):
        """Place or close orders based on signal."""
        pos = self.open_position

        if signal == 1 and pos is None:
            await self.open_long(current_price)

        elif signal == -1 and pos is None:
            await self.open_short(current_price)

        elif signal == 0 and pos is not None:
            await self.close_position("Signal went flat")

        elif pos is not None and signal != pos["direction"]:
            # Reversal — close then re-enter
            await self.close_position("Signal reversal")
            if signal == 1:
                await self.open_long(current_price)
            elif signal == -1:
                await self.open_short(current_price)

    # ── Order Helpers ─────────────────────────────────────────────────────────

    async def open_long(self, price: float):
        contracts = self._size_position(price)
        logger.info(f"LONG {contracts} GC @ ~{price:.2f}")
        await self._place_order("Buy", contracts, price)
        self.open_position = {"direction": 1, "contracts": contracts,
                               "entry_price": price}

    async def open_short(self, price: float):
        contracts = self._size_position(price)
        logger.info(f"SHORT {contracts} GC @ ~{price:.2f}")
        await self._place_order("Sell", contracts, price)
        self.open_position = {"direction": -1, "contracts": contracts,
                               "entry_price": price}

    async def close_position(self, reason: str = ""):
        if self.open_position is None:
            return
        pos = self.open_position
        action = "Sell" if pos["direction"] == 1 else "Buy"
        logger.info(f"CLOSE {pos['contracts']} GC ({reason})")
        await self._place_order(action, pos["contracts"])
        self.open_position = None

    async def _place_order(self, action: str, quantity: int, price: float = None):
        """Submit order to TopstepX. Skips in demo mode."""
        if self.mode == "demo":
            logger.info(f"[DEMO] {action} {quantity} contracts @ "
                        f"{'market' if price is None else price:.2f}")
            return

        if self.client is None:
            logger.error("Not connected to ProjectX API")
            return

        try:
            await self.client.place_order(
                account_id=self.account_id,
                contract_id=self.contract_id,
                action=action,
                order_type="Market",
                quantity=quantity,
            )
            logger.success(f"Order submitted: {action} {quantity} GC")
        except Exception as e:
            logger.error(f"Order failed: {e}")

    # ── Position Sizing ───────────────────────────────────────────────────────

    def _size_position(self, price: float) -> int:
        """
        ATR-based position sizing.
        Risk 1% of account per trade, stop = 1.5x ATR.
        """
        if len(self.bar_buffer) < 20:
            return RISK["min_contracts"]

        df = pd.DataFrame(list(self.bar_buffer)).set_index("timestamp")
        atr = _compute_atr(df, period=14)

        account = 100_000  # Track actual balance from API in production
        risk_dollars = account * RISK["risk_pct_per_trade"]
        stop_dollars = atr * RISK["atr_multiplier"] * GC_CONTRACT["point_value"]

        contracts = int(risk_dollars / max(stop_dollars, 1))
        contracts = max(RISK["min_contracts"],
                        min(contracts, RISK["max_contracts"]))

        logger.debug(f"Sizing: ATR={atr:.2f} | Risk=${risk_dollars:.0f} | "
                     f"Stop=${stop_dollars:.0f} | Contracts={contracts}")
        return contracts

    # ── Main Run ──────────────────────────────────────────────────────────────

    async def run(self):
        if self.mode == "live":
            await self.connect()

        logger.info("Starting data feed...")
        await self.start_data_feed()


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Simple ATR calculation from OHLCV DataFrame."""
    import pandas_ta as ta
    if len(df) < period + 1:
        return df["close"].iloc[-1] * 0.005  # Fallback: 0.5% of price
    atr_series = ta.atr(df["high"], df["low"], df["close"], period)
    return float(atr_series.iloc[-1]) if atr_series is not None else 10.0


# ── Entry Point ───────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="GC Gold Futures Trading Bot")
    parser.add_argument("--mode", choices=["demo", "live"], default="demo",
                        help="demo = paper trading, live = real money")
    args = parser.parse_args()

    mode = args.mode or TRADING_MODE
    logger.info(f"Starting GC bot in {mode.upper()} mode")

    bot = GoldTradingBot(mode=mode)

    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
        await bot.close_position("Manual shutdown")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        await bot.close_position("Fatal error")


if __name__ == "__main__":
    asyncio.run(main())
