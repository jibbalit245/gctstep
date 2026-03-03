"""
utils/questdb_client.py
QuestDB helper — table creation, writes, and common queries.
QuestDB runs locally via Docker. See README for setup.

Usage:
    from utils.questdb_client import QuestDBClient
    qdb = QuestDBClient()
    qdb.create_tables()
    qdb.write_bars(df)
"""

import pandas as pd
import psycopg2
import questdb.ingress as qi
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import QUESTDB


class QuestDBClient:
    """Thin wrapper around QuestDB HTTP + Postgres wire protocol."""

    def __init__(self):
        self.pg_conn_str = (
            f"host={QUESTDB['host']} port={QUESTDB['pg_port']} "
            f"user={QUESTDB['user']} password={QUESTDB['password']} "
            f"dbname=qdb"
        )
        self.ilp_uri = (
            f"http::addr={QUESTDB['host']}:{QUESTDB['http_port']};"
            f"username={QUESTDB['user']};password={QUESTDB['password']};"
        )

    # ── Schema ────────────────────────────────────────────────────────────────

    def create_tables(self):
        """Create all required tables. Safe to run multiple times (IF NOT EXISTS)."""
        statements = [
            """
            CREATE TABLE IF NOT EXISTS gc_bars (
                ts          TIMESTAMP,
                open        DOUBLE,
                high        DOUBLE,
                low         DOUBLE,
                close       DOUBLE,
                volume      LONG,
                contract    SYMBOL CAPACITY 10
            ) TIMESTAMP(ts) PARTITION BY MONTH WAL;
            """,
            """
            CREATE TABLE IF NOT EXISTS gc_features (
                ts              TIMESTAMP,
                rsi_14          DOUBLE,
                atr_14          DOUBLE,
                ema_diff_20_50  DOUBLE,
                vol_regime      DOUBLE,
                dxy_ret         DOUBLE,
                real_yield      DOUBLE,
                vix_level       DOUBLE,
                cot_nc_net_pct  DOUBLE,
                lgbm_prob       DOUBLE,
                lstm_prob       DOUBLE,
                ensemble_signal INT
            ) TIMESTAMP(ts) PARTITION BY MONTH WAL;
            """,
            """
            CREATE TABLE IF NOT EXISTS gc_trades (
                ts              TIMESTAMP,
                direction       INT,
                contracts       INT,
                entry_price     DOUBLE,
                exit_price      DOUBLE,
                pnl             DOUBLE,
                session_pnl     DOUBLE,
                total_pnl       DOUBLE
            ) TIMESTAMP(ts) PARTITION BY MONTH WAL;
            """,
        ]

        conn = psycopg2.connect(self.pg_conn_str)
        cur  = conn.cursor()
        for stmt in statements:
            cur.execute(stmt)
        conn.commit()
        cur.close()
        conn.close()
        logger.success("QuestDB tables created (or already exist)")

    # ── Writes ────────────────────────────────────────────────────────────────

    def write_bars(self, df: pd.DataFrame, contract: str = "GC"):
        """
        Write OHLCV bars to QuestDB using ILP (fast ingress protocol).
        df must have DatetimeIndex (UTC) and columns: open, high, low, close, volume
        """
        with qi.Sender.from_uri(self.ilp_uri) as sender:
            for ts, row in df.iterrows():
                sender.row(
                    "gc_bars",
                    symbols={"contract": contract},
                    columns={
                        "open":   float(row["open"]),
                        "high":   float(row["high"]),
                        "low":    float(row["low"]),
                        "close":  float(row["close"]),
                        "volume": int(row["volume"]),
                    },
                    at=qi.TimestampNanos(int(ts.value)),
                )
        logger.info(f"Wrote {len(df):,} bars to QuestDB gc_bars")

    def write_feature_row(self, ts, features: dict, signal: int):
        """Write a single feature + signal row (called on each bar)."""
        with qi.Sender.from_uri(self.ilp_uri) as sender:
            sender.row(
                "gc_features",
                columns={**{k: float(v) for k, v in features.items()},
                         "ensemble_signal": signal},
                at=qi.TimestampNanos(int(pd.Timestamp(ts).value)),
            )

    def write_trade(self, ts, direction: int, contracts: int,
                    entry: float, exit_price: float,
                    pnl: float, session_pnl: float, total_pnl: float):
        """Log a completed trade."""
        with qi.Sender.from_uri(self.ilp_uri) as sender:
            sender.row(
                "gc_trades",
                columns={
                    "direction":    direction,
                    "contracts":    contracts,
                    "entry_price":  entry,
                    "exit_price":   exit_price,
                    "pnl":          pnl,
                    "session_pnl":  session_pnl,
                    "total_pnl":    total_pnl,
                },
                at=qi.TimestampNanos(int(pd.Timestamp(ts).value)),
            )

    # ── Queries ───────────────────────────────────────────────────────────────

    def query(self, sql: str) -> pd.DataFrame:
        """Run any SQL query and return a DataFrame."""
        conn = psycopg2.connect(self.pg_conn_str)
        df   = pd.read_sql(sql, conn)
        conn.close()
        return df

    def get_recent_bars(self, n: int = 300) -> pd.DataFrame:
        """Get last N bars for ML inference."""
        df = self.query(f"""
            SELECT ts, open, high, low, close, volume
            FROM gc_bars
            ORDER BY ts DESC
            LIMIT {n}
        """)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts").sort_index()
        return df

    def get_performance_summary(self) -> dict:
        """Rolling performance metrics from logged trades."""
        df = self.query("""
            SELECT
                count(*)           as total_trades,
                sum(pnl)           as total_pnl,
                avg(pnl)           as avg_pnl,
                max(total_pnl)     as peak_equity_gain,
                min(total_pnl)     as max_drawdown_point,
                sum(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winners
            FROM gc_trades
        """)
        if df.empty:
            return {}
        row = df.iloc[0].to_dict()
        row["win_rate"] = row["winners"] / max(row["total_trades"], 1)
        return row

    def health_check(self) -> bool:
        """Verify QuestDB is running and accessible."""
        try:
            self.query("SELECT 1")
            logger.info("QuestDB: connected ✓")
            return True
        except Exception as e:
            logger.error(f"QuestDB connection failed: {e}")
            logger.error("Is QuestDB running? Try: docker start questdb")
            return False


if __name__ == "__main__":
    client = QuestDBClient()
    if client.health_check():
        client.create_tables()
