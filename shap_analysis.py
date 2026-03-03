"""
execution/risk_guard.py
Enforces ALL Topstep Trading Combine rules in real-time.
This runs before EVERY order. Violations = account closure.
"""

from datetime import datetime, time
import pytz
from loguru import logger
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TOPSTEP


class TradingHaltedError(Exception):
    """Raised when a Topstep rule violation is detected."""
    pass


class TopstepRiskGuard:
    """
    Stateful risk manager. Track session P&L and enforce all rules.

    Usage:
        guard = TopstepRiskGuard(account_balance=100_000)
        guard.check_all()                          # Before each order
        guard.update_pnl(realized_pnl=250.0)      # After each fill
    """

    CT = pytz.timezone("America/Chicago")

    def __init__(self, account_balance: float = 100_000.0):
        self.account_balance  = account_balance
        self.session_pnl      = 0.0
        self.total_pnl        = 0.0
        self.daily_pnl_log    = []   # List of daily P&Ls for consistency check
        self.is_halted        = False
        self.halt_reason      = ""

    # ── Core check — call before every order ──────────────────────────────────

    def check_all(self) -> None:
        """
        Run all rule checks. Raises TradingHaltedError if any rule is violated.
        Call this before placing any order.
        """
        if self.is_halted:
            raise TradingHaltedError(f"Trading halted: {self.halt_reason}")

        self._check_daily_loss()
        self._check_trailing_drawdown()
        self._check_close_time()
        self._check_weekend()

    def check_position_size(self, requested_contracts: int) -> int:
        """
        Validate and cap contract size to Topstep limits.
        Returns the allowed number of contracts.
        """
        max_allowed = TOPSTEP["max_contracts"]
        if requested_contracts > max_allowed:
            logger.warning(f"Contract cap: {requested_contracts} → {max_allowed}")
            return max_allowed
        return requested_contracts

    # ── State updates ─────────────────────────────────────────────────────────

    def update_pnl(self, realized_pnl: float) -> None:
        """Call after each trade fill with the P&L of that trade."""
        self.session_pnl += realized_pnl
        self.total_pnl   += realized_pnl
        logger.debug(f"P&L update: +${realized_pnl:.2f} | "
                     f"Session: ${self.session_pnl:.2f} | "
                     f"Total: ${self.total_pnl:.2f}")

    def end_of_day(self) -> None:
        """Call at session close (3:58 PM CT). Logs daily P&L."""
        self.daily_pnl_log.append(self.session_pnl)
        logger.info(f"Session closed | Day P&L: ${self.session_pnl:.2f} | "
                    f"Total P&L: ${self.total_pnl:.2f}")
        self.session_pnl = 0.0

    # ── Individual rule checks ─────────────────────────────────────────────────

    def _check_daily_loss(self) -> None:
        limit = TOPSTEP["daily_loss_limit"]
        if self.session_pnl <= limit:
            self._halt(f"Daily loss limit hit: ${self.session_pnl:.2f} "
                       f"(limit: ${limit:.2f})")

        # Warning buffer at 75% of limit
        warn_level = limit * 0.75
        if self.session_pnl <= warn_level:
            remaining = abs(limit) - abs(self.session_pnl)
            logger.warning(f"⚠️  Approaching daily loss limit | "
                           f"${remaining:.2f} remaining before halt")

    def _check_trailing_drawdown(self) -> None:
        limit = TOPSTEP["trailing_drawdown"]
        if self.total_pnl <= limit:
            self._halt(f"Trailing drawdown limit hit: ${self.total_pnl:.2f} "
                       f"(limit: ${limit:.2f})")

    def _check_close_time(self) -> None:
        """No positions after 3:58 PM CT (metals session)."""
        now_ct = datetime.now(self.CT)
        close_time = time(15, 58)
        if now_ct.time() >= close_time:
            self._halt(f"Market close time reached ({now_ct.strftime('%H:%M')} CT)")

    def _check_weekend(self) -> None:
        """No positions over the weekend."""
        now_ct = datetime.now(self.CT)
        # Friday after 3:58 PM through Sunday
        if now_ct.weekday() == TOPSTEP["close_day"]:  # Friday
            close_time = time(15, 58)
            if now_ct.time() >= close_time:
                self._halt("Weekend — no positions allowed (Friday close)")
        elif now_ct.weekday() in (5, 6):  # Saturday, Sunday
            self._halt("Weekend — market closed")

    def check_consistency_rule(self, proposed_trade_pnl: float) -> bool:
        """
        Topstep: no single day can exceed 50% of your total profit target.
        Call before closing a winning trade.
        Returns False if the trade would violate the consistency rule.
        """
        if self.total_pnl <= 0:
            return True  # No profit yet, no constraint

        limit = TOPSTEP["consistency_limit"]
        target = TOPSTEP["profit_target"]
        projected_day = self.session_pnl + proposed_trade_pnl
        max_day = target * limit

        if projected_day > max_day:
            logger.warning(f"Consistency rule: projected day P&L ${projected_day:.2f} "
                           f"would exceed {limit:.0%} of target (${max_day:.2f}). "
                           f"Consider reducing position size.")
            return False
        return True

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _halt(self, reason: str) -> None:
        self.is_halted  = True
        self.halt_reason = reason
        logger.critical(f"🛑 TRADING HALTED: {reason}")
        raise TradingHaltedError(reason)

    @property
    def should_close_soon(self) -> bool:
        """True if within 10 minutes of forced close time."""
        now_ct = datetime.now(self.CT)
        warn_time = time(15, 48)
        close_time = time(15, 58)
        return warn_time <= now_ct.time() < close_time

    @property
    def status(self) -> dict:
        """Current risk status summary."""
        return {
            "halted":         self.is_halted,
            "halt_reason":    self.halt_reason,
            "session_pnl":    self.session_pnl,
            "total_pnl":      self.total_pnl,
            "daily_loss_remaining": TOPSTEP["daily_loss_limit"] - self.session_pnl,
            "drawdown_remaining":   TOPSTEP["trailing_drawdown"] - self.total_pnl,
            "combine_progress": f"{self.total_pnl:.2f} / {TOPSTEP['profit_target']:.2f}",
        }
