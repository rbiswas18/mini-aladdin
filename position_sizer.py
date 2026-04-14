"""
position_sizer.py — ATR-Based Position Sizing + Risk Manager
Calculates safe position sizes and enforces account-level risk rules.
"""

import logging
from datetime import date, datetime
from typing import Optional

import pandas as pd
import ta

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Calculates position sizes based on ATR (Average True Range).

    Why ATR? Because different stocks have different volatility.
    NVDA moves 3-4% per day. MSFT moves 1-2%. A fixed 5% stop
    means very different things for each. ATR normalizes this.
    """

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range for a DataFrame of OHLCV data.

        Args:
            df: DataFrame with High, Low, Close columns
            period: ATR lookback period (default 14 days)

        Returns:
            pd.Series of ATR values
        """
        try:
            atr = ta.volatility.AverageTrueRange(
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                window=period
            )
            return atr.average_true_range()
        except Exception as e:
            logger.error(f"ATR calculation failed: {e}")
            # Fallback: simple daily range average
            return (df["High"] - df["Low"]).rolling(period).mean()

    def calculate_position_size(
        self,
        price: float,
        atr: float,
        account_value: float,
        risk_pct: float = 0.01,
        atr_multiplier: float = 1.5,
        max_position_pct: float = 0.5,
    ) -> dict:
        """
        Calculate position size based on ATR stop distance and account risk.

        Logic:
        - Stop price = entry price - (ATR × multiplier)
        - Stop distance = entry price - stop price
        - Max loss = account_value × risk_pct
        - Shares = max_loss / stop_distance

        Args:
            price: Current stock price (entry price)
            atr: Current ATR value
            account_value: Total account value in USD
            risk_pct: Max % of account to risk per trade (default 1%)
            atr_multiplier: Stop distance in ATR units (default 1.5)
            max_position_pct: Max % of account in one position (default 50%)

        Returns:
            dict with shares, position_value, stop_price, max_loss, risk_pct
        """
        if price <= 0 or atr <= 0 or account_value <= 0:
            return {
                "shares": 0.0,
                "position_value": 0.0,
                "stop_price": 0.0,
                "max_loss": 0.0,
                "risk_pct": 0.0,
                "error": "Invalid inputs"
            }

        stop_distance = atr * atr_multiplier
        stop_price = price - stop_distance
        max_loss = account_value * risk_pct
        max_position_value = account_value * max_position_pct

        if stop_distance <= 0:
            return {
                "shares": 0.0,
                "position_value": 0.0,
                "stop_price": stop_price,
                "max_loss": max_loss,
                "risk_pct": risk_pct,
                "error": "Stop distance too small"
            }

        shares = max_loss / stop_distance

        # Cap at max position size
        position_value = shares * price
        if position_value > max_position_value:
            shares = max_position_value / price
            position_value = max_position_value

        actual_max_loss = shares * stop_distance
        actual_risk_pct = actual_max_loss / account_value

        return {
            "shares": round(shares, 4),
            "position_value": round(position_value, 2),
            "stop_price": round(stop_price, 4),
            "max_loss": round(actual_max_loss, 2),
            "risk_pct": round(actual_risk_pct * 100, 3),  # as %
        }


class RiskManager:
    """
    Account-level risk controls for safe trading.

    Enforces:
    - Max positions at once
    - Max capital per position
    - Daily loss limit
    - Drawdown-based trading pause

    With $1,200 capital, these rules are non-negotiable.
    """

    def __init__(
        self,
        account_value: float,
        max_positions: int = 2,
        max_position_pct: float = 0.5,
        daily_loss_limit_pct: float = 0.02,
        drawdown_pause_pct: float = 0.08,
        risk_per_trade_pct: float = 0.01,
    ):
        """
        Args:
            account_value: Starting/current account value
            max_positions: Max simultaneous open positions
            max_position_pct: Max % of account in one position
            daily_loss_limit_pct: Stop trading if daily loss exceeds this %
            drawdown_pause_pct: Pause trading if account drops this % from peak
            risk_per_trade_pct: Target risk per trade as % of account
        """
        self.account_value = account_value
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.drawdown_pause_pct = drawdown_pause_pct
        self.risk_per_trade_pct = risk_per_trade_pct

        self.peak_value = account_value
        self.daily_losses = 0.0
        self.open_positions = 0
        self.daily_reset_date: Optional[date] = None
        self.paused = False
        self.pause_reason = ""

        self.sizer = PositionSizer()

    def can_trade(self) -> bool:
        """
        Check if trading is allowed given current risk state.

        Returns:
            True if all risk checks pass
        """
        # Auto-reset daily counters if new day
        today = date.today()
        if self.daily_reset_date != today:
            self.reset_daily()

        # Check drawdown pause
        drawdown = (self.peak_value - self.account_value) / self.peak_value
        if drawdown >= self.drawdown_pause_pct:
            self.paused = True
            self.pause_reason = f"Drawdown {drawdown:.1%} exceeds limit {self.drawdown_pause_pct:.1%}"
            return False

        # Check daily loss limit
        daily_loss_pct = self.daily_losses / self.account_value
        if daily_loss_pct >= self.daily_loss_limit_pct:
            self.pause_reason = f"Daily loss {daily_loss_pct:.1%} exceeds limit {self.daily_loss_limit_pct:.1%}"
            return False

        # Check max positions
        if self.open_positions >= self.max_positions:
            self.pause_reason = f"Max positions ({self.max_positions}) reached"
            return False

        self.paused = False
        self.pause_reason = ""
        return True

    def get_position_size(self, price: float, atr: float) -> dict:
        """
        Get recommended position size for a new trade.

        Args:
            price: Entry price
            atr: Current ATR

        Returns:
            Position sizing dict from PositionSizer
        """
        return self.sizer.calculate_position_size(
            price=price,
            atr=atr,
            account_value=self.account_value,
            risk_pct=self.risk_per_trade_pct,
            max_position_pct=self.max_position_pct,
        )

    def record_loss(self, amount: float):
        """Record a realized loss for daily tracking."""
        if amount > 0:
            self.daily_losses += amount
            self.account_value -= amount
            logger.info(f"Loss recorded: ${amount:.2f} | Daily losses: ${self.daily_losses:.2f}")

    def record_trade_open(self):
        """Track that a new position was opened."""
        self.open_positions += 1

    def record_trade_close(self, pnl: float):
        """
        Record trade closure and update peak value tracking.

        Args:
            pnl: Profit (positive) or loss (negative)
        """
        self.open_positions = max(0, self.open_positions - 1)
        self.account_value += pnl

        # Track daily losses separately (do NOT subtract from account again)
        if pnl < 0:
            self.daily_losses += abs(pnl)  # track for daily limit check only

        # Update peak
        if self.account_value > self.peak_value:
            self.peak_value = self.account_value

        logger.info(f"Trade closed: PnL=${pnl:.2f} | Account: ${self.account_value:.2f}")

    def reset_daily(self):
        """Reset daily counters. Call at market open each day."""
        self.daily_losses = 0.0
        self.daily_reset_date = date.today()
        logger.info("Daily risk counters reset")

    def get_status(self) -> dict:
        """Get full risk status snapshot."""
        drawdown = (self.peak_value - self.account_value) / self.peak_value
        daily_loss_pct = self.daily_losses / self.account_value if self.account_value > 0 else 0

        return {
            "account_value": round(self.account_value, 2),
            "peak_value": round(self.peak_value, 2),
            "drawdown_pct": round(drawdown * 100, 2),
            "drawdown_limit_pct": round(self.drawdown_pause_pct * 100, 2),
            "daily_losses": round(self.daily_losses, 2),
            "daily_loss_pct": round(daily_loss_pct * 100, 2),
            "daily_loss_limit_pct": round(self.daily_loss_limit_pct * 100, 2),
            "open_positions": self.open_positions,
            "max_positions": self.max_positions,
            "can_trade": self.can_trade(),
            "paused": self.paused,
            "pause_reason": self.pause_reason,
            "risk_per_trade_pct": round(self.risk_per_trade_pct * 100, 2),
            "max_loss_per_trade": round(self.account_value * self.risk_per_trade_pct, 2),
        }


if __name__ == "__main__":
    rm = RiskManager(account_value=1200)
    print("Risk Status:", rm.get_status())
    print("Can trade:", rm.can_trade())

    # Example position size for NVDA at $900, ATR=$25
    size = rm.get_position_size(price=900, atr=25)
    print("Position size:", size)
