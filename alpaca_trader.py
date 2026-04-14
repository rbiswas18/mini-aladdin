"""
alpaca_trader.py — Alpaca Paper/Live Trading Integration
Handles order execution, position tracking, and account info via Alpaca API.
Paper trading by default — flip ALPACA_PAPER=false in .env to go live.
"""

import logging
import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Alpaca SDK import (graceful fallback if not installed)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed. Run: pip install alpaca-py")


class AlpacaTrader:
    """
    Alpaca trading integration for paper and live trading.

    Paper mode (default): safe simulation with real market data.
    Live mode: real money — only activate after validating paper results.

    Setup:
        1. Create free account at https://alpaca.markets
        2. Get API keys from dashboard
        3. Add to .env:
           ALPACA_API_KEY=your_key
           ALPACA_SECRET_KEY=your_secret
           ALPACA_PAPER=true   # set false for live trading
    """

    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.is_paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"
        self.available = False

        if not ALPACA_AVAILABLE:
            logger.warning("alpaca-py package not installed.")
            return

        if not self.api_key or not self.secret_key:
            logger.warning(
                "Alpaca API keys not set. Add ALPACA_API_KEY and ALPACA_SECRET_KEY to .env. "
                "Get free keys at https://alpaca.markets"
            )
            return

        try:
            self.trading_client = TradingClient(
                self.api_key, self.secret_key, paper=self.is_paper
            )
            self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
            self.available = True
            mode = "PAPER" if self.is_paper else "🔴 LIVE"
            logger.info(f"Alpaca connected [{mode}]")
        except Exception as e:
            logger.error(f"Alpaca connection failed: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Account
    # ──────────────────────────────────────────────────────────────────────

    def get_account(self) -> dict:
        """Get account info: buying power, portfolio value, cash."""
        if not self.available:
            return self._unavailable("get_account")
        try:
            acct = self.trading_client.get_account()
            return {
                "status": "ok",
                "mode": "paper" if self.is_paper else "live",
                "buying_power": float(acct.buying_power),
                "portfolio_value": float(acct.portfolio_value),
                "cash": float(acct.cash),
                "equity": float(acct.equity),
                "currency": acct.currency,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        if not self.available:
            return []
        try:
            positions = self.trading_client.get_all_positions()
            return [
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc) * 100,
                    "side": p.side.value,
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"get_positions failed: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[dict]:
        """Get position for a specific symbol. Returns None if no position."""
        positions = self.get_positions()
        symbol = symbol.upper()
        for p in positions:
            if p["symbol"] == symbol:
                return p
        return None

    # ──────────────────────────────────────────────────────────────────────
    # Orders
    # ──────────────────────────────────────────────────────────────────────

    def place_market_order(
        self,
        symbol: str,
        side: str,                    # "buy" or "sell"
        qty: Optional[float] = None,
        notional: Optional[float] = None,  # dollar amount (fractional shares)
    ) -> dict:
        """
        Place a market order.

        Args:
            symbol: Ticker (e.g. 'AAPL')
            side: 'buy' or 'sell'
            qty: Number of shares (use this OR notional)
            notional: Dollar amount to buy/sell (fractional shares)

        Returns:
            dict with order details or error
        """
        if not self.available:
            return self._unavailable("place_market_order")

        symbol = symbol.upper()
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        try:
            request = MarketOrderRequest(
                symbol=symbol,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                qty=qty if qty else None,
                notional=round(notional, 2) if notional else None,
            )
            order = self.trading_client.submit_order(request)
            logger.info(f"Order placed: {side.upper()} {symbol} | id={order.id}")
            return {
                "status": "ok",
                "order_id": str(order.id),
                "symbol": order.symbol,
                "side": order.side.value,
                "qty": str(order.qty),
                "type": "market",
                "submitted_at": str(order.submitted_at),
            }
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return {"status": "error", "message": str(e)}

    def close_position(self, symbol: str) -> dict:
        """Close entire position for a symbol (market sell all shares)."""
        if not self.available:
            return self._unavailable("close_position")
        try:
            symbol = symbol.upper()
            result = self.trading_client.close_position(symbol)
            logger.info(f"Position closed: {symbol}")
            return {"status": "ok", "symbol": symbol, "order_id": str(result.id)}
        except Exception as e:
            logger.error(f"close_position failed: {e}")
            return {"status": "error", "message": str(e)}

    def get_orders(self, status: str = "all", limit: int = 20) -> list[dict]:
        """Get recent orders."""
        if not self.available:
            return []
        try:
            from alpaca.trading.requests import GetOrdersRequest
            request = GetOrdersRequest(status=status, limit=limit)
            orders = self.trading_client.get_orders(filter=request)
            return [
                {
                    "order_id": str(o.id),
                    "symbol": o.symbol,
                    "side": o.side.value,
                    "qty": str(o.qty),
                    "status": o.status.value,
                    "submitted_at": str(o.submitted_at),
                    "filled_avg_price": str(o.filled_avg_price),
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"get_orders failed: {e}")
            return []

    # ──────────────────────────────────────────────────────────────────────
    # Risk Management — Stop Loss
    # ──────────────────────────────────────────────────────────────────────

    def check_stop_losses(self, stop_loss_pct: float = 5.0) -> list[dict]:
        """
        Check all open positions and return those that have breached stop-loss.

        Args:
            stop_loss_pct: Maximum allowed loss % before triggering stop (default: 5%)

        Returns:
            List of positions that need to be closed
        """
        positions = self.get_positions()
        triggered = []
        for p in positions:
            loss_pct = p["unrealized_plpc"]
            if loss_pct <= -stop_loss_pct:
                triggered.append({
                    **p,
                    "stop_loss_pct": stop_loss_pct,
                    "action": "CLOSE — stop loss triggered",
                })
                logger.warning(
                    f"STOP LOSS triggered: {p['symbol']} | "
                    f"Loss: {loss_pct:.2f}% | Threshold: -{stop_loss_pct}%"
                )
        return triggered

    def enforce_stop_losses(self, stop_loss_pct: float = 5.0) -> list[dict]:
        """
        Check and automatically close positions that breach stop-loss threshold.

        Args:
            stop_loss_pct: Maximum allowed loss % (default: 5%)

        Returns:
            List of closed position results
        """
        triggered = self.check_stop_losses(stop_loss_pct)
        results = []
        for position in triggered:
            symbol = position["symbol"]
            logger.warning(f"Auto-closing {symbol} — stop loss at -{stop_loss_pct}%")
            result = self.close_position(symbol)
            result["reason"] = f"Stop loss triggered at {position['unrealized_plpc']:.2f}%"
            results.append(result)
        return results

    # ──────────────────────────────────────────────────────────────────────
    # Market Data
    # ──────────────────────────────────────────────────────────────────────

    def get_latest_quote(self, symbol: str) -> dict:
        """Get latest bid/ask quote from Alpaca data feed."""
        if not self.available:
            return self._unavailable("get_latest_quote")
        try:
            symbol = symbol.upper()
            request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quotes = self.data_client.get_stock_latest_quote(request)
            q = quotes[symbol]
            return {
                "symbol": symbol,
                "ask": float(q.ask_price),
                "bid": float(q.bid_price),
                "mid": round((float(q.ask_price) + float(q.bid_price)) / 2, 4),
                "timestamp": str(q.timestamp),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _unavailable(self, method: str) -> dict:
        return {
            "status": "error",
            "message": f"Alpaca not configured. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env to use {method}().",
        }

    def is_market_open(self) -> bool:
        """Check if US market is currently open."""
        if not self.available:
            return False
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception:
            return False


if __name__ == "__main__":
    trader = AlpacaTrader()
    print("Available:", trader.available)
    if trader.available:
        print("Account:", trader.get_account())
        print("Positions:", trader.get_positions())
        print("Market open:", trader.is_market_open())
