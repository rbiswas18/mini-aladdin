"""
portfolio_engine.py — Portfolio Orchestration Engine
The central nervous system of Mini-Aladdin.
Connects: signals → regime check → risk check → order → fill → state → disk

Run modes:
  - dry_run=True  → logs what it WOULD do, no orders placed (safe testing)
  - paper_mode=True → places real Alpaca paper orders (fake money, real market)
  - paper_mode=False → LIVE trading (real money — only after weeks of paper validation)
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# State dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PositionRecord:
    """Tracks one open position."""
    symbol: str
    entry_price: float
    shares: float
    stop_price: float
    take_profit_price: float
    entry_time: str
    strategy_name: str
    position_value: float
    atr_at_entry: float


@dataclass
class PortfolioState:
    """
    Full persistent state of the portfolio engine.
    Saved to JSON on disk after every action — survives restarts.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    open_positions: dict = field(default_factory=dict)      # symbol → PositionRecord dict
    last_scan_time: str = ""
    last_alerts: dict = field(default_factory=dict)         # symbol → last action
    trade_journal: list = field(default_factory=list)       # completed trades
    account_snapshot: dict = field(default_factory=lambda: {
        "value": 1200.0, "cash": 1200.0, "peak_value": 1200.0
    })

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PortfolioState":
        return cls(**d)


# ──────────────────────────────────────────────────────────────────────────────
# Portfolio Engine
# ──────────────────────────────────────────────────────────────────────────────

class PortfolioEngine:
    """
    Orchestrates the full trading loop:
    scan → check → size → order → fill → monitor → close → persist

    Default: paper_mode=True, dry_run=True (safest — just logs, no orders)
    """

    def __init__(
        self,
        symbols: list,
        strategy_name: str = "CombinedSignal",
        account_value: float = 1200.0,
        state_file: str = "portfolio_state.json",
        paper_mode: bool = True,
        max_positions: int = 2,
        risk_per_trade_pct: float = 0.01,
        stop_atr_multiplier: float = 1.5,
        take_profit_multiplier: float = 3.0,
    ):
        self.symbols = [s.upper() for s in symbols]
        self.strategy_name = strategy_name
        self.account_value = account_value
        self.state_file = Path(state_file)
        self.paper_mode = paper_mode
        self.max_positions = max_positions
        self.risk_per_trade_pct = risk_per_trade_pct
        self.stop_atr_multiplier = stop_atr_multiplier
        self.take_profit_multiplier = take_profit_multiplier

        # Load persistent state
        self.state = self.load_state()

        # Initialize subsystems
        self._init_subsystems()

    def _init_subsystems(self):
        """Initialize all connected modules."""
        try:
            from strategy_simple import build_strategy
            self.strategy = build_strategy(self.strategy_name)
        except Exception as e:
            logger.error(f"Strategy init failed: {e}")
            self.strategy = None

        try:
            from regime_filter import RegimeFilter, EarningsFilter
            self.regime_filter = RegimeFilter()
            self.earnings_filter = EarningsFilter()
        except Exception as e:
            logger.warning(f"Regime filter unavailable: {e}")
            self.regime_filter = None
            self.earnings_filter = None

        try:
            from position_sizer import PositionSizer, RiskManager
            self.sizer = PositionSizer()
            self.risk_manager = RiskManager(
                account_value=self.account_value,
                max_positions=self.max_positions,
                risk_per_trade_pct=self.risk_per_trade_pct,
            )
        except Exception as e:
            logger.warning(f"Risk manager unavailable: {e}")
            self.sizer = None
            self.risk_manager = None

        try:
            from alpaca_trader import AlpacaTrader
            self.trader = AlpacaTrader()
        except Exception as e:
            logger.warning(f"Alpaca trader unavailable: {e}")
            self.trader = None

        try:
            from validator import PaperTradeLog
            self.paper_log = PaperTradeLog("paper_trades.json")
        except Exception as e:
            self.paper_log = None

    # ──────────────────────────────────────────────────────────────────────────
    # Core scan
    # ──────────────────────────────────────────────────────────────────────────

    def run_scan(self) -> list[dict]:
        """
        Scan all symbols and generate action recommendations.

        Returns list of action dicts:
        {symbol, action, price, shares, stop_price, take_profit_price, reason, blocked_by}
        """
        if self.strategy is None:
            logger.error("No strategy loaded — cannot scan")
            return []

        actions = []
        today = date.today()
        end = today.strftime("%Y-%m-%d")
        start = (today - timedelta(days=365)).strftime("%Y-%m-%d")

        for symbol in self.symbols:
            try:
                # Fetch data
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start, end=end, auto_adjust=True)
                if df.empty or len(df) < 50:
                    logger.warning(f"{symbol}: insufficient data")
                    continue

                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                current_price = float(df["Close"].iloc[-1])

                # Generate signals
                signals_df = self.strategy.generate_signals(df.copy())
                last_signal = int(signals_df["signal"].iloc[-1])

                from strategy_simple import BUY, SELL, HOLD

                action_dict = {
                    "symbol": symbol,
                    "action": "BUY" if last_signal == BUY else "SELL" if last_signal == SELL else "HOLD",
                    "price": current_price,
                    "shares": 0.0,
                    "stop_price": 0.0,
                    "take_profit_price": 0.0,
                    "reason": "Strategy signal",
                    "blocked_by": None,
                }

                # For BUY signals: apply filters and sizing
                if last_signal == BUY:
                    # Already in position?
                    if symbol in self.state.open_positions:
                        action_dict["action"] = "HOLD"
                        action_dict["reason"] = "Already in position"
                        actions.append(action_dict)
                        continue

                    # Regime check
                    if self.regime_filter:
                        if not self.regime_filter.is_bullish():
                            action_dict["action"] = "HOLD"
                            action_dict["blocked_by"] = "Regime filter (market not bullish)"
                            actions.append(action_dict)
                            continue

                    # Earnings blackout
                    if self.earnings_filter:
                        if self.earnings_filter.is_earnings_blackout(symbol, today):
                            action_dict["action"] = "HOLD"
                            action_dict["blocked_by"] = f"Earnings blackout for {symbol}"
                            actions.append(action_dict)
                            continue

                    # Risk check
                    if self.risk_manager:
                        if not self.risk_manager.can_trade():
                            action_dict["action"] = "HOLD"
                            action_dict["blocked_by"] = f"Risk limit: {self.risk_manager.pause_reason}"
                            actions.append(action_dict)
                            continue

                    # Position sizing
                    if self.sizer:
                        import ta
                        atr_series = self.sizer.calculate_atr(df, period=14)
                        atr = float(atr_series.iloc[-1]) if not atr_series.empty else current_price * 0.02
                        sizing = self.sizer.calculate_position_size(
                            price=current_price,
                            atr=atr,
                            account_value=self.account_value,
                            risk_pct=self.risk_per_trade_pct,
                        )
                        action_dict["shares"] = sizing["shares"]
                        action_dict["stop_price"] = sizing["stop_price"]
                        action_dict["take_profit_price"] = round(
                            current_price + (current_price - sizing["stop_price"]) * self.take_profit_multiplier, 4
                        )
                        action_dict["atr"] = atr
                        action_dict["max_loss"] = sizing["max_loss"]
                    else:
                        # Fallback: fixed 5% stop
                        action_dict["shares"] = round((self.account_value * 0.5) / current_price, 4)
                        action_dict["stop_price"] = round(current_price * 0.95, 4)
                        action_dict["take_profit_price"] = round(current_price * 1.10, 4)

                # For SELL signals: only act if in position
                elif last_signal == SELL:
                    if symbol not in self.state.open_positions:
                        action_dict["action"] = "HOLD"
                        action_dict["reason"] = "No open position to close"

                actions.append(action_dict)

            except Exception as e:
                logger.error(f"Scan failed for {symbol}: {e}")

        self.state.last_scan_time = datetime.utcnow().isoformat()
        return actions

    # ──────────────────────────────────────────────────────────────────────────
    # Execute
    # ──────────────────────────────────────────────────────────────────────────

    def execute_scan(self, dry_run: bool = True) -> list[dict]:
        """
        Run scan and execute actionable signals.

        Args:
            dry_run: If True, only logs — no orders placed

        Returns:
            Execution log
        """
        actions = self.run_scan()
        execution_log = []

        for action in actions:
            symbol = action["symbol"]
            act = action["action"]

            if act == "HOLD":
                if action.get("blocked_by"):
                    logger.info(f"⏸️  {symbol}: BLOCKED — {action['blocked_by']}")
                continue

            if act == "BUY" and action["shares"] > 0:
                log_entry = self._execute_buy(symbol, action, dry_run)
                execution_log.append(log_entry)

            elif act == "SELL" and symbol in self.state.open_positions:
                log_entry = self._execute_sell(symbol, action["price"], "Strategy SELL signal", dry_run)
                execution_log.append(log_entry)

        self.save_state()
        return execution_log

    def _execute_buy(self, symbol: str, action: dict, dry_run: bool) -> dict:
        """Execute or simulate a BUY order."""
        price = action["price"]
        shares = action["shares"]
        stop_price = action["stop_price"]
        take_profit = action["take_profit_price"]

        log_entry = {
            "action": "BUY",
            "symbol": symbol,
            "price": price,
            "shares": shares,
            "stop_price": stop_price,
            "take_profit_price": take_profit,
            "dry_run": dry_run,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if dry_run:
            logger.info(f"🔵 [DRY RUN] BUY {shares:.4f} {symbol} @ ${price:.2f} | Stop: ${stop_price:.2f} | Target: ${take_profit:.2f}")
            log_entry["status"] = "dry_run"
        else:
            try:
                if self.trader and self.trader.available:
                    result = self.trader.place_market_order(symbol, "buy", notional=round(shares * price, 2))
                    log_entry["order_id"] = result.get("order_id")
                    log_entry["status"] = result.get("status")
                    logger.info(f"✅ BUY order placed: {symbol} | Order ID: {result.get('order_id')}")
                else:
                    log_entry["status"] = "alpaca_unavailable"
                    logger.warning(f"Alpaca not available — BUY {symbol} not executed")
            except Exception as e:
                log_entry["status"] = "error"
                log_entry["error"] = str(e)
                logger.error(f"BUY order failed for {symbol}: {e}")
                return log_entry

        # Record position in state
        position = PositionRecord(
            symbol=symbol,
            entry_price=price,
            shares=shares,
            stop_price=stop_price,
            take_profit_price=take_profit,
            entry_time=datetime.utcnow().isoformat(),
            strategy_name=self.strategy_name,
            position_value=round(shares * price, 2),
            atr_at_entry=action.get("atr", 0.0),
        )
        self.state.open_positions[symbol] = asdict(position)
        self.state.last_alerts[symbol] = "BUY"

        # Log to paper trade log
        if self.paper_log:
            from strategy_simple import BUY as BUY_SIGNAL
            self.paper_log.log_signal(symbol, BUY_SIGNAL, price, self.strategy_name)
            self.paper_log.log_fill(symbol, "buy", shares, price, signal_price=price)
            self.paper_log.save()

        # Telegram alert
        self._send_telegram_alert(
            f"{'🔵 [DRY RUN] ' if dry_run else '✅ '}BUY {symbol}\n"
            f"Price: ${price:.2f} | Shares: {shares:.4f}\n"
            f"Stop: ${stop_price:.2f} | Target: ${take_profit:.2f}\n"
            f"Max Loss: ${action.get('max_loss', 0):.2f}"
        )

        if self.risk_manager:
            self.risk_manager.record_trade_open()

        return log_entry

    def _execute_sell(self, symbol: str, price: float, reason: str, dry_run: bool) -> dict:
        """Execute or simulate a SELL/close order."""
        position = self.state.open_positions.get(symbol)
        if not position:
            return {"action": "SELL", "symbol": symbol, "status": "no_position"}

        entry_price = position["entry_price"]
        shares = position["shares"]
        pnl = (price - entry_price) * shares
        return_pct = (price - entry_price) / entry_price * 100

        log_entry = {
            "action": "SELL",
            "symbol": symbol,
            "price": price,
            "shares": shares,
            "pnl": round(pnl, 2),
            "return_pct": round(return_pct, 2),
            "reason": reason,
            "dry_run": dry_run,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if dry_run:
            logger.info(f"🔴 [DRY RUN] SELL {shares:.4f} {symbol} @ ${price:.2f} | PnL: ${pnl:.2f} ({return_pct:.1f}%) | Reason: {reason}")
            log_entry["status"] = "dry_run"
        else:
            try:
                if self.trader and self.trader.available:
                    result = self.trader.close_position(symbol)
                    log_entry["order_id"] = result.get("order_id")
                    log_entry["status"] = result.get("status")
                    logger.info(f"✅ Position closed: {symbol} | PnL: ${pnl:.2f}")
                else:
                    log_entry["status"] = "alpaca_unavailable"
            except Exception as e:
                log_entry["status"] = "error"
                log_entry["error"] = str(e)
                logger.error(f"SELL order failed for {symbol}: {e}")
                return log_entry

        # Update state
        del self.state.open_positions[symbol]
        self.state.last_alerts[symbol] = "SELL"

        # Record in journal
        self.state.trade_journal.append({
            "symbol": symbol,
            "entry_price": entry_price,
            "exit_price": price,
            "shares": shares,
            "pnl": round(pnl, 2),
            "return_pct": round(return_pct, 2),
            "entry_time": position["entry_time"],
            "exit_time": datetime.utcnow().isoformat(),
            "reason": reason,
            "strategy": position["strategy_name"],
        })

        # Log to paper trade log
        if self.paper_log:
            self.paper_log.log_close(symbol, pnl, return_pct / 100)
            self.paper_log.save()

        # Update risk manager
        if self.risk_manager:
            self.risk_manager.record_trade_close(pnl)

        icon = "✅" if pnl >= 0 else "❌"
        self._send_telegram_alert(
            f"{'🔴 [DRY RUN] ' if dry_run else icon + ' '}SELL {symbol}\n"
            f"Price: ${price:.2f} | PnL: ${pnl:.2f} ({return_pct:.1f}%)\n"
            f"Reason: {reason}"
        )

        return log_entry

    # ──────────────────────────────────────────────────────────────────────────
    # Stop monitoring
    # ──────────────────────────────────────────────────────────────────────────

    def check_stops(self, dry_run: bool = True) -> list[dict]:
        """
        Check all open positions against stop-loss and take-profit levels.
        Closes any that have breached their levels.
        """
        closed = []
        if not self.state.open_positions:
            return closed

        for symbol, pos in list(self.state.open_positions.items()):
            try:
                ticker = yf.Ticker(symbol)
                current_price = ticker.fast_info.last_price
                if not current_price:
                    continue

                stop = pos["stop_price"]
                target = pos["take_profit_price"]
                entry = pos["entry_price"]

                if current_price <= stop:
                    logger.warning(f"🛑 STOP LOSS triggered: {symbol} @ ${current_price:.2f} (stop: ${stop:.2f})")
                    result = self._execute_sell(symbol, current_price, f"Stop loss hit at ${current_price:.2f}", dry_run)
                    closed.append(result)

                elif current_price >= target:
                    logger.info(f"🎯 TAKE PROFIT triggered: {symbol} @ ${current_price:.2f} (target: ${target:.2f})")
                    result = self._execute_sell(symbol, current_price, f"Take profit hit at ${current_price:.2f}", dry_run)
                    closed.append(result)

                else:
                    change_pct = (current_price - entry) / entry * 100
                    logger.info(f"📊 {symbol}: ${current_price:.2f} ({change_pct:+.1f}%) | Stop: ${stop:.2f} | Target: ${target:.2f}")

            except Exception as e:
                logger.error(f"Stop check failed for {symbol}: {e}")

        if closed:
            self.save_state()

        return closed

    # ──────────────────────────────────────────────────────────────────────────
    # Reconciliation
    # ──────────────────────────────────────────────────────────────────────────

    def reconcile_with_alpaca(self) -> dict:
        """
        Compare internal state with actual Alpaca positions.
        Flags discrepancies after restarts or manual trades.
        """
        report = {
            "status": "ok",
            "internal_positions": list(self.state.open_positions.keys()),
            "alpaca_positions": [],
            "missing_from_state": [],
            "missing_from_alpaca": [],
            "discrepancies": [],
        }

        if not self.trader or not self.trader.available:
            report["status"] = "alpaca_unavailable"
            logger.warning("Cannot reconcile — Alpaca not configured")
            return report

        try:
            alpaca_positions = self.trader.get_positions()
            alpaca_symbols = [p["symbol"] for p in alpaca_positions]
            report["alpaca_positions"] = alpaca_symbols

            # In Alpaca but not in our state
            for sym in alpaca_symbols:
                if sym not in self.state.open_positions:
                    report["missing_from_state"].append(sym)
                    report["discrepancies"].append(f"{sym}: exists in Alpaca but not tracked internally")

            # In our state but not in Alpaca
            for sym in self.state.open_positions:
                if sym not in alpaca_symbols:
                    report["missing_from_alpaca"].append(sym)
                    report["discrepancies"].append(f"{sym}: tracked internally but not found in Alpaca")

            if report["discrepancies"]:
                report["status"] = "discrepancies_found"
                logger.warning(f"Reconciliation issues: {report['discrepancies']}")
            else:
                logger.info("✅ Reconciliation clean — state matches Alpaca")

        except Exception as e:
            report["status"] = "error"
            report["error"] = str(e)
            logger.error(f"Reconciliation failed: {e}")

        return report

    # ──────────────────────────────────────────────────────────────────────────
    # Status & persistence
    # ──────────────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Get full current system status."""
        journal = self.state.trade_journal
        wins = [t for t in journal if t.get("pnl", 0) > 0]
        losses = [t for t in journal if t.get("pnl", 0) <= 0]
        total_pnl = sum(t.get("pnl", 0) for t in journal)

        risk_status = {}
        if self.risk_manager:
            risk_status = self.risk_manager.get_status()

        return {
            "session_id": self.state.session_id,
            "open_positions": self.state.open_positions,
            "open_count": len(self.state.open_positions),
            "last_scan": self.state.last_scan_time,
            "account_snapshot": self.state.account_snapshot,
            "trade_journal": {
                "total_trades": len(journal),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": round(len(wins) / len(journal) * 100, 1) if journal else 0,
                "total_pnl": round(total_pnl, 2),
            },
            "risk_status": risk_status,
            "strategy": self.strategy_name,
            "symbols": self.symbols,
            "paper_mode": self.paper_mode,
        }

    def save_state(self):
        """Persist state to JSON file."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
            logger.debug(f"State saved to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load_state(self) -> PortfolioState:
        """Load state from JSON file, return fresh state if not found."""
        try:
            if self.state_file.exists():
                with open(self.state_file) as f:
                    data = json.load(f)
                logger.info(f"Loaded existing state from {self.state_file}")
                return PortfolioState.from_dict(data)
        except Exception as e:
            logger.warning(f"Could not load state: {e} — starting fresh")
        return PortfolioState()

    # ──────────────────────────────────────────────────────────────────────────
    # Alerts
    # ──────────────────────────────────────────────────────────────────────────

    def _send_telegram_alert(self, message: str):
        """Send Telegram alert. Silently skips if not configured."""
        import requests
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
                timeout=5,
            )
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Realistic backtest (aligned with live rules)
# ──────────────────────────────────────────────────────────────────────────────

def run_realistic_backtest(
    symbol: str,
    strategy_name: str = "CombinedSignal",
    start_date: str = "2022-01-01",
    end_date: str = "2024-01-01",
    initial_capital: float = 1200.0,
):
    """
    Runs a backtest that mirrors live trading rules:
    - Regime filter applied (only trade in bull markets)
    - ATR-based position sizing (not all-in)
    - Max 1 position at a time (realistic for $1,200)
    - Same stop logic used live

    This gives a more honest picture than the standard backtest.
    """
    from backtest_simple import run_backtest, BacktestResult
    from strategy_simple import build_strategy
    from regime_filter import RegimeAwareStrategy, RegimeFilter, EarningsFilter

    base_strategy = build_strategy(strategy_name)
    regime_filter = RegimeFilter()
    earnings_filter = EarningsFilter()

    # Wrap with regime awareness
    aware_strategy = RegimeAwareStrategy(
        strategy=base_strategy,
        symbol=symbol,
        use_regime=True,
        use_earnings_filter=True,
    )

    result = run_backtest(
        symbol=symbol,
        strategy=aware_strategy,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        commission=0.001,
        slippage=0.001,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
    )

    logger.info(f"Realistic backtest: {symbol} | {strategy_name}")
    logger.info(f"  Return: {result.metrics['total_return']}% | Sharpe: {result.metrics['sharpe_ratio']}")
    logger.info(f"  Trades: {result.metrics['total_trades']} | Win Rate: {result.metrics['win_rate']}%")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Daily session runner
# ──────────────────────────────────────────────────────────────────────────────

def run_daily_session(
    symbols: list = None,
    strategy_name: str = "CombinedSignal",
    account_value: float = 1200.0,
    dry_run: bool = True,
    stop_check_interval_minutes: int = 30,
):
    """
    Run one complete trading session.

    1. Check market hours
    2. Reset daily risk counters
    3. Reconcile with Alpaca
    4. Scan + execute signals
    5. Monitor stops every N minutes during market hours
    6. Send end-of-day summary

    Args:
        dry_run: True = log only, no real orders (default: True)
    """
    if symbols is None:
        from watchlist import DEFAULT_WATCHLIST
        symbols = DEFAULT_WATCHLIST

    engine = PortfolioEngine(
        symbols=symbols,
        strategy_name=strategy_name,
        account_value=account_value,
        dry_run=dry_run,
        paper_mode=True,
    )

    logger.info(f"{'='*60}")
    logger.info(f"Mini-Aladdin Daily Session — {'DRY RUN' if dry_run else 'PAPER TRADING'}")
    logger.info(f"Strategy: {strategy_name} | Symbols: {symbols}")
    logger.info(f"{'='*60}")

    # 1. Check market hours
    market_open = False
    if engine.trader and engine.trader.available:
        market_open = engine.trader.is_market_open()
    else:
        # Assume open during US market hours (9:30 AM - 4:00 PM ET)
        now_utc = datetime.utcnow()
        hour = now_utc.hour
        market_open = 13 <= hour <= 20  # approx ET hours in UTC

    if not market_open:
        logger.info("Market is closed. Running scan for preview only.")

    # 2. Reset daily risk counters
    if engine.risk_manager:
        engine.risk_manager.reset_daily()

    # 3. Reconcile
    recon = engine.reconcile_with_alpaca()
    logger.info(f"Reconciliation: {recon['status']}")
    if recon.get("discrepancies"):
        for d in recon["discrepancies"]:
            logger.warning(f"  ⚠️  {d}")

    # 4. Initial scan + execute
    logger.info("Running initial scan...")
    execution_log = engine.execute_scan(dry_run=dry_run)
    logger.info(f"Executed {len(execution_log)} actions")

    # 5. Stop monitoring loop (only during market hours)
    if market_open and engine.state.open_positions:
        check_count = 0
        max_checks = 14  # ~7 hours / 30 min = 14 checks
        logger.info(f"Starting stop monitor — checking every {stop_check_interval_minutes} min")

        while check_count < max_checks:
            time.sleep(stop_check_interval_minutes * 60)
            check_count += 1
            logger.info(f"Stop check #{check_count}")
            closed = engine.check_stops(dry_run=dry_run)
            if closed:
                logger.info(f"Closed {len(closed)} positions on stop/target")
            if not engine.state.open_positions:
                logger.info("No open positions remaining — monitoring done")
                break

    # 6. End of day summary
    status = engine.get_status()
    summary = (
        f"📊 Mini-Aladdin Daily Summary\n"
        f"Strategy: {strategy_name}\n"
        f"Open positions: {status['open_count']}\n"
        f"Total trades today: {status['trade_journal']['total_trades']}\n"
        f"Total PnL: ${status['trade_journal']['total_pnl']:,.2f}\n"
        f"Win rate: {status['trade_journal']['win_rate']}%\n"
        f"Mode: {'DRY RUN' if dry_run else 'PAPER'}"
    )
    logger.info(summary)
    engine._send_telegram_alert(summary)
    engine.save_state()

    return status


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "dry"

    if mode == "dry":
        print("Running DRY RUN scan...")
        engine = PortfolioEngine(
            symbols=["AAPL", "NVDA", "MSFT"],
            strategy_name="CombinedSignal",
        )
        actions = engine.run_scan()
        for a in actions:
            print(f"{a['symbol']}: {a['action']} @ ${a['price']:.2f}"
                  + (f" | blocked: {a['blocked_by']}" if a.get("blocked_by") else ""))
        print("\nStatus:", engine.get_status())

    elif mode == "backtest":
        print("Running realistic backtest...")
        result = run_realistic_backtest("NVDA", "RSIMeanReversion", "2022-01-01", "2024-01-01")
        print(f"Return: {result.metrics['total_return']}% | Sharpe: {result.metrics['sharpe_ratio']}")

    elif mode == "session":
        print("Running full daily session (dry run)...")
        run_daily_session(symbols=["AAPL", "NVDA", "MSFT"], dry_run=True)
