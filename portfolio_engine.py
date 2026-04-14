"""
portfolio_engine.py — Portfolio Orchestration Engine
The central nervous system of Mini-Aladdin.
Written by GPT-5.4 (3-part parallel generation).

Connects: signals → regime check → risk check → order → fill → state → disk

Run modes:
  python portfolio_engine.py dry      → dry scan (no orders)
  python portfolio_engine.py session  → full day session
  python portfolio_engine.py backtest → realistic backtest on NVDA
"""

# ── Part 1: Foundation ────────────────────────────────────────────────────────

import json, logging, os, time, uuid, requests
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Optional

import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PositionRecord:
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
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    open_positions: dict = field(default_factory=dict)
    last_scan_time: str = ""
    last_alerts: dict = field(default_factory=dict)
    trade_journal: list = field(default_factory=list)
    account_snapshot: dict = field(
        default_factory=lambda: {"value": 1200.0, "cash": 1200.0, "peak_value": 1200.0}
    )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PortfolioState":
        if not isinstance(d, dict):
            return cls()
        raw_open_positions = d.get("open_positions", {})
        open_positions = {}
        if isinstance(raw_open_positions, dict):
            for symbol, record in raw_open_positions.items():
                if isinstance(record, PositionRecord):
                    open_positions[symbol] = asdict(record)
                elif isinstance(record, dict):
                    open_positions[symbol] = record
                else:
                    logger.warning("Skipping invalid position record for %s", symbol)
        return cls(
            session_id=str(d.get("session_id") or str(uuid.uuid4())),
            open_positions=open_positions,
            last_scan_time=str(d.get("last_scan_time", "")),
            last_alerts=d.get("last_alerts", {}) if isinstance(d.get("last_alerts", {}), dict) else {},
            trade_journal=d.get("trade_journal", []) if isinstance(d.get("trade_journal", []), list) else [],
            account_snapshot=d.get("account_snapshot", {"value": 1200.0, "cash": 1200.0, "peak_value": 1200.0}),
        )


def _send_telegram_alert(message: str) -> None:
    """Send Telegram alert. Silent fallback if not configured."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id or not message:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=5,
        )
    except Exception:
        return


def _load_subsystems(engine: object) -> None:
    """Load all subsystems with graceful fallbacks."""
    try:
        from strategy_simple import build_strategy
        engine.strategy = build_strategy(engine.strategy_name)
    except Exception as e:
        engine.strategy = None
        logger.warning("Failed to load strategy: %s", e)

    try:
        from regime_filter import RegimeFilter
        engine.regime_filter = RegimeFilter()
    except Exception as e:
        engine.regime_filter = None
        logger.warning("Failed to load regime_filter: %s", e)

    try:
        from regime_filter import EarningsFilter
        engine.earnings_filter = EarningsFilter()
    except Exception as e:
        engine.earnings_filter = None
        logger.warning("Failed to load earnings_filter: %s", e)

    try:
        from position_sizer import PositionSizer
        engine.sizer = PositionSizer()
    except Exception as e:
        engine.sizer = None
        logger.warning("Failed to load sizer: %s", e)

    try:
        from position_sizer import RiskManager
        engine.risk_manager = RiskManager(
            account_value=engine.account_value,
            max_positions=engine.max_positions,
            risk_per_trade_pct=engine.risk_per_trade_pct,
        )
    except Exception as e:
        engine.risk_manager = None
        logger.warning("Failed to load risk_manager: %s", e)

    try:
        from alpaca_trader import AlpacaTrader
        engine.trader = AlpacaTrader()
    except Exception as e:
        engine.trader = None
        logger.warning("Failed to load trader: %s", e)

    try:
        from validator import PaperTradeLog
        engine.paper_log = PaperTradeLog("paper_trades.json")
    except Exception as e:
        engine.paper_log = None
        logger.warning("Failed to load paper_log: %s", e)


def save_state(state: PortfolioState, state_file: Path) -> None:
    """Save portfolio state to JSON file."""
    try:
        state_file = Path(state_file)
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with state_file.open("w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)
        logger.info("State saved to %s", state_file)
    except Exception as e:
        logger.error("Failed to save state: %s", e)


def load_state(state_file: Path) -> PortfolioState:
    """Load portfolio state from JSON file. Returns fresh state if not found."""
    try:
        state_file = Path(state_file)
        if not state_file.exists():
            return PortfolioState()
        with state_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("State loaded from %s", state_file)
        return PortfolioState.from_dict(data)
    except Exception as e:
        logger.error("Failed to load state: %s", e)
        return PortfolioState()


# ── Part 2: PortfolioEngine — init + scan + execute ───────────────────────────

class PortfolioEngine:
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
        self.symbols = list(symbols)
        self.strategy_name = strategy_name
        self.account_value = float(account_value)
        self.state_file = Path(state_file)
        self.paper_mode = bool(paper_mode)
        self.max_positions = int(max_positions)
        self.risk_per_trade_pct = float(risk_per_trade_pct)
        self.stop_atr_multiplier = float(stop_atr_multiplier)
        self.take_profit_multiplier = float(take_profit_multiplier)

        self.strategy = None
        self.regime_filter = None
        self.earnings_filter = None
        self.sizer = None
        self.risk_manager = None
        self.trader = None
        self.paper_log = None

        _load_subsystems(self)
        self.state = load_state(self.state_file)
        logger.info("PortfolioEngine initialized | symbols=%s | strategy=%s", self.symbols, self.strategy_name)

    def run_scan(self) -> list[dict]:
        """Scan all symbols. Apply regime, earnings, risk filters. Calculate ATR sizing for BUY signals."""
        actions: list[dict] = []
        today = date.today()
        end = today.strftime("%Y-%m-%d")
        start = (today - timedelta(days=365)).strftime("%Y-%m-%d")

        if self.strategy is None:
            logger.error("Strategy unavailable — cannot scan")
            self.state.last_scan_time = datetime.utcnow().isoformat()
            return actions

        for symbol in self.symbols:
            try:
                df = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)[["Open","High","Low","Close","Volume"]]
                if df.empty or len(df) < 50:
                    logger.warning("Skipping %s — insufficient data", symbol)
                    continue

                signals_df = self.strategy.generate_signals(df.copy())
                last_signal = int(signals_df["signal"].iloc[-1])
                current_price = float(df["Close"].iloc[-1])

                from strategy_simple import BUY, SELL, HOLD
                if last_signal == BUY:
                    action = "BUY"
                elif last_signal == SELL:
                    action = "SELL"
                else:
                    action = "HOLD"

                action_dict = {
                    "symbol": symbol,
                    "action": action,
                    "price": current_price,
                    "shares": 0.0,
                    "stop_price": 0.0,
                    "take_profit_price": 0.0,
                    "reason": "Strategy signal",
                    "blocked_by": None,
                }

                if action == "BUY":
                    if symbol in self.state.open_positions:
                        action_dict["action"] = "HOLD"
                        action_dict["reason"] = "Already in position"
                    elif len(self.state.open_positions) >= self.max_positions:
                        action_dict["action"] = "HOLD"
                        action_dict["blocked_by"] = "Max positions reached"
                    elif self.regime_filter is not None and not self.regime_filter.is_bullish():
                        action_dict["action"] = "HOLD"
                        action_dict["blocked_by"] = "Regime not bullish"
                    elif self.earnings_filter is not None and self.earnings_filter.is_earnings_blackout(symbol, today):
                        action_dict["action"] = "HOLD"
                        action_dict["blocked_by"] = "Earnings blackout"
                    elif self.risk_manager is not None and not self.risk_manager.can_trade():
                        action_dict["action"] = "HOLD"
                        action_dict["blocked_by"] = getattr(self.risk_manager, "pause_reason", "Risk limit")
                    elif self.sizer is not None:
                        atr_series = self.sizer.calculate_atr(df)
                        atr = float(atr_series.iloc[-1])
                        if atr <= 0:
                            action_dict["action"] = "HOLD"
                            action_dict["blocked_by"] = "Invalid ATR"
                        else:
                            sizing = self.sizer.calculate_position_size(
                                price=current_price, atr=atr,
                                account_value=self.account_value,
                                risk_pct=self.risk_per_trade_pct,
                            )
                            shares = float(sizing.get("shares", 0.0))
                            stop_price = float(sizing.get("stop_price", current_price - atr * self.stop_atr_multiplier))
                            take_profit_price = current_price + (current_price - stop_price) * self.take_profit_multiplier
                            if shares <= 0:
                                action_dict["action"] = "HOLD"
                                action_dict["blocked_by"] = "Position size <= 0"
                            else:
                                action_dict.update({
                                    "shares": shares,
                                    "stop_price": stop_price,
                                    "take_profit_price": round(take_profit_price, 4),
                                    "atr": atr,
                                    "max_loss": float(sizing.get("max_loss", 0.0)),
                                })

                elif action == "SELL":
                    if symbol not in self.state.open_positions:
                        action_dict["action"] = "HOLD"
                        action_dict["reason"] = "No open position"

                actions.append(action_dict)

            except Exception as e:
                logger.error("Error scanning %s: %s", symbol, e)

        self.state.last_scan_time = datetime.utcnow().isoformat()
        return actions

    def execute_scan(self, dry_run: bool = True) -> list[dict]:
        """Execute or simulate scan actions. Saves state after execution."""
        actions = self.run_scan()
        execution_log: list[dict] = []

        for action in actions:
            symbol = action["symbol"]
            if action["action"] == "HOLD":
                if action.get("blocked_by"):
                    logger.info("⏸️  %s: HOLD — %s", symbol, action["blocked_by"])
                continue
            if action["action"] == "BUY" and float(action.get("shares", 0.0)) > 0:
                execution_log.append(self._execute_buy(symbol, action, dry_run))
            elif action["action"] == "SELL" and symbol in self.state.open_positions:
                execution_log.append(self._execute_sell(symbol, float(action["price"]), "Strategy SELL signal", dry_run))

        save_state(self.state, self.state_file)
        return execution_log

    def _execute_buy(self, symbol: str, action: dict, dry_run: bool) -> dict:
        """Handle BUY order execution or simulation."""
        price = float(action["price"])
        shares = float(action["shares"])
        stop_price = float(action["stop_price"])
        take_profit_price = float(action["take_profit_price"])
        timestamp = datetime.utcnow().isoformat()

        log_entry = {
            "action": "BUY", "symbol": symbol, "price": price, "shares": shares,
            "stop_price": stop_price, "take_profit_price": take_profit_price,
            "dry_run": dry_run, "timestamp": timestamp,
        }

        if dry_run:
            logger.info("🔵 [DRY RUN] BUY %s @ $%.2f | Stop: $%.2f | Target: $%.2f", symbol, price, stop_price, take_profit_price)
            log_entry["status"] = "dry_run"
        else:
            try:
                result = self.trader.place_market_order(symbol, "buy", notional=round(shares * price, 2))
                log_entry["order_id"] = result.get("order_id") if isinstance(result, dict) else getattr(result, "id", None)
                log_entry["status"] = "submitted"
                logger.info("✅ BUY order placed: %s", symbol)
            except Exception as e:
                logger.error("BUY failed for %s: %s", symbol, e)
                log_entry["status"] = "error"
                log_entry["error"] = str(e)
                return log_entry

        position = PositionRecord(
            symbol=symbol, entry_price=price, shares=shares,
            stop_price=stop_price, take_profit_price=take_profit_price,
            entry_time=timestamp, strategy_name=self.strategy_name,
            position_value=round(shares * price, 2),
            atr_at_entry=float(action.get("atr", 0.0)),
        )
        self.state.open_positions[symbol] = asdict(position)
        self.state.last_alerts[symbol] = "BUY"

        if self.paper_log:
            try:
                from strategy_simple import BUY as BUY_SIG
                self.paper_log.log_signal(symbol, BUY_SIG, price, self.strategy_name)
                self.paper_log.log_fill(symbol, "buy", shares, price, signal_price=price)
                self.paper_log.save()
            except Exception:
                pass

        if self.risk_manager:
            try:
                self.risk_manager.record_trade_open()
            except Exception:
                pass

        _send_telegram_alert(
            f"{'🔵 [DRY RUN] ' if dry_run else '✅ '}BUY <b>{symbol}</b>\n"
            f"Price: ${price:.2f} | Shares: {shares:.4f}\n"
            f"Stop: ${stop_price:.2f} | Target: ${take_profit_price:.2f}\n"
            f"Max Loss: ${action.get('max_loss', 0):.2f}"
        )
        return log_entry

    def _execute_sell(self, symbol: str, price: float, reason: str, dry_run: bool) -> dict:
        """Handle SELL/close order execution or simulation."""
        position = self.state.open_positions.get(symbol)
        if not position:
            return {"action": "SELL", "symbol": symbol, "status": "no_position"}

        entry_price = float(position.get("entry_price", price))
        shares = float(position.get("shares", 0.0))
        pnl = (price - entry_price) * shares
        return_pct = (price - entry_price) / entry_price * 100 if entry_price else 0.0
        timestamp = datetime.utcnow().isoformat()

        log_entry = {
            "action": "SELL", "symbol": symbol, "price": price, "shares": shares,
            "pnl": round(pnl, 2), "return_pct": round(return_pct, 2),
            "reason": reason, "dry_run": dry_run, "timestamp": timestamp,
        }

        if dry_run:
            logger.info("🔴 [DRY RUN] SELL %s @ $%.2f | PnL: $%.2f (%.1f%%)", symbol, price, pnl, return_pct)
            log_entry["status"] = "dry_run"
        else:
            try:
                result = self.trader.close_position(symbol)
                log_entry["order_id"] = result.get("order_id") if isinstance(result, dict) else None
                log_entry["status"] = "submitted"
                logger.info("✅ Position closed: %s | PnL: $%.2f", symbol, pnl)
            except Exception as e:
                logger.error("SELL failed for %s: %s", symbol, e)
                log_entry["status"] = "error"
                log_entry["error"] = str(e)
                return log_entry

        del self.state.open_positions[symbol]
        self.state.last_alerts[symbol] = "SELL"

        self.state.trade_journal.append({
            "symbol": symbol, "entry_price": entry_price, "exit_price": price,
            "shares": shares, "pnl": round(pnl, 2), "return_pct": round(return_pct, 2),
            "entry_time": position.get("entry_time", ""), "exit_time": timestamp,
            "reason": reason, "strategy": position.get("strategy_name", self.strategy_name),
        })

        if self.paper_log:
            try:
                self.paper_log.log_close(symbol, pnl, return_pct / 100)
                self.paper_log.save()
            except Exception:
                pass

        if self.risk_manager:
            try:
                self.risk_manager.record_trade_close(pnl)
            except Exception:
                pass

        icon = "✅" if pnl >= 0 else "❌"
        _send_telegram_alert(
            f"{'🔴 [DRY RUN] ' if dry_run else icon + ' '}SELL <b>{symbol}</b>\n"
            f"Price: ${price:.2f} | PnL: ${pnl:.2f} ({return_pct:.1f}%)\n"
            f"Reason: {reason}"
        )
        return log_entry


# ── Part 3: Monitoring + top-level functions ──────────────────────────────────

def check_stops(self, dry_run: bool = True) -> list[dict]:
    """Check all open positions against stop-loss and take-profit levels."""
    if not self.state.open_positions:
        return []

    closed: list[dict] = []

    for symbol in list(self.state.open_positions.keys()):
        try:
            pos = self.state.open_positions.get(symbol)
            if not pos:
                continue
            ticker = yf.Ticker(symbol)
            fast_info = getattr(ticker, "fast_info", None)
            current_price = getattr(fast_info, "last_price", None) if fast_info else None
            if not current_price:
                continue

            stop = pos["stop_price"]
            target = pos["take_profit_price"]
            entry = pos["entry_price"]

            if current_price <= stop:
                logger.warning("🛑 STOP LOSS triggered: %s @ $%.2f", symbol, current_price)
                result = self._execute_sell(symbol, float(current_price), "Stop loss hit", dry_run)
                closed.append(result)
            elif current_price >= target:
                logger.info("🎯 TAKE PROFIT triggered: %s @ $%.2f", symbol, current_price)
                result = self._execute_sell(symbol, float(current_price), "Take profit hit", dry_run)
                closed.append(result)
            else:
                change_pct = ((float(current_price) - float(entry)) / float(entry)) * 100 if entry else 0.0
                logger.info("📊 %s: $%.2f (%+.1f%%)", symbol, float(current_price), change_pct)

        except Exception as e:
            logger.error("Stop check failed for %s: %s", symbol, e)

    if closed:
        save_state(self.state, self.state_file)
    return closed


def reconcile_with_alpaca(self) -> dict:
    """Compare internal state with actual Alpaca positions."""
    report = {
        "status": "ok",
        "internal_positions": list(self.state.open_positions.keys()),
        "alpaca_positions": [],
        "missing_from_state": [],
        "missing_from_alpaca": [],
        "discrepancies": [],
    }

    if not getattr(self, "trader", None) or not getattr(self.trader, "available", False):
        report["status"] = "alpaca_unavailable"
        logger.warning("Alpaca unavailable — skipping reconciliation")
        return report

    try:
        alpaca_positions = self.trader.get_positions() or []
        alpaca_symbols = [p["symbol"] for p in alpaca_positions]
        report["alpaca_positions"] = alpaca_symbols
        internal_symbols = report["internal_positions"]

        for sym in alpaca_symbols:
            if sym not in internal_symbols:
                msg = f"{sym} in Alpaca but not tracked internally"
                report["missing_from_state"].append(sym)
                report["discrepancies"].append(msg)
                logger.warning("Reconciliation: %s", msg)

        for sym in internal_symbols:
            if sym not in alpaca_symbols:
                msg = f"{sym} tracked internally but not in Alpaca"
                report["missing_from_alpaca"].append(sym)
                report["discrepancies"].append(msg)
                logger.warning("Reconciliation: %s", msg)

        if report["discrepancies"]:
            report["status"] = "discrepancies_found"
        else:
            logger.info("✅ Reconciliation clean")

    except Exception as e:
        report["status"] = "error"
        report["error"] = str(e)
        logger.error("Reconciliation failed: %s", e)

    return report


def get_status(self) -> dict:
    """Return full current system status snapshot."""
    journal = self.state.trade_journal
    wins = [t for t in journal if t.get("pnl", 0) > 0]
    total_pnl = sum(t.get("pnl", 0) for t in journal)
    total_trades = len(journal)
    win_rate = round(len(wins) / total_trades * 100, 2) if total_trades else 0.0
    risk_status = self.risk_manager.get_status() if self.risk_manager else {}

    return {
        "session_id": self.state.session_id,
        "open_positions": self.state.open_positions,
        "open_count": len(self.state.open_positions),
        "last_scan": self.state.last_scan_time,
        "account_snapshot": self.state.account_snapshot,
        "trade_journal": {
            "total": total_trades,
            "wins": len(wins),
            "losses": total_trades - len(wins),
            "win_rate": win_rate,
            "total_pnl": round(total_pnl, 2),
        },
        "risk_status": risk_status,
        "strategy": self.strategy_name,
        "symbols": self.symbols,
        "paper_mode": self.paper_mode,
    }


# Attach Part 3 methods to class
PortfolioEngine.check_stops = check_stops
PortfolioEngine.reconcile_with_alpaca = reconcile_with_alpaca
PortfolioEngine.get_status = get_status
PortfolioEngine.save_state = lambda self: save_state(self.state, self.state_file)
PortfolioEngine.load_state = lambda self: load_state(self.state_file)


# ── Top-level functions ───────────────────────────────────────────────────────

def run_daily_session(
    symbols=None,
    strategy_name: str = "CombinedSignal",
    account_value: float = 1200.0,
    dry_run: bool = True,
    stop_check_interval_minutes: int = 30,
):
    """Run one complete trading session with full orchestration loop."""
    if symbols is None:
        from watchlist import DEFAULT_WATCHLIST
        symbols = DEFAULT_WATCHLIST

    engine = PortfolioEngine(symbols=symbols, strategy_name=strategy_name, account_value=account_value, paper_mode=True)
    mode = "DRY RUN" if dry_run else "PAPER"
    logger.info("🚀 Daily session [%s] | Strategy=%s | Symbols=%s", mode, strategy_name, symbols)

    # Market hours check
    market_open = False
    try:
        if engine.trader and getattr(engine.trader, "available", False):
            market_open = bool(engine.trader.is_market_open())
        else:
            utc_hour = datetime.now(timezone.utc).hour
            market_open = 13 <= utc_hour <= 20
    except Exception as e:
        logger.warning("Market check failed: %s", e)
        market_open = False

    if not market_open:
        logger.info("Market closed — running preview scan only")

    # Reset daily risk
    if engine.risk_manager:
        try:
            engine.risk_manager.reset_daily()
        except Exception:
            pass

    # Reconcile + scan
    recon = engine.reconcile_with_alpaca()
    actions = engine.execute_scan(dry_run=dry_run)

    # Stop monitoring loop
    if market_open and engine.state.open_positions:
        for i in range(14):
            if not engine.state.open_positions:
                break
            logger.info("Stop check %d/14 in %d min", i + 1, stop_check_interval_minutes)
            time.sleep(stop_check_interval_minutes * 60)
            engine.check_stops(dry_run=dry_run)

    # Summary
    status = engine.get_status()
    summary = (
        f"📊 Mini-Aladdin Session Complete\n"
        f"Mode: {mode} | Strategy: {strategy_name}\n"
        f"Open positions: {status['open_count']}\n"
        f"Trades: {status['trade_journal']['total']} | Win rate: {status['trade_journal']['win_rate']}%\n"
        f"Total PnL: ${status['trade_journal']['total_pnl']:,.2f}"
    )
    logger.info(summary)
    _send_telegram_alert(summary)
    engine.save_state()
    return status


def run_realistic_backtest(
    symbol: str,
    strategy_name: str = "CombinedSignal",
    start_date: str = "2022-01-01",
    end_date: str = "2024-01-01",
    initial_capital: float = 1200.0,
):
    """
    Backtest with regime filter + ATR stop-loss applied.
    More honest than standard backtest — reflects live trading conditions.
    """
    from backtest_simple import run_backtest
    from strategy_simple import build_strategy
    from regime_filter import RegimeAwareStrategy

    base_strategy = build_strategy(strategy_name)
    aware_strategy = RegimeAwareStrategy(strategy=base_strategy, symbol=symbol, use_regime=True, use_earnings_filter=True)
    result = run_backtest(symbol, aware_strategy, start_date, end_date, initial_capital, 0.001, 0.001, stop_loss_pct=0.05, take_profit_pct=0.10)
    logger.info("Realistic backtest: %s | Return=%.2f%% | Sharpe=%.3f | Trades=%d",
                symbol, result.metrics["total_return"], result.metrics["sharpe_ratio"], result.metrics["total_trades"])
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "dry"

    if mode == "dry":
        print("Running DRY RUN scan...")
        engine = PortfolioEngine(symbols=["AAPL", "NVDA", "MSFT"], strategy_name="CombinedSignal")
        actions = engine.run_scan()
        for a in actions:
            blocked = f" | blocked: {a['blocked_by']}" if a.get("blocked_by") else ""
            print(f"{a['symbol']}: {a['action']} @ ${a['price']:.2f}{blocked}")
        print("\nStatus:", engine.get_status())

    elif mode == "session":
        print("Running full daily session (dry run)...")
        run_daily_session(symbols=["AAPL", "NVDA", "MSFT"], dry_run=True)

    elif mode == "backtest":
        print("Running realistic backtest on NVDA...")
        result = run_realistic_backtest("NVDA", "RSIMeanReversion", "2020-01-01", "2024-01-01")
        print(f"Return: {result.metrics['total_return']}% | Sharpe: {result.metrics['sharpe_ratio']}")
