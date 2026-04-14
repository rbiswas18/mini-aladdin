"""
validator.py — Walk-Forward Validation + Paper Trade Log
Tests strategy robustness across time periods and tracks paper trades.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from backtest_simple import run_backtest
from strategy_simple import Strategy

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward validation: tests a strategy across multiple time periods.

    Why this matters: A strategy that "works" on 3 years of NVDA data
    might only work because it caught one lucky trend. Walk-forward
    testing checks if it works CONSISTENTLY across different periods.

    Method:
    - Split total date range into n_splits windows
    - Each window: 70% in-sample (train), 30% out-of-sample (test)
    - Run backtest only on out-of-sample periods
    - Aggregate results → consistency score → verdict
    """

    def run(
        self,
        symbol: str,
        strategy: Strategy,
        start_date: str,
        end_date: str,
        n_splits: int = 5,
        initial_capital: float = 10_000.0,
    ) -> dict:
        """
        Run walk-forward validation.

        Args:
            symbol: Ticker symbol
            strategy: Strategy instance
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'
            n_splits: Number of time windows to test
            initial_capital: Starting capital per split

        Returns:
            Validation results dict with splits, averages, verdict
        """
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        total_days = (end_dt - start_dt).days

        if total_days < 200:
            return {
                "error": "Not enough data — need at least 200 days for walk-forward testing",
                "verdict": "UNRELIABLE",
                "is_robust": False,
            }

        window_days = total_days // n_splits
        splits = []

        for i in range(n_splits):
            window_start = start_dt + timedelta(days=i * window_days)
            window_end = window_start + timedelta(days=window_days)
            if window_end > end_dt:
                window_end = end_dt

            # 70/30 split
            split_days = (window_end - window_start).days
            train_days = int(split_days * 0.7)

            train_start = window_start
            train_end = window_start + timedelta(days=train_days)
            test_start = train_end
            test_end = window_end

            if (test_end - test_start).days < 20:
                continue  # skip tiny test periods

            try:
                result = run_backtest(
                    symbol=symbol,
                    strategy=strategy,
                    start_date=test_start.strftime("%Y-%m-%d"),
                    end_date=test_end.strftime("%Y-%m-%d"),
                    initial_capital=initial_capital,
                )
                splits.append({
                    "split": i + 1,
                    "train_start": train_start.strftime("%Y-%m-%d"),
                    "train_end": train_end.strftime("%Y-%m-%d"),
                    "test_start": test_start.strftime("%Y-%m-%d"),
                    "test_end": test_end.strftime("%Y-%m-%d"),
                    "metrics": result.metrics,
                })
            except Exception as e:
                logger.warning(f"Split {i+1} failed: {e}")
                splits.append({
                    "split": i + 1,
                    "train_start": train_start.strftime("%Y-%m-%d"),
                    "train_end": train_end.strftime("%Y-%m-%d"),
                    "test_start": test_start.strftime("%Y-%m-%d"),
                    "test_end": test_end.strftime("%Y-%m-%d"),
                    "metrics": None,
                    "error": str(e),
                })

        # Aggregate metrics
        valid_splits = [s for s in splits if s["metrics"] is not None]

        if not valid_splits:
            return {
                "splits": splits,
                "verdict": "UNRELIABLE",
                "is_robust": False,
                "error": "All splits failed",
            }

        returns = [s["metrics"]["total_return"] for s in valid_splits]
        sharpes = [s["metrics"]["sharpe_ratio"] for s in valid_splits]
        drawdowns = [s["metrics"]["max_drawdown"] for s in valid_splits]
        win_rates = [s["metrics"]["win_rate"] for s in valid_splits]

        avg_return = sum(returns) / len(returns)
        avg_sharpe = sum(sharpes) / len(sharpes)
        avg_drawdown = sum(drawdowns) / len(drawdowns)
        avg_win_rate = sum(win_rates) / len(win_rates)

        positive_splits = sum(1 for r in returns if r > 0)
        consistency_score = positive_splits / len(valid_splits)

        is_robust = consistency_score >= 0.6 and avg_sharpe >= 0.5

        if consistency_score >= 0.7 and avg_sharpe >= 0.8:
            verdict = "ROBUST"
        elif consistency_score >= 0.5 and avg_sharpe >= 0.3:
            verdict = "MARGINAL"
        else:
            verdict = "UNRELIABLE"

        return {
            "symbol": symbol,
            "strategy": strategy.name,
            "n_splits": n_splits,
            "valid_splits": len(valid_splits),
            "splits": splits,
            "avg_return": round(avg_return, 2),
            "avg_sharpe": round(avg_sharpe, 3),
            "avg_max_drawdown": round(avg_drawdown, 2),
            "avg_win_rate": round(avg_win_rate, 2),
            "consistency_score": round(consistency_score, 3),
            "positive_splits": positive_splits,
            "is_robust": is_robust,
            "verdict": verdict,
        }


class PaperTradeLog:
    """
    Logs every paper trade signal and simulated fill.
    Tracks expected vs actual fills to measure execution quality.
    Persists to JSON for review across sessions.
    """

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath or "paper_trades.json"
        self._log: list[dict] = []
        self._load_if_exists()

    def _load_if_exists(self):
        try:
            p = Path(self.filepath)
            if p.exists():
                with open(p) as f:
                    self._log = json.load(f)
        except Exception:
            self._log = []

    def log_signal(
        self,
        symbol: str,
        signal: int,
        price: float,
        strategy_name: str,
        timestamp: Optional[datetime] = None,
    ):
        """Log a trading signal."""
        from strategy_simple import BUY, SELL, HOLD
        action = "BUY" if signal == BUY else "SELL" if signal == SELL else "HOLD"
        self._log.append({
            "type": "signal",
            "symbol": symbol,
            "action": action,
            "signal_price": round(price, 4),
            "strategy": strategy_name,
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
        })

    def log_fill(
        self,
        symbol: str,
        side: str,
        shares: float,
        fill_price: float,
        timestamp: Optional[datetime] = None,
        order_id: Optional[str] = None,
        signal_price: Optional[float] = None,
    ):
        """Log a simulated order fill."""
        slippage = None
        if signal_price and signal_price > 0:
            slippage = round((fill_price - signal_price) / signal_price * 100, 4)

        self._log.append({
            "type": "fill",
            "symbol": symbol,
            "side": side,
            "shares": round(shares, 4),
            "fill_price": round(fill_price, 4),
            "signal_price": signal_price,
            "slippage_pct": slippage,
            "order_id": order_id,
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
        })

    def log_close(
        self,
        symbol: str,
        pnl: float,
        return_pct: float,
        timestamp: Optional[datetime] = None,
    ):
        """Log a trade closure with P&L."""
        self._log.append({
            "type": "close",
            "symbol": symbol,
            "pnl": round(pnl, 2),
            "return_pct": round(return_pct, 4),
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
        })

    def get_log(self) -> pd.DataFrame:
        """Get full log as DataFrame."""
        if not self._log:
            return pd.DataFrame()
        return pd.DataFrame(self._log)

    def get_summary(self) -> dict:
        """Get summary of all paper trades."""
        closes = [e for e in self._log if e["type"] == "close"]
        fills = [e for e in self._log if e["type"] == "fill"]

        if not closes:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_return_pct": 0.0,
                "avg_slippage_pct": 0.0,
            }

        total_pnl = sum(c["pnl"] for c in closes)
        winners = [c for c in closes if c["pnl"] > 0]
        win_rate = len(winners) / len(closes) * 100
        avg_return = sum(c["return_pct"] for c in closes) / len(closes)

        slippages = [f["slippage_pct"] for f in fills if f.get("slippage_pct") is not None]
        avg_slippage = sum(slippages) / len(slippages) if slippages else 0.0

        return {
            "total_trades": len(closes),
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "avg_return_pct": round(avg_return, 4),
            "avg_slippage_pct": round(avg_slippage, 4),
        }

    def save(self, filepath: Optional[str] = None):
        """Save log to JSON file."""
        path = filepath or self.filepath
        try:
            with open(path, "w") as f:
                json.dump(self._log, f, indent=2)
            logger.info(f"Paper trade log saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save log: {e}")

    def load(self, filepath: Optional[str] = None):
        """Load log from JSON file."""
        path = filepath or self.filepath
        try:
            with open(path) as f:
                self._log = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load log: {e}")

    def clear(self):
        """Clear all logs."""
        self._log = []


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit tab code (paste into app.py tabs)
# ─────────────────────────────────────────────────────────────────────────────

VALIDATE_TAB_CODE = '''
# ── Validate Tab ──────────────────────────────────────────────────────────────
# Add this as a new tab in app.py:
# tab1, tab2, tab3, tab4 = st.tabs(["📊 Backtest", "⚖️ Compare All", "⚡ Live Quote", "🔬 Validate"])

import streamlit as st
from validator import WalkForwardValidator, PaperTradeLog
from strategy_simple import build_strategy

st.header("🔬 Walk-Forward Validation")
st.caption("Tests if your strategy is consistently profitable across different time periods — not just lucky once.")

col1, col2 = st.columns(2)
with col1:
    val_symbol = st.text_input("Symbol to validate", value="AAPL").upper()
    val_strategy = st.selectbox("Strategy to validate", list(STRATEGY_REGISTRY.keys()), key="val_strategy")
with col2:
    val_splits = st.slider("Number of test periods", 3, 8, 5)
    val_capital = st.number_input("Capital per period ($)", value=10000, min_value=1000)

if st.button("🔬 Run Validation", type="primary"):
    with st.spinner("Running walk-forward validation... this takes ~30 seconds"):
        try:
            validator = WalkForwardValidator()
            strategy = build_strategy(val_strategy)
            results = validator.run(
                symbol=val_symbol,
                strategy=strategy,
                start_date=str(date.today() - timedelta(days=4*365)),
                end_date=str(date.today() - timedelta(days=1)),
                n_splits=val_splits,
                initial_capital=val_capital,
            )
            st.session_state["validation"] = results
        except Exception as e:
            st.error(f"Validation failed: {e}")

val = st.session_state.get("validation")
if val:
    verdict = val.get("verdict", "UNRELIABLE")
    if verdict == "ROBUST":
        st.success(f"✅ ROBUST — This strategy is consistently profitable ({val['consistency_score']:.0%} of periods)")
    elif verdict == "MARGINAL":
        st.warning(f"⚠️ MARGINAL — Mixed results. Use with caution ({val['consistency_score']:.0%} of periods profitable)")
    else:
        st.error(f"❌ UNRELIABLE — This strategy is inconsistent. Do not trade live ({val['consistency_score']:.0%} of periods profitable)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Return", f"{val.get('avg_return', 0):.1f}%")
    col2.metric("Avg Sharpe", f"{val.get('avg_sharpe', 0):.2f}")
    col3.metric("Avg Max Drawdown", f"{val.get('avg_max_drawdown', 0):.1f}%")
    col4.metric("Consistency", f"{val.get('consistency_score', 0):.0%}")

    st.divider()
    st.subheader("Per-Period Results")
    splits_data = []
    for s in val.get("splits", []):
        if s.get("metrics"):
            splits_data.append({
                "Period": s["split"],
                "Test Start": s["test_start"],
                "Test End": s["test_end"],
                "Return %": s["metrics"]["total_return"],
                "Sharpe": s["metrics"]["sharpe_ratio"],
                "Win Rate %": s["metrics"]["win_rate"],
                "Max DD %": s["metrics"]["max_drawdown"],
                "Trades": s["metrics"]["total_trades"],
            })
    if splits_data:
        st.dataframe(pd.DataFrame(splits_data), use_container_width=True)

st.divider()
st.subheader("📋 Paper Trade Log")
log = PaperTradeLog()
summary = log.get_summary()
if summary["total_trades"] > 0:
    col1, col2, col3 = st.columns(3)
    col1.metric("Paper Trades", summary["total_trades"])
    col2.metric("Win Rate", f"{summary['win_rate']}%")
    col3.metric("Total P&L", f"${summary['total_pnl']:,.2f}")
    st.dataframe(log.get_log(), use_container_width=True)
else:
    st.info("No paper trades logged yet. Paper trading will be added in the next update.")
'''


if __name__ == "__main__":
    from strategy_simple import build_strategy

    validator = WalkForwardValidator()
    strategy = build_strategy("RSIMeanReversion")
    results = validator.run("AAPL", strategy, "2021-01-01", "2024-01-01", n_splits=4)

    print(f"Verdict: {results['verdict']}")
    print(f"Consistency: {results['consistency_score']:.0%}")
    print(f"Avg Return: {results['avg_return']}%")
    print(f"Avg Sharpe: {results['avg_sharpe']}")
