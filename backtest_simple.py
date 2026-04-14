"""
backtest_simple.py — Lightweight Backtesting Engine
Pure pandas implementation. Works on Python 3.9+ with no heavy dependencies.
"""

import logging
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
import yfinance as yf

from strategy_simple import Strategy, build_strategy, STRATEGY_REGISTRY, BUY, SELL

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    symbol: str
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict
    signals_df: pd.DataFrame


def fetch_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol.upper())
    df = ticker.history(start=start, end=end, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def run_backtest(
    symbol: str,
    strategy: Strategy,
    start_date: str,
    end_date: str,
    initial_capital: float = 10_000.0,
    commission: float = 0.001,
    slippage: float = 0.001,
    stop_loss_pct: float = 0.0,   # 0 = disabled, 0.05 = 5% stop loss
    take_profit_pct: float = 0.0, # 0 = disabled, 0.10 = 10% take profit
) -> BacktestResult:

    df = fetch_data(symbol, start_date, end_date)
    signals_df = strategy.generate_signals(df)

    cash = initial_capital
    shares = 0.0
    equity_curve = []
    trades = []
    entry_price = 0.0
    entry_date = None

    for i, (date, row) in enumerate(signals_df.iterrows()):
        price = row["Close"] * (1 + slippage)
        signal = row["signal"]
        portfolio_value = cash + shares * row["Close"]
        equity_curve.append({"date": date, "value": portfolio_value})

        # BUY
        if signal == BUY and shares == 0 and cash > 0:
            shares_to_buy = (cash * (1 - commission)) / price
            shares = shares_to_buy
            entry_price = price
            entry_date = date
            cash = 0.0

        # Stop loss / take profit check
        if shares > 0 and entry_price > 0:
            change_pct = (row["Close"] - entry_price) / entry_price
            if stop_loss_pct > 0 and change_pct <= -stop_loss_pct:
                signal = SELL  # force exit
            elif take_profit_pct > 0 and change_pct >= take_profit_pct:
                signal = SELL  # force exit

        # SELL
        elif signal == SELL and shares > 0:
            sell_price = row["Close"] * (1 - slippage)
            proceeds = shares * sell_price * (1 - commission)
            pnl = proceeds - (shares * entry_price)
            ret = (sell_price - entry_price) / entry_price * 100
            trades.append({
                "Entry Date": entry_date,
                "Exit Date": date,
                "Entry Price": round(entry_price, 4),
                "Exit Price": round(sell_price, 4),
                "PnL": round(pnl, 2),
                "Return %": round(ret, 2),
            })
            cash = proceeds
            shares = 0.0
            entry_price = 0.0
            entry_date = None

    # Final value
    final_price = signals_df["Close"].iloc[-1]
    final_value = cash + shares * final_price
    equity_series = pd.Series(
        [e["value"] for e in equity_curve],
        index=[e["date"] for e in equity_curve],
        name="Portfolio Value"
    )

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    metrics = compute_metrics(equity_series, trades_df, initial_capital)

    return BacktestResult(
        symbol=symbol,
        strategy_name=strategy.name,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        equity_curve=equity_series,
        trades=trades_df,
        metrics=metrics,
        signals_df=signals_df,
    )


def compute_metrics(equity: pd.Series, trades: pd.DataFrame, initial_capital: float) -> dict:
    if len(equity) < 2:
        return {"total_return": 0.0, "cagr": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0,
                "win_rate": 0.0, "profit_factor": 0.0, "total_trades": 0,
                "final_value": initial_capital, "initial_capital": initial_capital}
    final_value = float(equity.iloc[-1])
    total_return = (final_value - initial_capital) / initial_capital * 100

    n_days = (equity.index[-1] - equity.index[0]).days
    n_years = n_days / 365.25
    cagr = ((final_value / initial_capital) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0

    daily_returns = equity.pct_change().dropna()
    sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() != 0 else 0.0

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max * 100
    max_drawdown = float(drawdown.min())

    total_trades = len(trades)
    if total_trades > 0:
        winners = trades[trades["PnL"] > 0]
        losers = trades[trades["PnL"] <= 0]
        win_rate = len(winners) / total_trades * 100
        gross_profit = winners["PnL"].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers["PnL"].sum()) if len(losers) > 0 else 0
        profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else 0.0
    else:
        win_rate = 0.0
        profit_factor = 0.0

    return {
        "total_return": round(total_return, 2),
        "cagr": round(cagr, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_drawdown, 2),
        "win_rate": round(win_rate, 2),
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "final_value": round(final_value, 2),
        "initial_capital": round(initial_capital, 2),
    }


def compare_strategies(
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 10_000.0,
) -> list[BacktestResult]:
    results = []
    for name in STRATEGY_REGISTRY:
        try:
            strategy = build_strategy(name)
            result = run_backtest(symbol, strategy, start_date, end_date, initial_capital)
            results.append(result)
        except Exception as e:
            logger.warning(f"{name} failed: {e}")
    results.sort(key=lambda r: r.metrics.get("sharpe_ratio", -999), reverse=True)
    return results


def results_to_df(results: list[BacktestResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"Strategy": r.strategy_name}
        row.update(r.metrics)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Strategy")


if __name__ == "__main__":
    strategy = build_strategy("MovingAverageCrossover")
    result = run_backtest("AAPL", strategy, "2022-01-01", "2024-01-01")
    print(f"Return: {result.metrics['total_return']}%")
    print(f"Sharpe: {result.metrics['sharpe_ratio']}")
    print(f"Trades: {result.metrics['total_trades']}")
