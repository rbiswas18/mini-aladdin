"""
backtest.py — Backtesting Engine
Uses VectorBT for fast, vectorized backtesting with realistic cost modeling.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import vectorbt as vbt

from data_fetch import get_provider
from strategy import Strategy, build_strategy, STRATEGY_REGISTRY, BUY, SELL

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """
    Container for backtest output data.
    Passed to the dashboard for visualization and to the agent for analysis.
    """
    symbol: str
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict
    signals_df: pd.DataFrame  # full bar data with signals


def _compute_metrics(portfolio: vbt.Portfolio, initial_capital: float) -> dict:
    """
    Extract key performance metrics from a VectorBT Portfolio object.

    Returns:
        dict with: total_return, cagr, sharpe_ratio, max_drawdown,
                   win_rate, profit_factor, total_trades
    """
    try:
        stats = portfolio.stats()
        total_return = float(portfolio.total_return())
        max_dd = float(portfolio.max_drawdown())

        # CAGR
        n_days = (portfolio.wrapper.index[-1] - portfolio.wrapper.index[0]).days
        n_years = n_days / 365.25
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

        # Sharpe (annualized, daily returns)
        daily_returns = portfolio.returns()
        sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() != 0 else 0.0

        # Trade stats
        trades_df = portfolio.trades.records_readable
        total_trades = len(trades_df)

        if total_trades > 0:
            winning = trades_df[trades_df["PnL"] > 0]
            losing = trades_df[trades_df["PnL"] <= 0]
            win_rate = len(winning) / total_trades
            gross_profit = winning["PnL"].sum() if len(winning) > 0 else 0
            gross_loss = abs(losing["PnL"].sum()) if len(losing) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        else:
            win_rate = 0.0
            profit_factor = 0.0

        return {
            "total_return": round(total_return * 100, 2),       # %
            "cagr": round(cagr * 100, 2),                       # %
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown": round(max_dd * 100, 2),             # %
            "win_rate": round(win_rate * 100, 2),               # %
            "profit_factor": round(profit_factor, 3),
            "total_trades": total_trades,
            "final_value": round(float(portfolio.final_value()), 2),
            "initial_capital": round(initial_capital, 2),
        }
    except Exception as e:
        logger.warning(f"Metrics computation error: {e}")
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "final_value": initial_capital,
            "initial_capital": initial_capital,
        }


class BacktestEngine:
    """
    Vectorized backtesting engine wrapping VectorBT.

    Features:
    - Configurable commission and slippage
    - Realistic position sizing (% of equity)
    - Clean BacktestResult output for dashboard + agent
    """

    def __init__(self, provider_name: str = "yfinance"):
        self.provider = get_provider(provider_name)

    def run(
        self,
        symbol: str,
        strategy: Strategy,
        start_date: str,
        end_date: str,
        initial_capital: float = 10_000.0,
        commission: float = 0.001,       # 0.1% per trade
        slippage: float = 0.001,         # 0.1% slippage
        size_pct: float = 1.0,           # fraction of portfolio per trade (1.0 = all-in)
    ) -> BacktestResult:
        """
        Run a backtest for a single strategy on a single symbol.

        Args:
            symbol: Ticker symbol (e.g. 'AAPL')
            strategy: Instantiated Strategy object
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'
            initial_capital: Starting portfolio value in USD
            commission: Fraction cost per trade (0.001 = 0.1%)
            slippage: Fraction price impact per trade
            size_pct: Fraction of portfolio to allocate per position

        Returns:
            BacktestResult dataclass
        """
        symbol = symbol.upper().strip()
        logger.info(f"Running backtest: {symbol} | {strategy.name} | {start_date} → {end_date}")

        # 1. Fetch data
        df = self.provider.get_bars(symbol, "1d", start_date, end_date)

        # 2. Generate signals
        signals_df = strategy.generate_signals(df)

        # 3. Extract entry/exit boolean arrays for VectorBT
        entries = signals_df["signal"] == BUY
        exits = signals_df["signal"] == SELL

        # Ensure at least one entry/exit
        if not entries.any():
            logger.warning(f"No BUY signals generated for {symbol} with {strategy.name}")

        close_prices = signals_df["Close"]

        # 4. Apply slippage to fill prices
        fill_price = close_prices * (1 + slippage)

        # 5. Run VectorBT portfolio simulation
        try:
            portfolio = vbt.Portfolio.from_signals(
                close=close_prices,
                entries=entries,
                exits=exits,
                size=size_pct,
                size_type="valuepercent",
                init_cash=initial_capital,
                fees=commission,
                sl_stop=None,
                freq="D",
            )
        except Exception as e:
            raise RuntimeError(f"VectorBT simulation failed: {e}")

        # 6. Build equity curve
        equity_curve = portfolio.value()
        equity_curve.name = "Portfolio Value"

        # 7. Build trades DataFrame
        try:
            trades_df = portfolio.trades.records_readable.copy()
            if not trades_df.empty:
                trades_df = trades_df[["Entry Timestamp", "Exit Timestamp", "Size", "Avg Entry Price",
                                       "Avg Exit Price", "PnL", "Return"]].copy()
                trades_df.columns = ["Entry Date", "Exit Date", "Size", "Entry Price", "Exit Price", "PnL", "Return %"]
                trades_df["Return %"] = (trades_df["Return %"] * 100).round(2)
                trades_df["PnL"] = trades_df["PnL"].round(2)
                trades_df["Entry Price"] = trades_df["Entry Price"].round(4)
                trades_df["Exit Price"] = trades_df["Exit Price"].round(4)
        except Exception as e:
            logger.warning(f"Could not build trades DataFrame: {e}")
            trades_df = pd.DataFrame()

        # 8. Compute metrics
        metrics = _compute_metrics(portfolio, initial_capital)

        return BacktestResult(
            symbol=symbol,
            strategy_name=strategy.name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            equity_curve=equity_curve,
            trades=trades_df,
            metrics=metrics,
            signals_df=signals_df,
        )

    def compare_strategies(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 10_000.0,
        strategy_names: Optional[list] = None,
    ) -> list[BacktestResult]:
        """
        Run all (or specified) strategies on the same symbol/period and return ranked results.

        Args:
            symbol: Ticker symbol
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'
            initial_capital: Starting capital
            strategy_names: List of strategy names to compare (default: all in registry)

        Returns:
            List of BacktestResult objects, sorted by Sharpe ratio descending
        """
        if strategy_names is None:
            strategy_names = list(STRATEGY_REGISTRY.keys())

        results = []
        for name in strategy_names:
            try:
                strategy = build_strategy(name)
                result = self.run(symbol, strategy, start_date, end_date, initial_capital)
                results.append(result)
                logger.info(f"  {name}: Return={result.metrics['total_return']}%, Sharpe={result.metrics['sharpe_ratio']}")
            except Exception as e:
                logger.warning(f"Strategy {name} failed: {e}")

        # Rank by Sharpe ratio
        results.sort(key=lambda r: r.metrics.get("sharpe_ratio", -999), reverse=True)
        return results


def results_to_comparison_df(results: list[BacktestResult]) -> pd.DataFrame:
    """
    Convert a list of BacktestResult objects into a comparison DataFrame.
    Useful for dashboard display.
    """
    rows = []
    for r in results:
        row = {"Strategy": r.strategy_name}
        row.update(r.metrics)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Strategy")


if __name__ == "__main__":
    engine = BacktestEngine()
    strategy = build_strategy("MovingAverageCrossover", fast_period=10, slow_period=50)
    result = engine.run("SPY", strategy, "2022-01-01", "2024-01-01")

    print(f"\n{'='*50}")
    print(f"Backtest: {result.symbol} | {result.strategy_name}")
    print(f"{'='*50}")
    for k, v in result.metrics.items():
        print(f"  {k:20s}: {v}")

    print(f"\nTrades ({result.metrics['total_trades']}):")
    print(result.trades.head(10))
