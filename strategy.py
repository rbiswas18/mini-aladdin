"""
strategy.py — Strategy Definitions
Pluggable strategy classes that generate BUY/SELL/HOLD signals.
Each strategy uses pandas-ta for indicator calculation.
"""

import abc
import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)

# Signal constants
BUY = 1
SELL = -1
HOLD = 0


@dataclass
class StrategyConfig:
    """Serializable strategy configuration (used by agent layer)."""
    name: str
    params: dict = field(default_factory=dict)
    symbol: str = "SPY"
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"


class Strategy(abc.ABC):
    """
    Abstract base class for all trading strategies.

    Subclasses implement generate_signals() to produce entry/exit signals
    given a DataFrame of OHLCV bars.
    """

    def __init__(self, **params):
        self.params = params

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @abc.abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from OHLCV data.

        Args:
            df: DataFrame with columns Open, High, Low, Close, Volume

        Returns:
            DataFrame with original columns plus:
                - 'signal': int column (1=BUY, -1=SELL, 0=HOLD)
                - indicator columns used by this strategy
        """

    def _validate_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate input DataFrame has required columns."""
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
        return df.copy()


class MovingAverageCrossover(Strategy):
    """
    Moving Average Crossover Strategy.

    BUY when fast MA crosses above slow MA.
    SELL when fast MA crosses below slow MA.
    HOLD otherwise.

    Params:
        fast_period (int): Period for fast moving average. Default: 10
        slow_period (int): Period for slow moving average. Default: 50
        ma_type (str): 'sma' or 'ema'. Default: 'ema'
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 50, ma_type: str = "ema"):
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")
        super().__init__(fast_period=fast_period, slow_period=slow_period, ma_type=ma_type)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type.lower()

    @property
    def name(self) -> str:
        return f"MA Crossover ({self.fast_period}/{self.slow_period} {self.ma_type.upper()})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_df(df)

        if self.ma_type == "ema":
            df["fast_ma"] = ta.ema(df["Close"], length=self.fast_period)
            df["slow_ma"] = ta.ema(df["Close"], length=self.slow_period)
        else:
            df["fast_ma"] = ta.sma(df["Close"], length=self.fast_period)
            df["slow_ma"] = ta.sma(df["Close"], length=self.slow_period)

        df["signal"] = HOLD

        # Crossover: fast crosses above slow → BUY
        bullish_cross = (df["fast_ma"] > df["slow_ma"]) & (df["fast_ma"].shift(1) <= df["slow_ma"].shift(1))
        # Crossunder: fast crosses below slow → SELL
        bearish_cross = (df["fast_ma"] < df["slow_ma"]) & (df["fast_ma"].shift(1) >= df["slow_ma"].shift(1))

        df.loc[bullish_cross, "signal"] = BUY
        df.loc[bearish_cross, "signal"] = SELL

        df.dropna(subset=["fast_ma", "slow_ma"], inplace=True)
        return df


class RSIMeanReversion(Strategy):
    """
    RSI Mean Reversion Strategy.

    BUY when RSI drops below oversold threshold.
    SELL when RSI rises above overbought threshold.

    Params:
        rsi_period (int): RSI lookback period. Default: 14
        oversold (int): RSI level below which to BUY. Default: 30
        overbought (int): RSI level above which to SELL. Default: 70
    """

    def __init__(self, rsi_period: int = 14, oversold: int = 30, overbought: int = 70):
        if oversold >= overbought:
            raise ValueError("oversold must be less than overbought")
        super().__init__(rsi_period=rsi_period, oversold=oversold, overbought=overbought)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    @property
    def name(self) -> str:
        return f"RSI Mean Reversion ({self.rsi_period}, {self.oversold}/{self.overbought})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_df(df)

        df["rsi"] = ta.rsi(df["Close"], length=self.rsi_period)
        df["signal"] = HOLD

        # Enter when RSI crosses back above oversold from below
        buy_signal = (df["rsi"] < self.oversold) & (df["rsi"].shift(1) >= self.oversold)
        sell_signal = (df["rsi"] > self.overbought) & (df["rsi"].shift(1) <= self.overbought)

        # Simpler version: direct threshold
        buy_signal = df["rsi"] < self.oversold
        sell_signal = df["rsi"] > self.overbought

        df.loc[buy_signal, "signal"] = BUY
        df.loc[sell_signal, "signal"] = SELL

        df.dropna(subset=["rsi"], inplace=True)
        return df


class MACDStrategy(Strategy):
    """
    MACD Signal Line Crossover Strategy.

    BUY when MACD line crosses above signal line.
    SELL when MACD line crosses below signal line.

    Params:
        fast (int): Fast EMA period. Default: 12
        slow (int): Slow EMA period. Default: 26
        signal (int): Signal line smoothing period. Default: 9
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(fast=fast, slow=slow, signal=signal)
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    @property
    def name(self) -> str:
        return f"MACD ({self.fast}/{self.slow}/{self.signal_period})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_df(df)

        macd_result = ta.macd(df["Close"], fast=self.fast, slow=self.slow, signal=self.signal_period)

        if macd_result is None or macd_result.empty:
            raise RuntimeError("MACD calculation failed — not enough data points.")

        macd_col = f"MACD_{self.fast}_{self.slow}_{self.signal_period}"
        signal_col = f"MACDs_{self.fast}_{self.slow}_{self.signal_period}"
        hist_col = f"MACDh_{self.fast}_{self.slow}_{self.signal_period}"

        df["macd"] = macd_result[macd_col]
        df["macd_signal"] = macd_result[signal_col]
        df["macd_hist"] = macd_result[hist_col]
        df["signal"] = HOLD

        bullish = (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
        bearish = (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))

        df.loc[bullish, "signal"] = BUY
        df.loc[bearish, "signal"] = SELL

        df.dropna(subset=["macd", "macd_signal"], inplace=True)
        return df


# Registry: maps strategy name strings to classes + default params
STRATEGY_REGISTRY: dict[str, dict[str, Any]] = {
    "MovingAverageCrossover": {
        "class": MovingAverageCrossover,
        "default_params": {"fast_period": 10, "slow_period": 50, "ma_type": "ema"},
        "param_defs": {
            "fast_period": {"type": "int", "min": 2, "max": 50, "default": 10},
            "slow_period": {"type": "int", "min": 10, "max": 200, "default": 50},
            "ma_type": {"type": "select", "options": ["ema", "sma"], "default": "ema"},
        },
    },
    "RSIMeanReversion": {
        "class": RSIMeanReversion,
        "default_params": {"rsi_period": 14, "oversold": 30, "overbought": 70},
        "param_defs": {
            "rsi_period": {"type": "int", "min": 2, "max": 30, "default": 14},
            "oversold": {"type": "int", "min": 10, "max": 40, "default": 30},
            "overbought": {"type": "int", "min": 60, "max": 90, "default": 70},
        },
    },
    "MACDStrategy": {
        "class": MACDStrategy,
        "default_params": {"fast": 12, "slow": 26, "signal": 9},
        "param_defs": {
            "fast": {"type": "int", "min": 3, "max": 20, "default": 12},
            "slow": {"type": "int", "min": 10, "max": 50, "default": 26},
            "signal": {"type": "int", "min": 3, "max": 20, "default": 9},
        },
    },
}


def build_strategy(name: str, **params) -> Strategy:
    """
    Instantiate a strategy from the registry by name.

    Args:
        name: Strategy name (must be in STRATEGY_REGISTRY)
        **params: Strategy-specific parameters (overrides defaults)

    Returns:
        Instantiated Strategy object
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY.keys())}")

    entry = STRATEGY_REGISTRY[name]
    merged_params = {**entry["default_params"], **params}
    return entry["class"](**merged_params)


if __name__ == "__main__":
    from data_fetch import get_provider

    provider = get_provider("yfinance")
    df = provider.get_bars("SPY", "1d", "2022-01-01", "2024-01-01")

    for strategy_name in STRATEGY_REGISTRY:
        s = build_strategy(strategy_name)
        result = s.generate_signals(df.copy())
        buy_count = (result["signal"] == BUY).sum()
        sell_count = (result["signal"] == SELL).sum()
        print(f"{s.name}: {buy_count} BUY signals, {sell_count} SELL signals")
