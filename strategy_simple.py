"""
strategy_simple.py — Strategy Definitions (compatible with Python 3.9-3.11)
Uses 'ta' library instead of pandas-ta for broader compatibility.
"""

import abc
import pandas as pd
import ta

BUY = 1
SELL = -1
HOLD = 0


class Strategy(abc.ABC):
    def __init__(self, **params):
        self.params = params

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def _validate_df(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return df.copy()


class MovingAverageCrossover(Strategy):
    """Buy when fast EMA crosses above slow EMA. Sell on crossunder."""

    def __init__(self, fast_period=10, slow_period=50):
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")
        super().__init__(fast_period=fast_period, slow_period=slow_period)
        self.fast_period = fast_period
        self.slow_period = slow_period

    @property
    def name(self):
        return f"MA Crossover ({self.fast_period}/{self.slow_period})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_df(df)
        df["fast_ma"] = ta.trend.ema_indicator(df["Close"], window=self.fast_period)
        df["slow_ma"] = ta.trend.ema_indicator(df["Close"], window=self.slow_period)
        df["signal"] = HOLD

        bullish = (df["fast_ma"] > df["slow_ma"]) & (df["fast_ma"].shift(1) <= df["slow_ma"].shift(1))
        bearish = (df["fast_ma"] < df["slow_ma"]) & (df["fast_ma"].shift(1) >= df["slow_ma"].shift(1))

        df.loc[bullish, "signal"] = BUY
        df.loc[bearish, "signal"] = SELL
        df.dropna(subset=["fast_ma", "slow_ma"], inplace=True)
        return df


class RSIMeanReversion(Strategy):
    """Buy when RSI is oversold. Sell when overbought."""

    def __init__(self, rsi_period=14, oversold=30, overbought=70):
        super().__init__(rsi_period=rsi_period, oversold=oversold, overbought=overbought)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    @property
    def name(self):
        return f"RSI ({self.rsi_period}, {self.oversold}/{self.overbought})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_df(df)
        df["rsi"] = ta.momentum.rsi(df["Close"], window=self.rsi_period)
        df["signal"] = HOLD
        df.loc[df["rsi"] < self.oversold, "signal"] = BUY
        df.loc[df["rsi"] > self.overbought, "signal"] = SELL
        df.dropna(subset=["rsi"], inplace=True)
        return df


class MACDStrategy(Strategy):
    """Buy on MACD bullish crossover. Sell on bearish crossover."""

    def __init__(self, fast=12, slow=26, signal=9):
        super().__init__(fast=fast, slow=slow, signal=signal)
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    @property
    def name(self):
        return f"MACD ({self.fast}/{self.slow}/{self.signal_period})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_df(df)
        macd = ta.trend.MACD(df["Close"], window_fast=self.fast, window_slow=self.slow, window_sign=self.signal_period)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["signal"] = HOLD

        bullish = (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
        bearish = (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))

        df.loc[bullish, "signal"] = BUY
        df.loc[bearish, "signal"] = SELL
        df.dropna(subset=["macd", "macd_signal"], inplace=True)
        return df


class BollingerBandsStrategy(Strategy):
    """
    Bollinger Bands Breakout Strategy.
    BUY when price touches lower band (oversold).
    SELL when price touches upper band (overbought).
    """
    def __init__(self, window=20, std_dev=2.0):
        super().__init__(window=window, std_dev=std_dev)
        self.window = window
        self.std_dev = std_dev

    @property
    def name(self):
        return f"Bollinger Bands ({self.window}, {self.std_dev}σ)"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_df(df)
        indicator = ta.volatility.BollingerBands(
            df["Close"], window=self.window, window_dev=self.std_dev
        )
        df["bb_upper"] = indicator.bollinger_hband()
        df["bb_lower"] = indicator.bollinger_lband()
        df["bb_mid"] = indicator.bollinger_mavg()
        df["signal"] = HOLD
        df.loc[df["Close"] <= df["bb_lower"], "signal"] = BUY
        df.loc[df["Close"] >= df["bb_upper"], "signal"] = SELL
        df.dropna(subset=["bb_upper", "bb_lower"], inplace=True)
        return df


class MomentumStrategy(Strategy):
    """
    Momentum Strategy.
    BUY when price is making new N-day highs (strength).
    SELL when price breaks below N-day low.
    """
    def __init__(self, window=20):
        super().__init__(window=window)
        self.window = window

    @property
    def name(self):
        return f"Momentum ({self.window}d)"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_df(df)
        df["rolling_high"] = df["Close"].rolling(window=self.window).max()
        df["rolling_low"] = df["Close"].rolling(window=self.window).min()
        df["signal"] = HOLD
        # BUY when today's close equals the N-day high (breakout)
        breakout_up = df["Close"] >= df["rolling_high"]
        breakout_down = df["Close"] <= df["rolling_low"]
        df.loc[breakout_up, "signal"] = BUY
        df.loc[breakout_down, "signal"] = SELL
        df.dropna(subset=["rolling_high", "rolling_low"], inplace=True)
        return df


class CombinedSignalStrategy(Strategy):
    """
    Combined Signal Strategy — only trades when majority of strategies agree.
    Higher confidence = higher win rate.
    Uses MA Crossover + RSI + MACD voting.
    """
    def __init__(self, required_votes=2):
        super().__init__(required_votes=required_votes)
        self.required_votes = required_votes
        self._strategies = [
            MovingAverageCrossover(fast_period=10, slow_period=50),
            RSIMeanReversion(rsi_period=14, oversold=35, overbought=65),
            MACDStrategy(fast=12, slow=26, signal=9),
        ]

    @property
    def name(self):
        return f"Combined Signal ({self.required_votes}/3 votes)"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_df(df)
        vote_df = pd.DataFrame(index=df.index)

        for s in self._strategies:
            try:
                result = s.generate_signals(df.copy())
                vote_df[s.name] = result["signal"].reindex(df.index).fillna(HOLD)
            except Exception:
                vote_df[s.name] = HOLD

        buy_votes = (vote_df == BUY).sum(axis=1)
        sell_votes = (vote_df == SELL).sum(axis=1)

        df["signal"] = HOLD
        df.loc[buy_votes >= self.required_votes, "signal"] = BUY
        df.loc[sell_votes >= self.required_votes, "signal"] = SELL
        return df


STRATEGY_REGISTRY = {
    "MovingAverageCrossover": {
        "class": MovingAverageCrossover,
        "default_params": {"fast_period": 10, "slow_period": 50},
        "param_defs": {
            "fast_period": {"type": "int", "min": 2, "max": 50, "default": 10},
            "slow_period": {"type": "int", "min": 10, "max": 200, "default": 50},
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
    "BollingerBands": {
        "class": BollingerBandsStrategy,
        "default_params": {"window": 20, "std_dev": 2.0},
        "param_defs": {
            "window": {"type": "int", "min": 5, "max": 50, "default": 20},
        },
    },
    "Momentum": {
        "class": MomentumStrategy,
        "default_params": {"window": 20},
        "param_defs": {
            "window": {"type": "int", "min": 5, "max": 60, "default": 20},
        },
    },
    "CombinedSignal": {
        "class": CombinedSignalStrategy,
        "default_params": {"required_votes": 2},
        "param_defs": {
            "required_votes": {"type": "int", "min": 1, "max": 3, "default": 2},
        },
    },
}


def build_strategy(name: str, **params) -> Strategy:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY.keys())}")
    entry = STRATEGY_REGISTRY[name]
    merged = {**entry["default_params"], **params}
    return entry["class"](**merged)
