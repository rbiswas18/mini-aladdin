"""
strategy_pro.py — Professional-grade combination strategies for Trading Alpha.

These strategies avoid single-indicator retail logic and instead combine trend,
volume, volatility regime, and multi-horizon confirmation signals commonly used
in discretionary prop and systematic trading workflows.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import ta

from strategy_simple import Strategy, BUY, SELL, HOLD


class TrendVolumeConfirmationStrategy(Strategy):
    """
    Trend + Volume Confirmation (TVF).

    Professional logic:
    Strong directional trades are more reliable when price trend, participation,
    and trend strength all align. A move above or below the medium-term EMA is
    not enough on its own because low-volume moves are often fragile. This
    strategy only fires when price is aligned with trend, current volume is well
    above its rolling average, and ADX confirms that the market is actually
    trending rather than chopping sideways.

    BUY:
        - Close > EMA(ema_period)
        - Volume > volume_multiplier * average volume over volume_window
        - ADX(adx_period) > adx_threshold

    SELL:
        - Close < EMA(ema_period)
        - Volume > volume_multiplier * average volume over volume_window
        - ADX(adx_period) > adx_threshold
    """

    def __init__(
        self,
        ema_period: int = 50,
        volume_multiplier: float = 1.5,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        volume_window: int = 20,
    ):
        if ema_period < 2:
            raise ValueError("ema_period must be >= 2")
        if volume_window < 2:
            raise ValueError("volume_window must be >= 2")
        if adx_period < 2:
            raise ValueError("adx_period must be >= 2")
        if volume_multiplier <= 0:
            raise ValueError("volume_multiplier must be > 0")
        if adx_threshold < 0:
            raise ValueError("adx_threshold must be >= 0")

        super().__init__(
            ema_period=ema_period,
            volume_multiplier=volume_multiplier,
            adx_period=adx_period,
            adx_threshold=adx_threshold,
            volume_window=volume_window,
        )
        self.ema_period = ema_period
        self.volume_multiplier = volume_multiplier
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.volume_window = volume_window

    @property
    def name(self) -> str:
        return (
            f"Trend + Volume Confirmation "
            f"({self.ema_period} EMA, {self.volume_multiplier}x vol, ADX {self.adx_threshold})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_df(df)

        if df.empty:
            raise ValueError("Input DataFrame is empty")

        df["ema_trend"] = ta.trend.ema_indicator(df["Close"], window=self.ema_period)
        df["avg_volume"] = df["Volume"].rolling(window=self.volume_window, min_periods=self.volume_window).mean()
        df["adx"] = ta.trend.adx(
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            window=self.adx_period,
        )
        df["signal"] = HOLD

        volume_confirmed = df["Volume"] > (df["avg_volume"] * self.volume_multiplier)
        strong_trend = df["adx"] > self.adx_threshold

        bullish = (df["Close"] > df["ema_trend"]) & volume_confirmed & strong_trend
        bearish = (df["Close"] < df["ema_trend"]) & volume_confirmed & strong_trend

        df.loc[bullish, "signal"] = BUY
        df.loc[bearish, "signal"] = SELL
        df.dropna(subset=["ema_trend", "avg_volume", "adx"], inplace=True)
        return df


class MeanReversionRegimeFilterStrategy(Strategy):
    """
    Mean Reversion with Regime Filter (MRRF).

    Professional logic:
    Mean reversion performs best in compressed, range-bound markets and tends to
    fail badly in expanding or directional regimes. This strategy first checks
    that volatility is low using Bollinger Band width, then looks for RSI
    extremes, and finally requires price to remain close to a 20-period fair
    value proxy (SMA used here as a practical VWAP approximation on daily bars).
    If volatility expands beyond a defined threshold, the strategy stands down
    completely rather than forcing reversion trades into a trend.

    BUY:
        - RSI < rsi_oversold
        - Bollinger Band width < bb_threshold
        - Price deviation from SMA(sma_period) <= price_tolerance

    SELL:
        - RSI > rsi_overbought
        - Bollinger Band width < bb_threshold
        - Price deviation from SMA(sma_period) <= price_tolerance

    HOLD override:
        - If Bollinger Band width > bb_trend_threshold, do not mean revert
    """

    def __init__(
        self,
        rsi_period: int = 14,
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        bb_threshold: float = 0.05,
        bb_trend_threshold: float = 0.08,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        sma_period: int = 20,
        price_tolerance: float = 0.02,
    ):
        if rsi_period < 2:
            raise ValueError("rsi_period must be >= 2")
        if bb_period < 2:
            raise ValueError("bb_period must be >= 2")
        if sma_period < 2:
            raise ValueError("sma_period must be >= 2")
        if bb_std_dev <= 0:
            raise ValueError("bb_std_dev must be > 0")
        if bb_threshold <= 0:
            raise ValueError("bb_threshold must be > 0")
        if bb_trend_threshold <= bb_threshold:
            raise ValueError("bb_trend_threshold must be greater than bb_threshold")
        if not 0 < price_tolerance < 1:
            raise ValueError("price_tolerance must be between 0 and 1")
        if rsi_oversold >= rsi_overbought:
            raise ValueError("rsi_oversold must be less than rsi_overbought")

        super().__init__(
            rsi_period=rsi_period,
            bb_period=bb_period,
            bb_std_dev=bb_std_dev,
            bb_threshold=bb_threshold,
            bb_trend_threshold=bb_trend_threshold,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            sma_period=sma_period,
            price_tolerance=price_tolerance,
        )
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.bb_threshold = bb_threshold
        self.bb_trend_threshold = bb_trend_threshold
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.sma_period = sma_period
        self.price_tolerance = price_tolerance

    @property
    def name(self) -> str:
        return (
            f"Mean Reversion + Regime Filter "
            f"(RSI {self.rsi_period}, BB {self.bb_period}, tol {self.price_tolerance:.0%})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_df(df)

        if df.empty:
            raise ValueError("Input DataFrame is empty")

        bb = ta.volatility.BollingerBands(
            close=df["Close"],
            window=self.bb_period,
            window_dev=self.bb_std_dev,
        )

        df["rsi"] = ta.momentum.rsi(df["Close"], window=self.rsi_period)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, pd.NA)
        df["sma_fair_value"] = ta.trend.sma_indicator(df["Close"], window=self.sma_period)
        df["price_deviation"] = ((df["Close"] - df["sma_fair_value"]).abs() / df["sma_fair_value"].abs().replace(0, pd.NA))
        df["signal"] = HOLD

        sideways_regime = df["bb_width"] < self.bb_threshold
        trending_regime = df["bb_width"] > self.bb_trend_threshold
        near_fair_value = df["price_deviation"] <= self.price_tolerance

        bullish = (df["rsi"] < self.rsi_oversold) & sideways_regime & near_fair_value
        bearish = (df["rsi"] > self.rsi_overbought) & sideways_regime & near_fair_value

        df.loc[bullish, "signal"] = BUY
        df.loc[bearish, "signal"] = SELL
        df.loc[trending_regime, "signal"] = HOLD

        df.dropna(
            subset=["rsi", "bb_upper", "bb_lower", "bb_mid", "bb_width", "sma_fair_value", "price_deviation"],
            inplace=True,
        )
        return df


class MultiTimeframeMomentumStrategy(Strategy):
    """
    Multi-Timeframe Momentum (MTM).

    Professional logic:
    Institutional momentum models often require agreement across horizons so a
    short-term burst is not mistaken for durable trend persistence. This
    strategy combines fast rate-of-change, medium-horizon rate-of-change, and a
    slow trend anchor. It only trades when all three horizons agree, reducing
    whipsaw from mixed or transitional conditions.

    BUY:
        - ROC(short_roc) > 0
        - ROC(medium_roc) > 0
        - Close > EMA(ema_period)

    SELL:
        - ROC(short_roc) < 0
        - ROC(medium_roc) < 0
        - Close < EMA(ema_period)

    HOLD:
        - Any mixed reading across the three horizons
    """

    def __init__(self, short_roc: int = 5, medium_roc: int = 20, ema_period: int = 100):
        if short_roc < 1:
            raise ValueError("short_roc must be >= 1")
        if medium_roc <= short_roc:
            raise ValueError("medium_roc must be greater than short_roc")
        if ema_period < 2:
            raise ValueError("ema_period must be >= 2")

        super().__init__(short_roc=short_roc, medium_roc=medium_roc, ema_period=ema_period)
        self.short_roc = short_roc
        self.medium_roc = medium_roc
        self.ema_period = ema_period

    @property
    def name(self) -> str:
        return f"Multi-Timeframe Momentum ({self.short_roc}/{self.medium_roc}, EMA {self.ema_period})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_df(df)

        if df.empty:
            raise ValueError("Input DataFrame is empty")

        df["roc_short"] = ta.momentum.roc(df["Close"], window=self.short_roc)
        df["roc_medium"] = ta.momentum.roc(df["Close"], window=self.medium_roc)
        df["ema_long"] = ta.trend.ema_indicator(df["Close"], window=self.ema_period)
        df["signal"] = HOLD

        bullish = (df["roc_short"] > 0) & (df["roc_medium"] > 0) & (df["Close"] > df["ema_long"])
        bearish = (df["roc_short"] < 0) & (df["roc_medium"] < 0) & (df["Close"] < df["ema_long"])

        df.loc[bullish, "signal"] = BUY
        df.loc[bearish, "signal"] = SELL
        df.dropna(subset=["roc_short", "roc_medium", "ema_long"], inplace=True)
        return df


PRO_STRATEGY_REGISTRY: dict[str, dict[str, Any]] = {
    "TrendVolumeConfirmation": {
        "class": TrendVolumeConfirmationStrategy,
        "default_params": {
            "ema_period": 50,
            "volume_multiplier": 1.5,
            "adx_period": 14,
            "adx_threshold": 25.0,
            "volume_window": 20,
        },
        "param_defs": {
            "ema_period": {"type": "int", "min": 5, "max": 200, "default": 50},
            "volume_multiplier": {"type": "float", "min": 0.5, "max": 5.0, "default": 1.5},
            "adx_period": {"type": "int", "min": 5, "max": 50, "default": 14},
            "adx_threshold": {"type": "float", "min": 5.0, "max": 60.0, "default": 25.0},
            "volume_window": {"type": "int", "min": 5, "max": 60, "default": 20},
        },
    },
    "MeanReversionRegimeFilter": {
        "class": MeanReversionRegimeFilterStrategy,
        "default_params": {
            "rsi_period": 14,
            "bb_period": 20,
            "bb_std_dev": 2.0,
            "bb_threshold": 0.05,
            "bb_trend_threshold": 0.08,
            "rsi_oversold": 35.0,
            "rsi_overbought": 65.0,
            "sma_period": 20,
            "price_tolerance": 0.02,
        },
        "param_defs": {
            "rsi_period": {"type": "int", "min": 2, "max": 30, "default": 14},
            "bb_period": {"type": "int", "min": 5, "max": 60, "default": 20},
            "bb_std_dev": {"type": "float", "min": 1.0, "max": 4.0, "default": 2.0},
            "bb_threshold": {"type": "float", "min": 0.01, "max": 0.20, "default": 0.05},
            "bb_trend_threshold": {"type": "float", "min": 0.02, "max": 0.30, "default": 0.08},
            "rsi_oversold": {"type": "float", "min": 10.0, "max": 45.0, "default": 35.0},
            "rsi_overbought": {"type": "float", "min": 55.0, "max": 90.0, "default": 65.0},
            "sma_period": {"type": "int", "min": 5, "max": 60, "default": 20},
            "price_tolerance": {"type": "float", "min": 0.005, "max": 0.10, "default": 0.02},
        },
    },
    "MultiTimeframeMomentum": {
        "class": MultiTimeframeMomentumStrategy,
        "default_params": {
            "short_roc": 5,
            "medium_roc": 20,
            "ema_period": 100,
        },
        "param_defs": {
            "short_roc": {"type": "int", "min": 1, "max": 20, "default": 5},
            "medium_roc": {"type": "int", "min": 5, "max": 60, "default": 20},
            "ema_period": {"type": "int", "min": 20, "max": 250, "default": 100},
        },
    },
}


def build_pro_strategy(name: str, **params) -> Strategy:
    """
    Instantiate a professional strategy from PRO_STRATEGY_REGISTRY.

    Args:
        name: Registry key for the strategy.
        **params: Parameter overrides.

    Returns:
        Configured Strategy instance.
    """
    if name not in PRO_STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(PRO_STRATEGY_REGISTRY.keys())}"
        )

    entry = PRO_STRATEGY_REGISTRY[name]
    merged_params = {**entry["default_params"], **params}
    return entry["class"](**merged_params)
