"""ARIMA-based price direction forecast provider for Trading Alpha.

This module forecasts the next trading day's closing price using ARIMA when
available, with a NumPy linear regression fallback.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from statsmodels.tsa.arima.model import ARIMA

    HAS_STATSMODELS = True
except Exception:
    ARIMA = None
    HAS_STATSMODELS = False


class ARIMAForecastProvider:
    """Forecast short-term price direction using ARIMA or linear regression.

    Results are cached for 4 hours per symbol/lookback pair.
    """

    CACHE_TTL_SECONDS = 4 * 60 * 60
    _cache: Dict[str, Dict[str, Any]] = {}

    def analyze(self, symbol: str, lookback_days: int = 60) -> dict:
        """Forecast next-day direction for a stock.

        Args:
            symbol: Stock ticker symbol.
            lookback_days: Number of recent trading days to analyze.

        Returns:
            A dictionary with keys: signal, score, details, summary, error.
        """
        normalized_symbol = (symbol or "").strip().upper()
        if not normalized_symbol:
            return self._error_result(symbol, "Symbol is required")
        if lookback_days < 20:
            lookback_days = 20

        cache_key = f"{normalized_symbol}:{lookback_days}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            history = yf.Ticker(normalized_symbol).history(period=f"{max(lookback_days + 10, 30)}d")
            history = self._normalize_history_frame(history)
            if history.empty or "Close" not in history:
                raise ValueError("No price history available")

            closes = history["Close"].dropna().tail(lookback_days)
            if isinstance(closes, pd.DataFrame):
                if closes.empty:
                    raise ValueError("No closing price history available")
                closes = closes.iloc[:, 0]
            elif isinstance(closes, pd.Series) and closes.empty:
                raise ValueError("No closing price history available")
            if len(closes) < 20:
                raise ValueError("Insufficient price history for forecast")

            prices = closes.to_numpy(dtype=float)
            current_price = float(prices[-1])
            method = "ARIMA(5,1,0)" if HAS_STATSMODELS else "linear_regression_fallback"

            if HAS_STATSMODELS:
                try:
                    model = ARIMA(prices, order=(5, 1, 0))
                    fitted = model.fit()
                    forecast_price = float(fitted.forecast(steps=1)[0])
                except Exception as exc:
                    result = {
                        "signal": 0,
                        "score": 0.0,
                        "details": {
                            "symbol": normalized_symbol,
                            "method": "ARIMA(5,1,0)",
                            "lookback_days": lookback_days,
                            "current_price": current_price,
                        },
                        "summary": f"{normalized_symbol} ARIMA forecast was inconclusive due to model convergence issues.",
                        "error": str(exc),
                    }
                    self._set_cache(cache_key, result)
                    return result
            else:
                forecast_price = self._linear_regression_forecast(prices[-10:])

            recent_returns = np.diff(prices[-21:]) / prices[-21:-1]
            volatility = float(np.std(recent_returns)) if len(recent_returns) > 0 else 0.0
            expected_return = (forecast_price - current_price) / current_price if current_price else 0.0

            if expected_return > volatility:
                signal = 1
                direction = "up"
            elif expected_return < -volatility:
                signal = -1
                direction = "down"
            else:
                signal = 0
                direction = "sideways"

            score = 0.0
            if volatility > 0:
                score = min(abs(expected_return) / volatility, 1.0)
            score = round(float(score), 4)

            result = {
                "signal": signal,
                "score": score,
                "details": {
                    "symbol": normalized_symbol,
                    "method": method,
                    "lookback_days": lookback_days,
                    "current_price": current_price,
                    "forecast_price": forecast_price,
                    "expected_return": expected_return,
                    "recent_volatility_1std": volatility,
                },
                "summary": (
                    f"{normalized_symbol} forecast points {direction} tomorrow. "
                    f"Expected return is {expected_return:.2%} versus 1σ daily volatility of {volatility:.2%}."
                ),
                "error": None,
            }
            self._set_cache(cache_key, result)
            return result
        except Exception as exc:
            result = self._error_result(normalized_symbol, str(exc))
            self._set_cache(cache_key, result)
            return result

    @staticmethod
    def _normalize_history_frame(history: Any) -> pd.DataFrame:
        if history is None:
            return pd.DataFrame()
        if isinstance(history, pd.Series):
            if history.empty:
                return pd.DataFrame()
            history = history.to_frame(name=history.name or "value")
        if not isinstance(history, pd.DataFrame):
            try:
                history = pd.DataFrame(history)
            except Exception:
                return pd.DataFrame()
        if history.empty:
            return history

        normalized = history.copy()
        if isinstance(normalized.index, pd.DatetimeIndex):
            if normalized.index.tz is not None:
                normalized.index = normalized.index.tz_convert(None)
            else:
                normalized.index = normalized.index.tz_localize(None)
        return normalized

    @staticmethod
    def _linear_regression_forecast(prices: np.ndarray) -> float:
        """Forecast the next price using a linear trend over recent prices."""
        x = np.arange(len(prices), dtype=float)
        slope, intercept = np.polyfit(x, prices.astype(float), 1)
        next_x = float(len(prices))
        return float(slope * next_x + intercept)

    def _get_cached(self, key: str) -> Optional[dict]:
        """Return cached result if still fresh."""
        entry = self._cache.get(key)
        if not entry:
            return None
        if time.time() - entry["timestamp"] > self.CACHE_TTL_SECONDS:
            self._cache.pop(key, None)
            return None
        return entry["result"]

    def _set_cache(self, key: str, result: dict) -> None:
        """Store a result in cache."""
        self._cache[key] = {"timestamp": time.time(), "result": result}

    @staticmethod
    def _error_result(symbol: str, error: str) -> dict:
        """Build a standardized error result."""
        return {
            "signal": 0,
            "score": 0.0,
            "details": {"symbol": (symbol or "").strip().upper()},
            "summary": f"Unable to complete forecast analysis for {(symbol or '').strip().upper() or 'unknown symbol'}.",
            "error": error,
        }
