"""Fundamental analysis provider for Trading Alpha.

This module evaluates a stock's fundamental financial health using free data
from ``yfinance`` only. It converts a small set of accounting and valuation
metrics into a simple directional signal:

- ``+1``: bullish fundamentals
- ``0``: neutral fundamentals
- ``-1``: bearish fundamentals
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import yfinance as yf


class FundamentalsProvider:
    """Analyze stock fundamentals with a simple rules-based scoring model.

    The provider fetches company metrics from ``yfinance.Ticker(symbol).info``,
    scores each available criterion as bullish, neutral, or bearish, and then
    returns a normalized signal dictionary.

    Results are cached in memory for 24 hours to reduce repeated network calls.
    """

    CACHE_TTL_SECONDS = 24 * 60 * 60

    def __init__(self) -> None:
        """Initialize the provider with an in-memory 24-hour cache."""
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def analyze(self, symbol: str) -> Dict[str, Any]:
        """Analyze a stock's fundamental strength.

        Args:
            symbol: Stock ticker symbol, for example ``"AAPL"``.

        Returns:
            A dictionary with the structure:

            .. code-block:: python

                {
                    "signal": int,
                    "score": float,
                    "details": dict,
                    "summary": str,
                    "error": str | None,
                }

            Notes:
                - ``signal`` is ``+1``, ``0``, or ``-1``.
                - ``score`` is confidence on a ``0.0`` to ``1.0`` scale.
                - The signed normalized score is included in
                  ``details["normalized_score"]``.
                - Missing metrics are skipped gracefully.
                - Any unexpected exception returns a neutral result with an
                  error message.
        """
        clean_symbol = (symbol or "").strip().upper()
        if not clean_symbol:
            return self._error_result(symbol, "Symbol is required.")

        cached = self._get_cached(clean_symbol)
        if cached is not None:
            return cached

        try:
            ticker = yf.Ticker(clean_symbol)
            info = ticker.info or {}

            metrics = {
                "trailing_pe": self._to_float(info.get("trailingPE")),
                "forward_pe": self._to_float(info.get("forwardPE")),
                "price_to_book": self._to_float(info.get("priceToBook")),
                "debt_to_equity": self._to_float(info.get("debtToEquity")),
                "return_on_equity": self._to_float(info.get("returnOnEquity")),
                "revenue_growth": self._to_float(info.get("revenueGrowth")),
                "earnings_growth": self._to_float(info.get("earningsGrowth")),
                "free_cash_flow": self._to_float(info.get("freeCashflow")),
                "current_ratio": self._to_float(info.get("currentRatio")),
            }

            criteria: Dict[str, Optional[int]] = {
                "pe_ratio": self._score_pe(metrics["trailing_pe"], metrics["forward_pe"]),
                "roe": self._score_threshold(metrics["return_on_equity"], bullish_gt=0.15, bearish_lt=0.05),
                "revenue_growth": self._score_threshold(metrics["revenue_growth"], bullish_gt=0.10, bearish_lt=0.0),
                "earnings_growth": self._score_threshold(metrics["earnings_growth"], bullish_gt=0.15, bearish_lt=-0.05),
                "debt_to_equity": self._score_threshold(metrics["debt_to_equity"], bullish_lt=0.5, bearish_gt=2.0),
                "free_cash_flow": self._score_cash_flow(metrics["free_cash_flow"]),
                "current_ratio": self._score_threshold(metrics["current_ratio"], bullish_gt=1.5, bearish_lt=1.0),
            }

            available_scores = [value for value in criteria.values() if value is not None]
            if available_scores:
                normalized_score = sum(available_scores) / len(available_scores)
            else:
                normalized_score = 0.0

            if normalized_score > 0.2:
                signal = 1
                label = "BULLISH"
            elif normalized_score < -0.2:
                signal = -1
                label = "BEARISH"
            else:
                signal = 0
                label = "NEUTRAL"

            confidence = abs(normalized_score)
            summary = self._build_summary(clean_symbol, metrics, criteria, label)

            result = {
                "signal": signal,
                "score": round(confidence, 4),
                "details": {
                    "metrics": metrics,
                    "criteria_scores": criteria,
                    "available_criteria": len(available_scores),
                    "normalized_score": round(normalized_score, 4),
                    "signal_label": label,
                },
                "summary": summary,
                "error": None,
            }
            self._cache[clean_symbol] = (time.time(), result)
            return result
        except Exception as exc:  # pragma: no cover - defensive error handling
            result = self._error_result(clean_symbol, str(exc))
            self._cache[clean_symbol] = (time.time(), result)
            return result

    def get_signal(self, symbol: str) -> int:
        """Return only the integer signal for a ticker symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            ``+1`` for bullish, ``0`` for neutral, or ``-1`` for bearish.
        """
        return int(self.analyze(symbol).get("signal", 0))

    def _get_cached(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return a cached analysis result if it is still fresh."""
        cached = self._cache.get(symbol)
        if not cached:
            return None

        cached_at, result = cached
        if time.time() - cached_at < self.CACHE_TTL_SECONDS:
            return result

        self._cache.pop(symbol, None)
        return None

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        """Convert a value to ``float`` safely, returning ``None`` if invalid."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _score_threshold(
        value: Optional[float],
        *,
        bullish_gt: Optional[float] = None,
        bullish_lt: Optional[float] = None,
        bearish_gt: Optional[float] = None,
        bearish_lt: Optional[float] = None,
    ) -> Optional[int]:
        """Score a metric using upper and lower bullish/bearish thresholds."""
        if value is None:
            return None

        if bullish_gt is not None and value > bullish_gt:
            return 1
        if bullish_lt is not None and value < bullish_lt:
            return 1
        if bearish_gt is not None and value > bearish_gt:
            return -1
        if bearish_lt is not None and value < bearish_lt:
            return -1
        return 0

    @staticmethod
    def _score_pe(trailing_pe: Optional[float], forward_pe: Optional[float]) -> Optional[int]:
        """Score valuation using trailing P/E, falling back to forward P/E."""
        pe_value = trailing_pe if trailing_pe is not None else forward_pe
        if pe_value is None:
            return None
        if pe_value < 25:
            return 1
        if pe_value > 40:
            return -1
        return 0

    @staticmethod
    def _score_cash_flow(free_cash_flow: Optional[float]) -> Optional[int]:
        """Score free cash flow as positive, negative, or unavailable."""
        if free_cash_flow is None:
            return None
        if free_cash_flow > 0:
            return 1
        if free_cash_flow < 0:
            return -1
        return 0

    @staticmethod
    def _build_summary(
        symbol: str,
        metrics: Dict[str, Optional[float]],
        criteria: Dict[str, Optional[int]],
        label: str,
    ) -> str:
        """Build a one-line plain-English summary for the analysis result."""
        positives = []
        negatives = []

        if criteria.get("roe") == 1:
            positives.append("high ROE")
        elif criteria.get("roe") == -1:
            negatives.append("weak ROE")

        if criteria.get("revenue_growth") == 1:
            positives.append("healthy revenue growth")
        elif criteria.get("revenue_growth") == -1:
            negatives.append("declining revenue")

        if criteria.get("earnings_growth") == 1:
            positives.append("strong earnings growth")
        elif criteria.get("earnings_growth") == -1:
            negatives.append("weak earnings growth")

        if criteria.get("debt_to_equity") == 1:
            positives.append("manageable debt")
        elif criteria.get("debt_to_equity") == -1:
            negatives.append("heavy debt")

        if criteria.get("free_cash_flow") == 1:
            positives.append("positive FCF")
        elif criteria.get("free_cash_flow") == -1:
            negatives.append("negative FCF")

        if criteria.get("current_ratio") == 1:
            positives.append("solid liquidity")
        elif criteria.get("current_ratio") == -1:
            negatives.append("weak liquidity")

        if criteria.get("pe_ratio") == 1:
            positives.append("reasonable valuation")
        elif criteria.get("pe_ratio") == -1:
            negatives.append("stretched valuation")

        descriptors = []
        if positives:
            descriptors.append(", ".join(positives[:3]))
        if negatives:
            prefix = "but " if positives else ""
            descriptors.append(prefix + ", ".join(negatives[:3]))

        tone_map = {
            "BULLISH": "Strong fundamentals",
            "BEARISH": "Weak fundamentals",
            "NEUTRAL": "Mixed fundamentals",
        }
        tone = tone_map.get(label, "Mixed fundamentals")

        if descriptors:
            return f"{symbol}: {tone} - {'; '.join(descriptors)}. Signal: {label}"
        if all(value is None for value in metrics.values()):
            return f"{symbol}: Insufficient fundamental data from yfinance. Signal: {label}"
        return f"{symbol}: {tone}. Signal: {label}"

    @staticmethod
    def _error_result(symbol: str, error_message: str) -> Dict[str, Any]:
        """Return a neutral result payload for invalid input or runtime errors."""
        safe_symbol = (symbol or "").strip().upper() or "UNKNOWN"
        return {
            "signal": 0,
            "score": 0.0,
            "details": {
                "metrics": {},
                "criteria_scores": {},
                "available_criteria": 0,
                "normalized_score": 0.0,
                "signal_label": "NEUTRAL",
            },
            "summary": f"{safe_symbol}: Fundamental analysis unavailable. Signal: NEUTRAL",
            "error": error_message,
        }
