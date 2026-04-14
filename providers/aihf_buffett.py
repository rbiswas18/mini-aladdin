"""Buffett-style quality screen provider for Trading Alpha.

This module evaluates a stock against a simple Warren Buffett-inspired quality
checklist using free Yahoo Finance fundamentals.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import yfinance as yf


class BuffettProvider:
    """Run a Buffett-style quality screen on a stock.

    The provider checks seven business quality criteria and returns a
    standardized signal payload. Results are cached for 24 hours per symbol.
    """

    CACHE_TTL_SECONDS = 24 * 60 * 60
    _cache: Dict[str, Dict[str, Any]] = {}

    def analyze(self, symbol: str) -> dict:
        """Analyze a stock using Buffett-style quality criteria.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            A dictionary with keys: signal, score, details, summary, error.
        """
        normalized_symbol = (symbol or "").strip().upper()
        if not normalized_symbol:
            return self._error_result(symbol, "Symbol is required")

        cached = self._get_cached(normalized_symbol)
        if cached is not None:
            return cached

        try:
            info = yf.Ticker(normalized_symbol).info or {}
            criteria = {
                "roe": {
                    "label": "ROE > 15%",
                    "value": self._safe_float(info.get("returnOnEquity")),
                    "passed": False,
                },
                "debt_to_equity": {
                    "label": "Debt/Equity < 0.8",
                    "value": self._safe_float(info.get("debtToEquity")),
                    "passed": False,
                },
                "profit_margin": {
                    "label": "Profit margin > 15%",
                    "value": self._safe_float(info.get("profitMargins")),
                    "passed": False,
                },
                "current_ratio": {
                    "label": "Current ratio > 1.5",
                    "value": self._safe_float(info.get("currentRatio")),
                    "passed": False,
                },
                "free_cashflow": {
                    "label": "Positive free cash flow",
                    "value": self._safe_float(info.get("freeCashflow")),
                    "passed": False,
                },
                "revenue_growth": {
                    "label": "Revenue growth positive",
                    "value": self._safe_float(info.get("revenueGrowth")),
                    "passed": False,
                },
                "trailing_eps": {
                    "label": "EPS consistently positive",
                    "value": self._safe_float(info.get("trailingEps")),
                    "passed": False,
                },
            }

            criteria["roe"]["passed"] = bool(
                criteria["roe"]["value"] is not None and criteria["roe"]["value"] > 0.15
            )
            criteria["debt_to_equity"]["passed"] = bool(
                criteria["debt_to_equity"]["value"] is not None
                and criteria["debt_to_equity"]["value"] < 80
            )
            criteria["profit_margin"]["passed"] = bool(
                criteria["profit_margin"]["value"] is not None
                and criteria["profit_margin"]["value"] > 0.15
            )
            criteria["current_ratio"]["passed"] = bool(
                criteria["current_ratio"]["value"] is not None
                and criteria["current_ratio"]["value"] > 1.5
            )
            criteria["free_cashflow"]["passed"] = bool(
                criteria["free_cashflow"]["value"] is not None
                and criteria["free_cashflow"]["value"] > 0
            )
            criteria["revenue_growth"]["passed"] = bool(
                criteria["revenue_growth"]["value"] is not None
                and criteria["revenue_growth"]["value"] > 0
            )
            criteria["trailing_eps"]["passed"] = bool(
                criteria["trailing_eps"]["value"] is not None
                and criteria["trailing_eps"]["value"] > 0
            )

            passed_count = sum(1 for item in criteria.values() if item["passed"])
            total_count = len(criteria)
            score = round(passed_count / total_count, 4)

            if passed_count >= 6:
                signal = 1
                signal_text = "QUALITY BUY"
                disposition = "Buffett would buy"
            elif passed_count >= 3:
                signal = 0
                signal_text = "BORDERLINE"
                disposition = "Borderline"
            else:
                signal = -1
                signal_text = "AVOID"
                disposition = "Buffett would avoid"

            strengths = [item["label"] for item in criteria.values() if item["passed"]]
            weaknesses = [item["label"] for item in criteria.values() if not item["passed"]]

            summary_parts = [
                f"{normalized_symbol} passes {passed_count}/{total_count} Buffett criteria.",
            ]
            if strengths:
                summary_parts.append(f"Strengths: {', '.join(strengths[:3])}.")
            if weaknesses and signal != 1:
                summary_parts.append(f"Weak spots: {', '.join(weaknesses[:2])}.")
            summary_parts.append(f"Signal: {signal_text}.")

            result = {
                "signal": signal,
                "score": score,
                "details": {
                    "symbol": normalized_symbol,
                    "passed_count": passed_count,
                    "total_count": total_count,
                    "disposition": disposition,
                    "criteria": criteria,
                },
                "summary": " ".join(summary_parts),
                "error": None,
            }
            self._set_cache(normalized_symbol, result)
            return result
        except Exception as exc:
            result = self._error_result(normalized_symbol, str(exc))
            self._set_cache(normalized_symbol, result)
            return result

    def _get_cached(self, symbol: str) -> Optional[dict]:
        """Return cached result if it has not expired."""
        entry = self._cache.get(symbol)
        if not entry:
            return None
        if time.time() - entry["timestamp"] > self.CACHE_TTL_SECONDS:
            self._cache.pop(symbol, None)
            return None
        return entry["result"]

    def _set_cache(self, symbol: str, result: dict) -> None:
        """Store result in cache."""
        self._cache[symbol] = {"timestamp": time.time(), "result": result}

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Safely convert a value to float."""
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _error_result(symbol: str, error: str) -> dict:
        """Build a standardized error result."""
        return {
            "signal": 0,
            "score": 0.0,
            "details": {"symbol": (symbol or "").strip().upper()},
            "summary": f"Unable to complete Buffett analysis for {(symbol or '').strip().upper() or 'unknown symbol'}.",
            "error": error,
        }
