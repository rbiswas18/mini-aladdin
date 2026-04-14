"""Valuation signal provider for Trading Alpha.

This module implements a simple intrinsic value model using free data from
`yfinance`. It combines four valuation methods and returns a standardized
analysis payload.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import yfinance as yf


class ValuationProvider:
    """Analyze whether a stock appears undervalued or overvalued.

    The provider uses four valuation methods:
    1. Trailing P/E versus a sector average
    2. PEG ratio
    3. Price-to-book ratio
    4. Simple discounted cash flow estimate

    Results are cached for 24 hours per symbol.
    """

    CACHE_TTL_SECONDS = 24 * 60 * 60
    _cache: Dict[str, Dict[str, Any]] = {}

    SECTOR_PE_AVERAGES = {
        "tech": 28.0,
        "technology": 28.0,
        "finance": 15.0,
        "financial services": 15.0,
        "healthcare": 22.0,
        "consumer": 20.0,
        "consumer cyclical": 20.0,
        "consumer defensive": 20.0,
        "energy": 12.0,
        "default": 20.0,
    }

    def analyze(self, symbol: str) -> dict:
        """Analyze a symbol and return a standardized valuation signal.

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
            ticker = yf.Ticker(normalized_symbol)
            info = ticker.info or {}

            current_price = self._safe_float(
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
            )
            if current_price is None or current_price <= 0:
                hist = ticker.history(period="5d")
                if not hist.empty:
                    current_price = self._safe_float(hist["Close"].dropna().iloc[-1])

            if current_price is None or current_price <= 0:
                raise ValueError("Unable to determine current price")

            details: Dict[str, Any] = {
                "symbol": normalized_symbol,
                "current_price": current_price,
                "sector": info.get("sector"),
                "methods": {},
            }

            votes = []

            pe_signal, pe_details = self._analyze_pe(info)
            details["methods"]["pe_vs_sector"] = pe_details
            if pe_signal is not None:
                votes.append(pe_signal)

            peg_signal, peg_details = self._analyze_peg(info)
            details["methods"]["peg_ratio"] = peg_details
            if peg_signal is not None:
                votes.append(peg_signal)

            pb_signal, pb_details = self._analyze_price_to_book(info)
            details["methods"]["price_to_book"] = pb_details
            if pb_signal is not None:
                votes.append(pb_signal)

            dcf_signal, dcf_details = self._analyze_dcf(info, current_price)
            details["methods"]["dcf"] = dcf_details
            if dcf_signal is not None:
                votes.append(dcf_signal)

            if not votes:
                result = {
                    "signal": 0,
                    "score": 0.0,
                    "details": details,
                    "summary": f"{normalized_symbol} could not be valued with the available Yahoo Finance data.",
                    "error": None,
                }
                self._set_cache(normalized_symbol, result)
                return result

            positives = sum(1 for vote in votes if vote == 1)
            negatives = sum(1 for vote in votes if vote == -1)
            neutrals = sum(1 for vote in votes if vote == 0)

            if positives > negatives:
                signal = 1
                agreement = positives
                stance = "undervalued"
            elif negatives > positives:
                signal = -1
                agreement = negatives
                stance = "overvalued"
            else:
                signal = 0
                agreement = max(neutrals, positives, negatives)
                stance = "fairly valued"

            score = round(agreement / len(votes), 4)
            details["vote_counts"] = {
                "undervalued": positives,
                "overvalued": negatives,
                "fair": neutrals,
                "total": len(votes),
            }

            result = {
                "signal": signal,
                "score": score,
                "details": details,
                "summary": (
                    f"{normalized_symbol} screens as {stance}. "
                    f"{agreement}/{len(votes)} valuation methods agree."
                ),
                "error": None,
            }
            self._set_cache(normalized_symbol, result)
            return result
        except Exception as exc:
            result = self._error_result(normalized_symbol, str(exc))
            self._set_cache(normalized_symbol, result)
            return result

    def _analyze_pe(self, info: Dict[str, Any]) -> tuple[Optional[int], Dict[str, Any]]:
        """Evaluate trailing P/E against a hardcoded sector average."""
        trailing_pe = self._safe_float(info.get("trailingPE"))
        sector = str(info.get("sector") or "").strip().lower()
        sector_avg = self.SECTOR_PE_AVERAGES.get(sector)
        if sector_avg is None:
            sector_avg = self.SECTOR_PE_AVERAGES["default"]

        details = {
            "trailing_pe": trailing_pe,
            "sector": info.get("sector"),
            "sector_average_pe": sector_avg,
            "signal": 0,
            "reason": "Insufficient data",
        }

        if trailing_pe is None or trailing_pe <= 0:
            return None, details

        if trailing_pe < sector_avg * 0.8:
            details["signal"] = 1
            details["reason"] = "P/E is more than 20% below sector average"
            return 1, details
        if trailing_pe > sector_avg * 1.2:
            details["signal"] = -1
            details["reason"] = "P/E is more than 20% above sector average"
            return -1, details

        details["reason"] = "P/E is near sector average"
        return 0, details

    def _analyze_peg(self, info: Dict[str, Any]) -> tuple[Optional[int], Dict[str, Any]]:
        """Evaluate valuation using PEG ratio."""
        peg_ratio = self._safe_float(info.get("pegRatio"))
        details = {
            "peg_ratio": peg_ratio,
            "signal": 0,
            "reason": "Insufficient data",
        }

        if peg_ratio is None or peg_ratio <= 0:
            return None, details
        if peg_ratio < 1.0:
            details["signal"] = 1
            details["reason"] = "PEG ratio is below 1.0"
            return 1, details
        if peg_ratio > 2.0:
            details["signal"] = -1
            details["reason"] = "PEG ratio is above 2.0"
            return -1, details

        details["reason"] = "PEG ratio is in a neutral range"
        return 0, details

    def _analyze_price_to_book(self, info: Dict[str, Any]) -> tuple[Optional[int], Dict[str, Any]]:
        """Evaluate valuation using price-to-book ratio."""
        price_to_book = self._safe_float(info.get("priceToBook"))
        details = {
            "price_to_book": price_to_book,
            "signal": 0,
            "reason": "Insufficient data",
        }

        if price_to_book is None or price_to_book <= 0:
            return None, details
        if price_to_book < 1.5:
            details["signal"] = 1
            details["reason"] = "Price/book is below 1.5"
            return 1, details
        if price_to_book > 5.0:
            details["signal"] = -1
            details["reason"] = "Price/book is above 5.0"
            return -1, details

        details["reason"] = "Price/book is in a neutral range"
        return 0, details

    def _analyze_dcf(self, info: Dict[str, Any], current_price: float) -> tuple[Optional[int], Dict[str, Any]]:
        """Estimate intrinsic value using a simple DCF model."""
        free_cashflow = self._safe_float(info.get("freeCashflow"))
        shares_outstanding = self._safe_float(info.get("sharesOutstanding"))
        growth_rate = 0.10
        terminal_growth = 0.03
        discount_rate = 0.10

        details = {
            "free_cashflow": free_cashflow,
            "shares_outstanding": shares_outstanding,
            "growth_rate": growth_rate,
            "terminal_growth": terminal_growth,
            "discount_rate": discount_rate,
            "intrinsic_value": None,
            "signal": 0,
            "reason": "Insufficient data",
        }

        if (
            free_cashflow is None
            or shares_outstanding is None
            or free_cashflow <= 0
            or shares_outstanding <= 0
        ):
            return None, details

        fcf_per_share = free_cashflow / shares_outstanding
        projected_value = 0.0
        next_fcf = fcf_per_share

        for year in range(1, 6):
            next_fcf *= 1 + growth_rate
            projected_value += next_fcf / ((1 + discount_rate) ** year)

        if discount_rate <= terminal_growth:
            return None, details

        terminal_fcf = next_fcf * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        discounted_terminal = terminal_value / ((1 + discount_rate) ** 5)
        intrinsic_value = projected_value + discounted_terminal

        details["intrinsic_value"] = intrinsic_value
        details["fcf_per_share"] = fcf_per_share

        if current_price < intrinsic_value * 0.9:
            details["signal"] = 1
            details["reason"] = "Market price is more than 10% below DCF value"
            return 1, details
        if current_price > intrinsic_value * 1.1:
            details["signal"] = -1
            details["reason"] = "Market price is more than 10% above DCF value"
            return -1, details

        details["reason"] = "Market price is close to DCF value"
        return 0, details

    def _get_cached(self, symbol: str) -> Optional[dict]:
        """Return cached result if still valid."""
        entry = self._cache.get(symbol)
        if not entry:
            return None
        if time.time() - entry["timestamp"] > self.CACHE_TTL_SECONDS:
            self._cache.pop(symbol, None)
            return None
        return entry["result"]

    def _set_cache(self, symbol: str, result: dict) -> None:
        """Store a result in the in-memory cache."""
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
            "summary": f"Unable to complete valuation analysis for {(symbol or '').strip().upper() or 'unknown symbol'}.",
            "error": error,
        }
