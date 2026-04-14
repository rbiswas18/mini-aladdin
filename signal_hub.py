"""signal_hub.py - Weighted multi-source signal aggregation for Trading Alpha.

The Signal Hub combines technical strategy outputs and optional provider signals
into a normalized signal score (NSS), then applies market regime filtering to
produce trade-ready recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json

import pandas as pd
import yfinance as yf

from regime_filter import RegimeFilter
from strategy_simple import BUY, HOLD, SELL, STRATEGY_REGISTRY
from strategy_pro import PRO_STRATEGY_REGISTRY

logger = logging.getLogger(__name__)

SIGNAL_WEIGHTS: Dict[str, float] = {
    # Technical strategies
    "MultiTimeframeMomentum": 1.2,
    "Momentum": 1.2,
    "MeanReversionRegimeFilter": 1.0,
    "MovingAverageCrossover": 0.8,
    "RSIMeanReversion": 0.8,
    "MACDStrategy": 0.8,
    "BollingerBands": 0.8,
    "CombinedSignal": 0.8,
    "TrendVolumeConfirmation": 1.0,
    # Providers
    "fundamentals": 1.5,
    "valuation": 1.3,
    "sentiment": 1.0,
    "buffett": 1.2,
    "arima_forecast": 0.9,
}


@dataclass
class ProviderConfig:
    name: str
    attr_name: str
    module_name: str
    class_name: str


class SignalHub:
    """Central signal aggregator for Trading Alpha."""

    CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "signal_hub"
    CACHE_VERSION = 1
    DEFAULT_LOOKBACK_DAYS = 365
    PROVIDER_CONFIGS = [
        ProviderConfig("fundamentals", "fundamental", "providers.aihf_fundamental", "FundamentalsProvider"),
        ProviderConfig("valuation", "valuation", "providers.aihf_valuation", "ValuationProvider"),
        ProviderConfig("sentiment", "sentiment", "providers.aihf_sentiment", "SentimentProvider"),
        ProviderConfig("buffett", "buffett", "providers.aihf_buffett", "BuffettProvider"),
        ProviderConfig("arima_forecast", "arima_forecast", "providers.arima_forecast", "ARIMAForecastProvider"),
    ]

    def __init__(self, symbols: list, use_providers: bool = True, use_regime: bool = True):
        self.symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
        self.use_providers = use_providers
        self.use_regime = use_regime
        self.regime_filter = RegimeFilter() if use_regime else None
        self.last_scan_results: Dict[str, Dict[str, Any]] = {}
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        self._provider_errors: Dict[str, str] = {}
        for config in self.PROVIDER_CONFIGS:
            setattr(self, config.attr_name, self._load_provider(config))

    def _load_provider(self, config: ProviderConfig) -> Any:
        """Lazily import and instantiate a provider if available."""
        if not self.use_providers:
            return None

        try:
            module = __import__(config.module_name, fromlist=[config.class_name])
            provider_class = getattr(module, config.class_name)
            return provider_class()
        except ImportError as exc:
            self._provider_errors[config.name] = f"unavailable: {exc}"
            logger.info("Provider %s unavailable: %s", config.name, exc)
            return None
        except Exception as exc:  # pragma: no cover - defensive
            self._provider_errors[config.name] = f"init failed: {exc}"
            logger.warning("Provider %s failed to initialize: %s", config.name, exc)
            return None

    def _cache_path(self, symbol: str, cache_date: date) -> Path:
        return self.CACHE_DIR / f"{symbol}_{cache_date.isoformat()}.json"

    def _serialize_cache(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return json.loads(json.dumps(payload, default=self._json_default))

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if pd.isna(value):
            return None
        return str(value)

    def _read_cached_result(self, symbol: str, cache_date: date) -> Optional[Dict[str, Any]]:
        path = self._cache_path(symbol, cache_date)
        if not path.exists():
            return None

        try:
            payload = json.loads(path.read_text())
            if payload.get("cache_version") != self.CACHE_VERSION:
                return None
            return payload.get("result")
        except Exception as exc:
            logger.warning("Failed reading cache for %s: %s", symbol, exc)
            return None

    def _write_cached_result(self, symbol: str, cache_date: date, result: Dict[str, Any]) -> None:
        path = self._cache_path(symbol, cache_date)
        payload = {
            "cache_version": self.CACHE_VERSION,
            "cached_at": datetime.utcnow().isoformat(),
            "result": self._serialize_cache(result),
        }
        try:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        except Exception as exc:
            logger.warning("Failed writing cache for %s: %s", symbol, exc)

    @staticmethod
    def _normalize_signal(value: Any) -> int:
        """Normalize provider/strategy output to -1, 0, or +1."""
        if value is None:
            return 0
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized in {"BUY", "BULL", "BULLISH", "LONG", "POSITIVE", "+1"}:
                return 1
            if normalized in {"SELL", "BEAR", "BEARISH", "SHORT", "NEGATIVE", "-1"}:
                return -1
            if normalized in {"HOLD", "NEUTRAL", "0"}:
                return 0
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0
        if numeric > 0:
            return 1
        if numeric < 0:
            return -1
        return 0

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
            return result if pd.notna(result) else default
        except (TypeError, ValueError):
            return default

    def _strategy_items(self) -> List[tuple[str, dict]]:
        items: List[tuple[str, dict]] = []
        items.extend(STRATEGY_REGISTRY.items())
        items.extend(PRO_STRATEGY_REGISTRY.items())
        return items

    def collect_signals(self, symbol: str, df: pd.DataFrame) -> list[dict]:
        """Collect signals from all technical strategies and optional providers."""
        signals: List[Dict[str, Any]] = []

        if df is None or df.empty:
            logger.warning("No price data available for %s", symbol)
            return signals

        for name, entry in self._strategy_items():
            weight = SIGNAL_WEIGHTS.get(name, 1.0)
            try:
                strategy = entry["class"](**entry.get("default_params", {}))
                strategy_df = strategy.generate_signals(df.copy())
                signal_value = HOLD
                if strategy_df is not None and not strategy_df.empty and "signal" in strategy_df.columns:
                    signal_value = self._normalize_signal(strategy_df["signal"].iloc[-1])

                summary = self._technical_summary(name, signal_value)
                signals.append(
                    {
                        "name": name,
                        "source": "technical",
                        "signal": signal_value,
                        "weight": weight,
                        "summary": summary,
                    }
                )
            except Exception as exc:
                logger.warning("Technical strategy %s failed for %s: %s", name, symbol, exc)
                signals.append(
                    {
                        "name": name,
                        "source": "technical",
                        "signal": HOLD,
                        "weight": weight,
                        "summary": f"{name}: unavailable ({exc})",
                        "error": str(exc),
                    }
                )

        if self.use_providers:
            signals.extend(self._collect_provider_signals(symbol, df))

        return signals

    def _collect_provider_signals(self, symbol: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
        provider_signals: List[Dict[str, Any]] = []

        for config in self.PROVIDER_CONFIGS:
            provider = getattr(self, config.attr_name, None)
            weight = SIGNAL_WEIGHTS.get(config.name, 1.0)

            if provider is None:
                reason = self._provider_errors.get(config.name, "provider unavailable")
                provider_signals.append(
                    {
                        "name": config.name,
                        "source": "provider",
                        "signal": HOLD,
                        "weight": weight,
                        "details": {"available": False, "reason": reason},
                        "summary": f"{config.name}: unavailable",
                        "error": reason,
                    }
                )
                continue

            try:
                result = self._run_provider(provider, symbol, df)
                signal_value = self._normalize_signal(result.get("signal", HOLD))
                provider_signals.append(
                    {
                        "name": config.name,
                        "source": "provider",
                        "signal": signal_value,
                        "weight": weight,
                        "details": result.get("details", {}),
                        "summary": result.get("summary") or self._provider_summary(config.name, signal_value),
                    }
                )
            except Exception as exc:
                logger.warning("Provider %s failed for %s: %s", config.name, symbol, exc)
                provider_signals.append(
                    {
                        "name": config.name,
                        "source": "provider",
                        "signal": HOLD,
                        "weight": weight,
                        "details": {"error": str(exc)},
                        "summary": f"{config.name}: unavailable ({exc})",
                        "error": str(exc),
                    }
                )

        return provider_signals

    def _run_provider(self, provider: Any, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Run a provider using whichever method it exposes."""
        call_order: List[tuple[str, Callable[..., Any]]] = []
        for method_name in ("analyze", "get_signal", "predict", "forecast", "evaluate"):
            method = getattr(provider, method_name, None)
            if callable(method):
                call_order.append((method_name, method))

        if not call_order:
            raise AttributeError(f"Provider {provider.__class__.__name__} has no supported interface")

        for method_name, method in call_order:
            try:
                if method_name == "get_signal":
                    raw = method(symbol)
                else:
                    try:
                        raw = method(symbol, df)
                    except TypeError:
                        raw = method(symbol)
                return self._normalize_provider_result(raw)
            except TypeError:
                continue

        raise RuntimeError(f"Could not execute provider {provider.__class__.__name__}")

    @staticmethod
    def _normalize_provider_result(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            result = dict(raw)
            result.setdefault("details", {})
            result.setdefault("summary", "")
            result.setdefault("signal", result.get("value", HOLD))
            return result
        return {"signal": raw, "details": {}, "summary": ""}

    @staticmethod
    def _technical_summary(name: str, signal: int) -> str:
        label = "BULLISH" if signal > 0 else "BEARISH" if signal < 0 else "NEUTRAL"
        return f"{name}: {label}"

    @staticmethod
    def _provider_summary(name: str, signal: int) -> str:
        label = "BULLISH" if signal > 0 else "BEARISH" if signal < 0 else "NEUTRAL"
        return f"{name}: {label}"

    def aggregate(self, signals: list[dict]) -> dict:
        """Aggregate weighted signals into a normalized signal score (NSS)."""
        if not signals:
            return {
                "nss": 0.0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "total_weight": 0.0,
                "breakdown": [],
                "recommendation": "HOLD",
            }

        weighted_sum = 0.0
        total_weight = 0.0
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        breakdown: List[Dict[str, Any]] = []

        for item in signals:
            signal_value = self._normalize_signal(item.get("signal", HOLD))
            weight = abs(self._safe_float(item.get("weight", 0.0)))
            contribution = signal_value * weight
            weighted_sum += contribution
            total_weight += weight

            if signal_value > 0:
                bullish_count += 1
            elif signal_value < 0:
                bearish_count += 1
            else:
                neutral_count += 1

            breakdown.append(
                {
                    "name": item.get("name"),
                    "source": item.get("source", "unknown"),
                    "signal": signal_value,
                    "weight": round(weight, 4),
                    "contribution": round(contribution, 4),
                    "summary": item.get("summary", ""),
                }
            )

        nss = weighted_sum / total_weight if total_weight else 0.0
        nss = max(-1.0, min(1.0, round(nss, 4)))

        return {
            "nss": nss,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "total_weight": round(total_weight, 4),
            "breakdown": breakdown,
            "recommendation": self._recommendation_from_nss(nss),
        }

    @staticmethod
    def _recommendation_from_nss(nss: float) -> str:
        if nss >= 0.50:
            return "STRONG BUY"
        if nss >= 0.30:
            return "BUY"
        if nss <= -0.50:
            return "STRONG SELL"
        if nss <= -0.30:
            return "SELL"
        return "HOLD"

    def _fetch_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch one year of daily bars via yfinance."""
        end_ts = pd.Timestamp(end_date).normalize() if end_date else pd.Timestamp.utcnow().normalize()
        start_ts = pd.Timestamp(start_date).normalize() if start_date else end_ts - pd.Timedelta(days=self.DEFAULT_LOOKBACK_DAYS)

        try:
            df = yf.download(
                symbol,
                start=start_ts.strftime("%Y-%m-%d"),
                end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df = df.rename(columns={col: str(col).title() for col in df.columns})
            df = df[[col for col in ["Open", "High", "Low", "Close", "Volume"] if col in df.columns]].dropna()
            return df
        except Exception as exc:
            logger.error("Failed to fetch data for %s: %s", symbol, exc)
            return pd.DataFrame()

    def _resolve_regime(self, check_date: Optional[date] = None) -> str:
        if not self.use_regime or self.regime_filter is None:
            return "UNKNOWN"

        try:
            return self.regime_filter.get_regime(check_date)
        except Exception as exc:
            logger.warning("Regime detection fallback engaged: %s", exc)

        try:
            spy_df = self.regime_filter._get_spy_data()
            if spy_df.empty:
                return "UNKNOWN"

            index = pd.DatetimeIndex(spy_df.index)
            if index.tz is not None:
                index = index.tz_localize(None)
            spy_df = spy_df.copy()
            spy_df.index = index

            target_date = pd.Timestamp(check_date or date.today()).normalize()
            available = spy_df.index[spy_df.index <= target_date]
            if len(available) == 0:
                return "UNKNOWN"

            row = spy_df.loc[available[-1]]
            price = self._safe_float(row.get("Close"), default=float("nan"))
            sma200 = self._safe_float(row.get("sma200"), default=float("nan"))
            sma50 = self._safe_float(row.get("sma50"), default=float("nan"))
            vol = self._safe_float(row.get("realized_vol"), default=float("nan"))

            if pd.isna(price) or pd.isna(sma200) or pd.isna(sma50):
                return "UNKNOWN"
            if not pd.isna(vol) and vol > self.regime_filter.vol_threshold:
                return "BEAR" if price < sma200 else "CHOPPY"
            if price < sma200:
                return "BEAR"
            if price < sma50:
                return "CHOPPY"
            return "BULL"
        except Exception as fallback_exc:
            logger.warning("Regime detection failed: %s", fallback_exc)
            return "UNKNOWN"

    @staticmethod
    def _should_trade(nss: float, regime: str) -> bool:
        if nss >= 0.30 and regime == "BULL":
            return True
        if nss <= -0.30 and regime == "BEAR":
            return True
        return False

    @staticmethod
    def _trade_direction(nss: float) -> str:
        if nss >= 0.30:
            return "LONG"
        if nss <= -0.30:
            return "SHORT"
        return "HOLD"

    def scan(self, start_date: str = None, end_date: str = None) -> dict[str, dict]:
        """Run a full scan across all tracked symbols."""
        results: Dict[str, Dict[str, Any]] = {}
        cache_date = pd.Timestamp(end_date).date() if end_date else date.today()

        for symbol in self.symbols:
            cached = self._read_cached_result(symbol, cache_date)
            if cached is not None:
                results[symbol] = cached
                continue

            try:
                df = self._fetch_data(symbol, start_date=start_date, end_date=end_date)
                if df.empty:
                    raise ValueError("No market data returned")

                signals = self.collect_signals(symbol, df)
                aggregate = self.aggregate(signals)
                regime = self._resolve_regime(df.index[-1].date() if not df.empty else cache_date)
                should_trade = self._should_trade(aggregate["nss"], regime)

                result = {
                    "symbol": symbol,
                    "as_of": (df.index[-1].date().isoformat() if not df.empty else cache_date.isoformat()),
                    "nss": aggregate["nss"],
                    "recommendation": aggregate["recommendation"],
                    "signals": signals,
                    "breakdown": aggregate["breakdown"],
                    "bullish_count": aggregate["bullish_count"],
                    "bearish_count": aggregate["bearish_count"],
                    "neutral_count": aggregate["neutral_count"],
                    "total_weight": aggregate["total_weight"],
                    "regime": regime,
                    "trade_direction": self._trade_direction(aggregate["nss"]),
                    "should_trade": should_trade,
                    "errors": [item.get("error") for item in signals if item.get("error")],
                }
                results[symbol] = result
                self._write_cached_result(symbol, cache_date, result)
            except Exception as exc:
                logger.error("Scan failed for %s: %s", symbol, exc)
                results[symbol] = {
                    "symbol": symbol,
                    "as_of": cache_date.isoformat(),
                    "nss": 0.0,
                    "recommendation": "HOLD",
                    "signals": [],
                    "breakdown": [],
                    "bullish_count": 0,
                    "bearish_count": 0,
                    "neutral_count": 0,
                    "total_weight": 0.0,
                    "regime": self._resolve_regime(cache_date),
                    "trade_direction": "HOLD",
                    "should_trade": False,
                    "errors": [str(exc)],
                }

        self.last_scan_results = results
        return results

    def get_trade_candidates(self, min_nss: float = 0.30) -> list[dict]:
        """Return tradeable symbols sorted by absolute NSS descending."""
        scan_results = self.last_scan_results or self.scan()
        candidates = []

        for symbol, result in scan_results.items():
            nss = self._safe_float(result.get("nss"))
            if abs(nss) < float(min_nss):
                continue
            if not result.get("should_trade", False):
                continue
            candidates.append(result)

        return sorted(candidates, key=lambda item: abs(self._safe_float(item.get("nss"))), reverse=True)

    def format_report(self, scan_results: dict) -> str:
        """Build a plain-English scan report."""
        report_date = datetime.utcnow().strftime("%Y-%m-%d")
        lines = [
            f"📊 Trading Alpha Signal Report — {report_date}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
        ]

        for symbol, result in scan_results.items():
            nss = self._safe_float(result.get("nss"))
            recommendation = result.get("recommendation", "HOLD")
            regime = result.get("regime", "UNKNOWN")
            header_emoji = "✅" if result.get("should_trade") else "⏸️"
            lines.append(f"{symbol}: NSS {nss:+.2f} → {recommendation} {header_emoji}")
            lines.append(
                f"  Technical: {result.get('bullish_count', 0)} bullish, {result.get('bearish_count', 0)} bearish"
            )

            provider_names = ["fundamentals", "valuation", "sentiment", "buffett", "arima_forecast"]
            signals_by_name = {item.get("name"): item for item in result.get("signals", [])}
            for provider_name in provider_names:
                item = signals_by_name.get(provider_name)
                if not item:
                    continue
                label = "BULLISH" if self._normalize_signal(item.get("signal")) > 0 else "BEARISH" if self._normalize_signal(item.get("signal")) < 0 else "NEUTRAL"
                summary = item.get("summary", "").strip()
                if summary and ":" in summary:
                    detail = summary.split(":", 1)[1].strip()
                else:
                    detail = summary or "no additional detail"
                pretty_name = provider_name.replace("_", " ").title()
                lines.append(f"  {pretty_name}: {label} ({detail})")

            regime_emoji = "✅" if regime == "BULL" else "⚠️" if regime == "BEAR" else "⏸️"
            lines.append(f"  Regime: {regime} market {regime_emoji}")

            errors = result.get("errors") or []
            if errors:
                lines.append(f"  Notes: {len(errors)} source issue(s) encountered")
            lines.append("")

        return "\n".join(lines).strip()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    hub = SignalHub(symbols=["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"])
    results = hub.scan()
    print(hub.format_report(results))
    print("\nTop candidates:")
    for candidate in hub.get_trade_candidates():
        print(f"- {candidate['symbol']}: NSS {candidate['nss']:+.2f} | {candidate['recommendation']} | Regime {candidate['regime']}")
