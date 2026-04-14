"""
data_fetch.py — Market Data Layer
Provides a unified interface for fetching US stock market data.
Supports yfinance (free) with a Polygon.io stub for future upgrade.
"""

import os
import abc
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

VALID_TIMEFRAMES = {"1d", "1h", "15m", "5m", "1m"}


class MarketDataProvider(abc.ABC):
    """Abstract base class for all market data providers."""

    @abc.abstractmethod
    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for a symbol.

        Args:
            symbol: Ticker symbol (e.g. 'AAPL')
            timeframe: Bar interval — '1d', '1h', '15m', '5m', '1m'
            start: Start date string 'YYYY-MM-DD'
            end: End date string 'YYYY-MM-DD'

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            Index: DatetimeIndex (UTC-aware)
        """

    @abc.abstractmethod
    def get_latest_quote(self, symbol: str) -> dict:
        """
        Fetch the latest quote for a symbol.

        Args:
            symbol: Ticker symbol (e.g. 'AAPL')

        Returns:
            dict with keys: symbol, price, change, change_pct, volume, timestamp
        """


class YFinanceProvider(MarketDataProvider):
    """
    Market data provider backed by yfinance (Yahoo Finance).
    Caches results as Parquet files to avoid redundant network calls.
    """

    TIMEFRAME_MAP = {
        "1d": "1d",
        "1h": "1h",
        "15m": "15m",
        "5m": "5m",
        "1m": "1m",
    }

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: str = "2020-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars using yfinance with local Parquet cache.

        Intraday data (< 1d) is not cached due to yfinance's rolling
        availability window. Daily data is cached indefinitely and
        refreshed only when the requested end date is today.
        """
        if timeframe not in VALID_TIMEFRAMES:
            raise ValueError(f"Invalid timeframe '{timeframe}'. Choose from {VALID_TIMEFRAMES}")

        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        symbol = symbol.upper().strip()
        cache_path = CACHE_DIR / f"{symbol}_{timeframe}_{start}_{end}.parquet"

        # Use cache for daily data only
        if timeframe == "1d" and cache_path.exists():
            logger.info(f"Loading {symbol} from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
            return df

        logger.info(f"Fetching {symbol} [{timeframe}] {start} → {end} from yfinance")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start,
                end=end,
                interval=self.TIMEFRAME_MAP[timeframe],
                auto_adjust=True,   # adjusts for splits/dividends
                actions=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data for {symbol}: {e}")

        if df.empty:
            raise ValueError(f"No data returned for {symbol} between {start} and {end}")

        # Normalize columns
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"

        # Cache daily data
        if timeframe == "1d":
            df.to_parquet(cache_path)
            logger.info(f"Cached {symbol} daily bars to {cache_path}")

        return df

    def get_latest_quote(self, symbol: str) -> dict:
        """
        Fetch the latest available quote for a symbol.

        Returns a dict with price, change, change_pct, volume, timestamp.
        """
        symbol = symbol.upper().strip()
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info

            price = getattr(info, "last_price", None)
            prev_close = getattr(info, "previous_close", None)
            volume = getattr(info, "last_volume", None)

            if price is None:
                # Fallback: fetch last 2 days of daily data
                df = self.get_bars(symbol, "1d", start=(datetime.today() - timedelta(days=5)).strftime("%Y-%m-%d"))
                if df.empty:
                    raise ValueError(f"No quote data for {symbol}")
                price = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else price
                volume = int(df["Volume"].iloc[-1])

            change = round(price - prev_close, 4) if prev_close else 0.0
            change_pct = round((change / prev_close) * 100, 2) if prev_close else 0.0

            return {
                "symbol": symbol,
                "price": round(float(price), 4),
                "change": change,
                "change_pct": change_pct,
                "volume": int(volume) if volume else 0,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to fetch quote for {symbol}: {e}")


class PolygonProvider(MarketDataProvider):
    """
    Stub for Polygon.io integration.
    Upgrade path: set POLYGON_API_KEY in .env and implement methods below.
    Free tier: 15-minute delayed data. $29/mo for real-time.
    """

    def __init__(self):
        self.api_key = os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "POLYGON_API_KEY not set. Add it to your .env file. "
                "Sign up free at https://polygon.io"
            )

    def get_bars(self, symbol, timeframe, start, end) -> pd.DataFrame:
        raise NotImplementedError(
            "PolygonProvider.get_bars() not yet implemented. "
            "Use YFinanceProvider for now."
        )

    def get_latest_quote(self, symbol) -> dict:
        raise NotImplementedError(
            "PolygonProvider.get_latest_quote() not yet implemented. "
            "Use YFinanceProvider for now."
        )


def get_provider(name: str = "yfinance") -> MarketDataProvider:
    """
    Factory function to get a market data provider by name.

    Args:
        name: 'yfinance' (default, free) or 'polygon' (requires API key)

    Returns:
        Instantiated MarketDataProvider
    """
    providers = {
        "yfinance": YFinanceProvider,
        "polygon": PolygonProvider,
    }
    if name not in providers:
        raise ValueError(f"Unknown provider '{name}'. Choose from: {list(providers.keys())}")
    return providers[name]()


if __name__ == "__main__":
    # Quick smoke test
    provider = get_provider("yfinance")
    df = provider.get_bars("AAPL", "1d", "2023-01-01", "2024-01-01")
    print(df.tail())
    quote = provider.get_latest_quote("AAPL")
    print(quote)
