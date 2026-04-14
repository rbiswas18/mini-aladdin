"""
regime_filter.py — Market Regime Detection + Earnings Blackout
Prevents trading in bad market conditions and around earnings announcements.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
import ta

from strategy_simple import Strategy, BUY, SELL, HOLD

logger = logging.getLogger(__name__)


class RegimeFilter:
    """
    Detects the current market regime using SPY as a proxy for the overall market.

    Regimes:
    - BULL: SPY above 200-day SMA, 50-day SMA trending up, low volatility
    - BEAR: SPY below 200-day SMA
    - CHOPPY: Mixed signals or high volatility
    """

    def __init__(self, vol_threshold: float = 0.02):
        """
        Args:
            vol_threshold: Daily realized vol above this = CHOPPY (default 2%)
        """
        self.vol_threshold = vol_threshold
        self._spy_cache: Optional[pd.DataFrame] = None
        self._cache_date: Optional[date] = None

    def _get_spy_data(self) -> pd.DataFrame:
        """Fetch SPY data with caching (refreshes once per day)."""
        today = date.today()
        if self._spy_cache is not None and self._cache_date == today:
            return self._spy_cache

        try:
            spy = yf.Ticker("SPY")
            df = spy.history(period="2y", auto_adjust=True)
            if df.empty:
                raise ValueError("No SPY data returned")
            df["sma200"] = ta.trend.sma_indicator(df["Close"], window=200)
            df["sma50"] = ta.trend.sma_indicator(df["Close"], window=50)
            df["realized_vol"] = df["Close"].pct_change().rolling(20).std()
            self._spy_cache = df
            self._cache_date = today
            return df
        except Exception as e:
            logger.error(f"Failed to fetch SPY data: {e}")
            return pd.DataFrame()

    def get_regime(self, check_date: Optional[date] = None) -> str:
        """
        Get market regime for a given date.

        Args:
            check_date: Date to check (default: today)

        Returns:
            "BULL", "BEAR", or "CHOPPY"
        """
        df = self._get_spy_data()
        if df.empty:
            logger.warning("No SPY data — defaulting to BULL")
            return "BULL"

        if check_date is None:
            check_date = date.today()

        # Find nearest available date
        check_ts = pd.Timestamp(check_date)
        available = df.index[df.index <= check_ts]
        if len(available) == 0:
            return "BULL"

        row = df.loc[available[-1]]

        price = row["Close"]
        sma200 = row.get("sma200")
        sma50 = row.get("sma50")
        vol = row.get("realized_vol")

        if pd.isna(sma200) or pd.isna(sma50):
            return "BULL"  # not enough data, assume bullish

        # High volatility = choppy regardless of trend
        if not pd.isna(vol) and vol > self.vol_threshold:
            if price < sma200:
                return "BEAR"
            return "CHOPPY"

        # Price below 200-day = bear
        if price < sma200:
            return "BEAR"

        # Price above 200-day but below 50-day = choppy
        if price < sma50:
            return "CHOPPY"

        return "BULL"

    def is_bullish(self, check_date: Optional[date] = None) -> bool:
        """Returns True if regime is BULL."""
        return self.get_regime(check_date) == "BULL"

    def get_regime_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Get regime for every date in a DataFrame index.
        Returns a Series of "BULL"/"BEAR"/"CHOPPY" strings.
        """
        spy_df = self._get_spy_data()
        if spy_df.empty:
            return pd.Series("BULL", index=df.index)

        regimes = []
        for ts in df.index:
            available = spy_df.index[spy_df.index <= ts]
            if len(available) == 0:
                regimes.append("BULL")
                continue
            row = spy_df.loc[available[-1]]
            price = row["Close"]
            sma200 = row.get("sma200")
            sma50 = row.get("sma50")
            vol = row.get("realized_vol")

            if pd.isna(sma200) or pd.isna(sma50):
                regimes.append("BULL")
                continue
            if not pd.isna(vol) and vol > self.vol_threshold:
                regimes.append("BEAR" if price < sma200 else "CHOPPY")
            elif price < sma200:
                regimes.append("BEAR")
            elif price < sma50:
                regimes.append("CHOPPY")
            else:
                regimes.append("BULL")

        return pd.Series(regimes, index=df.index)


class EarningsFilter:
    """
    Blocks trading 1 day before and 1 day after earnings announcements.
    Uses yfinance calendar data.
    """

    def __init__(self):
        self._cache: dict = {}

    def _get_earnings_dates(self, symbol: str) -> list[date]:
        """Fetch earnings dates for a symbol."""
        if symbol in self._cache:
            return self._cache[symbol]

        try:
            ticker = yf.Ticker(symbol)
            cal = ticker.calendar
            earnings_dates = []

            if cal is not None and not cal.empty:
                for col in cal.columns:
                    if "Earnings" in str(col):
                        for val in cal[col]:
                            try:
                                if pd.notna(val):
                                    earnings_dates.append(pd.Timestamp(val).date())
                            except Exception:
                                pass

            # Also try earnings_dates attribute
            try:
                ed = ticker.earnings_dates
                if ed is not None and not ed.empty:
                    for ts in ed.index:
                        earnings_dates.append(ts.date())
            except Exception:
                pass

            self._cache[symbol] = list(set(earnings_dates))
            return self._cache[symbol]
        except Exception as e:
            logger.warning(f"Could not fetch earnings for {symbol}: {e}")
            return []

    def is_earnings_blackout(self, symbol: str, check_date: date) -> bool:
        """
        Returns True if check_date is within 1 day of an earnings announcement.
        """
        try:
            earnings_dates = self._get_earnings_dates(symbol)
            for ed in earnings_dates:
                if abs((check_date - ed).days) <= 1:
                    return True
            return False
        except Exception:
            return False


class RegimeAwareStrategy(Strategy):
    """
    Wrapper that adds regime filtering and earnings blackout to any strategy.
    Only generates signals during BULL regimes and outside earnings windows.
    Drop-in replacement for any Strategy.
    """

    def __init__(
        self,
        strategy: Strategy,
        symbol: str = "SPY",
        use_regime: bool = True,
        use_earnings_filter: bool = True,
    ):
        super().__init__()
        self.strategy = strategy
        self.symbol = symbol
        self.use_regime = use_regime
        self.use_earnings_filter = use_earnings_filter
        self.regime_filter = RegimeFilter()
        self.earnings_filter = EarningsFilter()

    @property
    def name(self) -> str:
        filters = []
        if self.use_regime:
            filters.append("Regime")
        if self.use_earnings_filter:
            filters.append("Earnings")
        suffix = f" [{'+'.join(filters)} Filter]" if filters else ""
        return f"{self.strategy.name}{suffix}"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # Get base signals from underlying strategy
        df = self.strategy.generate_signals(df)

        if self.use_regime:
            regime_series = self.regime_filter.get_regime_series(df)
            # Only block NEW ENTRIES (BUY) on non-bull days
            # NEVER block protective exits (SELL signals)
            non_bull_mask = regime_series != "BULL"
            buy_mask = df["signal"] == BUY
            df.loc[non_bull_mask & buy_mask, "signal"] = HOLD
            df["regime"] = regime_series

        if self.use_earnings_filter:
            for ts in df.index:
                # Only block BUY entries around earnings, never SELL exits
                if df.loc[ts, "signal"] == BUY:
                    if self.earnings_filter.is_earnings_blackout(self.symbol, ts.date()):
                        df.loc[ts, "signal"] = HOLD

        return df


if __name__ == "__main__":
    rf = RegimeFilter()
    print(f"Current regime: {rf.get_regime()}")
    print(f"Is bullish: {rf.is_bullish()}")

    ef = EarningsFilter()
    print(f"AAPL earnings blackout today: {ef.is_earnings_blackout('AAPL', date.today())}")
