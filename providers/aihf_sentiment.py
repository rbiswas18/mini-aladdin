"""
aihf_sentiment.py - Free news sentiment provider for Trading Alpha.

Sources:
- Google News RSS (primary)
- yfinance news (fallback, if available)

Sentiment engine:
- NLTK VADER (preferred)
- Simple keyword scorer (fallback)

All public methods return:
{
    "signal": int,        # +1, 0, or -1
    "score": float,       # -1.0 to +1.0
    "details": dict,      # headlines analyzed, counts, source, cache info
    "summary": str,       # plain English summary
    "error": str | None
}
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional
from urllib.error import URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 4 * 60 * 60  # 4 hours
POSITIVE_THRESHOLD = 0.15
NEGATIVE_THRESHOLD = -0.15

POSITIVE_WORDS = [
    "surge",
    "rally",
    "beat",
    "strong",
    "growth",
    "record",
    "profit",
    "gain",
    "upgrade",
]
NEGATIVE_WORDS = [
    "crash",
    "fall",
    "miss",
    "weak",
    "loss",
    "cut",
    "downgrade",
    "concern",
    "risk",
    "decline",
]


class SentimentProvider:
    def __init__(self, max_articles: int = 10):
        self.max_articles = max(1, int(max_articles))
        self.cache_dir = Path(__file__).resolve().parent.parent / "data" / "cache" / "sentiment"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._vader = self._init_vader()

    def analyze(self, symbol: str, company_name: str = None) -> dict:
        symbol = (symbol or "").upper().strip()
        company_name = (company_name or "").strip() or None

        if not symbol:
            return self._result(
                signal=0,
                score=0.0,
                details={"articles": [], "article_count": 0},
                summary="No symbol provided.",
                error="Symbol is required.",
            )

        cached = self._read_cache(symbol, company_name)
        if cached is not None:
            return cached

        errors: List[str] = []

        try:
            articles, source = self._fetch_articles(symbol, company_name)
        except Exception as exc:
            logger.exception("Sentiment fetch failed for %s", symbol)
            articles, source = [], "unavailable"
            errors.append(str(exc))

        if not articles:
            result = self._result(
                signal=0,
                score=0.0,
                details={
                    "articles": [],
                    "article_count": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": 0,
                    "source": source,
                    "method": "none",
                    "cached": False,
                },
                summary=f"{symbol}: No recent news articles found. Signal: NEUTRAL",
                error="; ".join(errors) if errors else "No articles found.",
            )
            self._write_cache(symbol, company_name, result)
            return result

        try:
            analysis = self._score_articles(articles)
            summary = self._build_summary(symbol, analysis)
            result = self._result(
                signal=analysis["signal"],
                score=analysis["score"],
                details={
                    "articles": analysis["articles"],
                    "article_count": analysis["article_count"],
                    "positive_count": analysis["positive_count"],
                    "negative_count": analysis["negative_count"],
                    "neutral_count": analysis["neutral_count"],
                    "source": source,
                    "method": analysis["method"],
                    "cached": False,
                },
                summary=summary,
                error="; ".join(errors) if errors else None,
            )
        except Exception as exc:
            logger.exception("Sentiment scoring failed for %s", symbol)
            result = self._result(
                signal=0,
                score=0.0,
                details={
                    "articles": articles,
                    "article_count": len(articles),
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": len(articles),
                    "source": source,
                    "method": "error",
                    "cached": False,
                },
                summary=f"{symbol}: Sentiment analysis failed. Signal: NEUTRAL",
                error=str(exc),
            )

        self._write_cache(symbol, company_name, result)
        return result

    def _fetch_articles(self, symbol: str, company_name: Optional[str]) -> tuple[list[dict], str]:
        queries = [f"{symbol} stock"]
        if company_name:
            queries.insert(0, f'"{company_name}" stock')

        all_articles: List[dict] = []
        seen = set()
        last_error = None

        for query in queries:
            try:
                for article in self._fetch_google_news_rss(query):
                    key = (article.get("title", "").strip().lower(), article.get("description", "").strip().lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    all_articles.append(article)
                    if len(all_articles) >= self.max_articles:
                        return all_articles[: self.max_articles], "google_news_rss"
            except Exception as exc:
                last_error = exc

        if all_articles:
            return all_articles[: self.max_articles], "google_news_rss"

        try:
            yf_articles = self._fetch_yfinance_news(symbol)
            if yf_articles:
                return yf_articles[: self.max_articles], "yfinance"
        except Exception as exc:
            if last_error:
                raise RuntimeError(f"Google News RSS failed: {last_error}; yfinance fallback failed: {exc}")
            raise RuntimeError(f"yfinance fallback failed: {exc}")

        if last_error:
            raise RuntimeError(f"Google News RSS failed: {last_error}")
        raise RuntimeError("No news articles found from available free sources.")

    def _fetch_google_news_rss(self, query: str) -> list[dict]:
        url = (
            "https://news.google.com/rss/search?"
            f"q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        )

        try:
            import feedparser  # type: ignore

            parsed = feedparser.parse(url)
            entries = getattr(parsed, "entries", []) or []
            articles = []
            for entry in entries[: self.max_articles]:
                title = self._clean_text(getattr(entry, "title", ""))
                description = self._clean_text(
                    getattr(entry, "summary", "") or getattr(entry, "description", "")
                )
                if title:
                    articles.append({"title": title, "description": description})
            if articles:
                return articles
        except Exception as exc:
            logger.debug("feedparser unavailable or failed: %s", exc)

        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
            },
        )
        try:
            with urlopen(request, timeout=15) as response:
                raw = response.read()
        except URLError as exc:
            raise RuntimeError(f"Failed to fetch Google News RSS: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Unexpected RSS fetch error: {exc}") from exc

        try:
            root = ET.fromstring(raw)
        except ET.ParseError as exc:
            raise RuntimeError(f"Failed to parse Google News RSS XML: {exc}") from exc

        articles = []
        for item in root.findall("./channel/item")[: self.max_articles]:
            title = self._clean_text(item.findtext("title", default=""))
            description = self._clean_text(item.findtext("description", default=""))
            if title:
                articles.append({"title": title, "description": description})
        return articles

    def _fetch_yfinance_news(self, symbol: str) -> list[dict]:
        try:
            import yfinance as yf  # type: ignore
        except Exception as exc:
            raise RuntimeError("yfinance is not installed.") from exc

        ticker = yf.Ticker(symbol)
        news_items = getattr(ticker, "news", None) or []

        articles = []
        for item in news_items[: self.max_articles]:
            title = self._clean_text(item.get("title", ""))
            description = self._clean_text(
                item.get("summary")
                or item.get("snippet")
                or item.get("content")
                or ""
            )
            if title:
                articles.append({"title": title, "description": description})
        return articles

    def _init_vader(self):
        try:
            import nltk  # type: ignore
            from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore

            try:
                analyzer = SentimentIntensityAnalyzer()
                return analyzer
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
                return SentimentIntensityAnalyzer()
        except Exception as exc:
            logger.debug("VADER unavailable, using keyword fallback: %s", exc)
            return None

    def _score_articles(self, articles: list[dict]) -> dict:
        if self._vader is not None:
            return self._score_with_vader(articles)
        return self._score_with_keywords(articles)

    def _score_with_vader(self, articles: list[dict]) -> dict:
        scored_articles = []
        compounds = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for article in articles:
            text = f"{article.get('title', '')}. {article.get('description', '')}".strip()
            scores = self._vader.polarity_scores(text)
            compound = float(scores.get("compound", 0.0))
            label = self._label_from_score(compound)

            if label == "positive":
                positive_count += 1
            elif label == "negative":
                negative_count += 1
            else:
                neutral_count += 1

            compounds.append(compound)
            scored_articles.append(
                {
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "score": round(compound, 4),
                    "label": label,
                }
            )

        avg_score = sum(compounds) / len(compounds) if compounds else 0.0
        signal = self._signal_from_score(avg_score)
        return {
            "signal": signal,
            "score": round(avg_score, 4),
            "articles": scored_articles,
            "article_count": len(scored_articles),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "method": "vader",
        }

    def _score_with_keywords(self, articles: list[dict]) -> dict:
        scored_articles = []
        article_scores = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}".lower()
            positive_hits = sum(len(re.findall(rf"\b{re.escape(word)}\b", text)) for word in POSITIVE_WORDS)
            negative_hits = sum(len(re.findall(rf"\b{re.escape(word)}\b", text)) for word in NEGATIVE_WORDS)

            total_hits = positive_hits + negative_hits
            if total_hits == 0:
                score = 0.0
            else:
                score = (positive_hits - negative_hits) / total_hits

            label = self._label_from_score(score)
            if label == "positive":
                positive_count += 1
            elif label == "negative":
                negative_count += 1
            else:
                neutral_count += 1

            article_scores.append(score)
            scored_articles.append(
                {
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "score": round(score, 4),
                    "label": label,
                    "positive_hits": positive_hits,
                    "negative_hits": negative_hits,
                }
            )

        avg_score = sum(article_scores) / len(article_scores) if article_scores else 0.0
        signal = self._signal_from_score(avg_score)
        return {
            "signal": signal,
            "score": round(avg_score, 4),
            "articles": scored_articles,
            "article_count": len(scored_articles),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "method": "keyword_fallback",
        }

    def _build_summary(self, symbol: str, analysis: dict) -> str:
        total = analysis.get("article_count", 0)
        pos = analysis.get("positive_count", 0)
        neg = analysis.get("negative_count", 0)
        neu = analysis.get("neutral_count", 0)
        score = float(analysis.get("score", 0.0))
        signal = analysis.get("signal", 0)

        if signal > 0:
            outlook = "BULLISH"
        elif signal < 0:
            outlook = "BEARISH"
        else:
            outlook = "NEUTRAL"

        dominant = max(
            [(pos, "positive"), (neg, "negative"), (neu, "neutral")],
            key=lambda x: x[0],
        )[1]

        if total == 0:
            return f"{symbol}: No articles analyzed. Signal: NEUTRAL"

        return (
            f"{symbol}: {pos}/{total} articles positive, {neg}/{total} negative, "
            f"{neu}/{total} neutral. Dominant tone: {dominant}. "
            f"Avg sentiment {score:+.2f}. Signal: {outlook}"
        )

    def _read_cache(self, symbol: str, company_name: Optional[str]) -> Optional[dict]:
        cache_path = self._cache_path(symbol, company_name)
        if not cache_path.exists():
            return None

        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            created_at = float(payload.get("created_at", 0))
            if time.time() - created_at > CACHE_TTL_SECONDS:
                return None

            result = payload.get("result")
            if isinstance(result, dict):
                details = result.setdefault("details", {})
                details["cached"] = True
                details["cache_age_seconds"] = int(time.time() - created_at)
                return result
        except Exception as exc:
            logger.debug("Failed to read sentiment cache %s: %s", cache_path, exc)
        return None

    def _write_cache(self, symbol: str, company_name: Optional[str], result: dict) -> None:
        cache_path = self._cache_path(symbol, company_name)
        try:
            payload = {
                "created_at": time.time(),
                "symbol": symbol,
                "company_name": company_name,
                "result": result,
            }
            cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("Failed to write sentiment cache %s: %s", cache_path, exc)

    def _cache_path(self, symbol: str, company_name: Optional[str]) -> Path:
        key = f"{symbol}|{company_name or ''}|{self.max_articles}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text or ""
        text = unescape(text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _signal_from_score(score: float) -> int:
        if score > POSITIVE_THRESHOLD:
            return 1
        if score < NEGATIVE_THRESHOLD:
            return -1
        return 0

    @staticmethod
    def _label_from_score(score: float) -> str:
        if score > POSITIVE_THRESHOLD:
            return "positive"
        if score < NEGATIVE_THRESHOLD:
            return "negative"
        return "neutral"

    @staticmethod
    def _result(signal: int, score: float, details: dict, summary: str, error: Optional[str]) -> dict:
        return {
            "signal": int(signal),
            "score": round(float(score), 4),
            "details": details if isinstance(details, dict) else {},
            "summary": summary,
            "error": error,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    provider = SentimentProvider(max_articles=8)
    print(json.dumps(provider.analyze("NVDA", company_name="NVIDIA"), indent=2))
