"""News keyword monitoring collector for Korean financial RSS feeds.

Monitors 연합뉴스, 한경, 매경 RSS feeds for keyword hits
related to geopolitical events, oil, and critical market triggers.
Returns scored signals with urgency levels for the regime gate.
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

from kospi_corr.collectors.base import BaseCollector
from kospi_corr.domain.errors import DataProviderError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Korean news RSS sources
# ---------------------------------------------------------------------------

RSS_SOURCES: dict[str, str] = {
    "연합뉴스": "https://www.yna.co.kr/rss/economy.xml",
    "한국경제": "https://www.hankyung.com/feed/economy",
    "매일경제": "https://www.mk.co.kr/rss/30100041/",
}

# ---------------------------------------------------------------------------
# Keyword categories (from indicators.json)
# ---------------------------------------------------------------------------

KEYWORD_CATEGORIES: dict[str, list[str]] = {
    "geopolitical": ["이란", "트럼프", "미국", "전쟁", "제재", "관세"],
    "oil_related": ["유가", "원유", "OPEC", "감산", "증산", "셰일"],
    "market_critical": ["금리", "연준", "Fed", "반도체", "수출규제", "환율"],
}

# Category importance weights for sentiment scoring
_CATEGORY_WEIGHTS: dict[str, float] = {
    "geopolitical": 1.5,
    "oil_related": 1.2,
    "market_critical": 1.0,
}

# Maximum article age (hours) for urgency boosting
_URGENCY_FRESH_HOURS = 2
_URGENCY_RECENT_HOURS = 6

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ArticleHit:
    """A single article that matched keywords."""
    title: str
    link: str
    source: str
    published: datetime | None
    matched_keywords: list[str]
    matched_categories: list[str]
    keyword_density: float  # hits / title_length


@dataclass
class NewsSignal:
    """Aggregated news monitoring signal."""
    sentiment_score: float        # 0.0 (calm) → 1.0 (maximum alert)
    keyword_hits: dict[str, int]  # category → total hit count
    top_articles: list[ArticleHit]
    urgency_level: str            # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    collected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_articles_scanned: int = 0
    error_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict for JSON / DataFrame conversion."""
        return {
            "sentiment_score": self.sentiment_score,
            "keyword_hits": self.keyword_hits,
            "urgency_level": self.urgency_level,
            "top_articles_count": len(self.top_articles),
            "total_articles_scanned": self.total_articles_scanned,
            "collected_at": self.collected_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# RSS parsing helpers (using requests + xml.etree — no feedparser)
# ---------------------------------------------------------------------------

_KST = timezone(timedelta(hours=9))

# Common RSS date formats
_DATE_FORMATS = [
    "%a, %d %b %Y %H:%M:%S %z",     # RFC 822 with tz
    "%a, %d %b %Y %H:%M:%S %Z",     # RFC 822 with tz name
    "%Y-%m-%dT%H:%M:%S%z",           # ISO 8601
    "%Y-%m-%dT%H:%M:%S",             # ISO 8601 no tz
    "%Y-%m-%d %H:%M:%S",             # Simple datetime
    "%a, %d %b %Y %H:%M:%S",        # RFC 822 no tz
]


def _parse_date(raw: str | None) -> datetime | None:
    """Best-effort parse of RSS pubDate / published fields."""
    if not raw:
        return None
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_KST)
            return dt
        except ValueError:
            continue
    return None


def _fetch_rss(source_name: str, url: str, timeout: int = 10) -> list[dict[str, str]]:
    """Fetch and parse an RSS feed, returning list of {title, link, pubDate}.

    Handles both RSS 2.0 (<item>) and Atom (<entry>) feeds.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("RSS fetch failed for %s (%s): %s", source_name, url, exc)
        raise DataProviderError(
            f"RSS fetch failed for {source_name}: {exc}",
            source="news_rss",
            symbol=source_name,
        ) from exc

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        logger.warning("RSS parse failed for %s: %s", source_name, exc)
        raise DataProviderError(
            f"RSS XML parse failed for {source_name}: {exc}",
            source="news_rss",
            symbol=source_name,
        ) from exc

    articles: list[dict[str, str]] = []

    # Handle namespace-prefixed Atom feeds
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    # Try RSS 2.0 first: //channel/item
    items = root.findall(".//item")
    if items:
        for item in items:
            title_el = item.find("title")
            link_el = item.find("link")
            date_el = item.find("pubDate")
            articles.append({
                "title": (title_el.text or "").strip() if title_el is not None else "",
                "link": (link_el.text or "").strip() if link_el is not None else "",
                "pubDate": (date_el.text or "").strip() if date_el is not None else "",
            })
    else:
        # Try Atom: //entry
        entries = root.findall(".//entry") or root.findall(".//atom:entry", ns)
        for entry in entries:
            title_el = entry.find("title") or entry.find("atom:title", ns)
            link_el = entry.find("link") or entry.find("atom:link", ns)
            date_el = (
                entry.find("published")
                or entry.find("updated")
                or entry.find("atom:published", ns)
                or entry.find("atom:updated", ns)
            )
            link_href = ""
            if link_el is not None:
                link_href = link_el.get("href", "") or (link_el.text or "")
            articles.append({
                "title": (title_el.text or "").strip() if title_el is not None else "",
                "link": link_href.strip(),
                "pubDate": (date_el.text or "").strip() if date_el is not None else "",
            })

    return articles


# ---------------------------------------------------------------------------
# Keyword matching engine
# ---------------------------------------------------------------------------

def _match_keywords(
    text: str,
    categories: dict[str, list[str]] | None = None,
) -> tuple[list[str], list[str], float]:
    """Match keywords against text.

    Returns (matched_keywords, matched_categories, density).
    Density = total_keyword_hits / max(len(text), 1).
    """
    if categories is None:
        categories = KEYWORD_CATEGORIES

    matched_keywords: list[str] = []
    matched_categories: list[str] = []
    total_hits = 0

    for category, keywords in categories.items():
        cat_hit = False
        for kw in keywords:
            # Case-insensitive match for English terms like OPEC, Fed
            count = len(re.findall(re.escape(kw), text, re.IGNORECASE))
            if count > 0:
                matched_keywords.append(kw)
                total_hits += count
                cat_hit = True
        if cat_hit:
            matched_categories.append(category)

    density = total_hits / max(len(text), 1)
    return matched_keywords, matched_categories, density


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _compute_recency_boost(pub_date: datetime | None, now: datetime) -> float:
    """Boost factor based on how recent the article is.

    Returns a multiplier in [1.0, 2.0].
    """
    if pub_date is None:
        return 1.0

    age_hours = (now - pub_date).total_seconds() / 3600
    if age_hours < 0:
        age_hours = 0

    if age_hours <= _URGENCY_FRESH_HOURS:
        return 2.0
    elif age_hours <= _URGENCY_RECENT_HOURS:
        return 1.5
    elif age_hours <= 24:
        return 1.2
    else:
        return 1.0


def _determine_urgency(sentiment_score: float) -> str:
    """Map a 0-1 sentiment score to an urgency level."""
    if sentiment_score >= 0.75:
        return "CRITICAL"
    elif sentiment_score >= 0.50:
        return "HIGH"
    elif sentiment_score >= 0.25:
        return "MEDIUM"
    else:
        return "LOW"


# ---------------------------------------------------------------------------
# NewsCollector
# ---------------------------------------------------------------------------

class NewsCollector(BaseCollector):
    """Monitors Korean financial news RSS feeds for keyword-based signals.

    Scans 연합뉴스, 한경, 매경 RSS for geopolitical, oil, and
    market-critical keywords.  Produces a :class:`NewsSignal` with
    a composite sentiment score, keyword hit counts, top articles,
    and an urgency level for the regime gate.

    Parameters
    ----------
    sources : dict[str, str] | None
        Override RSS source map (name → URL).  Defaults to built-in
        Korean financial news sources.
    keywords : dict[str, list[str]] | None
        Override keyword category map.  Defaults to indicators.json keywords.
    request_timeout : int
        HTTP timeout per RSS feed in seconds.
    max_articles_per_source : int
        Cap articles processed per source to avoid slow scoring on
        very long feeds.
    """

    def __init__(
        self,
        sources: dict[str, str] | None = None,
        keywords: dict[str, list[str]] | None = None,
        request_timeout: int = 10,
        max_articles_per_source: int = 50,
    ) -> None:
        self._sources = sources or RSS_SOURCES
        self._keywords = keywords or KEYWORD_CATEGORIES
        self._timeout = request_timeout
        self._max_per_source = max_articles_per_source

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    def collect(self, start: date, end: date) -> pd.DataFrame:
        """Collect news signal data as a single-row DataFrame.

        The ``start`` / ``end`` parameters are accepted for interface
        compatibility but news collection is inherently point-in-time
        (latest RSS snapshot).

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame indexed by collection timestamp with
            columns: sentiment_score, urgency_level, geopolitical_hits,
            oil_related_hits, market_critical_hits, total_articles_scanned.
        """
        signal = self.collect_signal()
        row = {
            "sentiment_score": signal.sentiment_score,
            "urgency_level": signal.urgency_level,
            "total_articles_scanned": signal.total_articles_scanned,
        }
        # Flatten keyword hits into separate columns
        for cat in self._keywords:
            row[f"{cat}_hits"] = signal.keyword_hits.get(cat, 0)

        idx = pd.DatetimeIndex([signal.collected_at], name="date")
        return pd.DataFrame([row], index=idx)

    # ------------------------------------------------------------------
    # Core collection logic
    # ------------------------------------------------------------------

    def collect_signal(self) -> NewsSignal:
        """Fetch all RSS feeds and produce a scored NewsSignal."""
        now = datetime.now(timezone.utc)
        all_hits: list[ArticleHit] = []
        category_hits: dict[str, int] = {cat: 0 for cat in self._keywords}
        total_scanned = 0
        error_sources: list[str] = []

        for source_name, url in self._sources.items():
            try:
                raw_articles = _fetch_rss(source_name, url, timeout=self._timeout)
            except DataProviderError:
                error_sources.append(source_name)
                continue

            articles = raw_articles[: self._max_per_source]
            total_scanned += len(articles)

            for art in articles:
                title = art.get("title", "")
                if not title:
                    continue

                matched_kw, matched_cats, density = _match_keywords(
                    title, self._keywords
                )
                if not matched_kw:
                    continue

                pub_dt = _parse_date(art.get("pubDate"))
                hit = ArticleHit(
                    title=title,
                    link=art.get("link", ""),
                    source=source_name,
                    published=pub_dt,
                    matched_keywords=matched_kw,
                    matched_categories=matched_cats,
                    keyword_density=density,
                )
                all_hits.append(hit)

                for cat in matched_cats:
                    cat_kw_count = sum(
                        1 for kw in matched_kw if kw in self._keywords.get(cat, [])
                    )
                    category_hits[cat] = category_hits.get(cat, 0) + cat_kw_count

        # Score
        sentiment = self._compute_sentiment(all_hits, category_hits, now, total_scanned)
        urgency = _determine_urgency(sentiment)

        # Sort hits by density * recency for top articles
        def _sort_key(h: ArticleHit) -> float:
            recency = _compute_recency_boost(h.published, now)
            cat_weight = max(
                (_CATEGORY_WEIGHTS.get(c, 1.0) for c in h.matched_categories),
                default=1.0,
            )
            return h.keyword_density * recency * cat_weight

        all_hits.sort(key=_sort_key, reverse=True)
        top_articles = all_hits[:10]

        return NewsSignal(
            sentiment_score=round(sentiment, 4),
            keyword_hits=category_hits,
            top_articles=top_articles,
            urgency_level=urgency,
            collected_at=now,
            total_articles_scanned=total_scanned,
            error_sources=error_sources,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_sentiment(
        self,
        hits: list[ArticleHit],
        category_hits: dict[str, int],
        now: datetime,
        total_scanned: int,
    ) -> float:
        """Compute composite sentiment score in [0.0, 1.0].

        Factors:
        1. Weighted keyword hit ratio (hits / total articles)
        2. Category diversity (more categories hit → higher)
        3. Recency boost (recent articles count more)
        4. Keyword density across matched articles
        """
        if total_scanned == 0 or not hits:
            return 0.0

        # Factor 1: Weighted hit intensity
        weighted_hits = sum(
            category_hits.get(cat, 0) * _CATEGORY_WEIGHTS.get(cat, 1.0)
            for cat in self._keywords
        )
        # Normalize: if every article has 2 weighted keyword hits, score=1.0
        max_expected = total_scanned * 2.0
        hit_intensity = min(weighted_hits / max_expected, 1.0)

        # Factor 2: Category diversity (0-1)
        active_categories = sum(1 for v in category_hits.values() if v > 0)
        diversity = active_categories / max(len(self._keywords), 1)

        # Factor 3: Average recency boost of matched articles
        recency_scores = [
            _compute_recency_boost(h.published, now) for h in hits
        ]
        avg_recency = sum(recency_scores) / len(recency_scores) if recency_scores else 1.0
        # Normalize from [1.0, 2.0] to [0.0, 1.0]
        recency_factor = (avg_recency - 1.0)

        # Factor 4: Average keyword density
        avg_density = sum(h.keyword_density for h in hits) / len(hits) if hits else 0.0
        # Typical Korean title is ~30 chars; 2 keywords in 30 chars ≈ 0.067
        density_factor = min(avg_density / 0.1, 1.0)

        # Weighted combination
        score = (
            0.40 * hit_intensity
            + 0.20 * diversity
            + 0.25 * recency_factor
            + 0.15 * density_factor
        )
        return min(max(score, 0.0), 1.0)
