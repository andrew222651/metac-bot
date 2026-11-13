"""Bounded, read-only financial and prediction-market research."""

import asyncio
from datetime import datetime, timezone
import logging
import os
import re
from typing import Any
from urllib.parse import quote

import httpx

from .runtime import is_real_env
from .provider_errors import ProviderUnavailable

# FRED requires its key in the query string; httpx logs complete URLs at INFO.
logging.getLogger("httpx").setLevel(logging.WARNING)


def _retrieved_at() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _get(url: str, params: dict[str, Any] | None = None) -> Any:
    try:
        async with asyncio.timeout(25):
            async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()
    except httpx.HTTPStatusError as exc:
        raise ProviderUnavailable(
            f"Data provider returned HTTP {exc.response.status_code}; check credentials, quota, and request parameters",
            status_code=exc.response.status_code,
        ) from None
    except (httpx.HTTPError, ValueError, TimeoutError):
        # URLs and response bodies can contain API keys; do not expose them.
        raise ProviderUnavailable(
            "Data provider request failed or returned invalid JSON"
        ) from None


def _identifier(value: str, pattern: str) -> str:
    value = value.strip()
    if not re.fullmatch(pattern, value):
        raise ValueError("Invalid data identifier")
    return value


async def yahoo_series(symbol: str) -> dict:
    symbol = _identifier(symbol, r"[A-Za-z0-9^=._-]{1,40}")
    data = await _get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}",
        {"range": "1y", "interval": "1d"},
    )
    try:
        result = data["chart"]["result"][0]
        meta = result["meta"]
        observations = [
            {"time": datetime.fromtimestamp(t, timezone.utc).isoformat(), "close": v}
            for t, v in zip(
                result.get("timestamp", []), result["indicators"]["quote"][0]["close"]
            )
            if v is not None
        ]
    except (KeyError, IndexError, TypeError, ValueError):
        raise ProviderUnavailable("Yahoo Finance returned no usable series") from None
    return {
        "provider": "Yahoo Finance",
        "symbol": symbol,
        "source_url": f"https://finance.yahoo.com/quote/{quote(symbol, safe='')}/",
        "retrieved_at": _retrieved_at(),
        "currency": meta.get("currency"),
        "exchange_timezone": meta.get("exchangeTimezoneName"),
        "observations": observations,
        "note": "Daily close values may be delayed, unadjusted, incomplete, or revised; the latest bar may be in progress.",
    }


async def fred_series(series_id: str) -> dict:
    series_id = _identifier(series_id, r"[A-Za-z0-9_]{1,40}")
    key = os.getenv("FRED_API_KEY", "").strip()
    if not key or not is_real_env("FRED_API_KEY"):
        raise ProviderUnavailable("FRED_API_KEY is not configured")
    data = await _get(
        "https://api.stlouisfed.org/fred/series/observations",
        {
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 100,
        },
    )
    if not isinstance(data, dict) or not isinstance(data.get("observations"), list):
        raise ProviderUnavailable("FRED returned no usable series")
    return {
        "provider": "FRED",
        "series_id": series_id,
        "source_url": f"https://fred.stlouisfed.org/series/{series_id}",
        "retrieved_at": _retrieved_at(),
        "observations": data["observations"],
        "note": "Observation dates are not release dates; values may be revised. A dot means missing data.",
    }


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3}


async def _market_source(source: str, query: str) -> dict:
    if source == "kalshi":
        words = _words(query)
        events = []
        cursor = None
        pages = 0
        # This is a bounded catalogue scan, not a complete search; callers may
        # also use the exact-ticker tool after finding a contract elsewhere.
        for _ in range(3):
            params = {"status": "open", "limit": 200, "with_nested_markets": "true"}
            if cursor:
                params["cursor"] = cursor
            data = await _get(
                "https://api.elections.kalshi.com/trade-api/v2/events", params
            )
            if not isinstance(data, dict) or not isinstance(data.get("events"), list):
                raise RuntimeError("Invalid Kalshi response")
            pages += 1
            for event in data["events"]:
                if not isinstance(event, dict):
                    continue
                title = event.get("title") or ""
                if isinstance(title, str) and words & _words(title):
                    events.append(event)
            cursor = data.get("cursor")
            if not cursor:
                break
        events.sort(key=lambda e: len(words & _words(e["title"])), reverse=True)
        markets = [
            {
                "event": e.get("title"),
                "event_ticker": e.get("event_ticker"),
                "markets": [
                    {
                        k: m.get(k)
                        for k in (
                            "ticker",
                            "title",
                            "rules_primary",
                            "close_time",
                            "status",
                            "yes_bid_dollars",
                            "yes_ask_dollars",
                            "last_price_dollars",
                            "volume_fp",
                        )
                    }
                    for m in e.get("markets", [])[:20]
                    if isinstance(m, dict)
                ],
            }
            for e in events[:10]
        ]
        return {
            "provider": source,
            "retrieved_at": _retrieved_at(),
            "candidates": markets,
            "catalogue_pages_scanned": pages,
            "catalogue_incomplete": bool(cursor),
        }
    if source == "manifold":
        data = await _get(
            "https://api.manifold.markets/v0/search-markets",
            {"term": query, "filter": "open", "limit": 10},
        )
        if not isinstance(data, list):
            raise RuntimeError("Invalid Manifold response")
        markets = [
            {
                k: x.get(k)
                for k in (
                    "id",
                    "question",
                    "url",
                    "outcomeType",
                    "probability",
                    "answers",
                    "closeTime",
                    "volume",
                    "totalLiquidity",
                    "token",
                )
            }
            for x in data
            if isinstance(x, dict)
        ]
    elif source == "polymarket":
        data = await _get(
            "https://gamma-api.polymarket.com/public-search",
            {"q": query, "limit_per_type": 10, "events_status": "active"},
        )
        if not isinstance(data, dict) or not isinstance(data.get("events", []), list):
            raise RuntimeError("Invalid Polymarket response")
        markets = [
            {
                "event": e.get("title"),
                "slug": e.get("slug"),
                "url": f"https://polymarket.com/event/{quote(e['slug'], safe='')}"
                if isinstance(e.get("slug"), str)
                else None,
                "markets": [
                    {
                        k: m.get(k)
                        for k in (
                            "question",
                            "slug",
                            "outcomes",
                            "outcomePrices",
                            "endDate",
                            "volume",
                            "liquidity",
                            "closed",
                        )
                    }
                    for m in e.get("markets", [])[:20]
                    if isinstance(m, dict)
                ],
            }
            for e in data.get("events", [])
            if isinstance(e, dict)
        ]
    elif source == "predictit":
        data = await _get("https://www.predictit.org/api/marketdata/all/")
        if not isinstance(data, dict) or not isinstance(data.get("markets"), list):
            raise RuntimeError("Invalid PredictIt response")
        words = _words(query)
        candidates = [
            m
            for m in data["markets"]
            if isinstance(m, dict) and words & _words(m.get("name", ""))
        ]
        candidates.sort(
            key=lambda m: len(words & _words(m.get("name", ""))), reverse=True
        )
        markets = [
            {
                k: m.get(k)
                for k in ("id", "name", "url", "status", "timeStamp", "contracts")
            }
            for m in candidates[:10]
        ]
    else:
        raise ValueError("Unknown prediction market source")
    return {"provider": source, "retrieved_at": _retrieved_at(), "candidates": markets}


async def prediction_markets(query: str) -> dict:
    if not query.strip() or len(query) > 500:
        raise ValueError("Query must contain 1–500 characters")
    sources = ("manifold", "polymarket", "predictit", "kalshi")
    results = await asyncio.gather(
        *(_market_source(s, query) for s in sources), return_exceptions=True
    )
    return {
        "query": query,
        "sources": [
            r if isinstance(r, dict) else {"provider": s, "error": type(r).__name__}
            for s, r in zip(sources, results)
        ],
        "note": "Search results are unverified candidates, not equivalent forecasts. Check resolution criteria, dates, market liquidity, and price freshness; never assume a price is a probability for this question.",
    }


async def kalshi_market(ticker: str) -> dict:
    ticker = _identifier(ticker, r"[A-Za-z0-9._-]{1,100}")
    data = await _get(f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}")
    if not isinstance(data, dict) or not isinstance(data.get("market"), dict):
        raise ProviderUnavailable("Invalid Kalshi response")
    market = data["market"]
    return {
        "provider": "kalshi",
        "retrieved_at": _retrieved_at(),
        "market": {
            k: market.get(k)
            for k in (
                "ticker",
                "event_ticker",
                "title",
                "rules_primary",
                "rules_secondary",
                "close_time",
                "status",
                "yes_bid_dollars",
                "yes_ask_dollars",
                "last_price_dollars",
                "volume_fp",
                "open_interest_fp",
            )
        },
        "note": "Bid, ask and last trade are distinct; compare rules and dates before using.",
    }
