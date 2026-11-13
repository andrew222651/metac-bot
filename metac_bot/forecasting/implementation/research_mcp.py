"""Optional research MCP, separate from Metaculus forecast validation."""

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Annotated, Literal

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from ...infrastructure import market_data, research
from ...infrastructure.paths import ROOT
from ...infrastructure.provider_errors import ProviderUnavailable
from ..strategy.research import RESEARCH_INSTRUCTIONS

load_dotenv(ROOT / ".env")
mcp = FastMCP("research", instructions=RESEARCH_INSTRUCTIONS)
Query = Annotated[str, Field(min_length=1, max_length=12000)]
logger = logging.getLogger(__name__)
_cooldowns: dict[str, tuple[float, str]] = {}


async def _optional_result(provider: str, call: Callable[[], Awaitable[dict]]) -> dict:
    """Treat an optional provider failure as not using that service."""
    now = time.monotonic()
    until, reason = _cooldowns.get(provider, (0, ""))
    if now < until:
        return {
            "provider": provider,
            "status": "unavailable",
            "reason": reason,
            "guidance": "Use another source and report the limitation; retry later.",
        }
    try:
        return await call()
    except ProviderUnavailable as exc:
        reason = str(exc)
        # 429 may be transient rate limiting, not necessarily exhausted credits.
        seconds = {402: 300, 429: 30}.get(exc.status_code, 0)
        if seconds:
            _cooldowns[provider] = (time.monotonic() + seconds, reason)
        logger.warning(
            "Optional research provider %s unavailable: %s", provider, reason
        )
        return {
            "provider": provider,
            "status": "unavailable",
            "reason": reason,
            "guidance": "Use another source and report the limitation; retry later.",
        }


@mcp.tool()
def available_providers() -> dict[str, bool]:
    """List configured AskNews/Perplexity/Exa credentials (not a connectivity test)."""
    return research.available_providers()


@mcp.tool()
async def asknews_research(
    query: Query, depth: Literal["news", "low", "medium", "high"] = "news"
) -> dict:
    """Get AskNews recent/historical news summaries or low/medium/high deep research.

    Uses ASKNEWS_API_KEY or ASKNEWS_CLIENT_ID + ASKNEWS_SECRET. Results are
    provider-generated, with inline sources. Higher depths use more time/credits.
    """
    return await _optional_result(
        "asknews", lambda: research.asknews_research(query, depth)
    )


@mcp.tool()
async def perplexity_research(
    query: Query, effort: Literal["fast", "low", "medium", "high", "xhigh"] = "fast"
) -> dict:
    """Get provider-generated research with sources using PERPLEXITY_API_KEY.

    Uses Perplexity's Agent API. fast replaces the old Sonar/Pro lookup;
    high is deep research. Higher effort consumes more time/credits.
    """
    return await _optional_result(
        "perplexity", lambda: research.perplexity_research(query, effort)
    )


@mcp.tool()
async def exa_search(
    query: Query,
    num_results: Annotated[int, Field(ge=1, le=10)] = 5,
    include_domains: list[str] | None = None,
) -> dict:
    """Search Exa for URLs, dates, and page text using EXA_API_KEY.

    Optional domain restrictions prioritize primary sources. Returns up to 6000
    characters per page; synthesize the evidence yourself in OpenCode.
    """
    return await _optional_result(
        "exa", lambda: research.exa_search(query, num_results, include_domains)
    )


@mcp.tool()
async def yahoo_financial_series(
    symbol: Annotated[str, Field(min_length=1, max_length=40)],
) -> dict:
    """Retrieve up to a year of dated daily Yahoo Finance closes for a known ticker."""
    return await _optional_result("yahoo", lambda: market_data.yahoo_series(symbol))


@mcp.tool()
async def fred_financial_series(
    series_id: Annotated[str, Field(min_length=1, max_length=40)],
) -> dict:
    """Retrieve recent FRED observations for a known series ID; requires FRED_API_KEY."""
    return await _optional_result("fred", lambda: market_data.fred_series(series_id))


@mcp.tool()
async def prediction_market_search(
    query: Annotated[str, Field(min_length=1, max_length=500)],
) -> dict:
    """Find unverified candidates across four markets; Kalshi scans only part of its catalogue."""
    return await market_data.prediction_markets(query)


@mcp.tool()
async def kalshi_market(
    ticker: Annotated[str, Field(min_length=1, max_length=100)],
) -> dict:
    """Retrieve the rules and prices of a known Kalshi market ticker."""
    return await _optional_result("kalshi", lambda: market_data.kalshi_market(ticker))


if __name__ == "__main__":
    mcp.run(transport="stdio")
