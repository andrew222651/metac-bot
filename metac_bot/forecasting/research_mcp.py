"""Optional research MCP, separate from Metaculus forecast validation."""

from typing import Annotated, Literal

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from ..infrastructure import research
from ..infrastructure.paths import ROOT

load_dotenv(ROOT / ".env")
mcp = FastMCP(
    "research",
    instructions=(
        "Optional research providers. Call available_providers to check configuration. "
        "Load the forecasting-research skill for source selection and synthesis. "
        "AskNews and Perplexity return provider-generated research; verify cited sources. "
        "Exa returns source text for OpenCode to synthesize."
    ),
)
Query = Annotated[str, Field(min_length=1, max_length=12000)]


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
    return await research.asknews_research(query, depth)


@mcp.tool()
async def perplexity_research(
    query: Query, effort: Literal["fast", "low", "medium", "high", "xhigh"] = "fast"
) -> dict:
    """Get provider-generated research with sources using PERPLEXITY_API_KEY.

    Uses Perplexity's Agent API. fast replaces the old Sonar/Pro lookup;
    high is deep research. Higher effort consumes more time/credits.
    """
    return await research.perplexity_research(query, effort)


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
    return await research.exa_search(query, num_results, include_domains)


if __name__ == "__main__":
    mcp.run(transport="stdio")
