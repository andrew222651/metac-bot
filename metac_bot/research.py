"""Optional research services called by OpenCode through MCP."""

import asyncio
import contextlib
import os
import sys
from typing import Literal

import httpx

from .runtime import is_real_env


def available_providers() -> dict[str, bool]:
    """Report configured credentials without exposing their values."""
    return {
        "asknews": is_real_env("ASKNEWS_API_KEY")
        or (is_real_env("ASKNEWS_CLIENT_ID") and is_real_env("ASKNEWS_SECRET")),
        "perplexity": is_real_env("PERPLEXITY_API_KEY"),
        "exa": is_real_env("EXA_API_KEY"),
    }


def require_provider(provider: str) -> None:
    if not available_providers()[provider]:
        raise ValueError(
            f"{provider} is not configured in the research MCP environment or project .env"
        )


async def post_json(
    provider: str, url: str, key: str, body: dict, timeout: float
) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(
                url, headers={"Authorization": f"Bearer {key}"}, json=body
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # Do not forward provider bodies or authentication headers to the model.
            raise RuntimeError(
                f"{provider} returned HTTP {exc.response.status_code}; check credentials, quota, and request parameters"
            ) from None
        except httpx.RequestError:
            raise RuntimeError(f"{provider} request failed or timed out") from None
        return response.json()


async def asknews_research(
    query: str, depth: Literal["news", "low", "medium", "high"] = "news"
) -> dict:
    """News summaries or the original forecasting-tools deep-research presets."""
    require_provider("asknews")
    if depth not in ("news", "low", "medium", "high"):
        raise ValueError("depth must be news, low, medium, or high")
    # Import lazily: Exa/Perplexity do not need forecasting-tools initialized.
    with contextlib.redirect_stdout(sys.stderr):
        from forecasting_tools import AskNewsSearcher
    searcher = AskNewsSearcher(
        client_id=os.getenv("ASKNEWS_CLIENT_ID")
        if is_real_env("ASKNEWS_CLIENT_ID")
        else None,
        client_secret=os.getenv("ASKNEWS_SECRET")
        if is_real_env("ASKNEWS_SECRET")
        else None,
        api_key=os.getenv("ASKNEWS_API_KEY")
        if is_real_env("ASKNEWS_API_KEY")
        else None,
    )
    preset = (
        "asknews/news-summaries"
        if depth == "news"
        else f"asknews/deep-research/{depth}-depth"
    )
    try:
        async with asyncio.timeout(600):
            text = await searcher.call_preconfigured_version(preset, query)
    except Exception as exc:
        raise RuntimeError(
            f"AskNews research failed ({type(exc).__name__}); check credentials and quota"
        ) from None
    return {
        "provider": "asknews",
        "mode": depth,
        "text": text,
        "provider_generated": True,
    }


async def perplexity_research(
    query: str, effort: Literal["fast", "low", "medium", "high", "xhigh"] = "fast"
) -> dict:
    """Search-grounded synthesis using Perplexity's current Agent API."""
    require_provider("perplexity")
    if effort not in ("fast", "low", "medium", "high", "xhigh"):
        raise ValueError("Invalid Perplexity effort")
    data = await post_json(
        "Perplexity",
        "https://api.perplexity.ai/v1/agent",
        os.environ["PERPLEXITY_API_KEY"],
        {"input": query, "effort": effort},
        timeout=600,
    )
    if data.get("status") != "completed":
        raise RuntimeError(
            f"Perplexity did not complete the research (status={data.get('status')})"
        )
    messages = [
        item for item in data.get("output", []) if item.get("type") == "message"
    ]
    text = "\n\n".join(
        part["text"]
        for item in messages
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )
    if not text.strip():
        raise RuntimeError("Perplexity returned no research text")
    return {
        "provider": "perplexity",
        "provider_generated": True,
        "text": text,
        "messages": messages,
        "sources": [
            source
            for item in data.get("output", [])
            if item.get("type") == "search_results"
            for source in item.get("results", [])
        ],
        "usage": data.get("usage"),
    }


async def exa_search(
    query: str, num_results: int = 5, include_domains: list[str] | None = None
) -> dict:
    """Retrieve source text; OpenCode performs the synthesis."""
    require_provider("exa")
    if not 1 <= num_results <= 10:
        raise ValueError("num_results must be between 1 and 10")
    body = {
        "query": query,
        "type": "auto",
        "numResults": num_results,
        "contents": {"text": {"maxCharacters": 6000}},
    }
    if include_domains:
        body["includeDomains"] = include_domains
    data = await post_json(
        "Exa", "https://api.exa.ai/search", os.environ["EXA_API_KEY"], body, timeout=90
    )
    return {
        "provider": "exa",
        "results": data.get("results", []),
        "notices": data.get("notices", []),
        "cost": data.get("costDollars"),
    }
