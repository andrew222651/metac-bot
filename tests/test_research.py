import json
import os
import unittest
from typing import Literal
from unittest.mock import AsyncMock, patch

import httpx

from metac_bot.infrastructure import research
from metac_bot.forecasting.implementation.research_mcp import mcp


class ResearchTests(unittest.IsolatedAsyncioTestCase):
    def test_availability_requires_real_complete_credentials(self) -> None:
        with patch.dict(
            os.environ,
            {"ASKNEWS_CLIENT_ID": "client", "EXA_API_KEY": "REPLACE_ME"},
            clear=True,
        ):
            self.assertEqual(
                research.available_providers(),
                {"asknews": False, "perplexity": False, "exa": False},
            )
            os.environ["ASKNEWS_SECRET"] = "test-secret"
            self.assertTrue(research.available_providers()["asknews"])

    async def test_missing_key_fails_before_network(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(research, "post_json") as post,
        ):
            with self.assertRaisesRegex(ValueError, "not configured"):
                await research.exa_search("query")
            post.assert_not_called()

    async def test_exa_returns_sources_without_requesting_synthesis(self) -> None:
        def handle(request: httpx.Request) -> httpx.Response:
            self.assertEqual(str(request.url), "https://api.exa.ai/search")
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            body = json.loads(request.content)
            self.assertEqual(body["includeDomains"], ["example.org"])
            self.assertEqual(body["numResults"], 3)
            self.assertEqual(body["contents"], {"text": {"maxCharacters": 6000}})
            return httpx.Response(
                200,
                json={"results": [{"url": "https://example.org", "text": "Evidence"}]},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        with (
            patch.dict(os.environ, {"EXA_API_KEY": "test-key"}, clear=True),
            patch.object(research.httpx, "AsyncClient", return_value=client),
        ):
            result = await research.exa_search("query", 3, ["example.org"])
        self.assertEqual(result["results"][0]["text"], "Evidence")

    async def test_perplexity_keeps_sources_annotations_and_usage(self) -> None:
        source = {"url": "https://example.org", "date": "2026-01-01"}
        annotation = {"type": "url_citation", "url": "https://example.org"}

        def handle(request: httpx.Request) -> httpx.Response:
            self.assertEqual(str(request.url), "https://api.perplexity.ai/v1/agent")
            self.assertEqual(
                json.loads(request.content), {"input": "query", "effort": "high"}
            )
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "output": [
                        {"type": "search_results", "results": [source]},
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "Answer [1]",
                                    "annotations": [annotation],
                                }
                            ],
                        },
                    ],
                    "usage": {"total_tokens": 123},
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        with (
            patch.dict(os.environ, {"PERPLEXITY_API_KEY": "test-key"}, clear=True),
            patch.object(research.httpx, "AsyncClient", return_value=client),
        ):
            result = await research.perplexity_research("query", "high")
        self.assertEqual(result["sources"], [source])
        self.assertEqual(
            result["messages"][0]["content"][0]["annotations"], [annotation]
        )
        self.assertTrue(result["provider_generated"])
        self.assertEqual(result["usage"]["total_tokens"], 123)

    async def test_incomplete_perplexity_response_is_not_success(self) -> None:
        with (
            patch.dict(os.environ, {"PERPLEXITY_API_KEY": "test-key"}, clear=True),
            patch.object(
                research,
                "post_json",
                new_callable=AsyncMock,
                return_value={"status": "failed"},
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "did not complete"):
                await research.perplexity_research("query")

    async def test_provider_error_does_not_expose_response_body(self) -> None:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    401, text="sensitive provider error test-key"
                )
            )
        )
        with patch.object(research.httpx, "AsyncClient", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "HTTP 401") as error:
                await research.post_json(
                    "Exa", "https://api.exa.ai/search", "test-key", {}, 90
                )
        self.assertNotIn("test-key", str(error.exception))

    async def test_asknews_uses_original_presets(self) -> None:
        with (
            patch.dict(os.environ, {"ASKNEWS_API_KEY": "test-key"}, clear=True),
            patch("forecasting_tools.AskNewsSearcher") as searcher,
        ):
            searcher.return_value.call_preconfigured_version = AsyncMock(
                return_value="Research with sources"
            )
            presets: list[tuple[Literal["news", "low", "medium", "high"], str]] = [
                ("news", "asknews/news-summaries"),
                ("low", "asknews/deep-research/low-depth"),
                ("medium", "asknews/deep-research/medium-depth"),
                ("high", "asknews/deep-research/high-depth"),
            ]
            for depth, preset in presets:
                result = await research.asknews_research("query", depth)
                searcher.return_value.call_preconfigured_version.assert_awaited_with(
                    preset, "query"
                )
                self.assertEqual(result["text"], "Research with sources")

    async def test_mcp_catalog_exposes_bounded_provider_tools(self) -> None:
        tools = {tool.name: tool for tool in await mcp.list_tools()}
        self.assertEqual(
            set(tools),
            {
                "available_providers",
                "asknews_research",
                "perplexity_research",
                "exa_search",
            },
        )
        self.assertEqual(
            tools["exa_search"].inputSchema["properties"]["num_results"]["maximum"], 10
        )
        self.assertIn(
            "high", tools["asknews_research"].inputSchema["properties"]["depth"]["enum"]
        )
