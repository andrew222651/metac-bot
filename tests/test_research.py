import json
import os
import unittest
from typing import Literal, cast
from unittest.mock import AsyncMock, patch

import httpx
from mcp.types import TextContent

from metac_bot.infrastructure import research
from metac_bot.forecasting.implementation import research_mcp
from metac_bot.forecasting.implementation.research_mcp import mcp
from metac_bot.infrastructure.provider_errors import ProviderUnavailable


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
            with self.assertRaisesRegex(ProviderUnavailable, "not configured"):
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

    async def test_asknews_credit_failure_is_safe_and_cools_down(self) -> None:
        error = httpx.HTTPStatusError(
            "secret-provider-body test-key",
            request=httpx.Request("POST", "https://example.org"),
            response=httpx.Response(402),
        )
        search = AsyncMock(side_effect=error)
        with (
            patch.dict(os.environ, {"ASKNEWS_API_KEY": "test-key"}, clear=True),
            patch.dict(research_mcp._cooldowns, {}, clear=True),
            patch("forecasting_tools.AskNewsSearcher") as searcher,
            patch.object(research_mcp.time, "monotonic", return_value=100),
        ):
            searcher.return_value.call_preconfigured_version = search
            first_output = await mcp.call_tool("asknews_research", {"query": "test"})
            second_output = await mcp.call_tool("asknews_research", {"query": "other"})
            first = json.loads(cast(list[TextContent], first_output)[0].text)
            second = json.loads(cast(list[TextContent], second_output)[0].text)
            self.assertEqual(first, second)
            self.assertEqual(first["status"], "unavailable")
            self.assertIn("HTTP 402", first["reason"])
            self.assertIn("another source", first["guidance"])
            self.assertNotIn("test-key", str(first))
            self.assertNotIn("secret-provider-body", str(first))
            search.assert_awaited_once()

    async def test_asknews_constructor_failure_is_sanitized(self) -> None:
        with (
            patch.dict(os.environ, {"ASKNEWS_API_KEY": "test-key"}, clear=True),
            patch.dict(research_mcp._cooldowns, {}, clear=True),
            patch(
                "forecasting_tools.AskNewsSearcher", side_effect=ValueError("test-key")
            ),
        ):
            result = await research_mcp.asknews_research("test")
        self.assertEqual(result["status"], "unavailable")
        self.assertNotIn("test-key", str(result))

    async def test_payment_and_rate_limit_cooldowns_expire_and_are_independent(
        self,
    ) -> None:
        for status, delay in ((402, 300), (429, 30)):
            with (
                self.subTest(status=status),
                patch.dict(research_mcp._cooldowns, {}, clear=True),
            ):
                call = AsyncMock(
                    side_effect=[
                        ProviderUnavailable(f"Exa returned HTTP {status}", status),
                        {"provider": "exa", "results": ["success"]},
                    ]
                )
                other = AsyncMock(return_value={"provider": "perplexity", "text": "ok"})
                with patch.object(
                    research_mcp.time, "monotonic", return_value=100
                ) as clock:
                    first = await research_mcp._optional_result("exa", call)
                    second = await research_mcp._optional_result("exa", call)
                    alternative = await research_mcp._optional_result(
                        "perplexity", other
                    )
                    self.assertEqual(first, second)
                    self.assertEqual(first["status"], "unavailable")
                    self.assertEqual(alternative["text"], "ok")
                    call.assert_awaited_once()
                    clock.return_value += delay
                    result = await research_mcp._optional_result("exa", call)
                    self.assertEqual(result["results"], ["success"])
                    self.assertEqual(call.await_count, 2)

    async def test_other_failures_do_not_trigger_cooldown(self) -> None:
        with patch.dict(research_mcp._cooldowns, {}, clear=True):
            call = AsyncMock(
                side_effect=[
                    ProviderUnavailable("service unavailable", 503),
                    {"provider": "exa", "results": []},
                ]
            )
            self.assertEqual(
                (await research_mcp._optional_result("exa", call))["status"],
                "unavailable",
            )
            self.assertEqual(
                (await research_mcp._optional_result("exa", call))["results"], []
            )
            self.assertEqual(call.await_count, 2)

    async def test_exa_and_perplexity_http_payment_errors_are_safe(self) -> None:
        for name, tool in (
            ("EXA_API_KEY", "exa_search"),
            ("PERPLEXITY_API_KEY", "perplexity_research"),
        ):
            with (
                self.subTest(tool=tool),
                patch.dict(research_mcp._cooldowns, {}, clear=True),
            ):
                client = httpx.AsyncClient(
                    transport=httpx.MockTransport(
                        lambda request: httpx.Response(
                            402, text="secret-key and provider body"
                        )
                    )
                )
                with (
                    patch.dict(os.environ, {name: "secret-key"}, clear=True),
                    patch.object(research.httpx, "AsyncClient", return_value=client),
                ):
                    output = await mcp.call_tool(tool, {"query": "test"})
                    result = json.loads(cast(list[TextContent], output)[0].text)
                self.assertEqual(result["status"], "unavailable")
                self.assertIn("HTTP 402", result["reason"])
                self.assertNotIn("secret-key", str(result))

    async def test_invalid_provider_arguments_remain_validation_errors(self) -> None:
        with patch.dict(os.environ, {"EXA_API_KEY": "test-key"}, clear=True):
            with self.assertRaises(ValueError):
                await research_mcp.exa_search("query", num_results=11)

    async def test_mcp_catalog_exposes_bounded_provider_tools(self) -> None:
        tools = {tool.name: tool for tool in await mcp.list_tools()}
        self.assertEqual(
            set(tools),
            {
                "available_providers",
                "asknews_research",
                "perplexity_research",
                "exa_search",
                "yahoo_financial_series",
                "fred_financial_series",
                "prediction_market_search",
                "kalshi_market",
            },
        )
        self.assertEqual(
            tools["exa_search"].inputSchema["properties"]["num_results"]["maximum"], 10
        )
        self.assertIn(
            "high", tools["asknews_research"].inputSchema["properties"]["depth"]["enum"]
        )
