"""Exercise both MCP servers over their real stdio transports."""

import asyncio
import sys
import unittest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from metac_bot.infrastructure.paths import ROOT


class McpEntrypointTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_mcp_launch_commands_expose_tools(self) -> None:
        servers = (
            (
                "metac_bot.mcp_server",
                {
                    "get_forecast_job",
                    "record_forecast",
                    "get_bayesian_stage",
                    "record_bayesian_stage",
                },
            ),
            (
                "metac_bot.research_mcp",
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
            ),
        )

        async def list_tools(module: str) -> set[str]:
            params = StdioServerParameters(
                command=sys.executable, args=["-m", module], cwd=ROOT
            )
            async with asyncio.timeout(60):
                async with stdio_client(params) as (reader, writer):
                    async with ClientSession(reader, writer) as session:
                        await session.initialize()
                        result = await session.list_tools()
            return {tool.name for tool in result.tools}

        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(list_tools(module)) for module, _ in servers]
        for (module, expected), task in zip(servers, tasks):
            with self.subTest(module=module):
                self.assertEqual(task.result(), expected)
