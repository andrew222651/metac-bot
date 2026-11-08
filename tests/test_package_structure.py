import asyncio
import sys
import unittest
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from metac_bot.forecasting import jobs
from metac_bot.forecasting.prompts import forecast_prompt, stage_prompt
from metac_bot.forecasting.roles import MODEL_ROLES, ROLES
from metac_bot.infrastructure.paths import ROOT


class PackageStructureTests(unittest.TestCase):
    def test_project_paths_are_independent_of_module_depth(self):
        self.assertEqual(ROOT, Path(__file__).resolve().parents[1])
        self.assertEqual(jobs.ROOT, ROOT)
        self.assertEqual(jobs.JOBS, ROOT / "forecast_runs")

    def test_prompts_preserve_role_and_standalone_instructions(self):
        self.assertEqual(set(MODEL_ROLES), {"economic-modeler", "causal-modeler"})
        self.assertIn(
            ROLES["economic-modeler"],
            stage_prompt("proposal", "economic-modeler", "stage", one_off=False),
        )
        self.assertIn(
            "do not redo the initial research",
            stage_prompt("review", "skeptic", "stage", one_off=False),
        )
        self.assertNotIn(
            "standalone",
            stage_prompt("decision", "adjudicator", "stage", one_off=False),
        )
        self.assertIn(
            "standalone user-supplied",
            stage_prompt("decision", "adjudicator", "stage", one_off=True),
        )
        self.assertIn("job job-id", forecast_prompt("job-id", one_off=False))
        self.assertNotIn("standalone", forecast_prompt("job-id", one_off=False))
        self.assertIn(
            "do not look it up on Metaculus", forecast_prompt("job-id", one_off=True)
        )


class McpEntrypointTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_mcp_launch_commands_expose_tools(self):
        for module, expected in (
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
                },
            ),
        ):
            with self.subTest(module=module):
                params = StdioServerParameters(
                    command=sys.executable, args=["-m", module], cwd=ROOT
                )
                async with asyncio.timeout(60):
                    async with stdio_client(params) as (reader, writer):
                        async with ClientSession(reader, writer) as session:
                            await session.initialize()
                            result = await session.list_tools()
                self.assertEqual({tool.name for tool in result.tools}, expected)
