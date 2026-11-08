import asyncio
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from metac_bot.infrastructure.opencode_service import (
    _copy_config,
    _verify_search,
    _wait_for_mcp,
    connect_for_run,
)
from metac_bot.infrastructure.paths import ROOT


class FakeProcess:
    def __init__(self):
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(
            b"server listening on http://127.0.0.1:43210\nserver password fake-password\n"
        )
        self.returncode = None

    def terminate(self):
        self.returncode = 0
        self.stdout.feed_eof()

    def kill(self):
        self.terminate()

    async def wait(self):
        return 0


class PrivateServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ready = patch(
            "metac_bot.infrastructure.opencode_service._wait_for_mcp",
            new_callable=AsyncMock,
        )
        self.ready = ready.start()
        self.addCleanup(ready.stop)

    async def test_no_explicit_override_starts_private_server(self):
        client = AsyncMock()
        client.__aenter__.return_value = client
        factory = MagicMock(return_value=client)
        process = FakeProcess()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=process,
            ) as start,
        ):
            async with connect_for_run(
                api_timeout=15,
                startup_timeout=10,
                client_factory=factory,
            ) as actual:
                self.assertIs(actual, client)
                self.assertEqual(factory.call_args.args[0], "http://127.0.0.1:43210")
                start.assert_awaited_once()
                assert start.await_args is not None
                config_path = (
                    Path(start.await_args.kwargs["env"]["XDG_CONFIG_HOME"])
                    / "opencode/opencode.jsonc"
                )
                servers = json.loads(config_path.read_text())["mcp"]["servers"]
                self.assertEqual(set(servers), {"metaculus", "research"})
                config = json.loads(config_path.read_text())
                self.assertIn(
                    str(ROOT / "metac_bot/forecasting/opencode/skills"),
                    config["skills"],
                )
                for name in ("causal-modeler", "economic-modeler"):
                    self.assertEqual(
                        (config_path.parent / "agents" / f"{name}.md").read_bytes(),
                        (
                            ROOT
                            / "metac_bot/forecasting/opencode/agents"
                            / f"{name}.md"
                        ).read_bytes(),
                    )
                self.ready.assert_awaited_once()
            self.assertEqual(process.returncode, 0)

    async def test_mcp_failure_stops_private_server(self):
        self.ready.side_effect = RuntimeError("MCP failed")
        process = FakeProcess()
        client = AsyncMock()
        client.__aenter__.return_value = client
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=process,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "MCP failed"):
                async with connect_for_run(
                    api_timeout=15,
                    startup_timeout=10,
                    client_factory=MagicMock(return_value=client),
                ):
                    self.fail("must not start a forecast")
        self.assertEqual(process.returncode, 0)

    async def test_in_memory_database_still_starts_private_server(self):
        process = FakeProcess()
        client = AsyncMock()
        client.__aenter__.return_value = client
        with (
            patch.dict(os.environ, {"OPENCODE_DB": ":memory:"}, clear=True),
            patch(
                "asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=process,
            ) as start,
        ):
            async with connect_for_run(
                api_timeout=15,
                startup_timeout=10,
                client_factory=MagicMock(return_value=client),
            ):
                assert start.await_args is not None
                self.assertEqual(
                    start.await_args.kwargs["env"]["OPENCODE_DB"], ":memory:"
                )

    async def test_explicit_key_uses_isolated_database_and_does_not_change_user_config(
        self,
    ):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "user-config/opencode/opencode.jsonc"
            original.parent.mkdir(parents=True)
            original.write_text(
                '{ // preserved\n "websearch": {"provider": "exa"}, "model": "openrouter/user" }'
            )
            original_text = original.read_text()
            process = FakeProcess()
            client = AsyncMock()
            client.__aenter__.return_value = client
            client.request.side_effect = [
                {"data": {"active": True}},
                [{"info": {"websearch": {"provider": "tavily"}}}],
            ]
            factory = MagicMock(return_value=client)
            env = {
                "OPENROUTER_API_KEY": "test-private-key",
                "OPENCODE_SEARCH_PROVIDER": "tavily",
                "METAC_BOT_HOST_CONFIG_HOME": str(root / "user-config"),
                "METAC_BOT_HOST_DATA_HOME": str(root / "user-data"),
            }
            with (
                patch.dict(os.environ, env, clear=True),
                patch(
                    "asyncio.create_subprocess_exec",
                    new_callable=AsyncMock,
                    return_value=process,
                ) as start,
            ):
                async with connect_for_run(
                    api_timeout=15, startup_timeout=10, client_factory=factory
                ):
                    assert start.await_args is not None
                    process_env = start.await_args.kwargs["env"]
                    self.assertEqual(process_env["OPENCODE_DB"], ":memory:")
                    self.assertNotEqual(
                        process_env["XDG_DATA_HOME"], str(root / "user-data")
                    )
                    self.assertNotIn("test-private-key", repr(start.await_args.args))
                    private_config = (
                        Path(process_env["XDG_CONFIG_HOME"]) / "opencode/opencode.jsonc"
                    )
                    self.assertEqual(
                        json.loads(private_config.read_text())["websearch"],
                        {"provider": "tavily"},
                    )
                    self.assertEqual(
                        json.loads(private_config.read_text())["model"],
                        "openrouter/user",
                    )
                self.assertEqual(original.read_text(), original_text)
                self.assertFalse(private_config.exists())
                first = client.request.await_args_list[0]
                self.assertEqual(first.args[:2], ("POST", "/api/credential"))
                self.assertEqual(
                    first.kwargs["json"]["value"]["key"], "test-private-key"
                )
                self.assertEqual(process.returncode, 0)

    async def test_search_only_reuses_saved_account_data_without_creating_credentials(
        self,
    ):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            process = FakeProcess()
            client = AsyncMock()
            client.__aenter__.return_value = client
            client.request.return_value = [{"info": {"websearch": {"provider": "exa"}}}]
            env = {
                "OPENCODE_SEARCH_PROVIDER": "exa",
                "METAC_BOT_HOST_CONFIG_HOME": str(root / "config"),
                "METAC_BOT_HOST_DATA_HOME": str(root / "user-data"),
            }
            with (
                patch.dict(os.environ, env, clear=True),
                patch(
                    "asyncio.create_subprocess_exec",
                    new_callable=AsyncMock,
                    return_value=process,
                ) as start,
            ):
                async with connect_for_run(
                    api_timeout=15,
                    startup_timeout=10,
                    client_factory=MagicMock(return_value=client),
                ):
                    assert start.await_args is not None
                    process_env = start.await_args.kwargs["env"]
                    self.assertEqual(
                        process_env["XDG_DATA_HOME"], str(root / "user-data")
                    )
                    self.assertNotIn("OPENCODE_DB", process_env)
                self.assertEqual(client.request.await_args.args[0], "GET")
                self.assertEqual(client.request.await_count, 1)

    async def test_research_environment_uses_private_server_with_existing_accounts(
        self,
    ):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            process = FakeProcess()
            client = AsyncMock()
            client.__aenter__.return_value = client
            env = {
                "EXA_API_KEY": "test-key",
                "METAC_BOT_HOST_CONFIG_HOME": str(root / "config"),
                "METAC_BOT_HOST_DATA_HOME": str(root / "user-data"),
            }
            with (
                patch.dict(os.environ, env, clear=True),
                patch(
                    "asyncio.create_subprocess_exec",
                    new_callable=AsyncMock,
                    return_value=process,
                ) as start,
            ):
                async with connect_for_run(
                    api_timeout=15,
                    startup_timeout=10,
                    client_factory=MagicMock(return_value=client),
                ):
                    assert start.await_args is not None
                    self.assertEqual(
                        start.await_args.kwargs["env"]["EXA_API_KEY"], "test-key"
                    )
                    self.assertEqual(
                        start.await_args.kwargs["env"]["XDG_DATA_HOME"],
                        str(root / "user-data"),
                    )
                client.request.assert_not_awaited()

    async def test_search_override_fails_if_higher_precedence_config_conflicts(self):
        client = AsyncMock()
        client.request.return_value = [
            {"info": {"websearch": {"provider": "exa"}}},
            {"info": {"websearch": {"provider": "tavily"}}},
        ]
        with self.assertRaisesRegex(RuntimeError, "prevents the requested"):
            await _verify_search(client, "exa")

    async def test_explicit_remote_server_is_not_used(self):
        env = {
            "OPENCODE_SERVER_URL": "https://example.invalid",
            "OPENCODE_SERVER_PASSWORD": "example-secret",
        }
        process = FakeProcess()
        client = AsyncMock()
        client.__aenter__.return_value = client
        with (
            patch.dict(os.environ, env, clear=True),
            patch(
                "asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=process,
            ) as start,
        ):
            async with connect_for_run(
                api_timeout=15,
                startup_timeout=10,
                client_factory=MagicMock(return_value=client),
            ):
                assert start.await_args is not None
                server_env = start.await_args.kwargs["env"]
                self.assertNotIn("OPENCODE_SERVER_URL", server_env)
                self.assertNotIn("OPENCODE_SERVER_PASSWORD", server_env)
            self.assertEqual(process.returncode, 0)

    async def test_wait_for_mcp_requires_both_connected(self):
        client = AsyncMock()
        client.request.side_effect = [
            {"data": []},
            {
                "data": [
                    {"name": "metaculus", "status": {"status": "connected"}},
                    {"name": "research", "status": {"status": "pending"}},
                ]
            },
            {
                "data": [
                    {"name": name, "status": {"status": "connected"}}
                    for name in ("metaculus", "research")
                ]
            },
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _wait_for_mcp(client, 5)
        self.assertEqual(client.request.await_count, 3)

    async def test_wait_for_mcp_fails_on_connection_error(self):
        client = AsyncMock()
        client.request.return_value = {
            "data": [{"name": "metaculus", "status": {"status": "failed"}}]
        }
        with self.assertRaisesRegex(RuntimeError, "metaculus"):
            await _wait_for_mcp(client, 5)

    async def test_wait_for_mcp_times_out(self):
        client = AsyncMock()
        client.request.return_value = {"data": []}
        with self.assertRaisesRegex(TimeoutError, "did not connect"):
            await _wait_for_mcp(client, 0.01)

    def test_copy_config_dereferences_symlink_before_override(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original/opencode"
            source.mkdir(parents=True)
            real = root / "shared-config.jsonc"
            real.write_text('{"websearch":{"provider":"exa"},"model":"provider/model"}')
            (source / "opencode.jsonc").symlink_to(real)
            _copy_config(root / "original", root / "copy", "tavily")
            copied = root / "copy/opencode/opencode.jsonc"
            self.assertFalse(copied.is_symlink())
            self.assertEqual(
                json.loads(copied.read_text())["websearch"]["provider"], "tavily"
            )
            self.assertEqual(
                json.loads(real.read_text())["websearch"]["provider"], "exa"
            )

    def test_copy_config_preserves_user_agents_and_skill_sources(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original/opencode"
            (source / "agents").mkdir(parents=True)
            (source / "agents/custom.md").write_text("User agent")
            (source / "agents/causal-modeler.md").write_text("User version")
            (source / "opencode.jsonc").write_text('{"skills": ["/user/skills"]}')
            _copy_config(root / "original", root / "copy", None)
            copied = root / "copy/opencode"
            self.assertEqual((copied / "agents/custom.md").read_text(), "User agent")
            self.assertEqual(
                (source / "agents/causal-modeler.md").read_text(), "User version"
            )
            self.assertEqual(
                (copied / "agents/causal-modeler.md").read_bytes(),
                (
                    ROOT / "metac_bot/forecasting/opencode/agents/causal-modeler.md"
                ).read_bytes(),
            )
            self.assertEqual(
                json.loads((copied / "opencode.jsonc").read_text())["skills"],
                ["/user/skills", str(ROOT / "metac_bot/forecasting/opencode/skills")],
            )
