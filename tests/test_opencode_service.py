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
    connect_for_run,
)


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
    async def test_no_explicit_override_uses_shared_service(self):
        client = AsyncMock()
        client.__aenter__.return_value = client
        factory = MagicMock(return_value=client)
        discover = MagicMock(return_value=("http://shared", None))
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("asyncio.create_subprocess_exec") as start,
        ):
            async with connect_for_run(
                api_timeout=15,
                startup_timeout=10,
                discover=discover,
                client_factory=factory,
            ) as actual:
                self.assertIs(actual, client)
            discover.assert_called_once()
            factory.assert_called_once_with("http://shared", None, timeout=15)
            start.assert_not_called()

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

    async def test_explicit_remote_server_and_override_fail_closed(self):
        env = {
            "OPENCODE_SERVER_URL": "https://example.invalid",
            "OPENROUTER_API_KEY": "test-key",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch("asyncio.create_subprocess_exec") as start,
        ):
            with self.assertRaisesRegex(ValueError, "cannot be applied"):
                async with connect_for_run(api_timeout=15, startup_timeout=10):
                    self.fail("must not connect")
            start.assert_not_called()

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
