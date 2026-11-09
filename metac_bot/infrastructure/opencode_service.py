"""Start and stop a private OpenCode server for each forecasting run."""

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlsplit

import httpx
import json5

from .environment import ConnectionSettings
from .paths import ROOT
from .opencode import OpenCode
from .runtime import is_real_env

logger = logging.getLogger(__name__)


def _copy_config(
    source_home: Path, destination_home: Path, search_provider: str | None
) -> None:
    """Copy the user's config without exposing the original files to writes."""
    source = source_home / "opencode"
    destination = destination_home / "opencode"
    destination_home.mkdir(parents=True)
    if source.is_dir():

        def ignore_service_files(directory: str, names: list[str]) -> list[str]:
            return (
                [
                    name
                    for name in names
                    if name.startswith("service") and name.endswith(".json")
                ]
                if Path(directory) == source
                else []
            )

        shutil.copytree(
            source,
            destination,
            symlinks=False,
            ignore_dangling_symlinks=True,
            ignore=ignore_service_files,
        )
    else:
        destination.mkdir()
    path = destination / "opencode.jsonc"
    try:
        config = (
            json5.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
        )
        if not isinstance(config, dict):
            raise ValueError("OpenCode config is not an object")
    except (ValueError, OSError):
        raise RuntimeError(
            "Cannot read the OpenCode configuration for the private server"
        ) from None
    mcp = json5.loads((ROOT / "metac_bot/opencode-mcp.jsonc").read_text())
    config.setdefault("mcp", {}).setdefault("servers", {}).update(mcp["mcp"]["servers"])
    bot_config = ROOT / "metac_bot/forecasting/strategy/opencode"
    shutil.copytree(bot_config / "agents", destination / "agents", dirs_exist_ok=True)
    skills_path = str(bot_config / "skills")
    skills = config.setdefault("skills", [])
    if skills_path not in skills:
        skills.append(skills_path)
    if search_provider is not None:
        config["websearch"] = {"provider": search_provider}
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


async def _startup(
    process: asyncio.subprocess.Process, timeout: float
) -> tuple[str, str]:
    assert process.stdout is not None
    url = password = None
    try:
        async with asyncio.timeout(timeout):
            while not (url and password):
                line = await process.stdout.readline()
                if not line:
                    raise RuntimeError(
                        "Private OpenCode server exited before it was ready"
                    )
                text = line.decode(errors="replace").strip()
                if text.startswith("server listening on "):
                    url = text.removeprefix("server listening on ")
                elif text.startswith("server password "):
                    password = text.removeprefix("server password ")
    except TimeoutError as exc:
        raise TimeoutError(
            f"Private OpenCode startup exceeded {timeout} seconds"
        ) from exc
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in (
        "127.0.0.1",
        "localhost",
        "::1",
    ):
        raise RuntimeError("Private OpenCode server did not bind to loopback")
    return url, password


async def _drain(stream: asyncio.StreamReader) -> None:
    while await stream.read(8192):
        pass


async def _verify_search(client: OpenCode, provider: str) -> None:
    entries = await client.request(
        "GET", "/api/config", params={"location[directory]": str(ROOT)}
    )
    configured = [
        entry["info"]["websearch"]
        for entry in entries
        if isinstance(entry.get("info"), dict) and "websearch" in entry["info"]
    ]
    if not configured or configured[-1] != {"provider": provider}:
        raise RuntimeError(
            "OpenCode project or managed configuration prevents the requested search-provider override"
        )


async def _wait_for_mcp(client: OpenCode, timeout: float) -> None:
    """Wait until the bot's tools are ready before starting an agent session."""
    required = {"metaculus", "research"}
    try:
        async with asyncio.timeout(timeout):
            while True:
                response = await client.request(
                    "GET", "/api/mcp", params={"location[directory]": str(ROOT)}
                )
                statuses = {
                    item["name"]: item["status"]["status"]
                    for item in response.get("data", [])
                    if item.get("name") in required
                }
                failed = {
                    name: status
                    for name, status in statuses.items()
                    if status not in ("connected", "pending")
                }
                if failed:
                    raise RuntimeError(
                        f"Forecast MCP servers failed to connect: {failed}"
                    )
                if all(statuses.get(name) == "connected" for name in required):
                    return
                await asyncio.sleep(0.5)
    except TimeoutError as exc:
        raise TimeoutError(
            "Forecast MCP servers did not connect before the startup deadline"
        ) from exc


@asynccontextmanager
async def connect_for_run(
    *,
    api_timeout: float,
    startup_timeout: float,
    client_factory: Callable[..., OpenCode] = OpenCode,
) -> AsyncGenerator[OpenCode]:
    """Run an authenticated private server and clean it up after the forecast."""
    settings = ConnectionSettings()
    key = (
        settings.OPENROUTER_API_KEY.get_secret_value()
        if is_real_env("OPENROUTER_API_KEY") and settings.OPENROUTER_API_KEY
        else None
    )
    search = settings.OPENCODE_SEARCH_PROVIDER
    config_home = Path(
        settings.METAC_BOT_HOST_CONFIG_HOME
        or settings.XDG_CONFIG_HOME
        or Path.home() / ".config"
    )
    data_home = Path(
        settings.METAC_BOT_HOST_DATA_HOME
        or settings.XDG_DATA_HOME
        or Path.home() / ".local/share"
    )
    with tempfile.TemporaryDirectory(prefix="metac-bot-opencode-") as directory:
        temp = Path(directory)
        _copy_config(config_home, temp / "config", search)
        env = os.environ.copy()
        for name in (
            "OPENCODE_SERVER_URL",
            "OPENCODE_SERVER_USERNAME",
            "OPENCODE_SERVER_PASSWORD",
            "OPENCODE_SERVICE_FILE",
        ):
            env.pop(name, None)
        env["XDG_CONFIG_HOME"] = str(temp / "config")
        env["XDG_STATE_HOME"] = str(temp / "state")
        if key:
            # A separate in-memory database prevents changing the user's active account.
            env["XDG_DATA_HOME"] = str(temp / "data")
            env["OPENCODE_DB"] = ":memory:"
            logger.info(
                "Using an isolated OpenCode server for the explicit OpenRouter key; other saved provider accounts are unavailable"
            )
        else:
            env["XDG_DATA_HOME"] = str(data_home)
            logger.info("Using a private OpenCode server for this run")
        process = await asyncio.create_subprocess_exec(
            "opencode",
            "serve",
            "--hostname",
            "127.0.0.1",
            "--port",
            "0",
            cwd=ROOT,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        drain = None
        try:
            url, password = await _startup(process, startup_timeout)
            assert process.stdout is not None
            drain = asyncio.create_task(_drain(process.stdout))
            async with client_factory(
                url, httpx.BasicAuth("opencode", password), timeout=api_timeout
            ) as client:
                if key:
                    response = await client.request(
                        "POST",
                        "/api/credential",
                        json={
                            "integrationID": "openrouter",
                            "value": {"type": "key", "key": key},
                        },
                    )
                    if not response.get("data", {}).get("active"):
                        raise RuntimeError(
                            "Could not activate the explicit OpenRouter credential on the private server"
                        )
                if search:
                    await _verify_search(client, search)
                await _wait_for_mcp(client, startup_timeout)
                yield client
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=10)
                except TimeoutError:
                    process.kill()
                    await process.wait()
            if drain:
                drain.cancel()
                await asyncio.gather(drain, return_exceptions=True)
