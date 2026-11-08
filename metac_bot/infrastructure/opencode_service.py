"""Connect to the shared service or isolate explicit provider overrides."""

import asyncio
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
from .opencode import OpenCode, discover_service
from .runtime import is_real_env

logger = logging.getLogger(__name__)

SERVICE_ENVIRONMENT = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "EXA_API_KEY",
    "TAVILY_API_KEY",
    "FIRECRAWL_API_KEY",
    "PARALLEL_API_KEY",
    "TINYFISH_API_KEY",
    "ASKNEWS_API_KEY",
    "ASKNEWS_CLIENT_ID",
    "ASKNEWS_SECRET",
    "ASKNEWS_CACHE_MODE",
    "PERPLEXITY_API_KEY",
)


def _copy_config(
    source_home: Path, destination_home: Path, search_provider: str | None
) -> None:
    """Copy the user's config without exposing the original files to writes."""
    source = source_home / "opencode"
    destination = destination_home / "opencode"
    destination_home.mkdir(parents=True)
    if source.is_dir():

        def ignore_service_files(directory, names):
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
    if search_provider is None:
        return
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
    config["websearch"] = {"provider": search_provider}
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


async def _startup(process, timeout: float) -> tuple[str, str]:
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


async def _drain(stream) -> None:
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


@asynccontextmanager
async def connect_for_run(
    *,
    api_timeout: float,
    startup_timeout: float,
    discover=discover_service,
    client_factory=OpenCode,
):
    """Apply explicit overrides privately; otherwise use the user's shared service."""
    settings = ConnectionSettings()
    key = (
        settings.OPENROUTER_API_KEY.get_secret_value()
        if is_real_env("OPENROUTER_API_KEY") and settings.OPENROUTER_API_KEY
        else None
    )
    search = settings.OPENCODE_SEARCH_PROVIDER
    service_environment = any(is_real_env(name) for name in SERVICE_ENVIRONMENT)
    if not key and not search and not service_environment:
        url, auth = discover()
        async with client_factory(url, auth, timeout=api_timeout) as client:
            yield client
        return
    if settings.OPENCODE_SERVER_URL:
        raise ValueError(
            "Explicit provider/search environment settings cannot be applied to OPENCODE_SERVER_URL without modifying that server; use the local service instead"
        )
    if not key and settings.OPENCODE_DB == ":memory:":
        raise ValueError(
            "A private search override cannot reuse a shared in-memory OpenCode database"
        )

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
            logger.info(
                "Using a private OpenCode server for explicit provider/search environment settings"
            )
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
