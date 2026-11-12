"""Small V2 HTTP client using the existing OpenCode service and its credentials."""

import asyncio
from pathlib import Path
import time
from typing import Any, Self

import httpx

from .models import model_reference


class OpenCode:
    def __init__(
        self,
        url: str,
        auth: httpx.Auth | tuple[str, str] | None = None,
        *,
        timeout: float = 30,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.http = httpx.AsyncClient(
            base_url=url, auth=auth, timeout=timeout, transport=transport
        )

    async def __aenter__(self) -> Self:
        try:
            info = await self.request("GET", "/api/info")
            if not info.get("version", "").startswith("2."):
                raise RuntimeError("This bot requires OpenCode V2")
        except BaseException:
            await self.http.aclose()
            raise
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.http.aclose()

    async def request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self.http.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else None

    async def create_session(
        self,
        directory: Path,
        title: str,
        model: str | None = None,
        *,
        agent: str | None = None,
    ) -> str:
        body = {"location": {"directory": str(directory)}, "title": title}
        if agent:
            body["agent"] = agent
        if model:
            body["model"] = model_reference(model)
        session = await self.request("POST", "/api/session", json=body)
        return session["data"]["id"]

    async def prompt(self, session_id: str, text: str) -> Any:
        return await self.request(
            "POST", f"/api/session/{session_id}/prompt", json={"text": text}
        )

    async def wait(self, session_id: str, timeout: float = 900) -> None:
        # Long-poll for completion while reporting pending approvals, rather than
        # silently approving them or hanging an unattended Actions job.
        waiter = asyncio.create_task(
            self.request(
                "POST",
                f"/api/experimental/session/{session_id}/wait",
                timeout=timeout,
            )
        )
        deadline = time.monotonic() + timeout
        try:
            while not waiter.done():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Session {session_id} exceeded {timeout} seconds"
                    )
                await asyncio.wait({waiter}, timeout=min(2, remaining, timeout / 2))
                if waiter.done():
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Session {session_id} exceeded {timeout} seconds"
                    )
                try:
                    async with asyncio.timeout(remaining):
                        permissions = await self.request(
                            "GET", f"/api/session/{session_id}/permission"
                        )
                        if permissions.get("data"):
                            raise RuntimeError(
                                f"Session {session_id} needs permission approval in OpenCode"
                            )
                        forms = await self.request(
                            "GET", f"/api/session/{session_id}/form"
                        )
                        if forms.get("data"):
                            raise RuntimeError(
                                f"Session {session_id} needs input in OpenCode"
                            )
                except TimeoutError as exc:
                    raise TimeoutError(
                        f"Session {session_id} exceeded {timeout} seconds"
                    ) from exc
            try:
                await waiter
            except httpx.TimeoutException as exc:
                raise TimeoutError(
                    f"Session {session_id} exceeded {timeout} seconds"
                ) from exc
        except BaseException:
            try:
                async with asyncio.timeout(5):
                    await self.request(
                        "POST", f"/api/session/{session_id}/interrupt", json={}
                    )
            except Exception:
                pass
            raise
        finally:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    async def messages(self, session_id: str) -> Any:
        return await self.request(
            "GET",
            f"/api/session/{session_id}/message",
            params={"order": "desc", "limit": "20"},
        )
