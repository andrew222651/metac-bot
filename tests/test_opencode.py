import asyncio
import json
from pathlib import Path
import unittest

import httpx
from metac_bot.infrastructure.opencode import OpenCode


class OpenCodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_api_timeout_is_configured(self):
        def handle(request):
            self.assertEqual(request.extensions["timeout"]["connect"], 45)
            self.assertEqual(request.extensions["timeout"]["read"], 45)
            return httpx.Response(200, json={"version": "2.0.27"})

        async with OpenCode(
            "http://test", timeout=45, transport=httpx.MockTransport(handle)
        ):
            pass

    async def test_named_agent_is_selected_without_overriding_model_or_permissions(
        self,
    ):
        def handle(request):
            if request.url.path == "/api/info":
                return httpx.Response(200, json={"version": "2.0.27"})
            body = json.loads(request.content)
            self.assertEqual(body["agent"], "causal-modeler")
            self.assertNotIn("model", body)
            self.assertNotIn("permissions", body)
            return httpx.Response(200, json={"data": {"id": "ses_test"}})

        async with OpenCode(
            "http://test", transport=httpx.MockTransport(handle)
        ) as client:
            await client.create_session(
                Path("/project"), "Test", agent="causal-modeler"
            )

    async def test_v2_session_prompt_and_wait_contract(self):
        calls = []

        def handle(request):
            calls.append((request.method, request.url.path))
            if request.url.path == "/api/info":
                return httpx.Response(200, json={"version": "2.0.27"})
            if request.url.path == "/api/session":
                body = json.loads(request.content)
                self.assertEqual(body["location"], {"directory": "/project"})
                self.assertNotIn("permissions", body)
                self.assertNotIn("agent", body)
                self.assertEqual(
                    body["model"],
                    {
                        "providerID": "openrouter",
                        "id": "openai/example",
                        "variant": "high",
                    },
                )
                return httpx.Response(200, json={"data": {"id": "ses_test"}})
            if request.url.path.endswith("/prompt"):
                self.assertEqual(json.loads(request.content), {"text": "Forecast"})
                return httpx.Response(200, json={"data": {"id": "msg_test"}})
            if request.url.path.endswith("/wait"):
                return httpx.Response(204)
            self.fail(f"Unexpected request {request.url}")

        async with OpenCode(
            "http://test", transport=httpx.MockTransport(handle)
        ) as client:
            sid = await client.create_session(
                Path("/project"), "Test", "openrouter/openai/example#high"
            )
            await client.prompt(sid, "Forecast")
            await client.wait(sid, 5)
        self.assertIn(("POST", "/api/experimental/session/ses_test/wait"), calls)

    async def test_timeout_interrupts_without_retrying_prompt(self):
        interrupted = []

        async def handle(request):
            if request.url.path.endswith("/wait"):
                await asyncio.sleep(10)
                return httpx.Response(204)
            if request.url.path.endswith("/permission"):
                return httpx.Response(200, json={"data": []})
            if request.url.path.endswith("/interrupt"):
                interrupted.append(True)
                return httpx.Response(200, json={"interrupted": True})
            return httpx.Response(200, json={"version": "2.0.27"})

        async with OpenCode(
            "http://test", transport=httpx.MockTransport(handle)
        ) as client:
            with self.assertRaises(TimeoutError):
                await client.wait("ses_test", 0.02)
        self.assertEqual(interrupted, [True])

    async def test_session_deadline_bounds_slow_permission_poll(self):
        interrupted = []

        async def handle(request):
            if request.url.path.endswith("/wait") or request.url.path.endswith(
                "/permission"
            ):
                await asyncio.sleep(10)
                return httpx.Response(200, json={"data": []})
            if request.url.path.endswith("/interrupt"):
                interrupted.append(True)
            return httpx.Response(200, json={"version": "2.0.27"})

        async with OpenCode(
            "http://test", timeout=30, transport=httpx.MockTransport(handle)
        ) as client:
            with self.assertRaisesRegex(TimeoutError, "exceeded"):
                await asyncio.wait_for(client.wait("ses_test", 0.02), timeout=1)
        self.assertEqual(interrupted, [True])

    async def test_permission_requests_are_not_autoapproved(self):
        calls = []

        async def handle(request):
            calls.append(request.url.path)
            if request.url.path.endswith("/wait"):
                await asyncio.sleep(10)
                return httpx.Response(204)
            if request.url.path.endswith("/permission"):
                return httpx.Response(200, json={"data": [{"id": "perm_test"}]})
            return httpx.Response(200, json={"version": "2.0.27"})

        async with OpenCode(
            "http://test", transport=httpx.MockTransport(handle)
        ) as client:
            with self.assertRaisesRegex(RuntimeError, "needs permission"):
                await client.wait("ses_test", 0.5)
        self.assertTrue(any(path.endswith("/interrupt") for path in calls))
        self.assertFalse(any(path.endswith("/reply") for path in calls))

    async def test_missing_search_provider_form_is_reported(self):
        async def handle(request):
            if request.url.path.endswith("/wait"):
                await asyncio.sleep(10)
                return httpx.Response(204)
            if request.url.path.endswith("/permission"):
                return httpx.Response(200, json={"data": []})
            if request.url.path.endswith("/form"):
                return httpx.Response(200, json={"data": [{"id": "form_search"}]})
            return httpx.Response(200, json={"version": "2.0.27"})

        async with OpenCode(
            "http://test", transport=httpx.MockTransport(handle)
        ) as client:
            with self.assertRaisesRegex(RuntimeError, "needs input"):
                await client.wait("ses_test", 0.5)
