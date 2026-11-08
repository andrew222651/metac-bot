"""Verify service startup for the Termux launcher without starting a service."""

import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def test_only_top_level_cli_starts_the_service(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            log = base / "calls"
            for name, code in {
                "opencode": """#!/bin/bash
printf 'opencode %s\\n' "$*" >> "$FAKE_LOG"
if [[ "$*" == 'debug paths state' ]]; then printf '/test-state\\n'; fi
""",
                "proot-distro": """#!/bin/bash
printf 'proot-distro %s\\n' "$*" >> "$FAKE_LOG"
""",
            }.items():
                path = base / name
                path.write_text(code)
                path.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{base}:/usr/bin:/bin",
                "FAKE_LOG": str(log),
                "TERMUX_VERSION": "0.118",
            }
            for name in (
                "PD_CONTAINER",
                "OPENCODE_SERVICE_FILE",
                "OPENCODE_SERVER_URL",
            ):
                env.pop(name, None)
            script = ROOT / "scripts/python.sh"
            subprocess.run(
                ["bash", str(script), "-m", "metac_bot", "forecast", "Question?"],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            calls = log.read_text().splitlines()
            self.assertIn("opencode service start", calls)
            self.assertTrue(
                any(
                    "OPENCODE_SERVICE_FILE=/test-state/service.json" in c for c in calls
                )
            )
            log.unlink()
            subprocess.run(
                ["bash", str(script), "-m", "metac_bot.mcp_server"],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertNotIn("opencode service start", log.read_text().splitlines())
