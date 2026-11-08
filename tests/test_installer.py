"""Exercise installer decisions without changing the user's containers."""

import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.log = self.base / "calls.log"
        self.bin = self.base / "bin"
        self.bin.mkdir()
        self.executable(
            "uv",
            """#!/bin/bash
printf 'uv %s\\n' "$*" >> "$FAKE_LOG"
""",
        )
        self.executable(
            "proot-distro",
            """#!/bin/bash
printf 'proot-distro %s\\n' "$*" >> "$FAKE_LOG"
if [[ "$1" == list ]]; then
  if [[ "$FAKE_CONTAINER" == present ]]; then printf 'debian\\n'; fi
  exit 0
fi
if [[ "$1" == login && "$*" == *'test -x /usr/bin/python3'* ]]; then
  [[ "$FAKE_PYTHON" != missing ]]
  exit $?
fi
if [[ "$1" == login && "$*" == *'/usr/bin/python3 -c'* ]]; then
  [[ "$FAKE_PYTHON" != incompatible ]]
  exit $?
fi
exit 0
""",
        )
        self.stubs = self.base / "stubs"
        self.stubs.mkdir()
        for name in ("uv", "proot-distro"):
            shutil.copy2(self.bin / name, self.stubs / name)
        self.executable(
            "pkg",
            """#!/bin/bash
printf 'pkg %s\\n' "$*" >> "$FAKE_LOG"
if [[ "${FAKE_PKG_FAIL:-}" == 1 ]]; then exit 1; fi
shift 2
for package in "$@"; do
  cp "$FAKE_STUBS/$package" "$FAKE_BIN/$package"
  chmod +x "$FAKE_BIN/$package"
done
""",
        )

    def executable(self, name, text):
        path = self.bin / name
        path.write_text(text)
        path.chmod(0o755)

    def run_installer(
        self, *, termux=True, container="present", python="present", args=()
    ):
        env = {
            **os.environ,
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "FAKE_LOG": str(self.log),
            "FAKE_STUBS": str(self.stubs),
            "FAKE_BIN": str(self.bin),
            "FAKE_CONTAINER": container,
            "FAKE_PYTHON": python,
            "TERMUX_VERSION": "0.118" if termux else "",
            "METAC_BOT_VENV": "/root/.venvs/custom",
        }
        env.pop("PD_CONTAINER", None)
        return subprocess.run(
            ["bash", str(ROOT / "scripts/install.sh"), *args],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def calls(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def test_fresh_termux_creates_container_and_installs_python(self):
        result = self.run_installer(container="missing", python="missing")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertIn("proot-distro install debian:trixie", calls)
        self.assertIn("proot-distro login debian -- apt-get update", calls)
        self.assertTrue(
            any(
                "apt-get install -y python3 python3-venv ca-certificates" in call
                for call in calls
            )
        )
        self.assertTrue(
            any(
                "UV_PROJECT_ENVIRONMENT=/root/.venvs/custom" in call
                and "uv sync --locked --inexact --python /usr/bin/python3" in call
                for call in calls
            )
        )

    def test_existing_termux_container_is_reused_without_apt(self):
        result = self.run_installer(args=("--all-groups",))
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertFalse(
            any(" install debian" in call or "apt-get" in call for call in calls)
        )
        self.assertFalse(any(call.startswith("pkg ") for call in calls))
        self.assertTrue(
            any(
                "uv sync --locked --inexact" in call and "--all-groups" in call
                for call in calls
            )
        )

    def test_incompatible_existing_python_is_not_replaced(self):
        result = self.run_installer(python="incompatible")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("needs Python 3.11", result.stderr)
        self.assertFalse(
            any("apt-get" in call or "uv sync" in call for call in self.calls())
        )

    def test_linux_uses_same_command_without_proot(self):
        result = self.run_installer(termux=False, args=("--all-groups",))
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].startswith("uv sync --locked --inexact --project "))
        self.assertIn("--all-groups", calls[0])

    def test_termux_installs_missing_uv_and_proot(self):
        (self.bin / "uv").unlink()
        (self.bin / "proot-distro").unlink()
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertIn("pkg install -y uv proot-distro", calls)
        self.assertTrue(any("uv sync --locked --inexact" in call for call in calls))

    def test_termux_installs_only_the_missing_tool(self):
        (self.bin / "uv").unlink()
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("pkg install -y uv", self.calls())
        self.assertNotIn("pkg install -y uv proot-distro", self.calls())

    def test_termux_installs_missing_proot_only(self):
        (self.bin / "proot-distro").unlink()
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("pkg install -y proot-distro", self.calls())

    def test_termux_reports_missing_package_manager(self):
        (self.bin / "uv").unlink()
        (self.bin / "pkg").unlink()
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Termux package manager", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_linux_still_requires_uv(self):
        (self.bin / "uv").unlink()
        result = self.run_installer(termux=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("uv is required on Linux/macOS", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_invalid_arguments_fail_without_changing_environment(self):
        result = self.run_installer(args=("--unexpected",))
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.calls(), [])
