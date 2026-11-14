"""Exercise GitHub workflow configuration without secrets or network access."""

import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

import yaml

from metac_bot.infrastructure.environment import EnvironmentSettings

ROOT = Path(__file__).resolve().parents[1]
ACTION = yaml.safe_load((ROOT / ".github/actions/run-forecast/action.yaml").read_text())


def step_script(workflow: list[dict], name: str) -> str:
    return next(step["run"] for step in workflow if step.get("name") == name)


class CiConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.log = self.base / "calls"
        self.bin = self.base / "bin"
        self.bin.mkdir()
        for name in ("opencode", "uv"):
            path = self.bin / name
            extra = (
                'printf "ENV_TIMEOUT=%s\\n" "${METAC_BOT_SESSION_TIMEOUT:-}" >> "$FAKE_LOG"\n'
                'printf "ENV_CONCURRENCY=%s\\n" "${METAC_BOT_ENSEMBLE_CONCURRENCY:-}" >> "$FAKE_LOG"\n'
                if name == "uv"
                else ""
            )
            path.write_text(
                '#!/bin/bash\nprintf \'%s\\n\' "$@" >> "$FAKE_LOG"\n' + extra
            )
            path.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "XDG_CONFIG_HOME": str(self.base / "config"),
            "FAKE_LOG": str(self.log),
            "OPENCODE_MODEL": "provider/model#low",
            "OPENCODE_SEARCH_PROVIDER": "exa",
        }

    def run_step(self, script: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-e", "-c", script],
            cwd=ROOT,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def test_validation_does_not_touch_opencode_configuration_or_start_shared_service(
        self,
    ) -> None:
        target = self.base / "config/opencode/opencode.json"
        target.parent.mkdir(parents=True)
        target.write_text('{"custom": true}')
        result = self.run_step(
            step_script(ACTION["runs"]["steps"], "Validate forecast settings")
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(target.read_text(), '{"custom": true}')
        self.assertFalse(self.log.exists())

    def test_provider_and_model_are_validated_before_service_start(self) -> None:
        script = step_script(ACTION["runs"]["steps"], "Validate forecast settings")
        self.env["OPENCODE_SEARCH_PROVIDER"] = 'random"},"permissions":[]'
        result = self.run_step(script)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())
        self.env["OPENCODE_SEARCH_PROVIDER"] = "random"
        self.env["OPENCODE_MODEL"] = ""
        self.assertNotEqual(self.run_step(script).returncode, 0)
        self.assertFalse(self.log.exists())

    def test_forecast_values_remain_environment_data_not_shell_code(self) -> None:
        marker = self.base / "injected"
        self.env.update(
            {
                "FORECAST_MODE": "test_questions",
                "METAC_BOT_SESSION_TIMEOUT": f"$(touch {marker})",
                "METAC_BOT_ENSEMBLE_CONCURRENCY": "3",
            }
        )
        result = self.run_step(step_script(ACTION["runs"]["steps"], "Forecast"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(marker.exists())
        args = self.log.read_text().splitlines()
        self.assertIn(f"ENV_TIMEOUT=$(touch {marker})", args)
        self.assertIn("ENV_CONCURRENCY=3", args)
        self.assertNotIn("--timeout", args)

    def test_forecast_uses_defaults_when_environment_values_are_missing(self) -> None:
        self.env["FORECAST_MODE"] = "tournament"
        self.env.pop("METAC_BOT_SESSION_TIMEOUT", None)
        self.env.pop("METAC_BOT_ENSEMBLE_CONCURRENCY", None)
        result = self.run_step(step_script(ACTION["runs"]["steps"], "Forecast"))
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.log.read_text().splitlines()
        self.assertNotIn("--timeout", args)
        self.assertNotIn("--ensemble-concurrency", args)

    def test_forecast_passes_both_modes_to_runner(self) -> None:
        for mode in ("test_questions", "tournament"):
            with self.subTest(mode=mode):
                self.log.unlink(missing_ok=True)
                self.env["FORECAST_MODE"] = mode
                result = self.run_step(step_script(ACTION["runs"]["steps"], "Forecast"))
                self.assertEqual(result.returncode, 0, result.stderr)
                args = self.log.read_text().splitlines()
                self.assertEqual(args[args.index("--mode") + 1], mode)

    def test_schedule_is_enabled_by_default_and_can_be_disabled_without_blocking_dispatch(
        self,
    ) -> None:
        data = yaml.safe_load((ROOT / ".github/workflows/run_bot.yaml").read_text())
        self.assertTrue(data[True]["schedule"])
        self.assertIn("workflow_dispatch", data[True])
        self.assertEqual(
            data["jobs"]["forecast_job"]["if"],
            "${{ github.event_name != 'schedule' || vars.METAC_BOT_SCHEDULE_ENABLED != 'false' }}",
        )

    def test_run_bot_workflow_offers_dispatch_inputs(self) -> None:
        data = yaml.safe_load((ROOT / ".github/workflows/run_bot.yaml").read_text())
        # PyYAML's YAML 1.1 parser interprets the key "on" as True.
        inputs = data[True]["workflow_dispatch"]["inputs"]
        self.assertEqual(
            set(inputs),
            {
                "mode",
                "model",
                "search_provider",
                "session_timeout",
                "ensemble_concurrency",
            },
        )
        run = next(
            step
            for step in data["jobs"]["forecast_job"]["steps"]
            if step.get("name") == "Run bot"
        )
        for key in (
            "OPENCODE_MODEL",
            "OPENCODE_SEARCH_PROVIDER",
            "METAC_BOT_SESSION_TIMEOUT",
            "METAC_BOT_ENSEMBLE_CONCURRENCY",
        ):
            self.assertIn("inputs.", run["env"][key])
        runtime_settings = {
            name
            for name in EnvironmentSettings.model_fields
            if name.startswith("METAC_BOT_")
            and name
            not in {
                "METAC_BOT_VENV",
                "METAC_BOT_HOST_CONFIG_HOME",
                "METAC_BOT_HOST_DATA_HOME",
            }
        }
        self.assertTrue(runtime_settings.issubset(run["env"]))
