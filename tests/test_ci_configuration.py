"""Exercise GitHub workflow configuration without secrets or network access."""

import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

import yaml

from metac_bot.infrastructure.environment import EnvironmentSettings

ROOT = Path(__file__).resolve().parents[1]
ACTION = yaml.safe_load((ROOT / ".github/actions/run-forecast/action.yaml").read_text())
REVIEW = yaml.safe_load((ROOT / ".github/workflows/review_bot.yaml").read_text())


def step_script(workflow, name):
    return next(step["run"] for step in workflow if step.get("name") == name)


class CiConfigurationTests(unittest.TestCase):
    def setUp(self):
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

    def run_step(self, script):
        return subprocess.run(
            ["bash", "-e", "-c", script],
            cwd=ROOT,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def test_ci_template_keeps_provider_as_environment_substitution(self):
        template = json.loads((ROOT / ".github/opencode-ci.jsonc").read_text())
        self.assertEqual(
            template["websearch"]["provider"], "{env:OPENCODE_SEARCH_PROVIDER}"
        )
        self.assertNotIn("model", template)
        result = self.run_step(step_script(ACTION["runs"]["steps"], "Start OpenCode"))
        self.assertEqual(result.returncode, 0, result.stderr)
        target = self.base / "config/opencode/opencode.jsonc"
        self.assertEqual(json.loads(target.read_text()), template)
        self.assertEqual(
            self.log.read_text().splitlines(), ["service", "start", "mcp", "list"]
        )

    def test_existing_config_is_not_overwritten(self):
        target = self.base / "config/opencode/opencode.json"
        target.parent.mkdir(parents=True)
        target.write_text('{"custom": true}')
        result = self.run_step(step_script(ACTION["runs"]["steps"], "Start OpenCode"))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(target.read_text(), '{"custom": true}')
        self.assertFalse(self.log.exists())

    def test_provider_and_model_are_validated_before_service_start(self):
        script = step_script(ACTION["runs"]["steps"], "Start OpenCode")
        self.env["OPENCODE_SEARCH_PROVIDER"] = 'random"},"permissions":[]'
        result = self.run_step(script)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())
        self.env["OPENCODE_SEARCH_PROVIDER"] = "random"
        self.env["OPENCODE_MODEL"] = ""
        self.assertNotEqual(self.run_step(script).returncode, 0)
        self.assertFalse(self.log.exists())

    def test_forecast_values_remain_environment_data_not_shell_code(self):
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

    def test_forecast_uses_defaults_when_environment_values_are_missing(self):
        self.env["FORECAST_MODE"] = "tournament"
        self.env.pop("METAC_BOT_SESSION_TIMEOUT", None)
        self.env.pop("METAC_BOT_ENSEMBLE_CONCURRENCY", None)
        result = self.run_step(step_script(ACTION["runs"]["steps"], "Forecast"))
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.log.read_text().splitlines()
        self.assertNotIn("--timeout", args)
        self.assertNotIn("--ensemble-concurrency", args)

    def test_review_days_rejects_shell_syntax(self):
        marker = self.base / "injected"
        self.env["REVIEW_DAYS"] = f"$(touch {marker})"
        script = step_script(REVIEW["jobs"]["review_job"]["steps"], "Review forecasts")
        result = self.run_step(script)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(marker.exists())
        self.assertFalse(self.log.exists())

    def test_review_days_is_passed_as_a_literal_argument(self):
        self.env["REVIEW_DAYS"] = "45"
        script = step_script(REVIEW["jobs"]["review_job"]["steps"], "Review forecasts")
        result = self.run_step(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.log.read_text().splitlines()
        self.assertEqual(args[args.index("--resolved-since") + 1], "45")

    def test_all_forecasting_workflows_offer_dispatch_inputs(self):
        for name in (
            "test_bot.yaml",
            "run_bot_on_tournament.yaml",
            "run_bot_on_metaculus_cup.yaml",
        ):
            data = yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
            # PyYAML's YAML 1.1 parser interprets the key "on" as True.
            inputs = data[True]["workflow_dispatch"]["inputs"]
            self.assertEqual(
                set(inputs),
                {"model", "search_provider", "session_timeout", "ensemble_concurrency"},
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
