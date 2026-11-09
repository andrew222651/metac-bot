"""Verify the code-check workflow remains configured."""

from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]


class ChecksWorkflowTests(unittest.TestCase):
    def test_code_checks_run_on_push(self) -> None:
        workflows = ROOT / ".github/workflows"
        workflow = yaml.safe_load((workflows / "checks.yaml").read_text())
        # PyYAML treats the YAML 1.1 key "on" as True.
        self.assertIn("push", workflow[True])
        steps = workflow["jobs"]["checks"]["steps"]
        commands = [step["run"] for step in steps if "run" in step]
        for command in (
            "ruff format --check",
            "ruff check",
            "ty check",
        ):
            self.assertTrue(any(command in item for item in commands), command)
        self.assertTrue(
            any("unittest discover -s tests -v" in item for item in commands)
        )
        self.assertTrue(
            any("unittest discover -s tests/slow -v" in item for item in commands)
        )
        fast = next(step for step in steps if step.get("name") == "Fast tests")
        slow = next(step for step in steps if step.get("name") == "Slow tests")
        self.assertEqual(fast["id"], "fast_tests")
        self.assertEqual(
            slow["if"], "${{ !cancelled() && steps.fast_tests.outcome != 'skipped' }}"
        )
        # Local discovery must not recurse into the slow suite.
        self.assertFalse((ROOT / "tests/slow/__init__.py").exists())

    def test_forecasts_are_scheduled_and_test_can_be_run_manually(self) -> None:
        workflows = ROOT / ".github/workflows"
        workflow = yaml.safe_load((workflows / "run_bot.yaml").read_text())
        triggers = workflow[True]
        self.assertEqual(triggers["schedule"], [{"cron": "7,27,47 * * * *"}])
        mode = triggers["workflow_dispatch"]["inputs"]["mode"]
        self.assertEqual(mode["default"], "test_questions")
        self.assertEqual(mode["options"], ["test_questions", "tournament"])
        step = next(
            item
            for item in workflow["jobs"]["forecast_job"]["steps"]
            if item.get("name") == "Run bot"
        )
        self.assertEqual(step["uses"], "./.github/actions/run-forecast")
        self.assertEqual(step["with"]["mode"], "${{ inputs.mode || 'tournament' }}")
        self.assertEqual(
            workflow["concurrency"]["group"],
            "${{ github.workflow }}-${{ inputs.mode || 'tournament' }}",
        )
