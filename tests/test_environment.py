import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from dotenv import dotenv_values

from metac_bot.infrastructure.cli import parse_args
from metac_bot.infrastructure.environment import (
    EnvironmentSettings,
    environment_template,
)


class EnvironmentTests(unittest.TestCase):
    def test_environment_settings_and_cli_precedence(self) -> None:
        env = {
            "OPENCODE_MODEL": "openrouter/env-model#low",
            "METAC_BOT_SESSION_TIMEOUT": "321",
            "METAC_BOT_API_TIMEOUT": "41",
            "METAC_BOT_SERVICE_START_TIMEOUT": "55",
            "METAC_BOT_SINGLE_AGENT": "true",
            "METAC_BOT_ENSEMBLE_CONCURRENCY": "4",
            # Retired settings in old deployments no longer impose a quorum.
            "METAC_BOT_ENSEMBLE_MIN_MEMBERS": "5",
            "METAC_BOT_ENSEMBLE_REVIEW_ROUNDS": "2",
            "METAC_BOT_ENSEMBLE_TOLERANCE": "0.03",
            "METAC_BOT_ENSEMBLE_PROPOSAL_TIMEOUT": "301",
            "METAC_BOT_ENSEMBLE_REVIEW_TIMEOUT": "302",
            "METAC_BOT_ENSEMBLE_DECISION_TIMEOUT": "303",
            "METAC_BOT_ENSEMBLE_MODELS": '{"skeptic":"provider/env","generalist":"provider/base"}',
            "METAC_BOT_ENSEMBLE_ADJUDICATOR_MODEL": "provider/env-judge",
        }
        with patch.dict(os.environ, env, clear=True):
            config = parse_args(["forecast", "Question?"])
            self.assertEqual(config.model, "openrouter/env-model#low")
            self.assertEqual(config.timeout, 321)
            self.assertEqual(config.api_timeout, 41)
            self.assertEqual(config.service_start_timeout, 55)
            self.assertTrue(config.single_agent)
            self.assertEqual(config.ensemble_concurrency, 4)
            self.assertFalse(hasattr(config, "ensemble_min_members"))
            self.assertEqual(config.ensemble_review_rounds, 2)
            self.assertEqual(config.ensemble_tolerance, 0.03)
            self.assertEqual(config.ensemble_proposal_timeout, 301)
            self.assertEqual(config.ensemble_review_timeout, 302)
            self.assertEqual(config.ensemble_decision_timeout, 303)
            self.assertEqual(
                config.ensemble_models,
                {"skeptic": "provider/env", "generalist": "provider/base"},
            )
            self.assertEqual(config.ensemble_adjudicator_model, "provider/env-judge")
            override = parse_args(
                [
                    "forecast",
                    "Question?",
                    "--model",
                    "provider/cli",
                    "--timeout",
                    "123",
                    "--no-single-agent",
                    "--ensemble-model",
                    "skeptic=provider/cli-skeptic",
                ]
            )
            self.assertEqual(override.model, "provider/cli")
            self.assertEqual(override.timeout, 123)
            self.assertFalse(override.single_agent)
            self.assertEqual(
                override.ensemble_models,
                {"skeptic": "provider/cli-skeptic", "generalist": "provider/base"},
            )

    def test_help_lists_every_registered_name_without_showing_secrets(self) -> None:
        root = Path(__file__).resolve().parents[1]
        env = {**os.environ, "OPENROUTER_API_KEY": "never-print-this-test-secret"}
        for args in (["--help"], ["forecast", "--help"]):
            with self.subTest(args=args):
                result = subprocess.run(
                    [sys.executable, "-m", "metac_bot", *args],
                    cwd=root,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                )
                self.assertNotIn(
                    env["OPENROUTER_API_KEY"], result.stdout + result.stderr
                )
                for name in EnvironmentSettings.model_fields:
                    self.assertIn(name, result.stdout)

    def test_generated_template_covers_settings_without_using_environment_values(
        self,
    ) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "never-print-this-test-secret",
                "METAC_BOT_SESSION_TIMEOUT": "invalid",
            },
            clear=True,
        ):
            template = environment_template()
        self.assertNotIn("never-print-this-test-secret", template)
        self.assertNotIn("invalid", template)
        self.assertEqual(dotenv_values(stream=io.StringIO(template)), {})
        for name, field in EnvironmentSettings.model_fields.items():
            with self.subTest(name=name):
                self.assertEqual(
                    sum(
                        line.startswith(f"# {name}=") for line in template.splitlines()
                    ),
                    1,
                )
                if field.description:
                    self.assertIn(field.description, template)
        self.assertIn("# METAC_BOT_SESSION_TIMEOUT=900", template)
        self.assertIn("# METAC_BOT_SINGLE_AGENT=false", template)
        self.assertIn("# METACULUS_TOKEN=", template)

    def test_generate_env_command_prints_template_and_exits(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "metac_bot", "generate-env"],
            cwd=root,
            env={
                **os.environ,
                "OPENROUTER_API_KEY": "never-print-this-test-secret",
                "METAC_BOT_SESSION_TIMEOUT": "invalid",
            },
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        self.assertEqual(result.stdout, environment_template())
        self.assertNotIn("never-print-this-test-secret", result.stdout + result.stderr)

    def test_explicit_cli_value_overrides_an_invalid_environment_value(self) -> None:
        with patch.dict(
            os.environ, {"METAC_BOT_SESSION_TIMEOUT": "invalid"}, clear=True
        ):
            config = parse_args(["forecast", "Question?", "--timeout", "120"])
            self.assertEqual(config.timeout, 120)

    def test_empty_ensemble_model_option_clears_invalid_environment_mapping(
        self,
    ) -> None:
        with patch.dict(
            os.environ, {"METAC_BOT_ENSEMBLE_MODELS": "not-json"}, clear=True
        ):
            config = parse_args(["forecast", "Question?", "--empty-ensemble-model"])
            self.assertEqual(config.ensemble_models, {})

    def test_settings_repr_hides_secret_values(self) -> None:
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "never-print-this-test-secret"},
            clear=True,
        ):
            settings = EnvironmentSettings()
            self.assertNotIn("never-print-this-test-secret", repr(settings))
