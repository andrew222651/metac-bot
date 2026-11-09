import os
from typing import Any, cast
import unittest
from unittest.mock import patch

from metac_bot.infrastructure.cli import parse_args
from metac_bot.infrastructure.config import RunConfig


class ConfigTests(unittest.TestCase):
    def test_defaults_and_role_models(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = parse_args(
                [
                    "--dry-run",
                    "--ensemble-model",
                    "skeptic=openrouter/openai/example#high",
                ]
            )
        self.assertTrue(config.dry_run)
        self.assertEqual(
            config.ensemble_models, {"skeptic": "openrouter/openai/example#high"}
        )
        self.assertEqual(config.ensemble_min_members, 3)
        self.assertEqual(config.ensemble_review_rounds, 3)

    def test_cli_defaults_match_run_config(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(parse_args([]), RunConfig())

    def test_model_defaults_to_environment(self) -> None:
        with patch.dict(os.environ, {"OPENCODE_MODEL": "openai/example"}, clear=True):
            self.assertEqual(parse_args([]).model, "openai/example")
            self.assertEqual(
                parse_args(["--model", "other/model"]).model, "other/model"
            )
        with patch.dict(os.environ, {"OPENCODE_MODEL": ""}, clear=True):
            self.assertIsNone(parse_args([]).model)

    def test_invalid_options_fail_before_running(self) -> None:
        for argv in [
            ["--mode", "metaculus_cup"],
            ["--timeout", "nan"],
            ["--timeout", "inf"],
            ["--api-timeout", "0"],
            ["--ensemble-review-timeout", "-1"],
            ["--limit", "0"],
            ["--model", "/model"],
            ["--ensemble-adjudicator-model", "bad"],
            ["--ensemble-model", "skeptic=provider/"],
            ["--ensemble-model", "bad=provider/model"],
            [
                "--ensemble-model",
                "skeptic=provider/model",
                "--ensemble-model",
                "skeptic=other/model",
            ],
        ]:
            with self.subTest(argv=argv), patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    parse_args(argv)

    def test_help_exits_without_starting_a_run(self) -> None:
        with self.assertRaises(SystemExit) as exc:
            parse_args(["--help"])
        self.assertEqual(exc.exception.code, 0)

    def test_session_and_stage_timeout_options(self) -> None:
        config = parse_args(
            [
                "--session-timeout",
                "600",
                "--api-timeout",
                "45",
                "--ensemble-proposal-timeout",
                "700",
                "--ensemble-review-timeout",
                "800",
                "--ensemble-decision-timeout",
                "900",
            ]
        )
        self.assertEqual(config.timeout, 600)
        self.assertEqual(config.api_timeout, 45)
        self.assertEqual(config.ensemble_proposal_timeout, 700)
        self.assertEqual(config.ensemble_review_timeout, 800)
        self.assertEqual(config.ensemble_decision_timeout, 900)

    def test_repeated_member_models_are_preserved(self) -> None:
        config = parse_args(
            [
                "--ensemble-model",
                "skeptic=provider/a",
                "--ensemble-model",
                "base-rate=provider/b",
            ]
        )
        self.assertEqual(
            config.ensemble_models, {"skeptic": "provider/a", "base-rate": "provider/b"}
        )

    def test_shared_options_match_between_commands(self) -> None:
        options = [
            "--model",
            "provider/default",
            "--session-timeout",
            "600",
            "--api-timeout",
            "40",
            "--single-agent",
            "--ensemble-concurrency",
            "3",
            "--ensemble-min-members",
            "4",
            "--ensemble-review-rounds",
            "2",
            "--ensemble-tolerance",
            "0.02",
            "--ensemble-proposal-timeout",
            "500",
            "--ensemble-review-timeout",
            "700",
            "--ensemble-decision-timeout",
            "800",
            "--ensemble-model",
            "skeptic=provider/other",
            "--ensemble-adjudicator-model",
            "provider/judge",
        ]
        tournament = parse_args(["--dry-run", *options])
        one_off = parse_args(["forecast", "Question?", *options])
        for name, expected in vars(tournament).items():
            with self.subTest(option=name):
                self.assertEqual(getattr(one_off, name), expected)

    def test_programmatic_config_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid tournament mode"):
            RunConfig(mode=cast(Any, "metaculus_cup"))
        with self.assertRaises(ValueError):
            RunConfig(ensemble_min_members=7)
        self.assertEqual(RunConfig(ensemble_min_members=6).ensemble_min_members, 6)
        with self.assertRaises(ValueError):
            RunConfig(ensemble_models={"quantitative": "provider/model"})
        with self.assertRaises(ValueError):
            RunConfig(ensemble_models={"unknown": "provider/model"})
