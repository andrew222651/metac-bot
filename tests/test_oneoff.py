import io
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, patch

from forecasting_tools import MultipleChoiceQuestion

from metac_bot.infrastructure import runner
from metac_bot.forecasting.implementation import jobs
from metac_bot.infrastructure.cli import parse_args
from metac_bot.infrastructure.config import OneOffConfig


class OneOffCliTests(unittest.TestCase):
    def test_parse_multiple_choice_and_model(self) -> None:
        config = parse_args(
            [
                "forecast",
                "Which party wins?",
                "--option",
                "PP",
                "--option",
                "PSOE",
                "--option",
                "Vox",
                "--option",
                "other",
                "--model",
                "openrouter/openai/gpt-6.1-terra#low",
            ]
        )
        self.assertIsInstance(config, OneOffConfig)
        assert isinstance(config, OneOffConfig)
        self.assertEqual(config.options, ("PP", "PSOE", "Vox", "other"))
        self.assertEqual(config.model, "openrouter/openai/gpt-6.1-terra#low")
        self.assertTrue(config.dry_run)
        self.assertFalse(config.single_agent)

    def test_binary_question_requires_no_options(self) -> None:
        config = parse_args(["forecast", "Will it happen?", "--single-agent"])
        self.assertIsInstance(config, OneOffConfig)
        assert isinstance(config, OneOffConfig)
        self.assertEqual(config.options, ())
        self.assertTrue(config.single_agent)

    def test_one_off_exposes_ensemble_configuration(self) -> None:
        config = parse_args(
            [
                "forecast",
                "Which?",
                "--option",
                "A",
                "--option",
                "B",
                "--ensemble-concurrency",
                "3",
                "--ensemble-review-rounds",
                "2",
                "--ensemble-model",
                "skeptic=provider/skeptic",
                "--ensemble-adjudicator-model",
                "provider/judge",
                "--session-timeout",
                "601",
                "--api-timeout",
                "42",
                "--ensemble-proposal-timeout",
                "701",
                "--ensemble-review-timeout",
                "801",
                "--ensemble-decision-timeout",
                "901",
            ]
        )
        self.assertIsInstance(config, OneOffConfig)
        self.assertEqual(config.ensemble_concurrency, 3)
        self.assertEqual(config.ensemble_review_rounds, 2)
        self.assertEqual(config.ensemble_models, {"skeptic": "provider/skeptic"})
        self.assertEqual(config.ensemble_adjudicator_model, "provider/judge")
        self.assertEqual(config.timeout, 601)
        self.assertEqual(config.api_timeout, 42)
        self.assertEqual(config.ensemble_proposal_timeout, 701)
        self.assertEqual(config.ensemble_review_timeout, 801)
        self.assertEqual(config.ensemble_decision_timeout, 901)

    def test_rejects_invalid_or_publishing_options(self) -> None:
        cases = [
            ["forecast", " "],
            ["forecast", "Question", "--option", "only"],
            ["forecast", "Question", "--option", "A", "--option", "a"],
            ["forecast", "Question", "--no-dry-run"],
        ]
        for argv in cases:
            with self.subTest(argv=argv), patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    parse_args(argv)
        with self.assertRaisesRegex(ValueError, "cannot be published"):
            OneOffConfig(question="Question", dry_run=False)


class OneOffRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_off_uses_ensemble_without_metaculus_or_publishing(self) -> None:
        config = OneOffConfig(
            question="Which party?", options=("PP", "PSOE"), model="provider/model"
        )
        client = AsyncMock()
        client.__aenter__.return_value = client
        result = {
            "prediction": {"options": {"PP": 0.7, "PSOE": 0.3}},
            "research": "Research",
            "rationale": "Rationale",
        }
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
            patch.object(runner, "MetaculusClient") as metaculus,
            patch.object(runner, "check_environment") as check_environment,
            patch.object(runner, "connect_for_run", return_value=client) as connect,
            patch.object(
                runner, "run_ensemble", new_callable=AsyncMock, return_value=result
            ) as ensemble,
            patch.object(runner, "publish_prediction") as publish,
            redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(await runner.run_one_off(config), 0)
            metaculus.assert_not_called()
            check_environment.assert_not_called()
            ensemble.assert_awaited_once()
            connect.assert_called_once_with(
                api_timeout=config.api_timeout,
                startup_timeout=config.service_start_timeout,
                client_factory=runner.OpenCode,
            )
            publish.assert_not_called()
            assert ensemble.await_args is not None
            question = ensemble.await_args.args[1]
            self.assertIsInstance(question, MultipleChoiceQuestion)
            self.assertEqual(question.options, ["PP", "PSOE"])
            self.assertTrue(question.custom_metadata["one_off"])
            self.assertIn("PP: 70.0%", output.getvalue())
            self.assertIn("Research", output.getvalue())

    async def test_one_off_single_agent_saves_without_publishing(self) -> None:
        config = OneOffConfig(question="Will it happen?", single_agent=True)
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.create_session.return_value = "ses_test"
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
        ):

            async def record(*_args: object) -> None:
                job_id = next(Path(directory).iterdir()).name
                jobs.save_forecast(
                    job_id, {"probability": 0.6}, "Research", "Rationale"
                )

            client.wait.side_effect = record
            with (
                patch.object(runner, "MetaculusClient") as metaculus,
                patch.object(runner, "connect_for_run", return_value=client),
                patch.object(runner, "publish_prediction") as publish,
                redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(await runner.run_one_off(config), 0)
                metaculus.assert_not_called()
                publish.assert_not_called()
                self.assertIn("Yes: 60.0%", output.getvalue())
                self.assertIn(
                    "do not look it up on Metaculus", client.prompt.await_args.args[1]
                )
