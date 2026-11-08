import io
import os
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, patch

from forecasting_tools import BinaryReport, MultipleChoiceQuestion, MultipleChoiceReport

from metac_bot import runner
from metac_bot.forecasting import jobs
from metac_bot.infrastructure.cli import parse_args
from metac_bot.infrastructure.config import OneOffConfig
from metac_bot.infrastructure.opencode_service import SERVICE_ENVIRONMENT


class OneOffCliTests(unittest.TestCase):
    def test_parse_multiple_choice_and_model(self):
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

    def test_binary_question_requires_no_options(self):
        config = parse_args(["forecast", "Will it happen?", "--single-agent"])
        self.assertIsInstance(config, OneOffConfig)
        assert isinstance(config, OneOffConfig)
        self.assertEqual(config.options, ())
        self.assertTrue(config.single_agent)

    def test_one_off_exposes_ensemble_configuration(self):
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

    def test_rejects_invalid_or_publishing_options(self):
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
    def setUp(self):
        environment = patch.dict(
            os.environ,
            {
                **dict.fromkeys(SERVICE_ENVIRONMENT, ""),
                "OPENROUTER_API_KEY": "",
                "OPENCODE_SEARCH_PROVIDER": "",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

    async def test_one_off_uses_ensemble_without_metaculus_or_publishing(self):
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
            patch.object(
                runner, "discover_service", return_value=("http://test", None)
            ),
            patch.object(runner, "OpenCode", return_value=client) as client_factory,
            patch.object(
                runner, "run_ensemble", new_callable=AsyncMock, return_value=result
            ) as ensemble,
            patch.object(
                MultipleChoiceReport,
                "publish_report_to_metaculus",
                new_callable=AsyncMock,
            ) as publish,
            redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(await runner.run_one_off(config), 0)
            metaculus.assert_not_called()
            check_environment.assert_not_called()
            ensemble.assert_awaited_once()
            client_factory.assert_called_once_with(
                "http://test", None, timeout=config.api_timeout
            )
            publish.assert_not_awaited()
            assert ensemble.await_args is not None
            question = ensemble.await_args.args[1]
            self.assertIsInstance(question, MultipleChoiceQuestion)
            self.assertEqual(question.options, ["PP", "PSOE"])
            self.assertTrue(question.custom_metadata["one_off"])
            self.assertIn("PP: 70.0%", output.getvalue())
            self.assertIn("Research", output.getvalue())

    async def test_one_off_single_agent_saves_without_publishing(self):
        config = OneOffConfig(question="Will it happen?", single_agent=True)
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.create_session.return_value = "ses_test"
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
        ):

            async def record(*_args):
                job_id = next(Path(directory).iterdir()).name
                jobs.save_forecast(
                    job_id, {"probability": 0.6}, "Research", "Rationale"
                )

            client.wait.side_effect = record
            with (
                patch.object(runner, "MetaculusClient") as metaculus,
                patch.object(
                    runner, "discover_service", return_value=("http://test", None)
                ),
                patch.object(runner, "OpenCode", return_value=client),
                patch.object(
                    BinaryReport, "publish_report_to_metaculus", new_callable=AsyncMock
                ) as publish,
                redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(await runner.run_one_off(config), 0)
                metaculus.assert_not_called()
                publish.assert_not_awaited()
                self.assertIn("Yes: 60.0%", output.getvalue())
                self.assertIn(
                    "do not look it up on Metaculus", client.prompt.await_args.args[1]
                )
