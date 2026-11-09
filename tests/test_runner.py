from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from forecasting_tools import BinaryQuestion, BinaryReport, NumericQuestion
from forecasting_tools.data_models.data_organizer import DataOrganizer
from test_bayesian import FakeOpenCode
from metac_bot.infrastructure import runner as main
from metac_bot.forecasting.implementation import jobs
from metac_bot.infrastructure.config import RunConfig
from metac_bot.forecasting.strategy.principles import SUPERFORECASTING_PRINCIPLES


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_numeric_default_ensemble_reaches_publication(self) -> None:
        question = NumericQuestion(
            question_text="How many?",
            lower_bound=0,
            upper_bound=100,
            open_lower_bound=False,
            open_upper_bound=False,
        )
        report_type = DataOrganizer.get_report_type_for_question_type(type(question))
        client = FakeOpenCode()
        metaculus = MagicMock()
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
            patch.object(
                report_type, "publish_report_to_metaculus", new_callable=AsyncMock
            ) as publish,
        ):
            result = await main.forecast_question(
                cast(main.OpenCode, client), metaculus, question, RunConfig()
            )
            self.assertNotIsInstance(result, Exception)
            publish.assert_awaited_once()
            self.assertEqual(len(client.sessions), 14)

    async def test_environment_failure_returns_without_starting_opencode(self) -> None:
        with (
            patch.object(main, "check_environment", side_effect=SystemExit(1)),
            patch.object(main, "connect_for_run") as connect,
        ):
            self.assertEqual(await main.run(RunConfig()), 1)
            connect.assert_not_called()

    async def test_tournament_selection_skips_previous_and_deduplicates(self) -> None:
        seen = BinaryQuestion(
            question_text="Already forecast",
            id_of_post=1,
            id_of_question=1,
            already_forecasted=True,
        )
        new = BinaryQuestion(
            question_text="New",
            id_of_post=2,
            id_of_question=2,
            already_forecasted=False,
        )
        client = MagicMock()
        client.CURRENT_AI_COMPETITION_ID = 1
        client.CURRENT_MINIBENCH_ID = 2
        client.get_all_open_questions_from_tournament.side_effect = [[seen, new], [new]]
        self.assertEqual(await main.select_questions(client, RunConfig()), [new])
        self.assertEqual(client.get_all_open_questions_from_tournament.call_count, 2)

    async def test_empty_tournament_does_not_start_opencode(self) -> None:
        client = MagicMock()
        client.get_all_open_questions_from_tournament.return_value = []
        with (
            patch.object(main, "MetaculusClient", return_value=client),
            patch.object(main, "check_environment"),
            patch.object(main, "print_startup_banner"),
            patch.object(main, "print_run_summary_banner"),
            patch.object(main, "connect_for_run") as connect,
        ):
            self.assertEqual(await main.run(RunConfig()), 0)
            connect.assert_not_called()

    async def exercise(self, dry_run: bool, record: bool = True) -> tuple[int, int]:
        question = BinaryQuestion(
            question_text="Test",
            id_of_post=1,
            id_of_question=2,
            already_forecasted=False,
        )
        metaculus = MagicMock()
        # Same question appears in two tournaments: only forecast once.
        metaculus.get_all_open_questions_from_tournament.return_value = [question]
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.create_session.return_value = "ses_test"
        client.messages.return_value = {"data": []}

        async def record_result(*args: object) -> None:
            if record:
                job_id = next(jobs.JOBS.iterdir()).name
                jobs.save_forecast(job_id, {"probability": 0.4}, "Research", "Reason")

        client.wait.side_effect = record_result
        args = RunConfig(
            mode="tournament", timeout=5, dry_run=dry_run, single_agent=True
        )
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
            patch.object(main, "check_environment"),
            patch.object(main, "print_startup_banner"),
            patch.object(main, "print_run_summary_banner"),
            patch.object(main, "MetaculusClient", return_value=metaculus),
            patch.object(main, "connect_for_run", return_value=client),
            patch.object(
                BinaryReport, "publish_report_to_metaculus", new_callable=AsyncMock
            ) as publish,
        ):
            result = await main.run(args)
            self.assertEqual(client.create_session.await_count, 1)
            self.assertIn(SUPERFORECASTING_PRINCIPLES, client.prompt.await_args.args[1])
            return result, publish.await_count

    async def test_dry_run_never_publishes(self) -> None:
        self.assertEqual(await self.exercise(dry_run=True), (0, 0))

    async def test_valid_forecast_is_published_once(self) -> None:
        self.assertEqual(await self.exercise(dry_run=False), (0, 1))

    async def test_prose_without_recorded_forecast_fails(self) -> None:
        self.assertEqual(await self.exercise(dry_run=False, record=False), (1, 0))

    async def test_binary_ensemble_dry_run_saves_without_publishing(self) -> None:
        question = BinaryQuestion(question_text="Test", id_of_post=1, id_of_question=2)
        metaculus = MagicMock()
        metaculus.get_all_open_questions_from_tournament.return_value = [question]
        client = AsyncMock()
        client.__aenter__.return_value = client
        args = RunConfig(mode="test_questions", limit=1, timeout=5, dry_run=True)
        result = {
            "prediction": {"probability": 0.4},
            "research": "Evidence",
            "rationale": "Bayesian update",
        }
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
            patch.object(main, "check_environment"),
            patch.object(main, "print_startup_banner"),
            patch.object(main, "print_run_summary_banner"),
            patch.object(main, "MetaculusClient", return_value=metaculus),
            patch.object(main, "connect_for_run", return_value=client),
            patch.object(
                main, "run_ensemble", new_callable=AsyncMock, return_value=result
            ) as ensemble,
            patch.object(
                BinaryReport, "publish_report_to_metaculus", new_callable=AsyncMock
            ) as publish,
        ):
            self.assertEqual(await main.run(args), 0)
            ensemble.assert_awaited_once()
            publish.assert_not_awaited()
            job_id = next(Path(directory).iterdir()).name
            self.assertEqual(jobs.read_report(job_id).prediction, 0.4)
