import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from forecasting_tools import BinaryQuestion

from metac_bot.forecasting.implementation import jobs
from metac_bot.infrastructure import deadlines, runner
from metac_bot.infrastructure.config import RunConfig


class DeadlineTests(IsolatedAsyncioTestCase):
    def question(self, seconds: float) -> BinaryQuestion:
        return BinaryQuestion(
            question_text="Test",
            close_time=datetime.now(timezone.utc) + timedelta(seconds=seconds),
        )

    async def test_too_close_is_skipped_before_job_and_model(self) -> None:
        client = AsyncMock()
        with patch.object(runner, "create_job") as create:
            result = await runner.forecast_question(
                client, None, self.question(40), RunConfig()
            )
        self.assertIsInstance(result, RuntimeError)
        create.assert_not_called()
        client.create_session.assert_not_awaited()

    async def test_dry_run_does_not_enforce_closing(self) -> None:
        self.assertIsNone(
            deadlines.forecast_budget(self.question(-60), RunConfig(dry_run=True))
        )

    async def test_short_budget_caps_both_ensemble_stages(self) -> None:
        config = RunConfig(ensemble_concurrency=2)
        self.assertEqual(deadlines.stage_timeout_cap(600, config), 100)

    async def test_late_close_prevents_publication(self) -> None:
        question = self.question(200)
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
            patch.object(runner, "remaining_seconds", return_value=-1),
            patch.object(
                runner,
                "run_ensemble",
                new=AsyncMock(
                    return_value={
                        "prediction": {"probability": 0.4},
                        "research": "Research",
                        "rationale": "Reason",
                    }
                ),
            ),
            patch.object(runner, "publish_prediction") as publish,
        ):
            result = await runner.forecast_question(
                AsyncMock(), MagicMock(), question, RunConfig()
            )
        self.assertIsInstance(result, RuntimeError)
        publish.assert_not_called()

    async def test_whole_forecast_timeout_does_not_publish(self) -> None:
        async def slow(*args: object, **kwargs: object) -> None:
            await asyncio.sleep(10)

        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
            patch.object(runner, "forecast_budget", return_value=0.01),
            patch.object(runner, "MIN_FORECAST_BUDGET", 0),
            patch.object(runner, "run_ensemble", side_effect=slow),
            patch.object(runner, "publish_prediction") as publish,
        ):
            result = await runner.forecast_question(
                AsyncMock(), MagicMock(), self.question(300), RunConfig()
            )
        self.assertIsInstance(result, TimeoutError)
        publish.assert_not_called()

    async def test_live_selection_prioritizes_close_before_limit(self) -> None:
        later, sooner = self.question(600), self.question(300)
        later.id_of_post, later.id_of_question = 1, 1
        sooner.id_of_post, sooner.id_of_question = 2, 2
        client = MagicMock()
        client.get_all_open_questions_from_tournament.side_effect = [
            [later, sooner],
            [],
        ]
        selected = await runner.select_questions(client, RunConfig(limit=1))
        self.assertEqual(selected, [sooner])
