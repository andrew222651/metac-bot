"""Select questions, produce forecasts, and publish validated reports."""

import asyncio
import logging
from typing import Any, cast

from forecasting_tools import (
    BinaryQuestion,
    ForecastReport,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    PredictedOptionList,
)

from ..forecasting.implementation.agent import run_agent
from ..forecasting.implementation.jobs import (
    create_job,
    job_path,
    read_report,
    save_forecast,
    write_json,
)
from ..forecasting.strategy.prompts import forecast_prompt
from ..forecasting.strategy.workflow import run_ensemble, uses_ensemble
from .config import OneOffConfig, RunConfig
from .deadlines import (
    MIN_FORECAST_BUDGET,
    forecast_budget,
    is_closed,
    remaining_seconds,
    stage_timeout_cap,
)
from .opencode import OpenCode
from .opencode_service import connect_for_run
from .runtime import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
)

logger = logging.getLogger(__name__)


async def select_questions(
    metaculus: MetaculusClient, args: RunConfig
) -> list[MetaculusQuestion]:
    """Fetch and deduplicate the questions selected by the run mode."""
    tournaments = {
        "tournament": [
            metaculus.CURRENT_AI_COMPETITION_ID,
            metaculus.CURRENT_MINIBENCH_ID,
        ],
        "test_questions": ["bot-testing-area"],
    }[args.mode]
    questions = {}
    for tournament in tournaments:
        for question in await asyncio.to_thread(
            metaculus.get_all_open_questions_from_tournament, tournament
        ):
            if args.mode == "tournament" and question.already_forecasted:
                continue
            questions[(question.id_of_post, question.id_of_question)] = question
    selected = list(questions.values())
    if not args.dry_run:

        def close_sort_key(question: MetaculusQuestion) -> float:
            remaining = remaining_seconds(question)
            return remaining if remaining is not None else float("inf")

        selected.sort(key=close_sort_key)
    if args.limit:
        selected = selected[: args.limit]
    return selected


async def forecast_question(
    opencode: OpenCode,
    metaculus: MetaculusClient | None,
    question: MetaculusQuestion,
    args: RunConfig,
) -> ForecastReport | Exception:
    """Run one question independently; return either its report or its error."""
    budget = forecast_budget(question, args)
    if not args.dry_run and is_closed(question):
        return RuntimeError(f"Question {question.page_url} is already closed")
    if budget is not None and budget < MIN_FORECAST_BUDGET:
        return RuntimeError(
            f"Question {question.page_url} is too close to closing to forecast safely"
        )
    job_id = create_job(question)
    session_id = None
    try:
        async with asyncio.timeout(budget):
            await _produce_forecast(opencode, question, args, job_id, budget)
        report = read_report(job_id)
        write_json(
            job_path(job_id) / "report.json",
            report.model_dump(mode="json", serialize_as_any=True),
        )
        if not args.dry_run:
            remaining = remaining_seconds(question)
            if is_closed(question) or (remaining is not None and remaining <= 0):
                raise RuntimeError("Question closed before publication")
            await report.publish_report_to_metaculus(metaculus)
        write_json(job_path(job_id) / "result.json", {"published": not args.dry_run})
        return report
    except Exception as exc:
        session_path = job_path(job_id) / "session.json"
        if session_path.exists():
            from ..forecasting.implementation.jobs import read_json

            session_id = read_json(session_path).get("session_id")
        logger.exception(
            "Forecast failed for %s (session %s)", question.page_url, session_id
        )
        write_json(
            job_path(job_id) / "error.json",
            {"error": str(exc), "session_id": session_id},
        )
        return exc


async def _produce_forecast(
    opencode: OpenCode,
    question: MetaculusQuestion,
    args: RunConfig,
    job_id: str,
    budget: float | None,
) -> None:
    session_id = None
    cap = stage_timeout_cap(budget, args) if budget is not None else float("inf")
    if uses_ensemble(question, single_agent=args.single_agent):
        logger.info("Bayesian ensemble for %s: job=%s", question.page_url, job_id)
        result = await run_ensemble(
            opencode,
            question,
            job_id,
            model=args.model,
            timeout=min(args.timeout, cap),
            concurrency=args.ensemble_concurrency,
            min_members=args.ensemble_min_members,
            review_rounds=args.ensemble_review_rounds,
            tolerance=args.ensemble_tolerance,
            member_models=args.ensemble_models,
            adjudicator_model=args.ensemble_adjudicator_model,
            stage_timeouts={
                "proposal": min(args.ensemble_proposal_timeout or args.timeout, cap),
                "review": min(args.ensemble_review_timeout or args.timeout, cap),
                "decision": args.ensemble_decision_timeout,
            },
        )
        save_forecast(job_id, **result)
    else:

        def record_session(value: str) -> None:
            nonlocal session_id
            session_id = value
            write_json(job_path(job_id) / "session.json", {"session_id": value})
            logger.info(
                "Question %s: job=%s session=%s", question.page_url, job_id, value
            )

        try:
            await run_agent(
                opencode,
                title=f"Forecast: {question.question_text[:100]}",
                model=args.model,
                timeout=min(args.timeout, cap),
                on_session=record_session,
                prompt=forecast_prompt(job_id, one_off=isinstance(args, OneOffConfig)),
            )
        except TimeoutError as timeout_error:
            if session_id is None:
                raise
            try:
                read_report(job_id)
            except Exception:
                raise timeout_error from None
            write_json(
                job_path(job_id) / "session.json",
                {"session_id": session_id, "timeout_recovered": True},
            )
            logger.warning(
                "Using recorded forecast after session %s timed out", session_id
            )
    if not (job_path(job_id) / "forecast.json").exists():
        messages = await opencode.messages(cast(str, session_id))
        write_json(job_path(job_id) / "messages.json", messages)
        raise RuntimeError(
            f"OpenCode did not record a forecast; inspect session {session_id}"
        )


async def run(args: RunConfig) -> int:
    try:
        check_environment(strict=True, require_llm_key=False)
    except SystemExit as exc:
        return int(exc.code or 0)
    print_startup_banner(args.mode, will_publish=not args.dry_run)
    metaculus = MetaculusClient()
    selected = await select_questions(metaculus, args)
    reports = []
    if not selected:
        print_run_summary_banner(reports, will_publish=not args.dry_run)
        return 0

    async with connect_for_run(
        api_timeout=args.api_timeout,
        startup_timeout=args.service_start_timeout,
        client_factory=OpenCode,
    ) as opencode:
        for question in selected:
            reports.append(await forecast_question(opencode, metaculus, question, args))
    print_run_summary_banner(reports, will_publish=not args.dry_run)
    return int(any(isinstance(report, BaseException) for report in reports))


def one_off_question(args: OneOffConfig) -> BinaryQuestion | MultipleChoiceQuestion:
    """Build a local question snapshot without querying Metaculus."""
    data: dict[str, Any] = {
        "question_text": args.question,
        "resolution_criteria": args.criteria,
        "background_info": args.background,
        "fine_print": (
            "This is a standalone user-supplied forecast. Do not assume it corresponds to a "
            "Metaculus question. If no separate resolution criteria are provided, "
            "use the question wording and explicitly state any ambiguities."
        ),
        "custom_metadata": {"source": "user-supplied one-off", "one_off": True},
    }
    if args.options:
        data["options"] = list(args.options)
        return MultipleChoiceQuestion.model_validate(data)
    return BinaryQuestion.model_validate(data)


async def run_one_off(args: OneOffConfig) -> int:
    """Forecast only the supplied question; never fetch or publish to Metaculus."""
    question = one_off_question(args)
    print("Running a local one-off forecast (no publishing).", flush=True)
    async with connect_for_run(
        api_timeout=args.api_timeout,
        startup_timeout=args.service_start_timeout,
        client_factory=OpenCode,
    ) as opencode:
        report = await forecast_question(opencode, None, question, args)
    if isinstance(report, BaseException):
        print(f"Forecast failed: {report}")
        return 1
    print("\nForecast:")
    if isinstance(report.prediction, PredictedOptionList):
        for name, probability in report.prediction.to_dict().items():
            print(f"  {name}: {probability:.1%}")
    else:
        print(f"  Yes: {report.prediction:.1%}")
        print(f"  No: {1 - report.prediction:.1%}")
    print(f"\n{report.explanation}")
    return 0
