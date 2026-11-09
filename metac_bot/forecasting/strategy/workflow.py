"""The forecasting strategy: independent proposals, adjudication, and review."""

import asyncio
from typing import Any, Literal, TypeGuard

from forecasting_tools import BinaryQuestion, MetaculusQuestion, MultipleChoiceQuestion

from ..implementation.bayesian import candidates_from_proposals
from ..implementation.ensemble import EnsembleRunner, EnsembleSettings
from .aggregation import make_result, posterior_change
from .roles import MODEL_ROLES, ROLES


def uses_ensemble(
    question: MetaculusQuestion, *, single_agent: bool
) -> TypeGuard[BinaryQuestion | MultipleChoiceQuestion]:
    return (
        isinstance(question, (BinaryQuestion, MultipleChoiceQuestion))
        and not single_agent
    )


def require_quorum(
    runner: EnsembleRunner,
    stages: list[dict | None],
    stage: Literal["proposal", "review"],
    round_number: int | None = None,
) -> list[dict]:
    completed = [result for result in stages if result is not None]
    if len(completed) < runner.settings.min_members:
        description = (
            "proposals"
            if stage == "proposal"
            else f"reviews succeeded in round {round_number}"
        )
        if stage == "proposal":
            description += " succeeded"
        raise RuntimeError(
            f"Only {len(completed)} Bayesian {description}; need {runner.settings.min_members}. "
            f"{runner.describe_failures(stage, round_number=round_number)}"
        )
    required_roles = set(MODEL_ROLES) & set(ROLES)
    missing = required_roles - {result["role"] for result in completed}
    if missing:
        noun = "proposals" if stage == "proposal" else "reviews"
        raise RuntimeError(
            f"Required modeling {noun} failed: {', '.join(sorted(missing))}. "
            f"{runner.describe_failures(stage, missing, round_number)}"
        )
    return completed


async def run_workflow(runner: EnsembleRunner) -> dict:
    """Run independent proposals and review rounds; never publish from here."""
    proposals = require_quorum(
        runner,
        await asyncio.gather(
            *(
                runner.run_stage(
                    "proposal", role, {"role": role, "instructions": instructions}
                )
                for role, instructions in ROLES.items()
            )
        ),
        "proposal",
    )
    candidates = candidates_from_proposals(proposals, runner.options)
    initial, current = await runner.decide(
        "initial-adjudicator",
        {"proposals": proposals, "candidates": candidates, "round": 0},
        candidates,
    )
    runner.artifacts["rounds"].append(
        {"round": 0, "decision_stage": initial["stage_id"], **current}
    )
    runner.persist()

    converged = False
    for round_number in range(1, runner.settings.review_rounds + 1):
        context = {
            "round": round_number,
            "proposals": proposals,
            "candidates": candidates,
            "current": current,
        }
        reviews = require_quorum(
            runner,
            await asyncio.gather(
                *(
                    runner.run_stage("review", proposal["role"], context)
                    for proposal in proposals
                )
            ),
            "review",
            round_number,
        )
        final, updated = await runner.decide(
            f"adjudicator-round-{round_number}",
            {**context, "reviews": reviews},
            candidates,
        )
        delta = posterior_change(current, updated, runner.options)
        current = updated
        accepted = all(review["result"]["accepts_current"] for review in reviews)
        converged = accepted and delta <= runner.settings.tolerance
        runner.artifacts["rounds"].append(
            {
                "round": round_number,
                "decision_stage": final["stage_id"],
                "review_stages": [item["stage_id"] for item in reviews],
                "all_reviewers_accept_previous": accepted,
                "posterior_change": delta,
                **current,
            }
        )
        runner.persist()
        if converged:
            break

    prediction = runner.prediction(current)
    runner.artifacts.update(
        converged=converged,
        posterior=current["posterior"],
        published_prediction=prediction,
    )
    runner.persist()
    if not converged:
        raise RuntimeError(
            f"Bayesian ensemble did not converge within {runner.settings.review_rounds} review rounds; no forecast published"
        )
    return make_result(proposals, candidates, current, prediction, converged)


async def run_ensemble(
    opencode: Any,
    question: BinaryQuestion | MultipleChoiceQuestion,
    job_id: str,
    *,
    model: str | None = None,
    timeout: float = 900,
    concurrency: int = 2,
    min_members: int = 3,
    review_rounds: int = 3,
    tolerance: float = 0.01,
    member_models: dict[str, str] | None = None,
    adjudicator_model: str | None = None,
    stage_timeouts: dict[str, float | None] | None = None,
) -> dict:
    settings = EnsembleSettings(
        model=model,
        timeout=timeout,
        concurrency=concurrency,
        min_members=min_members,
        review_rounds=review_rounds,
        tolerance=tolerance,
        member_models=member_models or {},
        adjudicator_model=adjudicator_model,
        stage_timeouts=stage_timeouts or {},
    )
    return await run_workflow(EnsembleRunner(opencode, question, job_id, settings))
