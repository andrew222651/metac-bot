"""The forecasting strategy: independent posteriors, pooling, and review."""

import asyncio
from typing import Any
import json

from forecasting_tools import MetaculusQuestion

from ..implementation.ensemble import EnsembleRunner, EnsembleSettings
from .pooling import pool_distance, pool_posteriors, publication_prediction
from .roles import ROLES


def uses_ensemble(question: MetaculusQuestion, *, single_agent: bool) -> bool:
    return not single_agent


async def run_workflow(runner: EnsembleRunner) -> dict:
    """Pool independent Bayesian judgments, then review and revise them."""
    proposals = [
        result
        for result in await asyncio.gather(
            *(
                runner.run_stage(
                    "proposal", role, {"role": role, "instructions": instructions}
                )
                for role, instructions in ROLES.items()
            )
        )
        if result is not None
    ]
    if not proposals:
        raise RuntimeError(
            f"No Bayesian proposals succeeded; {runner.describe_failures('proposal')}"
        )
    weights = [1 / len(proposals)] * len(proposals)
    posteriors = {item["role"]: item["result"]["posterior"] for item in proposals}
    current = pool_posteriors(runner.question, list(posteriors.values()), weights)
    runner.artifacts["weights"] = dict(zip(posteriors, weights))
    runner.artifacts["rounds"].append(
        {"round": 0, "posteriors": dict(posteriors), "consensus": current.prediction()}
    )
    runner.persist()

    converged = False
    previous_reviews: list[dict] = []
    reviews_unavailable = False
    for round_number in range(1, runner.settings.review_rounds + 1):
        context = {
            "round": round_number,
            "proposals": proposals,
            "posteriors": dict(posteriors),
            "weights": dict(zip(posteriors, weights)),
            "current": current.prediction(),
            "previous_reviews": previous_reviews,
        }
        reviews = [
            result
            for result in await asyncio.gather(
                *(
                    runner.run_stage("review", proposal["role"], context)
                    for proposal in proposals
                )
            )
            if result is not None
        ]
        for review in reviews:
            posteriors[review["role"]] = review["result"]["posterior"]
        previous_reviews = reviews
        updated = pool_posteriors(runner.question, list(posteriors.values()), weights)
        delta = pool_distance(
            current, updated, error=min(0.001, runner.settings.tolerance / 10)
        )
        current = updated
        accepted = (
            all(review["result"]["accepts_current"] for review in reviews)
            if reviews
            else None
        )
        converged = accepted is True and delta <= runner.settings.tolerance
        reviews_unavailable = not reviews
        runner.artifacts["rounds"].append(
            {
                "round": round_number,
                "review_stages": [item["stage_id"] for item in reviews],
                "all_reviewers_accept_previous": accepted,
                "posterior_change": delta,
                "posteriors": dict(posteriors),
                "consensus": current.prediction(),
            }
        )
        runner.persist()
        if converged or reviews_unavailable:
            break

    posterior = current.prediction()
    prediction = publication_prediction(current)
    runner.artifacts.update(
        converged=converged,
        reviews_unavailable=reviews_unavailable,
        posterior=posterior,
        published_prediction=prediction,
    )
    runner.persist()
    if not converged and not reviews_unavailable:
        raise RuntimeError(
            f"Bayesian ensemble did not converge within {runner.settings.review_rounds} review rounds; no forecast published"
        )
    research = "\n\n".join(
        f"### {item['role']}\n{item['result']['research']}" for item in proposals
    )
    review_summary = (
        f"All {len(previous_reviews)} completed reviewers accepted the consensus and "
        f"the change met the tolerance; {len(proposals) - len(previous_reviews)} "
        "reviewers did not complete the final round."
        if converged
        else "No reviewers completed the final round. The latest available pooled "
        "judgments are reported without establishing convergence; earlier completed "
        "reviews, if any, are retained."
    )
    rationale = (
        "Equal-weight linear pooling of agents' Bayesian judgments; shared evidence "
        "is not treated as independent, and the pool is not a verified Bayesian posterior. "
        "Quantiles use piecewise-linear CDF interpolation and the documented tail assumptions. "
        f"{len(proposals)} of {len(ROLES)} proposals completed. {review_summary}\n\n"
        "## Agent proposals, priors, sources, and models\n\n```json\n"
        + json.dumps(proposals, indent=2)
        + "\n```\n\n## Ensemble deliberation and revisions\n\n```json\n"
        + json.dumps(runner.artifacts, indent=2)
        + "\n```"
    )
    return {"prediction": prediction, "research": research, "rationale": rationale}


async def run_ensemble(
    opencode: Any,
    question: MetaculusQuestion,
    job_id: str,
    *,
    model: str | None = None,
    timeout: float = 900,
    concurrency: int = 2,
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
        review_rounds=review_rounds,
        tolerance=tolerance,
        member_models=member_models or {},
        adjudicator_model=adjudicator_model,
        stage_timeouts=stage_timeouts or {},
    )
    return await run_workflow(EnsembleRunner(opencode, question, job_id, settings))
