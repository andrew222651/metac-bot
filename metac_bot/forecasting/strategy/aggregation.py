"""Bayesian aggregation and reporting policy for the forecasting strategy."""

import json

from ..implementation.bayesian import calculate_choice_posterior, calculate_posterior


def aggregate(
    decision: dict, candidates: list[dict], options: list[str] | None
) -> dict:
    """Recompute the posterior from the adjudicated prior and evidence groups."""
    if options is None:
        return calculate_posterior(decision, candidates)
    return calculate_choice_posterior(decision, candidates, options)


def posterior_change(previous: dict, current: dict, options: list[str] | None) -> float:
    if options is None:
        return abs(current["posterior"] - previous["posterior"])
    return max(
        abs(current["posterior"][option] - previous["posterior"][option])
        for option in options
    )


def make_result(
    proposals: list[dict],
    candidates: list[dict],
    current: dict,
    prediction: dict,
    converged: bool,
) -> dict:
    research = "\n\n".join(
        f"### {p['role']}\n{p['result']['research']}" for p in proposals
    )
    models = {
        p["role"]: p["result"]["model"] for p in proposals if p["result"].get("model")
    }
    rationale = (
        "This is a judgment-based Bayesian evidence update, not an objectively verified posterior. "
        "The adjudicator groups overlapping sources and estimates each group's likelihood ratio "
        "conditional on previous groups. Dependence and likelihoods are uncertain model judgments.\n\n"
        f"Prior: {json.dumps(current['prior'])}. Posterior: {json.dumps(current['posterior'])}. "
        f"Prediction after probability bounds: {json.dumps(prediction)}. "
        f"Posterior-change convergence criterion met: {converged}.\n\n"
        "## Final evidence ledger and assumptions\n\n```json\n"
        + json.dumps(current["decision"], indent=2)
        + "\n```\n\n## Evidence candidates and sources\n\n```json\n"
        + json.dumps(candidates, indent=2)
        + "\n```\n\n## Economic and causal models\n\n```json\n"
        + json.dumps(models, indent=2)
        + "\n```"
    )
    return {"prediction": prediction, "research": research, "rationale": rationale}
