"""Auditable, judgment-based Bayesian updating for binary forecasts.

The model supplies conditional likelihood ratios, not automatically verified
probabilities. Grouping and conditional dependence remain model judgments.
"""

import math
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from .roles import MODEL_ROLES


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Evidence(StrictModel):
    claim: str = Field(min_length=1, max_length=4000)
    source_urls: list[HttpUrl] = Field(min_length=1, max_length=8)
    source_date: str = Field(min_length=1, max_length=100)
    likelihood_ratio: float = Field(gt=0, ge=math.exp(-20), le=math.exp(20))
    rationale: str = Field(min_length=1, max_length=4000)
    limitations: str = Field(min_length=1, max_length=4000)


class ModelArtifact(StrictModel):
    kind: Literal["economic", "causal"]
    applicable: bool
    applicability_reason: str = Field(min_length=1, max_length=4000)
    structure: str = Field(min_length=1, max_length=12000)
    equations: str = Field(min_length=1, max_length=12000)
    inputs_and_sources: str = Field(min_length=1, max_length=12000)
    inference_and_implementation: str = Field(min_length=1, max_length=12000)
    result_and_uncertainty: str = Field(min_length=1, max_length=12000)
    limitations: str = Field(min_length=1, max_length=12000)


class Proposal(StrictModel):
    prior: float = Field(gt=0, lt=1)
    prior_rationale: str = Field(min_length=1, max_length=4000)
    evidence: list[Evidence] = Field(max_length=20)
    research: str = Field(min_length=1, max_length=16000)
    model: ModelArtifact | None = None


class ChoiceEvidence(StrictModel):
    claim: str = Field(min_length=1, max_length=4000)
    source_urls: list[HttpUrl] = Field(min_length=1, max_length=8)
    source_date: str = Field(min_length=1, max_length=100)
    likelihoods: dict[str, float]
    rationale: str = Field(min_length=1, max_length=4000)
    limitations: str = Field(min_length=1, max_length=4000)


class ChoiceProposal(StrictModel):
    prior: dict[str, float]
    prior_rationale: str = Field(min_length=1, max_length=4000)
    evidence: list[ChoiceEvidence] = Field(max_length=20)
    research: str = Field(min_length=1, max_length=16000)
    model: ModelArtifact | None = None


class EvidenceGroup(StrictModel):
    candidate_ids: list[str] = Field(min_length=1, max_length=100)
    conditional_likelihood_ratio: float = Field(gt=0, ge=math.exp(-20), le=math.exp(20))
    rationale: str = Field(min_length=1, max_length=4000)
    dependence_assumptions: str = Field(min_length=1, max_length=4000)


class Exclusion(StrictModel):
    candidate_id: str
    reason: str = Field(min_length=1, max_length=4000)


class Decision(StrictModel):
    prior: float = Field(gt=0, lt=1)
    prior_rationale: str = Field(min_length=1, max_length=4000)
    groups: list[EvidenceGroup] = Field(max_length=100)
    excluded: list[Exclusion] = Field(max_length=100)
    limitations: str = Field(min_length=1, max_length=4000)


class ChoiceEvidenceGroup(StrictModel):
    candidate_ids: list[str] = Field(min_length=1, max_length=100)
    conditional_likelihoods: dict[str, float]
    rationale: str = Field(min_length=1, max_length=4000)
    dependence_assumptions: str = Field(min_length=1, max_length=4000)


class ChoiceDecision(StrictModel):
    prior: dict[str, float]
    prior_rationale: str = Field(min_length=1, max_length=4000)
    groups: list[ChoiceEvidenceGroup] = Field(max_length=100)
    excluded: list[Exclusion] = Field(max_length=100)
    limitations: str = Field(min_length=1, max_length=4000)


class Review(StrictModel):
    accepts_current: bool
    critique: str = Field(min_length=1, max_length=12000)
    proposed_changes: str = Field(min_length=1, max_length=12000)
    limitations: str = Field(min_length=1, max_length=4000)


SCHEMAS = {"proposal": Proposal, "decision": Decision, "review": Review}
CHOICE_SCHEMAS = {
    "proposal": ChoiceProposal,
    "decision": ChoiceDecision,
    "review": Review,
}


def normalize_url(url: str) -> str:
    parts = urlsplit(url)
    query = urlencode(
        sorted(
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}
        )
    )
    host = (parts.hostname or "").lower()
    port = parts.port
    if port and not (
        (parts.scheme == "https" and port == 443)
        or (parts.scheme == "http" and port == 80)
    ):
        host += f":{port}"
    return urlunsplit(
        (parts.scheme.lower(), host, parts.path.rstrip("/") or "/", query, "")
    )


def candidates_from_proposals(
    proposals: list[dict], options: list[str] | None = None
) -> list[dict]:
    result = []
    for index, entry in enumerate(proposals):
        proposal = (ChoiceProposal if options is not None else Proposal).model_validate(
            entry["result"]
        )
        for item_index, item in enumerate(proposal.evidence):
            result.append(
                {
                    "id": f"p{index + 1}-e{item_index + 1}",
                    "role": entry["role"],
                    **item.model_dump(mode="json"),
                }
            )
    return result


def validate_partition(
    parsed: Decision | ChoiceDecision, candidates: list[dict]
) -> None:
    by_id = {item["id"]: item for item in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("Duplicate candidate IDs")
    assigned = set()
    source_groups = {}
    for index, group in enumerate(parsed.groups):
        for candidate_id in group.candidate_ids:
            if candidate_id not in by_id or candidate_id in assigned:
                raise ValueError(f"Unknown or repeated candidate ID: {candidate_id}")
            assigned.add(candidate_id)
            for url in by_id[candidate_id]["source_urls"]:
                normalized = normalize_url(str(url))
                previous = source_groups.setdefault(normalized, index)
                if previous != index:
                    raise ValueError(
                        "The same source cannot occur in multiple evidence groups"
                    )
    for exclusion in parsed.excluded:
        if exclusion.candidate_id not in by_id or exclusion.candidate_id in assigned:
            raise ValueError(
                f"Unknown or repeated candidate ID: {exclusion.candidate_id}"
            )
        assigned.add(exclusion.candidate_id)
    if assigned != set(by_id):
        raise ValueError("Every candidate must be grouped or excluded with a reason")


def calculate_posterior(decision: dict, candidates: list[dict]) -> dict:
    """Recompute from the prior each time; never apply updates to the last round."""
    parsed = Decision.model_validate(decision)
    validate_partition(parsed, candidates)
    log_odds = math.log(parsed.prior) - math.log1p(-parsed.prior)
    log_odds += sum(
        math.log(group.conditional_likelihood_ratio) for group in parsed.groups
    )
    # Stable logistic avoids overflow for extreme but valid input.
    posterior = (
        1 / (1 + math.exp(-log_odds))
        if log_odds >= 0
        else math.exp(log_odds) / (1 + math.exp(log_odds))
    )
    return {
        "prior": parsed.prior,
        "log_odds": log_odds,
        "posterior": posterior,
        "published_probability": max(0.01, min(0.99, posterior)),
        "decision": parsed.model_dump(mode="json"),
    }


def validate_choice_weights(
    weights: dict[str, float], options: list[str], *, prior: bool
) -> None:
    if (
        len(options) < 2
        or len(set(options)) != len(options)
        or set(weights) != set(options)
    ):
        raise ValueError(
            "Probabilities or likelihoods must include every exact option once"
        )
    if any(not math.isfinite(value) or value <= 0 for value in weights.values()):
        raise ValueError(
            "All probabilities and likelihoods must be finite and positive"
        )
    if prior and not math.isclose(sum(weights.values()), 1, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Prior probabilities must sum to 1")
    if not prior and any(
        value < math.exp(-20) or value > math.exp(20) for value in weights.values()
    ):
        raise ValueError("Likelihood weights must be between exp(-20) and exp(20)")


def calculate_choice_posterior(
    decision: dict, candidates: list[dict], options: list[str]
) -> dict:
    """Update mutually exclusive options by ordered conditional evidence likelihoods."""
    parsed = ChoiceDecision.model_validate(decision)
    validate_partition(parsed, candidates)
    validate_choice_weights(parsed.prior, options, prior=True)
    for group in parsed.groups:
        validate_choice_weights(group.conditional_likelihoods, options, prior=False)
    log_weights = {
        option: math.log(parsed.prior[option])
        + sum(
            math.log(group.conditional_likelihoods[option]) for group in parsed.groups
        )
        for option in options
    }
    max_weight = max(log_weights.values())
    weights = {
        option: math.exp(value - max_weight) for option, value in log_weights.items()
    }
    total = sum(weights.values())
    posterior = {option: value / total for option, value in weights.items()}
    return {
        "prior": parsed.prior,
        "log_weights": log_weights,
        "posterior": posterior,
        "decision": parsed.model_dump(mode="json"),
    }


def validate_stage(
    stage: str, value: dict, context: dict, options: list[str] | None = None
) -> dict:
    schemas = CHOICE_SCHEMAS if options is not None else SCHEMAS
    if stage not in schemas:
        raise ValueError("Unknown Bayesian stage")
    parsed = schemas[stage].model_validate(value).model_dump(mode="json")
    if stage == "proposal" and context.get("role") in MODEL_ROLES:
        expected = MODEL_ROLES[context["role"]]
        if parsed["model"] is None or parsed["model"]["kind"] != expected:
            raise ValueError(
                f"The {context['role']} must record a {expected} model artifact"
            )
    if options is not None and stage == "proposal":
        validate_choice_weights(parsed["prior"], options, prior=True)
        for evidence in parsed["evidence"]:
            validate_choice_weights(evidence["likelihoods"], options, prior=False)
    if stage == "decision":
        if options is None:
            calculate_posterior(parsed, context["candidates"])
        else:
            calculate_choice_posterior(parsed, context["candidates"], options)
    return parsed
