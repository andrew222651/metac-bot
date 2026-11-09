"""Schemas and validation for distributional ensemble stages."""

from pydantic import Field, HttpUrl
from forecasting_tools import MetaculusQuestion

from ..strategy.pooling import posterior_schema, validate_posterior
from ..strategy.roles import MODEL_ROLES
from .bayesian import ModelArtifact, StrictModel


class SourceObservation(StrictModel):
    claim: str = Field(min_length=1, max_length=4000)
    source_urls: list[HttpUrl] = Field(min_length=1, max_length=8)
    source_date: str = Field(min_length=1, max_length=100)
    reasoning: str = Field(min_length=1, max_length=4000)
    limitations: str = Field(min_length=1, max_length=4000)


class PosteriorProposal(StrictModel):
    prior: dict
    posterior: dict
    prior_rationale: str = Field(min_length=1, max_length=4000)
    reasoning: str = Field(min_length=1, max_length=12000)
    research: str = Field(min_length=1, max_length=16000)
    evidence: list[SourceObservation] = Field(max_length=20)
    limitations: str = Field(min_length=1, max_length=4000)
    model: ModelArtifact | None = None


class PosteriorReview(StrictModel):
    accepts_current: bool
    posterior: dict
    critique: str = Field(min_length=1, max_length=12000)
    reasoning: str = Field(min_length=1, max_length=12000)
    limitations: str = Field(min_length=1, max_length=4000)


SCHEMAS = {"proposal": PosteriorProposal, "review": PosteriorReview}


def stage_schema(question: MetaculusQuestion, stage: str) -> dict:
    schema = SCHEMAS[stage].model_json_schema()
    schema["properties"]["posterior"] = posterior_schema(question)
    if stage == "proposal":
        schema["properties"]["prior"] = posterior_schema(question)
    return schema


def validate_result(
    question: MetaculusQuestion, stage: str, value: dict, context: dict
) -> dict:
    if stage not in SCHEMAS:
        raise ValueError("Unknown distributional ensemble stage")
    result = SCHEMAS[stage].model_validate(value).model_dump(mode="json")
    validate_posterior(question, result["posterior"])
    if stage == "proposal":
        validate_posterior(question, result["prior"])
        if context.get("role") in MODEL_ROLES:
            expected = MODEL_ROLES[context["role"]]
            if result["model"] is None or result["model"]["kind"] != expected:
                raise ValueError(
                    f"The {context['role']} must record a {expected} model artifact"
                )
    return result
