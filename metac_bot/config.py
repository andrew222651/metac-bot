"""Shared, validated configuration for the runner and ensemble."""

from dataclasses import dataclass, field
import math
from typing import Literal

ROLES = {
    "generalist": "Evaluate the full question and balanced scenarios.",
    "base-rate": "Focus on reference classes, historical frequencies and prior selection.",
    "current-evidence": "Focus on recent developments and authoritative primary sources.",
    "skeptic": "Challenge key assumptions and seek credible counterevidence.",
    "quantitative": "Focus on measurable trends, explicit likelihoods and uncertainty.",
    "economic-modeler": (
        "Load the economic-modeling and modeling-python skills and build an explicit economic model, "
        "such as a DSGE-style model when appropriate. Record the required model artifact."
    ),
    "causal-modeler": (
        "Load the bayesian-causal-modeling and modeling-python skills and build an explicit Bayesian causal model. "
        "Record the required model artifact."
    ),
}

MODEL_ROLES = {"economic-modeler": "economic", "causal-modeler": "causal"}


def model_reference(model: str | None) -> dict[str, str] | None:
    """Validate a provider/model string and convert it to the V2 API shape."""
    if model is None:
        return None
    provider, separator, name = model.partition("/")
    identifier, variant_separator, variant = name.partition("#")
    if not separator or not provider or not identifier:
        raise ValueError(
            "Model must be provider/model, optionally followed by #variant"
        )
    if variant_separator and not variant:
        raise ValueError("Model variant must not be empty")
    result = {"providerID": provider, "id": identifier}
    if variant_separator:
        result["variant"] = variant
    return result


def validate_model(model: str | None) -> None:
    model_reference(model)


@dataclass(frozen=True)
class RunConfig:
    mode: Literal["tournament", "metaculus_cup", "test_questions"] = "tournament"
    dry_run: bool = False
    limit: int | None = None
    model: str | None = None
    timeout: float = 900
    api_timeout: float = 30
    single_agent: bool = False
    ensemble_concurrency: int = 2
    ensemble_min_members: int = 3
    ensemble_review_rounds: int = 3
    ensemble_tolerance: float = 0.01
    ensemble_proposal_timeout: float | None = None
    ensemble_review_timeout: float | None = None
    ensemble_decision_timeout: float | None = None
    ensemble_models: dict[str, str] = field(default_factory=dict)
    ensemble_adjudicator_model: str | None = None

    def __post_init__(self):
        if self.mode not in ("tournament", "metaculus_cup", "test_questions"):
            raise ValueError("Invalid tournament mode")
        if (
            not math.isfinite(self.timeout)
            or self.timeout <= 0
            or not math.isfinite(self.api_timeout)
            or self.api_timeout <= 0
            or (self.limit is not None and self.limit <= 0)
        ):
            raise ValueError("Timeouts and --limit must be finite and positive")
        for value in (
            self.ensemble_proposal_timeout,
            self.ensemble_review_timeout,
            self.ensemble_decision_timeout,
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("Ensemble stage timeouts must be finite and positive")
        if (
            self.ensemble_concurrency < 1
            or not 1 <= self.ensemble_min_members <= len(ROLES)
            or not 1 <= self.ensemble_review_rounds <= 5
            or not 0 < self.ensemble_tolerance < 1
        ):
            raise ValueError("Invalid ensemble configuration")
        validate_model(self.model)
        validate_model(self.ensemble_adjudicator_model)
        for role, model in self.ensemble_models.items():
            if role not in ROLES:
                raise ValueError(f"Invalid ensemble role: {role}")
            validate_model(model)


@dataclass(frozen=True)
class OneOffConfig(RunConfig):
    """A user-supplied question; publication is never permitted."""

    dry_run: bool = True
    question: str = ""
    options: tuple[str, ...] = ()
    criteria: str | None = None
    background: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if not self.dry_run:
            raise ValueError("One-off forecasts cannot be published")
        if not self.question.strip():
            raise ValueError("Question must not be empty")
        if self.options:
            if len(self.options) < 2 or any(not item.strip() for item in self.options):
                raise ValueError(
                    "Provide at least two nonempty options, or none for a binary question"
                )
            if len({item.strip().casefold() for item in self.options}) != len(
                self.options
            ):
                raise ValueError("Options must be distinct")
