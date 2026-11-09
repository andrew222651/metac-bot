"""Pool coherent posterior distributions without assuming independent agents."""

from dataclasses import dataclass
import math
from typing import Any

from forecasting_tools import (
    BinaryQuestion,
    ConditionalQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from jsonschema import validate

from ..implementation.domain import finite_number, object_schema, prediction_schema
from .distributions import (
    ContinuousDistribution,
    cdf_distance_bounds,
    distribution_from_percentiles,
    distribution_prediction,
    pool_distributions,
)


def posterior_schema(question: MetaculusQuestion) -> dict:
    """An agent's conditional posterior specifies a joint, not a free child."""
    if isinstance(question, ConditionalQuestion):
        if not isinstance(question.parent, BinaryQuestion):
            raise ValueError("Conditional ensemble requires a binary condition")
        return object_schema(
            {
                "parent": prediction_schema(question.parent),
                "prediction_yes": posterior_schema(question.question_yes),
                "prediction_no": posterior_schema(question.question_no),
            }
        )
    return prediction_schema(question)


def validate_posterior(question: MetaculusQuestion, prediction: dict) -> None:
    validate(prediction, posterior_schema(question))
    if isinstance(question, BinaryQuestion):
        probability = finite_number(prediction["probability"])
        if not 0 <= probability <= 1:
            raise ValueError("Probability must lie between 0 and 1")
    elif isinstance(question, MultipleChoiceQuestion):
        values = [finite_number(prediction["options"][key]) for key in question.options]
        if not all(0 <= p <= 1 for p in values) or not math.isclose(
            sum(values), 1, rel_tol=0, abs_tol=1e-6
        ):
            raise ValueError("Option probabilities must be nonnegative and sum to 1")
    elif isinstance(question, (NumericQuestion, DateQuestion)):
        distribution_from_percentiles(question, prediction)
    elif isinstance(question, ConditionalQuestion):
        validate_posterior(question.parent, prediction["parent"])
        validate_posterior(question.question_yes, prediction["prediction_yes"])
        validate_posterior(question.question_no, prediction["prediction_no"])
    else:
        raise ValueError(f"Unsupported question type: {type(question).__name__}")


@dataclass(frozen=True)
class Pool:
    """A pooled measure; numeric pools retain the full mixture CDF."""

    question: MetaculusQuestion
    value: Any

    def prediction(self) -> dict:
        question = self.question
        if isinstance(question, BinaryQuestion):
            return {"probability": self.value}
        if isinstance(question, MultipleChoiceQuestion):
            return {"options": self.value}
        if isinstance(question, (NumericQuestion, DateQuestion)):
            return distribution_prediction(question, self.value)
        if isinstance(question, ConditionalQuestion):
            parent, yes, no, child = self.value
            return {
                "parent": parent.prediction(),
                "prediction_yes": yes.prediction(),
                "prediction_no": no.prediction(),
                "child": child.prediction(),
            }
        raise ValueError("Unsupported question type")


def pool_posteriors(
    question: MetaculusQuestion, predictions: list[dict], weights: list[float]
) -> Pool:
    if not predictions or len(predictions) != len(weights):
        raise ValueError("A weight is required for every posterior")
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("Weights must be finite and nonnegative")
    total = sum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Weights must have positive finite total")
    weights = [w / total for w in weights]
    for prediction in predictions:
        validate_posterior(question, prediction)
    if isinstance(question, BinaryQuestion):
        return Pool(
            question, sum(w * p["probability"] for w, p in zip(weights, predictions))
        )
    if isinstance(question, MultipleChoiceQuestion):
        return Pool(
            question,
            {
                option: sum(
                    w * p["options"][option] for w, p in zip(weights, predictions)
                )
                for option in question.options
            },
        )
    if isinstance(question, (NumericQuestion, DateQuestion)):
        return Pool(
            question,
            pool_distributions(
                [
                    distribution_from_percentiles(question, prediction)
                    for prediction in predictions
                ],
                weights,
            ),
        )
    if isinstance(question, ConditionalQuestion):
        parent_predictions = [p["parent"] for p in predictions]
        parents = [p["probability"] for p in parent_predictions]
        yes_weights = [w * p for w, p in zip(weights, parents)]
        no_weights = [w * (1 - p) for w, p in zip(weights, parents)]
        if sum(yes_weights) == 0 or sum(no_weights) == 0:
            raise ValueError(
                "Cannot derive a conditional distribution for a zero-probability branch"
            )
        yes_predictions = [p["prediction_yes"] for p in predictions]
        no_predictions = [p["prediction_no"] for p in predictions]
        yes = pool_posteriors(question.question_yes, yes_predictions, yes_weights)
        no = pool_posteriors(question.question_no, no_predictions, no_weights)
        child = pool_posteriors(
            question.child, yes_predictions + no_predictions, yes_weights + no_weights
        )
        return Pool(
            question,
            (
                pool_posteriors(question.parent, parent_predictions, weights),
                yes,
                no,
                child,
            ),
        )
    raise ValueError(f"Unsupported question type: {type(question).__name__}")


def pool_distance(first: Pool, second: Pool, *, error: float = 0.001) -> float:
    """Return a conservative upper bound on the largest distribution change."""
    question = first.question
    if isinstance(question, BinaryQuestion):
        return abs(first.value - second.value)
    if isinstance(question, MultipleChoiceQuestion):
        return max(abs(first.value[o] - second.value[o]) for o in question.options)
    if isinstance(question, (NumericQuestion, DateQuestion)):
        assert isinstance(first.value, ContinuousDistribution)
        assert isinstance(second.value, ContinuousDistribution)
        return cdf_distance_bounds(first.value, second.value, error=error)[1]
    if isinstance(question, ConditionalQuestion):
        return max(
            pool_distance(a, b, error=error) for a, b in zip(first.value, second.value)
        )
    raise ValueError("Unsupported question type")


def publication_prediction(pool: Pool) -> dict:
    """Apply platform binary bounds while preserving conditional coherence.

    Numeric/date publication uses a quantile representation of the full pooled
    CDF, so it is an approximation. The unbounded pool remains in the ledger.
    """
    question = pool.question
    if isinstance(question, BinaryQuestion):
        return {"probability": max(0.01, min(0.99, pool.value))}
    if isinstance(question, ConditionalQuestion):
        parent, yes, no, _ = pool.value
        parent_prediction = publication_prediction(parent)
        yes_prediction = publication_prediction(yes)
        no_prediction = publication_prediction(no)
        probability = parent_prediction["probability"]

        def agent_prediction(q: MetaculusQuestion, value: dict) -> dict:
            if not isinstance(q, ConditionalQuestion):
                return value
            return {
                "parent": value["parent"],
                "prediction_yes": agent_prediction(
                    q.question_yes, value["prediction_yes"]
                ),
                "prediction_no": agent_prediction(
                    q.question_no, value["prediction_no"]
                ),
            }

        child = pool_posteriors(
            question.child,
            [
                agent_prediction(question.question_yes, yes_prediction),
                agent_prediction(question.question_no, no_prediction),
            ],
            [probability, 1 - probability],
        )
        return {
            "parent": parent_prediction,
            "prediction_yes": yes_prediction,
            "prediction_no": no_prediction,
            "child": publication_prediction(child),
        }
    return pool.prediction()
