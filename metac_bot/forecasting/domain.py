"""Deterministic forecast validation and Metaculus serialization (no LLM calls)."""

from datetime import datetime, timezone
import math
from typing import Any

from forecasting_tools import (
    BinaryQuestion,
    ConditionalPrediction,
    ConditionalQuestion,
    DateQuestion,
    ForecastReport,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
)
from forecasting_tools.data_models.data_organizer import DataOrganizer


def load_question(data: dict) -> MetaculusQuestion:
    data = dict(data)
    if data["question_type"] == "conditional":
        for name in ("parent", "child", "question_yes", "question_no"):
            data[name] = load_question(data[name])
    for cls in DataOrganizer.get_all_question_types():
        if cls.model_fields["question_type"].default == data["question_type"]:
            return cls.model_validate(data)
    raise ValueError(f"Unsupported question type: {data['question_type']}")


def object_schema(properties: dict) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def prediction_schema(question: MetaculusQuestion) -> dict:
    probability = {"type": "number", "minimum": 0, "maximum": 1}
    if isinstance(question, BinaryQuestion):
        return object_schema({"probability": probability})
    if isinstance(question, MultipleChoiceQuestion):
        return object_schema(
            {"options": object_schema({o: probability for o in question.options})}
        )
    if isinstance(question, ConditionalQuestion):
        return object_schema(
            {
                "parent": prediction_schema(question.parent),
                "child": prediction_schema(question.child),
                "prediction_yes": prediction_schema(question.question_yes),
                "prediction_no": prediction_schema(question.question_no),
            }
        )
    if isinstance(question, (NumericQuestion, DateQuestion)):
        value = (
            {"type": "string", "description": "ISO 8601 date/time, UTC"}
            if isinstance(question, DateQuestion)
            else {"type": "number"}
        )
        return object_schema(
            {
                "percentiles": {
                    "type": "array",
                    "minItems": 2,
                    "items": object_schema({"percentile": probability, "value": value}),
                }
            }
        )
    raise ValueError(f"Unsupported question type: {type(question).__name__}")


def finite_number(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Expected a finite number")
    return float(value)


def build_prediction(question: MetaculusQuestion, prediction: dict) -> Any:
    # Validate the exact question-specific shape before translating to SDK models.
    from jsonschema import validate

    validate(prediction, prediction_schema(question))
    if isinstance(question, BinaryQuestion):
        return max(0.01, min(0.99, finite_number(prediction["probability"])))
    if isinstance(question, MultipleChoiceQuestion):
        options = prediction["options"]
        if not math.isclose(
            sum(finite_number(p) for p in options.values()), 1, abs_tol=1e-6
        ):
            raise ValueError("Option probabilities must sum to 1")
        return PredictedOptionList(
            predicted_options=[
                {"option_name": name, "probability": options[name]}
                for name in question.options
            ]
        )
    if isinstance(question, ConditionalQuestion):
        return ConditionalPrediction(
            parent=build_prediction(question.parent, prediction["parent"]),
            child=build_prediction(question.child, prediction["child"]),
            prediction_yes=build_prediction(
                question.question_yes, prediction["prediction_yes"]
            ),
            prediction_no=build_prediction(
                question.question_no, prediction["prediction_no"]
            ),
        )
    if not isinstance(question, (NumericQuestion, DateQuestion)):
        raise ValueError(f"Unsupported question type: {type(question).__name__}")
    percentiles = []
    for item in prediction["percentiles"]:
        value = item["value"]
        if isinstance(question, DateQuestion):
            date = datetime.fromisoformat(value.replace("Z", "+00:00"))
            value = (
                date.replace(tzinfo=timezone.utc).timestamp()
                if date.tzinfo is None
                else date.timestamp()
            )
        percentiles.append(
            Percentile(
                percentile=finite_number(item["percentile"]), value=finite_number(value)
            )
        )
    for left, right in zip(percentiles, percentiles[1:]):
        if left.percentile >= right.percentile or left.value >= right.value:
            raise ValueError("Percentiles and their values must be strictly increasing")
    lower = (
        question.lower_bound.timestamp()
        if isinstance(question, DateQuestion)
        else question.lower_bound
    )
    upper = (
        question.upper_bound.timestamp()
        if isinstance(question, DateQuestion)
        else question.upper_bound
    )
    if not question.open_lower_bound and percentiles[0].value < lower:
        raise ValueError("Forecast value is below the question's closed lower bound")
    if not question.open_upper_bound and percentiles[-1].value > upper:
        raise ValueError("Forecast value is above the question's closed upper bound")
    distribution = NumericDistribution.from_question(percentiles, question)
    distribution.get_cdf()  # Check interpolation now, before accepting a forecast.
    return distribution


def build_report(
    question: MetaculusQuestion, prediction: dict, research: str, rationale: str
) -> ForecastReport:
    if not research.strip() or not rationale.strip():
        raise ValueError("Research and rationale must both be nonempty")
    report_type = DataOrganizer.get_report_type_for_question_type(type(question))
    return report_type(
        question=question,
        prediction=build_prediction(question, prediction),
        explanation=f"# Research\n\n{research}\n\n# Forecast\n\n{rationale}",
    )
