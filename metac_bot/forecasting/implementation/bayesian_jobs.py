"""Isolated stage artifacts exposed to agents by opaque UUID."""

from forecasting_tools import MetaculusQuestion, MultipleChoiceQuestion

from . import jobs
from .bayesian import CHOICE_SCHEMAS, SCHEMAS, validate_stage
from .deliberation import stage_schema, validate_result


def stage_options(stage_id: str) -> list[str] | None:
    question = jobs.read_question(stage_id)
    return question.options if isinstance(question, MultipleChoiceQuestion) else None


def _stage_state(stage_id: str) -> tuple[dict, list[str] | None]:
    path = jobs.job_path(stage_id)
    return jobs.read_json(path / "stage.json"), stage_options(stage_id)


def create_stage(question: MetaculusQuestion, stage: str, context: dict) -> str:
    if stage not in SCHEMAS:
        raise ValueError("Unknown stage")
    stage_id = jobs.create_job(question)
    jobs.write_json(
        jobs.job_path(stage_id) / "stage.json", {"stage": stage, "context": context}
    )
    return stage_id


def get_stage(stage_id: str) -> dict:
    data, options = _stage_state(stage_id)
    if data["context"].get("method") == "distributional":
        return {
            "stage_id": stage_id,
            "stage": data["stage"],
            "context": data["context"],
            "question": jobs.get_job(stage_id)["question"],
            "result_schema": stage_schema(jobs.read_question(stage_id), data["stage"]),
        }
    schemas = CHOICE_SCHEMAS if options is not None else SCHEMAS
    schema = schemas[data["stage"]].model_json_schema()
    if options is not None and data["stage"] in ("proposal", "decision"):

        def vector_schema(*, prior: bool) -> dict:
            value = {"type": "number", "exclusiveMinimum": 0}
            if prior:
                value["maximum"] = 1
            return {
                "type": "object",
                "properties": {option: dict(value) for option in options},
                "required": options,
                "additionalProperties": False,
            }

        schema["properties"]["prior"] = vector_schema(prior=True)
        if data["stage"] == "proposal":
            schema["$defs"]["ChoiceEvidence"]["properties"]["likelihoods"] = (
                vector_schema(prior=False)
            )
        else:
            schema["$defs"]["ChoiceEvidenceGroup"]["properties"][
                "conditional_likelihoods"
            ] = vector_schema(prior=False)
    return {
        "stage_id": stage_id,
        "stage": data["stage"],
        "context": data["context"],
        "question": jobs.get_job(stage_id)["question"],
        "result_schema": schema,
    }


def record_stage(stage_id: str, result: dict) -> dict:
    path = jobs.job_path(stage_id)
    stage, options = _stage_state(stage_id)
    if stage["context"].get("method") == "distributional":
        validated = validate_result(
            jobs.read_question(stage_id), stage["stage"], result, stage["context"]
        )
    else:
        validated = validate_stage(stage["stage"], result, stage["context"], options)
    jobs.write_json(path / "stage_result.json", validated)
    return {"saved": True, "stage_id": stage_id}


def read_stage(stage_id: str) -> dict:
    path = jobs.job_path(stage_id)
    stage, options = _stage_state(stage_id)
    result = jobs.read_json(path / "stage_result.json")
    if stage["context"].get("method") == "distributional":
        return validate_result(
            jobs.read_question(stage_id), stage["stage"], result, stage["context"]
        )
    return validate_stage(stage["stage"], result, stage["context"], options)
