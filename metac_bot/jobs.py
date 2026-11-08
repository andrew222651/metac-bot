"""Shared job snapshots and validated forecast artifacts for the MCP and runner."""

import json
from pathlib import Path
from uuid import UUID, uuid4

from .domain import build_report, load_question, prediction_schema

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "forecast_runs"


def job_path(job_id: str) -> Path:
    if str(UUID(job_id)) != job_id:
        raise ValueError("Invalid job ID")
    return JOBS / job_id


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, indent=2, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def create_job(question) -> str:
    job_id = str(uuid4())
    path = job_path(job_id)
    path.mkdir(parents=True)
    write_json(
        path / "question.json", question.model_dump(mode="json", serialize_as_any=True)
    )
    return job_id


def read_question(job_id: str):
    return load_question(read_json(job_path(job_id) / "question.json"))


def get_job(job_id: str) -> dict:
    question = read_question(job_id)
    data = question.model_dump(mode="json", serialize_as_any=True)

    # The raw API response duplicates the question and can contain community forecasts.
    def clean(value):
        if isinstance(value, dict):
            return {
                k: clean(v)
                for k, v in value.items()
                if k
                not in (
                    "api_json",
                    "community_prediction_at_access_time",
                    "previous_forecasts",
                )
            }
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    return {
        "job_id": job_id,
        "question": clean(data),
        "prediction_schema": prediction_schema(question),
    }


def save_forecast(job_id: str, prediction: dict, research: str, rationale: str) -> dict:
    report = build_report(read_question(job_id), prediction, research, rationale)
    write_json(
        job_path(job_id) / "forecast.json",
        {
            "prediction": prediction,
            "research": research,
            "rationale": rationale,
        },
    )
    return {
        "saved": True,
        "job_id": job_id,
        "prediction": report.model_dump(mode="json")["prediction"],
    }


def read_report(job_id: str):
    data = read_json(job_path(job_id) / "forecast.json")
    return build_report(read_question(job_id), **data)
