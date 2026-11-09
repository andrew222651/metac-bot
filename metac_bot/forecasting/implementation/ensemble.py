"""Execute ensemble stages and persist their progress."""

import asyncio
from dataclasses import dataclass, field
from typing import Any

from forecasting_tools import MetaculusQuestion

from ..strategy.prompts import Stage, stage_prompt
from ..strategy.roles import MODEL_ROLES
from .agent import run_agent
from .bayesian_jobs import create_stage, read_stage
from .jobs import job_path, write_json


@dataclass(frozen=True)
class EnsembleSettings:
    """Execution settings shared by all stages of one ensemble."""

    model: str | None = None
    timeout: float = 900
    concurrency: int = 2
    min_members: int = 3
    review_rounds: int = 3
    tolerance: float = 0.01
    member_models: dict[str, str] = field(default_factory=dict)
    adjudicator_model: str | None = None
    stage_timeouts: dict[str, float | None] = field(default_factory=dict)

    def model_for(self, stage: Stage, role: str) -> str | None:
        override = (
            self.adjudicator_model
            if stage == "decision"
            else self.member_models.get(role)
        )
        return override or self.model

    def timeout_for(self, stage: Stage) -> float:
        return self.stage_timeouts.get(stage) or self.timeout


class EnsembleRunner:
    """Run stages and keep the on-disk progress ledger in sync."""

    def __init__(
        self,
        opencode: Any,
        question: MetaculusQuestion,
        job_id: str,
        settings: EnsembleSettings,
    ) -> None:
        self.opencode = opencode
        self.question = question
        self.settings = settings
        self.path = job_path(job_id)
        self.semaphore = asyncio.Semaphore(settings.concurrency)
        self.artifacts: dict[str, Any] = {
            "stages": [],
            "rounds": [],
            "min_members": settings.min_members,
            "convergence_tolerance": settings.tolerance,
        }

    def persist(self) -> None:
        write_json(self.path / "ensemble.json", self.artifacts)

    def describe_failures(
        self,
        stage: Stage,
        labels: set[str] | None = None,
        round_number: int | None = None,
    ) -> str:
        failures = [
            entry
            for entry in self.artifacts["stages"]
            if entry["stage"] == stage
            and entry["status"] == "failed"
            and (labels is None or entry["label"] in labels)
            and (round_number is None or entry.get("round") == round_number)
        ]
        return "; ".join(
            f"{entry['label']}: {entry.get('error', 'unknown error')}"
            for entry in failures
        )

    def _prompt(self, stage: Stage, label: str, stage_id: str) -> str:
        return stage_prompt(
            stage,
            label,
            stage_id,
            one_off=bool(self.question.custom_metadata.get("one_off")),
        )

    async def run_stage(self, stage: Stage, label: str, context: dict) -> dict | None:
        stage_id = create_stage(
            self.question, stage, {**context, "method": "distributional"}
        )
        entry = {
            "stage_id": stage_id,
            "stage": stage,
            "label": label,
            "status": "created",
            "round": context.get("round"),
        }
        self.artifacts["stages"].append(entry)
        self.persist()
        try:
            async with self.semaphore:
                model = self.settings.model_for(stage, label)
                timeout = self.settings.timeout_for(stage)
                agent = (
                    label
                    if stage in ("proposal", "review") and label in MODEL_ROLES
                    else None
                )

                def record_session(session: str) -> None:
                    entry.update(
                        session_id=session,
                        status="running",
                        model=model,
                        agent=agent,
                        timeout=timeout,
                    )
                    self.persist()

                try:
                    session = await run_agent(
                        self.opencode,
                        title=f"Bayesian {stage}: {label}",
                        model=model,
                        timeout=timeout,
                        on_session=record_session,
                        agent=agent,
                        prompt=self._prompt(stage, label, stage_id),
                    )
                except TimeoutError as timeout_error:
                    if "session_id" not in entry:
                        raise
                    try:
                        result = read_stage(stage_id)
                    except Exception:
                        raise timeout_error from None
                    session = entry["session_id"]
                    entry["timeout_recovered"] = True
                else:
                    result = read_stage(stage_id)
                entry.update(status="complete", result=result)
                self.persist()
                return {
                    "role": label,
                    "stage_id": stage_id,
                    "session_id": session,
                    "result": result,
                }
        except Exception as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            self.persist()
            return None
