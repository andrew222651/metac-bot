"""Orchestrate independent evidence proposals and shared-posterior reviews."""

import asyncio
from dataclasses import dataclass, field
import json
from typing import Any, Literal

from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion

from .agent import run_agent
from .bayesian import (
    calculate_choice_posterior,
    calculate_posterior,
    candidates_from_proposals,
)
from .bayesian_jobs import create_stage, read_stage
from .config import MODEL_ROLES, ROLES
from .domain import build_prediction
from .jobs import job_path, write_json

Stage = Literal["proposal", "review", "decision"]


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
        question: BinaryQuestion | MultipleChoiceQuestion,
        job_id: str,
        settings: EnsembleSettings,
    ) -> None:
        self.opencode = opencode
        self.question = question
        self.settings = settings
        self.options = (
            question.options if isinstance(question, MultipleChoiceQuestion) else None
        )
        self.path = job_path(job_id)
        self.semaphore = asyncio.Semaphore(settings.concurrency)
        self.artifacts: dict[str, Any] = {
            "stages": [],
            "rounds": [],
            "min_members": settings.min_members,
            "convergence_tolerance": settings.tolerance,
        }

    def _persist(self) -> None:
        write_json(self.path / "ensemble.json", self.artifacts)

    def _describe_failures(
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
        if stage == "proposal":
            instructions = ROLES.get(label, "")
        elif stage == "review":
            instructions = (
                f"Review the supplied work from the {label} perspective; "
                "do not redo the initial research or rebuild the model."
            )
        else:
            instructions = ""
        standalone = (
            "This is a standalone user-supplied question: do not look it up on Metaculus "
            "or ask for a Metaculus token."
            if self.question.custom_metadata.get("one_off")
            else ""
        )
        return (
            f"Load the bayesian-ensemble skill and complete your {stage} stage {stage_id}. "
            f"{instructions} "
            "Use metaculus.get_bayesian_stage and metaculus.record_bayesian_stage. "
            "Follow the provided JSON schema, correct validation errors, and record your result. "
            "The Python runner computes and stores the forecast; do not publish or edit files. "
            f"{standalone}"
        )

    async def _run_stage(self, stage: Stage, label: str, context: dict) -> dict | None:
        stage_id = create_stage(self.question, stage, context)
        entry = {
            "stage_id": stage_id,
            "stage": stage,
            "label": label,
            "status": "created",
            "round": context.get("round"),
        }
        self.artifacts["stages"].append(entry)
        self._persist()
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
                    self._persist()

                session = await run_agent(
                    self.opencode,
                    title=f"Bayesian {stage}: {label}",
                    model=model,
                    timeout=timeout,
                    on_session=record_session,
                    agent=agent,
                    prompt=self._prompt(stage, label, stage_id),
                )
                result = read_stage(stage_id)
                entry["status"] = "complete"
                self._persist()
                return {
                    "role": label,
                    "stage_id": stage_id,
                    "session_id": session,
                    "result": result,
                }
        except Exception as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            self._persist()
            return None

    def _require_quorum(
        self,
        stages: list[dict | None],
        stage: Literal["proposal", "review"],
        round_number: int | None = None,
    ) -> list[dict]:
        completed = [result for result in stages if result is not None]
        if len(completed) < self.settings.min_members:
            description = (
                "proposals"
                if stage == "proposal"
                else f"reviews succeeded in round {round_number}"
            )
            if stage == "proposal":
                description += " succeeded"
            raise RuntimeError(
                f"Only {len(completed)} Bayesian {description}; need {self.settings.min_members}. "
                f"{self._describe_failures(stage, round_number=round_number)}"
            )
        required_roles = set(MODEL_ROLES) & set(ROLES)
        missing = required_roles - {result["role"] for result in completed}
        if missing:
            noun = "proposals" if stage == "proposal" else "reviews"
            raise RuntimeError(
                f"Required modeling {noun} failed: {', '.join(sorted(missing))}. "
                f"{self._describe_failures(stage, missing, round_number)}"
            )
        return completed

    async def _decide(
        self, label: str, context: dict, candidates: list[dict]
    ) -> tuple[dict, dict]:
        stage = await self._run_stage("decision", label, context)
        if stage is None:
            if context["round"] == 0:
                message = "Bayesian initial adjudication failed"
            else:
                message = f"Bayesian adjudication failed in round {context['round']}"
            raise RuntimeError(
                f"{message}: {self._describe_failures('decision', {label}, context['round'])}"
            )
        if self.options is None:
            result = calculate_posterior(stage["result"], candidates)
        else:
            result = calculate_choice_posterior(
                stage["result"], candidates, self.options
            )
        return stage, result

    def _posterior_change(self, previous: dict, current: dict) -> float:
        if self.options is None:
            return abs(current["posterior"] - previous["posterior"])
        return max(
            abs(current["posterior"][option] - previous["posterior"][option])
            for option in self.options
        )

    def _prediction(self, current: dict) -> dict:
        if self.options is None:
            return {"probability": current["published_probability"]}
        return {
            "options": build_prediction(
                self.question, {"options": current["posterior"]}
            ).to_dict()
        }

    async def run(self) -> dict:
        """Return research, rationale and probabilities; never publish from here."""
        proposals = self._require_quorum(
            await asyncio.gather(
                *(
                    self._run_stage(
                        "proposal", role, {"role": role, "instructions": instructions}
                    )
                    for role, instructions in ROLES.items()
                )
            ),
            "proposal",
        )
        candidates = candidates_from_proposals(proposals, self.options)
        initial, current = await self._decide(
            "initial-adjudicator",
            {"proposals": proposals, "candidates": candidates, "round": 0},
            candidates,
        )
        self.artifacts["rounds"].append(
            {"round": 0, "decision_stage": initial["stage_id"], **current}
        )
        self._persist()

        converged = False
        for round_number in range(1, self.settings.review_rounds + 1):
            context = {
                "round": round_number,
                "proposals": proposals,
                "candidates": candidates,
                "current": current,
            }
            reviews = self._require_quorum(
                await asyncio.gather(
                    *(
                        self._run_stage("review", proposal["role"], context)
                        for proposal in proposals
                    )
                ),
                "review",
                round_number,
            )
            final, updated = await self._decide(
                f"adjudicator-round-{round_number}",
                {**context, "reviews": reviews},
                candidates,
            )
            delta = self._posterior_change(current, updated)
            current = updated
            accepted = all(review["result"]["accepts_current"] for review in reviews)
            converged = accepted and delta <= self.settings.tolerance
            self.artifacts["rounds"].append(
                {
                    "round": round_number,
                    "decision_stage": final["stage_id"],
                    "review_stages": [item["stage_id"] for item in reviews],
                    "all_reviewers_accept_previous": accepted,
                    "posterior_change": delta,
                    **current,
                }
            )
            self._persist()
            if converged:
                break

        prediction = self._prediction(current)
        self.artifacts.update(
            converged=converged,
            posterior=current["posterior"],
            published_prediction=prediction,
        )
        self._persist()
        if not converged:
            raise RuntimeError(
                f"Bayesian ensemble did not converge within {self.settings.review_rounds} review rounds; no forecast published"
            )
        return _make_result(proposals, candidates, current, prediction, converged)


def _make_result(
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


async def run_ensemble(
    opencode,
    question,
    job_id,
    *,
    model=None,
    timeout=900,
    concurrency=2,
    min_members=3,
    review_rounds=3,
    tolerance=0.01,
    member_models=None,
    adjudicator_model=None,
    stage_timeouts=None,
):
    settings = EnsembleSettings(
        model=model,
        timeout=timeout,
        concurrency=concurrency,
        min_members=min_members,
        review_rounds=review_rounds,
        tolerance=tolerance,
        member_models=member_models or {},
        adjudicator_model=adjudicator_model,
        stage_timeouts=stage_timeouts or {},
    )
    return await EnsembleRunner(opencode, question, job_id, settings).run()
