import asyncio
from collections.abc import Iterable
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from forecasting_tools import (
    BinaryQuestion,
    ConditionalQuestion,
    DateQuestion,
    DiscreteQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from metac_bot.forecasting.implementation import jobs
from metac_bot.forecasting.implementation.bayesian import (
    calculate_choice_posterior,
    calculate_posterior,
    candidates_from_proposals,
    validate_stage,
)
from metac_bot.forecasting.implementation.bayesian_jobs import (
    get_stage,
    read_stage,
    record_stage,
)
from metac_bot.forecasting.implementation.ensemble import (
    EnsembleRunner,
    EnsembleSettings,
)
from metac_bot.forecasting.strategy.principles import SUPERFORECASTING_PRINCIPLES
from metac_bot.forecasting.strategy.roles import ROLES
from metac_bot.forecasting.strategy.workflow import run_ensemble
from metac_bot.infrastructure.opencode import InputRequiredError


def proposal(url: str = "https://example.org/a") -> dict:
    return {
        "prior": 0.2,
        "prior_rationale": "Historical baseline",
        "research": "Synthetic test research, not a real source.",
        "evidence": [
            {
                "claim": "Observation",
                "source_urls": [url],
                "source_date": "2026-01-01",
                "likelihood_ratio": 2,
                "rationale": "Illustrative likelihood",
                "limitations": "Uncertain",
            }
        ],
    }


def decision(ids: list[str], lr: float = 2) -> dict:
    return {
        "prior": 0.2,
        "prior_rationale": "Common prior",
        "groups": [
            {
                "candidate_ids": ids,
                "conditional_likelihood_ratio": lr,
                "rationale": "One joint observation",
                "dependence_assumptions": "All overlap",
            }
        ],
        "excluded": [],
        "limitations": "Likelihood is a judgment",
    }


def choice_proposal(options: list[str]) -> dict:
    result = proposal()
    result["prior"] = {option: 1 / len(options) for option in options}
    evidence = result["evidence"][0]
    evidence.pop("likelihood_ratio")
    evidence["likelihoods"] = {
        option: float(len(options) - i) for i, option in enumerate(options)
    }
    return result


def choice_decision(ids: list[str], options: list[str]) -> dict:
    return {
        "prior": {option: 1 / len(options) for option in options},
        "prior_rationale": "Equal prior for the synthetic fixture",
        "groups": [
            {
                "candidate_ids": ids,
                "conditional_likelihoods": {
                    option: float(len(options) - i) for i, option in enumerate(options)
                },
                "rationale": "Joint likelihood",
                "dependence_assumptions": "All overlap",
            }
        ],
        "excluded": [],
        "limitations": "Synthetic likelihoods",
    }


def model_artifact(kind: str) -> dict:
    return {
        "kind": kind,
        "applicable": True,
        "applicability_reason": "Relevant",
        "structure": "X causes Y",
        "equations": "Y = f(X)",
        "inputs_and_sources": "Synthetic assumptions",
        "inference_and_implementation": "Not executed",
        "result_and_uncertainty": "Illustrative",
        "limitations": "Uncalibrated",
    }


class BayesianTests(unittest.TestCase):
    def test_ensemble_settings_choose_role_models_and_stage_timeouts(self) -> None:
        settings = EnsembleSettings(
            model="provider/default",
            timeout=600,
            member_models={"skeptic": "provider/skeptic"},
            adjudicator_model="provider/judge",
            stage_timeouts={"review": 900},
        )
        self.assertEqual(settings.model_for("proposal", "skeptic"), "provider/skeptic")
        self.assertEqual(settings.model_for("review", "generalist"), "provider/default")
        self.assertEqual(settings.model_for("decision", "judge"), "provider/judge")
        self.assertEqual(settings.timeout_for("proposal"), 600)
        self.assertEqual(settings.timeout_for("review"), 900)

    def setUp(self) -> None:
        self.candidates = candidates_from_proposals(
            [
                {"role": "a", "result": proposal("https://example.org/a?utm_source=x")},
                {"role": "b", "result": proposal("https://example.org/a")},
            ]
        )

    def test_bayes_and_recompute_from_prior(self) -> None:
        update = decision(["p1-e1", "p2-e1"])
        first = calculate_posterior(update, self.candidates)
        second = calculate_posterior(update, self.candidates)
        self.assertAlmostEqual(first["posterior"], 1 / 3)
        self.assertEqual(first, second)

    def test_categorical_update_and_scale_invariance(self) -> None:
        options = ["A", "B", "C"]
        candidates = candidates_from_proposals(
            [{"role": "a", "result": choice_proposal(options)}], options
        )
        update = choice_decision(["p1-e1"], options)
        update["prior"] = {"A": 0.5, "B": 0.3, "C": 0.2}
        update["groups"][0]["conditional_likelihoods"] = {"A": 2, "B": 1, "C": 0.5}
        result = calculate_choice_posterior(update, candidates, options)
        self.assertAlmostEqual(result["posterior"]["A"], 1 / 1.4)
        self.assertAlmostEqual(result["posterior"]["B"], 0.3 / 1.4)
        self.assertAlmostEqual(result["posterior"]["C"], 0.1 / 1.4)
        update["groups"][0]["conditional_likelihoods"] = {"A": 4, "B": 2, "C": 1}
        scaled = calculate_choice_posterior(update, candidates, options)
        for option in options:
            self.assertAlmostEqual(
                result["posterior"][option], scaled["posterior"][option]
            )

    def test_categorical_vectors_are_complete_and_valid(self) -> None:
        options = ["A", "B"]
        candidate = candidates_from_proposals(
            [{"role": "a", "result": choice_proposal(options)}], options
        )
        for weights in [
            {"A": 1},
            {"A": 1, "B": 0},
            {"A": 1, "B": math.nan},
            {"A": 1, "C": 1},
        ]:
            update = choice_decision(["p1-e1"], options)
            update["groups"][0]["conditional_likelihoods"] = weights
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                calculate_choice_posterior(update, candidate, options)
        update = choice_decision(["p1-e1"], options)
        update["prior"] = {"A": 0.2, "B": 0.2}
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            calculate_choice_posterior(update, candidate, options)

    def test_categorical_duplicate_sources_are_not_independent(self) -> None:
        options = ["A", "B"]
        candidates = candidates_from_proposals(
            [
                {"role": "a", "result": choice_proposal(options)},
                {"role": "b", "result": choice_proposal(options)},
            ],
            options,
        )
        update = choice_decision(["p1-e1"], options)
        update["groups"].append({**update["groups"][0], "candidate_ids": ["p2-e1"]})
        with self.assertRaisesRegex(ValueError, "same source"):
            calculate_choice_posterior(update, candidates, options)

    def test_duplicate_source_cannot_be_independent(self) -> None:
        update = decision(["p1-e1"])
        update["groups"].append({**update["groups"][0], "candidate_ids": ["p2-e1"]})
        with self.assertRaisesRegex(ValueError, "same source"):
            calculate_posterior(update, self.candidates)

    def test_candidates_must_all_be_assigned_once(self) -> None:
        for update in [
            decision(["p1-e1"]),
            decision(["p1-e1", "p1-e1", "p2-e1"]),
            decision(["p1-e1", "p2-e1", "p3-e1"]),
        ]:
            with self.assertRaises(ValueError):
                calculate_posterior(update, self.candidates)

    def test_excluded_candidate_and_invalid_numbers(self) -> None:
        update = decision(["p1-e1"])
        update["excluded"] = [{"candidate_id": "p2-e1", "reason": "Duplicate"}]
        self.assertAlmostEqual(
            calculate_posterior(update, self.candidates)["posterior"], 1 / 3
        )
        for invalid in [0, -1, math.inf, math.nan]:
            update["groups"][0]["conditional_likelihood_ratio"] = invalid
            with self.assertRaises(ValueError):
                calculate_posterior(update, self.candidates)

    def test_extreme_posterior_is_clipped_only_for_publication(self) -> None:
        update = decision(["p1-e1", "p2-e1"], math.exp(20))
        result = calculate_posterior(update, self.candidates)
        self.assertGreater(result["posterior"], 0.99)
        self.assertEqual(result["published_probability"], 0.99)

    def test_modeler_must_record_the_right_model_kind(self) -> None:
        for role, kind in [
            ("economic-modeler", "economic"),
            ("causal-modeler", "causal"),
        ]:
            with self.subTest(role=role):
                data = proposal()
                with self.assertRaisesRegex(ValueError, "model artifact"):
                    validate_stage("proposal", data, {"role": role})
                data["model"] = model_artifact(
                    "causal" if kind == "economic" else "economic"
                )
                with self.assertRaisesRegex(ValueError, "model artifact"):
                    validate_stage("proposal", data, {"role": role})
                data["model"] = model_artifact(kind)
                self.assertEqual(
                    validate_stage("proposal", data, {"role": role})["model"]["kind"],
                    kind,
                )


class FakeOpenCode:
    def __init__(self, fail_labels: Iterable[str] = ()) -> None:
        self.sessions = {}
        self.fail_labels = set(fail_labels)
        self.active = 0
        self.max_active = 0
        self.prompts = []

    async def create_session(
        self,
        directory: Path,
        title: str,
        model: str | None,
        *,
        agent: str | None = None,
    ) -> str:
        session = f"s{len(self.sessions)}"
        self.sessions[session] = {"title": title, "model": model, "agent": agent}
        return session

    async def prompt(self, session: str, text: str) -> None:
        match = re.search(r"stage ([0-9a-f-]{36})", text)
        assert match is not None
        self.sessions[session]["stage_id"] = match.group(1)
        self.prompts.append(text)

    async def wait(self, session: str, timeout: float) -> None:
        self.sessions[session]["timeout"] = timeout
        self.active += 1
        self.max_active = max(self.active, self.max_active)
        try:
            await asyncio.sleep(0.001)
            title = self.sessions[session]["title"]
            if any(title.endswith(label) for label in self.fail_labels):
                raise RuntimeError("simulated failure")
            stage_id = self.sessions[session]["stage_id"]
            stage = get_stage(stage_id)
            options = stage["question"].get("options")
            if stage["stage"] == "proposal":

                def forecast(question: dict) -> dict:
                    if question["question_type"] == "conditional":
                        return {
                            "parent": forecast(question["parent"]),
                            "prediction_yes": forecast(question["question_yes"]),
                            "prediction_no": forecast(question["question_no"]),
                        }
                    if question["question_type"] in ("numeric", "discrete"):
                        return {
                            "percentiles": [
                                {"percentile": 0.1, "value": 10},
                                {"percentile": 0.9, "value": 90},
                            ]
                        }
                    if question["question_type"] == "date":
                        return {
                            "percentiles": [
                                {"percentile": 0.1, "value": "2027-02-01T00:00:00Z"},
                                {"percentile": 0.9, "value": "2027-11-01T00:00:00Z"},
                            ]
                        }
                    if question["question_type"] == "multiple_choice":
                        names = question["options"]
                        return {
                            "options": {
                                name: (len(names) - i) / sum(range(1, len(names) + 1))
                                for i, name in enumerate(names)
                            }
                        }
                    return {"probability": 1 / 3}

                posterior = forecast(stage["question"])
                prior = (
                    {"options": {option: 1 / len(options) for option in options}}
                    if options
                    else posterior
                )
                result = {
                    "prior": prior,
                    "posterior": posterior,
                    "prior_rationale": "Historical baseline",
                    "reasoning": "Bayesian update",
                    "research": "Synthetic research",
                    "limitations": "Uncertain",
                    "evidence": [
                        {
                            "claim": "Observation",
                            "source_urls": ["https://example.org/a"],
                            "source_date": "2026-01-01",
                            "reasoning": "Joint evidence",
                            "limitations": "Uncertain",
                        }
                    ],
                }
                if stage["context"]["role"] == "economic-modeler":
                    result["model"] = model_artifact("economic")
                elif stage["context"]["role"] == "causal-modeler":
                    result["model"] = model_artifact("causal")
            elif stage["stage"] == "review":
                result = {
                    "accepts_current": True,
                    "posterior": stage["context"]["posteriors"][title.split(": ")[-1]],
                    "critique": "Check assumptions",
                    "reasoning": "Keep",
                    "limitations": "Uncertain",
                }
            else:
                ids = [item["id"] for item in stage["context"]["candidates"]]
                result = choice_decision(ids, options) if options else decision(ids)
            record_stage(stage_id, result)
        finally:
            self.active -= 1


class EnsembleTests(unittest.IsolatedAsyncioTestCase):
    async def test_saved_review_with_input_request_is_used(self) -> None:
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
        ):
            question = BinaryQuestion(question_text="Synthetic")
            job_id = jobs.create_job(question)
            client = FakeOpenCode()
            original_wait = client.wait

            async def review_with_input(session: str, timeout: float) -> None:
                title = client.sessions[session]["title"]
                if title in {
                    "Bayesian review: skeptic",
                    "Bayesian review: economic-modeler",
                    "Bayesian review: causal-modeler",
                }:
                    raise RuntimeError("Review failed")
                await original_wait(session, timeout)
                if title == "Bayesian review: base-rate":
                    raise InputRequiredError("needs input")

            with patch.object(client, "wait", side_effect=review_with_input):
                result = await run_ensemble(client, question, job_id)
            self.assertAlmostEqual(result["prediction"]["probability"], 1 / 3)
            artifact = jobs.read_json(jobs.job_path(job_id) / "ensemble.json")
            recovered = [s for s in artifact["stages"] if s.get("input_recovered")]
            self.assertEqual(len(recovered), 1)
            self.assertEqual(recovered[0]["label"], "base-rate")
            self.assertEqual(recovered[0]["status"], "complete")
            self.assertEqual(len(artifact["rounds"][1]["review_stages"]), 3)
            self.assertTrue(artifact["converged"])

    async def test_input_without_valid_record_fails(self) -> None:
        for malformed in (False, True):
            with (
                self.subTest(malformed=malformed),
                TemporaryDirectory() as directory,
                patch.object(jobs, "JOBS", Path(directory)),
            ):
                question = BinaryQuestion(question_text="Synthetic")
                client = FakeOpenCode()
                runner = EnsembleRunner(
                    client, question, jobs.create_job(question), EnsembleSettings()
                )

                async def request_input(session: str, seconds: float) -> None:
                    if malformed:
                        stage_id = client.sessions[session]["stage_id"]
                        jobs.write_json(
                            jobs.job_path(stage_id) / "stage_result.json",
                            {"posterior": {}},
                        )
                    raise InputRequiredError("needs input")

                with patch.object(client, "wait", side_effect=request_input):
                    result = await runner.run_stage(
                        "proposal", "generalist", {"role": "generalist"}
                    )
                self.assertIsNone(result)
                self.assertEqual(runner.artifacts["stages"][0]["status"], "failed")
                self.assertIn(
                    "InputRequiredError", runner.artifacts["stages"][0]["error"]
                )

    async def test_recorded_stages_survive_session_timeout(self) -> None:
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
        ):
            question = BinaryQuestion(question_text="Synthetic")
            job_id = jobs.create_job(question)
            client = FakeOpenCode()
            original_wait = client.wait

            async def recorded_then_timeout(session: str, timeout: float) -> None:
                await original_wait(session, timeout)
                if client.sessions[session]["title"].endswith("generalist"):
                    raise TimeoutError("session expired")

            with patch.object(client, "wait", side_effect=recorded_then_timeout):
                result = await run_ensemble(client, question, job_id, timeout=10)
            self.assertAlmostEqual(result["prediction"]["probability"], 1 / 3)
            artifact = jobs.read_json(jobs.job_path(job_id) / "ensemble.json")
            recovered = [
                stage for stage in artifact["stages"] if stage.get("timeout_recovered")
            ]
            self.assertEqual(
                {stage["stage"] for stage in recovered}, {"proposal", "review"}
            )
            self.assertTrue(all(stage["status"] == "complete" for stage in recovered))
            self.assertTrue(
                all(
                    "session time budget is 10 seconds" in prompt
                    for prompt in client.prompts
                )
            )

    async def test_timeout_without_valid_stage_does_not_count(self) -> None:
        for malformed in (False, True):
            with (
                self.subTest(malformed=malformed),
                TemporaryDirectory() as directory,
                patch.object(jobs, "JOBS", Path(directory)),
            ):
                question = BinaryQuestion(question_text="Synthetic")
                client = FakeOpenCode()
                runner = EnsembleRunner(
                    client, question, jobs.create_job(question), EnsembleSettings()
                )

                async def timeout(session: str, seconds: float) -> None:
                    if malformed:
                        stage_id = client.sessions[session]["stage_id"]
                        jobs.write_json(
                            jobs.job_path(stage_id) / "stage_result.json",
                            {"posterior": {}},
                        )
                    raise TimeoutError("session expired")

                with patch.object(client, "wait", side_effect=timeout):
                    result = await runner.run_stage(
                        "proposal", "generalist", {"role": "generalist"}
                    )
                self.assertIsNone(result)
                self.assertEqual(runner.artifacts["stages"][0]["status"], "failed")
                self.assertIn("TimeoutError", runner.artifacts["stages"][0]["error"])

    async def test_non_timeout_failure_does_not_recover_recorded_stage(self) -> None:
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
        ):
            question = BinaryQuestion(question_text="Synthetic")
            client = FakeOpenCode()
            runner = EnsembleRunner(
                client, question, jobs.create_job(question), EnsembleSettings()
            )
            original_wait = client.wait

            async def failed_after_record(session: str, timeout: float) -> None:
                await original_wait(session, timeout)
                raise RuntimeError("permission required")

            with patch.object(client, "wait", side_effect=failed_after_record):
                result = await runner.run_stage(
                    "proposal", "generalist", {"role": "generalist"}
                )
            self.assertIsNone(result)
            self.assertEqual(runner.artifacts["stages"][0]["status"], "failed")

    async def test_revisions_are_pooled_and_prior_reviews_are_shared(self) -> None:
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
        ):
            question = BinaryQuestion(question_text="Synthetic")
            job_id = jobs.create_job(question)
            client = FakeOpenCode()
            original_wait = client.wait
            seen_reviews = []

            async def revise(session: str, timeout: float) -> None:
                if "review:" not in client.sessions[session]["title"]:
                    return await original_wait(session, timeout)
                stage_id = client.sessions[session]["stage_id"]
                context = get_stage(stage_id)["context"]
                role = client.sessions[session]["title"].split(": ")[-1]
                seen_reviews.append(
                    (context["round"], len(context["previous_reviews"]))
                )
                posterior = context["posteriors"][role]
                if context["round"] == 1 and role == "generalist":
                    posterior = {"probability": 0.9}
                record_stage(
                    stage_id,
                    {
                        "accepts_current": context["round"] > 1,
                        "posterior": posterior,
                        "critique": "Consider the evidence",
                        "reasoning": "Updated after review",
                        "limitations": "Uncertain",
                    },
                )

            with patch.object(client, "wait", side_effect=revise):
                result = await run_ensemble(client, question, job_id, review_rounds=2)
            self.assertAlmostEqual(
                result["prediction"]["probability"],
                (0.9 + (len(ROLES) - 1) / 3) / len(ROLES),
            )
            self.assertEqual(seen_reviews.count((1, 0)), len(ROLES))
            self.assertEqual(seen_reviews.count((2, len(ROLES))), len(ROLES))
            artifact = __import__("json").loads(
                (jobs.job_path(job_id) / "ensemble.json").read_text()
            )
            self.assertEqual(len(artifact["rounds"]), 3)
            self.assertTrue(artifact["converged"])

    async def test_numeric_date_and_conditional_ensembles_save_reports(self) -> None:
        numeric = NumericQuestion(
            question_text="How many?",
            lower_bound=0,
            upper_bound=100,
            open_lower_bound=True,
            open_upper_bound=True,
        )
        discrete = DiscreteQuestion(
            question_text="How many discrete outcomes?",
            lower_bound=0,
            upper_bound=100,
            open_lower_bound=False,
            open_upper_bound=False,
            cdf_size=11,
        )
        date = DateQuestion(
            question_text="When?",
            lower_bound="2027-01-01T00:00:00Z",
            upper_bound="2028-01-01T00:00:00Z",
            open_lower_bound=False,
            open_upper_bound=False,
        )
        parent = BinaryQuestion(question_text="Condition?")
        conditional = ConditionalQuestion(
            question_text="Conditional?",
            parent=parent,
            child=numeric,
            question_yes=numeric,
            question_no=numeric,
        )
        for question in (numeric, discrete, date, conditional):
            with (
                self.subTest(question=type(question).__name__),
                TemporaryDirectory() as directory,
                patch.object(jobs, "JOBS", Path(directory)),
            ):
                job_id = jobs.create_job(question)
                client = FakeOpenCode()
                result = await run_ensemble(client, question, job_id)
                jobs.save_forecast(job_id, **result)
                report = jobs.read_report(job_id)
                self.assertIs(type(report.question), type(question))
                self.assertEqual(len(client.sessions), 2 * len(ROLES))

    async def test_multiple_choice_runs_complete_ensemble(self) -> None:
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
        ):
            question = MultipleChoiceQuestion(
                question_text="Synthetic", options=["A", "B", "C", "D"]
            )
            job_id = jobs.create_job(question)
            client = FakeOpenCode()
            result = await run_ensemble(client, question, job_id, concurrency=2)
            self.assertEqual(len(client.sessions), 2 * len(ROLES))
            expected = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
            for option in expected:
                self.assertAlmostEqual(
                    result["prediction"]["options"][option], expected[option]
                )
            artifact = __import__("json").loads(
                (jobs.job_path(job_id) / "ensemble.json").read_text()
            )
            self.assertTrue(artifact["converged"])
            self.assertEqual(artifact["rounds"][1]["posterior_change"], 0)
            stage = get_stage(artifact["stages"][0]["stage_id"])
            self.assertEqual(
                stage["result_schema"]["properties"]["prior"]["properties"]["options"][
                    "required"
                ],
                question.options,
            )

    async def test_complete_round_and_independence(self) -> None:
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
        ):
            question = BinaryQuestion(question_text="Synthetic")
            job_id = jobs.create_job(question)
            client = FakeOpenCode()
            result = await run_ensemble(
                client,
                question,
                job_id,
                concurrency=2,
                model="provider/default",
                member_models={"skeptic": "provider/skeptic"},
                adjudicator_model="provider/judge",
                timeout=10,
                stage_timeouts={"proposal": 20, "review": 30, "decision": 40},
            )
            self.assertAlmostEqual(result["prediction"]["probability"], 1 / 3)
            self.assertLessEqual(client.max_active, 2)
            self.assertEqual(len(client.sessions), 2 * len(ROLES))
            for session in client.sessions.values():
                stage = session["title"].split()[1].rstrip(":")
                self.assertEqual(
                    session["timeout"],
                    {"proposal": 20, "review": 30, "decision": 40}[stage],
                )
            self.assertTrue(
                all(SUPERFORECASTING_PRINCIPLES in prompt for prompt in client.prompts)
            )
            skeptic = [
                s for s in client.sessions.values() if s["title"].endswith("skeptic")
            ]
            self.assertEqual(
                [s["model"] for s in skeptic], ["provider/skeptic", "provider/skeptic"]
            )
            self.assertFalse(
                any("decision:" in s["title"] for s in client.sessions.values())
            )
            artifact = __import__("json").loads(
                (jobs.job_path(job_id) / "ensemble.json").read_text()
            )
            self.assertTrue(artifact["converged"])
            self.assertEqual(artifact["rounds"][1]["posterior_change"], 0)
            self.assertIn('"kind": "economic"', result["rationale"])
            self.assertIn('"kind": "causal"', result["rationale"])
            self.assertTrue(
                any(
                    "economic-modeling and modeling-python skills" in prompt
                    for prompt in client.prompts
                )
            )
            self.assertTrue(
                any(
                    "bayesian-causal-modeling and modeling-python skills" in prompt
                    for prompt in client.prompts
                )
            )
            for role in ("economic-modeler", "causal-modeler"):
                sessions = [
                    s for s in client.sessions.values() if s["title"].endswith(role)
                ]
                self.assertEqual([s["agent"] for s in sessions], [role, role])
            for stage in artifact["stages"][: len(ROLES)]:
                context = get_stage(stage["stage_id"])["context"]
                self.assertNotIn("proposals", context)
                self.assertNotIn("prior", context)

    async def test_one_successful_proposal_and_review_is_enough(self) -> None:
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
        ):
            question = BinaryQuestion(question_text="Synthetic")
            job_id = jobs.create_job(question)
            client = FakeOpenCode(set(ROLES) - {"generalist"})
            result = await run_ensemble(client, question, job_id)
            self.assertAlmostEqual(result["prediction"]["probability"], 1 / 3)
            artifact = jobs.read_json(jobs.job_path(job_id) / "ensemble.json")
            self.assertEqual(artifact["weights"], {"generalist": 1.0})
            self.assertTrue(artifact["converged"])
            self.assertIn("1 of 6 proposals completed", result["rationale"])

    async def test_one_successful_review_is_enough(self) -> None:
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
        ):
            question = BinaryQuestion(question_text="Synthetic")
            job_id = jobs.create_job(question)
            client = FakeOpenCode()
            original_wait = client.wait

            async def fail_other_reviews(session: str, timeout: float) -> None:
                title = client.sessions[session]["title"]
                if title.startswith("Bayesian review:") and not title.endswith(
                    "generalist"
                ):
                    raise RuntimeError("Review failed")
                await original_wait(session, timeout)

            with patch.object(client, "wait", side_effect=fail_other_reviews):
                result = await run_ensemble(client, question, job_id)
            artifact = jobs.read_json(jobs.job_path(job_id) / "ensemble.json")
            self.assertTrue(artifact["converged"])
            self.assertEqual(len(artifact["rounds"][1]["review_stages"]), 1)
            self.assertIn("5 reviewers did not complete", result["rationale"])

    async def test_all_proposals_fail_without_forecast(self) -> None:
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
        ):
            question = BinaryQuestion(question_text="Synthetic")
            job_id = jobs.create_job(question)
            with self.assertRaisesRegex(
                RuntimeError, "No Bayesian proposals succeeded"
            ):
                await run_ensemble(
                    FakeOpenCode(ROLES),
                    question,
                    job_id,
                )
            self.assertFalse((jobs.job_path(job_id) / "forecast.json").exists())

    async def test_any_role_can_fail_proposal(self) -> None:
        for role in ROLES:
            with (
                self.subTest(role=role),
                TemporaryDirectory() as directory,
                patch.object(jobs, "JOBS", Path(directory)),
            ):
                question = BinaryQuestion(question_text="Synthetic")
                job_id = jobs.create_job(question)
                result = await run_ensemble(FakeOpenCode({role}), question, job_id)
                self.assertAlmostEqual(result["prediction"]["probability"], 1 / 3)
                artifact = jobs.read_json(jobs.job_path(job_id) / "ensemble.json")
                self.assertNotIn(role, artifact["weights"])
                self.assertEqual(len(artifact["weights"]), len(ROLES) - 1)
                self.assertTrue(artifact["converged"])

    async def test_any_role_can_fail_review(self) -> None:
        for role in ROLES:
            with (
                self.subTest(role=role),
                TemporaryDirectory() as directory,
                patch.object(jobs, "JOBS", Path(directory)),
            ):
                question = BinaryQuestion(question_text="Synthetic")
                job_id = jobs.create_job(question)
                client = FakeOpenCode()
                original_wait = client.wait

                async def fail_review(session: str, timeout: float) -> None:
                    title = client.sessions[session]["title"]
                    if title == f"Bayesian review: {role}":
                        raise RuntimeError("Review failure")
                    await original_wait(session, timeout)
                    if title == f"Bayesian proposal: {role}":
                        stage_id = client.sessions[session]["stage_id"]
                        proposal_result = read_stage(stage_id)
                        proposal_result["posterior"] = {"probability": 0.5}
                        record_stage(stage_id, proposal_result)

                with patch.object(client, "wait", side_effect=fail_review):
                    result = await run_ensemble(client, question, job_id)
                self.assertAlmostEqual(
                    result["prediction"]["probability"],
                    (0.5 + (len(ROLES) - 1) / 3) / len(ROLES),
                )
                artifact = jobs.read_json(jobs.job_path(job_id) / "ensemble.json")
                self.assertEqual(
                    artifact["rounds"][1]["posteriors"][role],
                    artifact["rounds"][0]["posteriors"][role],
                )
                self.assertEqual(
                    len(artifact["rounds"][1]["review_stages"]), len(ROLES) - 1
                )
                self.assertTrue(artifact["converged"])

    async def test_all_reviews_fail_uses_unreviewed_pool(self) -> None:
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
        ):
            question = BinaryQuestion(question_text="Synthetic")
            job_id = jobs.create_job(question)
            client = FakeOpenCode()
            original_wait = client.wait

            async def fail_reviews(session: str, timeout: float) -> None:
                if "review:" in client.sessions[session]["title"]:
                    raise RuntimeError("Review failure")
                await original_wait(session, timeout)

            with patch.object(client, "wait", side_effect=fail_reviews):
                result = await run_ensemble(client, question, job_id)
            self.assertAlmostEqual(result["prediction"]["probability"], 1 / 3)
            artifact = jobs.read_json(jobs.job_path(job_id) / "ensemble.json")
            self.assertFalse(artifact["converged"])
            self.assertTrue(artifact["reviews_unavailable"])
            self.assertIsNone(artifact["rounds"][1]["all_reviewers_accept_previous"])
            self.assertIn("No reviewers completed the final round", result["rationale"])
            self.assertIn("without establishing convergence", result["rationale"])

    async def test_later_round_without_reviews_retains_earlier_revision(self) -> None:
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
        ):
            question = BinaryQuestion(question_text="Synthetic")
            job_id = jobs.create_job(question)
            client = FakeOpenCode()
            original_wait = client.wait

            async def revise_then_fail(session: str, timeout: float) -> None:
                if "review:" not in client.sessions[session]["title"]:
                    await original_wait(session, timeout)
                    return
                stage_id = client.sessions[session]["stage_id"]
                context = get_stage(stage_id)["context"]
                if context["round"] == 2:
                    raise RuntimeError("Review failed")
                record_stage(
                    stage_id,
                    {
                        "accepts_current": False,
                        "posterior": {"probability": 0.6},
                        "critique": "Revise",
                        "reasoning": "Updated",
                        "limitations": "Uncertain",
                    },
                )

            with patch.object(client, "wait", side_effect=revise_then_fail):
                result = await run_ensemble(client, question, job_id, review_rounds=2)
            self.assertAlmostEqual(result["prediction"]["probability"], 0.6)
            artifact = jobs.read_json(jobs.job_path(job_id) / "ensemble.json")
            self.assertFalse(artifact["converged"])
            self.assertTrue(artifact["reviews_unavailable"])
            self.assertIn("earlier completed reviews", result["rationale"])

    async def test_stable_posterior_without_acceptance_fails_closed(self) -> None:
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
        ):
            question = BinaryQuestion(question_text="Synthetic")
            job_id = jobs.create_job(question)
            client = FakeOpenCode()
            original_wait = client.wait

            async def disagree(session: str, timeout: float) -> None:
                if "review:" not in client.sessions[session]["title"]:
                    return await original_wait(session, timeout)
                stage_id = client.sessions[session]["stage_id"]
                record_stage(
                    stage_id,
                    {
                        "accepts_current": False,
                        "posterior": {"probability": 1 / 3},
                        "critique": "Disagree",
                        "reasoning": "Revise prior",
                        "limitations": "Uncertain",
                    },
                )

            with patch.object(client, "wait", side_effect=disagree):
                with self.assertRaisesRegex(RuntimeError, "did not converge"):
                    await run_ensemble(client, question, job_id, review_rounds=1)
            self.assertFalse((jobs.job_path(job_id) / "forecast.json").exists())
