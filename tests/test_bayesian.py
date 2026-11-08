import asyncio
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion
from metac_bot import jobs
from metac_bot.bayesian import (
    calculate_choice_posterior,
    calculate_posterior,
    candidates_from_proposals,
    validate_stage,
)
from metac_bot.bayesian_jobs import get_stage, record_stage
from metac_bot.ensemble import EnsembleSettings, run_ensemble
from metac_bot.forecasting_principles import SUPERFORECASTING_PRINCIPLES


def proposal(url="https://example.org/a"):
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


def decision(ids, lr=2):
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


def choice_proposal(options):
    result = proposal()
    result["prior"] = {option: 1 / len(options) for option in options}
    evidence = result["evidence"][0]
    evidence.pop("likelihood_ratio")
    evidence["likelihoods"] = {
        option: float(len(options) - i) for i, option in enumerate(options)
    }
    return result


def choice_decision(ids, options):
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


def model_artifact(kind):
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
    def test_ensemble_settings_choose_role_models_and_stage_timeouts(self):
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

    def setUp(self):
        self.candidates = candidates_from_proposals(
            [
                {"role": "a", "result": proposal("https://example.org/a?utm_source=x")},
                {"role": "b", "result": proposal("https://example.org/a")},
            ]
        )

    def test_bayes_and_recompute_from_prior(self):
        update = decision(["p1-e1", "p2-e1"])
        first = calculate_posterior(update, self.candidates)
        second = calculate_posterior(update, self.candidates)
        self.assertAlmostEqual(first["posterior"], 1 / 3)
        self.assertEqual(first, second)

    def test_categorical_update_and_scale_invariance(self):
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

    def test_categorical_vectors_are_complete_and_valid(self):
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

    def test_categorical_duplicate_sources_are_not_independent(self):
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

    def test_duplicate_source_cannot_be_independent(self):
        update = decision(["p1-e1"])
        update["groups"].append({**update["groups"][0], "candidate_ids": ["p2-e1"]})
        with self.assertRaisesRegex(ValueError, "same source"):
            calculate_posterior(update, self.candidates)

    def test_candidates_must_all_be_assigned_once(self):
        for update in [
            decision(["p1-e1"]),
            decision(["p1-e1", "p1-e1", "p2-e1"]),
            decision(["p1-e1", "p2-e1", "p3-e1"]),
        ]:
            with self.assertRaises(ValueError):
                calculate_posterior(update, self.candidates)

    def test_excluded_candidate_and_invalid_numbers(self):
        update = decision(["p1-e1"])
        update["excluded"] = [{"candidate_id": "p2-e1", "reason": "Duplicate"}]
        self.assertAlmostEqual(
            calculate_posterior(update, self.candidates)["posterior"], 1 / 3
        )
        for invalid in [0, -1, math.inf, math.nan]:
            update["groups"][0]["conditional_likelihood_ratio"] = invalid
            with self.assertRaises(ValueError):
                calculate_posterior(update, self.candidates)

    def test_extreme_posterior_is_clipped_only_for_publication(self):
        update = decision(["p1-e1", "p2-e1"], math.exp(20))
        result = calculate_posterior(update, self.candidates)
        self.assertGreater(result["posterior"], 0.99)
        self.assertEqual(result["published_probability"], 0.99)

    def test_modeler_must_record_the_right_model_kind(self):
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
    def __init__(self, fail_labels=()):
        self.sessions = {}
        self.fail_labels = set(fail_labels)
        self.active = 0
        self.max_active = 0
        self.prompts = []

    async def create_session(self, directory, title, model, *, agent=None):
        session = f"s{len(self.sessions)}"
        self.sessions[session] = {"title": title, "model": model, "agent": agent}
        return session

    async def prompt(self, session, text):
        match = re.search(r"stage ([0-9a-f-]{36})", text)
        assert match is not None
        self.sessions[session]["stage_id"] = match.group(1)
        self.prompts.append(text)

    async def wait(self, session, timeout):
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
                result = choice_proposal(options) if options else proposal()
                if stage["context"]["role"] == "economic-modeler":
                    result["model"] = model_artifact("economic")
                elif stage["context"]["role"] == "causal-modeler":
                    result["model"] = model_artifact("causal")
            elif stage["stage"] == "review":
                result = {
                    "accepts_current": True,
                    "critique": "Check assumptions",
                    "proposed_changes": "Keep",
                    "limitations": "Uncertain",
                }
            else:
                ids = [item["id"] for item in stage["context"]["candidates"]]
                result = choice_decision(ids, options) if options else decision(ids)
            record_stage(stage_id, result)
        finally:
            self.active -= 1


class EnsembleTests(unittest.IsolatedAsyncioTestCase):
    async def test_multiple_choice_runs_complete_ensemble(self):
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
            self.assertEqual(len(client.sessions), 16)
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
                stage["result_schema"]["properties"]["prior"]["required"],
                question.options,
            )

    async def test_complete_round_and_independence(self):
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
            self.assertEqual(len(client.sessions), 16)
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
            self.assertTrue(
                all(
                    s["model"] == "provider/judge"
                    for s in client.sessions.values()
                    if "decision:" in s["title"]
                )
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
            for stage in artifact["stages"][:7]:
                context = get_stage(stage["stage_id"])["context"]
                self.assertNotIn("proposals", context)
                self.assertNotIn("prior", context)

    async def test_quorum_failure_never_creates_forecast(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
        ):
            question = BinaryQuestion(question_text="Synthetic")
            job_id = jobs.create_job(question)
            with self.assertRaisesRegex(RuntimeError, "Only 2 Bayesian proposals"):
                await run_ensemble(
                    FakeOpenCode(
                        {
                            "generalist",
                            "skeptic",
                            "quantitative",
                            "economic-modeler",
                            "causal-modeler",
                        }
                    ),
                    question,
                    job_id,
                )
            self.assertFalse((jobs.job_path(job_id) / "forecast.json").exists())

    async def test_required_modeler_failure_prevents_result_despite_quorum(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
        ):
            question = BinaryQuestion(question_text="Synthetic")
            job_id = jobs.create_job(question)
            with self.assertRaisesRegex(
                RuntimeError, "Required modeling proposals failed: causal-modeler"
            ) as exc:
                await run_ensemble(FakeOpenCode({"causal-modeler"}), question, job_id)
            self.assertIn("simulated failure", str(exc.exception))
            self.assertFalse((jobs.job_path(job_id) / "forecast.json").exists())

    async def test_required_modeler_review_failure_prevents_result_despite_quorum(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
        ):
            question = BinaryQuestion(question_text="Synthetic")
            job_id = jobs.create_job(question)
            client = FakeOpenCode()
            original_wait = client.wait

            async def fail_causal_review(session, timeout):
                if (
                    client.sessions[session]["title"]
                    == "Bayesian review: causal-modeler"
                ):
                    raise RuntimeError("Review failure")
                await original_wait(session, timeout)

            with patch.object(client, "wait", side_effect=fail_causal_review):
                with self.assertRaisesRegex(
                    RuntimeError, "Required modeling reviews failed: causal-modeler"
                ):
                    await run_ensemble(client, question, job_id)
            self.assertFalse((jobs.job_path(job_id) / "forecast.json").exists())

    async def test_review_quorum_failure_prevents_result(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
        ):
            question = BinaryQuestion(question_text="Synthetic")
            job_id = jobs.create_job(question)
            client = FakeOpenCode()
            original_wait = client.wait

            async def fail_reviews(session, timeout):
                if "review:" in client.sessions[session]["title"]:
                    raise RuntimeError("Review failure")
                await original_wait(session, timeout)

            with patch.object(client, "wait", side_effect=fail_reviews):
                with self.assertRaisesRegex(RuntimeError, "Bayesian reviews succeeded"):
                    await run_ensemble(client, question, job_id)
            self.assertFalse((jobs.job_path(job_id) / "forecast.json").exists())

    async def test_stable_posterior_without_acceptance_fails_closed(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
        ):
            question = BinaryQuestion(question_text="Synthetic")
            job_id = jobs.create_job(question)
            client = FakeOpenCode()
            original_wait = client.wait

            async def disagree(session, timeout):
                if "review:" not in client.sessions[session]["title"]:
                    return await original_wait(session, timeout)
                stage_id = client.sessions[session]["stage_id"]
                record_stage(
                    stage_id,
                    {
                        "accepts_current": False,
                        "critique": "Disagree",
                        "proposed_changes": "Revise prior",
                        "limitations": "Uncertain",
                    },
                )

            with patch.object(client, "wait", side_effect=disagree):
                with self.assertRaisesRegex(RuntimeError, "did not converge"):
                    await run_ensemble(client, question, job_id, review_rounds=1)
            self.assertFalse((jobs.job_path(job_id) / "forecast.json").exists())
