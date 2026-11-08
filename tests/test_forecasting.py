import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from forecasting_tools import (
    BinaryQuestion,
    ConditionalQuestion,
    DateQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from metac_bot.domain import build_prediction, load_question
from metac_bot import jobs


class ForecastValidationTests(unittest.TestCase):
    def test_binary_clamps_and_rejects_nonfinite(self):
        question = BinaryQuestion(question_text="Will it happen?")
        self.assertEqual(build_prediction(question, {"probability": 0}), 0.01)
        self.assertEqual(build_prediction(question, {"probability": 1}), 0.99)
        for value in [math.nan, math.inf, -0.1, 1.1, True]:
            with self.subTest(value=value), self.assertRaises(Exception):
                build_prediction(question, {"probability": value})

    def test_multiple_choice_exact_options_and_total(self):
        question = MultipleChoiceQuestion(question_text="Which?", options=["A", "B"])
        self.assertEqual(
            build_prediction(question, {"options": {"A": 0.4, "B": 0.6}}).to_dict(),
            {"A": 0.4, "B": 0.6},
        )
        for options in [{"A": 1}, {"A": 0.5, "C": 0.5}, {"A": 0.4, "B": 0.4}]:
            with self.subTest(options=options), self.assertRaises(Exception):
                build_prediction(question, {"options": options})

    def test_numeric_cdf_and_order(self):
        question = NumericQuestion(
            question_text="How many?",
            lower_bound=0,
            upper_bound=100,
            open_lower_bound=False,
            open_upper_bound=False,
        )
        prediction = {
            "percentiles": [
                {"percentile": 0.1, "value": 10},
                {"percentile": 0.9, "value": 90},
            ]
        }
        cdf = build_prediction(question, prediction).get_cdf()
        self.assertEqual(len(cdf), question.cdf_size)
        self.assertTrue(all(a.percentile <= b.percentile for a, b in zip(cdf, cdf[1:])))
        with self.assertRaisesRegex(ValueError, "closed lower bound"):
            build_prediction(
                question,
                {
                    "percentiles": [
                        {"percentile": 0.1, "value": -5},
                        {"percentile": 0.9, "value": 90},
                    ]
                },
            )
        prediction["percentiles"].reverse()
        with self.assertRaises(ValueError):
            build_prediction(question, prediction)

    def test_date_values_are_utc(self):
        question = DateQuestion(
            question_text="When?",
            lower_bound="2027-01-01T00:00:00Z",
            upper_bound="2028-01-01T00:00:00Z",
            open_lower_bound=False,
            open_upper_bound=False,
        )
        result = build_prediction(
            question,
            {
                "percentiles": [
                    {"percentile": 0.1, "value": "2027-02-01"},
                    {"percentile": 0.9, "value": "2027-11-01T00:00:00Z"},
                ]
            },
        )
        self.assertTrue(result.is_date)
        self.assertEqual(result.declared_percentiles[0].value, 1801440000)

    def test_conditional_snapshot_preserves_subtypes(self):
        binary = BinaryQuestion(question_text="Condition?", id_of_question=10)
        numeric = NumericQuestion(
            question_text="Outcome?",
            lower_bound=0,
            upper_bound=100,
            open_lower_bound=True,
            open_upper_bound=True,
        )
        question = ConditionalQuestion(
            question_text="Conditional",
            parent=binary,
            child=numeric,
            question_yes=numeric,
            question_no=numeric,
        )
        restored = load_question(
            question.model_dump(mode="json", serialize_as_any=True)
        )
        assert isinstance(restored, ConditionalQuestion)
        self.assertIsInstance(restored.child, NumericQuestion)
        percentiles = {
            "percentiles": [
                {"percentile": 0.1, "value": 10},
                {"percentile": 0.9, "value": 90},
            ]
        }
        result = build_prediction(
            restored,
            {
                "parent": {"probability": 0.5},
                "child": percentiles,
                "prediction_yes": percentiles,
                "prediction_no": percentiles,
            },
        )
        self.assertEqual(result.parent, 0.5)
        self.assertEqual(result.prediction_yes.cdf_size, 201)

    def test_job_is_bound_to_snapshot_and_validation_precedes_save(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(jobs, "JOBS", Path(directory)),
        ):
            question = BinaryQuestion(
                question_text="Question", id_of_question=42, api_json={"secret": "omit"}
            )
            job_id = jobs.create_job(question)
            self.assertNotIn("api_json", jobs.get_job(job_id)["question"])
            with self.assertRaises(Exception):
                jobs.save_forecast(job_id, {"probability": 7}, "Research", "Reason")
            self.assertFalse((jobs.job_path(job_id) / "forecast.json").exists())
            jobs.save_forecast(job_id, {"probability": 0.3}, "Research", "Reason")
            self.assertEqual(jobs.read_report(job_id).question.id_of_question, 42)
            self.assertEqual(jobs.read_report(job_id).prediction, 0.3)
            with self.assertRaises(ValueError):
                jobs.get_job("../../outside")

    def test_failed_atomic_write_preserves_original_and_cleans_temporary_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            jobs.write_json(path, {"version": "original"})
            with patch.object(Path, "replace", side_effect=OSError("write failed")):
                with self.assertRaisesRegex(OSError, "write failed"):
                    jobs.write_json(path, {"version": "replacement"})
            self.assertEqual(jobs.read_json(path), {"version": "original"})
            self.assertEqual(list(Path(directory).iterdir()), [path])
