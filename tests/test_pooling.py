import unittest

from forecasting_tools import (
    BinaryQuestion,
    ConditionalQuestion,
    MultipleChoiceQuestion,
)
from metac_bot.forecasting.strategy.pooling import (
    pool_distance,
    pool_posteriors,
    publication_prediction,
)


class PoolTests(unittest.TestCase):
    def test_repeated_identical_judgments_do_not_compound_evidence(self) -> None:
        question = BinaryQuestion(question_text="Will it happen?")
        forecast = {"probability": 0.8}
        pooled = pool_posteriors(question, [forecast] * 7, [1] * 7)
        self.assertAlmostEqual(pooled.prediction()["probability"], 0.8)

    def test_binary_and_choice_pool(self) -> None:
        question = BinaryQuestion(question_text="Will it happen?")
        result = pool_posteriors(
            question, [{"probability": 0.2}, {"probability": 0.8}], [1, 3]
        )
        self.assertAlmostEqual(result.prediction()["probability"], 0.65)
        identical = pool_posteriors(question, [{"probability": 0.65}], [1])
        self.assertAlmostEqual(pool_distance(result, identical), 0)
        choice = MultipleChoiceQuestion(question_text="Which?", options=["A", "B"])
        result = pool_posteriors(
            choice,
            [
                {"options": {"A": 0.2, "B": 0.8}},
                {"options": {"A": 0.6, "B": 0.4}},
            ],
            [1, 1],
        )
        self.assertAlmostEqual(result.prediction()["options"]["A"], 0.4)
        with self.assertRaises(ValueError):
            pool_posteriors(choice, [{"options": {"A": 0.2, "B": 0.1}}], [1])

    def test_conditional_pool_uses_joint_weights(self) -> None:
        parent = BinaryQuestion(question_text="Condition?")
        child = BinaryQuestion(question_text="Outcome?")
        question = ConditionalQuestion(
            question_text="Conditional?",
            parent=parent,
            child=child,
            question_yes=child,
            question_no=child,
        )
        forecasts = [
            {
                "parent": {"probability": 0.9},
                "prediction_yes": {"probability": 0.8},
                "prediction_no": {"probability": 0.2},
            },
            {
                "parent": {"probability": 0.1},
                "prediction_yes": {"probability": 0.2},
                "prediction_no": {"probability": 0.8},
            },
        ]
        result = pool_posteriors(question, forecasts, [1, 1]).prediction()
        self.assertAlmostEqual(result["parent"]["probability"], 0.5)
        self.assertAlmostEqual(result["prediction_yes"]["probability"], 0.74)
        self.assertAlmostEqual(result["prediction_no"]["probability"], 0.74)
        self.assertAlmostEqual(result["child"]["probability"], 0.74)

    def test_zero_probability_branch_fails_closed(self) -> None:
        parent = BinaryQuestion(question_text="Condition?")
        child = BinaryQuestion(question_text="Outcome?")
        question = ConditionalQuestion(
            question_text="Conditional?",
            parent=parent,
            child=child,
            question_yes=child,
            question_no=child,
        )
        forecast = {
            "parent": {"probability": 1},
            "prediction_yes": {"probability": 0.7},
            "prediction_no": {"probability": 0.3},
        }
        with self.assertRaisesRegex(ValueError, "zero-probability branch"):
            pool_posteriors(question, [forecast], [1])

    def test_publication_bounds_preserve_conditional_coherence(self) -> None:
        parent = BinaryQuestion(question_text="Condition?")
        child = BinaryQuestion(question_text="Outcome?")
        question = ConditionalQuestion(
            question_text="Conditional?",
            parent=parent,
            child=child,
            question_yes=child,
            question_no=child,
        )
        forecast = {
            "parent": {"probability": 0.999},
            "prediction_yes": {"probability": 1},
            "prediction_no": {"probability": 0},
        }
        pool = pool_posteriors(question, [forecast], [1])
        result = publication_prediction(pool)
        self.assertEqual(result["parent"]["probability"], 0.99)
        self.assertEqual(result["prediction_yes"]["probability"], 0.99)
        self.assertEqual(result["prediction_no"]["probability"], 0.01)
        self.assertAlmostEqual(
            result["child"]["probability"], 0.99 * 0.99 + 0.01 * 0.01
        )
        self.assertAlmostEqual(pool.prediction()["child"]["probability"], 0.999)
