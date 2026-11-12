"""The forecasting strategy selects the existing supported workflows."""

import unittest

from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion
from scipy.stats import norm

from metac_bot.forecasting.strategy.aggregation import posterior_change
from metac_bot.forecasting.strategy.principles import SUPERFORECASTING_PRINCIPLES
from metac_bot.forecasting.strategy.workflow import uses_ensemble


class StrategyTests(unittest.TestCase):
    def test_forecasting_guidance_prioritizes_accuracy(self) -> None:
        self.assertIn("accurate probability distribution", SUPERFORECASTING_PRINCIPLES)
        self.assertIn(
            "calibration alone is not the objective", SUPERFORECASTING_PRINCIPLES
        )

    def test_agents_are_encouraged_to_use_available_scipy(self) -> None:
        self.assertIn("basic statistical modeling", SUPERFORECASTING_PRINCIPLES)
        self.assertIn("SciPy", SUPERFORECASTING_PRINCIPLES)
        self.assertAlmostEqual(norm.cdf(0), 0.5)

    def test_agents_are_informed_of_modeling_libraries(self) -> None:
        for library in (
            "NumPy",
            "SciPy",
            "SymPy",
            "pandas",
            "statsmodels",
            "NetworkX",
            "PyMC",
            "ArviZ",
        ):
            with self.subTest(library=library):
                self.assertIn(library, SUPERFORECASTING_PRINCIPLES)

    def test_agents_can_discuss_auxiliary_probabilities(self) -> None:
        self.assertIn(
            "auxiliary probabilities for related events", SUPERFORECASTING_PRINCIPLES
        )
        self.assertIn(
            "rather than as additional target forecasts", SUPERFORECASTING_PRINCIPLES
        )

    def test_ensemble_selection(self) -> None:
        binary = BinaryQuestion(question_text="Will it happen?")
        choice = MultipleChoiceQuestion(question_text="Which?", options=["A", "B"])
        numeric = NumericQuestion(
            question_text="How many?",
            lower_bound=0,
            upper_bound=10,
            open_lower_bound=False,
            open_upper_bound=False,
        )
        for question in (binary, choice, numeric):
            with self.subTest(question=type(question).__name__):
                self.assertTrue(uses_ensemble(question, single_agent=False))
                self.assertFalse(uses_ensemble(question, single_agent=True))

    def test_convergence_uses_largest_choice_change(self) -> None:
        previous = {"posterior": {"A": 0.5, "B": 0.3, "C": 0.2}}
        current = {"posterior": {"A": 0.4, "B": 0.35, "C": 0.25}}
        self.assertAlmostEqual(
            posterior_change(previous, current, ["A", "B", "C"]), 0.1
        )
        self.assertAlmostEqual(
            posterior_change({"posterior": 0.4}, {"posterior": 0.6}, None), 0.2
        )
