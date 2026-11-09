"""The forecasting strategy selects the existing supported workflows."""

import unittest

from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion

from metac_bot.forecasting.strategy.aggregation import posterior_change
from metac_bot.forecasting.strategy.workflow import uses_ensemble


class StrategyTests(unittest.TestCase):
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
        for question in (binary, choice):
            with self.subTest(question=type(question).__name__):
                self.assertTrue(uses_ensemble(question, single_agent=False))
                self.assertFalse(uses_ensemble(question, single_agent=True))
        self.assertFalse(uses_ensemble(numeric, single_agent=False))

    def test_convergence_uses_largest_choice_change(self) -> None:
        previous = {"posterior": {"A": 0.5, "B": 0.3, "C": 0.2}}
        current = {"posterior": {"A": 0.4, "B": 0.35, "C": 0.25}}
        self.assertAlmostEqual(
            posterior_change(previous, current, ["A", "B", "C"]), 0.1
        )
        self.assertAlmostEqual(
            posterior_change({"posterior": 0.4}, {"posterior": 0.6}, None), 0.2
        )
