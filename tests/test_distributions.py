import unittest

from forecasting_tools import DateQuestion, NumericQuestion
from metac_bot.forecasting.strategy.distributions import (
    cdf_distance_bounds,
    distribution_from_percentiles,
    distribution_prediction,
    pool_distributions,
)


class DistributionTests(unittest.TestCase):
    def numeric(self, *, opened: bool = False) -> NumericQuestion:
        return NumericQuestion(
            question_text="How many?",
            lower_bound=0,
            upper_bound=100,
            open_lower_bound=opened,
            open_upper_bound=opened,
        )

    def test_closed_distribution_and_pool(self) -> None:
        question = self.numeric()
        first = distribution_from_percentiles(
            question,
            {
                "percentiles": [
                    {"percentile": 0.25, "value": 25},
                    {"percentile": 0.75, "value": 75},
                ]
            },
        )
        second = distribution_from_percentiles(
            question,
            {
                "percentiles": [
                    {"percentile": 0.25, "value": 10},
                    {"percentile": 0.75, "value": 50},
                ]
            },
        )
        pooled = pool_distributions([first, second], [1, 3])
        self.assertAlmostEqual(first.cdf(40), 0.4)
        self.assertAlmostEqual(pooled.cdf(40), 0.25 * 0.4 + 0.75 * 0.625)
        self.assertAlmostEqual(pooled.cdf(pooled.quantile(0.7)), 0.7)
        self.assertEqual(first.cdf(-100), 0)
        self.assertEqual(first.cdf(200), 1)
        low, high = cdf_distance_bounds(first, second)
        self.assertLessEqual(low, high)
        self.assertLessEqual(high - low, 0.00100001)
        self.assertLessEqual(low, 0.25)
        self.assertGreaterEqual(high, 0.25)

    def test_open_tails_are_not_truncated(self) -> None:
        question = self.numeric(opened=True)
        dist = distribution_from_percentiles(
            question,
            {
                "percentiles": [
                    {"percentile": 0.1, "value": 10},
                    {"percentile": 0.9, "value": 90},
                ]
            },
        )
        self.assertGreater(dist.cdf(-50), 0)
        self.assertLess(dist.cdf(150), 1)
        self.assertLess(dist.quantile(0.01), 0)
        self.assertGreater(dist.quantile(0.99), 100)
        prediction = distribution_prediction(question, dist)
        self.assertEqual(len(prediction["percentiles"]), 99)

    def test_date_distribution(self) -> None:
        question = DateQuestion(
            question_text="When?",
            lower_bound="2027-01-01T00:00:00Z",
            upper_bound="2028-01-01T00:00:00Z",
            open_lower_bound=False,
            open_upper_bound=False,
        )
        distribution = distribution_from_percentiles(
            question,
            {
                "percentiles": [
                    {"percentile": 0.25, "value": "2027-03-01T00:00:00Z"},
                    {"percentile": 0.75, "value": "2027-09-01T00:00:00Z"},
                ]
            },
        )
        self.assertTrue(
            distribution_prediction(question, distribution)["percentiles"][0][
                "value"
            ].endswith("+00:00")
        )

    def test_invalid_weights_are_rejected(self) -> None:
        question = self.numeric()
        dist = distribution_from_percentiles(
            question,
            {
                "percentiles": [
                    {"percentile": 0.2, "value": 20},
                    {"percentile": 0.8, "value": 80},
                ]
            },
        )
        invalid_weights: list[list[float]] = [[], [0], [-1], [float("nan")]]
        for weights in invalid_weights:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                pool_distributions([dist], weights)
