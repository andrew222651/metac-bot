import unittest
from unittest.mock import MagicMock, call

from forecasting_tools import (
    BinaryQuestion,
    ConditionalQuestion,
    DateQuestion,
    DiscreteQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)

from metac_bot.forecasting.implementation.domain import build_report
from metac_bot.infrastructure.runner import publish_prediction


class PublicationTests(unittest.TestCase):
    def test_binary_and_multiple_choice_publish_only_predictions(self) -> None:
        examples = [
            (
                BinaryQuestion(question_text="Binary", id_of_post=1, id_of_question=2),
                {"probability": 0.4},
                "post_binary_question_prediction",
                (2, 0.4),
            ),
            (
                MultipleChoiceQuestion(
                    question_text="Choice",
                    id_of_post=1,
                    id_of_question=3,
                    options=["A", "B"],
                ),
                {"options": {"A": 0.7, "B": 0.3}},
                "post_multiple_choice_question_prediction",
                (3, {"A": 0.7, "B": 0.3}),
            ),
        ]
        for question, prediction, method, arguments in examples:
            with self.subTest(question=question.question_text):
                client = MagicMock()
                report = build_report(question, prediction, "Research", "Rationale")
                publish_prediction(report, client)
                getattr(client, method).assert_called_once_with(*arguments)
                self.assertEqual(
                    client.method_calls, [call.__getattr__(method)(*arguments)]
                )

    def test_numeric_discrete_and_date_publish_cdf_only(self) -> None:
        questions = [
            NumericQuestion(
                question_text="Numeric",
                id_of_post=1,
                id_of_question=2,
                lower_bound=0,
                upper_bound=100,
                open_lower_bound=False,
                open_upper_bound=False,
            ),
            DiscreteQuestion(
                question_text="Discrete",
                id_of_post=1,
                id_of_question=3,
                lower_bound=0,
                upper_bound=100,
                open_lower_bound=False,
                open_upper_bound=False,
                cdf_size=11,
            ),
            DateQuestion(
                question_text="Date",
                id_of_post=1,
                id_of_question=4,
                lower_bound="2027-01-01T00:00:00Z",
                upper_bound="2028-01-01T00:00:00Z",
                open_lower_bound=False,
                open_upper_bound=False,
            ),
        ]
        for question in questions:
            with self.subTest(question=question.question_text):
                values = (
                    ["2027-03-01T00:00:00Z", "2027-09-01T00:00:00Z"]
                    if isinstance(question, DateQuestion)
                    else [25, 75]
                )
                prediction = {
                    "percentiles": [
                        {"percentile": 0.25, "value": values[0]},
                        {"percentile": 0.75, "value": values[1]},
                    ]
                }
                report = build_report(question, prediction, "Research", "Rationale")
                expected = [point.percentile for point in report.prediction.get_cdf()]
                for no_size in (False, True):
                    with self.subTest(no_size=no_size):
                        client = MagicMock()
                        candidate = report.model_copy(deep=True)
                        if no_size:
                            candidate.prediction.cdf_size = None
                        publish_prediction(candidate, client)
                        self.assertEqual(
                            client.method_calls,
                            [
                                call.post_numeric_question_prediction(
                                    question.id_of_question, expected
                                )
                            ],
                        )

    def test_conditional_publishes_only_both_branch_predictions(self) -> None:
        def binary(name: str, question_id: int) -> BinaryQuestion:
            return BinaryQuestion(
                question_text=name, id_of_post=1, id_of_question=question_id
            )

        question = ConditionalQuestion(
            question_text="Conditional",
            parent=binary("Parent", 1),
            child=binary("Child", 2),
            question_yes=binary("Yes", 3),
            question_no=binary("No", 4),
        )
        report = build_report(
            question,
            {
                "parent": {"probability": 0.5},
                "child": {"probability": 0.6},
                "prediction_yes": {"probability": 0.7},
                "prediction_no": {"probability": 0.3},
            },
            "Research",
            "Rationale",
        )
        client = MagicMock()
        publish_prediction(report, client)
        self.assertEqual(
            client.method_calls,
            [
                call.post_binary_question_prediction(3, 0.7),
                call.post_binary_question_prediction(4, 0.3),
            ],
        )

    def test_missing_identifiers_fail_without_posting(self) -> None:
        for post_id, question_id in ((None, 2), (1, None)):
            with self.subTest(post_id=post_id, question_id=question_id):
                question = BinaryQuestion(
                    question_text="Binary",
                    id_of_post=post_id,
                    id_of_question=question_id,
                )
                report = build_report(
                    question, {"probability": 0.4}, "Research", "Rationale"
                )
                client = MagicMock()
                with self.assertRaises(ValueError):
                    publish_prediction(report, client)
                self.assertEqual(client.method_calls, [])
