"""Contract checks for the optional LightningRod integration."""

import importlib.util
import unittest
from unittest.mock import MagicMock, patch


@unittest.skipUnless(
    importlib.util.find_spec("lightningrod"), "optional integration not installed"
)
class LightningRodTests(unittest.TestCase):
    def test_cyclopts_seed_option_and_legacy_alias(self):
        from integrations import main_lightningrod_eval as example

        for option in ("--max-seeds", "--max-questions"):
            with (
                self.subTest(option=option),
                patch.object(example, "load_dotenv"),
                patch.object(example, "LightningRod"),
                patch.object(example, "run_news_eval", return_value=[]) as run,
                patch.object(example, "print_results"),
            ):
                example.app([option, "3"], result_action="return_value")
                self.assertEqual(run.call_args.args[1], 3)

    def test_pipeline_uses_the_sdk_seed_parameter(self):
        from integrations import main_lightningrod_eval as example

        client = MagicMock()
        client.transforms.run.return_value.download.return_value = []
        example.run_news_eval(client, 4)
        self.assertEqual(client.transforms.run.call_args.kwargs["max_seeds"], 4)
        self.assertNotIn("max_questions", client.transforms.run.call_args.kwargs)
