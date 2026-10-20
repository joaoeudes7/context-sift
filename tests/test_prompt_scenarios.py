import re
import unittest

from context_sift import Compactor
from context_sift.prompt_scenarios import scenarios

_MODEL = "models/context-sift"


def _contains_exact(text: str, value: str) -> bool:
    """Match a whole value, never a protected substring embedded in another token."""
    return re.search(r"(?<![\w.])" + re.escape(value) + r"(?!\w)", text) is not None


class PromptScenarioTests(unittest.TestCase):
    def test_scenarios_are_long_pure_text_and_contain_critical_information(self):
        items = scenarios()
        self.assertEqual(len(items), 3)
        for item in items:
            self.assertGreater(len(item.text), 10_000)
            self.assertNotIn("```", item.text)
            self.assertTrue(all(value in item.text for value in item.critical))

    def test_agent_scenarios_keep_every_critical_value(self):
        """Structural protection + condition gates must not drop a critical span."""
        compactor = Compactor(_MODEL)
        for scenario in scenarios():
            compact = compactor(scenario.text)
            missing = [
                value for value in scenario.critical if not _contains_exact(compact, value)
            ]
            self.assertEqual(missing, [], scenario.name)

    def test_checkpoint_projection_keeps_every_critical_value(self):
        """Checkpoint mode is the only OpenCode path and lands in persisted memory."""
        compactor = Compactor(_MODEL)
        for scenario in scenarios():
            compact = compactor.checkpoint_report(scenario.text).text
            missing = [
                value for value in scenario.critical if not _contains_exact(compact, value)
            ]
            self.assertEqual(missing, [], scenario.name)


if __name__ == "__main__":
    unittest.main()
