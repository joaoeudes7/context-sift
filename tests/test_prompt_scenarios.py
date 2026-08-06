import unittest

from context_sift.prompt_scenarios import scenarios


class PromptScenarioTests(unittest.TestCase):
    def test_scenarios_are_long_pure_text_and_contain_critical_information(self):
        items = scenarios()
        self.assertEqual(len(items), 3)
        for item in items:
            self.assertGreater(len(item.text), 10_000)
            self.assertNotIn("```", item.text)
            self.assertTrue(all(value in item.text for value in item.critical))


if __name__ == "__main__":
    unittest.main()
