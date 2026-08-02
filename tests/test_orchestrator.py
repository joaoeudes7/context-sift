import unittest

from compact_dataset.orchestrator import (
    DatasetError,
    fingerprint,
    parse_examples,
    repeated_sentences,
    simhash,
    validate_item,
)


class OrchestratorTest(unittest.TestCase):
    def test_parse_supported_plain_json_shapes(self) -> None:
        item = '{"language":"en","source":"source","target":"target"}'
        self.assertEqual(len(parse_examples(item)), 1)
        self.assertEqual(len(parse_examples(f"[{item}]")), 1)
        self.assertEqual(len(parse_examples(f'```json\n{{"examples":[{item}]}}\n```')), 1)

    def test_normalized_fingerprint_and_simhash_match(self) -> None:
        self.assertEqual(fingerprint("Hello   WORLD"), fingerprint("hello world"))
        self.assertEqual(simhash("One two three four"), simhash("one  two three four"))

    def test_repeated_sentence_detection(self) -> None:
        self.assertTrue(repeated_sentences("Keep exact important rules. Keep exact important rules."))
        self.assertFalse(repeated_sentences("Keep rules. Remove filler."))

    def test_valid_item_preserves_protected_values(self) -> None:
        source = " ".join(
            ["Please prepare detailed migration guidance"] * 20
            + ["for Project Atlas by 2027 using https://example.com and keep `user_id` unchanged"]
        )
        target = " ".join(
            ["Prepare migration guidance"] * 8
            + ["Project Atlas 2027 https://example.com keep `user_id` unchanged"]
        )
        language, _, _, protected = validate_item(
            {"language": "en", "source": source, "target": target, "protected_spans": []},
            0.25,
            0.40,
            100,
            300,
        )
        self.assertEqual(language, "en")
        self.assertIn("2027", protected)

    def test_missing_protected_value_rejected(self) -> None:
        source = " ".join(["Explain all requirements carefully"] * 30 + ["Deadline 2027"])
        target = " ".join(["Explain requirements"] * 10)
        with self.assertRaises(DatasetError):
            validate_item(
                {"language": "en", "source": source, "target": target},
                0.20,
                0.50,
                100,
                300,
            )


if __name__ == "__main__":
    unittest.main()
