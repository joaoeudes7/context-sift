from __future__ import annotations

import unittest

from compact_dataset.evaluation import evaluate_rows, is_token_subsequence, valuable_token_recall


class EvaluationGateTests(unittest.TestCase):
    def test_passes_exact_extractive_compaction(self) -> None:
        rows = [
            {
                "source": "Please remove this long irrelevant filler before API must never expose password extra words now today.",
                "output": "API must never expose password.",
                "protected_spans": ["password"],
            },
            {
                "source": "Long irrelevant context and repeated filler says deploy at 15/08/2026 after approval today and delay.",
                "output": "deploy at 15/08/2026 after approval.",
                "protected_spans": ["15/08/2026"],
            },
        ]
        result = evaluate_rows(rows, min_ratio=0.25, max_ratio=0.40)
        self.assertTrue(result.passed)
        self.assertEqual(result.protected_span_recall, 1.0)
        self.assertEqual(result.protected_span_corruptions, 0)
        self.assertEqual(result.extractive_validity, 1.0)
        self.assertGreaterEqual(result.median_compression_ratio, 0.25)
        self.assertLessEqual(result.median_compression_ratio, 0.40)

    def test_fails_missing_or_corrupted_protected_span(self) -> None:
        result = evaluate_rows([
            {
                "source": "Filler API must keep code `user_id` and logs now.",
                "output": "API keep user_id logs.",
                "protected_spans": ["`user_id`"],
            }
        ], min_ratio=0.0, max_ratio=1.0)
        self.assertFalse(result.passed)
        self.assertLess(result.protected_span_recall, 1.0)
        self.assertGreater(result.protected_span_corruptions, 0)

    def test_fails_non_extractive_output(self) -> None:
        result = evaluate_rows([
            {"source": "API keeps logs safely.", "output": "API securely keeps logs.", "protected_spans": []}
        ], min_ratio=0.0, max_ratio=2.0)
        self.assertFalse(result.passed)
        self.assertEqual(result.extractive_validity, 0.0)

    def test_fails_ratio_outside_default_gate(self) -> None:
        result = evaluate_rows([
            {"source": "one two three four five six seven eight", "output": "one", "protected_spans": []}
        ], min_ratio=0.25, max_ratio=0.40)
        self.assertFalse(result.passed)
        self.assertLess(result.median_compression_ratio, 0.25)

    def test_fails_when_valuable_target_content_is_lost(self) -> None:
        result = evaluate_rows([{
            "source": "API must never expose password and must retain audit logs.",
            "target": "API never expose password retain audit logs",
            "output": "API must never expose password.",
        }], min_ratio=0.0, max_ratio=1.0)
        self.assertFalse(result.passed)
        self.assertLess(result.valuable_token_recall, 0.8)

    def test_valuable_recall_uses_ordered_target_tokens(self) -> None:
        self.assertEqual(valuable_token_recall("API keep logs", "API must keep logs."), 1.0)

    def test_token_subsequence_allows_whitespace_changes_but_not_reordering(self) -> None:
        self.assertTrue(is_token_subsequence("API\nkeeps logs.", "API keeps logs."))
        self.assertFalse(is_token_subsequence("API keeps logs.", "logs API."))

    def test_default_gate_accepts_variable_compression_ratio(self) -> None:
        result = evaluate_rows([{
            "source": "one two three four five six seven eight nine ten",
            "output": "one",
        }])
        self.assertTrue(result.passed)

    def test_rejects_empty_dataset_and_bad_thresholds(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluate_rows([])
        with self.assertRaisesRegex(ValueError, "threshold"):
            evaluate_rows([{"source": "x", "output": "x"}], min_ratio=0.5, max_ratio=0.2)


if __name__ == "__main__":
    unittest.main()
