import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from context_sift.alignment import AlignmentError, align_pair, tokenize_with_offsets


class AlignmentTests(unittest.TestCase):
    def test_tokenization_preserves_offsets_and_round_trips_tokens(self):
        text = "API não expõe `password` em /v1/users."
        tokens = tokenize_with_offsets(text)

        self.assertEqual(
            [text[token.start : token.end] for token in tokens],
            [token.text for token in tokens],
        )
        self.assertEqual(
            [token.text for token in tokens],
            ["API", "não", "expõe", "`password`", "em", "/v1/users", "."],
        )

    def test_alignment_labels_lcs_and_forces_protected_source_span(self):
        result = align_pair(
            "Please keep API password secret forever.",
            "API secret",
            protected_spans=["password"],
            min_target_coverage=1.0,
        )

        self.assertEqual(result.tokens, ["Please", "keep", "API", "password", "secret", "forever", "."])
        self.assertEqual(result.labels, [0, 0, 1, 1, 1, 0, 0])
        self.assertEqual(result.target_coverage, 1.0)
        self.assertEqual(result.protected_tokens, 1)

    def test_alignment_is_casefolded_and_handles_duplicate_tokens_in_order(self):
        result = align_pair("Alpha beta alpha gamma", "alpha alpha gamma", min_target_coverage=1.0)
        self.assertEqual(result.labels, [1, 0, 1, 1])

    def test_alignment_rejects_low_target_coverage(self):
        with self.assertRaisesRegex(AlignmentError, "coverage"):
            align_pair("keep this", "invented words", min_target_coverage=0.6)

    def test_dataset_builder_writes_jsonl_and_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "pairs.jsonl"
            output = Path(directory) / "pruner.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "ok", "language": "en", "source": "Please keep API secret.", "target": "API secret", "protected_spans": []}),
                        json.dumps({"id": "bad", "language": "en", "source": "keep this", "target": "new content", "protected_spans": []}),
                    ]
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [sys.executable, "scripts/build_pruner_dataset.py", str(source), str(output), "--min-target-coverage", "1"],
                check=True,
                capture_output=True,
                text=True,
            )

            row = json.loads(output.read_text(encoding="utf-8"))
            stats = json.loads(completed.stdout)
            self.assertEqual(row["id"], "ok")
            self.assertEqual(row["tokens"], ["Please", "keep", "API", "secret", "."])
            self.assertEqual(row["labels"], [0, 0, 1, 1, 0])
            self.assertEqual(row["offsets"], [[0, 6], [7, 11], [12, 15], [16, 22], [22, 23]])
            self.assertEqual(row["stats"]["target_coverage"], 1.0)
            self.assertEqual(stats, {"accepted": 1, "rejected": 1, "total": 2})


if __name__ == "__main__":
    unittest.main()
