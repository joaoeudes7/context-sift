import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from context_sift.clause_dataset import build_clause_row, split_clauses


class ClauseDatasetTests(unittest.TestCase):
    def test_split_is_unicode_language_agnostic(self) -> None:
        source = "第一の事実。第二の事実！ السبب واضح؟ الحل يعمل؛ Следующий факт. Новый факт."
        clauses = split_clauses(source)
        self.assertEqual(len(clauses), 6)
        self.assertEqual("".join(source[item.start:item.end] for item in clauses).replace(" ", ""), source.replace(" ", ""))

    def test_split_preserves_exact_text_and_offsets(self):
        source = "API rápida, cache ativa; senha nunca exposta."
        clauses = split_clauses(source)

        self.assertEqual(
            [source[item.start:item.end] for item in clauses],
            [item.text for item in clauses],
        )
        self.assertEqual(
            [item.text for item in clauses],
            ["API rápida,", "cache ativa;", "senha nunca exposta."],
        )

    def test_labels_clauses_and_reports_oracle_ratio(self):
        row = build_clause_row(
            "API rápida, cache desativada; senha nunca exposta.",
            "API rápida senha nunca exposta",
            min_target_coverage=1,
        )

        self.assertEqual(row["labels"], [1, 0, 1])
        self.assertEqual(row["stats"]["kept_units"], 2)
        self.assertLess(row["stats"]["oracle_token_ratio"], 1)
        self.assertIn("oracle_gap", row["stats"])

    def test_forces_one_keep_when_threshold_rejects_all(self):
        row = build_clause_row(
            "alpha beta, gamma delta.",
            "alpha",
            min_target_coverage=1,
            keep_threshold=1,
        )

        self.assertEqual(sum(row["labels"]), 1)

    def test_cli_filters_bad_pair_and_emits_aggregate_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "pairs.jsonl"
            output = Path(directory) / "clauses.jsonl"
            source.write_text("\n".join([
                json.dumps({"id": "ok", "source": "keep this, drop that.", "target": "keep this"}),
                json.dumps({"id": "bad", "source": "known only.", "target": "invented text"}),
            ]), encoding="utf-8")

            completed = subprocess.run(
                [sys.executable, "scripts/build_clause_dataset.py", str(source), str(output), "--min-target-coverage", "1"],
                check=True,
                capture_output=True,
                text=True,
            )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            summary = json.loads(completed.stdout)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], "ok")
            self.assertEqual(rows[0]["labels"], [1, 0])
            self.assertEqual(summary["accepted"], 1)
            self.assertEqual(summary["rejected"], 1)
            self.assertIsNotNone(summary["mean_oracle_token_ratio"])


if __name__ == "__main__":
    unittest.main()
