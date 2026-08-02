from __future__ import annotations

import unittest


class WordPrunerTests(unittest.TestCase):
    def test_hash_ids_are_stable_and_casefolded(self) -> None:
        from compact_dataset.word_pruner import token_id

        self.assertEqual(token_id("API"), token_id("api"))
        self.assertNotEqual(token_id("api"), token_id("logs"))

    def test_sentence_selection_never_returns_word_fragments(self) -> None:
        from compact_dataset.word_pruner import render_sentences, sentence_ranges

        text = "Remove filler words. API não expõe `password`. Keep logs!"
        ranges = sentence_ranges(text)
        self.assertEqual(
            render_sentences(text, ranges, [False, True, False]),
            "API não expõe `password`.",
        )

    def test_sentence_split_does_not_break_url_or_decimal(self) -> None:
        from compact_dataset.word_pruner import render_sentences, sentence_ranges

        text = "Use https://api.example.com/v1 at 1.5 ms. Keep logs."
        ranges = sentence_ranges(text)
        self.assertEqual(len(ranges), 2)
        self.assertEqual(render_sentences(text, ranges, [True, False]),
                         "Use https://api.example.com/v1 at 1.5 ms.")

    def test_clause_split_uses_semicolon_and_comma_boundaries(self) -> None:
        from compact_dataset.word_pruner import render_sentences, sentence_ranges

        text = "Drop context, keep API rule; remove filler."
        ranges = sentence_ranges(text)
        self.assertEqual(len(ranges), 3)
        self.assertEqual(render_sentences(text, ranges, [False, True, False]),
                         "keep API rule;")

    def test_protected_span_forces_whole_sentence(self) -> None:
        from compact_dataset.word_pruner import force_protected_sentences, sentence_ranges

        text = "Drop this. Deadline 15/08/2026. Drop that."
        ranges = sentence_ranges(text)
        keep = [False] * len(ranges)
        force_protected_sentences(text, ranges, keep, ["15/08/2026"])
        self.assertEqual(keep, [False, True, False])

    def test_sentence_labels_keep_sentence_with_target_content(self) -> None:
        from compact_dataset.word_pruner import sentence_keep_labels

        source = "Background details. API must not expose password. Extra filler."
        target = "API must not expose password."
        self.assertEqual(sentence_keep_labels(source, target), [0, 1, 0])

    def test_sentence_labels_rank_partial_extractive_content(self) -> None:
        from compact_dataset.word_pruner import sentence_keep_labels

        source = "API service handles users and logs. Unrelated background details."
        target = "API users logs"
        self.assertEqual(sentence_keep_labels(source, target), [1, 0])

    def test_model_stays_in_small_parameter_budget(self) -> None:
        from compact_dataset.word_pruner import WordSentencePruner

        count = WordSentencePruner().parameter_count()
        self.assertTrue(1_000_000 <= count <= 2_000_000)

    def test_model_emits_one_logit_per_clause(self) -> None:
        import mlx.core as mx
        from compact_dataset.word_pruner import WordSentencePruner

        model = WordSentencePruner(embedding_dim=8, hidden_dim=12)
        token_ids = mx.array([[[1, 2, 0], [3, 4, 5]]], dtype=mx.int32)
        word_mask = token_ids != 0
        logits = model(token_ids, word_mask)
        mx.eval(logits)
        self.assertEqual(logits.shape, (1, 2, 2))

    def test_training_split_is_deterministic_and_disjoint(self) -> None:
        from compact_dataset.word_pruner import split_rows

        rows = [{"id": str(index)} for index in range(10)]
        train, validation = split_rows(rows, 0.2, 7)
        self.assertEqual(len(train), 8)
        self.assertEqual(len(validation), 2)
        self.assertFalse({row["id"] for row in train} & {row["id"] for row in validation})
        self.assertEqual((train, validation), split_rows(rows, 0.2, 7))


if __name__ == "__main__":
    unittest.main()
