import unittest

from compact_dataset.rules import compress_rules


class RulesTest(unittest.TestCase):
    def test_removes_english_and_portuguese_fillers(self) -> None:
        result = compress_rules(
            "Please note that API latency matters. "
            "É importante notar que erros importam."
        )
        self.assertEqual(result.text, "API latency matters. erros importam.")
        self.assertLess(result.output_token_count, result.input_token_count)

    def test_deduplicates_exact_sentences_and_lines(self) -> None:
        result = compress_rules("Keep logs. Keep logs.\nDeploy now\nDeploy now")
        self.assertEqual(result.text, "Keep logs.\nDeploy now")

    def test_preserves_protected_values(self) -> None:
        source = (
            "Please make sure that API must never expose `user_id` at "
            "https://example.com/v1 before 2027-08-02; use maxRetries."
        )
        result = compress_rules(source)
        for value in (
            "must",
            "never",
            "`user_id`",
            "https://example.com/v1",
            "2027-08-02",
            "maxRetries",
        ):
            self.assertIn(value, result.text)
            self.assertIn(value, [span.text for span in result.protected_spans])

    def test_only_deletes_source_tokens_and_keeps_order(self) -> None:
        source = "Actually alpha, in order to beta; gamma."
        result = compress_rules(source)
        source_tokens = source.replace(",", "").replace(";", "").replace(".", "").split()
        output_tokens = result.text.replace(",", "").replace(";", "").replace(".", "").split()
        positions = [source_tokens.index(token) for token in output_tokens]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(result.text, "alpha, beta; gamma.")

    def test_empty_input(self) -> None:
        result = compress_rules("")
        self.assertEqual(result.text, "")
        self.assertEqual((result.input_token_count, result.output_token_count), (0, 0))
        self.assertEqual(result.protected_spans, ())

    def test_numeric_span_never_crosses_paragraph(self) -> None:
        result = compress_rules("Reference 32/2\n\nExpositio follows.")
        spans = [span.text for span in result.protected_spans]
        self.assertIn("32/2", spans)
        self.assertNotIn("32/2\n\nExpositio", spans)

    def test_numeric_span_does_not_consume_following_noun(self) -> None:
        spans = [span.text for span in compress_rules("Population: 100  Jews.").protected_spans]
        self.assertIn("100", spans)
        self.assertNotIn("100  Jews", spans)

    def test_protects_relative_repository_paths(self) -> None:
        result = compress_rules("Edit src/auth/session.ts and tests/auth/session.test.ts.")
        spans = [span.text for span in result.protected_spans]
        self.assertIn("src/auth/session.ts", spans)
        self.assertIn("tests/auth/session.test.ts", spans)
        self.assertIn("src/auth/session.ts", result.text)
        self.assertIn("tests/auth/session.test.ts", result.text)

    def test_protects_complete_uuid(self) -> None:
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        result = compress_rules(f"trace_id={uuid}")
        self.assertIn(uuid, result.text)
        self.assertIn(uuid, [span.text for span in result.protected_spans])

    def test_protects_agent_handoff_state(self) -> None:
        source = (
            "Never read .env. Failed attempt used asyncio.Lock. "
            "Run tests/auth/test_rotation.py::test_concurrent_refresh. "
            "User rejected lowering security checks."
        )
        spans = [span.text for span in compress_rules(source).protected_spans]
        for value in (
            ".env", "Failed", "asyncio.Lock",
            "tests/auth/test_rotation.py::test_concurrent_refresh", "rejected", "security",
        ):
            self.assertIn(value, spans)

        self.assertIn(
            "tests/auth/test_rotation.py::test_concurrent_refresh",
            compress_rules(source).text,
        )
        self.assertIn("read .env", compress_rules(source).text)

    def test_protects_declared_goal_across_languages(self) -> None:
        spans = [
            span.text
            for span in compress_rules(
                "User goal: fix refresh-token reuse. Objetivo: adicionar regressão."
            ).protected_spans
        ]
        self.assertIn("User goal:", spans)
        self.assertIn("Objetivo:", spans)

    def test_protects_labeled_sentence_and_technical_tokens_next_to_japanese(self) -> None:
        source = "目的：更新する。調査でsrc/auth/token.pyとasyncio.Lockを確認。"
        spans = [span.text for span in compress_rules(source).protected_spans]
        self.assertIn("目的：", spans)
        self.assertIn("src/auth/token.py", spans)
        self.assertIn("asyncio.Lock", spans)

    def test_protects_multilingual_rejection_state(self) -> None:
        source = "Utilisateur rejeté. Benutzer abgelehnt. ユーザーは拒否しました。用户拒绝降低检查。"
        spans = [span.text.casefold() for span in compress_rules(source).protected_spans]
        for value in ("rejeté", "abgelehnt", "拒否", "拒绝"):
            self.assertIn(value.casefold(), spans)


if __name__ == "__main__":
    unittest.main()
