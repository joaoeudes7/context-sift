import unittest

from compact_dataset.wikipedia_long import build_long_rows, chunk_text


class WikipediaLongTests(unittest.TestCase):
    def test_chunk_requires_long_context(self) -> None:
        text = "\n\n".join(["A" * 6_000] * 5)
        chunks = chunk_text(text, min_chars=10_000, max_chars=20_000)
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) >= 10_000 for chunk in chunks))

    def test_labels_and_target_are_copy_only(self) -> None:
        text = "\n\n".join(["First fact 2026. Supporting detail. " * 200] * 8)
        rows = build_long_rows("Title", "https://example.com", text)
        self.assertTrue(rows)
        row = rows[0]
        self.assertEqual(len(row["units"]), len(row["labels"]))
        self.assertTrue(any(row["labels"]))
        self.assertTrue(all(unit["text"] in row["source"] for unit in row["units"]))
        self.assertTrue(all(token in row["source"] for token in row["target"].split()))


if __name__ == "__main__":
    unittest.main()
