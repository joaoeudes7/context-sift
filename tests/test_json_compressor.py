"""Tests for the JSON compressor module."""

import json
import unittest

from context_sift.json_compressor import compact_json, _compress_json_array, _is_error_item


class JsonCompressorTests(unittest.TestCase):
    def test_passthrough_non_json(self) -> None:
        text = "This is plain text with no JSON."
        self.assertEqual(compact_json(text), text)

    def test_passthrough_short_array(self) -> None:
        items = [{"id": 1}, {"id": 2}, {"id": 3}]
        text = json.dumps(items)
        result = compact_json(text)
        self.assertEqual(json.loads(result), items)

    def test_large_array_keeps_first_last(self) -> None:
        items = [{"id": i, "value": f"item-{i}"} for i in range(50)]
        text = json.dumps(items)
        result = compact_json(text)
        parsed = json.loads(result)
        self.assertLess(len(parsed), len(items))
        self.assertLess(len(result), len(text))
        # First 3 kept
        self.assertEqual(parsed[0]["id"], 0)
        self.assertEqual(parsed[1]["id"], 1)
        self.assertEqual(parsed[2]["id"], 2)
        # Last 2 kept
        last_real = [p for p in parsed if "_summary" not in p]
        self.assertEqual(last_real[-1]["id"], 49)

    def test_error_items_preserved(self) -> None:
        items = [{"id": i, "status": "ok"} for i in range(20)]
        items.append({"id": 20, "status": "error", "message": "timeout"})
        text = json.dumps(items)
        result = compact_json(text)
        parsed = json.loads(result)
        # Error item should be present
        statuses = [p.get("status") for p in parsed if "_summary" not in p]
        self.assertIn("error", statuses)

    def test_deduplication(self) -> None:
        items = [{"type": "log", "msg": "same"}] * 30
        text = json.dumps(items)
        result = compact_json(text)
        parsed = json.loads(result)
        self.assertLess(len(parsed), len(items))
        # Summary should mention deduplication
        summary = [p for p in parsed if "_summary" in p]
        self.assertEqual(len(summary), 1)

    def test_summary_line_present(self) -> None:
        items = [{"key": f"val{i}"} for i in range(25)]
        text = json.dumps(items)
        result = compact_json(text)
        parsed = json.loads(result)
        summary = [p for p in parsed if "_summary" in p]
        self.assertEqual(len(summary), 1)
        self.assertIn("25 items total", summary[0]["_summary"])

    def test_compression_ratio_large_array(self) -> None:
        items = [{"id": i, "name": f"user-{i}", "email": f"user{i}@example.com", "active": True} for i in range(100)]
        text = json.dumps(items)
        result = compact_json(text)
        ratio = len(result) / len(text)
        self.assertLess(ratio, 0.40)  # >60% reduction

    def test_compression_ratio_nested_objects(self) -> None:
        items = [{"id": i, "data": {"nested": {"deep": f"value-{i}"}}} for i in range(60)]
        text = json.dumps(items)
        result = compact_json(text)
        ratio = len(result) / len(text)
        self.assertLess(ratio, 0.50)  # >50% reduction

    def test_single_object_large_string_value(self) -> None:
        obj = {"code": "x" * 500, "result": "ok"}
        text = json.dumps(obj)
        result = compact_json(text)
        self.assertLess(len(result), len(text))
        parsed = json.loads(result)
        self.assertIn("...", parsed["code"])

    def test_embedded_json_in_text(self) -> None:
        prefix = "API response: "
        items = [{"id": i} for i in range(40)]
        text = prefix + json.dumps(items)
        result = compact_json(text)
        self.assertTrue(result.startswith(prefix))
        self.assertLess(len(result), len(text))

    def test_is_error_item(self) -> None:
        self.assertTrue(_is_error_item({"status": "error"}))
        self.assertTrue(_is_error_item({"level": "fatal"}))
        self.assertTrue(_is_error_item({"error": "timeout"}))
        self.assertFalse(_is_error_item({"status": "ok"}))
        self.assertFalse(_is_error_item({"id": 1}))

    def test_compress_array_small_unchanged(self) -> None:
        items = [{"a": 1}, {"a": 2}]
        self.assertEqual(_compress_json_array(items), items)

    def test_compress_array_preserves_max_keep(self) -> None:
        items = [{"id": i} for i in range(20)]
        result = _compress_json_array(items, max_keep=4)
        real_items = [r for r in result if "_summary" not in r]
        self.assertLessEqual(len(real_items), 4 + 2)  # first(3) + last(2) + errors


if __name__ == "__main__":
    unittest.main()
