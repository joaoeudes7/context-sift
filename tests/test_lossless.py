"""Tests for lossless compaction strategies ported from Headroom."""

import unittest

from context_sift.lossless import (
    collapse_runs,
    expand_runs,
    fold_repeated_blocks,
    unfold_repeated_blocks,
    strip_ansi,
    search_heading,
    search_unheading,
    search_dir_heading,
    path_heading,
    diff_strip_index,
    compact_logs,
)


class LosslessTests(unittest.TestCase):
    def test_strip_ansi(self) -> None:
        text = "\x1b[31mERROR\x1b[0m: something failed"
        self.assertEqual(strip_ansi(text), "ERROR: something failed")

    def test_collapse_runs_basic(self) -> None:
        text = "line1\nline1\nline1\nline2\n"
        result = collapse_runs(text)
        self.assertIn("... (repeated 3 times)", result)
        self.assertEqual(expand_runs(result), text)

    def test_collapse_runs_single_not_collapsed(self) -> None:
        text = "line1\nline2\nline3\n"
        self.assertEqual(collapse_runs(text), text)

    def test_fold_repeated_blocks(self) -> None:
        block = "config: key1=val1\nconfig: key2=val2\nconfig: key3=val3\n"
        text = block + "other: x\n" + block
        result = fold_repeated_blocks(text)
        self.assertIn("repeats 3 lines from", result)
        self.assertLess(len(result), len(text))
        self.assertEqual(unfold_repeated_blocks(result), text)

    def test_search_heading(self) -> None:
        text = "src/main.py:1:import os\nsrc/main.py:2:x = 1\nsrc/utils.py:1:def foo():\n"
        result = search_heading(text)
        self.assertIn("src/main.py", result)
        self.assertIn("src/utils.py", result)
        self.assertLess(len(result), len(text))
        self.assertEqual(search_unheading(result), text)

    def test_search_dir_heading(self) -> None:
        text = "src/main.py:1:import os\nsrc/utils.py:1:def foo():\nlib/core.py:1:class X:\n"
        result = search_dir_heading(text)
        self.assertIn("src/", result)
        self.assertLess(len(result), len(text))

    def test_path_heading(self) -> None:
        text = "src/main.py\nsrc/utils.py\nlib/core.py\n"
        result = path_heading(text)
        self.assertIn("src/", result)
        self.assertLess(len(result), len(text))

    def test_diff_strip_index(self) -> None:
        text = "index abc1234..def5678 100644\n--- a/file.py\n+++ b/file.py\n"
        result = diff_strip_index(text)
        self.assertNotIn("index", result)
        self.assertIn("--- a/file.py", result)

    def test_compact_logs_errors_kept(self) -> None:
        lines = [f"INFO: processing item {i}" for i in range(20)]
        lines.append("ERROR: something failed")
        lines.extend([f"INFO: done {i}" for i in range(5)])
        text = "\n".join(lines)
        result = compact_logs(text)
        self.assertIn("ERROR: something failed", result)
        self.assertLess(len(result), len(text))

    def test_compact_logs_ansi_stripped(self) -> None:
        text = "\x1b[32mINFO\x1b[0m: ok\n" * 15
        result = compact_logs(text)
        self.assertNotIn("\x1b", result)

    def test_compact_logs_warnings_deduped(self) -> None:
        lines = ["WARNING: connection slow"] * 10
        lines.append("ERROR: timeout")
        text = "\n".join(lines)
        result = compact_logs(text)
        self.assertLess(result.count("WARNING"), 10)


if __name__ == "__main__":
    unittest.main()
