import unittest
from unittest.mock import Mock, patch

from context_sift.clause_dataset import split_clauses
from context_sift.msc import CompactorService, join_kept, preserve_table_headers


class CompactorServiceTests(unittest.TestCase):
    @patch("context_sift.msc.Compactor")
    def test_default_model_requires_no_argument(self, compactor_type: Mock) -> None:
        compactor_type.return_value._runtime.clear_cache = Mock()

        service = CompactorService()

        self.assertTrue(str(compactor_type.call_args.args[0]).endswith("models/context-sift"))
        service.stop()

    @patch("context_sift.msc.Compactor")
    def test_loads_once_for_many_calls_and_stops_idempotently(self, compactor_type: Mock) -> None:
        compactor = compactor_type.return_value
        compactor.side_effect = lambda text: f"compact:{text}"
        compactor._runtime.clear_cache = Mock()
        service = CompactorService("model", autostart=False)

        service.start().start()
        self.assertEqual(service("one"), "compact:one")
        self.assertEqual(service("two"), "compact:two")
        self.assertEqual(compactor_type.call_count, 1)

        service.stop()
        service.stop()
        self.assertFalse(service.running)
        compactor._runtime.clear_cache.assert_called_once_with()
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            service("three")

    @patch("context_sift.msc.Compactor")
    def test_context_manager_starts_and_stops(self, compactor_type: Mock) -> None:
        compactor_type.return_value._runtime.clear_cache = Mock()
        service = CompactorService("model", autostart=False)

        with service as active:
            self.assertIs(active, service)
            self.assertTrue(service.running)

        self.assertFalse(service.running)

    def test_preserve_table_headers_keeps_separator_when_a_row_survives(self) -> None:
        text = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
        clauses = split_clauses(text)
        keep = [False] * len(clauses)
        keep[-1] = True  # only the last row is selected
        keep = preserve_table_headers(clauses, keep)
        self.assertTrue(keep[0], "header must be restored")
        self.assertTrue(keep[1], "separator must be restored")

    def test_join_kept_preserves_markdown_structure(self) -> None:
        text = "# Title\n\n- alpha\n- beta\n\n```python\ndef f():\n    return 1\n```"
        clauses = split_clauses(text)
        out = join_kept(text, clauses, [True] * len(clauses))
        self.assertIn("# Title", out)
        self.assertIn("- alpha\n- beta", out)
        self.assertIn("    return 1", out)
        self.assertEqual(out.count("\n"), text.count("\n"))

    @patch("context_sift.msc.Compactor")
    def test_checkpoint_report_routes_to_compactor_and_stops(self, compactor_type: Mock) -> None:
        from context_sift.msc import CompactionResult

        compactor = compactor_type.return_value
        compactor.checkpoint_report.side_effect = lambda text: CompactionResult(f"ckpt:{text}")
        compactor._runtime.clear_cache = Mock()
        service = CompactorService("model", autostart=False)
        service.start()

        self.assertEqual(service.checkpoint_report("one").text, "ckpt:one")
        self.assertEqual(service.checkpoint_report("one").text, "ckpt:one")
        self.assertEqual(compactor.checkpoint_report.call_count, 2)

        service.stop()
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            service.checkpoint_report("two")

    @patch("context_sift.msc.Compactor")
    def test_always_compact_flag_reaches_compactor(self, compactor_type: Mock) -> None:
        compactor_type.return_value._runtime.clear_cache = Mock()

        service = CompactorService(always_compact=True)

        self.assertTrue(compactor_type.call_args.kwargs["always_compact"])
        service.stop()


if __name__ == "__main__":
    unittest.main()
