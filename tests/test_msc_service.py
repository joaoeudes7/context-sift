import unittest
from unittest.mock import Mock, patch

from context_sift.msc import CompactorService


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

    @patch("context_sift.msc.Compactor")
    def test_always_compact_flag_reaches_compactor(self, compactor_type: Mock) -> None:
        compactor_type.return_value._runtime.clear_cache = Mock()

        service = CompactorService(always_compact=True)

        self.assertTrue(compactor_type.call_args.kwargs["always_compact"])
        service.stop()


if __name__ == "__main__":
    unittest.main()
