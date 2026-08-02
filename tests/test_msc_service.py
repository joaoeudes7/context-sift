import unittest
from unittest.mock import Mock, patch

from compact_dataset.msc import CompactorService


class CompactorServiceTests(unittest.TestCase):
    @patch("compact_dataset.msc.Compactor")
    def test_default_model_requires_no_argument(self, compactor_type: Mock) -> None:
        compactor_type.return_value._runtime.clear_cache = Mock()

        service = CompactorService()

        self.assertTrue(str(compactor_type.call_args.args[0]).endswith("models/context-sift"))
        service.stop()

    @patch("compact_dataset.msc.Compactor")
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

    @patch("compact_dataset.msc.Compactor")
    def test_context_manager_starts_and_stops(self, compactor_type: Mock) -> None:
        compactor_type.return_value._runtime.clear_cache = Mock()
        service = CompactorService("model", autostart=False)

        with service as active:
            self.assertIs(active, service)
            self.assertTrue(service.running)

        self.assertFalse(service.running)


if __name__ == "__main__":
    unittest.main()
