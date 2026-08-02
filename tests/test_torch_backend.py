import json
import unittest
from pathlib import Path
from unittest.mock import Mock


class TorchBackendTests(unittest.TestCase):
    model_path = Path("models/context-sift")

    def test_cpu_backend_loads_production_weights(self) -> None:
        from compact_dataset.torch_backend import TorchFastMinimumContextRNN

        model = TorchFastMinimumContextRNN(self.model_path / "model.safetensors")
        logits, context = model([[1, 20, 2], [1, 42, 91, 2]])

        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(tuple(context.shape), (2, 256))
        self.assertTrue(model.torch.isfinite(logits).all())

    def test_auto_device_prefers_cuda_then_cpu(self) -> None:
        from compact_dataset.torch_backend import resolve_device

        torch = Mock()
        torch.cuda.is_available.return_value = True
        self.assertEqual(resolve_device(torch, None), "cuda")
        torch.cuda.is_available.return_value = False
        self.assertEqual(resolve_device(torch, None), "cpu")

    def test_service_runs_repeated_cpu_calls_and_stops(self) -> None:
        from compact_dataset import CompactorService

        text = "Keep exact path src/auth/session.ts and timeout 30 seconds. " * 8
        service = CompactorService(backend="torch", device="cpu")
        first = service(text)
        second = service(text)
        service.stop()

        self.assertEqual(first, second)
        self.assertLessEqual(len(first), len(text))
        self.assertIn("src/auth/session.ts", first)
        self.assertIn("30", first)
        self.assertFalse(service.running)

    def test_zero_parameter_service_auto_detects_runtime(self) -> None:
        from compact_dataset import CompactorService

        service = CompactorService()
        self.assertIn(service._compactor.backend, ("mlx", "torch"))
        service("Keep src/app.py and timeout 30 seconds. " * 6)
        service.stop()

    def test_parameter_count_matches_config(self) -> None:
        from safetensors.torch import load_file

        config = json.loads((self.model_path / "config.json").read_text())
        weights = load_file(self.model_path / "model.safetensors")

        self.assertEqual(sum(value.numel() for value in weights.values()), config["parameters"])

    def test_unavailable_cuda_fails_explicitly(self) -> None:
        import torch
        from compact_dataset.torch_backend import TorchFastMinimumContextRNN

        if torch.cuda.is_available():
            self.skipTest("CUDA is available")
        with self.assertRaisesRegex(RuntimeError, "CUDA requested"):
            TorchFastMinimumContextRNN(
                self.model_path / "model.safetensors", device="cuda"
            )


if __name__ == "__main__":
    unittest.main()
