import json
import unittest
from pathlib import Path
from unittest.mock import Mock


class TorchBackendTests(unittest.TestCase):
    model_path = Path("models/context-sift")

    def test_cpu_backend_loads_production_weights(self) -> None:
        from context_sift.torch_backend import TorchFastMinimumContextRNN

        model = TorchFastMinimumContextRNN(self.model_path / "model.safetensors")
        logits, context = model([[1, 20, 2], [1, 42, 91, 2]])

        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(tuple(context.shape), (2, 256))
        self.assertTrue(model.torch.isfinite(logits).all())

    def test_auto_device_prefers_cuda_then_cpu(self) -> None:
        from context_sift.torch_backend import resolve_device

        torch = Mock()
        torch.cuda.is_available.return_value = True
        self.assertEqual(resolve_device(torch, None), "cuda")
        torch.cuda.is_available.return_value = False
        self.assertEqual(resolve_device(torch, None), "cpu")

    def test_service_runs_repeated_cpu_calls_and_stops(self) -> None:
        from context_sift import CompactorService

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

    def test_short_text_is_exact_unless_always_compact(self) -> None:
        from context_sift import CompactorService

        text = "Repeated instruction. Repeated instruction."
        service = CompactorService(backend="torch", device="cpu")
        forced = CompactorService(backend="torch", device="cpu", always_compact=True)
        try:
            self.assertEqual(service(text), text)
            self.assertLess(len(forced(text)), len(text))
        finally:
            service.stop()
            forced.stop()

    def test_zero_parameter_service_auto_detects_runtime(self) -> None:
        from context_sift import CompactorService

        service = CompactorService()
        self.assertIn(service._compactor.backend, ("mlx", "torch"))
        service("Keep src/app.py and timeout 30 seconds. " * 6)
        service.stop()

    def test_parameter_count_matches_config(self) -> None:
        from safetensors.torch import load_file

        config = json.loads((self.model_path / "config.json").read_text())
        weights = load_file(self.model_path / "model.safetensors")

        self.assertEqual(sum(value.numel() for value in weights.values()), config["parameters"])

    def test_torch_backend_matches_mlx_numerically(self) -> None:
        try:
            import mlx.core as mx
            import torch
            from safetensors import safe_open
        except ImportError as error:
            self.skipTest(f"mlx/torch/safetensors unavailable: {error}")
        from context_sift.msc_model import FastMinimumContextRNN
        from context_sift.torch_backend import TorchFastMinimumContextRNN

        config = json.loads((self.model_path / "config.json").read_text())
        torch_model = TorchFastMinimumContextRNN(self.model_path / "model.safetensors", device="cpu")
        mlx_model = FastMinimumContextRNN(
            config["vocab_size"], config["embedding_dim"], config["hidden_dim"]
        )
        weights = {}
        with safe_open(self.model_path / "model.safetensors", framework="numpy") as stream:
            for key in stream.keys():
                weights[key] = mx.array(stream.get_tensor(key))
        mlx_model.load_weights(list(weights.items()))

        units = [[1, 20, 2], [1, 42, 91, 2], [3, 7, 9, 5, 11]]
        with torch.no_grad():
            torch_logits, torch_context = torch_model(units)
        mlx_logits, mlx_context = mlx_model([mx.array(unit, dtype=mx.int32) for unit in units])
        mx.eval(mlx_logits, mlx_context)
        mlx_logits = torch.tensor(mlx_logits.tolist())
        mlx_context = torch.tensor(mlx_context.tolist())
        torch_logits = torch.tensor(torch_logits)
        torch_context = torch.tensor(torch_context)

        self.assertTrue(
            torch.allclose(torch_logits, mlx_logits, atol=1e-5),
            msg=f"logit diff max={float((torch_logits - mlx_logits).abs().max())}",
        )
        self.assertTrue(
            torch.allclose(torch_context, mlx_context, atol=1e-4),
            msg=f"context diff max={float((torch_context - mlx_context).abs().max())}",
        )

    def test_unavailable_cuda_fails_explicitly(self) -> None:
        import torch
        from context_sift.torch_backend import TorchFastMinimumContextRNN

        if torch.cuda.is_available():
            self.skipTest("CUDA is available")
        with self.assertRaisesRegex(RuntimeError, "CUDA requested"):
            TorchFastMinimumContextRNN(
                self.model_path / "model.safetensors", device="cuda"
            )


if __name__ == "__main__":
    unittest.main()
