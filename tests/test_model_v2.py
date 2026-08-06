"""Tests for improved model architecture and utilities."""

import unittest

from context_sift.msc_model import (
    FastMinimumContextRNN,
    FastMinimumContextRNNV2,
    adaptive_threshold,
    quantize_int8,
)


class ModelArchitectureTests(unittest.TestCase):
    def test_v2_has_norm_and_attention(self) -> None:
        model = FastMinimumContextRNNV2(vocab_size=100)
        params = dict(model.parameters())
        # Nested structure: params[group][param_name]
        self.assertIn("norm", params)
        self.assertIn("weight", params["norm"])
        self.assertIn("attention_query", params)
        self.assertIn("weight", params["attention_query"])
        self.assertIn("attention_key", params)
        self.assertIn("attention_value", params)

    def test_v2_parameter_count(self) -> None:
        model = FastMinimumContextRNNV2(vocab_size=8000)
        count = model.parameter_count()
        # V2 should have ~758K params (vs 660K legacy)
        self.assertGreater(count, 700_000)
        self.assertLess(count, 850_000)

    def test_legacy_parameter_count(self) -> None:
        model = FastMinimumContextRNN(vocab_size=8000)
        count = model.parameter_count()
        self.assertGreater(count, 600_000)
        self.assertLess(count, 700_000)

    def test_v2_output_shape(self) -> None:
        import mlx.core as mx
        model = FastMinimumContextRNNV2(vocab_size=100)
        units = [mx.array([1, 2, 3], dtype=mx.int32) for _ in range(5)]
        logits, contextual = model(units)
        mx.eval(logits, contextual)
        self.assertEqual(logits.shape, (5,))
        self.assertEqual(contextual.shape, (5, 256))  # hidden_dim*2

    def test_legacy_output_shape(self) -> None:
        import mlx.core as mx
        model = FastMinimumContextRNN(vocab_size=100)
        units = [mx.array([1, 2, 3], dtype=mx.int32) for _ in range(5)]
        logits, contextual = model(units)
        mx.eval(logits, contextual)
        self.assertEqual(logits.shape, (5,))
        self.assertEqual(contextual.shape, (5, 256))


class AdaptiveThresholdTests(unittest.TestCase):
    def test_low_variance_tightens(self) -> None:
        import mlx.core as mx
        # Tight cluster of logits
        logits = mx.array([0.5, 0.51, 0.49, 0.505, 0.495])
        threshold = adaptive_threshold(logits, base_threshold=0.28)
        self.assertLess(threshold, 0.28)

    def test_high_variance_loosens(self) -> None:
        import mlx.core as mx
        # Very spread logits to ensure high variance
        logits = mx.array([0.0, 0.2, 0.5, 0.8, 1.0])
        threshold = adaptive_threshold(logits, base_threshold=0.28)
        self.assertGreater(threshold, 0.28)

    def test_bounds(self) -> None:
        import mlx.core as mx
        logits = mx.array([0.5, 0.5, 0.5])
        threshold = adaptive_threshold(logits)
        self.assertGreaterEqual(threshold, 0.1)
        self.assertLessEqual(threshold, 0.6)


class QuantizeTests(unittest.TestCase):
    def test_quantize_reduces_precision(self) -> None:
        import mlx.core as mx
        model = FastMinimumContextRNN(vocab_size=100)
        quantized = quantize_int8(model)
        # Linear weights should be int8
        self.assertEqual(quantized["keep_head.weight"].dtype, mx.int8)
        self.assertIn("keep_head.weight_scale", quantized)
        # Embedding stays float32
        self.assertEqual(quantized["embedding.weight"].dtype, mx.float32)


if __name__ == "__main__":
    unittest.main()
