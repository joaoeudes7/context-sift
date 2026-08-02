import unittest


class MSCModelTests(unittest.TestCase):
    def test_model_is_tiny_and_emits_one_score_per_unit(self) -> None:
        import mlx.core as mx

        from compact_dataset.msc_model import MinimumContextRNN

        model = MinimumContextRNN(vocab_size=8_000)
        units = [mx.array([1, 20, 2]), mx.array([1, 30, 40, 2])]
        logits, semantics = model(units)
        mx.eval(logits, semantics)
        self.assertEqual(logits.shape, (2,))
        self.assertEqual(semantics.shape, (2, 64))
        self.assertLess(model.parameter_count(), 2_000_000)

    def test_rejects_empty_document(self) -> None:
        from compact_dataset.msc_model import MinimumContextRNN

        with self.assertRaisesRegex(ValueError, "at least one unit"):
            MinimumContextRNN(vocab_size=8_000)([])

    def test_fast_model_stays_below_one_million_parameters(self) -> None:
        import mlx.core as mx

        from compact_dataset.msc_model import FastMinimumContextRNN

        model = FastMinimumContextRNN(8_000)
        logits, _ = model([mx.array([1, 20, 2]), mx.array([1, 30, 40, 2])])
        mx.eval(logits)
        self.assertEqual(logits.shape, (2,))
        self.assertLess(model.parameter_count(), 1_000_000)


if __name__ == "__main__":
    unittest.main()
