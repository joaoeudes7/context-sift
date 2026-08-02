"""Public text-to-text Minimum Sufficient Context library API."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock

from compact_dataset.clause_dataset import split_clauses
from compact_dataset.git_diff import compact_git_diff
from compact_dataset.payloads import compact_base64
from compact_dataset.rules import compress_rules


DEFAULT_MODEL_PATH = Path("models/context-sift")


class Compactor:
    """Load once; compact many prompts before downstream LLM prefill."""

    def __init__(
        self, model_path: str | Path = DEFAULT_MODEL_PATH, *, threshold: float | None = None,
        window_units: int = 64,
    ) -> None:
        import mlx.core as mx
        import sentencepiece as spm

        from compact_dataset.msc_model import FastMinimumContextRNN, MinimumContextRNN

        self._mx = mx
        path = Path(model_path)
        config = json.loads((path / "config.json").read_text())
        self.threshold = float(threshold if threshold is not None else config.get("threshold", 0.5))
        self.window_units = window_units
        self.tokenizer = spm.SentencePieceProcessor(model_file=str(path / "tokenizer.model"))
        if config["architecture"] == "FastMinimumContextRNN":
            self.model = FastMinimumContextRNN(
                config["vocab_size"], config["embedding_dim"], config["hidden_dim"]
            )
        else:
            self.model = MinimumContextRNN(
                config["vocab_size"], config["embedding_dim"], config["hidden_dim"],
                config["projection_dim"],
            )
        self.model.load_weights(str(path / "model.safetensors"))

    def __call__(self, text: str) -> str:
        if any(line.startswith("diff --git ") for line in text.splitlines()):
            return compact_git_diff(text)
        text = compress_rules(compact_base64(text)).text
        if len(text) < 200:
            return text
        clauses = split_clauses(text)
        if not clauses:
            return text
        units = [
            self._mx.array(
                self.tokenizer.encode(clause.text, out_type=int, add_bos=True, add_eos=True)[:256],
                dtype=self._mx.int32,
            )
            for clause in clauses
        ]
        scores: list[float] = []
        for start in range(0, len(units), self.window_units):
            logits, _ = self.model(units[start:start + self.window_units])
            probabilities = self._mx.sigmoid(logits)
            self._mx.eval(probabilities)
            scores.extend(float(score) for score in probabilities.tolist())
        keep = [score >= self.threshold for score in scores]
        protected = compress_rules(text).protected_spans
        keep = [
            selected or any(clause.start < span.end and clause.end > span.start for span in protected)
            for clause, selected in zip(clauses, keep)
        ]
        if not any(keep):
            keep[max(range(len(scores)), key=scores.__getitem__)] = True
        output = " ".join(clause.text.strip() for clause, selected in zip(clauses, keep) if selected)
        return output if len(output) < len(text) else text


class CompactorService:
    """Reusable process-local compactor; load once, stop explicitly."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        threshold: float | None = None,
        window_units: int = 64,
        autostart: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.window_units = window_units
        self._lock = RLock()
        self._compactor: Compactor | None = None
        if autostart:
            self.start()

    @property
    def running(self) -> bool:
        return self._compactor is not None

    def start(self) -> CompactorService:
        """Load model once. Repeated calls are no-ops."""
        with self._lock:
            if self._compactor is None:
                self._compactor = Compactor(
                    self.model_path,
                    threshold=self.threshold,
                    window_units=self.window_units,
                )
        return self

    def __call__(self, text: str) -> str:
        with self._lock:
            if self._compactor is None:
                raise RuntimeError("CompactorService is stopped; call start() before use")
            return self._compactor(text)

    def stop(self) -> None:
        """Release model references and MLX cache. Repeated calls are safe."""
        import gc

        with self._lock:
            if self._compactor is None:
                return
            mx = self._compactor._mx
            self._compactor = None
            gc.collect()
            mx.clear_cache()

    def __enter__(self) -> CompactorService:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()
