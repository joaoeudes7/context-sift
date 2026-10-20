"""Public text-to-text Minimum Sufficient Context library API."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Literal

from context_sift.clause_dataset import Clause, split_clauses
from context_sift.checkpoint import keep_for_checkpoint, overlaps
from context_sift.git_diff import compact_git_diff
from context_sift.json_compressor import compact_json
from context_sift.lossless import collapse_spaces, compact_logs
from context_sift.payloads import compact_base64
from context_sift.rules import compress_rules, protected_spans, semantic_risk_spans


_PACKAGE_MODEL_PATH = Path(__file__).with_name("models") / "context-sift"
_REPOSITORY_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "context-sift"
DEFAULT_MODEL_PATH = _PACKAGE_MODEL_PATH if _PACKAGE_MODEL_PATH.exists() else _REPOSITORY_MODEL_PATH
Backend = Literal["auto", "mlx", "torch"]


def join_kept(text: str, clauses: list[Clause], keep: list[bool]) -> str:
    """Reassemble kept clauses using the source's own separators.

    A block boundary (newline) in the source is preserved, so markdown headings,
    lists, tables and code fences do not collapse into one space-joined line. A
    boundary introduced only by a dropped clause degrades to a single space.
    """
    parts: list[str] = []
    previous: int | None = None
    for index, (clause, selected) in enumerate(zip(clauses, keep)):
        if not selected:
            continue
        if previous is not None:
            between = text[clauses[previous].end:clause.start]
            if index == previous + 1 and not between.strip():
                parts.append(between)  # adjacent: keep exact whitespace/indentation
            else:
                parts.append("\n" if "\n" in between else " ")
        parts.append(clause.text.strip())
        previous = index
    return "".join(parts)


def preserve_table_headers(clauses: list[Clause], keep: list[bool]) -> list[bool]:
    """Keep a markdown table's header + separator whenever any of its rows survives.

    Dropping the ``|---|`` separator (or the header) turns the table into plain
    text. If the model keeps any row of a contiguous table block, its first two
    lines are forced back so the block still parses as a table.
    """
    def is_row(text: str) -> bool:
        return text.lstrip().startswith("|")

    index = 0
    while index < len(clauses):
        if not is_row(clauses[index].text):
            index += 1
            continue
        end = index
        while end < len(clauses) and is_row(clauses[end].text):
            end += 1
        if any(keep[index:end]):
            for forced in range(index, min(index + 2, end)):
                keep[forced] = True
        index = end
    return keep


@dataclass(frozen=True, slots=True)
class CompactionResult:
    """Outcome of one compaction, for detection and accounting.

    ``text`` is the compacted text (what callers send downstream). ``units`` is
    the clause count scored, ``kept`` how many were retained, ``protected`` how
    many were kept by a protected span rather than the model, and ``ratio`` the
    output/input character ratio (1.0 = no change).
    """

    text: str
    units: int = 0
    kept: int = 0
    protected: int = 0
    ratio: float = 1.0


def _ratio(output: str, source: str) -> float:
    return round(len(output) / len(source), 4) if source else 1.0


class Compactor:
    """Load once; compact many prompts before downstream LLM prefill."""

    def __init__(
        self, model_path: str | Path = DEFAULT_MODEL_PATH, *, threshold: float | None = None,
        window_units: int = 64, backend: Backend = "auto", device: str | None = None,
        always_compact: bool = False,
    ) -> None:
        import sentencepiece as spm
        path = Path(model_path)
        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"model not found: {path} (missing config.json)")
        config = json.loads(config_path.read_text())
        self.threshold = float(threshold if threshold is not None else config.get("threshold", 0.5))
        self.window_units = window_units
        self.always_compact = always_compact
        self.tokenizer = spm.SentencePieceProcessor(model_file=str(path / "tokenizer.model"))
        if backend not in ("auto", "mlx", "torch"):
            raise ValueError("backend must be auto, mlx, or torch")
        if backend == "auto":
            backend = "mlx" if platform.system() == "Darwin" and platform.machine() == "arm64" else "torch"
            if backend == "mlx":
                try:
                    import mlx.core as mx
                    mx.default_device()
                except (ImportError, RuntimeError):
                    backend = "torch"
        self.backend = backend
        if backend == "mlx":
            import mlx.core as mx
            from context_sift.msc_model import (
                FastMinimumContextRNN,
                FastMinimumContextRNNV2,
                MinimumContextRNN,
            )

            self._runtime = mx
            arch = config["architecture"]
            if arch == "FastMinimumContextRNNV2":
                self.model = FastMinimumContextRNNV2(
                    config["vocab_size"], config["embedding_dim"], config["hidden_dim"]
                )
            elif arch == "FastMinimumContextRNN":
                self.model = FastMinimumContextRNN(
                    config["vocab_size"], config["embedding_dim"], config["hidden_dim"]
                )
            else:
                self.model = MinimumContextRNN(
                    config["vocab_size"], config["embedding_dim"], config["hidden_dim"],
                    config["projection_dim"],
                )
            self.model.load_weights(str(path / "model.safetensors"))
        else:
            from context_sift.torch_backend import TorchFastMinimumContextRNN

            self.model = TorchFastMinimumContextRNN(
                path / "model.safetensors", device=device
            )
            self._runtime = self.model

    def __call__(self, text: str) -> str:
        return self.compact_report(text).text

    def compact_report(self, text: str) -> CompactionResult:
        original = text
        if any(line.startswith("diff --git ") for line in text.splitlines()):
            output = compact_git_diff(text)
            return CompactionResult(output, ratio=_ratio(output, original))
        if len(text) < 10:
            return CompactionResult(text)
        if not self.always_compact and len(text) < 200:
            return CompactionResult(text)
        text = compress_rules(compact_json(compact_base64(compact_logs(text)))).text
        if not self.always_compact and len(text) < 200:
            return CompactionResult(text, ratio=_ratio(text, original))
        clauses = split_clauses(text)
        if not clauses:
            return CompactionResult(text, ratio=_ratio(text, original))
        token_units = [
            self.tokenizer.encode(clause.text, out_type=int, add_bos=True, add_eos=True)[:256]
            for clause in clauses
        ]
        units = token_units
        if self.backend == "mlx":
            units = [self._runtime.array(unit, dtype=self._runtime.int32) for unit in token_units]
        scores: list[float] = []
        for start in range(0, len(units), self.window_units):
            logits, _ = self.model(units[start:start + self.window_units])
            if self.backend == "mlx":
                probabilities = self._runtime.sigmoid(logits)
                self._runtime.eval(probabilities)
                scores.extend(float(score) for score in probabilities.tolist())
            else:
                scores.extend(self.model.sigmoid_values(logits))
        keep = [score >= self.threshold for score in scores]
        protected = protected_spans(text) + semantic_risk_spans(text)
        forced = [
            any(clause.start < span.end and clause.end > span.start for span in protected)
            for clause in clauses
        ]
        keep = [selected or force for selected, force in zip(keep, forced)]
        keep = preserve_table_headers(clauses, keep)
        if not any(keep):
            keep[max(range(len(scores)), key=scores.__getitem__)] = True
        output = join_kept(text, clauses, keep)
        if len(output) >= len(text):
            output = text
        return CompactionResult(
            output,
            units=len(clauses),
            kept=sum(keep),
            protected=sum(forced),
            ratio=_ratio(output, original),
        )

    def checkpoint_report(self, text: str) -> CompactionResult:
        """Project a session into a checkpoint: completed work + decisions + next steps.

        Deterministic and extractive — no model call. Dropped narration is still
        recoverable from persisted history; this only shapes what is sent on.
        """
        original = text
        if any(line.startswith("diff --git ") for line in text.splitlines()):
            output = compact_git_diff(text)
            return CompactionResult(output, ratio=_ratio(output, original))
        if len(text) < 10:
            return CompactionResult(text)
        text = compress_rules(compact_json(compact_base64(compact_logs(text)))).text
        clauses = split_clauses(text)
        if not clauses:
            return CompactionResult(text, ratio=_ratio(text, original))
        # Recall-first: a checkpoint lands in persisted memory, so a meaning-changing
        # anchor keeps its whole sentence however long it runs. The character cap that
        # protects the compression ratio of the generic path would drop the tail of a
        # long condition ("... por lo que se revirtió"), leaving the head unreadable.
        protected = protected_spans(text) + semantic_risk_spans(text, max_chars=0)
        flags = [overlaps(clause, protected) for clause in clauses]
        keep = keep_for_checkpoint(clauses, protected)
        if not any(keep):
            keep = [True] * len(clauses)  # nothing status-like: keep everything
        output = join_kept(text, clauses, keep)
        if len(output) >= len(text):
            output = text
        return CompactionResult(
            output,
            units=len(clauses),
            kept=sum(keep),
            protected=sum(flags),
            ratio=_ratio(output, original),
        )


class CompactorService:
    """Reusable process-local compactor; load once, stop explicitly."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        threshold: float | None = None,
        window_units: int = 64,
        backend: Backend = "auto",
        device: str | None = None,
        always_compact: bool = False,
        autostart: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.window_units = window_units
        self.backend = backend
        self.device = device
        self.always_compact = always_compact
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
                    backend=self.backend,
                    device=self.device,
                    always_compact=self.always_compact,
                )
        return self

    def __call__(self, text: str) -> str:
        with self._lock:
            if self._compactor is None:
                raise RuntimeError("CompactorService is stopped; call start() before use")
            return self._compactor(text)

    def compact_report(self, text: str) -> CompactionResult:
        with self._lock:
            if self._compactor is None:
                raise RuntimeError("CompactorService is stopped; call start() before use")
            return self._compactor.compact_report(text)

    def checkpoint_report(self, text: str) -> CompactionResult:
        with self._lock:
            if self._compactor is None:
                raise RuntimeError("CompactorService is stopped; call start() before use")
            return self._compactor.checkpoint_report(text)

    def stop(self) -> None:
        """Release model references and MLX cache. Repeated calls are safe."""
        import gc

        with self._lock:
            if self._compactor is None:
                return
            runtime = self._compactor._runtime
            self._compactor = None
            gc.collect()
            runtime.clear_cache()

    def __enter__(self) -> CompactorService:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()
