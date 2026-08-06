"""Word-aware sentence pruner: model scores tokens, output keeps whole sentences."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from context_sift.alignment import align_pair, tokenize_with_offsets
from context_sift.clause_dataset import split_clauses


VOCAB_SIZE = 8192
@dataclass(frozen=True, slots=True)
class TextRange:
    start: int
    end: int


def token_id(token: str) -> int:
    digest = hashlib.blake2b(token.casefold().encode("utf-8"), digest_size=8).digest()
    return 1 + int.from_bytes(digest) % (VOCAB_SIZE - 1)


def split_rows(
    rows: list[dict[str, Any]], validation_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_ratio))
    return shuffled[validation_count:], shuffled[:validation_count]


def sentence_ranges(text: str) -> list[TextRange]:
    return [TextRange(clause.start, clause.end) for clause in split_clauses(text)]


def render_sentences(text: str, ranges: list[TextRange], keep: list[bool]) -> str:
    return " ".join(
        text[item.start : item.end].strip()
        for item, selected in zip(ranges, keep)
        if selected
    )


def force_protected_sentences(
    text: str, ranges: list[TextRange], keep: list[bool], spans: list[str]
) -> None:
    for span in spans:
        start = 0
        while span and (index := text.find(span, start)) >= 0:
            end = index + len(span)
            for sentence_index, item in enumerate(ranges):
                if item.start < end and item.end > index:
                    keep[sentence_index] = True
            start = end


def sentence_keep_labels(source: str, target: str) -> list[int]:
    ranges = sentence_ranges(source)
    alignment = align_pair(source, target, min_target_coverage=0.0)
    scores: list[float] = []
    for item in ranges:
        indexes = [
            index
            for index, (start, end) in enumerate(alignment.offsets)
            if start < item.end and end > item.start
        ]
        kept = sum(alignment.labels[index] for index in indexes)
        scores.append(kept / len(indexes) if indexes else 0.0)
    if not scores or max(scores) == 0:
        return [0] * len(scores)
    cutoff = max(scores) * 0.5
    return [int(score >= cutoff and score > 0) for score in scores]


class WordSentencePruner(nn.Module):
    """Mean word embeddings per clause, then BiGRU over clauses."""

    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 192):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim)
        self.forward_gru = nn.GRU(embedding_dim, hidden_dim)
        self.backward_gru = nn.GRU(embedding_dim, hidden_dim)
        self.classifier = nn.Linear(2 * hidden_dim, 2)

    def __call__(self, token_ids: mx.array, word_mask: mx.array) -> mx.array:
        embedded = self.embedding(token_ids)
        weights = word_mask[..., None].astype(embedded.dtype)
        counts = mx.maximum(weights.sum(axis=-2), mx.array(1, dtype=embedded.dtype))
        clauses = (embedded * weights).sum(axis=-2) / counts
        forward = self.forward_gru(clauses)
        backward = self.backward_gru(clauses[:, ::-1, :])[:, ::-1, :]
        return self.classifier(mx.concatenate([forward, backward], axis=-1))

    def parameter_count(self) -> int:
        return sum(value.size for _, value in tree_flatten(self.parameters()))


def prune_sentences(
    model: WordSentencePruner,
    text: str,
    protected_spans: list[str] | None = None,
    threshold: float = 0.5,
) -> str:
    tokens = tokenize_with_offsets(text)
    ranges = sentence_ranges(text)
    if not tokens or not ranges:
        return text.strip()
    clause_tokens = [
        [token_id(token.text) for token in tokens if token.start < item.end and token.end > item.start]
        for item in ranges
    ]
    width = max(len(values) for values in clause_tokens)
    padded = [values + [0] * (width - len(values)) for values in clause_tokens]
    token_ids = mx.array([padded], dtype=mx.int32)
    logits = model(token_ids, token_ids != 0)[0]
    probabilities = mx.softmax(logits, axis=-1)[:, 1]
    mx.eval(probabilities)
    scores = [float(value) for value in probabilities.tolist()]
    keep = [score >= threshold for score in scores]
    force_protected_sentences(text, ranges, keep, protected_spans or [])
    if not any(keep):
        keep[max(range(len(scores)), key=scores.__getitem__)] = True
    return render_sentences(text, ranges, keep)
