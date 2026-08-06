"""Tiny hierarchical recurrent encoder for Minimum Sufficient Context."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class MinimumContextRNN(nn.Module):
    """Shared subword encoder; emits unit salience and semantic projections."""

    def __init__(
        self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 128,
        projection_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.local_forward = nn.GRU(embedding_dim, hidden_dim)
        self.local_backward = nn.GRU(embedding_dim, hidden_dim)
        self.document_forward = nn.GRU(hidden_dim * 2, hidden_dim)
        self.document_backward = nn.GRU(hidden_dim * 2, hidden_dim)
        self.keep_head = nn.Linear(hidden_dim * 2, 1)
        self.semantic_head = nn.Linear(hidden_dim * 2, projection_dim)

    def _unit(self, token_ids: mx.array) -> mx.array:
        embedded = self.embedding(token_ids[None, :])
        forward = self.local_forward(embedded)[0, -1]
        backward = self.local_backward(embedded[:, ::-1, :])[0, -1]
        return mx.concatenate([forward, backward])

    def __call__(self, units: list[mx.array]) -> tuple[mx.array, mx.array]:
        if not units:
            raise ValueError("document must contain at least one unit")
        local = mx.stack([self._unit(unit) for unit in units])[None, :, :]
        forward = self.document_forward(local)[0]
        backward = self.document_backward(local[:, ::-1, :])[0, ::-1]
        contextual = mx.concatenate([forward, backward], axis=-1)
        keep_logits = self.keep_head(contextual).squeeze(-1)
        semantics = self.semantic_head(contextual)
        semantics = semantics / mx.maximum(mx.linalg.norm(semantics, axis=-1, keepdims=True), 1e-6)
        return keep_logits, semantics

    def parameter_count(self) -> int:
        return sum(value.size for _, value in tree_flatten(self.parameters()))


class FastMinimumContextRNN(nn.Module):
    """Mean subword embeddings + document BiGRU; optimized MSC runtime."""

    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.document_forward = nn.GRU(embedding_dim, hidden_dim)
        self.document_backward = nn.GRU(embedding_dim, hidden_dim)
        self.keep_head = nn.Linear(hidden_dim * 2, 1)

    def __call__(self, units: list[mx.array]) -> tuple[mx.array, mx.array]:
        if not units:
            raise ValueError("document must contain at least one unit")
        width = max(unit.shape[0] for unit in units)
        padded = mx.stack([mx.pad(unit, [(0, width - unit.shape[0])]) for unit in units])
        mask = padded != 3
        embedded = self.embedding(padded)
        weights = mask[..., None].astype(embedded.dtype)
        pooled = (embedded * weights).sum(axis=1) / mx.maximum(
            weights.sum(axis=1), mx.array(1, dtype=embedded.dtype)
        )
        sequence = pooled[None, :, :]
        forward = self.document_forward(sequence)[0]
        backward = self.document_backward(sequence[:, ::-1, :])[0, ::-1]
        contextual = mx.concatenate([forward, backward], axis=-1)
        return self.keep_head(contextual).squeeze(-1), contextual

    def parameter_count(self) -> int:
        return sum(value.size for _, value in tree_flatten(self.parameters()))
