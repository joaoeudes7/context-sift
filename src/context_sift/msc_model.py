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
    """Mean subword embeddings + document BiGRU; optimized MSC runtime.

    Legacy architecture (660K params). Backward compatible with existing weights.
    """

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


class FastMinimumContextRNNV2(nn.Module):
    """Improved FastMinimumContextRNN with layer norm + self-attention.

    Pareto-optimized architecture (~758K params):
    - Embedding → mean pooling (mask token 3) → BiGRU
    - LayerNorm on BiGRU output (training stability)
    - Single-head self-attention over clause representations (long-range deps)
    - Linear keep_head for clause scoring

    Requires new training. Does not load legacy weights.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.document_forward = nn.GRU(embedding_dim, hidden_dim)
        self.document_backward = nn.GRU(embedding_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.attention_query = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention_key = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention_value = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.keep_head = nn.Linear(hidden_dim * 2, 1)

    def _self_attention(self, x: mx.array) -> mx.array:
        """Single-head self-attention with residual connection.

        x: (batch, seq_len, hidden*2) or (seq_len, hidden*2)
        Returns: same shape as input
        """
        q = self.attention_query(x)
        k = self.attention_key(x)
        v = self.attention_value(x)
        scale = mx.array(q.shape[-1] ** -0.5, dtype=q.dtype)
        # Handle 2D (single batch) and 3D inputs
        if q.ndim == 2:
            scores = mx.matmul(q, k.T) * scale
        else:
            scores = mx.matmul(q, k.transpose(0, 2, 1)) * scale
        weights = mx.softmax(scores, axis=-1)
        attended = mx.matmul(weights, v)
        return x + attended

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
        contextual = self.norm(contextual)
        contextual = self._self_attention(contextual)
        return self.keep_head(contextual).squeeze(-1), contextual

    def parameter_count(self) -> int:
        return sum(value.size for _, value in tree_flatten(self.parameters()))


def adaptive_threshold(logits: mx.array, base_threshold: float = 0.28) -> float:
    """Compute adaptive threshold based on logit distribution.

    Low variance → tighten (more selective). High variance → loosen (keep more).
    """
    mean = mx.mean(logits)
    std = mx.std(logits)
    mx.eval(mean, std)
    mean_val = float(mean.item())
    std_val = float(std.item())
    adjustment = max(-0.1, min(0.1, std_val * 0.3 - 0.1))
    return max(0.1, min(0.6, base_threshold + adjustment))


def quantize_int8(model: FastMinimumContextRNN | FastMinimumContextRNNV2) -> dict[str, mx.array]:
    """Post-training INT8 quantization for linear layer weights.

    Embeddings and GRU weights stay FP32 for numerical stability.
    """
    params = dict(tree_flatten(model.parameters()))
    quantized = {}
    for name, value in params.items():
        if "keep_head" in name or "attention" in name:
            scale = mx.max(mx.abs(value)) / 127.0
            quantized[name] = mx.clip(mx.round(value / scale), -127, 127).astype(mx.int8)
            quantized[f"{name}_scale"] = scale
        else:
            quantized[name] = value
    return quantized
