"""Portable CPU/CUDA inference for ContextSift MLX weights."""

from __future__ import annotations

from pathlib import Path


def resolve_device(torch, requested: str | None) -> str:
    """Choose CUDA when available, otherwise portable CPU."""
    if requested is not None:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


class TorchFastMinimumContextRNN:
    """Exact inference recurrence for MLX FastMinimumContextRNN weights.

    Supports both legacy (660K) and improved (758K) architectures:
    - Legacy: Embedding → mean pooling → BiGRU → keep_head
    - Improved (V2): + LayerNorm + self-attention
    """

    def __init__(self, weights_path: str | Path, *, device: str | None = None) -> None:
        import torch
        from safetensors.torch import load_file

        device = resolve_device(torch, device)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        self.torch = torch
        self.device = torch.device(device)
        self.weights = load_file(str(weights_path), device=str(self.device))
        self._has_norm = "norm.weight" in self.weights
        self._has_attention = "attention_query.weight" in self.weights
        self._is_v2 = self._has_norm and self._has_attention

    def _gru(self, sequence, prefix: str):
        torch = self.torch
        weights = self.weights
        wx = weights[f"{prefix}.Wx"]
        wh = weights[f"{prefix}.Wh"]
        bias = weights[f"{prefix}.b"]
        hidden_bias = weights[f"{prefix}.bhn"]
        hidden_size = hidden_bias.shape[0]
        hidden = torch.zeros(sequence.shape[0], hidden_size, device=self.device)
        outputs = []
        for value in sequence.unbind(dim=1):
            projected = torch.nn.functional.linear(value, wx, bias)
            hidden_projected = torch.nn.functional.linear(hidden, wh)
            reset_update = torch.sigmoid(
                projected[:, : 2 * hidden_size]
                + hidden_projected[:, : 2 * hidden_size]
            )
            reset, update = reset_update.chunk(2, dim=-1)
            candidate = torch.tanh(
                projected[:, 2 * hidden_size :]
                + reset * (hidden_projected[:, 2 * hidden_size :] + hidden_bias)
            )
            hidden = (1 - update) * candidate + update * hidden
            outputs.append(hidden)
        return torch.stack(outputs, dim=1)

    def _layer_norm(self, x):
        """Apply LayerNorm if weights exist (improved architecture)."""
        if not self._has_norm:
            return x
        torch = self.torch
        weight = self.weights["norm.weight"]
        bias = self.weights["norm.bias"]
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias)

    def _self_attention(self, x):
        """Apply self-attention if weights exist (improved architecture).

        x: (batch, seq_len, hidden*2)
        Returns: (batch, seq_len, hidden*2)
        """
        if not self._has_attention:
            return x
        torch = self.torch
        wq = self.weights["attention_query.weight"]
        bq = self.weights["attention_query.bias"]
        wk = self.weights["attention_key.weight"]
        bk = self.weights["attention_key.bias"]
        wv = self.weights["attention_value.weight"]
        bv = self.weights["attention_value.bias"]

        q = torch.nn.functional.linear(x, wq, bq)
        k = torch.nn.functional.linear(x, wk, bk)
        v = torch.nn.functional.linear(x, wv, bv)

        scale = q.shape[-1] ** -0.5
        scores = torch.bmm(q, k.transpose(1, 2)) * scale
        weights = torch.softmax(scores, dim=-1)
        attended = torch.bmm(weights, v)

        return x + attended

    def __call__(self, units: list[list[int]]):
        torch = self.torch
        if not units:
            raise ValueError("document must contain at least one unit")
        with torch.inference_mode():
            width = max(map(len, units))
            padded = torch.tensor(
                [unit + [0] * (width - len(unit)) for unit in units],
                dtype=torch.long,
                device=self.device,
            )
            # Preserve trained MLX behavior: padding is token 0 while mask excludes token 3.
            mask = padded != 3
            embedded = torch.nn.functional.embedding(padded, self.weights["embedding.weight"])
            weights = mask.unsqueeze(-1).to(embedded.dtype)
            pooled = (embedded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            sequence = pooled.unsqueeze(0)
            forward = self._gru(sequence, "document_forward")[0]
            backward = self._gru(sequence.flip(1), "document_backward")[0].flip(0)
            contextual = torch.cat((forward, backward), dim=-1)
            # Layer norm (improved architecture)
            contextual = self._layer_norm(contextual)
            # Self-attention (improved architecture)
            contextual = self._self_attention(contextual)
            logits = torch.nn.functional.linear(
                contextual,
                self.weights["keep_head.weight"],
                self.weights["keep_head.bias"],
            ).squeeze(-1)
            return logits, contextual

    def sigmoid_values(self, logits) -> list[float]:
        return self.torch.sigmoid(logits).cpu().tolist()

    def clear_cache(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()
