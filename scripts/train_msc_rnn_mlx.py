#!/usr/bin/env python3
"""Train tiny hierarchical MSC RNN on MPS/MLX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sentencepiece as spm
from mlx.utils import tree_flatten

from context_sift.msc_model import FastMinimumContextRNN, MinimumContextRNN
from context_sift.msc_training import TrainingWindow, split_grouped_rows, training_windows


def loss_fn(model: nn.Module, units: list[mx.array], labels: mx.array) -> mx.array:
    logits, _ = model(units)
    targets = labels.astype(logits.dtype)
    losses = mx.maximum(logits, 0) - logits * targets + mx.log1p(mx.exp(-mx.abs(logits)))
    positives = mx.maximum(targets.sum(), mx.array(1, dtype=targets.dtype))
    negatives = mx.maximum((1 - targets).sum(), mx.array(1, dtype=targets.dtype))
    weights = targets * (targets.size / (2 * positives)) + (1 - targets) * (targets.size / (2 * negatives))
    return (losses * weights).mean()


def arrays(window: TrainingWindow) -> tuple[list[mx.array], mx.array]:
    return [mx.array(unit, dtype=mx.int32) for unit in window.units], mx.array(window.labels, dtype=mx.float32)


def validation_loss(model: MinimumContextRNN, windows: list[TrainingWindow]) -> float:
    losses = []
    for window in windows:
        units, labels = arrays(window)
        loss = loss_fn(model, units, labels)
        mx.eval(loss)
        losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", default=[Path("data/wikipedia/long.jsonl")])
    parser.add_argument("--tokenizer", type=Path, default=Path("models/context-sift/tokenizer.model"))
    parser.add_argument("--output", type=Path, default=Path("models/context-sift-next"))
    parser.add_argument("--initial-model", type=Path, default=Path("models/context-sift/model.safetensors"))
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-label-ratio", type=float, default=0.8)
    parser.add_argument("--reverse-probability", type=float, default=0.5)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    initial_config_path = args.initial_model.parent / "config.json" if args.initial_model else None
    initial_config = (
        json.loads(initial_config_path.read_text(encoding="utf-8"))
        if initial_config_path and initial_config_path.exists()
        else {}
    )
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    rows = []
    for path in args.data:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                rows.append(json.loads(line))
    def usable(row: dict) -> bool:
        if "labels" in row and row["labels"]:
            return sum(int(value) for value in row["labels"]) / len(row["labels"]) <= args.max_label_ratio
        return True

    filtered = [row for row in rows if usable(row)]
    train_rows, valid_rows = split_grouped_rows(filtered, 0.1, args.seed)
    train = [window for row in train_rows for window in training_windows(row, tokenizer)]
    valid = [window for row in valid_rows for window in training_windows(row, tokenizer)]
    if len(train) < 10 or not valid:
        raise SystemExit("need at least 10 training windows and one validation group")
    rng = random.Random(args.seed)
    rng.shuffle(train)
    model = FastMinimumContextRNN(tokenizer.vocab_size()) if args.fast else MinimumContextRNN(tokenizer.vocab_size())
    if args.initial_model:
        model.load_weights(str(args.initial_model))
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)
    value_and_grad = nn.value_and_grad(model, loss_fn)
    initial = validation_loss(model, valid[:20])
    best = initial
    best_weights = dict(tree_flatten(model.parameters()))
    for iteration in range(1, args.iterations + 1):
        window = train[rng.randrange(len(train))]
        units, labels = arrays(window)
        if rng.random() < args.reverse_probability:
            units = list(reversed(units))
            labels = labels[::-1]
        loss, gradients = value_and_grad(model, units, labels)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        if iteration == 1 or iteration % 20 == 0:
            current = validation_loss(model, valid[:20])
            print(json.dumps({"iteration": iteration, "train_loss": float(loss.item()), "valid_loss": current}))
            if current < best:
                best = current
                best_weights = {name: mx.array(value) for name, value in tree_flatten(model.parameters())}
    model.load_weights(list(best_weights.items()))
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(args.output / "model.safetensors"))
    shutil.copy2(args.tokenizer, args.output / "tokenizer.model")
    config = {
        "name": initial_config.get("name", "ContextSift"),
        "architecture": "FastMinimumContextRNN" if args.fast else "MinimumContextRNN",
        "vocab_size": tokenizer.vocab_size(),
        "embedding_dim": 64 if args.fast else 128, "hidden_dim": 128,
        "projection_dim": None if args.fast else 64,
        "parameters": model.parameter_count(), "training_windows": len(train),
        "validation_windows": len(valid), "initial_valid_loss": initial, "best_valid_loss": best,
        "excluded_rows": len(rows) - len(filtered), "reverse_probability": args.reverse_probability,
        "initial_model": str(args.initial_model) if args.initial_model else None,
        "threshold": initial_config.get("threshold", 0.5),
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    reloaded = FastMinimumContextRNN(tokenizer.vocab_size()) if args.fast else MinimumContextRNN(tokenizer.vocab_size())
    reloaded.load_weights(str(args.output / "model.safetensors"))
    probe_units, _ = arrays(valid[0])
    logits, _ = reloaded(probe_units)
    mx.eval(logits)
    if not bool(mx.all(mx.isfinite(logits)).item()):
        raise RuntimeError("reloaded model produced non-finite logits")
    print(json.dumps({**config, "reload": True, "finite_logits": True}))


if __name__ == "__main__":
    main()
