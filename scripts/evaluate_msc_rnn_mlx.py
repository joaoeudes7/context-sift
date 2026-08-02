#!/usr/bin/env python3
"""Evaluate MSC recall/retention and positional robustness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import sentencepiece as spm

from compact_dataset.msc_model import FastMinimumContextRNN, MinimumContextRNN
from compact_dataset.msc_training import split_grouped_rows, training_windows


def metrics(scores: list[float], labels: list[int], threshold: float) -> dict[str, float]:
    predicted = [score >= threshold for score in scores]
    tp = sum(prediction and label for prediction, label in zip(predicted, labels))
    fp = sum(prediction and not label for prediction, label in zip(predicted, labels))
    positives = sum(labels)
    return {
        "threshold": threshold,
        "recall": tp / positives if positives else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "retention": sum(predicted) / len(predicted),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/context-sift"))
    parser.add_argument("--data", type=Path, default=Path("data/wikipedia/long.jsonl"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    config = json.loads((args.model / "config.json").read_text())
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.model / "tokenizer.model"))
    model = (
        FastMinimumContextRNN(config["vocab_size"], config["embedding_dim"], config["hidden_dim"])
        if config["architecture"] == "FastMinimumContextRNN"
        else MinimumContextRNN(
            config["vocab_size"], config["embedding_dim"], config["hidden_dim"], config["projection_dim"]
        )
    )
    model.load_weights(str(args.model / "model.safetensors"))
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line]
    _, valid_rows = split_grouped_rows(rows, 0.1, args.seed)
    windows = [window for row in valid_rows for window in training_windows(row, tokenizer)]
    scores: list[float] = []
    reverse_scores: list[float] = []
    labels: list[int] = []
    for window in windows:
        units = [mx.array(unit, dtype=mx.int32) for unit in window.units]
        logits, _ = model(units)
        reversed_logits, _ = model(list(reversed(units)))
        probabilities = mx.sigmoid(logits)
        reversed_probabilities = mx.sigmoid(reversed_logits)[::-1]
        mx.eval(probabilities, reversed_probabilities)
        scores.extend(float(value) for value in probabilities.tolist())
        reverse_scores.extend(float(value) for value in reversed_probabilities.tolist())
        labels.extend(window.labels)
    candidates = [metrics(scores, labels, step / 100) for step in range(1, 100)]
    eligible = [item for item in candidates if item["recall"] >= 0.95]
    selected = min(eligible, key=lambda item: item["retention"]) if eligible else max(candidates, key=lambda item: item["recall"])
    compression_candidates = [item for item in candidates if item["retention"] <= 0.70]
    target_30 = max(compression_candidates, key=lambda item: item["recall"], default=None)
    reverse = metrics(reverse_scores, labels, selected["threshold"])
    agreement = sum((left >= selected["threshold"]) == (right >= selected["threshold"])
                    for left, right in zip(scores, reverse_scores)) / len(scores)
    print(json.dumps({
        "validation_groups": len(valid_rows), "windows": len(windows), "units": len(labels),
        "forward": selected, "target_30_percent_reduction": target_30,
        "reversed": reverse, "prediction_agreement": agreement,
        "position_recall_drop": selected["recall"] - reverse["recall"],
    }, indent=2))


if __name__ == "__main__":
    main()
