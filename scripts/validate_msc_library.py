#!/usr/bin/env python3
"""Validate public Compactor across long and multilingual corpora; append compact history."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median
import time

from context_sift import Compactor
from context_sift.evaluation import valuable_token_recall
from context_sift.rules import compress_rules


def aggregate(items: list[dict]) -> dict:
    reductions = [item["reduction"] for item in items]
    latencies = [item["milliseconds"] for item in items]
    result = {
        "examples": len(items),
        "median_reduction": median(reductions),
        "min_reduction": min(reductions),
        "median_ms": median(latencies),
        "p95_ms": sorted(latencies)[max(0, round(0.95 * len(latencies)) - 1)],
        "protected_recall": sum(item["kept_spans"] for item in items) / max(1, sum(item["spans"] for item in items)),
    }
    recalls = [item["target_recall"] for item in items if "target_recall" in item]
    if recalls:
        result.update({"mean_target_recall": sum(recalls) / len(recalls), "min_target_recall": min(recalls)})
    return result


def measure(compactor: Compactor, source: str, target: str | None = None) -> dict:
    spans = [span.text for span in compress_rules(source).protected_spans]
    start = time.perf_counter()
    output = compactor(source)
    elapsed = (time.perf_counter() - start) * 1_000
    result = {
        "reduction": 1 - len(output) / max(1, len(source)), "milliseconds": elapsed,
        "spans": len(spans), "kept_spans": sum(span in output for span in spans),
    }
    if target is not None:
        result["target_recall"] = valuable_token_recall(target, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/context-sift"))
    parser.add_argument("--long", type=Path, default=Path("data/wikipedia/long.jsonl"))
    parser.add_argument("--aligned", type=Path, default=Path("data/wikipedia/aligned.jsonl"))
    parser.add_argument("--long-per-language", type=int, default=10)
    parser.add_argument("--history", type=Path, default=Path("reports/validation_history.jsonl"))
    args = parser.parse_args()
    compactor = Compactor(args.model)
    by_language: dict[str, list[dict]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for line in args.long.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        language = str(row["language"])
        if counts[language] < args.long_per_language:
            by_language[language].append(measure(compactor, row["source"], row["target"]))
            counts[language] += 1
    aligned_by_language: dict[str, list[dict]] = defaultdict(list)
    for line in args.aligned.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for language, page in row["pages"].items():
            aligned_by_language[language].append(measure(compactor, page["text"]))
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "long": {language: aggregate(items) for language, items in sorted(by_language.items())},
        "aligned": {language: aggregate(items) for language, items in sorted(aligned_by_language.items())},
    }
    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
