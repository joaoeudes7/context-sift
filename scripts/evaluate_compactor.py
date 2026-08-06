#!/usr/bin/env python3
"""Evaluate JSONL compactor predictions and enforce release gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from context_sift.evaluation import evaluate_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate JSONL rows containing source, output, and optional protected_spans"
    )
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--min-recall", type=float, default=1.0)
    parser.add_argument("--min-ratio", type=float)
    parser.add_argument("--max-ratio", type=float)
    parser.add_argument("--min-value-recall", type=float, default=0.8)
    args = parser.parse_args()
    if (args.min_ratio is None) != (args.max_ratio is None):
        parser.error("--min-ratio and --max-ratio must be provided together")

    rows = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = evaluate_rows(
        rows,
        min_recall=args.min_recall,
        min_ratio=args.min_ratio,
        max_ratio=args.max_ratio,
        min_value_recall=args.min_value_recall,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
