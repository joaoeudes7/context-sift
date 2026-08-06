#!/usr/bin/env python3
"""Gate exact policy recall and useful compression for a candidate model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from context_sift import Compactor


def normalize(text: str) -> str:
    text = re.sub(r"\s+([:：?!。！？])", r"\1", text)
    return " ".join(text.split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/context-sift-next"))
    parser.add_argument("--data", type=Path, default=Path("data/validation/prompts/policy_rules.jsonl"))
    args = parser.parse_args()
    compactor = Compactor(args.model, backend="torch", device="cpu")
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line]
    critical = missing = source_chars = output_chars = 0
    failures = []
    for row in rows:
        output = compactor(row["source"])
        normalized_output = normalize(output)
        lost = [value for value in row["critical"] if normalize(value) not in normalized_output]
        critical += len(row["critical"])
        missing += len(lost)
        source_chars += len(row["source"])
        output_chars += len(output)
        if lost:
            failures.append({"id": row["id"], "missing": lost})
    result = {
        "rows": len(rows),
        "critical_recall": 1 - missing / critical,
        "reduction": 1 - output_chars / source_chars,
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
