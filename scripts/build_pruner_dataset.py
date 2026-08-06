#!/usr/bin/env python3
"""Build token-level extractive labels from source/target JSONL pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from context_sift.alignment import AlignmentError, align_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-target-coverage", type=float, default=0.8)
    args = parser.parse_args()

    accepted = rejected = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as output:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                alignment = align_pair(
                    row["source"],
                    row["target"],
                    protected_spans=row.get("protected_spans", ()),
                    min_target_coverage=args.min_target_coverage,
                )
            except AlignmentError:
                rejected += 1
                continue
            output.write(json.dumps({
                "id": row.get("id"),
                "language": row.get("language"),
                "source": row["source"],
                "target": row["target"],
                "tokens": alignment.tokens,
                "labels": alignment.labels,
                "offsets": alignment.offsets,
                "stats": {
                    "target_coverage": alignment.target_coverage,
                    "kept_tokens": alignment.kept_tokens,
                    "protected_tokens": alignment.protected_tokens,
                },
            }, ensure_ascii=False) + "\n")
            accepted += 1
    print(json.dumps({"accepted": accepted, "rejected": rejected, "total": accepted + rejected}))


if __name__ == "__main__":
    main()
