#!/usr/bin/env python3
"""Convert source/target JSONL into clause-level KEEP/DROP JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from context_sift.alignment import AlignmentError
from context_sift.clause_dataset import build_clause_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-target-coverage", type=float, default=0.8)
    parser.add_argument("--keep-threshold", type=float, default=0.5)
    args = parser.parse_args()

    accepted = rejected = total = 0
    oracle_ratios: list[float] = []
    oracle_gaps: list[float] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as input_file, args.output.open(
        "w", encoding="utf-8"
    ) as output_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            total += 1
            try:
                item = json.loads(line)
                row = build_clause_row(
                    str(item["source"]),
                    str(item["target"]),
                    protected_spans=item.get("protected_spans", ()),
                    min_target_coverage=args.min_target_coverage,
                    keep_threshold=args.keep_threshold,
                )
            except (AlignmentError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                rejected += 1
                continue
            row["id"] = item.get("id", str(line_number))
            row["language"] = item.get("language")
            row["target"] = str(item["target"])
            stats = row["stats"]
            oracle_ratios.append(stats["oracle_token_ratio"])
            oracle_gaps.append(stats["oracle_gap"])
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            accepted += 1

    mean_ratio = sum(oracle_ratios) / accepted if accepted else None
    mean_gap = sum(oracle_gaps) / accepted if accepted else None
    print(json.dumps({
        "accepted": accepted,
        "rejected": rejected,
        "total": total,
        "mean_oracle_token_ratio": mean_ratio,
        "mean_oracle_gap": mean_gap,
    }))


if __name__ == "__main__":
    main()
