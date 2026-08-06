"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from context_sift.orchestrator import BuildConfig, DatasetBuilder, DatasetError


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Generate compact-text training JSONL.")
    command.add_argument("--output", type=Path, default=Path("data/train.jsonl"))
    command.add_argument("--count", type=int, default=100)
    command.add_argument("--batch-size", type=int, default=4)
    command.add_argument("--concurrency", type=int, default=4)
    command.add_argument("--min-source-words", type=int, default=180)
    command.add_argument("--max-source-words", type=int, default=700)
    command.add_argument("--min-ratio", type=float, default=0.25)
    command.add_argument("--max-ratio", type=float, default=0.40)
    command.add_argument("--model", default="openrouter/free")
    command.add_argument("--seed", type=int, default=7)
    command.add_argument(
        "--plain-json",
        action="store_true",
        help="Request JSON through prompting when model lacks response_format support.",
    )
    return command


def main() -> None:
    args = parser().parse_args()
    if args.count < 1 or args.batch_size < 1 or args.concurrency < 1:
        raise SystemExit("count, batch-size, and concurrency must be positive")
    config = BuildConfig(
        output=args.output,
        count=args.count,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        min_source_words=args.min_source_words,
        max_source_words=args.max_source_words,
        min_ratio=args.min_ratio,
        max_ratio=args.max_ratio,
        model=args.model,
        seed=args.seed,
        structured_output=not args.plain_json,
    )
    try:
        count = asyncio.run(DatasetBuilder(config).build())
    except DatasetError as error:
        raise SystemExit(str(error)) from error
    print(f"dataset ready: {count} examples in {config.output}")


if __name__ == "__main__":
    main()
