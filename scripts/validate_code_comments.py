#!/usr/bin/env python3
"""Measure whether generic text compaction is safe for commented source code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from context_sift import Compactor


def without_comments(source: str) -> str:
    output: list[str] = []
    block_comment = False
    for raw in source.splitlines(keepends=True):
        line = raw
        if block_comment:
            if "*/" in line:
                block_comment = False
            continue
        stripped = line.lstrip()
        if stripped.startswith("/*"):
            block_comment = "*/" not in line
            continue
        if stripped.startswith("//"):
            continue
        output.append(line)
    return "".join(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/context-sift"))
    parser.add_argument("--source", type=Path, default=Path("data/validation/code/commented_service.ts"))
    parser.add_argument("--output", type=Path, default=Path("reports/code_samples/commented_service.compact.ts"))
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8")
    compactor = Compactor(args.model)
    start = time.perf_counter()
    output = compactor(source)
    elapsed = (time.perf_counter() - start) * 1_000
    source_code = without_comments(source)
    output_code = without_comments(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(json.dumps({
        "source_chars": len(source),
        "output_chars": len(output),
        "reduction": 1 - len(output) / len(source),
        "milliseconds": elapsed,
        "code_bytes_preserved": source_code == output_code,
        "source_code_chars": len(source_code),
        "output_code_chars": len(output_code),
        "output": output,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
