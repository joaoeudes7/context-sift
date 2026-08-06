#!/usr/bin/env python3
"""Compact aligned multilingual handoffs and report exact semantic-anchor retention."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import unicodedata

from context_sift import Compactor
from validate_prompt_scenarios import contains_exact


def contains_natural_phrase(text: str, phrase: str) -> bool:
    def normalize(value: str) -> str:
        value = unicodedata.normalize("NFKC", value).casefold()
        value = re.sub(r"\s+([,;:!?])", r"\1", value)
        return " ".join(value.split())
    return normalize(phrase) in normalize(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/context-sift"))
    parser.add_argument("--data", type=Path, default=Path("data/validation/prompts/multilingual_handoffs.jsonl"))
    parser.add_argument("--history", type=Path, default=Path("reports/multilingual_prompt_validation.jsonl"))
    parser.add_argument("--outputs", type=Path, default=Path("reports/multilingual_prompt_samples"))
    parser.add_argument("--repetitions", type=int, default=80)
    args = parser.parse_args()
    compactor = Compactor(args.model)
    args.outputs.mkdir(parents=True, exist_ok=True)
    results = []
    for line in args.data.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        source = row["source"] + "\n" + "\n".join([row["repetition"]] * args.repetitions)
        start = time.perf_counter()
        output = compactor(source)
        milliseconds = (time.perf_counter() - start) * 1_000
        missing_anchors = [value for value in row["anchors"] if not contains_exact(output, value)]
        missing_phrases = [
            value for value in row["critical_phrases"]
            if not contains_natural_phrase(output, value)
        ]
        missing = [*missing_anchors, *missing_phrases]
        required_count = len(row["anchors"]) + len(row["critical_phrases"])
        source_tokens = len(compactor.tokenizer.encode(source))
        output_tokens = len(compactor.tokenizer.encode(output))
        (args.outputs / f"{row['language']}.compact.txt").write_text(output, encoding="utf-8")
        results.append({
            "language": row["language"], "source_tokens": source_tokens,
            "output_tokens": output_tokens, "token_reduction": 1 - output_tokens / source_tokens,
            "critical_recall": 1 - len(missing) / required_count, "missing": missing,
            "milliseconds": milliseconds,
        })
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "model": str(args.model), "results": results}
    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
