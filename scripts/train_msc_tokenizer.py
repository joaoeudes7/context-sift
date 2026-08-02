#!/usr/bin/env python3
"""Train shared language-agnostic Unigram tokenizer with byte fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import sentencepiece as spm


def texts(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row.get("source"), str):
            yield row["source"]
        for page in row.get("pages", {}).values():
            if isinstance(page.get("text"), str):
                yield page["text"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", default=[
        Path("data/train.jsonl"), Path("data/wikipedia/long.jsonl"),
        Path("data/wikipedia/aligned.jsonl"),
    ])
    parser.add_argument("--output", type=Path, default=Path("models/context-sift-tokenizer-next"))
    parser.add_argument("--vocab-size", type=int, default=8_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output, delete=True) as corpus:
        documents = 0
        for path in args.data:
            for text in texts(path):
                corpus.write(text.replace("\x00", " ") + "\n")
                documents += 1
        corpus.flush()
        prefix = args.output / "tokenizer"
        spm.SentencePieceTrainer.train(
            input=corpus.name,
            model_prefix=str(prefix),
            model_type="unigram",
            vocab_size=args.vocab_size,
            byte_fallback=True,
            character_coverage=0.9995,
            normalization_rule_name="nmt_nfkc",
            split_digits=True,
            hard_vocab_limit=False,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            unk_id=0,
            minloglevel=2,
        )
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.output / "tokenizer.model"))
    probes = ["mesma informação", "same understanding", "同じ意味", "نفس المعنى", "src/auth/session.ts"]
    if any(tokenizer.decode(tokenizer.encode(probe)) != probe for probe in probes):
        raise RuntimeError("tokenizer round-trip failed")
    print(json.dumps({"documents": documents, "vocab_size": tokenizer.vocab_size(), "round_trip": True}))


if __name__ == "__main__":
    main()
