#!/usr/bin/env python3
"""Blind semantic comparison of real original/compacted multilingual pairs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import httpx

from context_sift import Compactor


API_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM = """You audit one semantic equivalence pair between ORIGINAL and COMPACT text.
ORIGINAL is authoritative. Grammar and style do not matter. Compact fragments are valid.
Use only facts present in this ORIGINAL. Do not use outside knowledge or expect details absent from ORIGINAL.
Check whether an AI reading only COMPACT reaches the same central understanding.
Penalize missing important facts, causal/temporal relations, conditions, negation, uncertainty,
methods, results, limitations, or contradictions. Ignore expendable examples and repetition.
Return JSON only: {"id":"...","central_preserved":true,
"information_recall":0.0,"contradictions":[],"missing_critical":[],"reason":"..."}.
information_recall measures important semantic information, not word overlap.
"""


def parse_json(content: str) -> dict:
    start = content.find("{")
    if start < 0:
        raise ValueError("judge returned no JSON")
    value, _ = json.JSONDecoder().raw_decode(content[start:])
    return value


def judge_pair(client: httpx.Client, model: str, sample: dict) -> tuple[dict, str]:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = client.post(API_URL, json={
                "model": model, "temperature": 0,
                "messages": [{"role": "system", "content": SYSTEM}, {
                    "role": "user", "content": json.dumps(sample, ensure_ascii=False),
                }],
                "max_tokens": 1_000, "reasoning": {"enabled": False},
            })
            if response.status_code == 429:
                time.sleep(min(2 ** attempt, 30))
                continue
            response.raise_for_status()
            body = response.json()
            value = parse_json(body["choices"][0]["message"]["content"])
            required = {"id", "central_preserved", "information_recall", "contradictions", "missing_critical", "reason"}
            if value.keys() != required or value["id"] != sample["id"]:
                raise ValueError("judge returned invalid schema")
            if not isinstance(value["information_recall"], (int, float)) or not 0 <= value["information_recall"] <= 1:
                raise ValueError("judge information_recall must be 0..1")
            return value, str(body.get("model", model))
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f"judge failed after retries: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/context-sift"))
    parser.add_argument("--judge", default="inclusionai/ling-2.6-flash")
    parser.add_argument("--data", type=Path, default=Path("data/wikipedia/aligned-9lang.jsonl"))
    parser.add_argument("--history", type=Path, default=Path("reports/semantic_validation.jsonl"))
    args = parser.parse_args()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    compactor = Compactor(args.model_path)
    samples = {}
    for line in args.data.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for language, page in row["pages"].items():
            if language not in samples:
                original = page["text"]
                compact = compactor(original)
                samples[language] = {
                    "id": language, "original": original, "compact": compact,
                }
    headers = {"Authorization": f"Bearer {api_key}", "HTTP-Referer": "https://github.com/compact-llm-summary"}
    evaluations = {}
    judge_models = set()
    with httpx.Client(headers=headers, timeout=180) as client:
        for language, sample in samples.items():
            evaluations[language], judge_model = judge_pair(client, args.judge, sample)
            judge_models.add(judge_model)
            time.sleep(1)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "compactor": str(args.model_path), "judge": sorted(judge_models), "corrects_batch_contamination": True,
        "evaluations": [{
            **evaluations[language], "language": language,
            "source_sha256": hashlib.sha256(sample["original"].encode()).hexdigest(),
            "reduction": 1 - len(sample["compact"]) / len(sample["original"]),
        } for language, sample in samples.items()],
    }
    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
