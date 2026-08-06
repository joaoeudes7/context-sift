#!/usr/bin/env python3
"""Validate original→compact quality on four real non-Wikipedia domains."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import time

import httpx

from context_sift import Compactor
from context_sift.rules import compress_rules
from judge_msc_pairs import API_URL, SYSTEM, parse_json


SOURCES = {
    "scientific": "https://arxiv.org/abs/1706.03762",
    "logs": "https://github.com/logpai/loghub/blob/master/HDFS/HDFS_2k.log",
    "git_diff": "https://github.com/anomalyco/opencode/commit/d2bd7ea",
    "agent_prompt": "https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/session/prompt/anthropic.txt",
}
RUBRICS = {
    "scientific": "Preserve problem, method, architecture, equations/parameters, experimental setup, numerical results, comparisons, conclusions, and limitations.",
    "logs": "Preserve event types, sequence, severity, component/block identifiers, anomalies, failures, and operational outcome.",
    "git_diff": "Preserve every file/hunk header and every added/deleted line exactly; explain behavioral change from remaining context.",
    "agent_prompt": "Preserve instructions, prohibitions, conditions, tool rules, workflows, commands, paths, URLs, and precedence.",
}


class PaperText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "nav"}:
            self.skip += 1
        elif tag in {"p", "h1", "h2", "h3", "li", "tr", "figure"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav"} and self.skip:
            self.skip -= 1
        elif tag in {"p", "h1", "h2", "h3", "li", "tr", "figure"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)

    def text(self) -> str:
        lines = (re.sub(r"\s+", " ", line).strip() for line in "".join(self.parts).splitlines())
        return "\n".join(line for line in lines if line)


def load_samples(root: Path, log_chars: int) -> dict[str, str]:
    parser = PaperText()
    parser.feed((root / "attention.html").read_text(encoding="utf-8"))
    return {
        "scientific": parser.text(),
        "logs": (root / "hdfs.log").read_text(encoding="utf-8")[:log_chars],
        "git_diff": (root / "opencode.diff").read_text(encoding="utf-8"),
        "agent_prompt": (root / "opencode-anthropic.txt").read_text(encoding="utf-8"),
    }


def exact_recall(required: list[str], output: str) -> float:
    return sum(item in output for item in required) / max(1, len(required))


def domain_required(kind: str, source: str) -> list[str]:
    lines = source.splitlines()
    if kind == "git_diff":
        return [line for line in lines if line.startswith(("diff --git ", "--- ", "+++ ", "@@ "))
                or (line.startswith(("+", "-")) and not line.startswith(("+++", "---")))]
    if kind == "logs":
        return [line for line in lines if re.search(r"\b(?:WARN|ERROR|FATAL)\b", line)]
    return [span.text for span in compress_rules(source).protected_spans]


def judge(client: httpx.Client, model: str, sample: dict) -> tuple[dict, str]:
    required = {"id", "central_preserved", "information_recall", "contradictions", "missing_critical", "reason"}
    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "central_preserved": {"type": "boolean"},
            "information_recall": {"type": "number", "minimum": 0, "maximum": 1},
            "contradictions": {"type": "array", "items": {"type": "string"}},
            "missing_critical": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "required": sorted(required),
        "additionalProperties": False,
    }
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.post(API_URL, json={
                "model": model,
                "temperature": 0,
                "messages": [{"role": "system", "content": SYSTEM}, {
                    "role": "user", "content": json.dumps(sample, ensure_ascii=False),
                }],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "semantic_audit", "strict": True, "schema": schema},
                },
                "max_tokens": 1_500,
                "reasoning": {"enabled": False},
            })
            response.raise_for_status()
            body = response.json()
            result = parse_json(body["choices"][0]["message"]["content"])
            if result.keys() != required or result["id"] != sample["id"]:
                raise ValueError("judge returned invalid schema")
            return result, str(body.get("model", model))
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f"judge failed after retries: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/context-sift"))
    parser.add_argument("--data", type=Path, default=Path("data/validation/real"))
    parser.add_argument("--history", type=Path, default=Path("reports/domain_validation.jsonl"))
    parser.add_argument("--samples", type=Path, default=Path("reports/domain_samples"))
    parser.add_argument("--judge", default="inclusionai/ling-2.6-flash")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--log-chars", type=int, default=50_000)
    args = parser.parse_args()

    compactor = Compactor(args.model)
    args.samples.mkdir(parents=True, exist_ok=True)
    results = []
    pairs = []
    for kind, source in load_samples(args.data, args.log_chars).items():
        start = time.perf_counter()
        compact = compactor(source)
        milliseconds = (time.perf_counter() - start) * 1_000
        required = domain_required(kind, source)
        source_tokens = len(compactor.tokenizer.encode(source))
        compact_tokens = len(compactor.tokenizer.encode(compact))
        (args.samples / f"{kind}.compact.txt").write_text(compact, encoding="utf-8")
        results.append({
            "id": kind,
            "source": SOURCES[kind],
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "compact_sha256": hashlib.sha256(compact.encode()).hexdigest(),
            "source_chars": len(source),
            "compact_chars": len(compact),
            "char_reduction": 1 - len(compact) / max(1, len(source)),
            "source_tokens": source_tokens,
            "compact_tokens": compact_tokens,
            "token_reduction": 1 - compact_tokens / max(1, source_tokens),
            "milliseconds": milliseconds,
            "required_items": len(required),
            "required_exact_recall": exact_recall(required, compact),
        })
        pairs.append({"id": kind, "rubric": RUBRICS[kind], "original": source, "compact": compact})

    judge_models: set[str] = set()
    if not args.skip_judge:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY is required unless --skip-judge")
        headers = {"Authorization": f"Bearer {api_key}", "HTTP-Referer": "https://github.com/compact-llm-summary"}
        with httpx.Client(headers=headers, timeout=240) as client:
            for result, pair in zip(results, pairs):
                evaluation, actual_model = judge(client, args.judge, pair)
                result["semantic_judge"] = evaluation
                judge_models.add(actual_model)
                time.sleep(1)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "judge_models": sorted(judge_models),
        "results": results,
    }
    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
