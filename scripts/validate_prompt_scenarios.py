#!/usr/bin/env python3
"""Validate long pure-text agent prompts and cross-model handoffs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time

import httpx

from context_sift import Compactor
from context_sift.prompt_scenarios import PromptScenario, scenarios
from validate_real_domains import judge


RUBRIC = (
    "Judge whether a new coding model can continue correctly using COMPACT alone. Preserve active goal, "
    "instruction precedence, permissions/prohibitions, decisions and reasons, completed work, failed or "
    "superseded attempts, exact identifiers/paths/commands, unresolved choices, and next action. "
    "Repeated narration and tool descriptions are expendable."
)


def contains_exact(text: str, value: str) -> bool:
    return re.search(r"(?<![\w.])" + re.escape(value) + r"(?!\w)", text) is not None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/context-sift"))
    parser.add_argument("--history", type=Path, default=Path("reports/prompt_validation.jsonl"))
    parser.add_argument("--outputs", type=Path, default=Path("reports/prompt_samples"))
    parser.add_argument("--judge", default="inclusionai/ling-2.6-flash")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skill", type=Path, action="append", default=[])
    args = parser.parse_args()

    compactor = Compactor(args.model)
    args.outputs.mkdir(parents=True, exist_ok=True)
    items = scenarios()
    if args.skill:
        skill_text = "\n\n".join(path.read_text(encoding="utf-8") for path in args.skill)
        handoff = next(item.text for item in items if item.name == "cross_model_session_handoff")
        items.append(PromptScenario(
            "real_skills_plus_model_handoff",
            f"{skill_text}\n\n# SESSION TO CONTINUE\n{handoff}",
            (
                "ACTIVE EVERY RESPONSE",
                'Off only: "stop caveman" / "normal mode"',
                "Drop caveman for: security warnings",
                "Default to context-mode for ALL commands",
                "Always console.log/print your findings",
                "Don't re-index data already in context",
                "ALWAYS use `filename` parameter",
                "fix intermittent refresh-token reuse rejection",
                "AUTH-417",
                "rotate_atomically",
                "User explicitly rejected lowering security checks",
            ),
        ))
    results = []
    pairs = []
    for scenario in items:
        start = time.perf_counter()
        compact = compactor(scenario.text)
        elapsed = (time.perf_counter() - start) * 1_000
        source_tokens = len(compactor.tokenizer.encode(scenario.text))
        output_tokens = len(compactor.tokenizer.encode(compact))
        missing = [value for value in scenario.critical if not contains_exact(compact, value)]
        (args.outputs / f"{scenario.name}.compact.txt").write_text(compact, encoding="utf-8")
        results.append({
            "id": scenario.name,
            "source_chars": len(scenario.text),
            "compact_chars": len(compact),
            "source_tokens": source_tokens,
            "compact_tokens": output_tokens,
            "token_reduction": 1 - output_tokens / source_tokens,
            "milliseconds": elapsed,
            "critical_recall": 1 - len(missing) / len(scenario.critical),
            "missing_critical": missing,
        })
        pairs.append({
            "id": scenario.name,
            "rubric": RUBRIC,
            "original": scenario.text,
            "compact": compact,
        })

    judge_models: set[str] = set()
    if not args.skip_judge:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY is required unless --skip-judge")
        headers = {"Authorization": f"Bearer {api_key}", "HTTP-Referer": "https://github.com/compact-llm-summary"}
        with httpx.Client(headers=headers, timeout=240) as client:
            for result, pair in zip(results, pairs):
                result["semantic_judge"], actual_model = judge(client, args.judge, pair)
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
