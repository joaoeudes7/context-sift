#!/usr/bin/env python3
"""Annotate existing sources with universal Minimum Sufficient Context labels."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx

from compact_dataset.msc_annotation import materialize_annotation, source_chunks, teacher_units


API_URL = "https://openrouter.ai/api/v1/chat/completions"
ROLES = "core, action, constraint, evidence, mechanism, identifier, detail, noise"
SYSTEM = f"""Label text for Minimum Sufficient Context training.
Goal: smallest representation preserving same understanding, decisions, central message, facts, relations,
conditions, negations, actions, uncertainty, and reproducibility. No fixed compression ratio.
Grammar, prose quality, and style do not matter. Terse caveman fragments are valid when meaning stays intact.
Never retain tokens merely to improve fluency.
Assign roles only to useful cited units; omitted units become noise/drop. Roles: {ROLES}.
Select evidence units sufficient for all central/required facts. Protected units must be selected.
Every selected unit must uniquely support a selected fact, relation, condition, action, uncertainty, or identifier;
otherwise omit it. If two units convey the same fact, select the cheaper sufficient evidence. Do not preserve background
merely because it is true. Dense information may remain large; redundant information should shrink aggressively.
Facts must be grounded only in supplied unit IDs. Never invent facts or rewrite source units.
Return JSON only with: central_message, unit_roles, facts, verification_questions.
Each fact: id, claim, evidence_unit_ids, central, required.
Each fact/relation may cite at most 8 units. Cite no unit unless it supports that fact.
Each question: question, answer_fact_ids. At least one fact must be central.
"""


def parse_json(content: str) -> dict[str, Any]:
    start = content.find("{")
    if start < 0:
        raise ValueError("teacher returned no JSON object")
    value, _ = json.JSONDecoder().raw_decode(content[start:])
    if not isinstance(value, dict):
        raise ValueError("teacher JSON must be object")
    return value


async def annotate(
    client: httpx.AsyncClient, model: str, source: str, temperature: float = 0.0
) -> tuple[dict[str, Any], str]:
    units = teacher_units(source)
    prompt = json.dumps({"units": [
        {"id": unit["id"], "text": unit["text"], "protected": unit["protected"]}
        for unit in units
    ]}, ensure_ascii=False)
    response = await client.post(API_URL, json={
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 6_000,
        "reasoning": {"enabled": False},
    })
    response.raise_for_status()
    body = response.json()
    annotation = parse_json(body["choices"][0]["message"]["content"])
    return materialize_annotation(source, annotation), str(body.get("model", model))


async def run(args: argparse.Namespace) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    sources = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    existing = {
        json.loads(line)["id"] for line in args.output.read_text(encoding="utf-8").splitlines()
    } if args.output.exists() else set()
    selected = []
    for row in sources:
        for chunk_index, source in enumerate(source_chunks(str(row["source"]), args.max_source_chars)):
            chunk_id = f"{row['id']}:{chunk_index}"
            if chunk_id not in existing:
                selected.append({**row, "id": chunk_id, "source": source})
            if len(selected) >= args.count:
                break
        if len(selected) >= args.count:
            break
    headers = {"Authorization": f"Bearer {api_key}", "HTTP-Referer": "https://github.com/compact-llm-summary"}
    accepted = rejected = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(headers=headers, timeout=180) as client:
        with args.output.open("a", encoding="utf-8") as output:
            for row in selected:
                try:
                    result, teacher_model = await annotate(
                        client, args.model, str(row["source"]), args.temperature
                    )
                    result.update({"id": str(row["id"]), "language": row.get("language"), "teacher_model": teacher_model})
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    accepted += 1
                except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    rejected += 1
                    print(f"rejected {row.get('id')}: {error}")
                await asyncio.sleep(args.delay)
    print(json.dumps({"accepted": accepted, "rejected": rejected, "output": str(args.output)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/msc/train.jsonl"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--model", default="nvidia/nemotron-3-nano-30b-a3b:free")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-source-chars", type=int, default=8_000)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
