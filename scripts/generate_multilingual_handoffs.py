#!/usr/bin/env python3
"""Generate aligned multilingual model-handoff validation text through OpenRouter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import httpx


API_URL = "https://openrouter.ai/api/v1/chat/completions"
LANGUAGES = {
    "pt": "Portuguese", "es": "Spanish", "fr": "French", "de": "German",
    "ru": "Russian", "ar": "Arabic", "ja": "Japanese", "zh": "Simplified Chinese",
}
SOURCE = """Session handoff from model Aurora to model Boreal. User goal: fix intermittent refresh-token reuse rejection and add regression coverage. Constraint: do not change public API or database schema. Repository is /workspace/identity. Investigation found race in src/auth/rotate_token.py: validation and revocation are separate transactions. Decision: use existing repository method rotate_atomically instead of adding a lock. Reason: method already performs compare-and-swap and works across processes. Completed: reproduced failure with tests/auth/test_rotation.py::test_concurrent_refresh; inspected migration history; confirmed no schema change needed. Failed attempt: process-local asyncio.Lock passed one worker but failed multi-worker test, so it was reverted. No production files have been edited. Pending: patch TokenService.refresh to call rotate_atomically, retain audit event refresh.reuse_detected, then run pytest tests/auth/test_rotation.py -q. User explicitly rejected lowering security checks. Relevant issue: AUTH-417. Expected behavior: exactly one concurrent request succeeds; others return token_reused without invalidating the new token. Do not repeat investigation unless evidence conflicts. Earlier discussion repeatedly concluded the race is cross-process. The local lock is superseded."""
ANCHORS = [
    "/workspace/identity", "src/auth/rotate_token.py", "rotate_atomically", "asyncio.Lock",
    "tests/auth/test_rotation.py::test_concurrent_refresh", "TokenService.refresh",
    "refresh.reuse_detected", "pytest tests/auth/test_rotation.py -q", "AUTH-417", "token_reused",
]
SENTENCES = SOURCE.split(". ")
CRITICAL_INDEXES = (1, 2, 5, 6, 8, 10, 11, 13)


def translate(client: httpx.Client, model: str, code: str, name: str) -> dict:
    placeholders = {anchor: f"__MSC_ANCHOR_{index}__" for index, anchor in enumerate(ANCHORS)}
    protected_sentences = SENTENCES[:]
    for anchor in sorted(ANCHORS, key=len, reverse=True):
        protected_sentences = [sentence.replace(anchor, placeholders[anchor]) for sentence in protected_sentences]
    schema = {
        "type": "object",
        "properties": {
            "sentences": {
                "type": "array", "minItems": len(SENTENCES), "maxItems": len(SENTENCES),
                "items": {"type": "string"},
            },
            "repetition": {"type": "string"},
        },
        "required": ["sentences", "repetition"],
        "additionalProperties": False,
    }
    prompt = f"""Translate each item in SENTENCES into {name}. Return exactly {len(SENTENCES)} items in same order. Keep every fact. Copy every __MSC_ANCHOR_N__ placeholder byte-exact.
repetition: one natural sentence meaning prior discussion confirms cross-process race and local lock is superseded.
SENTENCES:\n{json.dumps(protected_sentences, ensure_ascii=False)}"""
    missing: list[str] = ANCHORS
    for attempt in range(4):
        response = client.post(API_URL, json={
            "model": model, "temperature": 0,
            "messages": [{"role": "system", "content": "Faithful technical translator. JSON only."},
                         {"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "handoff_translation", "strict": True, "schema": schema,
            }},
            "max_tokens": 3_000, "reasoning": {"enabled": False},
        })
        if response.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        value = json.loads(response.json()["choices"][0]["message"]["content"])
        translated = value["sentences"]
        source = "\n".join(translated)
        missing_placeholders = [placeholder for placeholder in placeholders.values() if placeholder not in source]
        if missing_placeholders:
            missing = missing_placeholders
            continue
        for anchor, placeholder in placeholders.items():
            translated = [sentence.replace(placeholder, anchor) for sentence in translated]
        source = "\n".join(translated)
        missing = [anchor for anchor in ANCHORS if anchor not in source]
        if not missing:
            return {
                "language": code, "source": source,
                "critical_phrases": [translated[index] for index in CRITICAL_INDEXES],
                "repetition": value["repetition"], "anchors": ANCHORS,
            }
    raise RuntimeError(f"translation failed: {code}; altered anchors: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="inclusionai/ling-2.6-flash")
    parser.add_argument("--fallback-model", default="openrouter/free")
    parser.add_argument("--output", type=Path, default=Path("data/validation/prompts/multilingual_handoffs.jsonl"))
    args = parser.parse_args()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    english = {
        "language": "en", "source": SOURCE,
        "critical_phrases": [SENTENCES[index] for index in CRITICAL_INDEXES],
        "repetition": "Prior discussion confirms the race is cross-process and the local lock is superseded.",
        "anchors": ANCHORS,
    }
    rows = [english]
    headers = {"Authorization": f"Bearer {api_key}", "HTTP-Referer": "https://github.com/compact-llm-summary"}
    failures = {}
    with httpx.Client(headers=headers, timeout=180) as client:
        for code, name in LANGUAGES.items():
            try:
                rows.append(translate(client, args.model, code, name))
            except RuntimeError:
                try:
                    rows.append(translate(client, args.fallback_model, code, name))
                except RuntimeError as error:
                    failures[code] = str(error)
            time.sleep(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({
        "rows": len(rows), "languages": [row["language"] for row in rows],
        "failures": failures,
    }))


if __name__ == "__main__":
    main()
