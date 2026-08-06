"""Concurrent, resumable OpenRouter synthetic-data generation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openrouter/free"
LANGUAGES = ("en", "pt")
TOKEN_RE = re.compile(r"[\w@./:+%-]+", re.UNICODE)
PROTECTED_RE = re.compile(
    r"https?://\S+|`[^`]+`|\b\d+(?:[.,]\d+)*(?:%|[A-Za-z]+)?\b"
)
_BAND_MASK = (1 << 32) - 1

SYSTEM_PROMPT = """You create supervised examples for a tiny bilingual prompt compactor.
Return only schema-valid JSON. Each source must be a realistic, self-contained user prompt,
not an article summary request. Each target must:
- use terse caveman-style fragments;
- retain every rule, request, target, constraint, negation, name, number, date, URL, and code span;
- remove filler and repeated meaning;
- never add facts;
- remain in source language;
- use 25-40% as many words as source.
Create varied software, business, planning, research, and everyday prompts. Avoid unsafe or private data.
"""


@dataclass(frozen=True, slots=True)
class BuildConfig:
    output: Path
    count: int = 100
    batch_size: int = 4
    concurrency: int = 4
    min_source_words: int = 180
    max_source_words: int = 700
    min_ratio: float = 0.25
    max_ratio: float = 0.40
    max_attempts: int = 5
    model: str = MODEL
    seed: int = 7
    structured_output: bool = True

    def __post_init__(self) -> None:
        if self.count < 1 or self.batch_size < 1 or self.concurrency < 1 or self.max_attempts < 1:
            raise ValueError("count, batch-size, concurrency, and max-attempts must be positive")
        if self.min_source_words < 1:
            raise ValueError("min-source-words must be positive")
        if self.min_source_words > self.max_source_words:
            raise ValueError("min-source-words must not exceed max-source-words")
        if not 0 < self.min_ratio <= 1:
            raise ValueError("min-ratio must be in (0, 1]")
        if self.min_ratio > self.max_ratio:
            raise ValueError("min-ratio must not exceed max-ratio")


@dataclass(frozen=True, slots=True)
class Example:
    id: str
    language: str
    source: str
    target: str
    protected_spans: list[str]
    teacher_model: str
    created_at: int


class DatasetError(RuntimeError):
    """Dataset generation or validation failed."""


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def words(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def simhash(text: str) -> int:
    """Return stable 64-bit SimHash for cheap near-duplicate checks."""
    weights = [0] * 64
    tokens = words(normalize(text))
    shingles = (" ".join(tokens[i : i + 3]) for i in range(max(1, len(tokens) - 2)))
    for shingle in shingles:
        value = int.from_bytes(hashlib.blake2b(shingle.encode(), digest_size=8).digest())
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    return sum(1 << bit for bit, weight in enumerate(weights) if weight >= 0)


def repeated_sentences(text: str) -> bool:
    sentences = [normalize(part) for part in re.split(r"[.!?\n]+", text) if len(words(part)) >= 4]
    return len(sentences) != len(set(sentences))


def parse_examples(content: str) -> list[dict[str, Any]]:
    if content.strip().startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    parsed = json.loads(content)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and isinstance(parsed.get("examples"), list):
        return parsed["examples"]
    if isinstance(parsed, dict) and {"language", "source", "target"} <= parsed.keys():
        return [parsed]
    raise DatasetError("response JSON has no examples")


def validate_item(
    item: dict[str, Any],
    min_ratio: float,
    max_ratio: float,
    min_source_words: int,
    max_source_words: int,
) -> tuple[str, str, str, list[str]]:
    try:
        language = str(item["language"]).strip().lower()
        source = str(item["source"]).strip()
        target = str(item["target"]).strip()
        spans_value = item.get("protected_spans", [])
        if isinstance(spans_value, str):
            spans_value = [spans_value]
        claimed = [str(span).strip() for span in spans_value]
    except (KeyError, TypeError) as error:
        raise DatasetError("missing example field") from error

    source_count = len(words(source))
    target_count = len(words(target))
    ratio = target_count / source_count if source_count else math.inf
    if language not in LANGUAGES:
        raise DatasetError(f"unsupported language: {language}")
    if not min_source_words <= source_count <= max_source_words:
        raise DatasetError(f"source words outside bounds: {source_count}")
    if not min_ratio <= ratio <= max_ratio:
        raise DatasetError(f"compression ratio outside bounds: {ratio:.2f}")
    if repeated_sentences(target):
        raise DatasetError("target repeats sentence")

    extracted = PROTECTED_RE.findall(source)
    protected = list(dict.fromkeys(span for span in [*claimed, *extracted] if span))
    missing = [span for span in extracted if normalize(span) not in normalize(target)]
    if missing:
        raise DatasetError(f"target misses protected spans: {missing[:3]}")
    return language, source, target, protected


class DatasetBuilder:
    def __init__(self, config: BuildConfig, api_key: str | None = None) -> None:
        self.config = config
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise DatasetError("OPENROUTER_API_KEY is required")
        self._write_lock = asyncio.Lock()
        self._seen: set[str] = set()
        self._simhash_bands: dict[int, dict[int, list[int]]] = {0: {}, 1: {}}
        self._accepted = 0
        self._rng = random.Random(config.seed)

    async def build(self) -> int:
        self._load_existing()
        remaining = max(0, self.config.count - self._accepted)
        if remaining == 0:
            return self._accepted

        request_count = math.ceil(remaining / self.config.batch_size)
        semaphore = asyncio.Semaphore(self.config.concurrency)
        timeout = httpx.Timeout(120.0, connect=20.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            tasks = [self._request_batch(client, semaphore, index) for index in range(request_count)]
            await asyncio.gather(*tasks)
        return self._accepted

    def _load_existing(self) -> None:
        try:
            lines = self.config.output.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                source = str(item["source"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise DatasetError(f"invalid JSONL line {line_number}") from error
            self._seen.add(fingerprint(source))
            self._index_simhash(simhash(source))
            self._accepted += 1

    def _index_simhash(self, value: int) -> None:
        for band, shift in ((0, 32), (1, 0)):
            key = (value >> shift) & _BAND_MASK
            self._simhash_bands[band].setdefault(key, []).append(value)

    def _is_near_duplicate(self, value: int) -> bool:
        """Exact Hamming<=3 check, indexed to avoid scanning all history.

        Two 64-bit values at Hamming distance <=3 agree on at least one 32-bit
        half within <=1 flipped bit, so probing each half plus its 32 one-bit
        neighbors finds every match without false positives.
        """
        for band, shift in ((0, 32), (1, 0)):
            base = (value >> shift) & _BAND_MASK
            for flip in range(33):
                key = base if flip == 0 else base ^ (1 << (flip - 1))
                for existing in self._simhash_bands[band].get(key, ()):
                    if (value ^ existing).bit_count() <= 3:
                        return True
        return False

    async def _request_batch(
        self, client: httpx.AsyncClient, semaphore: asyncio.Semaphore, batch_index: int
    ) -> None:
        async with semaphore:
            for attempt in range(self.config.max_attempts):
                try:
                    response = await client.post(
                        API_URL,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "https://github.com/compact-llm-summary",
                            "X-Title": "Compact LLM Dataset Builder",
                        },
                        json=self._payload(batch_index),
                    )
                    if response.status_code == 429:
                        delay = float(response.headers.get("Retry-After", 2**attempt))
                        await asyncio.sleep(min(delay, 60.0))
                        continue
                    response.raise_for_status()
                    await self._save_response(response.json())
                    return
                except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    if attempt + 1 == self.config.max_attempts:
                        print(f"batch {batch_index} failed: {error}")
                        return
                    await asyncio.sleep(min(2**attempt + self._rng.random(), 30.0))

    def _payload(self, batch_index: int) -> dict[str, Any]:
        count = self.config.batch_size
        schema = {
            "name": "compaction_examples",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "examples": {
                        "type": "array",
                        "minItems": count,
                        "maxItems": count,
                        "items": {
                            "type": "object",
                            "properties": {
                                "language": {"type": "string", "enum": list(LANGUAGES)},
                                "source": {"type": "string"},
                                "target": {"type": "string"},
                                "protected_spans": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["language", "source", "target", "protected_spans"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["examples"],
                "additionalProperties": False,
            },
        }
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Create {count} unique examples, balanced English and Portuguese. "
                        f"Each source: {self.config.min_source_words}-{self.config.max_source_words} words. "
                        f"Batch diversity nonce: {self.config.seed}-{batch_index}. "
                        'Set language to exactly "en" or "pt", never "en|pt". '
                        'Return {"examples":[{"language":"en","source":"...",'
                        '"target":"...","protected_spans":["..."]}]}.'
                    ),
                },
            ],
            "temperature": 0.9,
            "max_tokens": min(16_000, count * int(self.config.max_source_words * 1.8 + 256)),
        }
        if self.config.structured_output:
            payload["response_format"] = {"type": "json_schema", "json_schema": schema}
            payload["provider"] = {"require_parameters": True}
        else:
            payload["reasoning"] = {"enabled": False}
        return payload

    async def _save_response(self, body: dict[str, Any]) -> None:
        model = str(body.get("model", self.config.model))
        content = body["choices"][0]["message"]["content"]
        candidates: list[Example] = []
        for item in parse_examples(content):
            try:
                language, source, target, protected = validate_item(
                    item,
                    self.config.min_ratio,
                    self.config.max_ratio,
                    self.config.min_source_words,
                    self.config.max_source_words,
                )
            except DatasetError as error:
                print(f"rejected example: {error}")
                continue
            candidates.append(
                Example(
                    id=fingerprint(source)[:16],
                    language=language,
                    source=source,
                    target=target,
                    protected_spans=protected,
                    teacher_model=model,
                    created_at=int(time.time()),
                )
            )

        async with self._write_lock:
            self.config.output.parent.mkdir(parents=True, exist_ok=True)
            with self.config.output.open("a", encoding="utf-8") as output:
                for example in candidates:
                    if self._accepted >= self.config.count:
                        break
                    source_hash = fingerprint(example.source)
                    source_simhash = simhash(example.source)
                    near_duplicate = self._is_near_duplicate(source_simhash)
                    if source_hash in self._seen or near_duplicate:
                        continue
                    output.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
                    output.flush()
                    self._seen.add(source_hash)
                    self._index_simhash(source_simhash)
                    self._accepted += 1
