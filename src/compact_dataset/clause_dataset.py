"""Build clause-level KEEP/DROP supervision from extractive summaries."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from compact_dataset.alignment import Alignment, align_pair, tokenize_with_offsets


_CLAUSE_RE = re.compile(
    r".+?(?:[!?;。！？；؟؛]+|[,،](?=\s)|\.(?=\s|$)|(?=\n|$))",
    re.UNICODE,
)


@dataclass(frozen=True, slots=True)
class Clause:
    text: str
    start: int
    end: int


def split_clauses(text: str) -> list[Clause]:
    """Split at clause punctuation while preserving exact source offsets."""
    clauses = []
    for match in _CLAUSE_RE.finditer(text):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            clauses.append(Clause(text[start:end], start, end))
    return clauses


def _token_indexes(clause: Clause, alignment: Alignment) -> list[int]:
    return [
        index
        for index, (start, end) in enumerate(alignment.offsets)
        if start < clause.end and end > clause.start
    ]


def build_clause_row(
    source: str,
    target: str,
    *,
    protected_spans: Iterable[str] = (),
    min_target_coverage: float = 0.8,
    keep_threshold: float = 0.5,
) -> dict[str, object]:
    """Return clause units, labels, and oracle compression statistics."""
    if not 0 <= keep_threshold <= 1:
        raise ValueError("keep_threshold must be between 0 and 1")
    alignment = align_pair(
        source,
        target,
        protected_spans=protected_spans,
        min_target_coverage=min_target_coverage,
    )
    clauses = split_clauses(source)
    if not clauses:
        raise ValueError("source must contain a clause")

    indexes = [_token_indexes(clause, alignment) for clause in clauses]
    scores = [
        sum(alignment.labels[index] for index in unit) / len(unit) if unit else 0.0
        for unit in indexes
    ]
    labels = [int(score >= keep_threshold and score > 0) for score in scores]
    if not any(labels):
        labels[max(range(len(scores)), key=scores.__getitem__)] = 1

    kept_indexes = {
        index
        for label, unit in zip(labels, indexes)
        if label
        for index in unit
    }
    source_tokens = len(alignment.tokens)
    oracle_ratio = len(kept_indexes) / source_tokens
    target_ratio = len(tokenize_with_offsets(target)) / source_tokens
    return {
        "source": source,
        "units": [
            {"text": clause.text, "range": [clause.start, clause.end]}
            for clause in clauses
        ],
        "labels": labels,
        "stats": {
            "source_tokens": source_tokens,
            "kept_units": sum(labels),
            "unit_count": len(clauses),
            "keep_ratio": sum(labels) / len(labels),
            "target_token_ratio": target_ratio,
            "oracle_token_ratio": oracle_ratio,
            "oracle_gap": oracle_ratio - target_ratio,
            "target_coverage": alignment.target_coverage,
        },
    }
