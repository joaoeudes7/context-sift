"""Convert extractive source/target pairs into token KEEP/DROP labels."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Sequence


TOKEN_RE = re.compile(
    r"`[^`\n]+`|https?://[^\s]+|(?:/[\w.@+%-]+)+|[\w]+(?:['’-][\w]+)*|[^\w\s]",
    re.UNICODE,
)


class AlignmentError(ValueError):
    """Pair cannot produce trustworthy extractive labels."""


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Alignment:
    tokens: list[str]
    labels: list[int]
    offsets: list[list[int]]
    target_coverage: float
    kept_tokens: int
    protected_tokens: int


def tokenize_with_offsets(text: str) -> list[Token]:
    """Tokenize without losing source character positions."""
    tokens: list[Token] = []
    for match in TOKEN_RE.finditer(text):
        value = match.group()
        end = match.end()
        if (value.startswith("/") or value.startswith("http")) and value[-1:] in ".,;!?":
            tokens.append(Token(value[:-1], match.start(), end - 1))
            tokens.append(Token(value[-1], end - 1, end))
        else:
            tokens.append(Token(value, match.start(), end))
    return tokens


def _normal(token: str) -> str:
    return token.casefold()


def _protected_ranges(source: str, spans: Iterable[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    folded = source.casefold()
    for span in spans:
        needle = str(span).strip().casefold()
        if not needle:
            continue
        start = 0
        while (index := folded.find(needle, start)) >= 0:
            ranges.append((index, index + len(needle)))
            start = index + len(needle)
    return ranges


def _rule_protected_spans(source: str) -> Sequence[str]:
    """Use rules module when present; keep alignment independently usable."""
    try:
        from compact_dataset.rules import compress_rules
    except (ImportError, AttributeError):
        return ()
    return [span.text for span in compress_rules(source).protected_spans]


def align_pair(
    source: str,
    target: str,
    *,
    protected_spans: Iterable[str] = (),
    min_target_coverage: float = 0.8,
) -> Alignment:
    """Label source tokens by exact normalized LCS coverage of target tokens."""
    if not 0 <= min_target_coverage <= 1:
        raise ValueError("min_target_coverage must be between 0 and 1")
    source_tokens = tokenize_with_offsets(source)
    target_tokens = tokenize_with_offsets(target)
    if not source_tokens or not target_tokens:
        raise AlignmentError("source and target must contain tokens")

    source_norm = [_normal(token.text) for token in source_tokens]
    target_norm = [_normal(token.text) for token in target_tokens]
    matcher = SequenceMatcher(None, source_norm, target_norm, autojunk=False)
    labels = [0] * len(source_tokens)
    matched_target = 0
    for source_start, _, size in matcher.get_matching_blocks():
        for index in range(source_start, source_start + size):
            labels[index] = 1
        matched_target += size

    coverage = matched_target / len(target_tokens)
    if coverage < min_target_coverage:
        raise AlignmentError(
            f"target coverage {coverage:.3f} below threshold {min_target_coverage:.3f}"
        )

    spans = [*protected_spans, *_rule_protected_spans(source)]
    ranges = _protected_ranges(source, spans)
    protected_indexes: set[int] = set()
    for index, token in enumerate(source_tokens):
        if any(token.start < end and token.end > start for start, end in ranges):
            labels[index] = 1
            protected_indexes.add(index)

    return Alignment(
        tokens=[token.text for token in source_tokens],
        labels=labels,
        offsets=[[token.start, token.end] for token in source_tokens],
        target_coverage=coverage,
        kept_tokens=sum(labels),
        protected_tokens=len(protected_indexes),
    )
