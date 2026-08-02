"""End-to-end quality gates for extractive prompt compactors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from statistics import median
from typing import Iterable

from compact_dataset.alignment import tokenize_with_offsets
from compact_dataset.rules import compress_rules


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    examples: int
    protected_spans: int
    retained_protected_spans: int
    protected_span_recall: float
    protected_span_corruptions: int
    median_compression_ratio: float
    extractive_validity: float
    valuable_token_recall: float
    passed: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def _tokens(text: str) -> list[str]:
    return [token.text for token in tokenize_with_offsets(text)]


def is_token_subsequence(source: str, output: str) -> bool:
    """True when every output token occurs in source order."""
    source_tokens = iter(_tokens(source))
    return all(any(candidate == wanted for candidate in source_tokens) for wanted in _tokens(output))


def valuable_token_recall(target: str, output: str) -> float:
    """Ordered recall of teacher/oracle tokens retained in output."""
    wanted = [token.casefold() for token in _tokens(target)]
    actual = [token.casefold() for token in _tokens(output)]
    if not wanted:
        return 1.0
    matcher = SequenceMatcher(None, actual, wanted, autojunk=False)
    matched = sum(size for _, _, size in matcher.get_matching_blocks())
    return matched / len(wanted)


def protected_spans(row: dict) -> list[str]:
    explicit = [str(span) for span in row.get("protected_spans", []) if str(span)]
    detected = [span.text for span in compress_rules(str(row["source"])).protected_spans]
    return list(dict.fromkeys([*explicit, *detected]))


def evaluate_rows(
    rows: Iterable[dict],
    *,
    min_recall: float = 1.0,
    min_ratio: float | None = None,
    max_ratio: float | None = None,
    min_value_recall: float = 0.80,
) -> EvaluationResult:
    materialized = list(rows)
    if not materialized:
        raise ValueError("evaluation dataset is empty")
    ratio_gate_valid = (
        (min_ratio is None and max_ratio is None)
        or (min_ratio is not None and max_ratio is not None and 0 <= min_ratio <= max_ratio)
    )
    if not 0 <= min_recall <= 1 or not 0 <= min_value_recall <= 1 or not ratio_gate_valid:
        raise ValueError("invalid gate thresholds")

    total_spans = retained_spans = 0
    ratios: list[float] = []
    extractive = 0
    value_recalls: list[float] = []
    for row in materialized:
        source = str(row["source"])
        output = str(row["output"])
        spans = protected_spans(row)
        total_spans += len(spans)
        retained_spans += sum(span in output for span in spans)
        source_count = len(_tokens(source))
        ratios.append(len(_tokens(output)) / source_count if source_count else 0.0)
        extractive += int(is_token_subsequence(source, output))
        if "target" in row:
            value_recalls.append(valuable_token_recall(str(row["target"]), output))

    recall = retained_spans / total_spans if total_spans else 1.0
    compression = median(ratios)
    validity = extractive / len(materialized)
    value_recall = sum(value_recalls) / len(value_recalls) if value_recalls else 1.0
    corruptions = total_spans - retained_spans
    passed = (
        recall >= min_recall
        and (min_ratio is None or min_ratio <= compression <= max_ratio)
        and validity == 1.0
        and value_recall >= min_value_recall
        and corruptions == 0
    )
    return EvaluationResult(
        examples=len(materialized),
        protected_spans=total_spans,
        retained_protected_spans=retained_spans,
        protected_span_recall=recall,
        protected_span_corruptions=corruptions,
        median_compression_ratio=compression,
        extractive_validity=validity,
        valuable_token_recall=value_recall,
        passed=passed,
    )
