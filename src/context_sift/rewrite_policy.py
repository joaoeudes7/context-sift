"""Hard validation gates for bounded abstractive rewriting."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from context_sift.rules import compress_rules


@dataclass(frozen=True, slots=True)
class RewriteValidation:
    allowed: bool
    reasons: tuple[str, ...]


def _critical(text: str) -> set[str]:
    return {
        span.text
        for span in compress_rules(text).protected_spans
        if not span.text.endswith((":", "："))
    }


def _line_subsequence(source: str, output: str) -> bool:
    lines = iter(source.splitlines())
    return all(any(candidate == wanted for candidate in lines) for wanted in output.splitlines())


def _leaves(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _leaves(item)]
    if isinstance(value, list):
        return [leaf for item in value for leaf in _leaves(item)]
    return [value]


def validate_rewrite(source: str, output: str, source_type: str) -> RewriteValidation:
    reasons: list[str] = []
    if source_type == "binary":
        return RewriteValidation(False, ("binary_requires_decoder",))
    if source_type in {"code", "git_diff"}:
        if not _line_subsequence(source, output):
            reasons.append("code_not_copy_only")
        return RewriteValidation(not reasons, tuple(reasons))
    if source_type == "toon" and source != output:
        return RewriteValidation(False, ("toon_parser_required",))
    if source_type == "json":
        try:
            source_json, output_json = json.loads(source), json.loads(output)
        except json.JSONDecodeError:
            return RewriteValidation(False, ("invalid_json",))
        source_leaves = _leaves(source_json)
        if any(leaf not in source_leaves for leaf in _leaves(output_json)):
            reasons.append("invented_json_value")

    source_critical = _critical(source)
    output_critical = _critical(output)
    missing = source_critical - output_critical
    invented = output_critical - source_critical
    if missing:
        reasons.append("missing_critical_value")
    if invented:
        reasons.append("invented_critical_value")
    if not output.strip():
        reasons.append("empty_output")
    return RewriteValidation(not reasons, tuple(reasons))
