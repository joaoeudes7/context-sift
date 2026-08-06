"""Grounded KEEP/DROP/MERGE/REPLACE operation validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from context_sift.rewrite_policy import validate_rewrite


class Operation(StrEnum):
    KEEP = "keep"
    DROP = "drop"
    MERGE = "merge"
    REPLACE = "replace"


@dataclass(frozen=True, slots=True)
class SourceUnit:
    id: str
    text: str
    fact_ids: frozenset[str]
    protected: bool = False


@dataclass(frozen=True, slots=True)
class Transformation:
    operation: Operation
    source_unit_ids: tuple[str, ...]
    output: str
    covered_fact_ids: frozenset[str]


def validate_transformations(
    units: list[SourceUnit], transformations: list[Transformation], required_facts: set[str]
) -> list[str]:
    """Ensure compact rewrites remain grounded and cover mandatory meaning."""
    errors: list[str] = []
    by_id = {unit.id: unit for unit in units}
    consumed: set[str] = set()
    covered: set[str] = set()
    for index, item in enumerate(transformations):
        ids = set(item.source_unit_ids)
        if not ids or ids - by_id.keys():
            errors.append(f"unknown_sources:{index}")
            continue
        if consumed & ids:
            errors.append(f"source_reused:{index}")
        consumed.update(ids)
        source_facts = set().union(*(by_id[unit_id].fact_ids for unit_id in ids))
        if item.covered_fact_ids - source_facts:
            errors.append(f"invented_fact_coverage:{index}")
        if item.operation == Operation.KEEP:
            unit_id = item.source_unit_ids[0] if len(item.source_unit_ids) == 1 else ""
            if not unit_id or item.output != by_id[unit_id].text:
                errors.append(f"invalid_keep:{index}")
        elif item.operation == Operation.DROP:
            if item.output or any(by_id[unit_id].protected for unit_id in ids):
                errors.append(f"invalid_drop:{index}")
        elif item.operation == Operation.MERGE:
            if len(ids) < 2 or not item.output.strip():
                errors.append(f"invalid_merge:{index}")
        elif item.operation == Operation.REPLACE:
            if len(ids) != 1 or not item.output.strip():
                errors.append(f"invalid_replace:{index}")
        if item.operation in {Operation.MERGE, Operation.REPLACE}:
            source = "\n".join(by_id[unit_id].text for unit_id in item.source_unit_ids)
            if not validate_rewrite(source, item.output, "text").allowed:
                errors.append(f"unsafe_rewrite:{index}")
        if item.operation != Operation.DROP:
            covered.update(item.covered_fact_ids)
    if missing := required_facts - covered:
        errors.append(f"missing_required_facts:{sorted(missing)}")
    return errors
