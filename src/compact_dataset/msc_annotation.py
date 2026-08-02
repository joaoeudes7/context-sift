"""Build locally grounded MSC supervision from teacher labels."""

from __future__ import annotations

import re
from typing import Any

from compact_dataset.clause_dataset import split_clauses
from compact_dataset.supervision import UnitRole, validate_supervision


_HARD_LITERAL_RE = re.compile(
    r"`[^`\n]+`|\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"
    r"|\b(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]+\b",
    re.IGNORECASE,
)
_ROLE_ALIASES = {
    "central_claim": "core",
    "supporting_evidence": "evidence",
    "required_reproduction": "mechanism",
    "context": "detail",
    "redundant": "noise",
}


def source_chunks(source: str, max_chars: int) -> list[str]:
    """Split long teacher input at nearby whitespace without losing source bytes."""
    if max_chars < 200:
        raise ValueError("max_chars must be at least 200")
    chunks = []
    start = 0
    while len(source) - start > max_chars:
        floor = start + int(max_chars * 0.7)
        end = max(source.rfind("\n", floor, start + max_chars), source.rfind(" ", floor, start + max_chars))
        end = end + 1 if end >= floor else start + max_chars
        chunks.append(source[start:end])
        start = end
    if start < len(source):
        chunks.append(source[start:])
    return chunks


def teacher_units(source: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"u{index}", "start": clause.start, "end": clause.end, "text": clause.text,
            "protected": bool(_HARD_LITERAL_RE.search(clause.text)),
        }
        for index, clause in enumerate(split_clauses(source))
    ]


def materialize_annotation(source: str, annotation: dict[str, Any]) -> dict[str, Any]:
    """Attach teacher labels to exact local units and validate factual coverage."""
    units = teacher_units(source)
    raw_roles = annotation["unit_roles"]
    roles = (
        {str(unit_id): str(role) for unit_id, role in raw_roles.items()}
        if isinstance(raw_roles, dict)
        else {str(item["unit_id"]): str(item["role"]) for item in raw_roles}
    )
    valid_ids = {unit["id"] for unit in units}
    if roles.keys() - valid_ids:
        raise ValueError("teacher assigned role to unknown unit")
    for unit in units:
        role = roles.get(unit["id"], UnitRole.NOISE)
        unit["role"] = UnitRole(_ROLE_ALIASES.get(str(role), str(role))).value
    facts = annotation["facts"]
    evidenced = {str(unit_id) for fact in facts for unit_id in fact["evidence_unit_ids"]}
    selected = evidenced | {unit["id"] for unit in units if unit["protected"]}
    row = {
        "source": source,
        "central_message": annotation["central_message"],
        "units": units,
        "target_unit_ids": [unit["id"] for unit in units if unit["id"] in selected],
        "facts": facts,
        "verification_questions": annotation.get("verification_questions", []),
    }
    if errors := validate_supervision(row):
        raise ValueError("; ".join(errors))
    return row
