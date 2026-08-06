"""Verifiable supervision schema for central-message preservation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class UnitRole(StrEnum):
    CORE = "core"
    ACTION = "action"
    CONSTRAINT = "constraint"
    EVIDENCE = "evidence"
    MECHANISM = "mechanism"
    IDENTIFIER = "identifier"
    DETAIL = "detail"
    NOISE = "noise"


@dataclass(frozen=True, slots=True)
class Unit:
    id: str
    start: int
    end: int
    text: str
    role: UnitRole
    protected: bool = False


@dataclass(frozen=True, slots=True)
class Fact:
    id: str
    claim: str
    evidence_unit_ids: tuple[str, ...]
    central: bool = False
    required: bool = False


@dataclass(frozen=True, slots=True)
class VerificationQuestion:
    question: str
    answer_fact_ids: tuple[str, ...]


def validate_supervision(row: dict[str, Any]) -> list[str]:
    """Reject supervision that cannot be traced back to literal source units."""
    errors: list[str] = []
    source = row.get("source")
    if not isinstance(source, str) or not source:
        return ["source_missing"]
    if not isinstance(row.get("central_message"), str) or not row["central_message"].strip():
        errors.append("central_message_missing")

    units: dict[str, Unit] = {}
    for value in row.get("units", []):
        try:
            unit = Unit(
                id=str(value["id"]), start=int(value["start"]), end=int(value["end"]),
                text=str(value["text"]), role=UnitRole(value["role"]),
                protected=bool(value.get("protected", False)),
            )
        except (KeyError, TypeError, ValueError):
            errors.append("invalid_unit")
            continue
        if unit.id in units:
            errors.append(f"duplicate_unit:{unit.id}")
        elif unit.start < 0 or unit.end <= unit.start or source[unit.start:unit.end] != unit.text:
            errors.append(f"unit_evidence_mismatch:{unit.id}")
        else:
            units[unit.id] = unit

    selected = {str(value) for value in row.get("target_unit_ids", [])}
    unknown_selected = selected - units.keys()
    if unknown_selected:
        errors.append(f"unknown_target_units:{sorted(unknown_selected)}")
    protected = {unit.id for unit in units.values() if unit.protected}
    if missing := protected - selected:
        errors.append(f"missing_protected_units:{sorted(missing)}")

    facts: dict[str, Fact] = {}
    for value in row.get("facts", []):
        try:
            fact = Fact(
                id=str(value["id"]), claim=str(value["claim"]).strip(),
                evidence_unit_ids=tuple(str(item) for item in value["evidence_unit_ids"]),
                central=bool(value.get("central", False)),
                required=bool(value.get("required", False)),
            )
        except (KeyError, TypeError):
            errors.append("invalid_fact")
            continue
        if not fact.claim or not fact.evidence_unit_ids:
            errors.append(f"ungrounded_fact:{fact.id}")
        elif len(fact.evidence_unit_ids) > 8:
            errors.append(f"fact_evidence_not_atomic:{fact.id}")
        elif set(fact.evidence_unit_ids) - units.keys():
            errors.append(f"unknown_fact_evidence:{fact.id}")
        elif fact.id in facts:
            errors.append(f"duplicate_fact:{fact.id}")
        else:
            facts[fact.id] = fact
            if (fact.central or fact.required) and not set(fact.evidence_unit_ids) & selected:
                errors.append(f"required_fact_not_selected:{fact.id}")
    if not any(fact.central for fact in facts.values()):
        errors.append("central_fact_missing")
    evidenced = {unit_id for fact in facts.values() for unit_id in fact.evidence_unit_ids}
    if unsupported := selected - evidenced - protected:
        errors.append(f"selected_without_fact:{sorted(unsupported)}")

    for value in row.get("verification_questions", []):
        question = str(value.get("question", "")).strip()
        answer_ids = {str(item) for item in value.get("answer_fact_ids", [])}
        if not question or not answer_ids:
            errors.append("invalid_verification_question")
        elif answer_ids - facts.keys():
            errors.append(f"unknown_question_facts:{sorted(answer_ids - facts.keys())}")
        elif not all(set(facts[fact_id].evidence_unit_ids) & selected for fact_id in answer_ids):
            errors.append(f"unanswerable_verification_question:{question}")
    return errors
