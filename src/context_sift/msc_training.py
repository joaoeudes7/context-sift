"""Convert heterogeneous extractive labels into MSC recurrent training windows."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Protocol

from context_sift.clause_dataset import split_clauses


class Tokenizer(Protocol):
    def encode(self, text: str, **kwargs) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class TrainingWindow:
    id: str
    units: list[list[int]]
    labels: list[int]


def group_key(row: dict[str, Any]) -> str:
    """Keep chunks and translations of one source concept in one split."""
    if row.get("group_id"):
        return str(row["group_id"])
    metadata = row.get("metadata", {})
    if metadata.get("url"):
        return str(metadata["url"])
    return str(row.get("id", "row")).rsplit(":", 1)[0]


def split_grouped_rows(
    rows: list[dict[str, Any]], validation_ratio: float, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)
    ranked = sorted(
        groups,
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).digest(),
    )
    validation_count = max(1, round(len(ranked) * validation_ratio))
    validation_keys = set(ranked[:validation_count])
    train = [row for key, values in groups.items() if key not in validation_keys for row in values]
    valid = [row for key, values in groups.items() if key in validation_keys for row in values]
    return train, valid


def training_windows(
    row: dict[str, Any], tokenizer: Tokenizer, *, max_units: int = 64, max_unit_tokens: int = 256,
) -> list[TrainingWindow]:
    source = str(row["source"])
    raw_units = row.get("units")
    if raw_units:
        texts = [str(unit["text"]) for unit in raw_units]
    else:
        texts = [clause.text for clause in split_clauses(source)]
    if "target_unit_ids" in row:
        if not raw_units:
            raise ValueError("target_unit_ids require row units")
        selected = {str(value) for value in row["target_unit_ids"]}
        labels = [int(str(unit.get("id", f"u{index}")) in selected) for index, unit in enumerate(raw_units)]
    elif "labels" in row:
        labels = [int(value) for value in row["labels"]]
    else:
        from context_sift.word_pruner import sentence_keep_labels

        labels = sentence_keep_labels(source, str(row["target"]))
    if not texts or len(texts) != len(labels):
        raise ValueError("units and labels must be non-empty and aligned")
    encoded = [
        tokenizer.encode(text, out_type=int, add_bos=True, add_eos=True)[:max_unit_tokens]
        for text in texts
    ]
    row_id = str(row.get("id", "row"))
    return [
        TrainingWindow(f"{row_id}:{start // max_units}", encoded[start:start + max_units], labels[start:start + max_units])
        for start in range(0, len(encoded), max_units)
    ]
