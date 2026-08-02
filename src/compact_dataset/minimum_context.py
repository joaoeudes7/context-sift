"""Greedy minimum sufficient context selection by factual coverage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Candidate:
    id: str
    token_cost: int
    fact_ids: frozenset[str]
    protected: bool = False


def select_minimum_context(
    candidates: list[Candidate],
    fact_weights: dict[str, float],
    required_facts: set[str],
    *,
    min_optional_gain: float = 0.0,
) -> list[str]:
    """Cover required facts, then keep optional information worth its token cost."""
    selected = {item.id for item in candidates if item.protected}
    covered = set().union(*(item.fact_ids for item in candidates if item.id in selected)) if selected else set()
    available = {item.id: item for item in candidates if item.id not in selected}

    while not required_facts <= covered:
        missing = required_facts - covered
        ranked = [
            (sum(fact_weights.get(fact, 1.0) for fact in item.fact_ids & missing) / max(1, item.token_cost), item)
            for item in available.values()
            if item.fact_ids & missing
        ]
        if not ranked:
            raise ValueError(f"required facts have no evidence: {sorted(missing)}")
        _, best = max(ranked, key=lambda value: (value[0], -value[1].token_cost, value[1].id))
        selected.add(best.id)
        covered.update(best.fact_ids)
        del available[best.id]

    while min_optional_gain > 0:
        ranked = [
            (sum(fact_weights.get(fact, 1.0) for fact in item.fact_ids - covered) / max(1, item.token_cost), item)
            for item in available.values()
        ]
        gain, best = max(ranked, default=(0.0, None), key=lambda value: value[0])
        if best is None or gain < min_optional_gain:
            break
        selected.add(best.id)
        covered.update(best.fact_ids)
        del available[best.id]

    order = {item.id: index for index, item in enumerate(candidates)}
    return sorted(selected, key=order.__getitem__)
