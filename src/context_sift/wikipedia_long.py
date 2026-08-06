"""Create long, copy-only clause supervision from Wikipedia articles."""

from __future__ import annotations

from context_sift.clause_dataset import split_clauses
from context_sift.rules import compress_rules


def chunk_text(text: str, min_chars: int = 25_000, max_chars: int = 50_000) -> list[str]:
    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        if current and size + len(paragraph) + 2 > max_chars:
            if size >= min_chars:
                chunks.append("\n\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph) + 2
    if current and size >= min_chars:
        chunks.append("\n\n".join(current))
    return chunks


def build_long_rows(title: str, url: str, text: str) -> list[dict[str, object]]:
    rows = []
    for chunk_index, source in enumerate(chunk_text(text)):
        units = split_clauses(source)
        protected = compress_rules(source).protected_spans
        paragraph_starts = {0}
        start = 0
        while (index := source.find("\n\n", start)) >= 0:
            paragraph_starts.add(index + 2)
            start = index + 2
        labels = [
            int(
                any(unit.start <= position < unit.end for position in paragraph_starts)
                or any(unit.start < span.end and unit.end > span.start for span in protected)
            )
            for unit in units
        ]
        if units and not any(labels):
            labels[0] = 1
        target = " ".join(unit.text for unit, keep in zip(units, labels) if keep)
        rows.append({
            "id": f"wiki-{title}-{chunk_index}",
            "source": source,
            "target": target,
            "units": [
                {"text": unit.text, "range": [unit.start, unit.end]} for unit in units
            ],
            "labels": labels,
            "protected_spans": [span.text for span in protected],
            "metadata": {"title": title, "url": url, "chunk": chunk_index},
        })
    return rows
