"""Lossless-for-changes compaction of unified git diffs."""

from __future__ import annotations


_META_PREFIXES = (
    "diff --git ",
    "index ",
    "--- ",
    "+++ ",
    "@@ ",
    "new file mode ",
    "deleted file mode ",
    "similarity index ",
    "rename from ",
    "rename to ",
)


def compact_git_diff(text: str, context_lines: int = 2) -> str:
    """Keep all diff metadata/changes and nearby context, byte-exact per line."""
    lines = text.splitlines()
    if not any(line.startswith("diff --git ") for line in lines):
        return text
    keep = {
        index
        for index, line in enumerate(lines)
        if line.startswith(_META_PREFIXES)
        or (line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
    }
    change_indexes = [
        index
        for index, line in enumerate(lines)
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    for index in change_indexes:
        keep.update(range(max(0, index - context_lines), min(len(lines), index + context_lines + 1)))
        if index + 1 < len(lines) and lines[index + 1] == "\\ No newline at end of file":
            keep.add(index + 1)
    return "\n".join(line for index, line in enumerate(lines) if index in keep)
