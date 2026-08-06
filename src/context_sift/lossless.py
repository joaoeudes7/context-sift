"""Lossless compaction strategies ported from Headroom.

Pure stdlib, no ML. Each transform is reversible or semantically safe:
- collapse_runs: collapse repeated lines (syslog convention)
- fold_repeated_blocks: collapse multi-line blocks that repeat earlier content
- strip_ansi: remove ANSI color codes
- search_heading / search_dir_heading: fold grep output
- path_heading: fold file path listings
- diff_strip_index: strip git diff index lines
- compact_logs: compress log/build output (level scoring, stack traces, dedup)
"""

from __future__ import annotations

import re


# ── ANSI ────────────────────────────────────────────────────────────────

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_MULTI_SPACE_RE = re.compile(r"[^\S\n]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)


def strip_ansi(text: str) -> str:
    """Remove ANSI CSI/SGR (color) escape sequences."""
    return _ANSI_RE.sub("", text)


def collapse_spaces(text: str) -> str:
    """Normalize whitespace: collapse runs of spaces/tabs, trim trailing, collapse blank lines.

    - Multiple spaces/tabs → single space (preserves newlines)
    - Trailing whitespace per line removed
    - 3+ consecutive newlines → 2 newlines
    - Preserves leading indentation (code-safe)
    """
    lines = text.splitlines()
    result = []
    for line in lines:
        # Preserve leading whitespace (indentation)
        stripped = line.lstrip()
        leading = line[: len(line) - len(stripped)] if stripped else ""
        # Collapse internal spaces only
        collapsed = _MULTI_SPACE_RE.sub(" ", stripped)
        result.append(leading + collapsed)
    text = "\n".join(result)
    text = _TRAILING_WS_RE.sub("", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text


# ── Output trimming ────────────────────────────────────────────────────

_CEREMONY_RE = re.compile(
    r"^(?:"
    r"(?:Sure|OK|Okay|Alright|Got it|Let me|I'll|I will|Here's|Here is|"
    r"Great|Thanks|Thank you|Absolutely|Certainly|Of course|No problem|"
    r"Let's|Shall we|I'd be happy to|I can help with that|"
    r"Claro|Claro que sim|Com certeza|Sem problemas|"
    r"Por supuesto|Desde luego|Sin problema|"
    r"Bien sûr|Pas de problème|D'accord)"
    r"[,.! ]+"
    r")",
    re.IGNORECASE,
)
_ECHO_NGRAM = 8


def _echo_ratio(output_text: str, context_text: str, n: int = _ECHO_NGRAM) -> float:
    """Fraction of output n-grams already present in context (measured waste)."""
    out_words = output_text.split()
    if len(out_words) < n:
        return 0.0
    ctx_words = context_text.split()
    if len(ctx_words) < n:
        return 0.0
    ctx_grams = {" ".join(ctx_words[i:i + n]) for i in range(len(ctx_words) - n + 1)}
    out_grams = [" ".join(out_words[i:i + n]) for i in range(len(out_words) - n + 1)]
    if not out_grams:
        return 0.0
    return sum(1 for g in out_grams if g in ctx_grams) / len(out_grams)


def trim_output(text: str, context: str = "") -> str:
    """Trim redundant output: ceremony preambles, echoed code, trailing filler.

    Designed for model output trimming (input-side, no proxy needed):
    - Strips "Sure! Let me..." ceremony preambles
    - Removes trailing "Let me know if..." / "Hope this helps!" filler
    - Drops lines that echo context (high n-gram overlap)
    - Collapses consecutive blank lines

    Args:
        text: Model output to trim.
        context: Original prompt/context (for echo detection).
    """
    lines = text.splitlines()
    result = []
    skip_leading = True

    for line in lines:
        stripped = line.strip()

        # Skip ceremony preamble lines
        if skip_leading and stripped and _CEREMONY_RE.match(stripped):
            continue

        # Stop skipping once we hit real content
        if skip_leading and stripped and not _CEREMONY_RE.match(stripped):
            skip_leading = False

        # Skip trailing filler
        if re.match(
            r"^(?:Let me know|Hope this helps|Feel free to ask|"
            r"Let me know if you|If you need anything|Happy to help|"
            r"Não hesite|Dúvida|Fique à vontade|"
            r"N'hésitez pas|Besoin d'autre|"
            r"Zögern Sie nicht|Lassen Sie mich wissen)",
            stripped,
            re.IGNORECASE,
        ):
            continue

        # Skip echoed lines (high overlap with context)
        if context and stripped and len(stripped.split()) >= _ECHO_NGRAM:
            ratio = _echo_ratio(stripped, context)
            if ratio > 0.6:
                continue

        result.append(line)

    output = "\n".join(result)
    output = _MULTI_NEWLINE_RE.sub("\n\n", output).strip()
    return output if len(output) < len(text) else text


# ── Run collapse ────────────────────────────────────────────────────────

_RUN_MARKER_RE = re.compile(r"^\.\.\. \(repeated (\d+) times\)$")
_BLOCK_MARKER_RE = re.compile(r"^\.\.\. \(repeats (\d+) lines from (\d+) lines back\)$")

_FOLD_MIN_BLOCK = 3
_FOLD_MAX_BLOCK = 64
_FOLD_MAX_CANDIDATES = 8
_FOLD_MAX_LINES = 20_000


def _split_keep_trailing(text: str) -> tuple[list[str], bool]:
    if text == "":
        return [], False
    had_trailing = text.endswith("\n")
    body = text[:-1] if had_trailing else text
    return body.split("\n"), had_trailing


def _join(lines: list[str], had_trailing: bool) -> str:
    out = "\n".join(lines)
    if had_trailing:
        out += "\n"
    return out


def collapse_runs(text: str) -> str:
    """Collapse runs of >=2 identical consecutive lines.

    A run of N lines becomes the line once followed by
    ``... (repeated N times)``.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        j = i
        while j + 1 < n and lines[j + 1] == lines[i]:
            j += 1
        run_len = j - i + 1
        if run_len >= 2:
            out.append(lines[i])
            out.append(f"... (repeated {run_len} times)")
        else:
            out.append(lines[i])
        i = j + 1
    return _join(out, had_trailing)


def expand_runs(text: str) -> str:
    """Exact inverse of collapse_runs."""
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if i + 1 < n:
            m = _RUN_MARKER_RE.match(lines[i + 1])
            if m:
                count = int(m.group(1))
                out.extend([line] * count)
                i += 2
                continue
        out.append(line)
        i += 1
    return _join(out, had_trailing)


# ── Block fold ──────────────────────────────────────────────────────────


def _remember(positions: dict[str, list[int]], line: str, index: int) -> None:
    bucket = positions.setdefault(line, [])
    bucket.append(index)
    if len(bucket) > _FOLD_MAX_CANDIDATES:
        del bucket[0]


def fold_repeated_blocks(text: str) -> str:
    """Collapse multi-line blocks that repeat earlier content into back-refs.

    A block of K consecutive lines (K >= 3) that exactly reproduces K lines
    seen D lines earlier becomes ``... (repeats K lines from D lines back)``.
    """
    lines, had_trailing = _split_keep_trailing(text)
    n = len(lines)
    if n < _FOLD_MIN_BLOCK * 2 or n > _FOLD_MAX_LINES:
        return text
    positions: dict[str, list[int]] = {}
    out: list[str] = []
    i = 0
    while i < n:
        best_len = 0
        best_dist = 0
        for q in reversed(positions.get(lines[i], ())):
            max_len = min(_FOLD_MAX_BLOCK, n - i, i - q)
            length = 0
            while length < max_len and lines[q + length] == lines[i + length]:
                length += 1
            if length > best_len:
                best_len = length
                best_dist = i - q
        if best_len >= _FOLD_MIN_BLOCK:
            marker = f"... (repeats {best_len} lines from {best_dist} lines back)"
            block_chars = sum(len(lines[i + k]) + 1 for k in range(best_len))
            if block_chars > len(marker) + 1:
                out.append(marker)
                for k in range(best_len):
                    _remember(positions, lines[i + k], i + k)
                i += best_len
                continue
        _remember(positions, lines[i], i)
        out.append(lines[i])
        i += 1
    return _join(out, had_trailing)


def unfold_repeated_blocks(text: str) -> str:
    """Exact inverse of fold_repeated_blocks."""
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    for line in lines:
        m = _BLOCK_MARKER_RE.match(line)
        if m:
            length, dist = int(m.group(1)), int(m.group(2))
            start = len(out) - dist
            if start >= 0 and length <= dist:
                out.extend(out[start : start + length])
                continue
        out.append(line)
    return _join(out, had_trailing)


# ── Grep/search heading ────────────────────────────────────────────────

_GREP_ROW_RE = re.compile(r"^(?P<path>[^\n:]+):(?P<line>\d+):(?P<content>.*)$")
_HEADING_ROW_RE = re.compile(r"^(?P<line>\d+):(?P<content>.*)$")
_DIR_DATA_RE = re.compile(r"^(?P<base>[^/\n:]+):(?P<line>\d+):(?P<content>.*)$")
_PATH_ROW_RE = re.compile(r"^(?P<dir>(?:\.{0,2}/)?(?:[^/\s:]+/)+)(?P<base>[^/\s:]+)$")


def search_heading(text: str) -> str:
    """Fold grep path:line:content rows into ripgrep --heading form."""
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    current_path: str | None = None
    for line in lines:
        m = _GREP_ROW_RE.match(line)
        if m:
            path = m.group("path")
            if path != current_path:
                out.append(path)
                current_path = path
            out.append(f"{m.group('line')}:{m.group('content')}")
        else:
            out.append(line)
            current_path = None
    return _join(out, had_trailing)


def search_unheading(text: str) -> str:
    """Exact inverse of search_heading."""
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    current_path: str | None = None
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        data = _HEADING_ROW_RE.match(line)
        if current_path is not None and data:
            out.append(f"{current_path}:{data.group('line')}:{data.group('content')}")
            i += 1
            continue
        if not data and i + 1 < n and _HEADING_ROW_RE.match(lines[i + 1]):
            current_path = line
            i += 1
            continue
        current_path = None
        out.append(line)
        i += 1
    return _join(out, had_trailing)


def search_dir_heading(text: str) -> str:
    """Fold grep rows by directory (repeated dir across files)."""
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    current_dir: str | None = None
    for line in lines:
        m = _GREP_ROW_RE.match(line)
        if m and "/" in m.group("path"):
            path = m.group("path")
            cut = path.rindex("/") + 1
            dir_part, base = path[:cut], path[cut:]
            if dir_part != current_dir:
                out.append(dir_part)
                current_dir = dir_part
            out.append(f"{base}:{m.group('line')}:{m.group('content')}")
        else:
            out.append(line)
            current_dir = None
    return _join(out, had_trailing)


def path_heading(text: str) -> str:
    """Fold file path listing into directory + basenames."""
    lines, had_trailing = _split_keep_trailing(text)
    if sum(1 for ln in lines if _PATH_ROW_RE.match(ln)) < 2:
        return text
    out: list[str] = []
    current: str | None = None
    for line in lines:
        m = _PATH_ROW_RE.match(line)
        if m:
            d = m.group("dir")
            if d != current:
                out.append(d)
                current = d
            out.append(m.group("base"))
        else:
            out.append(line)
            current = None
    return _join(out, had_trailing)


# ── Diff index strip ────────────────────────────────────────────────────

_DIFF_INDEX_RE = re.compile(r"^index [0-9a-fA-F]+\.\.[0-9a-fA-F]+( [0-7]+)?$")


def diff_strip_index(text: str) -> str:
    """Drop index sha..sha lines from unified diff."""
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out = [line for line in lines if not _DIFF_INDEX_RE.match(line)]
    return _join(out, had_trailing)


# ── Log compression ─────────────────────────────────────────────────────

_LOG_LEVEL_RE = re.compile(
    r"\b(FATAL|ERROR|ERR|FAILED|FAIL|CRITICAL|WARN(?:ING)?|INFO|DEBUG|TRACE)\b",
    re.IGNORECASE,
)
_STACK_TRACE_RE = re.compile(
    r"^\s*(Traceback \(most recent call last\)|"
    r"at .+\(.+:\d+:\d+\)|"
    r"File \".+\", line \d+|"
    r".*Error:|.*Exception:|.*Failed to)",
    re.IGNORECASE,
)
_SUMMARY_RE = re.compile(
    r"(passed|failed|error|ok|skipped|total|summary|result)",
    re.IGNORECASE,
)


def _score_log_line(line: str) -> float:
    """Score a log line by importance. Higher = more important."""
    m = _LOG_LEVEL_RE.search(line)
    if m:
        level = m.group(1).upper()
        if level in ("FATAL", "ERROR", "ERR", "FAILED", "FAIL", "CRITICAL"):
            return 1.0
        if level in ("WARN", "WARNING"):
            return 0.7
        if level in ("INFO",):
            return 0.3
        if level in ("DEBUG", "TRACE"):
            return 0.1
    if _STACK_TRACE_RE.match(line):
        return 0.8
    if _SUMMARY_RE.search(line):
        return 0.9
    return 0.2


def compact_logs(text: str) -> str:
    """Compress log/build output: keep errors, warnings, summaries, first/last.

    Strategy:
    - Keep all ERROR/FATAL/CRITICAL lines (100%)
    - Keep first and last WARNING lines
    - Keep stack trace heads (message + first 3 frames)
    - Keep summary/pass/fail lines
    - Deduplicate similar warnings
    - Collapse repeated lines
    - Strip ANSI color codes
    """
    text = strip_ansi(text)
    lines, had_trailing = _split_keep_trailing(text)
    if not lines or len(lines) < 10:
        return text

    scored = [(i, _score_log_line(line)) for i, line in enumerate(lines)]
    keep = set()

    # Always keep high-score lines
    for i, score in scored:
        if score >= 0.7:
            keep.add(i)

    # Keep first and last 2 lines for context
    for i in range(min(2, len(lines))):
        keep.add(i)
    for i in range(max(0, len(lines) - 2), len(lines)):
        keep.add(i)

    # Keep stack trace heads (traceback + first 3 non-blank frames)
    in_trace = False
    trace_count = 0
    for i, line in enumerate(lines):
        if "Traceback (most recent call last)" in line:
            in_trace = True
            trace_count = 0
            keep.add(i)
            continue
        if in_trace:
            if trace_count < 3 and line.strip():
                keep.add(i)
                trace_count += 1
            elif not line.strip() or _LOG_LEVEL_RE.search(line):
                in_trace = False

    # Deduplicate warnings/info: keep first occurrence only
    seen_warnings: set[str] = set()
    for i in sorted(keep):
        if scored[i][1] < 0.8:
            normalized = re.sub(r"\d+", "N", lines[i]).strip()
            if normalized in seen_warnings:
                keep.discard(i)
            else:
                seen_warnings.add(normalized)
    for i, score in scored:
        if 0.2 <= score < 0.8 and i not in keep:
            normalized = re.sub(r"\d+", "N", lines[i]).strip()
            if normalized not in seen_warnings:
                seen_warnings.add(normalized)
                keep.add(i)

    # Build output
    result = []
    for i, line in enumerate(lines):
        if i in keep:
            result.append(line)
        elif i > 0 and i - 1 in keep:
            result.append("  ...")

    output = _join(result, had_trailing)
    if len(output) >= len(text):
        return text
    return output
