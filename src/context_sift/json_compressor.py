"""Compress JSON arrays and objects before prompt scoring.

Strategy inspired by Headroom SmartCrusher + recursive_json:
- Detect JSON arrays of objects (API responses, tool outputs)
- Keep first/last N items for pagination context
- Keep error/status items (100%)
- Deduplicate near-identical objects
- Summarize remainder with count + stats
- Recursive routing: find embedded JSON spans in text and compress each
"""

from __future__ import annotations

import hashlib
import json

_OPEN = "[{"
_CLOSE = "]}"
_PAIR = {"}": "{", "]": "["}


def _obj_hash(obj: dict) -> str:
    """Stable hash for deduplication (order-insensitive for keys)."""
    return hashlib.md5(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def _is_error_item(obj: dict) -> bool:
    """Detect error/status items that must be preserved."""
    if not isinstance(obj, dict):
        return False
    for key in ("error", "status", "level", "severity"):
        val = str(obj.get(key, "")).lower()
        if val in ("error", "fatal", "critical", "failed", "fail", "exception", "err", "timeout"):
            return True
    return False


def _summarize_array(items: list, original_count: int) -> str:
    """Produce a compact summary line for the dropped items."""
    if not items:
        return ""
    keys = set()
    for item in items[:10]:
        if isinstance(item, dict):
            keys.update(item.keys())
    key_hint = f" keys=[{','.join(sorted(keys)[:8])}]" if keys else ""
    return f"[array: {original_count} items total, {len(items)} summarized{key_hint}]"


def _compress_json_array(items: list, max_keep: int = 6) -> list:
    """Compress a JSON array of objects. Returns the compressed list."""
    if len(items) <= max_keep:
        return items

    result = []
    seen_hashes: set[str] = set()

    # First items (pagination context)
    first_n = min(3, len(items) // 4 + 1)
    for item in items[:first_n]:
        result.append(item)
        seen_hashes.add(_obj_hash(item) if isinstance(item, dict) else str(item))

    # Last items (recency context)
    last_n = min(2, len(items) // 4 + 1)
    for item in items[-last_n:]:
        h = _obj_hash(item) if isinstance(item, dict) else str(item)
        if h not in seen_hashes:
            result.append(item)
            seen_hashes.add(h)

    # Error items (100% preservation)
    for item in items:
        if _is_error_item(item):
            h = _obj_hash(item) if isinstance(item, dict) else str(item)
            if h not in seen_hashes:
                result.append(item)
                seen_hashes.add(h)

    # Summary line for dropped items
    summary = _summarize_array(items, len(items))
    if summary:
        result.append({"_summary": summary})

    return result


def _match_span(text: str, start: int) -> int | None:
    """Index just past the balanced JSON container opening at start."""
    stack: list[str] = []
    in_str = esc = False
    for j in range(start, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in _OPEN:
            stack.append(ch)
        elif ch in _CLOSE:
            if not stack or stack[-1] != _PAIR[ch]:
                return None
            stack.pop()
            if not stack:
                return j + 1
    return None


def _find_embedded_json_spans(text: str) -> list[tuple[int, int]]:
    """Find balanced JSON spans at any offset in text."""
    spans: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] in _OPEN:
            end = _match_span(text, i)
            if end is not None:
                spans.append((i, end))
                i = end
                continue
        i += 1
    return spans


def _has_routable_json(span: str) -> bool:
    """True if span contains an array of objects somewhere."""
    try:
        v = json.loads(span)
    except (ValueError, TypeError):
        return False
    if isinstance(v, list) and len(v) >= 2:
        return sum(isinstance(e, dict) for e in v) >= 0.8 * len(v)
    if isinstance(v, dict):
        for val in v.values():
            if isinstance(val, list) and len(val) >= 2:
                if sum(isinstance(e, dict) for e in val) >= 0.8 * len(val):
                    return True
    return False


def _try_parse_json(text: str) -> tuple[str, object | None]:
    """Detect JSON in text, return (prefix, parsed) or (text, None).

    Handles text that is purely JSON or has leading/trailing prose around JSON.
    """
    stripped = text.strip()

    # Pure JSON (most common for tool outputs)
    if stripped.startswith(("[", "{")):
        try:
            return "", json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            pass

    # Text with embedded JSON block
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        idx = text.find(start_char)
        if idx < 0:
            continue
        # Find matching closing bracket
        depth = 0
        in_string = False
        escape = False
        for i in range(idx, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\" and in_string:
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[idx : i + 1])
                        return text[:idx], parsed
                    except (json.JSONDecodeError, ValueError):
                        break
        break

    return text, None


def _compress_json_value(parsed: object) -> str | None:
    """Compress a parsed JSON value. Returns compacted string or None if no benefit."""
    if isinstance(parsed, list):
        compressed = _compress_json_array(parsed)
        result = json.dumps(compressed, ensure_ascii=False, separators=(",", ":"))
        original = json.dumps(parsed, ensure_ascii=False)
        return result if len(result) < len(original) else None

    if isinstance(parsed, dict):
        compressed = {}
        for key, value in parsed.items():
            if isinstance(value, str) and len(value) > 200:
                compressed[key] = value[:100] + f"...[{len(value)} chars]"
            elif isinstance(value, list) and len(value) > 3:
                compressed[key] = _compress_json_array(value)
            else:
                compressed[key] = value
        result = json.dumps(compressed, ensure_ascii=False, separators=(",", ":"))
        original = json.dumps(parsed, ensure_ascii=False)
        return result if len(result) < len(original) else None

    return None


def compact_json(text: str) -> str:
    """Compress JSON arrays/objects in text. Returns compacted text.

    Handles both top-level JSON and embedded JSON spans in text.
    If no JSON is detected, returns text unchanged.
    """
    # Try top-level JSON first
    prefix, parsed = _try_parse_json(text)
    if parsed is not None:
        compressed = _compress_json_value(parsed)
        if compressed is not None:
            original = json.dumps(parsed, ensure_ascii=False)
            suffix = text[len(prefix) + len(original) :]
            return prefix + compressed + suffix
        return text

    # Try embedded JSON spans (recursive routing)
    spans = _find_embedded_json_spans(text)
    if not spans:
        return text

    # Skip if the whole text is one JSON span (already handled above)
    if len(spans) == 1 and spans[0] == (0, len(text.strip())):
        return text

    parts: list[str] = []
    last = 0
    changed = False
    for a, b in spans:
        chunk = text[a:b]
        try:
            parsed = json.loads(chunk)
        except (json.JSONDecodeError, ValueError):
            continue
        if not _has_routable_json(chunk):
            continue
        compressed = _compress_json_value(parsed)
        if compressed is not None and len(compressed) < len(chunk):
            parts.append(text[last:a])
            parts.append(compressed)
            last = b
            changed = True

    if not changed:
        return text
    parts.append(text[last:])
    return "".join(parts)
