"""Deterministic, extractive text compaction."""

from __future__ import annotations

from dataclasses import dataclass
import re


_TOKEN_RE = re.compile(r"\w+(?:[-']\w+)*|https?://\S+|`[^`]*`", re.UNICODE)
_PROTECTED_RE = re.compile(
    r"```[\s\S]*?```"
    r"|(?:^|(?<=[.!?。！？]\s))[^.!?。！？:：\n]{1,48}[:：]"
    r"|`[^`\n]+`"
    r"|https?://[^\s<>]+"
    r"|\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"
    r"|(?<![A-Za-z0-9_.-])\.[A-Za-z_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_-])?(?=$|[\s,;:!?)}\]]|\.(?=\s|$)|[^\x00-\x7F])"
    r"|(?<![A-Za-z0-9_.-])/?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]*[A-Za-z0-9_-](?:::[A-Za-z0-9_.-]*[A-Za-z0-9_-])*(?=$|[\s,;:!?)}\]]|\.(?=\s|$)|[^\x00-\x7F])"
    r"|\b\d+(?:[./,:-]\d+)*(?:[ \t]*(?:%|ms|sec|seconds?|bytes?|KB|MB|GB|TB|Hz|kHz|MHz|GHz|mm|cm|km|m|kg|g|USD|EUR|BRL))?\b"
    r"|\b(?:não|nunca|jamais|sem|deve|devem|precisa|precisam|obrigatório|proibido"
    r"|not|never|no|without|must|shall|required|should|cannot|can't|don't|won't"
    r"|rejected|refused|reverted|superseded|failed|pending|blocked|warning|security"
    r"|rejeté|refusé|annulé|échoué|attente|bloqué"
    r"|abgelehnt|verweigert|rückgängig|fehlgeschlagen|ausstehend|blockiert"
    r"|rechazado|rechazada|revertido|falló|pendiente|bloqueado"
    r"|отклонил|отклонено|отказался|отменён|неудачная|ожидается|заблокирован"
    r"|رفض|فشل|معلق|محظور"
    r"|rejeitado|recusado|revertido|substituído|falhou|pendente|bloqueado|aviso|segurança)\b"
    r"|拒否|失敗|保留|禁止|拒绝|失败|待处理"
    r"|(?<![A-Za-z0-9_])(?-i:[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+|[a-z]+(?:[A-Z][A-Za-z0-9]*)+|[a-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)(?![A-Za-z0-9_])",
    re.IGNORECASE | re.UNICODE | re.MULTILINE,
)

_FILLERS = (
    "please make sure that",
    "please make sure",
    "please note that",
    "it is important to note that",
    "it should be noted that",
    "as a matter of fact",
    "in order to",
    "basically",
    "actually",
    "por favor certifique-se de que",
    "por favor certifique-se",
    "é importante observar que",
    "é importante notar que",
    "vale a pena lembrar que",
    "de modo geral",
    "basicamente",
)
_FILLER_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(item) for item in sorted(_FILLERS, key=len, reverse=True)) + r")(?!\w)",
    re.IGNORECASE | re.UNICODE,
)
_UNIT_RE = re.compile(
    r".+?(?:[!?]+|\.(?=\s+(?:[A-ZÀ-Ý]|$)|$)|(?=\n|$))",
    re.UNICODE,
)


@dataclass(frozen=True, slots=True)
class ProtectedSpan:
    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class RuleCompressionResult:
    text: str
    input_token_count: int
    output_token_count: int
    protected_spans: tuple[ProtectedSpan, ...]


def _token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def _protected_spans(text: str) -> tuple[ProtectedSpan, ...]:
    """Return ordered, non-overlapping protected source ranges."""
    spans: list[ProtectedSpan] = []
    for match in _PROTECTED_RE.finditer(text):
        if spans and match.start() < spans[-1].end:
            continue
        spans.append(ProtectedSpan(match.group(), match.start(), match.end()))
    return tuple(spans)


def _overlaps(start: int, end: int, spans: tuple[ProtectedSpan, ...]) -> bool:
    return any(start < span.end and end > span.start for span in spans)


def _duplicate_ranges(text: str) -> list[tuple[int, int]]:
    seen: set[str] = set()
    removed: list[tuple[int, int]] = []
    for match in _UNIT_RE.finditer(text):
        unit = match.group()
        if unit.isspace():
            continue
        key = " ".join(re.findall(r"\w+", unit.casefold(), re.UNICODE))
        if not key:
            continue
        if key in seen:
            removed.append(match.span())
        else:
            seen.add(key)
    return removed


def _cleanup(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,;:!?]|\.(?![A-Za-z_]))", r"\1", text)
    text = re.sub(r"([,;])(?:\s*[,;])+", r"\1", text)
    text = re.sub(r"(^|\n)\s*[,;:]\s*", r"\1", text)
    return text.strip()


def compress_rules(text: str) -> RuleCompressionResult:
    """Remove known filler and exact duplicate units without generating words."""
    protected = _protected_spans(text)
    removals = _duplicate_ranges(text)
    removals.extend(
        match.span()
        for match in _FILLER_RE.finditer(text)
        if not _overlaps(match.start(), match.end(), protected)
    )

    keep = [True] * len(text)
    for start, end in removals:
        for index in range(start, end):
            keep[index] = False
    compacted = _cleanup("".join(char for index, char in enumerate(text) if keep[index]))
    return RuleCompressionResult(
        text=compacted,
        input_token_count=_token_count(text),
        output_token_count=_token_count(compacted),
        protected_spans=protected,
    )
