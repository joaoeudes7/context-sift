"""Deterministic, extractive text compaction."""

from __future__ import annotations

from dataclasses import dataclass
import re


_TOKEN_RE = re.compile(r"\w+(?:[-']\w+)*|https?://\S+|`[^`]*`", re.UNICODE)
#: Meaning-changing anchors: negations, permissions, prohibitions, decision state
#: and condition/sequence gates. Kept in their own pattern so they can expand to a
#: whole sentence (a negation without its object is meaningless) without dragging
#: every technical anchor (paths, numbers, URLs) into an expensive sentence span.
_SEMANTIC_RISK_PATTERN = (
    r"\b(?:não|nunca|jamais|sem|deve|devem|precisa|precisam|obrigatório|proibido"
    r"|not|never|no|without|must|shall|required|require(?:s|d|ments?)?|should|cannot|can't|don't|won't"
    r"|request|question|allowed|permit|permitted|forbidden|override"
    r"|solicitud|pregunta|nunca|sin|debe|permitido|permitida|prohibido|prohibida|prevalecen"
    r"|demande|question|jamais|sans|doit|autorisé|autorisée|interdit|interdite|priment"
    r"|auftrag|frage|niemals|nicht|ohne|muss|erlaubt|verboten|vorrang"
    r"|запрос|вопрос|никогда|не|без|должен|разрешена|разрешено|запрещено|безопасности|важнее"
    r"|طلب|سؤال|لا|دون|يجب|يسمح|ممنوع|أمان|تتقدم"
    r"|rejected|refused|reverted|superseded|failed|pending|blocked|warning|security"
    r"|rejet\w*|refus\w*|annul\w*|échou\w*|attente|bloqu\w*"
    r"|abgelehnt|verweigert|rückgängig|fehlgeschlagen|ausstehend|blockiert"
    r"|rechaz\w*|revert\w*|revirt\w*|fall(?:ó|o|a|aron|id[oa])|pendiente|bloqu\w*"
    r"|отклон\w*|отказ\w*|отмен\w*|неудач\w*|ожида\w*|заблокирован\w*"
    r"|رفض|فشل|معلق|محظور"
    r"|rejeit\w*|recus\w*|revert\w*|substitu\w*|falh\w*|pendente|bloqu\w*|aviso|segurança)\b"
    r"|\b(?:only after|only when|only if|not before|unless|provided that|prior approval|approval is required"
    r"|requires? (?:user |explicit )?approval|pending (?:user |explicit )?approval|awaiting approval"
    r"|requer(?:em|id[oa]s?)?|exige(?:m)?|requiere(?:n)?|requiert|erfordert|benötigt|требует(?:ся)?|нужно|يتطلب|必要|要する|需要"
    r"|somente após|só após|apenas depois|a menos que|desde que|aprovação prévia|requer aprovação|aguardando aprovação"
    r"|solo después|solo tras|a menos que|siempre que|aprobación previa|requiere aprobación"
    r"|seulement après|à moins que|à condition que|sous réserve|approbation préalable"
    r"|erst nach|erst wenn|es sei denn|sofern|vorherige (?:genehmigung|zustimmung)"
    r"|только после|только тогда|если не|при условии|предварительн\w+ (?:одобрен|согласован)"
    r"|فقط بعد|فقط عندما|إلا إذا|بشرط|موافقة مسبقة"
    r"|承認後|承認を得|の場合のみ|仅当|除非|经批准|前提是)(?:\b|(?=[^\x00-\x7F]))"
    r"|拒否|失敗|保留|禁止|拒绝|失败|待处理|してはいけない|ならない|許可|優先|依頼|質問"
    r"|不得|不要|必须|允许|安全|优先|请求|问题|[?？]"
    r"|\b(?:tool inventory|available tools|available capabilities|inventário de ferramentas|ferramentas disponíveis)\b"
)

_PROTECTED_RE = re.compile(
    r"```[\s\S]*?```"
    r"|->|→"
    r"|(?:^|(?<=\n)|(?<=[.!?。！？]\s))[ \t]*(?:[-*][ \t]+)?(?:you should|you must|you need to|can you|could you|would you|do you"
    r"|do|run|search|find|identify|locate|determine|check|verify|analyze|compare|list|explain"
    r"|create|update|fix|add|remove|keep|use|execute"
    r"|faça|faca|execute|atualize|corrija|adicione|remova|mantenha|use"
    r"|haz|ejecuta|actualiza|corrige|añade|elimina|mantén|usa"
    r"|faites|exécutez|mettez|corrigez|ajoutez|supprimez|gardez|utilisez"
    r"|mach|führe|aktualisiere|korrigiere|füge|entferne|behalte|verwende"
    r"|сделай|выполни|обнови|исправь|добавь|удали|сохрани|используй"
    r"|افعل|نفذ|حدّث|أصلح|أضف|احذف|احتفظ|استخدم|実行|更新|修正|追加|削除|保持|使用|请|执行|更新|修复|添加|删除|保留|使用)(?:\b|(?=[^\x00-\x7F]))"
    r"|(?:^|(?<=[.!?。！？]\s))[^.!?。！？:：\n]{1,48}[:：]"
    r"|`[^`\n]+`"
    r"|https?://[^\s<>]+"
    r"|\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"
    r"|(?<![A-Za-z0-9_.-])\.[A-Za-z_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_-])?(?=$|[\s,;:!?)}\]]|\.(?=\s|$)|[^\x00-\x7F])"
    r"|(?<![A-Za-z0-9_.-])/?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]*[A-Za-z0-9_-](?:::[A-Za-z0-9_.-]*[A-Za-z0-9_-])*(?=$|[\s,;:!?)}\]]|\.(?=\s|$)|[^\x00-\x7F])"
    r"|\b\d+(?:[./,:-]\d+)*(?:[ \t]*(?:%|ms|sec|seconds?|bytes?|KB|MB|GB|TB|Hz|kHz|MHz|GHz|mm|cm|km|m|kg|g|USD|EUR|BRL))?\b"
    r"|" + _SEMANTIC_RISK_PATTERN +
    r"|(?<![A-Za-z0-9_])(?-i:[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+|[a-z]+(?:[A-Z][A-Za-z0-9]*)+|[a-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)(?![A-Za-z0-9_])",
    re.IGNORECASE | re.UNICODE | re.MULTILINE,
)
_SEMANTIC_RISK_RE = re.compile(_SEMANTIC_RISK_PATTERN, re.IGNORECASE | re.UNICODE | re.MULTILINE)

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
_FILLER_PATTERN = sorted(_FILLERS, key=len, reverse=True)
_FILLER_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(item) for item in _FILLER_PATTERN) + r")(?!\w)",
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


_SENTENCE_STOPS = "!?。！？\n"


def _is_sentence_boundary(text: str, index: int) -> bool:
    """True when ``text[index]`` terminates a sentence.

    A plain ``.`` only ends a sentence when whitespace follows it, so decimals
    (``3.12``), filenames (``capture.py``) and qualified symbols do not split a
    protected sentence in half.
    """
    char = text[index]
    if char in _SENTENCE_STOPS:
        return True
    if char == ".":
        following = text[index + 1] if index + 1 < len(text) else " "
        return following in " \t\n"
    return False


def protected_spans(text: str) -> tuple[ProtectedSpan, ...]:
    """Exact protected anchors (no sentence expansion), ordered and non-overlapping."""
    return _protected_spans(text)


def _expand_to_sentence(text: str, start: int, end: int) -> tuple[int, int]:
    while start > 0 and not _is_sentence_boundary(text, start - 1):
        start -= 1
    while start < end and text[start].isspace():
        start += 1
    while end < len(text) and not _is_sentence_boundary(text, end):
        end += 1
    if end < len(text):
        end += 1
    return start, end


def semantic_risk_spans(
    text: str, *, max_chars: int = 128
) -> tuple[ProtectedSpan, ...]:
    """Expand meaning-changing anchors to their sentence, capped at ``max_chars``.

    A negation, prohibition or condition gate is only interpretable with the
    clause it governs, so its whole sentence is protected. The cap keeps a long
    prose sentence from being dragged in whole: past the cap the anchor falls
    back to its minimal span (still protected upstream), so dense prose keeps
    its compression ratio.
    """
    expanded: list[ProtectedSpan] = []
    for match in _SEMANTIC_RISK_RE.finditer(text):
        if expanded and match.start() < expanded[-1].end:
            continue
        start, end = _expand_to_sentence(text, match.start(), match.end())
        if max_chars and end - start > max_chars:
            continue
        if expanded and start <= expanded[-1].end:
            previous = expanded[-1]
            expanded[-1] = ProtectedSpan(text[previous.start:end], previous.start, end)
        else:
            expanded.append(ProtectedSpan(text[start:end], start, end))
    return tuple(expanded)


def protected_sentence_spans(text: str) -> tuple[ProtectedSpan, ...]:
    """Expand every protected anchor so models cannot delete part of its sentence."""
    expanded: list[ProtectedSpan] = []
    for span in _protected_spans(text):
        start, end = _expand_to_sentence(text, span.start, span.end)
        if expanded and start <= expanded[-1].end:
            previous = expanded[-1]
            expanded[-1] = ProtectedSpan(text[previous.start:end], previous.start, end)
        else:
            expanded.append(ProtectedSpan(text[start:end], start, end))
    return tuple(expanded)


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


_INDENT_RE = re.compile(r"^[ \t]*")


def _cleanup(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        indent = _INDENT_RE.match(line).group()
        body = line[len(indent):]
        if not body:
            lines.append("")  # whitespace-only line: drop the padding, keep the break
            continue
        lines.append(indent + re.sub(r"[ \t]+", " ", body).rstrip())
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+([,;:!?]|\.(?![A-Za-z_]))", r"\1", text)
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
