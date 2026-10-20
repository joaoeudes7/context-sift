"""Project a conversation into a checkpoint: what was done, what is next.

The compactor's default mode keeps enough context to *continue* a task. A
checkpoint asks a narrower question: when a long session is folded, which
clauses carry the state a fresh model needs -- completed work, decisions with
their reasons, constraints, and pending/next actions -- and which are narration
that can go. The projection is deterministic and extractive: it selects source
clauses, never rewrites them.
"""

from __future__ import annotations

import re

from context_sift.clause_dataset import Clause
from context_sift.rules import ProtectedSpan


# Completed work: past-tense action verbs across EN/PT.
_DONE_RE = re.compile(
    r"(?<!\w)(?:done|completed|finished|implemented|added|removed|updated|created|fixed"
    r"|resolved|reverted|ran|tested|validated|passed|failed|merged|wrote|patched"
    r"|conclu[íi]|feito|implementei|adicionei|removi|atualizei|criei|corrigi|resolvi|reverti"
    r"|rodei|testei|validei|passei|falhou|mesclei|revisei)(?!\w)",
    re.IGNORECASE | re.UNICODE,
)

# Pending / next steps.
_NEXT_RE = re.compile(
    r"(?<!\w)(?:next|pending|todo|to do|remaining|follow-?up|after that|once .+ (?:pass|green)"
    r"|pr[óo]xim[oa]s?|pendente|a seguir|falta|em seguida|depois disso|por fazer)(?!\w)",
    re.IGNORECASE | re.UNICODE,
)

# Decisions and their reasons.
_DECISION_RE = re.compile(
    r"(?<!\w)(?:decision|decided|chose|choose|instead|rather than|reason|because"
    r"|decidimos|decidi|escolhi|em vez de|porque|raz[ãa]o)(?!\w)",
    re.IGNORECASE | re.UNICODE,
)

# Pure narration / ceremony: dropped unless it also carries an anchor.
_NOISE_RE = re.compile(
    r"(?<!\w)(?:great question|happy to|take a look|have a look|i'll start|let me start|sure"
    r"|of course|certainly|thanks|thank you|appreciate|no problem|ok|okay|continue"
    r"|unrelated|turned out|just to be safe|for completeness|double[- ]?checked"
    r"|com certeza|claro|obrigad[oa]|valeu|sem problemas|tudo bem|pode continuar)(?!\w)",
    re.IGNORECASE | re.UNICODE,
)

# Tool/log filler: omission markers and pure pass rows. Failure rows stay.
_MARKER_RE = re.compile(r"^\.{2,}$")
_PASS_ROW_RE = re.compile(r"^(?:collected \d+ items|.*\bPASSED\b)$", re.IGNORECASE)
_FAIL_RE = re.compile(r"(?<!\w)(?:fail(?:ed|ure)?|error|exception|fatal)(?!\w)", re.IGNORECASE)

# Constraints and limits: permission, exclusivity and numeric caps. A checkpoint
# that drops "only the maintainers can deploy" loses a rule, not narration.
_CONSTRAINT_RE = re.compile(
    r"(?<!\w)(?:only|restrict(?:ed|ion|ing)?|read-?only|at (?:most|least)|no (?:more|fewer|less) than"
    r"|cap(?:ped|s)?|limits?|limited|maximum|minimum|prohibit(?:ed)?|forbid(?:den)?|den(?:y|ied|ies)"
    r"|exclusiv\w+|somente|apenas|restrit\w+|no máximo|no mínimo|proibid\w+|obrigat[óo]ri\w+|exigid\w+)(?!\w)",
    re.IGNORECASE | re.UNICODE,
)

# A bare label with nothing after it ("Here is the test run:", "Resultados:") is
# structure, not status. A label that names status ("Pending:", "Decision:")
# still matches _NEXT/_DECISION and is kept.
_LABEL_RE = re.compile(r"^[^.!?。！？:：\n]{1,48}[:：]$")


def _is_tool_noise(text: str) -> bool:
    stripped = text.strip()
    if _MARKER_RE.match(stripped):
        return True
    if _FAIL_RE.search(stripped):
        return False
    return bool(_PASS_ROW_RE.match(stripped))


def overlaps(clause: Clause, spans: tuple[ProtectedSpan, ...]) -> bool:
    return any(clause.start < span.end and clause.end > span.start for span in spans)


def keep_for_checkpoint(
    clauses: list[Clause], protected: tuple[ProtectedSpan, ...]
) -> list[bool]:
    """Keep completed work, decisions, next steps, failures, constraints and anchors."""
    keep: list[bool] = []
    for clause in clauses:
        text = clause.text
        anchored = overlaps(clause, protected)
        status = bool(
            _NEXT_RE.search(text)
            or _DONE_RE.search(text)
            or _DECISION_RE.search(text)
            or _FAIL_RE.search(text)
            or _CONSTRAINT_RE.search(text)
        )
        if _is_tool_noise(text):
            keep.append(False)
            continue
        # A bare label carries no status by itself; drop it unless it names status.
        if _LABEL_RE.match(text.strip()) and not status:
            keep.append(False)
            continue
        # In a checkpoint narration goes even when a generic anchor (e.g. the
        # word "question") happens to sit inside it — unless it is real status.
        if _NOISE_RE.search(text) and not status:
            keep.append(False)
            continue
        keep.append(status or anchored)
    return keep
