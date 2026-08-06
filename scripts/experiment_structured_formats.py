#!/usr/bin/env python3
"""Measure when compact graph-like rewrites are safe and token-efficient."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import sentencepiece as spm


MODEL = Path("models/context-sift/tokenizer.model")


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    family: str
    source: str
    candidate: str
    anchors: tuple[str, ...] = ()
    required_edges: tuple[str, ...] = ()
    candidate_edges: tuple[str, ...] = ()
    required_modes: tuple[str, ...] = ()
    candidate_modes: tuple[str, ...] = ()


CASES = (
    Case("en_pipeline", "sequence", "Write the code, then test it, review it, and finally deploy it.",
         "write -> test -> review -> deploy", required_edges=("write>test", "test>review", "review>deploy"),
         candidate_edges=("write>test", "test>review", "review>deploy")),
    Case("pt_pipeline", "sequence", "Primeiro faça o bolo. Depois venda, receba o dinheiro e repita o processo.",
         "fazer bolo -> vender -> receber dinheiro -> repetir(fazer bolo)",
         required_edges=("bolo>vender", "vender>dinheiro", "dinheiro>repeat"),
         candidate_edges=("bolo>vender", "vender>dinheiro", "dinheiro>repeat")),
    Case("en_condition", "branch", "After authentication, show the dashboard if valid; otherwise show an error.",
         "auth -> dashboard if valid | error if invalid", required_edges=("auth>dashboard:valid", "auth>error:invalid"),
         candidate_edges=("auth>dashboard:valid", "auth>error:invalid"), required_modes=("if",), candidate_modes=("if",)),
    Case("en_dependency", "dependency", "Deployment requires tests, and tests require a successful build.",
         "deploy requires tests; tests require build", required_edges=("deploy:requires:tests", "tests:requires:build"),
         candidate_edges=("deploy:requires:tests", "tests:requires:build")),
    Case("en_dependency_wrong", "unsafe_dependency", "Deployment requires tests, and tests require a successful build.",
         "build -> tests -> deploy", required_edges=("deploy:requires:tests", "tests:requires:build"),
         candidate_edges=("build>tests", "tests>deploy")),
    Case("en_parallel_wrong", "unsafe_parallel", "Run lint and tests in parallel, then deploy after both finish.",
         "lint -> tests -> deploy", required_edges=("lint||tests", "both>deploy"),
         candidate_edges=("lint>tests", "tests>deploy"), required_modes=("both", "parallel")),
    Case("en_effect_words", "effect", "Better code increases quality and reduces bugs.",
         "code => quality up; bugs down", required_edges=("code>quality:up", "code>bugs:down"),
         candidate_edges=("code>quality:up", "code>bugs:down")),
    Case("en_effect_symbols", "effect_symbols", "Better code increases quality and reduces bugs.",
         "code => quality↑; bugs↓", required_edges=("code>quality:up", "code>bugs:down"),
         candidate_edges=("code>quality:up", "code>bugs:down")),
    Case("es_rules", "rules", "Debes conservar todas las rutas. Nunca expongas claves API. Puedes borrar comentarios repetidos.",
         "MUST conservar rutas; NEVER exponer claves API; MAY borrar comentarios repetidos",
         anchors=("rutas", "claves API"), required_modes=("MUST", "NEVER", "MAY"), candidate_modes=("MUST", "NEVER", "MAY")),
    Case("fr_state", "state", "Le service utilise le port 8080, son état est actif et le délai maximal est de 30 secondes.",
         "service: port=8080; état=actif; délai_max=30 secondes", anchors=("8080", "actif", "30 secondes")),
    Case("de_code", "code_description", "Die Datei src/auth/token.py wurde geändert. refresh_token prüft reuse_detected vor dem Speichern.",
         "src/auth/token.py: refresh_token -> prüft reuse_detected -> speichern",
         anchors=("src/auth/token.py", "refresh_token", "reuse_detected"), required_edges=("check>save",), candidate_edges=("check>save",)),
    Case("en_agent_flow", "large_graph",
         "The agent reads AGENTS.md before editing. It inspects the repository, changes only the required files, runs unit tests, and reports the result. Deployment is allowed only when every test passes.",
         "agent: read AGENTS.md -> inspect repo -> edit required files -> run tests -> report; deploy only if all tests pass",
         anchors=("AGENTS.md",),
         required_edges=("read>inspect", "inspect>edit", "edit>test", "test>report", "deploy:if:all-pass"),
         candidate_edges=("read>inspect", "inspect>edit", "edit>test", "test>report", "deploy:if:all-pass"),
         required_modes=("only if",), candidate_modes=("only if",)),
    Case("pt_records", "repeated_records",
         "O serviço auth usa a porta 8080 e está ativo. O serviço billing usa a porta 8081 e está pausado. O serviço search usa a porta 8082 e está ativo.",
         "serviço|porta|estado; auth|8080|ativo; billing|8081|pausado; search|8082|ativo",
         anchors=("auth", "8080", "billing", "8081", "search", "8082")),
    Case("pt_long_agent", "long_sequence",
         "Antes de alterar o serviço, leia AGENTS.md e verifique o estado com git status --short. Abra src/auth/token.py, localize refresh_token e confirme reuse_detected antes da gravação. Edite somente arquivos necessários. Execute PYTHONPATH=src python3 -m unittest discover -s tests -v. Se falhar, não faça deploy: corrija a causa e repita. Se passar, informe arquivos, causa e testes. Nunca exponha OPENROUTER_API_KEY nem crie commit sem autorização explícita.",
         "FLUXO: ler AGENTS.md -> git status --short -> abrir src/auth/token.py:refresh_token -> confirmar reuse_detected antes de gravar -> editar somente arquivos necessários -> executar `PYTHONPATH=src python3 -m unittest discover -s tests -v` -> se falhar: corrigir causa -> repetir; se passar: informar arquivos+causa+testes. REGRAS: NUNCA expor OPENROUTER_API_KEY; NUNCA deploy se falhar; NUNCA commit sem autorização explícita.",
         anchors=("AGENTS.md", "git status --short", "src/auth/token.py", "refresh_token", "reuse_detected", "PYTHONPATH=src python3 -m unittest discover -s tests -v", "OPENROUTER_API_KEY"),
         required_edges=("read>status", "status>inspect", "inspect>check", "check>edit", "edit>test", "fail>fix", "fix>rerun", "pass>report"),
         candidate_edges=("read>status", "status>inspect", "inspect>check", "check>edit", "edit>test", "fail>fix", "fix>rerun", "pass>report"),
         required_modes=("somente arquivos necessários", "NUNCA deploy se falhar", "NUNCA commit sem autorização explícita")),
    Case("es_long_branch", "long_branch",
         "Primero conserva rutas, comandos, números e identificadores exactos. Si el usuario pide una explicación, inspecciona los archivos y responde sin modificar nada. Si pide corregir, reproduce el fallo, cambia únicamente el código necesario y ejecuta las pruebas. Si fallan, identifica la causa raíz, corrige y repite. Solo con todas las pruebas aprobadas puede recomendar despliegue; nunca despliegues sin permiso explícito. Está prohibido mostrar secretos, borrar datos o ejecutar git reset --hard. Informa resultado, archivos, pruebas y riesgos.",
         "REGLAS: conservar rutas/comandos/números/IDs exactos; NUNCA mostrar secretos, borrar datos ni ejecutar `git reset --hard`; despliegue requiere permiso explícito + pruebas aprobadas. RAMA: explicación -> inspeccionar -> responder sin modificar | corrección -> reproducir -> cambiar solo código necesario -> probar -> si falla: causa raíz -> corregir -> repetir -> si aprueba: recomendar despliegue. INFORME: resultado; archivos; pruebas; riesgos.",
         anchors=("git reset --hard",),
         required_edges=("explain>inspect", "inspect>answer", "fix>reproduce", "reproduce>edit", "edit>test", "fail>cause", "cause>fix", "fix>rerun", "pass>recommend"),
         candidate_edges=("explain>inspect", "inspect>answer", "fix>reproduce", "reproduce>edit", "edit>test", "fail>cause", "cause>fix", "fix>rerun", "pass>recommend"),
         required_modes=("sin modificar", "solo código necesario", "permiso explícito", "NUNCA")),
    Case("de_long_dependencies", "long_dependency",
         "Der Build startet erst nach Installation der Abhängigkeiten und validierter Konfiguration. Integrationstests benötigen erfolgreichen Build und Testdatenbank. Sicherheitsscan benötigt den Build und läuft parallel. Signieren ist nur nach erfolgreichen Tests und Scan erlaubt. Veröffentlichung von version 1.4.2 braucht Signatur und ausdrückliche Freigabe. Bei Fehler bleibt version 1.4.1 aktiv. API_TOKEN darf nie in Logs erscheinen; artifacts/context-sift-1.4.2.whl bleibt unverändert.",
         "ABHÄNGIG: Abhängigkeiten installieren + Konfiguration validieren -> Build; Build -> Integrationstests mit Testdatenbank || Sicherheitsscan; Tests erfolgreich + Scan erfolgreich -> `artifacts/context-sift-1.4.2.whl` signieren; Signatur + ausdrückliche Freigabe -> version 1.4.2 veröffentlichen. FEHLER => version 1.4.1 bleibt aktiv. NIE API_TOKEN loggen; Artefakt unverändert.",
         anchors=("version 1.4.2", "version 1.4.1", "API_TOKEN", "artifacts/context-sift-1.4.2.whl"),
         required_edges=("install+config>build", "build>integration", "db>integration", "build>security", "integration||security", "pass>sign", "sign+approval>publish", "failure>keep"),
         candidate_edges=("install+config>build", "build>integration", "db>integration", "build>security", "integration||security", "pass>sign", "sign+approval>publish", "failure>keep"),
         required_modes=("||", "ausdrückliche Freigabe", "NIE API_TOKEN", "unverändert")),
    Case("fr_long_table", "long_state_table",
         "Le service auth écoute sur le port 8080, utilise deux réplicas, a un délai de 30 secondes, reste actif et dépend de redis-auth:6379. Le service billing écoute sur 8081, utilise un réplica, a un délai de 45 secondes, reste suspendu et dépend de postgres-billing:5432. Le service search écoute sur 8082, utilise quatre réplicas, a un délai de 20 secondes, reste actif et dépend de opensearch:9200. Aucune valeur ne peut changer.",
         "service|port|réplicas|délai|état|dépendance; auth|8080|2|30s|actif|redis-auth:6379; billing|8081|1|45s|suspendu|postgres-billing:5432; search|8082|4|20s|actif|opensearch:9200. RÈGLE: aucune valeur changée.",
         anchors=("auth", "8080", "2", "30", "redis-auth:6379", "billing", "8081", "1", "45", "postgres-billing:5432", "search", "8082", "4", "20", "opensearch:9200"),
         required_edges=("auth>redis", "billing>postgres", "search>opensearch"), candidate_edges=("auth>redis", "billing>postgres", "search>opensearch"),
         required_modes=("aucune valeur changée",)),
    Case("en_long_handoff", "long_handoff",
         "Previous agent investigated intermittent refresh-token reuse rejection. User goal remains: fix the defect and add regression coverage. src/auth/token.py writes the rotated token before checking reuse_detected, causing concurrent inconsistent state. Production files remain unchanged. Uncommitted tests/test_refresh_token.py::test_concurrent_reuse_is_rejected_once fails. pytest tests/test_refresh_token.py -q reported 1 failed and 12 passed. Next agent must inspect the diff, move the check before persistence, preserve the public API, run focused test, then full suite. Never delete the regression test, change schema, expose REFRESH_TOKEN_SECRET, commit, or deploy without explicit approval.",
         "GOAL: fix intermittent refresh-token reuse rejection; add regression coverage. CAUSE: src/auth/token.py persists before reuse_detected check => concurrent inconsistent state. STATE: production unchanged; uncommitted tests/test_refresh_token.py::test_concurrent_reuse_is_rejected_once fails; `pytest tests/test_refresh_token.py -q` => 1 failed, 12 passed. NEXT: inspect diff -> check before persistence -> preserve public API -> focused test -> full suite. NEVER delete regression test/change schema/expose REFRESH_TOKEN_SECRET/commit/deploy without explicit approval.",
         anchors=("src/auth/token.py", "reuse_detected", "tests/test_refresh_token.py", "test_concurrent_reuse_is_rejected_once", "pytest tests/test_refresh_token.py -q", "1 failed", "12 passed", "REFRESH_TOKEN_SECRET"),
         required_edges=("persist-before-check>inconsistent", "inspect>move", "move>focused", "focused>full"), candidate_edges=("persist-before-check>inconsistent", "inspect>move", "move>focused", "focused>full"),
         required_modes=("production unchanged", "preserve public API", "NEVER", "explicit approval")),
    Case("en_exact_command", "unsafe_exact", "Only after approval, run git checkout -- src/config.py.",
         "approval -> git checkout src/config.py", anchors=("git checkout -- src/config.py",),
         required_modes=("only after",), candidate_modes=()),
)


def main() -> None:
    tokenizer = spm.SentencePieceProcessor(model_file=str(MODEL))
    rows = []
    for case in CASES:
        source_tokens = len(tokenizer.encode(case.source))
        candidate_tokens = len(tokenizer.encode(case.candidate))
        saved = source_tokens - candidate_tokens
        reduction = saved / source_tokens
        invalid_source_anchors = [value for value in case.anchors if value not in case.source]
        missing_anchors = [value for value in case.anchors if value not in case.candidate]
        missing_edges = sorted(set(case.required_edges) - set(case.candidate_edges))
        invented_edges = sorted(set(case.candidate_edges) - set(case.required_edges))
        candidate_folded = case.candidate.casefold()
        missing_modes = [mode for mode in case.required_modes if mode.casefold() not in candidate_folded]
        oracle_consistent = not (
            invalid_source_anchors or missing_anchors or missing_edges or invented_edges or missing_modes
        )
        efficient = saved >= 8 and reduction >= 0.15
        rows.append({
            **asdict(case), "source_tokens": source_tokens, "candidate_tokens": candidate_tokens,
            "saved_tokens": saved, "reduction": round(reduction, 4),
            "invalid_source_anchors": invalid_source_anchors, "missing_anchors": missing_anchors,
            "missing_edges": missing_edges,
            "invented_edges": invented_edges, "missing_modes": missing_modes,
            "oracle_consistent": oracle_consistent, "efficient": efficient,
            "use": oracle_consistent and efficient,
        })

    consistent = [row for row in rows if row["oracle_consistent"]]
    summary = {
        "cases": len(rows),
        "oracle_consistent_cases": len(consistent),
        "usable_cases": sum(row["use"] for row in rows),
        "mean_consistent_reduction": round(
            sum(row["reduction"] for row in consistent) / len(consistent), 4
        ),
        "gate": "oracle_consistent and saved_tokens >= 8 and reduction >= 0.15",
    }
    assert not next(row for row in rows if row["id"] == "en_parallel_wrong")["oracle_consistent"]
    assert not next(row for row in rows if row["id"] == "en_exact_command")["oracle_consistent"]
    print(json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
