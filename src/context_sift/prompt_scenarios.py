"""Deterministic long-text scenarios for agent-context validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptScenario:
    name: str
    text: str
    critical: tuple[str, ...]


def _long(base: str, repeated: str, copies: int = 80) -> str:
    return base + "\n\n" + "\n".join(repeated for _ in range(copies))


def scenarios() -> list[PromptScenario]:
    """Representative pure-text contexts; Kilo-like case is synthetic, not copied."""
    kilo = _long(
        """You are a coding agent operating in plan mode. Analyze before editing. Global instructions prohibit reading .env and credentials.json. Project instructions require Python 3.12, pytest, and surgical changes. Directory instructions under services/payments require idempotency tests. Project rules override global style preferences when they conflict. Read tools are allowed. Editing requires user approval while plan mode is active. Shell commands that delete files are denied. Switch to code mode only after plan approval. Current request: diagnose duplicate payment capture without changing production data. Repository root is /workspace/checkout. Relevant files are services/payments/capture.py and tests/payments/test_capture.py. MCP server payments-docs is read-only. Preserve transaction ID 550e8400-e29b-41d4-a716-446655440000. Never expose API keys.""",
        "Agent reminder: inspect relevant files, follow loaded rules, update plan, avoid speculative edits, and report evidence.",
    )
    handoff = _long(
        """Session handoff from model Aurora to model Boreal. User goal: fix intermittent refresh-token reuse rejection and add regression coverage. Constraint: do not change public API or database schema. Repository is /workspace/identity. Investigation found race in src/auth/rotate_token.py: validation and revocation are separate transactions. Decision: use existing repository method rotate_atomically instead of adding a lock. Reason: method already performs compare-and-swap and works across processes. Completed: reproduced failure with tests/auth/test_rotation.py::test_concurrent_refresh; inspected migration history; confirmed no schema change needed. Failed attempt: process-local asyncio.Lock passed one worker but failed multi-worker test, so it was reverted. No production files have been edited. Pending: patch TokenService.refresh to call rotate_atomically, retain audit event refresh.reuse_detected, then run pytest tests/auth/test_rotation.py -q. User explicitly rejected lowering security checks. Relevant issue: AUTH-417. Expected behavior: exactly one concurrent request succeeds; others return token_reused without invalidating the new token. Do not repeat investigation unless evidence conflicts.""",
        "Earlier discussion repeatedly concluded the race is cross-process. The local lock is superseded. Preserve current decision and next action.",
        copies=95,
    )
    tools = _long(
        """Agent task: prepare read-only diagnosis for deployment failure. Tool inventory includes repository search, file read, CI status, issue lookup, and Kubernetes documentation search. Never invoke write, deploy, restart, merge, or secret-reading operations. Workspace /workspace/billing-api. CI run 84219 failed only in integration-payments. Tool evidence: dependency resolution selected cryptography 44.0.1 while lock file requires 43.0.3. pyproject.toml allows any version below 45; uv.lock pins 43.0.3. Decision needed from user: either regenerate uv.lock or constrain pyproject.toml. No action authorized. Relevant paths: pyproject.toml, uv.lock, .github/workflows/integration.yml. Secret names may be mentioned, values must never be returned. Next response must explain root cause and both options with tradeoffs.""",
        "Repeated tool description: repository search is read-only and accepts query plus path. CI status is read-only and accepts run identifier.",
    )
    return [
        PromptScenario("kilo_like_layered_prompt", kilo, (
            ".env", "pytest", "services/payments/capture.py", "plan mode",
            "Switch to code mode only after plan approval",
            "550e8400-e29b-41d4-a716-446655440000", "Never expose API keys",
        )),
        PromptScenario("cross_model_session_handoff", handoff, (
            "fix intermittent refresh-token reuse rejection", "AUTH-417", "rotate_atomically",
            "asyncio.Lock", "tests/auth/test_rotation.py::test_concurrent_refresh",
            "User explicitly rejected lowering security checks", "token_reused",
        )),
        PromptScenario("tool_heavy_read_only_session", tools, (
            "84219", "cryptography 44.0.1", "43.0.3", "pyproject.toml", "uv.lock", "No action authorized",
            "Tool inventory includes repository search, file read, CI status, issue lookup, and Kubernetes documentation search",
            "Never invoke write, deploy, restart, merge, or secret-reading operations",
        )),
    ]
