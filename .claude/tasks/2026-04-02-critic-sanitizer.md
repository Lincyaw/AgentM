# Task: CriticSanitizer — LLM-Based Checks

**Plan**: [investigation-sanitizer](../plans/2026-04-02-investigation-sanitizer.md)
**Phase**: 3 — CriticSanitizer
**Design**: [investigation-sanitizer](../designs/investigation-sanitizer.md) §8, §13 Q1
**Depends on**: [investigation-tracker](2026-04-02-investigation-tracker.md)

## Scope

Implement semantic checks that require LLM judgment. Each check sends a focused prompt to a cheap model and parses structured output.

## Deliverables

### critic_sanitizer.py — `src/agentm/scenarios/rca/sanitizer/critic_sanitizer.py`

**Prompt templates** (one per check):
- C3: causal direction — input: hypothesis + deep_analyze findings → output: `{ proven: bool, missing_evidence: str }`
- J1: signal misread — input: hypothesis evidence + rejection reason → output: `{ misread: bool, reasoning: str }`
- J4: symptom-as-cause — input: root cause service + upstream profiles → output: `{ likely_symptom: bool, suspect_upstream: list[str] }`
- P2: dispatch quality — input: task instruction text → output: `{ quality: str, missing: list[str] }`

**Pydantic response schemas** for each check's structured output.

**CriticSanitizer class**:
- Constructor: `model` (ModelProtocol), configurable severity/disabled codes
- Implements `Sanitizer` protocol
- `check(trigger, ...)` for sync execution (pre_finalize: C3, J1, J4)
- `check_async(trigger, ...)` for async execution (dispatch: P2) — launches `asyncio.Task`
- `collect_async_results() -> list[SanitizerFinding]` — drains completed async tasks
- Internal: builds prompt from stores/tracker data, calls `model.with_structured_output().ainvoke()`, maps response to `SanitizerFinding`

**Key considerations**:
- J1 only runs when a hypothesis was rejected after having ≥3 evidence items (skip otherwise — no misread risk)
- J4 only runs when confirmed hypothesis exists and root cause service has upstream in profile store
- P2 async tasks are fire-and-forget with timeout; if they fail, no finding is produced (silent)

### Tests — `tests/unit/test_critic_sanitizer.py`

- Mock LLM responses for each check
- Verify correct prompt construction (contains expected service names, metrics, etc.)
- Verify response parsing: proven=False → BLOCK finding, proven=True → no finding
- Async P2: verify task is created, collect_async_results returns findings after completion
- Edge cases: no confirmed hypothesis → J1/J4 checks skip silently
- LLM failure: verify graceful handling (no finding, no crash)
