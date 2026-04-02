# Task: SanitizerMiddleware — Integration Layer

**Plan**: [investigation-sanitizer](../plans/2026-04-02-investigation-sanitizer.md)
**Phase**: 4 — Middleware
**Design**: [investigation-sanitizer](../designs/investigation-sanitizer.md) §4, §9, §13
**Depends on**: [code-sanitizer](2026-04-02-code-sanitizer.md), [critic-sanitizer](2026-04-02-critic-sanitizer.md)

## Scope

Wire CodeSanitizer and CriticSanitizer into the orchestrator loop via MiddlewareBase hooks. Handle trigger routing, finding injection, finalize gate, degradation, and trajectory recording.

## Deliverables

### middleware.py — `src/agentm/scenarios/rca/sanitizer/middleware.py`

**SanitizerMiddleware(MiddlewareBase)**:

#### on_tool_call hook
- Intercept tool calls by name, record events in InvestigationTracker:
  - `dispatch_agent` → `dispatch` event (extract agent_id, task_type, target services from args)
  - `update_hypothesis` → `hypothesis_change` event (extract id, status)
  - `remove_hypothesis` → `hypothesis_change` event (status="removed")
  - `update_service_profile` / `query_service_profile` → `tool_call` event
- On `dispatch_agent`: trigger async P2 critic check (pass task instruction)
- Call `call_next` to execute the actual tool (sanitizer is non-blocking for tool execution)
- On tool result from `dispatch_agent`/`check_tasks` containing task completion: record `task_complete` event (extract task_type, verdict if verify)

#### on_llm_end hook
- Parse response content for `<decision>finalize</decision>`
- **If finalize detected**:
  1. Run CodeSanitizer with trigger="pre_finalize"
  2. Run CriticSanitizer with trigger="pre_finalize" (sync)
  3. Apply budget-aware severity degradation (LoopContext check)
  4. Apply retry degradation (3-attempt rule per code+target pair)
  5. Collect BLOCK findings
  6. If BLOCK findings: strip `<decision>finalize</decision>` from response content, queue finalize_blocked message
  7. Record sanitizer event in TrajectoryCollector
- **Every round**: run CodeSanitizer trigger="every_round" (J2 drift only)
- **Every N rounds** (periodic_interval): run CodeSanitizer trigger="periodic"
- **On hypothesis_change** (detected from on_tool_call state): run CodeSanitizer trigger="hypothesis_change"
- Queue all findings for injection in next on_llm_start

#### on_llm_start hook
- Collect completed async critic results via `critic_sanitizer.collect_async_results()`
- Merge with queued findings from on_llm_end
- If any findings: format as XML (design §9) and append as human message
- Clear finding queue

#### State tracking
- `_pending_findings: list[SanitizerFinding]` — accumulated between on_llm_end and on_llm_start
- `_block_attempts: dict[tuple[str, str], int]` — (code, target) → consecutive block count for degradation
- `_hypothesis_changed_this_round: bool` — flag set in on_tool_call, consumed in on_llm_end
- `_periodic_counter: int` — round counter for periodic trigger

#### Finding formatting
- `_format_findings(findings) -> str` — XML format per design §9
- `_format_finalize_blocked(findings) -> str` — blocking message per design §9

#### TrajectoryCollector integration
- Record `"sanitizer"` events after each check cycle (design §13 Q3)

### Tests — `tests/unit/test_sanitizer_middleware.py`

- **Tool interception**: on_tool_call records correct events for each intercepted tool
- **Finalize gate**: on_llm_end with finalize tag + BLOCK finding → tag stripped, findings queued
- **Finalize pass**: on_llm_end with finalize tag + no BLOCK → response unchanged
- **Finding injection**: on_llm_start with queued findings → message appended with XML format
- **Degradation**: same BLOCK finding 3 times → severity becomes WARN on 4th
- **Budget degradation**: coverage BLOCK with exhausted budget → WARN; process BLOCK → stays BLOCK
- **Periodic trigger**: findings produced every N rounds
- **Async critic drain**: mock async results appear in on_llm_start injection
- **Trajectory recording**: verify trajectory.record called with sanitizer event type
