# Design Audit Findings

**Created**: 2026-03-08
**Status**: ALL RESOLVED — ready for implementation

---

## Scope

Systematic audit of all 5 design documents + index.yaml for inconsistencies, anti-patterns, and implementation risks.

---

## CRITICAL Issues

### ~~C1. Send() parallel model contradiction~~ → RESOLVED

**Resolution**: Replaced with Async TaskManager model. Orchestrator is `create_react_agent`, Sub-Agents are independently compiled subgraphs invoked asynchronously via TaskManager.

### ~~C2. `ExecutorState` definition conflict: `dataclass` vs `TypedDict`~~ → RESOLVED

**Resolution**: Unified to `TypedDict`. DiagnosticNotebook remains `@dataclass` as a field value. Added `compression_refs: list[CompressionRef]` field.

### ~~C3. `Phase` name collision~~ → RESOLVED

**Resolution**: Renamed PhaseManager's `Phase` dataclass to `PhaseDefinition` in generic-state-wrapper.md. Orchestrator's `Phase` Enum unchanged.

### ~~C4. `CompressionRef` storage location~~ → RESOLVED

**Resolution**: Stored in state (`compression_refs` field in `ExecutorState` and `SubAgentState`). All references to "checkpoint metadata" updated. Verification warning replaced with decision note.

---

## HIGH Issues

### ~~H1. `VerificationResult` string literal~~ → RESOLVED

**Resolution**: VerificationResult completely redesigned (verdict Enum + natural language report). No more string literals.

### ~~H2. `current_phase` field type inconsistency~~ → RESOLVED

**Resolution**: `BaseExecutorState.current_phase` stays `str`. `ExecutorState.current_phase` is `Phase` Enum. All illustrative code uses `Phase.GENERATION` etc.

### ~~H3. `index.yaml` outdated description~~ → RESOLVED

**Resolution**: Updated `StateSchemaFactory` → `State Schema Registry` in generic_state_wrapper description.

### ~~H4. `create_react_agent` deprecation warning~~ → RESOLVED

**Resolution**: Changed to soft note: "verify import path at implementation time". Removed definitive deprecation claim.

### ~~H5. `ExecutorState` defined twice~~ → RESOLVED

**Resolution**: Second definition (Results Aggregation section) was removed when Async Task Dispatch section was rewritten.

### ~~H6. `dispatch_task` tool undefined~~ → RESOLVED

**Resolution**: Replaced with `dispatch_agent` tool with full `@tool` signature in TaskManager section.

---

## MEDIUM Issues

### ~~M1. Sub-Agent return format inconsistent between phases~~ → RESOLVED

**Resolution**: Both Phase 1 and Phase 3 return natural language diagnostic reports. Phase 3 additionally wraps in VerificationResult JSON with structured `verdict` field. Format controlled by prompt guidelines, not rigid schema.

### M2. `REMOVE_ALL_MESSAGES` may not exist → DEFERRED

**Status**: Verify at implementation time. If not available, iterate message IDs.

### ~~M3. Frontend WebSocket single connection~~ → RESOLVED

**Resolution**: TaskManager uses `Set[WebSocket]` with broadcast to all connected clients.

### ~~M4. Memory Extraction `phases` config format~~ → RESOLVED

**Resolution**: Added explicit `phases` top-level key to RCA scenario config in system-design-overview.md. All system types now declare phases consistently.

### ~~M5. `trajectory_system` no design document~~ → RESOLVED

**Resolution**: index.yaml now points to orchestrator.md (Trajectory Registry section). No separate document needed.

---

## LOW Issues — DEFERRED to implementation

### L1. Notebook mutation vs immutability

Illustrative code mutates notebook. Use `dataclasses.replace()` or document as exception at implementation time.

### L2. AgentPool loading approach

sub-agent.md AgentPool now loads from `scenario_config.agents` (updated in previous commit).

### L3. `knowledge_list` efficiency

Verify `store.search()` without query parameter behavior at implementation time.
