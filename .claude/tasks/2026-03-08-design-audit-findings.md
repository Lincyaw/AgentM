# Design Audit Findings

**Created**: 2026-03-08
**Status**: PENDING — to be discussed and resolved before implementation

---

## Scope

Systematic audit of all 5 design documents + index.yaml for inconsistencies, anti-patterns, and implementation risks.

**Documents audited**:
- system-design-overview.md
- orchestrator.md
- sub-agent.md
- generic-state-wrapper.md
- frontend-architecture.md
- index.yaml

---

## CRITICAL Issues

### ~~C1. Send() parallel model contradiction~~ → RESOLVED

**Resolution**: Replaced with Async TaskManager model. Orchestrator is `create_react_agent`, Sub-Agents are independently compiled subgraphs invoked asynchronously via TaskManager. See orchestrator.md for updated design.

### C2. `ExecutorState` definition conflict: `dataclass` vs `TypedDict`

- **orchestrator.md** (line 128): `ExecutorState` defined as `@dataclass`
- **system-design-overview.md** (line 107-113): `ExecutorState` defined as `TypedDict`
- **sub-agent.md** (line 24): Referenced as `TypedDict`
- **generic-state-wrapper.md** (line 44-53): `BaseExecutorState` defined as `TypedDict`

**Risk**: LangGraph's `StateGraph` requires state to be `TypedDict` (or `BaseModel`). `@dataclass` cannot be used directly as state schema — will cause compilation failure.

**Fix**: Unify to `TypedDict`. `DiagnosticNotebook` internally can be `@dataclass`, but `ExecutorState` itself must be `TypedDict`.

### C3. `Phase` name collision between orchestrator.md and generic-state-wrapper.md

- **orchestrator.md**: Defines `Phase` as an Enum (`EXPLORATION`, `GENERATION`, `VERIFICATION`, `CONFIRMATION`)
- **generic-state-wrapper.md** (line 108-110): Defines `Phase` as a dataclass (`name: str, description: str, handler: Callable`) for PhaseManager

**Risk**: Two different concepts using the same name. Naming collision inevitable at implementation.

**Fix**: Rename PhaseManager's phase definition to `PhaseDefinition` or `PhaseConfig`.

### C4. `CompressionRef` storage location undecided

- **orchestrator.md** initially says stored in checkpoint metadata, then has a verification warning that LangGraph doesn't support nodes writing custom metadata
- Recommended Option 1 (store in state), but other parts of the document still reference "checkpoint metadata"
- `drill_down_compressed_range` function uses `compression_ref` parameter without explaining how to get it from state

**Fix**: Unify to Option 1. Add `compression_refs: list[CompressionRef]` field to `ExecutorState` / `SubAgentState`. Update all references.

---

## HIGH Issues

### H1. `VerificationResult` example still has string literal

orchestrator.md (line ~802):
```python
verdict="confirmed"  # Should be Verdict.CONFIRMED
```

All status values should use Enum as previously agreed.

### H2. `current_phase` field type inconsistency

- `ExecutorState.current_phase` in orchestrator.md is `Phase` Enum type
- But phase assignment in multiple return dicts uses strings: `"current_phase": "generation"`, etc.
- `BaseExecutorState.current_phase` in generic-state-wrapper.md is `str` type

**Fix**: Unify. If `BaseExecutorState` keeps `str` (different system types have different phases), then `HypothesisDrivenState` should override to `Phase` type. All return dicts should use `Phase.GENERATION` etc.

### H3. `index.yaml` description for `generic_state_wrapper` is outdated

```yaml
description: "...Shared BaseOrchestrator, PhaseManager, StateSchemaFactory"
```

`StateSchemaFactory` was changed to `State Schema Registry` (`STATE_SCHEMAS` dict), but index.yaml not updated.

### H4. Sub-Agent `create_react_agent` deprecation warning may be inaccurate

- **sub-agent.md** (line 45): Says `create_react_agent` is deprecated, moved to `langchain.agents` as `create_agent`
- But immediately after (line 48): Code still uses `from langgraph.prebuilt import create_react_agent`
- As of May 2025, `create_react_agent` is still the primary API in `langgraph.prebuilt`

**Fix**: Remove deprecation warning or mark as "verify at implementation time".

### H5. `ExecutorState` defined twice in orchestrator.md with conflicts

- Line 128-138: As `@dataclass` with complete fields
- Line 1022-1026: As `TypedDict` in "Results Aggregation" section, adding `agent_results: Annotated[list[dict], operator.add]`

Two definitions with different types and fields.

### H6. `dispatch_task` tool undefined

Scenario config and multiple examples reference `dispatch_task` tool, but no `@tool` signature is defined. Only `check_agents`, `inject_instruction`, `abort_agent` have definitions.

**Note**: With the new Async TaskManager model, this is now `dispatch_agent` tool.

---

## MEDIUM Issues

### M1. Sub-Agent return format inconsistent between phases

- Phase 1 (Exploration): Returns raw JSON data (`{"cpu": 0.85, ...}`)
- Phase 3 (Verification): Returns structured `VerificationResult` (three-block structure)

No mechanism specified for how the Sub-Agent knows which mode to use.

**Fix**: Clarify that task instruction or `depth` parameter determines output format.

### M2. `REMOVE_ALL_MESSAGES` may not exist

orchestrator.md (line ~833):
```python
"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
```
Annotated as "✅ LangGraph Verified" but needs verification at implementation time.

### M3. Frontend WebSocket supports only single connection

FastAPI setup registers only one websocket. Multiple browser windows would overwrite each other.

**Fix**: Change to `Set[WebSocket]` for multiple connections, broadcast to all.

### M4. Memory Extraction `phases` config format doesn't match scenario.yaml structure

generic-state-wrapper.md has a standalone `phases` top-level key, but system-design-overview.md's scenario config doesn't have this key. RCA scenario controls phases via `orchestrator.feature_gates`.

**Fix**: Unify phase declaration approach.

### M5. `trajectory_system` still has no design document

index.yaml: `trajectory_system.design: null`. Trajectory concepts are scattered across multiple documents but no dedicated design covering:
- Complete trajectory data format
- RL transition reward function design
- Hierarchical trace serialization/deserialization
- Exact mapping between trajectory and checkpoints

---

## LOW Issues

### L1. Notebook mutation conflicts with immutability principle

Code examples heavily mutate notebook objects directly. While these are "illustrative" code, they conflict with coding-style rules (always create new objects, never mutate).

**Fix**: At implementation time, use immutable patterns (`dataclasses.replace()`) or explicitly document Notebook as a mutation exception.

### L2. `AgentPool` loads from independent yaml files vs scenario.yaml inline

sub-agent.md's `AgentPool` (line ~310) globs `config_dir/*.yaml`, but actual design defines agents inline in `scenario.yaml`.

### L3. `knowledge_list` uses `store.search()` to enumerate entries — may be inefficient

For structural browsing, using semantic search to list all entries is not optimal. Need to verify `store.search()` behavior without a `query` parameter.
