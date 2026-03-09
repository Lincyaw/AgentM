# Plan: Context Compression, History Recall & Cross-Session Memory

**Date**: 2026-03-09
**Status**: DRAFT
**Prereq**: [Stub Implementation](2026-03-08-stub-implementation.md) — COMPLETED
**Design refs**: [Orchestrator §Compression](../designs/orchestrator.md), [Generic State Wrapper §Memory Extraction](../designs/generic-state-wrapper.md)

## Overview

Stub implementation is complete — all modules compile and 178 tests pass. Three major feature gaps remain between design and implementation:

1. **Compression + Recall** (intra-task) — Sub-Agent/Orchestrator context compression are pass-through stubs; `recall_history` returns a fixed string
2. **Knowledge Store wiring** (cross-task infrastructure) — Tool functions exist but Store backend is not injected
3. **Memory Extraction Agent** (cross-session) — Entire `memory_extraction` system type is `NotImplementedError`

This plan addresses all three in dependency order across three phases (A → B → C).

---

## Current State Analysis

| Component | Design Doc | Code Status | Gap |
|-----------|-----------|-------------|-----|
| Sub-Agent LLM compression | orchestrator.md §Sub-Agent Layer | `compression.py:53` — pass-through, no LLM call | No summary generation, no `CompressionRef` |
| Orchestrator Notebook compression | orchestrator.md §Orchestrator Layer | `compress_completed_phase()` exists but never called | Not integrated into ReAct loop |
| `recall_history` tool | orchestrator.md §History Recall | `orchestrator.py:158` — returns fixed string | No checkpoint traversal, no LLM retrieval |
| Knowledge Store backend | generic-state-wrapper.md §Knowledge Store | `knowledge.py` tools done, `builder.py:278` passes `store=None` | Store not created or injected |
| Knowledge tools in RCA | scenario.yaml:34 | Commented out | Blocked by Store wiring |
| Memory Extraction system | generic-state-wrapper.md §Memory Extraction | `builder.py:291` — `NotImplementedError` | Zero implementation |
| StateGraph mode | generic-state-wrapper.md §Base Orchestrator | Not implemented | Required by memory_extraction |
| Prompt templates | scenario.yaml references | Not verified | May need creation |

---

## Phase A: Compression + Recall (Intra-Task Memory)

### Rationale

Compression is the foundation — without it, `recall_history` has nothing to recall, and long-running RCA tasks will exhaust the context window. This phase makes the existing pass-through stubs functional.

### Dependency Graph

```
A1: Sub-Agent LLM Compression
  │  (compression.py — add LLM summarization call)
  │
  ├── A2: CompressionRef State Integration
  │     (hooks.py — write CompressionRef to state after compression)
  │     (models/state.py — verify compression_refs field)
  │
  └── A3: recall_history Implementation
        (tools/orchestrator.py — checkpoint traversal + LLM retrieval)
        (builder.py — inject graph/config into recall_history)
        │
        A4: Orchestrator Notebook Compression
           (agents/orchestrator.py — integrate compress_completed_phase)
           (core/prompt.py — compression-aware Notebook formatting)
```

### Task A1: Sub-Agent LLM Compression

**Files**: `src/agentm/core/compression.py`
**Size**: M
**Depends**: —

Current `sub_agent_compression_hook` and `build_compression_hook` detect when token count exceeds threshold but just pass messages through unchanged. Implement actual LLM-based summarization:

1. When threshold exceeded, call compression model to generate a structured summary of older messages
2. Keep last `preserve_latest_n` messages (default 2) uncompressed for continuity
3. Return `{'llm_input_messages': [summary_message] + recent_messages}` — state messages stay intact per LangGraph design
4. Add `CompressionConfig` fields: `preserve_latest_n` (currently missing from schema)

**Key constraint**: Must use `llm_input_messages` return key (not `messages`) to preserve full history in checkpoints. Design doc §Sub-Agent Layer explicitly states this.

**Test**: Add unit test verifying summary generation with mock LLM; existing `test_compression_recall.py` (Layer 2) covers integration.

### Task A2: CompressionRef State Integration

**Files**: `src/agentm/agents/hooks.py`, `src/agentm/models/state.py`
**Size**: S
**Depends**: A1

After compression occurs in A1, write a `CompressionRef` to the agent's state:

1. `hooks.py` `build_compression_hook` — after compression, append `CompressionRef` to `state["compression_refs"]`
2. Verify `SubAgentState` and `HypothesisDrivenState` both have `compression_refs: list[CompressionRef]` field
3. `CompressionRef` needs `from_checkpoint_id` and `to_checkpoint_id` — these come from the checkpointer's current state. The hook runs as `pre_model_hook` which receives state but not config. Investigate: can we get checkpoint IDs from state metadata, or do we need to pass config into the hook?

**Design doc reference**: orchestrator.md §Checkpoint-Based Compression Recovery — `CompressionRef` is stored in state, accessible via `state["compression_refs"]`.

**Risk**: HIGH — `pre_model_hook` receives `state` dict but may not have checkpoint IDs. May need to store step counts or timestamps as proxy identifiers, then resolve to checkpoint IDs during recall. Check LangGraph `pre_model_hook` API.

### Task A3: recall_history Full Implementation

**Files**: `src/agentm/tools/orchestrator.py`, `src/agentm/builder.py`
**Size**: L
**Depends**: A1, A2

Replace the stub `recall_history` with the full implementation from orchestrator.md §History Recall:

1. Move `recall_history` into `create_orchestrator_tools` factory — it needs access to `graph` and `config` (for checkpoint traversal)
2. Read `state["compression_refs"]` to find compressed checkpoint ranges
3. Use `graph.get_state_history(config)` to traverse the checkpoint chain
4. Filter checkpoints to the compressed range (from_id → to_id)
5. Extract all messages from those checkpoints
6. Call compression/retrieval model with the query + extracted messages
7. Return the relevant information

**Key API**: `drill_down_compressed_range(graph, config, compression_ref)` — design doc provides illustrative implementation.

**Risk**: MEDIUM — Need to verify LangGraph checkpoint traversal API (`get_state_history`, `StateSnapshot` shape). Mock in tests.

**Test**: Extend `tests/snapshot/test_compression_recall.py` P7-P8 to verify recall returns meaningful content.

### Task A4: Orchestrator Notebook Compression

**Files**: `src/agentm/agents/orchestrator.py`, `src/agentm/core/prompt.py`
**Size**: M
**Depends**: A1

Integrate the already-implemented `compress_completed_phase()` into the Orchestrator's prompt pipeline:

1. In the prompt callable (passed to `create_react_agent`), check if the formatted Notebook content exceeds the token threshold
2. If so, call `compress_completed_phase()` for completed phases before formatting
3. Update `format_notebook_for_llm()` (in prompt.py or notebook.py) to use `PhaseSummary` for compressed phases instead of raw `exploration_history`
4. This is Orchestrator-level compression (Layer 2 in design doc) — independent of Sub-Agent compression (Layer 1)

**Design doc note**: orchestrator.md §Implementation Note says "Orchestrator-level Notebook compression (Layer 2) is deferred." This task implements it.

---

## Phase B: Knowledge Store Wiring (Cross-Task Infrastructure)

### Rationale

Knowledge tools are fully implemented but disconnected. Wiring the LangGraph Store backend enables cross-task knowledge retrieval in RCA — a prerequisite for the Memory Extraction system in Phase C.

### Dependency Graph

```
B1: Store Creation in Builder
  │  (builder.py — create InMemoryStore or PostgresStore)
  │  (config/schema.py — add store config fields if missing)
  │
  └── B2: Knowledge Tools Activation
        (builder.py — inject store into knowledge.set_store())
        (scenario.yaml — uncomment knowledge tools)
        │
        └── B3: Knowledge Tools Integration Test
              (tests/ — verify search/list/read against real Store)
```

### Task B1: Store Backend Creation in Builder

**Files**: `src/agentm/builder.py`, `src/agentm/config/schema.py`
**Size**: M
**Depends**: —

1. Add LangGraph Store creation to `AgentSystemBuilder.build()`:
   - `InMemoryStore` for development/testing (default)
   - `AsyncPostgresStore` for production (config-driven, optional)
2. Check `config/schema.py` for `StorageConfig.store` — add if missing (backend, connection string, embedding config)
3. Pass created store to `create_orchestrator()` (currently `store=None`)
4. Call `knowledge.set_store(store)` before tool creation

**Design doc reference**: generic-state-wrapper.md §Store Configuration.

### Task B2: Knowledge Tools Activation

**Files**: `src/agentm/builder.py`, `config/scenarios/rca_hypothesis/scenario.yaml`
**Size**: S
**Depends**: B1

1. In `builder.py`, register knowledge tools (search/list/read) in the orchestrator's tool list
2. Uncomment `knowledge_search`, `knowledge_list`, `knowledge_read` in `scenario.yaml`
3. Ensure knowledge tools are accessible as `StructuredTool` via the existing tool binding mechanism

### Task B3: Knowledge Store Integration Test

**Files**: `tests/unit/test_knowledge_store_integration.py`
**Size**: S
**Depends**: B1, B2

1. Test with `InMemoryStore`: write entry → search → list → read → delete
2. Verify semantic search returns relevant results
3. Verify path browsing (`list_namespaces`) works correctly
4. Test the full RCA workflow: knowledge tools available to Orchestrator

---

## Phase C: Memory Extraction Agent (Cross-Session Memory)

### Rationale

This is the most ambitious phase — implementing an entirely new agent system type. It enables AgentM to learn from past RCA tasks and build a persistent knowledge base. Phase B must be complete first (Knowledge Store is the output target).

### Dependency Graph

```
C1: StateGraph Mode Infrastructure
  │  (builder.py — build_graph_system() skeleton)
  │  (agents/ — BaseOrchestrator, PhaseManager integration)
  │
  ├── C2: MemoryExtractionOrchestrator
  │     (agents/memory_orchestrator.py — 4-phase StateGraph)
  │     (Sub-Agent definitions: trajectory_analyst, pattern_extractor, knowledge_writer)
  │
  ├── C3: Trajectory Reading Tools
  │     (tools/trajectory_reader.py — read_trajectory, get_checkpoint_history, compare_trajectories)
  │
  └── C4: Memory Extraction Scenario Config
        (config/scenarios/memory_extraction/ — scenario.yaml + prompts)
        │
        └── C5: End-to-End Integration
              (tests/ — RCA → Memory Extraction → Knowledge Store → next RCA)
```

### Task C1: StateGraph Mode Infrastructure

**Files**: `src/agentm/builder.py`, `src/agentm/agents/base_orchestrator.py` (new)
**Size**: L
**Depends**: Phase B complete

1. Implement `build_graph_system()` in builder.py — currently raises `NotImplementedError`
2. Create `BaseOrchestrator` abstract class with:
   - PhaseManager integration (transition validation)
   - Sub-Agent dispatch (shared with ReAct mode via TaskManager)
   - `_decide_next_phase()` abstract method
3. Build a `StateGraph` with phase nodes from config:
   - Each phase becomes a node
   - `_decide_next_phase()` drives conditional edges
   - Phase `on_enter` / `on_exit` hooks for compression triggers

**Design doc reference**: generic-state-wrapper.md §Base Orchestrator, §Phase Management.

**Risk**: HIGH — Architectural complexity. StateGraph construction from config is a new pattern. Start with hardcoded memory_extraction graph, then generalize.

### Task C2: MemoryExtractionOrchestrator

**Files**: `src/agentm/agents/memory_orchestrator.py` (new)
**Size**: L
**Depends**: C1

Implement the four-phase memory extraction flow:

1. **COLLECT** — Load target trajectories from checkpoint store by thread_id
2. **ANALYZE** — Dispatch trajectory_analyst and pattern_extractor Sub-Agents
3. **EXTRACT** — Synthesize findings into `KnowledgeEntry` objects, deduplicate against existing
4. **REFINE** — Compare with existing knowledge, merge or create new entries, write to Store

Phase transition logic:
- `collect → analyze` (always)
- `analyze → analyze` (if more trajectories to process) or `analyze → extract`
- `extract → refine` (always)
- `refine → []` (terminal)

Sub-Agent roles:
- `trajectory_analyst` — reads trajectory data, identifies decision points, phase durations
- `pattern_extractor` — finds recurring failures, common root causes across trajectories
- `knowledge_writer` — writes/updates knowledge entries with proper confidence levels

### Task C3: Trajectory Reading Tools

**Files**: `src/agentm/tools/trajectory_reader.py` (new)
**Size**: M
**Depends**: C1

Tools for Memory Extraction Sub-Agents to read historical RCA trajectories:

1. `read_trajectory(thread_id)` — Read a completed RCA task's trajectory from checkpoint store
2. `get_checkpoint_history(thread_id, from_step?, to_step?)` — Browse checkpoint chain for a thread
3. `compare_trajectories(thread_ids)` — Side-by-side comparison of multiple RCA runs

These tools read from the same checkpoint store that RCA tasks write to.

### Task C4: Memory Extraction Scenario Configuration

**Files**: `config/scenarios/memory_extraction/scenario.yaml` (new), `config/scenarios/memory_extraction/prompts/` (new)
**Size**: M
**Depends**: C2, C3

Create the scenario configuration and prompt templates:

1. `scenario.yaml` — system_type: memory_extraction, orchestrator_mode: graph, phase definitions, agent definitions, tool lists
2. Prompt templates:
   - `prompts/orchestrator_system.j2` — Memory extraction system prompt
   - `prompts/agents/trajectory_analyst.j2`
   - `prompts/agents/pattern_extractor.j2`
   - `prompts/agents/knowledge_writer.j2`

**Design doc reference**: generic-state-wrapper.md §Configuration — complete YAML example provided.

### Task C5: End-to-End Integration Test

**Files**: `tests/integration/test_memory_extraction_e2e.py` (new)
**Size**: L
**Depends**: C1-C4, Phase B

Full pipeline test:

1. Run a mock RCA task → produces trajectory + checkpoints
2. Run Memory Extraction on that trajectory → produces KnowledgeEntry items in Store
3. Run a second RCA task → Orchestrator uses `knowledge_search` to find relevant patterns
4. Verify the knowledge influences hypothesis generation

This is the ultimate acceptance test for the cross-session memory loop.

---

## Parallelization Map

```
Phase A (intra-task):
  Wave A1: A1 (Sub-Agent compression)
  Wave A2: A2 (CompressionRef) + A4 (Notebook compression)  [parallel]
  Wave A3: A3 (recall_history)  [depends on A1, A2]

Phase B (cross-task infra):
  Wave B1: B1 (Store creation)           [can start parallel with late Phase A]
  Wave B2: B2 (tools activation) + B3 (integration test)  [parallel after B1]

Phase C (cross-session):
  Wave C1: C1 (StateGraph infra) + C3 (trajectory tools)  [parallel]
  Wave C2: C2 (MemoryExtractionOrchestrator)
  Wave C3: C4 (config + prompts)
  Wave C4: C5 (E2E test)
```

**Estimated total**: 13 tasks, ~3 waves of parallelizable work per phase.

---

## Risk Assessment

| Risk | Level | Phase | Mitigation |
|------|-------|-------|-----------|
| `pre_model_hook` cannot access checkpoint IDs | HIGH | A2 | Use step count as proxy; resolve to checkpoint ID during recall |
| LangGraph Store embedding setup complexity | MEDIUM | B1 | Start with `InMemoryStore` (no embeddings); add Postgres + embeddings later |
| StateGraph from config — new pattern | HIGH | C1 | Hardcode memory_extraction graph first, then generalize |
| Trajectory checkpoint format assumptions | MEDIUM | C3 | Verify with real checkpoint data from a mock RCA run |
| LLM compression quality | LOW | A1 | Use structured prompt; quality tuning is iterative |

---

## Acceptance Criteria

### Phase A
- [ ] Sub-Agent compression hook calls LLM and produces a summary when threshold exceeded
- [ ] `CompressionRef` is appended to `state["compression_refs"]` after compression
- [ ] `recall_history("query")` traverses checkpoints and returns relevant information
- [ ] Orchestrator Notebook uses `PhaseSummary` for completed phases
- [ ] All existing 178 tests continue to pass
- [ ] `tests/snapshot/test_compression_recall.py` P7-P8 verify compression + recall cycle

### Phase B
- [ ] `AgentSystemBuilder.build()` creates a LangGraph Store instance
- [ ] Knowledge tools (`search`, `list`, `read`) available to RCA Orchestrator
- [ ] `knowledge_search` returns results from Store with semantic similarity
- [ ] Integration test: write → search → read round-trip

### Phase C
- [ ] `orchestrator_mode: "graph"` no longer raises `NotImplementedError`
- [ ] Memory Extraction system processes a completed RCA trajectory
- [ ] `KnowledgeEntry` items written to Store with correct confidence levels
- [ ] E2E test: RCA → Memory Extraction → Knowledge Store → next RCA retrieval
- [ ] Coverage >= 80% on new modules

---

## Related Documents

- [Orchestrator Design — Compression & Recall](../designs/orchestrator.md)
- [Generic State Wrapper — Memory Extraction](../designs/generic-state-wrapper.md)
- [Testing Strategy](../designs/testing-strategy.md)
- [Builder Design](../designs/builder.md)
