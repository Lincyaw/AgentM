**Status**: HISTORICAL — describes the pre-v2 architecture removed in Phase 2.5 (2026-04-30).
The current architecture lives in [pluggable-architecture.md](../pluggable-architecture.md) and
[extension-as-scenario.md](../extension-as-scenario.md).

---

# Design: Agent Memory Scope

**Status**: DRAFT
**Created**: 2026-03-31

---

## 1. Overview

Agent Memory Scope provides per-agent persistent memory that survives across runs, enabling agents to accumulate knowledge, preferences, and behavioral instructions over time. It is a simplified, scoped application of the existing [MemoryVault](memory-vault.md) storage infrastructure.

---

## 2. Background & Motivation

### 2.1 Problem

AgentM currently has two memory mechanisms:

1. **Intra-run**: CompressionMiddleware summarizes conversation history within a single run.
2. **Cross-run knowledge**: MemoryVault stores domain knowledge (skills, episodic, concepts) that agents can search and read via vault tools.

Neither provides **per-agent persistent memory** — the ability for a specific agent to remember learnings, preferences, and behavioral corrections across independent runs. If an agent learns "always check connection pool metrics before diagnosing timeout" during run N, that insight is lost by run N+1 unless manually encoded.

### 2.2 Inspiration: Claude Code's Three-Layer Memory

Claude Code implements a three-scope memory system:

| Scope | Persistence | Sharing | Storage |
|-------|------------|---------|---------|
| `user` | Cross-project | User-only | `~/.claude/MEMORY.md` |
| `project` | Per-project | Team (git-tracked) | `<project>/.claude/MEMORY.md` |
| `local` | Per-project+machine | Machine-only | `<project>/.claude/.local/MEMORY.md` |

Key implementation insight: memory is injected as a **system prompt suffix**. The prompt contains behavioral instructions plus MEMORY.md index content. The agent reads/writes memory using dedicated tools, but influence happens through prompt injection at every LLM call.

### 2.3 What Agent Memory Enables

- **Cross-run learning**: Agent accumulates patterns ("service X frequently has connection pool issues")
- **Behavioral correction persistence**: User feedback ("don't use grep -r on large directories") survives across runs
- **Agent specialization**: Over time, each agent develops memory tailored to its role

---

## 3. Design

### 3.1 Memory Scope Enum

```python
class MemoryScope(StrEnum):
    AGENT = "agent"         # Per-agent, persistent across runs
    SCENARIO = "scenario"   # Shared across agents within a scenario type
    PROJECT = "project"     # Global project-level memory (cross-scenario)
```

| Scope | Lifetime | Visibility | Use Case |
|-------|----------|------------|----------|
| `AGENT` | Persists across runs | Single agent identity | Agent-specific learnings |
| `SCENARIO` | Persists across runs | All agents in same scenario type | Scenario-level patterns |
| `PROJECT` | Persists across runs | All agents, all scenarios | Project-wide conventions |

Scope resolution order (highest priority first): `AGENT > SCENARIO > PROJECT`.

### 3.2 Directory Structure

```
<memory_root>/
  project/
    MEMORY.md                          # Project-wide memory index
    notes/
      prefer-structured-logging.md
  scenario/
    rca/
      MEMORY.md                        # RCA scenario shared memory
      notes/
        cascading-failure-pattern.md
  agent/
    rca-orchestrator/
      MEMORY.md                        # Per-agent memory index
      notes/
        always-check-pool-metrics.md
```

`MEMORY.md` at each scope level serves as the **index file** — a concise summary of all memory entries. Individual notes under `notes/` contain detailed content.

### 3.3 MEMORY.md Format

```markdown
# Agent Memory: rca-orchestrator

- [Pool Metrics First](notes/always-check-pool-metrics.md) — always check connection pool before diagnosing timeout
- [Brief Summaries](notes/user-prefers-brief-summaries.md) — user prefers concise bullet-point summaries
```

Format: `[title](relative-path) — one-line description`. Human-readable, git-trackable, cheap to inject into prompts.

### 3.4 AgentMemoryConfig

```python
@dataclass
class AgentMemoryConfig:
    enabled: bool = True
    memory_root: str = "./.agent-memory"
    scopes: list[MemoryScope] = field(
        default_factory=lambda: [MemoryScope.AGENT, MemoryScope.SCENARIO, MemoryScope.PROJECT]
    )
    max_prompt_entries: int = 50
    agent_identity: str = ""              # Derived from config if empty
    scenario_name: str = ""               # Derived from config if empty
```

### 3.5 `load_agent_memory_prompt()`

Constructs a prompt fragment containing behavioral instructions and memory index content from all applicable scopes:

```xml
<agent_memory>
You have persistent memory that accumulates across runs. The entries below
represent learnings, corrections, and preferences from previous sessions.
Treat them as strong priors — follow them unless the current task explicitly
contradicts them.

<memory scope="project">
- user prefers concise bullet-point summaries over verbose explanations
- environment uses Kubernetes 1.28 on AWS EKS
</memory>

<memory scope="scenario" name="rca">
- cascading failures are common — always check upstream dependencies
</memory>

<memory scope="agent" name="rca-orchestrator">
- always check connection pool metrics before diagnosing timeout issues
</memory>

To save a new learning, use `memory_write`. To review or update existing
entries, use `memory_read` and `memory_edit`.
</agent_memory>
```

The function reads each applicable `MEMORY.md`, extracts description lines, and assembles them into XML-tagged blocks. Agent scope entries appear last (highest priority).

---

## 4. Prompt Injection — MemoryMiddleware

### Decision: MemoryMiddleware (new MiddlewareBase subclass)

| Option | Pros | Cons |
|--------|------|------|
| A: Static injection at construction | Simple | Cannot reflect mid-run memory updates |
| B: **MemoryMiddleware (on_llm_start)** | Follows SkillMiddleware pattern; supports dirty-flag refresh | Slight per-call overhead |

**Recommendation: Option B.** Consistent with `SkillMiddleware`. Memory prompt is cached and only re-read when a write occurs (dirty flag).

### Middleware Stack Position

```
DynamicContextMiddleware → LoopDetectionMiddleware → CompressionMiddleware
  → SkillMiddleware → MemoryMiddleware → [scenario middleware] → TrajectoryMiddleware
```

### Builder Integration

Phases 2 and 4 of `build_agent_system()` instantiate `MemoryMiddleware` and insert it into the middleware stack when `AgentMemoryConfig.enabled` is true.

---

## 5. Read/Write Tools

### Decision: Dedicated memory tools with vault parser backend

| Option | Pros | Cons |
|--------|------|------|
| A: Reuse vault tools | No new tools | No scope enforcement, manual index maintenance |
| B: **Dedicated tools** | Clean API, scope enforcement, auto-maintained index | More tools to register |

**Recommendation: Option B.** Memory has stricter invariants (mandatory index, scope isolation). Internally uses vault's parser utilities but not the full `MarkdownVault` class.

### Tool Specifications

| Tool | Parameters | Description |
|------|-----------|-------------|
| `memory_read` | `scope?: MemoryScope`, `path?: str` | Read memory index or a specific note |
| `memory_write` | `title: str`, `description: str`, `body?: str`, `scope?: MemoryScope` | Create entry, auto-update MEMORY.md index. Default: AGENT scope |
| `memory_edit` | `path: str`, `operation: str`, `params: dict` | Edit existing note (`replace_string`, `set_description`) |
| `memory_delete` | `path: str` | Remove entry and its index line |

**Scope restriction**: Agent can write to `AGENT` and `SCENARIO` scopes. `PROJECT` scope is read-only at runtime (written by humans or offline processes).

### Relationship with MemoryVault

| Aspect | MemoryVault | Agent Memory |
|--------|-------------|--------------|
| Purpose | Domain knowledge (skills, concepts) | Agent behavioral memory (learnings, preferences) |
| Storage | `<kb_dir>/vault/` with `.vault.db` | `<memory_root>/` with flat markdown |
| Index | SQLite (FTS5 + vec0 + link graph) | MEMORY.md index files (plain markdown) |
| Tools | 10 vault tools | 4 memory tools |
| Sharing | Shared across all agents | Scoped (project / scenario / agent) |

Agent memory **reuses vault's parser module** for frontmatter parsing and section editing. This is a code-level dependency, not architectural coupling.

---

## 6. Configuration

### scenario.yaml

```yaml
agents:
  orchestrator:
    memory:
      enabled: true
      scopes: [agent, scenario, project]
      max_prompt_entries: 50
  worker:
    memory:
      enabled: true
      scopes: [agent, scenario]
      max_prompt_entries: 20
```

### system.yaml

```yaml
memory:
  root: "./.agent-memory"
  project_scope: true
```

---

## 7. Impact Analysis

| Document | Impact |
|----------|--------|
| [memory-vault.md](memory-vault.md) | No changes. Architecturally separate. |
| [agent-harness.md](agent-harness.md) | Add `MemoryMiddleware` to middleware catalog |
| [system-design-overview.md](system-design-overview.md) | Add memory config to config hierarchy |
| builder.py | Phases 2 and 4 instantiate MemoryMiddleware |

---

## 8. Constraints and Decisions

| Decision | Rationale |
|----------|-----------|
| Three scopes (agent/scenario/project) | Maps to Claude Code's proven model, adapted for multi-agent |
| MEMORY.md as index file | Human-readable, git-trackable, cheap to inject |
| Flat markdown (no SQLite) | Memory is small (tens of entries), FTS unnecessary |
| MemoryMiddleware for injection | Consistent with SkillMiddleware, supports mid-run updates |
| Dedicated tools (not vault reuse) | Scope enforcement, auto-index maintenance |
| PROJECT scope read-only at runtime | Prevents agents overwriting team-shared conventions |
| Agent identity from config key | No manual naming required |

---

## 9. Open Questions

- Should agent identity be config key (`orchestrator`) or scenario-qualified (`rca-orchestrator`)?
- Should memory entries have TTL or confidence decay?
- Should `memory_search` tool exist, or is the index sufficient (expected < 100 entries)?
- How to handle memory conflicts when same agent identity used across scenario types?
- Should PROJECT scope MEMORY.md be committed to git?

---

## 10. Related Concepts

- [Memory Vault](memory-vault.md) — Shared domain knowledge; agent memory reuses parser module
- [Agent Harness](agent-harness.md) — Middleware protocol for prompt injection
- [System Design Overview](system-design-overview.md) — Config hierarchy, builder phases
- [Sub-Agent](sub-agent.md) — Workers receive memory via MemoryMiddleware