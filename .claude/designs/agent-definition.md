**Status**: HISTORICAL — describes the pre-v2 architecture removed in Phase 2.5 (2026-04-30).
The current architecture lives in [pluggable-architecture.md](pluggable-architecture.md) and
[extension-as-scenario.md](extension-as-scenario.md).

---

# Design: Declarative Agent Definition

**Status**: DRAFT
**Created**: 2026-03-31

---

## 1. Overview

`AgentDefinition` is a frozen dataclass that fully describes a single agent's identity, behavior, and constraints. It replaces `AgentConfig` as the canonical agent descriptor, loaded from either **Markdown frontmatter** files or **YAML** blocks, unifying agent metadata and system prompt into a single artifact.

---

## 2. Motivation

### 2.1 Current Problems

1. **Scattered configuration** — Agent identity is split across two locations: structural parameters in `scenario.yaml` under `agents.worker`, and the system prompt in a separate `.j2` file referenced by `prompt:`. Understanding what an agent *is* requires reading both files and mentally merging them.

2. **Single-worker assumption** — `scenario.yaml` defines `agents: dict[str, AgentConfig]`, but in practice every scenario uses only `agents.worker`. There is no first-class support for multiple named agent types within the same scenario.

3. **Organic precedent** — The `trajectory_analysis` scenario already uses Markdown frontmatter files (`workers/trajectory-reader.md`, `workers/analysis-critic.md`) with `name` and `description` in YAML frontmatter and the system prompt as the body. But this pattern is ad-hoc.

4. **Inspiration from Claude Code** — Claude Code defines agents as single Markdown files with YAML frontmatter (name, description, tools, model) and the body as the system prompt. One file = one agent.

### 2.2 Goals

- **One file, one agent** — All agent metadata and system prompt in a single file.
- **Multiple named agents per scenario** — Replace `agents: dict[str, AgentConfig]` with `dict[str, AgentDefinition]` loaded from an `agents/` directory.
- **Backward compatibility** — Existing `scenario.yaml` `agents:` blocks continue to work during migration.
- **Jinja2 support preserved** — System prompts still support template variables.

---

## 3. Design Details

### 3.1 AgentDefinition Dataclass

```python
@dataclass(frozen=True)
class AgentDefinition:
    """Complete, immutable definition of a single agent."""

    # --- Identity ---
    name: str                              # Unique name within scenario
    description: str = ""                  # Human-readable purpose

    # --- Behavior ---
    task_type: str | None = None           # Default task_type for dispatch routing
    system_prompt: str = ""                # Full system prompt text (may contain Jinja2)

    # --- Model ---
    model: str = ""                        # Model name (resolved against system.yaml)
    temperature: float = 0.0

    # --- Tools ---
    tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    include_think_tool: bool = True

    # --- Execution ---
    max_steps: int = 20
    timeout: int = 120
    tool_call_budget: int | None = None

    # --- Advanced ---
    skills: list[str] = field(default_factory=list)
    tool_settings: dict[str, dict[str, Any]] = field(default_factory=dict)

    # --- Source metadata ---
    source_file: Path | None = None
```

**Key differences from `AgentConfig`**:

| Aspect | `AgentConfig` (current) | `AgentDefinition` (new) |
|--------|------------------------|------------------------|
| Type | Pydantic `BaseModel` (mutable) | Frozen `dataclass` (immutable) |
| System prompt | External `.j2` file path in `prompt` field | Inline `system_prompt` text |
| Task-type prompts | Separate `task_type_prompts` dict of paths | Each task-type is its own `AgentDefinition` |
| Execution config | Nested `ExecutionConfig` object | Flat fields |
| Identity | Implicit (dict key in `agents:`) | Explicit `name` field |

### 3.2 Loading Formats

#### Format A: Markdown Frontmatter (preferred for new agents)

```markdown
---
name: trajectory-reader
description: Read and analyze agent execution trajectories
task_type: read
model: gpt-5.1-mini
temperature: 0.7
tools: [jq_query, vault_read, vault_search, load_case_data]
max_steps: 60
timeout: 180
skills:
  - skill/diagnose-sql
---

You are a trajectory analysis worker. You analyze one agent execution
trajectory and return structured findings.

## Workflow

1. Detect format...
2. Build global picture...
```

Frontmatter is parsed as YAML → `AgentDefinition` fields. Body (after closing `---`) becomes `system_prompt`. Body is treated as Jinja2 template, rendered at worker creation time.

#### Format B: YAML block (backward compatible)

Existing `scenario.yaml` `agents:` blocks are converted via `agent_config_to_definition()`:

1. Base `worker` entry → `AgentDefinition` with `name="worker"`.
2. If `prompt` is a file path, loaded and stored as `system_prompt`.
3. `task_type_prompts` entries remain as overlay prompts (current behavior preserved).
4. `execution.*` fields flattened into top-level.

### 3.3 Loading Function

```python
def load_agent_definitions(
    scenario_dir: Path,
    scenario_config: ScenarioConfig | None = None,
) -> dict[str, AgentDefinition]:
    """Load agent definitions from a scenario directory.

    Sources (later overrides earlier):
    1. Markdown files in scenarios/<name>/agents/*.md
    2. YAML agents: block in scenario.yaml (converted)
    """
```

Discovery:
1. Scan `scenarios/<name>/agents/` for `*.md` files.
2. Parse frontmatter + body → `AgentDefinition`.
3. If `scenario_config` provided, convert its `agents:` block.
4. Merge: Markdown definitions take precedence over YAML-converted ones with the same name.

### 3.4 Directory Structure

Current:
```
config/scenarios/rca_hypothesis/
├── scenario.yaml
├── prompts/
│   ├── orchestrator_system.j2
│   └── task_types/
│       ├── scout.j2
│       └── verify.j2
```

New (coexists):
```
config/scenarios/rca_hypothesis/
├── scenario.yaml                    # agents: block still works
├── agents/                          # NEW: one .md per agent
│   ├── scout.md
│   ├── verifier.md
│   └── deep-analyzer.md
├── prompts/
│   └── orchestrator_system.j2       # orchestrator stays as .j2
```

The `agents/` directory is optional. When absent, falls back to `agents:` block in `scenario.yaml`.

### 3.5 Jinja2 Template Support

Available template variables:

| Variable | Type | Description |
|----------|------|-------------|
| `agent_id` | `str` | Runtime agent ID |
| `tools_description` | `str` | Formatted list of available tools |

Scenario-specific variables injected via `ScenarioWiring`. File extension is always `.md` — Jinja2 rendering is transparent.

### 3.6 Source Layering

| Layer | Location | Purpose |
|-------|----------|---------|
| **Built-in** | `src/agentm/defaults/agents/` (future) | SDK-provided defaults |
| **Scenario** | `config/scenarios/<name>/agents/` | Scenario-specific agents |
| **User override** | Environment or runtime injection | Per-run customization |

Merge semantics: **full replacement** (no field-level merge). If two layers define the same name, higher-priority completely replaces lower.

### 3.7 WorkerLoopFactory Integration

```python
class WorkerLoopFactory:
    def __init__(
        self,
        agent_definitions: dict[str, AgentDefinition],  # replaces ScenarioConfig
        tool_registry: ToolRegistry,
        *,
        default_agent: str = "worker",
        extra_tools: list[Tool] | None = None,
        extra_middleware: list[Any] | None = None,
        trajectory: TrajectoryCollector | None = None,
        answer_schemas: dict[str, type[BaseModel]] | None = None,
    ) -> None
```

**Dispatch routing** (when `dispatch_agent(task_type="scout")` is called):

1. If `AgentDefinition` with `name="scout"` exists → use it directly.
2. Otherwise → fall back to `default_agent` definition + task_type overlay prompt.

This allows gradual migration.

### 3.8 Migration Path

1. **Phase 1** (this design): Introduce `AgentDefinition`, `load_agent_definitions()`, and `agent_config_to_definition()`.
2. **Phase 2**: Update `WorkerLoopFactory` to accept `dict[str, AgentDefinition]`. Builder converts `AgentConfig` when `agents/` directory absent.
3. **Phase 3** (future): Deprecate `AgentConfig` and `agents:` block in `scenario.yaml`.

---

## 4. Constraints and Decisions

| Decision | Rationale |
|----------|-----------|
| Frozen dataclass, not Pydantic | Pure data, loaded once. No runtime validation needed. |
| Markdown body = system prompt | One file = one agent. No file-path indirection. |
| `.md` extension even with Jinja2 | Agent files are primarily Markdown for readability. |
| Flat fields (no nested ExecutionConfig) | Simple and readable. |
| Full replacement layering | Field-level merge is error-prone. Full replacement is predictable. |
| `task_type` field enables dispatch routing | Orchestrator dispatches by task_type; factory matches to definition. |

---

## 5. Impact Assessment

### sub-agent.md

- Section 2 (WorkerLoopFactory): Update constructor signature and dispatch routing.
- Section 9 (Configuration): Replace `AgentConfig` YAML example with Markdown example.

### system-design-overview.md

- Configuration System: Add `agents/` directory to scenario layout.

### builder.py

- Pre-phase or Phase 1: Call `load_agent_definitions()`.
- Phase 2: Pass `dict[str, AgentDefinition]` to `WorkerLoopFactory`.

### Existing scenarios migration effort

| Scenario | Migration Work |
|----------|---------------|
| `trajectory_analysis` | Rename `workers/` to `agents/`, add missing frontmatter fields |
| `trajectory_judger` | Move prompt to `agents/worker.md`, add frontmatter |
| `rca_hypothesis` | Create `agents/scout.md`, `agents/verifier.md`, `agents/deep-analyzer.md` |
| `general_purpose` | Create `agents/executor.md` |

---

## 6. Open Questions

- Should dedup/compression/loop-detection config be in agent frontmatter or remain scenario-level? (Recommend: keep in `scenario.yaml` — operational concerns, not agent identity.)
- Should the orchestrator also be an AgentDefinition? (Defer — orchestrator has unique fields.)
- Task-type routing vs. named routing precedence? (Proposed: name → task_type → default.)
- Unknown frontmatter keys: strict reject or lenient ignore? (Recommend: lenient for forward-compat.)

---

## 7. Related Concepts

- [Sub-Agent (Worker)](sub-agent.md) — `AgentDefinition` replaces configuration portion
- [System Design Overview](system-design-overview.md) — Config system updates
- [Builder](../index.yaml) — `build_agent_system()` integration
- [Tool Filter](tool-filter.md) — `tools`/`disallowed_tools` fields consumed by tool filter
- [Permission Mode](permission-mode.md) — Permission mode field on agent definition