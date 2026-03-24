# Design: Trajectory Analysis

**Status**: DRAFT
**Created**: 2026-03-24
**Last Updated**: 2026-03-24

**Supersedes**: Memory Extraction sections in [generic-state-wrapper.md](generic-state-wrapper.md)

---

## Overview

Trajectory Analysis is a **skill-driven agent system** for analyzing completed agent execution trajectories. It replaces the former `memory_extraction` scenario with a general-purpose analysis framework where **skills** determine what the system looks for and what it produces.

The core insight: reading trajectories → analyzing with a specific lens → producing structured output is a general pattern. Different analysis needs (knowledge extraction, failure diagnosis, quality audit, feature tagging) differ only in **what to look for** and **how to report it** — not in system architecture.

### Design Goals

| Goal | Description |
|------|-------------|
| Skill-driven analysis | Analysis behavior determined entirely by skill content, not code |
| Progressive disclosure | Three-tier loading: catalog (~100 tokens/skill) → SKILL.md (<5000 tokens) → references/ (on demand) |
| Zero-code extensibility | New analysis skill = new directory in vault with SKILL.md + references/. No code changes. |
| Agent Skills compatible | Skill format follows the [Agent Skills specification](https://agentskills.io/specification) |
| Dual activation modes | Model-driven (orchestrator picks) or config-driven (scenario.yaml specifies) |

---

## Relationship to Former `memory_extraction`

| Aspect | Former `memory_extraction` | New `trajectory_analysis` |
|--------|--------------------------|--------------------------|
| **Scope** | One fixed use case (knowledge extraction) | Any trajectory analysis task via skills |
| **Phases** | Hardcoded 4: collect → analyze → extract → refine | Simplified 2: analyze → synthesize |
| **State** | `KnowledgeEntry`, `extracted_patterns`, `existing_knowledge` | Generic: `analysis_results: list[dict]` |
| **Prompt** | 145-line monolithic orchestrator_system.j2 | ~20-line skeleton + SKILL.md (<500 lines) + references/ (on demand) |
| **Extensibility** | New use case = new scenario code | New use case = new SKILL.md file in vault |
| **system_type** | `"memory_extraction"` | `"trajectory_analysis"` |

The former `memory_extraction` functionality becomes the `memory-extraction` skill — the first and most mature skill in the trajectory analysis system.

---

## Skill Architecture

### What Is a Skill

A skill is a **directory in the vault** containing instructions, reference material, and examples that tell the orchestrator how to analyze trajectories. Skills follow the [Agent Skills specification](https://agentskills.io/specification):

```
vault/skill/trajectory-analysis/
├── memory-extraction/
│   ├── SKILL.md                    # Required: metadata + core instructions (<500 lines)
│   ├── references/                 # Optional: loaded on demand via vault_read
│   │   ├── what-to-extract.md
│   │   ├── entry-quality.md
│   │   ├── categories.md
│   │   └── dedup-strategy.md
│   └── examples/                   # Optional: concrete good/bad examples
│       ├── good-entry.md
│       └── bad-entry.md
│
├── failure-diagnosis/
│   ├── SKILL.md
│   ├── references/
│   │   ├── failure-taxonomy.md
│   │   └── root-cause-checklist.md
│   └── examples/
│       └── sample-report.md
│
└── quality-audit/
    ├── SKILL.md
    └── references/
        └── scoring-rubric.md
```

### SKILL.md Format

```markdown
---
name: memory-extraction
description: >
  Extract reusable diagnostic knowledge from RCA trajectories into the
  vault. Use when processing completed RCA runs to build the knowledge
  store, even if the user just says "learn from these runs."
---

# Memory Extraction

[Core instructions — workflow, decision tree, gotchas]
[Links to references/ and examples/ for deeper guidance]
```

Frontmatter fields (per Agent Skills spec):

| Field | Required | Purpose |
|-------|----------|---------|
| `name` | Yes | Identifier, must match directory name. Lowercase, hyphens. Max 64 chars. |
| `description` | Yes | When to activate this skill. Max 1024 chars. Imperative phrasing ("Use when..."). |

### Progressive Disclosure

Three tiers, loaded at different times:

| Tier | Content | When loaded | Token budget | How loaded |
|------|---------|-------------|--------------|------------|
| **1. Catalog** | `name` + `description` from each SKILL.md frontmatter | Strategy init (graph compile time) | ~50-100 per skill | Strategy scans vault |
| **2. Instructions** | SKILL.md body (frontmatter stripped) | Orchestrator activates skill (first LLM round) | < 5000 tokens | `vault_read("skill/trajectory-analysis/{name}/SKILL.md")` |
| **3. Resources** | references/, examples/ | During analysis, when SKILL.md links direct | Varies | `vault_read("skill/trajectory-analysis/{name}/references/foo.md")` |

Key principle from Agent Skills best practices: **SKILL.md carries the entire burden of activation and core guidance.** Gotchas and decision trees belong in SKILL.md (always loaded). Detailed rubrics, taxonomies, and examples belong in references/ (loaded on demand).

### Skill Content Best Practices

From [agentskills.io best practices](https://agentskills.io/skill-creation/best-practices):

- **Add what the agent lacks, omit what it knows** — no explaining what trajectories are
- **Provide defaults, not menus** — one clear workflow path, alternatives mentioned briefly
- **Gotchas are highest-value content** — keep in SKILL.md, not in references/
- **Match specificity to fragility** — prescriptive for vault_write sequences, flexible for analysis approach
- **Favor procedures over declarations** — teach how to analyze, not what specific answers to produce
- **Keep SKILL.md under 500 lines** — move detailed rubrics to references/

---

## Activation Modes

### Model-Driven Activation (Default)

The orchestrator sees an `<available_skills>` catalog in its system prompt and decides which skill to load based on the task description.

```xml
<available_skills>
<skill>
  <name>memory-extraction</name>
  <description>Extract reusable diagnostic knowledge from RCA trajectories
    into the vault. Use when processing completed RCA runs to build the
    knowledge store.</description>
  <path>skill/trajectory-analysis/memory-extraction</path>
</skill>
<skill>
  <name>failure-diagnosis</name>
  <description>Analyze why an RCA investigation failed or was suboptimal.
    Use when a completed RCA produced wrong results or took too long.</description>
  <path>skill/trajectory-analysis/failure-diagnosis</path>
</skill>
</available_skills>
```

The orchestrator activates a skill by calling `vault_read(path="skill/trajectory-analysis/{name}/SKILL.md")`, receiving the body content. It then follows the skill's workflow, loading references/ on demand.

### Config-Driven Activation (Batch/CI)

For unattended runs, `scenario.yaml` specifies the skill directly:

```yaml
system:
  type: "trajectory_analysis"
  skill: "memory-extraction"     # Skip discovery, pre-activate this skill
```

When `skill` is set, the Strategy pre-loads the SKILL.md body and injects it into `format_context` from the first round. The orchestrator does not see an `<available_skills>` catalog — it receives the skill content directly.

### When to Use Which

| Mode | Use case |
|------|----------|
| Model-driven | Interactive: user describes what they want, orchestrator picks |
| Config-driven | Batch processing, CI pipelines, scheduled jobs |

---

## State Schema

```python
class TrajectoryAnalysisState(BaseExecutorState):
    """Generic state for all trajectory analysis skills.

    Inherits messages, task_id, task_description, current_phase
    from BaseExecutorState.
    """

    source_trajectories: list[str]                          # Thread IDs to analyze
    skill_name: str                                          # Active skill (set after activation)
    analysis_results: Annotated[list[dict], operator.add]    # Worker findings (generic shape)
    structured_output: Optional[Any]                         # Final output (skill-specific)
```

No skill-specific fields (no `knowledge_entries`, no `extracted_patterns`). Skills that need intermediate data use the LLM's message history and tool calls, not state fields.

---

## Phase Pipeline

Two phases instead of four:

```python
phases = {
    "analyze": PhaseDefinition(
        name="analyze",
        description="Activate skill, read trajectories, dispatch workers, collect findings",
        next_phases=["synthesize"],
    ),
    "synthesize": PhaseDefinition(
        name="synthesize",
        description="Aggregate findings, produce output, write artifacts",
        next_phases=[],
    ),
}
```

| Phase | What happens | Actor |
|-------|-------------|-------|
| **analyze** | Orchestrator activates skill (if model-driven), reads trajectories, dispatches workers for parallel analysis, collects results | Orchestrator + Workers |
| **synthesize** | Orchestrator aggregates worker findings, produces structured output, writes artifacts (vault entries, reports) as skill directs | Orchestrator |

The skill's SKILL.md determines the detailed workflow within each phase. The phases are markers, not rigid constraints — the orchestrator interleaves freely within the ReAct loop.

---

## Orchestrator Prompt

### System Prompt Skeleton (orchestrator_system.j2)

```jinja2
You are a trajectory analysis system. You read completed agent execution
trajectories and produce structured analysis based on your active skill.

<tools>
- read_trajectory(thread_id) — full message history of a past run
- get_checkpoint_history(thread_id, limit?) — checkpoint metadata
- vault_read(path) — read a vault entry (skill references, existing knowledge)
- vault_write(path, frontmatter, body) — write to vault
- vault_search(query, limit?, mode?) — search vault
- vault_list(path?, depth?) — list vault entries
- vault_edit(path, operation, params) — edit a vault entry
- dispatch_agent(agent_id, task, task_type) — launch a worker
- check_tasks(request) — collect worker results
</tools>

{{ skill_section }}
```

### Skill Section Rendering

**Model-driven mode** (no `skill` in config):

```jinja2
<skill_usage>
1. Read the task description to understand what analysis is needed.
2. Review the available skills and pick the best match.
3. Load the skill: vault_read(path="<skill_path>/SKILL.md")
4. Follow the skill's workflow. Load references via vault_read as directed.
</skill_usage>

{{ available_skills_xml }}
```

**Config-driven mode** (`skill: "memory-extraction"` in config):

```jinja2
<active_skill name="{{ skill_name }}">
{{ skill_body }}

Skill directory: skill/trajectory-analysis/{{ skill_name }}
Relative paths in this skill resolve against the skill directory.
</active_skill>
```

---

## Strategy

### TrajectoryAnalysisStrategy

```python
class TrajectoryAnalysisStrategy:
    """ReasoningStrategy for skill-driven trajectory analysis.

    At init time:
    - Scans vault for skills under skill/trajectory-analysis/
    - Reads SKILL.md frontmatter (name + description) for the catalog
    - If config specifies a skill, pre-loads its body

    At runtime:
    - format_context() injects either the catalog or the pre-loaded skill
    - should_terminate() returns True when current_phase == "synthesize"
    """

    @property
    def name(self) -> str:
        return "trajectory_analysis"

    def initial_state(self, task_id, task_description) -> TrajectoryAnalysisState:
        return TrajectoryAnalysisState(
            messages=[HumanMessage(content=task_description)],
            task_id=task_id,
            task_description=task_description,
            current_phase="analyze",
            source_trajectories=[],
            skill_name="",
            analysis_results=[],
            structured_output=None,
        )

    def format_context(self, state) -> str:
        # If config-driven: return pre-loaded skill body + source trajectories
        # If model-driven: return available_skills catalog + source trajectories
        ...

    def phase_definitions(self) -> dict[str, PhaseDefinition]:
        return {"analyze": ..., "synthesize": ...}

    def should_terminate(self, state) -> bool:
        return state.get("current_phase") == "synthesize"

    def state_schema(self) -> type:
        return TrajectoryAnalysisState
```

### Skill Discovery

At Strategy init time, the vault is scanned:

```python
def _discover_skills(self, vault: MarkdownVault) -> list[SkillCatalogEntry]:
    """Scan vault for trajectory analysis skills.

    Reads SKILL.md frontmatter (name + description) for each skill
    directory under skill/trajectory-analysis/.
    """
    entries = []
    for note in vault.list("skill/trajectory-analysis", depth=1):
        skill_md = vault.read(f"{note}/SKILL.md")
        if skill_md is None:
            continue
        fm = skill_md["frontmatter"]
        entries.append(SkillCatalogEntry(
            name=fm.get("name", ""),
            description=fm.get("description", ""),
            path=note,
        ))
    return entries
```

`SkillCatalogEntry` is a simple dataclass — not a new abstraction, just a container for the catalog data:

```python
@dataclass
class SkillCatalogEntry:
    name: str
    description: str
    path: str   # vault path, e.g. "skill/trajectory-analysis/memory-extraction"
```

---

## Output Schema

### Generic AnalysisReport

A single output schema serves all skills:

```python
class AnalysisReport(BaseModel):
    """Final output of a trajectory analysis run."""

    skill: str = Field(description="Which analysis skill was applied")
    source_count: int = Field(description="Number of trajectories analyzed")
    findings: list[dict] = Field(
        description="Key findings from the analysis (skill-specific shape)"
    )
    artifacts: list[str] = Field(
        description="Paths of artifacts created (vault entries, reports, etc.)"
    )
    summary: str = Field(
        description="Prose summary of the analysis"
    )
```

`findings` is `list[dict]` — intentionally untyped. Each skill's SKILL.md instructs the LLM what structure to put in findings. This avoids per-skill schema proliferation while still producing structured output.

Skills that need a specialized output schema (e.g., the existing `KnowledgeSummary`) can register theirs in `OUTPUT_SCHEMAS` and reference it in scenario.yaml's `output.schema_name`. The generic `AnalysisReport` is the default.

---

## Worker Definition

### File-Based Worker Roles

Workers are defined as **markdown files** following the same pattern as Claude Code's subagent definitions (`.claude/agents/*.md`). Each file has frontmatter (name, description) and a body (role instructions):

```
config/scenarios/trajectory_analysis/
├── workers/
│   └── trajectory-reader.md    # The default (and only) worker role
├── prompts/
│   └── orchestrator_system.j2
└── scenario.yaml
```

### Worker File Format

```markdown
---
name: trajectory-reader
description: >
  Read and analyze agent execution trajectories. Returns structured
  findings based on the orchestrator's dispatch instruction.
---

You are a trajectory analysis worker. You read one agent execution
trajectory and return structured findings.

## Workflow

1. `read_trajectory(thread_id)` — load full message history
2. `get_checkpoint_history(thread_id)` — inspect phase/step metadata
3. Analyze the trajectory through the lens described in your task instruction
4. Search existing vault knowledge for context: `vault_search(query=...)`
5. Return your findings in the structured format your task requests

## Gotchas

- **Read the full trajectory before analyzing** — don't jump to conclusions
  from the first few messages
- **Distinguish correlation from causation** — "X happened before Y"
  does not mean X caused Y
- **Report what you observe, don't prescribe** — the orchestrator decides
  what to do with your findings
- **Check existing knowledge** — vault_search before reporting a "new"
  pattern that may already be documented

## Output

Return `findings`: your analysis structured as requested by the task.
Be specific — use exact service names, metric values, and timestamps
from the trajectory. Omit reasoning steps; report only findings.
```

### How It Works

The worker file is loaded once at graph compile time as the worker's base system prompt. The **orchestrator's dispatch instruction** (the `task` parameter in `dispatch_agent`) carries the skill-specific analysis lens:

```
# Memory extraction skill dispatches:
dispatch_agent(agent_id="worker", task="Read trajectory thread-001 and
  identify transferable diagnostic principles. For each principle, search
  vault for similar existing entries.", task_type="analyze")

# Failure diagnosis skill dispatches:
dispatch_agent(agent_id="worker", task="Read trajectory thread-001 and
  identify where the investigation went wrong. What signals were missed?
  What reasoning errors occurred?", task_type="analyze")
```

Same worker, same base prompt, different instruction. The skill's SKILL.md guides the orchestrator to write the right dispatch instruction for its analysis lens. This means:

- **One worker definition** serves all skills
- **No per-skill worker prompts** — the orchestrator prompt (shaped by SKILL.md) determines what the worker does
- **New skills don't need new worker files** — they just need good orchestrator-level instructions

### scenario.yaml Worker Config

```yaml
agents:
  worker:
    model: "gpt-5.1-mini"
    temperature: 0.2
    prompt: "workers/trajectory-reader.md"    # File-based role definition
    tools:
      - read_trajectory
      - get_checkpoint_history
      - vault_read
      - vault_search
      - vault_list
      - vault_backlinks
      - vault_traverse
    execution:
      max_steps: 20
      timeout: 120
```

The `prompt` field points to the worker role file (relative to scenario directory). At compile time, the builder reads this file, strips frontmatter, and uses the body as the worker's system prompt — the same flow as the existing `AgentConfig.prompt` field, now pointing to a richer markdown file instead of a bare j2 template.

---

## Scenario Configuration

```yaml
# config/scenarios/trajectory_analysis/scenario.yaml
system:
  type: "trajectory_analysis"
  # skill: "memory-extraction"    # Uncomment for config-driven mode

orchestrator:
  model: "gpt-5.1"
  temperature: 0.3
  orchestrator_mode: "node"
  max_rounds: 20

  prompts:
    system: "prompts/orchestrator_system.j2"

  tools:
    - dispatch_agent
    - check_tasks
    - vault_write
    - vault_read
    - vault_edit
    - vault_delete
    - vault_rename
    - vault_search
    - vault_list
    - vault_backlinks
    - vault_traverse
    - vault_lint
    - read_trajectory
    - get_checkpoint_history

  output:
    prompt: "prompts/output/analysis_report.j2"
    schema_name: "AnalysisReport"

agents:
  worker:
    model: "gpt-5.1-mini"
    temperature: 0.2
    prompt: "workers/trajectory-reader.md"    # File-based worker role
    tools:
      - read_trajectory
      - get_checkpoint_history
      - vault_read
      - vault_search
      - vault_list
      - vault_backlinks
      - vault_traverse
    execution:
      max_steps: 20
      timeout: 120
```

---

## Migration from memory_extraction

### Code Changes

| Component | Change |
|-----------|--------|
| `scenarios/memory_extraction/` | Rename to `scenarios/trajectory_analysis/` |
| `MemoryExtractionStrategy` | Replace with `TrajectoryAnalysisStrategy` |
| `MemoryExtractionState` | Replace with `TrajectoryAnalysisState` |
| `memory_extraction/enums.py` | Remove (`KnowledgeCategory`, `KnowledgeConfidence` — purely prompt-guided via skill content) |
| `memory_extraction/data.py` | Remove (`KnowledgeEntry`, `KnowledgeEvidence` — purely prompt-guided, no state-level types) |
| `memory_extraction/config.py` | Remove (`MemoryFeatureGates` — no skill-specific feature gates needed) |
| `memory_extraction/formatters.py` | Replace with generic catalog/skill formatter in strategy |
| `memory_extraction/answer_schemas.py` | Replace with single generic `AnalyzeAnswer` |
| `memory_extraction/output.py` | Replace with generic `AnalysisReport` |
| `scenarios/__init__.py` | Update `discover()`: `register_mem()` → `register_ta()` |
| `config/scenarios/memory_extraction/` | Rename to `config/scenarios/trajectory_analysis/` |
| `config/scenarios/.../prompts/orchestrator_system.j2` | Replace with generic skeleton |
| `config/scenarios/.../prompts/task_types/extract.j2` | Remove (replaced by `workers/trajectory-reader.md`) |

### Content Migration

The current 145-line `orchestrator_system.j2` becomes the `memory-extraction` skill:

| Current location | New location |
|-----------------|-------------|
| `orchestrator_system.j2` § purpose, strategy, discipline | `vault/skill/trajectory-analysis/memory-extraction/SKILL.md` |
| `orchestrator_system.j2` § what_to_extract | `vault/.../references/what-to-extract.md` |
| `orchestrator_system.j2` § entry_quality | `vault/.../references/entry-quality.md` |
| `orchestrator_system.j2` § entry_categories | `vault/.../references/categories.md` |
| `orchestrator_system.j2` § confidence_levels | `vault/.../references/confidence-levels.md` |
| `orchestrator_system.j2` § entry_format | `vault/.../references/entry-format.md` |
| `worker extract.j2` § what_makes_a_good_entry | `vault/.../references/entry-quality.md` (merged) |
| `worker extract.j2` § discipline | `vault/.../SKILL.md` gotchas section |
| `worker extract.j2` (base role) | `config/.../workers/trajectory-reader.md` (generic, shared by all skills) |

### Registry Changes

| Registry | Before | After |
|----------|--------|-------|
| `STATE_SCHEMAS` | `"memory_extraction"` → `MemoryExtractionState` | `"trajectory_analysis"` → `TrajectoryAnalysisState` |
| `_STRATEGY_INSTANCES` | `"memory_extraction"` → `MemoryExtractionStrategy()` | `"trajectory_analysis"` → `TrajectoryAnalysisStrategy()` |
| `ANSWER_SCHEMA` | `collect`, `analyze`, `extract`, `refine` | `analyze` (single generic worker type) |
| `OUTPUT_SCHEMAS` | `KnowledgeSummary` | `AnalysisReport` |

---

## Constraints and Decisions

| Decision | Rationale | Alternative |
|----------|-----------|-------------|
| Skills in vault, not config dir | Vault already has search, list, read tools; orchestrator can discover and load skills without new infrastructure | Separate skills/ directory with dedicated file-read tool |
| Generic `list[dict]` for findings | Avoids per-skill schema proliferation; skill prompt guides structure | Per-skill Pydantic schema registered in OUTPUT_SCHEMAS |
| Two phases (analyze/synthesize) | Skill determines detailed workflow; phases are markers, not constraints | Keep 4 phases from memory_extraction (over-structured for generic use) |
| Model-driven activation as default | More flexible; matches Agent Skills standard pattern | Config-driven only (simpler but rigid) |
| Single worker task_type ("analyze") | Orchestrator dispatch instruction carries skill-specific lens | Per-skill task_type with separate prompts (more config overhead) |
| SKILL.md < 500 lines | Agent Skills spec recommendation; forces progressive disclosure | No limit (risk of context bloat) |
| File-based worker role (`workers/trajectory-reader.md`) | Follows Claude Code subagent pattern; self-documenting; metadata + instructions in one file | Inline j2 template (less self-documenting) |
| All data types prompt-guided | `KnowledgeEntry`, `KnowledgeEvidence` etc. defined in skill content, not Python code. LLM follows the skill's output format instructions. | Keep Pydantic/dataclass types for validation (adds code per skill) |

---

## Open Questions

- [ ] Should `KnowledgeSummary` be preserved as an alternate output schema alongside `AnalysisReport` for backward compatibility?

---

## Related Concepts

- [System Architecture](system-design-overview.md) — Overall multi-agent framework
- [Generic SDK Wrapper](generic-state-wrapper.md) — ReasoningStrategy, state registry, builder
- [Memory Vault](memory-vault.md) — Vault infrastructure that hosts skills and knowledge
- [Builder](builder.md) — AgentSystemBuilder that compiles the graph
- [Orchestrator](orchestrator.md) — Node-based orchestrator that executes the analysis
