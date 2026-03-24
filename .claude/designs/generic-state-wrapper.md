# Design: Generic SDK Wrapper

**Status**: DRAFT
**Last Updated**: 2026-03-08

---

> **Code conventions**: Same as [orchestrator.md](orchestrator.md#code-conventions-in-this-document). Data structures (`dataclass`, `Enum`, `TypedDict`) are **normative**; function logic is **illustrative**.

## Overview

The SDK Wrapper is the **implementation framework** underlying all AgentM agent systems. The core insight:

> Different agent systems (hypothesis-driven RCA, sequential diagnosis, memory extraction, etc.) share the same Supervisor + Subgraph architecture. The differences are only in **state schema**, **phase definitions**, and **configuration**. A generic framework supports all of them with zero code changes — only config + prompt files.

```
Generic SDK Wrapper (shared code)
├─ AgentSystemBuilder      — Unified build() interface
├─ State Schema Registry   — Concrete state classes per system type
├─ TaskManager             — Async Sub-Agent lifecycle (shared by all)
├─ Configuration System    — YAML loading, validation
└─ Two architecture modes:
   ├─ ReAct mode (create_react_agent)
   │   └─ For: hypothesis_driven (RCA)
   │   └─ Phases: Notebook state markers, LLM-driven
   └─ StateGraph mode (custom graph with phase nodes)
       └─ For: memory_extraction, sequential, decision_tree
       └─ Phases: Explicit nodes, PhaseManager-driven
```

---

## State Schema

### Base State (shared by all systems)

```python
from typing import TypedDict, Annotated
import operator

class BaseExecutorState(TypedDict):
    """Fields shared by all agent systems."""
    messages: Annotated[list[BaseMessage], add_messages]
    task_id: str
    task_description: str
    current_phase: str
```

### System-Specific States

```python
class HypothesisDrivenState(BaseExecutorState):
    """Hypothesis-driven RCA state."""
    notebook: DiagnosticNotebook
    current_hypothesis: Optional[str]

class SequentialDiagnosisState(BaseExecutorState):
    """Sequential step-by-step diagnosis state."""
    steps: Annotated[list[dict], operator.add]
    current_step_index: int

class MemoryExtractionState(BaseExecutorState):
    """Cross-task knowledge extraction state."""
    source_trajectories: list[str]          # Thread IDs of completed RCA tasks
    extracted_patterns: Annotated[list[dict], operator.add]  # Patterns found by Sub-Agents
    knowledge_entries: list[KnowledgeEntry]  # Final refined knowledge
    existing_knowledge: list[KnowledgeEntry] # Current knowledge base (for dedup/refine)

class DecisionTreeState(BaseExecutorState):
    """Decision tree classification state."""
    decision_path: list[str]
    current_node_id: str
    feature_values: dict[str, Any]
```

### State Schema Registry

Each system type maps to its concrete State class. The registry is a simple lookup — no dynamic TypedDict creation needed.

```python
STATE_SCHEMAS: dict[str, type] = {
    "hypothesis_driven": HypothesisDrivenState,
    "sequential": SequentialDiagnosisState,
    "memory_extraction": MemoryExtractionState,
    "decision_tree": DecisionTreeState,
}

def get_state_schema(system_type: str) -> type:
    """Look up the state schema for a system type."""
    if system_type not in STATE_SCHEMAS:
        raise ValueError(f"Unknown system type: {system_type}. Available: {list(STATE_SCHEMAS.keys())}")
    return STATE_SCHEMAS[system_type]
```

To add a new system type: define a new State class inheriting from `BaseExecutorState`, then register it in `STATE_SCHEMAS`.

---

## Phase Management

```python
@dataclass
class PhaseDefinition:
    name: str
    description: str
    handler: Callable
    next_phases: list[str]
    on_enter: Optional[Callable] = None   # e.g., compress previous phase
    on_exit: Optional[Callable] = None

class PhaseManager:
    def __init__(self, phases: dict[str, PhaseDefinition], initial_phase: str):
        self.phases = phases
        self.current_phase = initial_phase

    def transition_to(self, next_phase: str):
        if next_phase not in self.phases[self.current_phase].next_phases:
            raise ValueError(f"Invalid transition: {self.current_phase} → {next_phase}")
        self.current_phase = next_phase

    @classmethod
    def from_config(cls, config: dict) -> "PhaseManager":
        """Build PhaseManager from YAML config."""
        ...
```

### Phase Definitions by System Type

| System | Phases | Flow |
|--------|--------|------|
| **hypothesis_driven** | exploration → generation → verification → confirmation | Loops on verification; may cycle back |
| **sequential** | step_1 → step_2 → ... → step_N | Linear, no branching |
| **memory_extraction** | collect → analyze → extract → refine | See [Memory Extraction System](#memory-extraction-system) |
| **decision_tree** | node evaluation → branch → node evaluation → ... → leaf | Tree traversal |

---

## Base Orchestrator

> **Note**: BaseOrchestrator and PhaseManager are used by **StateGraph-based systems** (linear scenarios like memory_extraction, sequential, decision_tree). **ReAct-based systems** (like hypothesis_driven RCA) use `create_react_agent` directly — they don't use BaseOrchestrator or PhaseManager. Phase management in ReAct systems happens through the LLM's own reasoning, with phases recorded as Notebook state markers.

```python
class BaseOrchestrator(ABC):
    """Shared Orchestrator logic for all systems."""

    def __init__(self, state_schema, phase_manager, sub_agents, config):
        self.state_schema = state_schema
        self.phase_manager = phase_manager
        self.sub_agents = sub_agents
        self.config = config

    @abstractmethod
    async def _decide_next_phase(self, state: dict) -> str:
        """System-specific phase transition logic."""
        pass

    async def dispatch_to_agent(self, agent_id: str, task: dict) -> dict:
        """Dispatch task to Sub-Agent (shared)."""
        return await self.sub_agents[agent_id].ainvoke(task)
```

### System-Specific Orchestrators

```python
class HypothesisDrivenOrchestrator(BaseOrchestrator):
    async def _decide_next_phase(self, state):
        notebook = state["notebook"]
        if notebook.confirmed_hypothesis:
            return "confirmation"
        return "hypothesis_verification"

class MemoryExtractionOrchestrator(BaseOrchestrator):
    async def _decide_next_phase(self, state):
        if not state["extracted_patterns"]:
            return "analyze"  # Still analyzing trajectories
        if not state["knowledge_entries"]:
            return "extract"  # Patterns found, need to synthesize
        return "refine"       # Refine against existing knowledge
```

---

## Memory Extraction System

> **⚠️ SUPERSEDED**: This section describes the original `memory_extraction` scenario design. It has been replaced by the [Trajectory Analysis](trajectory-analysis.md) scenario — a skill-driven, general-purpose trajectory analysis framework where memory extraction is one of many pluggable skills. The content below is retained for historical reference only.

The Memory Extraction system is a concrete agent system built on the same SDK Wrapper. It processes completed RCA trajectories to build a cross-task knowledge base.

### Architecture

```
Memory Extraction System (system_type: "memory_extraction")
│
├─ Orchestrator (MemoryExtractionOrchestrator)
│   ├─ Drives 4 phases: collect → analyze → extract → refine
│   ├─ Feature gates: most RCA-specific gates OFF
│   └─ Tools: search_knowledge_store, update_knowledge_store
│
├─ Sub-Agents
│   ├─ trajectory_analyst    — Read and analyze historical trajectories
│   ├─ pattern_extractor     — Identify failure patterns across trajectories
│   └─ knowledge_writer      — Write/update knowledge entries
│
└─ Knowledge Store (LangGraph Store)
    ├─ failure_patterns      — Common failure modes
    ├─ diagnostic_skills     — Effective diagnostic strategies
    └─ system_knowledge      — Domain-specific facts
```

### Comparison with RCA System

| Aspect | hypothesis_driven (RCA) | memory_extraction (Memory) |
|--------|------------------------|---------------------------|
| **Architecture** | ReAct (create_react_agent) | StateGraph (PhaseManager) |
| **Input** | Incident report / alert | Completed RCA trajectory (thread_id) |
| **Output** | Root cause + recommendations | Knowledge entries (patterns, skills) |
| **State** | DiagnosticNotebook + Hypotheses | Extracted patterns + Knowledge entries |
| **Phases** | explore → hypothesize → verify → confirm | collect → analyze → extract → refine |
| **Sub-Agent role** | Collect live system data | Analyze historical trajectory data |
| **Feature gates** | adversarial_review, parallel_verification, etc. | All RCA-specific gates OFF |
| **Compression** | Active (long tool call sequences) | Typically not needed (reading, not executing) |
| **Shared framework** | BaseOrchestrator, PhaseManager, Send API, Checkpoint, Stream | Same |

### Four-Phase Flow

```
Phase 1: COLLECT
  │  Load target trajectories from checkpoint store
  │  Orchestrator selects which completed RCA threads to analyze
  │  Sub-Agents read trajectory data via get_state_history()
  ↓
Phase 2: ANALYZE
  │  Sub-Agents examine trajectories for patterns
  │  Each Sub-Agent focuses on different aspects:
  │    - trajectory_analyst: execution flow, phase durations, decision points
  │    - pattern_extractor: recurring failures, common root causes
  │  Results aggregated via Annotated[list, operator.add]
  ↓
Phase 3: EXTRACT
  │  Orchestrator synthesizes Sub-Agent findings into KnowledgeEntry objects
  │  Deduplication against existing_knowledge
  │  Categorization: failure_pattern | diagnostic_skill | system_knowledge
  ↓
Phase 4: REFINE
  │  Compare with existing knowledge base
  │  Update existing entries (add evidence, refine descriptions)
  │  or create new entries
  │  Write to Knowledge Store
```

### Knowledge Data Structures

```python
from enum import Enum


class KnowledgeCategory(str, Enum):
    """Top-level classification of knowledge entries."""
    FAILURE_PATTERN = "failure_pattern"
    DIAGNOSTIC_SKILL = "diagnostic_skill"
    SYSTEM_KNOWLEDGE = "system_knowledge"


class KnowledgeConfidence(str, Enum):
    """Confidence level of a knowledge entry, determined by evidence strength.

    Memory Agent assigns this during extraction based on how the knowledge
    was established:
    - FACT: Verified causal relationship (hypothesis confirmed through RCA)
    - PATTERN: Observed correlation across multiple trajectories (statistical, not causal)
    - HEURISTIC: Inferred strategy or rule of thumb from diagnostic experience
    """
    FACT = "fact"            # Confirmed causal relationship, e.g., "pool full → API timeout"
    PATTERN = "pattern"      # Observed correlation, e.g., "CPU >80% often accompanies slow queries"
    HEURISTIC = "heuristic"  # Empirical strategy, e.g., "check connection pool before disk I/O"


@dataclass
class KnowledgeEntry:
    id: str
    path: str                         # Full namespace path, e.g., "failure_pattern/database/connection_pool_exhaustion"
    category: KnowledgeCategory
    confidence: KnowledgeConfidence   # How reliable this knowledge is
    domain: str                       # Second-level grouping, e.g., "database", "network", "memory"
    title: str                        # "Database connection pool exhaustion"
    description: str                  # Detailed description
    evidence: list[KnowledgeEvidence] # Source trajectories and supporting data
    tags: list[str]                   # ["database", "connection_pool", "timeout"]
    related_entries: list[str]        # Paths of related entries (e.g., a pattern → its effective skill)
    created_at: str
    updated_at: str
    frequency: int                    # How many times this pattern has been observed

@dataclass
class KnowledgeEvidence:
    source_thread_id: str             # Which RCA task this came from
    source_checkpoint_range: Optional[tuple[str, str]]  # (from_id, to_id)
    relevant_data: dict               # Key data points from that trajectory
    summary: str                      # Brief description of how this evidence supports the entry
```

### Knowledge Confidence Model

Memory Agent assigns a confidence level to each extracted knowledge entry based on how it was established during RCA:

| Confidence | Meaning | Source in Trajectory | Example |
|-----------|---------|---------------------|---------|
| **FACT** | Verified causal relationship | Hypothesis confirmed (Phase 4) with supporting evidence | "When connection pool is 100% full, API response > 5s" |
| **PATTERN** | Observed correlation, not proven causal | Recurring co-occurrence across multiple trajectories | "CPU > 80% is often accompanied by slow query increase" |
| **HEURISTIC** | Effective strategy from experience | Diagnostic path that led to faster resolution | "Check connection pool before disk I/O — faster to confirm/eliminate" |

**How RCA Orchestrator uses confidence**:

| RCA Phase | Preferred Confidence | Rationale |
|-----------|---------------------|-----------|
| Phase 1 (Exploration) | HEURISTIC, PATTERN | Guide which agents to dispatch first; broad relevance |
| Phase 2 (Hypothesis Generation) | FACT, PATTERN | Generate hypotheses from known causes and correlations |
| Phase 3 (Verification) | FACT only | Only trust confirmed causal knowledge when evaluating evidence |

**Confidence promotion**: A PATTERN entry can be promoted to FACT when a new RCA trajectory confirms the causal relationship. The Memory Agent handles this during the REFINE phase by comparing new verification results against existing pattern entries.

### Knowledge Store: Filesystem-Like Namespace

The Knowledge Store uses LangGraph Store as backend, with a **hierarchical namespace** design that mirrors a filesystem. This gives Agents two complementary retrieval modes:

1. **Semantic search** — Fuzzy, similarity-based ("find database-related failures")
2. **Path browsing** — Precise, structure-based ("list what's under `/failure_pattern/database/`")

This is analogous to how humans use a filesystem: search to find approximate location, then `ls` to confirm context, then open the specific file. Agents can cross-validate search results by browsing the structural neighborhood.

#### Namespace Hierarchy

```
knowledge/                              ← root namespace
├── failure_pattern/                    ← category
│   ├── database/                       ← domain
│   │   ├── connection_pool_exhaustion  ← entry (KnowledgeEntry)
│   │   ├── slow_query_lock_contention
│   │   └── replication_lag
│   ├── network/
│   │   ├── dns_resolution_timeout
│   │   └── tcp_connection_reset
│   └── memory/
│       └── gc_pressure_oom
├── diagnostic_skill/
│   ├── database/
│   │   └── connection_pool_analysis
│   └── general/
│       └── timeline_correlation
└── system_knowledge/
    ├── architecture/
    │   └── service_dependency_map
    └── limits/
        └── connection_pool_defaults
```

#### Mapping to LangGraph Store

> **✅ LangGraph Verified**: All Store operations used in this design are natively supported:
> - `store.put(namespace, key, value, index=[...])` — Writes with embedding index ✅
> - `store.get(namespace, key)` — Direct read by path ✅
> - `store.search(namespace_prefix, query, filter, limit)` — Semantic search + structured filter ✅
> - `store.list_namespaces(prefix, max_depth, limit)` — Filesystem-like browsing ✅
> - `store.delete(namespace, key)` — Delete entry ✅
>
> **Note on `list_namespaces` usage**: The `max_depth` parameter truncates namespace paths to the specified depth. For listing immediate children of `("knowledge", "failure_pattern")`, use `max_depth=3` (prefix length + 1). The design's `max_depth=len(namespace) + 1` is correct.

LangGraph Store uses namespace tuples. The path maps directly:

```python
# Path: "failure_pattern/database/connection_pool_exhaustion"
# → namespace: ("knowledge", "failure_pattern", "database")
# → key: "connection_pool_exhaustion"
# → value: KnowledgeEntry as dict

store.put(
    namespace=("knowledge", "failure_pattern", "database"),
    key="connection_pool_exhaustion",
    value=asdict(entry),
    index=["title", "description", "tags"],  # Fields to embed for semantic search
)
```

#### Store Configuration

```python
from langgraph.store.postgres import AsyncPostgresStore

store = AsyncPostgresStore(
    conn_string="postgresql://...",
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
        "fields": ["title", "description"],  # Default fields to embed
    },
)
await store.setup()
```

### Knowledge Categories

| Category | What It Contains | Typical Confidence | How RCA Uses It |
|----------|-----------------|-------------------|----------------|
| **failure_pattern** | Recurring failure modes with symptoms, root causes, and resolution | FACT or PATTERN | Orchestrator searches during Phase 2 (hypothesis generation) to propose hypotheses based on historical precedent |
| **diagnostic_skill** | Effective investigation strategies (which tools to use, in what order, for what symptoms) | HEURISTIC | Orchestrator references during Phase 1 (exploration) and Phase 3 (verification) to optimize investigation |
| **system_knowledge** | Domain facts about the target system (architecture, dependencies, known limits) | FACT | Injected into Orchestrator's system prompt for context |

### Integration with RCA System: Knowledge Tools

The RCA Orchestrator gets three tools for interacting with the Knowledge Store, mirroring filesystem operations:

> **Implementation Note**: Knowledge tools (`knowledge_search`, `knowledge_list`, `knowledge_read`, `knowledge_write`) are implemented but **disabled** in the RCA scenario config. The LangGraph Store backend needs to be wired by the builder before they can be activated. The tool implementations exist in `src/agentm/tools/knowledge.py` but the Store instance is not yet injected at runtime.

```python
@tool
def knowledge_search(
    query: str,
    path: str = "/",
    filter: Optional[dict] = None,
    limit: int = 5,
) -> list[dict]:
    """Semantic search over the knowledge base.

    Use this to find relevant historical patterns, skills, or facts
    based on natural language description. Results are ranked by relevance.

    To narrow the search scope, specify a path prefix.

    Args:
        query: Natural language description of what you're looking for.
            Examples:
            - "database connection timeout under high load"
            - "how to diagnose memory leaks in Java services"
        path: Path prefix to search within. Defaults to "/" (all knowledge).
            Examples:
            - "/failure_pattern/database/" — only database failure patterns
            - "/diagnostic_skill/" — only diagnostic skills
        filter: Optional structured filter.
            Examples:
            - {"frequency": {"$gte": 3}} — patterns observed 3+ times
            - {"tags": {"$eq": "timeout"}}
        limit: Maximum number of results to return.
    """
    namespace_prefix = path_to_namespace(path)
    results = store.search(
        namespace_prefix=namespace_prefix,
        query=query,
        filter=filter,
        limit=limit,
    )
    return [format_search_result(r) for r in results]


@tool
def knowledge_list(path: str = "/") -> dict:
    """List the structure of the knowledge base at a given path.

    Use this to understand what knowledge exists and how it's organized.
    Returns both sub-directories (sub-namespaces) and entries at the path.

    This is useful for:
    - Understanding the overall knowledge structure
    - Browsing a specific domain to see available patterns
    - Cross-validating search results by checking neighboring entries

    Args:
        path: Path to list. Defaults to "/" (root).
            Examples:
            - "/" — show top-level categories
            - "/failure_pattern/" — show all domains under failure patterns
            - "/failure_pattern/database/" — show all database failure patterns
    """
    namespace = path_to_namespace(path)

    # List sub-namespaces (like sub-directories)
    children = store.list_namespaces(
        prefix=namespace,
        max_depth=len(namespace) + 1,  # Only immediate children
    )

    # List entries at this path (like files in the directory)
    entries = store.search(
        namespace_prefix=namespace,
        limit=100,
    )

    return {
        "path": path,
        "sub_paths": [namespace_to_path(ns) for ns in children],
        "entries": [{
            "key": e.key,
            "title": e.value.get("title", ""),
            "confidence": e.value.get("confidence", ""),
            "frequency": e.value.get("frequency", 0),
        } for e in entries],
    }


@tool
def knowledge_read(path: str) -> dict:
    """Read a specific knowledge entry by its full path.

    Use this to get the complete details of a knowledge entry,
    including full description, all evidence, and related entries.

    Args:
        path: Full path to the entry.
            Example: "/failure_pattern/database/connection_pool_exhaustion"
    """
    namespace, key = path_to_namespace_and_key(path)
    item = store.get(namespace=namespace, key=key)
    if item is None:
        raise ToolException(f"Knowledge entry not found: {path}")
    return item.value
```

#### Path Utility Functions

```python
STORE_ROOT = ("knowledge",)

def path_to_namespace(path: str) -> tuple[str, ...]:
    """Convert filesystem-like path to LangGraph Store namespace tuple.
    "/" → ("knowledge",)
    "/failure_pattern/database/" → ("knowledge", "failure_pattern", "database")
    """
    parts = [p for p in path.strip("/").split("/") if p]
    return STORE_ROOT + tuple(parts)

def path_to_namespace_and_key(path: str) -> tuple[tuple[str, ...], str]:
    """Split path into namespace + key.
    "/failure_pattern/database/connection_pool_exhaustion"
    → (("knowledge", "failure_pattern", "database"), "connection_pool_exhaustion")
    """
    parts = [p for p in path.strip("/").split("/") if p]
    return STORE_ROOT + tuple(parts[:-1]), parts[-1]

def namespace_to_path(namespace: tuple[str, ...]) -> str:
    """Convert namespace tuple back to path string.
    ("knowledge", "failure_pattern", "database") → "/failure_pattern/database/"
    """
    parts = namespace[len(STORE_ROOT):]  # Strip root prefix
    return "/" + "/".join(parts) + "/" if parts else "/"
```

#### Write Tools (Memory Agent Only)

The RCA Orchestrator has **read-only** access (search, list, read). The Memory Extraction system's Sub-Agents additionally have **write** access:

```python
@tool
def knowledge_write(
    path: str,
    entry: dict,
    merge: bool = False,
) -> str:
    """Create or update a knowledge entry.

    Args:
        path: Full path for the entry.
            Example: "/failure_pattern/database/connection_pool_exhaustion"
        entry: KnowledgeEntry fields as dict.
        merge: If True and entry exists, merge evidence and increment frequency.
               If False and entry exists, overwrite.
    """
    namespace, key = path_to_namespace_and_key(path)
    if merge:
        existing = store.get(namespace=namespace, key=key)
        if existing:
            entry = merge_entries(existing.value, entry)
    store.put(namespace=namespace, key=key, value=entry,
              index=["title", "description"])
    return f"Written: {path}"


@tool
def knowledge_delete(path: str) -> str:
    """Delete a knowledge entry. Use when an entry is outdated or incorrect.

    Args:
        path: Full path of the entry to delete.
    """
    namespace, key = path_to_namespace_and_key(path)
    store.delete(namespace=namespace, key=key)
    return f"Deleted: {path}"
```

#### Tool Access by System Type

| Tool | RCA Orchestrator | Memory Extraction Sub-Agents |
|------|:---:|:---:|
| `knowledge_search` | read | read |
| `knowledge_list` | read | read |
| `knowledge_read` | read | read |
| `knowledge_write` | — | write |
| `knowledge_delete` | — | write |

#### Agent Workflow Example

```
RCA Orchestrator (Phase 2: Hypothesis Generation)
  │
  ├─ 1. Semantic search with confidence filter:
  │     knowledge_search("API timeout under high load",
  │                      filter={"confidence": {"$eq": "fact"}})
  │     → returns: [{path: "/failure_pattern/database/connection_pool_exhaustion",
  │                  confidence: "fact", score: 0.92, ...}]
  │     # Note: repeat with {"confidence": {"$eq": "pattern"}} if needed.
  │     # LangGraph Store supports $eq, $ne, $gt, $gte, $lt, $lte — but NOT $in.
  │
  ├─ 2. Browse neighborhood: knowledge_list("/failure_pattern/database/")
  │     → returns: {sub_paths: [], entries: [
  │         {key: "connection_pool_exhaustion", title: "...", frequency: 5, confidence: "fact"},
  │         {key: "slow_query_lock_contention", title: "...", frequency: 3, confidence: "pattern"},
  │         {key: "replication_lag", title: "...", frequency: 1, confidence: "pattern"},
  │     ]}
  │     → Agent sees: "connection_pool_exhaustion" is a confirmed FACT (highest confidence),
  │       other patterns are correlations worth considering
  │
  ├─ 3. Read full detail: knowledge_read("/failure_pattern/database/connection_pool_exhaustion")
  │     → returns full KnowledgeEntry with evidence, confidence, related_entries, etc.
  │
  ├─ 4. Follow related: knowledge_read("/diagnostic_skill/database/connection_pool_analysis")
  │     → returns the recommended diagnostic approach (HEURISTIC confidence)
  │
  └─ 5. Generate hypothesis with historical context and confidence:
        "H1: Connection pool exhaustion (FACT, 5 precedents, diagnostic skill available)"
```

### Trigger Modes

| Mode | When | Description |
|------|------|-------------|
| **Post-task** | After each RCA task completes | Automatically process the latest trajectory |
| **Batch** | Scheduled (e.g., daily) | Process multiple recent trajectories at once |
| **Manual** | On-demand | User triggers analysis of specific trajectories |

### Configuration

```yaml
# scenarios/memory_extraction/scenario.yaml
system:
  type: "memory_extraction"

orchestrator:
  model: "gpt-4o"
  temperature: 0.3
  orchestrator_mode: "graph"

  prompts:
    system: "prompts/orchestrator_system.j2"

  feature_gates:
    adversarial_review: false
    parallel_verification: false
    auto_refine_partial: false
    dedup_against_existing: true
    auto_merge_similar: true
    min_evidence_for_pattern: 2

  tools:
    - knowledge_search
    - knowledge_list
    - knowledge_read
    - knowledge_write
    - knowledge_delete

  compression:
    enabled: false

phases:
  collect:
    handler: "phase_collect_trajectories"
    next_phases: [analyze]
  analyze:
    handler: "phase_analyze_trajectories"
    next_phases: [analyze, extract]
  extract:
    handler: "phase_extract_knowledge"
    next_phases: [refine]
  refine:
    handler: "phase_refine_knowledge"
    next_phases: []

agents:
  trajectory_analyst:
    model: "gpt-4o-mini"
    prompt: "prompts/agents/trajectory_analyst.j2"
    tools:
      - read_trajectory
      - get_checkpoint_history
      - knowledge_search
      - knowledge_list
  pattern_extractor:
    model: "gpt-4o"
    prompt: "prompts/agents/pattern_extractor.j2"
    tools:
      - knowledge_search
      - knowledge_list
      - knowledge_read
      - compare_trajectories
  knowledge_writer:
    model: "gpt-4o"
    prompt: "prompts/agents/knowledge_writer.j2"
    tools:
      - knowledge_search
      - knowledge_list
      - knowledge_read
      - knowledge_write
      - knowledge_delete

```

> **Note**: Knowledge Store configuration (backend, embedding, index) is in `system.yaml` under `storage.store`, not in the scenario file. The scenario only references knowledge tools — the store itself is global infrastructure.

### Usage

```python
# Build Memory Extraction system — same API as RCA
memory_system = AgentSystemBuilder.build(
    system_type="memory_extraction",
    config=load_yaml("config/scenarios/memory_extraction.yaml"),
)

# Post-task: process a completed RCA trajectory
result = await memory_system.execute({
    "task": "Extract knowledge from completed RCA",
    "source_trajectories": ["thread-rca-001", "thread-rca-002"],
})

# The knowledge store is now updated and available to future RCA tasks
# via the search_knowledge tool
```

---

## AgentSystemBuilder

> **Implementation Note**: Only `react` mode (via `create_react_agent`) is currently implemented. Other modes (`graph` for StateGraph-based systems like memory_extraction, sequential, decision_tree) raise `NotImplementedError`. The `AgentSystemBuilder.build()` method validates `orchestrator_mode` and delegates to `build_react_system()` for `react` mode.

```python
class AgentSystemBuilder:
    """Unified entry point for building any agent system.

    Internally selects the appropriate architecture based on system_type:
    - ReAct-based (create_react_agent): For exploratory, non-linear scenarios like RCA
    - StateGraph-based (custom graph with phase nodes): For linear, deterministic scenarios like Memory Extraction

    External callers use the same interface regardless of internal architecture.
    """

    @staticmethod
    def build(system_type: str, config: ScenarioConfig) -> AgentSystem:
        state_schema = get_state_schema(system_type)

        # Select architecture based on system type characteristics
        if config.orchestrator_mode == "react":
            # RCA, exploratory scenarios: LLM freely interleaves phases
            return build_react_system(state_schema, config)
        else:
            # Memory extraction, sequential: explicit phase nodes
            return build_graph_system(state_schema, config)

class AgentSystem:
    """Unified interface for all agent systems."""
    async def execute(self, input_data: dict) -> dict: ...
    async def stream(self, input_data: dict): ...
```

---

## Building a New System

Adding a new agent system type requires:

1. **State class** — Define a new `TypedDict` inheriting from `BaseExecutorState`, register in `STATE_SCHEMAS`
2. **Architecture mode** — Choose `react` (exploratory) or `graph` (linear) in scenario.yaml `orchestrator_mode`
3. **YAML config** — Define agents, tools, prompts, feature gates
4. **If StateGraph mode**: Define phase handlers and register with PhaseManager
5. **If ReAct mode**: Design the system prompt with diagnostic methodology guidelines

---

## Related Documents

- [System Architecture](system-design-overview.md) — Overall system design
- [Orchestrator](orchestrator.md) — Orchestrator design, hypothesis flow, compression, recall
- [Sub-Agent](sub-agent.md) — Sub-Agent architecture and configuration
