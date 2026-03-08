# Design: Generic SDK Wrapper

**Status**: DRAFT
**Last Updated**: 2026-03-08

---

## Overview

The SDK Wrapper is the **implementation framework** underlying all AgentM agent systems. The core insight:

> Different agent systems (hypothesis-driven RCA, sequential diagnosis, memory extraction, etc.) share the same Supervisor + Subgraph architecture. The differences are only in **state schema**, **phase definitions**, and **configuration**. A generic framework supports all of them with zero code changes — only config + prompt files.

```
Generic SDK Wrapper (shared code)
├─ StateSchemaFactory     — Dynamic state schema generation
├─ PhaseManager           — Phase lifecycle and transitions
├─ BaseOrchestrator       — Supervisor routing, Sub-Agent dispatch
├─ Trajectory Recording   — Checkpoint, streaming, export
└─ Configuration System   — YAML loading, validation, hot-reload
         ↓
┌────────────────────────────────────────────────────────┐
│  Concrete Agent Systems (config-driven)                 │
├────────────────────────────────────────────────────────┤
│                                                         │
│  hypothesis_driven     — RCA with hypothesis reasoning  │
│  sequential            — Step-by-step diagnosis         │
│  memory_extraction     — Cross-task knowledge building  │
│  decision_tree         — Classification-based diagnosis │
│  custom                — User-defined system            │
│                                                         │
└────────────────────────────────────────────────────────┘
```

---

## State Schema

### Base State (shared by all systems)

```python
from typing import TypedDict, Annotated
import operator

class BaseExecutorState(TypedDict):
    """Fields shared by all agent systems."""
    messages: Annotated[list, operator.add]
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

### StateSchemaFactory

```python
class StateSchemaFactory:
    """Generate state schemas from config."""

    SYSTEM_FIELDS = {
        "hypothesis_driven": {
            "notebook": DiagnosticNotebook,
            "current_hypothesis": Optional[str],
        },
        "sequential": {
            "steps": Annotated[list[dict], operator.add],
            "current_step_index": int,
        },
        "memory_extraction": {
            "source_trajectories": list[str],
            "extracted_patterns": Annotated[list[dict], operator.add],
            "knowledge_entries": list[KnowledgeEntry],
            "existing_knowledge": list[KnowledgeEntry],
        },
        "decision_tree": {
            "decision_path": list[str],
            "current_node_id": str,
            "feature_values": dict,
        },
    }

    @staticmethod
    def create(system_type: str, custom_fields: dict = None) -> type:
        base = {"messages": Annotated[list, operator.add],
                "task_id": str, "task_description": str, "current_phase": str}
        system = StateSchemaFactory.SYSTEM_FIELDS.get(system_type, {})
        all_fields = {**base, **system, **(custom_fields or {})}
        return TypedDict("ExecutorState", all_fields)
```

---

## Phase Management

```python
@dataclass
class Phase:
    name: str
    description: str
    handler: Callable
    next_phases: list[str]
    on_enter: Optional[Callable] = None   # e.g., compress previous phase
    on_exit: Optional[Callable] = None

class PhaseManager:
    def __init__(self, phases: dict[str, Phase], initial_phase: str):
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
@dataclass
class KnowledgeEntry:
    id: str
    path: str                         # Full namespace path, e.g., "failure_pattern/database/connection_pool_exhaustion"
    category: Literal["failure_pattern", "diagnostic_skill", "system_knowledge"]
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

| Category | What It Contains | How RCA Uses It |
|----------|-----------------|----------------|
| **failure_pattern** | Recurring failure modes with symptoms, root causes, and resolution | Orchestrator searches during Phase 2 (hypothesis generation) to propose hypotheses based on historical precedent |
| **diagnostic_skill** | Effective investigation strategies (which tools to use, in what order, for what symptoms) | Orchestrator references during Phase 1 (exploration) and Phase 3 (verification) to optimize investigation |
| **system_knowledge** | Domain facts about the target system (architecture, dependencies, known limits) | Injected into Orchestrator's system prompt for context |

### Integration with RCA System: Knowledge Tools

The RCA Orchestrator gets three tools for interacting with the Knowledge Store, mirroring filesystem operations:

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
        "entries": [{"key": e.key, "title": e.value.get("title", ""), "frequency": e.value.get("frequency", 0)} for e in entries],
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
  ├─ 1. Semantic search: knowledge_search("API timeout under high load")
  │     → returns: [{path: "/failure_pattern/database/connection_pool_exhaustion", score: 0.92, ...},
  │                 {path: "/failure_pattern/network/dns_resolution_timeout", score: 0.71, ...}]
  │
  ├─ 2. Browse neighborhood: knowledge_list("/failure_pattern/database/")
  │     → returns: {sub_paths: [], entries: [
  │         {key: "connection_pool_exhaustion", title: "...", frequency: 5},
  │         {key: "slow_query_lock_contention", title: "...", frequency: 3},
  │         {key: "replication_lag", title: "...", frequency: 1},
  │     ]}
  │     → Agent sees: "connection_pool_exhaustion" is the most frequent database pattern,
  │       and there are other database patterns to consider
  │
  ├─ 3. Read full detail: knowledge_read("/failure_pattern/database/connection_pool_exhaustion")
  │     → returns full KnowledgeEntry with evidence, related_entries, etc.
  │
  ├─ 4. Follow related: knowledge_read("/diagnostic_skill/database/connection_pool_analysis")
  │     → returns the recommended diagnostic approach for this pattern
  │
  └─ 5. Generate hypothesis with historical context:
        "H1: Connection pool exhaustion (5 historical precedents, recommended skill available)"
```

### Trigger Modes

| Mode | When | Description |
|------|------|-------------|
| **Post-task** | After each RCA task completes | Automatically process the latest trajectory |
| **Batch** | Scheduled (e.g., daily) | Process multiple recent trajectories at once |
| **Manual** | On-demand | User triggers analysis of specific trajectories |

### Configuration

```yaml
# config/scenarios/memory_extraction.yaml
system:
  type: "memory_extraction"

orchestrator:
  model: "gpt-4o"
  temperature: 0.3

  feature_gates:
    # RCA-specific gates — all OFF
    adversarial_review: false
    parallel_verification: false
    auto_refine_partial: false
    # Memory-specific gates
    dedup_against_existing: true
    auto_merge_similar: true
    min_evidence_for_pattern: 2   # Require N occurrences before creating a pattern

  tools:
    - search_knowledge_store
    - update_knowledge_store

  compression:
    enabled: false    # Memory extraction tasks are typically short

phases:
  collect:
    handler: "phase_collect_trajectories"
    next_phases: [analyze]
  analyze:
    handler: "phase_analyze_trajectories"
    next_phases: [analyze, extract]   # May loop for more analysis
  extract:
    handler: "phase_extract_knowledge"
    next_phases: [refine]
  refine:
    handler: "phase_refine_knowledge"
    next_phases: []

sub_agents:
  trajectory_analyst:
    model: "gpt-4o-mini"
    tools:
      - read_trajectory          # Read checkpoint history from completed RCA threads
      - get_checkpoint_history
      - knowledge_search          # Search existing knowledge (read-only)
      - knowledge_list
  pattern_extractor:
    model: "gpt-4o"
    tools:
      - knowledge_search          # Search for dedup / cross-reference
      - knowledge_list            # Browse structure
      - knowledge_read            # Read full entries
      - compare_trajectories
  knowledge_writer:
    model: "gpt-4o"
    tools:
      - knowledge_search          # Check before writing
      - knowledge_list
      - knowledge_read
      - knowledge_write           # Create / update entries
      - knowledge_delete          # Remove obsolete entries

knowledge_store:
  backend: "langgraph_store"
  namespace_root: "knowledge"
  persistence: "postgres"           # "memory" | "sqlite" | "postgres"
  index:
    dims: 1536
    embed: "openai:text-embedding-3-small"
    fields: ["title", "description"]  # Fields to embed for semantic search
```

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

## Building a New System

Adding a new agent system type requires:

1. **YAML config** — Define phases, agents, feature gates
2. **State fields** — Add system-specific fields to `StateSchemaFactory.SYSTEM_FIELDS`
3. **Phase handlers** — Implement the phase functions
4. **Orchestrator subclass** — Implement `_decide_next_phase`

No changes to the core framework (StateSchemaFactory, PhaseManager, BaseOrchestrator).

---

## Related Documents

- [System Architecture](system-design-overview.md) — Overall system design
- [Orchestrator](orchestrator.md) — Orchestrator design, hypothesis flow, compression, recall
- [Sub-Agent](sub-agent.md) — Sub-Agent architecture and configuration
