# Library Anti-Pattern Fixes

> Status: **Draft — pending review**
> Date: 2026-03-15
> Goal: Make AgentM usable as a Python library (not just a CLI tool)

## Problem Summary

AgentM currently can only be used via its CLI entry point. Multiple architectural patterns prevent programmatic library usage: module-level global state, `sys.exit()` in core paths, hardcoded paths/models, no custom exceptions, and Rich-based output instead of logging.

---

## Fix 1 — P0: Eliminate Module-Level Global State

### 1a. `knowledge.py` → `KnowledgeStore` class

**Current**: 7 module-level variables (`_base_dir`, `_entries`, `_inv_index`, `_embeddings`, `_model`, `_model_load_failed`, `_lock`) with an `init()` function that resets them all via `global`.

**Problem**: Only one knowledge store per process. Two library users in the same process collide.

**Fix**:

```python
# src/agentm/tools/knowledge.py

class KnowledgeStore:
    """File-system knowledge backend with inverted index + hybrid search."""

    def __init__(self, base_dir: str = "./knowledge") -> None:
        self._base_dir = Path(base_dir)
        self._entries: dict[str, dict] = {}
        self._inv_index: dict[str, set[str]] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._model: Any = None
        self._model_load_failed: bool = False
        self._lock = threading.Lock()
        self._init_store()

    def _init_store(self) -> None:
        """Scan existing files and build indexes."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_json()
        # ... (same scan logic as current init(), but using self.*)

    def write(self, path: str, entry: dict[str, Any]) -> str: ...
    def read(self, path: str) -> str: ...
    def delete(self, path: str) -> str: ...
    def list(self, request: str, path: str = "/", depth: int = 1) -> str: ...
    def search(self, query: str, path: str = "/", limit: int = 5, mode: str = "hybrid") -> str: ...
```

**Backward compatibility**: Keep module-level functions as thin wrappers around a default instance:

```python
# Module-level convenience (backward compat for CLI)
_default_store: KnowledgeStore | None = None

def init(base_dir: str = "./knowledge") -> None:
    global _default_store
    _default_store = KnowledgeStore(base_dir)

def knowledge_write(path: str, entry: dict[str, Any]) -> str:
    if _default_store is None:
        raise RuntimeError("Knowledge store not initialized — call init() first")
    return _default_store.write(path, entry)

# ... same pattern for read, delete, list, search
```

**Builder change** (`builder.py:342`):

```python
# Before
knowledge_module.init(base_dir="./knowledge")

# After — store instance is created and passed through
knowledge_store = KnowledgeStore(base_dir="./knowledge")
# ... pass to tool closures
```

**Files changed**: `src/agentm/tools/knowledge.py`, `src/agentm/builder.py`

---

### 1b. `memory.py` → `MemoryStore` class

**Current**: Module-level `_db_path` set by builder, `_get_conn()` creates new connection each time.

**Fix**:

```python
# src/agentm/tools/memory.py

class MemoryStore:
    """Read-only access to trajectory checkpoints via SQLite."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def read_trajectory(self, thread_id: str) -> str: ...
    def get_checkpoint_history(self, thread_id: str, limit: int = 50) -> str: ...
```

**Backward compat**: Same pattern — keep `set_db_path()` and module-level functions as wrappers.

**Files changed**: `src/agentm/tools/memory.py`, `src/agentm/builder.py`

---

### 1c. `server/app.py` → Scoped WebSocket clients

**Current**: `_websocket_clients: set[WebSocket] = set()` at module level, mutated without any synchronization.

**Fix**: Move `_websocket_clients` into `app.state` inside `create_dashboard_app()`:

```python
def create_dashboard_app(...) -> FastAPI:
    app = FastAPI(title="AgentM Dashboard")
    app.state.websocket_clients: set[WebSocket] = set()
    # ...
```

Update `broadcast_event` to accept `app` or the clients set as argument, or make it a method on a small `Broadcaster` class:

```python
class Broadcaster:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    def add(self, ws: WebSocket) -> None:
        self._clients.add(ws)

    def remove(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def broadcast(self, event: dict[str, Any]) -> None:
        if "timestamp" not in event:
            event["timestamp"] = datetime.now().isoformat()
        if "mode" not in event:
            event["mode"] = "updates"
        # ... serialize + send to self._clients
```

This also fixes the iteration-during-mutation race condition.

**Files changed**: `src/agentm/server/app.py`, `src/agentm/cli/run.py` (adjust `broadcast_event` references)

---

## Fix 2 — P0: Remove `sys.exit()` from Non-CLI Code

**Current locations** (`cli/run.py`):
- Line 74: `sys.exit(1)` on data init failure
- Line 297: `sys.exit(1)` on missing thread_id
- Line 495: `sys.exit(1)` on trajectory file not found
- Line 506: `sys.exit(1)` on missing thread_id in trajectory
- Line 539: `sys.exit(1)` on data init failure (resume)
- Line 569: `sys.exit(1)` on checkpoint read error
- Line 577: `sys.exit(1)` on no checkpoints found
- Line 596: `sys.exit(1)` on invalid checkpoint index

**Fix**: Replace all `sys.exit(1)` calls with custom exceptions (see Fix 3). The only place `sys.exit` should appear is in `cli/main.py`:

```python
# cli/main.py
def main() -> None:
    load_dotenv()
    try:
        app()
    except AgentMError as e:
        console.print(f"[red]ERROR: {e}[/]")
        sys.exit(1)
```

In `cli/run.py`, replace patterns like:

```python
# Before
if "error" in init_info:
    console.print(f"[red]ERROR: {init_info['error']}[/]")
    sys.exit(1)

# After
if "error" in init_info:
    raise DataInitError(init_info["error"])
```

**Files changed**: `src/agentm/cli/run.py`, `src/agentm/cli/main.py`

---

## Fix 3 — P1: Custom Exception Hierarchy

**Current**: All exceptions are built-in (`ValueError`, `KeyError`, `RuntimeError`). Library users cannot selectively catch AgentM errors.

**Fix**: Create `src/agentm/exceptions.py`:

```python
class AgentMError(Exception):
    """Base exception for all AgentM errors."""

class ConfigError(AgentMError):
    """Configuration loading or validation error."""

class DataInitError(AgentMError):
    """Data directory initialization failed."""

class ToolError(AgentMError):
    """Tool execution error."""

class AgentError(AgentMError):
    """Agent execution error."""

class CheckpointError(AgentMError):
    """Checkpoint read/write error."""

class StoreNotInitializedError(AgentMError):
    """A store (knowledge/memory) was used before initialization."""
```

**Migration**: Replace key `raise` sites:
- `knowledge.py:246` `RuntimeError` → `StoreNotInitializedError`
- `config/loader.py:23` `KeyError` → `ConfigError`
- `builder.py:406` `ValueError` → `ConfigError`
- `cli/run.py` all `sys.exit(1)` → appropriate exception subclass

**Files changed**: New `src/agentm/exceptions.py`, plus `knowledge.py`, `memory.py`, `builder.py`, `config/loader.py`, `cli/run.py`

---

## Fix 4 — P1: Externalize Hardcoded Paths and Model Names

### 4a. `TOOLS_DIR` in `builder.py`

**Current** (`builder.py:242`):
```python
TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "tools"
```
Assumes project directory structure — breaks when installed as a library.

**Fix**: Accept `tools_dir` as a parameter to `AgentSystemBuilder.build()`:

```python
@staticmethod
def build(
    system_type: str,
    scenario_config: ScenarioConfig,
    system_config: Any | None = None,
    existing_thread_id: str | None = None,
    tools_dir: Path | str | None = None,   # NEW
) -> AgentSystem:
    if tools_dir is None:
        tools_dir = Path(__file__).resolve().parent.parent.parent / "config" / "tools"
    tools_dir = Path(tools_dir)
    # ...
```

This preserves backward compat (default behavior unchanged) while allowing library users to specify their own tool definitions directory.

### 4b. Hardcoded model names in `compression.py`

**Current**:
- `count_tokens()` defaults to `model="gpt-5.1"` (line 46)
- `_summarize_messages()` defaults to `model="gpt-5.1-mini"` (line 89)
- `sub_agent_compression_hook()` uses these defaults (line 188)

**Fix**: The `build_compression_hook` already takes `CompressionConfig` which has `compression_model`. The problem is the standalone `sub_agent_compression_hook` function and the defaults in `count_tokens` / `_summarize_messages`.

```python
# Change defaults to be more generic
def count_tokens(messages: list[Any], model: str = "gpt-4o") -> int:
    """Count tokens. Model is used for tokenizer selection only."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    # ...
```

For `sub_agent_compression_hook` — it's a standalone function that doesn't take config. Since `build_compression_hook` (which takes config) exists and is the intended API, mark `sub_agent_compression_hook` as deprecated:

```python
def sub_agent_compression_hook(state: dict[str, Any]) -> dict[str, Any]:
    """Deprecated: use build_compression_hook(config) instead."""
    # keep existing behavior for backward compat
```

### 4c. Hardcoded model in `orchestrator.py`

**Current** (`orchestrator.py:198`): `_recall_model_name = "gpt-5.1-mini"` as fallback.

And line 581: `ChatOpenAI(model="gpt-5.1-mini", ...)` — this is in the **dead code** block (the second duplicate set of functions after `return tools` on line 322). This dead code should be deleted entirely.

**Fix**:
1. Delete the dead code block (lines 324-605) — it's unreachable after `return tools` on line 322.
2. The fallback `_recall_model_name = "gpt-5.1-mini"` (line 198) is acceptable since it's overridden by config when available. But we should use the config-resolved name (line 298 already does this correctly in the live code).

### 4d. `project_root` derivation in `cli/run.py`

**Current** (`cli/run.py:43`):
```python
project_root = Path(config_path).resolve().parent.parent
```
Assumes `config_path` is always at `<project>/config/system.yaml`.

**Fix**: Add `project_root` as an explicit optional parameter:

```python
async def run_investigation(
    # ... existing params ...
    project_root: str | Path | None = None,   # NEW
) -> None:
    if project_root is None:
        project_root = Path(config_path).resolve().parent.parent
    else:
        project_root = Path(project_root).resolve()
```

**Files changed**: `src/agentm/builder.py`, `src/agentm/core/compression.py`, `src/agentm/tools/orchestrator.py`, `src/agentm/cli/run.py`

---

## Fix 5 — P2: Builder Decomposition

**Current**: `AgentSystemBuilder.build()` is a 190-line static method that does everything.

**Problem**: Library users who want to customize one component (e.g., swap the agent pool) must rewrite the entire build.

**Fix**: Decompose into composable steps while keeping `build()` as the convenience "batteries-included" path:

```python
class AgentSystemBuilder:
    """Builder with composable steps."""

    def __init__(self, system_type: str, scenario_config: ScenarioConfig) -> None:
        self.system_type = system_type
        self.scenario_config = scenario_config
        self.system_config: Any | None = None
        self.tool_registry: ToolRegistry | None = None
        self.agent_pool: AgentPool | None = None
        self.task_manager: TaskManager | None = None
        self.knowledge_store: KnowledgeStore | None = None
        self.memory_store: MemoryStore | None = None
        self.trajectory: TrajectoryCollector | None = None
        self.thread_id: str = ""
        self._tools_dir: Path | None = None

    def with_system_config(self, config: Any) -> AgentSystemBuilder:
        self.system_config = config
        return self

    def with_tool_registry(self, registry: ToolRegistry) -> AgentSystemBuilder:
        self.tool_registry = registry
        return self

    def with_agent_pool(self, pool: AgentPool) -> AgentSystemBuilder:
        self.agent_pool = pool
        return self

    def with_knowledge_store(self, store: KnowledgeStore) -> AgentSystemBuilder:
        self.knowledge_store = store
        return self

    def with_tools_dir(self, path: Path | str) -> AgentSystemBuilder:
        self._tools_dir = Path(path)
        return self

    def build(self) -> AgentSystem:
        """Build with defaults for anything not explicitly set."""
        if self.tool_registry is None:
            self.tool_registry = self._create_default_tool_registry()
        if self.agent_pool is None:
            self.agent_pool = self._create_default_agent_pool()
        # ... etc
        return self._assemble()

    # Class method for backward compat
    @classmethod
    def build_default(
        cls,
        system_type: str,
        scenario_config: ScenarioConfig,
        system_config: Any | None = None,
        existing_thread_id: str | None = None,
    ) -> AgentSystem:
        """One-shot build — same as current build() API."""
        builder = cls(system_type, scenario_config)
        if system_config is not None:
            builder.with_system_config(system_config)
        if existing_thread_id:
            builder.thread_id = existing_thread_id
        return builder.build()
```

**Usage as library**:

```python
from agentm.builder import AgentSystemBuilder
from agentm.tools.knowledge import KnowledgeStore

# Customize just the knowledge store
system = (
    AgentSystemBuilder("hypothesis_driven", scenario_cfg)
    .with_system_config(sys_cfg)
    .with_knowledge_store(KnowledgeStore("/my/custom/path"))
    .build()
)
```

**Backward compat**: Rename current `build()` → `build_default()` class method; new `build()` is the instance method. All existing call sites use the class method syntax `AgentSystemBuilder.build(...)` which maps to `build_default`.

**Files changed**: `src/agentm/builder.py`, `src/agentm/cli/run.py` (update call sites)

---

## Fix 6 — P2: Core Logic → `logging`, Rich → CLI Only

**Current**: All output goes through `Rich.Console.print()` in `cli/run.py`. Core modules don't log at all.

**Fix**: Add standard logging to core modules, keep Rich in CLI layer only.

```python
# src/agentm/core/task_manager.py (example)
import logging

logger = logging.getLogger("agentm.core.task_manager")

class TaskManager:
    async def submit(self, ...):
        logger.info("Task %s submitted: agent=%s type=%s", task_id, agent_id, task_type)
        # ...
```

```python
# src/agentm/builder.py
import logging

logger = logging.getLogger("agentm.builder")

class AgentSystemBuilder:
    def build(self) -> AgentSystem:
        logger.info("Building %s system with mode=%s", self.system_type, orch_mode)
        # ...
```

The CLI layer (`cli/run.py`) configures logging at startup:

```python
import logging

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(name)s %(levelname)s %(message)s")
```

Library users can then integrate with their own logging:

```python
import logging
logging.getLogger("agentm").setLevel(logging.WARNING)
```

**Files changed**: Add `logger` to `builder.py`, `core/task_manager.py`, `core/trajectory.py`, `tools/knowledge.py`. Adjust `cli/run.py` to initialize logging.

---

## Fix 7 — P3: Dead Code Cleanup

`src/agentm/tools/orchestrator.py` has **duplicate function definitions** after `return tools` on line 322. Lines 324-605 are unreachable dead code (exact duplicates of lines 35-321, with one difference: line 581 uses hardcoded `"gpt-5.1-mini"` instead of `_recall_model_name`).

**Fix**: Delete lines 324-605.

Also in `memory.py:29-31`, `set_checkpointer()` is a no-op shim:

```python
def set_checkpointer(checkpointer: Any) -> None:
    """No-op shim kept for call-site compatibility. Use set_db_path() instead."""
    pass
```

Check if any call site still uses it. If not, delete it.

**Files changed**: `src/agentm/tools/orchestrator.py`, `src/agentm/tools/memory.py`

---

## Execution Order

| Phase | Fixes | Risk | Effort |
|-------|-------|------|--------|
| 1 | Fix 7 (dead code), Fix 3 (exceptions) | Low — additive | Small |
| 2 | Fix 2 (sys.exit), Fix 4a/4c (paths, dead hardcode) | Low — mechanical | Small |
| 3 | Fix 1a/1b (knowledge/memory stores) | Medium — refactor | Medium |
| 4 | Fix 1c (websocket), Fix 4b/4d (model/path defaults) | Low | Small |
| 5 | Fix 5 (builder decomposition) | Medium — API change | Large |
| 6 | Fix 6 (logging) | Low — additive | Medium |

Each phase is independently shippable. Tests should be run after each phase.

---

## Public API Surface (Post-Fix)

After these changes, library users get:

```python
from agentm.builder import AgentSystemBuilder
from agentm.tools.knowledge import KnowledgeStore
from agentm.tools.memory import MemoryStore
from agentm.config.loader import load_system_config, load_scenario_config
from agentm.exceptions import AgentMError, ConfigError, ToolError

# 1. Load config
sys_cfg = load_system_config("path/to/system.yaml")
scn_cfg = load_scenario_config("path/to/scenario.yaml")

# 2. Build (simple path)
system = AgentSystemBuilder.build_default(
    system_type="hypothesis_driven",
    scenario_config=scn_cfg,
    system_config=sys_cfg,
)

# 3. Build (customized)
system = (
    AgentSystemBuilder("hypothesis_driven", scn_cfg)
    .with_system_config(sys_cfg)
    .with_knowledge_store(KnowledgeStore("/custom/path"))
    .with_tools_dir("/my/tools")
    .build()
)

# 4. Execute
async with system:
    result = await system.execute({"messages": [HumanMessage(content="...")]})

# 5. Error handling
try:
    system = AgentSystemBuilder.build_default(...)
except ConfigError as e:
    print(f"Bad config: {e}")
except AgentMError as e:
    print(f"AgentM error: {e}")
```
