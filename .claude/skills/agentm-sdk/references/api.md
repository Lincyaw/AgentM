# Atom API Reference

Detailed API for atoms interacting with the SDK via `AtomAPI`.
Read this when writing or editing atoms.

## Operations — the environment abstraction

Operations are how atoms act on the target environment (local host,
sandbox, or any future backend). The `local_backend` atom binds the
default local backend's services:

```yaml
- module: agentm.extensions.builtin.local_backend
```

Everything is consumed through the service registry with the role
descriptors from `agentm.core.abi.roles`:

- Shell commands → `BASH_OPERATIONS_SERVICE` (`operations:bash`)
- Host-pinned shell (never the sandbox) → `HOST_BASH_OPERATIONS`
- Environment bundle (snapshot/close) → `ENVIRONMENT_OPERATIONS`
- File reads → `RESOURCE_READER` role
- Git-tracked file writes → `RESOURCE_WRITER` role

### BashOperations

```python
from agentm.core.abi import BASH_OPERATIONS_SERVICE, BashOperations

bash_ops = api.services.require(BASH_OPERATIONS_SERVICE, BashOperations)
result = await bash_ops.exec(
    cmd, cwd=api.ctx.cwd, timeout=30.0,
    env={"KEY": "val"},       # optional
    on_data=callback,          # optional streaming
    signal=cancel_event,       # optional cancellation
)
# result: ExecResult(stdout, stderr, exit_code, timed_out)
```

### Common mistakes

```python
# WRONG — bypasses the environment abstraction
with open(path, "rb") as f: data = f.read()
subprocess.run(["grep", "-r", pattern, "."])

# RIGHT
writer = api.services.require_role(RESOURCE_WRITER)
data = await writer.read(path)
result = await bash_ops.exec(f"grep -r {shlex.quote(pattern)} .", cwd=api.ctx.cwd)
```

### Lazy-resolve pattern

Declare hard dependencies in `MANIFEST.requires` (the solver orders
installs so they exist), and resolve per call rather than caching the
instance at install — the binding may be replaced over the session's
lifetime:

```python
def _bash_ops() -> BashOperations:
    return api.services.require(BASH_OPERATIONS_SERVICE, BashOperations)
```

---

## Registering tools

```python
from agentm.core.abi import FunctionTool, TextContent, ToolResult

api.register_tool(
    FunctionTool(
        name="my_tool",
        description="What the LLM sees.",
        parameters={...},              # JSON Schema dict
        fn=my_async_handler,           # async (dict) -> ToolResult
        metadata={"file_op": "read"},  # optional, for compaction tracking
    )
)
```

### Tool result convention

```python
# Success
ToolResult(content=[TextContent(type="text", text=output)], is_error=False)

# Error — LLM sees this and can retry
ToolResult(content=[TextContent(type="text", text=msg)], is_error=True)
```

Never raise exceptions from tool handlers — catch and return `is_error=True`.

### Pydantic-derived schemas

If you have a Pydantic model, don't hand-write JSON Schema:

```python
from agentm.core.lib import pydantic_to_tool_schema
schema = pydantic_to_tool_schema(MyModel)
```

This strips `$defs`, inlines refs, and produces a clean schema compatible
with all providers. Never use `model.model_json_schema()` directly — it
produces output most providers reject.

---

## ResourceWriter — git-tracked writes

For writes to user-visible files (code, memory, artifacts):

```python
from agentm.core.abi import RESOURCE_WRITER

writer = api.services.require_role(RESOURCE_WRITER)

await writer.write(path, content_bytes, rationale="why")
await writer.replace(path, old_bytes, new_bytes, rationale="why")
await writer.delete(path, rationale="why")
data = await writer.read(path)
```

ResourceWriter adds git-commit semantics. In sandbox mode, it is
transparently replaced by a sandbox-backed writer. Don't bypass it
with direct `open()` / `Path.write_bytes()` for user-visible files.

---

## Events

### Subscribing

```python
from agentm.core.abi.events import TurnEndEvent

def _on_turn_end(event: TurnEndEvent) -> None:
    ...

api.on(TurnEndEvent.CHANNEL, _on_turn_end)
```

### Key events

| Event | Fires when | Use for |
|-------|-----------|---------|
| `BeforeRunEvent` | Before agent loop | System prompt injection |
| `SessionReadyEvent` | Session initialized | One-time setup |
| `TurnStartEvent` / `TurnEndEvent` | Each LLM turn | Per-turn bookkeeping |
| `ToolCallEvent` / `ToolResultEvent` | Tool invocation | Gating, logging |
| `ContextEvent` | Context assembly | Per-turn context injection |
| `BeforeSendEvent` | Before LLM call | Last-chance message edit |
| `SessionShutdownEvent` | Session ending | Cleanup |

### Handler priorities

Use `BusPriority.PRE` -> `BusPriority.NORMAL` -> `BusPriority.POST`.
Use `PRE` for gates, `POST` for logging.

```python
from agentm.core.abi import BusPriority
api.on(ToolCallEvent.CHANNEL, my_gate, priority=BusPriority.PRE)
```

---

## Services — inter-atom communication

Atoms communicate through the typed `ServiceRegistry`, **never through
imports**:

```python
# Provider atom
api.services.register("my_service", my_service_object, MyProtocol)

# Consumer atom (declare dependency so provider loads first)
MANIFEST = ExtensionManifest(requires=("provider_atom",), ...)

def install(api: AtomAPI, config: Mapping[str, JsonValue]) -> None:
    svc = api.services.get("my_service", MyProtocol)
```

Service scopes control inheritance to child sessions:
- `"session"` — local to this session only
- `"tree"` — inherited by child sessions (default)
- `"host"` — host-owned injection point, inherited
- `"process"` — process-wide, inherited
- `"resource"` — tied to an external resource boundary, inherited
