---
name: agentm-sdk
description: >
  AgentM SDK development guide — atom contract, Operations abstraction, event
  system, service communication, CLI conventions, scenario authoring, logging,
  structured output, and config resolution. Use whenever writing, editing, or
  reviewing code under src/agentm/ (atoms, core, gateway), contrib/scenarios/
  (manifests), or contrib/extensions/ (workspace-member atoms). Also trigger
  when creating new atoms, modifying MANIFEST declarations, registering tools
  or events, touching FileOperations / BashOperations / ResourceWriter, writing
  CLI subcommands, configuring model profiles, or when a code change looks like
  it might bypass the SDK's existing abstractions. If you catch yourself about
  to write raw os.stat / open() / subprocess.run in an atom, or importing
  openai/anthropic directly instead of going through the provider layer, stop
  and read this first.
---

# AgentM SDK Guide

This skill exists because agents working on this codebase tend to reinvent
mechanisms the SDK already provides. Before writing new infrastructure, check
whether the SDK already has it.

## Architecture in 30 seconds

```
atoms  (extensions/builtin/*.py)     ← you write these
  ↕  ExtensionAPI (core/abi/)        ← the only API you use
substrate (core/runtime/)            ← you never import from here
```

Atoms are plugins. Each atom gets an `ExtensionAPI` handle (`api`) in its
`install()` function. Everything an atom needs — tools, events, operations,
services, sessions — is accessed through `api`. If you find yourself importing
from `core.runtime`, you are doing it wrong.

---

## 1. The atom contract (§11)

Every atom is a single `.py` file that exports:

```python
MANIFEST = ExtensionManifest(
    name="my_atom",
    description="What this atom does.",
    registers=("tool:my_tool",),  # what it provides
    requires=(),                   # what it needs from other atoms
)

def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    # all wiring happens here
```

### Import rules (enforced by `extensions.validate`)

| Allowed | Forbidden |
|---------|-----------|
| `agentm.core.abi.*` | `agentm.core.runtime.*` |
| `agentm.core.lib.*` | other atoms (`agentm.extensions.builtin.X`) |
| `agentm.extensions.ExtensionManifest` | `agentm.core._internal` |
| stdlib, third-party libs | |

### Config resolution

Atom config comes from three sources (highest wins):

```
CLI --set overrides  >  env AGENTM_<ATOM>_<KEY>  >  manifest config:
```

Declare accepted keys in `MANIFEST.config_schema` (JSON Schema). Use
`config.get(key, default)` in `install()` — defaults live there, not in
the schema.

---

## 2. Operations — the environment abstraction

The `Operations` bundle is how atoms interact with the target environment
(local host, K8s sandbox, or any future backend). The unified `operations`
atom selects the backend via `config.backend`:

```yaml
- module: agentm.extensions.builtin.operations
  config:
    backend: local       # or "agent_env"
```

### When to use Operations

**Use Operations** for LLM-triggered actions on the target environment:
- Reading/writing user files → `api.get_operations().file`
- Running shell commands → `api.get_operations().bash`

**Don't use Operations** for agent infrastructure:
- Reading skill files, prompt templates, config — host resources
- Writing to `.agentm/` internal state (catalog, traces, eval runs)

### FileOperations

```python
file_ops = api.get_operations().file

data = await file_ops.read_file(path)           # bytes
stat = await file_ops.stat(path)                # FileStat(size, mtime_ns, is_file, is_dir)
exists = await file_ops.access(path)            # bool
is_file = await file_ops.is_file(path)          # bool
is_dir = await file_ops.is_dir(path)            # bool
entries = await file_ops.list_dir(path)         # list[str]
await file_ops.write_file(path, data)           # write bytes
await file_ops.makedirs(path, exist_ok=True)    # create directories
```

### BashOperations

```python
bash_ops = api.get_operations().bash

result = await bash_ops.exec(
    cmd, cwd=api.cwd, timeout=30.0,
    env={"KEY": "val"},       # optional
    on_data=callback,          # optional streaming
    signal=cancel_event,       # optional cancellation
)
# result: ExecResult(stdout, stderr, exit_code, timed_out)
```

### Common mistakes

```python
# WRONG — bypasses the Operations abstraction
stat = os.stat(path)
with open(path, "rb") as f: data = f.read()
subprocess.run(["grep", "-r", pattern, "."])

# RIGHT
stat = await file_ops.stat(path)
data = await file_ops.read_file(path)
result = await bash_ops.exec(f"grep -r {shlex.quote(pattern)} .", cwd=api.cwd)
```

### Lazy-resolve pattern

At install time, Operations may not be registered yet. Defer to first use:

```python
def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    _cache: list[FileOperations] = []

    def _get_file_ops() -> FileOperations:
        if not _cache:
            _cache.append(api.get_operations().file)
        return _cache[0]
```

---

## 3. Registering tools

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
from agentm.core.lib import pydantic_to_openai_tool_schema
schema = pydantic_to_openai_tool_schema(MyModel)
```

This strips `$defs`, inlines refs, and produces a clean schema compatible
with all providers. Never use `model.model_json_schema()` directly — it
produces output most providers reject.

---

## 4. ResourceWriter — git-tracked writes

For writes to user-visible files (code, memory, artifacts):

```python
writer = api.get_resource_writer()

await writer.write(path, content_bytes, rationale="why")
await writer.replace(path, old_bytes, new_bytes, rationale="why")
await writer.delete(path, rationale="why")
data = await writer.read(path)
```

ResourceWriter adds git-commit semantics. In sandbox mode, it is
transparently replaced by a sandbox-backed writer. Don't bypass it
with direct `open()` / `Path.write_bytes()` for user-visible files.

---

## 5. Events

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
| `BeforeAgentStartEvent` | Before agent loop | System prompt injection |
| `SessionReadyEvent` | Session initialized | One-time setup |
| `TurnStartEvent` / `TurnEndEvent` | Each LLM turn | Per-turn bookkeeping |
| `ToolCallEvent` / `ToolResultEvent` | Tool invocation | Gating, logging |
| `ContextEvent` | Context assembly | Per-turn context injection |
| `BeforeSendToLLMEvent` | Before LLM call | Last-chance message edit |
| `SessionShutdownEvent` | Session ending | Cleanup |
| `ResourcesDiscoverEvent` | Startup scan | Contributing skill/resource paths |

### Handler priorities

`PRE` → `NORMAL` → `POST`. Use `PRE` for gates, `POST` for logging.

```python
api.on(ToolCallEvent.CHANNEL, my_gate, priority="PRE")
```

---

## 6. Services — inter-atom communication

Atoms communicate through named services, **never through imports**:

```python
# Provider atom
api.set_service("my_service", my_service_object)

# Consumer atom (declare dependency so provider loads first)
MANIFEST = ExtensionManifest(requires=("provider_atom",), ...)

def install(api, config):
    svc = api.get_service("my_service")
```

---

## 7. LLM provider layer

The SDK wraps LLM providers (`llm_openai`, `llm_anthropic`). Never import
`openai` or `anthropic` directly in atoms — the provider layer handles:

- Message encoding (tool calls, images, system prompts)
- Streaming with backpressure
- Retry with exponential backoff (via the `retry_policy` atom)
- `reasoning_effort` mapping across providers
- `config.toml` model profiles

### Model profiles (`~/.agentm/config.toml`)

```toml
default_model = "doubao"

[models.doubao]
provider = "openai"
model = "doubao-seed-2-0-pro-260215"
base_url = "https://ark.cn-beijing.volces.com/api/v3"
api_key = "..."
context_window = 131072
reasoning_effort = "high"

[models.doubao.extra_body]
thinking = { type = "enabled" }
```

Precedence: CLI flag > env / `.env` > `config.toml` profile.

### Structured output

Don't use `openai.beta.chat.completions.parse()` or Anthropic's tool-use
for structured extraction. The SDK's provider layer already handles
`response_format` and JSON schema conversion. If an atom needs structured
LLM output, use the provider's built-in support through `ProviderConfig`.

---

## 8. CLI conventions (typer)

The CLI uses `typer` with a single `app` instance in `cli.py`.

```python
@app.command()
def my_subcommand(
    name: Annotated[str, typer.Argument(help="...")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    ...
```

Rules:
- Use `typer.BadParameter` for input validation errors (exit code 2)
- Use `raise SystemExit(1)` for runtime failures
- stdout for machine-readable output, stderr for human messages
- `--format ndjson|table|text` when output has multiple consumers
- Non-interactive by default (no prompts, no `input()`)

---

## 9. Logging and diagnostics

Atoms use **two channels** depending on audience:

### For observability / debugging (developers)

```python
import logging
logger = logging.getLogger(__name__)
logger.warning("something unexpected: %s", detail)
```

### For user-visible diagnostics (surfaced in TUI/trace)

```python
from agentm.core.abi.events import DiagnosticEvent

await api.events.emit(
    DiagnosticEvent.CHANNEL,
    DiagnosticEvent(level="warning", source="my_atom", message="...")
)
```

Don't use `print()`. Don't write to stdout from atoms — the CLI owns stdout.

---

## 10. Scenario authoring

A scenario is a YAML manifest at `contrib/scenarios/<name>/manifest.yaml`.
It composes atoms:

```yaml
name: my_scenario
description: What this scenario does.
extensions:
  # operations MUST be first
  - module: agentm.extensions.builtin.operations
    config:
      backend: local

  - module: agentm.extensions.builtin.file_tools
  - module: agentm.extensions.builtin.tool_bash
  - module: agentm.extensions.builtin.observability
```

### Rules

- `operations` atom listed **first** — other atoms depend on it
- Floor atoms (`compaction_prompts`, `slash_commands`) are auto-mounted;
  don't list them
- Scenario-specific logic goes in scenario-local atoms under
  `contrib/scenarios/<name>/`, **never** in `src/agentm/core/`
- Config for atoms is inline:
  ```yaml
  - module: agentm.extensions.builtin.system_prompt
    config:
      prompt: "You are a helpful assistant."
  ```

---

## 11. Quick reference: "I want to X"

| I want to... | Use this |
|--------------|----------|
| Read a user file | `api.get_operations().file.read_file(path)` |
| Write a user file (git-tracked) | `api.get_resource_writer().write(path, data)` |
| Check file metadata | `api.get_operations().file.stat(path)` |
| Run a shell command | `api.get_operations().bash.exec(cmd, cwd=api.cwd)` |
| Register a tool for the LLM | `api.register_tool(FunctionTool(...))` |
| Listen to events | `api.on(Event.CHANNEL, handler)` |
| Share state between atoms | `api.set_service(name, obj)` |
| Consume another atom's state | `api.get_service(name)` |
| Inject system prompt content | Handle `BeforeAgentStartEvent` |
| Inject per-turn context | Handle `ContextEvent` |
| Generate JSON Schema from Pydantic | `pydantic_to_openai_tool_schema(Model)` |
| Spawn a child agent | `api.spawn_child_session(config)` |
| Emit a user-visible diagnostic | Emit `DiagnosticEvent` |
| Log for debugging | `logging.getLogger(__name__)` |
| Get the working directory | `api.cwd` |
| Get the session ID | `api.session.get_session_id()` |

## 12. Anti-patterns checklist

1. **Direct filesystem I/O in tool handlers** — Use `FileOperations` / `BashOperations`.
2. **Importing `core.runtime`** — Use `core.abi` only.
3. **Importing another atom** — Use `api.get_service()`.
4. **Hand-writing JSON Schema next to a Pydantic model** — Use `pydantic_to_openai_tool_schema`.
5. **`subprocess.run()` for shell commands** — Use `BashOperations.exec()`.
6. **Resolving Operations at install time** — Use lazy-resolve (may not be registered yet).
7. **Raising exceptions from tool handlers** — Catch and return `ToolResult(is_error=True)`.
8. **Scenario-specific logic in `src/agentm/core/`** — Belongs in `contrib/scenarios/<name>/`.
9. **Direct `openai` / `anthropic` imports in atoms** — Use the provider layer.
10. **`print()` or stdout writes in atoms** — Use `logging` or `DiagnosticEvent`.
11. **Preset enums for subjective fields** — Use free-text + LLM-decided.
12. **Config defaults in schema** — Defaults live in `config.get(key, default)`, not the JSON Schema.
