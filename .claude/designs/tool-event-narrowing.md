# Design: Per-Tool Typed Events and Narrowing

**Status**: DRAFT
**Created**: 2026-05-01
**Last Updated**: 2026-05-01

## Overview

Replace the single `ToolCallEvent` / `ToolResultEvent` (with stringly-typed `tool_name`) with a sealed family of per-tool dataclasses (`BashToolCallEvent`, `ReadToolCallEvent`, `GrepToolCallEvent`, ...), plus narrowing helpers (`is_bash_tool_call`, `is_grep_tool_result`, ...) so handlers can dispatch on tool identity with full static + runtime type safety. Pi-mono reference: [`packages/coding-agent/src/core/extensions/types.ts:740-940`](/tmp/pi-analysis/pi-mono/packages/coding-agent/src/core/extensions/types.ts).

## Motivation

The current AgentM kernel emits one `ToolCallEvent { tool_name: str, args: dict[str, Any] }`. Every handler that wants to do something tool-specific (e.g., `permission` denying `bash` calls; `file_mutation_queue` serializing `edit`/`write`) does:

```python
def on_tool_call(event):
    if event.tool_name == "bash":
        cmd = event.args["command"]    # untyped dict access; typo risk
        ...
```

Three problems compound:

1. `args` is `dict[str, Any]` — no static guarantee of `command: str`. Refactors are silent.
2. Handlers can't be expressed with `match event:` for narrowing because every tool produces structurally identical events.
3. As we add `grep`, `find`, `ls`, `tool_submit_plan`, etc., the proliferation of string-keyed dispatch grows linearly. Pi solves this with a discriminated union + TypeScript type guards. We can match the ergonomics with Python 3.12 dataclasses + `match` + `isinstance`.

## Design Details

### New event hierarchy

In `core/kernel/events.py` (add alongside the existing `ToolCallEvent` / `ToolResultEvent`):

```python
# Base classes — keep the existing ToolCallEvent and ToolResultEvent as
# the *generic* fallback for custom tools. Per-tool variants subclass them.

@dataclass(slots=True)
class ToolCallEvent(Event):
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    # Mutability contract unchanged.

# Per-tool subclasses, one per built-in tool. Each carries a typed `input`
# field referencing a TypedDict that mirrors the tool's JSON schema.

class BashInput(TypedDict, total=False):
    command: str
    timeout: float

@dataclass(slots=True)
class BashToolCallEvent(ToolCallEvent):
    input: BashInput   # alias view onto args; same dict, narrower type

# Similar for: ReadToolCallEvent, EditToolCallEvent, WriteToolCallEvent,
# GrepToolCallEvent, FindToolCallEvent, LsToolCallEvent.
```

The `input` field is the same dict as `args` (no copy) — just typed. Mutating `input["command"] = ...` mutates `args` too. This preserves the documented mutability contract ([extension-as-scenario.md §10b](extension-as-scenario.md#10b1-new-events-to-add-non-breaking-pure-additions)) without forcing a copy.

`tool_name` on the subclass is a `Literal["bash"]` (Python supports this via class-level annotation; runtime value is set by the kernel when dispatching). This gives `match`-statement exhaustiveness when callers pin the kernel to known tool names.

### Narrowing helpers

```python
def is_bash_tool_call(e: ToolCallEvent) -> TypeGuard[BashToolCallEvent]:
    return isinstance(e, BashToolCallEvent)

def is_read_tool_call(e: ToolCallEvent) -> TypeGuard[ReadToolCallEvent]: ...
# ... one per tool

# Symmetric set for ToolResultEvent: BashToolResultEvent etc., is_bash_tool_result.
```

`typing.TypeGuard` (or `typing.TypeIs` in 3.13+) gives mypy the narrowing it needs. At runtime these are `isinstance` checks — the same pattern AgentM already uses for kernel/harness events.

### Idiomatic Python alternative: `match` statement

Because every per-tool event is a dataclass, callers can also use:

```python
def on_tool_call(event: ToolCallEvent) -> None:
    match event:
        case BashToolCallEvent(input={"command": cmd}) if cmd.startswith("rm "):
            return {"block": True, "reason": "rm denied"}
        case ReadToolCallEvent(input={"path": p}):
            audit.log_read(p)
        case _:
            pass
```

This is the cleaner pattern and **the recommended style going forward**. The `is_*` helpers exist for handlers that prefer early-return / chained `if` style.

### Construction site (kernel loop)

In `core/kernel/loop.py` where `ToolCallEvent` is instantiated today, swap to a small dispatch:

```python
_TOOL_EVENT_TYPES: dict[str, type[ToolCallEvent]] = {
    "bash": BashToolCallEvent,
    "read": ReadToolCallEvent,
    "edit": EditToolCallEvent,
    "write": WriteToolCallEvent,
    "grep": GrepToolCallEvent,
    "find": FindToolCallEvent,
    "ls": LsToolCallEvent,
}

def make_tool_call_event(tool_call_id: str, tool_name: str, args: dict) -> ToolCallEvent:
    cls = _TOOL_EVENT_TYPES.get(tool_name, ToolCallEvent)
    if cls is ToolCallEvent:
        return ToolCallEvent(tool_call_id=tool_call_id, tool_name=tool_name, args=args)
    return cls(tool_call_id=tool_call_id, tool_name=tool_name, args=args, input=args)
```

Custom tools registered by user extensions still get the base `ToolCallEvent`. They can opt into typed events by registering their own dataclass and adding it to a registry (deferred — see Open Questions).

The same pattern applies to `ToolResultEvent`: per-tool subclasses with a typed `details` field (currently `Any`).

### Migration path for existing handlers

The new types subclass the existing ones, so:

```python
# Old code — still works; new event still isinstance(event, ToolCallEvent).
def handler(event):
    if event.tool_name == "bash":
        cmd = event.args["command"]

# New code — preferred.
def handler(event):
    if is_bash_tool_call(event):
        cmd = event.input["command"]   # statically typed
```

No breaking change. The migration is a sweep across `extensions/builtin/*.py` to convert `event.tool_name == "x" and event.args[...]` patterns to `is_x_tool_call(event)` / `match` form.

### Scope of the migration

Files that currently do `event.tool_name == "..."` checks (per `git grep`-able patterns) — the implementer must convert them in a coordinated PR (analogous to Phase 2.0b). Estimated scope:

- `extensions/builtin/permission.py`
- `extensions/builtin/dedup.py`
- `extensions/builtin/file_mutation_queue.py`
- `extensions/builtin/tool_result_budget.py`
- `extensions/builtin/trajectory.py`
- Any test under `tests/unit/extensions/builtin/*/` that constructs or asserts on `ToolCallEvent`.
- Scenario YAMLs do **not** change — they reference module paths, not event types.

### Why subclassing, not a tagged union (Algebraic Data Type)

Considered options:

**Option A — single class, `Literal["bash", ...]` `tool_name`, `args: BashArgs | ReadArgs | ...`** — relies on `Literal` for narrowing. Mypy can narrow on `event.tool_name == "bash"`, but the user-facing API still has all tools' args reachable; the discriminated-union story works less well than full subclasses.

**Option B — per-tool subclasses (this design)** — cleanest match for `match` statements; isinstance is fast; subclasses can carry tool-specific helpers; keeps the existing single-class fallback for custom tools.

**Option C — `TaggedUnion[Bash, Read, ...]`** — requires dataclass library tricks (e.g., `pydantic` discriminated unions). Heavy.

Option B wins on simplicity, isinstance friendliness, and zero new deps.

## Interface Definition

```python
# core/kernel/events.py — additions

class BashInput(TypedDict, total=False): ...
class ReadInput(TypedDict, total=False): ...
class EditInput(TypedDict, total=False): ...
class WriteInput(TypedDict, total=False): ...
class GrepInput(TypedDict, total=False): ...
class FindInput(TypedDict, total=False): ...
class LsInput(TypedDict, total=False): ...

@dataclass(slots=True)
class BashToolCallEvent(ToolCallEvent):
    input: BashInput
# ... one per tool

@dataclass(slots=True)
class BashToolResultEvent(ToolResultEvent):
    input: BashInput
    details: BashToolDetails | None = None
# ... one per tool

# Narrowing helpers
def is_bash_tool_call(e: ToolCallEvent) -> TypeGuard[BashToolCallEvent]: ...
def is_bash_tool_result(e: ToolResultEvent) -> TypeGuard[BashToolResultEvent]: ...
# ... one per tool
```

`*ToolDetails` types live next to their tool atoms (where the structured data is produced) and are imported into `kernel/events.py`. **Exception to the layer-purity rule**: `kernel/events.py` is allowed to import the details dataclasses because they are pure data. The implementer phase locks the exact module name (probably `core/kernel/tool_details.py` to keep imports flat).

## Acceptance Scenarios

1. `match event: case BashToolCallEvent(input={"command": cmd}): ...` narrows `cmd` to `str` under mypy.
2. Existing `event.tool_name == "bash"` handlers keep working unchanged (subclass relationship preserved).
3. `is_grep_tool_call(event)` returns `True` exactly when `event.tool_name == "grep"`.
4. A custom user-registered tool still produces a base `ToolCallEvent`; `match` falls through to `case ToolCallEvent()`.
5. Mutating `event.input["command"]` from a handler mutates `event.args["command"]` (same dict).
6. Per-tool subclasses appear in `__init__.py`'s `__all__` — visible to extension authors via plain import.

## Related Concepts

- [pluggable-architecture.md](pluggable-architecture.md) §3.5 — event taxonomy
- [extension-as-scenario.md](extension-as-scenario.md) §10b.1 — event additions list
- [search-tools.md](search-tools.md) — adds `grep`/`find`/`ls` events
- [edit-diff.md](edit-diff.md) — the upgraded `tool_edit` carries richer `EditToolDetails`

## Constraints and Decisions

| Decision | Rationale | Alternative |
|---|---|---|
| Per-tool dataclass subclasses | Cleanest `match` ergonomics; isinstance-friendly | Tagged union via pydantic / Literal-tagged single class |
| `input` is an alias for `args` (same dict object) | Preserves mutability contract; zero copy | Copy `args` to typed `input` — breaks mutation semantics |
| Custom tools fall back to base `ToolCallEvent` | Don't force every third-party tool to declare a dataclass | Mandatory subclass registration — friction |
| Migration as one coordinated PR (Phase 2.0b-style) | Many atoms touch this; piecemeal would be worse | Per-atom incremental — sustained churn |
| `TypedDict, total=False` for input shapes | Tools accept partial args at the JSON-schema level | `dataclass` for input — clashes with mutation-in-place |

## Out of Scope

- Auto-generating the `input` `TypedDict` from the tool's JSON schema. The implementer hand-writes them; future tooling can lint that they match.
- Generic event-handler decorators with type-driven dispatch (`@on(BashToolCallEvent)` style). Defer; `is_*` + `match` are enough.
- Strict tool-event registry that rejects unknown `tool_name` values. We allow custom tools by design.

## Open Questions

- [ ] Should custom-tool authors be able to register their own typed event subclass via `api.register_tool_event_type(name, cls)`? Defer — no real consumer yet. The base `ToolCallEvent` covers them.
- [ ] Do we want `ToolCallEventResult`-style typed *return* values from handlers? Currently handlers return `dict | None`. Recommendation: defer; the dict shape is small and stable.
- [ ] Where does `*ToolDetails` live to avoid layer-purity churn? Recommendation: `core/kernel/tool_details.py` so atoms can import it and the kernel can too.
