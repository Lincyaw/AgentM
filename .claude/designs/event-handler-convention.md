# Event Handler Convention — Typed Fields, Not Return Dicts

All bidirectional communication between event handlers and the runtime
goes through **typed fields on the event object**. Handlers mutate the
event; the runtime reads it after `emit()` returns. Handler return
values are a deprecated backward-compat path — new code never
introduces new return-dict keys.

Related: [handler-priority](handler-priority.md) (dispatch order),
[pluggable-architecture](pluggable-architecture.md) §3.5 (event bus),
[agent-loop](agent-loop.md) (decide_turn_action resolution).

## Motivation

The original design used handler return values (`{"system": "..."}`,
`{"block": True, "cause": ...}`) as the runtime's data source. This
caused a class of silent bugs: handlers that mutated `event.system`
without returning the dict had their system prompts silently dropped.
Five handlers across builtin and contrib had this bug.

Return dicts are also untyped — handler authors can't discover available
keys without reading runtime source. Typed event fields are
IDE-discoverable, documented, and self-describing.

Mature frameworks (Gin, Express, Koa) uniformly use the
context/event object as the single communication channel. We adopt
the same pattern.

## The Rule

> **Handler reads/writes event fields. Runtime reads event after emit.
> Return values are ignored for new channels.**

## Field Design Patterns

Every event field falls into one of three categories:

### 1. Last-Wins (replacement)

Handler sets the field; if multiple handlers set it, the last one's
value is what the runtime sees. Used for singular values where only
one answer makes sense.

```python
@dataclass(slots=True)
class BeforeAgentStartEvent(Event):
    system: str | None = None     # last mutation wins
    veto: TerminationCause | None = None  # first non-None wins (runtime checks)
```

Handler:
```python
def handler(event: BeforeAgentStartEvent) -> None:
    event.system = f"{event.system or ''}\n\n{my_block}"
```

For **append-style** handlers (system prompt injection), the handler
reads the current value, appends, and writes back. This naturally
chains because all handlers share the same event object in
registration order.

### 2. Accumulator (multi-handler contribution)

Multiple handlers each contribute items. The runtime reads the
accumulated list after all handlers run.

```python
@dataclass(slots=True)
class DecideTurnActionEvent(Event):
    observation: Observation
    injections: list[list[AgentMessage]] = field(default_factory=list)
    stop: Stop | None = None
    step: bool = False
```

Handler:
```python
def handler(event: DecideTurnActionEvent) -> None:
    event.injections.append(reminder_messages)
```

Runtime resolves the accumulated fields with the same lattice as
today: injections > stop > step > default.

### 3. First-Match (short-circuit)

The first handler to set the field "claims" the event. Downstream
handlers can check whether it's already claimed.

```python
@dataclass(slots=True)
class InputEvent(Event):
    text: str
    handled: bool = False
    messages: list[AgentMessage] | None = None
```

Handler:
```python
def handler(event: InputEvent) -> None:
    if event.handled:
        return  # another handler already claimed this
    if event.text.startswith("/"):
        event.handled = True
        event.messages = process_slash(event.text)
```

## Backward Compatibility

The runtime reads event fields as the primary path. If no handler
mutated the relevant field, the runtime falls back to reading
return values via `collect_system_replacement` / `collect_start_veto`.
This preserves backward compat for existing handlers during
migration.

Once all handlers are migrated, the `collect_*` helpers can be
removed.

## Checklist for New Events

When adding a new event channel:

1. Define a `@dataclass(slots=True)` event class with typed fields
   for every piece of data the runtime will read back.
2. Use `None` defaults for optional/unset fields.
3. Use `field(default_factory=list)` for accumulator fields.
4. Document each field's semantics: last-wins, accumulator, or
   first-match.
5. Runtime reads event fields after `await bus.emit()` — never
   reads return values.
6. Export the event class from `agentm.core.abi`.

## Migration Status

| Channel | Event Class | Return-dict keys | Status |
|---------|-----------|-----------------|--------|
| `before_agent_start` | `BeforeAgentStartEvent` | `system`, `block`+`cause` | migrating |
| `decide_turn_action` | `DecideTurnActionEvent` | `LoopAction` (typed) | planned |
| `input` | raw `dict` (no class) | `handled`, `messages` | planned |
| all others | typed Event subclasses | (notification-only, no return consumed) | done |
