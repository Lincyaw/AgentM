# Event Handler Convention — One Typed State Channel

Every event hook has one authoritative state channel: typed fields on the
event object. A handler mutates only fields declared by the event's
`HookContract.mutable_fields`; the emitter reads those final fields after
`emit()` returns.

Related: [handler-priority](handler-priority.md), [pluggable-architecture](pluggable-architecture.md)
§3.5, and [agent-loop](agent-loop.md).

## Why one state channel

The old `before_agent_start` contract accepted both event mutation and
return dictionaries. Multiple handlers could therefore update different
copies of the same logical value: later handlers saw `event.system`, while
the runtime could choose a returned `{"system": ...}` value. The result
depended on handler order and could silently discard prompt fragments.

The current contract removes that compatibility path. State that later
handlers or the emitter consume always lives on the event instance.

## Hook contract

Mutable event classes declare their writable fields mechanically:

```python
@dataclass(slots=True)
class BeforeSendToLlmEvent(Event):
    HOOK: ClassVar[HookContract] = HookContract(
        effects=("observe", "mutate_messages", "replace_model"),
        mutation_contract="The final provider request may be adjusted.",
        mutable_fields=("messages", "model", "tools", "system"),
    )

    messages: list[AgentMessage]
    model: Model
    tools: list[Tool]
    system: str | None
```

`mutation_contract` documents the semantics for people.
`mutable_fields` is the machine-readable source of truth used by runtime
validation, observability, adaptation metadata, and static analysis.

An explicit `return_contract` is separate. It is allowed only where a hook
returns a control decision or replacement value rather than mutating event
state, such as tool-call blocking or a `ToolResult` replacement. It must not
duplicate a mutable field channel.

## Runtime enforcement

`EventBus` validates every typed event by default:

1. Before each handler, deep-copy every event field not listed in
   `mutable_fields`.
2. Run the handler serially.
3. Compare the read-only fields.
4. If an undeclared field changed, restore it and raise immediately.

For observation-only events, `mutable_fields` is empty, so every payload field
is read-only and any handler-side change fails. This keeps “observe” a runtime
contract rather than a documentation convention.

Serial dispatch is load-bearing: each handler sees the final mutations made
by every earlier handler on that channel.

## Emitter rule

The emitter binds a mutable event to a local variable, emits it, and consumes
only the event's final fields:

```python
before_send = BeforeSendToLlmEvent(
    messages=messages,
    model=model,
    tools=tools,
    system=system,
)
await bus.emit(BeforeSendToLlmEvent.CHANNEL, before_send)
tool_index = {tool.name: tool for tool in before_send.tools}
stream_fn(
    before_send.messages,
    before_send.model,
    before_send.tools,
    before_send.system,
)
```

Using `messages`, `model`, `tools`, or a value derived from them before the
emit is state drift. It can make the provider request, executor view,
telemetry, and prompt dump disagree.

## Field patterns

- Last wins: singular values such as `system`, `model`, or `veto`.
- Accumulator: mutable lists where handlers append contributions.
- First claim: a handler checks an unset field before claiming it, such as
  `InputEvent.handled`.

The event class documents which pattern applies. No separate return
dictionary mirrors these fields.

## Mechanical gates

The repository enforces this convention at three layers:

- `AM015 event-source-drift` rejects inline mutable event construction and
  stale source values used after dispatch.
- `AM016 hook-contract-integrity` rejects missing, unknown, or inconsistent
  `mutable_fields`.
- `EventBus` catches handler-side writes to undeclared fields at runtime and
  restores the event before failing.

CI runs these checks over both `src/` and `contrib/`.

## Checklist for a new mutable hook

1. Define a slotted typed event class.
2. Declare every writable field in `HookContract.mutable_fields`.
3. Keep `mutation_contract` and `effects` consistent with those fields.
4. Bind the event to a local before dispatch.
5. Read every downstream value from the event after dispatch.
6. Do not add a return contract that duplicates mutable event state.
7. Add or update a fail-stop test only when the invariant is load-bearing.
