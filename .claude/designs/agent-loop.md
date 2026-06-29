# Agent Loop — Per-Turn Decision Protocol

The single source of truth for AgentM's loop termination semantics. The kernel `AgentLoop.run` produces exactly one decision per turn, expressed as a sum type. This document describes that decision, the contract extensions see, and the resolution lattice the kernel applies.

Related: [pluggable-architecture](pluggable-architecture.md) §3 (the loop is part of the kernel ABI), [sub-agent-lifecycle](sub-agent-lifecycle.md) (the canonical extension that overrides the default termination).

## Per-Turn Decision

Each iteration of `AgentLoop.run` ends with the kernel emitting one [`DecideTurnActionEvent`](../../src/agentm/core/abi/events.py) on the `decide_turn_action` channel. The event carries a [`TurnObservation`](../../src/agentm/core/abi/events.py) snapshot:

| field | type | meaning |
|---|---|---|
| `turn_index` | `int` | zero-based turn number that just completed |
| `assistant_message` | `AssistantMessage \| None` | the model's reply for this turn; `None` only on kernel-imposed paths (signal/max_turns) where no LLM call ran |
| `tool_outcomes` | `list[ToolOutcome]` | one outcome per executed tool call, in call order; empty if the assistant turn had no tool calls |
| `default_action` | `LoopAction` | what the kernel will do if no handler overrides |

Handlers may return a `LoopAction` (or `None` for "no opinion") to override `default_action`. The kernel then runs `_resolve_action(default, returns)` and acts on the resolved action.

## Sum Types

### `ToolOutcome` — what a tool returned

| variant | rationale |
|---|---|
| `ToolContinue(result)` | normal tool execution — bare `ToolResult` returns get auto-wrapped to this |
| `ToolTerminate(result, reason)` | tool succeeded and asks the loop to terminate; `reason` is opaque to the kernel and surfaced through the `ToolTerminated` cause |

Tools may return either a bare `ToolResult` or a `ToolOutcome`. Bare results normalize to `ToolContinue`, so legacy tools keep working. The terminal-tool case (e.g. `submit_final_report` in the RCA scenario) returns `ToolTerminate` directly from its `fn`.

**`reason` namespace convention.** `ToolTerminate.reason` is opaque to the kernel — it is surfaced verbatim through `ToolTerminated.reason`. To prevent accidental collisions when two scenarios pick the same bare label, prefix the reason with the extension or scenario short name and a colon: `"rca:final-report-submitted"`, `"plan_mode:plan-accepted"`, `"code_review:approved"`. The kernel does not enforce the convention, but observability indexers should treat the reason as opaque inside the namespace.

### Tool Execution Boundary

`AgentLoop` does not call `tool.execute` inline. Every foreground tool call
goes through `core/abi/tool_executor.py::execute_tool_call`, which applies a
substrate-owned execution boundary before the tool coroutine is awaited.

Execution domain is declared as optional tool metadata, not a new required
Protocol field:

```python
metadata = {"execution_domain": "thread"}
```

Supported values:

| domain | status | meaning |
|---|---|---|
| `event_loop` | default | run the tool coroutine behind an explicit `asyncio.Task` on the session event loop |
| `thread` | implemented | run the tool coroutine inside a worker-thread event loop; forward the abort signal cooperatively |
| `process` | implemented | run the tool coroutine in a spawned child process; signal/cancel terminates, then kills and joins the child if needed |
| `sandbox` | reserved | fail loudly until the sandbox executor exists |

The `thread` domain protects the core session event loop from a blocking tool,
but it is not a kill boundary: cancelling the outer await cannot forcibly stop a
Python thread. Tools still need to honor the forwarded signal. The `process`
domain is the killable boundary for untrusted/blocking tool code: the parent
waits on a result pipe plus the session signal, and on signal/cancel asks the
child to stop, then escalates through `terminate()` and `kill()` before joining
it. Process-domain tools should be pure with respect to the parent runtime:
child writes to the bus/session/API state do not mutate the parent. Because the
process executor uses the multiprocessing `spawn` start method, process-domain
tool objects must be pickleable/importable. `sandbox` is the later
resource-isolated variant. Wrapper atoms such as `background_exec` should keep
their wrapper logic on `event_loop` when touching session state, but execute the
wrapped tool through `execute_tool_call` so the wrapped tool's own domain is
preserved.

### LLM Preflight Prompt Debugging

The last inspection point before bytes leave AgentM is immediately after
`before_send_to_llm` and immediately before `StreamFn` is called. Two
environment variables expose that boundary:

| env | behavior |
|---|---|
| `AGENTM_LLM_PROMPT_DUMP=1` | Emit an `agentm.diagnostic` record with `source="llm_prompt_dump"` containing the final system prompt, messages, model, and tool schemas, then continue to the provider. |
| `AGENTM_LLM_PROMPT_DRY_RUN=1` | Emit the same diagnostic record, skip the provider call, and synthesize a normal end-turn assistant message. |

The diagnostic is intentionally emitted from the kernel loop rather than from
workflow/sub-agent code, so it captures the fully assembled prompt after
scenario atoms, `skill_loader`, `system_prompt`, context handlers, and tool
filters have all run. Payload strings, sequences, and nested objects are
bounded so a large paper or tool schema does not make the trace unusable.

Use `agentm trace logs --session <session> --format ndjson` and filter for
`agentm.diagnostic.source == "llm_prompt_dump"` to inspect the payload. For a
workflow tree, inspect the child session that corresponds to the worker whose
prompt you want.

### `TerminationCause` — why the loop is stopping

| variant | `final` | rationale |
|---|---|---|
| `ModelEndTurn` | `False` | assistant message had no tool_calls — model voluntarily finished |
| `ToolTerminated(tool_name, reason)` | `False` | a tool returned `ToolTerminate`; identifies which tool fired |
| `ProviderTruncated(kind)` | `False` | provider's stop_reason was `max_tokens` or `error` |
| `ProviderProtocolViolation(detail)` | `False` | provider said `tool_use` but no tool_calls were extracted — surfaces parser/provider disagreement distinctly instead of silently downgrading to `ModelEndTurn` |
| `MaxTurnsExhausted` | `True` | loop hit its turn cap |
| `SignalAborted` | `True` | external `asyncio.Event` was set |
| `BudgetExhausted` | `True` | harness short-circuited the next prompt because cost budget was tripped |

`final` controls override semantics: when `True`, no extension `LoopAction` return is honored. The hook still fires for observability, but `_resolve_action` ignores everything but the default. This matches the operational reality of these causes — extensions can cap budget but not un-cap it once tripped, and "I want one more turn" is not a meaningful answer to "the loop has been signalled to abort".

### `LoopAction` — what to do next

| variant | rationale |
|---|---|
| `Step` | continue to the next turn with current messages (default after a successful tools-and-results round) |
| `Stop(cause)` | terminate the loop with the given cause; `_finish_with_cause` emits `agent_end` |
| `Inject(messages)` | continue to next turn after appending these messages; used by extensions to override a default `Stop` (e.g. inject a continuation prompt instead of terminating) |

## Backlog: sum-typify the remaining handler return conventions

The kernel currently uses four different return-shape conventions across
its event channels:

| channel | return shape | notes |
|---|---|---|
| `decide_turn_action` | `LoopAction \| None` | sealed sum (this redesign) |
| `tool_call` | `{"block": True, "reason": str} \| None` | dict literal |
| `context` | `list[AgentMessage] \| {"messages": [...]} \| None` | dual shape |
| `tool_result` | bare `ToolResult \| None` | no wrapper |

The sum-type approach for `decide_turn_action` is strictly better: typed,
discoverable through autocomplete, and the resolution lattice
(`Inject > Stop > Step`) is explicit instead of buried in folklore. Extending
the same pattern to the other three channels — defining `ToolCallAction`
(`Allow | Block(reason)`), `ContextAction` (`Keep | Replace(messages)`),
`ToolResultAction` (`Keep | Replace(result)`) — is the natural follow-up and
will land in a separate PR. It is not blocking PR #65 cleanup because each
channel has only 1–2 in-tree handlers and the dict shapes have not caused
incidents yet, but it is the only way to scale to 10+ extensions without
re-reading every helper to remember which channel takes which shape.

## Default Action Computation

`_default_action_with_names(assistant_msg, paired_outcomes)` (in `core/abi/loop.py`) decides the kernel's default in this priority order:

1. **Any tool returned `ToolTerminate`?** → `Stop(ToolTerminated(tool_name=..., reason=...))`. First terminal tool wins so the cause maps to the *first* terminal call in the turn.
2. **No tool calls at all?** → map the provider's `assistant_msg.stop_reason`:
   - `max_tokens` → `Stop(ProviderTruncated(kind="max_tokens"))`
   - `error` → `Stop(ProviderTruncated(kind="error"))`
   - `tool_use` (no calls extracted) → `Stop(ProviderProtocolViolation(detail=...))`
   - anything else → `Stop(ModelEndTurn())`
3. **Tools ran successfully and none asked to terminate** → `Step()`

For kernel-imposed terminations (`SignalAborted`, `MaxTurnsExhausted`), the kernel calls `_terminate(...)` directly with the cause — the hook still fires (with `Stop(cause)` as the default and `tool_outcomes=[]`) so observability sees every termination, but overrides are ignored because `cause.final` is `True`.

## Resolution Lattice

`_resolve_action(default, returns)` reconciles the kernel default with handler-supplied overrides:

1. **`final` causes shadow everything.** If `default = Stop(cause)` and `cause.final` is `True`, return `default` regardless of what handlers returned.
2. **Among handler `LoopAction` returns:**
   - Any `Inject` wins. Messages from all `Inject` returns are concatenated in registration order. Extensions stack rather than fight.
   - Else any `Stop` wins. If multiple `Stop`s were returned, the last one's cause is used (latest authoritative voice).
   - Else `Step` if returned.
   - Else the default applies.

Handler returns that are not `LoopAction` instances (including `None`) are ignored — `None` is the standard "no opinion" signal.

```
                        kernel computes default
                                  │
                                  ▼
                   ┌──── Stop(cause) where cause.final ──── default wins ──→ terminate
                   │
                   └─── any other default
                                  │
                                  ▼
                       collect Inject/Stop/Step returns
                                  │
                                  ▼
                  ┌───── any Inject? ─── concat messages, continue
                  │
                  └───── any Stop? ───── use last Stop's cause, terminate
                                  │
                                  ▼
                       any Step? or default applies
```

## Why Not Flags/Strings?

The previous design encoded "what happens after each turn" via 5 separate signals that had to be combined with `if/elif`:

1. `assistant_msg.stop_reason` (provider's raw string)
2. presence/absence of `tool_calls` in the assistant message
3. `BeforeAgentEndEvent.stop_reason` (one of 6 string literals)
4. handler return shape `{"cancel": bool, "append": list}`
5. multi-handler resolution rules (`cancel`-OR + `append`-concat) that lived in folklore comments

Failure modes the redesign fixes:

- **Silent miscategorization.** Provider says `tool_use` but no tool calls were extracted → previously mapped to `end_turn` because there were no tool blocks to dispatch on. Now surfaces as `ProviderProtocolViolation`, which observability can flag as a parser bug.
- **No "this tool wants to terminate" signal.** Tools that completed the user's task (e.g. `submit_final_report`) had to either let the model decide to end the turn (which it might refuse) or fight the loop via `before_agent_end`. Now they return `ToolTerminate` and the kernel does the right thing without scenario glue.
- **Override cardinality buried in comments.** The `cancel`-OR-with-`append`-concat resolution rule was implicit in `_collect_before_agent_end_decision`. Now it's a typed lattice (`Inject > Stop > Step`) with a single explicit `_resolve_action` function.
- **No symmetric observability for `final` causes.** Previously, max_turns and aborted paths bypassed `before_agent_end` entirely, so trajectory subscribers couldn't see those terminations through the same channel. Now `decide_turn_action` always fires.

## Migration Notes

For anyone who used the old `before_agent_end` / `cancel+append` API:

| old | new |
|---|---|
| `api.on("before_agent_end", handler)` | `api.on("decide_turn_action", handler)` |
| `BeforeAgentEndEvent.messages` | `event.observation.assistant_message` (just the latest turn) |
| `BeforeAgentEndEvent.stop_reason` | inspect `event.observation.default_action` — `Stop(ModelEndTurn)` is voluntary, `Stop(ToolTerminated)` is a terminal-tool exit, etc. |
| `return {"cancel": True, "append": [msg]}` | `return Inject(messages=[msg])` |
| `return {"cancel": True}` (no append) | `return Inject(messages=[])` — but consider whether you really want to keep the loop alive without telling the model anything new |
| `event.messages.append(msg); return None` (mutate then exit) | `return Inject(messages=[msg])` — costs one extra LLM turn, but the loop still terminates cleanly when the model has nothing more to do; the in-place mutation is no longer supported |

For tools that want to declare the loop done:

```python
# Old: return ToolResult and hope the model decides to end the turn
async def submit_final_report(args) -> ToolResult:
    return ToolResult(content=[...])

# New: return ToolTerminate; kernel emits Stop(ToolTerminated(...))
async def submit_final_report(args) -> ToolTerminate:
    return ToolTerminate(
        result=ToolResult(content=[...]),
        reason="rca-final-report-submitted",
    )
```

Extensions that need to know "did the loop terminate cleanly?" should pattern-match on the cause:

```python
def on_agent_end(event: AgentEndEvent) -> None:
    cause = event.cause
    if isinstance(cause, (ModelEndTurn, ToolTerminated)):
        record_success()
    elif isinstance(cause, MaxTurnsExhausted):
        record_max_turns()
    # ... etc
```

The serialized observability format carries the cause as `attributes.cause = {<dataclass fields>}`; the indexer's `_extract_stop_reason` derives a canonical label from the cause's discriminating fields without importing kernel types.
