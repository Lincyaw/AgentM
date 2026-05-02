# Sub-Agent Lifecycle

Status: design (not yet implemented)
Owner: subagent + orchestration

Reaches into: `core/abi/events.py`, `core/abi/loop.py`, `extensions/builtin/sub_agent.py`,
`harness/events.py` (re-export only).

## Problem

`extensions/builtin/sub_agent.py` already lets a parent dispatch a child `AgentSession`
via `dispatch_agent` and queries its status via `check_tasks`. The dispatch is
asynchronous: `dispatch_agent` returns immediately with a `task_id`; the child runs in
the background; results are not delivered to the parent unless the parent explicitly
polls.

Three concrete failure modes followed from this in the rca scenario:

1. **Parent exits with unread findings.** The kernel terminates the loop the moment the
   model emits a text-only response (`AgentLoop.run` line 327: empty `tool_calls` →
   emit `agent_end`). If a child is still running or has just completed, its findings
   are silently dropped and the child gets a 5-second grace before being aborted by
   `_ChildTaskManager.on_session_shutdown`. The first rca trial showed this exactly: the
   model dispatched three workers, said "Let me record the working hypothesis while
   they run," and ended.

2. **Polling burns the parent's turn budget.** Even when the model does poll, every
   `check_tasks` call costs one parent LLM round-trip. With workers doing ten or twenty
   tool calls of real investigation, the parent exhausts `LoopConfig.max_turns` (32)
   waiting on no-op status snapshots. Second rca trial: 33 parent tool calls, 30 of them
   `check_tasks` returning `running`, no progress on the actual hypothesis.

3. **Findings are not retrievable.** Even after fixing the two above with a blocking
   `check_tasks` and a hand-strengthened prompt, the original `check_tasks` payload
   only returned `{task_id, status, error, final_message_count}` — no actual text from
   the child. The parent had no API to reach into a completed child's output.

The mechanical fix for (3) (add `final_text` to the payload, walked the child's last
assistant message) and (2) (block-on-progress in `check_tasks`) ship today. The
remaining structural problem is (1): an agent that asks a worker for help and then
forgets to listen is a fragile contract.

## Principle: runtime guards the boundary, agent owns the decision

The decision of *when* to consume a child's findings is identical in shape to a Claude
Code session deciding when to stop and ask its user. In both cases:

- The reasoner (LLM) judges sufficiency, branching ambiguity, and commitment cost.
- The runtime does not impose timing.
- The runtime does enforce one floor: don't take an action whose unintended consequence
  is irreversible. For Claude Code, that's "don't `git push --force` without auth."
  For sub-agents, it is "don't `agent_end` while pending child findings are unread."

The minimum-viable contract follows: the runtime guarantees that completed child
findings reach the parent's message stream before the parent is allowed to terminate.
Beyond that floor, *the parent decides*: it can keep working, it can synchronously
wait, it can abort, or it can finalize.

## Mechanism

### One new core event

`core.abi.events.BeforeAgentEndEvent`, mutable, fired immediately before
`AgentLoop.run` declares `agent_end`:

```python
@dataclass(slots=True)
class BeforeAgentEndEvent(Event):
    """Fires after the LLM emits a text-only assistant turn but before the loop
    declares ``agent_end``. Handlers may cancel the exit and append messages to
    keep the loop alive for another turn.

    Mutability: this event is intentionally **not frozen**. Handlers return either
    ``None`` (allow exit) or ``{cancel: True, append: list[AgentMessage]}`` to
    cancel the exit and inject one or more user-role messages into ``messages``
    before the next turn begins. Multiple handlers may cancel; appended messages
    are concatenated in handler-registration order.
    """

    messages: list[AgentMessage]   # the live history; do not mutate in-place
    stop_reason: Literal["end_turn", "max_turns"]
```

`AgentLoop.run` change is roughly ten lines: when the assistant message has no
tool-use blocks, emit `before_agent_end` first; if any handler returned
`{cancel: True, append}`, append the messages and continue the loop instead of
firing `agent_end`. `max_turns` is still respected — handlers cannot extend the
budget; only the natural `end_turn` path is interceptable.

### `sub_agent` extension uses the hook to enforce the floor

`_ChildTaskManager` already tracks `running` / `completed` / `aborted` / `error` per
task. We add a `read` flag (default false) flipped when the parent has consumed a
task's `final_text`, plus a single `before_agent_end` handler:

```
on before_agent_end(messages, stop_reason):
    pending  = tasks where status in {completed, error, aborted} and not read
    running  = tasks where status == running

    if not pending and not running:
        return None                        # allow exit

    notification_blocks = []

    for task in pending:
        notification_blocks.append(format_completed(task))
        task.read = True

    for task in running:
        notification_blocks.append(format_pending(task))

    if not notification_blocks:
        return None

    return {
        "cancel": True,
        "append": [user_message(notification_blocks)],
    }
```

Two structured XML-ish blocks per task. They live inside one synthesized user
message so the kernel sees them as a single conversational turn:

```xml
<subagent_result task_id="abc123" purpose="Map service topology">
... worker's final text ...
</subagent_result>

<subagent_pending task_id="def456" purpose="Forensic on ts-auth-service" />
```

The parent is now woken up with: completed siblings' findings as plain content,
plus a list of still-running children for it to decide about. The next LLM call
sees them in the message history; it picks: write the final report (ignore
pending), call `wait_subagent(task_id)` to block on a specific one,
call `check_tasks` to block until any one progresses, or `abort_task` to kill.

### `check_tasks` and `wait_subagent` are explicit waits

Both are tools the LLM may call *during* the loop, when it knows it needs the
answer to make the next move. Distinct from the `before_agent_end` floor, which
fires unconditionally.

- `check_tasks()` blocks until at least one running child changes state (already
  implemented as part of (2)). Returns the full table with `final_text` for
  completed entries; flips their `read` flag.
- `wait_subagent(task_id)` blocks until that specific child reaches a terminal
  state. Returns the same shape as one row of `check_tasks`. Flips `read`.

There is no separate "pop result" tool. Reading via `check_tasks` /
`wait_subagent` consumes the read flag the same way `before_agent_end` does;
findings cannot be delivered twice.

### `dispatch_agent` is unchanged

It remains async and returns `{task_id, status: running, purpose}` immediately.
Already supports `subagent_type` resolution against persona files. The contract
"dispatch and forget — runtime delivers" is the new capability; the tool itself
needs no signature change.

## Three usage patterns, one mechanism

| Pattern             | What the LLM does                              | What the runtime does                                                                  |
|---------------------|------------------------------------------------|-----------------------------------------------------------------------------------------|
| Fire and forget     | Dispatch, then continue with own SQL / notes    | At end of turn-stream, `before_agent_end` injects every completed child's `final_text` |
| Explicit wait       | Dispatch, next turn `wait_subagent(id)`         | Blocks the tool call; returns final text on completion                                  |
| Steered fan-out     | Dispatch N, loop `check_tasks` reading as ready | Each `check_tasks` blocks until one more makes progress; `inject_instruction` to steer  |

The same `_ChildTaskManager` registry, the same `read` flag, the same
notification format. No mode switch.

## State machine

```
running ──(child finishes normally)──> completed (read=False)
running ──(child raises)──────────────> error     (read=False)
running ──(parent abort_task)─────────> aborted   (read=False)

(any terminal, read=False) ──(check_tasks / wait_subagent / before_agent_end)──> read=True
```

The `read` flag is the single source of truth. It is never reset; once a task's
findings have been delivered to the parent, they will not be redelivered.

## Failure modes and edge cases

- **Repeated `before_agent_end` cancels.** If the parent emits text after
  consuming notifications and tries to end again, `before_agent_end` fires
  again. Pending now empty (we just flipped them read), running may still
  hold workers. The risk is a cancel-loop: LLM trained to not repeat itself
  emits empty text → handler cancels → next turn LLM emits empty text again →
  same cancel.
  Decision: the LLM gets **one** "still-running notice" cancel. The
  `_ChildTaskManager` keeps a `_running_only_cancels: int` counter,
  incremented when the handler cancels solely because of running-no-pending,
  reset to zero when the parent does anything else (tool call, non-empty
  text). When the counter would hit 2, the handler instead **auto-aborts
  every running child** (calling `abort_task` for each), waits for their
  finalizers (bounded by the existing 5s grace), records aborted summaries
  via the same notification block, and **does not cancel** — the loop exits.
  This makes "agent gives up on pending workers" reachable in finite turns
  without a new tool, and turns the ambiguous second cancel into an explicit,
  loud abort visible in trajectory.

- **`max_turns` reached.** `before_agent_end.stop_reason == "max_turns"`. The
  hook still fires so findings are not silently lost, but the loop terminates
  after handler runs regardless of `cancel`. Equivalent to "you ran out of
  budget; here's what your workers had said." The handler still flips `read`.

- **Child crashes during dispatch.** `_run_child` catches and sets
  `status=error, error=<str>`. `before_agent_end` reports it like any other
  terminal state; the parent decides whether to retry or skip.

- **Parent shutdown with running children.** Existing
  `on_session_shutdown` grace logic (5 s) is unchanged. `before_agent_end` runs
  before shutdown, so well-behaved sessions never reach the grace cutoff with
  unread completions.

- **Two extensions both want to cancel `agent_end`.** `cost_budget` (e.g.) might
  later use the same hook. Multiple handlers' appends are concatenated; the
  kernel checks `cancel` as `any(handler returned cancel=True)`. Order is
  registration order — same as every other event. No special arbitration.

- **Dispatch from inside a child.** Allowed today (each session has its own
  sub_agent install). Each manager only sees its own children. Nested cancels
  follow the same rules at each level.

- **Handler raises.** EventBus already swallows handler exceptions (logs;
  contributes `None`). A buggy handler cannot prevent loop termination — the
  same fail-open behavior as every other event channel.

## What is *not* in this design

- **Auto-sleep / polling cadence.** No timers, no exponential backoff. The
  registry's terminal-state notifications are inherently
  edge-triggered through `_run_child.finalize` → `_ChildTask.task` future.
  `check_tasks` / `wait_subagent` await directly on those futures.

- **Re-delivery / dead-letter.** A finding consumed by `before_agent_end` and
  not actually attended to by the LLM is on the agent — exactly as a user-side
  message that gets ignored is on the agent.

- **Synchronous dispatch.** Considered (Claude Code shape); rejected because
  AgentM's kernel runs `tool_calls` sequentially (`loop.py:340 for tc in
  tool_calls`), so synchronous dispatch would lose parallel fan-out without
  also adding parallel tool execution. Kept as a follow-up if and when the
  kernel grows that primitive.

- **Cross-extension result routing.** The notification block is a literal
  user-role message. Extensions that want richer routing (e.g. inject only
  into a specific tool_use_id's apparent tool_result) are not supported;
  Anthropic's tool_use → tool_result pairing forbids late binding to a
  pre-existing id.

## Compatibility and migration

- `dispatch_agent` signature unchanged.
- `check_tasks` payload gains `final_text` (already shipping); blocking
  semantics now load-bearing rather than optional. Tests that asserted
  immediate return must adapt.
- `inject_instruction`, `abort_task` unchanged.
- `BeforeAgentEndEvent` is additive — extensions that don't register a
  handler observe today's behavior.
- `LoopConfig.max_turns` semantics unchanged. The hook does not bypass it.
- Scenario manifests do not need to opt in. Loading `sub_agent` enables the
  hook automatically.

## Effort estimate

- core: `BeforeAgentEndEvent` + ten lines in `AgentLoop.run`
- `sub_agent`: ~50 lines (read flag + handler + `wait_subagent` tool)
- prompt updates: trim rca/orchestrator.md polling section to match the new
  pattern table

Total: roughly 100–150 lines of code plus this design doc. No data migration.
