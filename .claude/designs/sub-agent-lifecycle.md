# Sub-Agent Lifecycle

Status: implemented (PR #62 lifecycle floor; PR #65 sum-type loop redesign)
Owner: subagent + orchestration

Reaches into: `core/abi/events.py`, `core/abi/loop.py`, `extensions/builtin/sub_agent.py`,
`harness/events.py` (re-export only). The lifecycle floor is implemented as a
`decide_turn_action` handler returning `LoopAction.Inject(messages=[...])` — see
[agent-loop.md](agent-loop.md) for the per-turn termination protocol.


## Issue #87 C18 file-shape decision

`extensions/builtin/sub_agent.py` stays a single builtin atom file in issue #87.
The next split review threshold is **1500 LOC or more**; until then the §11
single-file atom contract is stronger than a cosmetic decomposition. The atom
now spawns children through `ExtensionAPI.spawn_child_session(**kwargs)` rather
than constructing harness session config objects directly.

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

### Hook into the per-turn decision protocol

The lifecycle floor lives on `decide_turn_action` — the single per-turn
override channel defined in [agent-loop.md](agent-loop.md). Each turn, the
kernel computes a default `LoopAction` from the assistant message and any
tool outcomes, fires `DecideTurnActionEvent`, and resolves the result via the
`Inject > Stop > Step` lattice. To keep the loop alive when there are
unconsumed sibling findings, the handler returns
`Inject(messages=[user_msg])`; messages from concurrent handlers are
concatenated in registration order.

```python
# core/abi/events.py — abridged
@dataclass(slots=True, frozen=True)
class TurnObservation:
    turn_index: int
    assistant_message: AssistantMessage | None
    tool_outcomes: list[ToolOutcome]
    default_action: LoopAction          # what the kernel would do if no handler runs

@dataclass(slots=True, frozen=True)
class DecideTurnActionEvent(Event):
    observation: TurnObservation

# Handler signature: ``LoopAction | None``. ``None`` = no opinion.
# ``Inject(messages=[...])`` keeps the loop alive AND seeds the next turn.
# ``Stop(cause)`` opts to terminate; the kernel honors only causes whose
# ``final`` flag is False (final causes shadow all overrides).
```

### `sub_agent` extension uses the hook to enforce the floor

`_ChildTaskManager` tracks `running` / `completed` / `aborted` / `error` per
task plus a `read` flag (default `False`) flipped when the parent consumes a
task's `final_text`. The `decide_turn_action` handler:

```
on decide_turn_action(observation):
    # Only intervene when the kernel was about to stop voluntarily.
    if not isinstance(observation.default_action, Stop):
        return None
    if not isinstance(observation.default_action.cause, ModelEndTurn):
        return None                       # final causes / tool-driven stops untouched

    pending  = tasks where status in {completed, error, aborted} and not read
    running  = tasks where status == running

    if not pending and not running:
        return None                       # allow exit

    notification_blocks = []
    for task in pending:
        notification_blocks.append(format_completed(task))
        task.read = True
    for task in running:
        notification_blocks.append(format_pending(task))

    if not notification_blocks:
        return None

    return Inject(messages=[user_message(notification_blocks)])
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
answer to make the next move. Distinct from the `decide_turn_action` floor,
which fires unconditionally on every voluntary `Stop(ModelEndTurn)`.

- `check_tasks()` blocks until at least one running child changes state (already
  implemented as part of (2)). Returns the full table with `final_text` for
  completed entries; flips their `read` flag.
- `wait_subagent(task_id)` blocks until that specific child reaches a terminal
  state. Returns the same shape as one row of `check_tasks`. Flips `read`.

There is no separate "pop result" tool. Reading via `check_tasks` /
`wait_subagent` consumes the read flag the same way the lifecycle floor does;
findings cannot be delivered twice.

### `dispatch_agent` is unchanged

It remains async and returns `{task_id, status: running, purpose}` immediately.
Already supports `subagent_type` resolution against persona files. The contract
"dispatch and forget — runtime delivers" is the new capability; the tool itself
needs no signature change.

## Three usage patterns, one mechanism

| Pattern             | What the LLM does                              | What the runtime does                                                                       |
|---------------------|------------------------------------------------|----------------------------------------------------------------------------------------------|
| Fire and forget     | Dispatch, then continue with own SQL / notes    | On `Stop(ModelEndTurn)`, `decide_turn_action` Injects every completed child's `final_text`  |
| Explicit wait       | Dispatch, next turn `wait_subagent(id)`         | Blocks the tool call; returns final text on completion                                       |
| Steered fan-out     | Dispatch N, loop `check_tasks` reading as ready | Each `check_tasks` blocks until one more makes progress; `inject_instruction` to steer       |

The same `_ChildTaskManager` registry, the same `read` flag, the same
notification format. No mode switch.

## State machine

```
running ──(child finishes normally)──> completed (read=False)
running ──(child raises)──────────────> error     (read=False)
running ──(parent abort_task)─────────> aborted   (read=False)

(any terminal, read=False) ──(check_tasks / wait_subagent / decide_turn_action Inject)──> read=True
```

The `read` flag is the single source of truth. It is never reset; once a task's
findings have been delivered to the parent, they will not be redelivered.

## Failure modes and edge cases

- **Repeated `Inject` keep-alives.** If the parent emits text after consuming
  notifications and tries to end again, the handler fires again. Pending now
  empty (we just flipped them read), running may still hold workers. The risk
  is an Inject-loop: LLM trained to not repeat itself emits empty text →
  handler Injects → next turn LLM emits empty text again → same Inject.
  Decision: the LLM gets **one** "still-running notice" Inject. The
  `_ChildTaskManager` keeps a `_running_only_cancels: int` counter,
  incremented when the handler Injects solely because of running-no-pending,
  reset to zero when the parent does anything else (tool call, non-empty
  text). When the counter would hit 2, the handler instead **auto-aborts
  every running child** (calling `abort_task` for each), waits for their
  finalizers (bounded by the existing 5s grace), Injects the aborted
  summaries one last time, and lets the kernel exit on the next turn. This
  makes "agent gives up on pending workers" reachable in finite turns without
  a new tool, and turns the ambiguous second Inject into an explicit, loud
  abort visible in trajectory.

- **`max_turns` reached.** The kernel terminates with
  `Stop(MaxTurnsExhausted())`. `MaxTurnsExhausted.final` is `True`, so per
  the resolution lattice the handler's `Inject` would be ignored — but
  `decide_turn_action` still fires so observers see the termination. The
  notification block is therefore *not* injected for `max_turns`; pending
  findings stay marked unread for the next session to inherit (or for an
  out-of-band consumer). If the scenario must surface them, the orchestrator
  should drain `check_tasks` / `wait_subagent` *before* the budget runs out.

- **Child crashes during dispatch.** `_run_child` catches and sets
  `status=error, error=<str>`. The Inject notification reports it like any
  other terminal state; the parent decides whether to retry or skip.

- **Parent shutdown with running children.** Existing
  `on_session_shutdown` grace logic (5 s) is unchanged. The lifecycle floor
  runs before shutdown so well-behaved sessions never reach the grace cutoff
  with unread completions.

- **Two extensions both want to keep the loop alive.** `cost_budget` (e.g.)
  might later use the same hook. Per the `Inject > Stop > Step` lattice,
  every handler's `Inject` messages are concatenated in registration order
  into one combined `Inject(messages=[...])`. No handler silently wins; no
  special arbitration.

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

- **Re-delivery / dead-letter.** A finding consumed by the lifecycle floor
  Inject and not actually attended to by the LLM is on the agent — exactly
  as a user-side message that gets ignored is on the agent.

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
- The `decide_turn_action` channel is additive at the kernel level —
  extensions that don't register a handler observe today's behavior. Per
  [agent-loop.md](agent-loop.md), the kernel always fires
  `decide_turn_action` exactly once per turn and once per kernel-imposed
  termination, so observers see every transition through one channel.
- `LoopConfig.max_turns` semantics unchanged. `MaxTurnsExhausted.final` is
  `True`, so handler `Inject` returns are ignored at that boundary.
- Scenario manifests do not need to opt in. Loading `sub_agent` enables the
  handler automatically.
