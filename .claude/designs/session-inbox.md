# Session Inbox — One Entry Point for Everything That Reaches the Loop

Status: **accepted (2026-05-28); steps 1-5 implemented — SessionInbox spine, sub_agent findings routed through the inbox, background_exec, monitor (incl. condition-polling #178), persistent driver + interrupt-and-resume; follow-ups #177 (terminate-from-background) and #179 (one-shot CLI idle keep-alive) closed**
Owner: kernel + orchestration

Reaches into: `core/abi/loop.py` (the `decide_turn_action`/`Inject` drain seam),
`core/runtime/session.py` (`prompt`/`tick`, `_drain_pending_user_messages`),
`core/runtime/extension.py` (`send_user_message`), `extensions/builtin/sub_agent.py`
(the existing background prototype), plus two new atoms (`background_exec`, `monitor`).

Related: [agent-loop](agent-loop.md) (the per-turn decision protocol this rides on),
[sub-agent-lifecycle](sub-agent-lifecycle.md) (the working prototype of the background
pattern), [nested-session-task](nested-session-task.md), [pluggable-architecture](pluggable-architecture.md)
§3 (the loop is kernel ABI), [extension-as-scenario](extension-as-scenario.md) §11
(atoms are single files).

---

## Problem

We want the agent to gain the runtime affordances a long-horizon agent harness has
(mirroring Claude Code's own tools):

1. **Per-tool-call foreground/background** — the agent decides whether a call blocks
   the turn or runs detached.
2. **Auto-backgrounding** — a foreground call that overruns ~60s is moved to the
   background automatically, returning control to the agent immediately.
3. **Ticker** — a backgrounded unit reports its status to the main agent on an
   interval.
4. **Agent-defined monitor** — the agent registers its own subscriptions and
   scheduled wakeups (cf. `Monitor` / `ScheduleWakeup` / `CronCreate`).

These look like four features. They are not. **Every one of them reduces to a single
question: how does an event that happens between or during turns become a message
inside the agent loop?** Answer that once, well, and the four features become thin
producers on top of it.

### What exists today

- The loop is async but tool execution is **sequential and blocking**: each call is
  `await tool.execute(args, signal=signal)` (`core/abi/loop.py:706`); the turn does
  not end until every call returns.
- The **only** clean seam for getting an out-of-band message into the loop is the
  `decide_turn_action` channel returning `Inject(messages=[...])`
  (`core/abi/loop.py:599-620`, `Inject` at `core/abi/events.py:252`). A handler can
  turn a would-be `Stop(ModelEndTurn)` into `Inject`/`Step`, keeping the agent alive
  and splicing in new messages.
- `send_user_message` queues content (`core/runtime/extension.py:521-526`) drained
  only at **prompt boundaries** (`_drain_pending_user_messages`,
  `core/runtime/session.py:267,342,426`) — not per turn.
- `tick()` (`core/runtime/session.py:305`) is the resume-without-prompt entry: it
  fires `decide_turn_action` with default `Stop(NoPendingInput)` and runs the loop
  only if a handler injects.
- **`sub_agent` already implements the whole background pattern** as a tool set
  (`extensions/builtin/sub_agent.py`): `dispatch` spawns an `asyncio.create_task` and
  returns an immediate `{task_id, status:running}` ticket (`:737-750`);
  `check_tasks`/`wait_subagent` poll (`:752`/`:769`); `inject_instruction` pushes a
  message into a running child (`:789`); `abort` sets a signal (`:812`); and a
  `decide_turn_action` floor refuses to terminate while children have unread findings
  (`:834`). This is the prototype we generalize.
- There is **no timer/ticker anywhere in `core`**; `asyncio.create_task` appears only
  in `sub_agent`.

---

## Two orthogonal concepts (do not conflate)

| | What it is | Existing basis |
|---|---|---|
| **A. Background *work*** | a tool call / subagent running as an `asyncio.Task` while the agent keeps taking turns | `sub_agent` proves it works |
| **B. Out-of-band *signal* injection** | how an event between/during turns becomes a message inside the loop | exactly one path: `decide_turn_action` → `Inject` |

Ticker, task completion, monitor fires, wakeups, **and user input while the agent is
running** are all instances of B. The design failure mode is letting each feature
invent its own way to inject. They must all funnel through one mechanism.

---

## Decision: one entry (`SessionInbox`), one driver

A first-class `SessionInbox` is the single entry point for **every** message that
enters the agent loop — user input, background completion, ticker status, monitor /
wakeup fires, subagent findings, cross-agent messages. There is also a single
**driver** that runs the loop. These two are orthogonal and both unified:

- **Entry** = how a message *gets in* → always `inbox.push`.
- **Driving** = who runs the loop and how a caller learns "this round finished" →
  always one driver; callers either `await` a completion signal or subscribe to the
  bus.

`prompt()` does **not** bypass the inbox or grab its own driver — it is sugar:

```python
class SessionInbox:
    def push(self, item: InboxItem) -> None: ...      # producer side; task/thread-safe
    async def wait_nonempty(self) -> None: ...         # driver side; block while idle
    def drain(self) -> list[InboxItem]: ...            # loop side; take all at a turn boundary

@dataclass
class InboxItem:
    source: InboxSource            # mechanism-level routing enum (see below)
    payload: ...                   # rendered into a message according to `source`
    dedup_key: str | None = None   # producers that supersede their own prior item set this

# Single driver — one per session (a CLI process starts one too).
async def _driver(self):
    while not self._closed:
        await self.inbox.wait_nonempty()   # idle ⇒ block; no CPU / no LLM burn
        await self._loop.run(...)           # run until the loop returns at idle

# prompt() = sugar: push + await this round's agent_end. It does NOT drive.
async def prompt(self, text):
    done = self._completion_waiter()        # subscribe the next agent_end
    self.inbox.push(InboxItem(source="user", payload=text))
    await done
    return self.session_manager.get_messages()
```

This keeps: one entry (inbox), one driver (no re-entrant `run`), and CLI/SDK
ergonomics (`messages = await session.prompt(...)` still works). A one-shot CLI does
`start driver → push → await agent_end → close (stop driver) → exit`. A long-lived
host may use `prompt` *or* `inbox.push` + bus subscription — same mechanism, two
usages, not two APIs.

`source` is a **mechanism-level routing enum** — `user | background | ticker |
monitor | subagent | ...` — deciding how the item lands (a `UserMessage`, a synthetic
`tool_result`, or a `<system-reminder>`-wrapped note). It is objective plumbing, not a
subjective classification, so it does not violate the "no preset enums for subjective
fields" rule.

### Drain seam, and the generalized floor

A single runtime-owned handler on `decide_turn_action` (registered by the session, not
an atom — the inbox is substrate) calls `inbox.drain()` at each turn boundary, renders
each item per its `source`, and returns `Inject(messages=[...])`. This reuses the
existing seam (`core/abi/loop.py:599-620`) with **no change to the kernel `AgentLoop`**.

The rule generalizes `sub_agent`'s lifecycle floor: **at a turn boundary, if the inbox
is non-empty, drain + `Inject` and keep running, instead of `Stop`.** "Unread subagent
findings" becomes the special case "a finding landed in the inbox". `final=True`
causes (budget / signal / max_turns) still hard-win over a non-empty inbox — a hard
ceiling should be hard.

Completion-signal boundary: `prompt` awaits the *real* `Stop` + `agent_end` (inbox
empty **and** the model voluntarily ended via `ModelEndTurn`). While the floor keeps
re-running on a non-empty inbox, no `agent_end` fires, so `prompt` does not return
early. Well-defined under low-concurrency single-user (CLI); under high-concurrency
multi-source (long-lived host) "which round is mine" is inherently fuzzy, so callers
there use `inbox.push` + bus subscription.

### Cache discipline

Inbox items land as **append-only new messages**, so the prefix stays stable and the
KV/prefix cache survives. This is distinct from per-turn *reminders*, which rewrite
the tail of the last message and are regenerated each turn — reminders do **not** go
through the inbox.

`send_user_message` becomes a thin `inbox.push(source="user", ...)` wrapper so the
existing ABI and its callers (`sub_agent.inject_instruction`, the channels gateway)
keep working unchanged. `pending_user_messages` is deleted.

### No separate kernel `Park`

The earlier "optional `Park` primitive" (loop awaits the next event internally to stay
alive) is **absorbed by the driver's `wait_nonempty`**: the kernel loop stays pure
("one round runs to idle, then returns"); staying-alive lives in the driver's outer
`while`. When the agent is idle with background work still running, the driver blocks
on `wait_nonempty` and a ticker push wakes it.

---

## Producers (all pure single-file atoms)

### `background_exec` — auto-backgrounding + ticker

Wraps registered tools via `api.tools` mutation (the pattern `tool_filter` /
`file_mutation_queue` already use). Auto-backgrounding needs no coroutine "pause" — it
runs the call in a task from the start and stops *waiting*, not the task:

```python
task = asyncio.create_task(tool.execute(args, signal))
done, _ = await asyncio.wait({task}, timeout=interval)   # interval from atom config, default 60s
if task in done:
    return task.result()                                 # finished in time → normal return
register(task)                                            # else: leave it running
return ticket                                             # {task_id, status:"running", note:"moved to background"}
```

The ticket is the immediate `tool_result` for that call (satisfying the "every
tool_call gets a result this turn" protocol); the real result arrives later as an
inbox item (`source="background"`). Companion tools `check_background` /
`wait_background` / `cancel_background` are the generalization of `sub_agent`'s polling
tools.

**Ticker policy (resolved #2): milestone-driven + sparse heartbeat fallback.** The
ticker pushes immediately on a milestone (completion / error / new output / a
silence-too-long warning); routine "still running" progress is left to the agent's own
`check_background` pull. A sparse heartbeat fallback (e.g. one "still alive" every few
minutes of no activity) bounds the worst case. Every ticker item carries a `dedup_key`;
`push` **replaces** the same-key undrained item rather than stacking, so a stuck-in-a-
long-turn agent never finds a pile of stale status lines.

**Step-3 design decisions (2026-05-28).**
- **ABI `ExtensionAPI.post_inbox(*, source, payload, dedup_key=None)`** is the generic
  producer entry. `send_user_message` becomes `post_inbox(source="user", …)`;
  background_exec / monitor / the step-5 sub_agent rewrite all post through it.
  (ExtensionAPI gains a method only because a real producer now needs it.)
- **Wrapping scope.** background_exec is opt-in (a scenario lists it). At install it
  wraps every tool in `api.tools` with a transparent auto-bg shim:
  `asyncio.wait({task}, timeout)` (config, default 60s). Fast tools finish within the
  timeout and return normally — **no behaviour change; existing tool tests must stay
  green** — only an overrun returns the ticket. Optional `denylist` config (default
  empty) excludes tools that must never background.
- **Terminal tools.** A foreground completion (<timeout) returning `ToolTerminate`
  works unchanged. A *backgrounded* tool that ultimately returns `ToolTerminate`
  posts its completion with `InboxItem.terminal=True` (#177): the `context` drain
  seam records the terminate cause and the keep-alive floor returns
  `Stop(ToolTerminated)` once the message is delivered — symmetric with a
  foreground terminate, never a silent drop.
- **Completion + idle boundary.** Completion and ticker items
  `post_inbox(source="background")`; while the agent is actively taking turns the
  step-1 `context` drain injects them. **Idle auto-wakeup is step 5** (the persistent
  driver) — in step 3 the agent stays informed by being active or by calling
  `wait_background`. `cancel_background` is the first caller of
  `BackgroundTaskRegistry.cancel`.
- **render_item** handles all four sources: `"user"` (plain `UserMessage`),
  `"background"` / `"monitor"` / `"subagent"` (`<system-reminder source="...">`-wrapped
  `UserMessage`). Any other source raises `NotImplementedError`.

### `monitor` — agent-defined subscriptions and wakeups

Tools `schedule_wakeup(delay)` (one-shot timer → inbox push), `create_monitor(watch=…)`
(subscribe a bus channel or poll a condition, push on fire), `list_monitors`,
`cancel_monitor`. Each monitor is just another inbox producer. Maps to `ScheduleWakeup`
/ `Monitor` / `CronCreate`.

### `sub_agent` refactor — sit on the shared substrate

`sub_agent`'s background-task registry + completion injection are extracted into
`core.lib` (§11 forbids atom→atom imports, so the shared code is a non-atom utility,
not a peer atom). Both `background_exec` (unit = a tool coroutine) and `sub_agent`
(unit = a child session) sit on it. Its bespoke `decide_turn_action` floor is **deleted**
in favour of the generalized inbox drain rule (findings push to the inbox with
`source="subagent"`). **Subagents get loop/monitor for free**: a child is a full session
loading the same atoms, so `background_exec` + `monitor` apply recursively.
`inject_instruction` is already the `SendMessage` analog.

---

## Interrupt vs cooperative pending (resolved #1)

- **Default (no interrupt key)**: cooperative turn-boundary pending. Input added
  mid-turn queues in the inbox and is consumed at the next turn boundary; latency =
  the remaining turn time. Accepted.
- **Interrupt key**: Claude-Code-style **interrupt + resume-with-new-input** — abort
  the current turn (existing `signal` path, `core/abi/loop.py:440,667`), keep the
  conversation context, push the new input as a `source="user"` inbox item, and let
  the driver re-run. The inbox and `signal` cooperate: `signal` preempts the in-flight
  turn; the inbox carries the new input into the next run. (Caveat to handle in step 5:
  a turn aborted mid-flight may have already appended a partial assistant message — the
  host must leave the session in a clean, resumable state.)

---

## Resolved decisions (2026-05-28)

1. **Cooperative pending + interrupt-and-resume.** Default is turn-boundary
   cooperative; the interrupt key aborts the turn (via `signal`), preserves context,
   and resumes with the new input through the inbox.
2. **Ticker = milestone-driven + sparse heartbeat fallback, with `dedup_key` replace.**
3. **One entry + one driver; `prompt` is sugar.** No dual API, no separate `Park`; the
   `sub_agent` floor generalizes to "inbox non-empty ⇒ keep running".
4. **FIFO, no priority.** Drain takes all and injects them as one batch ordered by
   arrival ("earlier-happened, earlier-said"); user vs ticker priority adds nothing
   when the agent sees the whole batch in one turn. No config knob.
5. **Undrained-window persistence: MVP none.** Inbox items are mostly transient signals
   (ticker/monitor) that should regenerate after a restart; the user-input window
   before drain is tiny. Items enter the session log only once drained and landed as
   messages (existing `message_persisted`). Marked TODO.

---

## Implementation order

1. **Spine** (step 1): `SessionInbox` + per-turn drain handler
   (runtime-owned, on `decide_turn_action`, with the generalized "non-empty ⇒ Inject"
   floor); single driver; `prompt`/`tick` collapse to push + drive; `send_user_message`
   becomes an `inbox.push` wrapper; delete `pending_user_messages`. **Done.**
2. Extract `sub_agent`'s registry + completion-injection into `core.lib`; re-seat
   `sub_agent` on it; route findings through the inbox and delete its bespoke floor.
   **Done** — `sub_agent` now posts via `post_inbox(source="subagent")`.
3. `background_exec` (auto-background + ticker) on the shared substrate. **Done** —
   `extensions/builtin/background_exec.py` posts via `post_inbox(source="background")`.
4. `monitor` (schedule_wakeup / create_monitor) on the inbox. **Done** —
   `extensions/builtin/monitor.py` posts via `post_inbox(source="monitor")`.
5. Long-lived host driver loop + interrupt-and-resume; validate on one channel first.
   **Done** — the persistent driver + `prompt`/`tick` sugar + interrupt landed
   with #176. Follow-ups closed: terminate-from-background routes through
   `Stop(ToolTerminated)` via `InboxItem.terminal` (#177); `create_monitor`
   gained the condition-polling form (#178); the one-shot `agentm -p` CLI waits
   `AgentSession.idle()` (driver parked + inbox empty + no tracked background
   unit) before exiting so late completions are delivered, not dropped (#179).
