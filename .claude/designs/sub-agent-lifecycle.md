# Sub-Agent Lifecycle

Status: implemented; revised 2026-07-04 to remove parent-side wait/check tools
Owner: sub_agent + session_inbox

Reaches into: `extensions/builtin/sub_agent.py`,
`core/runtime/session_inbox.py`, `core/runtime/session.py`,
`core/lib/background_tasks.py`, `core/abi/session_config.py`.

## Current Contract

`dispatch_agent` starts a child `AgentSession` and returns immediately with:

```json
{"task_id": "...", "child_session_id": "...", "status": "running", "purpose": "..."}
```

The parent does not poll the child and does not synchronously wait for it.
The child is tracked as detached background work with
`ExtensionAPI.track_background()`. When the child reaches a terminal state,
`sub_agent` formats its finding as:

```xml
<subagent_result task_id="abc123" purpose="Map service topology">
  <summary>...</summary>
  <artifacts>
    <ref id="..." kind="..." title="..." />
  </artifacts>
</subagent_result>
```

and posts it to the parent inbox with `source="subagent"`. The persistent
session driver is parked on `SessionInbox.wait_nonempty()` while idle, so the
inbox push is both the notification and the wakeup. The runtime context-drain
handler renders the item as a `system-reminder` user message on the next turn.

There is intentionally no parent-side status surface. That class of tool made
the parent spend model turns doing status management and encouraged synchronous
orchestration. Completion is edge-triggered by the child finalizer instead.

## Principle

Sub-agent work is background work. Background work reports through the inbox.

The parent owns the investigation strategy: it can keep doing independent work,
send a follow-up instruction to a running child with `inject_instruction`, abort
with `abort_task`, or end its current turn and wait for the notification. The
runtime owns delivery: completed child findings are posted once through the
same message-entry path used by background tool completions and monitor fires.

This keeps the SDK mechanism small:

- `dispatch_agent` describes work and starts it.
- `inject_instruction` queues a message for the child's next prompt turn.
- `abort_task` sets the child's cooperative abort signal.
- `SessionInbox` delivers terminal findings.

## State Machine

```
running --(child finishes normally)--> completed --(post inbox)--> delivered
running --(child raises)-------------> error     --(post inbox)--> delivered
running --(parent abort_task)--------> aborted   --(post inbox)--> delivered
```

`delivered` means the finding has been pushed to the parent inbox. Once the
driver drains that item, it is persisted in the session log as a user-visible
system reminder. There is no separate read flag and no pop-result API.

## Failure Modes And Edges

- **Parent ends a turn while child still runs.** The parent session parks. The
  tracked child keeps the one-shot CLI alive through `AgentSession.idle()`, and
  long-lived hosts keep their event loop alive naturally. When the child
  finishes, its inbox push wakes the parent session.
- **Child crashes.** `_run_child` records `status="error"`, includes the error
  in the summary, and still posts a `<subagent_result>`.
- **Parent shutdown with running children.** `on_session_shutdown` waits up to
  the configured grace window, then sets abort signals and awaits cleanup.
- **Multiple children finish close together.** Each child posts with a stable
  per-task `dedup_key`; undrained replacement can only affect the same task.
  Different tasks remain separate inbox items and drain FIFO.
- **Max-turn / budget termination.** Final kernel causes still win for the
  current run. A later child completion enters through the inbox path as long
  as the session is still alive.

## Compatibility And Migration

- `dispatch_agent` still returns `child_session_id` and still does not accept
  parent-supplied budget overrides.
- Legacy parent-side status/check tools are removed from `sub_agent`.
- Scenarios that previously required explicit sub-agent waits should instruct
  the parent to continue independent work or let the session park until the
  `<subagent_result>` notification arrives.
- `inject_instruction` and `abort_task` are unchanged.

## Config Inheritance By Manifest Name

`sub_agent.available_inherited_extensions` may name a parent atom with only a
module path. When the inherited spec omits `config` or supplies an empty config,
`sub_agent` copies the already loaded parent atom's resolved config by manifest
name into the child session. Scenarios can therefore define data-tool limits and
exclusions once on the parent atom while still mounting the same module in
critic/worker sessions.
