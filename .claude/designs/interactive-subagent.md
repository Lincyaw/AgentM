# Interactive Subagent

**Status:** IMPLEMENTED (core mechanism, 2026-06-16). Model A ("collaborative
side-channel") chosen over takeover/dual-drive. The gateway routing + sub_agent
keep-alive + interactive child tabs are landed; the sidebar reorganization
(§"Implementation surface" item 3) is deferred UI polish — children remain in
switchable tabs for now, but each tab is now fully interactive.

## What landed

- **`gateway/child_registry.py`** — `ChildSessionRegistry`, an in-memory
  `session_id -> child AgentSession` map seeded as the `child_session_registry`
  service onto every session (`SessionManager`).
- **`GatewayRuntime.handle_inbound`** — an inbound whose `session_key` names a
  registered child is delivered to that child's `SessionInbox`
  (`source="user"`) via `_deliver_child_input`, or interrupts it
  (`_interrupt_child`), instead of the `get_or_create` chat path.
- **`sub_agent`** — registers each spawned child with the registry (by service
  name, §11); in interactive mode (registry present) keeps the child alive on
  finalize for post-task chat, and tears all children down + deregisters on
  parent `session_shutdown`. Outside the gateway (no registry) the legacy
  teardown-on-finalize path is unchanged.
- **Terminal (Go)** — each child tab gets its own wire-backed Controller
  (`NewChildController`) keyed to the child's session id, so a typed message in
  a sub-agent tab routes to that live child. `ChildManager` holds the
  `WireClient` + base `Identity` to build them.

## Goal

Let a human send messages to a *subagent* (a spawned child session) and read its
trajectory — not just watch it run. Today subagents are parent-owned workers
whose trajectory is forwarded onto the parent wire (`child_id`-stamped bodies)
and surfaced as switchable tabs in the Go terminal; the user can observe but not
talk to them.

## Key insight: a subagent is already a full session

Two facts make this far smaller than it first looks:

1. **A subagent *is* a complete `AgentSession`.** `sub_agent` spawns children via
   `api.spawn_child_session(...)` → a real `AgentSession` with its own bus, loop,
   and `SessionInbox` and a stable `session_id`.
2. **`SessionInbox` already unifies input.** An item with `source="user"` renders
   as a real `UserMessage`; `source="subagent"/"monitor"/"background"` render as
   `<system-reminder>` notes. So *posting a `source="user"` item into a child's
   inbox is exactly "the user said something to it."* See [[session_inbox]].
3. **Single-process gateway.** The gateway and every session (main + children)
   live in **one process** ([[single_process_gateway]]). Child session objects
   are reachable in principle; they are simply not *registered* with the gateway
   today (the parent's `sub_agent` atom holds them).

Therefore the feature is **addressing + plumbing**, not a new capability. No new
runtime mechanism is required.

## The unified model

> **Every session (main or child) is registered with the gateway. An inbound is
> routed by `session_id` to that session. Input is always delivered to that
> session's `SessionInbox` (`source="user"`). The loop is run by whoever owns it;
> if no one is running it (idle), the gateway kicks it once.**

This makes three situations one mechanism, with no special cases:

| situation | input delivery | who runs the turn |
|---|---|---|
| main session | inbox `source="user"` | gateway (as today) |
| **live** subagent (parent driving) | inbox `source="user"` | parent's `sub_agent` monitor loop drains it |
| **done** subagent (parent finished) | inbox `source="user"` | gateway kicks the loop (≈ resume) |

### Model A semantics (the three accepted refinements)

- **Caller-agnostic.** A subagent does not care *who* addressed it — main agent or
  human are identical input. So there is **no "user-injection" special path**; the
  human's message and the parent's `inject_instruction` ride the *same* inbox.
- **Parent perceives the human's input.** Because the human's message lands in the
  child's shared trajectory (forwarded `child_id`-stamped onto the parent wire) and
  is drained by the parent-driven loop, the parent naturally sees it. No extra
  notification path is required for MVP.
- **Continue after "death."** A child is not destroyed on task completion; its
  `session_id` persists. Continuing the conversation after the parent is done is
  just **driving that session_id directly** — i.e. the subagent has degraded into
  an ordinary gateway session. This is the user's "it's just a session" made literal.

## The one real subtlety: who drives the loop

A session runs a turn only when *someone* drives its loop. The main session is
gateway-driven; a **live** child is driven by the parent's `sub_agent` monitor
([[sub_agent_lifecycle]]). If the gateway also called `prompt()` on a
parent-driven child, two drivers would race one loop.

`SessionInbox` is exactly the decoupling that avoids this:

- **Input is *always* `inbox.push(source="user")`** — never a direct `prompt()`
  on a session another driver owns.
- Whoever's loop is running drains it: a live child's parent loop picks it up at
  its next turn boundary (and thus the parent perceives it).
- **If the target session has no active turn running, the gateway kicks the loop
  once** so the pushed item is processed. This is the only branch that needs a
  "is a loop currently running?" check.

This is consistent with [[session_inbox]]'s "one entry + one driver" rule: entry
is always the inbox; driving stays single.

## Implementation surface

1. **Gateway — register children, route by session_id.**
   - On child spawn (same process), register the child `AgentSession` in the
     gateway's session table keyed by its `session_id`. Mechanism: a gateway-
     provided child-session registry service (mirrors the existing
     `child_wire_forwarder` hand-off) that the child-session factory calls — the
     `sub_agent` atom reaches it by name, no atom→atom import (§11).
   - `handle_inbound` for a `session_key` that names a registered child →
     `inbox.push(source="user")` on that child; kick its loop iff idle.
   - A **live** child routes to the in-memory instance; a **done** child whose
     in-memory instance is gone falls through to the existing resume path
     (`resolve_session_state(resume=child_session_id)`) — same machinery, no
     special case.

2. **`sub_agent` atom — keep children resumable.**
   - Do not tear the child session down on finalize; leave it registered (or at
     least its transcript persisted) so post-task chat works.

3. **Terminal (Go peer) — address sessions by id.** *(landed on the tab model;
   sidebar reorg deferred.)*
   - **Done:** each child tab owns a wire-backed Controller keyed to the child's
     session id (`NewChildController`), so typing in a sub-agent tab sends an
     inbound with `session_key == child_id` and the gateway routes it to that
     live child. The child's trajectory is already delivered (`child_id`-stamped
     bodies) and rendered in the same tab.
   - **Deferred:** moving subagents out of tabs into a sidebar (main agent alone
     in tabs). The interactive mechanism does not depend on it — a switchable
     child tab is already a working interactive surface — so this is pure UI
     polish, tracked separately.

## Explicitly NOT needed

- **No new wire envelope/field.** An inbound already carries `session_key`; setting
  it to a child's id is the whole addressing story. (No `target_child_id`.)
- **No special "inject user message into child" gateway/atom service.** Reuse the
  inbox `source="user"` path that the main session uses.
- **No parent-`ExtensionAPI` hop.** Direct registration replaces the earlier
  "reach the child through the parent's API" detour that an earlier draft proposed.

## Open decisions for implementation

- **Idle detection.** How the gateway knows a target session has no active turn
  running (to decide kick vs. let-the-owner-drain). Likely a per-session "driver
  active" flag the runtime/driver exposes.
- **Lifecycle/eviction.** How long done children stay registered in memory before
  falling back to resume-from-store; eviction policy for many short-lived children.
- **Concurrency guard.** Ensure the kick path cannot start a second driver on a
  child the parent is mid-turn on (the idle check must be race-safe).

## Relationship to existing concepts

Builds directly on [[session_inbox]] (the input seam) and [[sub_agent_lifecycle]]
(child ownership), routed through [[single_process_gateway]], surfaced by the Go
terminal ([[textual_tui]]). Related to [[nested_session_task]] and [[agent_team]].
