# 2026-05-28 — Session Inbox

Design: [session-inbox.md](../designs/session-inbox.md) (accepted 2026-05-28).
Concept: `session_inbox` in `index.yaml`.

Goal: one first-class `SessionInbox` as the single entry point for every message
reaching the loop (user input + background completion + ticker + monitor + subagent
findings), plus one driver. Unlocks per-tool-call background, auto-backgrounding,
ticker status, agent-defined monitors — all as thin producers on the spine.

## Step 1 — the spine (this iteration)

Scope: build the inbox + unified message-entry + single driver. **Do not touch
`sub_agent`** (its bespoke floor coexists for now — both handlers' `Inject`/`Step`
returns reconcile via `core/abi/loop.py:317`). No `background_exec` / `monitor` yet.
**Do not modify the kernel `AgentLoop` (`core/abi/loop.py`)** — wire everything through
runtime-owned bus handlers (closures over the inbox + session_manager).

### Pieces

- `core/runtime/session_inbox.py` (new): `SessionInbox` (`push` / `drain` /
  `async wait_nonempty` / non-blocking emptiness check), `InboxItem(source, payload,
  dedup_key=None)`, `InboxSource` (str-based, open). `push` with a `dedup_key`
  **replaces** the same-key undrained item (no stacking). asyncio-safe `push` (plain
  list append) + an `asyncio.Event` backing `wait_nonempty`.
- `_render_item(item) -> AgentMessage`: step 1 handles `source="user"` →
  `UserMessage`. Other sources are stubs for later steps.
- One shared drain+persist helper on the session: for each drained item, render →
  append to the messages list **and** `session_manager.append_message(msg)` (injected
  messages must be persisted — same contract as `core/abi/loop.py:609` and
  `session.py:396`).
- Two runtime-owned bus handlers registered by the session (closures, not atoms):
  - **`context`** (turn start, `core/abi/loop.py:466`): drain+persist into the turn's
    message list. **This is the single message-entry point** — it is what lets the
    LLM see pending input on the very next turn.
  - **`decide_turn_action`** (turn end): if the inbox is non-empty, return `Step()`
    to keep the loop alive (the next turn's `context` drains it). This generalizes
    `sub_agent`'s floor. `final=True` causes still hard-win (`loop.py:301`).
- `prompt(text)` collapses to: `inbox.push(InboxItem("user", text))` → `_drive()`,
  where `_drive()` = drain+persist once at entry (so the originating message lands
  before `build_session_context`, no first-turn delay) → before_agent_start → `run`.
- `tick()` collapses to `_drive()` with no push (drains whatever is already queued;
  empty ⇒ `NoPendingInput`). Share the entry logic with `prompt`.
- `send_user_message` → `inbox.push(InboxItem("user", content))`. Delete
  `pending_user_messages` and `_drain_pending_user_messages`
  (`extension.py:415,521-526`; `session.py:118,267,342,426-440`; runtime scope field).
  Behaviour preserved: content still surfaces on the next turn — now via the inbox.

### Driver note

Step 1's "driver" is just `prompt`/`tick`'s internal `run` (run-to-idle). A
**persistent** driver (`while: await inbox.wait_nonempty(); await run()`) that wakes
an idle agent on a late background push is only needed once producers exist — deferred
to step 5 (host). `wait_nonempty` is implemented now but unused by step 1.

### Fail-stop tests (quality over quantity)

- inbox FIFO drain order + `dedup_key` replace semantics
- originating prompt message lands + is persisted (trace parity with pre-change)
- a `send_user_message` issued mid-run is seen on the next turn (per-turn drain)
- `tick`: empty inbox ⇒ `NoPendingInput`; non-empty ⇒ runs
- keep-alive floor: model wants `Stop(ModelEndTurn)` but inbox non-empty ⇒ loop
  continues
- existing `prompt`/`tick`/`send_user_message` tests still pass (regression net)

### Dev loop

`uv run ruff check`, `uv run mypy`, `uv run pytest --tb=short` on touched files.

## Later steps (see design doc)

2. Extract `sub_agent` registry + completion-injection → `core.lib`; route findings
   through the inbox (`source="subagent"`); delete `sub_agent`'s bespoke floor.
3. `background_exec` atom: auto-backgrounding (`asyncio.wait(timeout=60)`) + ticker
   (milestone-driven + sparse heartbeat, `dedup_key` replace) + `check/wait/cancel_background`.
   See the design doc's "Step-3 design decisions": ABI add
   `ExtensionAPI.post_inbox(source, payload, dedup_key)` (`send_user_message` →
   `post_inbox(source="user")`); opt-in atom wraps all `api.tools` transparently
   (overrun → ticket registered in core.lib registry); completion + ticker
   `post_inbox(source="background")` injected by the step-1 context drain while active
   (idle auto-wakeup = step 5); `cancel_background` is the first `registry.cancel`
   caller; `render_item` gains `source="background"`; terminal-from-background
   simplified (deferred to step 5); **existing tool tests must stay green**.
4. `monitor` atom: `schedule_wakeup(delay)` (one-shot asyncio.sleep →
   `post_inbox(source="monitor", dedup_key=monitor_id)`), `create_monitor(watch=…)`
   subscribing to a bus channel (on fire → `post_inbox`), `list_monitors`,
   `cancel_monitor` (per-monitor cancel; MUST NOT touch the shared session
   `signal` — same discipline as step-3 Major 2). `render_item` gains
   `source="monitor"`. **Lifecycle from day 1** (don't repeat step-3 Major 1):
   `SessionShutdownEvent` handler cancels every monitor task + clears
   subscriptions; `post_inbox` wrapped in `ExtensionStaleError` guard (step-3
   Major 3). Per-session in-memory state (transient — restart regenerates, per
   step-1 decision #5). Condition-polling form of `create_monitor` is deferred
   (MVP supports bus-channel subscription only).
5. Long-lived host driver loop + interrupt-and-resume (abort turn via `signal`,
   preserve context, resume with new inbox input); validate on one channel first.

## Progress

- **Step 1 DONE (2026-05-28).** Merged to `feat/session-inbox`: docs `322af8bd`,
  inbox cherry-picked `41bd3beb` (clean; deliberately excludes the 2 unrelated
  llmharness commits that sat between the worktree base 165c0383 and the cherry).
  code-reviewer verdict APPROVE-WITH-NITS (0 blocker / 0 major). Re-verified on the
  new base 3c79fb98: ruff + mypy clean, `tests/unit` 104 passed, inbox+sub_agent 10
  passed. Nits deferred: Nit2 (`_now` function-local import) → fold into step 2;
  Nit1 (tests reach `session._apis` for `send_user_message`) → step 5 once a public
  push handle lands.

- **Step 2 DONE (2026-05-28).** Merged FF to `feat/session-inbox` as `5bd02259`.
  reviewer CLEAN (0 blocker / 0 major; 2 take-or-leave nits: `cancel()` had zero
  callers — step 3 became its first; docstring ratio acceptable as boundary-contract
  doc). Re-verified on HEAD: 7 passed targeted (registry + sub_agent lifecycle);
  ruff/mypy clean.

- **Step 3 DONE (2026-05-28).** Merged FF to `feat/session-inbox` as `685998be`.
  Initial commit `9491700b` → review found 0 blocker / **4 major** (shutdown leak,
  cancel over-cancels under host signal, unhandled producer exceptions, slot
  invariant). Worker fix `52f5c9c6` → re-review found **1 new major** (refusal branch
  itself leaked the inner task in the shutdown race; masked by a cooperative-only
  test stub). Worker fix `685998be` adds bounded-grace-then-cancel + a
  non-cooperative-inner test. Final: 100 passed, ruff/mypy clean. New fail-stop
  tests: signal-isolated cancel under host signal, slot-non-negative, shutdown
  drain, stale-doesn't-crash, double-show-suppressed, wrap idempotency across two
  `agent_start` cycles, non-cooperative-inner refusal terminates. **Lessons carried
  to step 4**: lifecycle (SessionShutdownEvent handler) from day 1; cancel must
  never touch the shared kernel signal; producer paths must guard
  `ExtensionStaleError`.

## Step-order refinement (decided 2026-05-28)

Deleting `sub_agent`'s `decide_turn_action` floor and routing findings through the
inbox is **moved from step 2 to step 5**. Reason: the floor does two jobs —
(a) inject completed-but-unread findings, (b) keep the parent alive while children are
*still running*. The inbox + generalized keep-alive replaces (a), but (b) has no
replacement until the **persistent driver (step 5)** exists — in steps 2–4 `prompt`
is push+run-until-idle, so a parent that voluntarily Stops while children run would
let them be aborted (the exact failure sub-agent-lifecycle.md was written to prevent).

Therefore:
- **Step 2 (revised) = PURE REFACTOR.** Extract a generic background-task registry
  into `core.lib`; re-seat `sub_agent` on it with **no behaviour change** (floor
  untouched, findings still delivered via the floor, inbox not involved). Guarded by
  the existing sub_agent lifecycle/budget tests. Fold step-1 Nit2 here.
- **Step 5 (expanded)** = persistent driver + interrupt-resume + delete the sub_agent
  floor + route findings via `inbox(source="subagent")` + Nit1 public push handle.
