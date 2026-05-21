# 2026-05-21 — Resume-without-prompt

## Problem

`agentm --resume <sid>` requires a positional `prompt` today, so harness-
driven flows (e.g. `llmharness.replay.reminder_seed`) that supply the first
message via `Inject` on `DecideTurnActionEvent` cannot resume a session
without the user typing a synthetic placeholder. The CLI guard fires
before any extension can speak up.

This bites any event-driven / scheduled / continuation atom — not just
llmharness. The fix belongs in agentm core.

## Shape chosen: `AgentSession.tick()` (option b)

Rejected option (a): adding `text: str | None = None` to `AgentSession.prompt`.
The semantics of "no user message + still fire decide before LLM" diverge
sharply from `prompt`'s contract (build-user-message → fire LLM → decide).
Overloading `prompt(None)` to mean "different lifecycle" reads as an
accident, not a contract. A dedicated method is more discoverable, doesn't
risk a silent regression for embedders calling `prompt(text)` with an
empty-string sentinel, and keeps the `prompt` signature stable for any
downstream caller that constructed it from typing.

`tick()` semantics:

1. Drain pending user-message queue (harmless; preserves `send_user_message`).
2. Build live context + system prompt; fire `BeforeAgentStartEvent` so atoms
   can still veto / replace system as they do for `prompt`.
3. Emit `AgentStartEvent` then a synthetic `DecideTurnActionEvent` whose
   default action is `Stop(NoPendingInput())` (a new non-final cause).
4. Resolve the decision via the same `_resolve_action` lattice the kernel
   already uses.
   * If `Inject(messages=[...])` — append the injected messages to the
     working context and hand off to `AgentLoop.run` with the same model /
     tools / system. The loop appends the next assistant turn, decides
     again, and either stops naturally or continues.
   * Else — emit `AgentEndEvent` with `NoPendingInput()` and return the
     unchanged messages. No user message is persisted.

`NoPendingInput` is non-final so future extensions can still override the
default via `Inject` (matches reminder_seed's existing assumption).

## CLI surface

* `prompt` becomes optional (default `""`).
* After typer parses, the rule is: `prompt` OR `resume` OR `continue_recent`
  must be truthy. Otherwise raise `typer.BadParameter` with the existing
  helpful message.
* When `prompt` is empty and `(resume or continue_recent)` is set, the CLI
  calls `session.tick()` instead of `session.prompt(...)`. Everything else
  (final-output print, session-id footer, exit code) is unchanged.

## Out of scope

* No `AgentLoop.run` signature change. The synthetic decide-first cycle
  lives in `AgentSession.tick`; the loop itself only learns about it via
  the existing `decide_turn_action` channel.
* No ABI Protocol for `AgentSession.tick` — there is no Protocol for
  `AgentSession` in `core.abi` today (only the concrete class). Adding one
  is a separate, larger discussion.

## Tests added

* `tests/unit/test_resume_without_prompt.py`
  * CLI: prompt OR resume → ok; neither → BadParameter.
  * `tick()` no-injector → returns unchanged messages, emits AgentEnd with
    `NoPendingInput`, persists no new entry.
  * `tick()` with an Inject handler → injected user message lands, loop
    runs one assistant turn, no `NoPendingInput` cause surfaces.
* `contrib/extensions/llmharness/tests/test_replay_cli_resume.py`
  * Drives a tiny persisted session through the emitted resume command
    (in-process, stub provider), asserts `REMINDER_DELIVERED` entry written
    and one assistant turn ran.

## Addendum 2026-05-21 (post-review)

Corrections to "Tests added" above. The shipped tests deviated from the
plan and review surfaced both gaps; recording the reality here rather
than rewriting the original section keeps the design log honest.

* `contrib/extensions/llmharness/tests/test_replay_cli_resume.py` was
  **not** shipped. The equivalent integration was instead written as
  `contrib/extensions/llmharness/tests/test_reminder_seed_tick_integration.py`,
  which mounts `llmharness.replay.reminder_seed` on a stub-provider
  `AgentSession` and calls `tick()` directly (no subprocess, no CLI
  argv plumbing). The integration boundary it pins — that
  `reminder_seed` survives the tick semantics and persists
  `REMINDER_DELIVERED` after one assistant turn — is the load-bearing
  invariant from the harness side. The agentm-side CLI dispatch contract
  is pinned by the dispatch-spy tests below; together they cover the
  same ground as the proposed subprocess test without the stub-HTTP
  server scaffolding.
* `tests/unit/test_resume_without_prompt.py` gained three CLI
  **dispatch** tests (post-review fix): they drive `agentm.cli.run`
  with a session-spy patched in via `AgentSession.create` and assert
  which of `prompt` / `tick` was called for `resume`-only,
  `resume + prompt`, and fresh-`prompt` invocations. The original
  arg-parsing test only checked the guard's negative — it would have
  silently passed a regression that always routed through `tick`.

## Addendum 2026-05-21 (post-review, naming)

`_resolve_action` and `_default_action` in `core/abi/loop.py` were
promoted to public names (`resolve_loop_action`,
`default_loop_action`) and re-exported via `__all__`. The
``AgentSession.tick`` caller in this plan was the second external
caller, and a leading underscore stops carrying its meaning at that
point. The in-module `AgentLoop._dispatch_decision` caller and the
existing `test_decide_turn_action.py` were updated accordingly.
