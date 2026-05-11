# Plan: Gateway command layer + P0 hardening

Date: 2026-05-11
Branch: `feature/gateway-command-layer`
Designs: `designs/gateway-channels.md`, `designs/command-routing.md`

Two PRs. **This plan covers PR #1.** PR #2 (atom-as-command + runtime
scenario overlay + MANIFEST `mountable_via_command`) is locked in
`command-routing.md` §8 and gets its own plan file after PR #1 merges.

## Scope (PR #1)

### P0 hardening
1. `ChannelManager._init_channels`: raise `SystemExit` when a channel
   has `allow_from: []` (currently warns at runtime — bot stays online
   but never responds, hostile UX). Mirrors nanobot's
   `_validate_allow_from`. Test: instantiating the manager with an
   empty `allow_from` raises.
2. O(1) approval routing. Gateway gains
   `_approval_index: dict[approval_id, ApprovalBridge]`.
   `ApprovalBridge` registers/unregisters on create/resolve; the
   gateway's `_dispatch` queries the index first and falls back to
   broadcast on miss. The bridge needs a reference to the index in
   its ctor; gateway passes `self._approval_index`. Test: a click on a
   pending approval id resolves the right bridge directly without
   broadcasting.

### Command layer skeleton

Files under `contrib/channels/src/agentm_channels/commands/`:

- `__init__.py` — package
- `protocol.py` — `CommandHandler` Protocol, `CommandInvocation`,
  `CommandResult`, `CommandContext` dataclasses
- `registry.py` — discovery: builtin scan + markdown scan + skill scan
  + entry-point plugin scan (`agentm_channels.commands` group)
- `router.py` — `CommandRouter.try_dispatch(InboundMessage, ctx) →
  CommandResult | None`; parses, locates, runs, formats error replies
- `markdown_command.py` — `MarkdownPromptCommand` class wrapping a
  markdown file; `$ARGUMENTS` substitution
- `skill_command.py` — `SkillCommand` class; `_walk_skill_dirs()` for
  discovery; `_render_prompt(skill, args)` for expansion
- `builtins/__init__.py`
- `builtins/new.py` — `/new`: drop current route (await
  `ctx.drop_route()`), reply "session reset"
- `builtins/end.py` — `/end`: drop route + remove from
  `ChatSessionMap`, reply "session closed"
- `builtins/status.py` — `/status`: show
  `route_key`, current `session_id` (via `get_route_stats`), pending
  approval count
- `builtins/help.py` — `/help`: render the registry grouped by source

### Gateway wiring

`gateway.py` changes:

1. Construct `CommandRouter` in `Gateway.__init__`; expose
   `_approval_index` and pass it to each `ApprovalBridge` ctor.
2. `_dispatch`: after the `button_value` branch, before
   `_get_or_create_route`, dispatch through the router. Control
   results return early; prompt results rewrite `msg.content` via
   `dataclasses.replace`.
3. `_get_or_create_route`: pass `_approval_index` into the new
   `ApprovalBridge`. (Existing constructor signature changes —
   ApprovalBridge gains a required `index` kwarg.)
4. Helper `_command_context_for(route, msg)` builds a
   `CommandContext` per dispatch.

### Tests

Fail-stop only; do not exhaust framework guarantees.

- `tests/test_approval.py` extension: O(1) routing path is taken on
  hit; broadcast still works on miss.
- `tests/test_manager_allow_from.py` (new): empty `allow_from` →
  `SystemExit`.
- `tests/test_command_router.py` (new):
  - parsing: `/new`, `/skill:foo bar baz`, `/`, `/unknown`, `//path`
  - unknown command → user-visible reply, no LLM call (assert on stub channel)
  - `/new` calls `ctx.drop_route()` and does not reach the agent
  - skill command expands the body into the prompt that reaches the agent
  - markdown command expands `$ARGUMENTS`

### Docs

- Update `contrib/channels/README.md` with a "Commands" section
  listing builtin commands, markdown command location, skill
  expansion behavior, and a note that atom commands land in PR #2.

## Out of scope (PR #2 / future)

- `AtomCommand` + MANIFEST `mountable_via_command` + runtime scenario
  overlay + `/atom:*` commands → PR #2
- Sticky skills (`--pin`), filesystem rescan, namespace aliases → P2
- `/approve` / `/deny` text fallback → small PR after PR #1, blocked
  on `ApprovalBridge.try_resolve_text`
- Streaming `send_delta` → separate plan, blocked on AgentM SDK
  exposing token-delta events
- Group `respond_when: mention` policy → batch 3
- Bounded outbound queue + drop policy → batch 3

## Acceptance

1. `uv run --package agentm-channels pytest contrib/channels/tests/`
   passes (existing + new tests).
2. `uv run ruff check contrib/channels/src/` clean.
3. `uv run mypy contrib/channels/src/agentm_channels` clean (targeted
   `# type: ignore` permitted for duck-typed handler args).
4. Manual smoke: a stub-channel session can run `/new`, `/help`,
   `/status`, and a skill expansion end-to-end without touching a
   real LLM.

## Index updates

`gateway_channels` and `command_routing` concepts added in
`.claude/index.yaml`; cross-reference from `pluggable_architecture`
not added (those edges flow the other way — channels know about the
SDK, SDK does not know about channels).
