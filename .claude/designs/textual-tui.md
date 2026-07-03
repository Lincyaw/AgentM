# Design: Terminal TUI (wire-client live frontend)

**Status**: designing (rewrite for channels-v2; supersedes the in-process
`modes/textual_app.py` design deleted with the channels-v2 migration)
**Created**: 2026-05-01
**Last Updated**: 2026-05-28
**Builds on**: [single-process-gateway.md](single-process-gateway.md),
[observability.md](observability.md), [pluggable-architecture.md](pluggable-architecture.md)

---

## 1. Overview

`agentm-terminal` is the interactive terminal frontend. After the channels-v2
rewrite it is a **wire client** of the single-process gateway, not an in-process
presenter: the `AgentSession` lives inside the `agentm gateway` daemon; the TUI
is a separate process that connects over the wire (`unix://` locally, `ws://` /
`wss://` remote) and renders what the gateway pushes.

It serves **two roles at once** — the dual role the original TUI design defined:

1. **Conversation surface** — streamed assistant text, thinking, inline tool-call
   blocks (collapse-by-default), sub-agent blocks, approval prompts, a multi-line
   input with a slash-command palette, keyboard-first navigation.
2. **Control + observability surface for AgentM's runtime** — the framework's
   headline behavior is *self-modification*, and the runtime fires a rich event
   taxonomy (extension install/reload/unload, atom registration, injected user
   messages, managed-resource writes, compaction, budget). The TUI renders each as
   a header counter, a toast, or an `/<name>` modal — the user's window into what
   the framework is doing on their behalf.

The decisive architectural fact is the **process boundary**: the TUI cannot
subscribe to the session bus (it is in another process). It renders only what the
gateway serializes onto the wire. So "replicate the rich interactive experience"
is fundamentally *"re-expose the bus event stream over the wire, then render it."*

## 2. Motivation

The channels-v2 rewrite collapsed the old in-process Textual app (which subscribed
to `AgentSession.bus` directly) into a thin wire client. In doing so the live
experience was lost: the `wire_driver` atom (the bus→wire bridge) only forwards the
final assistant text on `turn_end`, plus approvals and diagnostics. Streaming
tokens, thinking, tool-call lifecycle, cost/usage, sub-agent activity, and every
control/observability event still fire on the in-process bus but **never cross the
process boundary**.

The goal is to restore (and modernize, modeled on Claude Code's interaction) the
full surface — which requires work in three layers, not just the TUI:
`wire_driver` (translate bus → wire) → gateway sink (deliver) → TUI (render).

## 3. Architecture — three layers across the process boundary

```
 gateway process                                   terminal process
┌───────────────────────────────┐                ┌──────────────────────────┐
│ AgentSession.bus               │                │ WireClient.on_outbound   │
│   stream_delta, tool_call,     │  wire frames   │   → router (metadata.kind │
│   tool_result, extension_*,    │  (outbound     │      → widget mutation)  │
│   api_register, resource_write │   envelopes)   │ → Textual widget tree    │
│        │                       │  ───────────►  │                          │
│   wire_driver (projector) ─────┼─ _emit_outbound│                          │
│        │                       │   ├ ephemeral → direct peer write         │
│   SqliteOutbox  ◄──────────────┼───┴ durable   → outbox.enqueue           │
└───────────────────────────────┘                └──────────────────────────┘
```

- **Layer 1 (server):** `wire_driver` projects bus events into `outbound` bodies;
  the gateway sink routes each by delivery class.
- **Layer 2 (wire):** the existing `outbound` envelope kind carries everything,
  discriminated by `metadata.kind`. No new envelope kind (the wire keeps its
  fixed kind set; adding one needs a spec amendment — see single-process-gateway §2.3).
- **Layer 3 (client):** a dispatch table maps `metadata.kind` → one widget mutation.

## 4. Wire streaming protocol (the contract)

### 4.1 Projector table

`wire_driver` is rebuilt around a **declarative projector table**
`_PROJECTORS: dict[channel, Callable[[Event], dict | None]]`. Each entry maps a bus
channel to a function producing a JSON-safe `outbound` body (or `None` to skip a
particular event). This mirrors the per-event `to_otel` projection pattern already
in `core/abi/events.py`: the *set of surfaced events and their wire shape lives in
one reviewable place*, and adding/dropping an event is a one-line change. The table
**omitting** a channel is the explicit allow-list decision.

Every projected body carries `metadata.kind` (the discriminator), the routing
address (`channel` / `chat_id` / `thread_id`), and `_session_key` (echoed back so a
multi-surface client attributes the frame). Streaming bodies also carry `turn_id`
so the client keys mutations to the right turn.

### 4.2 metadata.kind taxonomy

**(a) Conversation — ephemeral** (drive the transcript live):

| metadata.kind | source channel / payload type | body fields |
|---|---|---|
| `turn_start` | `turn_start` | turn_id, turn_index |
| `stream_text` | `stream_delta`·`TextDelta` | turn_id, text |
| `stream_thinking` | `stream_delta`·`ThinkingDelta` | turn_id, text |
| `tool_call` | `tool_call` | tool_call_id, name, args |
| `tool_result` | `tool_result` / `tool_error` | tool_call_id, ok, duration_ms, preview |
| `usage` | `llm_request_end` (+usage/cost source) | turn_id, tokens_in/out, cost, model, ctx_window |
| `child_start` / `child_end` | `child_session_start/end` | child_id, purpose, error? |

**(b) Control / observability — ephemeral** (toasts, header counters, modal snapshots):

`extension_install` (phase + trigger), `extension_reload` (`is_self_modify`,
old/new hash), `extension_unload`, `api_register` (kind=tool|command → counters +
`/tools`), `api_send_user_message` (injected UserTurn), `resource_write`,
`plan_submitted`, `after_compact`, `cost_budget_exceeded` (budget latch + `/budget`),
`session_ready` (initial tools/commands/model snapshot), `command_dispatched`,
`agent_end` (TerminationCause → why the turn stopped).

**(c) Durable** (must survive a disconnected peer — the reliability floor):

`assistant_text` (`turn_end` — the source-of-truth assistant reply, already
forwarded today), `approval_request` / `approval_resolved`, `diagnostic_error`.

**(d) NOT surfaced** (kernel-internal / persistence / veto / rewrite hooks — no UI
meaning): `before_send_to_llm`, `context`, `decide_turn_action`,
`before_agent_start`, `message_persisted`, `message_appended`,
`session_header_emitted`, `entry_appended`, `before_install_atom`,
`before_unload_atom`, `child_session_extending`, `resolve_subagent`,
`resources_discover`, `before_compact`, `session_shutdown`.

### 4.3 Delivery classes (the load-bearing split)

The gateway sink (`_GatewayRuntime._emit_outbound`) routes by delivery class:

- **Ephemeral (a + b):** encode the envelope and **write it directly to each routed
  peer's `transport_writer`**, best-effort — dropped if the peer is currently
  disconnected. Bypasses the SQLite outbox.
- **Durable (c):** `outbox.enqueue` as today — at-least-once, survives reconnect.

This split is load-bearing: a turn streams ~50 `stream_text` deltas/second; routing
those through the durable outbox would create thousands of SQLite rows per turn and
replay them on reconnect. Ephemeral frames are *live decoration*; the durable
`assistant_text` is the authoritative record a reconnecting client can rely on.

The discriminator is a small set of durable `metadata.kind`s (default = ephemeral),
or an `_ephemeral` flag the atom pops from the body. `wire_driver` stays oblivious
to transport; it only tags delivery class.

## 5. Interaction model (the replica target)

### 5.1 Layout (outside-in dock)

```
┌──────────────────────────────────────────────────────────────────────┐
│ StatusBar: model · ctx 12.3k/200k · in/out · $0.041 · ◐ thinking      │ dock top, h:1
│            N tools · M atoms · $⚠ (budget latch)                       │
├──────────────────────────────────────────────────────────────────────┤
│ Transcript (1fr scroll)                                            ▲   │
│   UserTurn          — user input (primary gutter)                  │   │
│   AssistantTurn[turn_id]                                           │   │
│     ThinkingBlock   — dim, collapsible                             │   │
│     AssistantText   — streamed markdown                            │   │
│     ToolBlock       — name ⟳/✓/✗ 142ms, collapsible args+result    │   │
│     SubagentBlock   — nested child transcript (accent gutter)      │   │
│     ApprovalBlock   — Allow / Allow-always / Deny                  ▼   │
├──────────────────────────────────────────────────────────────────────┤
│ PromptInput (multi-line, rounded, autosize)                            │ dock bottom, h:auto
│ CommandSuggestions (inline autocomplete, shown below input on /…)      │ h:auto
└──────────────────────────────────────────────────────────────────────┘
  Keybindings (⌃C/⌃D/⌃L/⌃Y/⌃E) stay active but are not rendered in a Footer
  hint bar — the input column is the bottom row.
              Toast(s): transient overlay for control/observability events
```

### 5.2 Widget tree

```
AgentMTui(App[int])
├── StatusBar(Static)              # dock top; reactive model/tokens/cost/phase/counters
├── Transcript(VerticalScroll)     # 1fr
│   ├── UserTurn(Static)           # primary gutter; system→you variant for injected
│   ├── AssistantTurn(Vertical)    # keyed by turn_id
│   │   ├── ThinkingBlock(Collapsible)
│   │   ├── AssistantText(Static[Markdown])
│   │   ├── ToolBlock(Collapsible) # generic; Edit/Write→diff, Bash→cmd+out, Read→path
│   │   └── SubagentBlock(Vertical)
│   └── ApprovalBlock(Vertical)
├── input-dock(Vertical)           # dock bottom, h:auto
│   ├── PromptInput(TextArea)       # the prompt
│   └── CommandSuggestions          # inline autocomplete, below input
└── Toast(Static)                  # transient overlay
                                   # no Footer; slash-command help is inline
                                   # autocomplete, not a modal palette
```

### 5.3 Design language

- **Attribution by gutter:** `border-left: thick $variable` per role
  (user / assistant / thinking / tool / subagent / system-injected). Single source
  in CSS — no Rich `Panel` borders (which double up).
- **Glyphs:** phase glyphs (`● idle`, `◐ thinking`, `◑ streaming`, `⚙ tool`,
  `⌥ subagent`); tool status (`⟳` running, `✓` ok, `✗` error).
- **Theme:** Textual built-in `textual-dark` / `textual-light` + CSS variables
  (`$primary`, `$accent`, `$warning`, `$error`, `$text-muted`). No hardcoded hex.
  `--theme dark|light` maps to the built-in names.
- **Markdown** for assistant text and markdownish tool output; plain otherwise.

### 5.4 Router: metadata.kind → widget mutation

A single `router.py` dispatch table is the only place wire shapes meet widgets:

| metadata.kind | mutation |
|---|---|
| `turn_start` | create `AssistantTurn` keyed by turn_id; phase=thinking |
| `stream_text` | append to active turn's `AssistantText` (20 Hz buffered flush) |
| `stream_thinking` | append to active turn's `ThinkingBlock`; auto-collapse on next phase |
| `tool_call` | insert a `ToolBlock` (name + args), status ⟳; phase=tool |
| `tool_result` | fill result + status ✓/✗ + duration; auto-collapse if short+ok |
| `usage` | update StatusBar tokens/cost/ctx |
| `child_start`/`child_end` | open/close a `SubagentBlock`; phase=subagent |
| `assistant_text` | finalize the turn's text (authoritative; replaces stream buffer) |
| `approval_*` | inline `ApprovalBlock` / resolve it |
| `extension_*`, `resource_write`, `after_compact`, `plan_submitted` | toast (self-modify louder) |
| `api_register`, `session_ready` | StatusBar counters + `/tools` `/extensions` snapshot |
| `api_send_user_message` | synthetic `UserTurn` (system→you gutter) |
| `cost_budget_exceeded` | budget latch in StatusBar + `/budget` snapshot |
| `diagnostic_*` | toast (warning/error) |

### 5.5 Controls

| Key | Action |
|---|---|
| Enter | submit (non-empty) |
| Shift+Enter | newline |
| `/` (col 0, empty) | command palette |
| Esc | interrupt in-flight turn (soft cancel); else clear draft; else toast "nothing to cancel" |
| Ctrl+C | cancel in-flight; double-tap within 1.5s → quit |
| Ctrl+D | quit |
| Ctrl+L | clear transcript (visual only; session preserved) |
| Ctrl+R | command palette |
| Ctrl+E | expand/collapse focused ToolBlock |
| Up/Down (empty input) | input history |
| PgUp/PgDn | scroll transcript |

**Interrupt** requires a gateway affordance: an inbound carrying a control marker
that cancels the detached `session.prompt` task (in the gateway's `handle_inbound`).
Until that lands, Esc degrades to a local no-op toast.

### 5.6 Streaming render strategy

Buffer `stream_text` / `stream_thinking` per active turn; a `set_interval(0.05, …)`
worker flushes the buffer into the active `AssistantText` (20 Hz, the same cadence
the old mode used — no per-token repaint). On `assistant_text` (turn_end), force a
final flush and replace the buffered text with the authoritative content.

## 6. Component decomposition (package shape)

Clean rewrite of `contrib/gateway-peers/terminal/src/agentm_terminal/`:

```
cli.py              # typer entry: connect/auth/transport resolution, frontend select
client.py           # WireClient <-> asyncio queues glue (send_inbound, outbound stream)
frontends/
  plain.py          # text/json renderer for non-TTY (scripts/agents) — reorg of renderer.py
  tui/
    app.py          # AgentMTui(App): compose + BINDINGS + lifecycle
    router.py       # metadata.kind -> widget mutation (§5.4)
    state.py        # reactive status + turn registry (keyed by turn_id)
    theme.py        # glyphs, gutter roles, CSS path
    app.tcss        # docks, gutters, theme variables
    widgets/        # transcript turns tools subagent approval status input toast
    modals/         # palette help info
```

Frontend selection reuses the existing TTY autodetect: TUI on a TTY, `text` when
piped, `json` for agent drivers (`--format` overrides). Transport selection reuses
`gateway.client_cli.resolve_connect` (unix / ws / wss already handled).

## 7. Interface definition

```python
# agentm_terminal/client.py
class TerminalClient:
    """Owns the WireClient; exposes an async outbound stream + send_inbound."""
    async def connect(self) -> None: ...
    async def send_inbound(self, body: dict) -> None: ...
    def outbound(self) -> AsyncIterator[dict]: ...   # yields outbound bodies
    async def close(self) -> None: ...

# agentm_terminal/frontends/tui/app.py
class AgentMTui(App[int]):
    def __init__(self, client: TerminalClient, *, sender_id: str, chat_id: str,
                 theme: str = "dark") -> None: ...

async def run_tui(client: TerminalClient, *, sender_id, chat_id, theme="dark") -> int: ...
```

Local default stays `unix://`; `DEFAULT_SOCKET_URL`, `agentm daemon socket`, and
the Go terminal peer default all point at the same per-user local daemon socket.
WS is a first-class explicit-opt-in transport the TUI fully supports.

## 8. Related concepts

- [single-process-gateway.md](single-process-gateway.md) — the wire protocol,
  `wire_driver` service contract, outbox delivery semantics, peer routing.
- [observability.md](observability.md) — the bus events this design surfaces are a
  subset of what observability records; the projector mirrors the `to_otel` pattern.
- [pluggable-architecture.md](pluggable-architecture.md) — `wire_driver` is a §11
  single-file atom; the TUI is a presenter peer, not core.

## 9. Constraints and decisions

| Decision | Rationale | Alternative |
|---|---|---|
| Stream over the existing `outbound` kind via `metadata.kind` | Keeps the fixed wire-kind set (spec-amendment to add a kind) | New per-event envelope kinds — rejected, churns the wire spec |
| Ephemeral vs durable delivery split | ~50 deltas/s would flood the durable outbox; live decoration must not be replayed on reconnect | All-durable (rejected: row explosion) / all-ephemeral (rejected: loses the reliable reply record) |
| Declarative projector table in `wire_driver` | One reviewable allow-list; one-line to add/drop an event; mirrors `to_otel` | Bespoke per-event hooks — rejected: scatters the contract |
| Dual role (chat + control/observability) | Self-modification is the headline behavior and is otherwise invisible | Chat-only TUI — rejected: hides what the framework does |
| Authoritative `assistant_text` replaces the stream buffer | Ephemeral deltas may be lossy; the durable turn_end text is the source of truth | Trust the stream buffer — rejected: a dropped delta corrupts the transcript |
| Clean rewrite of the terminal package | Existing `ui/textual.py` is a minimal placeholder; not worth preserving | Extend in place — rejected by user |
| Local default `unix://`, WS explicit opt-in | Zero-config peer-cred locally; WS first-class for remote | WS-default everywhere — rejected: forces token/anon auth on local |
| 20 Hz batched stream flush | Token rate ~50/s; per-token repaint wastes CPU and flickers | Per-token — rejected |
| Interrupt needs a gateway cancel affordance; degrade gracefully if absent | Cancel must reach the detached prompt task across the process boundary | Make it a hard prerequisite — rejected: decouples TUI landing from the gateway add |

## 10. Acceptance scenarios

| # | Scenario | Expected |
|---|---|---|
| T1 | Type a prompt | UserTurn appears; assistant streams; StatusBar idle→thinking→streaming→idle |
| T2 | Assistant calls a tool returning 200 lines | ToolBlock collapsed `name ✓ 12ms`; ⌃E expands |
| T3 | Mid-stream Esc | stream stops; partial text kept; phase→idle; input refocused |
| T4 | Agent self-modifies (reload, trigger=agent) | `★ self-modify` toast (louder than a human reload) |
| T5 | Sub-agent dispatched | nested SubagentBlock renders the child's stream |
| T6 | Extension injects a user message | synthetic UserTurn marked `system → you` |
| T7 | Peer disconnects mid-turn, reconnects | ephemeral deltas dropped; durable `assistant_text` delivered on reconnect |
| T8 | `/tools` after startup | InfoModal lists registered tools (from `api_register` / `session_ready`) |
| T9 | Piped: `printf '/help\n' \| agentm-terminal` | `json` frames on stdout unchanged (non-TTY) |
| T10 | Connect over `ws://` with token | identical behavior to `unix://` |

## 11. Open questions

1. **`usage` data source.** `llm_request_end` carries duration/chunk_count but not
   token totals; usage likely rides on `MessageEnd.message` / provider usage. Confirm
   the field at implementation before wiring the `usage` projector.
2. **Interrupt wire shape.** Reuse the inbound path with a control marker, or add a
   dedicated control frame? Lean on an inbound marker to avoid a new wire kind.
3. **Tool-specific renderers.** MVP ships a generic ToolBlock + diff/bash/read
   specializations; the full per-tool catalog is a follow-up.
4. **History persistence.** In-memory MVP; file-backed (`~/.local/state/agentm/history`)
   later if requested.

## 12. Implementation phases

- **Phase 0** — this design doc + `index.yaml` concept refresh (docs, self-mergeable).
- **Phase 1** — `wire_driver` projector + gateway ephemeral delivery routing +
  `OutboundMetaKind` extension (server; boundary review).
- **Phase 2** — TUI foundation: package rewrite, design language, base widgets,
  router wired to the durable kinds; WS verified; `renderer.py` → `frontends/plain.py`.
- **Phase 3** — live conversation rendering (category-a frames) + Esc-interrupt.
- **Phase 4** — control/observability rendering (category-b frames) + command palette,
  history, info modals, polish.
