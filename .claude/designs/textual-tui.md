# Design: Textual TUI

**Status**: implemented (sole interactive frontend; `rich.live` simple TUI deleted 2026-05-02; layout rebuilt outside-in 2026-05-02)
**Created**: 2026-05-01
**Last Updated**: 2026-05-09
**Builds on**: [pluggable-architecture.md](pluggable-architecture.md), [observability.md](observability.md)

---

## 1. Overview

The interactive frontend is a Textual-based TUI serving **two roles
simultaneously**:

1. **Conversation surface** — streamed assistant text, inline tool
   calls with collapse-by-default, slash-command picker, keyboard-first
   navigation. The visible prompt-and-reply loop.
2. **Control + observability surface for AgentM's runtime** — every
   pluggable subsystem (extension lifecycle, tool registry, cost
   budget, hot reloads, extension-injected user messages) emits bus
   events; the TUI subscribes and renders each one as either a header
   counter, a toast, or an `/<name>` modal. This is the user's window
   into what the framework is doing on their behalf — particularly
   self-modification, which is the framework's headline behavior and
   would otherwise be invisible.

It superseded the original `rich.live`-based `modes/interactive.py`
(deleted 2026-05-02) and is the only interactive mode reachable via
`agentm -i`.

The mode layer's contract from `pluggable-architecture.md` §5 stays
unchanged: `run(config: AgentSessionConfig) -> int` owns the session
lifecycle and subscribes to public bus channels only —
`stream_delta`, `tool_call`, `tool_result`, `child_session_start/end`,
`llm_request_*`, plus the new control/observability set
(`extension_install`, `extension_reload`, `api_register`,
`api_send_user_message`, `cost_budget_exceeded`). Textual replaces
rich-live as the rendering engine; nothing else moves.

## 2. Motivation

The current `interactive.py` is a proof of concept:

- **Single panel.** All assistant text + tool output share one `rich.live` panel; thinking blocks render in a sibling panel each turn but reset between turns. No history scroll-back beyond the terminal scrollback.
- **No multiline input.** `console.input("> ")` is single-line, no Enter-to-newline / Shift-Enter-to-submit, no slash completion, no history.
- **No collapse / expand.** Every `tool_call` and `tool_result` shows `_truncate(repr(args), 80)` — fine for `ls`, useless for `read` returning 200 lines.
- **No status awareness.** Model name shown once at startup; nothing surfaces token count, cost, current turn, or in-flight requests.
- **No interrupt control.** Ctrl-C cancels; there's no Esc-to-interrupt without exit, no "stop this tool" affordance.
- **No permission UX.** When `permission` atom rejects a tool call, the user sees a tool-result error string, not a prompt asking them to allow.

Textual addresses all six because that's what it is for: an actual TUI app framework with widget composition, focus management, key bindings, and reactive state. Migrating from `rich.live` is the natural next step once the streaming-bus plumbing has been validated (which the current minimal mode did).

## 3. Design Details

### 3.1 Layout

Outside-in dock layout per Textual's
[design-a-layout](https://textual.textualize.io/how-to/design-a-layout/)
guide. Top header docks first, footer + input bar dock from the bottom,
the conversation log fills the remaining `1fr`:

```
┌──────────────────────────────────────────────────────────────────────┐
│ StatusHeader: AgentM ▎model · turn N · in 12.3k · out 3.4k · $0.041 ·│  ← dock: top, h: 1
│              ● idle                                                  │
├──────────────────────────────────────────────────────────────────────┤
│ ConversationLog                                                  ▲  │  ← 1fr scrollable
│   UserTurn     — user message (blue gutter)                      │  │
│   AssistantTurn — "● assistant" label + streaming text +         │  │
│                   inline tool blocks (yellow gutter, single)     │  │
│   SubagentBlock — nested cyan-gutter Vertical                    │  │
│                                                                  ▼  │
├──────────────────────────────────────────────────────────────────────┤
│ InputBar (PromptInput TextArea, rounded border, autosize 3..8)       │  ← dock: bottom, h: auto
├──────────────────────────────────────────────────────────────────────┤
│ Footer: ⌃C Interrupt · ⌃D Quit · ⌃L Clear · ⌃R Commands · ⌃E Toggle  │  ← dock: bottom, h: 1
└──────────────────────────────────────────────────────────────────────┘
```

The previous layout — Vertical stack with a one-line `italic dim`
status row at the bottom — was unreadable: status got buried, fixed
elements weren't anchored, and the conversation log could shift when
emptied. Promoting status to a top dock-bar with `$primary` background,
adding the standard `Footer` for binding hints, and using
`border-left` gutters for message attribution (no Rich `Panel`) gives
each region a single-source-of-truth visual treatment.

No left/right sidebar in MVP. Claude Code's "skills picker", "history
browser", etc. are deliberately deferred — they belong in a later
iteration once the core interaction works.

### 3.2 Widget tree

```
AgentMApp(textual.App)
├── StatusHeader(Static)               # dock: top, h: 1, reactive
├── ConversationLog(VerticalScroll)    # 1fr middle
│   ├── UserTurn(Static)               # markdown body, blue border-left gutter
│   ├── TurnContainer(Vertical)        # assistant turn
│   │   ├── Static("● assistant")      # attribution label
│   │   ├── ThinkingBlock(Collapsible) # auto-collapsed when streaming ends
│   │   ├── AssistantTextBlock(Static) # streaming target (Markdown)
│   │   ├── ToolCallBlock(Collapsible) # yellow gutter, no double border
│   │   │   ├── title: "tool_name  ✓ 142ms"
│   │   │   └── body: args (yellow) + result (green/red)
│   │   └── SubagentBlock(Vertical)    # cyan gutter, native widgets
│   └── ...
├── InputBar(Container)                # dock: bottom, h: auto
│   └── PromptInput(TextArea)          # rounded border, multi-line, slash completion
├── Footer()                           # dock: bottom, h: 1, BINDINGS hints
├── CommandPaletteScreen(ModalScreen)  # Ctrl+R or "/" at empty input
└── HelpScreen(ModalScreen)            # /help
```

Key Textual idioms used:

- `dock: top/bottom` for fixed regions; `1fr` for the flex middle
  (Textual's "outside-in" layout pattern).
- `Footer` widget auto-renders any `BINDINGS` entries with `show=True`,
  so users see `⌃C ⌃D ⌃L ⌃R ⌃E` without opening `/help`.
- App-level theme switching via `self.theme = "textual-dark"` /
  `"textual-light"` (Textual built-in registered themes) — no custom
  `.theme-dark { background: #0f1115 }` overrides that fight the
  framework.
- `Collapsible` for tool blocks (one keystroke to expand). Default
  state: collapsed if result < 20 lines, expanded if it's a code/diff
  write.
- `reactive` for status state (`model`, `turn`, `tokens_in`,
  `tokens_out`, `cost_usd`, `phase: "idle"|"thinking"|"streaming"|"tool"|"subagent"`).
- `worker` for the `session.prompt(text)` coroutine so the UI stays
  responsive while the model streams.
- `border-left: thick $variable` (CSS, single source) instead of
  `rich.Panel(border_style=...)` plus `border-left` (which produced
  doubled borders in the previous design).

### 3.3 Event-bus → widget mapping

The bus events are the contract. Each event maps to exactly one widget mutation:

| Event | Source | UI effect |
|---|---|---|
| `stream_delta(TextDelta)` | `AgentLoop` | Dispatch table entry appends to active `AssistantTextBlock`. Refresh panel. |
| `stream_delta(ThinkingDelta)` | `AgentLoop` | Dispatch table entry appends to active `ThinkingBlock`. Block is auto-collapsed when streaming ends. |
| `stream_delta(ToolCallStart)` | `AgentLoop` | Dispatch table entry inserts a placeholder `ToolCallBlock` with name, no args yet. |
| `tool_call(name, args)` | `OperationsImpl` | Fill the placeholder's args; mark phase=`tool` in status. |
| `tool_result(result, duration_ms)` | `OperationsImpl` | Set the block's result body + header status glyph (`✓`/`✗`). Auto-collapse if short and successful. |
| `llm_request_start(model, ...)` | `AgentLoop` | Status phase=`thinking`. StatusHeader updates `model_name` and the phase glyph. |
| `llm_request_end(usage)` | `AgentLoop` | Status phase=`idle` if no tool ran. Update `tokens_in`/`tokens_out`/`cost_usd`. |
| `child_session_start(purpose)` | `sub_agent` | Insert a `SubagentBlock` (nested ConversationLog), phase=`subagent`. |
| `child_session_end(error?)` | `sub_agent` | Close the SubagentBlock; if error, render in red. |
| `extension_install(phase=*)` | harness | Tracked into the `/extensions` modal snapshot. `phase=error` also emits an error toast. Header counters update with `loaded` / `failed` totals. |
| `api_register(kind=command\|tool)` | `ExtensionAPI` | Captured into the `slash_commands` and `tools` snapshots. Header `N tools` counter updates. Late registrations (post-create reload) are routed through the live `handle_api_register`. |
| `extension_reload(is_self_modify, ok/error)` | `atom_reloader` | Toast. `is_self_modify=true` → `★ self-modify` warning toast (the framework's headline behavior is made visible). Human reloads stay informational. `error` → error toast. |
| `cost_budget_exceeded(used, limit, currency)` | `cost_budget` atom | Latch `_budget_state`; header shows `$… ⚠`; error toast names the cap and warns the next prompt will halt with `stop_reason='budget'`. `/budget` modal then shows current state. |
| `api_send_user_message(extension, content)` | `ExtensionAPI` (any extension calling `api.send_user_message`) | Render a synthetic `UserTurn` with `injected_from=<extension>` — yellow gutter and a `system → you` label so the user can tell it apart from a turn they typed. |
| `permission_request(tool, args)` | `permission` atom (future event) | Open inline `PermissionPrompt` modal — NOT a system modal — with Allow / Allow once / Deny buttons. |

The `permission_request` event does not exist yet; the design assumes it's added as part of this work or in a parallel task. If absent at implementation time, the TUI gracefully degrades — permission denials show as red tool results without an interactive prompt.

### 3.4 Slash commands and the command palette

When the user types `/` at column 0 of an empty input line, open the `CommandPalette` modal. It lists one `BuiltinCommandRegistry`: TUI-owned commands and commands observed from `ExtensionAPI.register_command` via `ApiRegisterEvent(kind="command")`. Filter on each keystroke; Enter selects; Esc cancels.

Commands prefixed with `/` and not in the registry pass through as raw text (so users can type `/something` literally if they want to). Registered extension commands use the same registry lookup path as built-ins; their handler delegates to `session.prompt("/<name> ...")` so the harness `slash_commands` atom remains the SDK-level dispatcher.

Built-in commands the TUI itself owns (not from extensions):

- `/quit`, `/exit`, `/q` — quit the app
- `/clear` — clear `ConversationLog` (does NOT reset session state; only the visible history)
- `/help` — overlay listing key bindings + registered slash commands
- `/copy-last` — yank last assistant text block to system clipboard (via `pyperclip` or fall back to OSC 52)
- `/extensions` — open `InfoModal` showing the snapshot of every
  `ExtensionInstallEvent` seen in this session (status glyph
  ⏳/✓/✗ + module path + last error). Snapshot-on-open: close and
  reopen to refresh.
- `/tools` — open `InfoModal` listing every tool the agent currently
  has registered (name, source extension, description). The snapshot
  is built from `ApiRegisterEvent(kind="tool")` — both create-time
  registrations (accumulated by `run()` before `AgentSession.create`)
  and runtime reloads.
- `/budget` — open `InfoModal` showing the current `cost_budget`
  state. If `cost_budget_exceeded` has not fired, displays a "Budget
  OK" hint; once exceeded, shows used/limit/currency and the
  next-prompt halt warning.

### 3.5 Key bindings

| Key | Action |
|---|---|
| `Enter` | Submit prompt (when input is non-empty) |
| `Shift+Enter` | Insert newline in input |
| `Esc` | If a prompt is in flight: send a "soft cancel" — emits a `user_interrupt` event the kernel can act on (cancels current `session.prompt` task); if input has draft text: clear it; otherwise: emit a `Nothing to cancel.` toast (an explicit acknowledgement instead of the previous silent no-op, which made users wonder if the app had stalled) |
| `Ctrl+C` | If prompt in flight: cancel; if pressed again within 1.5s: quit. If no prompt in flight: quit when log is empty; otherwise toast `Press Ctrl+C again within 1.5s to quit` and quit on the second press inside that window |
| `Ctrl+D` | Quit unconditionally |
| `Ctrl+L` | `/clear` |
| `Ctrl+R` | Open command palette |
| `Tab` | Move focus between regions (`PromptInput.tab_behavior = "focus"` makes Tab leave the input naturally; the prior App-level Tab binding was a no-op while the input had focus and is therefore removed) |
| `PageUp`/`PageDown` | Scroll log |
| `Up`/`Down` (when input is empty) | Cycle through prior user inputs |
| `Up`/`Down` (when typing) | Caret movement within the textarea |
| `Ctrl+E` (on focused ToolCallBlock) | Expand/collapse |

`BINDINGS` entries with `show=True` appear in the docked `Footer`, so
the user discovers `⌃C` / `⌃D` / `⌃L` / `⌃R` / `⌃E` without opening
`/help`.

### 3.6 Streaming render strategy

Textual's reactive system can repaint per-frame, but that's overkill for token streams arriving at ~50/s. Instead:

- Buffer `TextDelta` chunks in a `bytearray` per-turn.
- A `set_interval(0.05, ...)` worker flushes the buffer into the active `AssistantTextBlock`'s renderable, calling `block.update(buffered_text)` which Textual's diff engine handles efficiently.
- 20Hz refresh is the same cadence as the current `rich.live` mode. No subjective lag.
- On `llm_request_end`, force one final flush before the worker is cancelled.

Rendering the assistant text uses `rich.markdown.Markdown` (Textual passes Rich renderables straight through). Code blocks get syntax highlighting; tool blocks render their result body the same way if it looks like markdown, plain text otherwise (use a heuristic: contains `` ``` ``, `# `, or `- ` ⇒ markdown).

### 3.7 Theming

Textual ≥6 ships a built-in theme system. Activate one by setting
`self.theme = "textual-dark"` (or `"textual-light"`) — colors flow
through CSS variables (`$primary`, `$surface`, `$boost`, `$text`,
`$accent`, `$warning`, `$error`, `$text-muted`) and the framework
handles every render path including the `Footer`. The CLI flag
`--theme` accepts the short aliases `dark` / `light` which are mapped
to those built-in theme names. The previous design hardcoded hex
colors inside `.theme-dark { background: #0f1115 }` blocks; that
fought the framework on every redraw and made it impossible for users
to register a third theme without rewriting CSS.

Semantic role → CSS variable:

- assistant attribution label: `$accent` bold
- user-message gutter: `$primary`
- thinking: `$text-muted` italic
- tool name / pending: `$warning`
- tool success: green `✓` glyph (Rich-side, in body)
- tool failure: `$error`
- subagent indent: `$accent`
- status header: `$primary 60%` background, `$text` foreground

CSS lives in `src/agentm/modes/textual_app.tcss`.

### 3.8 Migration path (historical)

Originally landed alongside the legacy `rich.live` mode at
`modes/interactive.py`, selectable via `--tui {simple,textual}`. As of
2026-05-02 the simple mode is deleted and the Textual app is the sole
interactive frontend reachable via `agentm -i`. The `--tui` flag has
been removed.

## 4. Interface Definition

```python
# agentm/modes/textual_app.py

from textual.app import App
from textual.binding import Binding

from agentm.harness import AgentSessionConfig

class AgentMApp(App[int]):
    """Textual TUI for AgentM. Returns process exit code on exit."""

    CSS_PATH = "textual_app.tcss"
    BINDINGS = [
        Binding("ctrl+c", "interrupt_or_quit", show=False),
        Binding("ctrl+d", "quit", show=True, priority=True),
        Binding("ctrl+l", "clear_log", show=True),
        Binding("ctrl+r", "open_palette", show=True),
        # ... see §3.5
    ]

    def __init__(self, config: AgentSessionConfig, *, theme: str = "dark") -> None: ...

async def run(config: AgentSessionConfig, *, theme: str = "dark") -> int:
    """Owns the session lifecycle for one ``agentm -i`` invocation."""
```

CLI: a single `-i` / `--interactive` flag opens the Textual TUI; there
is no longer a frontend selector.

Dependencies: `textual>=0.85` is a hard requirement. Optional: `pyperclip` for `/copy-last` (graceful fallback to OSC 52 escape if absent).

## 5. Related Concepts

- [pluggable-architecture.md](pluggable-architecture.md) — modes/ layer contract; the new mode obeys §5 unchanged.
- [observability.md](observability.md) — the bus-events list this design renders is a strict subset of what observability already records. No new observability surface.

## 6. Constraints and Decisions

| Decision | Rationale | Alternative |
|---|---|---|
| Textual, not raw curses or prompt_toolkit | Textual provides composition + reactive + Collapsible widgets out of the box; CSS-style theming; works over SSH | prompt_toolkit (lower-level, more code); urwid (legacy, slower iteration) |
| ~~Keep simple mode as fallback~~ (reverted 2026-05-02) | Initially the rich-live mode stayed for environments where Textual misbehaves. In practice no such environment surfaced and maintaining two frontends doubled the test surface. Simple mode deleted. | — |
| Outside-in dock layout (top header, bottom footer + input, 1fr middle) | Per Textual's official layout guide. Anchors fixed regions so the conversation log can never push them out of position; Footer auto-renders BINDINGS so users discover keys without /help. | Vertical-stack-with-implicit-flex (the previous approach). Rejected: status row could shift on empty/clear; readability suffered |
| Single-source-of-truth gutters via CSS `border-left` | Previous design rendered user/subagent with `rich.Panel(border_style=...)` AND CSS `border-left` simultaneously, producing doubled borders. Now Rich Panel is gone; gutters live only in CSS. | Keep both. Rejected: double-rendered borders look broken |
| Built-in `textual-dark` / `textual-light` themes (not custom hex) | The previous `.theme-dark { background: #0f1115 }` overrides fought the framework's theming system on every redraw and forced anyone who wanted a third theme to rewrite CSS. Built-in themes use semantic CSS variables. | Hardcoded hex per theme class. Rejected: brittle, no extension point |
| No left sidebar in MVP | Single-pane is enough to validate the framework migration; sidebar is feature creep | Build full Claude-Code-style multi-pane. Rejected: too much surface for one issue |
| Streaming flushes at 20 Hz, not per-token | Token rate is ~50/s; per-token repaint is wasteful | Per-token. Rejected: CPU + screen flicker without visible benefit |
| Esc cancels in-flight prompt; on idle emits explicit toast | Bare-Esc-to-exit is a documented foot-gun, but the previous silent no-op on idle made the app look stalled. The toast acknowledges the keypress. | Esc exits / Esc silent. Both rejected as user-confusing |
| Ctrl+C double-tap-within-1.5s escalates to quit | Previously Ctrl+C while a prompt was running could only cancel — never quit — and forced the user to remember Ctrl+D. The 1.5s window matches shell convention. | Ctrl+C never quits while prompt runs. Rejected: muscle-memory friction |
| Commands not in registry pass through as text | User might want to type `/something` literally | Strict mode: reject unknown slashes. Rejected: surprises users |
| Defer permission-prompt UX gracefully | The atom doesn't yet emit the event we'd hook | Make it a hard requirement. Rejected: couples this issue to permission rework |
| Subscribe to bus events twice — once before `AgentSession.create` (snapshot dict), once after (live mutation) | Extension load fires `api_register` and `extension_install` *during* create, before the AgentMApp instance exists. A snapshot dict accumulates those into `__init__`; live handlers cover post-create reloads. | Subscribe only after create. Rejected: `/extensions` and `/tools` would be empty until the agent triggered a reload, which is the opposite of when the user wants to see them. |
| Generic `InfoModal` instead of bespoke modal classes | All three control modals (`/extensions`, `/tools`, `/budget`) just need title + Rich renderable. Builders return Rich tables; the modal stays one class. | Three subclasses. Rejected: ~60 lines of duplication |
| Self-modification reloads use a stronger toast severity than human reloads | `trigger ∈ {agent, propose_change_approved}` is the framework's signature behavior; making it noisier than a human-triggered reload aligns visual weight with stakes. | Same severity for all reloads. Rejected: trains user to ignore self-modification |

## 7. Acceptance Scenarios

| # | Scenario | Expected |
|---|----------|----------|
| T1 | Launch `agentm -i`; type "hello" + Enter | Single user turn appears; assistant streams a response; status header updates from `● idle` → `◐ thinking` → `◑ streaming` → `● idle` |
| T2 | Mid-stream press Esc | Stream stops; partial assistant text remains; status returns to `idle`; input bar regains focus |
| T3 | Assistant calls `read` tool returning 200 lines | Tool block appears collapsed with title `read  ✓ 12ms` (yellow gutter, single border); one keystroke (Ctrl+E with block focused) expands to show full args + result |
| T4 | Type `/` at start of empty input | Command palette opens; lists `/quit`, `/clear`, `/help` plus extension-registered commands; type filters; Enter inserts |
| T5 | Sub-agent dispatched | A `SubagentBlock` appears nested under the parent assistant turn, indented + cyan; child stream renders inside it |
| T6 | Press Shift+Enter in input | Newline inserted; input expands vertically; submission only fires on bare Enter |
| T7 | `Ctrl+L` | ConversationLog clears; session messages preserved (next prompt continues the conversation) |
| T8 | Rendering 500 turns | No visible lag on scroll; memory stays bounded (Textual's virtual scrolling handles this) |
| T9 | `agentm -i` dispatches to the Textual runner | `_run_interactive` builds a session config and invokes `modes.textual_app.run` (regression guard for the legacy `--tui` flag removal) |
| T10 | Run in a terminal without 24-bit color | Theme falls back to 16-color; layout remains usable |

## 8. Open Questions

1. **Where do tool diffs render?** `tool_edit` results are diffs; Textual has no native diff widget. Options: (a) render unified-diff text with `rich.syntax.Syntax(language="diff")`, (b) build a custom `DiffWidget` showing side-by-side hunks, (c) defer until later. Recommend (a) for MVP; (b) is its own design.

2. **Does `/clear` truly preserve session state?** Today `interactive.py` doesn't expose any reset mechanism. If the user expects "clear means restart conversation", we need a separate `/reset` verb that calls `session.reset_messages()`. Decide: `/clear` = visual only, `/reset` = also reset session. Document both.

3. **Permission-prompt event surface.** The TUI design assumes a `permission_request` event will be added so the prompt can be rendered inline. If we want the TUI to also work as a no-op fallback (current behavior, just-deny-and-show-error), no harness change is needed. Decide whether to land the event surface alongside this work or keep it as a follow-up; the TUI must function either way.

4. **History persistence.** Up/Down on empty input cycles prior user inputs (§3.5). In-memory only? Or a file `~/.local/state/agentm/history`? Match shell history behavior (file-backed) is friendlier but adds a state-file dependency. Recommend: in-memory MVP; file-backed in a follow-up when someone complains.

5. **Multi-turn copy of "the response".** `/copy-last` copies the last assistant text block. What about copying a specific tool result, or all of a turn? Defer; one verb is enough for MVP.

### 3.8 Presenter Dispatch Tables and Turn Identity

The presenter keeps wiring declarative:

- Live event subscriptions are `_EVENT_SUBSCRIPTIONS: (channel, method_name)[]` and registered with one loop in `run()`; adding a public event no longer requires a repeated `bus.on(...)` block.
- Stream deltas are rendered through `_DELTA_HANDLERS` and `_CHILD_DELTA_HANDLERS`, keyed by delta type. New provider delta classes can be registered without editing the main `handle_stream_delta` body.
- Status phases are typed as `Phase = Literal["idle", "thinking", "streaming", "tool", "subagent"]`. `StatusHeader` reads glyphs from a `Theme` protocol; `DEFAULT_THEME` preserves the existing glyphs.
- `AgentLoop` emits `turn_id`, monotone for the lifetime of the loop, alongside prompt-local `turn_index`. The TUI keys root `TurnContainer`s by `turn_id`; the old prompt-epoch workaround is deleted.
