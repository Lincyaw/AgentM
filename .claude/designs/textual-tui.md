# Design: Textual TUI

**Status**: implemented (sole interactive frontend; `rich.live` simple TUI deleted 2026-05-02)
**Created**: 2026-05-01
**Last Updated**: 2026-05-02
**Builds on**: [pluggable-architecture.md](pluggable-architecture.md), [observability.md](observability.md)

---

## 1. Overview

The interactive frontend is a Textual-based TUI that mirrors Claude Code's interaction style: streamed assistant text, inline tool calls with collapse-by-default, a slash-command picker, a persistent status line, and keyboard-first navigation. It superseded the original `rich.live`-based `modes/interactive.py` (deleted) and is now the only interactive mode reachable via `agentm -i`.

The mode layer's contract from `pluggable-architecture.md` §5 stays unchanged: `run(config: AgentSessionConfig) -> int` owns the session lifecycle, subscribes to public bus channels (`stream_delta`, `tool_call`, `tool_result`, `child_session_start/end`, `extension_install`, `llm_request_*`), never reaches into harness internals. Textual replaces rich-live as the rendering engine; nothing else moves.

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

Three regions. Vertical stack, top to bottom:

```
┌──────────────────────────────────────────────────────────────────────┐
│ ConversationLog                                                  ▲  │  ← scrollable
│   AssistantTurn — streaming text + thinking + nested tool blocks │  │
│   UserTurn     — user message                                    │  │
│   AssistantTurn ...                                              │  │
│   ...                                                            │  │
│                                                                  ▼  │
├──────────────────────────────────────────────────────────────────────┤
│ InputBar                                                             │  ← fixed, multi-line aware
│ > _                                                                  │
├──────────────────────────────────────────────────────────────────────┤
│ StatusLine: model · turn N · in: 12.3k · out: 3.4k · $0.041 · idle   │  ← single row
└──────────────────────────────────────────────────────────────────────┘
```

No left/right sidebar in MVP. Claude Code's "skills picker", "history browser", etc. are deliberately deferred — they belong in a later iteration once the core interaction works.

### 3.2 Widget tree

```
AgentMApp(textual.App)
├── ConversationLog(VerticalScroll)
│   ├── TurnContainer(Vertical)        # one per logical turn
│   │   ├── ThinkingBlock(Static)      # collapsed by default if non-empty
│   │   ├── AssistantTextBlock(Static) # streaming target
│   │   └── ToolCallBlock(Collapsible) # one per tool call
│   │       ├── header: "→ tool_name args_preview  [✓ 142ms]"
│   │       └── body: full args + full result (markdown if text)
│   └── ...
├── InputBar(Container)
│   └── PromptInput(TextArea)          # multi-line, slash completion
├── CommandPalette(ModalScreen)        # opens on "/" at start of empty line
└── StatusLine(Static)                 # reactive
```

Key Textual idioms used:

- `Collapsible` for tool blocks (one keystroke to expand). Default state: collapsed if result < 20 lines, expanded if it's a code/diff write.
- `reactive` for status state (`model`, `turn`, `tokens_in`, `tokens_out`, `cost_usd`, `phase: "idle"|"thinking"|"streaming"|"tool"`).
- `worker` for the `session.prompt(text)` coroutine so the UI stays responsive while the model streams.

### 3.3 Event-bus → widget mapping

The bus events are the contract. Each event maps to exactly one widget mutation:

| Event | Source | UI effect |
|---|---|---|
| `stream_delta(TextDelta)` | `AgentLoop` | Append to active `AssistantTextBlock`. Refresh panel. |
| `stream_delta(ThinkingDelta)` | `AgentLoop` | Append to active `ThinkingBlock`. Block is auto-collapsed when streaming ends. |
| `stream_delta(ToolCallStart)` | `AgentLoop` | Insert a placeholder `ToolCallBlock` with name, no args yet. |
| `tool_call(name, args)` | `OperationsImpl` | Fill the placeholder's args; mark phase=`tool` in status. |
| `tool_result(result, duration_ms)` | `OperationsImpl` | Set the block's result body + header status glyph (`✓`/`✗`). Auto-collapse if short and successful. |
| `llm_request_start(model, ...)` | `AgentLoop` | Status phase=`thinking`. Start a tick on the StatusLine spinner. |
| `llm_request_end(usage)` | `AgentLoop` | Status phase=`idle` if no tool ran. Update `tokens_in`/`tokens_out`/`cost_usd`. |
| `child_session_start(purpose)` | `sub_agent` | Insert a `SubagentBlock` (nested ConversationLog), phase=`subagent`. |
| `child_session_end(error?)` | `sub_agent` | Close the SubagentBlock; if error, render in red. |
| `extension_install(phase=error)` | harness | Toast notification at top of `ConversationLog`, stays for 5s. |
| `permission_request(tool, args)` | `permission` atom (future event) | Open inline `PermissionPrompt` modal — NOT a system modal — with Allow / Allow once / Deny buttons. |

The `permission_request` event does not exist yet; the design assumes it's added as part of this work or in a parallel task. If absent at implementation time, the TUI gracefully degrades — permission denials show as red tool results without an interactive prompt.

### 3.4 Slash commands and the command palette

When the user types `/` at column 0 of an empty input line, open the `CommandPalette` modal. It lists the slash commands registered with `ExtensionAPI.register_command` (already implemented). Filter on each keystroke; Enter selects; Esc cancels.

Commands prefixed with `/` and not in the registry pass through as raw text (so users can type `/something` literally if they want to).

Built-in commands the TUI itself owns (not from extensions):

- `/quit`, `/exit`, `/q` — quit the app
- `/clear` — clear `ConversationLog` (does NOT reset session state; only the visible history)
- `/help` — overlay listing key bindings + registered slash commands
- `/copy-last` — yank last assistant text block to system clipboard (via `pyperclip` or fall back to OSC 52)

### 3.5 Key bindings

| Key | Action |
|---|---|
| `Enter` | Submit prompt (when input is non-empty) |
| `Shift+Enter` | Insert newline in input |
| `Esc` | If a prompt is in flight: send a "soft cancel" — emits a `user_interrupt` event the kernel can act on (cancels current `session.prompt` task); if input has draft text: clear it; otherwise: do nothing (do NOT exit on bare Esc — that's a foot-gun) |
| `Ctrl+C` | If prompt in flight: same as Esc; if no prompt in flight: quit (with one-tap confirmation if `ConversationLog` is non-empty) |
| `Ctrl+D` | Quit unconditionally |
| `Ctrl+L` | `/clear` |
| `Ctrl+R` | Open command palette |
| `Tab` | Move focus between regions (input ↔ log) |
| `PageUp`/`PageDown` | Scroll log |
| `Up`/`Down` (when input is empty) | Cycle through prior user inputs |
| `Up`/`Down` (when typing) | Caret movement within the textarea |
| `Ctrl+E` (on focused ToolCallBlock) | Expand/collapse |

### 3.6 Streaming render strategy

Textual's reactive system can repaint per-frame, but that's overkill for token streams arriving at ~50/s. Instead:

- Buffer `TextDelta` chunks in a `bytearray` per-turn.
- A `set_interval(0.05, ...)` worker flushes the buffer into the active `AssistantTextBlock`'s renderable, calling `block.update(buffered_text)` which Textual's diff engine handles efficiently.
- 20Hz refresh is the same cadence as the current `rich.live` mode. No subjective lag.
- On `llm_request_end`, force one final flush before the worker is cancelled.

Rendering the assistant text uses `rich.markdown.Markdown` (Textual passes Rich renderables straight through). Code blocks get syntax highlighting; tool blocks render their result body the same way if it looks like markdown, plain text otherwise (use a heuristic: contains `` ``` ``, `# `, or `- ` ⇒ markdown).

### 3.7 Theming

Textual uses CSS-like styling. Ship two themes — `dark` (default) and `light` — selected via a CLI flag `--theme`. CSS lives in `src/agentm/modes/textual_app.tcss`.

Color palette mirrors Claude Code:
- assistant text: foreground default
- thinking: dim italic
- tool name: yellow
- tool success: green ✓
- tool failure: red ✗
- subagent indent: cyan
- status line: dim, single line, never wraps

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
| No left sidebar in MVP | Single-pane is enough to validate the framework migration; sidebar is feature creep | Build full Claude-Code-style multi-pane. Rejected: too much surface for one issue |
| Streaming flushes at 20 Hz, not per-token | Token rate is ~50/s; per-token repaint is wasteful | Per-token. Rejected: CPU + screen flicker without visible benefit |
| Esc cancels in-flight prompt, never exits | Bare-Esc-to-exit is a documented foot-gun | Esc exits if no prompt running. Rejected: muscle memory accidents |
| Commands not in registry pass through as text | User might want to type `/something` literally | Strict mode: reject unknown slashes. Rejected: surprises users |
| Defer permission-prompt UX gracefully | The atom doesn't yet emit the event we'd hook | Make it a hard requirement. Rejected: couples this issue to permission rework |

## 7. Acceptance Scenarios

| # | Scenario | Expected |
|---|----------|----------|
| T1 | Launch `agentm -i`; type "hello" + Enter | Single user turn appears; assistant streams a response; status line updates from `idle` → `thinking` → `streaming` → `idle` |
| T2 | Mid-stream press Esc | Stream stops; partial assistant text remains; status returns to `idle`; input bar regains focus |
| T3 | Assistant calls `read` tool returning 200 lines | Tool block appears collapsed showing `→ read {"path":"foo"}  [✓ 12ms]`; one keystroke (Ctrl+E with block focused) expands to show full content |
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
