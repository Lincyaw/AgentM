# Design: Go Terminal TUI (bubbletea)

**Status**: IMPLEMENTED, evolving
**Created**: 2026-05-31
**Updated**: 2026-07-03

## 0. Purpose

A Go-native terminal chat-client peer for the AgentM gateway, replacing the
Python Textual TUI. Built on bubbletea (Elm architecture), lipgloss (styling),
glamour (markdown), golang.org/x/term. Connects to `agentm gateway` over the
v2 wire protocol (single-process-gateway.md §2).

The TUI is a **dumb adapter** (single-process-gateway.md §5.1) — it sends
`inbound` envelopes and renders `outbound` ones. No session, scenario, or
atom knowledge.

The recommended user entrypoint is `agentm terminal`: the Python CLI starts or
reuses a local gateway daemon, waits for its unix socket, launches
`agentm-terminal --connect ...`, and leaves the Go binary as a pure wire
client. A lightweight Python supervisor owns the daemon process and restarts
the gateway worker when source/config files change, so SDK/session code updates
apply without manually restarting the terminal. Advanced users can still run
`agentm gateway` and `agentm-terminal --connect ...` separately.

## 1. Design principles

1. **Collapse by default, expand on demand.** Every non-essential block
   (thinking, tool args/result, approval details) shows a one-line summary.
   The user expands with Enter/Ctrl+E. Screen real estate is scarce.
2. **Observability is not optional.** Elapsed time, tok/s, context gauge,
   cost accumulator — all visible by default. The user should never wonder
   "is it stuck?" or "how much did that cost?".
3. **Block-level interaction.** The transcript is a list of typed blocks.
   Navigation, yank, search, bookmark all operate on blocks, not raw text
   lines.
4. **No widget tree.** Bubbletea re-renders the full screen each frame.
   State is a block list + dirty flags. A 20 Hz tick flushes streaming
   buffers; no per-token repaint.

## 2. Screen layout

```
┌─────────────────────────────────────────────────┐
│ ● idle · doubao · 14.2s · 38 tok/s              │
│ ctx [████████░░] 78%  · $0.12 / $1.47           │  ← StatusBar (2 lines)
│ terminal:default · 23m                           │
├─────────────────────────────────────────────────┤
│ ┃ › you                                          │
│ ┃ explain the codebase                           │
│ ┃                                                │
│ ┃ ● assistant                                    │
│ ┃ ▸ thinking ···························         │  ← collapsed
│ ┃ The codebase is structured as a pluggable...   │
│ ┃ ⚙ bash(ls -la)  ✓                             │  ← collapsed tool
│ ┃ ⌥ subagent: analysis  ✓                       │
│ ┃                                                │  ← Viewport
│ ┃ › you                                          │    (scrollable)
│ ┃ what about the tests?                          │
│ ┃                                                │
│ ┃ ● assistant                                    │
│ ┃ ◐ thinking...                                  │  ← live streaming
│ ┃                                                │
├─────────────────────────────────────────────────┤
│  /help                                           │  ← Suggestions
│  /clear                                          │    (0-N lines,
│  /status                                         │     conditional)
├─────────────────────────────────────────────────┤
│ > _                                              │  ← Input
└─────────────────────────────────────────────────┘

              ┌─────────────────────┐
              │ reload: wire_driver │  ← Toast (overlay,
              └─────────────────────┘    bottom-right)
```

Height budget: StatusBar 2 lines, Input 1-3 lines (textarea auto-grow),
Suggestions 0-N lines. Viewport gets **all remaining height**.

### 2.1 StatusBar (2 lines)

Line 1: phase glyph + phase name · model · elapsed · tok/s
Line 2: context gauge bar · cost (this turn / session total) · session key · session age

When idle (no turn in flight), elapsed and tok/s are hidden. The gauge and
cost are always visible.

```
◐ thinking · doubao · 14.2s · 38 tok/s
ctx [████████░░] 78%  · $0.12 / $1.47  · terminal:default · 23m
```

### 2.2 Viewport (scrollable transcript)

A `bubbles/viewport` over the concatenation of all rendered blocks. The left
edge has a thin **activity spine** — a `┃` gutter whose color reflects the
block type (dim for user, accent for assistant, yellow for tool, etc.).

Auto-scroll to bottom on new content. Manual scroll (mouse wheel / PgUp/Dn)
disengages auto-scroll; new content re-engages it.

### 2.3 Input

`bubbles/textarea` with:
- Enter: submit
- Ctrl+J: insert newline
- Auto-grow height (1 → 3 lines max), shrink back after submit
- Prompt prefix `> ` rendered via lipgloss, not part of the editable text

### 2.4 Suggestions

Inline below input (not a floating popup). Visible only when input starts
with `/` and hasn't been committed. Populated from `catalog.commands` +
`/clear`. Up/Down navigates, Tab/Enter accepts. Esc dismisses.

### 2.5 Toast

Rendered as an overlay on the viewport's bottom-right corner. Auto-dismiss
after 4s via `tea.Tick`. Multiple toasts stack vertically (newest at bottom).
Variants: info (dim), warn (yellow), selfmod (accent).

## 3. Block types

Every item in the transcript implements:

```go
type Block interface {
    Kind() string
    Collapsed() bool
    SetCollapsed(bool)
    Render(width int, focused bool) string
}
```

### 3.1 UserTurn

Always expanded. No collapse toggle.

```
› you
explain the codebase
```

### 3.2 SystemTurn

Always expanded. Source label in parentheses.

```
system → you  (wire_driver)
[reminder injected]
```

### 3.3 AssistantTurn

Container block. Holds optional ThinkingBlock + markdown text + inline
ToolBlocks + SubagentBlocks. The text portion is always visible; thinking
and tools are collapsed by default.

```
● assistant
▸ thinking ····························  ← collapsed (one line)
The codebase is structured as...         ← glamour-rendered markdown
⚙ bash(ls -la)  ✓                       ← collapsed tool (one line)
⚙ edit(src/main.go)  ✓                  ← collapsed tool
⌥ subagent: analysis  ✓                 ← always one line
```

Expanded thinking:

```
● assistant
▾ thinking
  The user is asking about the codebase structure.
  I should start by listing the top-level directories...
The codebase is structured as...
```

Expanded tool (with diff preview for edit/write):

```
⚙ edit(src/main.go)  ✓
  │ -    return nil
  │ +    return fmt.Errorf("connection refused: %w", err)
```

Expanded tool (generic):

```
⚙ bash(ls -la)  ✓
  total 48
  drwxr-xr-x  5 user user 4096 ...
  -rw-r--r--  1 user user 1234 ...
```

### 3.4 ToolBlock (inline within AssistantTurn)

One-line collapsed: `⚙ name(summary)  ✓/✗/⟳`
Expanded: title + args/result body. For `edit`/`write` tools, render a
colored diff instead of raw JSON.

### 3.5 ThinkingBlock (inline within AssistantTurn)

One-line collapsed: `▸ thinking ···` (dot-padded to fill width, dim)
Expanded: `▾ thinking` + full reasoning text (dim style)

### 3.6 SubagentBlock (inline within AssistantTurn)

Always one line: `⌥ subagent: purpose  ✓/✗/⟳`. No expand (no inner stream
visible via wire protocol).

### 3.7 ApprovalBlock

Default: one-line summary of what the agent wants to do.
Expanded (`?`): full tool args JSON.

```
⚠ Agent wants to: delete all .tmp files in /var/log
  [1] Approve   [2] Deny   [?] Details
```

## 4. Interaction modes

The TUI has one primary mode (normal) and several transient overlays.

### 4.1 Normal mode

Input has focus. Keys go to the textarea. Global bindings:

| Key | Action |
|---|---|
| Enter | Submit input |
| Ctrl+J | Insert newline in input |
| Ctrl+C ×1 | Clear input (if non-empty) or toast "press again to quit" |
| Ctrl+C ×2 (1.5s) | Quit |
| Ctrl+D | Quit |
| Ctrl+L | Clear transcript (keep session) |
| Ctrl+E | Toggle collapse on nearest tool/thinking block |
| Ctrl+Y | Copy last assistant reply (OSC 52) |
| Ctrl+B | Bookmark current scroll position |
| Ctrl+G | Open bookmark list overlay |
| Ctrl+F | Open search overlay |
| Ctrl+R | Open re-send overlay (recent messages) |
| Ctrl+S | Save code block to file (prompts for path) |
| Esc | Close overlay / interrupt in-flight / clear input |
| `[` / `]` | Jump to prev/next turn (block-level nav) |
| Up/Down | Input history (or suggestion nav when popup visible) |
| Tab | Accept suggestion |
| PgUp/PgDn | Scroll viewport |
| Mouse wheel | Scroll viewport |
| `?` / F1 | Help overlay (keybinding cheat sheet) |
| `1`-`9` on ApprovalBlock | Pick approval option |

### 4.2 Search overlay (Ctrl+F)

A single-line input appears above the input area. As the user types,
matching text in the viewport is highlighted. `n` / `N` (or Enter / Shift+Enter)
jump between matches. `Esc` closes the overlay and restores normal scroll
position if no match was selected.

```
┌─────────────────────────────────────────┐
│  ...transcript with [highlighted] matches│
├─────────────────────────────────────────┤
│ 🔍 search: connection_                  │  ← search input
├─────────────────────────────────────────┤
│ > _                                      │
└─────────────────────────────────────────┘
```

### 4.3 Yank overlay (Ctrl+Y with prefix, or `y` then number)

When the user presses a dedicated yank chord (Ctrl+Y yanks last reply as
the fast path), a numbered index appears next to each block. The user types
a number to copy that block's content via OSC 52.

### 4.4 Bookmark list (Ctrl+G)

A small overlay listing bookmarks (label + line summary). Enter jumps to
the bookmarked scroll position. `d` deletes a bookmark. Esc closes.

### 4.5 Re-send overlay (Ctrl+R)

Lists the last N user messages. Fuzzy filter as you type (reverse-i-search
style). Enter re-sends; Tab loads into input for editing.

### 4.6 Help overlay (? / F1)

A semi-transparent overlay covering the viewport showing all keybindings
grouped by category. Any key dismisses.

## 5. Wire integration

### 5.1 WireClient

Connects to the gateway over unix socket or WebSocket. Handles:
- Length-prefixed framing (4-byte big-endian + JSON)
- `hello` → `welcome` handshake
- Sends `inbound` envelopes (user text, button clicks, interrupt control)
- Receives `outbound` + `error` + `ping` envelopes
- Ack pump for durable outbound
- Reconnect with exponential backoff

Runs in a goroutine; delivers `OutboundMsg` to the bubbletea event loop
via a channel + `tea.Cmd`.

### 5.2 Router (wire kind → state mutation)

Maps `metadata.kind` to block mutations. Identical logic to the Python
router (router.py), expressed as a switch:

| Kind | Mutation |
|---|---|
| `turn_start` | New AssistantTurn block, phase→thinking |
| `stream_text` | Append to active turn text buffer, mark dirty |
| `stream_thinking` | Append to active turn thinking buffer, mark dirty |
| `tool_call` | New ToolBlock in active turn |
| `tool_result` | Update ToolBlock with result |
| `assistant_text` | Set final text on active turn, turn complete |
| `agent_end` | phase→idle, clear in-flight |
| `usage` | Accumulate tokens, compute tok/s |
| `session_ready` | Update model name, catalog (tools + commands) |
| `approval_request` | New ApprovalBlock |
| `approval_resolved` | Toast |
| `diagnostic_warning` | Toast (warn) |
| `diagnostic_error` | Toast (warn) |
| `child_start` | New SubagentBlock |
| `child_end` | Finish SubagentBlock |
| `cost_budget_exceeded` | Update cost, toast |
| `extension_install` (error) | Toast (warn) |
| `extension_reload` | Toast (selfmod if self-modify) |
| `api_register` | Update catalog |
| `after_compact` | Toast + update context gauge |

### 5.3 Streaming flush

A `tea.Tick` at 50ms (20 Hz). On each tick, if any buffer is dirty:
re-render the active turn's block, update viewport content, auto-scroll.
This prevents per-token repaints.

## 6. Utilities

### 6.1 Transcript search

State: `searchActive bool`, `searchQuery string`, `searchMatches []int` (block indices),
`searchCursor int`. Rendering: matched blocks get a highlight style on matching
substrings. Viewport scrolls to the current match.

### 6.2 Block navigation

`[` and `]` scan the block list for the prev/next turn-level block (UserTurn
or AssistantTurn, skipping inline ToolBlocks). Viewport scrolls to place that
block at the top.

### 6.3 Bookmarks

State: `bookmarks []Bookmark` where `Bookmark{blockIndex int, label string}`.
Ctrl+B adds the block nearest to the current viewport top. Ctrl+G opens the
overlay.

### 6.4 Re-send (Ctrl+R)

State: `resendActive bool`, `resendFilter string`, `resendCandidates []string`.
Candidates are the user's input history. Fuzzy-matched as the user types.

### 6.5 Code block extraction (Ctrl+S)

Scans the last assistant turn for fenced code blocks. If exactly one, prompts
for a file path (default: guessed from language + content). If multiple,
shows a numbered list first. Writes via `os.WriteFile`.

### 6.6 Diff preview in ToolBlock

For `edit` and `write` tool calls, parse `old_string` / `new_string` from
args (edit) or detect file content (write). Render as red/green diff lines
using lipgloss. For other tools, render args as truncated JSON.

### 6.7 OSC 52 clipboard

```go
func copyToClipboard(text string) {
    b64 := base64.StdEncoding.EncodeToString([]byte(text))
    fmt.Fprintf(os.Stderr, "\033]52;c;%s\a", b64)
}
```

### 6.8 OSC 8 hyperlinks

URLs in rendered text are wrapped in OSC 8 sequences so compatible terminals
make them clickable: `\033]8;;URL\033\\label\033]8;;\033\\`.

### 6.9 Desktop notification (OSC 777 / 99)

When a turn completes (agent_end) and elapsed > 10s, emit an OSC 777/99
notification so the terminal (if supported) shows a desktop alert.

### 6.10 Session export (/export)

The `/export` command (handled client-side like `/clear`) writes the
transcript to a markdown file. Each block becomes a markdown section with
appropriate formatting (code fences for tool output, blockquotes for
thinking, etc.). Includes a YAML frontmatter with session metadata (model,
total tokens, cost, duration).

### 6.11 Help overlay

A pre-rendered lipgloss-styled string showing all keybindings in two columns,
grouped: Navigation, Editing, Control, Utilities. Overlaid on the viewport
center. Any keypress dismisses.

### 6.12 Context gauge

The gateway reports context usage via `usage` frames (`input_tokens`). The
gauge is `used / estimated_context_window` rendered as a 10-char bar. The
context window size is taken from `session_ready.context_window` or defaults
to 128k.

### 6.13 Cost estimation

Token counts × per-model pricing (a small hardcoded table: input $/1M,
output $/1M). Updated on each `usage` frame. Displayed as `$this_turn / $session`.

## 7. Package structure

```
contrib/gateway-peers/terminal-go/
├── go.mod
├── go.sum
├── main.go                      // CLI entry (pflag/cobra)
├── internal/
│   ├── app/
│   │   ├── model.go             // root Model + Init + Update + View
│   │   ├── router.go            // wire kind → state mutation dispatch
│   │   ├── keys.go              // key bindings table
│   │   └── overlay.go           // search/bookmark/resend/help overlay logic
│   ├── blocks/
│   │   ├── block.go             // Block interface
│   │   ├── user.go              // UserTurn
│   │   ├── system.go            // SystemTurn
│   │   ├── assistant.go         // AssistantTurn (container)
│   │   ├── thinking.go          // ThinkingBlock
│   │   ├── tool.go              // ToolBlock + diff preview
│   │   ├── subagent.go          // SubagentBlock
│   │   └── approval.go          // ApprovalBlock
│   ├── components/
│   │   ├── statusbar.go         // StatusBar model + render
│   │   ├── input.go             // Input wrapper (textarea + history + prompt prefix)
│   │   ├── suggestions.go       // Command autocomplete list
│   │   └── toast.go             // Toast stack + overlay render
│   ├── theme/
│   │   └── theme.go             // Glyphs, labels, lipgloss styles, dark/light
│   ├── wire/
│   │   ├── client.go            // WireClient (goroutine, channel-based)
│   │   ├── envelope.go          // Envelope struct, marshal/unmarshal
│   │   ├── framing.go           // 4-byte length-prefix read/write
│   │   └── transport.go         // Unix socket + WebSocket dial
│   └── util/
│       ├── clipboard.go         // OSC 52
│       ├── hyperlink.go         // OSC 8
│       ├── notify.go            // OSC 777/99 desktop notification
│       ├── diff.go              // Simple red/green diff render
│       └── export.go            // /export markdown writer
```

## 8. Implementation phases

| Phase | Scope | Deliverable |
|---|---|---|
| P1 | Theme + block types + all render functions | All blocks render correctly with mock data |
| P2 | App skeleton: statusbar + viewport + input + layout + keybindings | Interactive TUI with hardcoded transcript |
| P3 | Components: suggestions, toast, overlays (search, help, bookmark, resend) | All overlays functional with mock data |
| P4 | Wire client: framing, envelope, transport, handshake | Connects to gateway, receives welcome |
| P5 | Router + integration: wire events → block mutations, streaming flush | End-to-end: send message, see streamed reply |
| P6 | Utilities: diff preview, OSC clipboard/links/notify, /export, code extract, cost/gauge | All utils working |
| P7 | Polish: glamour markdown, dark/light theme, edge cases, error handling | Ship-ready |

## 9. Dependencies

```
github.com/charmbracelet/bubbletea    v1
github.com/charmbracelet/bubbles      (viewport, textarea)
github.com/charmbracelet/lipgloss     v1
github.com/charmbracelet/glamour      (markdown rendering)
golang.org/x/term                      (terminal size, raw mode)
nhooyr.io/websocket                    (WebSocket transport)
```

No CGo. Single static binary.

## 10. Claude Code TUI observation log (2026-07-03)

This section summarizes direct TTY observations of Claude Code/CCR behavior so
future terminal UX changes have a concrete reference instead of relying on
memory. The complete structured observation log lives at
`docs/claude-code-tui-observations.md`. A second tmux/freeze reverse-engineering
pass with screenshot artifacts lives at
`docs/claude-code-tui-reverse-engineering.md` and
`.agent/tui-dev/claude-code-20260703/`. Raw ANSI captures are also stored under
the gitignored directory `.agentm/artifacts/ccr-tui/`:

- `.agentm/artifacts/ccr-tui/ccr-main-20260703-171453.typescript`
- `.agentm/artifacts/ccr-tui/real-dev-flow.typescript`
- `.agentm/artifacts/ccr-tui/agent-team-flow-command.typescript`

Reproduction commands used:

```bash
mkdir -p /tmp/agentm-ccr-ui
git init /tmp/agentm-ccr-ui
script -q .agentm/artifacts/ccr-tui/ccr-main-$(date +%Y%m%d-%H%M%S).typescript \
  ccr code --dangerously-skip-permissions --name agentm-ccr-logged
```

The sessions were driven by sending real keypresses through a PTY, including
`?`, `/`, `Esc`, `Ctrl+C`, `←`, `↓`, `Enter`, `Ctrl+T`, `Ctrl+O`, `Ctrl+E`,
and agents-panel task creation. Later captures exercised real tool/edit/test
work and a successful three-agent workflow, not only login-boundary states.

### 10.1 Main conversation surface

Claude Code keeps the default chat surface simple:

- A welcome/context card appears at the top on a fresh session.
- The composer is anchored at the bottom and remains the primary control.
- The status/help line is intentionally sparse. Idle mode shows entries like
  `? for shortcuts`, `← for agents`, and the model/effort indicator.
- Permission state has higher priority than generic help. With bypass mode
  enabled, the bottom line leads with `⏵⏵ bypass permissions on
  (shift+tab to cycle)` and only then shows secondary affordances.
- `--dangerously-skip-permissions` does not skip workspace trust. Trust is a
  separate one-time gate with `Yes, I trust this folder` / `No, exit`.

Implication for AgentM: status chrome should be priority-based, not a static
list of every available shortcut. Runtime safety/mode state outranks generic
navigation hints.

### 10.2 Inline discovery instead of modal-first UI

Claude Code uses inline affordances near the composer:

- `?` expands a compact shortcut cheat sheet in-place. It is not a modal and
  does not take the user away from the transcript.
- `/` opens an inline command/skill list below the composer and filters as the
  user types.
- `@` opens file path completion from the composer.
- `Esc` and `Ctrl+C` use second-press confirmation:
  - `Esc` on non-empty input shows an "again to clear" style prompt before
    clearing.
  - `Ctrl+C` first shows "Press Ctrl-C again to exit"; the second press exits.
- `←` uses the same pattern for context switch: first press hints "again for
  agents", second press opens the agents manager.

Observed shortcut sheet entries included:

| Key | Observed meaning |
|---|---|
| `!` | shell mode |
| `/` | commands |
| `@` | file paths |
| `/btw` | side question |
| `Shift+Enter` | newline |
| `Shift+Tab` | auto-accept edits / cycle mode depending on state |
| `Ctrl+O` | verbose output |
| `Ctrl+T` | tasks / agents view |
| `Alt+P` | switch model |
| `Ctrl+S` | stash prompt |
| `Ctrl+G` | edit in `$EDITOR` |
| `Ctrl+Z` | suspend |
| `/keybindings` | customize shortcuts |

Implication for AgentM: the default path should be composer-first. Help,
commands, files, and agent/session switching should be discoverable inline;
full dialogs remain useful for dense detail views but should not be the first
interaction for common commands.

### 10.3 Background agents/task manager

Claude Code's background model is not "one tab per worker" by default.
Opening agents/tasks presents a full-screen manager with a task list:

```text
Claude Code v2.1.195
Opus 4.8 (1M context) · /private/tmp/agentm-ccr-ui
3 awaiting input · 0 working · 0 completed

Needs input
✻ ?Create bg-panel.txt wit… login required — run /login 25s
✻ new session              send a prompt to start        52s
✻ bg-ui-sample             send a prompt to start         1m

Each row is its own Claude session. Open one to see its work.
Sessions keep running if you close the terminal.

❯ describe a task for a new session
```

Important behaviors:

- The manager is a separate view from the main chat, not persistent tab chrome.
- It summarizes counts by state: awaiting input, working, completed.
- Rows are grouped by state (`Needs input`, `Working`, etc.).
- Each row has a short title, status/error text, and age.
- The bottom composer creates a new background session from a task description.
- Opening a row switches to that session's normal chat transcript. Returning to
  the manager is done through the same `←` agents affordance.
- The agents-panel help is inline. Observed entries included `alt+1-2 to open`,
  `space to reply`, `ctrl+x to delete`, `ctrl+s to switch views`,
  `ctrl+t to pin to top`, `ctrl+r to rename`, `@ to mention`, `? to close`,
  and `esc to quit`.
- Starting a background session from CLI returns a compact receipt:

```text
backgrounded · 31a4aabe · bg-ui-sample (idle — send a prompt to start)
  claude agents             list sessions
  claude attach 31a4aabe    open in this terminal
  claude logs 31a4aabe      show recent output
  claude stop 31a4aabe      stop this session
```

Implication for AgentM: workflow/sub-agent sessions should be a task manager
concept, not ordinary tab proliferation. Tabs are still useful when the user
explicitly opens a background session, but spawned work should first appear as
compact status plus an agents/tasks view.

### 10.4 Real multi-agent workflow behavior

A later CCR capture forced a real workflow with three agents: planner,
implementer, and QA reviewer. The main session acted as orchestrator. It
spawned three background agents, waited for their results, integrated the
implementer's worktree output, ran tests and the CLI example, and summarized the
agents used.

Observed live UI:

```text
Running 3 agents…
 ├ Plan workflow planner
 ├ Implement workflow planner · 0 tool uses
 │  ⎿  Initializing…
 └ QA review requirements

3 background agents launched (↓ to manage)
```

Opening the picker with `↓` did not create tabs. It exposed an inline task
selector:

```text
↑/↓ to select · Enter to view
Enter to view · x to stop
```

Pressing `Enter` on the implementer switched into that agent's normal
transcript view. The implementer worked in an isolated git worktree under
`.claude/worktrees/agent-.../`; the main session later copied the result into
the primary temp repository and removed the worktree. The first cleanup attempt
failed because the worktree still had modified/untracked files; Claude retried
with forced worktree removal. This is the concrete git edge case AgentM should
handle in workflow cleanup.

The detailed transcript controls are two-level:

```text
Showing detailed transcript · ctrl+o to toggle · ctrl+e to show all verbose
Showing detailed transcript · ctrl+o to toggle · ctrl+e to collapse verbose
```

Implication for AgentM: workflow UX should distinguish three surfaces:

1. Main orchestrator transcript.
2. Inline task picker/status rows for spawned work.
3. Promoted child transcript view only after explicit selection.

### 10.5 Current AgentM terminal direction

The current AgentM terminal implementation follows this direction:

1. Keep the parent conversation primary. Child workflow sessions open as
   background activity and do not steal focus.
2. Collapse background-only child sessions out of tab chrome and render them as
   inline task rows below the composer/status surface. `↓` opens an inline
   picker, `Enter` views a selected task, `x` stops it, and `Ctrl+T`
   hides/shows the task rows.
3. Make the composer the control surface:
   `Enter` sends, busy `Enter` queues cooperatively, `Shift+Enter`/`Ctrl+J`
   inserts newline, `?` opens shortcuts, `/` opens commands, `@` opens files,
   `Esc` interrupts or double-clears input, and `Ctrl+C` exits on second press.
4. Preserve child output for explicit inspection. Routed workflow child streams
   are buffered per session, so selecting a completed task shows its final text
   and completion note instead of a blank promoted tab.
5. Keep transcript detail controls:
   `Ctrl+O` toggles detailed transcript blocks, while `Ctrl+E` toggles fully
   verbose output within detailed mode.
6. Preserve raw TTY captures for significant UX investigations in
   `.agentm/artifacts/<topic>/` and summarize only the decisions in tracked
   design docs.
7. Keep process supervision outside the Go peer. `agentm terminal` owns the
   local gateway daemon/supervisor lifecycle; `agentm-terminal` only connects
   to a gateway URL and renders the wire stream.
8. Remove the legacy right-side session sidebar from the primary AgentM
   terminal layout. Wide screens keep the transcript/composer primary; session
   metadata belongs in status, picker, or detail views rather than a persistent
   side panel.
9. Render transient notices as compact one-line status hints near the bottom
   edge. They should auto-dismiss quickly, truncate long logs/errors, and never
   occupy a large bordered bottom-right card.

The implementation still reuses tab/supervisor plumbing internally; the user
surface is main chat plus workflow rows/picker/detail. The 2026-07-03
terminal-go verification captures under `.agent/tui-dev/captures/` exercised a
real gateway workflow with two parallel child agents. The parent remained on
the main transcript, and the two selected task details rendered `ALPHA` and
`BETA` respectively after completion.
