# Claude Code TUI Reverse Engineering

Captured on 2026-07-03 with the local `tui-dev` skill. The goal is to turn
Claude Code's terminal behavior into a concrete reference for AgentM terminal
workflow UX.

## Capture Method

Tooling:

- `tmux` as the controlled terminal host.
- `tmux capture-pane -p -e` for ANSI captures.
- `freeze` for SVG screenshots.
- `ccr code --dangerously-skip-permissions` and `claude agents` for real
  Claude Code sessions.

Artifacts live under:

```text
.agent/tui-dev/claude-code-20260703/
├── captures/
└── screenshots/
```

The `captures/*.txt` files are plain terminal text. The `captures/*.ansi`
files preserve color/control sequences. The `screenshots/*.svg` files are
rendered from the captured terminal state.

## Artifact Index

| Screen | Capture | Screenshot | Purpose |
|---|---|---|---|
| Trust gate | `01-trust-gate.txt` | `01-trust-gate.svg` | Workspace safety gate before chat. |
| Main idle | `02-main-idle.txt` | `02-main-idle.svg` | Composer-first base layout. |
| Shortcut sheet | `03-shortcuts.txt` | `03-shortcuts.svg` | Inline `?` help surface. |
| Slash menu | `04-slash-menu.txt` | `04-slash-menu.svg` | Inline command/skill discovery. |
| Resource mentions | `05-resource-mentions.txt` | `05-resource-mentions.svg` | Unified `@` files/directories/agents list. |
| Esc confirmation | `06-esc-clear-confirm.txt` | `06-esc-clear-confirm.svg` | Double-Esc clear guard. |
| Ctrl-C confirmation | `07-ctrlc-confirm.txt` | `07-ctrlc-confirm.svg` | Double-Ctrl-C exit guard. |
| Background receipt | `08-background-receipt.txt` | none | CLI receipt for background session. |
| Agents manager | `09-agents-manager.txt` | `09-agents-manager.svg` | Full-screen background session manager. |
| Agents help | `10-agents-manager-help.txt` | `10-agents-manager-help.svg` | Row-action help in manager. |
| Background detail | `11-background-detail.txt` | `11-background-detail.svg` | Background row promoted to normal transcript. |
| Team trust | `12-team-trust-or-idle.txt` | `12-team-trust-or-idle.svg` | Trust gate for multi-agent sandbox. |
| Team prompt submitted | `13-team-prompt-submitted.txt` | `13-team-prompt-submitted.svg` | Submitted prompt while main is working. |
| Agents launched | `14-team-agents-launched.txt` | `14-team-agents-launched.svg` | Main starts three background agents. |
| Agent picker | `15-team-agent-picker.txt` | `15-team-agent-picker.svg` | Inline `↓` task picker. |
| Agent detail | `16-team-agent-detail.txt` | `16-team-agent-detail.svg` | Selected worker transcript. |
| Detailed transcript | `17-team-detail-ctrl-o.txt` | `17-team-detail-ctrl-o.svg` | `Ctrl+O` detailed transcript mode. |
| Verbose transcript | `18-team-detail-ctrl-e.txt` | `18-team-detail-ctrl-e.svg` | `Ctrl+E` full verbose mode. |
| Back to main | `19-team-back-to-main.txt` | `19-team-back-to-main.svg` | Main waiting for task outputs. |
| Agents finished | `20-team-complete.txt` | `20-team-complete.svg` | All agents finished, main still integrating. |
| Verification running | `21-team-final-summary.txt` | `21-team-final-summary.svg` | Tests/CLI/diff running. |
| Final summary start | `22-team-final-summary.txt` | `22-team-final-summary.svg` | Final summary starts. |
| Final idle | `23-team-final-idle.txt` | `23-team-final-idle.svg` | Completed response and idle composer. |

## Surface Model

Claude Code's UI is not organized around permanent tab chrome. It is organized
around a primary composer plus contextual task surfaces.

The useful mental model:

1. Main transcript is the anchor.
2. Background work appears as compact task rows.
3. A picker can temporarily select/open background work.
4. Opening a task promotes it to a normal transcript view.
5. Detailed transcript modes are opt-in and reversible.

This maps directly to the AgentM direction: the main agent should stay the
customer-facing orchestrator, while worker agents are background workflow rows
until explicitly opened.

## Static Surfaces

### Trust Gate

The trust gate is a full-screen pre-chat state. It is not part of the
conversation transcript and is not skipped by bypass-permissions mode.

Important details:

- Shows the exact workspace path.
- Explains that Claude can read, edit, and execute files there.
- Uses a numbered two-option selection.
- `Enter` confirms; `Esc` cancels.

AgentM implication: workspace/session trust is a separate state machine from
tool permission mode.

### Main Idle

The base layout is composer-first:

- Welcome/context panel at top.
- Chat composer anchored at bottom.
- Session name on the composer border.
- Permission mode and model/effort in the footer.
- Minimal persistent help.

The status line prioritizes current mode over full keybinding discoverability:

```text
⏵⏵ bypass permissions on (shift+tab to cycle) · ← for agents
```

AgentM implication: do not permanently render every command. Render the mode
that currently matters, and let `?`, `/`, and `@` handle discovery.

### Inline Discovery

`?`, `/`, and `@` are all inline surfaces near the composer.

`?` shows a compact key sheet, not a modal:

```text
! for shell mode        double tap esc to clear input      ctrl + shift + _ to undo
/ for commands          shift + tab to auto-accept edits   ctrl + z to suspend
@ for file paths        ctrl + o for verbose output        ctrl + v to paste images
/btw for side question  ctrl + t to toggle tasks           opt + p to switch model
```

`/` shows commands, skills, and project/user actions in one filtered list.

`@` is a unified resource list. It includes files, directories, and agents:

```text
+ README.md
+ AGENTS.md
+ .git/
* autoharness:code-reviewer (agent) – Reviews a diff or worktree branch...
* autoharness:dev-worker (agent) – Implements a feature, fix, or refactor...
* claude (agent) – Catch-all for any task...
```

AgentM implication: `@` should become a resource mention surface, not a
file-only completion.

### Second-Press Guards

Esc and Ctrl-C avoid destructive surprises:

- `Esc` closes transient UI first, then shows `Esc again to clear`.
- `Ctrl-C` shows `Press Ctrl-C again to exit`.

AgentM implication: interruption, clearing input, and exiting should be guarded
with inline second-press state, not modal dialogs.

## Background Agents Manager

`claude agents --cwd <repo>` opens a full-screen manager:

```text
Claude Code v2.1.199
Opus 4.8 (1M context) · /private/tmp/agentm-tui-ui.ybNvf2
1 awaiting input · 0 working · 0 completed

Needs input
✻ tmux-bg-manager  send a prompt to start  27s

❯ describe a task for a new session
```

The manager provides:

- State counts.
- Grouped rows.
- Row title, status, and age.
- A bottom composer for new background work.
- Contextual row actions:

```text
ctrl+r to rename          @ to mention            alt+1 to open    ? to close
ctrl+s to switch views    ctrl+t to pin to top    esc to quit
```

Opening a row switches into a regular transcript for that session. The manager
is not embedded inside the chat; it is a view switch.

AgentM implication: a future workflow manager should be full-screen or
main-surface-level, with row actions and a plain-language task composer.

## Multi-Agent Workflow

The real multi-agent task used three background agents: planner, implementer,
and QA reviewer. The main session started them and kept the orchestration
context.

Launch state:

```text
3 background agents launched (↓ to manage)
├ Plan task graph helper
├ Implement task graph helper
└ QA review task graph
```

The footer switched to workflow controls:

```text
ctrl+t to hide tasks · ← for agents · ↓ to manage
```

The bottom task rows showed:

```text
⏺ main
◯ general-purpose  Plan task graph helper       0s
◯ general-purpose  Implement task graph helper  0s
◯ general-purpose  QA review task graph         0s
```

Opening the picker with `↓` changed the footer:

```text
↑/↓ to select · Enter to view
```

Selecting a worker and pressing `Enter` promoted it into a normal transcript
view with its own title:

```text
──────────────────────── Implement task graph helper ──
Enter to view · x to stop · ctrl+x ctrl+k to stop all agents
```

AgentM implication: worker sessions should be inspectable on demand without
stealing the main conversation when spawned.

## Detail And Verbose Modes

Claude Code uses two levels of transcript expansion:

1. `Ctrl+O`: detailed transcript mode.
2. `Ctrl+E`: full verbose expansion inside detailed mode.

Footer states:

```text
Showing detailed transcript · ctrl+o to toggle · ctrl+e to show all verbose
Showing detailed transcript · ctrl+o to toggle · ctrl+e to collapse verbose
```

AgentM implication: do not make every tool payload fully visible by default.
Use a two-level disclosure model: normal summary, detailed, fully verbose.

## Workflow Completion

The main session waited for all background agents, then integrated outputs and
ran verification:

```text
Bash(python3 -m unittest discover -s tests)
Ran 5 tests in 0.000s
OK

Bash(python3 -m taskflow examples/tasks.json)
```

Final summary included:

- Agents used.
- Files changed.
- Test command and result.
- CLI output.
- Current working tree status.

The final screen also showed a non-blocking stop-hook error inline:

```text
Ran 3 stop hooks (ctrl+o to expand)
Stop hook error: Failed with non-blocking status code...
```

AgentM implication: workflow completion should distinguish the user task
result from non-blocking hook/cleanup errors. The latter should be visible but
must not obscure the main outcome.

## AgentM UX Targets

Copy these behaviors:

1. Main conversation stays primary.
2. Spawned workflow agents appear as background rows, not new tabs.
3. `↓` opens a task picker; `Enter` opens selected task detail.
4. `Ctrl+T` hides/shows task rows without cancelling work.
5. `Ctrl+O` and `Ctrl+E` provide two-level transcript expansion.
6. `?`, `/`, and `@` are inline composer surfaces.
7. Esc/Ctrl-C use inline second-press guards.
8. Background manager rows expose status, age, and contextual actions.
9. Final workflow summaries include agent roles, files, tests, CLI checks, and
   working tree status.
10. Non-blocking hook/cleanup errors remain visible as secondary events.

Avoid these in AgentM:

1. Visible tab proliferation for every spawned worker.
2. Modal-first help for common composer actions.
3. Treating files and agents as separate mention mechanisms.
4. Hiding retry/failure events from the normal transcript.
5. Losing main conversation focus when a workflow starts.
