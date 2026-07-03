# Claude Code TUI Observations

Observed on 2026-07-03 through real PTY sessions driven from Codex. The goal of
this document is to record Claude Code/CCR terminal behavior as a concrete UX
reference for AgentM terminal-go work.

A later tmux/freeze reverse-engineering pass with per-screen captures and SVG
screenshots is tracked in `docs/claude-code-tui-reverse-engineering.md` and
`.agent/tui-dev/claude-code-20260703/`.

Raw TTY captures are intentionally not committed because they contain ANSI
control sequences and local terminal state. They are stored under the gitignored
artifact directory:

- `.agentm/artifacts/ccr-tui/trust-gate.typescript`
- `.agentm/artifacts/ccr-tui/composer-menus.typescript`
- `.agentm/artifacts/ccr-tui/ccr-main-20260703-171453.typescript`
- `.agentm/artifacts/ccr-tui/background-receipt.txt`
- `.agentm/artifacts/ccr-tui/real-dev-flow.typescript`
- `.agentm/artifacts/ccr-tui/agent-team-flow.typescript`
- `.agentm/artifacts/ccr-tui/agent-team-flow-command.typescript`

## Capture Setup

Commands used:

```bash
mkdir -p /tmp/agentm-ccr-ui
git init /tmp/agentm-ccr-ui
printf 'hello\n' > /tmp/agentm-ccr-ui/sample.txt

script -q .agentm/artifacts/ccr-tui/composer-menus.typescript \
  ccr code --dangerously-skip-permissions --name composer-menus

script -q .agentm/artifacts/ccr-tui/ccr-main-20260703-171453.typescript \
  ccr code --dangerously-skip-permissions --name agentm-ccr-logged

ccr code --dangerously-skip-permissions --bg --name bg-receipt-sample \
  'Stay idle; this is only to record the background-agent receipt UI.' \
  | tee .agentm/artifacts/ccr-tui/background-receipt.txt

script -q .agentm/artifacts/ccr-tui/real-dev-flow.typescript \
  ccr code --dangerously-skip-permissions --name real-dev-flow

script -q .agentm/artifacts/ccr-tui/agent-team-flow-command.typescript \
  ccr code --dangerously-skip-permissions --name agent-team-flow-command
```

Observed versions in the captures:

- Claude Code `v2.1.195` and `v2.1.199`
- CCR wrapper launched with `ccr code`

## Trust Gate

First launch in an untrusted workspace shows a full-screen trust gate before
the chat UI.

Observed content:

```text
Accessing workspace:

/Users/bytedance

Quick safety check: Is this a project you created or one you trust?

Claude Code'll be able to read, edit, and execute files here.

Security guide

❯ 1. Yes, I trust this folder
  2. No, exit

Enter to confirm · Esc to cancel
```

Notes:

- `--dangerously-skip-permissions` does not skip the trust gate.
- The displayed workspace may be broader than the process working directory.
  In one CCR run launched from `/tmp/agentm-ccr-trust`, the trust screen showed
  `/Users/bytedance`.
- Selection uses normal TUI navigation and `Enter`; it should not be treated as
  a free-text prompt.

## Main Idle Layout

The default chat surface is composer-first.

Observed elements:

- Top welcome/context card.
- Bottom composer with a visible session name on the upper border when provided
  by `--name`.
- Sparse bottom status/help line.
- Model/effort indicator on the right.
- Permission mode indicator has high priority.

Observed examples:

```text
────────────────────────────────────────────── composer-menus ──
❯
────────────────────────────────────────────────────────────────
⏵⏵ bypass permissions on (shift+tab to cycle) · ← for agents
                                                  ● high · /effort
```

Without the bypass-permissions emphasis, the idle hint line uses entries like:

```text
? for shortcuts · ← for agents                 ● high · /effort
```

Implications:

- The status line is priority-based.
- Runtime mode/safety state displaces generic shortcuts.
- The UI does not permanently show a complete keybinding list.

## Inline Shortcuts

Pressing `?` from the main composer expands an inline shortcut sheet near the
composer. It is not a modal dialog and does not navigate away from the current
conversation.

Observed entries:

| Key | Meaning shown |
|---|---|
| `!` | shell mode |
| `/` | commands |
| `@` | file paths |
| `/btw` | side question |
| `Shift+Enter` | newline |
| `Shift+Tab` | auto-accept edits, or cycle permission/mode depending on context |
| `Ctrl+O` | verbose output |
| `Ctrl+T` | toggle tasks |
| `Opt+P` / `Alt+P` | switch model |
| `Ctrl+S` | stash prompt |
| `Ctrl+G` | edit in `$EDITOR` |
| `Ctrl+Z` | suspend |
| `Ctrl+V` | paste images |
| `/keybindings` | customize keybindings |
| double `Esc` | clear input |
| `Ctrl+Shift+_` | undo |

Behavioral notes:

- The shortcut sheet is inline and compact.
- It can be dismissed by continuing with another composer interaction.
- It does not use a centered overlay unless a deeper command requires one.

## Slash Commands

Typing `/` opens an inline command list below the composer.

Observed behavior:

- The list appears in-place under the input.
- Items include built-in commands, skills, and project/user commands.
- Each row has a command name and a short description.
- Long descriptions are truncated to fit the terminal width.
- The list filters as the user types.

Observed examples:

```text
/init                         Initialize a new CLAUDE.md file with codebase documentation
/autoharness:paper-review     (autoharness) Run the paper-writing three-pass...
/rca-failure-analysis         分析RCA（根因分析）Agent的执行结果...
/lark-doc                     飞书云文档（Docx / Wiki 文档，v2 API）...
/add-dir                      Add a new working directory
```

Design rule to copy:

- `/` is the primary command discovery affordance.
- A separate command palette can exist, but should not be required for common
  slash commands.

## Resource Mentions

Typing `@` opens a unified resource completion list.

Observed rows:

```text
+ tty/
+ .git/
+ sample.txt
* autoharness:code-reviewer       (agent) – Reviews a diff or worktree branch...
* autoharness:dev-worker          (agent) – Implements a feature, fix, or refactor...
* autoharness:merge-agent         (agent) – Integrates one or more worktree branches...
* claude                          (agent) – Catch-all for any task...
```

Notes:

- `@` is not only file completion. It is a unified mention surface for files,
  directories, and agents.
- File/directory entries use a different prefix from agent entries.
- Agent entries include a short role description.

Design rule to copy:

- Treat `@` as "mention a resource" rather than "attach file only".

## Esc Behavior

Observed sequence with an active completion/input:

1. First `Esc` closes the inline menu and preserves the typed input.
2. Another `Esc` on non-empty input shows an inline hint like:

```text
Esc again to clear
```

3. A follow-up `Esc` within the confirmation window clears the input.

Observed subtlety:

- The timing/state window is strict. In one capture, repeated `Esc` showed the
  hint again instead of clearing, which suggests the clear confirmation is
  stateful and not simply "any second Esc ever".

Design rule to copy:

- `Esc` should first cancel the transient UI, then interrupt active work, then
  ask for a second press before clearing user text. It should not silently send
  or discard typed input.

## Ctrl+C Behavior

Observed behavior:

- First `Ctrl+C` does not exit. It shows:

```text
Press Ctrl-C again to exit
```

- A second `Ctrl+C` in the confirmation window exits.
- If the second press is too late, the normal status line returns and another
  first press is required.

Design rule to copy:

- Exit confirmation should be inline and second-press based in the normal chat
  surface. A modal exit dialog is too heavy for the default case.

## Agents / Tasks Entry

The main composer advertises:

```text
← for agents
```

Observed behavior:

- First `←` shows a hint like "again for agents".
- Two left-arrow presses close together open the agents manager.
- Opening the agents manager performs a full view switch; it is not a small
  popup.

Design rule to copy:

- Accidental cross-view navigation is guarded by a second press.

## Background Agent CLI Receipt

Starting a background agent from CLI returns immediately with a compact receipt.

Observed output:

```text
backgrounded · 79aca865 · bg-receipt-sample (idle — send a prompt to start)
  claude agents             list sessions
  claude attach 79aca865    open in this terminal
  claude logs 79aca865      show recent output
  claude stop 79aca865      stop this session
```

Notes:

- A background agent can be created idle.
- The receipt is action-oriented: list, attach, logs, stop.
- It gives a short stable id for later operations.

Design rule to copy:

- Any background workflow launch should return a compact, actionable receipt
  rather than dumping internal process/session details.

## Agents Manager Layout

The agents manager is a full-screen task/session list.

Observed layout:

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

Core behaviors:

- Top summary counts sessions by state.
- Rows are grouped by state (`Needs input`, `Working`, `Completed`, etc.).
- Each row shows:
  - status glyph/spinner,
  - compact title,
  - status/error text,
  - age.
- Bottom composer creates a new background session from a plain-language task.
- The selected row drives contextual footer help.

Observed footer/help entries:

| Key | Meaning shown |
|---|---|
| `Enter` | open / collapse depending on row state |
| `Space` | reply |
| `Ctrl+X` | delete / delete all depending on context |
| `Ctrl+R` | rename |
| `Ctrl+S` | switch views |
| `Ctrl+T` | pin to top |
| `@` | mention |
| `Alt+1-2` | open numbered row |
| `?` | show/close agents help |
| `Esc` | quit agents view |

Design rule to copy:

- Background work should live in a task manager view with grouped state rows,
  not in always-visible tabs.

## Opening A Background Session

Opening a row in the agents manager switches into that session's normal chat
transcript.

Observed detail view:

```text
❯ ?Create bg-panel.txt with one sentence, then run sleep 15 before your final response...
⎿  Not logged in · Please run /login

✻ Cogitated for 0s
                                                   Not logged in · Run /login
────────────────────────────────────────── claude ──
❯
────────────────────────────────────────────────────
⏵⏵ bypass permissions on (shift+tab to cycle) · ← for agents
```

Notes:

- Opening a background item does not embed a nested panel.
- It becomes a normal chat session view.
- Returning to the manager uses the same `← for agents` affordance.

Design rule to copy:

- Background tasks should be inspectable as full sessions, but only after the
  user explicitly opens them.

## Failure / Needs Input State

The captured background task hit a login boundary:

```text
login required — run /login
```

This appeared in:

- the agents manager row status,
- the opened session transcript,
- the session status area.

Design rule to copy:

- Workflow failures or blocked states should surface both in the list row and
  inside the opened transcript. The row gives triage; the transcript gives
  details and a place to reply.

## Real Development Flow

`real-dev-flow.typescript` captured a real single-session implementation task
in `/tmp/agentm-ccr-realflow`.

Task summary:

- Implement `src/workflow_planner.py` with parse/order/batch helpers.
- Add tests.
- Add a README usage snippet.
- Run tests and fix failures.

Observed behavior:

- Claude read project guidance before editing (`AGENTS.md`, `README.md`).
- Tool calls render as compact transcript rows: `Bash(...)`, `Write(...)`,
  `Read(...)`, and `git diff` / `git status`.
- Long command output is folded with a `ctrl+o to expand` hint.
- The spinner/status line shows elapsed time and token direction/count.
- A failed command stays in the transcript as an error block:

```text
Bash(python -m unittest discover -s tests)
Error: Exit code 127
(eval):1: command not found: python
```

- Claude recovered by running `python3 -m unittest discover -s tests`.
- The final summary included changed files and the verification result.
- Stop hooks ran after the response. A non-blocking autoharness hook error was
  shown inline but did not mark the user task as failed.

External verification after the TTY session:

```text
python3 -m unittest discover -s tests
Ran 7 tests in 0.000s
OK
```

Design rule to copy:

- Failed tool calls should stay visible as first-class events, and successful
  retries should be easy to see without reading raw logs.

## Real Multi-Agent Workflow

`agent-team-flow-command.typescript` captured a real multi-agent development
workflow in `/tmp/agentm-ccr-teamflow.qdUhYK`. The prompt explicitly required
three agents/tasks: planner, implementer, and QA reviewer.

Observed launch sequence:

```text
⎿  ◻ Plan workflow planner implementation
   ◻ Implement workflow planner package
   ◻ Review workflow planner quality

Agent(Plan workflow planner)
Backgrounded agent (↓ to manage · ctrl+o to expand)

Running 3 agents…
 ├ Plan workflow planner
 ├ Implement workflow planner · 0 tool uses
 │  ⎿  Initializing…
 └ QA review requirements

3 background agents launched (↓ to manage)
```

Important behavior:

- The main session acted as orchestrator and retained the full context.
- Agent tasks started in the background and appeared as task/checklist rows,
  not as new visible tabs.
- The bottom area showed `main` plus background agent rows with role, title,
  and age.
- Individual agent completion was reported inline:

```text
Agent "Plan workflow planner" finished · 33s
Agent "QA review requirements" finished · 47s
Agent "Implement workflow planner" finished · 2m 25s
```

- The implementer worked in an isolated git worktree under
  `.claude/worktrees/agent-a8a7361850faf3546/`.
- The main session later copied/integrated the implementer's output into the
  primary temporary repository, ran tests, ran the CLI example, checked git
  status, and cleaned up the agent worktree.
- Worktree cleanup first failed because the worktree still had modified or
  untracked files. Claude retried with `git worktree remove --force`, which
  succeeded. This is a concrete workflow edge case for AgentM merge/cleanup
  logic.
- The final summary reported agents used, files changed, test result, and CLI
  check.

External verification after the TTY session:

```text
python3 -m unittest discover -s tests
Ran 17 tests in 0.064s
OK

python3 -m workflow_planner examples/tasks.json
{
  "order": ["spec", "api", "ui", "tests"],
  "batches": [["spec"], ["api", "ui"], ["tests"]],
  "owner_batches": [
    {"pm": ["spec"]},
    {"backend": ["api"], "frontend": ["ui"]},
    {"qa": ["tests"]}
  ]
}
```

The temporary repo still had normal work-in-progress changes:

```text
 M README.md
?? tests/
?? workflow_planner/
```

It also had a `.git/index.lock` artifact after the observed run. That should
be treated as a reminder that workflow cleanup needs to handle git lockfiles
and interrupted git operations explicitly.

## Agent Workflow Key Behavior

The multi-agent session was used to press real keys while agents were running.

Observed key behavior:

| Key | Observed behavior |
|---|---|
| `↓` | Opens an inline background-agent picker from the main chat. |
| `↑` / `↓` in picker | Moves selection; footer changes with selected row context. |
| `Enter` in picker | Opens the selected agent/session transcript in the same TUI surface. |
| `x` in picker | Footer advertises stop for the selected agent; not pressed in the capture. |
| `←` inside picker/detail | No visible action in this state. |
| `Ctrl+T` | Toggles task/picker visibility when in the normal main view; effect is stateful and context-dependent. |
| `Ctrl+O` | Toggles detailed transcript for folded tool/search/write output. |
| `Ctrl+E` in detailed transcript | Expands all verbose output; pressing again collapses verbose back to detailed mode. |
| `Ctrl+C` | First press asks for confirmation; double press exits. |

Observed detail-mode footer:

```text
Showing detailed transcript · ctrl+o to toggle · ctrl+e to show all verbose
Showing detailed transcript · ctrl+o to toggle · ctrl+e to collapse verbose
```

Observed picker footer variants:

```text
↑/↓ to select · Enter to view
Enter to view · x to stop
```

Design rules to copy:

- Keep background work inspectable without creating visible tabs.
- Make the task picker inline and reversible.
- `Enter` promotes a selected background task into a normal transcript view.
- `Ctrl+O` and `Ctrl+E` should form a two-level detail model: detailed vs.
  fully verbose.
- Stop/delete controls should be advertised contextually but not easy to hit by
  accident.

## Observed But Not Fully Exercised

These behaviors still need future capture:

- edit approval / auto-accept edit states in non-bypass permission mode,
- shell mode entered by `!`,
- model picker opened by `Alt+P`,
- `/btw` side-question flow,
- completed background task row from `claude agents`,
- attached logs through `claude logs <id>`.

## AgentM Copy Targets

The following behaviors are appropriate to copy directly:

1. Composer-first layout with minimal persistent chrome.
2. Priority-based bottom status line.
3. Inline `?`, `/`, and `@` discovery surfaces.
4. `@` as unified resource mention: files, dirs, agents, and later workflows.
5. Second-press confirmations for destructive/large state changes:
   `Esc`, `Ctrl+C`, and cross-view navigation.
6. Background task manager instead of automatic visible tabs.
7. Background launch receipt with short id and list/attach/logs/stop actions.
8. Opening a task promotes it to a normal session view; returning uses the same
   agents/tasks affordance.
9. Row-level state summaries for blocked/working/completed workflows.
10. Raw TTY capture as the standard artifact for UX investigations.
