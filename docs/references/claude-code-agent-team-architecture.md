# Claude Code Agent Team Architecture — Reverse-Engineering Analysis

**Purpose**: Reference document for AgentM development. Analyzes how Claude Code implements its multi-agent orchestration system, to inform our own design decisions.

**Date**: 2026-03-26
**Sources**: Official Anthropic docs, community reverse-engineering, binary analysis (cli.mjs v2.1.47)

---

## 1. Two-Level Agent Hierarchy

Claude Code distinguishes two levels of agents:

| Level | Name | Lifetime | Communication | Nesting |
|-------|------|----------|---------------|---------|
| **Subagent** | One-shot worker | Single tool call | Result returned to parent only | Cannot spawn children |
| **Teammate** | Persistent team member | Until shutdown | Peer-to-peer messaging | Cannot spawn children |

A **Subagent** is lightweight: it runs, returns a result, and is gone. A **Teammate** persists across multiple turns, has its own inbox, and can communicate with any other teammate on the team.

The main conversation (Team Lead) is the only entity that can spawn agents. Agents themselves cannot spawn sub-agents — the hierarchy is strictly two levels deep.

---

## 2. Runtime Architecture

```
┌─────────────────────────────────────────────────────┐
│              Claude Code Harness (Node.js)            │
│         (single cli.mjs ~20MB, obfuscated)           │
│                                                       │
│  ┌─────────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Main Session │  │ Agent A  │  │ Agent B  │        │
│  │ (Team Lead)  │  │ (worker) │  │ (worker) │        │
│  └──────┬──────┘  └────┬─────┘  └────┬─────┘        │
│         │               │              │              │
│         ▼               ▼              ▼              │
│  ┌──────────────────────────────────────────────┐    │
│  │        File-system message bus                │    │
│  │  ~/.claude/teams/{name}/inboxes/{agent}.json  │    │
│  │  ~/.claude/tasks/{name}/{id}.json             │    │
│  └──────────────────────────────────────────────┘    │
│         │                                             │
│         ▼                                             │
│  ┌──────────────────────────────────────────────┐    │
│  │  Claude API (remote, stateless)               │    │
│  └──────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### Key Insight: The LLM Is Not a Process

Claude (the LLM) is a stateless API endpoint. It does not "run" or "listen". The **Harness** — a Node.js CLI application — is the actual process that:

1. Maintains conversation history in memory
2. Calls the Claude API with accumulated messages
3. Intercepts tool calls from the response and executes them locally
4. Injects results (tool outputs, agent notifications) back into the message history
5. Repeats until the LLM emits `stop_reason: "end_turn"`

Every "agent" is just an independent conversation loop managed by the Harness. There is no persistent LLM process.

---

## 3. Spawn Backends

Teammates have three execution backends, selected by environment:

| Backend | Process Model | When Used |
|---------|--------------|-----------|
| **in-process** | Async task in same Node.js event loop | Default fallback |
| **tmux** | Independent `claude` CLI process per tmux pane | Auto-detected if in tmux session |
| **iTerm2** | Independent `claude` CLI process per split pane | macOS + iTerm2 only |

Default is `"auto"`: use tmux if available, otherwise in-process.

### In-Process Isolation

When running in-process, multiple agents share one Node.js process. Isolation is achieved via **`AsyncLocalStorage`** (Node.js built-in), which provides per-async-context storage:

```typescript
// Conceptual — fields stored in AsyncLocalStorage per agent
interface AgentContext {
  agentId: string;
  agentName: string;
  teamName: string;
  parentSessionId: string;
  color: string;           // terminal UI color coding
  planModeRequired: boolean;
}
```

### Out-of-Process (tmux / iTerm2)

Each teammate is a fully independent `claude` CLI process. Spawn is sequential — roughly 6-7 seconds between agents in a 4-agent team (each needs its own API handshake and system prompt loading).

---

## 4. Communication: File-System Message Passing

All inter-agent communication is **file-based**. This is a deliberate design choice: files are the only IPC mechanism that works identically across in-process, tmux, and iTerm2 backends.

### 4.1 Directory Layout

```
~/.claude/
├── teams/{team-name}/
│   ├── config.json                  # Team metadata + member roster
│   └── inboxes/
│       ├── team-lead.json           # Team lead's inbox
│       ├── researcher.json          # Teammate inbox
│       └── implementer.json         # Teammate inbox
│
└── tasks/{team-name}/
    ├── .lock                        # flock() mutex for task operations
    ├── .highwatermark               # Auto-increment counter for task IDs
    ├── 1.json                       # Task files
    ├── 2.json
    └── 3.json
```

### 4.2 Inbox Protocol

Each agent has a JSON array file as its inbox:

```json
[
  {
    "from": "team-lead",
    "text": "{\"type\":\"task_assignment\",\"taskId\":\"3\"}",
    "timestamp": "2026-03-26T10:15:00.000Z",
    "read": false
  },
  {
    "from": "researcher",
    "text": "I found the root cause — see task #2 results",
    "timestamp": "2026-03-26T10:17:30.000Z",
    "read": false
  }
]
```

**Write path**: Sender reads the entire file, deserializes, pushes a new entry, serializes, writes back. O(N) per message.

**Read path**: Receiver **polls** its inbox file at intervals. New (unread) messages are injected as synthetic conversation turns into the agent's API call.

There is no central message broker, no WebSocket, no push notification — purely file append + poll.

### 4.3 Message Types

| Type | Direction | Purpose |
|------|-----------|---------|
| Plain text | Any → Any | General communication |
| `idle_notification` | Agent → Lead | "I finished my turn, waiting for input" |
| `shutdown_request` | Lead → Agent | Graceful termination request |
| `shutdown_response` | Agent → Lead | Acknowledge shutdown (approving terminates the agent) |
| `task_assignment` | Lead → Agent | Assign a task |
| `plan_approval_request` | Agent → Lead | Request approval for a plan |
| `plan_approval_response` | Lead → Agent | Approve or reject with feedback |
| `broadcast` | Lead → All | Fan-out to all teammates |

### 4.4 Message Injection into Conversation

When the Harness detects new messages in an agent's inbox, it constructs a synthetic `role: "user"` message and appends it to the conversation history before the next API call:

```
[Agent notification from "researcher"]: I found the root cause — the connection pool
is exhausted due to a leaked goroutine in the health check handler.
```

From the LLM's perspective, this looks like the user typed something. The Harness controls what the LLM "sees" by manipulating the messages array.

---

## 5. Task System

### 5.1 Task File Format

Each task is a JSON file at `~/.claude/tasks/{team-name}/{id}.json`:

```json
{
  "id": "1",
  "subject": "Investigate memory leak in auth service",
  "description": "Check heap profiles from the last 24h...",
  "activeForm": "Investigating memory leak",
  "status": "in_progress",
  "owner": "researcher",
  "blocks": ["3"],
  "blockedBy": [],
  "metadata": {}
}
```

### 5.2 State Machine

```
pending ──→ in_progress ──→ completed
   │                            │
   └──────→ deleted ←───────────┘ (also from any state)
```

When a task transitions to `completed`, all tasks listing it in `blockedBy` are automatically unblocked.

### 5.3 Concurrency Control

Multiple agents may attempt to claim the same task simultaneously. The system uses **`flock()`** (POSIX file lock) on a `.lock` file in the task directory:

```typescript
// Conceptual locking flow
async function claimTask(taskId: string, agentName: string) {
  const lockFd = await open(`.lock`, 'r');
  await flock(lockFd, LOCK_EX);  // exclusive lock
  try {
    const task = JSON.parse(await readFile(`${taskId}.json`));
    if (task.owner) throw new Error('Already claimed');
    task.owner = agentName;
    task.status = 'in_progress';
    await writeFile(`${taskId}.json`, JSON.stringify(task));
  } finally {
    await flock(lockFd, LOCK_UN);
    await close(lockFd);
  }
}
```

### 5.4 ID Generation

The `.highwatermark` file stores the next available task ID as a plain integer. Read + increment + write under the same `flock()`.

---

## 6. Agent Lifecycle

### 6.1 Startup

```
TeamCreate
  → mkdir ~/.claude/teams/{name}/
  → mkdir ~/.claude/tasks/{name}/
  → write config.json (empty members array)

Agent(name="researcher", team_name="my-team")
  → spawn process/async-task
  → register in config.json members array
  → create inbox file: inboxes/researcher.json = []
  → inject initial prompt as first user message
  → start conversation loop
```

### 6.2 Steady State: The Turn Cycle

```
┌─→ Poll inbox for new messages
│   │
│   ▼
│   Compose messages array (history + new inbox messages)
│   │
│   ▼
│   Call Claude API
│   │
│   ▼
│   Process tool calls (Read, Write, Bash, SendMessage, TaskUpdate, ...)
│   │
│   ▼
│   LLM emits end_turn
│   │
│   ▼
│   Send idle_notification to team lead
│   │
│   ▼
│   Block waiting for new inbox message ←──────────────────┘
```

### 6.3 Idle / Wake Mechanism

- **Idle** is the normal resting state after each turn. It is NOT an error.
- An idle agent **can receive messages** — writing to its inbox wakes it on the next poll cycle.
- **Heartbeat timeout**: 5 minutes of no activity → agent marked as `inactive`, its tasks can be reclaimed by healthy teammates.
- **`TeammateIdle` hook**: Fires before idle. If hook exits with code 2, idle is prevented and the agent continues working.

### 6.4 Shutdown

```
Team Lead sends: SendMessage(to="researcher", message={type: "shutdown_request"})
  → written to researcher's inbox
  → researcher's next poll picks it up
  → researcher responds: {type: "shutdown_response", approve: true}
  → Harness terminates researcher's conversation loop
  → removed from config.json members array

After all teammates shut down:
  TeamDelete → rm -rf ~/.claude/teams/{name}/ and ~/.claude/tasks/{name}/
```

---

## 7. Foreground vs Background Subagents

Non-team subagents (one-shot workers) have two modes:

| Mode | Behavior | Use Case |
|------|----------|----------|
| **Foreground** | Blocks the parent's tool call until subagent completes | Need result before proceeding |
| **Background** | Returns immediately; result delivered as notification when done | Independent parallel work |

Background mode can be set at spawn time (`run_in_background: true`) or toggled mid-flight (user presses Ctrl+B). The result is injected into the parent conversation as a synthetic message when the subagent's conversation loop terminates.

---

## 8. Built-in Subagent Types

| Type | Model | Tools | Purpose |
|------|-------|-------|---------|
| `general-purpose` | Inherits parent | All | Full-capability worker |
| `Explore` | Haiku | Read-only (Glob, Grep, Read, Bash) | Fast codebase search |
| `Plan` | Inherits parent | Read-only | Architecture planning |
| Custom (`.claude/agents/`) | Configurable | Configurable | Project-specific roles |

Each type is defined by:
1. A system prompt (instructions, role, constraints)
2. A tool allowlist (what the agent can do)
3. A model selection (which Claude model to use)

Custom agents are defined as markdown files in `.claude/agents/` with YAML frontmatter specifying tools and model.

---

## 9. What Claude Code Does NOT Have

Understanding limitations is as important as understanding capabilities:

| Missing Feature | Implication |
|----------------|-------------|
| No shared memory/state | Agents communicate only via messages and task files; no shared data structure |
| No agent nesting | Agents cannot spawn sub-agents; hierarchy is flat (lead + workers) |
| No push notifications | All communication is pull-based (polling), introducing latency |
| No structured state machine | Agent behavior is prompt-guided, not graph-enforced |
| No checkpoint/resume | If the Harness crashes, team state on disk may be stale |
| No cross-team communication | Teams are isolated; agents in team A cannot message team B |

---

## 10. Design Principles Worth Adopting

### 10.1 File System as Universal IPC

**Principle**: Use the file system as the communication substrate.

**Why it works**: Files are the lowest common denominator. They work across processes, across terminals, even across machines (with NFS/shared volumes). No daemon, no port, no protocol negotiation.

**Trade-off**: Higher latency (polling interval), O(N) message writes, no delivery guarantees. Acceptable for human-in-the-loop workflows where seconds of latency don't matter.

### 10.2 Harness-Controlled Conversation

**Principle**: The LLM never "runs" — a local harness orchestrates API calls and injects context.

**Why it matters**: This gives the harness complete control over what the LLM sees. Agent notifications, tool results, system messages — all injected as synthetic conversation turns. The LLM doesn't need to know about the orchestration machinery.

### 10.3 Prompt-Guided, Not Graph-Enforced

**Principle**: Agent behavior is shaped by system prompts and tool availability, not by a rigid state machine.

**Why it matters**: Flexible enough for diverse tasks. The same agent infrastructure handles code review, bug fixing, research, and planning — just by changing the prompt and tool set.

**Our divergence**: AgentM uses LangGraph's structured state machines for scenarios where behavioral guarantees matter (e.g., hypothesis-driven RCA). Both approaches have merit; the choice depends on how much behavioral enforcement the domain requires.

### 10.4 Flat Hierarchy with Centralized Coordination

**Principle**: One team lead, N workers. No sub-teams, no delegation chains.

**Why it works**: Simple mental model, simple failure modes. The team lead is the single point of coordination, which avoids split-brain scenarios and makes debugging straightforward.

**Our divergence**: AgentM's TaskManager + asyncio model allows the Orchestrator to exercise finer-grained control (inject instructions, abort tasks) than Claude Code's message-passing model.

---

## 11. Comparison: Claude Code vs AgentM

| Dimension | Claude Code | AgentM |
|-----------|-------------|--------|
| **Orchestration** | Prompt-guided ReAct (no graph) | LangGraph StateGraph (structured) |
| **Agent dispatch** | `Agent` tool → Harness spawns process/async task | `dispatch_agent` tool → TaskManager spawns asyncio.Task |
| **Communication** | File-based inbox polling | In-process shared state + messages field |
| **State sharing** | None (agents are isolated) | Shared `messages` field in parent graph state |
| **Intervention** | SendMessage (text only) | `inject_instruction` (interrupt + Command resume) |
| **Task tracking** | File-based task list with flock() | In-memory TaskManager with ManagedTask dataclass |
| **Concurrency control** | POSIX flock() on filesystem | asyncio locks (in-process) |
| **Agent hierarchy** | Flat (lead + workers, no nesting) | Flat (orchestrator + sub-agents, no nesting) |
| **Behavioral enforcement** | Prompt + tool allowlist | Graph structure + phase markers + prompt |
| **Persistence** | File system (survives restarts, partially) | LangGraph checkpointer (full state snapshots) |
| **Process model** | Multi-process possible (tmux/iTerm2) | Single-process (asyncio) |

---

## 12. References

- [Orchestrate teams of Claude Code sessions — Official Docs](https://code.claude.com/docs/en/agent-teams)
- [Create custom subagents — Official Docs](https://code.claude.com/docs/en/sub-agents)
- [Reverse-Engineering Claude Code Agent Teams: Architecture and Protocol — DEV Community](https://dev.to/nwyin/reverse-engineering-claude-code-agent-teams-architecture-and-protocol-o49)
- [Claude Code's Hidden Multi-Agent System — paddo.dev](https://paddo.dev/blog/claude-code-hidden-swarm/)
- [Claude Code Swarms — Addy Osmani](https://addyosmani.com/blog/claude-code-agent-teams/)
- [From Tasks to Swarms: Agent Teams in Claude Code — alexop.dev](https://alexop.dev/posts/from-tasks-to-swarms-agent-teams-in-claude-code/)
- [Poking Around Claude Code (binary analysis)](https://leehanchung.github.io/blogs/2025/03/07/claude-code/)
- [Claude Code source deobfuscation — GitHub (archived)](https://github.com/ghuntley/claude-code-source-code-deobfuscation)
