# Design: Frontend Architecture

**Status**: DRAFT
**Last Updated**: 2026-03-07

---

> **Code conventions**: TypeScript `interface` definitions and API endpoint schemas are **normative**. React component code and FastAPI handler implementations are **illustrative** — implementations may use different code organization as long as the interface contracts are preserved.

## Overview

AgentM Dashboard is a **read-only observation panel** for monitoring multi-agent execution in real-time. It receives stream events from the backend via WebSocket and renders agent topology, execution progress, conversation flow, and debug state.

**Current scope**: RCA (hypothesis-driven) scenario only. Other scenarios extend via a plugin/component model — each scenario provides its own UI components.

---

## Visual Style

Reference implementation: RCAgentEval Debug Dashboard. Design tokens:

```javascript
const C = {
  bg:        '#05080a',      // Deep dark background
  panel:     'rgba(5, 8, 10, 0.85)',
  line:      '#2a3640',      // Border/divider
  text:      '#c0cdd6',      // Primary text
  muted:     '#5a6b7c',      // Secondary text
  orange:    '#ff6b35',      // LLM activity accent
  teal:      '#4db4b9',      // Primary accent (active, links)
  green:     '#22c55e',      // Success / confirmed
  red:       '#ef4444',      // Error / rejected
  yellow:    '#f59e0b',      // Tool calls / warning
  purple:    '#a855f7',      // Hypothesis / analysis
};
```

Key visual principles:
- **Dark terminal aesthetic**: monospace fonts, minimal decoration
- **Color-coded semantics**: teal=active, green=done, red=failed, orange=LLM, yellow=tool
- **Information density**: compact layout, expandable sections, JSON syntax highlighting
- **Keyboard-first**: `j/k` navigate, `/` search, `Esc` close panel

---

## Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Backend API | FastAPI | Async WebSocket support, pairs with LangGraph Python |
| Real-time | WebSocket | Full-duplex, low-latency event streaming |
| Frontend | React 18 | Component model, CDN delivery (single HTML file) |
| Topology graph | d3.js | Flexible graph rendering for agent connections |
| Styling | Inline styles + CSS vars | No build step needed, self-contained |

**Deployment**: Single static HTML file served by FastAPI, loaded via CDN `<script>` tags (React, Babel, d3). No build toolchain required.

---

## Page Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  AgentM Dashboard                     [Scenario ▾] [● CONNECTED]│
├──────┬──────────────────────────────────────────────────────────┤
│      │                                                          │
│ Nav  │  Main Content Area                                       │
│      │  (switches by nav selection)                             │
│ ┌──┐ │                                                          │
│ │⬡ │ │  ┌──────────────────────────────────────────────────┐   │
│ │  │ │  │                                                  │   │
│ ├──┤ │  │            Active Page Content                   │   │
│ │▶ │ │  │                                                  │   │
│ │  │ │  │                                                  │   │
│ ├──┤ │  │                                                  │   │
│ │◈ │ │  │                                                  │   │
│ │  │ │  │                                                  │   │
│ ├──┤ │  └──────────────────────────────────────────────────┘   │
│ │⚙ │ │                                                          │
│ └──┘ │                                                          │
└──────┴──────────────────────────────────────────────────────────┘

Nav icons:
  ⬡  Topology    — Agent connection graph
  ▶  Execution   — Timeline + agent detail
  ◈  Conversation — Phase-based conversation flow (scenario-specific)
  ⚙  Debug       — Checkpoint browser + state inspector
```

---

## Page 1: Agent Topology

Real-time agent connection graph rendered with d3.js force layout.

```
┌─────────────────────────────────────────────────────────────┐
│  Agent Topology                                              │
│                                                              │
│                 ┌────────────────┐                           │
│                 │  Orchestrator  │                           │
│                 │   ◌ running    │                           │
│                 │  phase: verify │                           │
│                 └──┬──────┬──┬──┘                           │
│               ┌────┘      │  └────┐                         │
│               ▼           ▼       ▼                         │
│        ┌──────────┐ ┌─────────┐ ┌──────────┐               │
│        │  infra   │ │   db    │ │   logs   │               │
│        │ ● done   │ │ ◌ run   │ │ ✗ fail   │               │
│        │ 12.3s    │ │ 8/30    │ │ timeout  │               │
│        └──────────┘ └─────────┘ └──────────┘               │
│                                                              │
│  ● completed   ◌ running   ✗ failed   ○ pending            │
└─────────────────────────────────────────────────────────────┘
```

### Node display

| Status | Color | Content |
|--------|-------|---------|
| pending | `muted` | Agent name |
| running | `teal` + pulse animation | Agent name, step progress (N/max), current phase |
| completed | `green` | Agent name, duration |
| failed | `red` | Agent name, error type |

### Edge display

- Orchestrator → Sub-Agent: solid line, shows dispatch direction
- Active edges (agent running): `teal` color
- Completed edges: `green` color, faded

### Data source

Topology is derived from two sources:
1. **Static structure**: Parsed from scenario YAML at startup via `GET /api/topology` (agent names, connections)
2. **Dynamic status**: Updated via WebSocket events (agent state transitions)

Agents start as `pending` (known from config but no events received yet). On first WebSocket event for an agent, status transitions to `running`.

---

## Frontend Data Model

The frontend maintains its own in-memory state, accumulated from WebSocket events. This is separate from the backend's `AgentRuntime` (which serves the Orchestrator's tools).

```typescript
/** Frontend-side agent tracking, accumulated from WebSocket events */
interface FrontendAgentState {
  agentId: string;
  status: "pending" | "running" | "completed" | "failed";

  // Progress (extracted from stream events)
  currentStep: number;           // Incremented on each tool call event
  maxSteps: number | null;       // From topology config (scenario YAML max_steps)
  startedAt: string | null;      // Timestamp of first event
  completedAt: string | null;
  durationSeconds: number | null;

  // Tool call history (accumulated from WebSocket events by frontend)
  toolCalls: ToolCallRecord[];

  // Latest messages (accumulated from WebSocket events)
  messages: AgentMessage[];
}

interface ToolCallRecord {
  name: string;
  status: "running" | "success" | "error";
  startedAt: string;
  completedAt: string | null;
  durationSeconds: number | null;
  argsPreview: string;           // Truncated args for display
  resultPreview: string | null;  // Truncated result for display
}

interface AgentMessage {
  type: "human" | "ai" | "tool";
  content: string;
  toolCallId: string | null;
  timestamp: string;
}
```

**Key design**: The frontend accumulates tool call history and messages directly from WebSocket events, rather than requesting them from the backend. The backend's `AgentRuntime` serves a different consumer (the Orchestrator's tools like `check_tasks`) with a different interface (status snapshots, progress summaries).

---

## Page 2: Execution Monitor

Left-right split layout: timeline on the left, selected agent detail on the right. The split pane is resizable via drag handle.

```
┌───────────────────────────────────────────────────────────────┐
│  Execution Monitor                              [/ Search]     │
├─────────────────────────┬─┬───────────────────────────────────┤
│  Timeline               │▐│  Agent Detail: database           │
│                         │▐│                                   │
│  00:00 ▸ START          │▐│  Status: ◌ running (step 8/30)   │
│  00:01 ▸ orchestrator   │▐│  Model:  gpt-4                   │
│          phase: explore │▐│  Tools:  4 configured             │
│  00:02 ▸ dispatch ×3    │▐│                                   │
│          infra, db, logs│▐│  ┌─ CURRENT SUMMARY ───────────┐ │
│  00:05 ● infra          │▐│  │ Querying slow_query_log,    │ │
│          12.3s done     │▐│  │ found 3 queries > 500ms     │ │
│  00:08 ✗ logs           │▐│  └─────────────────────────────┘ │
│          timeout        │▐│                                   │
│  00:10 ▸ orchestrator   │▐│  Tool Calls:                     │
│          check_tasks    │▐│  1. get_db_metrics    ✓  2.1s    │
│  00:12 ▸ orchestrator   │▐│  2. check_connections ✓  1.8s    │
│          inject → db    │▐│  3. analyze_slow_q... ◌  ...     │
│  00:20 ● db             │▐│                                   │
│          18.0s done     │▐│  Latest Messages:                │
│  00:22 ▸ orchestrator   │▐│  ┌─ Orch → DB ────────────────┐ │
│          phase: confirm │▐│  │ Verify H1: pool exhaustion │ │
│                         │▐│  └────────────────────────────┘ │
│  ┌─ FINAL OUTPUT ──────┐│▐│  ┌─ DB (tool call) ───────────┐ │
│  │ Root cause: DB pool ││▐│  │ analyze_slow_queries(...)  │ │
│  │ exhaustion. Pool... ││▐│  └────────────────────────────┘ │
│  └─────────────────────┘│▐│                                   │
│                         │▐│                                   │
│          [↓ BOTTOM]     │▐│                [COPY] [ESC close] │
├─────────────────────────┴─┴───────────────────────────────────┤
│  j/k: navigate   /: search   Esc: close panel                │
└───────────────────────────────────────────────────────────────┘
```

### Timeline (left pane)

Chronological event stream. Each entry is one of:

| Event Type | Display | Color |
|-----------|---------|-------|
| Orchestrator phase change | Phase name + description | `teal` |
| Agent dispatch | Agent names + task type | `purple` |
| Agent completed | Agent name + duration + result preview | `green` |
| Agent failed | Agent name + error summary | `red` |
| Orchestrator intervention | inject/abort + target + reason | `orange` |
| check_tasks call | Summary of what Orchestrator saw | `muted` |

Features:
- **Auto-scroll**: Follows new events, disabled on manual scroll-up. "↓ BOTTOM" button with `NEW` badge when paused.
- **Search**: `/` opens search bar, filters timeline entries by keyword
- **Clickable**: Click any agent-related entry to open its detail panel

### Agent Detail (right pane)

Selected agent's full execution trace:

- **Header**: agent name, task type tag, status indicator
- **Current Summary**: Lightweight model summary of agent activity (for running agents)
- **Tool Calls**: Sequential list with name, status (✓/◌/✗), duration
- **Messages**: Collapsible message cards (Orchestrator→Agent instructions, Agent tool calls, Agent responses)
- **Summary**: Final result card with JSON syntax highlighting (for completed agents)

### Interaction

- Click timeline entry → opens agent detail panel
- Drag handle between panes → resize
- `j/k` → navigate between timeline entries
- `Esc` → close detail panel

---

## Page 3: Conversation View (RCA Scenario Component)

Conversation flow rebuilt from trajectory events. This page is **scenario-specific** — it is provided by the RCA scenario plugin.

```
┌───────────────────────────────────────────────────────────────┐
│  Conversation View                   Phase: [All ▾] [⟳ Live]  │
│                                                               │
│  ┌─ User ──────────────────────────────────────────────────┐  │
│  │ Production API response time > 5s since 14:30 UTC       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌─ Phase 1: Exploration ──────────────────── teal ────────┐  │
│  │ Dispatched agents: infra, database, logs                │  │
│  │                                                         │  │
│  │  ┌─ infra (result) ──────────────────────────────────┐  │  │
│  │  │ ▸ {cpu: 0.85, memory: 0.4, disk_io: {...}}       │  │  │
│  │  └──────────────────────────────────────── 12.3s ────┘  │  │
│  │  ┌─ database (result) ──────────────────────────────┐   │  │
│  │  │ ▸ {pool_size: 100, active: 100, waiting: 45}     │   │  │
│  │  └──────────────────────────────────────── 18.0s ────┘  │  │
│  │  ┌─ logs ✗ ─────────────────────────────────────────┐   │  │
│  │  │ Failed: timeout after 120s                       │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌─ Phase 2: Hypothesis Generation ────────── purple ──────┐  │
│  │ Generated hypotheses:                                   │  │
│  │                                                         │  │
│  │  ┌──────┬────────────────────────────────┬──────────┐   │  │
│  │  │  ID  │ Description                    │  Status  │   │  │
│  │  ├──────┼────────────────────────────────┼───────────────┤   │  │
│  │  │  H1  │ DB connection pool exhaustion  │ ◌ investigating│   │  │
│  │  │  H2  │ Disk I/O bottleneck            │ ○ formed      │   │  │
│  │  └──────┴────────────────────────────────┴───────────────┘   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌─ Phase 3: Verification ─────────────────── orange ──────┐  │
│  │ Verifying H1 → database agent                           │  │
│  │                                                         │  │
│  │  Supporting: Pool full (100/100), 45 connections waiting│  │
│  │  Rejecting:  CPU load acceptable (0.45)                 │  │
│  │  Neutral:    Network latency normal                     │  │
│  │                                                         │  │
│  │  Verdict: ✅ CONFIRMED                                   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌─ Phase 4: Confirmation ─────────────────── green ───────┐  │
│  │ Root cause confirmed: H1 — DB connection pool exhaustion│  │
│  │                                                         │  │
│  │ Recommendation:                                         │  │
│  │ ▸ Increase pool_size from 100 to 200                    │  │
│  │ ▸ Add connection timeout of 30s                         │  │
│  │ ▸ Monitor waiting_connections metric                    │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

### Phase color coding

| Phase | Color | Left border |
|-------|-------|------------|
| Phase 1: Exploration | `teal` (#4db4b9) | 3px solid |
| Phase 2: Hypothesis Generation | `purple` (#a855f7) | 3px solid |
| Phase 3: Verification | `orange` (#ff6b35) | 3px solid |
| Phase 4: Confirmation | `green` (#22c55e) | 3px solid |

### Features

- **Phase filter**: Dropdown to show all phases or a specific one
- **Live mode**: Auto-scrolls to latest phase; disables on manual scroll
- **Collapsible agent results**: Sub-Agent return data shown as folded JSON cards, click to expand
- **Hypothesis table**: Inline table within Phase 2, status column updates in real-time

### Data reconstruction

The frontend reconstructs the conversation from trajectory events streamed via WebSocket. Events include `llm_start`, `llm_end`, `tool_call`, `tool_result`, and lifecycle events (`task_dispatch`, `task_complete`).

---

## Page 4: Debug Panel

Checkpoint browser and state inspector for post-hoc analysis, replay, and trajectory export.

```
┌───────────────────────────────────────────────────────────────┐
│  Debug Panel                                   [📥 Export]     │
├────────────────────────┬─┬────────────────────────────────────┤
│  Checkpoint History    │▐│  State Inspector                   │
│                        │▐│                                    │
│  ○ Step 0  input       │▐│  Step 5 — orchestrator             │
│  ○ Step 1  orchestrator│▐│  source: loop                      │
│  ○ Step 2  infra       │▐│                                    │
│  ○ Step 3  database    │▐│  ┌─ notebook ─────────────────┐   │
│  ○ Step 4  logs        │▐│  │ current_phase: verification│   │
│  ● Step 5  orchestrator│▐│  │ ▸ hypotheses (2 items)     │   │
│  ○ Step 6  database    │▐│  │ ▸ exploration_history (5)  │   │
│  ○ Step 7  orchestrator│▐│  └────────────────────────────┘   │
│                        │▐│                                    │
│  Actions:              │▐│  ┌─ messages (12 items) ──────┐   │
│  [◀ Replay from here]  │▐│  │ ▸ [0] HumanMessage         │   │
│  [⑂ Fork here]         │▐│  │ ▸ [1] AIMessage            │   │
│                        │▐│  │ ▸ [2] ToolMessage          │   │
│                        │▐│  │   ...                       │   │
│                        │▐│  └────────────────────────────┘   │
│                        │▐│                                    │
│                        │▐│  ┌─ compression_ref ──────────┐   │
│                        │▐│  │ original_length: 45        │   │
│                        │▐│  │ compressed_at: step 4      │   │
│                        │▐│  │ from_id: chk_002           │   │
│                        │▐│  │ to_id: chk_004             │   │
│                        │▐│  └────────────────────────────┘   │
│                        │▐│                                    │
│                        │▐│           [COPY JSON] [RAW VIEW]  │
├────────────────────────┴─┴────────────────────────────────────┤
│  Shortcuts: ↑↓ select step  Enter: inspect  r: replay        │
└───────────────────────────────────────────────────────────────┘
```

### Checkpoint list (left)

- Lists all checkpoints in chronological order
- Each entry shows: step number, source node name, source type (input/loop/update)
- Selected step highlighted with `teal` indicator
- Sub-agent checkpoints indented under parent

### State inspector (right)

- Tree view of the selected checkpoint's full state (JSON)
- Collapsible nested objects
- JSON syntax highlighting (keys=teal, strings=green, numbers=yellow, booleans=purple, null=muted)
- Copy button per section

### Actions

| Action | Description | Implementation |
|--------|-------------|---------------|
| Replay from here | Resume execution from selected checkpoint | `POST /api/tasks/{thread_id}/resume` with `{checkpoint_id}` |
| Fork here | Create a new branch from this checkpoint | `POST /api/tasks/{thread_id}/fork` with `{checkpoint_id, state_updates}` |
| Export trajectory | Download full trajectory as JSONL | `GET /api/tasks/{thread_id}/trajectory` |

> **Note**: These actions call the backend REST API, which internally uses `AgentSystem.resume()` for replay and `AgentSystem.fork()` for fork. See [system-design-overview.md](system-design-overview.md#failure-recovery) for the recovery mechanism.

---

## WebSocket Event Protocol

The frontend connects to `ws://{host}/ws` and receives JSON events from the AgentRuntime.

### Event envelope

```typescript
interface WebSocketEvent {
  agent_path: string[];     // e.g. ["orchestrator"] or ["database"]
  mode: "updates" | "custom";
  data: Record<string, any>;
  timestamp: string;        // ISO 8601
}
```

### Event types (derived from stream data)

Frontend infers event type from `agent_path` and `data` content:

| Condition | Interpreted As | UI Action |
|-----------|---------------|-----------|
| `agent_path = ["orchestrator"]` + `data.state` | Orchestrator state update | Update Conversation View, Topology phase |
| `agent_path = ["orchestrator"]` + `data.messages` | Orchestrator message | Append to timeline |
| `agent_path = [agent_id]` + first event for agent | Agent started | Add node to Topology, add timeline entry |
| `agent_path = [agent_id]` + `data.messages` with ToolMessage | Tool result | Update agent detail panel |
| `agent_path = [agent_id]` + terminal event | Agent completed/failed | Update Topology node, add timeline entry |
| `mode = "custom"` | Custom annotation | Display in timeline as info event |

### Connection management

```typescript
function useWebSocket(onEvent: (ev: WebSocketEvent) => void) {
  // Auto-reconnect with exponential backoff (1s → 10s max)
  // Connection indicator in header: green dot = connected, red = disconnected
  // Buffer events during reconnection gap
}
```

---

## Scenario Plugin Architecture

The Conversation View page is scenario-specific. Different scenarios provide their own components via a plugin interface.

### Plugin interface

```typescript
interface ScenarioPlugin {
  /** Unique scenario identifier, matches scenario.yaml name */
  id: string;

  /** Display name for the nav dropdown */
  label: string;

  /**
   * Conversation View component for this scenario.
   * Receives the full state and renders the scenario-specific view.
   */
  ConversationView: React.ComponentType<{
    state: ExecutionState;
    events: WebSocketEvent[];
  }>;

  /**
   * Parse scenario-specific state from raw WebSocket data.
   * Called on each orchestrator state update.
   */
  parseState: (data: Record<string, any>) => Record<string, any>;

  /**
   * Topology node decorator — add scenario-specific info to nodes.
   * e.g., RCA adds "phase: verify" label to Orchestrator node.
   */
  decorateTopologyNode?: (agentId: string, state: any) => {
    sublabel?: string;
    badge?: { text: string; color: string };
  };
}
```

### RCA scenario plugin

```typescript
const RCAPlugin: ScenarioPlugin = {
  id: "rca_hypothesis",
  label: "RCA Hypothesis-Driven",

  ConversationView: RCAConversationView,  // The Phase 1-4 view described above

  parseState: (data) => ({
    hypotheses: data.hypotheses,             // From HypothesisStore via trajectory events
    serviceProfiles: data.serviceProfiles,   // From ServiceProfileStore via trajectory events
    agents: data.agents,                     // From AgentRuntime status
  }),

  decorateTopologyNode: (agentId, state) => {
    if (agentId === "orchestrator" && state.currentPhase) {
      return { sublabel: `phase: ${state.currentPhase}` };
    }
    return {};
  },
};
```

### Registration

```typescript
// Plugins are registered at startup
const SCENARIO_PLUGINS: Record<string, ScenarioPlugin> = {
  rca_hypothesis: RCAPlugin,
  // Future: memory_extraction: MemoryExtractionPlugin,
  // Future: sequential: SequentialPlugin,
};
```

---

## Backend API

FastAPI endpoints serving the frontend:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the static HTML dashboard |
| `/ws` | WebSocket | Real-time event stream from AgentRuntime |
| `/api/topology` | GET | Static agent topology (from scenario config) |
| `/api/tasks/{thread_id}/state` | GET | Current graph state (for page refresh) |
| `/api/tasks/{thread_id}/history` | GET | Checkpoint history for Debug Panel |
| `/api/tasks/{thread_id}/history/{checkpoint_id}` | GET | Single checkpoint full state |
| `/api/tasks/{thread_id}/trajectory` | GET | Export full trajectory as JSONL |
| `/api/tasks/{thread_id}/resume` | POST | Resume from last or specific checkpoint |
| `/api/tasks/{thread_id}/fork` | POST | Fork from checkpoint with state modifications |

> **Note**: API paths use `/api/tasks/{thread_id}/...` consistently, matching the recovery API defined in [system-design-overview.md](system-design-overview.md#failure-recovery).

### Topology response

```typescript
// GET /api/topology
interface TopologyResponse {
  scenario_id: string;
  agents: {
    agent_id: string;
    model: string;
    tools: string[];
    max_steps: number;          // From config execution.max_steps
  }[];
}
```

### Checkpoint history response

```typescript
// GET /api/tasks/{thread_id}/history
interface CheckpointEntry {
  step: number;
  source: "input" | "loop" | "update" | "fork";
  checkpoint_id: string;
  node_name: string;            // Which node produced this checkpoint
  next_nodes: string[];         // Pending nodes to execute
}
```

### FastAPI server setup

```python
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from datetime import datetime

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    runner.register_websocket(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        runner.unregister_websocket(websocket)

@app.get("/")
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)

@app.get("/api/topology")
async def get_topology():
    return {
        "scenario_id": scenario_config.name,
        "agents": [
            {
                "agent_id": name,
                "model": agent.model,
                "tools": agent.tools,
                "max_steps": agent.execution.max_steps,
            }
            for name, agent in scenario_config.agents.items()
        ],
    }

@app.get("/api/tasks/{thread_id}/history")
async def get_checkpoints(thread_id: str):
    # Returns checkpoint history from TrajectoryCollector events
    # Implementation reads from trajectory file, not LangGraph state
    history = trajectory_store.get_history(thread_id)
    return [
        {
            "step": event.get("step"),
            "source": event.get("source"),
            "checkpoint_id": event.get("checkpoint_id"),
            "node_name": event.get("agent_path", ["unknown"])[-1],
            "timestamp": event.get("timestamp"),
        }
        for event in history
    ]

@app.get("/api/tasks/{thread_id}/history/{checkpoint_id}")
async def get_checkpoint_state(thread_id: str, checkpoint_id: str):
    # Returns state snapshot at a specific checkpoint
    # Implementation reads from trajectory events, not LangGraph state
    state = trajectory_store.get_checkpoint(thread_id, checkpoint_id)
    return {
        "step": state.get("step"),
        "source": state.get("source"),
        "node_name": state.get("agent_path", ["unknown"])[-1],
        "timestamp": state.get("timestamp"),
        "values": state.get("data", {}),
    }

@app.post("/api/tasks/{thread_id}/resume")
async def resume_from_checkpoint(thread_id: str, body: ResumeRequest):
    # Resume execution from a checkpoint
    # Implementation uses AgentSystem with saved state, not LangGraph
    result = await agent_system.resume(thread_id, body.checkpoint_id)
    return {"status": "resumed", "result": result}

@app.post("/api/tasks/{thread_id}/fork")
async def fork_from_checkpoint(thread_id: str, body: ForkRequest):
    # Fork execution from a checkpoint with state modifications
    # Implementation creates new AgentSystem from saved state
    new_thread_id = await agent_system.fork(
        thread_id, body.checkpoint_id, body.state_updates
    )
    return {"status": "forked", "thread_id": new_thread_id}
```

---

## Component Architecture

```
App
├── Header                          # Connection status, scenario selector
├── NavBar                          # Page switching (Topology/Execution/Conversation/Debug)
└── MainContent
    ├── TopologyPage                # d3.js agent graph
    │   ├── TopologyGraph           # Force-directed graph renderer
    │   └── TopologyLegend          # Status color legend
    ├── ExecutionPage               # Timeline + detail split view
    │   ├── Timeline                # Event stream (left pane)
    │   │   ├── TimelineEntry       # Single event row
    │   │   └── SearchBar           # Filter entries
    │   ├── DragHandle              # Resizable split pane
    │   └── AgentDetailPanel        # Selected agent detail (right pane)
    │       ├── AgentHeader         # Name, status, model
    │       ├── ToolCallList        # Sequential tool calls
    │       ├── MessageList         # Collapsible messages
    │       └── ResultCard          # Final result with JSON highlight
    ├── ConversationPage            # Scenario plugin mount point
    │   └── [ScenarioPlugin].ConversationView
    │       └── (RCA example):
    │           ├── PhaseCard       # Phase 1-4 container
    │           ├── AgentResultCard # Collapsible agent result
    │           └── HypothesisTable # Hypothesis status table
    └── DebugPage                   # Checkpoint browser
        ├── CheckpointList          # Step list (left pane)
        ├── DragHandle
        └── StateInspector          # JSON tree view (right pane)
            ├── JsonTreeView        # Recursive tree renderer
            └── ActionBar           # Replay/Fork/Export buttons

Shared components:
├── Tag                             # Color-coded label badge
├── CollapsibleContent              # Expandable text/JSON block
├── CopyButton                      # Copy to clipboard
├── JsonHighlighter                 # JSON syntax coloring
└── useWebSocket                    # WebSocket hook with reconnect
```

---

## Related Documents

- [System Architecture](system-design-overview.md) — Overall system design, frontend overview
- [Orchestrator](orchestrator.md) — AgentRuntime, WebSocket forwarding, conversation reconstruction
- [Sub-Agent](sub-agent.md) — Agent configuration that informs topology

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single HTML file (CDN React) | No build toolchain, easy deployment, matches reference implementation |
| Read-only observation panel | Keep frontend simple; intervention via API/CLI only |
| Scenario plugin for Conversation View | Different scenarios need different visualizations; plugin isolates scenario logic |
| WebSocket for all real-time data | Single connection, full-duplex, low overhead |
| d3.js for topology | Flexible force-directed layout, handles dynamic node addition |
| Dark terminal aesthetic | Matches reference style, high contrast for data-dense displays |
| Keyboard-first interaction | Developer-oriented tool, efficiency matters |
| REST API for debug actions | Replay/Fork are infrequent, don't need WebSocket; clear request-response |
| Inline styles | No build step, self-contained, matches reference approach |
