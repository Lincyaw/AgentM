# Design: Debug Console

**Status**: DRAFT
**Last Updated**: 2026-03-08

---

## Overview

The Debug Console is a Rich-based real-time terminal UI that provides a live dashboard during agent system execution. It displays three panels showing agent task status, recent tool calls, and the hypothesis board. The console consumes events from the [TrajectoryCollector](trajectory.md) via its listener mechanism.

---

## Panels

### 1. Agent Status

Displays the current state of all dispatched tasks:

| Column | Source |
|--------|--------|
| Task ID | `task_dispatch` event `task_id` |
| Agent | `task_dispatch` event `agent_id` |
| Status | Updated by `task_dispatch` (running), `task_complete` (completed), `task_fail` (failed) |
| Duration | Computed from dispatch timestamp to completion |

Running tasks are highlighted. Completed tasks show duration. Failed tasks show the error message.

### 2. Tool Timeline

Shows the last 20 tool calls with agent context:

| Column | Source |
|--------|--------|
| Agent | `tool_call` event `agent_path` |
| Tool | `tool_call` event `data.tool_name` |
| Args (truncated) | `tool_call` event `data.args` |
| Time | `tool_call` event `timestamp` |

New entries push older ones off the bottom. This provides a real-time view of what tools are being invoked across all agents.

### 3. Hypothesis Board

> **Note**: The Hypothesis Board is RCA-scenario-specific (not SDK core). It only displays data when running the `hypothesis_driven` scenario.

Displays the current hypothesis state:

| Column | Source |
|--------|--------|
| ID | `hypothesis_update` event `data.hypothesis_id` |
| Status | `hypothesis_update` event `data.status` |
| Description | `hypothesis_update` event `data.description` |

Statuses are color-coded: `formed` (yellow), `investigating` (blue), `confirmed` (green), `rejected` (red), `refined` (cyan), `inconclusive` (dim).

---

## Event Consumption

The Debug Console registers a single callback with the TrajectoryCollector:

```python
def on_trajectory_event(self, event_dict: dict) -> None:
    """Sync callback registered as a TrajectoryCollector listener.
    Dispatches to panel-specific handlers based on event_type."""
```

### Handled Event Types

| Event Type | Panel Updated | Action |
|-----------|---------------|--------|
| `task_dispatch` | Agent Status | Add new row with status "running" |
| `task_complete` | Agent Status | Update row to "completed" with duration |
| `task_fail` | Agent Status | Update row to "failed" with error |
| `tool_call` | Tool Timeline | Append to ring buffer (max 20 entries) |
| `hypothesis_update` | Hypothesis Board | Upsert hypothesis row by ID |

Other event types (`tool_result`, `llm_end`, `task_abort`) are silently ignored.

---

## Display

- **Rendering**: Rich `Live` display with automatic refresh at 4 FPS (`refresh_per_second=4`).
- **Layout**: Rich `Columns` layout arranging the three panels side by side (or stacked, depending on terminal width).
- **Tables**: Each panel is a Rich `Table` updated in-place on each event.

---

## Lifecycle

```python
class DebugConsole:
    def start(self) -> None:
        """Begin the Rich Live display. Call before agent execution starts."""

    def stop(self) -> None:
        """End the Rich Live display. Call after agent execution completes."""
```

The Builder (or CLI runner) manages the console lifecycle:
1. Create `DebugConsole` instance
2. Register `on_trajectory_event` as a TrajectoryCollector listener
3. Call `start()` before `AgentSystem.execute()`
4. Call `stop()` after execution completes (or on error)

The console does not own the TrajectoryCollector -- it is a passive consumer.

---

## Related Documents

- [Trajectory Collector](trajectory.md) -- Event source
- [SDK Consistency](sdk-consistency.md) -- build_agent_system() wires the console into the agent system
- [Frontend Architecture](frontend-architecture.md) -- Web-based observation dashboard (complementary to terminal UI)

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Rich library | Terminal UI with tables, colors, live updates -- no external dependencies beyond Rich |
| Three-panel layout | Covers the three primary concerns during debugging: task progress, tool activity, reasoning state |
| Sync callback | Rich is not async-native; sync callback avoids complexity of async rendering |
| 4 FPS refresh | Balances responsiveness with CPU usage; agent events are not sub-second frequency |
| Ring buffer for tool timeline | Bounded memory; only recent activity is relevant during live debugging |
| Passive consumer | Console does not affect agent execution; safe to disable without side effects |
