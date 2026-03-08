"""Rich-based real-time terminal display for AgentM debug mode."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


_STATUS_STYLES: dict[str, str] = {
    "running": "yellow",
    "completed": "green",
    "failed": "red",
    "pending": "dim",
}

_HYPOTHESIS_STYLES: dict[str, str] = {
    "confirmed": "green",
    "rejected": "red",
    "investigating": "yellow",
    "formed": "dim",
    "refined": "cyan",
    "inconclusive": "magenta",
    "removed": "dim strike",
}


class DebugConsole:
    """Rich-based real-time terminal display for AgentM debug mode.

    Shows three panels:
    1. Agent Status — task_id, agent, status, step, duration
    2. Tool Timeline — last N tool calls with name, args preview, duration
    3. Hypothesis Board — hypothesis id, status (color-coded), description
    """

    def __init__(self, verbose: bool = False) -> None:
        self._console = Console()
        self._live: Live | None = None
        self._verbose = verbose

        self._agents: dict[str, dict[str, Any]] = {}
        self._tool_calls: list[dict[str, Any]] = []
        self._hypotheses: dict[str, dict[str, Any]] = {}
        self._step = 0
        self._phase = "exploration"

    def start(self) -> None:
        """Start the Live display."""
        self._live = Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=4,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the Live display."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    async def on_event(self, event: dict[str, Any]) -> None:
        """Process an event envelope and update the display.

        Called as the on_event callback from AgentSystem.stream().
        """
        self._step = event.get("step", self._step)
        data = event.get("data", {})

        # Parse the LangGraph event dict (keys = node names)
        for node_name, node_data in data.items():
            if node_name == "__interrupt__":
                continue
            self._parse_node_data(node_name, node_data)

        if self._live is not None:
            self._live.update(self._build_layout())

    def on_trajectory_event(self, event: dict[str, Any]) -> None:
        """Process a trajectory event dict for display updates.

        Called from CLI when trajectory events are available.
        """
        event_type = event.get("event_type", "")
        evt_data = event.get("data", {})

        if event_type == "task_dispatch":
            tid = evt_data.get("task_id", "")
            self._agents[tid] = {
                "agent_id": evt_data.get("agent_id", ""),
                "status": "running",
                "step": 0,
                "duration": None,
            }
        elif event_type == "task_complete":
            tid = evt_data.get("task_id", "")
            if tid in self._agents:
                self._agents[tid]["status"] = "completed"
                self._agents[tid]["duration"] = evt_data.get("duration_seconds")
        elif event_type == "task_fail":
            tid = evt_data.get("task_id", "")
            if tid in self._agents:
                self._agents[tid]["status"] = "failed"
        elif event_type == "tool_call":
            self._tool_calls.append({
                "tool": evt_data.get("tool_name", ""),
                "args": str(evt_data.get("args", ""))[:60],
                "time": event.get("timestamp", "")[-12:],
                "agent": "/".join(event.get("agent_path", [])),
            })
            self._tool_calls = self._tool_calls[-20:]
        elif event_type == "hypothesis_update":
            hid = evt_data.get("hypothesis_id", "")
            self._hypotheses[hid] = {
                "status": evt_data.get("status", ""),
                "description": evt_data.get("description", "")[:60],
            }

        if self._live is not None:
            self._live.update(self._build_layout())

    def _parse_node_data(
        self, node_name: str, node_data: dict[str, Any]
    ) -> None:
        """Extract tool calls and LLM content from a node update."""
        messages = node_data.get("messages", [])
        for msg in messages:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")

            if role == "ai":
                tool_calls = getattr(msg, "tool_calls", [])
                for tc in tool_calls:
                    self._tool_calls.append({
                        "tool": tc.get("name", ""),
                        "args": str(tc.get("args", ""))[:60],
                        "time": f"step {self._step}",
                        "agent": "orchestrator",
                    })
                    self._tool_calls = self._tool_calls[-20:]
            elif role == "tool":
                tool_name = getattr(msg, "name", "")
                if tool_name in (
                    "spawn_worker", "wait_for_workers",
                    "update_hypothesis", "remove_hypothesis",
                ):
                    self._parse_tool_result(tool_name, content)

    def _parse_tool_result(self, tool_name: str, content: str) -> None:
        """Update agent/hypothesis state from tool results."""
        import json

        if tool_name == "spawn_worker":
            try:
                result = json.loads(content)
                tid = result.get("task_id", "")
                if tid:
                    self._agents[tid] = {
                        "agent_id": f"worker-{result.get('task_type', '?')}",
                        "status": "running",
                        "step": 0,
                        "duration": None,
                    }
            except (json.JSONDecodeError, TypeError):
                pass

        elif tool_name == "wait_for_workers":
            try:
                result = json.loads(content)
                for completed in result.get("completed", []):
                    tid = completed.get("task_id", "")
                    if tid in self._agents:
                        self._agents[tid]["status"] = "completed"
                        self._agents[tid]["duration"] = completed.get("duration_seconds")
                for failed in result.get("failed", []):
                    tid = failed.get("task_id", "")
                    if tid in self._agents:
                        self._agents[tid]["status"] = "failed"
            except (json.JSONDecodeError, TypeError):
                pass

        elif tool_name == "update_hypothesis":
            # Parse from the string format "Hypothesis H1 updated: status — description"
            if "updated:" in content:
                parts = content.split("updated:", 1)
                hid = parts[0].replace("Hypothesis ", "").strip()
                rest = parts[1].strip()
                status_desc = rest.split(" — ", 1)
                status = status_desc[0].strip() if status_desc else ""
                desc = status_desc[1].strip() if len(status_desc) > 1 else ""
                self._hypotheses[hid] = {
                    "status": status,
                    "description": desc[:60],
                }

    def _build_layout(self) -> Layout:
        """Build the three-panel layout."""
        layout = Layout()
        layout.split_column(
            Layout(self._build_agent_table(), name="agents", ratio=2),
            Layout(self._build_tool_timeline(), name="tools", ratio=2),
            Layout(self._build_hypothesis_board(), name="hypotheses", ratio=1),
        )
        return layout

    def _build_agent_table(self) -> Panel:
        table = Table(expand=True)
        table.add_column("Task ID", style="dim", max_width=12)
        table.add_column("Agent", style="cyan")
        table.add_column("Status")
        table.add_column("Duration", justify="right")
        for tid, info in self._agents.items():
            style = _STATUS_STYLES.get(info["status"], "")
            duration = f"{info['duration']:.1f}s" if info.get("duration") else "-"
            table.add_row(
                tid[:12],
                info["agent_id"],
                Text(info["status"], style=style),
                duration,
            )
        return Panel(table, title="Agent Status", border_style="blue")

    def _build_tool_timeline(self) -> Panel:
        table = Table(expand=True)
        table.add_column("Time", style="dim", max_width=12)
        table.add_column("Agent", style="cyan", max_width=20)
        table.add_column("Tool", style="yellow")
        table.add_column("Args", style="dim", max_width=40)
        for tc in reversed(self._tool_calls[-10:]):
            table.add_row(tc["time"], tc["agent"], tc["tool"], tc["args"])
        return Panel(table, title="Tool Timeline", border_style="yellow")

    def _build_hypothesis_board(self) -> Panel:
        table = Table(expand=True)
        table.add_column("ID", style="magenta")
        table.add_column("Status")
        table.add_column("Description", max_width=50)
        for hid, info in self._hypotheses.items():
            style = _HYPOTHESIS_STYLES.get(info["status"], "")
            table.add_row(
                hid,
                Text(info["status"], style=style),
                info["description"],
            )
        return Panel(table, title="Hypothesis Board", border_style="magenta")
