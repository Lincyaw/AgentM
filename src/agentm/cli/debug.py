"""CLI debug subcommand — post-hoc analysis of trajectory JSONL files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text


console = Console()


def analyze_trajectory(
    trajectory_file: str,
    show_summary: bool = False,
    show_timeline: bool = False,
    filter_agent: str | None = None,
    filter_type: str | None = None,
) -> None:
    """Load and analyze a trajectory JSONL file."""
    path = Path(trajectory_file)
    if not path.exists():
        console.print(f"[red]File not found: {path}[/]")
        return

    events = _load_events(path)
    console.print(f"Loaded [cyan]{len(events)}[/] events from [dim]{path.name}[/]")

    if filter_agent:
        events = [
            e for e in events
            if filter_agent in "/".join(e.get("agent_path", []))
        ]
        console.print(f"  Filtered by agent [cyan]{filter_agent}[/]: {len(events)} events")

    if filter_type:
        events = [e for e in events if e.get("event_type") == filter_type]
        console.print(f"  Filtered by type [cyan]{filter_type}[/]: {len(events)} events")

    if not events:
        console.print("[yellow]No events match the filters.[/]")
        return

    # Default: show both
    if not show_summary and not show_timeline:
        show_summary = True
        show_timeline = True

    console.print()

    if show_summary:
        _print_summary(events)
    if show_timeline:
        _print_timeline(events)


def _load_events(path: Path) -> list[dict]:
    """Load events from a JSONL file, skipping the metadata header line."""
    events: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                console.print(f"[yellow]Warning: invalid JSON at line {line_num}, skipping[/]")
                continue
            # Skip metadata header line (identified by _meta key)
            if "_meta" in data:
                continue
            events.append(data)
    return events


def _print_summary(events: list[dict]) -> None:
    """Print aggregate statistics."""
    type_counts: Counter[str] = Counter()
    agents: set[str] = set()
    hypotheses: dict[str, str] = {}
    task_durations: dict[str, float] = {}

    for e in events:
        et = e.get("event_type", "unknown")
        type_counts[et] += 1
        for a in e.get("agent_path", []):
            agents.add(a)
        if et == "hypothesis_update":
            hid = e.get("data", {}).get("hypothesis_id", "")
            status = e.get("data", {}).get("status", "")
            hypotheses[hid] = status
        if et == "task_complete":
            tid = e.get("data", {}).get("task_id", "")
            dur = e.get("data", {}).get("duration_seconds", 0)
            if tid and dur:
                task_durations[tid] = dur

    # Summary table
    table = Table(title="Trajectory Summary", expand=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Total events", str(len(events)))
    table.add_row("Unique agents", ", ".join(sorted(agents)) or "-")
    table.add_row("Event types", str(len(type_counts)))
    for et, count in type_counts.most_common():
        table.add_row(f"  {et}", str(count))
    console.print(table)
    console.print()

    # Hypothesis board
    if hypotheses:
        h_table = Table(title="Hypothesis Final States", expand=True)
        h_table.add_column("ID", style="magenta")
        h_table.add_column("Final Status")
        for hid, status in sorted(hypotheses.items()):
            style = {
                "confirmed": "green", "rejected": "red",
                "investigating": "yellow", "formed": "dim",
            }.get(status, "")
            h_table.add_row(hid, Text(status, style=style))
        console.print(h_table)
        console.print()

    # Task durations
    if task_durations:
        d_table = Table(title="Task Durations", expand=True)
        d_table.add_column("Task ID", style="dim", max_width=16)
        d_table.add_column("Duration", justify="right")
        for tid, dur in sorted(task_durations.items(), key=lambda x: x[1], reverse=True):
            d_table.add_row(tid[:16], f"{dur:.1f}s")
        console.print(d_table)
        console.print()


def _print_timeline(events: list[dict]) -> None:
    """Print chronological event timeline."""
    table = Table(title="Event Timeline", expand=True)
    table.add_column("Seq", style="dim", max_width=5)
    table.add_column("Time", style="dim", max_width=12)
    table.add_column("Agent", style="cyan", max_width=20)
    table.add_column("Type", style="yellow", max_width=20)
    table.add_column("Detail", max_width=60)

    for e in events:
        detail = _event_detail(e)
        et = e.get("event_type", "")
        type_style = {
            "tool_call": "yellow",
            "tool_result": "green",
            "task_dispatch": "cyan",
            "task_complete": "green",
            "task_fail": "red",
            "hypothesis_update": "magenta",
            "error": "red",
            "llm_end": "blue",
        }.get(et, "")

        table.add_row(
            str(e.get("seq", "")),
            e.get("timestamp", "")[-12:],
            "/".join(e.get("agent_path", [])),
            Text(et, style=type_style),
            detail,
        )

    console.print(table)


def _event_detail(event: dict) -> str:
    """Extract a human-readable detail string from an event."""
    et = event.get("event_type", "")
    data = event.get("data", {})

    if et == "tool_call":
        return f"{data.get('tool_name', '')}({str(data.get('args', ''))[:40]})"
    if et == "tool_result":
        return f"{data.get('tool_name', '')}: {(data.get('result') or data.get('result_preview', ''))[:40]}"
    if et == "hypothesis_update":
        return f"{data.get('hypothesis_id', '')} -> {data.get('status', '')}"
    if et in ("task_dispatch", "task_complete", "task_fail"):
        parts = [data.get("agent_id", "")]
        if et == "task_complete" and data.get("duration_seconds"):
            parts.append(f"{data['duration_seconds']:.1f}s")
        if et == "task_fail":
            parts.append(data.get("error_summary", "")[:30])
        return " | ".join(parts)
    if et == "llm_end":
        return (data.get("content") or data.get("content_preview", ""))[:60]
    if et == "error":
        return data.get("message", "")[:60]
    return str(data)[:60]
