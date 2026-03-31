"""Tool call concurrency partitioning.

Partitions a list of tool calls into sequential batches where each batch
is either a group of concurrency-safe tools (run in parallel) or a single
non-safe tool (run alone).
"""
from __future__ import annotations

import os

from agentm.core.tool import Tool

DEFAULT_MAX_TOOL_CONCURRENCY = 8


def partition_tool_calls(
    tool_calls: list[dict[str, object]],
    tools_dict: dict[str, Tool],
) -> list[tuple[bool, list[dict[str, object]]]]:
    """Partition tool calls into (is_concurrent, calls) groups.

    Consecutive concurrency_safe tools are batched together for parallel execution.
    Non-safe tools are isolated into single-item batches for serial execution.
    Unknown tools (not in tools_dict) are treated as non-safe.

    Args:
        tool_calls: List of tool call dicts, each with "name" and "args" keys.
        tools_dict: Mapping of tool name to Tool instance.

    Returns:
        List of (is_concurrent, batch) tuples.
    """
    batches: list[tuple[bool, list[dict[str, object]]]] = []
    for tc in tool_calls:
        name = tc.get("name", "")
        tool = tools_dict.get(str(name))
        safe = bool(getattr(tool, "concurrency_safe", False)) if tool else False
        if safe and batches and batches[-1][0]:
            # Extend the current concurrent batch
            batches[-1][1].append(tc)
        else:
            batches.append((safe, [tc]))
    return batches


def get_max_tool_concurrency() -> int:
    """Return the max concurrent tool calls (from env or default)."""
    env = os.environ.get("AGENTM_MAX_TOOL_CONCURRENCY", "")
    try:
        return int(env) if env else DEFAULT_MAX_TOOL_CONCURRENCY
    except ValueError:
        return DEFAULT_MAX_TOOL_CONCURRENCY
