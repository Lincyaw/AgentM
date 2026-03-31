"""Three-layer tool filtering for agent tool resolution.

Implements the filter pipeline:
    global disallowed -> whitelist -> agent disallowed

See design: .claude/designs/tool-filter.md
"""

from __future__ import annotations

import logging

from agentm.core.tool import Tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global disallowed list (SDK-enforced recursion guard)
# ---------------------------------------------------------------------------

WORKER_DISALLOWED_TOOLS: frozenset[str] = frozenset({
    "dispatch_agent",
    "check_tasks",
    "inject_instruction",
    "abort_task",
})


# ---------------------------------------------------------------------------
# Three-layer filter
# ---------------------------------------------------------------------------

def resolve_tools(
    available_tools: list[Tool],
    *,
    whitelist: list[str],
    disallowed_tools: list[str] | None = None,
    global_disallowed: frozenset[str] | None = None,
) -> list[Tool]:
    """Apply three-layer tool filtering.

    Args:
        available_tools: All tools from registry + extra_tools + think.
        whitelist: Tool names to include. ``["*"]`` means all.
        disallowed_tools: Tool names to exclude (agent-level).
        global_disallowed: Tool names to exclude (SDK-level).

    Returns:
        Filtered list of Tool instances, preserving original order.

    Raises:
        ValueError: If ``["*"]`` is mixed with other names, or if a
            whitelisted tool name is not found in available_tools.
    """
    effective_disallowed = disallowed_tools or []
    effective_global = global_disallowed or frozenset()

    # --- Validate wildcard usage ---
    if "*" in whitelist and len(whitelist) > 1:
        raise ValueError(
            '["*"] must be the sole element in whitelist; '
            f"got {whitelist!r}"
        )

    # --- Build name -> Tool index ---
    tool_index: dict[str, Tool] = {t.name: t for t in available_tools}

    # --- Layer 1: Global disallowed ---
    for name in effective_global:
        tool_index.pop(name, None)

    # --- Layer 2: Whitelist ---
    if whitelist == ["*"]:
        # Preserve original order (minus globally disallowed)
        result = [t for t in available_tools if t.name in tool_index]
    else:
        result = []
        for name in whitelist:
            if name not in tool_index:
                raise ValueError(
                    f"Whitelisted tool {name!r} not found in available tools. "
                    f"Available: {sorted(tool_index.keys())}"
                )
            result.append(tool_index[name])

    # --- Layer 3: Agent disallowed ---
    if effective_disallowed:
        disallowed_set = set(effective_disallowed)

        # Log redundant entries already removed by global disallowed
        redundant = disallowed_set & effective_global
        if redundant:
            logger.warning(
                "disallowed_tools contains names already in global disallowed: %s",
                sorted(redundant),
            )

        result = [t for t in result if t.name not in disallowed_set]

    if not result:
        logger.warning("All tools filtered out; agent will run as pure LLM reasoning.")

    return result
