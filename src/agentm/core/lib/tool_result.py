"""Surface internally-handled tool conditions back to the model.

When an atom tool *raises*, the kernel tags the result ``is_error=True`` and
the ``tool_error_messages`` atom feeds the model ``"Tool execution error:
{exc}"`` — so uncaught failures already reach the model. But when a tool
*catches* an exception and recovers (returning a normal-looking result), the
model never learns the recovery happened: the logic flow is hidden and the
model may build on a silently-degraded result.

:func:`with_model_note` appends a model-visible ``<system-reminder>`` block
to a :class:`ToolResult` so a caught-but-non-fatal condition still reaches
the model. It matches the wire format used by the session inbox
(``<system-reminder source="...">``) so the model sees one consistent
reminder shape regardless of origin.

Pair it with a ``logger`` call — the two channels are complementary: the log
locates the failure for a human operator, the note keeps the model honest
about what the tool actually did. Typical use inside an ``except`` that
recovers rather than re-raising::

    try:
        rows = await query(primary)
    except QueryError as exc:
        logger.warning("search: secondary index unavailable: {}", exc)
        return with_model_note(
            _ok(primary_rows),
            f"The secondary index was unavailable ({exc}); "
            "results may be incomplete.",
        )
"""

from __future__ import annotations

from agentm.core.abi.messages import TextContent
from agentm.core.abi.tool import ToolResult

# Default ``source`` attribute for tool-emitted reminders, matching the
# inbox's ``<system-reminder source="...">`` convention.
_TOOL_NOTE_SOURCE = "tool"


def with_model_note(
    result: ToolResult, note: str, *, source: str = _TOOL_NOTE_SOURCE
) -> ToolResult:
    """Append a model-visible ``<system-reminder>`` note to *result* in place.

    Use inside an ``except`` block that recovers from an error the model
    should still be aware of. The note is wrapped as
    ``<system-reminder source="{source}">...</system-reminder>`` and appended
    to ``result.content`` so it rides back to the model as part of the tool
    result. Returns the same ``result`` for call-site chaining.
    """
    result.content.append(
        TextContent(
            type="text",
            text=f'<system-reminder source="{source}">\n{note}\n</system-reminder>',
        )
    )
    return result


__all__ = ["with_model_note"]
