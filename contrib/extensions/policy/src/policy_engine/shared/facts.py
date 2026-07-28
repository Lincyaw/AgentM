"""Facts about the running session, as SQL a checklist item can be gated on.

A tag says a concern *resembles* this session. A fact says whether the situation
the concern presupposes has actually arrived. Both are needed: eleven firings of
one item were measured, and six of them landed in sessions that had run no test
at all, so the item's stated premise -- that green narrowed runs existed -- was
false every time.

Why placeholders instead of the ``policy.v_*`` views: those views are defined
against one hardcoded schema (``harbor_live``), so a run in any other schema is
invisible to them. That is the literal reason the data plane has had no runtime
consumer. Expanding a placeholder against the session's own schema needs no DDL,
works in every schema, and keeps the SQL an item carries readable.

An item's ``precondition`` is any SQL that returns rows when the precondition
holds and none when it does not. It may use these names:

``{calls}``
    one row per tool call: ``turn_index``, ``tool_name``, ``args`` (jsonb),
    ``cmd``, ``path``, ``exit_code``, ``stdout``, ``is_error``
``{bash}``
    the ``{calls}`` rows whose tool is bash
``{edits}``
    the ``{calls}`` rows that wrote a file
``:session_id``
    bound by the caller; an item never has to name the session
"""

from __future__ import annotations

import re

#: One row per tool call in this session, with its result flattened alongside.
#: ``tool_results`` carries the call and the result together, which is what
#: makes exit codes and stdout reachable from the same row as the command.
_CALLS = """(
    SELECT t.turn_index                                        AS turn_index,
           r->'call'->>'name'                                  AS tool_name,
           r->'call'->'arguments'                              AS args,
           r->'call'->'arguments'->>'cmd'                      AS cmd,
           r->'call'->'arguments'->>'path'                     AS path,
           (r->'result'->'extras'->>'exit_code')::int          AS exit_code,
           r->'result'->'extras'->>'stdout'                    AS stdout,
           (r->'result'->>'is_error') = 'true'                 AS is_error
      FROM {schema}.agentm_trajectory_turns t,
           LATERAL jsonb_array_elements(
               COALESCE(t.turn_json->'tool_results', '[]'::jsonb)) r
     WHERE t.session_id = %(session_id)s
)"""

_BASH = "(SELECT * FROM {calls} c WHERE c.tool_name = 'bash')"

_EDITS = (
    "(SELECT * FROM {calls} c WHERE c.tool_name IN ('edit', 'write') "
    "AND c.path IS NOT NULL)"
)

PLACEHOLDERS: tuple[str, ...] = ("{calls}", "{bash}", "{edits}")

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def expand(sql: str, schema: str) -> str:
    """A precondition with its placeholders resolved against one schema.

    Raises on a schema that is not a bare identifier: this string is
    interpolated, not bound, because a table name cannot be a parameter.
    """
    if not _IDENTIFIER.match(schema):
        raise ValueError(f"not a usable schema name: {schema!r}")
    calls = _CALLS.format(schema=f'"{schema}"')
    bash = _BASH.format(calls=calls)
    edits = _EDITS.format(calls=calls)
    return (
        sql.replace("{bash}", bash).replace("{edits}", edits).replace("{calls}", calls)
    )


def documentation() -> str:
    """What the compiler is told it may write against. Generated from the same
    fragments the runtime expands, so the prompt cannot drift from the schema."""
    return (
        "{calls}  one row per tool call: turn_index, tool_name, args (jsonb), "
        "cmd, path, exit_code, stdout, is_error\n"
        "{bash}   the {calls} rows whose tool_name is 'bash'\n"
        "{edits}  the {calls} rows that wrote a file (tool_name in "
        "('edit','write'), path not null)\n"
        "Bind nothing: the session is already scoped. Return rows when the "
        "precondition holds, none when it does not."
    )


__all__ = ["PLACEHOLDERS", "documentation", "expand"]
