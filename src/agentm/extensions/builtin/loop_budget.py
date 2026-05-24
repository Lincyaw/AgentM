"""Builtin ``loop_budget`` atom — sets the agent-loop turn / tool budget.

The loop budget is a *policy*, so it lives as an atom rather than a privileged
manifest field: a scenario that wants a hard ceiling lists this atom with
``config``, exactly like any other capability. The atom registers a
:class:`LoopConfig` under :data:`LOOP_BUDGET_SERVICE`; the session factory
reads it just before constructing the loop.

```yaml
extensions:
  - module: agentm.extensions.builtin.loop_budget
    config:
      max_turns: 128        # omit / null ⇒ no turn cap
      max_tool_calls: 400   # omit / null ⇒ no tool-call cap
```

Precedence: an explicit caller override (CLI ``--max-turns`` / SDK
``loop_config=``) wins over whatever this atom registers; with neither, the
substrate default (``LoopConfig()`` — no cap) applies.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import LoopConfig
from agentm.core.abi.extension import ExtensionAPI
from agentm.core.abi.roles import LOOP_BUDGET_SERVICE
from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(
    name="loop_budget",
    description="Sets the agent-loop turn / tool-call budget for the session.",
    registers=(),  # Registers a service, not a tool/event/role — empty by design.
    config_schema={
        "type": "object",
        "properties": {
            "max_turns": {"type": ["integer", "null"], "minimum": 1},
            "max_tool_calls": {"type": ["integer", "null"], "minimum": 1},
        },
        "additionalProperties": False,
    },
    requires=(),
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    max_turns = _positive_int_or_none(config, "max_turns")
    max_tool_calls = _positive_int_or_none(config, "max_tool_calls")
    api.set_service(
        LOOP_BUDGET_SERVICE,
        LoopConfig(max_turns=max_turns, max_tool_calls=max_tool_calls),
    )


def _positive_int_or_none(config: dict[str, Any], key: str) -> int | None:
    """Validate ``config[key]`` as a positive int (``None``/absent ⇒ no cap).

    Fail fast on a typo'd or non-positive value rather than silently dropping
    it — a silently-ignored budget would let two "identical" runs diverge.
    ``bool`` is rejected explicitly: ``isinstance(True, int)`` is True, so a
    stray ``max_turns: true`` would otherwise slip through as 1.
    """

    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"loop_budget: '{key}' must be a positive integer or null; got {value!r}"
        )
    return value
