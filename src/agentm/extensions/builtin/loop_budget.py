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

from pydantic import BaseModel

from agentm.core.abi import CommandSpec, LOOP_BUDGET_SERVICE, LoopConfig
from agentm.extensions import ExtensionManifest


class LoopBudgetConfig(BaseModel):
    max_turns: int | None = None
    max_tool_calls: int | None = None


MANIFEST = ExtensionManifest(
    name="loop_budget",
    description="Sets the agent-loop turn / tool-call budget for the session.",
    registers=("command:loop",),
    config_schema=LoopBudgetConfig,
    requires=(),
    api_version=1,
    tier=1,
)


class _LoopBudgetRuntime:
    def __init__(self, session: Any, config: LoopBudgetConfig) -> None:
        self._session = session
        self._loop_config = LoopConfig(
            max_turns=_positive_int_or_none_from_model(config.max_turns, "max_turns"),
            max_tool_calls=_positive_int_or_none_from_model(
                config.max_tool_calls, "max_tool_calls"
            ),
        )

    def install(self) -> None:
        self._session.services.register(LOOP_BUDGET_SERVICE, self._loop_config)
        self._session.services.register(
            "command:loop",
            CommandSpec(
                description="Show this session's agent-loop turn/tool budget.",
                handler=self.loop_command,
            ),
        )

    async def loop_command(self, _args: str, cmd_api: Any) -> None:
        cfg = self._loop_config
        from agentm.core.abi import DiagnosticEvent
        await self._session.bus.emit(
            DiagnosticEvent.CHANNEL,
            DiagnosticEvent(
                level="info", source="loop_budget",
                message=(
                    f"Loop budget: max_turns={_render_limit(cfg.max_turns)}, "
                    f"max_tool_calls={_render_limit(cfg.max_tool_calls)}."
                ),
            ),
        )


def install(session: Any, config: LoopBudgetConfig) -> None:
    _LoopBudgetRuntime(session, config).install()


def _render_limit(value: int | None) -> str:
    return str(value) if value is not None else "unlimited"


def _positive_int_or_none_from_model(value: int | None, key: str) -> int | None:
    """Validate a value as a positive int (``None`` ⇒ no cap).

    Fail fast on a non-positive value rather than silently dropping
    it — a silently-ignored budget would let two "identical" runs diverge.
    ``bool`` is rejected explicitly: ``isinstance(True, int)`` is True, so a
    stray ``max_turns: true`` would otherwise slip through as 1.
    """

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"loop_budget: '{key}' must be a positive integer or null; got {value!r}"
        )
    return value


__all__ = (
    "LoopBudgetConfig",
    "MANIFEST",
    "install",
)
