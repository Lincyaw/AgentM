"""Inject a ``purpose`` parameter into every registered tool's schema.

Forces the model to state *why* it is calling a tool (a few words or one
sentence). The field is stripped from ``args`` on the ``tool_call`` event
before the real tool executes, so existing tools need no changes.

Configurable via ``exclude`` to skip tools where a purpose field would be
noise (e.g. terminal tools like ``finish``).
"""

from __future__ import annotations

from typing import Any, Final

from loguru import logger
from pydantic import BaseModel

from agentm.core.abi import BeforeRunEvent, ToolCallEvent
from agentm.extensions import ChannelEffects, ExtensionManifest

_FIELD_NAME: Final = "purpose"
_FIELD_SCHEMA: Final[dict[str, Any]] = {
    "type": "string",
    "description": "Why you are calling this tool (a few words).",
}


class ToolPurposeConfig(BaseModel):
    model_config = {"extra": "allow"}

    exclude: list[str] = []


MANIFEST = ExtensionManifest(
    name="tool_purpose",
    description="Add a purpose parameter to every tool so the model states intent.",
    registers=("event:agent_start", "event:tool_call"),
    config_schema=ToolPurposeConfig,
    requires=(),
    effects={
        "agent_start": ChannelEffects(appends=("tools",)),
        "tool_call": ChannelEffects(mutates=("args",)),
    },
)


class _ToolPurposeRuntime:
    __slots__ = ("_api", "_exclude", "_injected")

    def __init__(self, session: Any, config: ToolPurposeConfig) -> None:
        self._session = session
        self._exclude = set(config.exclude)
        self._injected: set[str] = set()

    def install(self) -> None:
        self._session.bus.on(BeforeRunEvent.CHANNEL, self._on_agent_start)
        self._session.bus.on(ToolCallEvent.CHANNEL, self._on_tool_call)

    def _on_agent_start(self, _event: BeforeRunEvent) -> None:
        for tool in self._session.tools:
            if tool.name in self._exclude:
                continue
            props = tool.parameters.get("properties")
            if not isinstance(props, dict):
                continue
            if _FIELD_NAME in props:
                continue
            props[_FIELD_NAME] = _FIELD_SCHEMA
            req = tool.parameters.get("required")
            if isinstance(req, list) and _FIELD_NAME not in req:
                req.append(_FIELD_NAME)
            self._injected.add(tool.name)
        if self._injected:
            logger.debug("tool_purpose: injected into {} tools", len(self._injected))

    def _on_tool_call(self, event: ToolCallEvent) -> None:
        event.args.pop(_FIELD_NAME, None)


def install(session: Any, config: ToolPurposeConfig) -> None:
    _ToolPurposeRuntime(session, config).install()
