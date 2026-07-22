"""Deterministic provider atom for Harbor adapter behavior tests."""

from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic import BaseModel, ConfigDict

from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.manifest import AtomInstallPriority
from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
)
from agentm.core.abi.provider import ProviderConfig
from agentm.core.abi.session_api import AtomAPI
from agentm.core.abi.stream import AssistantStreamEvent, MessageEnd, Model
from agentm.core.abi.tool import Tool
from agentm.extensions import ExtensionManifest


class HarborProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    model: str
    api_key: str | None = None
    base_url: str | None = None


MANIFEST = ExtensionManifest(
    name="harbor_test_provider",
    description="Register a deterministic provider for Harbor host tests.",
    registers=("provider:harbor-profile",),
    config_schema=HarborProviderConfig,
    priority=AtomInstallPriority.PROVIDER,
)


async def _stream(
    *,
    messages: list[AgentMessage],
    model: Model,
    tools: list[Tool],
    system: str | None = None,
    signal: CancelSignal | None = None,
    thinking: str = "off",
) -> AsyncIterator[AssistantStreamEvent]:
    del messages, model, tools, system, signal, thinking
    content = (
        TextContent(
            type="text",
            text="harbor-provider-completed",
        ),
    )
    yield MessageEnd(
        message=AssistantMessage(
            role="assistant",
            content=content,
            timestamp=0.0,
            stop_reason="end_turn",
        )
    )


def install(api: AtomAPI, config: HarborProviderConfig) -> None:
    api.register_provider(
        config.name,
        ProviderConfig(
            stream_fn=_stream,
            model=Model(
                id=config.model,
                provider="harbor-test",
                context_window=4096,
                max_output_tokens=512,
            ),
            name=config.name,
        ),
    )


__all__ = ("HarborProviderConfig", "MANIFEST", "install")
