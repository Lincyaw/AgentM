"""Custom trigger fixture installed through the public atom API."""

from __future__ import annotations

from dataclasses import dataclass

from agentm.core.abi.messages import AgentMessage, text_message
from agentm.core.abi.session_api import AtomAPI


@dataclass(frozen=True, slots=True)
class CustomTrigger:
    value: str
    source: str = "sdk_custom"


class CustomTriggerCodec:
    def serialize(self, trigger: CustomTrigger) -> dict[str, object]:
        return {
            "__source__": trigger.source,
            "value": trigger.value,
        }

    def deserialize(self, data: dict[str, object]) -> CustomTrigger:
        value = data.get("value")
        if not isinstance(value, str):
            raise ValueError("custom trigger value must be a string")
        return CustomTrigger(value=value)


class CustomTriggerRenderer:
    def render(self, trigger: object) -> list[AgentMessage]:
        if not isinstance(trigger, CustomTrigger):
            raise TypeError("custom trigger codec did not rehydrate the trigger")
        return [text_message(f"custom:{trigger.value}")]


def install(api: AtomAPI, config: object) -> None:
    del config
    api.register_trigger_codec("sdk_custom", CustomTriggerCodec())
    api.register_trigger_renderer("sdk_custom", CustomTriggerRenderer())
