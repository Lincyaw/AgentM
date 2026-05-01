"""Inject RCA investigation state before each LLM call."""

from __future__ import annotations

from typing import Any

from agentm.core.kernel import AgentEndEvent, ContextEvent, TextContent, UserMessage
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

from agentm_rca.stores import HypothesisStore, ServiceProfileStore

MANIFEST = ExtensionManifest(
    name="dynamic_context",
    description="Inject RCA stores as a transient <current_state> message.",
    registers=("event:context", "event:agent_end"),
)

_STATE_PREFIX = "<current_state>\n"
_STATE_SUFFIX = "\n</current_state>"


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    hypothesis_store = _expect_hypothesis_store(config)
    profile_store = _expect_profile_store(config)

    def _inject(event: ContextEvent) -> list[Any] | None:
        event.messages[:] = [
            message for message in event.messages if not _is_state_message(message)
        ]
        state_text = _format_current_state(
            hypothesis_store=hypothesis_store,
            profile_store=profile_store,
        )
        event.messages.insert(
            0,
            UserMessage(
                role="user",
                content=[TextContent(type="text", text=state_text)],
                timestamp=0.0,
            ),
        )
        return None

    def _cleanup(event: AgentEndEvent) -> None:
        event.messages[:] = [
            message for message in event.messages if not _is_state_message(message)
        ]

    api.on("context", _inject)
    api.on("agent_end", _cleanup)


def _expect_hypothesis_store(config: dict[str, Any]) -> HypothesisStore:
    store = config.get("hypothesis_store")
    if not isinstance(store, HypothesisStore):
        raise TypeError(
            "dynamic_context.install requires config['hypothesis_store']=HypothesisStore"
        )
    return store


def _expect_profile_store(config: dict[str, Any]) -> ServiceProfileStore:
    store = config.get("profile_store")
    if not isinstance(store, ServiceProfileStore):
        raise TypeError(
            "dynamic_context.install requires config['profile_store']=ServiceProfileStore"
        )
    return store


def _format_current_state(
    *,
    hypothesis_store: HypothesisStore,
    profile_store: ServiceProfileStore,
) -> str:
    sections: list[str] = []
    hypothesis_section = hypothesis_store.format_for_llm()
    if hypothesis_section:
        sections.append(hypothesis_section)
    profile_section = profile_store.format_for_llm()
    if profile_section:
        sections.append(profile_section)
    body = "\n\n".join(sections) if sections else "(Investigation starting -- no data collected yet)"
    return f"{_STATE_PREFIX}{body}{_STATE_SUFFIX}"


def _is_state_message(message: Any) -> bool:
    if not isinstance(message, UserMessage):
        return False
    if len(message.content) != 1:
        return False
    content = message.content[0]
    return isinstance(content, TextContent) and content.text.startswith(_STATE_PREFIX)
