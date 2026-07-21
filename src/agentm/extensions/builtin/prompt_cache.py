# code-health: ignore-file[AM025] -- atom tools validate untyped tool, config, and service payloads
"""Prompt-cache and content-reference context policy."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace

from pydantic import BaseModel, ConfigDict, model_validator

from agentm.core.abi.compaction import ProjectionReport
from agentm.core.abi.context import BindableContextPolicy, PolicyContext
from agentm.core.abi.manifest import AtomInstallPriority
from agentm.core.abi.messages import AgentMessage
from agentm.core.abi.roles import (
    TRAJECTORY_STORE_SERVICE,
)
from agentm.core.abi.store import TrajectoryStore
from agentm.core.abi.trajectory import PromptCacheState, Turn
from agentm.core.abi.session_api import AtomAPI
from agentm.core.lib.async_cancel import await_known_outcome
from agentm.extensions import ExtensionManifest


class PromptCacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_key: str | None = None
    content_ref: str | None = None
    content_replacement_state_key: str | None = None
    tag_last_messages: int = 1
    provider: str | None = None

    @model_validator(mode="after")
    def _require_explicit_identity(self) -> "PromptCacheConfig":
        if not (
            self.cache_key or self.content_ref or self.content_replacement_state_key
        ):
            raise ValueError(
                "prompt_cache requires cache_key, content_ref, or "
                "content_replacement_state_key"
            )
        return self


MANIFEST = ExtensionManifest(
    name="prompt_cache",
    description="Attach deterministic prompt-cache/content-ref metadata to context.",
    registers=("context_policy:prompt_cache",),
    config_schema=PromptCacheConfig,
    requires=(),
    priority=AtomInstallPriority.CONTEXT,
)


class PromptCacheContextPolicy(BindableContextPolicy):
    """Tag context messages and persist provider-neutral cache state."""

    def __init__(self, config: PromptCacheConfig) -> None:
        self._config = config
        self._session_id = ""
        self._parent_session_id: str | None = None
        self._store: TrajectoryStore | None = None
        self._last_report = ProjectionReport(metadata={"policy": "prompt_cache"})

    def bind(self, ctx: PolicyContext) -> None:
        self._session_id = ctx.session_id
        self._parent_session_id = ctx.parent_session_id
        services = ctx.services or {}
        candidate = services.get(TRAJECTORY_STORE_SERVICE)
        if isinstance(candidate, TrajectoryStore):
            self._store = candidate

    async def transform(
        self,
        messages: list[AgentMessage],
        turns: Sequence[Turn],
    ) -> list[AgentMessage]:
        if not messages:
            return messages
        cache_key = self._cache_key(turns)
        content_ref = self._config.content_ref
        if cache_key is None and content_ref is None:
            return messages
        tagged = list(messages)
        count = max(0, self._config.tag_last_messages)
        if count:
            start = max(0, len(tagged) - count)
            for index in range(start, len(tagged)):
                tagged[index] = _tag_message(
                    tagged[index],
                    cache_key=cache_key,
                    content_ref=content_ref,
                )
        if cache_key is not None and self._store is not None and self._session_id:
            existing = await asyncio.to_thread(
                self._store.load_prompt_cache_state,
                self._session_id,
                cache_key,
            )
            head = await asyncio.to_thread(
                self._store.get_head,
                self._session_id,
            )
            if (
                existing is None
                and self._parent_session_id is not None
                and head is not None
                and head.logical_parent_id is not None
            ):
                inherited = await asyncio.to_thread(
                    self._store.load_prompt_cache_state,
                    self._parent_session_id,
                    cache_key,
                )
                if (
                    inherited is not None
                    and inherited.leaf_node_id == head.logical_parent_id
                ):
                    existing = replace(
                        inherited,
                        metadata={
                            **dict(inherited.metadata),
                            "inherited_from_session_id": self._parent_session_id,
                        },
                    )
            await await_known_outcome(
                asyncio.to_thread(
                    self._store.save_prompt_cache_state,
                    self._session_id,
                    PromptCacheState(
                        cache_key=cache_key,
                        leaf_node_id=(
                            head.node_id
                            if head is not None
                            else existing.leaf_node_id
                            if existing is not None
                            else None
                        ),
                        content_replacement_state_key=(
                            self._config.content_replacement_state_key
                            or (
                                existing.content_replacement_state_key
                                if existing is not None
                                else None
                            )
                        ),
                        branch_id=(
                            head.branch_id
                            if head is not None
                            else existing.branch_id
                            if existing is not None
                            else "main"
                        ),
                        head_id=(
                            head.head_id
                            if head is not None
                            else existing.head_id
                            if existing is not None
                            else "main"
                        ),
                        provider=(
                            self._config.provider
                            or (existing.provider if existing is not None else None)
                        ),
                        metadata={
                            **(dict(existing.metadata) if existing is not None else {}),
                            "turn_count": len(turns),
                            "tag_last_messages": count,
                        },
                    ),
                )
            )
        self._last_report = ProjectionReport(
            cache_keys=(cache_key,) if cache_key is not None else (),
            content_refs=(content_ref,) if content_ref is not None else (),
            metadata={"policy": "prompt_cache", "message_count": len(tagged)},
        )
        return tagged

    def explain(self) -> ProjectionReport:
        return self._last_report

    def _cache_key(self, turns: Sequence[Turn]) -> str | None:
        if self._config.cache_key:
            return self._config.cache_key
        if self._config.content_replacement_state_key:
            return f"content-replacement:{self._config.content_replacement_state_key}"
        del turns
        return None


def install(api: AtomAPI, config: PromptCacheConfig) -> None:
    api.register_context_policy(PromptCacheContextPolicy(config))


def _tag_message(
    message: AgentMessage,
    *,
    cache_key: str | None,
    content_ref: str | None,
) -> AgentMessage:
    tags = dict(message.meta.tags)
    if cache_key is not None:
        tags["cache_key"] = cache_key
    if content_ref is not None:
        tags["content_ref"] = content_ref
    return replace(message, meta=replace(message.meta, tags=tags))


__all__ = [
    "PromptCacheContextPolicy",
    "install",
    "MANIFEST",
]
