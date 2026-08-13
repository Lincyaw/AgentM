"""Install ledger — which atom registered what, and how to rebuild it.

Every tool, context policy, trigger renderer, trigger codec, and provider
a session holds was put there either by an atom installation or by the
embedder that composed the session.  This ledger records which, and that
one distinction does two jobs.

Rolling back a failed installation needs the before picture, so the
ledger snapshots itself before each atom runs.  Rebuilding a session for
a child or a fork needs the opposite: the atoms themselves are replayed
from their specs, so only the objects nobody claimed — the embedder's —
are carried over directly.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from agentm.core.abi.codec import CodecRegistry
from agentm.core.abi.context import ContextPolicy
from agentm.core.abi.session_api import ExtensionSpec
from agentm.core.abi.tool import Tool
from agentm.core.abi.trigger import TriggerRenderer

_DEFAULT_CONTEXT_POLICY_PRIORITY = 500


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    """Ownership bookkeeping captured before one atom installation."""

    tool_owners: dict[int, str | None]
    context_policy_owners: dict[int, str | None]
    context_policy_priorities: dict[int, int]
    trigger_renderer_owners: dict[str, str | None]
    trigger_codec_owners: dict[str, str | None]
    module_paths: tuple[str, ...]
    specs: tuple[ExtensionSpec, ...]


class InstallLedger:
    """Attribution of a session's registrations to the atom that made them."""

    def __init__(
        self,
        *,
        tools: Sequence[Tool] = (),
        context_policies: Sequence[ContextPolicy] = (),
        trigger_renderers: Collection[str] = (),
    ) -> None:
        self._tool_owners: dict[int, str | None] = {id(tool): None for tool in tools}
        self._context_policy_owners: dict[int, str | None] = {
            id(policy): None for policy in context_policies
        }
        self._context_policy_priorities: dict[int, int] = {
            id(policy): _DEFAULT_CONTEXT_POLICY_PRIORITY for policy in context_policies
        }
        self._trigger_renderer_owners: dict[str, str | None] = {
            source: None for source in trigger_renderers
        }
        self._trigger_codec_owners: dict[str, str | None] = {}
        self.module_paths: list[str] = []
        self._specs: list[ExtensionSpec] = []

    # --- Attribution ---

    def note_tool(self, tool: Tool, owner: str | None) -> None:
        self._tool_owners[id(tool)] = owner

    def note_context_policy(
        self,
        policy: ContextPolicy,
        owner: str | None,
        *,
        priority: int,
    ) -> None:
        self._context_policy_priorities[id(policy)] = priority
        self._context_policy_owners[id(policy)] = owner

    def drop_context_policy(self, policy: ContextPolicy) -> None:
        """Undo ``note_context_policy`` when binding the policy failed."""

        self._context_policy_priorities.pop(id(policy), None)
        self._context_policy_owners.pop(id(policy), None)

    def priority_of(self, policy: ContextPolicy) -> int:
        return self._context_policy_priorities[id(policy)]

    def note_trigger_renderer(self, source: str, owner: str | None) -> None:
        self._trigger_renderer_owners[source] = owner

    def note_trigger_codec(self, source: str, owner: str | None) -> None:
        self._trigger_codec_owners[source] = owner

    def record_installed(self, spec: ExtensionSpec) -> None:
        self.module_paths.append(spec.module_path)
        self._specs.append(ExtensionSpec(source=spec.source, config=spec.config))

    # --- Composition rebuild ---

    def external_tools(self, tools: Sequence[Tool]) -> list[Tool]:
        return [tool for tool in tools if self._tool_owners.get(id(tool)) is None]

    def external_context_policies(
        self,
        policies: Sequence[ContextPolicy],
    ) -> list[ContextPolicy]:
        return [
            policy
            for policy in policies
            if self._context_policy_owners.get(id(policy)) is None
        ]

    def external_trigger_renderers(
        self,
        renderers: dict[str, TriggerRenderer],
    ) -> dict[str, TriggerRenderer]:
        return {
            source: renderer
            for source, renderer in renderers.items()
            if self._trigger_renderer_owners.get(source) is None
        }

    def composition_codec(self, codec: CodecRegistry) -> CodecRegistry:
        return codec.copy_without_trigger_sources(
            {
                source
                for source, owner in self._trigger_codec_owners.items()
                if owner is not None
            }
        )

    def composition_extensions(
        self,
        *,
        excluded_module_paths: Collection[str] = (),
    ) -> list[ExtensionSpec]:
        return [
            ExtensionSpec(source=spec.source, config=spec.config)
            for spec in self._specs
            if spec.module_path not in excluded_module_paths
        ]

    # --- Install rollback ---

    def capture(self) -> LedgerSnapshot:
        return LedgerSnapshot(
            tool_owners=dict(self._tool_owners),
            context_policy_owners=dict(self._context_policy_owners),
            context_policy_priorities=dict(self._context_policy_priorities),
            trigger_renderer_owners=dict(self._trigger_renderer_owners),
            trigger_codec_owners=dict(self._trigger_codec_owners),
            module_paths=tuple(self.module_paths),
            specs=tuple(self._specs),
        )

    def restore(self, snapshot: LedgerSnapshot) -> None:
        self._tool_owners = dict(snapshot.tool_owners)
        self._context_policy_owners = dict(snapshot.context_policy_owners)
        self._context_policy_priorities = dict(snapshot.context_policy_priorities)
        self._trigger_renderer_owners = dict(snapshot.trigger_renderer_owners)
        self._trigger_codec_owners = dict(snapshot.trigger_codec_owners)
        self.module_paths = list(snapshot.module_paths)
        self._specs = list(snapshot.specs)


__all__ = ["InstallLedger", "LedgerSnapshot"]
