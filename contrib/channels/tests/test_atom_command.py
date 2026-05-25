"""Atom-as-command: two-gate discovery, install/uninstall/list dispatch.

Fail-stop tests for the contract documented in
``.claude/designs/command-routing.md`` §8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agentm_channels.bus import InboundMessage
from agentm_channels.commands.atom_command import (
    build_atom_commands,
    discover_mountable_atoms,
)
from agentm_channels.commands.protocol import CommandContext
from agentm_channels.commands.registry import (
    CommandRegistry,
)


# --- fake ExtensionAPI ---------------------------------------------------


@dataclass
class _InstallResult:
    ok: bool
    target_path: str | None = None
    error: str | None = None


@dataclass
class _UnloadResult:
    ok: bool
    error: str | None = None


@dataclass
class _AtomInfo:
    name: str


@dataclass
class _FakeAPI:
    """Minimal stand-in for ExtensionAPI.

    Records every ``install_atom`` / ``unload_atom`` call so tests can
    assert on the exact args the handler forwarded.
    """

    installs: list[dict[str, Any]] = field(default_factory=list)
    unloads: list[dict[str, Any]] = field(default_factory=list)
    listed: list[str] = field(default_factory=list)
    install_result: _InstallResult = field(
        default_factory=lambda: _InstallResult(
            ok=True, target_path="/tmp/.agentm/atoms/X.py"
        )
    )
    unload_result: _UnloadResult = field(
        default_factory=lambda: _UnloadResult(ok=True)
    )

    def install_atom(self, **kwargs: Any) -> _InstallResult:
        self.installs.append(kwargs)
        return self.install_result

    def unload_atom(self, **kwargs: Any) -> _UnloadResult:
        self.unloads.append(kwargs)
        return self.unload_result

    def list_atoms(self) -> list[_AtomInfo]:
        return [_AtomInfo(name=n) for n in self.listed]


def _ctx(
    *,
    api: _FakeAPI | None = None,
    registry: CommandRegistry | None = None,
) -> CommandContext:
    async def _drop() -> None: ...
    def _stats() -> dict[str, Any]:
        return {"session_id": None, "turn_count": 0, "pending_approvals": 0}
    reg = registry or CommandRegistry()

    def get_api() -> Any | None:
        return api
    return CommandContext(
        route_key="stub:c",
        channel="stub",
        chat_id="c",
        sender_id="u",
        drop_route=_drop,
        get_route_stats=_stats,
        list_commands=reg.all,
        approval_bridge=None,
        get_extension_api=get_api,
    )


def _inbound(content: str) -> InboundMessage:
    return InboundMessage(
        channel="stub", sender_id="u", chat_id="c", content=content
    )


# --- discovery ------------------------------------------------------------


def test_discovery_filters_by_manifest_opt_in() -> None:
    """No SDK atom should opt in by default — preserving backwards
    compatibility. The two-gate design says even with allow=['*'] the
    list is empty until atoms turn ``mountable_via_command`` on."""
    atoms = discover_mountable_atoms(allow=frozenset({"*"}))
    # If any SDK atom opts in later this assertion legitimately
    # fails — bump the test together with the opt-in decision.
    assert atoms == [], (
        "Expected no atoms with MANIFEST.mountable_via_command=True "
        "yet; got " + ", ".join(a.name for a in atoms)
    )




# --- registry gating ------------------------------------------------------








# --- handler behaviour ----------------------------------------------------




@pytest.mark.asyncio
async def test_install_unknown_atom_is_refused() -> None:
    handlers = {h.name: h for h in build_atom_commands(allow=frozenset({"*"}))}
    api = _FakeAPI()
    inv = _parse(_inbound("/atom:install does-not-exist"))
    result = await handlers["install"].handle(inv, _ctx(api=api))
    assert "not in the mountable-atom allow list" in result.outbound[0].content
    assert api.installs == []
















# --- end-to-end via router ------------------------------------------------




# --- helpers --------------------------------------------------------------


def _parse(msg: InboundMessage):
    from agentm_channels.commands import parse_invocation

    inv = parse_invocation(msg)
    assert inv is not None
    return inv
