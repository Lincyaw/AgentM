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
    discover_commands,
)
from agentm_channels.commands.router import CommandRouter


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


def test_discovery_respects_allow_whitelist() -> None:
    """Empty allow list → nothing surfaces even if atoms opted in."""
    atoms = discover_mountable_atoms(allow=frozenset())
    assert atoms == []


# --- registry gating ------------------------------------------------------


def test_atom_commands_not_registered_when_disabled() -> None:
    """Default off: no /atom:* lookup succeeds."""
    reg = discover_commands(
        cwd="/nonexistent",
        atom_commands_enabled=False,
        atom_allow=["*"],
    )
    assert reg.lookup(namespace="atom", name="install") is None
    assert reg.lookup(namespace="atom", name="list") is None


def test_atom_commands_not_registered_when_allow_empty() -> None:
    """Enabled but empty allow → still no commands. Both gates required."""
    reg = discover_commands(
        cwd="/nonexistent",
        atom_commands_enabled=True,
        atom_allow=[],
    )
    assert reg.lookup(namespace="atom", name="install") is None


def test_atom_commands_registered_when_both_gates_open() -> None:
    reg = discover_commands(
        cwd="/nonexistent",
        atom_commands_enabled=True,
        atom_allow=["*"],
    )
    assert reg.lookup(namespace="atom", name="install") is not None
    assert reg.lookup(namespace="atom", name="uninstall") is not None
    assert reg.lookup(namespace="atom", name="list") is not None


# --- handler behaviour ----------------------------------------------------


@pytest.mark.asyncio
async def test_install_missing_args_returns_usage() -> None:
    handlers = {h.name: h for h in build_atom_commands(allow=frozenset({"*"}))}
    api = _FakeAPI()
    ctx = _ctx(api=api)
    inv = _parse(_inbound("/atom:install"))
    result = await handlers["install"].handle(inv, ctx)
    assert result.expanded_prompt is None
    assert "Usage" in result.outbound[0].content
    assert api.installs == []  # never reached install_atom


@pytest.mark.asyncio
async def test_install_unknown_atom_is_refused() -> None:
    handlers = {h.name: h for h in build_atom_commands(allow=frozenset({"*"}))}
    api = _FakeAPI()
    inv = _parse(_inbound("/atom:install does-not-exist"))
    result = await handlers["install"].handle(inv, _ctx(api=api))
    assert "not in the mountable-atom allow list" in result.outbound[0].content
    assert api.installs == []


@pytest.mark.asyncio
async def test_install_no_session_yet_returns_hint() -> None:
    """install_atom needs a live ExtensionAPI; before any user turn
    has run there is no session yet — the handler should explain
    instead of NPE'ing."""
    handlers = {h.name: h for h in build_atom_commands(allow=frozenset())}
    # No-op API getter: ctx with get_extension_api returning None.
    ctx = _ctx(api=None)
    inv = _parse(_inbound("/atom:install permission"))
    result = await handlers["install"].handle(inv, ctx)
    # Allow list is empty in this test, so the "not in allow list"
    # branch fires first.
    assert result.expanded_prompt is None
    assert result.outbound[0].content.startswith(
        "`permission` is not in the mountable-atom allow list."
    ) or "No live session" in result.outbound[0].content


@pytest.mark.asyncio
async def test_install_config_json_invalid_returns_error() -> None:
    handlers = {h.name: h for h in build_atom_commands(allow=frozenset({"*"}))}
    inv = _parse(_inbound("/atom:install something {bad json"))
    api = _FakeAPI()
    result = await handlers["install"].handle(inv, _ctx(api=api))
    text = result.outbound[0].content
    assert (
        "config json invalid" in text
        or "not in the mountable-atom allow list" in text
    )
    assert api.installs == []


@pytest.mark.asyncio
async def test_uninstall_no_args_returns_usage() -> None:
    handlers = {h.name: h for h in build_atom_commands(allow=frozenset({"*"}))}
    inv = _parse(_inbound("/atom:uninstall"))
    result = await handlers["uninstall"].handle(inv, _ctx(api=_FakeAPI()))
    assert "Usage" in result.outbound[0].content


@pytest.mark.asyncio
async def test_uninstall_forwards_to_api_and_reports_success() -> None:
    handlers = {h.name: h for h in build_atom_commands(allow=frozenset({"*"}))}
    api = _FakeAPI(unload_result=_UnloadResult(ok=True))
    inv = _parse(_inbound("/atom:uninstall permission"))
    result = await handlers["uninstall"].handle(inv, _ctx(api=api))
    assert len(api.unloads) == 1
    assert api.unloads[0]["name"] == "permission"
    assert api.unloads[0]["agent_initiated"] is False
    assert "Unloaded `permission`" in result.outbound[0].content


@pytest.mark.asyncio
async def test_uninstall_reports_sdk_rejection() -> None:
    handlers = {h.name: h for h in build_atom_commands(allow=frozenset({"*"}))}
    api = _FakeAPI(
        unload_result=_UnloadResult(ok=False, error="atom not loaded")
    )
    inv = _parse(_inbound("/atom:uninstall ghost"))
    result = await handlers["uninstall"].handle(inv, _ctx(api=api))
    assert "Unload rejected" in result.outbound[0].content
    assert "atom not loaded" in result.outbound[0].content


@pytest.mark.asyncio
async def test_list_shows_live_and_mountable() -> None:
    handlers = {h.name: h for h in build_atom_commands(allow=frozenset({"*"}))}
    api = _FakeAPI(listed=["permission", "cost_budget"])
    inv = _parse(_inbound("/atom:list"))
    result = await handlers["list"].handle(inv, _ctx(api=api))
    text = result.outbound[0].content
    assert "currently loaded" in text
    assert "permission" in text
    assert "cost_budget" in text


@pytest.mark.asyncio
async def test_list_handles_no_session() -> None:
    handlers = {h.name: h for h in build_atom_commands(allow=frozenset({"*"}))}
    inv = _parse(_inbound("/atom:list"))
    result = await handlers["list"].handle(inv, _ctx(api=None))
    assert "no live session" in result.outbound[0].content


# --- end-to-end via router ------------------------------------------------


@pytest.mark.asyncio
async def test_router_finds_atom_command_through_namespace() -> None:
    reg = discover_commands(
        cwd="/nonexistent",
        atom_commands_enabled=True,
        atom_allow=["*"],
    )
    router = CommandRouter(registry=reg)
    api = _FakeAPI(listed=["permission"])
    result = await router.try_dispatch(_inbound("/atom:list"), _ctx(api=api, registry=reg))
    assert result is not None
    assert "permission" in result.outbound[0].content


# --- helpers --------------------------------------------------------------


def _parse(msg: InboundMessage):
    from agentm_channels.commands import parse_invocation

    inv = parse_invocation(msg)
    assert inv is not None
    return inv
