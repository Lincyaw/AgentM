"""Tests for the ``operations_local`` builtin atom and the related
``register_operations`` / freeze-time-assertion contract added in the
Stage 1 collapse-harness-into-core plan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import EventBus
from agentm.core.abi.extension import ExtensionLoadError
from agentm.core.runtime.extension import (
    _ExtensionAPIImpl,
    build_extension_api_scope,
)


class _SessionView:
    def get_messages(self) -> list[Any]:
        return []

    def get_branch(self) -> list[Any]:
        return []

    def get_leaf_id(self) -> str | None:
        return None

    def get_entry(self, entry_id: str) -> Any | None:
        del entry_id
        return None

    def get_loop_config(self) -> Any:
        return None

    def append_entry(self, type: str, payload: Any, parent_id: str | None = None) -> str:
        del type, payload, parent_id
        return "entry"


def _api(tmp_path: Path) -> _ExtensionAPIImpl:
    scope = build_extension_api_scope(
        bus=EventBus(),
        cwd=str(tmp_path),
        session_id="session",
        session=_SessionView(),
        tools=[],
        commands={},
        providers={},
        renderers={},
        pending_user_messages=[],
        model_getter=lambda: None,
        provider_getter=lambda: None,
    )
    return _ExtensionAPIImpl(scope)








@pytest.mark.asyncio
async def test_session_factory_freezes_when_no_ops_atom(tmp_path: Path) -> None:
    """If a scenario manifest does not list any atom that registers
    Operations, session creation must fail loudly at freeze time."""

    from agentm.core.runtime.session import AgentSession
    from agentm.core.abi.session_config import AgentSessionConfig

    # A minimal stub provider atom: registers a no-op provider so the
    # provider gate passes, but never touches Operations. The freeze-time
    # check should still reject the session because no Operations atom
    # ran.
    stub_provider_module = """
from agentm.core.abi import ProviderConfig
from agentm.extensions import ExtensionManifest


MANIFEST = ExtensionManifest(
    name="_stub_provider_no_ops",
    description="test stub",
    registers=("provider:stub",),
    config_schema={"type": "object", "properties": {}, "additionalProperties": False},
)


async def _stream(*args, **kwargs):  # pragma: no cover - never called
    if False:
        yield None


def install(api, config):
    del config
    api.register_provider(
        "stub",
        ProviderConfig(stream_fn=_stream, model="stub", name="stub"),
    )
"""

    stub_path = tmp_path / "_stub_provider_no_ops.py"
    stub_path.write_text(stub_provider_module)

    # Make the stub importable
    import sys

    sys.path.insert(0, str(tmp_path))
    try:
        config = AgentSessionConfig(
            cwd=str(tmp_path),
            # explicit empty extensions list — auto-discover would otherwise
            # pull in the entire builtin catalog (including operations_local)
            # and mask the freeze-time check we want to assert.
            no_extensions=True,
            provider=("_stub_provider_no_ops", {}),
        )
        with pytest.raises(ExtensionLoadError, match="no atom registered Operations"):
            await AgentSession.create(config)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("_stub_provider_no_ops", None)
