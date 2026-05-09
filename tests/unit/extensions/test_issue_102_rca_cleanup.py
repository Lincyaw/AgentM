from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from agentm.core.abi import EventBus
from agentm.core.abi.events import DiagnosticEvent
from agentm.harness.events import BeforeAgentStartEvent, SessionReadyEvent

import contrib.extensions.rcabench_contract as rcabench_contract
from agentm.extensions.builtin.sub_agent import _resolve_inherited_extensions


def test_sub_agent_inherits_parent_config_by_manifest_name() -> None:
    resolved = _resolve_inherited_extensions(
        ["duckdb_sql", "tool_read"],
        {
            "duckdb_sql": {"module": "agentm_rca.tools.duckdb_sql"},
            "tool_read": {"module": "agentm.extensions.builtin.tool_read"},
        },
        {
            "duckdb_sql": {
                "exclude": ["conclusion.parquet"],
                "row_limit": 200,
                "token_limit": 5000,
            },
            "tool_read": {},
        },
    )

    assert resolved == [
        (
            "agentm_rca.tools.duckdb_sql",
            {
                "exclude": ["conclusion.parquet"],
                "row_limit": 200,
                "token_limit": 5000,
            },
        ),
        ("agentm.extensions.builtin.tool_read", {}),
    ]


@pytest.mark.asyncio
async def test_rcabench_contract_warns_and_injects_placeholder_when_vendor_missing() -> (
    None
):
    sys.modules.pop("rcabench_platform", None)
    sys.modules.pop("rcabench_platform.v3", None)
    sys.modules.pop("rcabench_platform.v3.sdk", None)
    sys.modules.pop("rcabench_platform.v3.sdk.evaluation", None)
    sys.modules.pop("rcabench_platform.v3.sdk.evaluation.v2", None)

    bus = EventBus()
    diagnostics: list[DiagnosticEvent] = []
    handlers: dict[str, Any] = {}
    bus.on(DiagnosticEvent.CHANNEL, diagnostics.append)

    class _Api:
        events = bus

        def on(self, channel: str, handler: Any) -> None:
            handlers[channel] = handler

    await rcabench_contract.install(_Api(), {})  # type: ignore[arg-type]
    await handlers[SessionReadyEvent.CHANNEL](
        SessionReadyEvent(
            cwd=".",
            session_id="s",
            tool_names=(),
            command_names=(),
            extension_module_paths=(),
            model=None,
            root_session_id="s",
        )
    )
    event = BeforeAgentStartEvent(messages=[], system="base")

    handlers[BeforeAgentStartEvent.CHANNEL](event)

    assert diagnostics
    assert diagnostics[0].level == "warning"
    assert "rcabench-platform contract unavailable" in diagnostics[0].message
    assert event.system is not None
    assert '<contract status="unavailable"' in event.system
    assert event.system.endswith("base")


@pytest.mark.asyncio
async def test_rcabench_contract_injects_vendor_prompt() -> None:
    module_name = "rcabench_platform.v3.sdk.evaluation.v2"
    for prefix in [
        "rcabench_platform",
        "rcabench_platform.v3",
        "rcabench_platform.v3.sdk",
        "rcabench_platform.v3.sdk.evaluation",
    ]:
        sys.modules[prefix] = types.ModuleType(prefix)
    vendor = types.ModuleType(module_name)
    vendor.get_agent_contract_prompt = lambda: "SERVICE CONTRACT"  # type: ignore[attr-defined]
    sys.modules[module_name] = vendor

    bus = EventBus()
    handlers: dict[str, Any] = {}

    class _Api:
        events = bus

        def on(self, channel: str, handler: Any) -> None:
            handlers[channel] = handler

    try:
        await rcabench_contract.install(_Api(), {})  # type: ignore[arg-type]
        await handlers[SessionReadyEvent.CHANNEL](
            SessionReadyEvent(
                cwd=".",
                session_id="s",
                tool_names=(),
                command_names=(),
                extension_module_paths=(),
                model=None,
                root_session_id="s",
            )
        )
        event = BeforeAgentStartEvent(messages=[], system=None)

        handlers[BeforeAgentStartEvent.CHANNEL](event)
    finally:
        for prefix in [
            module_name,
            "rcabench_platform.v3.sdk.evaluation",
            "rcabench_platform.v3.sdk",
            "rcabench_platform.v3",
            "rcabench_platform",
        ]:
            sys.modules.pop(prefix, None)

    assert event.system is not None
    assert "<agent_contract>" in event.system
    assert "SERVICE CONTRACT" in event.system
