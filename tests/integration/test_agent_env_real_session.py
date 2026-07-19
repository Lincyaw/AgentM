"""Real ARL agent-env integration coverage through ``AgentSession.create``.

Opt in with ``AGENTM_TEST_AGENT_ENV_IMAGE=<image>``. The test creates a real
ARL managed session through a local gateway (default ``http://127.0.0.1:30080``),
loads the public AgentM operations/file/bash atoms, and verifies the
sandbox-backed operations through registered tools plus the ExtensionAPI ports.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import pytest

from agentm.core.abi import (
    TextContent,
    ToolResult,
)
from agentm.core.abi.session_api import AgentSessionConfig
from agentm.core.runtime.resource_loader import InMemoryResourceLoader
from agentm.core.runtime.session import AgentSession


_DEFAULT_LOCAL_GATEWAY_URL = "http://127.0.0.1:30080"
_RUN_AGENT_ENV = bool(os.environ.get("AGENTM_TEST_AGENT_ENV_IMAGE"))

_PROVIDER_SOURCE = '''
from __future__ import annotations
from collections.abc import AsyncIterator
from typing import Any
from agentm.core.abi import AssistantMessage, MessageEnd, Model, TextContent, ProviderConfig

class _Stream:
    def __call__(
        self,
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[Any]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Any]:
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="agent-env test provider")],
                timestamp=1.0,
                stop_reason="end_turn",
            )
        )

def install(api, config):
    api.register_provider(
        "agent_env_stub",
        ProviderConfig(
            stream_fn=_Stream(),
            model=Model(
                id="agent-env-stub",
                provider="agent_env_stub",
                context_window=4096,
                max_output_tokens=512,
            ),
            name="agent_env_stub",
        ),
    )
'''


@pytest.mark.slow
@pytest.mark.skipif(
    not _RUN_AGENT_ENV,
    reason=(
        "set AGENTM_TEST_AGENT_ENV_IMAGE to run real local ARL agent-env integration tests"
    ),
)
@pytest.mark.asyncio
async def test_agent_env_backend_real_session_through_agent_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_module = _write_stub_provider(tmp_path, monkeypatch)
    session_id = f"agent-env-test-{uuid.uuid4().hex}"
    gateway_url = os.environ.get(
        "AGENTM_TEST_AGENT_ENV_GATEWAY_URL",
        _DEFAULT_LOCAL_GATEWAY_URL,
    )
    assert _is_local_gateway(gateway_url), (
        "real agent-env integration tests must use a local gateway; "
        f"got {gateway_url!r}"
    )
    # The production atom intentionally supports AGENTM_AGENT_ENV_* and ARL_* env
    # fallbacks. This integration test is local-only, so remove remote defaults
    # that may be present in a developer shell.
    for env_var in (
        "AGENTM_AGENT_ENV_GATEWAY_URL",
        "AGENTM_AGENT_ENV_API_KEY",
        "AGENTM_AGENT_ENV_PROFILE",
        "ARL_GATEWAY_URL",
        "ARL_API_KEY",
    ):
        monkeypatch.delenv(env_var, raising=False)
    profile = os.environ.get("AGENTM_TEST_AGENT_ENV_PROFILE")
    api_key = os.environ.get("AGENTM_TEST_AGENT_ENV_API_KEY")
    create_timeout = float(os.environ.get("AGENTM_TEST_AGENT_ENV_CREATE_TIMEOUT", "600"))
    image = os.environ["AGENTM_TEST_AGENT_ENV_IMAGE"]
    pool_profile = profile or "default"
    existing_pools = _agent_env_pool_names(gateway_url, api_key)

    operations_config: dict[str, Any] = {
        "backend": "agent_env",
        "image": image,
        "experiment_id": session_id,
        "gateway_url": gateway_url,
        "work_dir": "/workspace",
        "timeout": 30.0,
        "create_timeout": create_timeout,
        "delete_on_shutdown": True,
    }
    if profile:
        operations_config["profile"] = profile
    if api_key:
        operations_config["api_key"] = api_key

    session: AgentSession | None = None
    try:
        session = await AgentSession.create(
            AgentSessionConfig(
                cwd="/workspace",
                session_id=session_id,
                provider=(provider_module, {}),
                extensions=[
                    ("agentm.extensions.builtin.operations", operations_config),
                    ("agentm.extensions.builtin.file_tools", {"require_read": False}),
                    ("agentm.extensions.builtin.tool_bash", {"default_timeout": 30.0}),
                ],
                resource_loader=InMemoryResourceLoader(),
                no_skills=True,
                no_prompt_templates=True,
            )
        )
        arl_session_id = session.get_service("agent_env.session_id")
        assert isinstance(arl_session_id, str)
        assert arl_session_id

        tools = {tool.name: tool for tool in session.tools}
        assert {"bash", "read", "write"} <= set(tools)

        bash_result = cast(
            ToolResult,
            await tools["bash"].execute(
                {
                    "cmd": (
                        "printf 'pwd=%s\\n' \"$PWD\"; "
                        "mkdir -p nested; "
                        "printf 'from-bash\\n' > nested/bash.txt"
                    ),
                    "timeout": 30.0,
                }
            ),
        )
        assert bash_result.is_error is False
        assert "pwd=/workspace" in _tool_text(bash_result)

        write_result = cast(
            ToolResult,
            await tools["write"].execute(
                {
                    "path": "nested/tool.txt",
                    "content": "from-write\n",
                    "rationale": "agent-env real integration test",
                }
            ),
        )
        assert write_result.is_error is False
        assert "Created 'nested/tool.txt'" in _tool_text(write_result)

        read_result = cast(
            ToolResult,
            await tools["read"].execute({"path": "nested/tool.txt"}),
        )
        assert read_result.is_error is False
        assert "from-write" in _tool_text(read_result)

        api = session.extension_api
        ops = api.get_operations()
        writer = api.get_resource_writer()

        direct = await ops.bash.exec(
            "cat nested/bash.txt nested/tool.txt",
            cwd="/workspace",
            timeout=30.0,
        )
        assert direct.exit_code == 0
        assert direct.stdout == b"from-bash\nfrom-write\n"

        write = await writer.write(
            "/workspace/nested/writer.bin",
            b"\x00writer-bytes\n",
            rationale="agent-env writer binary test",
        )
        assert write.error is None
        assert await writer.read("nested/writer.bin") == b"\x00writer-bytes\n"

        refused = await writer.write(
            "/tmp/outside-agent-env.txt",
            b"nope",
            rationale="agent-env outside refusal test",
        )
        assert refused.path_class == "constitution"
        assert refused.error is not None
        assert "outside" in refused.error
    finally:
        if session is not None:
            await session.shutdown()
        remaining_sessions, remaining_new_pools = _delete_agent_env_environment(
            gateway_url,
            api_key,
            session_id,
            existing_pools,
            image,
            pool_profile,
        )
        assert remaining_sessions == []
        assert remaining_new_pools == []


def _write_stub_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    package = f"agent_env_provider_{uuid.uuid4().hex[:8]}"
    package_dir = tmp_path / package
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "provider.py").write_text(_PROVIDER_SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return f"{package}.provider"


def _tool_text(result: ToolResult) -> str:
    return "\n".join(
        item.text for item in result.content if isinstance(item, TextContent)
    )


def _is_local_gateway(gateway_url: str) -> bool:
    host = urlparse(gateway_url).hostname
    return host in {"localhost", "127.0.0.1", "::1"}


def _agent_env_pool_names(gateway_url: str, api_key: str | None) -> set[str]:
    from arl import GatewayClient  # type: ignore[import-not-found]

    client = GatewayClient(base_url=gateway_url, api_key=api_key or "")
    try:
        return {pool.name for pool in client.list_pools()}
    finally:
        client.close()


def _delete_agent_env_environment(
    gateway_url: str,
    api_key: str | None,
    experiment_id: str,
    existing_pools: set[str],
    image: str,
    profile: str,
) -> tuple[list[str], list[str]]:
    from arl import GatewayClient  # type: ignore[import-not-found]

    client = GatewayClient(base_url=gateway_url, api_key=api_key or "")
    try:
        client.delete_experiment(experiment_id)
        remaining_sessions = [
            session.id
            for session in client.list_experiment_sessions(experiment_id)
        ]
        for pool in client.list_pools():
            if (
                pool.name not in existing_pools
                and pool.image == image
                and pool.profile == profile
                and pool.allocated_replicas == 0
            ):
                client.destroy_pool(pool.name)
        remaining_new_pools = [
            pool.name
            for pool in client.list_pools()
            if (
                pool.name not in existing_pools
                and pool.image == image
                and pool.profile == profile
            )
        ]
        return remaining_sessions, remaining_new_pools
    finally:
        client.close()
