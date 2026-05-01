"""Bug guard: without fast-fail, configuring ``inherit_extensions=['permission']``
without populating ``available_inherited_extensions`` silently drops the
inheritance, leaving the child without the policies the parent expected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentm.harness.extension import ExtensionLoadError
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


@pytest.mark.asyncio
async def test_install_raises_when_inherit_name_not_in_available_map(
    tmp_path: Path,
) -> None:
    config = AgentSessionConfig(
        cwd=str(tmp_path),
        extensions=[
            (
                "agentm.extensions.builtin.sub_agent",
                {"inherit_extensions": ["permission"]},
            )
        ],
        provider=("tests.unit.harness_v2._fixtures.fake_provider", {}),
        resource_loader=InMemoryResourceLoader(),
    )

    with pytest.raises(ExtensionLoadError) as exc_info:
        await AgentSession.create(config)

    assert "permission" in str(exc_info.value)
