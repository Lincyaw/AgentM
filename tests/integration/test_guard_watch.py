"""Fail-stop test: ``tool_guard_watch`` evidence floor (B-7).

The load-bearing property: an outlier production trace MUST NOT trigger
auto-rollback. Without the ``min_samples`` floor, transient noise
reverses every activation and the loop never converges. Spec
(per-task-evolution-loop.md §10 P5): "auto-rollback requires evidence
(>=k samples beyond regression threshold)".

We seed an ``activations.jsonl`` with a prior activate + a current
activate (rollback target = prior). We then seed N production trace
JSONL files under ``.agentm/observability/`` with high
``tool_error_count`` per turn. With ``min_samples=5``:

- 4 regressing traces → guard_watch sees only 4 samples beyond
  threshold → MUST NOT roll back.
- 5 regressing traces → meets the floor → MUST roll back.

Both cases boot a session with ``tool_guard_watch`` installed; on
``session_ready`` the watcher fires synchronously. We then read
``activations.jsonl`` and assert whether a ``kind="rollback"`` record
landed.
"""

from __future__ import annotations

import json
import sys
import time
import types
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import (
    AssistantStreamEvent,
    MessageEnd,
    Model,
    TextContent,
)
from agentm.core.abi.messages import AssistantMessage
from agentm.harness.extension import ProviderConfig
from agentm.harness.session import AgentSession, AgentSessionConfig

_PROVIDER_MODULE = "agentm._tests.gw_provider"


def _install_static_provider() -> str:
    if _PROVIDER_MODULE in sys.modules:
        return _PROVIDER_MODULE
    module = types.ModuleType(_PROVIDER_MODULE)

    async def _stream(
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
        del messages, model, tools, system, signal, thinking
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="ok")],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "gw-test",
            ProviderConfig(
                stream_fn=_stream,
                model=Model(
                    id="gw-test",
                    provider="fake",
                    context_window=16_000,
                    max_output_tokens=2_000,
                ),
                name="gw-test",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[_PROVIDER_MODULE] = module
    return _PROVIDER_MODULE


def _seed_activations(decisions_dir: Path) -> None:
    """Two activations: a prior (rollback target) and a current."""
    decisions_dir.mkdir(parents=True, exist_ok=True)
    path = decisions_dir / "activations.jsonl"
    records = [
        {
            "at": time.time() - 7200,
            "kind": "activate",
            "scenario": "format_fix",
            "atom": "tool_x",
            "candidate_id": "c_prior",
            "to_sha": "deadbeef0001",
            "by": "test_seed",
        },
        {
            "at": time.time() - 3600,
            "kind": "activate",
            "scenario": "format_fix",
            "atom": "tool_x",
            "candidate_id": "c_current",
            "to_sha": "deadbeef0002",
            "by": "test_seed",
        },
    ]
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


def _seed_regressing_traces(obs_dir: Path, count: int) -> None:
    """Each trace contains turn.summary records with tool_error_count
    high enough to push tool_error_rate above the default threshold.
    """
    obs_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        trace_id = uuid.uuid4().hex[:16]
        path = obs_dir / f"{trace_id}.jsonl"
        # 5 turns, 3 errors each -> rate = 0.6, well above default 0.20.
        lines = []
        for t in range(5):
            lines.append(
                json.dumps(
                    {
                        "schema": "otel/span/v0",
                        "kind": "turn.summary",
                        "trace_id": trace_id,
                        "name": f"turn:{t}",
                        "attributes": {
                            "turn_index": t,
                            "tool_calls": ["x"],
                            "tool_call_count": 1,
                            "tool_error_count": 3,
                        },
                        "status": {"code": "OK"},
                    },
                    sort_keys=True,
                )
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        # Stagger mtimes so the watcher's recency sort is deterministic.
        os_atime_mtime = time.time() - (count - i)
        import os as _os

        _os.utime(path, (os_atime_mtime, os_atime_mtime))


async def _open_and_close_session(tmp_path: Path) -> None:
    """Boot a session with tool_guard_watch installed. The session_ready
    handler fires the watcher; we shut down immediately after."""
    provider_module = _install_static_provider()
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(provider_module, {}),
            extensions=[
                (
                    "agentm.extensions.builtin.tool_guard_watch",
                    {
                        "target_scenario": "format_fix",
                        "recent_n": 20,
                        "min_samples": 5,
                        "regression_threshold": 0.20,
                        "auto_rollback": True,
                        # Disable cooldown for the test — we are seeding
                        # fresh logs that have no prior rollback so this
                        # is moot, but make the contract explicit.
                        "cooldown_seconds": 0.0,
                    },
                ),
            ],
        )
    )
    await session.shutdown()


def _read_activations(decisions_dir: Path) -> list[dict[str, Any]]:
    path = decisions_dir / "activations.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.asyncio
async def test_outlier_below_evidence_floor_does_not_rollback(
    tmp_path: Path,
) -> None:
    """4 regressing traces with min_samples=5 → MUST NOT roll back.
    This is the fail-stop: without the evidence floor, a single bad
    trace would reverse the activation and the loop never converges.
    """
    decisions_dir = tmp_path / ".agentm" / "decisions" / "format_fix"
    obs_dir = tmp_path / ".agentm" / "observability"
    _seed_activations(decisions_dir)
    _seed_regressing_traces(obs_dir, count=4)

    await _open_and_close_session(tmp_path)

    records = _read_activations(decisions_dir)
    kinds = [r["kind"] for r in records]
    assert "rollback" not in kinds, (
        f"4 samples below floor of 5 must NOT trigger auto-rollback; "
        f"got kinds={kinds}"
    )


@pytest.mark.asyncio
async def test_evidence_floor_met_triggers_rollback(tmp_path: Path) -> None:
    """5 regressing traces meet the min_samples=5 floor → rollback fires.
    The rollback record references the prior activation's candidate_id
    as the target.
    """
    decisions_dir = tmp_path / ".agentm" / "decisions" / "format_fix"
    obs_dir = tmp_path / ".agentm" / "observability"
    _seed_activations(decisions_dir)
    _seed_regressing_traces(obs_dir, count=5)

    await _open_and_close_session(tmp_path)

    records = _read_activations(decisions_dir)
    rollbacks = [r for r in records if r["kind"] == "rollback"]
    assert len(rollbacks) == 1, [r["kind"] for r in records]
    rb = rollbacks[0]
    assert rb["auto"] is True
    assert rb["atom"] == "tool_x"
    assert rb["rolled_back_from"] == "c_current"
    assert rb["rolled_back_to"] == "c_prior"
    assert rb["evidence"]["regressing_count"] >= 5
    assert rb["evidence"]["min_samples"] == 5
    assert rb["by"] == "tool_guard_watch"


@pytest.mark.asyncio
async def test_disabled_flag_skips_subscription(tmp_path: Path) -> None:
    """Operator override: ``disabled=true`` means the atom does not
    subscribe at all. Even with overwhelming regression evidence, no
    rollback record appears.
    """
    decisions_dir = tmp_path / ".agentm" / "decisions" / "format_fix"
    obs_dir = tmp_path / ".agentm" / "observability"
    _seed_activations(decisions_dir)
    _seed_regressing_traces(obs_dir, count=20)

    provider_module = _install_static_provider()
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(provider_module, {}),
            extensions=[
                (
                    "agentm.extensions.builtin.tool_guard_watch",
                    {
                        "target_scenario": "format_fix",
                        "disabled": True,
                    },
                ),
            ],
        )
    )
    await session.shutdown()

    records = _read_activations(decisions_dir)
    assert all(r["kind"] != "rollback" for r in records), (
        f"disabled=true must suppress rollback; got {[r['kind'] for r in records]}"
    )
