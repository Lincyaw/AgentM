"""Fail-stop test: ``baseline_fingerprint`` validation (B-10).

Closes the §10 P6 concurrent-tuner race. Two tuners that both eval against
HEAD and both submit ``activate`` would, without this guard, race past the
deployment gate against each other's pre-image. The first commit lands;
the second tuner's eval ran against a now-stale baseline (different
content) and its activation would silently overwrite the first.

Structural fix: tuner pipes its baseline ``git rev-parse HEAD -- <path>``
into the activate call; the atom re-runs the rev-parse at activate time
and rejects with ``stale_baseline`` if the SHA moved.

Without this: two tuners both win against pre-images that no longer exist.

Also asserted: a stale-baseline rejection does NOT increment the
``stop_after_no_improvement`` counter — operator error must not consume
the anti-thrash budget.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
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
from agentm.core.abi.extension import ProviderConfig
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session import AgentSession

_PROVIDER_MODULE = "agentm._tests.fp_provider"


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
            "fp-test",
            ProviderConfig(
                stream_fn=_stream,
                model=Model(
                    id="fp-test",
                    provider="fake",
                    context_window=16_000,
                    max_output_tokens=2_000,
                ),
                name="fp-test",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[_PROVIDER_MODULE] = module
    return _PROVIDER_MODULE


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


_ATOM_V1 = (
    "from agentm.core.abi import FunctionTool, TextContent, ToolResult\n"
    "from agentm.extensions import ExtensionManifest\n"
    "MANIFEST = ExtensionManifest(name='tool_x', description='x',"
    " registers=('tool:x',))\n"
    "async def _exec(args):\n"
    "    return ToolResult(content=[TextContent(type='text', text='v1')])\n"
    "def install(api, config):\n"
    "    api.register_tool(FunctionTool(name='x', description='x',"
    " parameters={'type':'object','properties':{}}, fn=_exec))\n"
)


_ATOM_V2 = _ATOM_V1.replace("'v1'", "'v2'")
_ATOM_V3 = _ATOM_V1.replace("'v1'", "'v3'")


def _seed_eval_runs(cwd: Path) -> None:
    eval_runs = cwd / ".agentm" / "eval_runs"
    eval_runs.mkdir(parents=True, exist_ok=True)
    (eval_runs / "er_b.jsonl").write_text(
        json.dumps(
            {
                "kind": "eval_run.summary",
                "eval_run_id": "er_b",
                "primary_score": 0.10,
                "primary_score_stderr": 0.05,
                "guard_metrics": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (eval_runs / "er_p.jsonl").write_text(
        json.dumps(
            {
                "kind": "eval_run.summary",
                "eval_run_id": "er_p",
                "primary_score": 0.95,
                "primary_score_stderr": 0.05,
                "guard_metrics": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _setup_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Init a real git repo with v1 of the atom committed. Returns
    (atom_path, scenario_dir)."""
    _git(tmp_path, "init", "-q", "-b", "agent-tests")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    scenario_dir = tmp_path / "contrib" / "scenarios" / "format_fix"
    scenario_dir.mkdir(parents=True)
    atom_path = scenario_dir / "tool_x.py"
    atom_path.write_text(_ATOM_V1, encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return atom_path, scenario_dir


@pytest.mark.asyncio
async def test_stale_baseline_fingerprint_rejected(tmp_path: Path) -> None:
    """Tuner A computes fingerprint at HEAD. Tuner B lands a v2 commit on
    disk. Tuner A then submits ``activate`` carrying the now-stale
    fingerprint — must reject with ``stale_baseline``.

    Pass means the rev-parse at activate time correctly detected the file
    moved out from under the tuner. Without this, A's gate would deploy
    against a pre-image that no longer exists in tree, silently
    overwriting B's commit.
    """
    atom_path, _ = _setup_repo(tmp_path)
    _seed_eval_runs(tmp_path)

    # Tuner A's recorded fingerprint (HEAD before B's intervening commit).
    fp_stale = _git(
        tmp_path, "rev-parse", "HEAD"
    ).stdout.strip()

    # Tuner B lands a competing change.
    atom_path.write_text(_ATOM_V2, encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "tuner-B v2")

    provider_module = _install_static_provider()
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(provider_module, {}),
            extensions=[

                ("agentm.extensions.builtin.operations_local", {}),
                ("contrib.extensions.changespec_validators", {}),
                (
                    "agentm.extensions.builtin.tool_propose_change",
                    {"target_scenario": "format_fix"},
                ),
            ],
        )
    )
    try:
        propose = next(t for t in session.tools if t.name == "propose_change")
        # Tuner A submits its activate carrying the now-stale fingerprint.
        result = await propose.execute(
            {
                "target": {
                    "kind": "atom_source",
                    "path": "tool_x.py",
                    "new_content": _ATOM_V3,
                    "target_atom": "tool_x",
                },
                "rationale": "tuner-A activate",
                "eval_run_baseline": "er_b",
                "eval_run_proposed": "er_p",
                "decision": "activate",
                "baseline_fingerprint": fp_stale,
            }
        )
        assert result.is_error is True
        text = result.content[0].text
        assert "stale_baseline" in text, text
        assert fp_stale[:8] in text  # contains "was <X>"

        # The on-disk atom is still v2 (tuner B's commit), NOT v3.
        assert atom_path.read_text(encoding="utf-8") == _ATOM_V2

        # The rejection is recorded as ``kind=stale_baseline`` (an
        # operator-error class), distinct from ``rejected``. This MUST
        # NOT be counted by the B-9 stop_after_no_improvement counter,
        # else two concurrent tuners could lock each other out via stale
        # races.
        log = (
            tmp_path
            / ".agentm"
            / "decisions"
            / "format_fix"
            / "activations.jsonl"
        )
        records = [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        kinds = [r["kind"] for r in records]
        assert "stale_baseline" in kinds
        assert "rejected" not in kinds  # operator-error, not a gate rejection
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_omitted_baseline_fingerprint_skips_check(tmp_path: Path) -> None:
    """Backward-compat: a call without ``baseline_fingerprint`` skips the
    rev-parse check entirely. Single-tuner ergonomics are preserved —
    nothing existing has to change to keep working.
    """
    atom_path, _ = _setup_repo(tmp_path)
    _seed_eval_runs(tmp_path)

    # Move HEAD on disk so a stale fingerprint *would* fail. Then call
    # without the fingerprint — must succeed (no check performed).
    atom_path.write_text(_ATOM_V2, encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "intervening")

    provider_module = _install_static_provider()
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(provider_module, {}),
            extensions=[

                ("agentm.extensions.builtin.operations_local", {}),
                ("contrib.extensions.changespec_validators", {}),
                (
                    "agentm.extensions.builtin.tool_propose_change",
                    {"target_scenario": "format_fix"},
                ),
            ],
        )
    )
    try:
        propose = next(t for t in session.tools if t.name == "propose_change")
        result = await propose.execute(
            {
                "target": {
                    "kind": "atom_source",
                    "path": "tool_x.py",
                    "new_content": _ATOM_V3,
                    "target_atom": "tool_x",
                },
                "rationale": "no-fingerprint activate",
                "eval_run_baseline": "er_b",
                "eval_run_proposed": "er_p",
                "decision": "activate",
                # baseline_fingerprint omitted: legacy single-tuner flow.
            }
        )
        # Gate passes (huge delta, low stderr). The fingerprint is opt-in
        # — its absence must not be a rejection reason.
        assert result.is_error is False, result.content[0].text
    finally:
        await session.shutdown()
