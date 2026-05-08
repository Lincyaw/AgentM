"""Subprocess bridge to AgentM (DORMANT in V0).

Shells out to the ``agentm`` CLI with a single JSON payload as the user
prompt and parses the trailing JSON from stdout. Used by :mod:`worker`
when ``LLMHARNESS_PROVIDER=agentm`` (the file-protocol path, originally
designed for the Claude Code adapter).

V0 STATUS — dormant. The 2026-05-08 cognitive-audit refactor moved the
diagnostic agent's prompt and extension list from
``scenarios/harness_monitor/manifest.yaml`` into the Python package
(:mod:`llmharness.audit`), and AgentM-on-AgentM now uses the in-process
:mod:`llmharness.adapters.agentm` extension which calls
``api.spawn_child_session`` directly — no subprocess, no scenario YAML
lookup. The legacy ``harness_monitor`` scenario YAML no longer exists.

Two ways to keep this bridge useful:

1. **Migrate to in-process audit**: add ``llmharness.adapters.agentm`` to
   the main agent's extensions (works only when the main agent is itself
   AgentM). This is the recommended path.
2. **Ship your own audit scenario YAML**: set ``LLMHARNESS_AGENTM_SCENARIO``
   to its name (resolved via ``<cwd>/scenarios/<name>/manifest.yaml``),
   and your scenario must accept the same input shape and emit the
   :class:`RawAuditOutput` JSON. Useful when the main agent is Claude
   Code (cannot host AgentM extensions in-process) and you still want
   AgentM-quality drift detection.

If neither env var nor migration is set, :func:`monitor_via_agentm`
raises :class:`AgentMError` with a clear pointer.

Env vars:
  LLMHARNESS_AGENTM_BIN       path to the agentm CLI (default: ``agentm``)
  LLMHARNESS_AGENTM_CWD       cwd for the agentm subprocess; the AgentM
                              checkout root scenarios are resolved against
  LLMHARNESS_AGENTM_MODEL     provider model id (passed via ``--model``)
  LLMHARNESS_AGENTM_TIMEOUT   seconds, default 120
  LLMHARNESS_AGENTM_SCENARIO  scenario name to invoke; required since the
                              built-in ``harness_monitor`` scenario was
                              removed by the V0 refactor
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audit import RawAuditOutput
from .audit import extract_json as _extract_json
from .schema import Event, Verdict


class AgentMError(RuntimeError):
    pass


@dataclass(frozen=True)
class _Config:
    bin: str
    cwd: str | None
    model: str | None
    timeout: float
    scenario: str | None

    @classmethod
    def from_env(cls) -> _Config:
        return cls(
            bin=os.environ.get("LLMHARNESS_AGENTM_BIN", "agentm"),
            cwd=os.environ.get("LLMHARNESS_AGENTM_CWD") or None,
            model=os.environ.get("LLMHARNESS_AGENTM_MODEL") or None,
            timeout=float(os.environ.get("LLMHARNESS_AGENTM_TIMEOUT", "120")),
            scenario=os.environ.get("LLMHARNESS_AGENTM_SCENARIO") or None,
        )


def _run_scenario(scenario: str, payload: dict[str, Any]) -> dict[str, Any]:
    cfg = _Config.from_env()
    prompt = json.dumps(payload, ensure_ascii=False)
    cmd = [cfg.bin, prompt, "--scenario", scenario, "--quiet"]
    if cfg.model:
        cmd += ["--model", cfg.model]

    try:
        proc = subprocess.run(
            cmd,
            cwd=cfg.cwd,
            input="",
            capture_output=True,
            text=True,
            timeout=cfg.timeout,
            check=False,
        )
    except FileNotFoundError as e:
        raise AgentMError(f"agentm binary not found: {cfg.bin}") from e
    except subprocess.TimeoutExpired as e:
        raise AgentMError(f"agentm scenario {scenario} timed out") from e

    if proc.returncode != 0:
        raise AgentMError(
            f"agentm scenario {scenario} exited {proc.returncode}: "
            f"{(proc.stderr or proc.stdout)[-500:]}"
        )

    data = _extract_json(proc.stdout)
    if data is None:
        raise AgentMError(
            f"agentm scenario {scenario} produced no parseable JSON; "
            f"stdout tail: {proc.stdout[-300:]}"
        )
    if not isinstance(data, dict):
        raise AgentMError(f"agentm scenario {scenario} returned non-object: {data!r}")
    return data


# ---------------------------------------------------------------------------
# Public API


def monitor_via_agentm(
    new_turns_payload: list[dict[str, Any]],
    history_events_tail: list[dict[str, Any]],
    next_event_id: int,
) -> tuple[list[Event], Verdict]:
    """One call: fold turns into events AND judge drift.

    Shape contract is :data:`llmharness.audit.AUDIT_SYSTEM_PROMPT` step 10.
    Parsing flows through :class:`RawAuditOutput` so this dormant subprocess
    path and the in-process V0 audit stay schema-locked. ``next_event_id``
    seeds id assignment — the LLM never emits ``id``.

    Raises :class:`AgentMError` immediately when ``LLMHARNESS_AGENTM_SCENARIO``
    is unset, since the legacy ``harness_monitor`` scenario YAML no longer
    exists. See module docstring for the migration path.
    """

    cfg = _Config.from_env()
    if cfg.scenario is None:
        raise AgentMError(
            "LLMHARNESS_AGENTM_SCENARIO is unset and the legacy 'harness_monitor' "
            "scenario was removed by the V0 cognitive-audit refactor. Either set "
            "LLMHARNESS_AGENTM_SCENARIO to your own audit scenario, or migrate to "
            "the in-process llmharness.adapters.agentm extension (recommended for "
            "AgentM-on-AgentM)."
        )

    payload = {
        "next_event_id": next_event_id,
        "history_events_tail": history_events_tail,
        "new_turns": new_turns_payload,
    }
    out = _run_scenario(cfg.scenario, payload)

    parsed = RawAuditOutput.from_dict(out)
    if parsed is None:
        raise AgentMError(
            f"monitor returned no parseable verdict; raw output keys: {sorted(out.keys())}"
        )
    return parsed.to_events(next_id=next_event_id), parsed.to_verdict()


def dump_distillation(
    distill_dir: Path,
    sid: str,
    *,
    monitor_input: dict[str, Any],
    monitor_events: list[Event],
    monitor_verdict: Verdict,
) -> None:
    """Append a single (input, output) record per tick for offline distillation."""

    distill_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "input": monitor_input,
        "output": {
            "events": [e.to_dict() for e in monitor_events],
            "verdict": monitor_verdict.to_dict(),
        },
    }
    path = distill_dir / f"{sid}.monitor.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
