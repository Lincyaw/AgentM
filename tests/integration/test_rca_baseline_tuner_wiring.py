"""Smoke test for the rca:baseline tuner wiring (no LLM required).

Asserts:

1. The tuner manifest loads cleanly via ``load_scenario_with_meta`` and
   declares ``task_class: rca_baseline_tuner``.
2. Each atom in the tuner manifest installs against a stub ExtensionAPI
   without raising — catches config-shape regressions early.
3. The eval ``tasks/*.yaml`` files parse and carry the schema the
   grader / ``tool_eval_run`` rely on.
4. The programmatic grader correctly scores a synthetic trace fixture
   that mirrors the rca:baseline emit shape (list_tables → query_sql →
   submit_final_report). Verifies both a matching verdict (score=1.0)
   and a wrong verdict (score=0.0).

Real-LLM end-to-end runs are deliberately NOT included — those require
API keys and live in ``test_per_task_evolution_e2e.py`` (skipped by
default).
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml

# Per CLAUDE.md "Testing philosophy": this entire file is wiring (manifest-
# load / atoms-install / yaml-parse) — none of these gate a fail-stop
# position. Mark the whole module ``smoke`` so iteration on fail-stop
# coverage can filter it out with ``pytest -m 'not smoke'``.
pytestmark = pytest.mark.smoke

# Lock the tests to the worktree root so manifest path resolution + the
# grader's CWD-relative .agentm/observability lookup both line up.
_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Stub ExtensionAPI — minimal surface needed by the six Phase-2 atoms +
# system_prompt + observability + tool_read at install time.
# ---------------------------------------------------------------------------


class _StubAPI:
    def __init__(self, cwd: Path) -> None:
        self.cwd = str(cwd)
        self.tools: list[Any] = []
        self.commands: dict[str, Any] = {}
        self.providers: dict[str, Any] = {}
        self.handlers: dict[str, list[Any]] = {}
        self.renderers: dict[str, Any] = {}
        # Some atoms (skill_loader, prompt_templates) reach for catalog /
        # service-registry — none load in the tuner manifest, but provide
        # the attrs in case future atoms appear.
        self.catalog = _NoopService()
        self._services: dict[str, Any] = {}

    def set_service(self, name: str, obj: Any) -> None:
        self._services[name] = obj

    def get_service(self, name: str) -> Any:
        return self._services.get(name)

    def on(self, channel: str, handler: Any, *, priority: int = 0) -> Any:
        self.handlers.setdefault(channel, []).append(handler)
        return lambda: None

    def register_tool(self, tool: Any) -> None:
        self.tools.append(tool)

    def register_command(self, name: str, spec: Any) -> None:
        self.commands[name] = spec

    def register_provider(self, name: str, config: Any) -> None:
        self.providers[name] = config

    def register_message_renderer(self, custom_type: str, renderer: Any) -> None:
        self.renderers[custom_type] = renderer

    def get_operations(self) -> Any:
        return _NoopService()

    def list_atoms(self) -> list[Any]:
        return []


class _NoopService:
    def __getattr__(self, name: str) -> Any:
        def _stub(*args: Any, **kwargs: Any) -> None:
            return None

        return _stub


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tuner_manifest_loads_cleanly() -> None:
    """Asserts the tuner manifest is parseable and declares the right
    task_class. Catches name-vs-dirname mismatches and YAML typos."""
    cwd = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        from agentm.extensions.loader import load_scenario_with_meta

        extensions, meta = load_scenario_with_meta("rca/tuner")
    finally:
        os.chdir(cwd)

    module_paths = [m for m, _cfg in extensions]
    expected = {
        "agentm.extensions.builtin.tool_read",
        "agentm.extensions.builtin.tool_query_traces",
        "agentm.extensions.builtin.tool_query_candidates",
        "agentm.extensions.builtin.tool_query_module_feedback",
        "agentm.extensions.builtin.tool_reflect",
        "agentm.extensions.builtin.tool_eval_run",
        "contrib.extensions.changespec_validators",
        "agentm.extensions.builtin.tool_propose_change",
        "agentm.extensions.builtin.system_prompt",
        "agentm.extensions.builtin.observability",
    }
    assert expected.issubset(set(module_paths)), (
        f"missing atoms: {expected - set(module_paths)}"
    )
    assert meta.get("task_class") == "rca_baseline_tuner"


def test_tuner_atoms_install_against_stub_api(tmp_path: Path) -> None:
    """Each atom's install() must run without raising on a stub API.
    This is the smallest fail-stop that covers config-shape regressions
    in the new manifest."""
    cwd = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        from agentm.extensions.loader import load_scenario_with_meta

        extensions, _meta = load_scenario_with_meta("rca/tuner")
    finally:
        os.chdir(cwd)

    api = _StubAPI(tmp_path)
    # observability needs a richer API surface (api.events, api.session_id,
    # api.catalog with hash methods); it's load-bearing at runtime but not
    # the regression we're guarding here. Skip it; the GEPA atoms are the
    # ones whose config-shape we care about catching early.
    skip_modules = {"agentm.extensions.builtin.observability"}
    installed = 0
    for module_path, config in extensions:
        if module_path in skip_modules:
            continue
        module = importlib.import_module(module_path)
        install_fn = getattr(module, "install", None)
        assert callable(install_fn), f"{module_path} has no install()"
        result = install_fn(api, config)
        if inspect.iscoroutine(result):
            asyncio.run(result)
        installed += 1
    # All six Phase-2 atoms + system_prompt + tool_read = 8 installs.
    assert installed >= 8
    # Each Phase-2 tool atom registers exactly one tool; tool_read also
    # registers one. system_prompt registers no tool but binds a handler.
    assert len(api.tools) >= 7


def test_eval_tasks_parse_and_have_required_keys() -> None:
    tasks_dir = (
        _REPO_ROOT / "contrib" / "scenarios" / "rca" / "eval" / "baseline" / "tasks"
    )
    files = sorted(tasks_dir.glob("*.yaml"))
    assert len(files) == 3, f"expected exactly 3 task YAMLs, got {files}"
    holdout_count = 0
    for path in files:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict), f"{path} did not parse to a dict"
        assert payload.get("task_class") == "rca_baseline"
        assert isinstance(payload.get("id"), str) and payload["id"]
        inp = payload.get("input") or {}
        assert isinstance(inp.get("user_message"), str) and inp["user_message"].strip()
        expected = payload.get("expected") or {}
        services = expected.get("expected_services") or []
        assert isinstance(services, list) and services, (
            f"{path} has empty expected_services"
        )
        assert isinstance(expected.get("fault_kind"), str)
        if payload.get("holdout") is True:
            holdout_count += 1
    # Sanity: at least one holdout exists so the gate has a holdout signal.
    assert holdout_count >= 1


def _load_grader() -> Any:
    grader_path = (
        _REPO_ROOT
        / "contrib"
        / "scenarios"
        / "rca"
        / "eval"
        / "baseline"
        / "grader.py"
    )
    spec = importlib.util.spec_from_file_location(
        f"_test_grader_{uuid.uuid4().hex[:6]}", grader_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.grade


def _write_synthetic_trace(
    obs_dir: Path,
    *,
    task_id: str,
    service: str,
    fault_kind: str,
    include_binder_error: bool,
) -> None:
    """Compose a tiny JSONL trace mimicking the rca:baseline emit shape:
    session.fingerprint (carries task_id) → emit:tool_call(list_tables) →
    [optional Binder Error tool_result] → emit:tool_call(query_sql) →
    emit:tool_call(submit_final_report root_causes=[{service, fault_kind}]).
    """
    obs_dir.mkdir(parents=True, exist_ok=True)
    trace_id = uuid.uuid4().hex
    path = obs_dir / f"{trace_id}.jsonl"
    now_ns = int(time.time() * 1e9)
    records: list[dict[str, Any]] = [
        {
            "schema": "otel/span/v0",
            "kind": "session.fingerprint",
            "trace_id": trace_id,
            "start_time_unix_nano": now_ns,
            "attributes": {
                "task_meta": {
                    "task_class": "rca_baseline",
                    "task_id": task_id,
                    "eval_run_id": "er_test",
                },
                "atoms": {},
            },
        },
        {
            "kind": "event.dispatch",
            "name": "emit:tool_call",
            "trace_id": trace_id,
            "attributes": {
                "channel": "tool_call",
                "event": {"tool_name": "list_tables", "args": {}},
            },
        },
    ]
    if include_binder_error:
        records.append(
            {
                "kind": "event.dispatch",
                "name": "emit:tool_result",
                "trace_id": trace_id,
                "attributes": {
                    "channel": "tool_result",
                    "event": {
                        "tool_name": "query_sql",
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        {
                                            "error": (
                                                "query failed: Binder Error: "
                                                'Referenced table \\"attr\\" not found!'
                                            ),
                                            "sql": "SELECT attr.http.response.status_code FROM abnormal_traces",
                                        }
                                    ),
                                }
                            ],
                            "is_error": True,
                        },
                    },
                },
            }
        )
    records.append(
        {
            "kind": "event.dispatch",
            "name": "emit:tool_call",
            "trace_id": trace_id,
            "attributes": {
                "channel": "tool_call",
                "event": {
                    "tool_name": "submit_final_report",
                    "args": {
                        "root_causes": [
                            {
                                "service": service,
                                "fault_kind": fault_kind,
                                "evidence": [
                                    {
                                        "kind": "metric",
                                        "sql": "SELECT 1",
                                        "claim": "synthetic",
                                    }
                                ],
                            }
                        ]
                    },
                },
            },
        }
    )
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def test_grader_scores_correct_verdict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic trace whose submit_final_report names the expected
    service + fault_kind must score 1.0 (full match)."""
    monkeypatch.chdir(tmp_path)
    _write_synthetic_trace(
        tmp_path / ".agentm" / "observability",
        task_id="01_mysql_corrupt",
        service="ts-station-service",
        fault_kind="network_corrupt",
        include_binder_error=True,
    )
    grade = _load_grader()
    task = {
        "id": "01_mysql_corrupt",
        "expected": {
            "expected_services": ["mysql", "ts-station-service"],
            "fault_kind": "network_corrupt",
        },
    }
    result = grade(task, "ignored")
    assert result["score"] == pytest.approx(1.0)
    # module_feedback should fire query_sql when Binder Error is present
    # in the trace — proves the per-module credit-assignment wire.
    assert "query_sql" in result["module_feedback"]


def test_grader_scores_wrong_verdict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verdict that names neither expected service nor fault_kind must
    score 0.0."""
    monkeypatch.chdir(tmp_path)
    _write_synthetic_trace(
        tmp_path / ".agentm" / "observability",
        task_id="01_mysql_corrupt",
        service="ts-foo-service",
        fault_kind="cpu_stress",
        include_binder_error=False,
    )
    grade = _load_grader()
    task = {
        "id": "01_mysql_corrupt",
        "expected": {
            "expected_services": ["mysql", "ts-station-service"],
            "fault_kind": "network_corrupt",
        },
    }
    result = grade(task, "ignored")
    assert result["score"] == pytest.approx(0.0)
    # No Binder Error in the trace → no query_sql module_feedback.
    assert "query_sql" not in result["module_feedback"]
