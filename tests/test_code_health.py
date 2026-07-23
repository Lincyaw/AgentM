"""Behavior contracts for project-specific architecture checks."""

from __future__ import annotations

from pathlib import Path

from agentm.code_health import check_file
from agentm.core.lib.redact import redact_config


def _issues_for(
    tmp_path: Path,
    source: str,
    *,
    relative: str,
) -> list[str]:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return [issue.rule for issue in check_file(path)]


def test_code_health_rejects_core_atom_policy(tmp_path: Path) -> None:
    rules = _issues_for(
        tmp_path,
        'FLOOR = ("agentm.extensions.builtin.retry_policy",)\n',
        relative="src/agentm/core/runtime/factory.py",
    )
    assert "AM017" in rules


def test_code_health_flags_subprocess_in_contrib_atom(tmp_path: Path) -> None:
    rules = _issues_for(
        tmp_path,
        "import subprocess\n\ndef run() -> None:\n    subprocess.run(['ls'])\n",
        relative="contrib/extensions/policy/src/policy_engine/tooling.py",
    )
    assert "AM004" in rules


def test_code_health_allows_subprocess_in_host_code(tmp_path: Path) -> None:
    for relative in (
        "src/agentm/cli/_tools.py",
        "contrib/scenarios/harbor/src/agentm_harbor/host_adapter.py",
    ):
        rules = _issues_for(
            tmp_path,
            "import subprocess\n\ndef run() -> None:\n    subprocess.run(['ls'])\n",
            relative=relative,
        )
        assert "AM004" not in rules


def test_code_health_rejects_scenario_module_preload(tmp_path: Path) -> None:
    rules = _issues_for(
        tmp_path,
        """\
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("local", "local.py")
sys.modules["local"] = object()
""",
        relative="src/agentm/scenarios.py",
    )
    assert rules.count("AM018") == 2


def test_code_health_rejects_legacy_extension_shape(tmp_path: Path) -> None:
    rules = _issues_for(
        tmp_path,
        """\
module = resolved.provider[0]
for module, config in resolved.extensions:
    consume(module, config)
""",
        relative="src/agentm/presenter.py",
    )
    assert rules.count("AM019") == 2


def test_code_health_rejects_raw_cli_config(tmp_path: Path) -> None:
    rules = _issues_for(
        tmp_path,
        "payload = dict(extension.config)\n",
        relative="src/agentm/cli/config.py",
    )
    assert "AM020" in rules


def test_code_health_rejects_dynamic_attribute_access(tmp_path: Path) -> None:
    rules = _issues_for(
        tmp_path,
        """\
value = getattr(subject, "value")
if hasattr(subject, "close"):
    subject.close()
""",
        relative="src/agentm/runtime.py",
    )
    assert rules.count("AM021") == 2


def test_code_health_rejects_runtime_type_checks(tmp_path: Path) -> None:
    rules = _issues_for(
        tmp_path,
        """\
import builtins

if isinstance(value, str):
    handle_text(value)
if builtins.isinstance(value, int):
    handle_number(value)
""",
        relative="src/agentm/runtime.py",
    )
    assert rules.count("AM025") == 2


def test_code_health_allows_runtime_type_check_with_precise_ignore(
    tmp_path: Path,
) -> None:
    rules = _issues_for(
        tmp_path,
        """\
if isinstance(value, str):  # code-health: ignore[AM025] -- wire validation boundary
    handle_text(value)
""",
        relative="src/agentm/runtime/wire.py",
    )
    assert "AM025" not in rules


def test_code_health_rejects_any_and_bare_dict_in_all_source(
    tmp_path: Path,
) -> None:
    rules = _issues_for(
        tmp_path,
        """\
from typing import Any

def decode(value: Any) -> dict:
    return value
""",
        relative="src/agentm/runtime/codec.py",
    )
    assert rules.count("AM022") == 1
    assert rules.count("AM023") == 1


def test_code_health_rejects_stdlib_logging(tmp_path: Path) -> None:
    rules = _issues_for(
        tmp_path,
        """\
import logging
from logging.handlers import RotatingFileHandler
""",
        relative="src/agentm/runtime.py",
    )
    assert rules.count("AM024") == 2


def test_code_health_rejects_hand_written_tool_schema_factory(
    tmp_path: Path,
) -> None:
    rules = _issues_for(
        tmp_path,
        """\
MANIFEST = object()

def _bash_parameters(default_timeout: float) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "cmd": {"type": "string"},
            "timeout": {"type": "number", "default": default_timeout},
        },
        "required": ["cmd"],
    }

def install(api, config):
    api.register_tool(_BashTool(parameters=_bash_parameters(120.0)))
""",
        relative="contrib/extensions/policy/src/policy_engine/__init__.py",
    )
    assert rules.count("AM011") == 1


def test_code_health_rejects_hand_written_tool_schema_in_async_atom(
    tmp_path: Path,
) -> None:
    rules = _issues_for(
        tmp_path,
        """\
MANIFEST = object()

async def install(api, config):
    api.register_tool(
        FunctionTool(parameters={"type": "object", "properties": {}}, fn=execute)
    )
""",
        relative="src/agentm/extensions/builtin/async_atom.py",
    )
    assert rules.count("AM011") == 1


def test_code_health_allows_pydantic_tool_schema(
    tmp_path: Path,
) -> None:
    rules = _issues_for(
        tmp_path,
        """\
MANIFEST = object()

class _Args(BaseModel):
    cmd: str

def install(api, config):
    api.register_tool(FunctionTool(parameters=_Args, fn=execute))
    api.register_tool(_CustomTool(parameters=pydantic_to_tool_schema(_Args)))
""",
        relative="contrib/extensions/policy/src/policy_engine/__init__.py",
    )
    assert "AM011" not in rules


def test_code_health_rejects_authoring_presenter_dependency(
    tmp_path: Path,
) -> None:
    rules = _issues_for(
        tmp_path,
        "from agentm.presenter.frontmatter import FrontmatterDocument\n",
        relative="src/agentm/authoring/artifacts.py",
    )
    assert "AM010" in rules


def test_code_health_rejects_core_and_backend_dependency_inversions(
    tmp_path: Path,
) -> None:
    core_rules = _issues_for(
        tmp_path,
        "from agentm.extensions.builtin.retry_policy import Retry\n",
        relative="src/agentm/core/runtime/session.py",
    )
    storage_rules = _issues_for(
        tmp_path,
        "from agentm.core.runtime.stores.memory import Store\n",
        relative="src/agentm/storage/trajectory/jsonl.py",
    )
    assert "AM010" in core_rules
    assert "AM010" in storage_rules


def test_code_health_allows_typed_dict_and_precise_ignores(
    tmp_path: Path,
) -> None:
    rules = _issues_for(
        tmp_path,
        """\
from typing import Any, Final  # code-health: ignore[AM022] -- vendor boundary

payload: Final[dict[str, object]] = {}
value = getattr(subject, "value")  # code-health: ignore[AM021] -- reflection
""",
        relative="src/agentm/runtime/adapter.py",
    )
    assert not rules


def test_code_health_allows_file_level_rule_ignore(tmp_path: Path) -> None:
    rules = _issues_for(
        tmp_path,
        """\
# code-health: ignore-file[AM022] -- untyped wire decoder boundary
from typing import Any

def decode(value: Any) -> object:
    return value
""",
        relative="src/agentm/runtime/wire.py",
    )
    assert not rules


def test_current_composition_boundaries_pass_code_health() -> None:
    paths = (
        Path("src/agentm/core/runtime/session_factory.py"),
        Path("src/agentm/scenarios.py"),
        Path("src/agentm/cli/_config.py"),
        Path("src/agentm/cli/_scenario.py"),
    )
    issues = [issue for path in paths for issue in check_file(path)]
    boundary_rules = {f"AM{number:03d}" for number in range(17, 25)}
    assert not [issue for issue in issues if issue.rule in boundary_rules]


def test_code_health_retains_main_branch_architecture_rules(
    tmp_path: Path,
) -> None:
    rules = _issues_for(
        tmp_path,
        """\
from pathlib import Path

def build(config):
    try:
        work()
    except Exception:
        return None
    return AgentSessionConfig(**config)

root = Path(".").resolve().parent
""",
        relative="src/agentm/core/runtime/sample.py",
    )
    assert {"AM001", "AM012", "AM014"} <= set(rules)


def test_user_visible_config_redacts_nested_credentials() -> None:
    assert redact_config(
        {
            "api_key": "secret",
            "api_key_env": "OPENAI_API_KEY",
            "default_headers": {
                "Authorization": "Bearer secret",
                "X-TT-LOGID": "agentm",
            },
            "nested": {"client_secret": "hidden"},
        }
    ) == {
        "api_key": "***",
        "api_key_env": "OPENAI_API_KEY",
        "default_headers": {
            "Authorization": "***",
            "X-TT-LOGID": "agentm",
        },
        "nested": {"client_secret": "***"},
    }
