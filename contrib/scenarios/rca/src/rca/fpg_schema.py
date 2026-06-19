"""Shared fpg schema helpers for the RCA scenario."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from rca import SCENARIO_ROOT


class FpgOutputConfig(BaseModel):
    """Config shared by fpg-backed RCA atoms."""

    model_config = ConfigDict(extra="forbid")

    profile_path: str | None = None


def resolve_profile_path(profile_path: str | None, *, scenario_dir: str | None) -> Path:
    """Resolve the fpg vocabulary profile used by RCA output validation."""

    if profile_path:
        path = Path(profile_path)
        if not path.is_absolute():
            base = Path(scenario_dir) if scenario_dir else SCENARIO_ROOT
            path = base / path
        return path.resolve()
    if scenario_dir:
        return (Path(scenario_dir) / "fpg_profile.toml").resolve()
    return (SCENARIO_ROOT / "fpg_profile.toml").resolve()


@lru_cache(maxsize=8)
def _schema_bundle(profile_path: str) -> Any:
    from fpg import build_schema, load_profile

    return build_schema(load_profile(profile_path))


def schema_bundle(profile_path: Path) -> Any:
    return _schema_bundle(str(profile_path))


def model_output_model(profile_path: Path) -> type[Any]:
    return schema_bundle(profile_path).ModelRCAOutput


def vocab_for_contract(profile_path: Path) -> dict[str, str]:
    return schema_bundle(profile_path).vocab_for_model()


def model_output_tool_schema(profile_path: Path) -> dict[str, Any]:
    """Return a tool-call-friendly JSON schema for ModelRCAOutput.

    Pydantic emits local ``$defs`` / ``$ref`` entries. Some model tool-call
    backends do not resolve references, so inline them before registration.
    """

    raw = model_output_model(profile_path).model_json_schema()
    return _inline_local_refs(raw)


def contract_json_schema(profile_path: Path) -> str:
    """Stable pretty JSON for the prompt contract block."""

    return json.dumps(model_output_tool_schema(profile_path), indent=2, sort_keys=True)


def _inline_local_refs(schema: dict[str, Any]) -> dict[str, Any]:
    source = deepcopy(schema)
    defs = source.pop("$defs", {})

    def expand(value: Any) -> Any:
        if isinstance(value, list):
            return [expand(item) for item in value]
        if not isinstance(value, dict):
            return value

        ref = value.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            key = ref.removeprefix("#/$defs/")
            target = defs.get(key)
            if isinstance(target, dict):
                merged = expand(deepcopy(target))
                for k, v in value.items():
                    if k != "$ref":
                        merged[k] = expand(v)
                return merged

        return {k: expand(v) for k, v in value.items() if k != "$defs"}

    expanded = expand(source)
    if not isinstance(expanded, dict):
        raise TypeError("expanded JSON schema root is not an object")
    return expanded


__all__ = [
    "FpgOutputConfig",
    "contract_json_schema",
    "model_output_model",
    "model_output_tool_schema",
    "resolve_profile_path",
    "vocab_for_contract",
]
