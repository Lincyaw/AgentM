"""Concrete host-side ``SessionSpecResolver`` implementation."""

# code-health: ignore-file[AM022] -- validates untyped TOML and environment input

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agentm.core.abi.provider import ProviderSessionIdentity
from agentm.core.abi.session_api import (
    AgentSessionConfig,
    ConfigSource,
    ConfigValueProvenance,
    ExtensionSpec,
    ResolvedSessionSpec,
    ScenarioSpec,
    normalize_extension_spec,
)


class DefaultSessionSpecResolver:
    """Resolve session config from explicit SDK fields, env, and TOML profiles."""

    def __init__(
        self,
        *,
        project_config: str | Path | None = None,
        user_config: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._project_config = Path(project_config) if project_config is not None else None
        self._user_config = Path(user_config) if user_config is not None else _default_user_config()
        self._env = dict(os.environ if env is None else env)

    def resolve(self, request: AgentSessionConfig) -> ResolvedSessionSpec:
        provenance: list[ConfigValueProvenance] = []
        user_config = _load_toml(self._user_config)
        project_config = _load_toml(self._project_config)

        scenario = _choose(
            ("explicit", request.scenario, "AgentSessionConfig.scenario"),
            ("env", self._env.get("AGENTM_SCENARIO"), "AGENTM_SCENARIO"),
            (
                "project_config",
                _get_path(project_config, ("default_scenario",)),
                str(self._project_config) if self._project_config is not None else None,
            ),
            (
                "user_config",
                _get_path(user_config, ("default_scenario",)),
                str(self._user_config) if self._user_config is not None else None,
            ),
        )
        if scenario.source is not None:
            scenario_name = _required_nonempty_str(
                scenario.value,
                "scenario",
            )
            provenance.append(
                _provenance(
                    "scenario",
                    scenario.source,
                    scenario.ref,
                    scenario_name,
                )
            )
        else:
            scenario_name = None

        extensions = self._resolve_extensions(
            request,
            scenario_name,
            provenance,
        )
        atom_config = self._resolve_atom_config(
            request,
            project_config,
            user_config,
            provenance,
        )
        provider, provider_identity = self._resolve_provider(
            request,
            project_config,
            user_config,
            provenance,
        )

        return ResolvedSessionSpec(
            scenario=scenario_name,
            extensions=tuple(extensions),
            atom_config=atom_config,
            provider=provider,
            provider_identity=provider_identity,
            value_provenance=tuple(provenance),
            provenance={
                "resolver": type(self).__name__,
                "project_config": str(self._project_config) if self._project_config is not None else None,
                "user_config": str(self._user_config) if self._user_config is not None else None,
            },
        )

    def _resolve_extensions(
        self,
        request: AgentSessionConfig,
        scenario: object,
        provenance: list[ConfigValueProvenance],
    ) -> list[ExtensionSpec]:
        if request.extensions is not None:
            extensions = [
                normalize_extension_spec(item)
                for item in request.extensions
            ]
            provenance.append(
                _provenance(
                    "extensions",
                    "explicit",
                    "AgentSessionConfig.extensions",
                    _extension_records(extensions),
                )
            )
            return extensions
        if isinstance(scenario, str) and request.scenario_loader is not None:
            loaded = request.scenario_loader(scenario)
            raw_extensions = (
                loaded.extensions
                if isinstance(loaded, ScenarioSpec)
                else loaded
            )
            extensions = [
                normalize_extension_spec(item)
                for item in raw_extensions
            ]
            provenance.append(
                _provenance(
                    "extensions",
                    "scenario_default",
                    scenario,
                    _extension_records(extensions),
                )
            )
            return extensions
        return []

    def _resolve_atom_config(
        self,
        request: AgentSessionConfig,
        project_config: Mapping[str, Any],
        user_config: Mapping[str, Any],
        provenance: list[ConfigValueProvenance],
    ) -> dict[str, dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for source, config, ref in (
            ("user_config", user_config, str(self._user_config) if self._user_config is not None else None),
            ("project_config", project_config, str(self._project_config) if self._project_config is not None else None),
        ):
            atoms = _get_path(config, ("atoms",))
            if isinstance(atoms, Mapping):
                for name, atom_config in atoms.items():
                    if isinstance(name, str) and isinstance(atom_config, Mapping):
                        merged.setdefault(name, {}).update(dict(atom_config))
                        provenance.append(
                            _provenance(
                                f"atom_config.{name}",
                                source,  # type: ignore[arg-type]
                                ref,
                                atom_config,
                            )
                        )
        for name, atom_config in request.atom_config_overrides.items():
            merged.setdefault(name, {}).update(dict(atom_config))
            provenance.append(
                _provenance(
                    f"atom_config.{name}",
                    "atom_override",
                    "AgentSessionConfig.atom_config_overrides",
                    atom_config,
                )
            )
        return merged

    def _resolve_provider(
        self,
        request: AgentSessionConfig,
        project_config: Mapping[str, Any],
        user_config: Mapping[str, Any],
        provenance: list[ConfigValueProvenance],
    ) -> tuple[ExtensionSpec | None, ProviderSessionIdentity | None]:
        if request.provider is not None:
            provider_spec = normalize_extension_spec(request.provider)
            module = provider_spec.module_path
            config = provider_spec.config
            provenance.append(
                _provenance(
                    "provider",
                    "explicit",
                    "AgentSessionConfig.provider",
                    _extension_record(provider_spec),
                )
            )
            raw_name = config.get("name")
            name = (
                _provider_name_from_module(module)
                if raw_name is None
                else _required_nonempty_str(raw_name, "provider.name")
            )
            model_id = _optional_str(config.get("model"))
            return provider_spec, ProviderSessionIdentity(
                name=name,
                model_id=model_id,
            )

        provider_name = _choose(
            ("env", self._env.get("AGENTM_PROVIDER"), "AGENTM_PROVIDER"),
            ("project_config", _get_path(project_config, ("default_provider",)), str(self._project_config) if self._project_config is not None else None),
            ("user_config", _get_path(user_config, ("default_provider",)), str(self._user_config) if self._user_config is not None else None),
            ("project_config", _get_path(project_config, ("default_model",)), str(self._project_config) if self._project_config is not None else None),
            ("user_config", _get_path(user_config, ("default_model",)), str(self._user_config) if self._user_config is not None else None),
        )
        if provider_name.source is None:
            return None, None
        provider = _required_nonempty_str(
            provider_name.value,
            "default provider",
        )

        project_profile = _provider_profile(provider, project_config)
        user_profile = _provider_profile(provider, user_config)
        project_ref = (
            str(self._project_config)
            if self._project_config is not None
            else None
        )
        user_ref = (
            str(self._user_config)
            if self._user_config is not None
            else None
        )
        model = _choose(
            ("env", self._env.get("AGENTM_MODEL"), "AGENTM_MODEL"),
            ("project_config", project_profile.get("model"), project_ref),
            ("user_config", user_profile.get("model"), user_ref),
        )
        provider_config = {**user_profile, **project_profile}
        for resolver_only_key in (
            "api_key",
            "api_key_env",
            "base_url",
            "model",
            "name",
            "provider",
        ):
            provider_config.pop(resolver_only_key, None)
        provider_config["name"] = provider
        if model.source is not None:
            model_id = _required_nonempty_str(
                model.value,
                "provider.model",
            )
            provider_config["model"] = model_id
            provenance.append(_provenance("provider.model", model.source, model.ref, model_id))
        api_key = _provider_api_key(
            provider,
            project_profile,
            user_profile,
            self._env,
            project_ref=project_ref,
            user_ref=user_ref,
        )
        if api_key.value is not None:
            api_key_value = _required_nonempty_str(
                api_key.value,
                "provider.api_key",
            )
            provider_config["api_key"] = api_key_value
            provenance.append(_provenance("provider.api_key", api_key.source or "provider_default", api_key.ref, api_key_value))
        base_url = _choose(
            ("env", self._env.get("AGENTM_BASE_URL"), "AGENTM_BASE_URL"),
            ("project_config", project_profile.get("base_url"), project_ref),
            ("user_config", user_profile.get("base_url"), user_ref),
        )
        if base_url.source is not None:
            base_url_value = _required_nonempty_str(
                base_url.value,
                "provider.base_url",
            )
            provider_config["base_url"] = base_url_value
            provenance.append(_provenance("provider.base_url", base_url.source, base_url.ref, base_url_value))
        provenance.append(
            _provenance("provider", provider_name.source or "provider_default", provider_name.ref, provider)
        )
        provider_type = (
            _optional_str(project_profile.get("provider"))
            or _optional_str(user_profile.get("provider"))
            or provider
        )
        module = _provider_module(provider_type)
        return ExtensionSpec.from_module(
            module,
            provider_config,
        ), ProviderSessionIdentity(
            name=provider,
            model_id=_optional_str(provider_config.get("model")),
        )


class _Choice:
    def __init__(self, source: ConfigSource | None, value: object, ref: str | None) -> None:
        self.source = source
        self.value = value
        self.ref = ref


def _choose(*candidates: tuple[ConfigSource, object, str | None]) -> _Choice:
    for source, value, ref in candidates:
        if value is not None and value != "":
            return _Choice(source, value, ref)
    return _Choice(None, None, None)


def _provider_profile(
    provider: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    profile = _get_path(config, ("providers", provider))
    if isinstance(profile, Mapping):
        merged.update(dict(profile))
    models = _get_path(config, ("models", provider))
    if isinstance(models, Mapping):
        merged.update(dict(models))
    return merged


def _provider_api_key(
    provider: str,
    project_profile: Mapping[str, Any],
    user_profile: Mapping[str, Any],
    env: Mapping[str, str],
    *,
    project_ref: str | None,
    user_ref: str | None,
) -> _Choice:
    project_env_name = project_profile.get("api_key_env")
    user_env_name = user_profile.get("api_key_env")
    if project_env_name is not None:
        env_name = _required_nonempty_str(
            project_env_name,
            "provider.api_key_env",
        )
    elif user_env_name is not None:
        env_name = _required_nonempty_str(
            user_env_name,
            "provider.api_key_env",
        )
    else:
        env_name = None
    candidates: list[tuple[ConfigSource, object, str | None]] = []
    if env_name is not None:
        candidates.append(("env", env.get(env_name), env_name))
    candidates.extend(
        [
            ("env", env.get("AGENTM_API_KEY"), "AGENTM_API_KEY"),
            ("env", env.get(f"{provider.upper()}_API_KEY"), f"{provider.upper()}_API_KEY"),
            (
                "project_config",
                project_profile.get("api_key"),
                project_ref,
            ),
            ("user_config", user_profile.get("api_key"), user_ref),
        ]
    )
    return _choose(*candidates)


def _provider_module(provider: str) -> str:
    if provider == "openai":
        return "agentm.extensions.builtin.llm_openai"
    if provider == "anthropic":
        return "agentm.extensions.builtin.llm_anthropic"
    return provider


def _provider_name_from_module(module: str) -> str:
    if module.endswith("llm_openai"):
        return "openai"
    if module.endswith("llm_anthropic"):
        return "anthropic"
    return module.rsplit(".", 1)[-1]


def _load_toml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return data if isinstance(data, dict) else {}


def _default_user_config() -> Path:
    home = os.environ.get("AGENTM_HOME")
    if home:
        return Path(home).expanduser() / "config.toml"
    return Path.home() / ".agentm" / "config.toml"


def _get_path(data: Mapping[str, Any], path: tuple[str, ...]) -> object:
    current: object = data
    for part in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return _required_nonempty_str(value, "provider.model")


def _required_nonempty_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _provenance(
    path: str,
    source: ConfigSource,
    source_ref: str | None,
    value: object,
) -> ConfigValueProvenance:
    return ConfigValueProvenance(
        path=path,
        source=source,
        source_ref=source_ref,
        value_fingerprint=_fingerprint(value),
    )


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _extension_record(spec: ExtensionSpec) -> dict[str, object]:
    return {
        "source": {
            "kind": spec.source.kind,
            "location": spec.source.location,
            "digest": spec.source.digest,
        },
        "config_keys": sorted(spec.config),
    }


def _extension_records(
    specs: list[ExtensionSpec],
) -> list[dict[str, object]]:
    return [_extension_record(spec) for spec in specs]


__all__ = ["DefaultSessionSpecResolver"]
