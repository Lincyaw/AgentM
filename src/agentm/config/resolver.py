"""Concrete host-side ``SessionSpecResolver`` implementation."""

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
            provenance.append(_provenance("scenario", scenario.source, scenario.ref, scenario.value))

        extensions = self._resolve_extensions(request, scenario.value, provenance)
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
            scenario=scenario.value if isinstance(scenario.value, str) else None,
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
            provenance.append(
                _provenance("extensions", "explicit", "AgentSessionConfig.extensions", request.extensions)
            )
            return [(module, dict(config)) for module, config in request.extensions]
        if isinstance(scenario, str) and request.scenario_loader is not None:
            loaded = request.scenario_loader(scenario)
            if hasattr(loaded, "extensions"):
                extensions = getattr(loaded, "extensions")
            else:
                extensions = loaded
            provenance.append(
                _provenance("extensions", "scenario_default", scenario, extensions)
            )
            return [(module, dict(config)) for module, config in extensions]
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
            module, config = request.provider
            provenance.append(
                _provenance("provider", "explicit", "AgentSessionConfig.provider", request.provider)
            )
            name = str(config.get("name") or _provider_name_from_module(module))
            model_id = _optional_str(config.get("model"))
            return (module, dict(config)), ProviderSessionIdentity(name=name, model_id=model_id)

        provider_name = _choose(
            ("env", self._env.get("AGENTM_PROVIDER"), "AGENTM_PROVIDER"),
            ("project_config", _get_path(project_config, ("default_provider",)), str(self._project_config) if self._project_config is not None else None),
            ("user_config", _get_path(user_config, ("default_provider",)), str(self._user_config) if self._user_config is not None else None),
        )
        provider = provider_name.value
        if not isinstance(provider, str) or not provider:
            return None, None

        profile = _provider_profile(provider, project_config, user_config)
        model = _choose(
            ("env", self._env.get("AGENTM_MODEL"), "AGENTM_MODEL"),
            ("project_config", profile.get("model"), str(self._project_config) if self._project_config is not None else None),
            ("user_config", profile.get("model"), str(self._user_config) if self._user_config is not None else None),
        )
        provider_config: dict[str, Any] = {key: value for key, value in profile.items() if isinstance(key, str)}
        provider_config["name"] = provider
        if isinstance(model.value, str):
            provider_config["model"] = model.value
            provenance.append(_provenance("provider.model", model.source or "provider_default", model.ref, model.value))
        api_key = _provider_api_key(provider, profile, self._env)
        if api_key.value is not None:
            provider_config["api_key"] = api_key.value
            provenance.append(_provenance("provider.api_key", api_key.source or "provider_default", api_key.ref, api_key.value))
        base_url = _choose(
            ("env", self._env.get("AGENTM_BASE_URL"), "AGENTM_BASE_URL"),
            ("project_config", profile.get("base_url"), str(self._project_config) if self._project_config is not None else None),
            ("user_config", profile.get("base_url"), str(self._user_config) if self._user_config is not None else None),
        )
        if isinstance(base_url.value, str):
            provider_config["base_url"] = base_url.value
            provenance.append(_provenance("provider.base_url", base_url.source or "provider_default", base_url.ref, base_url.value))
        provenance.append(
            _provenance("provider", provider_name.source or "provider_default", provider_name.ref, provider)
        )
        module = _provider_module(provider)
        return (module, provider_config), ProviderSessionIdentity(name=provider, model_id=_optional_str(provider_config.get("model")))


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
    project_config: Mapping[str, Any],
    user_config: Mapping[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for config in (user_config, project_config):
        profile = _get_path(config, ("providers", provider))
        if isinstance(profile, Mapping):
            merged.update(dict(profile))
        models = _get_path(config, ("models", provider))
        if isinstance(models, Mapping):
            merged.update(dict(models))
    return merged


def _provider_api_key(
    provider: str,
    profile: Mapping[str, Any],
    env: Mapping[str, str],
) -> _Choice:
    env_name = profile.get("api_key_env")
    candidates: list[tuple[ConfigSource, object, str | None]] = []
    if isinstance(env_name, str):
        candidates.append(("env", env.get(env_name), env_name))
    candidates.extend(
        [
            ("env", env.get("AGENTM_API_KEY"), "AGENTM_API_KEY"),
            ("env", env.get(f"{provider.upper()}_API_KEY"), f"{provider.upper()}_API_KEY"),
            ("project_config", profile.get("api_key"), None),
            ("user_config", profile.get("api_key"), None),
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
    return value if isinstance(value, str) else None


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


def _fingerprint(value: object) -> str | None:
    try:
        payload = json.dumps(value, sort_keys=True, default=repr).encode("utf-8")
    except TypeError:
        return None
    return "sha256:" + hashlib.sha256(payload).hexdigest()


__all__ = ["DefaultSessionSpecResolver"]
