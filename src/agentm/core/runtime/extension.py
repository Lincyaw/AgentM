"""Extension loader — import an atom module and call its install().

Atoms receive the Session object directly.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import time
from collections.abc import Awaitable
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ValidationError as PydanticValidationError

from agentm.core.abi.errors import ExtensionLoadError
from agentm.core.abi.events import ExtensionInstallEvent


_INSTALLING_EXTENSION: ContextVar[str | None] = ContextVar(
    "_installing_extension", default=None
)


def current_installing_extension() -> str:
    return _INSTALLING_EXTENSION.get() or ""


async def install_extension(
    api: Any,
    module_path: str,
    config: dict[str, Any] | None = None,
    *,
    trigger: str = "session_start",
) -> None:
    """Install one extension and emit the standard install lifecycle event."""

    resolved_config = dict(config or {})
    started_ns = time.perf_counter_ns()
    name = module_path.rsplit(".", 1)[-1]
    await api.bus.emit(
        ExtensionInstallEvent.CHANNEL,
        ExtensionInstallEvent(
            name=name,
            module_path=module_path,
            phase="start",
            config=resolved_config,
            trigger=trigger,
        ),
    )
    error: str | None = None
    try:
        result = load_extension(module_path, api, resolved_config)
        if inspect.isawaitable(result):
            await result
        api._record_installed_extension(module_path, resolved_config)
        logger.debug("installed atom: {}", module_path)
    except Exception as exc:
        error = str(exc)
        logger.exception("failed to install atom: {}", module_path)
        raise
    finally:
        await api.bus.emit(
            ExtensionInstallEvent.CHANNEL,
            ExtensionInstallEvent(
                name=name,
                module_path=module_path,
                phase="error" if error else "end",
                config=resolved_config,
                duration_ns=time.perf_counter_ns() - started_ns,
                trigger=trigger,
                error=error,
            ),
        )


def load_extension(
    module_path: str,
    api: Any,
    config: dict[str, Any],
    *,
    validate: bool = True,
) -> None | Awaitable[None]:
    """Import ``module_path`` and invoke its ``install(api, config)``.

    Returns whatever ``install`` returns:
    - ``None`` for sync extensions (caller need not await).
    - An awaitable for async extensions (caller must await).

    Raises ``ExtensionLoadError`` on any failure.
    """

    if validate:
        validate_extension_source(module_path)

    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001
        raise ExtensionLoadError(module_path, exc) from exc

    install = getattr(module, "install", None)
    if install is None or not callable(install):
        raise ExtensionLoadError(
            module_path,
            AttributeError(f"module {module_path!r} has no callable 'install' symbol"),
        )

    # Validate config via the schema declared by the atom manifest.
    resolved_config: Any = config
    manifest = getattr(module, "MANIFEST", None)
    if manifest is not None:
        schema_cls = getattr(manifest, "config_schema", None)
        if isinstance(schema_cls, type) and issubclass(
            schema_cls,
            PydanticBaseModel,
        ):
            try:
                resolved_config = schema_cls.model_validate(config)
            except PydanticValidationError as exc:
                raise ExtensionLoadError(
                    module_path,
                    ValueError(_format_config_validation_error(schema_cls, exc)),
                ) from exc

    token = _INSTALLING_EXTENSION.set(module_path)
    try:
        result = install(api, resolved_config)
    except Exception as exc:  # noqa: BLE001
        _INSTALLING_EXTENSION.reset(token)
        raise ExtensionLoadError(module_path, exc) from exc
    if not inspect.isawaitable(result):
        _INSTALLING_EXTENSION.reset(token)
        return None
    awaitable_result = result
    _INSTALLING_EXTENSION.reset(token)

    async def _await_install() -> None:
        inner_token = _INSTALLING_EXTENSION.set(module_path)
        try:
            await awaitable_result
        except Exception as exc:  # noqa: BLE001
            raise ExtensionLoadError(module_path, exc) from exc
        finally:
            _INSTALLING_EXTENSION.reset(inner_token)

    return _await_install()


def _format_config_validation_error(schema_cls: type[Any], exc: BaseException) -> str:
    errors = exc.errors() if hasattr(exc, "errors") else []
    missing: list[str] = []
    if isinstance(errors, list):
        for error in errors:
            if not isinstance(error, dict) or error.get("type") != "missing":
                continue
            loc = error.get("loc")
            if isinstance(loc, (tuple, list)):
                missing.append(".".join(str(part) for part in loc))
            elif loc:
                missing.append(str(loc))
    if missing:
        return (
            f"config for {schema_cls.__name__} is missing required field(s): "
            + ", ".join(missing)
        )
    return f"config for {schema_cls.__name__} is invalid: {exc}"


def validate_extension_source(module_path: str) -> None:
    """Run AST validation before importing an atom module."""

    try:
        from agentm.extensions.validate import (
            extension_helper_imports,
            validate_atom_file,
            validate_atom_package,
        )
    except ImportError as exc:
        raise ExtensionLoadError(
            module_path,
            RuntimeError("extension validator unavailable"),
        ) from exc

    try:
        spec = importlib.util.find_spec(module_path)
    except Exception as exc:  # noqa: BLE001
        raise ExtensionLoadError(module_path, exc) from exc
    if spec is None:
        raise ExtensionLoadError(
            module_path,
            ModuleNotFoundError(f"cannot resolve extension module {module_path!r}"),
        )

    issues = []
    visited: set[str] = {module_path}
    if spec.submodule_search_locations:
        for package_dir in spec.submodule_search_locations:
            issues.extend(
                validate_atom_package(
                    package_dir,
                    module_path=module_path,
                    known_extension_names=set(),
                )
            )
    elif spec.origin is not None:
        src_file = Path(spec.origin)
        if src_file.suffix != ".py":
            return
        if src_file.name == "__init__.py":
            issues = validate_atom_package(
                src_file.parent,
                module_path=module_path,
                known_extension_names=set(),
            )
        else:
            issues = validate_atom_file(
                src_file,
                module_path=module_path,
                known_extension_names=set(),
            )
            for helper in extension_helper_imports(src_file):
                issues.extend(
                    _validate_extension_helper_source(
                        helper,
                        visited=visited,
                        validate_atom_file=validate_atom_file,
                        validate_atom_package=validate_atom_package,
                        extension_helper_imports=extension_helper_imports,
                    )
                )
    _raise_blocking_validation_issues(module_path, issues)


def _validate_extension_helper_source(
    module_path: str,
    *,
    visited: set[str],
    validate_atom_file: Any,
    validate_atom_package: Any,
    extension_helper_imports: Any,
) -> list[Any]:
    if module_path in visited:
        return []
    visited.add(module_path)
    try:
        spec = importlib.util.find_spec(module_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"extension validator cannot inspect helper {module_path!r}"
        ) from exc
    if spec is None:
        raise ModuleNotFoundError(
            f"extension validator cannot resolve helper {module_path!r}"
        )

    issues: list[Any] = []
    if spec.submodule_search_locations:
        for package_dir in spec.submodule_search_locations:
            issues.extend(
                validate_atom_package(
                    package_dir,
                    module_path=module_path,
                    known_extension_names=set(),
                )
            )
    elif spec.origin is not None:
        src_file = Path(spec.origin)
        if src_file.suffix != ".py":
            return []
        if src_file.name == "__init__.py":
            issues.extend(
                validate_atom_package(
                    src_file.parent,
                    module_path=module_path,
                    known_extension_names=set(),
                )
            )
        else:
            issues.extend(
                validate_atom_file(
                    src_file,
                    module_path=module_path,
                    known_extension_names=set(),
                )
            )
            for helper in extension_helper_imports(src_file):
                issues.extend(
                    _validate_extension_helper_source(
                        helper,
                        visited=visited,
                        validate_atom_file=validate_atom_file,
                        validate_atom_package=validate_atom_package,
                        extension_helper_imports=extension_helper_imports,
                    )
                )
    return issues


def _raise_blocking_validation_issues(module_path: str, issues: list[Any]) -> None:
    blocking = [i for i in issues if i.severity == "error"]
    if not blocking:
        return
    msg = "; ".join(f"[{i.rule}] {i.message}" for i in blocking[:5])
    raise ExtensionLoadError(
        module_path,
        RuntimeError(f"contract violation: {msg}"),
    )


__all__ = [
    "ExtensionLoadError",
    "current_installing_extension",
    "install_extension",
    "load_extension",
    "validate_extension_source",
]
