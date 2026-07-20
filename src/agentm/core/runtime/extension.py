"""Extension loader — import an atom module and call its install().

Atoms receive the Session object directly.
"""

# code-health: ignore-file[AM022] -- validates dynamically imported Python plugin contracts

from __future__ import annotations

import importlib
import importlib.util
import hashlib
import inspect
import sys
import threading
import time
from collections.abc import Awaitable
from contextvars import ContextVar
from pathlib import Path
from types import ModuleType
from typing import Any

from loguru import logger
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ValidationError as PydanticValidationError

from agentm.core.abi.errors import ExtensionLoadError
from agentm.core.abi.events import ExtensionInstallEvent
from agentm.core.abi.manifest import ExtensionManifest
from agentm.core.abi.messages import thaw_json
from agentm.core.abi.session_api import (
    ExtensionSource,
    ExtensionSpec,
)
from agentm.extensions.validate import (  # code-health: ignore[AM010] -- constitution-listed contract mechanism
    ValidationIssue,
    extension_helper_imports,
    validate_atom_file,
    validate_atom_package,
)


_INSTALLING_EXTENSION: ContextVar[str | None] = ContextVar(
    "_installing_extension", default=None
)
_FILE_EXTENSION_LOAD_LOCK = threading.Lock()


def current_installing_extension() -> str:
    return _INSTALLING_EXTENSION.get() or ""


async def install_extension(
    api: Any,
    extension: ExtensionSpec | str,
    config: dict[str, Any] | None = None,
    *,
    trigger: str = "session_start",
) -> None:
    """Install one extension and emit the standard install lifecycle event."""

    spec = _coerce_extension_spec(extension, config)
    module_path = spec.module_path
    started_ns = time.perf_counter_ns()
    name = module_path.rsplit(".", 1)[-1]
    await api.bus.emit(
        ExtensionInstallEvent.CHANNEL,
        ExtensionInstallEvent(
            name=name,
            module_path=module_path,
            phase="start",
            trigger=trigger,
        ),
    )
    error: str | None = None
    try:
        result = load_extension(spec, api)
        if inspect.isawaitable(result):
            await result
        api._record_installed_extension(spec)
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
                duration_ns=time.perf_counter_ns() - started_ns,
                trigger=trigger,
                error=error,
            ),
        )


def load_extension(
    extension: ExtensionSpec | str,
    api: Any,
    config: dict[str, Any] | None = None,
    *,
    validate: bool = True,
) -> None | Awaitable[None]:
    """Load one extension source and invoke its ``install(api, config)``.

    Returns whatever ``install`` returns:
    - ``None`` for sync extensions (caller need not await).
    - An awaitable for async extensions (caller must await).

    Raises ``ExtensionLoadError`` on any failure.
    """

    spec = _coerce_extension_spec(extension, config)
    module_path = spec.module_path
    module = load_extension_module(spec, validate=validate)

    install = module.__dict__.get("install")
    if install is None or not callable(install):
        raise ExtensionLoadError(
            module_path,
            AttributeError(f"module {module_path!r} has no callable 'install' symbol"),
        )

    # Validate config via the schema declared by the atom manifest.
    resolved_config: Any = thaw_json(spec.config)
    manifest = module.__dict__.get("MANIFEST")
    if manifest is not None:
        if not isinstance(manifest, ExtensionManifest):
            raise ExtensionLoadError(
                module_path,
                TypeError("MANIFEST must be an ExtensionManifest"),
            )
        schema_cls = manifest.config_schema
        if isinstance(schema_cls, type) and issubclass(
            schema_cls,
            PydanticBaseModel,
        ):
            try:
                resolved_config = schema_cls.model_validate(resolved_config)
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


def _coerce_extension_spec(
    extension: ExtensionSpec | str,
    config: dict[str, Any] | None,
) -> ExtensionSpec:
    if isinstance(extension, ExtensionSpec):
        if config is not None:
            raise TypeError(
                "config must be carried by ExtensionSpec when a canonical "
                "extension is provided"
            )
        return extension
    if not isinstance(extension, str) or not extension:
        raise TypeError("extension must be an ExtensionSpec or module string")
    return ExtensionSpec.from_module(extension, config)


def load_extension_module(
    spec: ExtensionSpec,
    *,
    validate: bool = True,
) -> ModuleType:
    """Load and return the module identified by a canonical extension spec."""

    if not isinstance(spec, ExtensionSpec):
        raise TypeError("extension module load requires ExtensionSpec")
    if validate:
        validate_extension_source(spec.source)
    if spec.source.kind == "module":
        try:
            return importlib.import_module(spec.source.location)
        except Exception as exc:  # noqa: BLE001
            raise ExtensionLoadError(spec.module_path, exc) from exc
    return _load_file_extension_module(spec.source)


def _load_file_extension_module(source: ExtensionSource) -> ModuleType:
    module_path = source.module_name
    with _FILE_EXTENSION_LOAD_LOCK:
        existing = sys.modules.get(module_path)
        if existing is not None:
            return existing

        path = Path(source.location)
        content = _read_verified_file_source(source)
        try:
            code = compile(content, str(path), "exec")
        except Exception as exc:  # noqa: BLE001
            raise ExtensionLoadError(module_path, exc) from exc

        module = ModuleType(module_path)
        module.__file__ = str(path)
        module.__package__ = ""
        module.__loader__ = None
        module.__spec__ = importlib.util.spec_from_loader(
            module_path,
            loader=None,
            origin=str(path),
        )
        module.__dict__["__agentm_source_digest__"] = source.digest
        sys.modules[module_path] = module
        try:
            exec(code, module.__dict__)
        except Exception as exc:  # noqa: BLE001
            sys.modules.pop(module_path, None)
            raise ExtensionLoadError(module_path, exc) from exc
        return module


def _read_verified_file_source(source: ExtensionSource) -> bytes:
    path = Path(source.location)
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ExtensionLoadError(source.module_name, exc) from exc
    actual = "sha256:" + hashlib.sha256(content).hexdigest()
    if actual != source.digest:
        raise ExtensionLoadError(
            source.module_name,
            RuntimeError(
                f"extension source digest changed: {actual} != {source.digest}"
            ),
        )
    return content


def _format_config_validation_error(
    schema_cls: type[PydanticBaseModel],
    exc: PydanticValidationError,
) -> str:
    errors = exc.errors()
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


def validate_extension_source(source: ExtensionSource | str) -> None:
    """Run AST validation before importing an atom module."""

    if isinstance(source, str):
        source = ExtensionSource(kind="module", location=source)
    if not isinstance(source, ExtensionSource):
        raise TypeError(
            "extension validation requires ExtensionSource or module string"
        )
    module_path = source.module_name
    issues: list[ValidationIssue]

    if source.kind == "file":
        _read_verified_file_source(source)
        src_file = Path(source.location)
        issues = validate_atom_file(src_file)
        for helper in extension_helper_imports(src_file):
            issues.extend(
                _validate_extension_helper_source(
                    helper,
                    visited={module_path},
                )
            )
        _read_verified_file_source(source)
        _raise_blocking_validation_issues(module_path, issues)
        return

    try:
        module_spec = importlib.util.find_spec(source.location)
    except Exception as exc:  # noqa: BLE001
        raise ExtensionLoadError(module_path, exc) from exc
    if module_spec is None:
        raise ExtensionLoadError(
            module_path,
            ModuleNotFoundError(f"cannot resolve extension module {module_path!r}"),
        )

    issues = []
    visited: set[str] = {module_path}
    if module_spec.submodule_search_locations:
        for package_dir in module_spec.submodule_search_locations:
            issues.extend(validate_atom_package(package_dir))
    elif module_spec.origin is not None:
        src_file = Path(module_spec.origin)
        if src_file.suffix != ".py":
            return
        if src_file.name == "__init__.py":
            issues = validate_atom_package(src_file.parent)
        else:
            issues = validate_atom_file(src_file)
            for helper in extension_helper_imports(src_file):
                issues.extend(
                    _validate_extension_helper_source(
                        helper,
                        visited=visited,
                    )
                )
    _raise_blocking_validation_issues(module_path, issues)


def _validate_extension_helper_source(
    module_path: str,
    *,
    visited: set[str],
) -> list[ValidationIssue]:
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

    issues: list[ValidationIssue] = []
    if spec.submodule_search_locations:
        for package_dir in spec.submodule_search_locations:
            issues.extend(validate_atom_package(package_dir))
    elif spec.origin is not None:
        src_file = Path(spec.origin)
        if src_file.suffix != ".py":
            return []
        if src_file.name == "__init__.py":
            issues.extend(validate_atom_package(src_file.parent))
        else:
            issues.extend(validate_atom_file(src_file))
            for helper in extension_helper_imports(src_file):
                issues.extend(
                    _validate_extension_helper_source(
                        helper,
                        visited=visited,
                    )
                )
    return issues


def _raise_blocking_validation_issues(
    module_path: str,
    issues: list[ValidationIssue],
) -> None:
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
    "load_extension_module",
    "validate_extension_source",
]
