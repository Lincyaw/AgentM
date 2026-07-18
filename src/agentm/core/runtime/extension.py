"""Extension loader — import an atom module and call its install().

Stripped to the essential load_extension() function for v2.  The v1
_ExtensionAPIImpl and its mixin hierarchy are removed; atoms now receive
the v2 Session (or a compat adapter) directly.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Awaitable
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from agentm.core.abi._v1_compat import ExtensionLoadError

_INSTALLING_EXTENSION: ContextVar[str | None] = ContextVar(
    "_installing_extension", default=None
)


def current_installing_extension() -> str:
    return _INSTALLING_EXTENSION.get() or ""


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

    if validate:
        _validate_on_load(module, module_path)

    # Auto-validate config via MANIFEST.config_schema (Pydantic model class).
    resolved_config: Any = config
    manifest = getattr(module, "MANIFEST", None)
    if manifest is not None:
        schema_cls = getattr(manifest, "config_schema", None)
        if schema_cls is not None:
            try:
                from pydantic import BaseModel as _PydanticBase
                from pydantic import ValidationError as _PydanticValidationError

                if isinstance(schema_cls, type) and issubclass(schema_cls, _PydanticBase):
                    try:
                        resolved_config = schema_cls.model_validate(config)
                    except _PydanticValidationError as exc:
                        raise ExtensionLoadError(
                            module_path,
                            ValueError(_format_config_validation_error(schema_cls, exc)),
                        ) from exc
            except ImportError:
                pass

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


def _validate_on_load(module: Any, module_path: str) -> None:
    """Run AST validation on *module*'s source before ``install`` runs."""

    try:
        from agentm.extensions.validate import validate_atom_file, validate_atom_package
    except ImportError:
        return

    src_file_str = getattr(module, "__file__", None)
    if src_file_str is None:
        return
    src_file = Path(src_file_str)

    if src_file.name == "__init__.py":
        package_dir = src_file.parent
        issues = validate_atom_package(
            package_dir,
            module_path=module_path,
            known_extension_names=set(),
        )
    else:
        issues = validate_atom_file(
            src_file,
            module_path=module_path,
            known_extension_names=set(),
        )

    blocking = [i for i in issues if i.severity == "error"]
    if blocking:
        msg = "; ".join(f"[{i.rule}] {i.message}" for i in blocking[:5])
        raise ExtensionLoadError(
            module_path,
            RuntimeError(f"contract violation: {msg}"),
        )


__all__ = [
    "ExtensionLoadError",
    "current_installing_extension",
    "load_extension",
]
