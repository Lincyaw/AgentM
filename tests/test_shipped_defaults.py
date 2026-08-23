# code-health: ignore-file[AM025] -- reads pydantic field defaults of unknown type
"""Defaults that name something have to name something that is there.

A default is the configuration nobody writes down, so it is the one nobody
tries. ``goal``'s checker scenario defaulted to ``local``, which is not a
scenario -- and because the atom correctly refuses to accept a stop it could
not verify, the session did not fail. It looped: 86 turns and 123 failed
checker spawns against a goal that settles in 2, burning a real model call
every time round.
"""

from __future__ import annotations

import importlib
import pkgutil

import pydantic

import agentm.extensions.builtin as builtin
from agentm.scenarios import packaged_scenario_names


def _config_schemas() -> list[tuple[str, type[pydantic.BaseModel]]]:
    found: list[tuple[str, type[pydantic.BaseModel]]] = []
    for module in pkgutil.iter_modules(builtin.__path__):
        loaded = importlib.import_module(f"agentm.extensions.builtin.{module.name}")
        manifest = getattr(loaded, "MANIFEST", None)  # code-health: ignore[AM021]
        schema = None if manifest is None else manifest.config_schema
        if schema is not None:
            found.append((module.name, schema))
    return found


def test_every_shipped_scenario_default_resolves() -> None:
    """Any config field whose name says "scenario" must default to a real one.

    Read off the field names rather than a list kept here, so an atom that
    grows a second scenario setting is covered the day it does.
    """

    known = set(packaged_scenario_names())
    assert known, "no packaged scenarios at all"

    broken: list[str] = []
    for atom, schema in _config_schemas():
        for name, field in schema.model_fields.items():
            if "scenario" not in name:
                continue
            default = field.default
            if isinstance(default, str) and default and default not in known:
                broken.append(f"{atom}.{name}={default!r}")
    assert not broken, (
        f"defaults naming a scenario that does not exist: {broken}; known: "
        f"{sorted(known)}"
    )
