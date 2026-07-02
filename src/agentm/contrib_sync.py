"""Install bundled contrib resources into ``$AGENTM_HOME/contrib``."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from agentm.core.lib.user_config import agentm_home_dir


class SyncMode(str, Enum):
    copy = "copy"
    symlink = "symlink"


VALID_KINDS = frozenset({"scenarios", "extensions"})
DEMO_SCENARIOS = ("chatbot", "minimal", "local")
COPY_IGNORE = shutil.ignore_patterns(
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "logs",
    "*.log",
    "*.log.*",
)


@dataclass(frozen=True, slots=True)
class SyncRecord:
    kind: str
    source: str | None
    destination: str
    action: str
    reason: str | None = None


app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)


def _repo_contrib_root() -> Path | None:
    """Locate ``<checkout>/contrib`` when running from an editable checkout."""

    for parent in Path(__file__).parents:
        contrib = parent / "contrib"
        if (parent / "pyproject.toml").is_file() and contrib.is_dir():
            return contrib
    return None


def _packaged_scenarios_root() -> Path | None:
    try:
        from importlib.resources import files
    except Exception as exc:  # noqa: BLE001
        logger.debug("contrib_sync: importlib.resources unavailable: {}", exc)
        return None
    try:
        root = files("agentm.scenarios")
    except (ModuleNotFoundError, TypeError):
        return None
    try:
        concrete = Path(os.fspath(root))  # type: ignore[call-overload]
    except TypeError:
        return None
    return concrete if concrete.is_dir() else None


def _source_for_kind(kind: str, explicit_root: Path | None) -> Path | None:
    if explicit_root is not None:
        candidate = explicit_root / kind
        return candidate if candidate.is_dir() else None

    repo_root = _repo_contrib_root()
    if repo_root is not None:
        candidate = repo_root / kind
        if candidate.is_dir():
            return candidate

    if kind == "scenarios":
        return _packaged_scenarios_root()
    return None


def _packaged_scenario_dir(name: str) -> Path | None:
    root = _packaged_scenarios_root()
    if root is None:
        return None
    candidate = root / name
    return candidate if (candidate / "manifest.yaml").is_file() else None


def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def sync_contrib(
    *,
    mode: SyncMode = SyncMode.copy,
    overwrite: bool = False,
    kinds: list[str] | None = None,
    home: Path | None = None,
    source: Path | None = None,
) -> list[SyncRecord]:
    selected = kinds or sorted(VALID_KINDS)
    invalid = sorted(set(selected) - VALID_KINDS)
    if invalid:
        raise ValueError(f"unknown contrib kind(s): {', '.join(invalid)}")

    home_root = (home or agentm_home_dir()).expanduser()
    contrib_root = home_root / "contrib"
    explicit_root = source.expanduser().resolve() if source is not None else None
    records: list[SyncRecord] = []

    for kind in selected:
        destination = contrib_root / kind
        src = _source_for_kind(kind, explicit_root)
        if src is None:
            records.append(
                SyncRecord(
                    kind=kind,
                    source=None,
                    destination=str(destination),
                    action="skipped",
                    reason="source not found",
                )
            )
            continue

        if destination.exists() or destination.is_symlink():
            if not overwrite:
                records.append(
                    SyncRecord(
                        kind=kind,
                        source=str(src),
                        destination=str(destination),
                        action="skipped",
                        reason="destination exists",
                    )
                )
                continue
            _remove_existing(destination)

        destination.parent.mkdir(parents=True, exist_ok=True)
        if mode == SyncMode.symlink:
            destination.symlink_to(src, target_is_directory=True)
        else:
            shutil.copytree(src, destination, symlinks=True, ignore=COPY_IGNORE)
        records.append(
            SyncRecord(
                kind=kind,
                source=str(src),
                destination=str(destination),
                action=mode.value,
            )
        )

    return records


def sync_demo_scenarios(
    *,
    home: Path | None = None,
    overwrite: bool = False,
    names: list[str] | None = None,
) -> list[SyncRecord]:
    """Install the small packaged scenario demos into ``$AGENTM_HOME/contrib``.

    ``agentm setup`` uses this instead of syncing the full source ``contrib``
    tree so a first-run install exposes only simple examples users can edit:
    ``chatbot`` for the default full stack, ``minimal`` for the smallest
    runnable stack, and ``local`` for coding sessions.
    """

    selected = names or list(DEMO_SCENARIOS)
    home_root = (home or agentm_home_dir()).expanduser()
    scenarios_root = home_root / "contrib" / "scenarios"
    records: list[SyncRecord] = []

    for name in selected:
        destination = scenarios_root / name
        src = _packaged_scenario_dir(name)
        if src is None:
            records.append(
                SyncRecord(
                    kind="scenario",
                    source=None,
                    destination=str(destination),
                    action="skipped",
                    reason=f"demo scenario {name!r} not found",
                )
            )
            continue

        if destination.exists() or destination.is_symlink():
            if not overwrite:
                records.append(
                    SyncRecord(
                        kind="scenario",
                        source=str(src),
                        destination=str(destination),
                        action="skipped",
                        reason="destination exists",
                    )
                )
                continue
            _remove_existing(destination)

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, destination, symlinks=True, ignore=COPY_IGNORE)
        records.append(
            SyncRecord(
                kind="scenario",
                source=str(src),
                destination=str(destination),
                action="copy",
            )
        )

    return records


@app.command(name="sync")
def sync_cmd(
    mode: Annotated[
        SyncMode,
        typer.Option(
            "--mode",
            help="Install mode: copy for editable user config, symlink for local development.",
        ),
    ] = SyncMode.copy,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Replace existing ~/.agentm/contrib entries.",
        ),
    ] = False,
    kind: Annotated[
        list[str] | None,
        typer.Option(
            "--kind",
            help="Resource kind to sync: scenarios or extensions. Repeatable.",
        ),
    ] = None,
    home: Annotated[
        Path | None,
        typer.Option(
            "--home",
            help="Override AGENTM_HOME for this sync operation.",
        ),
    ] = None,
    source: Annotated[
        Path | None,
        typer.Option(
            "--source",
            help="Explicit contrib root containing scenarios/ and/or extensions/.",
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format: text or json.",
        ),
    ] = "text",
) -> None:
    """Install bundled contrib scenarios/extensions into ``~/.agentm/contrib``."""

    if output_format not in {"text", "json"}:
        raise typer.BadParameter("--format must be 'text' or 'json'")
    try:
        records = sync_contrib(
            mode=mode,
            overwrite=overwrite,
            kinds=kind,
            home=home,
            source=source,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if output_format == "json":
        print(json.dumps([asdict(record) for record in records], indent=2))
        return

    for record in records:
        source_text = f"{record.source} -> " if record.source else ""
        suffix = f" ({record.reason})" if record.reason else ""
        print(
            f"{record.action}: {record.kind}: "
            f"{source_text}{record.destination}{suffix}"
        )


__all__ = [
    "DEMO_SCENARIOS",
    "SyncMode",
    "SyncRecord",
    "sync_contrib",
    "sync_demo_scenarios",
    "app",
]
