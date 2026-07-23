# code-health: ignore-file[AM025] -- event payloads and checklist YAML are untyped at the boundary
"""``checklist_watch`` atom -- runtime checklist critique, form one.

Watches the live trajectory with structural triggers compiled from the
mined checklist and, when one fires, injects the instantiated check
question through the loop's ``Inject`` action. The nudge quotes the
agent's own commands; it never carries task answers.

Self-contained in state: subscribes to raw tool/decide events and keeps
in-memory session state, independent of the engine's persistence and IFG
pipeline. It does reuse the package's pure helpers (``loader`` path
resolution, ``loop_closure`` command normalization) so the live watcher
and the offline extractor share one definition of "same command" and its
field-bought lessons (wrapper unwrapping, pytest exit 4).

Two built-in triggers (the two proven in the turborepo rescue,
2026-07-23):

- ``narrow_only_validation``: the mutated scope has been validated at
  least ``min_narrow_runs`` times, every such run strictly extends a
  common core of tokens, and the core itself never ran. Fires mid-run.
- ``unresolved_red_at_stop``: the loop is about to stop while some failed
  execution was never superseded by an equal-or-broader green run. Fires
  on the stop decision.

Both are recall-oriented; wording stays practice-class. Trigger evaluation
is pure token-set algebra over the agent's own argv history -- no runner
enumeration. The only lexical heuristic (a ``test`` token marks a run as
validation) is monitored, not trusted, mirroring the purpose-lexicon
stance of the anomaly-detection design.
"""

from __future__ import annotations

import os
import shlex
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
    TextContent,
    UserMessage,
)
from agentm.core.abi.events import (
    DecideEvent,
    Inject,
    LoopAction,
    Stop,
    ToolResultEvent,
)
from agentm.extensions import ExtensionManifest

from .loader import resolve_policy_path
from .loop_closure import _template_tokens as _unwrap_tokens


class ChecklistWatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    checklist: str = "package:checklist_watch.yaml"
    max_injections: int = 3
    min_narrow_runs: int = 2


MANIFEST = ExtensionManifest(
    name="checklist_watch",
    description=(
        "Structural checklist triggers over the live trajectory; injects "
        "instantiated check questions via the loop Inject action."
    ),
    registers=(),
    config_schema=ChecklistWatchConfig,
    priority=AtomInstallPriority.POLICY,
)

# CALIBRATION DEBT (tracked in docs/policy-anomaly-detection.md, "Runtime
# watcher heuristics"): the two constants below are judgment calls, not
# calibrated values, and each has a known blind spot.
#
# _MUTATING_TOOLS binds the builtin file_tools names; mutations made via
# bash write-redirects are invisible to the watcher (the write-leak gap
# the coarse action model already monitors as Q1).
_MUTATING_TOOLS = frozenset({"write", "edit"})
# _GENERIC_SEGMENTS is a hand-picked stoplist for scope extraction; a
# repository whose real package is literally named e.g. ``lib`` loses its
# scope token. Replacement plan: derive genericness from cross-repo path
# frequency once the calibration corpus is large enough.
_GENERIC_SEGMENTS = frozenset(
    {"src", "lib", "test", "tests", "crates", "packages", "apps", "internal", "pkg"}
)
# Measured exclusions, not guesses: 126/127 removed the Prefect/Electric
# F1 false fires (2026-07-22 calibration); pytest exit 4 is handled in
# find_unresolved_reds (Paperless lesson, mirrors loop_closure).
_ENV_PROBE_EXITS = frozenset({126, 127})


@dataclass(slots=True)
class _ChecklistItem:
    item_id: str
    dimension: str
    check: str
    advice: str


@dataclass(slots=True)
class _Segment:
    """One shell segment of a bash call (split on &&, ;, ||, |).

    ``norm`` holds the wrapper-unwrapped tokens (shared ``loop_closure``
    normalization), used for supersession comparison so a failure under
    one wrapper can be superseded by a pass under another.
    """

    ordered: tuple[str, ...]
    tokens: frozenset[str]
    norm: tuple[str, ...]

    @property
    def raw(self) -> str:
        return " ".join(self.ordered)


@dataclass(slots=True)
class _ExecRecord:
    raw: str
    segments: tuple[_Segment, ...]
    exit_code: int | None


_CONNECTORS = frozenset({"&&", ";", "||", "|"})


def split_segments(raw: str) -> tuple[_Segment, ...]:
    """Split a shell command into logical segments; drop bare `cd` hops."""

    try:
        ordered = tuple(shlex.split(raw))
    except ValueError:
        ordered = tuple(raw.split())
    segments: list[_Segment] = []
    current: list[str] = []
    for token in (*ordered, "&&"):
        if token in _CONNECTORS:
            if current and current[0] != "cd":
                segments.append(
                    _Segment(
                        ordered=tuple(current),
                        tokens=frozenset(current),
                        norm=_unwrap_tokens(" ".join(current)),
                    )
                )
            current = []
        else:
            current.append(token)
    return tuple(segments)


@dataclass(slots=True)
class _Firing:
    item: _ChecklistItem
    facts: list[str]
    suggestion: str


def _env_enabled() -> bool:
    value = os.environ.get("AGENTM_CHECKLIST_WATCH_ENABLED", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_items(config_ref: str) -> dict[str, _ChecklistItem]:
    path = resolve_policy_path(config_ref, cwd=Path.cwd())
    if path is None:
        logger.error("checklist_watch: checklist not found: {}", config_ref)
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.error("checklist_watch: cannot load {}: {}", path, exc)
        return {}
    items: dict[str, _ChecklistItem] = {}
    for entry in (raw or {}).get("items", []):
        if not isinstance(entry, Mapping):
            continue
        item = _ChecklistItem(
            item_id=str(entry.get("id", "")),
            dimension=str(entry.get("dimension", "")),
            check=" ".join(str(entry.get("check", "")).split()),
            advice=" ".join(str(entry.get("advice", "")).split()),
        )
        if item.item_id:
            items[item.item_id] = item
    return items


def _scope_tokens(mutated_paths: list[str]) -> frozenset[str]:
    """Candidate scope names derived from mutated paths, no manifest needed."""

    tokens: set[str] = set()
    for raw_path in mutated_paths:
        parts = Path(raw_path).parts
        for part in parts[:-1]:
            if part and part not in _GENERIC_SEGMENTS and not part.startswith("/"):
                tokens.add(part)
        stem = Path(raw_path).stem
        if stem:
            tokens.add(stem)
    return frozenset(tokens)


def _looks_like_validation(segment: _Segment) -> bool:
    """Monitored heuristic: one word, not a runner enumeration."""

    return any("test" in token for token in segment.ordered[:4])


def _selector_tokens(segment: _Segment, scope: str) -> tuple[str, ...]:
    """Bare positional tokens that narrow a validation segment.

    A selector is a non-flag, non-connector token past the head pair that
    is neither the scope name nor an option value (does not follow a
    ``-``-prefixed token). Pure token shape; no runner knowledge.
    """

    selectors: list[str] = []
    previous = ""
    for position, token in enumerate(segment.ordered):
        is_candidate = (
            position >= 2
            and not token.startswith("-")
            and "=" not in token
            and token != scope
            and scope not in token
            and not previous.startswith("-")
        )
        if is_candidate:
            selectors.append(token)
        previous = token
    return tuple(selectors)


@dataclass(slots=True)
class _NarrowRun:
    segment: _Segment
    selectors: tuple[str, ...]


@dataclass(slots=True)
class _NarrowGroup:
    scope: str
    runs: list[_NarrowRun] = field(default_factory=list)


def find_narrow_only_groups(
    execs: list[_ExecRecord],
    mutated_paths: list[str],
    *,
    min_runs: int,
) -> list[_NarrowGroup]:
    """Mutated scopes whose every validation segment carries selectors.

    Fires for a scope when at least ``min_runs`` validation segments
    reference it, all of them narrowed by selector tokens, and no
    selector-free segment referencing the scope exists anywhere in the
    session.
    """

    scopes = _scope_tokens(mutated_paths)
    if not scopes:
        return []
    groups: dict[str, _NarrowGroup] = {}
    broad_seen: set[str] = set()
    for record in execs:
        for segment in record.segments:
            if not _looks_like_validation(segment):
                continue
            for scope in scopes & segment.tokens:
                selectors = _selector_tokens(segment, scope)
                if selectors:
                    groups.setdefault(scope, _NarrowGroup(scope=scope)).runs.append(
                        _NarrowRun(segment=segment, selectors=selectors)
                    )
                else:
                    broad_seen.add(scope)
    return [
        group
        for scope, group in groups.items()
        if scope not in broad_seen and len(group.runs) >= min_runs
    ]


def _pytest_usage_error(record: _ExecRecord) -> bool:
    """Pytest exit 4 is a mistyped selector, not a red suite (Paperless
    lesson; mirrors ``loop_closure.ExecCall.failed``)."""

    return record.exit_code == 4 and any(
        segment.norm[:1] == ("pytest",) for segment in record.segments
    )


def find_unresolved_reds(execs: list[_ExecRecord]) -> list[_ExecRecord]:
    """Failed validation runs never superseded by an equal-or-broader green.

    Segment-level over wrapper-unwrapped tokens: a later green validation
    segment supersedes a failed one when its normalized tokens are a
    subset (equal or broader scope), so ``uv run mix test`` green covers a
    plain ``mix test`` red. Only validation-shaped segments count on both
    sides; a failed grep is not a red light.
    """

    unresolved: list[_ExecRecord] = []
    for index, record in enumerate(execs):
        if record.exit_code is None or record.exit_code == 0:
            continue
        if record.exit_code in _ENV_PROBE_EXITS or _pytest_usage_error(record):
            continue
        failed_segments = [
            segment for segment in record.segments if _looks_like_validation(segment)
        ]
        if not failed_segments:
            continue
        later_greens = [
            later_segment
            for later in execs[index + 1 :]
            if later.exit_code == 0
            for later_segment in later.segments
            if _looks_like_validation(later_segment)
        ]
        superseded = all(
            any(
                frozenset(green.norm) <= frozenset(segment.norm)
                for green in later_greens
            )
            for segment in failed_segments
        )
        if not superseded:
            unresolved.append(record)
    return unresolved


def _clip_cmd(raw: str, limit: int = 120) -> str:
    collapsed = " ".join(raw.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3] + "..."


def _render_message(firing: _Firing) -> str:
    lines = [
        f"Process check from the validation monitor ({firing.item.dimension}):",
        firing.item.check,
        "",
        "Observed in this session:",
    ]
    lines.extend(f"- {fact}" for fact in firing.facts)
    lines.append("")
    lines.append(f"{firing.item.advice} {firing.suggestion}".strip())
    return "\n".join(lines)


class _ChecklistWatchRuntime:
    def __init__(self, api: AtomAPI, config: ChecklistWatchConfig) -> None:
        self._api = api
        self._config = config
        self._items = _load_items(config.checklist)
        self._mutated_paths: list[str] = []
        self._execs: list[_ExecRecord] = []
        self._fired_items: set[str] = set()
        self._injections = 0
        self._nudged_scopes: set[str] = set()

    def install(self) -> None:
        if not (self._config.enabled or _env_enabled()):
            # Off by default so un-intervened baseline batches stay clean;
            # enable per run via AGENTM_CHECKLIST_WATCH_ENABLED=true (read
            # here directly: the runtime has no generic env-config layer).
            logger.info("checklist_watch: disabled (baseline mode)")
            return
        if not self._items:
            logger.warning("checklist_watch: no items loaded; atom is inert")
            return
        self._api.on(ToolResultEvent.CHANNEL, self._on_tool_result)
        self._api.on(DecideEvent.CHANNEL, self._on_decide)
        logger.info(
            "checklist_watch: active with {} item(s), budget {}",
            len(self._items),
            self._config.max_injections,
        )

    # -- state tracking ----------------------------------------------------

    def _on_tool_result(self, event: ToolResultEvent) -> None:
        if event.result is not None and event.result.is_error:
            return
        if event.tool_name in _MUTATING_TOOLS:
            path = event.args.get("path") or event.args.get("file_path")
            if isinstance(path, str) and path:
                self._mutated_paths.append(path)
            return
        if event.tool_name == "bash":
            raw = event.args.get("cmd")
            if not isinstance(raw, str) or not raw.strip():
                return
            record = _ExecRecord(
                raw=raw,
                segments=split_segments(raw),
                exit_code=event.exit_code,
            )
            self._execs.append(record)
            self._log_compliance(record)

    def _log_compliance(self, record: _ExecRecord) -> None:
        # Telemetry only (actuator read-back for the rescue experiments):
        # never influences a decision. Escalation on non-compliance is the
        # planned form-two upgrade path.
        for scope in self._nudged_scopes:
            for segment in record.segments:
                if (
                    scope in segment.tokens
                    and _looks_like_validation(segment)
                    and not _selector_tokens(segment, scope)
                ):
                    logger.info(
                        "checklist_watch: compliance observed for scope {}: {}",
                        scope,
                        _clip_cmd(segment.raw),
                    )

    # -- decision ----------------------------------------------------------

    def _on_decide(self, event: DecideEvent) -> LoopAction | None:
        if self._injections >= self._config.max_injections:
            return None
        firing = self._evaluate_narrow_only()
        if firing is None and isinstance(event.observation.default_action, Stop):
            firing = self._evaluate_unresolved_red()
        if firing is None:
            return None
        self._fired_items.add(firing.item.item_id)
        self._injections += 1
        message = _render_message(firing)
        logger.info(
            "checklist_watch: injecting {} ({} chars, injection {}/{})",
            firing.item.item_id,
            len(message),
            self._injections,
            self._config.max_injections,
        )
        return Inject(
            messages=(
                UserMessage(
                    role="user",
                    content=[TextContent(type="text", text=message)],
                    timestamp=time.time(),
                ),
            )
        )

    def _evaluate_narrow_only(self) -> _Firing | None:
        item = self._items.get("narrow_only_validation")
        if item is None or item.item_id in self._fired_items:
            return None
        groups = find_narrow_only_groups(
            self._execs,
            self._mutated_paths,
            min_runs=self._config.min_narrow_runs,
        )
        if not groups:
            return None
        facts: list[str] = []
        for group in groups[:3]:
            self._nudged_scopes.add(group.scope)
            for run in group.runs[-2:]:
                facts.append(
                    f"`{_clip_cmd(run.segment.raw)}` "
                    f"(narrowing tokens: {', '.join(run.selectors)})"
                )
        shortest = min(
            (run for group in groups for run in group.runs),
            key=lambda run: len(run.segment.ordered),
        )
        suggestion = (
            f"For example, re-run `{_clip_cmd(shortest.segment.raw)}` without "
            f"{', '.join(shortest.selectors)}."
        )
        return _Firing(item=item, facts=facts, suggestion=suggestion)

    def _evaluate_unresolved_red(self) -> _Firing | None:
        item = self._items.get("unresolved_red_at_stop")
        if item is None or item.item_id in self._fired_items:
            return None
        unresolved = find_unresolved_reds(self._execs)
        if not unresolved:
            return None
        facts = [
            f"`{_clip_cmd(record.raw)}` (exit {record.exit_code}, never superseded)"
            for record in unresolved[-3:]
        ]
        return _Firing(item=item, facts=facts, suggestion="")


def install(api: AtomAPI, config: ChecklistWatchConfig) -> None:
    _ChecklistWatchRuntime(api, config).install()
