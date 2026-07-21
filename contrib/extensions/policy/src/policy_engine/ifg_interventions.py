# code-health: ignore-file[AM025] -- IFG metrics normalize untyped tool arguments and results
# code-health: ignore-file[AM021] -- metric facade dispatches typed snapshot fields by DSL name
"""Adaptive intervention-epoch metrics for IFG policy queries."""

from __future__ import annotations

import math
import posixpath
import re
from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import dataclass, field

from .source_parser import BashSegment, parse_bash_segments
from .source_semantics import analyze_bash_segment
from .types import ToolArgs


@dataclass(frozen=True, slots=True)
class InterventionSnapshot:
    """Normalized metrics for one mutation-to-validation epoch."""

    active: bool
    start_sequence: int
    end_sequence: int
    call_count: int
    known_files_before: int
    mutation_count: int
    mutation_span_calls: int
    mutation_density: float
    distinct_target_files: int
    dominant_target_share: float
    effective_targets: float
    novel_files: int
    frontier_pressure: float
    validation_attempts: int
    failed_validations: int
    untrusted_validations: int
    calls_since_last_mutation: int
    adaptive_support: int
    adaptive_patience: int
    expanding: bool
    target_drifting: bool
    unvalidated: bool
    observed_signals: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "active": self.active,
            "start_sequence": self.start_sequence,
            "end_sequence": self.end_sequence,
            "calls": self.call_count,
            "known_files_before": self.known_files_before,
            "mutations": self.mutation_count,
            "mutation_span_calls": self.mutation_span_calls,
            "mutation_density": round(self.mutation_density, 3),
            "target_files": self.distinct_target_files,
            "dominant_target_share": round(self.dominant_target_share, 3),
            "effective_targets": round(self.effective_targets, 3),
            "novel_files": self.novel_files,
            "frontier_pressure": round(self.frontier_pressure, 3),
            "validation_attempts": self.validation_attempts,
            "failed_validations": self.failed_validations,
            "untrusted_validations": self.untrusted_validations,
            "calls_since_last_mutation": self.calls_since_last_mutation,
            "adaptive_support": self.adaptive_support,
            "adaptive_patience": self.adaptive_patience,
            "signals": [
                name
                for name, present in (
                    ("expanding", self.expanding),
                    ("target_drifting", self.target_drifting),
                    ("unvalidated", self.unvalidated),
                )
                if present
            ],
            "observed_signals": list(self.observed_signals),
        }


@dataclass(slots=True)
class _InterventionEpoch:
    start_sequence: int
    calls_before: int
    known_files_before: int
    mutation_sequences: list[int] = field(default_factory=list)
    mutation_targets: Counter[str] = field(default_factory=Counter)
    novel_files: set[str] = field(default_factory=set)
    validation_attempts: int = 0
    failed_validations: int = 0
    untrusted_validations: int = 0
    reported_signals: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _ValidationObservation:
    attempted: bool
    trusted: bool
    passed: bool


class IfgInterventionState:
    """Incrementally tracks whether exploration converges after mutation."""

    def __init__(self, max_completed_epochs: int = 32) -> None:
        self._sequence = 0
        self._known_files: set[str] = set()
        self._active: _InterventionEpoch | None = None
        self._completed: deque[InterventionSnapshot] = deque(
            maxlen=max_completed_epochs
        )
        self._became_expanding = False
        self._became_target_drifting = False
        self._became_unvalidated = False

    def record(
        self,
        tool_name: str,
        args: ToolArgs,
        result: Mapping[str, object] | None,
    ) -> None:
        self._sequence += 1
        self._clear_transitions()

        failed = _result_failed(result)
        path = _tool_path(args)
        if path is not None and tool_name in {"read", "edit", "write"} and not failed:
            is_novel = path not in self._known_files
            if tool_name in {"edit", "write"}:
                self._record_mutation(path, is_novel=is_novel)
            elif self._active is not None and is_novel:
                self._active.novel_files.add(path)
            self._known_files.add(path)

        if tool_name == "bash":
            observation = _validation_observation(
                args,
                result,
                mutation_targets=(
                    self._active.mutation_targets if self._active is not None else None
                ),
            )
            if self._active is not None and observation.attempted:
                self._active.validation_attempts += 1
                if not observation.trusted:
                    self._active.untrusted_validations += 1
                elif observation.passed:
                    self._complete_active_epoch()
                    return
                else:
                    self._active.failed_validations += 1

        self._update_transitions()

    def active_snapshot(self) -> InterventionSnapshot | None:
        if self._active is None:
            return None
        return self._snapshot(self._active, active=True)

    def last_completed_snapshot(self) -> InterventionSnapshot | None:
        return self._completed[-1] if self._completed else None

    def summary(self) -> dict[str, object]:
        snapshot = self.active_snapshot() or self.last_completed_snapshot()
        summary = snapshot.as_dict() if snapshot is not None else {"active": False}
        summary["completed_epochs"] = len(self._completed)
        return summary

    @property
    def became_expanding(self) -> bool:
        return self._became_expanding

    @property
    def became_target_drifting(self) -> bool:
        return self._became_target_drifting

    @property
    def became_unvalidated(self) -> bool:
        return self._became_unvalidated

    def _record_mutation(self, path: str, *, is_novel: bool) -> None:
        if self._active is None:
            self._active = _InterventionEpoch(
                start_sequence=self._sequence,
                calls_before=self._sequence - 1,
                known_files_before=len(self._known_files),
            )
        self._active.mutation_sequences.append(self._sequence)
        self._active.mutation_targets[path] += 1
        if is_novel:
            self._active.novel_files.add(path)

    def _complete_active_epoch(self) -> None:
        if self._active is None:
            return
        self._completed.append(self._snapshot(self._active, active=False))
        self._active = None
        self._clear_transitions()

    def _update_transitions(self) -> None:
        epoch = self._active
        if epoch is None:
            return
        snapshot = self._snapshot(epoch, active=True)
        signals = {
            "expanding": snapshot.expanding,
            "target_drifting": snapshot.target_drifting,
            "unvalidated": snapshot.unvalidated,
        }
        for name, present in signals.items():
            if not present or name in epoch.reported_signals:
                continue
            epoch.reported_signals.add(name)
            if name == "expanding":
                self._became_expanding = True
            elif name == "target_drifting":
                self._became_target_drifting = True
            else:
                self._became_unvalidated = True

    def _snapshot(
        self,
        epoch: _InterventionEpoch,
        *,
        active: bool,
    ) -> InterventionSnapshot:
        mutation_count = len(epoch.mutation_sequences)
        first_mutation = epoch.mutation_sequences[0]
        last_mutation = epoch.mutation_sequences[-1]
        mutation_span = last_mutation - first_mutation + 1
        dominant_count = max(epoch.mutation_targets.values(), default=0)
        effective_targets = _effective_target_count(
            epoch.mutation_targets,
            mutation_count,
        )
        novel_files = len(epoch.novel_files)
        adaptive_support = max(1, math.ceil(math.sqrt(epoch.known_files_before)))
        adaptive_patience = _adaptive_patience(epoch)
        calls_since_mutation = self._sequence - last_mutation
        expanding = novel_files > mutation_count and novel_files > adaptive_support
        target_drifting = (
            mutation_count >= adaptive_support
            and effective_targets > math.sqrt(mutation_count)
        )
        unvalidated = calls_since_mutation > adaptive_patience
        return InterventionSnapshot(
            active=active,
            start_sequence=epoch.start_sequence,
            end_sequence=self._sequence,
            call_count=self._sequence - epoch.start_sequence + 1,
            known_files_before=epoch.known_files_before,
            mutation_count=mutation_count,
            mutation_span_calls=mutation_span,
            mutation_density=mutation_count / mutation_span,
            distinct_target_files=len(epoch.mutation_targets),
            dominant_target_share=dominant_count / mutation_count,
            effective_targets=effective_targets,
            novel_files=novel_files,
            frontier_pressure=novel_files / mutation_count,
            validation_attempts=epoch.validation_attempts,
            failed_validations=epoch.failed_validations,
            untrusted_validations=epoch.untrusted_validations,
            calls_since_last_mutation=calls_since_mutation,
            adaptive_support=adaptive_support,
            adaptive_patience=adaptive_patience,
            expanding=expanding,
            target_drifting=target_drifting,
            unvalidated=unvalidated,
            observed_signals=tuple(sorted(epoch.reported_signals)),
        )

    def _clear_transitions(self) -> None:
        self._became_expanding = False
        self._became_target_drifting = False
        self._became_unvalidated = False


class InterventionQuery:
    """DSL query facade over the active intervention epoch."""

    def __init__(self, state: IfgInterventionState) -> None:
        self._state = state

    def active(self) -> bool:
        return self._state.active_snapshot() is not None

    def mutation_count(self) -> int:
        return _metric(self._state.active_snapshot(), "mutation_count", 0)

    def mutation_density(self) -> float:
        return _metric(self._state.active_snapshot(), "mutation_density", 0.0)

    def distinct_target_files(self) -> int:
        return _metric(self._state.active_snapshot(), "distinct_target_files", 0)

    def dominant_target_share(self) -> float:
        return _metric(self._state.active_snapshot(), "dominant_target_share", 0.0)

    def effective_targets(self) -> float:
        return _metric(self._state.active_snapshot(), "effective_targets", 0.0)

    def novel_files(self) -> int:
        return _metric(self._state.active_snapshot(), "novel_files", 0)

    def frontier_pressure(self) -> float:
        return _metric(self._state.active_snapshot(), "frontier_pressure", 0.0)

    def calls_since_last_mutation(self) -> int:
        return _metric(self._state.active_snapshot(), "calls_since_last_mutation", 0)

    def became_expanding(self) -> bool:
        return self._state.became_expanding

    def became_target_drifting(self) -> bool:
        return self._state.became_target_drifting

    def became_unvalidated(self) -> bool:
        return self._state.became_unvalidated

    def summary(self) -> dict[str, object]:
        return self._state.summary()


def _metric[T](
    snapshot: InterventionSnapshot | None,
    name: str,
    default: T,
) -> T:
    if snapshot is None:
        return default
    return getattr(snapshot, name)


def _tool_path(args: ToolArgs) -> str | None:
    raw = args.get("path") or args.get("file_path")
    if not isinstance(raw, str) or not raw:
        return None
    return posixpath.normpath(raw)


def _result_failed(result: Mapping[str, object] | None) -> bool:
    if result is None:
        return False
    if result.get("error"):
        return True
    exit_code = _integer(result.get("exit_code"))
    return exit_code is not None and exit_code != 0


def _validation_observation(
    args: ToolArgs,
    result: Mapping[str, object] | None,
    *,
    mutation_targets: Mapping[str, int] | None,
) -> _ValidationObservation:
    raw_command = args.get("cmd") or args.get("command")
    if not isinstance(raw_command, str) or not raw_command.strip():
        return _ValidationObservation(False, False, False)

    segments = parse_bash_segments(raw_command)
    test_segments = [
        segment
        for segment in segments
        if analyze_bash_segment(segment).action_kind == "test"
    ]
    if not test_segments:
        return _ValidationObservation(False, False, False)

    outer_segments = [segment for segment in segments if segment.depth == 0]
    last_outer = max(outer_segments, key=lambda segment: segment.end_byte, default=None)
    structurally_trusted = any(
        _is_direct_final_test(segment, last_outer) for segment in test_segments
    )
    related = _validation_matches_targets(test_segments, mutation_targets or {})
    exit_code = _integer(result.get("exit_code")) if result else None
    trusted = structurally_trusted and related and exit_code is not None
    return _ValidationObservation(
        attempted=True,
        trusted=trusted,
        passed=trusted and exit_code == 0 and not _result_failed(result),
    )


def _is_direct_final_test(
    segment: BashSegment,
    last_outer: BashSegment | None,
) -> bool:
    return (
        last_outer is not None
        and segment.depth == 0
        and segment.pipeline_index is None
        and segment.start_byte == last_outer.start_byte
        and segment.end_byte == last_outer.end_byte
    )


_SOURCE_SUFFIX_RE = re.compile(
    r"(?:\.d)?\.(?:py|pyi|js|jsx|mjs|cjs|ts|tsx|mts|cts)$",
    re.IGNORECASE,
)


def _validation_matches_targets(
    test_segments: list[BashSegment],
    mutation_targets: Mapping[str, int],
) -> bool:
    explicit_components = {
        component
        for segment in test_segments
        for token in segment.argv[1:]
        if (component := _source_component(token)) is not None
    }
    if not explicit_components:
        return True
    mutation_components = {
        component
        for path in mutation_targets
        if (component := _source_component(path)) is not None
    }
    return bool(explicit_components & mutation_components)


def _source_component(raw_path: str) -> str | None:
    path = raw_path.strip("'\"")
    if not _SOURCE_SUFFIX_RE.search(path):
        return None
    name = posixpath.basename(_SOURCE_SUFFIX_RE.sub("", path))
    for suffix in (".test", ".spec", "_test"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if name.startswith("test_"):
        name = name.removeprefix("test_")
    return name or None


def _effective_target_count(targets: Counter[str], total: int) -> float:
    if total <= 0:
        return 0.0
    entropy = -sum(
        (count / total) * math.log(count / total) for count in targets.values()
    )
    return math.exp(entropy)


def _adaptive_patience(epoch: _InterventionEpoch) -> int:
    exploration_scale = math.sqrt(max(epoch.calls_before, 1))
    mutation_count = len(epoch.mutation_sequences)
    mutation_cadence = 0.0
    if mutation_count > 1:
        mutation_cadence = (
            epoch.mutation_sequences[-1] - epoch.mutation_sequences[0]
        ) / (mutation_count - 1)
    return max(1, math.ceil(max(exploration_scale, mutation_cadence)))


def _integer(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


__all__ = [
    "IfgInterventionState",
    "InterventionQuery",
    "InterventionSnapshot",
]
