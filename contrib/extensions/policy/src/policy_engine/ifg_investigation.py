# code-health: ignore-file[AM025] -- IFG evidence normalizes untyped tool arguments and results
"""Structural investigation evidence over the temporal IFG.

This module deliberately does not infer intent, hypothesis quality, or task
success.  It records graph/state transitions that a later diagnosis stage can
interpret with semantic context.
"""

from __future__ import annotations

import math
import posixpath
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from .source_parser import BashSegment, parse_bash_segments
from .source_semantics import analyze_bash_segment
from .types import ToolArgs


_EVIDENCE_UNCHANGED_REENTRY = "unchanged_anchor_reentry"
_EVIDENCE_ANCHOR_CHURN = "unchanged_anchor_churn"
_EVIDENCE_ARTIFACT_REPLACEMENT = "created_artifact_replacement"
_EVIDENCE_STATE_CYCLE = "unchanged_investigation_state_cycle"
_EXECUTION_ACTIONS = frozenset({"exec", "test"})


@dataclass(frozen=True, slots=True)
class InvestigationEvidence:
    """One structural investigation transition."""

    kind: str
    sequence: int
    paths: tuple[str, ...]
    related_sequences: tuple[int, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "sequence": self.sequence,
            "paths": list(self.paths),
            "related_sequences": list(self.related_sequences),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class _RepositoryArtifact:
    path: str
    first_sequence: int
    anchor_source: str
    revision: int = 0


@dataclass(slots=True)
class _CreatedArtifact:
    path: str
    created_sequence: int
    last_sequence: int
    execution_sequences: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _InvestigationPhase:
    phase_index: int
    end_sequence: int
    artifact_paths: tuple[str, ...]
    focus_state: tuple[tuple[str, int], ...]
    state_signature: tuple[tuple[str, int], ...]
    repository_generation: int

    @property
    def focus_paths(self) -> frozenset[str]:
        return frozenset(path for path, _revision in self.focus_state)


class IfgInvestigationState:
    """Collect semantic-free structural evidence with adaptive support."""

    def __init__(self, *, cwd: str | None = None) -> None:
        self._cwd = posixpath.normpath(cwd) if cwd else None
        self._canonicalize: Callable[[str], str] | None = None
        self._repository_contains: Callable[[str], bool] | None = None
        self._sequence = 0
        self._repository_generation = 0
        self._repository_artifacts: dict[str, _RepositoryArtifact] = {}
        self._created_artifacts: dict[str, _CreatedArtifact] = {}
        self._bash_support_counts: Counter[str] = Counter()
        self._bash_path_candidates: Counter[str] = Counter()
        self._last_focus_path: str | None = None
        self._last_focus_revision: dict[str, int] = {}
        self._generation_reentries = 0
        self._anchor_churn_reported = False
        self._phase_focus: set[str] = set()
        self._phases: list[_InvestigationPhase] = []
        self._seen_phase_states: dict[
            tuple[tuple[str, int], ...], _InvestigationPhase
        ] = {}
        self._last_executed_created: tuple[str, int, int] | None = None
        self._current_evidence: list[InvestigationEvidence] = []
        self._latest_evidence: dict[str, InvestigationEvidence] = {}
        self._evidence_counts: Counter[str] = Counter()

    def configure_repository(
        self,
        *,
        contains_file: Callable[[str], bool] | None,
        canonicalize: Callable[[str], str] | None = None,
    ) -> None:
        """Attach optional repository facts supplied by the live index."""

        self._repository_contains = contains_file
        self._canonicalize = canonicalize

    def record(
        self,
        tool_name: str,
        args: ToolArgs,
        result: Mapping[str, object] | None,
    ) -> None:
        self._sequence += 1
        self._current_evidence = []

        if tool_name in {"read", "edit", "write"}:
            self._record_structured_file_tool(tool_name, args, result)
        elif tool_name == "bash":
            self._record_bash(args, result)

    @property
    def became_unchanged_reentry(self) -> bool:
        return self._has_current(_EVIDENCE_UNCHANGED_REENTRY)

    @property
    def became_anchor_churn(self) -> bool:
        return self._has_current(_EVIDENCE_ANCHOR_CHURN)

    @property
    def became_artifact_replacement(self) -> bool:
        return self._has_current(_EVIDENCE_ARTIFACT_REPLACEMENT)

    @property
    def became_state_cycle(self) -> bool:
        return self._has_current(_EVIDENCE_STATE_CYCLE)

    def summary(self) -> dict[str, object]:
        latest_phase = self._phases[-1] if self._phases else None
        return {
            "sequence": self._sequence,
            "repository_artifacts": len(self._repository_artifacts),
            "created_artifacts": len(self._created_artifacts),
            "bash_supported_paths": len(self._bash_support_counts),
            "bash_support_observations": sum(self._bash_support_counts.values()),
            "bash_path_candidates": len(self._bash_path_candidates),
            "repository_generation": self._repository_generation,
            "generation_reentries": self._generation_reentries,
            "adaptive_reentry_support": self._adaptive_reentry_support(),
            "current_anchor": self._last_focus_path,
            "phase_focus": sorted(self._phase_focus),
            "completed_phases": len(self._phases),
            "latest_phase": (
                _phase_dict(latest_phase, previous=self._previous_phase())
                if latest_phase is not None
                else None
            ),
            "evidence_counts": dict(sorted(self._evidence_counts.items())),
            "current_evidence": [item.as_dict() for item in self._current_evidence],
            "latest_evidence": {
                kind: evidence.as_dict()
                for kind, evidence in sorted(self._latest_evidence.items())
            },
        }

    def churn_summary(self) -> dict[str, object]:
        """Return the compact evidence packet used for churn escalation."""

        evidence = self._latest_evidence.get(_EVIDENCE_ANCHOR_CHURN)
        return {
            "sequence": self._sequence,
            "repository_generation": self._repository_generation,
            "repository_artifacts": len(self._repository_artifacts),
            "created_artifacts": len(self._created_artifacts),
            "generation_reentries": self._generation_reentries,
            "adaptive_reentry_support": self._adaptive_reentry_support(),
            "current_anchor": self._last_focus_path,
            "phase_focus_size": len(self._phase_focus),
            "completed_phases": len(self._phases),
            "bash_support_observations": sum(self._bash_support_counts.values()),
            "bash_path_candidates": len(self._bash_path_candidates),
            "transition": evidence.as_dict() if evidence is not None else None,
        }

    def _record_structured_file_tool(
        self,
        tool_name: str,
        args: ToolArgs,
        result: Mapping[str, object] | None,
    ) -> None:
        if _result_failed(result):
            return
        raw_path = args.get("path") or args.get("file_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return
        path = self._path(raw_path)

        if tool_name == "read":
            created = self._created_artifacts.get(path)
            if created is not None:
                created.last_sequence = self._sequence
                return
            self._ensure_repository_artifact(path, anchor_source="read")
            self._touch_repository(path, mutated=False)
            return

        role = self._path_role(path)
        if tool_name == "edit" and role == "unknown":
            role = "repository"
            self._ensure_repository_artifact(path, anchor_source="edit")

        if role == "repository":
            self._ensure_repository_artifact(path, anchor_source="repository_index")
            self._touch_repository(path, mutated=True)
        else:
            self._record_created_artifact(path)

    def _record_bash(
        self,
        args: ToolArgs,
        result: Mapping[str, object] | None,
    ) -> None:
        raw_command = args.get("cmd") or args.get("command")
        if not isinstance(raw_command, str) or not raw_command.strip():
            return

        command_cwd = self._cwd
        command_succeeded = not _result_failed(result)
        for segment in parse_bash_segments(raw_command):
            if not segment.argv:
                continue
            semantics = analyze_bash_segment(segment)
            if semantics.command == "cd":
                command_cwd = self._cd_target(segment, command_cwd)
                continue

            if command_succeeded:
                for reference in semantics.path_refs:
                    path = self._path(reference.path, cwd=command_cwd)
                    self._record_bash_path(path, relation=reference.relation)

            if semantics.action_kind in _EXECUTION_ACTIONS:
                executed = self._created_artifacts_in_segment(
                    segment,
                    cwd=command_cwd,
                )
                if executed:
                    self._record_created_artifact_execution(executed)

    def _record_bash_path(self, path: str, *, relation: str) -> None:
        if path in self._repository_artifacts:
            self._bash_support_counts[path] += 1
            return

        if path in self._created_artifacts:
            self._bash_support_counts[path] += 1
            self._created_artifacts[path].last_sequence = self._sequence
            return

        self._bash_path_candidates[path] += 1
        if relation == "write" and self._path_role(path) != "repository":
            self._record_created_artifact(path)

    def _ensure_repository_artifact(self, path: str, *, anchor_source: str) -> None:
        if path in self._repository_artifacts:
            return
        self._repository_artifacts[path] = _RepositoryArtifact(
            path=path,
            first_sequence=self._sequence,
            anchor_source=anchor_source,
        )
        self._promote_bash_candidate(path)
        self._created_artifacts.pop(path, None)

    def _record_created_artifact(self, path: str) -> None:
        existing = self._created_artifacts.get(path)
        if existing is not None:
            existing.last_sequence = self._sequence
            return
        self._created_artifacts[path] = _CreatedArtifact(
            path=path,
            created_sequence=self._sequence,
            last_sequence=self._sequence,
        )
        self._promote_bash_candidate(path)

    def _promote_bash_candidate(self, path: str) -> None:
        observations = self._bash_path_candidates.pop(path, 0)
        if observations:
            self._bash_support_counts[path] += observations

    def _touch_repository(self, path: str, *, mutated: bool) -> None:
        artifact = self._repository_artifacts[path]
        if mutated:
            artifact.revision += 1
            self._repository_generation += 1
            self._generation_reentries = 0
            self._anchor_churn_reported = False

        previous_revision = self._last_focus_revision.get(path)
        if (
            not mutated
            and previous_revision is not None
            and self._last_focus_path is not None
            and self._last_focus_path != path
            and previous_revision == artifact.revision
        ):
            self._generation_reentries += 1
            metadata = {
                "revision": artifact.revision,
                "from_anchor": self._last_focus_path,
                "repository_generation": self._repository_generation,
                "generation_reentries": self._generation_reentries,
                "adaptive_support": self._adaptive_reentry_support(),
            }
            self._emit(
                InvestigationEvidence(
                    kind=_EVIDENCE_UNCHANGED_REENTRY,
                    sequence=self._sequence,
                    paths=(path,),
                    metadata=metadata,
                )
            )
            if (
                not self._anchor_churn_reported
                and self._generation_reentries > self._adaptive_reentry_support()
            ):
                self._anchor_churn_reported = True
                self._emit(
                    InvestigationEvidence(
                        kind=_EVIDENCE_ANCHOR_CHURN,
                        sequence=self._sequence,
                        paths=(path,),
                        metadata=metadata,
                    )
                )

        self._last_focus_path = path
        self._last_focus_revision[path] = artifact.revision
        self._phase_focus.add(path)

    def _record_created_artifact_execution(self, paths: Sequence[str]) -> None:
        unique_paths = tuple(dict.fromkeys(paths))
        if not unique_paths:
            return

        previous = self._last_executed_created
        if previous is not None:
            previous_path, previous_sequence, previous_generation = previous
            current_path = unique_paths[0]
            if (
                previous_path != current_path
                and previous_generation == self._repository_generation
            ):
                self._emit(
                    InvestigationEvidence(
                        kind=_EVIDENCE_ARTIFACT_REPLACEMENT,
                        sequence=self._sequence,
                        paths=(previous_path, current_path),
                        related_sequences=(previous_sequence,),
                        metadata={
                            "repository_generation": self._repository_generation,
                        },
                    )
                )

        for path in unique_paths:
            artifact = self._created_artifacts[path]
            artifact.execution_sequences.append(self._sequence)
            artifact.last_sequence = self._sequence
        self._last_executed_created = (
            unique_paths[-1],
            self._sequence,
            self._repository_generation,
        )
        self._close_phase(unique_paths)

    def _close_phase(self, artifact_paths: tuple[str, ...]) -> None:
        focus_state = tuple(
            sorted(
                (path, self._repository_artifacts[path].revision)
                for path in self._phase_focus
            )
        )
        state_signature = self._current_state_signature()
        phase = _InvestigationPhase(
            phase_index=len(self._phases),
            end_sequence=self._sequence,
            artifact_paths=artifact_paths,
            focus_state=focus_state,
            state_signature=state_signature,
            repository_generation=self._repository_generation,
        )
        self._phases.append(phase)
        self._phase_focus = set()

        if not state_signature:
            return
        previous = self._seen_phase_states.get(state_signature)
        if (
            previous is not None
            and previous.repository_generation == phase.repository_generation
        ):
            self._emit(
                InvestigationEvidence(
                    kind=_EVIDENCE_STATE_CYCLE,
                    sequence=self._sequence,
                    paths=tuple(path for path, _revision in state_signature),
                    related_sequences=(previous.end_sequence,),
                    metadata={
                        "first_phase": previous.phase_index,
                        "current_phase": phase.phase_index,
                        "previous_artifacts": list(previous.artifact_paths),
                        "current_artifacts": list(phase.artifact_paths),
                        "repository_generation": self._repository_generation,
                    },
                )
            )
        else:
            self._seen_phase_states[state_signature] = phase

    def _current_state_signature(self) -> tuple[tuple[str, int], ...]:
        path = self._last_focus_path
        if path is None:
            return ()
        artifact = self._repository_artifacts.get(path)
        if artifact is None:
            return ()
        return ((path, artifact.revision),)

    def _created_artifacts_in_segment(
        self,
        segment: BashSegment,
        *,
        cwd: str | None,
    ) -> tuple[str, ...]:
        referenced: list[str] = []
        for raw_token in segment.argv[1:]:
            token = _strip_shell_quotes(raw_token)
            if not token or token.startswith("-"):
                continue
            candidate = self._path(token, cwd=cwd)
            if candidate in self._created_artifacts:
                referenced.append(candidate)
        return tuple(dict.fromkeys(referenced))

    def _cd_target(self, segment: BashSegment, cwd: str | None) -> str | None:
        if len(segment.argv) < 2:
            return cwd
        target = _strip_shell_quotes(segment.argv[1])
        if not target:
            return cwd
        return self._path(target, cwd=cwd)

    def _path_role(self, path: str) -> str:
        if path in self._repository_artifacts:
            return "repository"
        if path in self._created_artifacts:
            return "created"
        if self._repository_contains is not None and self._repository_contains(path):
            return "repository"
        return "unknown"

    def _path(self, raw_path: str, *, cwd: str | None = None) -> str:
        path = _strip_shell_quotes(raw_path)
        base = cwd if cwd is not None else self._cwd
        if not posixpath.isabs(path) and base:
            path = posixpath.join(base, path)
        normalized = posixpath.normpath(path)
        if self._canonicalize is not None:
            return posixpath.normpath(self._canonicalize(normalized))
        return normalized

    def _emit(self, evidence: InvestigationEvidence) -> None:
        self._current_evidence.append(evidence)
        self._latest_evidence[evidence.kind] = evidence
        self._evidence_counts[evidence.kind] += 1

    def _has_current(self, kind: str) -> bool:
        return any(item.kind == kind for item in self._current_evidence)

    def _previous_phase(self) -> _InvestigationPhase | None:
        return self._phases[-2] if len(self._phases) > 1 else None

    def _adaptive_reentry_support(self) -> int:
        return max(1, math.ceil(math.sqrt(len(self._repository_artifacts))))


class InvestigationQuery:
    """DSL facade for neutral investigation evidence."""

    def __init__(self, state: IfgInvestigationState) -> None:
        self._state = state

    def became_unchanged_reentry(self) -> bool:
        return self._state.became_unchanged_reentry

    def became_anchor_churn(self) -> bool:
        return self._state.became_anchor_churn

    def became_artifact_replacement(self) -> bool:
        return self._state.became_artifact_replacement

    def became_state_cycle(self) -> bool:
        return self._state.became_state_cycle

    def summary(self) -> dict[str, object]:
        return self._state.summary()

    def churn_summary(self) -> dict[str, object]:
        return self._state.churn_summary()


def _phase_dict(
    phase: _InvestigationPhase,
    *,
    previous: _InvestigationPhase | None,
) -> dict[str, object]:
    return {
        "phase_index": phase.phase_index,
        "end_sequence": phase.end_sequence,
        "artifact_paths": list(phase.artifact_paths),
        "focus_state": [
            {"path": path, "revision": revision} for path, revision in phase.focus_state
        ],
        "state_signature": [
            {"path": path, "revision": revision}
            for path, revision in phase.state_signature
        ],
        "repository_generation": phase.repository_generation,
        "focus_relation": _focus_relation(previous, phase),
    }


def _focus_relation(
    previous: _InvestigationPhase | None,
    current: _InvestigationPhase,
) -> str | None:
    if previous is None:
        return None
    before = previous.focus_paths
    after = current.focus_paths
    if after == before:
        return "same"
    if after < before:
        return "contracted"
    if after > before:
        return "expanded"
    return "shifted"


def _result_failed(result: Mapping[str, object] | None) -> bool:
    if result is None:
        return False
    if result.get("error"):
        return True
    exit_code = result.get("exit_code")
    if exit_code is None or isinstance(exit_code, bool):
        return False
    try:
        return int(str(exit_code)) != 0
    except (TypeError, ValueError):
        return False


def _strip_shell_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "'\"":
        return stripped[1:-1]
    return stripped


__all__ = [
    "IfgInvestigationState",
    "InvestigationEvidence",
    "InvestigationQuery",
]
