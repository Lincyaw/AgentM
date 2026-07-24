# code-health: ignore-file[AM025] -- tool args are untyped at the boundary
"""Trajectory signals: running counters over raw tool events.

``TrajectoryState`` is fed one tool result at a time (O(1) upkeep) and can
answer the signal questions the trigger layer asks. Each evaluator returns
evidence objects — the exact segments and paths behind a verdict — so a
firing can always be audited and a message can quote the agent's own
commands.

Calibration (2026-07-24 emission audit, 28 sessions, 4 passes):
  narrow_only        3 fires, all correct, 0 pass-disturbance after fixes
  self_authored_only 3 fires (gitea/better-auth/immich), 0 on passes;
                     the stem-match definition removed the paperless and
                     electric false emissions of the earlier proxy
  under_validation   3 fires (posthog/harbor/firezone), 0 on passes
  unresolved_red     noisy (fires on passing sessions) — critic evidence
                     only, never auto-inject
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .commands import (
    ENV_PROBE_EXITS,
    ExecRecord,
    Segment,
    is_validation,
    scope_refs,
    scope_tokens,
    selector_tokens,
    split_segments,
    supersedes,
    usage_error,
)

# Mutations observed through the file tools. CALIBRATION DEBT: bash
# write-redirects are invisible (the write-leak gap the coarse action model
# monitors as Q1); tracked in docs/policy-anomaly-detection.md.
MUTATING_TOOLS = frozenset({"write", "edit"})


def _is_test_path(path: str) -> bool:
    """Stem-pattern match, not substring: `test/vitest.config.mjs` is a
    config, not a test file (2026-07-24 audit — the substring form let a
    config token defeat the self-authored-only signal)."""

    stem = Path(path).stem.lower()
    return stem.startswith(("test_", "spec_")) or stem.endswith(
        ("_test", "_spec", ".test", ".spec", "-test", "-spec")
    )


@dataclass(slots=True, frozen=True)
class NarrowRun:
    segment: Segment
    selectors: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class NarrowGroup:
    scope: str
    runs: tuple[NarrowRun, ...]


@dataclass(slots=True, frozen=True)
class SelfAuthoredEvidence:
    edited_test_stems: tuple[str, ...]
    green_runs: tuple[Segment, ...]


@dataclass(slots=True, frozen=True)
class UnderValidationEvidence:
    mutation_count: int
    validation_count: int
    validations: tuple[Segment, ...]


@dataclass(slots=True)
class TrajectoryState:
    """Running state; feed() is O(1), evaluators walk bounded history."""

    mutated_paths: list[str] = field(default_factory=list)
    edited_test_stems: set[str] = field(default_factory=set)
    execs: list[ExecRecord] = field(default_factory=list)
    turn: int = 0

    def feed_mutation(self, path: str) -> None:
        self.mutated_paths.append(path)
        if _is_test_path(path):
            self.edited_test_stems.add(Path(path).stem)

    def feed_bash(self, raw: str, exit_code: int | None) -> None:
        self.execs.append(
            ExecRecord(
                raw=raw,
                segments=split_segments(raw),
                exit_code=exit_code,
                turn=self.turn,
            )
        )

    @property
    def scopes(self) -> frozenset[str]:
        return scope_tokens(self.mutated_paths)

    # -- validation inventory ------------------------------------------------

    def validation_segments(self) -> list[tuple[ExecRecord, Segment]]:
        return [
            (record, segment)
            for record in self.execs
            for segment in record.segments
            if is_validation(segment)
        ]

    def green_validations(self) -> list[tuple[ExecRecord, Segment]]:
        return [
            (record, segment)
            for record, segment in self.validation_segments()
            if record.exit_code == 0
        ]

    # -- signals ---------------------------------------------------------------

    def narrow_only_groups(self, *, min_runs: int) -> list[NarrowGroup]:
        """Mutated scopes whose every validation run carries selectors."""

        scopes = self.scopes
        if not scopes:
            return []
        runs_by_scope: dict[str, list[NarrowRun]] = {}
        broad_seen: set[str] = set()
        for _record, segment in self.validation_segments():
            for scope in scope_refs(segment, scopes):
                selectors = selector_tokens(segment, scope)
                if selectors:
                    runs_by_scope.setdefault(scope, []).append(
                        NarrowRun(segment=segment, selectors=selectors)
                    )
                else:
                    broad_seen.add(scope)
        return [
            NarrowGroup(scope=scope, runs=tuple(runs))
            for scope, runs in runs_by_scope.items()
            if scope not in broad_seen and len(runs) >= min_runs
        ]

    def self_authored_only(self) -> SelfAuthoredEvidence | None:
        """Every green validation exercises only agent-edited test files.

        Stem-level: a green run naming a pre-existing test file (a stem the
        agent never edited) counts as an independent anchor and clears the
        signal, whether or not the run was selector-narrowed.
        """

        if not self.edited_test_stems:
            return None
        greens = self.green_validations()
        if not greens:
            return None
        scopes = self.scopes
        for _record, segment in greens:
            refs = scope_refs(segment, scopes)
            named_test_stems = {
                Path(token).stem for token in segment.ordered if _is_test_path(token)
            }
            independent_stems = named_test_stems - self.edited_test_stems
            if independent_stems:
                return None
            if not named_test_stems and not any(
                selector_tokens(segment, scope) for scope in refs
            ):
                # Broad scope-level green: independent by construction.
                return None
        return SelfAuthoredEvidence(
            edited_test_stems=tuple(sorted(self.edited_test_stems)),
            green_runs=tuple(segment for _r, segment in greens[-4:]),
        )

    def under_validation(
        self, *, min_mutations: int = 8, ratio: float = 0.25
    ) -> UnderValidationEvidence | None:
        """Mutation-heavy, validation-light at stop time.

        Calibrated on the 2026-07-23 batch: posthog edited 31 files and ran
        3 validations; firezone 56/4; harbor 18/3. Evaluate only at the stop
        decision — mid-run edit bursts before a test pass are normal rhythm
        (latched mid-run this fired on 17/28 sessions including passes).
        """

        mutations = len(self.mutated_paths)
        validations = self.validation_segments()
        if mutations < min_mutations:
            return None
        if len(validations) >= mutations * ratio:
            return None
        return UnderValidationEvidence(
            mutation_count=mutations,
            validation_count=len(validations),
            validations=tuple(segment for _r, segment in validations[-4:]),
        )

    def unresolved_reds(self) -> list[ExecRecord]:
        """Failed validation runs never superseded by an equal-or-broader
        green. Noisy on passing sessions — critic evidence, not auto-inject."""

        scopes = self.scopes
        unresolved: list[ExecRecord] = []
        for index, record in enumerate(self.execs):
            if record.exit_code is None or record.exit_code == 0:
                continue
            if record.exit_code in ENV_PROBE_EXITS or usage_error(record):
                continue
            failed = [s for s in record.segments if is_validation(s)]
            if not failed:
                continue
            later_greens = [
                seg
                for later in self.execs[index + 1 :]
                if later.exit_code == 0
                for seg in later.segments
                if is_validation(seg)
            ]
            if not all(
                any(supersedes(green, seg, scopes) for green in later_greens)
                for seg in failed
            ):
                unresolved.append(record)
        return unresolved

    def repeated_failures(self, *, min_count: int = 2) -> list[tuple[str, int]]:
        """Same validation head+scope failing repeatedly — critic evidence."""

        counts: dict[str, int] = {}
        scopes = self.scopes
        for record in self.execs:
            if not record.exit_code or record.exit_code in ENV_PROBE_EXITS:
                continue
            for segment in record.segments:
                if not is_validation(segment):
                    continue
                key = f"{segment.head} {' '.join(sorted(scope_refs(segment, scopes)))}"
                counts[key] = counts.get(key, 0) + 1
        return [(key, n) for key, n in counts.items() if n >= min_count]
