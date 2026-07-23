# code-health: ignore-file[AM025] -- classifies persisted JSON rows at a deserialization boundary
"""Loop-closure feature extraction over persisted policy tool events.

Implements the stage-1 feature set from ``docs/policy-anomaly-detection.md``:
coarse-grained action classes (mutate via dedicated tools, bash as search or
execution), repair-episode segmentation, and the F1-F8/Q1 features computed
from the policy SQLite tables.

No command vocabulary is required. A bash segment counts as search only when
the command schema recognizes it as a read (schema-backed classifications
carry ``confidence != 'low'``); every unknown non-control segment counts as
execution. Supersession, breadth, and oracle provenance are approximated by
three syntactic relations: template identity, token coverage between a later
passing run and an earlier failed run, and intersection of referenced paths
with the session mutation set.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

_CONTROL_FAMILIES = frozenset({"control"})
_EXEC_FAMILIES = frozenset({"exec", "test"})
_VERIFY_PURPOSE_WORDS = frozenset(
    {"test", "tests", "verify", "verification", "validate", "validation", "check"}
)


@dataclass(slots=True, frozen=True)
class ExecSegment:
    """One bash segment classified as execution."""

    head: str
    template: str
    tokens: tuple[str, ...]
    paths: tuple[str, ...]


@dataclass(slots=True)
class ExecCall:
    """A bash tool call containing at least one execution segment."""

    event_id: int
    turn: int
    exit_code: int | None
    is_error: bool
    purpose: str | None
    segments: list[ExecSegment] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        """True for a countable execution failure.

        Excluded as environment/usage probes rather than validation outcomes:
        exit 126/127 (shell "not executable" / "command not found"), and
        pytest exit 4 (pytest usage error such as a mistyped selector; the
        Paperless batch showed it firing F1 on a typo'd test-class name).
        """
        if self.exit_code is None:
            return self.is_error
        if self.exit_code in (126, 127):
            return False
        if self.exit_code == 4 and any(s.head == "pytest" for s in self.segments):
            return False
        return self.exit_code != 0

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.is_error

    @property
    def referenced_paths(self) -> tuple[str, ...]:
        return tuple(p for seg in self.segments for p in seg.paths)

    @property
    def families(self) -> tuple[tuple[str, ...], ...]:
        """Two-token command families of the exec segments (e.g. mix,test)."""
        return tuple(tuple(seg.tokens[:2]) for seg in self.segments if seg.tokens)


@dataclass(slots=True)
class Episode:
    """A maximal mutation run plus the executions that follow it."""

    start_turn: int
    execs: int = 0
    passed: int = 0


@dataclass(slots=True, frozen=True)
class Mutation:
    """A file mutation performed through a dedicated edit/write tool."""

    event_id: int
    turn: int
    path: str
    created: bool


def _json_mapping(raw: str | None) -> Mapping[str, object]:
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except ValueError:
        return {}
    return loaded if isinstance(loaded, Mapping) else {}


_WRAPPER_HEADS = ("uv", "run")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _template_tokens(template: str) -> tuple[str, ...]:
    """Tokenize a normalized template, unwrapping runner wrappers.

    ``PYTHONPATH=src python3 -m pytest a b`` and ``uv run pytest a b`` must
    both normalize to head ``pytest`` so that a failure under one wrapper can
    be superseded by a pass under another (Harbor showed the miss).
    """
    tokens = [
        tok
        for tok in template.split()
        if not tok.startswith("<") and not _ENV_ASSIGN.match(tok)
    ]
    while True:
        if tuple(tokens[:2]) == _WRAPPER_HEADS:
            del tokens[:2]
            continue
        if (
            len(tokens) >= 3
            and tokens[0] in ("python", "python3")
            and tokens[1] == "-m"
        ):
            del tokens[:2]
            continue
        break
    return tuple(tokens)


def _token_covers(pass_tok: str, fail_tok: str) -> bool:
    return (
        fail_tok == pass_tok
        or fail_tok.startswith(pass_tok)
        or fail_tok.endswith("/" + pass_tok)
    )


def _segment_supersedes(passing: ExecSegment, failed: ExecSegment) -> bool:
    if passing.head != failed.head:
        return False
    fail_tokens = failed.tokens + failed.paths
    for tok in passing.tokens + passing.paths:
        if tok == passing.head:
            continue
        if not any(_token_covers(tok, ft) for ft in fail_tokens):
            return False
    return True


def _call_supersedes(passing: ExecCall, failed: ExecCall) -> bool:
    return any(
        _segment_supersedes(p, f) for p in passing.segments for f in failed.segments
    )


_LINE_SELECTOR = re.compile(r":\d+(?::\d+)?$")


def _normalize_candidate(candidate: str) -> str | None:
    """Strip runner line selectors (``file.exs:917``); drop junk candidates."""
    if not candidate or any(ch.isspace() for ch in candidate):
        return None
    return _LINE_SELECTOR.sub("", candidate).lstrip("./")


def _path_matches(candidate: str, targets: Iterable[str]) -> bool:
    cand = _normalize_candidate(candidate)
    if cand is None:
        return False
    for target in targets:
        if target == cand or target.endswith("/" + cand) or cand.endswith("/" + target):
            return True
    return False


def _declared_verify(purpose: str | None) -> bool:
    if not purpose:
        return False
    words = {w.strip(".,;:!?()").lower() for w in purpose.split()}
    return bool(words & _VERIFY_PURPOSE_WORDS)


def _load_exec_calls(
    conn: sqlite3.Connection, session_id: str
) -> tuple[list[ExecCall], int, int, list[Mutation]]:
    """Return exec calls, bash call count, write-leak count, and mutations."""

    events = conn.execute(
        """
        SELECT id, turn, tool_name, tool_call_id, args_json, exit_code,
               processed_json
        FROM policy_tool_events
        WHERE session_id = ? AND phase = 'post'
        ORDER BY id
        """,
        (session_id,),
    ).fetchall()

    segments_by_call: dict[str, list[Mapping[str, object]]] = {}
    for row in conn.execute(
        """
        SELECT tool_call_id, action_kind, family, command, template, confidence,
               action_id
        FROM ifg_actions
        WHERE session_id = ? AND tool_call_id IS NOT NULL
        ORDER BY turn, segment_index
        """,
        (session_id,),
    ):
        segments_by_call.setdefault(row[0], []).append(
            {
                "family": row[2],
                "command": row[3] or "",
                "template": row[4] or (row[3] or ""),
                "confidence": row[5],
                "action_id": row[6],
            }
        )

    paths_by_action: dict[str, list[str]] = {}
    for row in conn.execute(
        "SELECT action_id, normalized_path FROM ifg_path_candidates"
        " WHERE session_id = ?",
        (session_id,),
    ):
        paths_by_action.setdefault(row[0], []).append(row[1])

    read_paths: set[str] = set()
    exec_calls: list[ExecCall] = []
    mutations: list[Mutation] = []
    bash_calls = 0
    write_leaks = 0

    for (
        event_id,
        turn,
        tool_name,
        tool_call_id,
        args_raw,
        exit_code,
        processed_raw,
    ) in events:
        args = _json_mapping(args_raw)
        processed = _json_mapping(processed_raw)
        purpose_value = args.get("purpose")
        purpose = purpose_value if isinstance(purpose_value, str) else None

        if tool_name in ("edit", "write"):
            path_value = args.get("path")
            if isinstance(path_value, str) and path_value:
                created = tool_name == "write" or path_value not in read_paths
                mutations.append(
                    Mutation(
                        event_id=event_id,
                        turn=turn,
                        path=path_value,
                        created=created,
                    )
                )
            continue

        if tool_name == "read":
            path_value = args.get("path")
            if isinstance(path_value, str):
                read_paths.add(path_value)
            continue

        if tool_name != "bash":
            continue

        bash_calls += 1
        segments = segments_by_call.get(tool_call_id or "", [])
        if any(seg["family"] == "write" for seg in segments):
            write_leaks += 1

        is_error = bool(processed.get("is_error"))
        call = ExecCall(
            event_id=event_id,
            turn=turn,
            exit_code=exit_code if isinstance(exit_code, int) else None,
            is_error=is_error,
            purpose=purpose,
        )
        for seg in segments:
            family = seg["family"]
            if family in _CONTROL_FAMILIES or family == "write":
                continue
            confidence = seg["confidence"]
            if family not in _EXEC_FAMILIES and confidence != "low":
                continue  # schema-recognized search/read segment
            template = seg["template"]
            action_id = seg["action_id"]
            if not isinstance(template, str) or not isinstance(action_id, str):
                continue
            tokens = _template_tokens(template)
            call.segments.append(
                ExecSegment(
                    head=tokens[0] if tokens else str(seg["command"]),
                    template=template,
                    tokens=tokens,
                    paths=tuple(paths_by_action.get(action_id, ())),
                )
            )
        if call.segments:
            exec_calls.append(call)

    return exec_calls, bash_calls, write_leaks, mutations


def extract_loop_closure(db_path: Path, session_id: str) -> Mapping[str, object]:
    """Compute the F1-F8/Q1 loop-closure feature vector for one session."""

    conn = sqlite3.connect(str(db_path))
    try:
        exec_calls, bash_calls, write_leaks, mutations = _load_exec_calls(
            conn, session_id
        )
        revert_row = conn.execute(
            "SELECT COALESCE(SUM(reverts_to_prior_hash), 0),"
            " COALESCE(SUM(write_count >= 3), 0)"
            " FROM policy_file_state WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()

    mutated_paths = {m.path for m in mutations}
    last_mutation_turn = max((m.turn for m in mutations), default=None)

    # F1: failed exec never superseded by a later passing same-or-broader run.
    unresolved: list[Mapping[str, object]] = []
    for i, call in enumerate(exec_calls):
        if not call.failed:
            continue
        superseded = any(
            later.passed and _call_supersedes(later, call)
            for later in exec_calls[i + 1 :]
        )
        if not superseded:
            unresolved.append(
                {
                    "turn": call.turn,
                    "templates": [seg.template for seg in call.segments],
                    "exit_code": call.exit_code,
                }
            )

    # F2: execution presence after the final mutation.
    post_mut_execs = (
        [
            c
            for c in exec_calls
            if last_mutation_turn is None or c.turn > last_mutation_turn
        ]
        if last_mutation_turn is not None
        else list(exec_calls)
    )
    no_exec_after_final_mutation = last_mutation_turn is not None and not post_mut_execs
    no_pass_exec_after_final_mutation = last_mutation_turn is not None and not any(
        c.passed for c in post_mut_execs
    )

    # F2r: stale family closure. A command family that failed and was later
    # closed by a pass, where the final mutation postdates that closing pass
    # and the family never ran again -- the final code state was never
    # exercised by the check that had discriminated it (Electric: mix test
    # closed at turn 96, four production edits followed, only mix format ran).
    stale_families: list[Mapping[str, object]] = []
    if last_mutation_turn is not None:
        family_fails: dict[tuple[str, ...], int] = {}
        family_passes: dict[tuple[str, ...], int] = {}
        family_referencing: set[tuple[str, ...]] = set()
        for call in exec_calls:
            call_refs = any(
                _normalize_candidate(p) is not None for p in call.referenced_paths
            )
            for fam in call.families:
                if len(fam) < 2:
                    continue  # single-token utilities (sort, jq) are not checks
                if call_refs:
                    family_referencing.add(fam)
                if call.failed:
                    family_fails[fam] = max(family_fails.get(fam, 0), call.turn)
                elif call.passed:
                    family_passes[fam] = max(family_passes.get(fam, 0), call.turn)
        for fam, fail_turn in family_fails.items():
            if fam not in family_referencing:
                continue  # never anchored to repo paths: probe noise, not a check
            pass_turn = family_passes.get(fam)
            if pass_turn is None or pass_turn < fail_turn:
                continue  # unresolved family: F1 territory, not staleness
            if pass_turn < last_mutation_turn:
                stale_families.append(
                    {
                        "family": " ".join(fam),
                        "last_pass_turn": pass_turn,
                        "last_mutation_turn": last_mutation_turn,
                    }
                )

    # F3/F4: oracle provenance of post-mutation executions, judged per
    # segment. Only segments whose command family has demonstrated
    # discrimination (failed at least once this session) speak for the
    # oracle; a formatter segment sharing a compound call with the test
    # segment must not pollute provenance (Electric's trailing
    # ``mix format ... && mix test ...`` showed the miss). When no
    # discriminating segment exists, all reference-bearing segments count.
    # When the post-mutation window has no references at all, fall back to
    # the last reference-bearing execution of the session.
    failed_families: set[tuple[str, ...]] = set()
    for call in exec_calls:
        if call.failed:
            failed_families.update(fam for fam in call.families if len(fam) >= 2)

    def _provenance_refs(calls: Sequence[ExecCall]) -> list[list[str]]:
        discriminating: list[list[str]] = []
        anything: list[list[str]] = []
        for call in calls:
            for seg in call.segments:
                refs = [p for p in seg.paths if _normalize_candidate(p) is not None]
                if not refs:
                    continue
                anything.append(refs)
                if tuple(seg.tokens[:2]) in failed_families:
                    discriminating.append(refs)
        return discriminating if discriminating else anything

    refs_by_call = _provenance_refs(post_mut_execs)
    if not refs_by_call:
        for call in reversed(exec_calls):
            fallback = _provenance_refs([call])
            if fallback:
                refs_by_call = fallback
                break
    referencing = [refs for refs in refs_by_call if refs]
    self_authored_only = bool(referencing) and all(
        all(_path_matches(p, mutated_paths) for p in refs) for refs in referencing
    )
    independent_refs = [
        p for refs in refs_by_call for p in refs if not _path_matches(p, mutated_paths)
    ]
    zero_independent_oracle = not independent_refs

    # F5/F7: declared-purpose features (null when the batch lacks purpose).
    has_purpose = any(c.purpose is not None for c in exec_calls)
    declared_verify_execs = [c for c in exec_calls if _declared_verify(c.purpose)]
    declared_verify_closure: bool | None = None
    if has_purpose:
        post_mut_verify = [
            c
            for c in declared_verify_execs
            if last_mutation_turn is None or c.turn > last_mutation_turn
        ]
        declared_verify_closure = not any(c.passed for c in post_mut_verify)

    # F6: repair episodes -- a maximal mutation run plus the executions that
    # follow it, closed when at least one of those executions passes.
    episodes: list[Episode] = []
    timeline: list[tuple[int, Mutation | ExecCall]] = sorted(
        [(m.event_id, m) for m in mutations] + [(c.event_id, c) for c in exec_calls],
        key=lambda item: item[0],
    )
    current: Episode | None = None
    saw_exec_since_mutation = False
    for _, entry in timeline:
        if isinstance(entry, Mutation):  # code-health: ignore[AM025]
            if current is None or saw_exec_since_mutation:
                current = Episode(start_turn=entry.turn)
                episodes.append(current)
                saw_exec_since_mutation = False
            continue
        if current is not None:
            current.execs += 1
            if entry.passed:
                current.passed += 1
            saw_exec_since_mutation = True
    unclosed = [e for e in episodes if not e.passed]

    return {
        "session_id": session_id,
        "counts": {
            "exec_calls": len(exec_calls),
            "bash_calls": bash_calls,
            "mutations": len(mutations),
            "mutated_paths": len(mutated_paths),
            "episodes": len(episodes),
        },
        "features": {
            "f1_terminal_unresolved": bool(unresolved),
            "f2_no_exec_after_final_mutation": no_exec_after_final_mutation,
            "f2_no_pass_exec_after_final_mutation": no_pass_exec_after_final_mutation,
            "f2_stale_family_closure": bool(stale_families),
            "f3_self_authored_oracle_only": self_authored_only,
            "f4_zero_independent_oracle": zero_independent_oracle,
            "f5_declared_verify_closure_missing": declared_verify_closure,
            "f6_unclosed_episode_ratio": (
                round(len(unclosed) / len(episodes), 3) if episodes else None
            ),
            "f6_final_episode_unclosed": (bool(episodes) and not episodes[-1].passed),
            "f8_hash_reverts": int(revert_row[0]),
            "f8_rewritten_paths": int(revert_row[1]),
            "q1_write_leak_rate": (
                round(write_leaks / bash_calls, 3) if bash_calls else 0.0
            ),
        },
        "evidence": {
            "last_mutation_turn": last_mutation_turn,
            "unresolved_failures": unresolved,
            "stale_families": stale_families,
            "post_mutation_exec_turns": [c.turn for c in post_mut_execs],
            "independent_path_refs": sorted(set(independent_refs))[:20],
            "declared_verify_exec_count": len(declared_verify_execs),
        },
    }
