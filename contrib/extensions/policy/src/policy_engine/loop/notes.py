# code-health: ignore-file[AM025] -- the model's answer arrives as untyped
# JSON; the isinstance checks here are where it becomes a RepositoryNote.
"""Diagnoses of one repository become notes for whoever reviews it next.

The stage exists because of a measurement. Twenty-one checks written for the
agent doing the work moved no graded score. One note written for its reviewer --
that the tests here run against a real database, so a mocked store proves
nothing about a change concerned with locking -- changed the reviewer's method on
the first attempt: it stood up a real Postgres rather than modelling the
interleaving it needed, and found a defect it had previously only argued for.

So the loop produces two kinds of artefact, and they are not interchangeable. A
``Candidate`` is sent to the agent mid-work and asks it to do something now. A
``RepositoryNote`` is handed to a reviewer and holds for as long as the
repository does.

Grouped by repository, and merged rather than appended. A repository with
fifteen notes has the same effect as one with none.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from loguru import logger

from agentm.core.abi import JsonValue

from .contracts import Diagnosis, RepositoryNote
from .runner import AgentRun, ResultTool, fan_out, load_stage_manifest

RESULT_TOOL = ResultTool(
    name="submit_notes",
    description=(
        "Record what a reviewer of this repository must be told. The whole "
        "set, merged, not additions to it. Call once."
    ),
    parameters={  # code-health: ignore[AM011]
        "type": "object",
        "properties": {
            "notes": {
                "type": "array",
                "description": "The merged set. Empty is a valid answer.",
                "items": {
                    "type": "object",
                    "properties": {
                        "situation": {
                            "type": "string",
                            "description": "What to put the change into, "
                            "reachable from a checkout: something to run, a "
                            "service to start, a second process to introduce.",
                        },
                        "without_it": {
                            "type": "string",
                            "description": "What a reviewer who did not know "
                            "this would conclude, and why that is wrong.",
                        },
                    },
                    "required": ("situation", "without_it"),
                },
            },
            "dropped": {
                "type": "string",
                "description": "Any note you removed from the existing set, "
                "and why the evidence no longer supports it.",
            },
        },
        "required": ("notes",),
    },
)


def group_by_repository(diagnoses: Sequence[Diagnosis]) -> dict[str, list[Diagnosis]]:
    """Diagnoses keyed by the repository they came from.

    The task name carries it: ``prefect-fix-resolve-race-condition`` and
    ``prefect-perf-...`` are the same repository seen twice, and a note learned
    from one applies to the other. Splitting on the first segment is crude and
    right for this bench; a source that names repositories properly should say
    so instead of relying on it.
    """
    grouped: dict[str, list[Diagnosis]] = {}
    for diagnosis in diagnoses:
        repository = diagnosis.task_name.split("-", 1)[0]
        if repository:
            grouped.setdefault(repository, []).append(diagnosis)
    return grouped


def build_prompt(
    repository: str,
    diagnoses: Sequence[Diagnosis],
    existing: Sequence[RepositoryNote],
) -> str:
    lines = [f"# Repository: {repository}", ""]
    if existing:
        lines += ["## Notes it already has", ""]
        for note in existing:
            lines += [f"- {note.situation}", f"  Otherwise: {note.without_it}", ""]
    else:
        lines += ["## Notes it already has", "", "(none)", ""]

    lines += [f"## Diagnoses from this repository ({len(diagnoses)})", ""]
    for diagnosis in diagnoses:
        lines += [f"### {diagnosis.case_id}", ""]
        if diagnosis.required_behaviour:
            lines += [f"Grading required: {diagnosis.required_behaviour}", ""]
        if diagnosis.agent_idea:
            lines += [f"The agent built: {diagnosis.agent_idea}", ""]
        if diagnosis.mechanism:
            lines += [f"Why that is wrong: {diagnosis.mechanism}", ""]
        if diagnosis.lesson:
            lines += [f"What was thought to carry: {diagnosis.lesson}", ""]
        if diagnosis.cause_class:
            lines += [f"Cause class: {diagnosis.cause_class}", ""]

    lines.append(
        "Return the merged set. A note that would fit an unrelated repository "
        "unchanged does not belong here; a note about one of these defects "
        "rather than about the condition that hid it does not either."
    )
    return "\n".join(lines)


async def notes_for(
    repository: str,
    diagnoses: Sequence[Diagnosis],
    existing: Sequence[RepositoryNote] = (),
    *,
    provider: str = "",
    user_config: str = "",
    agent: AgentRun | None = None,
) -> list[RepositoryNote]:
    run = agent or AgentRun(
        manifest=load_stage_manifest("noter"),
        result_tool=RESULT_TOOL,
        provider=provider,
        user_config=user_config,
    )
    payload = await run.run(build_prompt(repository, diagnoses, existing))
    if payload is None:
        return []

    dropped = payload.get("dropped")
    if isinstance(dropped, str) and dropped.strip():
        logger.info("notes[{}]: dropped -- {}", repository, dropped.strip())

    raw = payload.get("notes")
    cases = tuple(diagnosis.case_id for diagnosis in diagnoses)
    kept: list[RepositoryNote] = []
    entries: list[JsonValue] = raw if isinstance(raw, list) else []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        note = RepositoryNote(
            note_id=uuid.uuid4().hex[:10],
            repository=repository,
            situation=_text(entry, "situation"),
            without_it=_text(entry, "without_it"),
            from_cases=cases,
        )
        rejection = note.rejection()
        if rejection:
            logger.info("notes[{}]: rejected -- {}", repository, rejection)
            continue
        kept.append(note)

    logger.info(
        "notes[{}]: {} note(s) from {} diagnoses", repository, len(kept), len(diagnoses)
    )
    return kept


async def notes(
    diagnoses: Sequence[Diagnosis],
    existing: Sequence[RepositoryNote] = (),
    *,
    provider: str = "",
    user_config: str = "",
    concurrency: int = 4,
) -> list[RepositoryNote]:
    """Notes for every repository these diagnoses touch."""

    grouped = group_by_repository(diagnoses)
    if not grouped:
        return []
    held: dict[str, list[RepositoryNote]] = {}
    for note in existing:
        held.setdefault(note.repository, []).append(note)

    async def one(item: tuple[str, list[Diagnosis]]) -> list[RepositoryNote]:
        repository, group = item
        return await notes_for(
            repository,
            group,
            held.get(repository, []),
            provider=provider,
            user_config=user_config,
        )

    produced = await fan_out(
        sorted(grouped.items()),
        one,
        concurrency=concurrency,
        label="notes",
        noun="repositor(ies)",
    )
    return [note for batch in produced if batch for note in batch]


def render(notes_for_one: Sequence[RepositoryNote]) -> str:
    """The notes as a reviewer receives them: prose, no scaffolding.

    Written as the reviewer's own reading rather than as a record, because a
    numbered list of findings reads as somebody else's homework and gets
    skimmed.
    """
    parts = []
    for note in notes_for_one:
        body = note.situation.strip()
        if note.without_it.strip():
            body = f"{body}\n\n{note.without_it.strip()}"
        parts.append(body)
    return "\n\n".join(parts) + "\n" if parts else ""


def _text(raw: Mapping[str, JsonValue], key: str) -> str:
    value = raw.get(key)
    return value.strip() if isinstance(value, str) else ""


__all__ = [
    "apply_notes",
    "build_prompt",
    "group_by_repository",
    "notes",
    "notes_for",
    "render",
]


def apply_notes(all_notes: Sequence[RepositoryNote], task_root: Path) -> list[Path]:
    """Write each repository's notes where a review of it will find them.

    Beside the task, as ``review-notes.md``. That is the one place both halves
    already agree on: the bench adapter names it when it starts a run, and the
    review tool reads whatever it is handed. Nothing else has to know.

    Every task belonging to a repository gets the same file, because a note is
    about the repository and the next attempt may be at any of its tasks.
    """
    grouped: dict[str, list[RepositoryNote]] = {}
    for note in all_notes:
        grouped.setdefault(note.repository, []).append(note)

    written: list[Path] = []
    for repository, group in sorted(grouped.items()):
        body = render(group)
        if not body:
            continue
        for task_dir in sorted(task_root.glob(f"{repository}-*")):
            if not task_dir.is_dir():
                continue
            path = task_dir / "review-notes.md"
            path.write_text(body, encoding="utf-8")
            written.append(path)
    logger.info("notes: applied to {} task(s) under {}", len(written), task_root)
    return written
