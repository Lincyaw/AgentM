# code-health: ignore-file[AM025] -- the model's answer arrives as untyped
# JSON; the isinstance checks here are where it becomes a Note.
"""Diagnoses of one repository become notes for whoever reviews it next.

The loop produces two kinds of artefact, and they are not interchangeable. A
``Candidate`` is sent to the agent mid-work and asks it to do something now. A
``Note`` is handed to a reviewer and holds for as long as the repository does.

A note tells the reviewer what its method has to account for -- that the tests
here run against a real database, so a mocked store proves nothing about a
change concerned with locking -- and not what to do.

Grouped by repository, and merged rather than appended. A repository with
fifteen notes has the same effect as one with none.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from loguru import logger

from agentm.core.abi import JsonValue

from .contracts import Diagnosis, Note, NotesPayload
from .runner import AgentRun, ResultTool, fan_out, load_stage_manifest

RESULT_TOOL = ResultTool(
    name="submit_notes",
    description=(
        "Record what a reviewer of this repository must be told. The whole "
        "set, merged, not additions to it. Call once."
    ),
    payload=NotesPayload,
)


def group_for(
    diagnoses: Sequence[Diagnosis], scope: str = "repository"
) -> dict[str, list[Diagnosis]]:
    """Diagnoses keyed by what the note will be about.

    Two scopes, and the difference is not the key -- it is what the grouping
    makes visible.

    ``repository`` groups a codebase's own failures. The task name carries it:
    ``prefect-fix-resolve-race-condition`` and ``prefect-perf-...`` are one
    repository seen twice, and what is learned from either applies to both.
    Splitting on the first segment is crude and right for this bench; a source
    that names repositories properly should say so instead.

    ``model`` groups one model's failures **across** repositories, and the
    crossing is the whole method. A trait that shows up in one codebase is
    indistinguishable from a fact about that codebase; one that shows up in
    three is about whatever they have in common, which is the model. So a model
    group drawn from a single repository is dropped here rather than left for
    the noter to be careful about.
    """
    grouped: dict[str, list[Diagnosis]] = {}
    for diagnosis in diagnoses:
        key = (
            repository_of(diagnosis)
            if scope == "repository"
            else diagnosis.model_name.strip()
        )
        if key:
            grouped.setdefault(key, []).append(diagnosis)
    if scope != "model":
        return grouped
    spread = {
        key: group
        for key, group in grouped.items()
        if len({repository_of(d) for d in group}) >= 2
    }
    for key in grouped.keys() - spread.keys():
        logger.info(
            "notes[model:{}]: only one repository in evidence; not a model trait",
            key,
        )
    return spread


def repository_of(diagnosis: Diagnosis) -> str:
    return diagnosis.task_name.split("-", 1)[0]


def build_prompt(
    subject: str,
    diagnoses: Sequence[Diagnosis],
    existing: Sequence[Note],
    *,
    scope: str = "repository",
) -> str:
    if scope == "model":
        lines = [
            f"# Model: {subject}",
            "",
            (
                "These failures come from different repositories. Whatever "
                "they have in common is not a fact about any codebase -- it "
                "is how this model works. Write only what survives the "
                "crossing; anything you could only say about one of them "
                "belongs to that repository."
            ),
            "",
        ]
    else:
        lines = [f"# Repository: {subject}", ""]
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
        if diagnosis.unasked_question:
            lines += [
                f"The question nobody asked: {diagnosis.unasked_question}",
                "",
            ]
        if diagnosis.discriminating_answer:
            lines += [
                f"What asking it would have shown: {diagnosis.discriminating_answer}",
                "",
            ]
        if diagnosis.lesson:
            lines += [f"What was thought to carry: {diagnosis.lesson}", ""]
        if diagnosis.cause_class:
            lines += [f"Cause class: {diagnosis.cause_class}", ""]

    if scope == "model":
        lines.append(
            "Return the merged set. A note that names a file, a framework or a "
            "convention from one of these repositories is a repository note, "
            "however true; what belongs here is what this model does wherever "
            "it works."
        )
    else:
        lines.append(
            "Return the merged set. A note that would fit an unrelated "
            "repository unchanged does not belong here; a note about one of "
            "these defects rather than about the condition that hid it does "
            "not either.\n\n"
            "The unasked questions are your best material. They are phrased "
            "without hindsight on purpose, so a reviewer can ask one before "
            "knowing the answer -- which is what a reviewer needs and a "
            "statement of the defect is not. Where one generalises past the "
            "case it came from, a note that hands it over is worth more than "
            "a note that hands over what it revealed."
        )
    return "\n".join(lines)


async def notes_for(
    subject: str,
    diagnoses: Sequence[Diagnosis],
    existing: Sequence[Note] = (),
    *,
    scope: str = "repository",
    provider: str = "",
    user_config: str = "",
    agent: AgentRun | None = None,
) -> list[Note]:
    run = agent or AgentRun(
        manifest=load_stage_manifest("noter"),
        result_tool=RESULT_TOOL,
        provider=provider,
        user_config=user_config,
    )
    payload = await run.run(build_prompt(subject, diagnoses, existing, scope=scope))
    if payload is None:
        return []

    dropped = payload.get("dropped")
    if isinstance(dropped, str) and dropped.strip():
        logger.info("notes[{}:{}]: dropped -- {}", scope, subject, dropped.strip())

    raw = payload.get("notes")
    cases = tuple(diagnosis.case_id for diagnosis in diagnoses)
    kept: list[Note] = []
    entries: list[JsonValue] = raw if isinstance(raw, list) else []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        note = Note(
            note_id=uuid.uuid4().hex[:10],
            scope=scope,
            subject=subject,
            from_repositories=tuple({repository_of(d) for d in diagnoses}),
            situation=_text(entry, "situation"),
            without_it=_text(entry, "without_it"),
            from_cases=cases,
        )
        rejection = note.rejection()
        if rejection:
            logger.info("notes[{}:{}]: rejected -- {}", scope, subject, rejection)
            continue
        kept.append(note)

    logger.info(
        "notes[{}:{}]: {} note(s) from {} diagnoses",
        scope,
        subject,
        len(kept),
        len(diagnoses),
    )
    return kept


async def notes(
    diagnoses: Sequence[Diagnosis],
    existing: Sequence[Note] = (),
    *,
    scope: str = "repository",
    provider: str = "",
    user_config: str = "",
    concurrency: int = 4,
) -> list[Note]:
    """Notes for every repository these diagnoses touch."""

    grouped = group_for(diagnoses, scope)
    if not grouped:
        return []
    held: dict[str, list[Note]] = {}
    for note in existing:
        held.setdefault(note.subject, []).append(note)

    async def one(item: tuple[str, list[Diagnosis]]) -> list[Note]:
        subject, group = item
        return await notes_for(
            subject,
            group,
            held.get(subject, []),
            scope=scope,
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


def render(notes_for_one: Sequence[Note]) -> str:
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
    "group_for",
    "notes",
    "notes_for",
    "render",
    "repository_of",
]


def apply_notes(all_notes: Sequence[Note], task_root: Path) -> list[Path]:
    """Write each repository's notes where a review of it will find them.

    Beside the task, as ``review-notes.md``. That is the one place both halves
    already agree on: the bench adapter names it when it starts a run, and the
    review tool reads whatever it is handed. Nothing else has to know.

    Every task belonging to a repository gets the same file, because a note is
    about the repository and the next attempt may be at any of its tasks.
    """
    grouped: dict[str, list[Note]] = {}
    for note in all_notes:
        # Repository scope only. A model note is about the model wherever it
        # works, so there is no one repository's tasks to sit beside; carrying
        # it to a reviewer is a different delivery and not this function's.
        if note.scope == "repository" and note.subject:
            grouped.setdefault(note.subject, []).append(note)

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
