# code-health: ignore-file[AM025] -- the vocabulary is read from YAML, which
# has no type until something checks it; these are that check.
"""The predicate vocabulary, and the tagger prompt generated from it.

The vocabulary is the single source of truth for the tagger's tag set: the
runtime regenerates the tagger prompt from it, so a predicate cannot exist in
the checklist without the tagger knowing what to look for. The prune helpers
are the offline half, used when a checklist revision retires items.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class PredicateDef:
    name: str
    definition: str


# -- Vocabulary I/O ------------------------------------------------------------


def load_vocabulary(path: Path) -> dict[str, PredicateDef]:
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        return {}
    vocab: dict[str, PredicateDef] = {}
    for name, entry in raw.items():
        if isinstance(entry, Mapping):
            vocab[str(name)] = PredicateDef(
                name=str(name), definition=str(entry.get("definition", ""))
            )
        elif isinstance(entry, str):
            vocab[str(name)] = PredicateDef(name=str(name), definition=entry)
    return vocab


def save_vocabulary(vocab: dict[str, PredicateDef], path: Path) -> None:
    data = {
        name: {"definition": pred.definition} for name, pred in sorted(vocab.items())
    }
    path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))


# -- Tagger prompt generation from vocabulary ----------------------------------


_TAGGER_PREAMBLE = """\
A software-engineering agent solves coding tasks by calling tools. You watch it
work, one step at a time, and record what has become true of its trajectory.

You see the whole run so far: this conversation holds every step the agent has
taken and everything you said about them. Each new message carries the steps
taken since your last reply — usually a few, sometimes one.

Steps arrive as events, not contents. A step shows the files read, the files
edited with a line count, the commands run with their exit status, and whatever
the agent said. File bodies, diffs, and command output are not shown — they do
not decide any tag below. The first message also carries the task description.

Answer each message by calling the `record` tool, with two arguments.

**phase** (exactly one, for where the agent stands at the end of these steps):
- "exploring" — reading files, searching, running commands to
  understand the codebase or problem
- "diagnosing" — analyzing a specific issue, forming a hypothesis
  about root cause based on what it read
- "implementing" — editing or creating code files
- "validating" — running tests, builds, linters, or other checks
  to verify its changes work
- "concluding" — declaring done, summarizing what was changed

**tags** (may be empty):
Report only what became true *in these steps* and that you have not already
reported. A tag you emitted earlier stays in force — never repeat it. Many
messages warrant no tag at all; an empty array is a normal answer.

Several tags below quantify over the whole run — "all executed test commands",
"only agent-authored tests", "at least two runs", "differs from a previous run".
Judge those against every step you have seen, and emit the tag as soon as it
first holds. If the run so far leaves one of them uncertain, leave it out; you
will see more steps.

The tag names, with what each means:
"""

_TAGGER_FOOTER = """
Do not restate tags you have already emitted. Emit a tag only on clear
evidence: when the run so far does not settle it, omit it.
"""


def generate_tagger_prompt(vocab: dict[str, PredicateDef]) -> str:
    """Generate the tagger system prompt from the predicate vocabulary."""
    tag_lines = []
    for name, pred in sorted(vocab.items()):
        tag_lines.append(f'- "{name}" — {pred.definition}')
    return _TAGGER_PREAMBLE + "\n".join(tag_lines) + "\n" + _TAGGER_FOOTER


# -- Prune (select step) ------------------------------------------------------


def prune_items(
    checklist_path: Path,
    fitness: dict[str, float],
    threshold: float = 0.1,
) -> list[str]:
    """Remove items below fitness threshold. Returns pruned item IDs."""
    raw = yaml.safe_load(checklist_path.read_text(encoding="utf-8"))
    kept = []
    pruned_ids = []
    for item in raw.get("items", []):
        item_id = item.get("id", "")
        score = fitness.get(item_id, 1.0)
        if score >= threshold:
            kept.append(item)
        else:
            pruned_ids.append(item_id)
    raw["items"] = kept
    checklist_path.write_text(
        yaml.dump(raw, default_flow_style=False, allow_unicode=True, width=120)
    )
    return pruned_ids


def prune_vocabulary(
    vocab: dict[str, PredicateDef],
    checklist_path: Path,
) -> dict[str, PredicateDef]:
    """Remove predicates not referenced by any checklist item."""
    raw = yaml.safe_load(checklist_path.read_text(encoding="utf-8"))
    all_triggers = " ".join(
        str(item.get("when", {}).get("trigger", "")) for item in raw.get("items", [])
    )
    return {name: pred for name, pred in vocab.items() if name in all_triggers}
