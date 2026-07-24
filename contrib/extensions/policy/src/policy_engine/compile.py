"""Compile step: decompose checklist when_notes into trigger expressions.

Part of the evolution loop:
  mine → **compile** → deploy → evaluate → select → diversify

Reads checklist.yaml, calls the compiler agent on each item's when_note,
deduplicates proposed predicates into a vocabulary, and writes back
trigger expressions + vocabulary.yaml.

The vocabulary is the single source of truth for the tagger's tag set.
When predicates change, the tagger prompt is regenerated automatically.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger


@dataclass(frozen=True, slots=True)
class PredicateDef:
    name: str
    definition: str


@dataclass(frozen=True, slots=True)
class CompiledItem:
    item_id: str
    trigger_expr: str
    new_predicates: tuple[PredicateDef, ...]
    reasoning: str


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


# -- Compiler prompt -----------------------------------------------------------


def build_compiler_prompt(when_note: str, vocabulary: dict[str, PredicateDef]) -> str:
    vocab_lines = [f"- {n}: {p.definition}" for n, p in sorted(vocabulary.items())]
    vocab_text = (
        "\n".join(vocab_lines) if vocab_lines else "(empty — propose what you need)"
    )
    return (
        f"## When_note to decompose\n\n{when_note}\n\n"
        f"## Current predicate vocabulary\n\n{vocab_text}\n"
    )


def parse_compiler_result(raw_text: str) -> CompiledItem | None:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("compile: invalid JSON from compiler")
        return None
    if not isinstance(parsed, Mapping):
        return None

    trigger_expr = str(parsed.get("trigger_expr", "always"))
    reasoning = str(parsed.get("reasoning", ""))
    new_preds = []
    for p in parsed.get("new_predicates", []):
        if isinstance(p, Mapping) and p.get("name") and p.get("definition"):
            new_preds.append(
                PredicateDef(name=str(p["name"]), definition=str(p["definition"]))
            )
    return CompiledItem(
        item_id="",
        trigger_expr=trigger_expr,
        new_predicates=tuple(new_preds),
        reasoning=reasoning,
    )


def merge_vocabulary(
    vocab: dict[str, PredicateDef], new_predicates: Sequence[PredicateDef]
) -> dict[str, PredicateDef]:
    merged = dict(vocab)
    for pred in new_predicates:
        if pred.name not in merged:
            merged[pred.name] = pred
    return merged


# -- Checklist I/O -------------------------------------------------------------


def update_checklist_triggers(checklist_path: Path, compiled: dict[str, str]) -> None:
    raw = yaml.safe_load(checklist_path.read_text(encoding="utf-8"))
    for item in raw.get("items", []):
        item_id = item.get("id", "")
        if item_id in compiled:
            when = item.get("when", {})
            when["trigger"] = compiled[item_id]
            item["when"] = when
    checklist_path.write_text(
        yaml.dump(raw, default_flow_style=False, allow_unicode=True, width=120)
    )


# -- Tagger prompt generation from vocabulary ----------------------------------


_TAGGER_PREAMBLE = """\
A software-engineering agent solves coding tasks by calling tools.
You see one step of its work and classify what happened.

What you can see in each step:

- **Assistant reasoning** (optional): the agent's thinking, plans,
  conclusions, or claims — some agents output this, some don't.
- **Tool calls**, each with:
  - read(path, offset, limit) → the file content it read
  - edit(path, old_string, new_string) → what code it changed
  - write(path, content) → the new file it created
  - bash(cmd) → the command it ran, its stdout/stderr, and exit code
- **Task description** (on the first step only): what the agent was
  asked to do, provided as context for task-level tags.

From these, you can observe: which files the agent looked at, what
code it changed, what commands it ran and whether they succeeded or
failed, what the agent claimed or concluded, and what the task asks for.

Output a JSON object with two fields:

**phase** (exactly one):
- "exploring" — reading files, searching, running commands to
  understand the codebase or problem
- "diagnosing" — analyzing a specific issue, forming a hypothesis
  about root cause based on what it read
- "implementing" — editing or creating code files
- "validating" — running tests, builds, linters, or other checks
  to verify its changes work
- "concluding" — declaring done, summarizing what was changed

**tags** (array of strings, may be empty):
Tag only what is clearly supported by this step's visible content.
If the evidence is ambiguous, omit the tag. Tags must be from this list:
"""

_TAGGER_FOOTER = """
Do not invent tags outside this list.
On the first step, the task description is included — check for
task-level tags (those describing what kind of task this is).
On later steps, focus on what the agent did and said in this step.
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
