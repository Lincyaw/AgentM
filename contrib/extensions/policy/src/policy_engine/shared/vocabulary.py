# code-health: ignore-file[AM025] -- the vocabulary is read from YAML, which
# has no type until something checks it; these are that check.
"""The predicate vocabulary: what a trigger expression may name.

Shared because both halves hold an end of it. The runtime generates the tagger
prompt from it and evaluates trigger expressions over its names; the loop's
compile stage shows it to the gate-writing agent and accepts proposals against
it. One reader for both keeps a predicate from existing in a checklist without
the tagger knowing what to look for.

The prune helpers are the offline half, used when a checklist revision retires
items.
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


# -- Prune (offline, when a checklist revision retires items) ------------------


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


__all__ = [
    "PredicateDef",
    "load_vocabulary",
    "prune_items",
    "prune_vocabulary",
    "save_vocabulary",
]
