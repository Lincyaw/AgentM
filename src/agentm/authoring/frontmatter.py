# code-health: ignore-file[AM025] -- authoring parsers validate untyped frontmatter data
"""Strict YAML frontmatter document used by authoring workflows."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import yaml

FrontmatterValue = str | int | float | bool | None


def _empty_metadata() -> Mapping[str, FrontmatterValue]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class FrontmatterDocument:
    metadata: Mapping[str, FrontmatterValue] = field(default_factory=_empty_metadata)
    body: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.body, str):
            raise TypeError("frontmatter body must be a string")
        copied: dict[str, FrontmatterValue] = {}
        for key, value in self.metadata.items():
            if not isinstance(key, str) or not key:
                raise ValueError("frontmatter keys must be non-empty strings")
            if value is not None and not isinstance(
                value,
                (str, int, float, bool),
            ):
                raise TypeError(f"frontmatter value {key!r} must be a scalar")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"frontmatter value {key!r} must be finite")
            copied[key] = value
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(copied),
        )

    def render(self) -> str:
        if not self.metadata:
            return self.body
        header = yaml.safe_dump(
            dict(self.metadata),
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=True,
        ).rstrip("\n")
        return f"---\n{header}\n---\n{self.body}"

    @classmethod
    def parse(cls, text: str) -> "FrontmatterDocument":
        if not text.startswith("---\n"):
            return cls(body=text)
        closing = text.find("\n---\n", 4)
        if closing < 0:
            raise ValueError("frontmatter opening delimiter has no closing delimiter")
        raw_metadata = text[4:closing]
        body = text[closing + 5 :]
        loaded = yaml.safe_load(raw_metadata)
        if loaded is None:
            metadata: dict[str, FrontmatterValue] = {}
        elif isinstance(loaded, Mapping):
            metadata = {}
            for key, value in loaded.items():
                if not isinstance(key, str):
                    raise TypeError("frontmatter keys must be strings")
                if value is not None and not isinstance(
                    value,
                    (str, int, float, bool),
                ):
                    raise TypeError(f"frontmatter value {key!r} must be a scalar")
                metadata[key] = value
        else:
            raise TypeError("frontmatter header must be a YAML object")
        return cls(metadata=metadata, body=body)


__all__ = ["FrontmatterDocument", "FrontmatterValue"]
