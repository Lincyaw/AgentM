"""Small frontmatter helper for presenter-authored artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FrontmatterDocument:
    metadata: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)
    body: str = ""

    def render(self) -> str:
        if not self.metadata:
            return self.body
        lines = ["---"]
        for key, value in sorted(self.metadata.items()):
            if value is None:
                continue
            lines.append(f"{key}: {value}")
        lines.append("---")
        lines.append(self.body)
        return "\n".join(lines)

    @classmethod
    def parse(cls, text: str) -> "FrontmatterDocument":
        if not text.startswith("---\n"):
            return cls(body=text)
        _, _, rest = text.partition("---\n")
        raw_meta, separator, body = rest.partition("\n---\n")
        if not separator:
            return cls(body=text)
        metadata: dict[str, str] = {}
        for line in raw_meta.splitlines():
            key, sep, value = line.partition(":")
            if sep:
                metadata[key.strip()] = value.strip()
        return cls(metadata=metadata, body=body)


__all__ = ["FrontmatterDocument"]
