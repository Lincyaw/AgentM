"""Policy engine IFC — minimal literal substring taint tracking."""

from __future__ import annotations

import re
from collections import deque

from .types import ToolArgs


# ---------------------------------------------------------------------------
# Label definition
# ---------------------------------------------------------------------------


class TaintLabel:
    """A compiled IFC label definition."""

    __slots__ = ("name", "source_tools", "result_regex", "min_length")

    def __init__(
        self,
        name: str,
        source_tools: frozenset[str],
        result_regex: re.Pattern[str],
        min_length: int = 8,
    ) -> None:
        self.name = name
        self.source_tools = source_tools
        self.result_regex = result_regex
        self.min_length = min_length


# ---------------------------------------------------------------------------
# Taint store
# ---------------------------------------------------------------------------

_MAX_TAINTED_VALUES = 100


class TaintStore:
    """Tracks tainted literal values per label. Session-scoped."""

    __slots__ = ("_values",)

    def __init__(self) -> None:
        self._values: dict[str, deque[str]] = {}

    def add_values(self, label_name: str, values: list[str], min_length: int) -> None:
        if label_name not in self._values:
            self._values[label_name] = deque(maxlen=_MAX_TAINTED_VALUES)
        store = self._values[label_name]
        for v in values:
            if len(v) >= min_length and v not in store:
                store.append(v)

    def check_taint(self, text: str) -> set[str]:
        """Check if text contains any tainted values. Returns set of label names."""
        labels: set[str] = set()
        for label_name, values in self._values.items():
            for v in values:
                if v in text:
                    labels.add(label_name)
                    break
        return labels

    def has_any_taint(self) -> bool:
        return any(len(v) > 0 for v in self._values.values())


# ---------------------------------------------------------------------------
# IFC engine
# ---------------------------------------------------------------------------


class IFCEngine:
    """Minimal IFC: literal substring taint propagation."""

    __slots__ = ("_labels", "_store")

    def __init__(self, labels: list[TaintLabel] | None = None) -> None:
        self._labels = labels or []
        self._store = TaintStore()

    def process_tool_result(self, tool_name: str, result_text: str) -> None:
        """Check tool result against label source patterns, extract tainted values."""
        for label in self._labels:
            if label.source_tools and tool_name not in label.source_tools:
                continue
            matches = label.result_regex.findall(result_text)
            if matches:
                values: list[str] = []
                for m in matches:
                    if isinstance(m, tuple):
                        values.extend(v for v in m if v)
                    elif isinstance(m, str):
                        values.append(m)
                self._store.add_values(label.name, values, label.min_length)

    def check_event_taint(self, args: ToolArgs) -> set[str]:
        """Check if tool call args contain tainted values."""
        if not self._store.has_any_taint():
            return set()

        text = _flatten_args_to_text(args)
        return self._store.check_taint(text)


def _flatten_args_to_text(args: ToolArgs) -> str:
    """Recursively flatten args dict to a single searchable string."""
    parts: list[str] = []
    _flatten_value(args, parts)
    return " ".join(parts)


def _flatten_value(value: object, parts: list[str]) -> None:
    if isinstance(value, str):
        parts.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            _flatten_value(v, parts)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _flatten_value(v, parts)
    elif value is not None:
        parts.append(str(value))


# ---------------------------------------------------------------------------
# Label compilation from YAML
# ---------------------------------------------------------------------------


def compile_labels(labels_config: dict[str, dict[str, object]] | None) -> list[TaintLabel]:
    """Compile label definitions from policy YAML."""
    if not labels_config:
        return []

    compiled: list[TaintLabel] = []
    for name, config in labels_config.items():
        if not isinstance(config, dict):
            continue

        source = config.get("source", {})
        if not isinstance(source, dict):
            continue

        tool_pattern = str(source.get("tool", ""))
        tools = frozenset(tool_pattern.split("|")) if tool_pattern else frozenset()

        regex_str = str(source.get("result_matches", ""))
        if not regex_str:
            continue

        try:
            regex = re.compile(regex_str)
        except re.error:
            continue

        min_length = int(str(config.get("min_length", 8) or 8))
        compiled.append(TaintLabel(
            name=name,
            source_tools=tools,
            result_regex=regex,
            min_length=min_length,
        ))

    return compiled
