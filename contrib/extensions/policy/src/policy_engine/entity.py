# code-health: ignore-file[AM025] -- entity extraction normalizes untyped tool payloads
"""Policy engine entity registry — 3-layer extraction pipeline."""

from __future__ import annotations

import re

from .types import EMPTY, EntityRecord, Evidence, EvidenceList, ToolArgs, _EmptyType


# ---------------------------------------------------------------------------
# Character-class patterns for Layer 2 (lexical extraction)
# ---------------------------------------------------------------------------

_PATH_PATTERN = re.compile(
    r"(?:^|[\s\"'`,(])(/[a-zA-Z0-9_./-]{3,}|[a-zA-Z]:\\[^\s\"'`]+)"
)
_URL_PATTERN = re.compile(r"https?://[^\s\"'`)\]>]{5,}")
_SYMBOL_PATTERN = re.compile(
    r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+|[a-z]+(?:[A-Z][a-z0-9]+)+)\b"
)


# ---------------------------------------------------------------------------
# Trie for Layer 3 dictionary recall
# ---------------------------------------------------------------------------


class _Trie:
    """Simple trie for multi-pattern substring search."""

    __slots__ = ("children", "output", "fail")

    def __init__(self) -> None:
        self.children: dict[str, _Trie] = {}
        self.output: list[str] = []
        self.fail: _Trie | None = None

    @classmethod
    def build(cls, patterns: list[str]) -> _Trie:
        """Build an Aho-Corasick automaton from patterns."""
        root = cls()
        for pattern in patterns:
            node = root
            for ch in pattern:
                if ch not in node.children:
                    node.children[ch] = cls()
                node = node.children[ch]
            node.output.append(pattern)

        # Build failure links (BFS)
        from collections import deque

        queue: deque[_Trie] = deque()
        for child in root.children.values():
            child.fail = root
            queue.append(child)

        while queue:
            current = queue.popleft()
            for ch, child in current.children.items():
                queue.append(child)
                fail = current.fail
                while fail and ch not in fail.children:
                    fail = fail.fail
                child.fail = fail.children[ch] if fail and ch in fail.children else root
                if child.fail is child:
                    child.fail = root
                child.output = child.output + child.fail.output

        return root

    def search(self, text: str) -> set[str]:
        """Find all pattern occurrences in text. Returns matched patterns."""
        matches: set[str] = set()
        node = self
        root = self
        for ch in text:
            while node is not root and ch not in node.children:
                node = node.fail or root
            node = node.children.get(ch, root)
            if node.output:
                matches.update(node.output)
        return matches


# ---------------------------------------------------------------------------
# EntityRegistry
# ---------------------------------------------------------------------------


class EntityRegistry:
    """Evidence-based entity tracking with 3-layer extraction."""

    def __init__(self) -> None:
        self._entities: dict[str, EntityRecord] = {}
        self._trie: _Trie | None = None
        self._trie_dirty: bool = True

    def get(self, name: str) -> EvidenceList | _EmptyType:
        record = self._entities.get(name)
        if record is None:
            return EMPTY
        return record.evidence

    def entries(self) -> tuple[EntityRecord, ...]:
        return tuple(self._entities.values())

    def _ensure_entity(self, name: str, entity_type: str, turn: int) -> EntityRecord:
        record = self._entities.get(name)
        if record is None:
            record = EntityRecord(
                entity=name,
                entity_type=entity_type,
                first_seen_turn=turn,
                last_seen_turn=turn,
            )
            self._entities[name] = record
            self._trie_dirty = True
        return record

    def add_evidence(
        self,
        name: str,
        entity_type: str,
        evidence_type: str,
        turn: int,
        detail: str = "",
    ) -> None:
        record = self._ensure_entity(name, entity_type, turn)
        ev = Evidence(type=evidence_type, turn=turn, detail=detail)
        record.add_evidence(ev)

    # --- Layer 1: Structural extraction (typed tool schema fields) ---

    def extract_structural(self, tool_name: str, args: ToolArgs, turn: int) -> None:
        """Extract entities from known tool schema fields."""
        path = args.get("path") or args.get("file_path")
        if path and isinstance(path, str):
            self.add_evidence(
                path, "path", "structural", turn, f"{tool_name}(path={path})"
            )

        cmd = args.get("cmd") or args.get("command")
        if cmd and isinstance(cmd, str):
            self._extract_paths_from_cmd(cmd, turn, tool_name)

    def _extract_paths_from_cmd(self, cmd: str, turn: int, tool_name: str) -> None:
        for match in _PATH_PATTERN.finditer(cmd):
            path = match.group(1)
            if len(path) >= 4:
                self.add_evidence(
                    path,
                    "path",
                    "structural",
                    turn,
                    f"{tool_name} cmd references {path}",
                )

    # --- Layer 2: Lexical extraction (character-class tokenization) ---

    def extract_lexical(self, text: str, turn: int, source: str = "") -> None:
        """Extract entities from free text by character-class analysis."""
        if not text or len(text) > 50000:
            return

        for match in _PATH_PATTERN.finditer(text):
            path = match.group(1)
            if len(path) >= 4 and path not in self._entities:
                self.add_evidence(
                    path,
                    "path",
                    "lexical_match",
                    turn,
                    f"found in {source}" if source else "lexical",
                )

        for match in _URL_PATTERN.finditer(text):
            url = match.group(0)
            if url not in self._entities:
                self.add_evidence(
                    url,
                    "url",
                    "lexical_match",
                    turn,
                    f"found in {source}" if source else "lexical",
                )

        for match in _SYMBOL_PATTERN.finditer(text):
            sym = match.group(1)
            if len(sym) >= 5 and sym not in self._entities:
                self.add_evidence(
                    sym,
                    "symbol",
                    "lexical_match",
                    turn,
                    f"found in {source}" if source else "lexical",
                )

    # --- Layer 3: Session dictionary (trie recall) ---

    def _rebuild_trie(self) -> None:
        """Rebuild the Aho-Corasick trie from strong entities."""
        patterns = [
            name
            for name, rec in self._entities.items()
            if len(name) >= 5
            and (
                rec.evidence.has(type="structural")
                or rec.evidence.has(type="tool_success")
            )
        ]
        self._trie = _Trie.build(patterns) if patterns else None
        self._trie_dirty = False

    def extract_dict_recall(self, text: str, turn: int) -> None:
        """Detect references to known entities via session dictionary (trie)."""
        if not text:
            return

        if self._trie_dirty:
            self._rebuild_trie()

        if self._trie is None:
            return

        matches = self._trie.search(text)
        for name in matches:
            record = self._entities.get(name)
            if record is None:
                continue
            has_recall = any(
                e.type == "dict_recall" and e.turn == turn
                for e in record.evidence.records
            )
            if not has_recall:
                self.add_evidence(
                    name, record.entity_type, "dict_recall", turn, "referenced in text"
                )

    # --- Tool result processing ---

    def record_tool_success(self, tool_name: str, args: ToolArgs, turn: int) -> None:
        """Record successful tool execution as evidence."""
        path = args.get("path") or args.get("file_path")
        if path and isinstance(path, str):
            self.add_evidence(
                path, "path", "tool_success", turn, f"{tool_name}() succeeded"
            )

    def record_tool_failure(
        self, tool_name: str, args: ToolArgs, turn: int, error: str = ""
    ) -> None:
        """Record failed tool execution as evidence."""
        path = args.get("path") or args.get("file_path")
        if path and isinstance(path, str):
            self.add_evidence(
                path,
                "path",
                "tool_failure",
                turn,
                f"{tool_name}() failed: {error[:80]}",
            )

    # --- Public query interface ---

    def entity_evidence(self, name: str) -> EvidenceList | _EmptyType:
        return self.get(name)
