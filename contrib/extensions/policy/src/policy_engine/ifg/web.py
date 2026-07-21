# code-health: ignore-file[AM025] -- web snapshots normalize dynamic SQLite/JSON rows
"""Live, read-only web projection for persisted IFG sessions."""

from __future__ import annotations

import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
from pathlib import Path
import sqlite3
import time
from typing import ClassVar
from urllib.parse import parse_qs, urlparse
import webbrowser

from loguru import logger

from .schema import IFG_EXTRACTOR_VERSION

_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
_STATIC_CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}
_STATIC_ROUTES = {
    "/": "index.html",
    "/app.css": "app.css",
    "/app.js": "app.js",
    "/vendor/cytoscape.min.js": "vendor/cytoscape.min.js",
    "/vendor/CYTOSCAPE-LICENSE": "vendor/CYTOSCAPE-LICENSE",
}


def load_ifg_web_snapshot(
    db_path: Path,
    session_id: str,
    *,
    include_symbol_links: bool = True,
) -> dict[str, object]:
    """Build a compact three-lane graph from one persisted IFG session."""

    path = db_path.expanduser().resolve()
    connection = sqlite3.connect(
        f"{path.as_uri()}?mode=ro",
        uri=True,
        timeout=1.0,
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.execute("BEGIN")
        extractor_version = _latest_extractor_version(connection, session_id)
        nodes, action_to_tool, path_to_file, symbol_to_node = _load_nodes(
            connection,
            session_id,
            extractor_version,
        )
        edges = _load_edges(
            connection,
            session_id,
            extractor_version,
            action_to_tool=action_to_tool,
            path_to_file=path_to_file,
            symbol_to_node=symbol_to_node,
            include_symbol_links=include_symbol_links,
        )
    finally:
        connection.close()

    components = _annotate_components(nodes, edges)
    updated_at = max(
        [
            *(float(node.get("updatedAt", 0.0)) for node in nodes),
            *(float(edge.get("updatedAt", 0.0)) for edge in edges),
        ],
        default=0.0,
    )
    revision_material = (
        f"{session_id}:{extractor_version}:{len(nodes)}:{len(edges)}:{updated_at:.6f}"
    )
    revision = hashlib.sha1(revision_material.encode()).hexdigest()[:12]
    node_counts = _count_by(nodes, "type")
    edge_counts = _count_by(edges, "kind")
    tool_counts = _count_by(
        (node for node in nodes if node.get("type") == "action"),
        "tool",
    )
    return {
        "sessionId": session_id,
        "extractorVersion": extractor_version,
        "generatedAt": time.time(),
        "updatedAt": updated_at,
        "revision": revision,
        "nodes": nodes,
        "edges": edges,
        "components": components,
        "stats": {
            "nodes": len(nodes),
            "edges": len(edges),
            "components": len(components),
            "nodeTypes": node_counts,
            "edgeKinds": edge_counts,
            "tools": tool_counts,
        },
    }


def _latest_extractor_version(
    connection: sqlite3.Connection,
    session_id: str,
) -> str:
    row = connection.execute(
        """
        SELECT extractor_version, MAX(updated_at) AS latest
        FROM ifg_nodes
        WHERE session_id = ?
        GROUP BY extractor_version
        ORDER BY latest DESC
        LIMIT 1
        """,
        (session_id,),
    ).fetchone()
    return str(row["extractor_version"]) if row is not None else IFG_EXTRACTOR_VERSION


def _load_nodes(
    connection: sqlite3.Connection,
    session_id: str,
    extractor_version: str,
) -> tuple[
    list[dict[str, object]],
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:
    graph_rows = connection.execute(
        """
        SELECT node_id, node_type, stable_key, display_name,
               first_seen_turn, last_seen_turn, observation_count,
               source, confidence, metadata_json, updated_at
        FROM ifg_nodes
        WHERE session_id = ? AND extractor_version = ?
          AND node_type IN ('file', 'symbol')
        """,
        (session_id, extractor_version),
    ).fetchall()

    nodes: list[dict[str, object]] = []
    path_to_file: dict[str, str] = {}
    symbol_to_node: dict[str, str] = {}
    for row in graph_rows:
        metadata = _json_object(row["metadata_json"])
        node_type = str(row["node_type"])
        path = _string_or_none(metadata.get("path"))
        if node_type == "file" and not path:
            path = str(row["stable_key"])
        if node_type == "file" and path:
            path_to_file[path] = str(row["node_id"])
        if node_type == "symbol" and metadata.get("symbol_id"):
            symbol_to_node[str(metadata["symbol_id"])] = str(row["node_id"])
        nodes.append(
            {
                "id": str(row["node_id"]),
                "type": node_type,
                "label": _node_label(node_type, str(row["display_name"]), metadata),
                "displayName": str(row["display_name"]),
                "path": path,
                "shortPath": _short_path(path),
                "symbolKind": metadata.get("kind") or metadata.get("symbol_kind"),
                "validation": metadata.get("validation"),
                "firstSeenTurn": int(row["first_seen_turn"]),
                "lastSeenTurn": int(row["last_seen_turn"]),
                "observations": int(row["observation_count"]),
                "source": str(row["source"]),
                "confidence": str(row["confidence"]),
                "updatedAt": float(row["updated_at"]),
            }
        )

    action_rows = connection.execute(
        """
        SELECT action_id, tool_call_id, tool_name, action_kind, family,
               command, template, turn, event_id, confidence, created_at, updated_at
        FROM ifg_actions
        WHERE session_id = ? AND extractor_version = ?
        ORDER BY created_at, action_id
        """,
        (session_id, extractor_version),
    ).fetchall()
    event_rows = connection.execute(
        """
        WITH ranked AS (
            SELECT normalized_event_id, tool_call_id, tool_name, phase,
                   args_json, turn, ts, is_error, source, updated_at,
                   ROW_NUMBER() OVER (
                       PARTITION BY COALESCE(tool_call_id, normalized_event_id)
                       ORDER BY CASE phase WHEN 'post' THEN 0 ELSE 1 END,
                                updated_at DESC
                   ) AS row_rank
            FROM ifg_normalized_tool_events
            WHERE session_id = ? AND extractor_version = ?
        )
        SELECT * FROM ranked WHERE row_rank = 1 ORDER BY ts, normalized_event_id
        """,
        (session_id, extractor_version),
    ).fetchall()

    actions_by_call: dict[str, list[sqlite3.Row]] = {}
    action_to_tool: dict[str, str] = {}
    for row in action_rows:
        call_key = str(row["tool_call_id"] or f"action:{row['action_id']}")
        actions_by_call.setdefault(call_key, []).append(row)
        action_to_tool[str(row["action_id"])] = f"tool:{call_key}"

    seen_calls: set[str] = set()
    for event in event_rows:
        call_key = str(event["tool_call_id"] or event["normalized_event_id"])
        seen_calls.add(call_key)
        grouped_actions = actions_by_call.get(call_key, [])
        nodes.append(_tool_node(call_key, event, grouped_actions))

    for call_key, grouped_actions in actions_by_call.items():
        if call_key in seen_calls:
            continue
        nodes.append(_tool_node_from_actions(call_key, grouped_actions))

    return nodes, action_to_tool, path_to_file, symbol_to_node


def _tool_node(
    call_key: str,
    event: sqlite3.Row,
    actions: list[sqlite3.Row],
) -> dict[str, object]:
    args = _json_object(event["args_json"])
    tool_name = str(event["tool_name"])
    action_kinds = sorted({str(row["action_kind"]) for row in actions})
    commands = [str(row["command"]) for row in actions if row["command"]]
    return {
        "id": f"tool:{call_key}",
        "type": "action",
        "label": _tool_label(tool_name, args, _primary_command(actions, commands)),
        "displayName": tool_name,
        "tool": tool_name,
        "toolCallId": event["tool_call_id"],
        "phase": str(event["phase"]),
        "isError": bool(event["is_error"]),
        "turn": int(event["turn"]),
        "timestamp": float(event["ts"]),
        "actionKinds": action_kinds,
        "segmentCount": len(actions),
        "commands": commands,
        "args": _compact_value(args),
        "source": str(event["source"]),
        "confidence": _best_confidence(actions),
        "observations": max(len(actions), 1),
        "updatedAt": float(event["updated_at"]),
    }


def _tool_node_from_actions(
    call_key: str,
    actions: list[sqlite3.Row],
) -> dict[str, object]:
    first = actions[0]
    tool_name = str(first["tool_name"])
    commands = [str(row["command"]) for row in actions if row["command"]]
    action_kinds = sorted({str(row["action_kind"]) for row in actions})
    return {
        "id": f"tool:{call_key}",
        "type": "action",
        "label": _tool_label(tool_name, {}, _primary_command(actions, commands)),
        "displayName": tool_name,
        "tool": tool_name,
        "toolCallId": first["tool_call_id"],
        "phase": "unknown",
        "isError": False,
        "turn": int(first["turn"]),
        "timestamp": float(first["created_at"]),
        "actionKinds": action_kinds,
        "segmentCount": len(actions),
        "commands": commands,
        "args": {},
        "source": "ifg_actions",
        "confidence": _best_confidence(actions),
        "observations": max(len(actions), 1),
        "updatedAt": max(float(row["updated_at"]) for row in actions),
    }


def _load_edges(
    connection: sqlite3.Connection,
    session_id: str,
    extractor_version: str,
    *,
    action_to_tool: dict[str, str],
    path_to_file: dict[str, str],
    symbol_to_node: dict[str, str],
    include_symbol_links: bool,
) -> list[dict[str, object]]:
    aggregated: dict[tuple[str, str, str, str], dict[str, object]] = {}

    for row in connection.execute(
        """
        SELECT action_id, path, relation, source, confidence, created_at
        FROM ifg_action_file_edges
        WHERE session_id = ? AND extractor_version = ?
        """,
        (session_id, extractor_version),
    ):
        source_id = action_to_tool.get(str(row["action_id"]))
        target_id = path_to_file.get(str(row["path"]))
        if source_id and target_id:
            _add_edge(
                aggregated,
                kind="action-file",
                source_id=source_id,
                target_id=target_id,
                relation=str(row["relation"]),
                evidence_source=str(row["source"]),
                confidence=str(row["confidence"]),
                updated_at=float(row["created_at"]),
                optional=False,
            )

    for row in connection.execute(
        """
        SELECT action_id, symbol_id, relation, source, confidence, updated_at
        FROM ifg_action_symbol_edges
        WHERE session_id = ? AND extractor_version = ?
        """,
        (session_id, extractor_version),
    ):
        source_id = action_to_tool.get(str(row["action_id"]))
        target_id = symbol_to_node.get(str(row["symbol_id"]))
        if source_id and target_id:
            _add_edge(
                aggregated,
                kind="action-symbol",
                source_id=source_id,
                target_id=target_id,
                relation=str(row["relation"]),
                evidence_source=str(row["source"]),
                confidence=str(row["confidence"]),
                updated_at=float(row["updated_at"]),
                optional=False,
            )

    for row in connection.execute(
        """
        SELECT path, symbol_id, relation, source, confidence, updated_at
        FROM ifg_file_symbol_edges
        WHERE session_id = ? AND extractor_version = ?
        """,
        (session_id, extractor_version),
    ):
        file_id = path_to_file.get(str(row["path"]))
        source_id = symbol_to_node.get(str(row["symbol_id"]))
        if file_id and source_id:
            _add_edge(
                aggregated,
                kind="symbol-file",
                source_id=source_id,
                target_id=file_id,
                relation=_file_relation_label(str(row["relation"])),
                evidence_source=str(row["source"]),
                confidence=str(row["confidence"]),
                updated_at=float(row["updated_at"]),
                optional=False,
            )

    if include_symbol_links:
        symbol_link_rows = connection.execute(
            """
            SELECT from_symbol_id, to_symbol_id, relation, source, confidence,
                   updated_at
            FROM ifg_symbol_symbol_edges
            WHERE session_id = ? AND extractor_version = ?
            """,
            (session_id, extractor_version),
        )
    else:
        symbol_link_rows = ()

    for row in symbol_link_rows:
        source_id = symbol_to_node.get(str(row["from_symbol_id"]))
        target_id = symbol_to_node.get(str(row["to_symbol_id"]))
        if not source_id or not target_id:
            continue
        _add_edge(
            aggregated,
            kind="symbol-symbol",
            source_id=source_id,
            target_id=target_id,
            relation=str(row["relation"]),
            evidence_source=str(row["source"]),
            confidence=str(row["confidence"]),
            updated_at=float(row["updated_at"]),
            optional=True,
        )

    _add_symbol_bridge_edges(aggregated)

    return sorted(
        aggregated.values(),
        key=lambda edge: (
            str(edge["kind"]),
            str(edge["source"]),
            str(edge["target"]),
            str(edge["relation"]),
        ),
    )


def _add_edge(
    aggregated: dict[tuple[str, str, str, str], dict[str, object]],
    *,
    kind: str,
    source_id: str,
    target_id: str,
    relation: str,
    evidence_source: str,
    confidence: str,
    updated_at: float,
    optional: bool,
) -> None:
    key = (kind, source_id, target_id, relation)
    current = aggregated.get(key)
    if current is None:
        digest = hashlib.sha1("\x1f".join(key).encode()).hexdigest()[:20]
        aggregated[key] = {
            "id": f"display-edge:{digest}",
            "kind": kind,
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "confidence": confidence,
            "evidenceSources": [evidence_source],
            "evidenceCount": 1,
            "optional": optional,
            "updatedAt": updated_at,
        }
        return
    current["evidenceCount"] = int(current["evidenceCount"]) + 1
    current["updatedAt"] = max(float(current["updatedAt"]), updated_at)
    sources = current["evidenceSources"]
    if isinstance(sources, list) and evidence_source not in sources:
        sources.append(evidence_source)
    if _CONFIDENCE_RANK.get(confidence, -1) > _CONFIDENCE_RANK.get(
        str(current["confidence"]), -1
    ):
        current["confidence"] = confidence


def _add_symbol_bridge_edges(
    aggregated: dict[tuple[str, str, str, str], dict[str, object]],
) -> None:
    """Collapse action -> symbol -> file paths for the default two-lane view."""

    direct_pairs = {
        (str(edge["source"]), str(edge["target"]))
        for edge in aggregated.values()
        if edge["kind"] == "action-file"
    }
    actions_by_symbol: dict[str, list[dict[str, object]]] = {}
    files_by_symbol: dict[str, list[dict[str, object]]] = {}
    for edge in list(aggregated.values()):
        if edge["kind"] == "action-symbol":
            actions_by_symbol.setdefault(str(edge["target"]), []).append(edge)
        elif edge["kind"] == "symbol-file":
            files_by_symbol.setdefault(str(edge["source"]), []).append(edge)

    for symbol_id, action_edges in actions_by_symbol.items():
        for action_edge in action_edges:
            for file_edge in files_by_symbol.get(symbol_id, ()):
                source_id = str(action_edge["source"])
                target_id = str(file_edge["target"])
                if (source_id, target_id) in direct_pairs:
                    continue
                confidences = (
                    str(action_edge["confidence"]),
                    str(file_edge["confidence"]),
                )
                confidence = min(
                    confidences,
                    key=lambda value: _CONFIDENCE_RANK.get(value, -1),
                )
                _add_edge(
                    aggregated,
                    kind="action-file",
                    source_id=source_id,
                    target_id=target_id,
                    relation="via symbol",
                    evidence_source="symbol_bridge",
                    confidence=confidence,
                    updated_at=max(
                        float(action_edge["updatedAt"]),
                        float(file_edge["updatedAt"]),
                    ),
                    optional=False,
                )
                bridge = aggregated[("action-file", source_id, target_id, "via symbol")]
                bridge["derived"] = True


def _annotate_components(
    nodes: list[dict[str, object]],
    edges: list[dict[str, object]],
) -> list[dict[str, object]]:
    parents = {str(node["id"]): str(node["id"]) for node in nodes}

    def find(node_id: str) -> str:
        parent = parents[node_id]
        while parent != parents[parent]:
            parents[parent] = parents[parents[parent]]
            parent = parents[parent]
        while node_id != parent:
            next_id = parents[node_id]
            parents[node_id] = parent
            node_id = next_id
        return parent

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parents[right_root] = left_root
        else:
            parents[left_root] = right_root

    for edge in edges:
        if edge.get("optional"):
            continue
        source_id = str(edge["source"])
        target_id = str(edge["target"])
        if source_id in parents and target_id in parents:
            union(source_id, target_id)

    members: dict[str, list[dict[str, object]]] = {}
    for node in nodes:
        root = find(str(node["id"]))
        members.setdefault(root, []).append(node)

    edge_groups: dict[str, list[dict[str, object]]] = {root: [] for root in members}
    for edge in edges:
        if edge.get("optional"):
            continue
        source_id = str(edge["source"])
        target_id = str(edge["target"])
        if source_id not in parents or target_id not in parents:
            continue
        root = find(source_id)
        if root == find(target_id):
            edge_groups[root].append(edge)

    ranked = sorted(
        members,
        key=lambda root: (
            -len(members[root]),
            -len(edge_groups[root]),
            root,
        ),
    )
    components: list[dict[str, object]] = []
    for rank, root in enumerate(ranked, start=1):
        component_id = f"component:{hashlib.sha1(root.encode()).hexdigest()[:12]}"
        component_nodes = members[root]
        for node in component_nodes:
            node["componentId"] = component_id
            node["componentRank"] = rank
        for edge in edge_groups[root]:
            edge["componentId"] = component_id
        counts = _count_by(component_nodes, "type")
        tools = _count_by(
            (node for node in component_nodes if node.get("type") == "action"),
            "tool",
        )
        anchor = _component_anchor(component_nodes, edge_groups[root])
        components.append(
            {
                "id": component_id,
                "rank": rank,
                "label": _component_label(component_nodes, anchor),
                "anchorNodeId": anchor.get("id") if anchor else None,
                "nodes": len(component_nodes),
                "edges": len(edge_groups[root]),
                "actions": counts.get("action", 0),
                "symbols": counts.get("symbol", 0),
                "files": counts.get("file", 0),
                "tools": tools,
                "updatedAt": max(
                    [
                        *(
                            float(node.get("updatedAt", 0.0))
                            for node in component_nodes
                        ),
                        *(
                            float(edge.get("updatedAt", 0.0))
                            for edge in edge_groups[root]
                        ),
                    ],
                    default=0.0,
                ),
            }
        )

    for edge in edges:
        if edge.get("optional"):
            source = next(
                (node for node in nodes if node["id"] == edge["source"]), None
            )
            target = next(
                (node for node in nodes if node["id"] == edge["target"]), None
            )
            if (
                source
                and target
                and source.get("componentId") == target.get("componentId")
            ):
                edge["componentId"] = source.get("componentId")
            else:
                edge["crossComponent"] = True
    return components


def _component_anchor(
    nodes: list[dict[str, object]],
    edges: list[dict[str, object]],
) -> dict[str, object] | None:
    degree: dict[str, int] = {}
    for edge in edges:
        degree[str(edge["source"])] = degree.get(str(edge["source"]), 0) + 1
        degree[str(edge["target"])] = degree.get(str(edge["target"]), 0) + 1

    def score(node: dict[str, object]) -> tuple[int, int, int, str]:
        node_type = str(node.get("type", ""))
        symbol_kind = str(node.get("symbolKind") or "")
        type_score = 3 if node_type == "symbol" else 2 if node_type == "file" else 1
        if symbol_kind in {"module", "mention"}:
            type_score = 0
        return (
            type_score,
            degree.get(str(node["id"]), 0),
            int(node.get("observations", 0)),
            str(node.get("label", "")),
        )

    return max(nodes, key=score) if nodes else None


def _component_label(
    nodes: list[dict[str, object]],
    anchor: dict[str, object] | None,
) -> str:
    if anchor is None:
        return "Empty component"
    label = str(anchor.get("displayName") or anchor.get("label") or "Component")
    anchor_path = str(anchor.get("shortPath") or "")
    if anchor.get("type") == "symbol" and anchor_path:
        return f"{label} · {anchor_path}"
    file_node = next((node for node in nodes if node.get("type") == "file"), None)
    short_path = str(file_node.get("shortPath")) if file_node else ""
    if anchor.get("type") == "symbol" and short_path:
        return f"{label} · {short_path}"
    return label


def _node_label(
    node_type: str,
    display_name: str,
    metadata: dict[str, object],
) -> str:
    if node_type == "file":
        return _short_path(_string_or_none(metadata.get("path"))) or display_name
    return display_name


def _tool_label(
    tool_name: str,
    args: dict[str, object],
    primary_command: str,
) -> str:
    if tool_name == "bash":
        command = str(args.get("cmd") or args.get("command") or "")
        command_name = primary_command or _first_command(command)
        return f"bash · {command_name or 'shell'}"
    path = _string_or_none(
        args.get("path") or args.get("file_path") or args.get("filename")
    )
    return f"{tool_name} · {_short_path(path)}" if path else tool_name


def _primary_command(actions: list[sqlite3.Row], commands: list[str]) -> str:
    for action in actions:
        if action["command"] and action["action_kind"] not in {"control", "filter"}:
            return str(action["command"])
    return commands[0] if commands else ""


def _first_command(command: str) -> str:
    for token in command.replace("&&", " ").replace(";", " ").split():
        if token not in {"cd", "echo"} and not token.startswith("/"):
            return Path(token).name
    return ""


def _short_path(path: str | None, *, parts: int = 4) -> str:
    if not path:
        return ""
    clean = path.rstrip("/")
    values = [part for part in clean.split("/") if part]
    if len(values) <= parts:
        return clean
    return "/".join(values[-parts:])


def _file_relation_label(relation: str) -> str:
    return {
        "defines": "defined in",
        "exports": "exported by",
        "imports": "imported by",
    }.get(relation, relation)


def _best_confidence(rows: list[sqlite3.Row]) -> str:
    return max(
        (str(row["confidence"]) for row in rows),
        key=lambda value: _CONFIDENCE_RANK.get(value, -1),
        default="medium",
    )


def _count_by(
    rows: object,
    key: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:  # type: ignore[union-attr]
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _json_object(raw: object) -> dict[str, object]:
    if not raw:
        return {}
    try:
        value = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _compact_value(value: object, *, depth: int = 0) -> object:
    if depth >= 4:
        return "…"
    if isinstance(value, str):
        return value if len(value) <= 1000 else f"{value[:997]}…"
    if isinstance(value, dict):
        return {
            str(key): _compact_value(item, depth=depth + 1)
            for key, item in list(value.items())[:30]
        }
    if isinstance(value, list):
        return [_compact_value(item, depth=depth + 1) for item in value[:30]]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


def _string_or_none(value: object) -> str | None:
    return str(value) if value is not None and str(value) else None


class _IfgRequestHandler(BaseHTTPRequestHandler):
    db_path: ClassVar[Path]
    session_id: ClassVar[str]
    refresh_ms: ClassVar[int]

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        route = urlparse(self.path).path
        if route == "/api/graph":
            self._send_graph()
            return
        if route == "/api/health":
            self._send_json(
                {
                    "ok": True,
                    "sessionId": self.session_id,
                    "dbPath": str(self.db_path),
                }
            )
            return
        asset = _STATIC_ROUTES.get(route)
        if asset is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_asset(asset)

    def _send_graph(self) -> None:
        try:
            query = parse_qs(urlparse(self.path).query)
            include_symbol_links = query.get("symbol_links") == ["1"]
            snapshot = load_ifg_web_snapshot(
                self.db_path,
                self.session_id,
                include_symbol_links=include_symbol_links,
            )
            snapshot["refreshMs"] = self.refresh_ms
            if query.get("revision") == [snapshot["revision"]]:
                self._send_empty(HTTPStatus.NO_CONTENT)
                return
            self._send_json(snapshot)
        except (OSError, sqlite3.Error, ValueError) as exc:
            logger.exception("IFG web snapshot failed")
            self._send_json(
                {"error": str(exc), "sessionId": self.session_id},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _send_empty(self, status: HTTPStatus) -> None:
        self.send_response(status)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _send_json(
        self,
        payload: dict[str, object],
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_asset(self, asset: str) -> None:
        resource = files("policy_engine.web").joinpath(asset)
        try:
            body = resource.read_bytes()
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = _STATIC_CONTENT_TYPES.get(Path(asset).suffix, "text/plain")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format: str, *args: object) -> None:
        logger.debug("IFG web: {}", format % args)


def create_ifg_web_server(
    db_path: Path,
    session_id: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    refresh_ms: int = 1500,
) -> ThreadingHTTPServer:
    """Create, but do not start, the IFG HTTP server."""

    class Handler(_IfgRequestHandler):
        pass

    Handler.db_path = db_path.expanduser().resolve()
    Handler.session_id = session_id
    Handler.refresh_ms = refresh_ms
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def serve_ifg_web(
    db_path: Path,
    session_id: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    refresh_ms: int = 1500,
    open_browser: bool = False,
) -> str:
    """Serve the IFG viewer until interrupted and return its URL on shutdown."""

    server = create_ifg_web_server(
        db_path,
        session_id,
        host=host,
        port=port,
        refresh_ms=refresh_ms,
    )
    bound_host, bound_port = server.server_address[:2]
    visible_host = "127.0.0.1" if bound_host in {"0.0.0.0", "::"} else bound_host
    url = f"http://{visible_host}:{bound_port}/"
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return url


__all__ = [
    "create_ifg_web_server",
    "load_ifg_web_snapshot",
    "serve_ifg_web",
]
