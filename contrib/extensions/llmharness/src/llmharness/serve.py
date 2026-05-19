"""``llmharness serve`` — read-only HTTP server over a ``cases/`` root.

Exposes the on-disk layout written by ``llmharness-aggregate`` as a small
JSON / NDJSON API consumed by remote reviewers (e.g. the aegis-ui
``Case Review`` sub-app). Read-only, stdlib-only, CORS-enabled.

API
---

``GET  /api/health``
    ``{"root": "<abs path>", "case_count": N, "version": "<pkg ver>"}``

``GET  /api/cases``
    ``{"cases": [{"case_id": ..., "meta": {...}}, ...]}`` (alphabetic).

``GET  /api/cases/<case_id>/meta``                       — ``meta.json``
``GET  /api/cases/<case_id>/main_agent``                 — ``main_agent.jsonl``
``GET  /api/cases/<case_id>/verdicts``                   — ``verdicts.jsonl``
``GET  /api/cases/<case_id>/trajectory``                 — ``trajectory.jsonl``
``GET  /api/cases/<case_id>/firings/<phase>``            — ``{"files": [...]}``
``GET  /api/cases/<case_id>/firings/<phase>/<name>``     — one firing JSON
``GET  /api/cases/<case_id>/snapshots/<seq>``            — ``event_graph/after_extractor_<seq:03d>.json``

``<phase>`` is ``extractor`` or ``auditor``. ``<seq>`` is a positive integer.
Missing snapshots return 404 with a JSON body (the UI treats that as
"firing did not advance the graph").
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata as _metadata
from pathlib import Path
from typing import Any
from urllib.parse import unquote

_log = logging.getLogger("llmharness.serve")

_CASE_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
_FIRING_NAME_RE = re.compile(r"^\d+_turn_\d+\.json$")
_PHASES = frozenset({"extractor", "auditor"})


def _pkg_version() -> str:
    try:
        return _metadata.version("llmharness")
    except _metadata.PackageNotFoundError:
        return "0.0.0"


class _Handler(BaseHTTPRequestHandler):
    # Bound by ``make_server``.
    root: Path
    allow_origin: str

    # --- response helpers -------------------------------------------------

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", self.allow_origin)
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found", "path": str(path)})
            return
        except OSError as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        self.send_response(HTTPStatus.OK)
        self._cors()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _not_found(self, why: str) -> None:
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found", "detail": why})

    def _bad_request(self, why: str) -> None:
        self._send_json(HTTPStatus.BAD_REQUEST, {"error": "bad_request", "detail": why})

    # --- dispatch ---------------------------------------------------------

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors()
        self.end_headers()

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        parts = [unquote(p) for p in self.path.split("?", 1)[0].strip("/").split("/") if p]
        try:
            self._route(parts)
        except Exception as exc:  # pragma: no cover — defensive
            _log.exception("handler crashed: %s", exc)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _route(self, parts: list[str]) -> None:
        if parts[:1] != ["api"]:
            self._not_found("expected /api/...")
            return
        rest = parts[1:]
        if rest == ["health"]:
            self._handle_health()
            return
        if rest == ["cases"]:
            self._handle_list_cases()
            return
        if len(rest) >= 2 and rest[0] == "cases":
            case_id = rest[1]
            if not _CASE_ID_RE.match(case_id):
                self._bad_request("invalid case_id")
                return
            self._handle_case(case_id, rest[2:])
            return
        self._not_found("unknown route")

    # --- handlers ---------------------------------------------------------

    def _handle_health(self) -> None:
        try:
            count = sum(1 for p in self.root.iterdir() if (p / "meta.json").is_file())
        except OSError as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        self._send_json(
            HTTPStatus.OK,
            {"root": str(self.root), "case_count": count, "version": _pkg_version()},
        )

    def _handle_list_cases(self) -> None:
        cases: list[dict[str, Any]] = []
        try:
            entries = sorted(p for p in self.root.iterdir() if p.is_dir())
        except OSError as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        for d in entries:
            meta_path = d / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            cases.append({"case_id": d.name, "meta": meta})
        self._send_json(HTTPStatus.OK, {"cases": cases})

    def _handle_case(self, case_id: str, tail: list[str]) -> None:
        case_dir = self.root / case_id
        if not case_dir.is_dir():
            self._not_found(f"case {case_id} not found")
            return
        # ``GET /api/cases/<id>/{meta,main_agent,verdicts,trajectory}``
        flat: dict[tuple[str, ...], tuple[str, str]] = {
            ("meta",): ("meta.json", "application/json; charset=utf-8"),
            ("main_agent",): ("main_agent.jsonl", "application/x-ndjson; charset=utf-8"),
            ("verdicts",): ("verdicts.jsonl", "application/x-ndjson; charset=utf-8"),
            ("trajectory",): ("trajectory.jsonl", "application/x-ndjson; charset=utf-8"),
        }
        key = tuple(tail)
        if key in flat:
            file_name, content_type = flat[key]
            self._send_file(case_dir / file_name, content_type)
            return
        if len(tail) >= 1 and tail[0] == "firings":
            self._handle_firings(case_dir, tail[1:])
            return
        if len(tail) == 2 and tail[0] == "snapshots":
            self._handle_snapshot(case_dir, tail[1])
            return
        self._not_found("unknown case route")

    def _handle_firings(self, case_dir: Path, tail: list[str]) -> None:
        if not tail:
            self._bad_request("missing phase")
            return
        phase = tail[0]
        if phase not in _PHASES:
            self._bad_request(f"phase must be one of {sorted(_PHASES)}")
            return
        phase_dir = case_dir / phase
        if len(tail) == 1:
            try:
                files = sorted(
                    p.name for p in phase_dir.iterdir() if _FIRING_NAME_RE.match(p.name)
                )
            except FileNotFoundError:
                files = []
            self._send_json(HTTPStatus.OK, {"files": files})
            return
        if len(tail) == 2:
            name = tail[1]
            if not _FIRING_NAME_RE.match(name):
                self._bad_request("invalid firing file name")
                return
            self._send_file(phase_dir / name, "application/json; charset=utf-8")
            return
        self._not_found("unknown firings route")

    def _handle_snapshot(self, case_dir: Path, raw_seq: str) -> None:
        try:
            seq = int(raw_seq)
        except ValueError:
            self._bad_request("snapshot seq must be an integer")
            return
        if seq < 0:
            self._bad_request("snapshot seq must be non-negative")
            return
        name = f"after_extractor_{seq:03d}.json"
        self._send_file(case_dir / "event_graph" / name, "application/json; charset=utf-8")

    # --- logging ----------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:
        _log.info("%s - %s", self.address_string(), format % args)


def make_server(root: Path, host: str, port: int, allow_origin: str) -> ThreadingHTTPServer:
    """Build a ready-to-serve HTTP server over ``root``.

    Resolves ``root`` to an absolute path up-front and validates it points
    at an existing directory. Returns a ``ThreadingHTTPServer`` whose
    handler class is bound to this ``(root, allow_origin)`` pair.
    """
    abs_root = root.resolve()
    if not abs_root.is_dir():
        raise NotADirectoryError(f"--root is not a directory: {abs_root}")

    handler = type(
        "BoundHandler",
        (_Handler,),
        {"root": abs_root, "allow_origin": allow_origin},
    )
    return ThreadingHTTPServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llmharness serve",
        description="Read-only HTTP server over a llmharness-aggregate cases/ root.",
    )
    parser.add_argument("--root", required=True, help="Path to the cases/ directory to serve.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765).")
    parser.add_argument(
        "--allow-origin",
        default="*",
        help="Access-Control-Allow-Origin value (default: *).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        server = make_server(Path(args.root), args.host, args.port, args.allow_origin)
    except (NotADirectoryError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    host, port = server.server_address[:2]
    _log.info("serving %s on http://%s:%s (CORS: %s)", Path(args.root).resolve(), host, port, args.allow_origin)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log.info("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
