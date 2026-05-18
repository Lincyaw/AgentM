#!/usr/bin/env python3
"""Upload a cases/ directory tree to aegis-blob under `bucket/prefix/`.

The deployed `/cases` sub-app reads from the aegis-blob `shared` bucket
under `cases/<set-name>/`. This script presign-puts every regular file
in <local-dir> to `<bucket>/<prefix><case_id>/<rel-path>` in parallel.

Auth: pulls the JWT from ~/.aegisctl/config.yaml (current-context).

Why a Python uploader instead of a proper `aegisctl blob` subcommand?
The Go subcommand is the right long-term home, but for unblocking the
extractor-prompt iteration loop a 150-line uploader is enough. Replace
this with `aegisctl blob sync` when that exists.

Usage::

    scripts/upload_cases_to_blob.py \\
        --src ./cases \\
        --bucket shared \\
        --prefix cases/openrca-iter-baseline/ \\
        --concurrency 4

Exit codes: 0 ok, 2 partial failure, 3 auth failure, 4 bad args.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

import json
import ssl
import yaml

_AEGISCTL_CONFIG = Path.home() / ".aegisctl" / "config.yaml"


class AuthError(RuntimeError):
    pass


def _load_token_and_server() -> tuple[str, str]:
    if not _AEGISCTL_CONFIG.is_file():
        raise AuthError(f"missing {_AEGISCTL_CONFIG} — run `aegisctl auth login` first")
    cfg = yaml.safe_load(_AEGISCTL_CONFIG.read_text(encoding="utf-8")) or {}
    ctx_name = cfg.get("current-context")
    if not ctx_name:
        raise AuthError("no current-context in aegisctl config")
    ctx = (cfg.get("contexts") or {}).get(ctx_name)
    if not isinstance(ctx, dict):
        raise AuthError(f"context {ctx_name!r} not found in aegisctl config")
    token = ctx.get("token")
    server = ctx.get("server")
    if not token or not server:
        raise AuthError(f"context {ctx_name!r} is missing token/server")
    return str(token), str(server).rstrip("/")


# Caddy fronting the aegis API uses a self-signed cert — match the
# behaviour the rest of the toolchain expects.
_TLS_CTX = ssl.create_default_context()
_TLS_CTX.check_hostname = False
_TLS_CTX.verify_mode = ssl.CERT_NONE


def _http(method: str, url: str, token: str, *, json_body: Any = None) -> dict[str, Any]:
    data: bytes | None = None
    headers = {"Authorization": f"Bearer {token}"}
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, method=method, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, context=_TLS_CTX, timeout=60) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"{method} {url} -> {exc.code}: {body[:400]}") from exc
    if not raw:
        return {}
    return json.loads(raw)


_CONTENT_TYPES = {
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".md": "text/markdown",
    ".txt": "text/plain",
}


def _content_type_for(path: Path) -> str:
    return _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


def _walk_files(src: Path) -> list[Path]:
    out: list[Path] = []
    for root, dirs, files in os.walk(src):
        # Skip dot-dirs (`.DS_Store`, `.cache`, …) — never useful payload.
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in files:
            if fn.startswith("."):
                continue
            out.append(Path(root) / fn)
    out.sort()
    return out


def _upload_one(
    *,
    server: str,
    token: str,
    bucket: str,
    key: str,
    local_path: Path,
) -> tuple[Path, str | None]:
    """Returns (path, error_string_or_None)."""
    try:
        body = local_path.read_bytes()
        ct = _content_type_for(local_path)
        presign_resp = _http(
            "POST",
            f"{server}/api/v2/blob/buckets/{urllib.parse.quote(bucket)}/presign-put",
            token,
            json_body={
                "key": key,
                "content_type": ct,
                "content_length": len(body),
                "entity_kind": "llmharness-case",
            },
        )
        ps = presign_resp.get("presigned") or {}
        put_url = ps.get("url")
        put_method = (ps.get("method") or "PUT").upper()
        extra_headers = ps.get("headers") or {}
        if not put_url:
            return local_path, f"presign returned no url: {presign_resp}"

        # Localfs driver returns a relative `/api/v2/blob/raw/...` path;
        # rewrite to absolute against the configured server.
        if put_url.startswith("/"):
            put_url = f"{server}{put_url}"

        req = urllib.request.Request(put_url, method=put_method, data=body)
        for hk, hv in extra_headers.items():
            req.add_header(hk, hv)
        req.add_header("Content-Type", ct)
        with urllib.request.urlopen(req, context=_TLS_CTX, timeout=120) as resp:
            if resp.status >= 300:
                return local_path, f"PUT -> HTTP {resp.status}"
        return local_path, None
    except Exception as exc:  # noqa: BLE001 — surface message for batch report
        return local_path, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="local cases/ dir to upload")
    ap.add_argument("--bucket", default="shared")
    ap.add_argument(
        "--prefix",
        required=True,
        help="bucket-relative prefix; trailing slash auto-added if missing",
    )
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the destination keys without uploading.",
    )
    args = ap.parse_args()

    if not args.src.is_dir():
        print(f"--src must be a directory: {args.src}", file=sys.stderr)
        return 4

    prefix = args.prefix
    if not prefix.endswith("/"):
        prefix += "/"

    try:
        token, server = _load_token_and_server()
    except AuthError as exc:
        print(f"auth: {exc}", file=sys.stderr)
        return 3

    files = _walk_files(args.src)
    if not files:
        print(f"no files under {args.src}", file=sys.stderr)
        return 4

    print(
        f"=> uploading {len(files)} files to {args.bucket}/{prefix} "
        f"(server={server}, concurrency={args.concurrency})",
        file=sys.stderr,
    )

    def key_for(p: Path) -> str:
        rel = p.relative_to(args.src).as_posix()
        return f"{prefix}{rel}"

    if args.dry_run:
        for p in files:
            print(f"{p}  ->  {args.bucket}/{key_for(p)}")
        return 0

    failed: list[tuple[Path, str]] = []
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {
            pool.submit(
                _upload_one,
                server=server,
                token=token,
                bucket=args.bucket,
                key=key_for(p),
                local_path=p,
            ): p
            for p in files
        }
        done = 0
        for fut in cf.as_completed(futs):
            path, err = fut.result()
            done += 1
            if err:
                failed.append((path, err))
                print(f"  ! [{done}/{len(files)}] FAIL {path}: {err}", file=sys.stderr)
            else:
                print(f"  . [{done}/{len(files)}] {path.name}", file=sys.stderr)

    if failed:
        print(f"\nFAILED {len(failed)}/{len(files)}", file=sys.stderr)
        return 2

    print(
        f"\nDone. Point /cases Settings at bucket={args.bucket} prefix={prefix}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
