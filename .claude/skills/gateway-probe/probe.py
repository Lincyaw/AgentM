#!/usr/bin/env python
"""Ad-hoc gateway/wire debug probe.

Connect to a running ``agentm gateway`` as a synthetic chat-client peer over the
v2 wire protocol, optionally send inbound message(s), and dump the outbound
frames the gateway emits. This is the tool to reach for when a chat client
"doesn't see X" (a command, a tool, a status) — it shows EXACTLY what the
gateway puts on the wire, with no TUI in the way.

Run it through the repo venv so it imports the editable gateway code:

    uv run python .claude/skills/gateway-probe/probe.py --help

Examples
--------
# What command list does the gateway send on a fresh session? (no LLM call if
# you only send a gateway builtin like /help; a real prompt triggers a turn)
uv run python .claude/skills/gateway-probe/probe.py \
    --connect ws://127.0.0.1:8770 --chat-id probe-$RANDOM \
    --send "hi" --summary

# Dump only session_ready frames (the command catalog lives here):
uv run python .claude/skills/gateway-probe/probe.py --send "hi" --only session_ready

# Just listen on connect (does anything arrive before the first inbound?):
uv run python .claude/skills/gateway-probe/probe.py --listen 3
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
from collections import Counter
from typing import Any


async def _run(args: argparse.Namespace) -> int:
    from agentm.gateway.client import WireClient
    from agentm.gateway.client_cli import resolve_connect

    rc = resolve_connect(args.connect, tls_ca=args.tls_ca)
    transport = rc[1] if isinstance(rc, tuple) else rc

    frames: list[tuple[str, dict[str, Any]]] = []

    async def on_outbound(env: Any) -> None:
        body = env.body if isinstance(env.body, dict) else {}
        frames.append((getattr(env, "kind", "?"), body))

    wkw = {
        "transport": transport,
        "peer_name": args.peer_name,
        "token": args.token,
        "on_outbound": on_outbound,
    }
    wkw = {k: v for k, v in wkw.items()
           if k in inspect.signature(WireClient.__init__).parameters}
    client = WireClient(**wkw)
    await client.connect()

    if args.listen_on_connect:
        await asyncio.sleep(args.listen_on_connect)
        print(f"[frames before any send: {len(frames)}]")

    skw = inspect.signature(client.send_inbound).parameters
    for i, content in enumerate(args.send):
        call = {"session_key": f"terminal:{args.chat_id}",
                "scenario": args.scenario if i == 0 else None,
                "env_id": f"in-probe-{i}"}
        call = {k: v for k, v in call.items() if k in skw}
        await client.send_inbound(
            {"channel": "terminal", "sender_id": args.sender_id,
             "chat_id": args.chat_id, "content": content},
            **call,
        )

    # Stop early once a session_ready is seen unless asked to wait the full time.
    for _ in range(int(args.listen / 0.25)):
        if not args.wait_full and any(
            (b.get("metadata") or {}).get("kind") == "session_ready" for _, b in frames
        ):
            break
        await asyncio.sleep(0.25)

    only = set(args.only or [])
    for _env_kind, body in frames:
        meta = body.get("metadata") or {}
        kind = meta.get("kind")
        if only and kind not in only:
            continue
        print(json.dumps({
            "meta_kind": kind,
            "channel": body.get("channel"),
            "command_names": meta.get("command_names"),
            "content": (str(body.get("content") or ""))[:200],
        }, ensure_ascii=False))

    if args.summary:
        counts = Counter((b.get("metadata") or {}).get("kind") for _, b in frames)
        print("KIND_COUNTS:", dict(counts))
        sr = [(b.get("metadata") or {}).get("command_names")
              for _, b in frames if (b.get("metadata") or {}).get("kind") == "session_ready"]
        print("SESSION_READY_COUNT:", len(sr))
        if sr:
            print("COMMAND_NAMES:", sorted(sr[0] or []))

    # Clean up a throwaway session so probes don't leak sessions in the gateway.
    if args.end and args.send:
        try:
            call = {k: v for k, v in {"session_key": f"terminal:{args.chat_id}",
                    "env_id": "in-probe-end"}.items() if k in skw}
            await client.send_inbound(
                {"channel": "terminal", "sender_id": args.sender_id,
                 "chat_id": args.chat_id, "content": "/end"}, **call)
            await asyncio.sleep(0.5)
        except Exception:  # noqa: BLE001
            pass
    try:
        await client.close()
    except Exception:  # noqa: BLE001
        pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe a running agentm gateway over the wire.")
    ap.add_argument("--connect", default="ws://127.0.0.1:8770",
                    help="gateway address (ws://host:port or unix:///path.sock)")
    ap.add_argument("--token", default=None, help="auth token (omit for --bind-allow-anonymous/uid)")
    ap.add_argument("--tls-ca", default=None)
    ap.add_argument("--peer-name", default="wire-probe")
    ap.add_argument("--sender-id", default="probe")
    ap.add_argument("--chat-id", default="wire-probe",
                    help="use a FRESH id to force a new session; reuse to hit an existing one")
    ap.add_argument("--scenario", default="local")
    ap.add_argument("--send", action="append", default=[],
                    help="inbound content to send (repeatable). A non-command triggers an LLM turn.")
    ap.add_argument("--only", action="append", default=[],
                    help="print only these metadata.kind values (repeatable)")
    ap.add_argument("--summary", action="store_true", help="print kind counts + session_ready command_names")
    ap.add_argument("--listen", type=float, default=12.0, help="max seconds to wait for frames")
    ap.add_argument("--listen-on-connect", type=float, default=0.0,
                    help="seconds to listen before sending anything")
    ap.add_argument("--wait-full", action="store_true", help="always wait the full --listen window")
    ap.add_argument("--no-end", dest="end", action="store_false", help="do not /end the probe session")
    args = ap.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
