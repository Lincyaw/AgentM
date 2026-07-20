"""Pure redaction helpers for observability atoms.

Two responsibilities, both pure (no I/O, no logging, no side effects):

* :func:`redact_messages` — replace LLM prompt bodies with
  ``{chars, sha256_prefix}`` stubs so user secrets, pasted API keys, and
  private file contents never reach disk in the default observability path.
* :func:`redact_headers` — scrub auth-bearing HTTP header values
  (``Authorization``, ``X-API-Key``, etc.) while preserving the key names
  so operators can still see which credentials were configured.

Both helpers operate on **already-JSON-serialized dicts** (post
``agentm.core.lib.to_jsonable``) — keeping them at the dict layer means
they're orthogonal to dataclass shapes and trivial to unit-test from a
fabricated input.

Lives in ``core.lib`` because observability atoms and provider/logging
adapters need identical redaction semantics, while atom-to-atom and
atom-to-runtime imports remain forbidden. ``core.lib`` is the sanctioned
shared "stdlib for atoms".
"""

# code-health: ignore-file[AM022] -- recursively sanitizes heterogeneous diagnostic JSON

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

__all__ = [
    "SENSITIVE_HEADER_NAMES",
    "redact_config",
    "redact_headers",
    "redact_messages",
]


# Case-insensitive set of header names whose values must be replaced with
# ``"***"``. Picked from the headers commonly carrying bearer tokens / API
# keys in HTTP-style configs (OTLP, generic webhook sinks, etc.).
SENSITIVE_HEADER_NAMES: frozenset[str] = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "x-auth-token",
        "cookie",
    }
)
_SENSITIVE_CONFIG_NAMES: frozenset[str] = frozenset(
    {
        "api_key",
        "authorization",
        "client_secret",
        "cookie",
        "password",
        "secret",
        "token",
    }
)
_SENSITIVE_CONFIG_SUFFIXES: tuple[str, ...] = (
    "_api_key",
    "_auth_token",
    "_access_token",
    "_client_secret",
    "_password",
)


def _sha256_prefix(text: str) -> str:
    """First 16 hex chars of SHA-256 over ``text`` encoded as UTF-8.

    16 hex = 64 bits — enough to make collisions astronomically unlikely
    for the trace-debugging use case (correlating "same prompt sent twice")
    while not echoing enough of the hash to enable rainbow-table recovery
    of short secrets.
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _message_text_chars(content: Any) -> int:
    """Count textual characters in a serialized message ``content`` list.

    Accepts the JSON-serialized form (list of dicts) of any of:
    UserMessage / AssistantMessage / ToolResultMessage content blocks.
    Sums ``text`` lengths from text / thinking blocks, ``arguments`` JSON
    length from tool-call blocks, and recurses into ``content`` for
    tool-result blocks. Image blocks contribute their decoded byte length.
    """

    if not isinstance(content, list):
        return 0
    total = 0
    for block in content:
        if not isinstance(block, Mapping):
            continue
        btype = block.get("type")
        if btype in ("text", "thinking"):
            text = block.get("text")
            if isinstance(text, str):
                total += len(text)
        elif btype == "tool_call":
            args = block.get("arguments")
            total += len(_stable_serialize(args))
        elif btype == "tool_result":
            total += _message_text_chars(block.get("content"))
        elif btype == "image":
            # to_jsonable rendered bytes as {"type": "bytes", "base64": "..."}.
            data = block.get("data")
            if isinstance(data, Mapping):
                b64 = data.get("base64")
                if isinstance(b64, str):
                    total += len(b64)
    return total


def _stable_serialize(value: Any) -> str:
    """Deterministic string form for hashing. ``sort_keys=True`` so
    structurally-equal payloads hash identically across runs."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _redact_message(message: Any) -> dict[str, Any]:
    """Turn one serialized message dict into a {role, chars, sha256_prefix} stub.

    Non-dict inputs are represented by a stable hash of their encoded shape.
    """

    if not isinstance(message, Mapping):
        return {
            "role": "unknown",
            "chars": 0,
            "sha256_prefix": _sha256_prefix(_stable_serialize(message)),
        }
    role = message.get("role", "unknown")
    chars = _message_text_chars(message.get("content"))
    return {
        "role": role,
        "chars": chars,
        "sha256_prefix": _sha256_prefix(_stable_serialize(message)),
    }


def _redact_system(system: Any) -> Any:
    """Stub form for the ``system`` prompt. ``None`` passes through."""

    if system is None:
        return None
    if isinstance(system, str):
        return {"chars": len(system), "sha256_prefix": _sha256_prefix(system)}
    # Defensive: some atoms may pre-serialize system into a structured form.
    serialized = _stable_serialize(system)
    return {"chars": len(serialized), "sha256_prefix": _sha256_prefix(serialized)}


def redact_messages(payload: Any) -> Any:
    """Replace ``messages`` / ``system`` fields in a serialized event dict
    with ``{chars, sha256_prefix}`` stubs.

    Operates on the **already-JSON-serialized** form (post ``to_jsonable``).
    Returns a new dict; ``payload`` is not mutated. Unknown payload shapes
    pass through unchanged so this is safe to apply unconditionally to any
    event in the LLM channel.

    Tool *schemas* (``tools`` field) are preserved — they describe the
    available toolset, not secret-bearing invocation arguments.

    Operational metadata is preserved: ``model``, ``model_id``,
    ``turn_index``, ``message_count``, ``tool_count``, ``system_chars``,
    ``duration_ns``, ``error``, ``usage``, etc.
    """

    if not isinstance(payload, Mapping):
        return payload
    out: dict[str, Any] = dict(payload)
    if "messages" in out:
        raw = out["messages"]
        if isinstance(raw, list):
            out["messages"] = [_redact_message(m) for m in raw]
    if "system" in out:
        out["system"] = _redact_system(out["system"])
    # Per-event content fields (e.g. ApiSendUserMessageEvent.content) —
    # collapse to a chars/sha stub when present and string-shaped.
    if "content" in out:
        out["content"] = _redact_content_field(out["content"])
    return out


def _redact_content_field(content: Any) -> Any:
    """Stub a free-form ``content`` payload (str / dict / list)."""

    if content is None:
        return None
    if isinstance(content, str):
        return {"chars": len(content), "sha256_prefix": _sha256_prefix(content)}
    serialized = _stable_serialize(content)
    return {
        "chars": len(serialized),
        "sha256_prefix": _sha256_prefix(serialized),
        "shape": type(content).__name__,
    }


def redact_headers(headers: Any) -> Any:
    """Return a copy of ``headers`` with auth-bearing values replaced by ``"***"``.

    Case-insensitive match against :data:`SENSITIVE_HEADER_NAMES`. The key
    is preserved verbatim — operators need to know *which* header was set,
    just not its value. Non-mapping inputs (or ``None``) pass through.
    """

    if not isinstance(headers, Mapping):
        return headers
    out: dict[str, Any] = {}
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() in SENSITIVE_HEADER_NAMES:
            out[key] = "***"
        else:
            out[key] = value
    return out


def redact_config(value: Any) -> Any:
    """Recursively redact credentials from user-visible config data."""

    if isinstance(value, Mapping):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            normalized = key.lower() if isinstance(key, str) else ""
            if normalized in _SENSITIVE_CONFIG_NAMES or normalized.endswith(
                _SENSITIVE_CONFIG_SUFFIXES
            ):
                redacted[key] = "***"
            elif normalized.endswith("headers"):
                redacted[key] = redact_config(redact_headers(item))
            else:
                redacted[key] = redact_config(item)
        return redacted
    if isinstance(value, list):
        return [redact_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_config(item) for item in value)
    return value
