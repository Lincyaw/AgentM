"""Unit tests for :mod:`agentm.core.lib.redact`.

The redactor is the load-bearing guard between user-pasted secrets and the
``.agentm/observability/*.jsonl`` files that ship inside evidence bundles.
Tests assert the three invariants:

1. Secret strings inside a serialized ``BeforeSendToLlmEvent`` are gone
   from the redacted output (only ``chars`` + ``sha256_prefix`` remain).
2. The sha256 prefix is deterministic across calls for identical input
   (so trace consumers can correlate "same prompt sent twice").
3. ``redact_headers`` strips Authorization-class header *values* but
   preserves the *keys* — operators must still see which credentials
   were configured.
"""

from __future__ import annotations

from agentm.core.lib.redact import (
    SENSITIVE_HEADER_NAMES,
    redact_headers,
    redact_messages,
)


SECRET = "sk-proj-FAKE_SECRET_xxx"


def _user_message(text: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}],
        "timestamp": 1234.0,
    }


def _assistant_with_tool_call(text: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "tool_call",
                "id": "tc1",
                "name": "bash",
                "arguments": args,
            },
        ],
        "stop_reason": "end_turn",
    }


def _llm_event_dict(messages: list[dict], system: str | None = None) -> dict:
    """Shape that mirrors ``to_jsonable(BeforeSendToLlmEvent(...))``."""

    return {
        "messages": messages,
        "system": system,
        "model": {"id": "claude-x", "name": "x"},
        "tools": [{"name": "bash", "description": "run shell"}],
    }


def test_redact_messages_removes_secret_text() -> None:
    payload = _llm_event_dict(
        [_user_message(f"please ignore this: {SECRET}")],
        system=f"system prompt with {SECRET}",
    )

    redacted = redact_messages(payload)

    # Secret string MUST NOT appear anywhere in the redacted output.
    import json as _json

    flat = _json.dumps(redacted)
    assert SECRET not in flat
    # Stub shape is preserved with non-zero char count.
    assert redacted["messages"][0]["role"] == "user"
    assert redacted["messages"][0]["chars"] > 0
    assert len(redacted["messages"][0]["sha256_prefix"]) == 16
    # System prompt collapses to stub but keeps its char count.
    assert redacted["system"]["chars"] > 0
    assert len(redacted["system"]["sha256_prefix"]) == 16
    # Tool *schemas* are preserved verbatim — schemas are not secret.
    assert redacted["tools"] == payload["tools"]
    # Operational metadata (model) is preserved verbatim.
    assert redacted["model"] == payload["model"]


def test_redact_messages_strips_secret_inside_tool_call_arguments() -> None:
    """Secrets pasted as tool-call arguments must also be redacted."""

    payload = _llm_event_dict(
        [
            _assistant_with_tool_call(
                "calling bash",
                {"cmd": f"echo {SECRET}"},
            )
        ]
    )

    redacted = redact_messages(payload)

    import json as _json

    assert SECRET not in _json.dumps(redacted)
    # Tool-call body contributes to char count via the JSON-serialized
    # ``arguments`` — the stub records nonzero chars even though no plain
    # text block was present.
    assert redacted["messages"][0]["chars"] > len("calling bash")


def test_redact_messages_passes_through_unknown_shape() -> None:
    # Non-mapping inputs (e.g. a dataclass that wasn't serialized) return
    # unchanged so the redactor is safe to apply unconditionally.
    assert redact_messages("not a dict") == "not a dict"
    assert redact_messages(None) is None


def test_redact_messages_handles_none_system() -> None:
    payload = _llm_event_dict([_user_message("hello")], system=None)
    redacted = redact_messages(payload)
    assert redacted["system"] is None


def test_redact_messages_redacts_freeform_content_field() -> None:
    """``ApiSendUserMessageEvent.content`` is free-form; the redactor
    should reduce it to a chars/sha stub when present."""

    payload = {"extension": "x", "content": f"line one\n{SECRET}\n"}
    redacted = redact_messages(payload)
    assert redacted["extension"] == "x"
    assert redacted["content"]["chars"] == len(payload["content"])
    assert SECRET not in str(redacted)


def test_sha256_prefix_is_deterministic() -> None:
    """Same content → same prefix; lets consumers correlate duplicate
    prompts across runs without seeing the prompt itself."""

    payload = _llm_event_dict(
        [_user_message("hello world")], system="be helpful"
    )

    a = redact_messages(payload)
    b = redact_messages(payload)
    assert a == b
    # And the prefix is the canonical sha256-16 of the serialized message.
    import hashlib
    import json as _json

    expected = hashlib.sha256(
        _json.dumps(
            payload["messages"][0], sort_keys=True, default=str, ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()[:16]
    assert a["messages"][0]["sha256_prefix"] == expected


def test_redact_headers_strips_sensitive_values_any_casing() -> None:
    headers = {
        "Authorization": "Bearer SECRET_TOKEN_xyz",
        "x-api-key": "abcdef",
        "X-Auth-Token": "zzz",
        "Cookie": "session=privatevalue",
        "Content-Type": "application/json",
        "User-Agent": "agentm/0.1",
    }

    scrubbed = redact_headers(headers)

    # Sensitive values gone, replaced with "***".
    assert scrubbed["Authorization"] == "***"
    assert scrubbed["x-api-key"] == "***"
    assert scrubbed["X-Auth-Token"] == "***"
    assert scrubbed["Cookie"] == "***"
    # Non-sensitive headers pass through unchanged.
    assert scrubbed["Content-Type"] == "application/json"
    assert scrubbed["User-Agent"] == "agentm/0.1"
    # Original dict is not mutated.
    assert headers["Authorization"] == "Bearer SECRET_TOKEN_xyz"


def test_redact_headers_passes_through_non_mapping() -> None:
    assert redact_headers(None) is None
    assert redact_headers("not a dict") == "not a dict"


def test_sensitive_header_names_are_lowercase() -> None:
    """The matcher is case-insensitive; the canonical set is lowercase
    so the comparison is a simple ``key.lower() in SENSITIVE_HEADER_NAMES``."""

    assert all(name == name.lower() for name in SENSITIVE_HEADER_NAMES)
