"""How many times a provider actually calls the endpoint, counted at the wire.

The retry policy is a service an atom binds and both shipped providers consume
at construction. Everything about it was verified by reading the object off the
stream function -- which says it was wired, and nothing about what it does.

Counted here instead, against a local endpoint that answers with a retryable
status and records every request. Both providers, because they had the same
defect and there is no reason to trust that they were fixed the same way.
"""

from __future__ import annotations

import asyncio
import json
import socketserver
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

from agentm import AgentSession, AgentSessionConfig, ExtensionSpec

_HITS: list[str] = []
_STATUS = [429]


class _AlwaysRetryable(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
        _HITS.append(self.path)
        body = json.dumps({"error": {"message": "no", "type": "overloaded_error"}})
        self.send_response(_STATUS[0])
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args: object) -> None:
        del args


@pytest.fixture
def endpoint() -> Iterator[int]:
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _AlwaysRetryable)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()


def _provider(module: str, port: int) -> ExtensionSpec:
    return ExtensionSpec.from_module(
        f"agentm.extensions.builtin.{module}",
        config={
            "api_key": "probe",
            "base_url": f"http://127.0.0.1:{port}",
            "name": "probe",
            "model": "probe-model",
        },
    )


def _retry(max_retries: int) -> ExtensionSpec:
    return ExtensionSpec.from_module(
        "agentm.extensions.builtin.retry_policy",
        # A real schedule, compressed: the point is how many, not how long.
        config={"max_retries": max_retries, "base_delay": 0.01, "factor": 1.0},
    )


async def _requests_made(
    tmp_path: Path,
    extensions: list[ExtensionSpec],
) -> int:
    _HITS.clear()
    session = await AgentSession.create(
        AgentSessionConfig(cwd=str(tmp_path), extensions=extensions)
    )
    session.start()
    try:
        # The endpoint never succeeds, so the turn fails; what is under test is
        # what reached the wire before it did.
        await asyncio.wait_for(session.run("probe"), timeout=60)
    except Exception as exc:  # noqa: BLE001 - the failure is the point
        del exc
    finally:
        await session.shutdown()
    return len(_HITS)


@pytest.mark.parametrize("module", ["llm_openai", "llm_anthropic"])
@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.asyncio
async def test_a_declared_retry_budget_is_the_budget_that_is_spent(
    tmp_path: Path,
    endpoint: int,
    module: str,
    status: int,
) -> None:
    """The number in the config is the number of attempts.

    Both SDKs run their own retry loop -- ``DEFAULT_MAX_RETRIES`` is 2, so three
    attempts -- and the bound policy used to wrap *around* it, multiplying the
    two. A composition declaring ``max_retries=7`` sent 24 requests to a
    rate-limited endpoint on a backoff schedule nobody wrote, and the operator
    who wrote 7 had every reason to believe 8.

    ``503`` is here because the fix switches the SDK's loop off, and the SDK
    retried server errors that the policy's own predicate did not. A policy
    that covered less than the layer it replaced would read as a configuration
    and behave as a downgrade.
    """

    _STATUS[0] = status
    for declared in (0, 1, 2):
        made = await _requests_made(
            tmp_path, [_provider(module, endpoint), _retry(declared)]
        )
        assert made == declared + 1, (
            f"{module} at {status} declared {declared + 1} attempt(s) and made {made}"
        )


@pytest.mark.parametrize("module", ["llm_openai", "llm_anthropic"])
@pytest.mark.asyncio
async def test_a_composition_with_no_policy_keeps_the_sdk_default(
    tmp_path: Path,
    endpoint: int,
    module: str,
) -> None:
    """Declaring no policy has never meant "no retries", and still does not.

    The SDK's loop is switched off only when a policy is bound. Turning three
    attempts into one for every composition that never mentioned retries would
    be a fragility nobody asked for, so this pins the untouched case.
    """

    _STATUS[0] = 429
    assert await _requests_made(tmp_path, [_provider(module, endpoint)]) == 3
