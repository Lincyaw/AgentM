"""Acceptance #6 — single-writer property of the rca_hgraph_store atom.

A second ``claim_write_handle`` call (with any token, matching the original
or not) must raise. The read service stays usable to anyone who can fetch
``rca.hgraph.read`` from the ExtensionAPI.

These are the fail-stop tests for design §7.4. If the single-writer property
breaks, two atoms can race on the graph and the falsification gate's
preconditions become merely advisory.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentm_rca.hfsm.atoms import rca_hgraph_store


class _StubAPI:
    """Minimal ``ExtensionAPI`` shim covering only the calls this atom makes."""

    def __init__(self) -> None:
        self._services: dict[str, Any] = {}

    def set_service(self, name: str, obj: Any) -> None:
        self._services[name] = obj

    def get_service(self, name: str) -> Any:
        return self._services.get(name)


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    rca_hgraph_store._reset_for_tests()
    yield
    rca_hgraph_store._reset_for_tests()


def test_first_claim_returns_handle_second_claim_raises() -> None:
    api = _StubAPI()
    rca_hgraph_store.install(api, {})
    token = api.get_service("rca.hgraph.write_token")
    assert isinstance(token, str) and token

    handle = rca_hgraph_store.claim_write_handle(token)
    assert handle is not None

    # Second claim with the same (now-consumed) token must raise.
    with pytest.raises(RuntimeError):
        rca_hgraph_store.claim_write_handle(token)




def test_claim_with_unknown_token_raises_then_legit_token_still_works() -> None:
    api = _StubAPI()
    rca_hgraph_store.install(api, {})
    token = api.get_service("rca.hgraph.write_token")

    # An attempt with a bogus token must raise (and the bogus token is
    # marked claimed so a retry with the same bogus value still raises).
    with pytest.raises(RuntimeError):
        rca_hgraph_store.claim_write_handle("wrong")
    with pytest.raises(RuntimeError):
        rca_hgraph_store.claim_write_handle("wrong")

    # The legitimate token is unaffected by tampering on a different
    # token — the one-shot rule is scoped per token, not registry-wide,
    # so multiple sessions in the same process can each redeem their own
    # token (regression test for #156). Per-token poisoning still applies:
    # claiming the legit token twice raises on the second call.
    rca_hgraph_store.claim_write_handle(token)
    with pytest.raises(RuntimeError):
        rca_hgraph_store.claim_write_handle(token)


def test_multiple_installs_each_redeem_their_own_token() -> None:
    """Regression for #156 — process-wide registry must not block legit sessions.

    Two ``install`` calls in the same process must each surface a unique
    write token, and each token must redeem its corresponding handle.
    """
    api_a = _StubAPI()
    rca_hgraph_store.install(api_a, {})
    token_a = api_a.get_service("rca.hgraph.write_token")

    api_b = _StubAPI()
    rca_hgraph_store.install(api_b, {})
    token_b = api_b.get_service("rca.hgraph.write_token")

    assert token_a != token_b
    handle_a = rca_hgraph_store.claim_write_handle(token_a)
    handle_b = rca_hgraph_store.claim_write_handle(token_b)
    assert handle_a is not handle_b


