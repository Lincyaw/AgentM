"""Tests for PKCE generation — verifier+challenge S256 contract."""

from __future__ import annotations

import base64
import hashlib

from agentm.ai.oauth import generate_pkce


def test_pkce_pair_matches_s256_relationship() -> None:
    pair = generate_pkce()

    expected_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(pair.verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    assert pair.challenge == expected_challenge


def test_pkce_pairs_are_unique_per_call() -> None:
    pairs = {generate_pkce().verifier for _ in range(8)}
    assert len(pairs) == 8


def test_verifier_is_base64url_no_padding() -> None:
    verifier = generate_pkce().verifier
    assert "=" not in verifier
    assert "+" not in verifier and "/" not in verifier
    # 32 random bytes → 43 base64url chars
    assert len(verifier) == 43
