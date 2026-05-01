"""PKCE utilities — port of pi-mono `utils/oauth/pkce.ts`.

Uses Python stdlib ``secrets`` + ``hashlib`` instead of Web Crypto.
Output is base64url-encoded (no padding) per RFC 7636.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class PKCEPair:
    verifier: str
    challenge: str


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def generate_pkce() -> PKCEPair:
    """Return a fresh PKCE verifier+challenge pair (S256)."""

    verifier_bytes = secrets.token_bytes(32)
    verifier = _base64url(verifier_bytes)
    challenge = _base64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return PKCEPair(verifier=verifier, challenge=challenge)
