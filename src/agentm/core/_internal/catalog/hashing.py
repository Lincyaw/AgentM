"""Content hashing helpers for the catalog contract."""

from __future__ import annotations

import hashlib

from agentm.core.abi.catalog import ActiveSetFingerprint


def compute_atom_hash(source: str) -> str:
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return digest[:12]



def compute_active_set_fingerprint(
    loaded: dict[str, str],
    scenario: str | None,
    core_hash: str | None,
) -> ActiveSetFingerprint:
    atoms = {
        name: f"{name}@{content_hash}"
        for name, content_hash in sorted(loaded.items())
    }
    return {
        "core": f"core@{core_hash}" if core_hash is not None else None,
        "scenario": scenario,
        "atoms": atoms,
    }
