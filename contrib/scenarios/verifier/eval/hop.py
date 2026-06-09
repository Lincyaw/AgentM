"""Hop agent execution: run a single edge verification agent via SDK."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

from verdict import extract_hop_verdict

REPO = Path(__file__).resolve().parents[4]

# Re-run a hop that returns no verdict, granting extra tool-calls each try.
HOP_MAX_ATTEMPTS = 3
HOP_BUDGET_BUMP = 5

# ---------------------------------------------------------------------------
# Lazy import of the prompt builder from the verifier_hop scenario package.
# The scenario directory is not on sys.path, so we use importlib.util to
# load it by file path.
# ---------------------------------------------------------------------------

_prompt_module: ModuleType | None = None


def _get_prompt_module() -> ModuleType:
    global _prompt_module  # noqa: PLW0603
    if _prompt_module is not None:
        return _prompt_module
    prompt_path = REPO / "contrib" / "scenarios" / "verifier_hop" / "prompt.py"
    spec = importlib.util.spec_from_file_location("verifier_hop.prompt", prompt_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load prompt module from {prompt_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _prompt_module = mod
    return mod


def _build_hop_prompt(
    from_service: str,
    to_service: str,
    rel_type: str,
    fault_kind: str,
    injection_target: str,
    all_faults: list[tuple[str, str, str]],
    fault_docs: dict[str, str],
    is_infra: bool,
    upstream_evidence: dict | None,
) -> str:
    mod = _get_prompt_module()
    return mod.build_hop_prompt(  # type: ignore[no-any-return]
        from_service=from_service,
        to_service=to_service,
        rel_type=rel_type,
        fault_kind=fault_kind,
        injection_target=injection_target,
        all_faults=all_faults,
        fault_docs=fault_docs,
        is_infra=is_infra,
        upstream_evidence=upstream_evidence,
    )


# ---------------------------------------------------------------------------
# SDK session helpers
# ---------------------------------------------------------------------------


def _resolve_provider() -> tuple[str, dict[str, object]]:
    """Build a provider spec from the environment (config.toml profile).

    Mirrors the CLI's ``_resolve_provider_model_cwd`` logic: reads the
    ``AGENTM_MODEL`` env var (or falls back to the config.toml
    ``default_model``), resolves the profile, and builds the provider
    extension spec via the registry.
    """
    from agentm.ai import DEFAULT_PROVIDER_REGISTRY
    from agentm.core.lib.user_config import resolve_model_profile

    model_name = os.environ.get("AGENTM_MODEL")
    profile = resolve_model_profile(model_name)
    if profile is not None:
        build_config = profile.to_build_config()
        provider_id = os.environ.get("AGENTM_PROVIDER") or profile.provider
    else:
        registry = DEFAULT_PROVIDER_REGISTRY
        provider_id = os.environ.get("AGENTM_PROVIDER") or registry.default_provider().id
        build_config = {"model": model_name or registry.default_model(provider_id)}

    return DEFAULT_PROVIDER_REGISTRY.build(provider_id, build_config)


def _extract_verdict_from_messages(messages: list) -> dict | None:  # type: ignore[type-arg]
    """Extract the hop verdict from the session's final messages.

    The ``submit_hop_verdict`` tool returns a ``ToolTerminate`` whose result
    is serialised as a ``ToolResultMessage`` containing the JSON-encoded
    verdict. We scan backwards for the last non-error tool result whose
    text parses as a verdict dict.
    """
    for msg in reversed(messages):
        if getattr(msg, "role", None) != "tool_result":
            continue
        for block in getattr(msg, "content", []):
            if getattr(block, "type", None) != "tool_result":
                continue
            if getattr(block, "is_error", False):
                continue
            for inner in getattr(block, "content", []):
                if getattr(inner, "type", None) != "text":
                    continue
                text = getattr(inner, "text", "")
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(obj, dict) and "verdict" in obj:
                    return obj
    return None


async def _run_hop_async(
    data_dir: Path,
    hop_dir: Path,
    prompt: str,
    budget: int,
) -> dict | None:
    """Run a single hop session via the SDK and return the verdict."""
    from agentm.core.abi import LoopConfig
    from agentm.core.abi.session_config import AgentSessionConfig
    from agentm.core.runtime.session import AgentSession

    os.environ["AGENTM_PROJECT_ROOT"] = str(REPO)
    os.environ["AGENTM_RCA_DATA_DIR"] = str(data_dir)

    provider_spec = _resolve_provider()
    config = AgentSessionConfig(
        cwd=str(hop_dir),
        provider=provider_spec,
        scenario="verifier_hop",
        loop_config=LoopConfig(max_tool_calls=budget),
        auto_commit=False,
    )
    session = await AgentSession.create(config)
    try:
        messages = await session.prompt(prompt)
        return _extract_verdict_from_messages(messages)
    finally:
        await session.shutdown()


def run_hop(
    data_dir: Path,
    from_service: str,
    to_service: str,
    rel_type: str,
    fault_kind: str,
    injection_target: str,
    all_faults: list[tuple[str, str, str]],
    fault_docs: dict[str, str],
    out_dir: Path,
    budget: int,
    is_infra: bool = False,
    upstream_evidence: dict | None = None,
) -> dict | None:
    """Run one hop-agent and return its verdict dict (or None).

    *all_faults* is every injected fault as ``(kind, target, params)``;
    *fault_docs* maps each kind to its reference doc. *fault_kind*/
    *injection_target* name the fault this edge's BFS branch descended
    from — used only to order the docs (that fault's reference first);
    the prompt itself lists every fault flat, since a node downstream of
    two coexisting faults must be judged against all of them.
    """
    hop_dir = out_dir / "hops" / f"{from_service}__{to_service}"
    hop_dir.mkdir(parents=True, exist_ok=True)

    prompt = _build_hop_prompt(
        from_service=from_service,
        to_service=to_service,
        rel_type=rel_type,
        fault_kind=fault_kind,
        injection_target=injection_target,
        all_faults=all_faults,
        fault_docs=fault_docs,
        is_infra=is_infra,
        upstream_evidence=upstream_evidence,
    )

    obs_dir = hop_dir / ".agentm" / "observability"
    verdict: dict | None = None
    for attempt in range(HOP_MAX_ATTEMPTS):
        attempt_budget = budget + attempt * HOP_BUDGET_BUMP
        try:
            verdict = asyncio.run(
                _run_hop_async(data_dir, hop_dir, prompt, attempt_budget)
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"    sdk-error {from_service} -> {to_service} "
                f"(attempt {attempt + 1}/{HOP_MAX_ATTEMPTS}): {exc}"
            )
        # Fall back to JSONL extraction if SDK extraction missed it
        if verdict is None and obs_dir.exists():
            verdict = extract_hop_verdict(obs_dir)
        if verdict:
            break
        print(
            f"    no-result {from_service} -> {to_service} "
            f"(attempt {attempt + 1}/{HOP_MAX_ATTEMPTS}, "
            f"budget={attempt_budget})"
        )
    if verdict:
        (hop_dir / "verdict.json").write_text(
            json.dumps(verdict, ensure_ascii=False, indent=2)
        )
    return verdict
