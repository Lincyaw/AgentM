"""Run the trajectory grounding analysis (Pass 1-3.5) on AFTraj-2K and inspect it.

Unlike ``eval_auditor.py`` (which drives an LLM auditor), this exercises the
deterministic-plus-local-judgment grounding pipeline in ``trajectory_index``:
Pass 1 extract+classify, Pass 2a alias, Pass 2b coreference, Pass 3 def-use,
Pass 3.5 value fidelity — and reports the built index, not a verdict.

Usage:
    uv run python contrib/evals/aftraj/eval_grounding.py inspect --n 3 --domain coding
    uv run python contrib/evals/aftraj/eval_grounding.py stability --n 3 --runs 4
    uv run python contrib/evals/aftraj/eval_grounding.py run --n 6
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer

app = typer.Typer(help="Run trajectory grounding analysis on AFTraj-2K.")

_DEFAULT_DATA_DIR = Path.home() / "AoyangSpace/references/agent-foresight/AFTraj"
_DEFAULT_MODEL = "azure-gpt"
_DEFAULT_VOCAB = "multi_agent"


# ---------------------------------------------------------------------------
# AFTraj turns -> agentm messages (proper block structure)
# ---------------------------------------------------------------------------


def aftraj_to_messages(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map AFTraj turns to messages preserving block structure so the def/use +
    grounding tagging works: ``environment`` -> tool_result (grounded), agent
    ``action`` -> tool_call (use), thought/content -> text (mention)."""
    msgs: list[dict[str, Any]] = []
    for i, t in enumerate(turns):
        role = t.get("role", "unknown")
        content = (t.get("content") or "").strip()
        thought = (t.get("thought") or "").strip()
        action = t.get("action")
        if role == "user":
            msgs.append({"id": str(i), "role": "user",
                         "content": [{"type": "text", "text": content or "(empty)"}]})
        elif role == "environment":
            msgs.append({"id": str(i), "role": "tool_result",
                         "content": [{"type": "tool_result",
                                      "content": [{"type": "text", "text": content or "(empty)"}],
                                      "is_error": False}]})
        else:
            blocks: list[dict[str, Any]] = []
            if thought:
                blocks.append({"type": "text", "text": f"[Thought] {thought}"})
            if action:
                for it in (action if isinstance(action, list) else [action]):
                    if isinstance(it, dict):
                        blocks.append({"type": "tool_call", "name": str(it.get("name", "tool")),
                                       "arguments": it.get("arguments", {})})
                    else:
                        blocks.append({"type": "tool_call", "name": "action", "arguments": str(it)})
            if content:
                blocks.append({"type": "text", "text": content})
            msgs.append({"id": str(i), "role": "assistant",
                         "content": blocks or [{"type": "text", "text": "(empty)"}]})
    return msgs


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


async def build_index(msgs: list[dict[str, Any]], *, model: str, vocab: str, full: bool, run_id: str) -> Any:
    """Pass 1 (extract+classify); with ``full`` also run 2a/2b/3/3.5."""
    from trajectory_index.data import build_index_from_chunks, extract_incremental

    chunks = await extract_incremental(msgs, model=model, run_id=run_id, chunk_size=(4, 6), vocabulary=vocab)
    idx = build_index_from_chunks([chunks])
    if full:
        from agentm.core.runtime.session import AgentSession
        from trajectory_index.adjudicate import compare_values, resolve_aliases, resolve_references

        sf = AgentSession.create
        groups = await resolve_aliases(idx, model=model, apply=False, session_factory=sf)
        if groups:
            idx.apply_alias_merges(groups)
        await resolve_references(idx, model=model, apply=False, session_factory=sf)
        idx.build_dependencies()
        await compare_values(idx, model=model, apply=True, session_factory=sf)
    else:
        idx.build_dependencies()
    return idx


def _load(data_dir: Path, domain: str | None, n: int) -> list[Any]:
    df = pd.read_parquet(data_dir / "aftraj_unsafe.parquet")
    if domain:
        df = df[df.domain == domain]
    return [row for _, row in df.head(n).iterrows()]


def _turns(row: Any) -> list[dict[str, Any]]:
    t = row["turns"]
    return json.loads(t) if isinstance(t, str) else list(t)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def inspect(
    n: Annotated[int, typer.Option(help="trajectories per domain (or total if --domain)")] = 2,
    domain: Annotated[str, typer.Option(help="agentic|coding|math; empty = all three")] = "",
    fast: Annotated[bool, typer.Option(help="skip the LLM alias/coref/value passes for a "
                                            "quick Pass 1+3 peek (no anaphors/contradicted)")] = False,
    model: Annotated[str, typer.Option()] = _DEFAULT_MODEL,
    vocab: Annotated[str, typer.Option()] = _DEFAULT_VOCAB,
    data_dir: Annotated[Path, typer.Option()] = _DEFAULT_DATA_DIR,
) -> None:
    """Dump the complete built index (symbols + class + grounding + def-use edges).

    Runs the full pipeline by default so the index includes coreference anaphors and
    value-fidelity (contradicted) edges. --fast skips those three LLM passes for a
    quick look at just extraction + classification + name-axis def-use.
    """
    doms = [domain] if domain else ["agentic", "coding", "math"]
    rows = [r for d in doms for r in _load(data_dir, d, n)]

    async def go() -> None:
        idxs = await asyncio.gather(*[
            build_index(aftraj_to_messages(_turns(r)), model=model, vocab=vocab, full=not fast,
                        run_id=str(r.conv_id)[:16]) for r in rows
        ])
        for r, idx in zip(rows, idxs, strict=True):
            ks = int(r.mistake_step)
            print(f"\n{'='*72}\n{r.conv_id} [{r.domain}] k*={ks} ({r.mistake_agent}) validate={idx.validate() or 'OK'}")
            for s in idx.symbols.values():
                refs = idx.get_references(s.id)
                steps = ",".join(sorted({rr.step_id for rr in refs}, key=int))
                at_k = " <-k*" if any(rr.step_id == str(ks) for rr in refs) else ""
                g = any(rr.grounded for rr in refs)
                print(f"  {s.canonical_name[:32]:32} {s.kind:10} {s.entity_class:10} [{steps}] grounded={g}{at_k}")
            for d in idx.get_dependencies():
                if d.risk != "grounded":
                    print(f"    edge {idx.symbols[d.symbol_id].canonical_name[:28]:28} {d.def_step_id}->{d.use_step_id} v{d.def_version} {d.risk}")

    asyncio.run(go())


@app.command()
def stability(
    n: Annotated[int, typer.Option(help="trajectories per domain")] = 1,
    runs: Annotated[int, typer.Option(help="repeat count per trajectory")] = 4,
    domain: Annotated[str, typer.Option()] = "",
    model: Annotated[str, typer.Option()] = _DEFAULT_MODEL,
    vocab: Annotated[str, typer.Option()] = _DEFAULT_VOCAB,
    data_dir: Annotated[Path, typer.Option()] = _DEFAULT_DATA_DIR,
) -> None:
    """Extract each trajectory ``runs`` times; report entity_class stability on shared entities."""
    doms = [domain] if domain else ["agentic", "coding", "math"]
    rows = [r for d in doms for r in _load(data_dir, d, n)]

    async def one(msgs: list[dict[str, Any]], rid: str) -> dict[str, tuple[str, str]]:
        idx = await build_index(msgs, model=model, vocab=vocab, full=False, run_id=rid)
        return {s.canonical_name: (s.kind, s.entity_class) for s in idx.symbols.values()}

    async def go() -> None:
        tasks = []
        for r in rows:
            m = aftraj_to_messages(_turns(r))
            tasks += [one(m, f"{str(r.conv_id)[:12]}_{k}") for k in range(runs)]
        res = await asyncio.gather(*tasks)
        pos = 0
        for r in rows:
            maps = res[pos:pos + runs]
            pos += runs
            names = sorted(set().union(*[set(m) for m in maps]))
            shared = [nm for nm in names if all(nm in m for m in maps)]
            stable = sum(1 for nm in shared if len({m[nm][1] for m in maps}) == 1)
            print(f"\n### {r.conv_id} [{r.domain}] {runs} runs | entity_class stable {stable}/{len(shared)} shared")
            for nm in names:
                ecs = {m[nm][1] for m in maps if nm in m}
                if len(ecs) > 1:
                    cells = " ".join(f"{m.get(nm, ('-', '-'))[0]}/{m.get(nm, ('-', '-'))[1]}" for m in maps)
                    print(f"    UNSTABLE {nm[:34]:34} {cells}")

    asyncio.run(go())


@app.command()
def run(
    n: Annotated[int, typer.Option(help="trajectories per domain")] = 2,
    domain: Annotated[str, typer.Option()] = "",
    model: Annotated[str, typer.Option()] = _DEFAULT_MODEL,
    vocab: Annotated[str, typer.Option()] = _DEFAULT_VOCAB,
    data_dir: Annotated[Path, typer.Option()] = _DEFAULT_DATA_DIR,
) -> None:
    """Full pipeline; report per-trajectory signal counts (ungrounded / contradicted / anaphors)."""
    import collections

    doms = [domain] if domain else ["agentic", "coding", "math"]
    rows = [r for d in doms for r in _load(data_dir, d, n)]

    async def go() -> None:
        idxs = await asyncio.gather(*[
            build_index(aftraj_to_messages(_turns(r)), model=model, vocab=vocab, full=True,
                        run_id=str(r.conv_id)[:16]) for r in rows
        ])
        for r, idx in zip(rows, idxs, strict=True):
            deps = idx.get_dependencies()
            risk = collections.Counter(d.risk for d in deps)
            anaphors = sum(1 for rr in idx.references.values() if rr.form == "anaphor")
            print(f"{r.conv_id[:38]:38} [{r.domain}] k*={r.mistake_step:>2} | "
                  f"{len(idx.symbols)} syms {len(deps)} deps anaphors={anaphors} | {dict(risk)}")

    asyncio.run(go())


if __name__ == "__main__":
    app()
