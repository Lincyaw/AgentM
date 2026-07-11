"""AFTraj-2K grounding pipeline evaluation adapter.

Exercises the deterministic grounding pipeline (Pass 1-3.5) on AFTraj
trajectories and reports the built index, not a verdict.
"""

from __future__ import annotations

import asyncio
import collections
import json
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from agentm_eval.experiment import experiment_context
from agentm_eval.registry import register
from agentm_eval.result import TaskResult

from .auditor import aftraj_to_messages

_DEFAULT_DATA_DIR = Path.home() / "AoyangSpace/references/agent-foresight/AFTraj"
_DEFAULT_MODEL = "azure-gpt"
_DEFAULT_VOCAB = "multi_agent"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


async def build_grounding_index(
    msgs: list[dict[str, Any]], *, model: str, vocab: str, full: bool, run_id: str,
) -> Any:
    from agentm_eval.methods.index import build_index, extract_symbols

    chunks = await extract_symbols(
        msgs, model=model, run_id=run_id, chunk_size=(4, 6), vocabulary=vocab,
    )
    return await build_index(chunks, model=model, resolve=full)


def _load(data_dir: Path, domain: str | None, n: int) -> list[Any]:
    import pandas as pd

    df = pd.read_parquet(data_dir / "aftraj_unsafe.parquet")
    if domain:
        df = df[df.domain == domain]
    return [row for _, row in df.head(n).iterrows()]


def _turns(row: Any) -> list[dict[str, Any]]:
    t = row["turns"]
    return json.loads(t) if isinstance(t, str) else list(t)


# ---------------------------------------------------------------------------
# CLI adapter
# ---------------------------------------------------------------------------


class AftrajGroundingAdapter:
    name = "aftraj-grounding"
    description = "AFTraj-2K trajectory grounding pipeline evaluation"

    def create_cli(self) -> typer.Typer:
        cli = typer.Typer(
            name="aftraj-grounding",
            help="Run trajectory grounding analysis on AFTraj-2K.",
            add_completion=False,
        )

        @cli.command()
        def inspect(
            n: Annotated[int, typer.Option(help="Trajectories per domain")] = 2,
            domain: Annotated[str, typer.Option(help="agentic|coding|math")] = "",
            fast: Annotated[bool, typer.Option(help="Skip LLM alias/coref/value passes")] = False,
            model: Annotated[str, typer.Option()] = _DEFAULT_MODEL,
            vocab: Annotated[str, typer.Option()] = _DEFAULT_VOCAB,
            data_dir: Annotated[Path, typer.Option()] = _DEFAULT_DATA_DIR,
        ) -> None:
            """Dump the complete built index."""
            doms = [domain] if domain else ["agentic", "coding", "math"]
            rows = [r for d in doms for r in _load(data_dir, d, n)]

            async def go() -> None:
                idxs = await asyncio.gather(*[
                    build_grounding_index(
                        aftraj_to_messages(_turns(r)), model=model, vocab=vocab,
                        full=not fast, run_id=str(r.conv_id)[:16],
                    ) for r in rows
                ])
                for r, idx in zip(rows, idxs, strict=True):
                    ks = int(r.mistake_step)
                    print(
                        f"\n{'='*72}\n{r.conv_id} [{r.domain}] k*={ks} "
                        f"({r.mistake_agent}) validate={idx.validate() or 'OK'}"
                    )
                    for s in idx.symbols.values():
                        refs = idx.get_references(s.id)
                        steps = ",".join(sorted({rr.step_id for rr in refs}, key=int))
                        at_k = " <-k*" if any(rr.step_id == str(ks) for rr in refs) else ""
                        g = any(rr.grounded for rr in refs)
                        print(
                            f"  {s.canonical_name[:32]:32} {s.kind:10} "
                            f"{s.entity_class:10} [{steps}] grounded={g}{at_k}"
                        )
                    for d in idx.get_dependencies():
                        if d.risk != "grounded":
                            print(
                                f"    edge {idx.symbols[d.symbol_id].canonical_name[:28]:28} "
                                f"{d.def_step_id}->{d.use_step_id} v{d.def_version} {d.risk}"
                            )

            asyncio.run(go())

        @cli.command()
        def stability(
            n: Annotated[int, typer.Option(help="Trajectories per domain")] = 1,
            runs: Annotated[int, typer.Option(help="Repeat count per trajectory")] = 4,
            domain: Annotated[str, typer.Option()] = "",
            model: Annotated[str, typer.Option()] = _DEFAULT_MODEL,
            vocab: Annotated[str, typer.Option()] = _DEFAULT_VOCAB,
            data_dir: Annotated[Path, typer.Option()] = _DEFAULT_DATA_DIR,
        ) -> None:
            """Extract each trajectory multiple times; report entity_class stability."""
            doms = [domain] if domain else ["agentic", "coding", "math"]
            rows = [r for d in doms for r in _load(data_dir, d, n)]

            async def one(msgs: list[dict[str, Any]], rid: str) -> dict[str, tuple[str, str]]:
                idx = await build_grounding_index(msgs, model=model, vocab=vocab, full=False, run_id=rid)
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
                    print(
                        f"\n### {r.conv_id} [{r.domain}] {runs} runs | "
                        f"entity_class stable {stable}/{len(shared)} shared"
                    )
                    for nm in names:
                        ecs = {m[nm][1] for m in maps if nm in m}
                        if len(ecs) > 1:
                            cells = " ".join(
                                f"{m.get(nm, ('-', '-'))[0]}/{m.get(nm, ('-', '-'))[1]}"
                                for m in maps
                            )
                            print(f"    UNSTABLE {nm[:34]:34} {cells}")

            asyncio.run(go())

        @cli.command()
        def run(
            n: Annotated[int, typer.Option(help="Trajectories per domain")] = 2,
            domain: Annotated[str, typer.Option()] = "",
            model: Annotated[str, typer.Option()] = _DEFAULT_MODEL,
            vocab: Annotated[str, typer.Option()] = _DEFAULT_VOCAB,
            data_dir: Annotated[Path, typer.Option()] = _DEFAULT_DATA_DIR,
            exp_id: Annotated[Optional[str], typer.Option(help="Override experiment ID")] = None,
        ) -> None:
            """Full pipeline; report per-trajectory signal counts."""
            doms = [domain] if domain else ["agentic", "coding", "math"]
            rows = [r for d in doms for r in _load(data_dir, d, n)]

            with experiment_context(
                "aftraj-grounding", model=model, exp_id=exp_id,
                domain=domain, n=n,
            ) as exp:
                async def go() -> None:
                    idxs = await asyncio.gather(*[
                        build_grounding_index(
                            aftraj_to_messages(_turns(r)), model=model, vocab=vocab,
                            full=True, run_id=str(r.conv_id)[:16],
                        ) for r in rows
                    ])
                    for r, idx in zip(rows, idxs, strict=True):
                        deps = idx.get_dependencies()
                        risk = collections.Counter(d.risk for d in deps)
                        anaphors = sum(
                            1 for rr in idx.references.values() if rr.form == "anaphor"
                        )
                        line = (
                            f"{r.conv_id[:38]:38} [{r.domain}] k*={r.mistake_step:>2} | "
                            f"{len(idx.symbols)} syms {len(deps)} deps "
                            f"anaphors={anaphors} | {dict(risk)}"
                        )
                        print(line)
                        exp.record_result(TaskResult(
                            task_id=str(r.conv_id),
                            status="pass",
                            score={
                                "n_symbols": len(idx.symbols),
                                "n_deps": len(deps),
                                "n_anaphors": anaphors,
                                "risk_counts": dict(risk),
                            },
                            metadata={"domain": r.domain, "mistake_step": int(r.mistake_step)},
                        ).to_dict())

                asyncio.run(go())

        return cli


register("aftraj-grounding", AftrajGroundingAdapter.description, AftrajGroundingAdapter)
