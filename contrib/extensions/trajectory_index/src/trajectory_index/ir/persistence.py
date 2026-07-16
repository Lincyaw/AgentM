"""Index persistence — integrity validation + JSON dump/load.

Pure serialization: no pass logic. ``dump`` writes the full IR (validating
referential integrity first, warnings only); ``load`` rebuilds an index,
tolerating legacy field names and coercing every enum-valued field back
through the runtime value sets (an invalid value degrades to a safe default
with a log line, never a crash).

Functions take a ``TrajectoryIndex``; ``index.py`` exposes them as thin
methods so the public API is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ..pass3_folds.grounding import drives_defuse, grounded_from_kind
from .models import (
    _ACTION_OP_VALUES,
    _ENTITY_CLASS_VALUES,
    _FINDING_STATUS_VALUES,
    _REF_FORM_VALUES,
    _RISK_VALUES,
    Action,
    Claim,
    ClaimFinding,
    Constraint,
    ConstraintFinding,
    Dependency,
    Edge,
    EntityClass,
    FindingStatus,
    Location,
    Reference,
    RefForm,
    Relation,
    Risk,
    Step,
    StepRole,
    Symbol,
    normalize_name,
    stable_id,
)

if TYPE_CHECKING:
    from .index import TrajectoryIndex


def validate(index: TrajectoryIndex) -> list[str]:
    """Check referential integrity. Returns a list of error descriptions."""
    errors: list[str] = []
    for sym in index.symbols.values():
        if sym.definition_ref_id and sym.definition_ref_id not in index.references:
            errors.append(f"symbol {sym.id} definition_ref_id {sym.definition_ref_id} not found")
    for ref in index.references.values():
        if ref.symbol_id not in index.symbols:
            errors.append(f"reference {ref.id} points to missing symbol {ref.symbol_id}")
        if ref.grounds_ref_id and ref.grounds_ref_id not in index.references:
            errors.append(f"reference {ref.id} grounds_ref_id {ref.grounds_ref_id} not found")
    for rel in index.relations.values():
        if rel.from_symbol_id not in index.symbols:
            errors.append(f"relation {rel.id} from_symbol {rel.from_symbol_id} not found")
        if rel.to_symbol_id not in index.symbols:
            errors.append(f"relation {rel.id} to_symbol {rel.to_symbol_id} not found")
    for dep in index.dependencies.values():
        if dep.symbol_id not in index.symbols:
            errors.append(f"dependency {dep.id} symbol {dep.symbol_id} not found")
        if dep.def_ref_id not in index.references:
            errors.append(f"dependency {dep.id} def_ref {dep.def_ref_id} not found")
        if dep.use_ref_id not in index.references:
            errors.append(f"dependency {dep.id} use_ref {dep.use_ref_id} not found")
        if dep.def_ref_id in index.references and index.references[dep.def_ref_id].symbol_id != dep.symbol_id:
            errors.append(f"dependency {dep.id} def_ref belongs to {index.references[dep.def_ref_id].symbol_id}, not {dep.symbol_id}")
        if dep.use_ref_id in index.references and index.references[dep.use_ref_id].symbol_id != dep.symbol_id:
            errors.append(f"dependency {dep.id} use_ref belongs to {index.references[dep.use_ref_id].symbol_id}, not {dep.symbol_id}")
    return errors


def dump(index: TrajectoryIndex, path: str | Path) -> None:
    """Write the full index to a JSON file. Validates integrity first."""
    errors = validate(index)
    if errors:
        from loguru import logger

        for e in errors:
            logger.warning(f"index integrity: {e}")
    steps = [
        {
            "run_id": s.run_id,
            "step_id": s.step_id,
            "index": s.index,
            "role": s.role,
            "content": s.content,
            "tool_name": s.tool_name,
            "call_id": s.call_id,
            "timestamp": s.timestamp,
            "obs_regions": [list(span) for span in s.obs_regions],
            "metadata": dict(s.metadata) if s.metadata else {},
        }
        for s in index.steps.values()
    ]
    symbols = [
        {
            "id": s.id,
            "name": s.canonical_name,
            "kind": s.kind,
            "aliases": sorted(s.aliases) if s.aliases else [],
            "summary": s.summary,
            "definition_ref_id": s.definition_ref_id,
            "entity_class": s.entity_class,
            "namespace": str(s.metadata.get("namespace", "")),
            "metadata": dict(s.metadata) if s.metadata else {},
        }
        for s in index.symbols.values()
    ]
    refs = [
        {
            "id": r.id,
            "symbol_id": r.symbol_id,
            "run_id": r.run_id,
            "step_id": r.step_id,
            "start": r.location.start,
            "end": r.location.end,
            "text": r.text,
            "role": r.role,
            "kind": r.kind,
            "grounded": r.grounded,
            "grounds_ref_id": r.grounds_ref_id,
            "structured": r.structured,
            "form": r.form,
            "value": r.value,
            "resolved_from": r.resolved_from,
            "metadata": dict(r.metadata) if r.metadata else {},
        }
        for r in index.references.values()
    ]
    relations = [
        {
            "id": r.id,
            "from": r.from_symbol_id,
            "to": r.to_symbol_id,
            "type": r.type,
            "run_id": r.run_id,
            "step_id": r.step_id,
            "weight": r.weight,
            "metadata": dict(r.metadata) if r.metadata else {},
        }
        for r in index.relations.values()
    ]
    dependencies = [
        {
            "id": d.id,
            "symbol_id": d.symbol_id,
            "run_id": d.run_id,
            "def_step_id": d.def_step_id,
            "def_ref_id": d.def_ref_id,
            "def_version": d.def_version,
            "use_step_id": d.use_step_id,
            "use_ref_id": d.use_ref_id,
            "risk": d.risk,
            "grounded_by_step_id": d.grounded_by_step_id,
            "def_value": d.def_value,
            "use_value": d.use_value,
            "metadata": dict(d.metadata) if d.metadata else {},
        }
        for d in index.dependencies.values()
    ]
    claims = [
        {"id": c.id, "run_id": c.run_id, "step_id": c.step_id, "text": c.text, "role": c.role}
        for c in index.claims.values()
    ]
    edges = [
        {
            "id": e.id, "kind": e.kind, "run_id": e.run_id,
            "src": e.src, "dst": e.dst,
            "quote": e.quote, "evidence_position": e.evidence_position,
        }
        for e in index.edges.values()
    ]
    claim_findings = [
        {
            "claim_id": f.claim_id, "run_id": f.run_id, "step_id": f.step_id,
            "status": f.status, "edge_ids": list(f.edge_ids),
            "evidence_empty": f.evidence_empty,
        }
        for f in index.claim_findings
    ]
    constraints = [
        {
            "id": c.id,
            "description": c.description,
            "normalized": dict(c.normalized) if c.normalized else None,
        }
        for c in index.constraints.values()
    ]
    constraint_findings = [
        {
            "constraint_id": f.constraint_id,
            "candidate": f.candidate,
            "status": f.status,
            "evidence_step_ids": list(f.evidence_step_ids),
            "commit_step_id": f.commit_step_id,
            "first_assertion_step_id": f.first_assertion_step_id,
            "confidence": f.confidence,
            "confidence_source": f.confidence_source,
            "reason": f.reason,
        }
        for f in index.constraint_findings
    ]
    actions = [
        {
            "call_id": a.call_id,
            "step_id": a.step_id,
            "run_id": a.run_id,
            "tool_name": a.tool_name,
            "operation": a.operation,
            "targets": list(a.targets),
            "diffs": [list(d) for d in a.diffs],
        }
        for a in index.actions.values()
    ]
    value_flow = getattr(index, "_value_flow", None)
    if value_flow is None:
        from ..pass3_folds.value_flow import build_value_flow_sync

        value_flow = build_value_flow_sync(index)

    data = {
        "stats": {
            "steps": len(index.steps),
            "symbols": len(index.symbols),
            "references": len(index.references),
            "relations": len(index.relations),
            "dependencies": len(index.dependencies),
            "claims": len(index.claims),
            "edges": len(index.edges),
            "constraints": len(index.constraints),
            "constraint_findings": len(index.constraint_findings),
            "actions": len(index.actions),
            "value_timelines": len(value_flow["value_timelines"]),
            "iterations": len(value_flow["iterations"]),
            "constraint_checks": len(value_flow["constraint_checks"]),
            "indexed_message_count": index.indexed_message_count,
        },
        "steps": steps,
        "symbols": symbols,
        "references": refs,
        "relations": relations,
        "dependencies": dependencies,
        "claims": claims,
        "edges": edges,
        "claim_findings": claim_findings,
        "constraints": constraints,
        "constraint_findings": constraint_findings,
        "actions": actions,
        "value_flow": value_flow,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load(cls: type[TrajectoryIndex], path: str | Path) -> TrajectoryIndex:
    """Load an index from a JSON file written by :func:`dump`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    index = cls()
    index.indexed_message_count = data.get("stats", {}).get("indexed_message_count", 0)

    from loguru import logger as _log

    for s in data.get("steps", []):
        content = str(s.get("content", ""))
        raw_spans = s.get("obs_regions", s.get("obs_spans"))
        if isinstance(raw_spans, list):
            obs_regions = tuple(
                (int(sp[0]), int(sp[1]))
                for sp in raw_spans
                if isinstance(sp, list) and len(sp) == 2
            )
        elif s.get("provenance") == "observation":
            # legacy single-label format
            obs_regions = ((0, len(content)),)
        elif s.get("provenance") == "mixed" and isinstance(s.get("obs_offset"), int):
            obs_regions = ((int(s["obs_offset"]), len(content)),)
        else:
            obs_regions = ()
        step = Step(
            run_id=str(s.get("run_id", "")),
            step_id=str(s["step_id"]),
            index=int(s.get("index", 0)),
            role=str(s.get("role", StepRole.USER)),
            content=content,
            tool_name=s.get("tool_name") if isinstance(s.get("tool_name"), str) else None,
            call_id=s.get("call_id") if isinstance(s.get("call_id"), str) else None,
            timestamp=s.get("timestamp") if isinstance(s.get("timestamp"), int | float) else None,
            metadata=s.get("metadata", {}) if isinstance(s.get("metadata"), dict) else {},
            obs_regions=obs_regions,
        )
        index.add_step(step)

    for s in data.get("symbols", data.get("entities", [])):
        metadata = s.get("metadata", {}) if isinstance(s.get("metadata"), dict) else {}
        namespace = str(s.get("namespace") or metadata.get("namespace") or "")
        if namespace:
            metadata = {**metadata, "namespace": namespace}
        aliases = set(s.get("aliases", []))
        default_id = (
            stable_id("sym", namespace, normalize_name(str(s["name"])))
            if namespace
            else stable_id("sym", normalize_name(str(s["name"])))
        )
        symbol_id = str(s.get("id") or default_id)
        raw_ec = s.get("entity_class")
        entity_class: EntityClass = raw_ec if raw_ec in _ENTITY_CLASS_VALUES else "identifier"
        if raw_ec and raw_ec not in _ENTITY_CLASS_VALUES:
            _log.warning(f"load: symbol {s.get('name')!r} entity_class {raw_ec!r} invalid, defaulting to 'identifier'")
        symbol = Symbol(
            id=symbol_id,
            canonical_name=str(s["name"]).strip(),
            kind=str(s.get("kind", "unknown")),
            aliases=aliases,
            summary=s.get("summary") if isinstance(s.get("summary"), str) else None,
            entity_class=entity_class,
            metadata=metadata,
        )
        symbol.definition_ref_id = s.get("definition_ref_id", s.get("definition_mention_id"))
        index.symbols[symbol.id] = symbol
        index._index_symbol_name(symbol.id, symbol.canonical_name, namespace)
        for alias in symbol.aliases:
            index._index_symbol_name(symbol.id, alias, namespace)

    for r in data.get("references", data.get("mentions", [])):
        sym_id = r.get("symbol_id", r.get("entity_id", ""))
        sym = index.symbols.get(sym_id)
        if not sym:
            _log.warning(f"load: reference {r['id']} -> missing symbol {sym_id}, skipped")
            continue
        run_id = str(r.get("run_id", ""))
        step_id = str(r["step_id"])
        ref_step = index.steps.get((run_id, step_id))
        if not ref_step:
            ref_step = Step(
                run_id=run_id, step_id=step_id, index=0,
                role=StepRole.USER, content="",
            )
            index.add_step(ref_step)
        text = str(r.get("text", ""))
        start = int(r.get("start", r.get("offset", 0)))
        end = int(r.get("end", start + len(text)))
        ref_id = str(r.get("id") or stable_id("ref", run_id, step_id, start, end, sym.id))
        if ref_id in index.references:
            continue
        ref_kind = str(r.get("kind", r.get("mention_type", "unknown")))
        raw_grounded = r.get("grounded")
        grounded = bool(raw_grounded) if isinstance(raw_grounded, bool) else grounded_from_kind(ref_kind)
        structured = bool(r.get("structured", drives_defuse(sym.entity_class)))
        grounds_ref_id = r.get("grounds_ref_id") if isinstance(r.get("grounds_ref_id"), str) else None
        raw_form = r.get("form")
        form: RefForm = raw_form if raw_form in _REF_FORM_VALUES else "direct"
        ref = Reference(
            id=ref_id,
            symbol_id=sym.id,
            run_id=run_id,
            step_id=step_id,
            location=Location(run_id, step_id, start, end),
            text=text,
            role=str(r.get("role", ref_step.role)),
            kind=ref_kind,
            grounded=grounded,
            grounds_ref_id=grounds_ref_id,
            structured=structured,
            form=form,
            value=r.get("value") if isinstance(r.get("value"), str) else None,
            resolved_from=r.get("resolved_from") if isinstance(r.get("resolved_from"), str) else None,
            metadata=r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else {},
        )
        index.references[ref.id] = ref
        index._ref_ids_by_symbol[sym.id].append(ref.id)
        index._ref_ids_by_step[(run_id, step_id)].append(ref.id)

    for r in data.get("relations", []):
        from_s = index.symbols.get(r["from"])
        to_s = index.symbols.get(r["to"])
        if not from_s or not to_s:
            missing = [s for s, v in [("from", from_s), ("to", to_s)] if not v]
            _log.warning(f"load: relation {r['id']} missing {missing} symbol(s), skipped")
            continue
        run_id = str(r.get("run_id", ""))
        step_id = str(r["step_id"])
        rel_step = index.steps.get((run_id, step_id))
        if not rel_step:
            rel_step = Step(
                run_id=run_id, step_id=step_id, index=0,
                role=StepRole.USER, content="",
            )
            index.add_step(rel_step)
        rel_id = str(r.get("id") or stable_id("rel", from_s.id, to_s.id, r["type"]))
        if rel_id in index.relations:
            continue
        relation = Relation(
            id=rel_id,
            from_symbol_id=from_s.id,
            to_symbol_id=to_s.id,
            type=str(r["type"]),
            run_id=run_id,
            step_id=step_id,
            weight=float(r.get("weight", 1.0)),
            metadata=r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else {},
        )
        index.relations[relation.id] = relation
        index._relation_ids_by_symbol[from_s.id].append(relation.id)
        index._relation_ids_by_symbol[to_s.id].append(relation.id)

    for d in data.get("dependencies", []):
        sym_id = str(d.get("symbol_id", ""))
        if sym_id not in index.symbols:
            _log.warning(f"load: dependency {d.get('id')} -> missing symbol {sym_id}, skipped")
            continue
        raw_risk = d.get("risk")
        risk: Risk = raw_risk if raw_risk in _RISK_VALUES else "grounded"
        if raw_risk and raw_risk not in _RISK_VALUES:
            _log.warning(f"load: dependency {d.get('id')} risk {raw_risk!r} invalid, defaulting to 'grounded'")
        run_id = str(d.get("run_id", ""))
        def_step_id = str(d.get("def_step_id", ""))
        use_step_id = str(d.get("use_step_id", ""))
        dep_id = str(
            d.get("id") or stable_id("dep", run_id, def_step_id, use_step_id, sym_id)
        )
        if dep_id in index.dependencies:
            continue
        dep = Dependency(
            id=dep_id,
            symbol_id=sym_id,
            run_id=run_id,
            def_step_id=def_step_id,
            def_ref_id=str(d.get("def_ref_id", "")),
            def_version=int(d.get("def_version", 0)),
            use_step_id=use_step_id,
            use_ref_id=str(d.get("use_ref_id", "")),
            risk=risk,
            grounded_by_step_id=(
                d.get("grounded_by_step_id")
                if isinstance(d.get("grounded_by_step_id"), str) else None
            ),
            def_value=d.get("def_value") if isinstance(d.get("def_value"), str) else None,
            use_value=d.get("use_value") if isinstance(d.get("use_value"), str) else None,
            metadata=d.get("metadata", {}) if isinstance(d.get("metadata"), dict) else {},
        )
        index.dependencies[dep.id] = dep
        index._dep_ids_by_symbol[sym_id].append(dep.id)

    for c in data.get("claims", []):
        cid = str(c.get("id", ""))
        text = str(c.get("text", ""))
        if not cid or not text:
            continue
        index.claims[cid] = Claim(
            id=cid,
            run_id=str(c.get("run_id", "")),
            step_id=str(c.get("step_id", "")),
            text=text,
            role=str(c.get("role", "")),
        )

    for e in data.get("edges", []):
        eid = str(e.get("id", ""))
        if not eid:
            continue
        index.edges[eid] = Edge(
            id=eid,
            kind=str(e.get("kind", "")),
            run_id=str(e.get("run_id", "")),
            src=str(e.get("src", "")),
            dst=str(e.get("dst", "")),
            quote=str(e.get("quote", "")),
            evidence_position=str(e.get("evidence_position", "")),
        )

    for cf in data.get("claim_findings", []):
        index.claim_findings.append(ClaimFinding(
            claim_id=str(cf.get("claim_id", "")),
            run_id=str(cf.get("run_id", "")),
            step_id=str(cf.get("step_id", "")),
            status=str(cf.get("status", "unknown")),
            edge_ids=tuple(str(e) for e in cf.get("edge_ids", [])),
            evidence_empty=bool(cf.get("evidence_empty", cf.get("universe_empty", False))),
        ))

    for c in data.get("constraints", []):
        cid = str(c.get("id", ""))
        if not cid:
            continue
        normalized = c.get("normalized")
        index.constraints[cid] = Constraint(
            id=cid,
            description=str(c.get("description", "")),
            normalized=normalized if isinstance(normalized, dict) else None,
        )

    for f in data.get("constraint_findings", []):
        raw_status = f.get("status")
        status: FindingStatus = (
            raw_status if raw_status in _FINDING_STATUS_VALUES else "unknown"
        )
        if raw_status and raw_status not in _FINDING_STATUS_VALUES:
            _log.warning(
                f"load: constraint finding status {raw_status!r} invalid, defaulting to 'unknown'"
            )
        index.constraint_findings.append(ConstraintFinding(
            constraint_id=str(f.get("constraint_id", "")),
            candidate=str(f.get("candidate", "")),
            status=status,
            evidence_step_ids=tuple(
                str(s) for s in f.get("evidence_step_ids", []) if isinstance(s, str | int)
            ),
            commit_step_id=(
                f.get("commit_step_id")
                if isinstance(f.get("commit_step_id"), str) else None
            ),
            first_assertion_step_id=(
                f.get("first_assertion_step_id")
                if isinstance(f.get("first_assertion_step_id"), str) else None
            ),
            confidence=float(f.get("confidence", 1.0)),
            confidence_source=str(f.get("confidence_source", "")),
            reason=str(f.get("reason", "")),
        ))

    for a in data.get("actions", []):
        call_id = str(a.get("call_id", ""))
        if not call_id:
            continue
        raw_op = a.get("operation", "other")
        op = raw_op if raw_op in _ACTION_OP_VALUES else "other"
        raw_diffs = a.get("diffs", [])
        diffs = tuple(
            (str(d[0]), str(d[1]), str(d[2]))
            for d in raw_diffs
            if isinstance(d, list) and len(d) == 3
        )
        index.actions[call_id] = Action(
            call_id=call_id,
            step_id=str(a.get("step_id", "")),
            run_id=str(a.get("run_id", "")),
            tool_name=str(a.get("tool_name", "")),
            operation=op,
            targets=tuple(str(t) for t in a.get("targets", [])),
            diffs=diffs,
        )

    return index
