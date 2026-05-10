"""``llmharness review-html`` — generate a self-contained review page.

The page bundles trajectory + event graph + verdicts from one session JSONL
into a single HTML file with the data inlined as JSON. No build step, no
server, no network at view time except CDN-hosted Cytoscape.js.

Layout: trajectory list on the left, event DAG (Cytoscape + dagre) top-right,
verdict strip bottom-right. Clicking a message highlights every event whose
``source_turns`` contains it and vice versa; clicking a verdict highlights
its ``matched_event_ids``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .audit import entry_types as _et


def _content_text(content: Any) -> str:
    """Best-effort one-string view of a message's ``content`` blocks."""
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for b in content:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            parts.append(str(b.get("text", "")))
        elif t == "thinking":
            parts.append(f"[thinking] {b.get('text', '')}")
        elif t == "tool_call":
            args = b.get("arguments")
            try:
                args_s = json.dumps(args, ensure_ascii=False)
            except (TypeError, ValueError):
                args_s = str(args)
            parts.append(f"[tool_call] {b.get('name', '?')}({args_s})")
        elif t == "tool_result":
            inner = b.get("content")
            if isinstance(inner, list):
                parts.append(_content_text(inner))
            else:
                parts.append(f"[tool_result] {inner!r}")
    return "\n".join(p for p in parts if p)


def _collect_payload(session_path: Path) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    verdicts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    failure_kinds = {
        _et.EXTRACTOR_NO_CALL,
        _et.EXTRACTOR_ERROR,
        _et.EXTRACTOR_EMPTY,
        _et.AUDIT_NO_CALL,
        _et.AUDIT_ERROR,
    }
    with session_path.open("r", encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = rec.get("type")
            payload = rec.get("payload")
            if not isinstance(payload, dict):
                continue
            if t == _et.MESSAGE:
                idx = len(messages)
                messages.append(
                    {
                        "index": idx,
                        "role": payload.get("role", "?"),
                        "text": _content_text(payload.get("content")),
                    }
                )
            elif t == _et.AUDIT_EVENT:
                events.append(payload)
            elif t == _et.VERDICT:
                verdicts.append(payload)
            elif t in failure_kinds:
                failures.append({"kind": t, **payload})
    events.sort(key=lambda e: e.get("id", 0))
    return {
        "session_file": str(session_path),
        "messages": messages,
        "events": events,
        "verdicts": verdicts,
        "failures": failures,
    }


def build_review_html(session_path: Path, out_path: Path) -> None:
    payload = _collect_payload(session_path)
    data_json = json.dumps(payload, ensure_ascii=False)
    html = _HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>llmharness review</title>
<script src="https://unpkg.com/cytoscape@3.30.0/dist/cytoscape.min.js"></script>
<script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
<style>
  :root {
    --color-task: #2c7be5;
    --color-hypothesis: #6f42c1;
    --color-evidence: #00b386;
    --color-decision: #f6a609;
    --color-action: #5c6773;
    --color-reflection: #00b6d8;
    --color-conclusion: #d6336c;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 13px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #222; }
  #app { display: grid; grid-template-columns: 380px 1fr; height: 100vh; }
  #trajectory { overflow-y: auto; border-right: 1px solid #e3e3e3; padding: 8px 10px; }
  #right-pane { display: grid; grid-template-rows: 1fr 220px; min-width: 0; }
  #graph-wrap { position: relative; background: #fafbfc; border-bottom: 1px solid #e3e3e3; }
  #cy { width: 100%; height: 100%; }
  #verdicts { overflow-y: auto; padding: 8px 10px; background: #fcfcfc; }
  h2 { margin: 0 0 8px 0; font-size: 12px; text-transform: uppercase; color: #666; letter-spacing: 0.06em; }
  .header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; }
  .header .stats { font-size: 11px; color: #888; }
  .msg { padding: 6px 8px; margin: 4px 0; border-left: 3px solid #ddd; border-radius: 3px; cursor: pointer; }
  .msg:hover { background: #f0f6ff; }
  .msg.highlight { background: #fff3cd; border-left-color: #ff8800; }
  .msg.dim { opacity: 0.35; }
  .msg.user { border-left-color: var(--color-task); }
  .msg.assistant { border-left-color: var(--color-evidence); }
  .msg.tool_result { border-left-color: var(--color-decision); }
  .msg .role { font-weight: 600; font-size: 10px; text-transform: uppercase; color: #666; letter-spacing: 0.04em; }
  .msg .role .idx { color: #aaa; margin-left: 4px; }
  .msg .body { white-space: pre-wrap; word-break: break-word; margin-top: 2px; max-height: 5.8em; overflow: hidden; position: relative; }
  .msg.expanded .body { max-height: none; }
  .msg .body::after { content: ""; position: absolute; bottom: 0; left: 0; right: 0; height: 1.2em; background: linear-gradient(transparent, #fff); pointer-events: none; }
  .msg.expanded .body::after, .msg.highlight .body::after { display: none; }
  .verdict { padding: 6px 8px; margin: 4px 0; border-radius: 4px; cursor: pointer; border: 1px solid transparent; }
  .verdict.ok { background: #e8f5ee; border-color: #b6dec5; }
  .verdict.drift { background: #fbe2e2; border-color: #f0b3b3; }
  .verdict.highlight { outline: 2px solid #ff8800; }
  .verdict .head { font-weight: 600; }
  .verdict .meta { font-size: 11px; color: #555; }
  .verdict .reminder { margin-top: 4px; white-space: pre-wrap; font-size: 12px; }
  #legend { position: absolute; top: 8px; right: 8px; background: rgba(255,255,255,0.92); padding: 6px 8px; border: 1px solid #e3e3e3; border-radius: 4px; font-size: 11px; }
  #legend .chip { display: inline-block; padding: 1px 6px; margin: 2px 2px; border-radius: 8px; color: #fff; font-size: 10px; }
  #toolbar { position: absolute; top: 8px; left: 8px; background: rgba(255,255,255,0.92); padding: 6px 8px; border: 1px solid #e3e3e3; border-radius: 4px; font-size: 11px; }
  #toolbar button { font: inherit; padding: 2px 8px; cursor: pointer; }
  #detail { position: absolute; bottom: 8px; left: 8px; right: 8px; max-height: 35%; overflow: auto; background: rgba(255,255,255,0.97); border: 1px solid #ddd; border-radius: 4px; padding: 8px 10px; font-size: 12px; display: none; }
  #detail.show { display: block; }
  #detail .close { float: right; cursor: pointer; color: #888; }
</style>
</head>
<body>
<div id="app">
  <div id="trajectory">
    <div class="header"><h2>Trajectory</h2><span class="stats" id="msg-stats"></span></div>
    <div id="msg-list"></div>
  </div>
  <div id="right-pane">
    <div id="graph-wrap">
      <div id="cy"></div>
      <div id="toolbar">
        <button id="btn-reset">Reset highlight</button>
        <button id="btn-fit">Fit</button>
      </div>
      <div id="legend"></div>
      <div id="detail"><span class="close" id="detail-close">&times;</span><div id="detail-body"></div></div>
    </div>
    <div id="verdicts">
      <div class="header"><h2>Verdicts</h2><span class="stats" id="verdict-stats"></span></div>
      <div id="verdict-list"></div>
    </div>
  </div>
</div>
<script>
const DATA = __DATA_JSON__;
const KIND_COLORS = {
  task: getCSS("--color-task"),
  hypothesis: getCSS("--color-hypothesis"),
  evidence: getCSS("--color-evidence"),
  decision: getCSS("--color-decision"),
  action: getCSS("--color-action"),
  reflection: getCSS("--color-reflection"),
  conclusion: getCSS("--color-conclusion"),
};
function getCSS(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

// Index for fast joint lookup.
const eventsById = new Map(DATA.events.map(e => [e.id, e]));
const eventsByTurn = new Map(); // msg index -> [event id]
for (const e of DATA.events) {
  for (const t of (e.source_turns || [])) {
    if (!eventsByTurn.has(t)) eventsByTurn.set(t, []);
    eventsByTurn.get(t).push(e.id);
  }
}

// --- left pane: trajectory ----------------------------------------------
const msgList = document.getElementById("msg-list");
document.getElementById("msg-stats").textContent = `${DATA.messages.length} messages`;
for (const m of DATA.messages) {
  const div = document.createElement("div");
  div.className = `msg ${m.role}`;
  div.dataset.idx = m.index;
  div.innerHTML = `<div class="role">${escapeHtml(m.role)}<span class="idx">#${m.index}</span></div><div class="body"></div>`;
  div.querySelector(".body").textContent = m.text || "(empty)";
  div.addEventListener("click", (ev) => {
    if (ev.shiftKey) { div.classList.toggle("expanded"); return; }
    selectMessage(m.index);
  });
  msgList.appendChild(div);
}

// --- bottom pane: verdicts -----------------------------------------------
const verdictList = document.getElementById("verdict-list");
const driftN = DATA.verdicts.filter(v => v.drift).length;
document.getElementById("verdict-stats").textContent = `${DATA.verdicts.length} (drift=${driftN})`;
DATA.verdicts.forEach((v, i) => {
  const div = document.createElement("div");
  div.className = `verdict ${v.drift ? "drift" : "ok"}`;
  div.dataset.vi = i;
  const matched = v.matched_event_ids || [];
  div.innerHTML = `
    <div class="head">verdict #${i} — drift=${v.drift}${v.type ? " · " + escapeHtml(v.type) : ""}</div>
    <div class="meta">matched=[${matched.join(", ")}]${v.cited_cards && v.cited_cards.length ? " · cards=[" + v.cited_cards.map(escapeHtml).join(", ") + "]" : ""}</div>
    ${v.reminder ? `<div class="reminder">${escapeHtml(v.reminder)}</div>` : ""}
  `;
  div.addEventListener("click", () => selectVerdict(i));
  verdictList.appendChild(div);
});
if (DATA.verdicts.length === 0) {
  verdictList.innerHTML = "<em style='color:#888'>(no verdicts)</em>";
}

// --- legend --------------------------------------------------------------
const legend = document.getElementById("legend");
legend.innerHTML = Object.entries(KIND_COLORS).map(([k, c]) =>
  `<span class="chip" style="background:${c}">${k}</span>`
).join("");

// --- center pane: cytoscape graph ---------------------------------------
const elements = [];
for (const e of DATA.events) {
  elements.push({
    data: {
      id: "e" + e.id,
      eid: e.id,
      kind: e.kind,
      label: `#${e.id} ${e.kind}`,
      summary: e.summary || "",
      source_turns: e.source_turns || [],
      color: KIND_COLORS[e.kind] || "#888",
    }
  });
  for (const r of (e.refs || [])) {
    if (eventsById.has(r)) {
      elements.push({ data: { id: `e${r}-e${e.id}`, source: "e" + r, target: "e" + e.id } });
    }
  }
}

const cy = cytoscape({
  container: document.getElementById("cy"),
  elements,
  style: [
    {
      selector: "node",
      style: {
        "background-color": "data(color)",
        "label": "data(label)",
        "color": "#fff",
        "text-valign": "center",
        "text-halign": "center",
        "font-size": 10,
        "width": "label",
        "height": 22,
        "padding": "6px",
        "shape": "round-rectangle",
        "border-width": 1,
        "border-color": "rgba(0,0,0,0.2)",
      },
    },
    { selector: "edge", style: {
        "width": 1.4,
        "line-color": "#bbb",
        "target-arrow-color": "#bbb",
        "target-arrow-shape": "triangle",
        "curve-style": "bezier",
    }},
    { selector: ".highlight", style: {
        "border-width": 3, "border-color": "#ff8800",
    }},
    { selector: ".dim", style: { "opacity": 0.2 } },
    { selector: "edge.highlight", style: {
        "line-color": "#ff8800", "target-arrow-color": "#ff8800", "width": 2.5,
    }},
  ],
  layout: { name: "dagre", rankDir: "TB", nodeSep: 22, rankSep: 36 },
});

cy.on("tap", "node", (evt) => selectEvent(evt.target.data("eid")));
cy.on("tap", (evt) => { if (evt.target === cy) clearHighlights(); });
document.getElementById("btn-reset").onclick = clearHighlights;
document.getElementById("btn-fit").onclick = () => cy.fit(undefined, 30);
document.getElementById("detail-close").onclick = () => document.getElementById("detail").classList.remove("show");

// --- selection logic -----------------------------------------------------
function clearHighlights() {
  cy.elements().removeClass("highlight").removeClass("dim");
  document.querySelectorAll(".msg").forEach(el => el.classList.remove("highlight", "dim"));
  document.querySelectorAll(".verdict").forEach(el => el.classList.remove("highlight"));
  document.getElementById("detail").classList.remove("show");
}

function dimAllExcept(eventIds, msgIndices) {
  const eidSet = new Set(eventIds.map(id => "e" + id));
  cy.nodes().forEach(n => {
    if (eidSet.has(n.id())) n.removeClass("dim").addClass("highlight");
    else n.addClass("dim").removeClass("highlight");
  });
  cy.edges().forEach(e => {
    const inSel = eidSet.has(e.source().id()) && eidSet.has(e.target().id());
    if (inSel) e.removeClass("dim").addClass("highlight");
    else e.addClass("dim").removeClass("highlight");
  });
  const idxSet = new Set(msgIndices);
  document.querySelectorAll(".msg").forEach(el => {
    const idx = +el.dataset.idx;
    if (idxSet.has(idx)) { el.classList.add("highlight"); el.classList.remove("dim"); }
    else { el.classList.add("dim"); el.classList.remove("highlight"); }
  });
}

function ancestorsOf(eid, acc = new Set()) {
  if (acc.has(eid)) return acc;
  acc.add(eid);
  const e = eventsById.get(eid);
  if (!e) return acc;
  for (const r of (e.refs || [])) ancestorsOf(r, acc);
  return acc;
}
function descendantsOf(eid) {
  const acc = new Set([eid]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const e of DATA.events) {
      if (acc.has(e.id)) continue;
      if ((e.refs || []).some(r => acc.has(r))) { acc.add(e.id); changed = true; }
    }
  }
  return acc;
}

function selectMessage(idx) {
  const eids = eventsByTurn.get(idx) || [];
  if (eids.length === 0) {
    clearHighlights();
    document.querySelector(`.msg[data-idx="${idx}"]`).classList.add("highlight");
    showDetail(`<b>message #${idx}</b> — no extracted events reference this turn.`);
    return;
  }
  dimAllExcept(eids, [idx]);
  showDetail(`<b>message #${idx}</b> → ${eids.length} event(s): ${eids.map(e => `#${e}`).join(", ")}`);
}

function selectEvent(eid) {
  const family = new Set([...ancestorsOf(eid), ...descendantsOf(eid)]);
  const e = eventsById.get(eid);
  const turns = new Set();
  for (const id of family) {
    const ev = eventsById.get(id);
    if (ev) for (const t of (ev.source_turns || [])) turns.add(t);
  }
  dimAllExcept([...family], [...turns]);
  showDetail(
    `<b>event #${eid}</b> · ${escapeHtml(e.kind)}<br>`
    + `<i>${escapeHtml(e.summary || "")}</i><br>`
    + `<small>refs=[${(e.refs || []).join(", ")}] · source_turns=[${(e.source_turns || []).join(", ")}]<br>`
    + `ancestors=${ancestorsOf(eid).size - 1} · descendants=${descendantsOf(eid).size - 1}</small>`
  );
}

function selectVerdict(vi) {
  const v = DATA.verdicts[vi];
  document.querySelectorAll(".verdict").forEach(el => el.classList.toggle("highlight", +el.dataset.vi === vi));
  const matched = v.matched_event_ids || [];
  if (matched.length === 0) {
    cy.elements().removeClass("highlight").removeClass("dim");
    document.querySelectorAll(".msg").forEach(el => el.classList.remove("highlight", "dim"));
    showDetail(`<b>verdict #${vi}</b> — no matched_event_ids.${v.reminder ? "<br>" + escapeHtml(v.reminder) : ""}`);
    return;
  }
  const turns = new Set();
  for (const id of matched) {
    const ev = eventsById.get(id);
    if (ev) for (const t of (ev.source_turns || [])) turns.add(t);
  }
  dimAllExcept(matched, [...turns]);
  showDetail(`<b>verdict #${vi}</b> matched [${matched.join(", ")}]${v.reminder ? "<br>" + escapeHtml(v.reminder) : ""}`);
}

function showDetail(html) {
  const d = document.getElementById("detail");
  document.getElementById("detail-body").innerHTML = html;
  d.classList.add("show");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
}

// initial fit
cy.ready(() => cy.fit(undefined, 30));
</script>
</body>
</html>
"""


__all__ = ["build_review_html"]
