/* global cytoscape */
(function () {
  "use strict";

  const confidenceRank = { low: 0, medium: 1, high: 2 };
  const toolOrder = ["bash", "read", "edit", "write"];
  const state = {
    snapshot: null,
    revision: null,
    cy: null,
    paused: false,
    refreshMs: 1500,
    timer: null,
    allComponents: false,
    selectedComponents: new Set(),
    enabledTools: new Set(),
    confidence: "low",
    showSymbols: false,
    showModules: false,
    showMentions: false,
    showSymbolLinks: false,
    connectedOnly: true,
    query: "",
    selectedId: null,
    fitNext: true,
    neighborsOnly: null,
  };

  const dom = {
    allComponents: document.querySelector("#all-components"),
    closeInspector: document.querySelector("#close-inspector"),
    componentCount: document.querySelector("#component-count"),
    componentList: document.querySelector("#component-list"),
    componentPanel: document.querySelector("#component-panel"),
    connectedOnly: document.querySelector("#connected-only"),
    fitGraph: document.querySelector("#fit-graph"),
    graph: document.querySelector("#graph"),
    graphMessage: document.querySelector("#graph-message"),
    graphSearch: document.querySelector("#graph-search"),
    graphStage: document.querySelector(".graph-stage"),
    headlineStats: document.querySelector("#headline-stats"),
    inspectorContent: document.querySelector("#inspector-content"),
    inspectorPanel: document.querySelector("#inspector-panel"),
    inspectorTitle: document.querySelector("#inspector-title"),
    laneActions: document.querySelector("#lane-actions-count"),
    laneFiles: document.querySelector("#lane-files-count"),
    laneSymbols: document.querySelector("#lane-symbols-count"),
    largestComponent: document.querySelector("#largest-component"),
    latestComponent: document.querySelector("#latest-component"),
    pauseLive: document.querySelector("#pause-live"),
    sessionLabel: document.querySelector("#session-label"),
    showMentions: document.querySelector("#show-mentions"),
    showModules: document.querySelector("#show-modules"),
    showSymbolLinks: document.querySelector("#show-symbol-links"),
    showSymbols: document.querySelector("#show-symbols"),
    symbolOptions: document.querySelector("#symbol-options"),
    toggleComponents: document.querySelector("#toggle-components"),
    toggleInspector: document.querySelector("#toggle-inspector"),
    toolFilters: document.querySelector("#tool-filters"),
    visibleCount: document.querySelector("#visible-count"),
    zoomIn: document.querySelector("#zoom-in"),
    zoomOut: document.querySelector("#zoom-out"),
  };

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
  }

  function shortId(value) {
    if (!value) return "unknown";
    return value.length > 18 ? `${value.slice(0, 10)}…${value.slice(-5)}` : value;
  }

  function setStatus(kind, label) {
    let node = dom.headlineStats.querySelector(".live-state");
    if (!node) {
      node = el("span", "live-state");
      dom.headlineStats.append(node);
    }
    node.className = `live-state ${kind || ""}`.trim();
    node.textContent = label;
  }

  function renderHeadline() {
    const snapshot = state.snapshot;
    if (!snapshot) return;
    dom.headlineStats.replaceChildren();
    const visible = filteredData();
    const counts = visible.nodes.reduce((result, node) => {
      result[node.type] = (result[node.type] || 0) + 1;
      return result;
    }, {});
    const values = [[counts.action || 0, "actions"]];
    if (state.showSymbols) values.push([counts.symbol || 0, "symbols"]);
    values.push([counts.file || 0, "files"], [visible.edges.length, "links"]);
    values.forEach(([number, label]) => {
      const pair = el("span", "stat-pair");
      pair.append(el("b", "", number), el("span", "", label));
      dom.headlineStats.append(pair);
    });
    setStatus(state.paused ? "paused" : "", state.paused ? "Paused" : "Live");
  }

  function renderToolFilters() {
    const counts = state.snapshot ? state.snapshot.stats.tools : {};
    const tools = [...new Set([...toolOrder, ...Object.keys(counts)])]
      .filter((tool) => counts[tool])
      .sort((left, right) => {
        const li = toolOrder.indexOf(left);
        const ri = toolOrder.indexOf(right);
        return (li < 0 ? 99 : li) - (ri < 0 ? 99 : ri) || left.localeCompare(right);
      });
    if (state.enabledTools.size === 0) tools.forEach((tool) => state.enabledTools.add(tool));
    dom.toolFilters.replaceChildren();
    tools.forEach((tool) => {
      const label = el("label", "tool-toggle");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = state.enabledTools.has(tool);
      input.addEventListener("change", () => {
        if (input.checked) state.enabledTools.add(tool);
        else state.enabledTools.delete(tool);
        state.fitNext = true;
        renderGraph();
      });
      label.append(
        input,
        el("i", `tool-dot ${tool}`),
        el("span", "", tool),
        el("span", "tool-count", counts[tool] || 0),
      );
      dom.toolFilters.append(label);
    });
  }

  function renderComponents() {
    const components = state.snapshot ? state.snapshot.components : [];
    dom.componentCount.textContent = String(components.length);
    dom.componentList.replaceChildren();
    const fragment = document.createDocumentFragment();
    components.forEach((component) => {
      const row = el("label", "component-row");
      if (state.allComponents || state.selectedComponents.has(component.id)) row.classList.add("selected");
      const checkWrap = el("span", "component-check");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = state.allComponents || state.selectedComponents.has(component.id);
      checkbox.disabled = state.allComponents;
      checkbox.addEventListener("change", () => {
        state.allComponents = false;
        if (checkbox.checked) state.selectedComponents.add(component.id);
        else state.selectedComponents.delete(component.id);
        state.neighborsOnly = null;
        state.fitNext = true;
        syncComponentButtons();
        renderComponents();
        renderGraph();
      });
      checkWrap.append(checkbox);
      const copy = el("span", "component-copy");
      copy.append(
        el("span", "component-name", component.label),
        el(
          "span",
          "component-meta",
          state.showSymbols
            ? `${component.actions} ops · ${component.symbols} symbols · ${component.files} files`
            : `${component.actions} ops · ${component.files} files`,
        ),
      );
      row.append(checkWrap, copy, el("span", "component-size", `#${component.rank}`));
      fragment.append(row);
    });
    dom.componentList.append(fragment);
  }

  function syncComponentButtons() {
    const latest = defaultComponent(state.snapshot);
    dom.allComponents.classList.toggle("active", state.allComponents);
    dom.latestComponent.classList.toggle(
      "active",
      !state.allComponents && state.selectedComponents.size === 1
        && state.selectedComponents.has(latest?.id),
    );
    dom.largestComponent.classList.toggle(
      "active",
      !state.allComponents && state.snapshot && state.selectedComponents.size === 1
        && state.selectedComponents.has(state.snapshot.components[0]?.id),
    );
  }

  function defaultComponent(snapshot) {
    if (!snapshot) return null;
    const candidates = snapshot.components
      .filter((component) => component.actions > 0 && component.symbols > 0 && component.files > 0 && component.nodes <= 120)
      .sort((left, right) => right.updatedAt - left.updatedAt || right.actions - left.actions || left.rank - right.rank);
    return candidates[0]
      || snapshot.components.find((component) => component.nodes > 1)
      || snapshot.components[0]
      || null;
  }

  function selectedComponentIds() {
    if (!state.snapshot) return new Set();
    if (state.allComponents) return new Set(state.snapshot.components.map((item) => item.id));
    return new Set(state.selectedComponents);
  }

  function filteredData() {
    const snapshot = state.snapshot;
    if (!snapshot) return { nodes: [], edges: [] };
    const componentIds = selectedComponentIds();
    const minRank = confidenceRank[state.confidence];
    const query = state.query.trim().toLowerCase();
    const allNodes = new Map(snapshot.nodes.map((node) => [node.id, node]));
    let nodes = snapshot.nodes.filter((node) => {
      if (!componentIds.has(node.componentId)) return false;
      if ((confidenceRank[node.confidence] ?? 0) < minRank) return false;
      if (node.type === "action" && !state.enabledTools.has(node.tool)) return false;
      if (node.type === "symbol" && !state.showSymbols) return false;
      if (node.type === "symbol" && node.symbolKind === "module" && !state.showModules) return false;
      if (node.type === "symbol" && node.symbolKind === "mention" && !state.showMentions) return false;
      return true;
    });
    let nodeIds = new Set(nodes.map((node) => node.id));
    let edges = snapshot.edges.filter((edge) => {
      if (edge.optional && !state.showSymbolLinks) return false;
      if ((confidenceRank[edge.confidence] ?? 0) < minRank) return false;
      return nodeIds.has(edge.source) && nodeIds.has(edge.target);
    });

    if (state.neighborsOnly) {
      const keep = new Set([state.neighborsOnly]);
      edges.forEach((edge) => {
        if (edge.source === state.neighborsOnly || edge.target === state.neighborsOnly) {
          keep.add(edge.source);
          keep.add(edge.target);
        }
      });
      nodes = nodes.filter((node) => keep.has(node.id));
      nodeIds = new Set(nodes.map((node) => node.id));
      edges = edges.filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target));
    }

    if (query) {
      const matches = new Set(
        nodes
          .filter((node) => JSON.stringify(node).toLowerCase().includes(query))
          .map((node) => node.id),
      );
      const context = new Set(matches);
      edges.forEach((edge) => {
        if (matches.has(edge.source) || matches.has(edge.target)) {
          context.add(edge.source);
          context.add(edge.target);
        }
      });
      nodes = nodes.filter((node) => context.has(node.id));
      nodeIds = new Set(nodes.map((node) => node.id));
      edges = edges.filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target));
    }

    if (state.connectedOnly) {
      const connected = new Set();
      edges.forEach((edge) => {
        connected.add(edge.source);
        connected.add(edge.target);
      });
      nodes = nodes.filter((node) => connected.has(node.id));
    }

    return { nodes, edges, allNodes };
  }

  function elementClasses(node) {
    const classes = [`type-${node.type}`];
    if (node.tool) classes.push(`tool-${node.tool}`);
    if (node.symbolKind) classes.push(`symbol-${node.symbolKind}`);
    if (node.isError) classes.push("is-error");
    return classes.join(" ");
  }

  function computePositions(nodes, edges) {
    const degree = new Map(nodes.map((node) => [node.id, 0]));
    edges.forEach((edge) => {
      degree.set(edge.source, (degree.get(edge.source) || 0) + 1);
      degree.set(edge.target, (degree.get(edge.target) || 0) + 1);
    });
    const components = new Map();
    nodes.forEach((node) => {
      if (!components.has(node.componentId)) components.set(node.componentId, []);
      components.get(node.componentId).push(node);
    });
    const orderedComponents = [...components.entries()].sort((left, right) => {
      const lr = left[1][0]?.componentRank || 9999;
      const rr = right[1][0]?.componentRank || 9999;
      return lr - rr;
    });
    const positions = new Map();
    const x = { action: 150, symbol: 650, file: state.showSymbols ? 1130 : 850 };
    let offset = 40;
    orderedComponents.forEach(([, componentNodes]) => {
      const lanes = { action: [], symbol: [], file: [] };
      componentNodes.forEach((node) => lanes[node.type]?.push(node));
      Object.values(lanes).forEach((lane) => {
        lane.sort((left, right) => {
          if (left.type === "action") return (left.timestamp || 0) - (right.timestamp || 0);
          return (degree.get(right.id) || 0) - (degree.get(left.id) || 0)
            || String(left.label).localeCompare(String(right.label));
        });
      });
      const rows = Math.max(lanes.action.length, lanes.symbol.length, lanes.file.length, 1);
      Object.entries(lanes).forEach(([type, lane]) => {
        const laneHeight = Math.max(rows * 54, 170);
        const start = offset + Math.max(0, (laneHeight - lane.length * 54) / 2);
        lane.forEach((node, index) => positions.set(node.id, { x: x[type], y: start + index * 54 }));
      });
      offset += Math.max(rows * 54, 170) + 72;
    });
    return positions;
  }

  function cytoscapeStyle() {
    return [
      {
        selector: "node",
        style: {
          width: 172,
          height: 34,
          shape: "roundrectangle",
          "background-color": "#ffffff",
          "border-width": 1.5,
          "border-color": "#8794a0",
          label: "data(label)",
          color: "#27313a",
          "font-family": "SFMono-Regular, Consolas, monospace",
          "font-size": 10,
          "font-weight": 560,
          "text-wrap": "ellipsis",
          "text-max-width": 152,
          "text-valign": "center",
          "text-halign": "center",
          "overlay-opacity": 0,
        },
      },
      { selector: ".type-action", style: { width: 164, "border-color": "#75818c" } },
      { selector: ".tool-bash", style: { "border-color": "#c5533d", "border-width": 2 } },
      { selector: ".tool-read", style: { "border-color": "#147c8c", "border-width": 2 } },
      { selector: ".tool-edit", style: { "border-color": "#6758a4", "border-width": 2 } },
      { selector: ".tool-write", style: { "border-color": "#b53f68", "border-width": 2 } },
      { selector: ".type-symbol", style: { width: 188, "border-color": "#4a8b68", "background-color": "#fbfefc" } },
      { selector: ".type-file", style: { width: 222, "border-color": "#627384", "background-color": "#fbfcfe" } },
      { selector: ".symbol-mention", style: { "border-style": "dashed", "border-color": "#b07a6e" } },
      { selector: ".symbol-module", style: { "border-style": "dashed", "border-color": "#7b8d9f" } },
      { selector: ".is-error", style: { "background-color": "#fff2f0", "border-color": "#b42318" } },
      {
        selector: "edge",
        style: {
          width: "mapData(evidenceCount, 1, 8, 1, 3)",
          "line-color": "#9aa5af",
          "target-arrow-color": "#9aa5af",
          "target-arrow-shape": "triangle",
          "arrow-scale": 0.7,
          "curve-style": "bezier",
          opacity: 0.48,
          label: "",
          "overlay-opacity": 0,
        },
      },
      { selector: ".kind-action-symbol", style: { "line-color": "#6a9c82", "target-arrow-color": "#6a9c82" } },
      { selector: ".kind-action-file", style: { "line-color": "#7e91a2", "target-arrow-color": "#7e91a2" } },
      { selector: ".kind-symbol-file", style: { "line-color": "#718c7c", "target-arrow-color": "#718c7c" } },
      { selector: ".kind-symbol-symbol", style: { "line-style": "dashed", opacity: 0.25 } },
      {
        selector: "node:selected",
        style: {
          "border-width": 3,
          "border-color": "#1668c7",
          "background-color": "#eef6ff",
        },
      },
      {
        selector: "edge:selected, edge.context",
        style: {
          width: 2.5,
          opacity: 0.95,
          label: "data(relation)",
          color: "#44515e",
          "font-size": 8,
          "text-background-color": "#ffffff",
          "text-background-opacity": 0.9,
          "text-background-padding": 2,
          "text-rotation": "autorotate",
          "z-index": 20,
        },
      },
      { selector: ".faded", style: { opacity: 0.09 } },
    ];
  }

  function renderGraph() {
    if (!state.snapshot) return;
    const { nodes, edges } = filteredData();
    const positions = computePositions(nodes, edges);
    const elements = [
      ...nodes.map((node) => ({
        group: "nodes",
        data: node,
        classes: elementClasses(node),
        position: positions.get(node.id),
      })),
      ...edges.map((edge) => ({
        group: "edges",
        data: edge,
        classes: `kind-${edge.kind}`,
      })),
    ];

    const previousPan = state.cy ? state.cy.pan() : null;
    const previousZoom = state.cy ? state.cy.zoom() : null;
    if (!state.cy) {
      state.cy = cytoscape({
        container: dom.graph,
        elements,
        style: cytoscapeStyle(),
        layout: { name: "preset", fit: false },
        minZoom: 0.05,
        maxZoom: 2.5,
        wheelSensitivity: 0.18,
        boxSelectionEnabled: true,
      });
      bindGraphEvents();
    } else {
      state.cy.startBatch();
      state.cy.elements().remove();
      state.cy.add(elements);
      state.cy.endBatch();
      state.cy.layout({ name: "preset", fit: false }).run();
    }

    if (state.fitNext || previousPan === null || previousZoom === null) {
      fitGraph();
      state.fitNext = false;
    } else {
      state.cy.zoom(previousZoom);
      state.cy.pan(previousPan);
    }
    if (state.selectedId && state.cy.getElementById(state.selectedId).length) {
      state.cy.getElementById(state.selectedId).select();
    }
    const counts = nodes.reduce((acc, node) => {
      acc[node.type] = (acc[node.type] || 0) + 1;
      return acc;
    }, {});
    dom.laneActions.textContent = counts.action || 0;
    dom.laneSymbols.textContent = counts.symbol || 0;
    dom.laneFiles.textContent = counts.file || 0;
    dom.visibleCount.textContent = `${nodes.length} nodes · ${edges.length} edges`;
    dom.graphMessage.hidden = nodes.length > 0;
    dom.graphMessage.textContent = componentMessage();
    renderHeadline();
    renderInspector();
  }

  function componentMessage() {
    if (!state.snapshot?.nodes.length) return "No IFG rows for this session.";
    if (!state.allComponents && state.selectedComponents.size === 0) return "Select one or more components.";
    if (state.query) return "No nodes match the current search and filters.";
    return "No connected nodes match the current filters.";
  }

  function bindGraphEvents() {
    state.cy.on("tap", "node, edge", (event) => {
      state.selectedId = event.target.id();
      renderInspector();
      if (window.innerWidth <= 820) dom.inspectorPanel.classList.add("open");
    });
    state.cy.on("tap", (event) => {
      if (event.target === state.cy) {
        state.selectedId = null;
        state.cy.elements().unselect();
        renderInspector();
      }
    });
    state.cy.on("mouseover", "node", (event) => {
      const node = event.target;
      const neighborhood = node.closedNeighborhood();
      state.cy.elements().difference(neighborhood).addClass("faded");
      node.connectedEdges().addClass("context");
    });
    state.cy.on("mouseout", "node", () => {
      state.cy.elements().removeClass("faded context");
    });
  }

  function fitGraph() {
    if (!state.cy || state.cy.nodes().length === 0) return;
    let focus = state.cy.nodes();
    if (focus.length > 120) {
      focus = focus
        .sort((left, right) => right.degree(false) - left.degree(false))
        .slice(0, 72);
    }
    state.cy.fit(focus, 58);
    const zoom = Math.min(state.cy.zoom(), 1.15);
    state.cy.zoom({ level: zoom, renderedPosition: { x: dom.graph.clientWidth / 2, y: dom.graph.clientHeight / 2 } });
  }

  function renderInspector() {
    dom.inspectorContent.replaceChildren();
    if (!state.snapshot) return;
    const data = filteredData();
    const node = state.snapshot.nodes.find((item) => item.id === state.selectedId);
    const edge = state.snapshot.edges.find((item) => item.id === state.selectedId);
    if (node) {
      renderNodeInspector(node, data);
      return;
    }
    if (edge) {
      renderEdgeInspector(edge);
      return;
    }
    const components = selectedComponentIds();
    dom.inspectorTitle.textContent = "Selection";
    const copy = el("p", "empty-copy", `${components.size} component${components.size === 1 ? "" : "s"} active`);
    const list = propertyList([
      ["Visible nodes", data.nodes.length],
      ["Visible edges", data.edges.length],
      ["Revision", state.snapshot.revision],
      ["Extractor", state.snapshot.extractorVersion],
    ]);
    dom.inspectorContent.append(copy, list);
  }

  function renderNodeInspector(node, data) {
    dom.inspectorTitle.textContent = node.displayName || node.label;
    const badges = el("div", "inspect-badges");
    badges.append(el("span", "badge", node.type));
    if (node.tool) badges.append(el("span", "badge", node.tool));
    if (node.symbolKind) badges.append(el("span", "badge", node.symbolKind));
    if (node.confidence) badges.append(el("span", `badge ${node.confidence}`, node.confidence));
    dom.inspectorContent.append(badges);
    if (node.path) dom.inspectorContent.append(el("p", "inspect-path", node.path));

    const actions = el("div", "inspect-actions");
    const focusComponent = el("button", "", "Focus component");
    focusComponent.type = "button";
    focusComponent.addEventListener("click", () => {
      state.allComponents = false;
      state.selectedComponents = new Set([node.componentId]);
      state.neighborsOnly = null;
      state.fitNext = true;
      syncComponentButtons();
      renderComponents();
      renderGraph();
    });
    const focusNeighbors = el("button", "", state.neighborsOnly === node.id ? "Show component" : "Focus neighbors");
    focusNeighbors.type = "button";
    focusNeighbors.addEventListener("click", () => {
      state.neighborsOnly = state.neighborsOnly === node.id ? null : node.id;
      state.fitNext = true;
      renderGraph();
    });
    actions.append(focusComponent, focusNeighbors);
    dom.inspectorContent.append(actions);

    const ignored = new Set(["id", "label", "displayName", "componentId", "componentRank", "metadata", "updatedAt", "timestamp", "path"]);
    const properties = Object.entries(node)
      .filter(([key, value]) => !ignored.has(key) && value !== null && value !== undefined && value !== "")
      .map(([key, value]) => [humanize(key), displayValue(value)]);
    dom.inspectorContent.append(propertyList(properties));

    const connections = data.edges
      .filter((item) => item.source === node.id || item.target === node.id)
      .map((item) => {
        const otherId = item.source === node.id ? item.target : item.source;
        const other = data.nodes.find((candidate) => candidate.id === otherId);
        return { edge: item, node: other };
      })
      .filter((item) => item.node)
      .sort((left, right) => left.edge.relation.localeCompare(right.edge.relation));
    dom.inspectorContent.append(el("h3", "connection-heading", `Connections · ${connections.length}`));
    connections.slice(0, 80).forEach(({ edge: item, node: other }) => {
      const row = el("button", "connection-row");
      row.type = "button";
      row.append(
        el("span", "connection-relation", item.relation),
        el("span", "connection-name", other.label),
        el("span", "connection-count", item.evidenceCount > 1 ? `×${item.evidenceCount}` : ""),
      );
      row.addEventListener("click", () => {
        state.selectedId = other.id;
        const target = state.cy.getElementById(other.id);
        if (target.length) {
          state.cy.elements().unselect();
          target.select();
          state.cy.animate({ center: { eles: target }, duration: 180 });
        }
        renderInspector();
      });
      dom.inspectorContent.append(row);
    });
  }

  function renderEdgeInspector(edge) {
    dom.inspectorTitle.textContent = edge.relation;
    const source = state.snapshot.nodes.find((node) => node.id === edge.source);
    const target = state.snapshot.nodes.find((node) => node.id === edge.target);
    const badges = el("div", "inspect-badges");
    badges.append(
      el("span", "badge", edge.kind),
      el("span", `badge ${edge.confidence}`, edge.confidence),
    );
    dom.inspectorContent.append(
      badges,
      propertyList([
        ["From", source?.label || edge.source],
        ["To", target?.label || edge.target],
        ["Relation", edge.relation],
        ["Evidence", edge.evidenceCount],
        ["Sources", edge.evidenceSources.join(", ")],
        ["Optional", edge.optional],
      ]),
    );
  }

  function propertyList(rows) {
    const list = el("dl", "property-list");
    rows.forEach(([key, value]) => {
      list.append(el("dt", "", key), el("dd", "", displayValue(value)));
    });
    return list;
  }

  function humanize(value) {
    return value.replace(/([A-Z])/g, " $1").replace(/^./, (char) => char.toUpperCase());
  }

  function displayValue(value) {
    if (Array.isArray(value)) return value.join(", ");
    if (value && typeof value === "object") return JSON.stringify(value, null, 2);
    if (typeof value === "boolean") return value ? "yes" : "no";
    return String(value);
  }

  async function refresh(force = false) {
    window.clearTimeout(state.timer);
    if (state.paused && !force) {
      scheduleRefresh();
      return;
    }
    try {
      const params = new URLSearchParams();
      if (state.showSymbolLinks) params.set("symbol_links", "1");
      if (state.revision) params.set("revision", state.revision);
      const response = await fetch(`/api/graph?${params}`, { cache: "no-store" });
      if (response.status === 204) return;
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const snapshot = await response.json();
      if (snapshot.error) throw new Error(snapshot.error);
      state.refreshMs = snapshot.refreshMs || state.refreshMs;
      const changed = snapshot.revision !== state.revision;
      const hadSnapshot = Boolean(state.snapshot);
      const oldSelection = new Set(state.selectedComponents);
      state.snapshot = snapshot;
      state.revision = snapshot.revision;
      dom.sessionLabel.textContent = shortId(snapshot.sessionId);
      dom.sessionLabel.title = snapshot.sessionId;
      if (!hadSnapshot) {
        const first = defaultComponent(snapshot);
        if (first) state.selectedComponents.add(first.id);
        state.fitNext = true;
      } else {
        const available = new Set(snapshot.components.map((component) => component.id));
        state.selectedComponents = new Set([...oldSelection].filter((id) => available.has(id)));
        const fallback = defaultComponent(snapshot);
        if (!state.allComponents && state.selectedComponents.size === 0 && fallback) {
          state.selectedComponents.add(fallback.id);
          state.fitNext = true;
        }
      }
      if (changed || !hadSnapshot) {
        renderToolFilters();
        renderComponents();
        syncComponentButtons();
        renderGraph();
      }
    } catch (error) {
      if (!state.snapshot) {
        dom.graphMessage.hidden = false;
        dom.graphMessage.textContent = `IFG unavailable: ${error.message}`;
      }
      setStatus("error", "Disconnected");
    } finally {
      scheduleRefresh();
    }
  }

  function scheduleRefresh() {
    window.clearTimeout(state.timer);
    state.timer = window.setTimeout(refresh, state.refreshMs);
  }

  function bindControls() {
    dom.zoomIn.addEventListener("click", () => state.cy?.zoom({ level: state.cy.zoom() * 1.2, renderedPosition: { x: dom.graph.clientWidth / 2, y: dom.graph.clientHeight / 2 } }));
    dom.zoomOut.addEventListener("click", () => state.cy?.zoom({ level: state.cy.zoom() / 1.2, renderedPosition: { x: dom.graph.clientWidth / 2, y: dom.graph.clientHeight / 2 } }));
    dom.fitGraph.addEventListener("click", fitGraph);
    dom.pauseLive.addEventListener("click", () => {
      state.paused = !state.paused;
      dom.pauseLive.textContent = state.paused ? "▶" : "Ⅱ";
      dom.pauseLive.title = state.paused ? "Resume live updates" : "Pause live updates";
      renderHeadline();
      if (!state.paused) refresh();
    });
    dom.graphSearch.addEventListener("input", () => {
      state.query = dom.graphSearch.value;
      state.fitNext = true;
      renderGraph();
    });
    dom.connectedOnly.addEventListener("change", () => {
      state.connectedOnly = dom.connectedOnly.checked;
      state.fitNext = true;
      renderGraph();
    });
    dom.showModules.addEventListener("change", () => {
      state.showModules = dom.showModules.checked;
      state.fitNext = true;
      renderGraph();
    });
    dom.showMentions.addEventListener("change", () => {
      state.showMentions = dom.showMentions.checked;
      state.fitNext = true;
      renderGraph();
    });
    dom.showSymbolLinks.addEventListener("change", () => {
      state.showSymbolLinks = dom.showSymbolLinks.checked;
      state.revision = null;
      state.fitNext = true;
      refresh(true);
    });
    dom.showSymbols.addEventListener("change", () => {
      state.showSymbols = dom.showSymbols.checked;
      dom.symbolOptions.hidden = !state.showSymbols;
      dom.graphStage.classList.toggle("symbols-hidden", !state.showSymbols);
      if (!state.showSymbols) state.neighborsOnly = null;
      state.fitNext = true;
      renderComponents();
      renderGraph();
    });
    document.querySelectorAll(".confidence-option").forEach((button) => {
      button.addEventListener("click", () => {
        state.confidence = button.dataset.confidence;
        document.querySelectorAll(".confidence-option").forEach((item) => item.classList.toggle("active", item === button));
        state.fitNext = true;
        renderGraph();
      });
    });
    dom.largestComponent.addEventListener("click", () => {
      const first = state.snapshot?.components[0];
      state.allComponents = false;
      state.selectedComponents = new Set(first ? [first.id] : []);
      state.neighborsOnly = null;
      state.fitNext = true;
      syncComponentButtons();
      renderComponents();
      renderGraph();
    });
    dom.latestComponent.addEventListener("click", () => {
      const component = defaultComponent(state.snapshot);
      state.allComponents = false;
      state.selectedComponents = new Set(component ? [component.id] : []);
      state.neighborsOnly = null;
      state.fitNext = true;
      syncComponentButtons();
      renderComponents();
      renderGraph();
    });
    dom.allComponents.addEventListener("click", () => {
      state.allComponents = true;
      state.neighborsOnly = null;
      state.fitNext = true;
      syncComponentButtons();
      renderComponents();
      renderGraph();
    });
    dom.toggleComponents.addEventListener("click", () => dom.componentPanel.classList.toggle("open"));
    dom.toggleInspector.addEventListener("click", () => dom.inspectorPanel.classList.toggle("open"));
    dom.closeInspector.addEventListener("click", () => dom.inspectorPanel.classList.remove("open"));
    document.addEventListener("keydown", (event) => {
      if (event.key === "/" && document.activeElement !== dom.graphSearch) {
        event.preventDefault();
        dom.graphSearch.focus();
      }
      if (event.key === "Escape") {
        state.neighborsOnly = null;
        state.selectedId = null;
        dom.componentPanel.classList.remove("open");
        dom.inspectorPanel.classList.remove("open");
        if (state.cy) state.cy.elements().unselect();
        renderGraph();
      }
    });
    window.addEventListener("resize", () => {
      state.cy?.resize();
    });
  }

  bindControls();
  refresh();
})();
