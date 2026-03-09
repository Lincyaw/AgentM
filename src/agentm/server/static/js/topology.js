function TopologyPage({ agents, topology, scenarioState, plugin }) {
  const svgRef = useRef(null);
  const simRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 500 });

  useEffect(() => {
    const el = svgRef.current?.parentElement;
    if (el) {
      setDimensions({ width: el.clientWidth, height: el.clientHeight - 40 });
      const obs = new ResizeObserver(entries => {
        const { width, height } = entries[0].contentRect;
        setDimensions({ width, height: height - 40 });
      });
      obs.observe(el);
      return () => obs.disconnect();
    }
  }, []);

  useEffect(() => {
    if (!svgRef.current || !topology) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const { width, height } = dimensions;
    const g = svg.append('g');

    // Build nodes
    const nodes = [
      { id: 'orchestrator', label: 'Orchestrator', isOrchestrator: true },
      ...topology.agents.map(a => ({ id: a.agent_id, label: a.agent_id, maxSteps: a.max_steps })),
    ];
    const links = topology.agents.map(a => ({ source: 'orchestrator', target: a.agent_id }));

    // Force simulation
    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(160))
      .force('charge', d3.forceManyBody().strength(-400))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('y', d3.forceY(height / 2).strength(0.05));

    simRef.current = sim;

    // Edges
    const link = g.selectAll('.link')
      .data(links).enter().append('line')
      .attr('class', 'link')
      .attr('stroke', C.line)
      .attr('stroke-width', 1.5);

    // Node groups
    const node = g.selectAll('.node')
      .data(nodes).enter().append('g')
      .attr('class', 'node')
      .call(d3.drag()
        .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    // Node box
    node.append('rect')
      .attr('width', 140).attr('height', 56).attr('rx', 6)
      .attr('x', -70).attr('y', -28)
      .attr('fill', C.panel).attr('stroke', C.line).attr('stroke-width', 1);

    // Agent name
    node.append('text')
      .attr('text-anchor', 'middle').attr('y', -6)
      .attr('fill', C.text).attr('font-size', 13).attr('font-family', 'inherit')
      .text(d => d.label);

    // Status text
    node.append('text')
      .attr('class', 'status-text')
      .attr('text-anchor', 'middle').attr('y', 12)
      .attr('fill', C.muted).attr('font-size', 11).attr('font-family', 'inherit')
      .text('pending');

    // Status dot
    node.append('circle')
      .attr('class', 'status-dot')
      .attr('cx', -56).attr('cy', -16).attr('r', 4)
      .attr('fill', C.muted);

    sim.on('tick', () => {
      link
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    // Zoom
    svg.call(d3.zoom().scaleExtent([0.3, 3]).on('zoom', (e) => g.attr('transform', e.transform)));

    return () => sim.stop();
  }, [topology, dimensions]);

  // Update node appearances from live agent state
  useEffect(() => {
    if (!svgRef.current || !topology) return;
    const svg = d3.select(svgRef.current);

    svg.selectAll('.node').each(function(d) {
      const agent = agents[d.id];
      const status = agent?.status || 'pending';
      const color = STATUS_COLORS[status] || C.muted;

      d3.select(this).select('rect').attr('stroke', color);
      d3.select(this).select('.status-dot').attr('fill', color);

      let statusText = status;
      if (status === 'running' && agent?.currentStep) {
        const max = agent.maxSteps || d.maxSteps;
        statusText = max ? `${agent.currentStep}/${max}` : `step ${agent.currentStep}`;
      }
      if (status === 'completed' && agent?.durationSeconds) {
        statusText = `${agent.durationSeconds.toFixed(1)}s`;
      }
      if (status === 'failed' && agent?.error) {
        statusText = agent.error.slice(0, 15);
      }

      // Plugin decoration
      if (plugin?.decorateTopologyNode) {
        const dec = plugin.decorateTopologyNode(d.id, scenarioState);
        if (dec?.sublabel) statusText = dec.sublabel;
      }

      d3.select(this).select('.status-text').text(statusText).attr('fill', color);
    });

    // Update edge colors
    svg.selectAll('.link').each(function(d) {
      const agent = agents[d.target.id || d.target];
      const status = agent?.status || 'pending';
      d3.select(this).attr('stroke', STATUS_COLORS[status] || C.line);
    });
  }, [agents, scenarioState, plugin, topology]);

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '10px 16px', borderBottom: `1px solid ${C.line}`, display: 'flex', justifyContent: 'space-between' }}>
        <span style={{ fontWeight: 600 }}>Agent Topology</span>
        <TopologyLegend />
      </div>
      <div style={{ flex: 1, overflow: 'hidden' }}>
        <svg ref={svgRef} width={dimensions.width} height={dimensions.height} />
      </div>
    </div>
  );
}

function TopologyLegend() {
  const items = [
    { label: 'completed', color: C.green, icon: '\u25CF' },
    { label: 'running', color: C.teal, icon: '\u25CC' },
    { label: 'failed', color: C.red, icon: '\u2717' },
    { label: 'pending', color: C.muted, icon: '\u25CB' },
  ];
  return (
    <div style={{ display: 'flex', gap: 12, fontSize: 11, color: C.muted }}>
      {items.map(i => (
        <span key={i.label} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ color: i.color }}>{i.icon}</span> {i.label}
        </span>
      ))}
    </div>
  );
}
