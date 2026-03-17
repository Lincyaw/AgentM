function ExecutionPage({ agents, events }) {
  const [splitX, setSplitX] = useState(null);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [selectedIdx, setSelectedIdx] = useState(-1);
  const containerRef = useRef(null);

  const leftWidth = splitX || 260;

  const handleDrag = useCallback((x) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    setSplitX(Math.max(180, Math.min(x - rect.left, rect.width - 200)));
  }, []);

  // Sorted agent list — running first, then by name
  const agentList = useMemo(() => {
    const entries = Object.entries(agents);
    const order = { running: 0, pending: 1, completed: 2, failed: 3 };
    return entries.sort((a, b) => {
      const oa = order[a[1].status] ?? 1;
      const ob = order[b[1].status] ?? 1;
      if (oa !== ob) return oa - ob;
      return a[0].localeCompare(b[0]);
    });
  }, [agents]);

  // Auto-select first agent when list populates
  useEffect(() => {
    if (!selectedAgent && agentList.length > 0) {
      setSelectedAgent(agentList[0][0]);
      setSelectedIdx(0);
    }
  }, [agentList.length]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === 'j' || e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIdx(i => {
          const next = Math.min(i + 1, agentList.length - 1);
          if (agentList[next]) setSelectedAgent(agentList[next][0]);
          return next;
        });
      }
      if (e.key === 'k' || e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIdx(i => {
          const next = Math.max(i - 1, 0);
          if (agentList[next]) setSelectedAgent(agentList[next][0]);
          return next;
        });
      }
      if (e.key === 'Escape') setSelectedAgent(null);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [agentList]);

  return (
    <div ref={containerRef} style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '10px 16px', borderBottom: `1px solid ${C.line}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontWeight: 600 }}>Execution Monitor</span>
        <span style={{ fontSize: 11, color: C.muted }}>
          {agentList.length} agent{agentList.length !== 1 ? 's' : ''}
          {agentList.filter(([, a]) => a.status === 'running').length > 0 &&
            <span style={{ color: C.teal, marginLeft: 8 }}>
              {agentList.filter(([, a]) => a.status === 'running').length} running
            </span>
          }
        </span>
      </div>
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Agent list */}
        <div style={{ width: leftWidth, overflowY: 'auto', borderRight: `1px solid ${C.line}` }}>
          {agentList.length === 0 && (
            <div style={{ padding: 40, textAlign: 'center', color: C.muted, fontSize: 12 }}>
              Waiting for agents...
            </div>
          )}
          {agentList.map(([id, agent], i) => {
            const isSelected = selectedAgent === id;
            const status = agent.status || 'pending';
            const color = STATUS_COLORS[status] || C.muted;
            return (
              <div
                key={id}
                onClick={() => { setSelectedAgent(id); setSelectedIdx(i); }}
                style={{
                  padding: '8px 12px',
                  cursor: 'pointer',
                  background: isSelected ? C.teal + '15' : 'transparent',
                  borderLeft: isSelected ? `2px solid ${C.teal}` : '2px solid transparent',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                }}
              >
                <StatusDot status={status} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: isSelected ? C.text : C.muted, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {id}
                  </div>
                  <div style={{ fontSize: 11, color: C.muted, display: 'flex', gap: 8, alignItems: 'center', marginTop: 2 }}>
                    {agent.taskType && <span>{agent.taskType}</span>}
                    {status === 'running' && agent.currentStep > 0 && (
                      <span>step {agent.currentStep}{agent.maxSteps ? `/${agent.maxSteps}` : ''}</span>
                    )}
                    {status === 'completed' && agent.durationSeconds && (
                      <span style={{ color: C.green }}>{agent.durationSeconds.toFixed(1)}s</span>
                    )}
                    {status === 'failed' && (
                      <span style={{ color: C.red }}>{(agent.error || 'error').slice(0, 20)}</span>
                    )}
                    {!agent.taskType && status === 'pending' && <span>pending</span>}
                  </div>
                </div>
                <Tag text={status} color={color} />
              </div>
            );
          })}
        </div>

        {/* Drag + Detail */}
        {selectedAgent && (
          <>
            <DragHandle onDrag={handleDrag} />
            <AgentDetailPanel agent={agents[selectedAgent]} agentId={selectedAgent} events={events} onClose={() => setSelectedAgent(null)} />
          </>
        )}

        {!selectedAgent && agentList.length > 0 && (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: C.muted }}>
            Select an agent to view details
          </div>
        )}
      </div>
      <div style={{ padding: '4px 12px', borderTop: `1px solid ${C.line}`, color: C.muted, fontSize: 11 }}>
        j/k: navigate &nbsp; Esc: deselect
      </div>
    </div>
  );
}

// Role colors and labels for message viewer
const ROLE_STYLE = {
  system:  { label: 'SYSTEM', color: C.purple, bg: C.purple + '12', border: C.purple + '30' },
  human:   { label: 'USER',   color: C.teal,   bg: C.teal + '12',   border: C.teal + '30' },
  ai:      { label: 'AI',     color: C.orange,  bg: C.orange + '12', border: C.orange + '30' },
  tool:    { label: 'TOOL',   color: C.yellow,  bg: C.yellow + '12', border: C.yellow + '30' },
  unknown: { label: '?',      color: C.muted,   bg: C.muted + '12',  border: C.muted + '30' },
};

function MessageBubble({ msg, index }) {
  const [expanded, setExpanded] = useState(false);
  const style = ROLE_STYLE[msg.role] || ROLE_STYLE.unknown;
  const content = msg.content || '';
  const isLong = content.length > 600;
  const displayContent = (!expanded && isLong) ? content.slice(0, 600) + '...' : content;

  return (
    <div style={{ padding: '8px 12px', borderBottom: `1px solid ${C.line}`, background: style.bg }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        <span style={{ fontSize: 10, fontWeight: 700, color: style.color, padding: '1px 6px', border: `1px solid ${style.border}`, borderRadius: 3 }}>{style.label}</span>
        {msg.role === 'tool' && msg.name && (
          <span style={{ fontSize: 11, color: C.yellow }}>{msg.name}</span>
        )}
        {msg.role === 'tool' && msg.tool_call_id && (
          <span style={{ fontSize: 10, color: C.muted }}>id: {msg.tool_call_id}</span>
        )}
        <span style={{ fontSize: 10, color: C.muted }}>#{index + 1}</span>
        <div style={{ flex: 1 }} />
        <CopyButton text={content} />
      </div>
      <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 12, color: C.text, lineHeight: 1.6, margin: 0 }}>{displayContent}</pre>
      {isLong && (
        <button onClick={() => setExpanded(!expanded)} style={{ background: 'none', border: 'none', color: C.teal, cursor: 'pointer', fontSize: 11, fontFamily: 'inherit', padding: '4px 0 0 0' }}>
          {expanded ? 'Show less' : `Show all (${content.length} chars)`}
        </button>
      )}
      {msg.tool_calls && msg.tool_calls.length > 0 && (
        <div style={{ marginTop: 6 }}>
          <div style={{ color: C.muted, fontSize: 10, marginBottom: 2 }}>Tool calls:</div>
          {msg.tool_calls.map((tc, i) => (
            <div key={i} style={{ padding: '4px 8px', marginBottom: 2, border: `1px solid ${C.line}`, borderRadius: 4, fontSize: 11 }}>
              <span style={{ color: C.yellow, fontWeight: 600 }}>{tc.name}</span>
              {tc.args && Object.keys(tc.args).length > 0 && (
                <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontSize: 11, color: C.text, margin: '4px 0 0 0' }}>
                  {JSON.stringify(tc.args, null, 2)}
                </pre>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function MessageListView({ llmStartEvents }) {
  const [selectedTurn, setSelectedTurn] = useState(llmStartEvents.length - 1);
  const turn = llmStartEvents[selectedTurn];
  const fullMessages = turn?.data?.messages || [];

  useEffect(() => {
    setSelectedTurn(llmStartEvents.length - 1);
  }, [llmStartEvents.length]);

  if (llmStartEvents.length === 0) {
    return (
      <div style={{ padding: 40, textAlign: 'center', color: C.muted }}>No LLM calls yet</div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Turn selector */}
      <div style={{ padding: '6px 12px', borderBottom: `1px solid ${C.line}`, display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
        <span style={{ fontSize: 11, color: C.muted }}>LLM Turn:</span>
        <select
          value={selectedTurn}
          onChange={e => setSelectedTurn(Number(e.target.value))}
          style={{ background: C.bg, color: C.text, border: `1px solid ${C.line}`, borderRadius: 3, padding: '2px 6px', fontSize: 11, fontFamily: 'inherit' }}
        >
          {llmStartEvents.map((ev, i) => (
            <option key={i} value={i}>
              Turn {i + 1} — {ev.data?.new_message_count || ev.data?.message_count || 0} new / {ev.data?.message_count || 0} total
            </option>
          ))}
        </select>
        <span style={{ fontSize: 10, color: C.muted }}>
          {fullMessages.length} new messages
        </span>
      </div>
      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {fullMessages.map((msg, i) => (
          <MessageBubble key={i} msg={msg} index={i} />
        ))}
      </div>
    </div>
  );
}

function AgentDetailPanel({ agent, agentId, events, onClose }) {
  const [tab, setTab] = useState('events');
  const [filterText, setFilterText] = useState('');
  const scrollRef = useRef(null);
  const autoScrollRef = useRef(true);

  const agentEvents = useMemo(() => {
    return events.filter(ev => ev.event_type && (ev.agent_path || []).slice(-1)[0] === agentId);
  }, [events, agentId]);

  const llmStartEvents = useMemo(() => {
    return agentEvents.filter(ev => ev.event_type === 'llm_start');
  }, [agentEvents]);

  // Auto-scroll on new events
  useEffect(() => {
    const el = scrollRef.current;
    if (el && autoScrollRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [agentEvents.length]);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    autoScrollRef.current = el.scrollTop + el.clientHeight >= el.scrollHeight - 50;
  };

  const status = agent?.status || 'pending';
  const startTime = agentEvents.length > 0 ? new Date(agentEvents[0].timestamp) : null;

  function formatRelativeTime(ts) {
    if (!startTime) return '00:00';
    const diff = Math.floor((new Date(ts) - startTime) / 1000);
    const m = String(Math.floor(diff / 60)).padStart(2, '0');
    const s = String(diff % 60).padStart(2, '0');
    return `${m}:${s}`;
  }

  function getEventInfo(ev) {
    const data = ev.data || {};
    const eventType = ev.event_type;
    switch (eventType) {
      case 'llm_start': {
        const count = data.message_count || 0;
        const msgs = data.messages || [];
        const roles = msgs.map(m => m.role).filter(Boolean);
        const summary = `${count} messages [${roles.join(', ')}]`;
        return { icon: '\u25B7', detail: summary, color: C.orange };
      }
      case 'tool_call':
        return { icon: '\u2192', detail: data.tool_name || 'tool', color: C.yellow };
      case 'tool_result': {
        const result = typeof data.result === 'string' ? data.result : JSON.stringify(data.result || '');
        return { icon: '\u2190', detail: `${data.tool_name || 'tool'}: ${result}`, color: C.green };
      }
      case 'llm_end': {
        const content = typeof data.content === 'string' ? data.content : (data.text || '');
        const detail = content || (data.tool_calls?.length ? `${data.tool_calls.length} tool call(s)` : 'response');
        return { icon: '\u25C6', detail, color: C.purple };
      }
      case 'task_dispatch':
        return { icon: '\u25B6', detail: `${data.task_type || 'task'}: dispatched`, color: C.teal };
      case 'task_complete': {
        const dur = data.duration_seconds ? `${data.duration_seconds.toFixed(1)}s` : '';
        return { icon: '\u2713', detail: `completed ${dur}`.trim(), color: C.green };
      }
      case 'task_fail':
        return { icon: '\u2717', detail: data.error || 'error', color: C.red };
      case 'task_abort':
        return { icon: '\u2298', detail: data.reason || 'aborted', color: C.red };
      case 'hypothesis_update':
        return { icon: '\u25C8', detail: `${data.hypothesis_id || '?'} \u2192 ${data.status || '?'}`, color: C.purple };
      case 'state_update':
        return { icon: '\u21BB', detail: `phase: ${data.current_phase || '?'}`, color: C.teal };
      default:
        return { icon: '\u25B8', detail: eventType || '', color: C.teal };
    }
  }

  const tabs = [
    { id: 'events', label: `Events (${agentEvents.length})` },
    { id: 'messages', label: `Messages (${llmStartEvents.length})` },
    { id: 'tools', label: `Tools (${agent?.toolCalls?.length || 0})` },
  ];

  // Clear filter when switching tabs
  const handleTabSwitch = (id) => { setTab(id); setFilterText(''); };

  // Filtered events
  const filteredEvents = useMemo(() => {
    if (!filterText) return agentEvents;
    const q = filterText.toLowerCase();
    return agentEvents.filter(ev => {
      const eventType = (ev.event_type || '').toLowerCase();
      if (eventType.includes(q)) return true;
      const info = getEventInfo(ev);
      if ((info.detail || '').toLowerCase().includes(q)) return true;
      const dataStr = JSON.stringify(ev.data || {}).toLowerCase();
      return dataStr.includes(q);
    });
  }, [agentEvents, filterText]);

  // Filtered tools
  const filteredTools = useMemo(() => {
    const all = agent?.toolCalls || [];
    if (!filterText) return all;
    const q = filterText.toLowerCase();
    return all.filter(tc => {
      if ((tc.name || '').toLowerCase().includes(q)) return true;
      if ((tc.status || '').toLowerCase().includes(q)) return true;
      const argsStr = JSON.stringify(tc.args || {}).toLowerCase();
      if (argsStr.includes(q)) return true;
      const resultStr = typeof tc.result === 'string' ? tc.result : JSON.stringify(tc.result || '');
      return resultStr.toLowerCase().includes(q);
    });
  }, [agent?.toolCalls, filterText]);

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ padding: '10px 16px', borderBottom: `1px solid ${C.line}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <StatusDot status={status} />
          <span style={{ fontWeight: 600, fontSize: 14 }}>{agentId}</span>
          <Tag text={status} color={STATUS_COLORS[status]} />
          {agent?.taskType && <Tag text={agent.taskType} color={C.teal} />}
        </div>
        <button onClick={onClose} style={{ background: 'none', border: 'none', color: C.muted, cursor: 'pointer', fontSize: 14, fontFamily: 'inherit' }}>[ESC]</button>
      </div>

      {/* Instruction */}
      {agent?.instruction && (
        <div style={{ padding: '8px 16px', borderBottom: `1px solid ${C.line}22` }}>
          <div style={{ color: C.muted, fontSize: 11, fontWeight: 600, marginBottom: 4 }}>Instruction</div>
          <div style={{ fontSize: 12, color: C.text, whiteSpace: 'pre-wrap', maxHeight: 80, overflow: 'auto' }}>{agent.instruction}</div>
        </div>
      )}

      {/* Tab bar */}
      <div style={{ display: 'flex', borderBottom: `1px solid ${C.line}`, padding: '0 12px', alignItems: 'center' }}>
        {tabs.map(t => (
          <button
            key={t.id}
            onClick={() => handleTabSwitch(t.id)}
            style={{
              background: 'none', border: 'none', borderBottom: tab === t.id ? `2px solid ${C.teal}` : '2px solid transparent',
              color: tab === t.id ? C.teal : C.muted, padding: '6px 12px', cursor: 'pointer', fontSize: 12, fontFamily: 'inherit',
            }}
          >{t.label}</button>
        ))}
        <div style={{ flex: 1 }} />
        <input
          type="text"
          placeholder="Filter..."
          value={filterText}
          onChange={e => setFilterText(e.target.value)}
          style={{
            background: C.bg, color: C.text, border: `1px solid ${C.line}`,
            padding: '3px 8px', borderRadius: 3, fontSize: 11, fontFamily: 'inherit',
            width: 160, outline: 'none',
          }}
          onFocus={e => e.target.style.borderColor = C.teal}
          onBlur={e => e.target.style.borderColor = C.line}
        />
        {filterText && (
          <span style={{ fontSize: 10, color: C.muted, marginLeft: 6 }}>
            {tab === 'events' ? filteredEvents.length : filteredTools.length} match
          </span>
        )}
      </div>

      {/* Content */}
      <div ref={scrollRef} onScroll={handleScroll} style={{ flex: 1, overflowY: 'auto' }}>
        {tab === 'events' && (
          <div>
            {filteredEvents.map((ev, i) => {
              const info = getEventInfo(ev);
              const truncated = info.detail?.length > 80 ? info.detail.slice(0, 80) + '...' : info.detail;
              return (
                <CollapsibleContent
                  key={i}
                  title={
                    React.createElement('span', { style: { display: 'flex', gap: 8, alignItems: 'center' } },
                      React.createElement('span', { style: { color: C.muted, fontSize: 11, minWidth: 36 } }, formatRelativeTime(ev.timestamp)),
                      React.createElement('span', { style: { color: info.color } }, info.icon),
                      React.createElement('span', { style: { color: C.text, fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } }, truncated)
                    )
                  }
                >
                  {info.detail && (
                    <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontSize: 12, color: C.text, marginBottom: 8, lineHeight: 1.6 }}>
                      {info.detail}
                    </pre>
                  )}
                  <JsonCard data={ev.data} title={ev.event_type || 'Event data'} />
                </CollapsibleContent>
              );
            })}
            {filteredEvents.length === 0 && (
              <div style={{ padding: 40, textAlign: 'center', color: C.muted }}>{filterText ? 'No matching events' : 'No events yet'}</div>
            )}
          </div>
        )}

        {tab === 'messages' && (
          <MessageListView llmStartEvents={llmStartEvents} />
        )}

        {tab === 'tools' && (
          <div style={{ padding: '8px 0' }}>
            {filteredTools.map((tc, i) => (
              <CollapsibleContent key={i} title={
                React.createElement('span', { style: { display: 'flex', gap: 8, alignItems: 'center' } },
                  React.createElement('span', { style: { color: C.muted } }, `${i + 1}.`),
                  React.createElement('span', { style: { color: C.yellow } }, tc.name),
                  React.createElement('span', { style: { color: tc.status === 'success' ? C.green : tc.status === 'error' ? C.red : C.teal } },
                    tc.status === 'success' ? '\u2713' : tc.status === 'error' ? '\u2717' : '\u25CC'
                  ),
                  tc.durationSeconds ? React.createElement('span', { style: { color: C.muted } }, `${tc.durationSeconds.toFixed(1)}s`) : null
                )
              }>
                {tc.args && Object.keys(tc.args).length > 0 && (
                  <div style={{ marginBottom: 8 }}>
                    <div style={{ color: C.muted, fontSize: 10, marginBottom: 2 }}>Arguments:</div>
                    <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontSize: 11, color: C.text }}>
                      {JSON.stringify(tc.args, null, 2)}
                    </pre>
                  </div>
                )}
                {tc.result && (
                  <div>
                    <div style={{ color: C.muted, fontSize: 10, marginBottom: 2 }}>Result:</div>
                    <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontSize: 11, color: C.text }}>
                      {typeof tc.result === 'string' ? tc.result : JSON.stringify(tc.result, null, 2)}
                    </pre>
                  </div>
                )}
              </CollapsibleContent>
            ))}
            {filteredTools.length === 0 && (
              <div style={{ padding: 40, textAlign: 'center', color: C.muted }}>{filterText ? 'No matching tools' : 'No tool calls yet'}</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
