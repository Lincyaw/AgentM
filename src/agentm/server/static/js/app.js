const NAV_ITEMS = [
  { id: 'topology', icon: '\u2B21', label: 'Topology' },
  { id: 'execution', icon: '\u25B6', label: 'Execution' },
  { id: 'conversation', icon: '\u25C8', label: 'Conversation' },
  { id: 'debug', icon: '\u2699', label: 'Debug' },
];

function App() {
  const [page, setPage] = useState('topology');
  const [agents, setAgents] = useState({});
  const [events, setEvents] = useState([]);
  const [topology, setTopology] = useState(null);
  const [scenarioState, setScenarioState] = useState({});
  const [threadId, setThreadId] = useState(null);

  // Determine active plugin
  const plugin = useMemo(() => {
    if (topology?.scenario_id && SCENARIO_PLUGINS[topology.scenario_id]) {
      return SCENARIO_PLUGINS[topology.scenario_id];
    }
    return SCENARIO_PLUGINS.rca_hypothesis;
  }, [topology]);

  // Fetch topology on mount
  useEffect(() => {
    fetch('/api/topology').then(r => r.json()).then(data => {
      setTopology(data);
      if (data.thread_id) setThreadId(data.thread_id);
    }).catch(() => { });
  }, []);

  // WebSocket event handler
  const handleEvent = useCallback((event) => {
    setEvents(prev => [...prev, event]);

    const agentPath = event.agent_path || [];
    const agentId = agentPath.slice(-1)[0];
    if (!agentId) return;
    const data = event.data || {};
    const ts = event.timestamp;
    const eventType = event.event_type;

    // Update agent state
    setAgents(prev => {
      const existing = prev[agentId] || {
        agentId, status: 'pending', currentStep: 0, maxSteps: null,
        startedAt: null, completedAt: null, durationSeconds: null,
        toolCalls: [], messages: [], taskType: null, instruction: null,
      };

      const updated = { ...existing };

      // Mark as running on first event
      if (updated.status === 'pending') {
        updated.status = 'running';
        updated.startedAt = ts;
      }

      // Process trajectory events (event_type present)
      if (eventType) {
        switch (eventType) {
          case 'llm_start':
            updated.messages = [...updated.messages, {
              type: 'llm_start',
              eventType,
              content: `LLM input: ${data.message_count || 0} messages`,
              messages: data.messages || [],
              timestamp: ts,
            }];
            break;

          case 'tool_call':
            updated.toolCalls = [...updated.toolCalls, {
              name: data.tool_name || 'unknown',
              args: data.args || {},
              status: 'running',
              startedAt: ts,
              completedAt: null,
              durationSeconds: null,
              result: null,
            }];
            updated.currentStep = updated.toolCalls.length;
            break;

          case 'tool_result':
            updated.toolCalls = updated.toolCalls.map(tc => {
              if (tc.status === 'running' && (tc.name === data.tool_name || !data.tool_name)) {
                const dur = tc.startedAt ? (new Date(ts) - new Date(tc.startedAt)) / 1000 : null;
                return { ...tc, status: 'success', completedAt: ts, durationSeconds: dur, result: data.result || data.content || '' };
              }
              return tc;
            });
            updated.messages = [...updated.messages, {
              type: 'tool_result',
              eventType,
              toolName: data.tool_name || 'tool',
              content: data.result || data.content || '',
              timestamp: ts,
            }];
            break;

          case 'llm_end':
            updated.messages = [...updated.messages, {
              type: 'ai',
              eventType,
              content: data.content || data.text || '',
              toolCalls: data.tool_calls || [],
              timestamp: ts,
            }];
            break;

          case 'task_dispatch':
            updated.status = 'running';
            updated.taskType = data.task_type || null;
            updated.instruction = data.instruction || null;
            updated.messages = [...updated.messages, {
              type: 'task_dispatch',
              eventType,
              content: `Dispatched: ${data.task_type || 'task'}`,
              timestamp: ts,
            }];
            break;

          case 'task_complete':
            updated.status = 'completed';
            updated.completedAt = ts;
            updated.durationSeconds = data.duration_seconds || (updated.startedAt ? (new Date(ts) - new Date(updated.startedAt)) / 1000 : null);
            updated.result = data.result || data.content || '';
            updated.messages = [...updated.messages, {
              type: 'task_complete',
              eventType,
              content: data.result || data.content || 'completed',
              timestamp: ts,
            }];
            break;

          case 'task_fail':
            updated.status = 'failed';
            updated.error = data.error || 'unknown error';
            updated.messages = [...updated.messages, {
              type: 'task_fail',
              eventType,
              content: data.error || 'failed',
              timestamp: ts,
            }];
            break;

          case 'task_abort':
            updated.status = 'failed';
            updated.error = data.reason || 'aborted';
            updated.messages = [...updated.messages, {
              type: 'task_abort',
              eventType,
              content: data.reason || 'aborted',
              timestamp: ts,
            }];
            break;

          case 'hypothesis_update':
            updated.messages = [...updated.messages, {
              type: 'hypothesis_update',
              eventType,
              content: `${data.hypothesis_id || '?'} → ${data.status || '?'}`,
              timestamp: ts,
            }];
            break;

          default:
            updated.messages = [...updated.messages, {
              type: eventType,
              eventType,
              content: JSON.stringify(data),
              timestamp: ts,
            }];
            break;
        }
      } else {
        // Fallback: process raw broadcast events for messages array
        const messages = data.messages || [];
        for (const msg of messages) {
          if (!msg) continue;

          if (msg.tool_calls?.length) {
            for (const tc of msg.tool_calls) {
              updated.toolCalls = [...updated.toolCalls, {
                name: tc.name,
                args: tc.args || {},
                status: 'running',
                startedAt: ts,
                completedAt: null,
                durationSeconds: null,
                result: null,
              }];
            }
            updated.currentStep = updated.toolCalls.length;
          }

          if (msg.type === 'tool' && msg.tool_call_id) {
            updated.toolCalls = updated.toolCalls.map(tc => {
              if (tc.status === 'running') {
                const dur = tc.startedAt ? (new Date(ts) - new Date(tc.startedAt)) / 1000 : null;
                return { ...tc, status: 'success', completedAt: ts, durationSeconds: dur, result: (msg.content || '').slice(0, 200) };
              }
              return tc;
            });
          }

          updated.messages = [...updated.messages, {
            type: msg.type || 'unknown',
            eventType: null,
            content: msg.content || '',
            toolCallId: msg.tool_call_id || null,
            timestamp: ts,
          }];
        }

        // For non-message data (e.g., state updates with notebook)
        if (!messages.length) {
          for (const [nodeName, nodeData] of Object.entries(data)) {
            if (nodeData?.messages) {
              for (const msg of nodeData.messages) {
                updated.messages = [...updated.messages, {
                  type: msg.type || 'unknown',
                  eventType: null,
                  content: msg.content || '',
                  toolCallId: msg.tool_call_id || null,
                  timestamp: ts,
                }];
              }
            }
          }
        }
      }

      return { ...prev, [agentId]: updated };
    });

    // Update scenario state — two paths:
    // 1. state_update events carry full notebook (preferred, from builder.py)
    // 2. Incremental build from individual events (fallback for existing sessions)
    if (eventType === 'state_update' && data.notebook) {
      // Full notebook snapshot from backend
      if (plugin?.parseState) {
        const parsed = plugin.parseState(data);
        if (parsed && Object.keys(parsed).some(k => parsed[k] !== undefined)) {
          setScenarioState(prev => ({ ...prev, ...parsed }));
        }
      }
    } else if (eventType) {
      // Incremental: build notebook from discrete events
      setScenarioState(prev => {
        const nb = { ...(prev.notebook || {}) };

        if (eventType === 'task_dispatch') {
          nb.task_description = nb.task_description || data.instruction || data.task_type || '';
          nb.current_phase = nb.current_phase || 'exploration';
          // Track exploration history
          const history = [...(nb.exploration_history || [])];
          history.push({
            agent_id: data.agent_id || agentId,
            agent: data.agent_id || agentId,
            task_type: data.task_type || '',
            timestamp: ts,
          });
          nb.exploration_history = history;
        }

        if (eventType === 'task_complete') {
          // Collect agent results
          const collected = { ...(nb.collected_data || {}) };
          const resultAgent = data.agent_id || agentId;
          collected[resultAgent] = data.result || data.content || '';
          nb.collected_data = collected;
        }

        if (eventType === 'hypothesis_update') {
          const hypotheses = { ...(nb.hypotheses || {}) };
          const hid = data.hypothesis_id || data.id;
          if (hid) {
            hypotheses[hid] = {
              ...(hypotheses[hid] || {}),
              id: hid,
              description: data.description || hypotheses[hid]?.description || '',
              status: data.status || hypotheses[hid]?.status || 'formed',
              evidence: data.evidence || hypotheses[hid]?.evidence || [],
              counter_evidence: data.counter_evidence || hypotheses[hid]?.counter_evidence || [],
            };
            nb.hypotheses = hypotheses;
            nb.current_phase = data.status === 'confirmed' ? 'confirmation' : (data.status === 'investigating' || data.status === 'rejected') ? 'verification' : nb.current_phase;
            if (data.status === 'confirmed') {
              nb.confirmed_hypothesis = hid;
            }
          }
        }

        // Also catch orchestrator llm_end for phase detection
        if (eventType === 'llm_end' && agentPath[0] === 'orchestrator') {
          const content = (data.content || '').toLowerCase();
          if (content.includes('phase: generation') || content.includes('hypothesis generation')) {
            nb.current_phase = nb.current_phase || 'generation';
          }
        }

        return { ...prev, notebook: nb };
      });
    }

    // Try to extract thread_id from data
    if ((data.task_id || data.notebook?.task_id) && !threadId) {
      setThreadId(data.task_id || data.notebook?.task_id);
    }
  }, [plugin, threadId]);

  const wsStatus = useWebSocket(handleEvent);

  // Global keyboard shortcuts
  useEffect(() => {
    const handler = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === '1') setPage('topology');
      if (e.key === '2') setPage('execution');
      if (e.key === '3') setPage('conversation');
      if (e.key === '4') setPage('debug');
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  return (
    <>
      {/* Header */}
      <div style={{
        height: 40,
        padding: '0 16px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: `1px solid ${C.line}`,
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontWeight: 700, color: C.teal, fontSize: 14 }}>AgentM</span>
          <span style={{ color: C.muted, fontSize: 12 }}>Dashboard</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {plugin && <Tag text={plugin.label} color={C.purple} />}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11 }}>
            <span style={{
              width: 8, height: 8, borderRadius: '50%',
              background: wsStatus === 'connected' ? C.green : C.red,
            }} />
            <span style={{ color: wsStatus === 'connected' ? C.green : C.red }}>
              {wsStatus === 'connected' ? 'CONNECTED' : 'DISCONNECTED'}
            </span>
          </div>
        </div>
      </div>

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* NavBar */}
        <div style={{
          width: 48,
          borderRight: `1px solid ${C.line}`,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          paddingTop: 8,
          gap: 4,
          flexShrink: 0,
        }}>
          {NAV_ITEMS.map(item => {
            const isActive = page === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setPage(item.id)}
                title={item.label}
                style={{
                  width: 36,
                  height: 36,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: 'none',
                  borderRadius: 6,
                  cursor: 'pointer',
                  fontSize: 16,
                  fontFamily: 'inherit',
                  background: isActive ? C.teal + '20' : 'transparent',
                  color: isActive ? C.teal : C.muted,
                  transition: 'all 0.15s',
                }}
              >{item.icon}</button>
            );
          })}
        </div>

        {/* Main content */}
        <div style={{ flex: 1, overflow: 'hidden' }}>
          {page === 'topology' && <TopologyPage agents={agents} topology={topology} scenarioState={scenarioState} plugin={plugin} />}
          {page === 'execution' && <ExecutionPage agents={agents} events={events} />}
          {page === 'conversation' && <ConversationPage state={scenarioState} events={events} plugin={plugin} />}
          {page === 'debug' && <DebugPage threadId={threadId} />}
        </div>
      </div>
    </>
  );
}

// Mount
ReactDOM.createRoot(document.getElementById('root')).render(<App />);
