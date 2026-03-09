function RCAConversationView({ state, events }) {
  const [phaseFilter, setPhaseFilter] = useState('all');
  const [liveMode, setLiveMode] = useState(true);
  const scrollRef = useRef(null);

  const notebook = state.notebook || {};
  const phases = ['exploration', 'generation', 'verification', 'confirmation'];

  useEffect(() => {
    if (liveMode && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [state, liveMode]);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    if (el.scrollTop + el.clientHeight < el.scrollHeight - 50) setLiveMode(false);
  };

  // Build phase data
  const phaseData = useMemo(() => {
    const result = [];

    // Exploration
    if (notebook.exploration_history?.length > 0 || notebook.collected_data) {
      result.push({
        phase: 'exploration',
        label: 'Phase 1: Exploration',
        content: { agents: notebook.collected_data || {}, history: notebook.exploration_history || [] },
      });
    }

    // Generation
    if (notebook.hypotheses && Object.keys(notebook.hypotheses).length > 0) {
      result.push({
        phase: 'generation',
        label: 'Phase 2: Hypothesis Generation',
        content: { hypotheses: notebook.hypotheses },
      });
    }

    // Verification
    const verifying = notebook.hypotheses
      ? Object.values(notebook.hypotheses).filter(h => {
          const s = (h.status || '').toLowerCase();
          return s === 'investigating' || s === 'confirmed' || s === 'rejected';
        })
      : [];
    if (verifying.length > 0) {
      result.push({
        phase: 'verification',
        label: 'Phase 3: Verification',
        content: { hypotheses: verifying },
      });
    }

    // Confirmation
    if (notebook.confirmed_hypothesis) {
      const confirmed = notebook.hypotheses?.[notebook.confirmed_hypothesis];
      result.push({
        phase: 'confirmation',
        label: 'Phase 4: Confirmation',
        content: { confirmed, id: notebook.confirmed_hypothesis },
      });
    }

    return result;
  }, [notebook]);

  const visiblePhases = phaseFilter === 'all' ? phaseData : phaseData.filter(p => p.phase === phaseFilter);

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '10px 16px', borderBottom: `1px solid ${C.line}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontWeight: 600 }}>Conversation View</span>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <span style={{ fontSize: 11, color: C.muted }}>Phase:</span>
          <select
            value={phaseFilter}
            onChange={e => setPhaseFilter(e.target.value)}
            style={{ background: C.bg, color: C.text, border: `1px solid ${C.line}`, padding: '3px 8px', borderRadius: 3, fontFamily: 'inherit', fontSize: 12 }}
          >
            <option value="all">All</option>
            {phases.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
          <button
            onClick={() => setLiveMode(!liveMode)}
            style={{
              background: liveMode ? C.green + '20' : 'transparent',
              border: `1px solid ${liveMode ? C.green : C.line}`,
              color: liveMode ? C.green : C.muted,
              padding: '3px 10px', borderRadius: 3, cursor: 'pointer', fontSize: 11, fontFamily: 'inherit',
            }}
          >{liveMode ? '\u21BB Live' : '\u21BB Paused'}</button>
        </div>
      </div>
      <div ref={scrollRef} onScroll={handleScroll} style={{ flex: 1, overflowY: 'auto', padding: 16 }}>
        {/* User message */}
        {notebook.task_description && (
          <div style={{ background: C.panel, border: `1px solid ${C.line}`, borderRadius: 6, padding: '10px 14px', marginBottom: 16 }}>
            <div style={{ fontSize: 11, color: C.muted, marginBottom: 4 }}>User</div>
            <div>{notebook.task_description}</div>
          </div>
        )}

        {visiblePhases.map((pd, i) => (
          <PhaseCard key={i} phase={pd.phase} label={pd.label}>
            {pd.phase === 'exploration' && <ExplorationContent data={pd.content} />}
            {pd.phase === 'generation' && <HypothesisTable hypotheses={pd.content.hypotheses} />}
            {pd.phase === 'verification' && <VerificationContent hypotheses={pd.content.hypotheses} />}
            {pd.phase === 'confirmation' && <ConfirmationContent confirmed={pd.content.confirmed} id={pd.content.id} />}
          </PhaseCard>
        ))}

        {visiblePhases.length === 0 && (
          <div style={{ padding: 40, textAlign: 'center', color: C.muted }}>
            {notebook.current_phase ? `Current phase: ${notebook.current_phase}` : 'Waiting for conversation data...'}
          </div>
        )}
      </div>
    </div>
  );
}

function PhaseCard({ phase, label, children }) {
  return (
    <div style={{
      borderLeft: `3px solid ${PHASE_COLORS[phase] || C.muted}`,
      background: C.panel,
      border: `1px solid ${C.line}`,
      borderLeftWidth: 3,
      borderLeftColor: PHASE_COLORS[phase] || C.muted,
      borderRadius: 6,
      padding: '12px 14px',
      marginBottom: 16,
    }}>
      <div style={{ fontWeight: 600, color: PHASE_COLORS[phase] || C.text, marginBottom: 10, fontSize: 13 }}>{label}</div>
      {children}
    </div>
  );
}

function ExplorationContent({ data }) {
  const agents = data.agents || {};
  return (
    <div>
      {data.history?.length > 0 && (
        <div style={{ color: C.muted, fontSize: 12, marginBottom: 8 }}>
          Dispatched agents: {data.history.map(h => h.agent_id || h.agent).filter(Boolean).join(', ')}
        </div>
      )}
      {Object.entries(agents).map(([name, result]) => (
        <CollapsibleContent key={name} title={`${name} (result)`}>
          <div style={{ fontSize: 11 }}><JsonHighlighter data={result} /></div>
        </CollapsibleContent>
      ))}
    </div>
  );
}

function HypothesisTable({ hypotheses }) {
  const entries = typeof hypotheses === 'object' && !Array.isArray(hypotheses)
    ? Object.values(hypotheses) : (hypotheses || []);
  if (entries.length === 0) return <div style={{ color: C.muted }}>No hypotheses yet</div>;

  const statusColor = (s) => {
    const key = (s || '').toUpperCase();
    const sc = { FORMED: C.muted, INVESTIGATING: C.teal, CONFIRMED: C.green, REJECTED: C.red, REFINED: C.purple, INCONCLUSIVE: C.yellow };
    return sc[key] || C.muted;
  };

  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
      <thead>
        <tr style={{ borderBottom: `1px solid ${C.line}` }}>
          <th style={{ textAlign: 'left', padding: '4px 8px', color: C.muted, fontWeight: 500 }}>ID</th>
          <th style={{ textAlign: 'left', padding: '4px 8px', color: C.muted, fontWeight: 500 }}>Description</th>
          <th style={{ textAlign: 'left', padding: '4px 8px', color: C.muted, fontWeight: 500 }}>Status</th>
        </tr>
      </thead>
      <tbody>
        {entries.map((h, i) => (
          <tr key={h.id || i} style={{ borderBottom: `1px solid ${C.line}22` }}>
            <td style={{ padding: '4px 8px', color: C.purple }}>{h.id || `H${i + 1}`}</td>
            <td style={{ padding: '4px 8px' }}>{h.description}</td>
            <td style={{ padding: '4px 8px' }}><Tag text={(h.status || '').toUpperCase()} color={statusColor(h.status)} /></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function VerificationContent({ hypotheses }) {
  return (
    <div>
      {hypotheses.map((h, i) => (
        <div key={h.id || i} style={{ marginBottom: 10 }}>
          <div style={{ fontWeight: 600, color: C.orange, marginBottom: 4 }}>
            Verifying {h.id}: {h.description}
          </div>
          {h.evidence?.length > 0 && (
            <div style={{ fontSize: 12, color: C.green, marginLeft: 8 }}>
              Supporting: {h.evidence.join('; ')}
            </div>
          )}
          {h.counter_evidence?.length > 0 && (
            <div style={{ fontSize: 12, color: C.red, marginLeft: 8 }}>
              Rejecting: {h.counter_evidence.join('; ')}
            </div>
          )}
          <div style={{ marginTop: 4, marginLeft: 8 }}>
            <Tag
              text={(h.status || '').toUpperCase()}
              color={(() => { const s = (h.status || '').toLowerCase(); return s === 'confirmed' ? C.green : s === 'rejected' ? C.red : C.teal; })()}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function ConfirmationContent({ confirmed, id }) {
  if (!confirmed) return <div style={{ color: C.muted }}>Awaiting confirmation...</div>;
  return (
    <div>
      <div style={{ color: C.green, fontWeight: 600, marginBottom: 8 }}>
        Root cause confirmed: {id} &mdash; {confirmed.description}
      </div>
      {confirmed.evidence?.length > 0 && (
        <div style={{ fontSize: 12 }}>
          <div style={{ color: C.muted, marginBottom: 4 }}>Evidence:</div>
          {confirmed.evidence.map((e, i) => (
            <div key={i} style={{ paddingLeft: 8, color: C.text }}>&#9656; {e}</div>
          ))}
        </div>
      )}
    </div>
  );
}

// Plugin definition
const RCAPlugin = {
  id: 'rca_hypothesis',
  label: 'RCA Hypothesis-Driven',
  ConversationView: RCAConversationView,
  parseState: (data) => ({
    notebook: data.notebook || data,
    hypotheses: data.notebook?.hypotheses || data.hypotheses,
    currentPhase: data.notebook?.current_phase || data.current_phase,
    explorationHistory: data.notebook?.exploration_history,
    collectedData: data.notebook?.collected_data,
    phaseSummaries: data.notebook?.phase_summaries,
  }),
  decorateTopologyNode: (agentId, state) => {
    if (agentId === 'orchestrator' && state?.currentPhase) {
      return { sublabel: `phase: ${state.currentPhase}` };
    }
    return {};
  },
};

const SCENARIO_PLUGINS = {
  rca_hypothesis: RCAPlugin,
};

function ConversationPage({ state, events, plugin }) {
  if (!plugin) {
    return (
      <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: C.muted }}>
        No scenario plugin loaded
      </div>
    );
  }
  const View = plugin.ConversationView;
  return <View state={state} events={events} />;
}
