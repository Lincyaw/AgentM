function DebugPage({ threadId }) {
  const [checkpoints, setCheckpoints] = useState([]);
  const [selectedCp, setSelectedCp] = useState(null);
  const [cpState, setCpState] = useState(null);
  const [splitX, setSplitX] = useState(null);
  const [loading, setLoading] = useState(false);
  const [actionMsg, setActionMsg] = useState('');
  const containerRef = useRef(null);

  const leftWidth = splitX || 280;

  useEffect(() => {
    if (!threadId) return;
    const fetchHistory = () => {
      fetch(`/api/tasks/${threadId}/history`)
        .then(r => r.json())
        .then(setCheckpoints)
        .catch(() => {});
    };
    fetchHistory();
    const interval = setInterval(fetchHistory, 5000);
    return () => clearInterval(interval);
  }, [threadId]);

  const selectCheckpoint = async (cp) => {
    setSelectedCp(cp);
    if (!threadId) return;
    setLoading(true);
    try {
      const r = await fetch(`/api/tasks/${threadId}/history/${cp.checkpoint_id}`);
      const data = await r.json();
      setCpState(data);
    } catch {
      setCpState(null);
    }
    setLoading(false);
  };

  const handleReplay = async () => {
    if (!threadId || !selectedCp) return;
    setActionMsg('Replaying...');
    try {
      const r = await fetch(`/api/tasks/${threadId}/resume`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ checkpoint_id: selectedCp.checkpoint_id }),
      });
      const data = await r.json();
      setActionMsg(`Replay: ${data.status || 'done'}`);
    } catch (e) {
      setActionMsg(`Replay error: ${e.message}`);
    }
    setTimeout(() => setActionMsg(''), 3000);
  };

  const handleExport = () => {
    if (!threadId) return;
    window.open(`/api/tasks/${threadId}/trajectory`, '_blank');
  };

  const handleDrag = useCallback((x) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    setSplitX(Math.max(180, Math.min(x - rect.left, rect.width - 200)));
  }, []);

  // Keyboard
  useEffect(() => {
    const handler = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        const idx = checkpoints.findIndex(c => c.checkpoint_id === selectedCp?.checkpoint_id);
        const next = e.key === 'ArrowDown' ? Math.min(idx + 1, checkpoints.length - 1) : Math.max(idx - 1, 0);
        if (checkpoints[next]) selectCheckpoint(checkpoints[next]);
      }
      if (e.key === 'r') handleReplay();
      if (e.key === 'e') handleExport();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [checkpoints, selectedCp, threadId]);

  return (
    <div ref={containerRef} style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '10px 16px', borderBottom: `1px solid ${C.line}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontWeight: 600 }}>Debug Panel</span>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {actionMsg && <span style={{ color: C.teal, fontSize: 11 }}>{actionMsg}</span>}
          <button
            onClick={handleExport}
            style={{ background: 'none', border: `1px solid ${C.line}`, color: C.muted, padding: '3px 10px', borderRadius: 3, cursor: 'pointer', fontSize: 11, fontFamily: 'inherit' }}
          >Export</button>
        </div>
      </div>
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Checkpoint list */}
        <div style={{ width: leftWidth, overflowY: 'auto', borderRight: `1px solid ${C.line}` }}>
          {checkpoints.length === 0 && (
            <div style={{ padding: 20, color: C.muted, textAlign: 'center', fontSize: 12 }}>
              {threadId ? 'No checkpoints found' : 'No thread ID — start an investigation'}
            </div>
          )}
          {checkpoints.map((cp, i) => {
            const isSelected = selectedCp?.checkpoint_id === cp.checkpoint_id;
            return (
              <div
                key={cp.checkpoint_id || i}
                onClick={() => selectCheckpoint(cp)}
                style={{
                  padding: '6px 12px',
                  cursor: 'pointer',
                  background: isSelected ? C.teal + '15' : 'transparent',
                  borderLeft: isSelected ? `2px solid ${C.teal}` : '2px solid transparent',
                  display: 'flex',
                  gap: 8,
                  alignItems: 'center',
                  fontSize: 12,
                }}
              >
                <span style={{ color: isSelected ? C.teal : C.muted }}>{isSelected ? '\u25CF' : '\u25CB'}</span>
                <span>Step {cp.step ?? i}</span>
                <span style={{ color: C.muted }}>{cp.node_name}</span>
                {cp.source && <span style={{ color: C.muted, fontSize: 10 }}>{cp.source}</span>}
              </div>
            );
          })}
          {selectedCp && (
            <div style={{ padding: '12px', borderTop: `1px solid ${C.line}` }}>
              <div style={{ fontSize: 11, color: C.muted, marginBottom: 8 }}>Actions:</div>
              <button
                onClick={handleReplay}
                style={{
                  display: 'block', width: '100%', marginBottom: 6, padding: '6px',
                  background: 'transparent', border: `1px solid ${C.teal}`, color: C.teal,
                  borderRadius: 3, cursor: 'pointer', fontSize: 11, fontFamily: 'inherit',
                }}
              >&#9664; Replay from here</button>
            </div>
          )}
        </div>

        <DragHandle onDrag={handleDrag} />

        {/* State inspector */}
        <div style={{ flex: 1, overflowY: 'auto', padding: 12 }}>
          {loading && <div style={{ color: C.muted, textAlign: 'center', padding: 20 }}>Loading...</div>}
          {!loading && !cpState && (
            <div style={{ color: C.muted, textAlign: 'center', padding: 40 }}>Select a checkpoint to inspect</div>
          )}
          {!loading && cpState && (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <div>
                  <span style={{ fontWeight: 600 }}>Step {cpState.step}</span>
                  <span style={{ color: C.muted, marginLeft: 8 }}>&mdash; {cpState.node_name}</span>
                  {cpState.source && <span style={{ color: C.muted, marginLeft: 8, fontSize: 11 }}>({cpState.source})</span>}
                </div>
                <CopyButton text={cpState} />
              </div>

              {cpState.values && Object.entries(cpState.values).map(([key, val]) => (
                <CollapsibleContent key={key} title={key} defaultOpen={key === 'notebook'}>
                  <JsonCard data={val} title={key} />
                </CollapsibleContent>
              ))}

              {!cpState.values && cpState.error && (
                <div style={{ color: C.red, padding: 12 }}>{cpState.error}</div>
              )}
            </div>
          )}
        </div>
      </div>
      <div style={{ padding: '4px 12px', borderTop: `1px solid ${C.line}`, color: C.muted, fontSize: 11 }}>
        &#8593;&#8595;: select step &nbsp; Enter: inspect &nbsp; r: replay &nbsp; e: export
      </div>
    </div>
  );
}
