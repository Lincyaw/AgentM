const NAV_ITEMS = [
  { id: 'eval', icon: '\u2261', label: 'Eval' },
  { id: 'debug', icon: '\u2699', label: 'Debug' },
];

function App() {
  const [page, setPage] = useState('eval');
  const [threadId, setThreadId] = useState(null);

  // Fetch thread_id on mount
  useEffect(() => {
    fetch('/api/topology').then(r => r.json()).then(data => {
      if (data.thread_id) setThreadId(data.thread_id);
    }).catch(() => { });
  }, []);

  // WebSocket event handler
  const handleEvent = useCallback((event) => {
    // Eval channel events — forward to EvalPage
    if (event.channel === 'eval') {
      window.dispatchEvent(new CustomEvent('eval_event', { detail: event }));
      return;
    }
    const data = event.data || {};
    if ((data.task_id || data.notebook?.task_id) && !threadId) {
      setThreadId(data.task_id || data.notebook?.task_id);
    }
  }, [threadId]);

  const wsStatus = useWebSocket(handleEvent);

  // Global keyboard shortcuts
  useEffect(() => {
    const handler = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === '1') setPage('eval');
      if (e.key === '2') setPage('debug');
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
          {page === 'eval' && <EvalPage />}
          {page === 'debug' && <DebugPage threadId={threadId} />}
        </div>
      </div>
    </>
  );
}

// Mount
ReactDOM.createRoot(document.getElementById('root')).render(<App />);
