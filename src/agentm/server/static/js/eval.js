// Eval Dashboard — batch evaluation monitoring page

// ── Progress Bar ─────────────────────────────────────────────────────
function EvalProgressBar({ summary }) {
  const total = summary.total || 0;
  if (total === 0) return null;
  const segments = [
    { key: 'completed', count: summary.completed || 0, color: C.green, label: 'Completed' },
    { key: 'running', count: summary.running || 0, color: C.teal, label: 'Running' },
    { key: 'failed', count: summary.failed || 0, color: C.red, label: 'Failed' },
    { key: 'skipped', count: summary.skipped || 0, color: C.yellow, label: 'Skipped' },
    { key: 'pending', count: summary.pending || 0, color: C.muted, label: 'Pending' },
  ];
  const done = (summary.completed || 0) + (summary.failed || 0) + (summary.skipped || 0);
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;

  return (
    <div style={{ padding: '12px 16px', borderBottom: `1px solid ${C.line}` }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <span style={{ fontWeight: 600, fontSize: 14 }}>Eval Progress</span>
        <span style={{ color: C.muted, fontSize: 12 }}>{done}/{total} ({pct}%)</span>
      </div>
      {/* Bar */}
      <div style={{ height: 8, borderRadius: 4, background: C.line, display: 'flex', overflow: 'hidden', marginBottom: 10 }}>
        {segments.map(seg => seg.count > 0 && (
          <div key={seg.key} style={{
            width: `${(seg.count / total) * 100}%`,
            background: seg.color,
            transition: 'width 0.3s ease',
          }} />
        ))}
      </div>
      {/* Stats */}
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        {segments.map(seg => (
          <div key={seg.key} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: seg.color, display: 'inline-block' }} />
            <span style={{ color: C.muted }}>{seg.label}:</span>
            <span style={{ color: seg.color, fontWeight: 600 }}>{seg.count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Sample Table ─────────────────────────────────────────────────────
function EvalSampleTable({ samples, total, onSelect, selectedId, statusFilter, onStatusFilter, searchText, onSearch, offset, onPageChange, pageSize }) {
  const statuses = ['all', 'pending', 'running', 'completed', 'failed', 'skipped'];
  const totalPages = Math.ceil(total / pageSize);
  const currentPage = Math.floor(offset / pageSize) + 1;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Filter bar */}
      <div style={{ padding: '8px 12px', borderBottom: `1px solid ${C.line}`, display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        {statuses.map(s => {
          const isActive = statusFilter === s || (s === 'all' && !statusFilter);
          const color = s === 'all' ? C.teal : (STATUS_COLORS[s] || C.muted);
          return (
            <button
              key={s}
              onClick={() => onStatusFilter(s === 'all' ? null : s)}
              style={{
                background: isActive ? color + '20' : 'transparent',
                border: `1px solid ${isActive ? color : C.line}`,
                color: isActive ? color : C.muted,
                padding: '2px 10px',
                borderRadius: 3,
                cursor: 'pointer',
                fontSize: 11,
                fontFamily: 'inherit',
                fontWeight: isActive ? 600 : 400,
              }}
            >{s}</button>
          );
        })}
        <div style={{ flex: 1 }} />
        <input
          type="text"
          placeholder="Search samples..."
          value={searchText}
          onChange={e => onSearch(e.target.value)}
          style={{
            background: C.bg, color: C.text, border: `1px solid ${C.line}`,
            padding: '3px 8px', borderRadius: 3, fontSize: 11, fontFamily: 'inherit',
            width: 180, outline: 'none',
          }}
        />
        <span style={{ fontSize: 11, color: C.muted }}>{total} total</span>
      </div>
      {/* Table */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {samples.length === 0 && (
          <div style={{ padding: 40, textAlign: 'center', color: C.muted, fontSize: 12 }}>
            {total === 0 ? 'No samples registered yet' : 'No matching samples'}
          </div>
        )}
        {samples.map(s => {
          const isSelected = selectedId === s.sample_id;
          const statusColor = STATUS_COLORS[s.status] || C.muted;
          return (
            <div
              key={s.sample_id}
              onClick={() => onSelect(s.sample_id)}
              style={{
                padding: '6px 12px',
                cursor: 'pointer',
                background: isSelected ? C.teal + '15' : 'transparent',
                borderLeft: isSelected ? `2px solid ${C.teal}` : '2px solid transparent',
                borderBottom: `1px solid ${C.line}22`,
                display: 'flex',
                alignItems: 'center',
                gap: 10,
              }}
            >
              <StatusDot status={s.status} />
              <span style={{ fontSize: 12, color: C.muted, minWidth: 32 }}>#{s.dataset_index}</span>
              <span style={{ fontSize: 12, color: C.text, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {s.sample_id}
              </span>
              <span style={{ fontSize: 11, color: C.muted, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {s.data_dir}
              </span>
              {s.duration_seconds != null && (
                <span style={{ fontSize: 11, color: C.green, minWidth: 50, textAlign: 'right' }}>{s.duration_seconds.toFixed(1)}s</span>
              )}
              <Tag text={s.status} color={statusColor} />
            </div>
          );
        })}
      </div>
      {/* Pagination */}
      {totalPages > 1 && (
        <div style={{ padding: '6px 12px', borderTop: `1px solid ${C.line}`, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 8 }}>
          <button
            disabled={currentPage <= 1}
            onClick={() => onPageChange(offset - pageSize)}
            style={{ background: 'none', border: `1px solid ${C.line}`, color: currentPage <= 1 ? C.muted + '50' : C.muted, cursor: currentPage <= 1 ? 'default' : 'pointer', padding: '2px 8px', borderRadius: 3, fontSize: 11, fontFamily: 'inherit' }}
          >Prev</button>
          <span style={{ fontSize: 11, color: C.muted }}>
            Page {currentPage} / {totalPages}
          </span>
          <button
            disabled={currentPage >= totalPages}
            onClick={() => onPageChange(offset + pageSize)}
            style={{ background: 'none', border: `1px solid ${C.line}`, color: currentPage >= totalPages ? C.muted + '50' : C.muted, cursor: currentPage >= totalPages ? 'default' : 'pointer', padding: '2px 8px', borderRadius: 3, fontSize: 11, fontFamily: 'inherit' }}
          >Next</button>
        </div>
      )}
    </div>
  );
}

// ── Sample Detail ────────────────────────────────────────────────────
function EvalSampleDetail({ sampleId, onClose }) {
  const [info, setInfo] = useState(null);
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const pollRef = useRef(null);
  const scrollRef = useRef(null);

  // Fetch sample info
  useEffect(() => {
    if (!sampleId) return;
    setLoading(true);
    setEvents([]);
    fetch(`/api/eval/samples/${encodeURIComponent(sampleId)}`)
      .then(r => r.json())
      .then(data => { setInfo(data); setLoading(false); })
      .catch(() => setLoading(false));
  }, [sampleId]);

  // Fetch events + poll for running samples
  useEffect(() => {
    if (!sampleId) return;

    function fetchEvents() {
      fetch(`/api/eval/samples/${encodeURIComponent(sampleId)}/events`)
        .then(r => r.json())
        .then(data => {
          setEvents(data.events || []);
          // Stop polling once completed/failed/skipped
          if (data.status && data.status !== 'running' && data.status !== 'pending') {
            if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
          }
        })
        .catch(() => {});
    }

    fetchEvents();
    // Poll every 2s for running samples
    pollRef.current = setInterval(fetchEvents, 2000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [sampleId]);

  // Auto-scroll events
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [events.length]);

  if (loading) {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: C.muted }}>
        Loading...
      </div>
    );
  }

  if (!info) {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: C.muted }}>
        Sample not found
      </div>
    );
  }

  const statusColor = STATUS_COLORS[info.status] || C.muted;

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ padding: '10px 16px', borderBottom: `1px solid ${C.line}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <StatusDot status={info.status} />
          <span style={{ fontWeight: 600, fontSize: 14 }}>{info.sample_id}</span>
          <Tag text={info.status} color={statusColor} />
          {info.duration_seconds != null && (
            <span style={{ fontSize: 11, color: C.green }}>{info.duration_seconds.toFixed(1)}s</span>
          )}
        </div>
        <button onClick={onClose} style={{ background: 'none', border: 'none', color: C.muted, cursor: 'pointer', fontSize: 14, fontFamily: 'inherit' }}>[ESC]</button>
      </div>
      {/* Meta */}
      <div style={{ padding: '8px 16px', borderBottom: `1px solid ${C.line}22`, display: 'flex', gap: 16, fontSize: 11, color: C.muted, flexWrap: 'wrap' }}>
        <span>Index: <span style={{ color: C.text }}>{info.dataset_index}</span></span>
        <span>Dir: <span style={{ color: C.text }}>{info.data_dir}</span></span>
        {info.run_id && <span>Run: <span style={{ color: C.text }}>{info.run_id}</span></span>}
        {info.error && <span>Error: <span style={{ color: C.red }}>{info.error}</span></span>}
      </div>
      {/* Events */}
      <div style={{ padding: '6px 12px', borderBottom: `1px solid ${C.line}`, fontSize: 12, color: C.muted }}>
        Events: {events.filter(e => e.event_type !== 'llm_start').length}{info.status === 'running' && <span style={{ color: C.teal, marginLeft: 8, animation: 'pulse 1.5s infinite' }}>polling...</span>}
      </div>
      <div ref={scrollRef} style={{ flex: 1, overflowY: 'auto' }}>
        {events.length === 0 && (
          <div style={{ padding: 40, textAlign: 'center', color: C.muted, fontSize: 12 }}>
            {info.status === 'pending' ? 'Waiting to start...' : info.status === 'skipped' ? 'Sample was skipped' : 'No events recorded'}
          </div>
        )}
        {events.map((ev, i) => {
          const eventType = ev.event_type || '';
          const agentPath = ev.agent_path || [];
          const agentId = agentPath.slice(-1)[0] || '';
          const data = ev.data || {};
          const ts = ev.timestamp || '';
          const shortTs = ts.length > 19 ? ts.substring(11, 19) : ts;

          let icon = '\u25B8';
          let color = C.teal;
          let detail = eventType;
          let hidden = false;

          switch (eventType) {
            case 'llm_start': hidden = true; break;
            case 'tool_call': icon = '\u2192'; color = C.yellow; detail = data.tool_name || 'tool'; break;
            case 'tool_result': icon = '\u2190'; color = C.green; detail = `${data.tool_name || 'tool'}: ${(typeof data.result === 'string' ? data.result : '').slice(0, 80)}`; break;
            case 'llm_end': icon = '\u25C6'; color = C.purple; detail = (data.content || '').slice(0, 80) || `${(data.tool_calls || []).length} tool call(s)`; break;
            case 'task_dispatch': icon = '\u25B6'; color = C.teal; detail = `dispatch: ${data.task_type || 'task'}`; break;
            case 'task_complete': icon = '\u2713'; color = C.green; detail = 'completed'; break;
            case 'task_fail': icon = '\u2717'; color = C.red; detail = data.error || 'failed'; break;
          }

          if (hidden) return null;

          return (
            <CollapsibleContent
              key={i}
              title={
                React.createElement('span', { style: { display: 'flex', gap: 8, alignItems: 'center', fontSize: 12 } },
                  React.createElement('span', { style: { color: C.muted, fontSize: 11, minWidth: 50 } }, shortTs),
                  agentId && React.createElement('span', { style: { color: C.teal, fontSize: 11 } }, agentId),
                  React.createElement('span', { style: { color } }, icon),
                  React.createElement('span', { style: { color: C.text, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } }, detail)
                )
              }
            >
              <JsonCard data={ev.data} title={eventType} />
            </CollapsibleContent>
          );
        })}
      </div>
    </div>
  );
}

// ── Eval Page (main) ─────────────────────────────────────────────────
function EvalPage() {
  const [summary, setSummary] = useState({ total: 0, pending: 0, running: 0, completed: 0, failed: 0, skipped: 0 });
  const [samples, setSamples] = useState([]);
  const [total, setTotal] = useState(0);
  const [selectedId, setSelectedId] = useState(null);
  const [statusFilter, setStatusFilter] = useState(null);
  const [searchText, setSearchText] = useState('');
  const [offset, setOffset] = useState(0);
  const pageSize = 50;

  // Initial fetch
  useEffect(() => {
    fetch('/api/eval/status').then(r => r.json()).then(data => {
      if (data.enabled) {
        const { enabled, ...rest } = data;
        setSummary(rest);
      }
    }).catch(() => {});
  }, []);

  // Fetch samples with filters
  useEffect(() => {
    const params = new URLSearchParams({ offset: String(offset), limit: String(pageSize) });
    if (statusFilter) params.set('status', statusFilter);
    if (searchText) params.set('search', searchText);
    fetch(`/api/eval/samples?${params}`).then(r => r.json()).then(data => {
      setSamples(data.samples || []);
      setTotal(data.total || 0);
    }).catch(() => {});
  }, [offset, statusFilter, searchText]);

  // Reset offset when filters change
  useEffect(() => { setOffset(0); }, [statusFilter, searchText]);

  // Listen for WebSocket eval events (attached via window)
  useEffect(() => {
    function handleEvalEvent(e) {
      const event = e.detail;
      if (!event || event.channel !== 'eval') return;

      if (event.event_type === 'eval_snapshot') {
        const data = event.data || {};
        if (data.summary) setSummary(data.summary);
        if (data.samples) { setSamples(data.samples); setTotal(data.total || 0); }
        return;
      }

      if (event.event_type === 'sample_status') {
        const s = event.data || {};
        // Update summary
        fetch('/api/eval/status').then(r => r.json()).then(data => {
          if (data.enabled) {
            const { enabled, ...rest } = data;
            setSummary(rest);
          }
        }).catch(() => {});
        // Update sample in list
        setSamples(prev => {
          const idx = prev.findIndex(x => x.sample_id === s.sample_id);
          if (idx >= 0) {
            const updated = [...prev];
            updated[idx] = s;
            return updated;
          }
          // New sample — add to list if it passes current filter
          if (!statusFilter || s.status === statusFilter) {
            return [...prev, s];
          }
          return prev;
        });
      }
    }

    window.addEventListener('eval_event', handleEvalEvent);
    return () => window.removeEventListener('eval_event', handleEvalEvent);
  }, [statusFilter]);

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <EvalProgressBar summary={summary} />
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Sample list */}
        <div style={{ width: selectedId ? '40%' : '100%', minWidth: 300, borderRight: selectedId ? `1px solid ${C.line}` : 'none', display: 'flex', flexDirection: 'column', transition: 'width 0.2s' }}>
          <EvalSampleTable
            samples={samples}
            total={total}
            onSelect={setSelectedId}
            selectedId={selectedId}
            statusFilter={statusFilter}
            onStatusFilter={setStatusFilter}
            searchText={searchText}
            onSearch={setSearchText}
            offset={offset}
            onPageChange={setOffset}
            pageSize={pageSize}
          />
        </div>
        {/* Detail panel */}
        {selectedId && (
          <EvalSampleDetail
            sampleId={selectedId}
            onClose={() => setSelectedId(null)}
          />
        )}
      </div>
    </div>
  );
}
