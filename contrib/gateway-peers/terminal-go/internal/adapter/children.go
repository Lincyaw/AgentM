package adapter

import (
	"context"
	"log"
	"sync"

	tea "charm.land/bubbletea/v2"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/app"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/runtime"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/session"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/messages"
	"github.com/AoyangSpace/agentm-terminal/internal/wire"
)

// childSession is the per-child state the ChildManager owns: a dedicated
// *app.App (its own event fan-out drives a dedicated supervisor tab) and a
// Translator that paints that app's transcript from child-stamped wire bodies.
type childSession struct {
	id         string
	app        *app.App
	sess       *session.Session
	translator *Translator
	finished   bool
}

// ChildManager routes child-session wire bodies (those carrying
// metadata.child_id) into per-child sub-sessions so spawned sub-agents render
// in their own switchable cagent tab, instead of bleeding into the parent
// transcript.
//
// The bridge to the vendored TUI is the supervisor's SessionSpawner seam: the
// TUI owns the supervisor and only ever calls the spawner from its
// SpawnSessionMsg handler. So when a brand-new child_id appears, the manager
//
//  1. builds the child *app.App + session up front (keyed by child_id),
//  2. parks it in spawnQueue, and
//  3. sends a SpawnSessionMsg to the tea.Program.
//
// The TUI then calls supervisor.SpawnSession -> our Spawner, which pops the
// parked child app. The supervisor registers it as a background tab (routing
// key == child session ID == child_id) while keeping the parent conversation
// active. Every subsequent body for that child_id is fed into the child's
// Translator, whose EmitEvent fans out as RoutedMsg{SessionID: child_id} that
// the supervisor paints onto that tab.
//
// All public methods are concurrency-safe: Route/markStart/markEnd run on the
// wire-pump goroutine while Spawn runs on the bubbletea goroutine.
type ChildManager struct {
	workingDir string
	// client + baseID let each child tab drive its OWN wire-backed Controller,
	// addressed by the child's session id, so the human can chat with a live
	// sub-agent (interactive-subagent design). Without them a child tab would be
	// observe-only.
	client *wire.WireClient
	baseID Identity

	mu       sync.Mutex
	program  *tea.Program
	children map[string]*childSession
	// spawnQueue holds child apps awaiting a supervisor.SpawnSession pull, in
	// FIFO order keyed by child_id. A queued entry is removed once Spawn pops it.
	spawnQueue []*childSession
}

// NewChildManager builds a ChildManager. workingDir labels spawned child tabs'
// fallback title (the supervisor uses it when a session has no title) and is the
// non-empty WorkingDir the SpawnSessionMsg carries so the TUI does not open the
// working-dir picker. client + baseID are the wire handle and platform identity
// each child tab's Controller reuses (re-keyed to the child's session id) so a
// typed message routes to that live sub-agent.
func NewChildManager(workingDir string, client *wire.WireClient, baseID Identity) *ChildManager {
	return &ChildManager{
		workingDir: workingDir,
		client:     client,
		baseID:     baseID,
		children:   make(map[string]*childSession),
	}
}

// SetProgram wires the tea.Program the manager sends SpawnSessionMsg to. Called
// from main once tui.New has produced the program. Safe to call before any
// child appears.
func (m *ChildManager) SetProgram(p *tea.Program) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.program = p
}

// TryPop atomically pops the next queued child app, or returns nil if the
// queue is empty. Used by the adapter's multi-session spawner to distinguish
// sub-agent child spawns (queued via Start) from user-initiated new tabs.
func (m *ChildManager) TryPop() *childSession {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.spawnQueue) == 0 {
		return nil
	}
	cs := m.spawnQueue[0]
	m.spawnQueue = m.spawnQueue[1:]
	return cs
}

// Has reports whether child_id is a known child session. Used by the root
// translator to decide whether a body must be routed to a child tab.
func (m *ChildManager) Has(childID string) bool {
	if childID == "" {
		return false
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	_, ok := m.children[childID]
	return ok
}

// Start handles a child_start body: it creates the child sub-session (app +
// session keyed by child_id + translator), parks it for the spawner, and asks
// the TUI to open a tab for it. Idempotent on child_id.
func (m *ChildManager) Start(childID, purpose string) {
	if childID == "" {
		return
	}
	m.mu.Lock()
	if _, exists := m.children[childID]; exists {
		m.mu.Unlock()
		return
	}

	title := purpose
	if title == "" {
		title = "sub-agent"
	}
	// Key the child session ID to child_id so the supervisor routing key, the
	// tab identity and the wire child_id are one and the same.
	sess := session.New(
		session.WithID(childID),
		session.WithTitle(title),
		session.WithWorkingDir(m.workingDir),
	)
	// Build translator -> child Controller -> App in the same order the root
	// adapter uses (App needs the Controller; the Controller needs the
	// translator; the translator needs the App). The child Controller is keyed
	// to childID so a typed message in this tab is delivered to the live
	// sub-agent's inbox by the gateway, not the main conversation.
	tr := NewTranslator(nil, sess)
	var childApp *app.App
	if m.client != nil {
		childIdentity := m.baseID
		childIdentity.SessionKey = childID
		childIdentity.Scenario = ""
		ctrl := NewChildController(m.client, childIdentity, tr)
		childApp = app.New(context.Background(), sess, app.WithController(ctrl))
	} else {
		// No wire handle (mock/launch paths): observe-only child tab.
		childApp = app.New(context.Background(), sess)
	}
	tr.app = childApp
	cs := &childSession{
		id:         childID,
		app:        childApp,
		sess:       sess,
		translator: tr,
	}
	m.children[childID] = cs
	m.spawnQueue = append(m.spawnQueue, cs)
	p := m.program
	m.mu.Unlock()

	if p == nil {
		// No program yet (should not happen once SetProgram ran before the
		// stream); the child app is still queued so a later spawn picks it up.
		log.Printf("[adapter] child_start %s parked before program ready", childID)
		return
	}
	// Drive the TUI's normal new-tab path, which pulls our queued child app via
	// Spawner. Child workflow tabs open in the background so the user does not
	// lose the parent conversation that initiated the workflow.
	p.Send(messages.SpawnSessionMsg{WorkingDir: m.workingDir, Background: true})
}

// Route feeds one child-stamped body into the child's translator so it paints
// the child tab. Reports false when child_id is unknown (the caller then leaves
// the body on the parent path). A child_start has not necessarily produced a
// registered child yet when its trajectory races ahead; Start always precedes
// Route for a given child because both arrive on the ordered parent wire.
func (m *ChildManager) Route(childID string, body map[string]any) bool {
	m.mu.Lock()
	cs := m.children[childID]
	m.mu.Unlock()
	if cs == nil {
		return false
	}
	cs.translator.handleOutbound(body)
	return true
}

// End handles a child_end body: it marks the child finished and emits a
// terminal breadcrumb onto the child's own tab (stream stop + a completion
// note) so the tab reads as done while staying open for inspection.
func (m *ChildManager) End(childID, errStr string, finalMsgCount int64) {
	m.mu.Lock()
	cs := m.children[childID]
	if cs != nil {
		cs.finished = true
	}
	m.mu.Unlock()
	if cs == nil {
		return
	}

	tr := cs.translator
	// Close any open stream on the child so its spinner stops.
	if tr.streaming {
		tr.streaming = false
		tr.emit(runtime.StreamStopped(tr.sessionID(), tr.agentName, "child_end"))
	}
	if errStr != "" {
		tr.emit(runtime.Error("sub-agent failed: " + errStr))
		return
	}
	note := "✓ sub-agent completed"
	if finalMsgCount > 0 {
		note += " (" + itoa(finalMsgCount) + " messages)"
	}
	tr.emit(runtime.AgentChoice(tr.agentName, tr.sessionID(), note))
}

// itoa renders a non-negative int64 without pulling strconv into the hot path's
// import set elsewhere; kept local to children.go.
func itoa(n int64) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
