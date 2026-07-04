package adapter

import (
	"context"
	"errors"
	"log"
	"os"
	"sync"

	tea "charm.land/bubbletea/v2"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/app"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/runtime"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/session"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/skills"
	"github.com/AoyangSpace/agentm-terminal/internal/tui"
	"github.com/AoyangSpace/agentm-terminal/internal/wire"
)

// ErrNoQueuedChild is returned by the child manager's spawner when the TUI asks
// to spawn a session but no sub-agent child app is queued — i.e. a genuine
// user-driven "new tab", which a single-conversation wire peer does not support.
var ErrNoQueuedChild = errors.New("ag: no sub-agent session to open; multi-session spawning is not supported over the wire")

// tabSession holds the per-tab state for an independently-spawned gateway
// session (user-driven new tab via Ctrl+T / "+"). Each tab talks to a
// distinct session_key on the same wire connection.
type tabSession struct {
	sessionKey string
	app        *app.App
	sess       *session.Session
	translator *Translator
	controller *Controller
	children   *ChildManager
}

// Adapter binds a WireClient to a cagent *app.App: it constructs the App with a
// wire-backed Controller, owns the Translator, and pumps the client's outbound
// envelope channel into the Translator. The result is a fully-wired App the TUI
// can drive identically to cagent's local-runtime App.
type Adapter struct {
	App        *app.App
	Session    *session.Session
	translator *Translator
	controller *Controller
	client     *wire.WireClient
	children   *ChildManager

	// Multi-session support: session_key → tab for independently-spawned tabs.
	// The root session is NOT in this map (it's the fallback in routeOutbound).
	mu      sync.Mutex
	rootKey string                 // session_key of the initial/root session
	baseID  Identity               // template for new tab identities
	tabs    map[string]*tabSession // session_key → spawned tab
	program *tea.Program           // cached for new tab child managers
}

// New builds an Adapter for a connected WireClient.
//
//   - id carries the platform identity stamped on every inbound.
//   - firstMessage is an optional prompt auto-sent on launch (empty == none).
//   - appOpts are extra app.Opt values (e.g. app.WithReadOnly()).
//
// The returned Adapter's App is ready to pass to tui.New; call Start to begin
// pumping wire events into it.
func New(client *wire.WireClient, id Identity, firstMessage string, appOpts ...app.Opt) *Adapter {
	sess := session.New(session.WithTitle("agentm"))

	// The child manager routes spawned sub-agent trajectories (bodies stamped
	// with metadata.child_id) into their own switchable cagent tabs. Its
	// working-dir label seeds child tab titles and the SpawnSessionMsg payload.
	wd, _ := os.Getwd()
	children := NewChildManager(wd, client, id)

	// Build the App first so the Translator and Controller can reference it,
	// then attach the Controller via an option. The root translator delegates
	// child-stamped bodies to the child manager.
	tr := NewTranslator(nil, sess)
	tr.children = children
	ctrl := NewController(client, id, tr, firstMessage)

	opts := append([]app.Opt{app.WithController(ctrl)}, appOpts...)
	a := app.New(context.Background(), sess, opts...)
	tr.app = a

	// Seed the App's capability view from the welcome handshake so the model
	// picker and command palette work before the first message creates a
	// session (the session_ready frame later augments this with the scenario's
	// tools and in-session commands).
	seedFromCapabilities(a, client.Capabilities())

	// Register the OnReconnect callback so the App's capability view is
	// refreshed after a reconnection.
	client.OnReconnect = func() {
		seedFromCapabilities(a, client.Capabilities())
	}

	return &Adapter{
		App:        a,
		Session:    sess,
		translator: tr,
		controller: ctrl,
		client:     client,
		children:   children,
		rootKey:    id.SessionKey,
		baseID:     id,
		tabs:       make(map[string]*tabSession),
	}
}

// seedFromCapabilities projects the welcome-handshake capability block onto the
// App so the model picker and command palette are populated before any session
// exists. Tolerant of a nil/partial block: missing fields just leave the
// corresponding view empty (as before this frame existed).
func seedFromCapabilities(a *app.App, caps map[string]any) {
	if a == nil || caps == nil {
		return
	}
	models := stringSlice(caps["models"])
	model, _ := caps["model"].(string)
	commandNames := capabilityCommandNames(caps["commands"])
	// No tool names pre-session; session_ready supplies those later. Setting
	// empty slices on a fresh App is harmless (nothing to overwrite yet).
	a.SetAgentInfo(nil, commandNames, models, model)
	a.SetSkills(capabilitySkills(caps["skills"]))
}

// capabilitySkills extracts the skill catalog from the welcome capability block.
// Each entry is a {name, summary} map; the wire protocol carries no skill body,
// so only Name/Description are populated.
func capabilitySkills(v any) []skills.Skill {
	entries, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]skills.Skill, 0, len(entries))
	for _, e := range entries {
		m, ok := e.(map[string]any)
		if !ok {
			continue
		}
		name, _ := m["name"].(string)
		if name == "" {
			continue
		}
		summary, _ := m["summary"].(string)
		out = append(out, skills.Skill{Name: name, Description: summary})
	}
	return out
}

// capabilityCommandNames extracts the bare command names from the welcome
// capability block's command catalog (each entry is a {name, kind, summary}
// map). Non-conforming entries are skipped.
func capabilityCommandNames(v any) []string {
	entries, ok := v.([]any)
	if !ok {
		return nil
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		m, ok := e.(map[string]any)
		if !ok {
			continue
		}
		if name, ok := m["name"].(string); ok && name != "" {
			names = append(names, name)
		}
	}
	return names
}

// SetProgram hands the tea.Program to the child manager so it can drive the
// TUI's new-tab path when a sub-agent starts. main calls this after tui.New.
func (ad *Adapter) SetProgram(p *tea.Program) {
	ad.mu.Lock()
	ad.program = p
	ad.mu.Unlock()
	if ad.children != nil {
		ad.children.SetProgram(p)
	}
}

// Start launches the goroutine that drains the wire client's outbound channel
// into the Translator until ctx is cancelled or the client disconnects.
func (ad *Adapter) Start(ctx context.Context) {
	go ad.pump(ctx)
}

// ErrorSpawner is a non-nil SessionSpawner that always fails. Used by the mock
// launch path where no wire client exists and multi-session spawning is not
// meaningful. A non-nil spawner is mandatory because the TUI's tab-restore
// path calls it unconditionally for every persisted tab beyond the first (a
// nil spawner panics there); the error makes the TUI log and skip the tab.
func ErrorSpawner() tui.SessionSpawner {
	return func(ctx context.Context, workingDir string) (*app.App, *session.Session, func(), error) {
		_, _ = ctx, workingDir
		return nil, nil, nil, errors.New("ag: multi-session spawning is not supported without a gateway connection")
	}
}

// Spawner returns the wire peer's SessionSpawner. It supports two kinds of
// spawn: sub-agent child tabs (queued by ChildManager.Start) and user-initiated
// independent sessions (Ctrl+T / "+" button). The spawner tries child queues
// first; if none has a pending child, it creates a new independent session on
// the same wire connection with a fresh session_key.
func (ad *Adapter) Spawner() tui.SessionSpawner {
	return func(ctx context.Context, workingDir string) (*app.App, *session.Session, func(), error) {
		// 1. Check if any child manager (root or spawned tab) has a queued
		//    sub-agent child — these take priority over independent sessions
		//    because the SpawnSessionMsg was driven by a child_start frame.
		if cs := ad.tryPopChild(); cs != nil {
			return cs.app, cs.sess, func() {}, nil
		}
		// 2. User-initiated new tab: create an independent gateway session.
		return ad.createTab(ctx)
	}
}

// tryPopChild checks the root child manager and all spawned-tab child managers
// for a queued sub-agent child app. Returns nil if none is pending.
func (ad *Adapter) tryPopChild() *childSession {
	if ad.children != nil {
		if cs := ad.children.TryPop(); cs != nil {
			return cs
		}
	}
	ad.mu.Lock()
	defer ad.mu.Unlock()
	for _, tab := range ad.tabs {
		if tab.children != nil {
			if cs := tab.children.TryPop(); cs != nil {
				return cs
			}
		}
	}
	return nil
}

// createTab builds a new independent session on the same wire connection. The
// new session gets a unique session_key so the gateway creates a fresh
// AgentSession; outbound frames with that key are routed to this tab's
// translator by the pump.
func (ad *Adapter) createTab(_ context.Context) (*app.App, *session.Session, func(), error) {
	newKey := ad.rootKey + ":" + wire.NewID()

	wd, _ := os.Getwd()
	sess := session.New(session.WithTitle("agentm"))

	tabID := newKey // identity for the child manager
	childID := ad.baseID
	childID.SessionKey = newKey
	childID.Scenario = ad.baseID.Scenario

	children := NewChildManager(wd, ad.client, childID)

	tr := NewTranslator(nil, sess)
	tr.children = children
	ctrl := NewController(ad.client, childID, tr, "")

	a := app.New(context.Background(), sess, app.WithController(ctrl))
	tr.app = a

	seedFromCapabilities(a, ad.client.Capabilities())

	tab := &tabSession{
		sessionKey: newKey,
		app:        a,
		sess:       sess,
		translator: tr,
		controller: ctrl,
		children:   children,
	}

	ad.mu.Lock()
	ad.tabs[newKey] = tab
	p := ad.program
	ad.mu.Unlock()

	if p != nil {
		children.SetProgram(p)
	}

	cleanup := func() {
		ad.mu.Lock()
		delete(ad.tabs, tabID)
		ad.mu.Unlock()
	}

	return a, sess, cleanup, nil
}

// routeOutbound dispatches an outbound envelope to the translator that owns its
// session_key. Spawned tabs are checked first; the root translator is the
// fallback for the root session_key or any envelope with an empty/unknown key.
func (ad *Adapter) routeOutbound(env *wire.Envelope) {
	if env.SessionKey != "" {
		ad.mu.Lock()
		tab, ok := ad.tabs[env.SessionKey]
		ad.mu.Unlock()
		if ok {
			tab.translator.HandleEnvelope(env)
			return
		}
	}
	ad.translator.HandleEnvelope(env)
}

func (ad *Adapter) pump(ctx context.Context) {
	for {
		outbound := ad.client.Outbound()
		select {
		case <-ctx.Done():
			return
		case <-ad.client.Done():
			if err := ad.client.Err(); err != nil {
				log.Printf("[adapter] wire client closed: %v", err)
				ad.translator.emit(runtime.Error("gateway connection closed: " + err.Error()))
			}
			return
		case <-ad.client.Reconnecting():
			// The wire client is reconnecting. Wait for either a new outbound
			// channel (reconnect succeeded) or permanent closure.
			log.Printf("[adapter] wire client reconnecting, waiting for recovery...")
			ad.translator.emit(runtime.Warning("gateway disconnected; reconnecting…", ""))
			reconnected := ad.client.Reconnected()
			select {
			case <-ad.client.Done():
				if err := ad.client.Err(); err != nil {
					log.Printf("[adapter] wire client reconnect failed: %v", err)
					ad.translator.emit(runtime.Error("gateway reconnect failed: " + err.Error()))
				}
				return
			case <-ctx.Done():
				return
			case <-reconnected:
				ad.translator.emit(runtime.Warning("gateway reconnected", ""))
				// New outbound channel is available via Outbound() — loop back
				// and re-select with the fresh channel.
				continue
			}
		case env, ok := <-outbound:
			if !ok {
				select {
				case <-ad.client.Done():
					return
				default:
					continue
				}
			}
			ad.routeOutbound(env)
		}
	}
}
