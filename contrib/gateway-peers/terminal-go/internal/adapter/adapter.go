package adapter

import (
	"context"
	"errors"
	"log"
	"os"

	tea "charm.land/bubbletea/v2"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/app"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/session"
	"github.com/AoyangSpace/agentm-terminal/internal/tui"
	"github.com/AoyangSpace/agentm-terminal/internal/wire"
)

// ErrNoQueuedChild is returned by the child manager's spawner when the TUI asks
// to spawn a session but no sub-agent child app is queued — i.e. a genuine
// user-driven "new tab", which a single-conversation wire peer does not support.
var ErrNoQueuedChild = errors.New("agentm-terminal: no sub-agent session to open; multi-session spawning is not supported over the wire")

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
	children := NewChildManager(wd)

	// Build the App first so the Translator and Controller can reference it,
	// then attach the Controller via an option. The root translator delegates
	// child-stamped bodies to the child manager.
	tr := NewTranslator(nil, sess)
	tr.children = children
	ctrl := NewController(client, id, tr, firstMessage)

	opts := append([]app.Opt{app.WithController(ctrl)}, appOpts...)
	a := app.New(context.Background(), sess, opts...)
	tr.app = a

	return &Adapter{
		App:        a,
		Session:    sess,
		translator: tr,
		controller: ctrl,
		client:     client,
		children:   children,
	}
}

// SetProgram hands the tea.Program to the child manager so it can drive the
// TUI's new-tab path when a sub-agent starts. main calls this after tui.New.
func (ad *Adapter) SetProgram(p *tea.Program) {
	if ad.children != nil {
		ad.children.SetProgram(p)
	}
}

// Start launches the goroutine that drains the wire client's outbound channel
// into the Translator until ctx is cancelled or the client disconnects.
func (ad *Adapter) Start(ctx context.Context) {
	go ad.pump(ctx)
}

// ErrorSpawner is a non-nil SessionSpawner that always fails. The wire peer
// maps one TUI to one gateway conversation, so spawning a *second* independent
// session is not supported. A non-nil spawner is mandatory because the TUI's
// tab-restore path calls it unconditionally for every persisted tab beyond the
// first (a nil spawner panics there); the error makes the TUI log and skip the
// tab. Used by both the live and mock launch paths.
func ErrorSpawner() tui.SessionSpawner {
	return func(ctx context.Context, workingDir string) (*app.App, *session.Session, func(), error) {
		_, _ = ctx, workingDir
		return nil, nil, nil, errors.New("agentm-terminal: multi-session spawning is not supported over the wire")
	}
}

// Spawner returns the wire peer's SessionSpawner. It is backed by the child
// manager: the supervisor calls it from its SpawnSessionMsg handler, and the
// manager returns whichever sub-agent child app it queued for that spawn (or
// ErrNoQueuedChild for a user-driven new tab).
func (ad *Adapter) Spawner() tui.SessionSpawner {
	if ad.children != nil {
		return ad.children.Spawner()
	}
	return ErrorSpawner()
}

func (ad *Adapter) pump(ctx context.Context) {
	outbound := ad.client.Outbound()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ad.client.Done():
			if err := ad.client.Err(); err != nil {
				log.Printf("[adapter] wire client closed: %v", err)
			}
			return
		case env, ok := <-outbound:
			if !ok {
				return
			}
			ad.translator.HandleEnvelope(env)
		}
	}
}
