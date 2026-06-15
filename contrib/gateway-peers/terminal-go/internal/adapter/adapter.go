package adapter

import (
	"context"
	"errors"
	"log"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/app"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/session"
	"github.com/AoyangSpace/agentm-terminal/internal/tui"
	"github.com/AoyangSpace/agentm-terminal/internal/wire"
)

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

	// Build the App first so the Translator and Controller can reference it,
	// then attach the Controller via an option.
	tr := NewTranslator(nil, sess)
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

// Spawner returns the wire peer's SessionSpawner (see ErrorSpawner).
func (ad *Adapter) Spawner() tui.SessionSpawner {
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
