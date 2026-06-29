// Package app defines the controller seam the TUI drives.
//
// In cagent this package wires a runtime.Runtime, a session.Session and a
// title generator into the App the TUI talks to. In the agentm-terminal peer
// the runtime is the gateway wire protocol; the real implementation of these
// methods is supplied later by internal/adapter. This file exists so the
// cagent-derived TUI compiles against a stable App surface: the exported
// methods are the subset the gateway-backed TUI still needs, while the bodies
// are stubs (zero value / not-implemented error) until the adapter takes over.
//
// Only the symbols the TUI references are exposed:
//
//	App, New, WithReadOnly,
//	(*App).Run, CompactSession, ResolveInput, RunBangCommand, RunSkillFork,
//	UpdateSessionTitle, CurrentAgentSkills,
//	CurrentAgentCommands, IsReadOnly, ShouldExitAfterFirstResponse,
//	SkillCommandFork, Session,
//	ErrTitleGenerating, ErrNothingToUndo.
package app

import (
	"context"
	"errors"
	"slices"
	"sync"

	tea "charm.land/bubbletea/v2"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/config/types"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/runtime"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/session"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/skills"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/tools"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/messages"
)

// Controller is the behavioural seam between the vendored cagent App surface
// (which the TUI drives, see pkg/tui) and the live backend. In cagent the
// backend is a local runtime.Runtime; in the agentm-terminal peer it is the
// gateway wire protocol, implemented by internal/adapter.
//
// Only the methods that actually need to reach the backend are listed here;
// everything else on App stays a local no-op/zero stub (model switching,
// snapshots, skills/commands enumeration — surfaces the wire protocol does not
// expose). A nil Controller keeps App behaving as the phase-1 stub.
//
// The Controller never sends events back directly: it pushes runtime.Event
// values into the App via App.EmitEvent, which fans them out to every
// SubscribeWith consumer (the TUI supervisor).
type Controller interface {
	// Run sends one user turn to the backend.
	Run(ctx context.Context, cancel context.CancelFunc, message string, attachments []messages.Attachment)
	// RunCooperative sends one user turn without interrupting an active run.
	RunCooperative(ctx context.Context, cancel context.CancelFunc, message string, attachments []messages.Attachment)
	// RunWithMessage sends a pre-built message (e.g. with attachments).
	RunWithMessage(ctx context.Context, cancel context.CancelFunc, msg *session.Message)
	// CompactSession asks the backend to summarise + compact history.
	CompactSession(ctx context.Context, cancel context.CancelFunc, additionalPrompt string)
	// Resume delivers a tool-confirmation decision to a waiting tool call.
	Resume(req runtime.ResumeRequest)
	// UpdateSessionTitle sets and persists the session title.
	UpdateSessionTitle(ctx context.Context, title string) error
	// RunBangCommand runs a shell command and surfaces its output.
	RunBangCommand(ctx context.Context, command string)
	// NewSession starts a fresh session on the backend.
	NewSession()
	// FirstMessage returns the queued first message, if any, to send on launch.
	FirstMessage() (content string, ok bool)
	// SwitchModel asks the backend to switch the active model profile by name.
	SwitchModel(name string)
}

// UndoSnapshotResult reports the outcome of an undo/reset operation.
type UndoSnapshotResult struct {
	RestoredFiles int
}

// ErrTitleGenerating is returned when attempting to set a title while
// generation is in progress.
var ErrTitleGenerating = errors.New("title generation in progress, please wait")

// ErrNothingToUndo is returned when an undo/reset is requested but no snapshot
// is available to restore.
var ErrNothingToUndo = errors.New("nothing to undo")

// App is the controller the TUI drives. The adapter supplies the live
// implementation via a Controller; without one the methods are stubs.
type App struct {
	session                *session.Session
	exitAfterFirstResponse bool
	readOnly               bool

	controller Controller

	// agentInfoMu guards the session_ready-sourced agent capability view
	// (tool/command names, model list, active model). The translator writes it
	// from the wire-reader goroutine; the TUI reads it from the bubbletea
	// goroutine when opening /tools, /skills (commands) and the model picker.
	agentInfoMu  sync.Mutex
	toolNames    []string
	commandNames []string
	modelNames   []string
	activeModel  string
	skills       []skills.Skill

	// events is the raw event stream the backend pushes into via EmitEvent;
	// startFanOut drains it and scatters every event to each SubscribeWith
	// consumer. Mirrors cagent pkg/app's fan-out so the vendored TUI's
	// supervisor sees the same subscription semantics.
	events     chan runtime.Event
	subsMu     sync.Mutex
	subs       []chan tea.Msg
	fanoutOnce sync.Once
}

const (
	eventsBufferSize     = 256
	subscriberBufferSize = 1024
)

// Opt is an option for creating a new App.
type Opt func(*App)

// WithReadOnly marks the session as read-only: the conversation history is
// displayed but no new messages can be sent to the LLM.
func WithReadOnly() Opt {
	return func(a *App) {
		a.readOnly = true
	}
}

// WithController wires the live backend the App delegates to. The adapter
// passes its wire-backed Controller here.
func WithController(c Controller) Opt {
	return func(a *App) {
		a.controller = c
	}
}

// WithExitAfterFirstResponse makes ShouldExitAfterFirstResponse report true.
func WithExitAfterFirstResponse() Opt {
	return func(a *App) {
		a.exitAfterFirstResponse = true
	}
}

// New creates a new App for the given session.
//
// The cagent original also threads a runtime.Runtime and a title generator;
// those belong to the adapter, which supplies a Controller via WithController.
func New(ctx context.Context, sess *session.Session, opts ...Opt) *App {
	_ = ctx
	app := &App{
		session: sess,
		events:  make(chan runtime.Event, eventsBufferSize),
	}
	for _, opt := range opts {
		opt(app)
	}
	return app
}

// EmitEvent pushes a runtime event into the App's stream so it fans out to
// every SubscribeWith consumer. Called by the adapter's translator. Non-blocking
// on a full buffer (the event is dropped) so a slow TUI cannot stall the wire
// reader.
func (a *App) EmitEvent(ev runtime.Event) {
	if ev == nil {
		return
	}
	select {
	case a.events <- ev:
	default:
	}
}

// Run one agent loop. Delegates to the wire-backed controller.
func (a *App) Run(ctx context.Context, cancel context.CancelFunc, message string, attachments []messages.Attachment) {
	if a.controller != nil {
		a.controller.Run(ctx, cancel, message, attachments)
	}
}

// RunCooperative sends one user turn without interrupting an active backend run.
func (a *App) RunCooperative(ctx context.Context, cancel context.CancelFunc, message string, attachments []messages.Attachment) {
	if a.controller != nil {
		a.controller.RunCooperative(ctx, cancel, message, attachments)
	}
}

// CompactSession generates a summary and compacts the session history.
func (a *App) CompactSession(ctx context.Context, cancel context.CancelFunc, additionalPrompt string) {
	if a.controller != nil {
		a.controller.CompactSession(ctx, cancel, additionalPrompt)
	}
}

// ResolveInput resolves user input (skill commands first, then agent commands)
// into the content ready to send to the agent. Stub: returns the input unchanged.
func (a *App) ResolveInput(ctx context.Context, input string) string {
	_ = ctx
	return input
}

// RunBangCommand runs a shell command and surfaces its output as a runtime
// event. Delegates to the wire-backed controller.
func (a *App) RunBangCommand(ctx context.Context, command string) {
	if a.controller != nil {
		a.controller.RunBangCommand(ctx, command)
	}
}

// RunSkillFork dispatches a fork-mode skill in an isolated sub-session.
// The wire protocol has no fork-skill notion; treat it as a normal turn that
// names the skill and task so the gateway scenario can dispatch it.
func (a *App) RunSkillFork(ctx context.Context, cancel context.CancelFunc, skillName, task string, attachments []messages.Attachment) {
	if a.controller != nil {
		a.controller.Run(ctx, cancel, "/"+skillName+" "+task, attachments)
	}
}

// UpdateSessionTitle updates the current session's title and persists it.
func (a *App) UpdateSessionTitle(ctx context.Context, title string) error {
	if a.controller != nil {
		return a.controller.UpdateSessionTitle(ctx, title)
	}
	return nil
}

// CurrentAgentSkills returns the available skills for the current agent,
// sourced from the welcome-handshake capability block (skills are gateway
// commands under the "skill" namespace). Empty until SetSkills runs.
func (a *App) CurrentAgentSkills() []skills.Skill {
	a.agentInfoMu.Lock()
	defer a.agentInfoMu.Unlock()
	return slices.Clone(a.skills)
}

// SetSkills records the skill catalog projected from the welcome handshake.
// Each entry carries a name and a one-line summary (the wire protocol does not
// carry skill bodies, so only the listing fields are populated).
func (a *App) SetSkills(skillList []skills.Skill) {
	a.agentInfoMu.Lock()
	defer a.agentInfoMu.Unlock()
	a.skills = slices.Clone(skillList)
}

// SetAgentInfo records the capability view projected from a session_ready
// frame: the tool/command names, the selectable model-profile names and the
// active model. The translator calls this so /tools, /skills (commands) and the
// model picker populate. Slices are cloned to decouple from the caller's
// decoded wire maps.
func (a *App) SetAgentInfo(toolNames, commandNames, modelNames []string, activeModel string) {
	a.agentInfoMu.Lock()
	defer a.agentInfoMu.Unlock()
	a.toolNames = slices.Clone(toolNames)
	a.commandNames = slices.Clone(commandNames)
	a.modelNames = slices.Clone(modelNames)
	a.activeModel = activeModel
}

// CurrentAgentCommands returns the commands for the active agent, sourced from
// the most recent session_ready frame. The wire protocol carries only command
// names, so each command has an empty body (name-only completion entries).
func (a *App) CurrentAgentCommands(ctx context.Context) types.Commands {
	_ = ctx
	a.agentInfoMu.Lock()
	defer a.agentInfoMu.Unlock()
	if len(a.commandNames) == 0 {
		return nil
	}
	cmds := make(types.Commands, len(a.commandNames))
	for _, name := range a.commandNames {
		cmds[name] = types.Command{}
	}
	return cmds
}

// SkillCommandFork reports whether input is a slash command for a fork-mode
// skill and, if so, returns the skill name and task. Stub: always reports false.
func (a *App) SkillCommandFork(ctx context.Context, input string) (skillName, task string, ok bool) {
	_, _ = ctx, input
	return "", "", false
}

// IsReadOnly reports whether the session is read-only.
func (a *App) IsReadOnly() bool {
	return a.readOnly
}

// ShouldExitAfterFirstResponse reports whether the app should exit after the
// first assistant response completes.
func (a *App) ShouldExitAfterFirstResponse() bool {
	return a.exitAfterFirstResponse
}

// Session returns the current session.
func (a *App) Session() *session.Session {
	return a.session
}

// SetSession replaces the current session view-model.
func (a *App) SetSession(sess *session.Session) {
	if sess != nil {
		a.session = sess
	}
}

// SubscribeWith subscribes to app events using a custom send function.
// Multiple concurrent subscribers are supported: a single fan-out goroutine
// drains the event stream and dispatches a copy to each one. Slow subscribers
// drop events rather than block the bus. Mirrors cagent pkg/app.SubscribeWith.
func (a *App) SubscribeWith(ctx context.Context, send func(tea.Msg)) {
	ch := make(chan tea.Msg, subscriberBufferSize)
	a.addSubscriber(ch)
	defer a.removeSubscriber(ch)

	a.fanoutOnce.Do(a.startFanOut)

	for {
		select {
		case <-ctx.Done():
			return
		case msg := <-ch:
			send(msg)
		}
	}
}

func (a *App) addSubscriber(ch chan tea.Msg) {
	a.subsMu.Lock()
	defer a.subsMu.Unlock()
	a.subs = append(a.subs, ch)
}

func (a *App) removeSubscriber(ch chan tea.Msg) {
	a.subsMu.Lock()
	defer a.subsMu.Unlock()
	a.subs = slices.DeleteFunc(a.subs, func(c chan tea.Msg) bool { return c == ch })
}

// startFanOut runs once per App. It scatters every event to all
// currently-registered subscribers. Sends are non-blocking; if a subscriber's
// buffer is full the event is dropped for that subscriber so one slow consumer
// cannot stall the others.
func (a *App) startFanOut() {
	go func() {
		for ev := range a.events {
			var msg tea.Msg = ev
			a.subsMu.Lock()
			subs := slices.Clone(a.subs)
			a.subsMu.Unlock()
			for _, ch := range subs {
				select {
				case ch <- msg:
				default:
				}
			}
		}
	}()
}

// SnapshotsEnabled reports whether session snapshots (undo) are available.
// Stub: returns false until the adapter takes over.
func (a *App) SnapshotsEnabled() bool {
	return false
}

// SendFirstMessage returns a command that sends the first message of the
// session. When the controller has a queued first message it is emitted as a
// SendMsg so it flows through the normal TUI send path (queueing, title
// generation, event fan-out). Returns nil when there is no first message.
func (a *App) SendFirstMessage() tea.Cmd {
	if a.controller == nil {
		return nil
	}
	content, ok := a.controller.FirstMessage()
	if !ok {
		return nil
	}
	return func() tea.Msg {
		return messages.SendMsg{Content: content}
	}
}

// RunWithMessage runs one agent loop seeded with a pre-built message.
// Delegates to the wire-backed controller.
func (a *App) RunWithMessage(ctx context.Context, cancel context.CancelFunc, msg *session.Message) {
	if a.controller != nil {
		a.controller.RunWithMessage(ctx, cancel, msg)
	}
}

// Resume delivers a tool-confirmation decision to a waiting tool call.
// Delegates to the wire-backed controller.
func (a *App) Resume(req runtime.ResumeRequest) {
	if a.controller != nil {
		a.controller.Resume(req)
	}
}

// TogglePause toggles the runtime pause state. Stub: reports not paused and
// not supported until the adapter takes over.
func (a *App) TogglePause() (paused, supported bool) {
	return false, false
}

// NewSession starts a fresh session. Delegates to the wire-backed controller.
func (a *App) NewSession() {
	if a.controller != nil {
		a.controller.NewSession()
	}
}

// SwitchAgent switches the active agent. Stub: returns nil until the adapter
// takes over.
func (a *App) SwitchAgent(agentName string) error {
	_ = agentName
	return nil
}

// SetCurrentAgentModel overrides the active agent's model by forwarding a
// "/model <name>" command to the gateway (which owns the switch_model command).
// An empty ref (the picker's "select default" path) is ignored since the wire
// protocol has no notion of clearing an override.
func (a *App) SetCurrentAgentModel(ctx context.Context, modelRef string) error {
	_ = ctx
	if modelRef == "" {
		return nil
	}
	if a.controller != nil {
		a.controller.SwitchModel(modelRef)
	}
	return nil
}

// SupportsModelSwitching reports whether the current backend can switch models.
// True once a session_ready frame advertised at least one selectable model.
func (a *App) SupportsModelSwitching() bool {
	a.agentInfoMu.Lock()
	defer a.agentInfoMu.Unlock()
	return len(a.modelNames) > 0
}

// CycleAgentThinkingLevel advances the active agent's thinking-effort level.
// The returned string is the new effort-level name (e.g. "high"); empty here.
// Stub: returns the zero level and nil until the adapter takes over.
func (a *App) CycleAgentThinkingLevel(ctx context.Context) (string, error) {
	_ = ctx
	return "", nil
}

// AvailableModels lists the models selectable for the current agent, sourced
// from the session_ready model-profile names. The wire protocol carries only
// names (no catalog/pricing metadata), so each choice is a bare name with its
// IsCurrent flag set to match the active model.
func (a *App) AvailableModels(ctx context.Context) []runtime.ModelChoice {
	_ = ctx
	a.agentInfoMu.Lock()
	defer a.agentInfoMu.Unlock()
	if len(a.modelNames) == 0 {
		return nil
	}
	choices := make([]runtime.ModelChoice, 0, len(a.modelNames))
	for _, name := range a.modelNames {
		choices = append(choices, runtime.ModelChoice{
			Name:      name,
			Ref:       name,
			IsCurrent: name == a.activeModel,
		})
	}
	return choices
}

// PermissionsInfo returns the team-level permission patterns. Stub: returns
// nil until the adapter takes over.
func (a *App) PermissionsInfo() *runtime.PermissionsInfo {
	return nil
}

// CurrentAgentTools returns the tools available to the active agent, sourced
// from the most recent session_ready frame. The wire protocol carries only tool
// names, so each tool is a name-only definition (no schema/description).
func (a *App) CurrentAgentTools(ctx context.Context) ([]tools.Tool, error) {
	_ = ctx
	a.agentInfoMu.Lock()
	defer a.agentInfoMu.Unlock()
	if len(a.toolNames) == 0 {
		return nil, nil
	}
	ts := make([]tools.Tool, 0, len(a.toolNames))
	for _, name := range a.toolNames {
		ts = append(ts, tools.Tool{Name: name})
	}
	return ts, nil
}

// ResolveCommand converts a /command into its prompt text. Stub: returns the
// input unchanged until the adapter takes over.
func (a *App) ResolveCommand(ctx context.Context, userInput string) string {
	_ = ctx
	return userInput
}

// LookupCommand parses userInput as a /command invocation. Stub: always reports
// no match until the adapter takes over.
func (a *App) LookupCommand(ctx context.Context, userInput string) (types.Command, string, bool) {
	_, _ = ctx, userInput
	return types.Command{}, "", false
}

// TrackCurrentAgentModel records the active agent's current model. Stub: no-op.
func (a *App) TrackCurrentAgentModel(model string) {
	_ = model
}

// PlainTextTranscript renders the session as plain text. Stub: returns the
// empty string until the adapter takes over.
func (a *App) PlainTextTranscript() string {
	return ""
}

// ExportHTML writes the session transcript to an HTML file. Stub: returns an
// empty path and nil until the adapter takes over.
func (a *App) ExportHTML(ctx context.Context, filename string) (string, error) {
	_, _ = ctx, filename
	return "", nil
}

// RegenerateSessionTitle regenerates the session title. Stub: returns nil until
// the adapter takes over.
func (a *App) RegenerateSessionTitle(ctx context.Context) error {
	_ = ctx
	return nil
}

// SessionStore returns the session store. Stub: returns nil until the adapter
// supplies a real store.
func (a *App) SessionStore() session.Store {
	return nil
}

// ReplaceSession swaps the active session. Stub: replaces the in-memory
// reference only; the adapter wires persistence.
func (a *App) ReplaceSession(ctx context.Context, sess *session.Session) {
	_ = ctx
	a.session = sess
}

// UndoLastSnapshot restores the most recent snapshot. Stub: returns a zero
// result and nil until the adapter takes over.
func (a *App) UndoLastSnapshot(ctx context.Context) (UndoSnapshotResult, error) {
	_ = ctx
	return UndoSnapshotResult{}, nil
}

// ResetSnapshot restores history, keeping the given number of entries.
// Stub: returns a zero result and nil until the adapter takes over.
func (a *App) ResetSnapshot(ctx context.Context, keep int) (UndoSnapshotResult, error) {
	_, _ = ctx, keep
	return UndoSnapshotResult{}, nil
}

// ListSnapshots returns the available snapshot indices. Stub: returns nil until
// the adapter takes over.
func (a *App) ListSnapshots() []int {
	return nil
}
