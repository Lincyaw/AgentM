// Package tui provides the top-level TUI model with tab and session management.
package tui

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/app"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/audio/transcribe"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/history"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/paths"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/runtime"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/session"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/userconfig"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/version"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/animation"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/commands"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/completion"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/editor"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/editor/completions"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/notification"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/spinner"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/statusbar"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/tabbar"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/dialog"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/internal/termfeatures"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/messages"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/page/chat"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/service"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/service/supervisor"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/service/tuistate"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
)

// SessionSpawner creates new sessions with their own runtime.
// This is an alias to the supervisor package's SessionSpawner type.
type SessionSpawner = supervisor.SessionSpawner

// FocusedPanel represents which panel is currently focused
type FocusedPanel string

const (
	PanelContent FocusedPanel = "content"
	PanelEditor  FocusedPanel = "editor"

	// resizeHandleWidth is the width of the draggable center portion of the resize handle
	resizeHandleWidth = 8
	// appPaddingHorizontal is total horizontal padding from AppStyle (left + right)
	appPaddingHorizontal = 2 * styles.AppPadding
	// minEditorLines keeps the default composer compact while preserving a
	// draggable multiline editor.
	minEditorLines = 2
)

// Model is the top-level TUI model that wraps the chat page.
type appModel struct {
	supervisor *supervisor.Supervisor
	tabBar     *tabbar.TabBar
	tuiStore   *tuistate.Store

	// Per-session chat pages (kept alive for streaming continuity)
	chatPages     map[string]chat.Page
	sessionStates map[string]*service.SessionState

	// Per-session editors (preserved across tab switches for draft text)
	editors map[string]editor.Editor

	// Active session (convenience pointers to the currently visible session)
	application  *app.App
	sessionState *service.SessionState
	chatPage     chat.Page
	editor       editor.Editor

	// Shared history for command history across all editors
	history *history.History

	// UI components
	notification notification.Manager
	dialogMgr    dialog.Manager
	statusBar    statusbar.StatusBar
	completions  completion.Manager

	// startupWarnings are surfaced once from Init after construction-time
	// recovery paths have finished.
	startupWarnings []string

	// Speech-to-text
	transcriber  Transcriber
	transcriptCh chan string // bridges transcriber goroutine → Bubble Tea event loop

	// Working state indicator (resize handle spinner)
	workingSpinner spinner.Spinner

	// animFrame is the current animation frame, used to rotate the window
	// title spinner so that tmux can detect pane activity.
	animFrame int

	// Window state
	wWidth, wHeight int
	width, height   int

	// Content area height (height minus editor, tab bar, resize handle, status bar)
	contentHeight int
	// Bottom-surface height from the last layout pass.
	bottomSurfaceLayoutHeight int

	// Editor resize state
	editorLines      int
	isDragging       bool
	isHoveringHandle bool

	// Focus state
	focusedPanel FocusedPanel

	lastExitRequest     time.Time
	lastEscClearRequest time.Time

	// Claude Code-style bottom activity surface. Background workflow sessions
	// stay routed by supervisor, while shell/tool and monitor activity stays as
	// rows instead of being promoted into tabs.
	mainSessionID            string
	bottomActivityRowsHidden bool
	workflowTaskPickerOpen   bool
	workflowTaskPickerIndex  int
	workflowTranscripts      map[string]string
	workflowVisible          map[string]bool
	backgroundActivities     map[string]backgroundActivity
	shortcutSheetOpen        bool
	transcriptDetailed       bool
	transcriptVerbose        bool

	// keyboardEnhancements stores the last keyboard enhancements message
	keyboardEnhancements *tea.KeyboardEnhancementsMsg

	// keyboardEnhancementsSupported tracks whether the terminal supports keyboard enhancements
	keyboardEnhancementsSupported bool

	// program holds a reference to the tea.Program so that we can
	// perform a full terminal release/restore cycle on focus events.
	program *tea.Program

	// dockerDesktop is true when running inside Docker Desktop's terminal
	// (TERM_PROGRAM=docker_desktop). Focus reporting and the terminal
	// release/restore cycle on tab switch are only enabled in this
	// environment.
	dockerDesktop bool

	// focused tracks whether the terminal currently has focus. Used to
	// filter spurious FocusMsg events (RestoreTerminal re-enables focus
	// reporting and delivers one even though we never blurred). Starts
	// at the zero value (false) so the first FocusMsg is treated as a
	// real focus event — in Docker Desktop that runs the release/restore
	// cycle which re-emits terminal mode escape sequences.
	focused bool

	// tickPaused is true while we should drop animation.TickMsg events
	// (and let the tick chain die). Set on BlurMsg and cleared on the
	// next real FocusMsg. Tracked separately from `focused` so that ticks
	// keep flowing at startup even before any focus event arrives — some
	// terminals never send FocusMsg.
	tickPaused bool

	// pendingRestores maps runtime tab IDs (supervisor routing keys) to
	// persisted session-store IDs. When a tab with a pending restore is first
	// switched to, the persisted session is loaded via replaceActiveSession —
	// the same code path as the /sessions command.
	//
	// This map also serves as the authoritative source for "which persisted
	// session ID does this tab represent?" until the restore completes, at
	// which point the app's live session ID takes over.
	pendingRestores map[string]string

	// pendingSidebarCollapsed maps runtime tab IDs to their persisted sidebar
	// collapsed state. Consumed when a chat page is first created for a
	// restored tab (in handleSwitchTab) and then removed from the map.
	pendingSidebarCollapsed map[string]bool

	// stashedDialogs holds background dialog instances that were on screen
	// when the user navigated away from a tab. The dialog instance preserves
	// in-progress input so that returning to the tab restores the same dialog rather than
	// rebuilding a fresh one from the originating runtime event.
	//
	// The stored event is matched against the supervisor's pending event on
	// return: if they no longer match (because the agent superseded the
	// prompt) the stashed dialog is discarded and a fresh one is built.
	stashedDialogs map[string]stashedDialog

	// pendingActiveTab is the tab ID to switch to on Init(). Set when the
	// previously focused tab differs from the initial tab.
	pendingActiveTab string

	ready bool
	err   error

	// hideSidebar hides the sidebar and disables the ctrl+b toggle.
	hideSidebar bool

	// buildCommandCategories is a function that returns the list of command categories.
	buildCommandCategories func(context.Context, tea.Model) []commands.Category

	appName    string
	appVersion string

	// disabledCommands holds slash commands to hide and disable.
	// Normalized to start with "/".
	disabledCommands map[string]bool
}

// Transcriber is the speech-to-text interface used by the TUI. It is an
// interface (rather than the concrete *transcribe.Transcriber) so that tests
// can inject a fake implementation via WithTranscriber and so that the TUI
// does not depend on a concrete audio backend.
type Transcriber interface {
	Start(ctx context.Context, handler transcribe.TranscriptHandler) error
	Stop()
	IsRunning() bool
	IsSupported() bool
}

// Option configures the TUI.
type Option func(*appModel)

// WithHideSidebar hides the chat sidebar. The rest of the chrome (tab bar,
// status bar, dialogs) remains visible. The user cannot bring the sidebar
// back via the TUI.
func WithHideSidebar() Option {
	return func(m *appModel) {
		m.hideSidebar = true
	}
}

// WithAppName sets the application name.
//
// If not provided, defaults to "AgentM Terminal".
func WithAppName(name string) Option {
	return func(m *appModel) {
		m.appName = name
	}
}

// WithVersion sets the application version.
//
// If not provided, defaults to version.Version.
func WithVersion(v string) Option {
	return func(m *appModel) {
		m.appVersion = v
	}
}

// WithDisabledCommands hides and disables the given slash commands so they
// are stripped from the command palette, the slash-command parser, and
// completion. Each entry is normalized to start with "/" (so "cost" and
// "/cost" are equivalent) and lower-cased to match the registered slash
// command names (so "/Cost" and "/cost" are equivalent).
func WithDisabledCommands(slashCommands []string) Option {
	return func(m *appModel) {
		if len(slashCommands) == 0 {
			return
		}
		if m.disabledCommands == nil {
			m.disabledCommands = make(map[string]bool, len(slashCommands))
		}
		for _, c := range slashCommands {
			c = strings.ToLower(strings.TrimSpace(c))
			if c == "" {
				continue
			}
			if !strings.HasPrefix(c, "/") {
				c = "/" + c
			}
			m.disabledCommands[c] = true
		}
	}
}

// WithCommandBuilder builds the command categories shown in the command
// palette from the given function. It overrides the default command category
// builder. To include the default commands, the given function should call
// commands.BuildCommandCategories and merge the result with its own.
//
// The tea.Model passed to the builder function must not be accessed during
// the build call itself - it should only be captured for use within command
// Execute functions. There is no guarantee that the tea.Model holds all
// dependencies during the build phase, which may cause [core.Resolve] to panic.
func WithCommandBuilder(
	fn func(context.Context, tea.Model) []commands.Category,
) Option {
	return func(m *appModel) {
		m.buildCommandCategories = fn
	}
}

// WithTranscriber overrides the speech-to-text backend used by the TUI. This
// is intended for tests that need to exercise speech handlers without
// connecting to a real audio device or external API.
func WithTranscriber(t Transcriber) Option {
	return func(m *appModel) {
		if t != nil {
			m.transcriber = t
		}
	}
}

// New creates a new Model.
func New(ctx context.Context, spawner SessionSpawner, initialApp *app.App, initialWorkingDir string, cleanup func(), opts ...Option) tea.Model {
	// Initialize supervisor
	sv := supervisor.New(spawner)

	// Initialize tab bar with configurable title length from user settings
	tabTitleMaxLen := userconfig.Get().GetTabTitleMaxLength()
	tb := tabbar.New(tabTitleMaxLen)

	// Initialize tab store
	var ts *tuistate.Store
	var tsErr error
	startupWarnings := []string{}
	ts, tsErr = tuistate.New()
	if tsErr != nil {
		slog.WarnContext(ctx, "Failed to open TUI state store, tabs won't persist", "error", tsErr)
		startupWarnings = append(startupWarnings, "TUI state unavailable; tabs won't persist.")
	}

	// Initialize shared command history
	historyStore, err := history.NewAtDir(paths.GetDataDir())
	if err != nil {
		slog.WarnContext(ctx, "Failed to initialize command history", "error", err)
		startupWarnings = append(startupWarnings, "Command history unavailable.")
	}

	initialSessionState := service.NewSessionState(initialApp.Session())
	sessID := initialApp.Session().ID

	m := &appModel{
		buildCommandCategories: func(ctx context.Context, model tea.Model) []commands.Category {
			if model != nil {
				if m, ok := model.(*appModel); ok && m.application != nil {
					return commands.BuildCommandCategories(ctx, m.application)
				}
			}
			return commands.BuildCommandCategories(ctx, initialApp)
		},
		supervisor:                    sv,
		tabBar:                        tb,
		tuiStore:                      ts,
		chatPages:                     map[string]chat.Page{},
		editors:                       map[string]editor.Editor{},
		sessionStates:                 map[string]*service.SessionState{sessID: initialSessionState},
		application:                   initialApp,
		sessionState:                  initialSessionState,
		mainSessionID:                 sessID,
		workflowTranscripts:           map[string]string{},
		workflowVisible:               map[string]bool{},
		backgroundActivities:          map[string]backgroundActivity{},
		history:                       historyStore,
		pendingRestores:               make(map[string]string),
		pendingSidebarCollapsed:       make(map[string]bool),
		stashedDialogs:                make(map[string]stashedDialog),
		notification:                  notification.New(),
		dialogMgr:                     dialog.New(),
		completions:                   completion.New(),
		startupWarnings:               startupWarnings,
		transcriber:                   transcribe.New(os.Getenv("OPENAI_API_KEY")),
		workingSpinner:                spinner.New(spinner.ModeSpinnerOnly, styles.SpinnerDotsHighlightStyle),
		focusedPanel:                  PanelEditor,
		editorLines:                   minEditorLines,
		keyboardEnhancementsSupported: termfeatures.SupportsModifiedEnter(os.Getenv),
		dockerDesktop:                 os.Getenv("TERM_PROGRAM") == "docker_desktop",
		appName:                       "AgentM Terminal",
		appVersion:                    version.Version,
		hideSidebar:                   true,
	}

	// Apply options
	for _, opt := range opts {
		opt(m)
	}

	// Create initial editor (after options are applied so command builder is set)
	initialEditor := editor.New(historyStore, m.editorOpts()...)
	m.editors[sessID] = initialEditor
	m.editor = initialEditor

	// Create initial chat page after options are applied.
	initialChatPage := chat.New(initialApp, initialSessionState, m.chatPageOpts()...)
	m.chatPages[sessID] = initialChatPage
	m.chatPage = initialChatPage

	// Initialize status bar (pass m as help provider)
	m.statusBar = statusbar.New(m, statusbar.WithTitle(""))

	// Add the initial session to the supervisor
	sv.AddSession(ctx, initialApp, initialApp.Session(), initialWorkingDir, cleanup)

	// Restore persisted tabs or persist the initial one.
	m.restoreTabs(ctx, ts, sv, spawner, initialApp, sessID, initialWorkingDir)

	// Initialize tab bar with current tabs
	tabs, activeIdx := sv.GetTabs()
	m.syncTabChrome(tabs, activeIdx)

	// Make sure to stop on context cancellation.
	// Note: chatPages/editors cleanup is handled by cleanupAll() on the
	// normal exit path (ExitConfirmedMsg). We don't iterate those maps
	// here to avoid racing with the Bubble Tea event loop.
	go func() {
		<-ctx.Done()
		if ts != nil {
			_ = ts.Close()
		}
		sv.Shutdown()
	}()

	return m
}

// Resolve implements dependency resolution for the appModel.
// See core.Resolve for additional information.
func (m *appModel) Resolve(v any) any {
	switch v.(type) {
	case **app.App:
		return m.application
	case **service.SessionState:
		return m.sessionState
	case *chat.Page:
		return m.chatPage
	case *editor.Editor:
		return m.editor
	}

	return nil
}

// SetProgram sets the tea.Program for the supervisor to send routed messages.
func (m *appModel) SetProgram(p *tea.Program) {
	m.program = p
	m.supervisor.SetProgram(p)
}

// reapplyKeyboardEnhancements forwards the cached keyboard enhancements message
// to the active chat page and editor so new/replaced instances pick up the
// terminal's key disambiguation support.
func (m *appModel) reapplyKeyboardEnhancements() {
	if m.keyboardEnhancements == nil {
		return
	}
	_ = m.updateChatCmd(*m.keyboardEnhancements)
	_ = m.updateEditorCmd(*m.keyboardEnhancements)
}

func (m *appModel) commandCategories() []commands.Category {
	categories := m.buildCommandCategories(context.Background(), m)
	if len(m.disabledCommands) == 0 {
		return categories
	}
	filtered := make([]commands.Category, 0, len(categories))
	for _, cat := range categories {
		items := make([]commands.Item, 0, len(cat.Commands))
		for _, item := range cat.Commands {
			if m.disabledCommands[item.SlashCommand] {
				continue
			}
			items = append(items, item)
		}
		if len(items) == 0 {
			continue
		}
		cat.Commands = items
		filtered = append(filtered, cat)
	}
	return filtered
}

// refreshCommandInputs rebuilds and injects the active command parser/completion
// providers into the current chat + editor pair.
func (m *appModel) refreshCommandInputs() tea.Cmd {
	categories := m.commandCategories()

	if m.chatPage != nil {
		m.chatPage.SetCommandParser(commands.NewParser(categories...))
	}
	if m.editor != nil {
		return m.editor.SetCompletions(
			completions.NewCommandCompletion(categories),
			completions.NewResourceCompletion(m.availableAgentDetails),
		)
	}

	return nil
}

// chatPageOpts returns the chat.PageOption slice derived from the current
// appModel configuration.
func (m *appModel) chatPageOpts() []chat.PageOption {
	opts := []chat.PageOption{
		chat.WithCommandParser(commands.NewParser(m.commandCategories()...)),
	}

	if m.hideSidebar {
		opts = append(opts, chat.WithHideSidebar())
	}
	return opts
}

// editorOpts returns the editor.Option slice derived from the current appModel.
func (m *appModel) editorOpts() []editor.Option {
	opts := []editor.Option{
		editor.WithCompletions(
			completions.NewCommandCompletion(m.commandCategories()),
			completions.NewResourceCompletion(m.availableAgentDetails),
		),
	}
	if m.application.IsReadOnly() {
		opts = append(opts, editor.WithReadOnly())
	}
	return opts
}

func (m *appModel) availableAgentDetails() []runtime.AgentDetails {
	if m == nil || m.sessionState == nil {
		return nil
	}
	return m.sessionState.AvailableAgents()
}

func (m *appModel) tabBarHeight() int {
	if m.tabBar.CanCollapseIntoBackgroundChrome(m.mainSessionID) {
		return 0
	}
	return m.tabBar.Height()
}

func (m *appModel) statusBarHeight() int {
	if m.completions.Open() {
		return 0
	}
	return m.statusBar.Height()
}

func (m *appModel) syncTabChrome(tabs []messages.TabInfo, activeIdx int) bool {
	prevHeight := m.tabBarHeight()
	m.tabBar.SetTabs(tabs, activeIdx)
	nextHeight := m.tabBarHeight()
	m.statusBar.SetActivity(m.backgroundActivityText())
	return nextHeight != prevHeight
}

func (m *appModel) backgroundActivityText() string {
	workflowText := ""
	if m.tabBar.HasOnlyInactiveBackgroundTabs() {
		workflowText = m.workflowBackgroundText()
	}
	if !m.bottomActivityRowsHidden {
		return workflowText
	}
	return joinBackgroundStatusParts(workflowText, m.backgroundActivityCountText())
}

func (m *appModel) workflowBackgroundText() string {
	total, running, needsAttention := m.tabBar.BackgroundStats()
	if total == 0 {
		return ""
	}

	switch {
	case needsAttention > 0:
		noun := "workflow"
		verb := "needs"
		if needsAttention != 1 {
			noun = "workflows"
			verb = "need"
		}
		return fmt.Sprintf("%d %s %s input (Ctrl+n)", needsAttention, noun, verb)
	case running > 0:
		noun := "workflow"
		if running != 1 {
			noun = "workflows"
		}
		return fmt.Sprintf("%d %s running (Ctrl+n)", running, noun)
	default:
		noun := "workflow"
		if total != 1 {
			noun = "workflows"
		}
		return fmt.Sprintf("%d %s done (Ctrl+n)", total, noun)
	}
}

// initSessionComponents creates a new chat page, session state, and editor for
// the given app and stores them in the per-session maps under tabID. The active
// convenience pointers (m.chatPage, m.sessionState, m.editor) are also updated.
func (m *appModel) initSessionComponents(tabID string, a *app.App, sess *session.Session) {
	cp, ss, ed := m.createSessionComponents(tabID, a, sess)

	m.application = a
	m.sessionState = ss
	m.chatPage = cp
	m.editor = ed
}

func (m *appModel) createSessionComponents(tabID string, a *app.App, sess *session.Session) (chat.Page, *service.SessionState, editor.Editor) {
	ss := service.NewSessionState(sess)
	cp := chat.New(a, ss, m.chatPageOpts()...)
	ed := editor.New(m.history, m.editorOpts()...)

	m.chatPages[tabID] = cp
	m.sessionStates[tabID] = ss
	m.editors[tabID] = ed

	return cp, ss, ed
}

// initAndFocusComponents returns a batch of commands that initializes and focuses
// the active chat page and editor, then resizes everything.
func (m *appModel) initAndFocusComponents() tea.Cmd {
	m.reapplyKeyboardEnhancements()
	return tea.Batch(
		m.chatPage.Init(),
		m.editor.Init(),
		m.editor.Focus(),
		m.resizeAll(),
	)
}

func (m *appModel) withStartupWarnings(cmds ...tea.Cmd) tea.Cmd {
	if len(m.startupWarnings) == 0 {
		return tea.Batch(cmds...)
	}
	for _, warning := range m.startupWarnings {
		cmds = append(cmds, notification.WarningCmd(warning))
	}
	m.startupWarnings = nil
	return tea.Batch(cmds...)
}

// Init initializes the model.
func (m *appModel) Init() tea.Cmd {
	// If a different tab should be active on startup, switch to it directly.
	// The initial tab's pending restore stays lazy — it will be loaded via
	// handleSwitchTab when the user eventually opens it, just like every
	// other non-active restored tab.
	if m.pendingActiveTab != "" {
		tabID := m.pendingActiveTab
		m.pendingActiveTab = ""
		_, switchCmd := m.handleSwitchTab(tabID)
		return m.withStartupWarnings(m.dialogMgr.Init(), switchCmd)
	}

	// If the initial tab has a pending session restore, go through
	// replaceActiveSession — the same code path as the /sessions command.
	activeID := m.supervisor.ActiveID()
	if oldSessionID, ok := m.pendingRestores[activeID]; ok {
		delete(m.pendingRestores, activeID)
		if store := m.application.SessionStore(); store != nil {
			if sess, err := store.GetSession(context.Background(), oldSessionID); err == nil {
				_, cmd := m.replaceActiveSession(context.Background(), sess)

				if m.tuiStore != nil && sess.WorkingDir != "" {
					if err := m.tuiStore.UpdateTabWorkingDir(context.Background(), oldSessionID, sess.WorkingDir); err != nil {
						slog.Warn("Failed to update persisted working dir", "error", err)
					}
				}

				cmd = tea.Batch(cmd, m.applySidebarCollapsed(activeID))
				m.persistActiveTab(sess.ID)

				return m.withStartupWarnings(m.dialogMgr.Init(), cmd)
			}
		}
	}

	return m.withStartupWarnings(
		m.dialogMgr.Init(),
		m.chatPage.Init(),
		m.editor.Init(),
		m.editor.Focus(),
		m.application.SendFirstMessage(),
	)
}

// Update handles messages.
func (m *appModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	// --- Routing & Animation ---

	case messages.RoutedMsg:
		return m.handleRoutedMsg(msg)

	case animation.TickMsg:
		// Drop the tick (and let the chain die) while we're blurred.
		// animation.StartTick re-arms the chain on the next FocusMsg so
		// spinners resume immediately when the user comes back.
		if m.tickPaused {
			return m, nil
		}
		cmds := []tea.Cmd{m.updateChatCmd(msg)}
		// Update working spinner
		if m.chatPage.IsWorking() {
			model, cmd := m.workingSpinner.Update(msg)
			m.workingSpinner = model.(spinner.Spinner)
			cmds = append(cmds, cmd)
		}
		// Track frame for window-title spinner (tmux activity detection)
		m.animFrame = msg.Frame
		// Forward frame to tab bar for running indicator animation
		m.tabBar.SetAnimFrame(msg.Frame)
		if animation.HasActive() {
			cmds = append(cmds, animation.StartTick())
		}
		return m, tea.Batch(cmds...)

	// --- Tab management ---

	case messages.TabsUpdatedMsg:
		tabChromeChanged := m.syncTabChrome(msg.Tabs, msg.ActiveIdx)
		m.syncWorkflowTaskPickerState()
		bottomSurfaceHeightChanged := m.bottomSurfaceHeight(m.width) != m.bottomSurfaceLayoutHeight
		if tabChromeChanged || bottomSurfaceHeightChanged {
			cmd := m.resizeAll()
			return m, cmd
		}
		return m, nil

	case messages.SpawnSessionMsg:
		return m.handleSpawnSession(msg.WorkingDir, msg.Background)

	case messages.SwitchTabMsg:
		return m.handleSwitchTab(msg.SessionID)

	case messages.CloseTabMsg:
		return m.handleCloseTab(msg.SessionID)

	case messages.ReorderTabMsg:
		return m.handleReorderTab(msg)

	case messages.ToggleSidebarMsg:
		if m.hideSidebar {
			return m, nil
		}
		if m.tuiStore != nil {
			persistedID := m.persistedSessionID(m.supervisor.ActiveID())
			if err := m.tuiStore.ToggleSidebarCollapsed(context.Background(), persistedID); err != nil {
				slog.Warn("Failed to persist sidebar collapsed state", "error", err)
			}
		}
		return m, nil

	// --- Focus requests from content view ---

	case messages.RequestFocusMsg:
		switch msg.Target {
		case messages.PanelMessages:
			if m.focusedPanel != PanelContent {
				m.focusedPanel = PanelContent
				m.statusBar.InvalidateCache()
				m.editor.Blur()
			}
			if msg.ClickX != 0 || msg.ClickY != 0 {
				return m, m.chatPage.FocusMessageAt(msg.ClickX, msg.ClickY)
			}
			return m, m.chatPage.FocusMessages()
		case messages.PanelSidebarTitle:
			if m.focusedPanel != PanelContent {
				m.focusedPanel = PanelContent
				m.statusBar.InvalidateCache()
				m.chatPage.BlurMessages()
				m.editor.Blur()
			}
			return m, nil
		case messages.PanelEditor:
			if m.focusedPanel != PanelEditor {
				m.focusedPanel = PanelEditor
				m.statusBar.InvalidateCache()
				m.chatPage.BlurMessages()
				return m, m.editor.Focus()
			}
		}
		return m, nil

	// --- Working state from content view ---

	case messages.WorkingStateChangedMsg:
		return m.handleWorkingStateChanged(msg)

	// --- Statusbar invalidation ---

	case messages.InvalidateStatusBarMsg:
		m.statusBar.InvalidateCache()
		return m, nil

	case completion.OpenMsg, completion.CloseMsg:
		cmd := m.updateCompletionsCmd(msg)
		return m, tea.Batch(cmd, m.resizeAll())

	case completion.OpenedMsg:
		return m, m.resizeAll()

	case completion.ClosedMsg:
		cmd := m.updateEditorCmd(msg)
		return m, tea.Batch(cmd, m.resizeAll())

	// --- Window / Terminal ---

	case tea.WindowSizeMsg:
		m.wWidth, m.wHeight = msg.Width, msg.Height
		cmd := m.handleWindowResize(msg.Width, msg.Height)
		return m, cmd

	case tea.BlurMsg:
		m.focused = false
		m.tickPaused = true
		return m, nil

	case tea.FocusMsg:
		// Filter spurious FocusMsg: RestoreTerminal re-enables focus
		// reporting which delivers a FocusMsg even when we never blurred.
		if m.focused {
			return m, nil
		}
		m.focused = true

		var cmds []tea.Cmd
		if m.tickPaused {
			// Re-arm the tick chain that died while we were blurred.
			m.tickPaused = false
			if animation.HasActive() {
				cmds = append(cmds, animation.StartTick())
			}
		}
		if m.dockerDesktop && m.program != nil {
			// Docker Desktop: the terminal may have lost all mode state (alt
			// screen, mouse tracking, keyboard enhancements, background
			// color, etc.). A full release/restore cycle re-emits every mode
			// sequence and forces a complete repaint.
			cmds = append(cmds, func() tea.Msg {
				_ = m.program.ReleaseTerminal()
				_ = m.program.RestoreTerminal()
				return nil
			})
		}
		return m, tea.Batch(cmds...)

	case tea.KeyboardEnhancementsMsg:
		m.keyboardEnhancements = &msg
		m.keyboardEnhancementsSupported = msg.Flags != 0 || termfeatures.SupportsModifiedEnter(os.Getenv)
		m.statusBar.InvalidateCache()
		return m, tea.Batch(m.updateChatCmd(msg), m.updateEditorCmd(msg))

	// --- Keyboard input ---

	case tea.KeyPressMsg:
		return m.handleKeyPress(msg)

	case tea.PasteMsg:
		if m.dialogMgr.Open() {
			return m.forwardDialog(msg)
		}
		// When inline editing a past message, forward paste to the chat page
		// so the messages component can insert content into the inline textarea.
		if m.chatPage.IsInlineEditing() {
			return m.forwardChat(msg)
		}
		// Forward paste to editor
		return m.forwardEditor(msg)

	// --- Mouse ---

	case tea.MouseClickMsg:
		return m.handleMouseClick(msg)

	case tea.MouseMotionMsg:
		return m.handleMouseMotion(msg)

	case tea.MouseReleaseMsg:
		return m.handleMouseRelease(msg)

	case messages.WheelCoalescedMsg:
		return m.handleWheelCoalesced(msg)

	// --- Dialog lifecycle ---

	case dialog.OpenDialogMsg, dialog.CloseDialogMsg:
		return m.forwardDialog(msg)

	case dialog.ExitConfirmedMsg:
		m.cleanupAll()
		return m, tea.Quit

	case dialog.RuntimeResumeMsg:
		m.application.Resume(msg.Request)
		return m, nil

	case dialog.MultiChoiceResultMsg:
		if msg.DialogID == dialog.ToolRejectionDialogID {
			if msg.Result.IsCancelled {
				return m, nil
			}
			resumeMsg := dialog.HandleToolRejectionResult(msg.Result)
			if resumeMsg != nil {
				return m, tea.Sequence(
					core.CmdHandler(dialog.CloseDialogMsg{}),
					core.CmdHandler(*resumeMsg),
				)
			}
		}
		return m, nil

	// --- Terminal bell ---

	case messages.BellMsg:
		// Ring the terminal bell to alert the user that an inactive tab needs attention.
		// The BEL character (\a) is written to stderr which is typically the terminal.
		_, _ = fmt.Fprint(os.Stderr, "\a")
		return m, nil

	// --- Notifications ---

	case notificationCopiedMsg:
		m.notification = m.notification.MarkCopied(msg.ID)
		return m, nil

	case notification.ShowMsg, notification.HideMsg, notification.DismissMsg, notification.AutoHideMsg:
		updated, cmd := m.notification.Update(msg)
		m.notification = updated
		return m, cmd

	// --- Runtime event specializations ---

	case *runtime.TeamInfoEvent:
		m.sessionState.SetAvailableAgents(msg.AvailableAgents)
		m.sessionState.SetCurrentAgentName(msg.CurrentAgent)
		return m.forwardChat(msg)

	case *runtime.AgentInfoEvent:
		m.sessionState.SetCurrentAgentName(msg.AgentName)
		m.application.TrackCurrentAgentModel(msg.Model)
		chatModel, cmd := m.forwardChat(msg)
		if refreshCmd := m.refreshCommandInputs(); refreshCmd != nil {
			return chatModel, tea.Batch(cmd, refreshCmd)
		}
		return chatModel, cmd

	case *runtime.SessionTitleEvent:
		m.sessionState.SetSessionTitle(msg.Title)
		return m.forwardChat(msg)

	case *runtime.BackgroundActivityEvent:
		return m.handleBackgroundActivity(msg)

	// --- New session (slash command /new) ---

	case messages.NewSessionMsg:
		// /new spawns a new tab when a session spawner is configured.
		return m.handleSpawnSession("", false)

	case messages.ClearSessionMsg:
		// /clear resets the current tab with a fresh session in the same working dir.
		return m.handleClearSession()

	// --- Exit ---

	case messages.ExitSessionMsg:
		// If multiple tabs are open, close only the current tab instead of
		// quitting the entire application (see #2373).
		if m.supervisor != nil && m.supervisor.Count() > 1 {
			return m.handleCloseTab(m.supervisor.ActiveID())
		}
		m.cleanupAll()
		return m, tea.Quit

	case messages.ExitAfterFirstResponseMsg:
		m.cleanupAll()
		return m, tea.Quit

	// --- SendMsg from editor ---

	case messages.SendMsg:
		// Forward send messages to the active content view
		if m.history != nil && !msg.BypassQueue {
			_ = m.history.Add(msg.Content)
		}
		return m.forwardChat(msg)

	// --- File attachments (routed to editor) ---

	case messages.InsertFileRefMsg:
		if err := m.editor.AttachFile(msg.FilePath); err != nil {
			slog.Warn("failed to attach file", "path", msg.FilePath, "error", err)
			return m, nil
		}
		return m, notification.SuccessCmd("File attached: " + msg.FilePath)

	// --- Agent management ---

	case messages.SwitchAgentMsg:
		return m.handleSwitchAgent(msg.AgentName)

	// --- Session browser ---

	case messages.OpenSessionBrowserMsg:
		return m.handleOpenSessionBrowser()

	case messages.OpenSessionBrowserWithDataMsg:
		return m.handleOpenSessionBrowserWithData(msg.Sessions)

	case messages.LoadSessionMsg:
		return m.handleLoadSession(msg.SessionID)

	case messages.BranchFromEditMsg:
		return m.handleBranchFromEdit(msg)

	case messages.ForkSessionMsg:
		return m.handleForkSession()

	// --- Session commands (slash commands, command palette) ---

	case messages.ToggleYoloMsg:
		return m.handleToggleYolo()

	case messages.TogglePauseMsg:
		return m.handleTogglePause()

	case messages.ToggleHideToolResultsMsg:
		return m.handleToggleHideToolResults()

	case messages.ToggleSplitDiffMsg:
		return m.handleToggleSplitDiff()

	case messages.CompactSessionMsg:
		return m.handleCompactSession(msg.AdditionalPrompt)

	case messages.CopySessionToClipboardMsg:
		return m.handleCopySessionToClipboard()

	case messages.CopyLastResponseToClipboardMsg:
		return m.handleCopyLastResponseToClipboard()

	case messages.UndoSnapshotMsg:
		return m.handleUndoSnapshot()

	case messages.ShowSnapshotsDialogMsg:
		return m.handleShowSnapshotsDialog()

	case messages.ResetSnapshotMsg:
		return m.handleResetSnapshot(msg.Keep)

	case messages.ExportSessionMsg:
		return m.handleExportSession(msg.Filename)

	case messages.ToggleSessionStarMsg:
		sessionID := msg.SessionID
		if sessionID == "" {
			if sess := m.application.Session(); sess != nil {
				sessionID = sess.ID
			} else {
				return m, nil
			}
		}
		return m.handleToggleSessionStar(sessionID)

	case messages.DeleteSessionMsg:
		return m.handleDeleteSession(msg.SessionID)

	case messages.SetSessionTitleMsg:
		return m.handleSetSessionTitle(msg.Title)

	case messages.RegenerateTitleMsg:
		return m.handleRegenerateTitle()

	case messages.ShowCostDialogMsg:
		return m.handleShowCostDialog()

	case messages.ShowPermissionsDialogMsg:
		return m.handleShowPermissionsDialog()

	case messages.ShowToolsDialogMsg:
		return m.handleShowToolsDialog()

	case messages.ShowSkillsDialogMsg:
		return m.handleShowSkillsDialog()

	case messages.AgentCommandMsg:
		return m.handleAgentCommand(msg.Command)

	case messages.StartShellMsg:
		return m.startShell()

	// --- Model picker ---

	case messages.OpenModelPickerMsg:
		return m.handleOpenModelPicker()

	case messages.ChangeModelMsg:
		return m.handleChangeModel(msg.ModelRef)

	// --- Theme picker ---

	case messages.OpenThemePickerMsg:
		return m.handleOpenThemePicker()

	case messages.ChangeThemeMsg:
		return m.handleChangeTheme(msg.ThemeRef)

	case messages.ThemePreviewMsg:
		return m.handleThemePreview(msg.ThemeRef)

	case messages.ThemeCancelPreviewMsg:
		return m.handleThemeCancelPreview(msg.OriginalRef)

	case messages.ThemeChangedMsg:
		return m.applyThemeChanged()

	case messages.ThemeFileChangedMsg:
		return m.handleThemeFileChanged(msg.ThemeRef)

	// --- Speech-to-text ---

	case messages.StartSpeakMsg:
		if !m.transcriber.IsSupported() {
			return m, notification.InfoCmd("Speech-to-text is only supported on macOS")
		}
		return m.handleStartSpeak()

	case messages.StopSpeakMsg:
		return m.handleStopSpeak()

	case messages.SpeakTranscriptMsg:
		m.editor.InsertText(msg.Delta)
		cmd := m.waitForTranscript()
		return m, cmd

	// --- File attachments ---

	case messages.AttachFileMsg:
		return m.handleAttachFile(msg.FilePath)

	case messages.SendAttachmentMsg:
		if m.application.IsReadOnly() {
			return m, notification.WarningCmd("Session is read-only. No new messages can be sent.")
		}
		m.application.RunWithMessage(context.Background(), nil, msg.Content)
		return m, nil

	// --- URL opening ---

	case messages.OpenURLMsg:
		return m.handleOpenURL(msg.URL)

	// --- Errors ---

	case error:
		m.err = msg
		return m, nil

	default:
		// Handle runtime events for active session
		if event, isRuntimeEvent := msg.(runtime.Event); isRuntimeEvent {
			if agentName := event.GetAgentName(); agentName != "" {
				m.sessionState.SetCurrentAgentName(agentName)
			}
			return m.forwardChat(msg)
		}

		// Forward to dialog if open (and to chat in parallel)
		if m.dialogMgr.Open() {
			return m, tea.Batch(m.updateDialogCmd(msg), m.updateChatCmd(msg))
		}

		// Forward to completion manager, editor, and chat page in parallel
		return m, tea.Batch(m.updateCompletionsCmd(msg), m.updateEditorCmd(msg), m.updateChatCmd(msg))
	}
}

// handleRoutedMsg processes messages routed to specific sessions.
func (m *appModel) handleRoutedMsg(msg messages.RoutedMsg) (tea.Model, tea.Cmd) {
	activeID := m.supervisor.ActiveID()
	m.recordWorkflowTranscript(msg.SessionID, msg.Inner)
	if ev, ok := msg.Inner.(*runtime.BackgroundActivityEvent); ok {
		if ev.SessionID == "" {
			ev.SessionID = msg.SessionID
		}
		return m.handleBackgroundActivity(ev)
	}

	if msg.SessionID == activeID {
		// Active session: forward through Update for full processing (spinners, cmds, etc.)
		return m.Update(msg.Inner)
	}

	// Background session: update its chat page directly so streaming content accumulates.
	// UI-only cmds (spinners, scroll) are discarded since the page isn't visible.
	chatPage, ok := m.chatPages[msg.SessionID]
	var initCmd tea.Cmd
	if !ok {
		runner := m.supervisor.GetRunner(msg.SessionID)
		if runner == nil || runner.App == nil {
			return m, nil
		}
		var ed editor.Editor
		chatPage, _, ed = m.createSessionComponents(msg.SessionID, runner.App, runner.App.Session())
		initCmd = tea.Batch(chatPage.Init(), ed.Init())
	}

	// Update session state for inactive sessions
	if event, isRuntimeEvent := msg.Inner.(runtime.Event); isRuntimeEvent {
		if sessionState, ok := m.sessionStates[msg.SessionID]; ok {
			if agentName := event.GetAgentName(); agentName != "" {
				sessionState.SetCurrentAgentName(agentName)
			}
		}
	}

	// Update the inactive chat page (discard cmds — UI effects aren't needed for hidden pages)
	updated, _ := chatPage.Update(msg.Inner)
	m.chatPages[msg.SessionID] = updated.(chat.Page)
	return m, initCmd
}

// handleWorkingStateChanged updates the editor working indicator and resize handle spinner.
func (m *appModel) handleWorkingStateChanged(msg messages.WorkingStateChangedMsg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	// Update editor working state
	cmds = append(cmds, m.editor.SetWorking(msg.Working))

	// Start/stop working spinner
	if msg.Working {
		cmds = append(cmds, m.workingSpinner.Init())
	} else {
		m.workingSpinner.Stop()
	}

	return m, tea.Batch(cmds...)
}

// handleWindowResize handles window resize.
func (m *appModel) handleWindowResize(width, height int) tea.Cmd {
	m.wWidth, m.wHeight = width, height

	m.statusBar.SetWidth(width)
	m.tabBar.SetWidth(width - appPaddingHorizontal)

	m.width = width
	m.height = height

	if !m.ready {
		m.ready = true
	}

	return m.resizeAll()
}

// resizeAll recalculates all component sizes based on current window dimensions.
func (m *appModel) resizeAll() tea.Cmd {
	var cmds []tea.Cmd

	width, height := m.width, m.height
	innerWidth := width - appPaddingHorizontal

	// Calculate chrome height (everything that isn't content or editor).
	bottomSurfaceHeight := m.bottomSurfaceHeight(width)
	m.bottomSurfaceLayoutHeight = bottomSurfaceHeight
	chromeHeight := m.tabBarHeight() + m.statusBarHeight() + bottomSurfaceHeight + 1 // +1 for resize handle

	// Calculate editor height
	minLines := minEditorLines
	maxLines := max(minLines, (height-6)/2)
	m.editorLines = max(minLines, min(m.editorLines, maxLines))

	targetEditorHeight := m.editorLines - 1
	cmds = append(cmds, m.editor.SetSize(innerWidth, targetEditorHeight))
	_, editorHeight := m.editor.GetSize()
	// The editor's View() adds MarginBottom(1) which isn't included in GetSize(),
	// so account for it in the layout calculation.
	editorRenderedHeight := editorHeight + 1

	// Content gets remaining space
	m.contentHeight = max(1, height-chromeHeight-editorRenderedHeight)
	cmds = append(cmds, m.chatPage.SetSize(width, m.contentHeight))

	cmds = append(cmds, m.updateDialogCmd(tea.WindowSizeMsg{Width: width, Height: height}))

	m.completions.SetEditorBottom(editorRenderedHeight + m.statusBarHeight() + bottomSurfaceHeight)
	m.completions.Update(tea.WindowSizeMsg{Width: width, Height: height})

	m.notification.SetSize(width, height)

	return tea.Batch(cmds...)
}
