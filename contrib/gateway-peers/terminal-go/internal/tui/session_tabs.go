package tui

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"time"

	tea "charm.land/bubbletea/v2"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/runtime"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/session"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/notification"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/spinner"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/dialog"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/messages"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/page/chat"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
)

// handleOpenSessionBrowser opens the session browser dialog.
func (m *appModel) handleOpenSessionBrowser() (tea.Model, tea.Cmd) {
	store := m.application.SessionStore()
	if store == nil {
		// Gateway mode: send bare /resume to fetch the session list from the
		// gateway. The gateway returns a session_list outbound that the
		// translator converts into OpenSessionBrowserWithDataMsg.
		return m, core.CmdHandler(messages.SendMsg{Content: "/resume", BypassQueue: true})
	}

	sessions, err := store.GetSessionSummaries(context.Background())
	if err != nil {
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to load sessions: %v", err))
	}
	if len(sessions) == 0 {
		return m, notification.InfoCmd("No previous sessions found")
	}

	return m, core.CmdHandler(dialog.OpenDialogMsg{
		Model: dialog.NewSessionBrowserDialog(sessions),
	})
}

// handleOpenSessionBrowserWithData opens the session browser with pre-fetched
// session data from the gateway (no local store needed).
func (m *appModel) handleOpenSessionBrowserWithData(sessions []session.Summary) (tea.Model, tea.Cmd) {
	if len(sessions) == 0 {
		return m, notification.InfoCmd("No previous sessions found")
	}
	return m, tea.Batch(
		core.CmdHandler(dialog.OpenDialogMsg{
			Model: dialog.NewSessionBrowserDialog(sessions),
		}),
		m.settleSessionBrowserControlCmd(),
	)
}

func (m *appModel) settleSessionBrowserControlCmd() tea.Cmd {
	sessionID := ""
	if sess := m.application.Session(); sess != nil {
		sessionID = sess.ID
	}
	agentName := ""
	if m.sessionState != nil {
		agentName = m.sessionState.CurrentAgentName()
	}
	return tea.Tick(50*time.Millisecond, func(time.Time) tea.Msg {
		return runtime.StreamStopped(sessionID, agentName, "command")
	})
}

// handleLoadSession loads a saved session into the current tab (if empty) or a new tab.
func (m *appModel) handleLoadSession(sessionID string) (tea.Model, tea.Cmd) {
	store := m.application.SessionStore()
	if store == nil {
		// Gateway mode: send /resume <id> to switch sessions server-side.
		return m, core.CmdHandler(messages.SendMsg{
			Content:     "/resume " + sessionID,
			BypassQueue: true,
		})
	}

	sess, err := store.GetSession(context.Background(), sessionID)
	if err != nil {
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to load session: %v", err))
	}

	// Check if this session is already open in another tab — switch instead of duplicating.
	if tabID := m.findTabByPersistedID(sessionID); tabID != "" {
		return m.handleSwitchTab(tabID)
	}

	// Determine working directory from the loaded session.
	workingDir := sess.WorkingDir
	if workingDir == "" {
		workingDir = m.application.Session().WorkingDir
	}
	ctx := context.Background()

	// If the current session is empty (no messages, no title — the default state
	// when opening the TUI or creating a new tab), replace it in-place instead of
	// spawning yet another tab.
	currentSess := m.application.Session()
	if len(currentSess.Messages) == 0 && currentSess.Title == "" {
		activeID := m.supervisor.ActiveID()
		oldPersistedID := m.persistedSessionID(activeID)

		model, cmd := m.replaceActiveSession(ctx, sess)

		// Update tuistate: replace old persisted ID with the loaded session's ID
		if m.tuiStore != nil {
			if err := m.tuiStore.UpdateTabSessionID(ctx, oldPersistedID, sess.ID); err != nil {
				slog.WarnContext(ctx, "Failed to update tab session ID after in-place load", "error", err)
			}
			if sess.WorkingDir != "" {
				if err := m.tuiStore.UpdateTabWorkingDir(ctx, sess.ID, sess.WorkingDir); err != nil {
					slog.WarnContext(ctx, "Failed to update tab working dir after in-place load", "error", err)
				}
			}
		}
		m.persistActiveTab(sess.ID)
		return model, cmd
	}

	slog.DebugContext(ctx, "Loading session into new tab", "session_id", sessionID)

	// Spawn a new tab.
	newSessionID, err := m.supervisor.SpawnSession(ctx, workingDir)
	if err != nil {
		return m, notification.ErrorCmd("Failed to create tab: " + err.Error())
	}

	// Persist the new tab using the loaded session's persisted ID (not the ephemeral tab ID).
	if m.tuiStore != nil {
		if err := m.tuiStore.AddTab(ctx, sess.ID, workingDir); err != nil {
			slog.WarnContext(ctx, "Failed to persist loaded session tab", "error", err)
		}
	}

	// Switch to the new tab so m.application points to the new app.
	model, switchCmd := m.handleSwitchTab(newSessionID)

	// Replace the blank session with the loaded one and rebuild all components.
	m.application.ReplaceSession(ctx, sess)
	m.initSessionComponents(newSessionID, m.application, sess)

	if sess.Title != "" {
		m.supervisor.SetRunnerTitle(newSessionID, sess.Title)
	}

	m.persistActiveTab(sess.ID)

	return model, tea.Batch(
		switchCmd,
		m.initAndFocusComponents(),
	)
}

// replaceActiveSession replaces the current (empty) tab's session with a loaded one in-place.
// If the loaded session's working directory differs from the runner's current one,
// a fresh runtime is spawned via the supervisor so that tools operate in the correct directory.
func (m *appModel) replaceActiveSession(ctx context.Context, sess *session.Session) (tea.Model, tea.Cmd) {
	activeID := m.supervisor.ActiveID()

	slog.DebugContext(ctx, "Replacing empty session in-place", "tab_id", activeID, "loaded_session", sess.ID)

	// Cleanup old editor for the active session
	if ed, ok := m.editors[activeID]; ok {
		ed.Cleanup()
	}

	// If the loaded session's working directory differs from the runner's,
	// we need a fresh runtime whose tools operate in the correct directory.
	runner := m.supervisor.GetRunner(activeID)
	sessWorkingDir := sess.WorkingDir
	if sessWorkingDir != "" && runner != nil && sessWorkingDir != runner.WorkingDir {
		newApp, _, spawnCleanup, err := m.supervisor.Spawner()(ctx, sessWorkingDir)
		if err == nil {
			slog.DebugContext(ctx, "Respawning runtime for working dir mismatch",
				"tab_id", activeID,
				"old_dir", runner.WorkingDir,
				"new_dir", sessWorkingDir)
			m.supervisor.ReplaceRunnerApp(ctx, activeID, newApp, sessWorkingDir, spawnCleanup)
			m.application = newApp
		} else {
			slog.WarnContext(ctx, "Failed to respawn runtime for working dir, using existing",
				"working_dir", sessWorkingDir, "error", err)
		}
	}

	// Replace the session in the app and rebuild all per-session components.
	m.application.ReplaceSession(ctx, sess)
	m.initSessionComponents(activeID, m.application, sess)

	if sess.Title != "" {
		m.supervisor.SetRunnerTitle(activeID, sess.Title)
	}

	cmd := m.initAndFocusComponents()
	return m, cmd
}

// handleClearSession resets the current tab by creating a fresh session
// in the same working directory.
func (m *appModel) handleClearSession() (tea.Model, tea.Cmd) {
	activeID := m.supervisor.ActiveID()

	// Cleanup old editor for the active session.
	if ed, ok := m.editors[activeID]; ok {
		ed.Cleanup()
	}

	// Create a fresh visible session in the same app, preserving the working
	// dir. The gateway still starts a fresh backend session, but /clear should
	// not expose its /new implementation detail in the transcript.
	oldSess := m.application.Session()
	workingDir := ""
	toolsApproved := false
	if oldSess != nil {
		workingDir = oldSess.WorkingDir
		toolsApproved = oldSess.ToolsApproved
	}
	m.application.ClearSession()
	m.suppressClearNoticeUntil = time.Now().Add(2 * time.Second)
	newSess := session.New(session.WithWorkingDir(workingDir), session.WithToolsApproved(toolsApproved))
	m.application.SetSession(newSess)

	// Rebuild all per-session UI components.
	m.initSessionComponents(activeID, m.application, newSess)
	m.dialogMgr = dialog.New()
	m.localPanelOpen = false
	m.localPanelCommand = ""
	m.localPanelDismissNotice = ""
	m.localSettingsTab = settingsTabStatus
	m.localSettingsBodyFocused = false
	m.localConfigSelected = false
	m.localConfigSearch = ""
	m.localHelpTab = helpTabGeneral
	m.localPermissionsTab = permissionsTabAllow
	m.localPermissionsSelected = false
	m.localPermissionsMode = permissionsModeList
	m.localPermissionRuleInput = ""
	m.localPermissionRuleDraft = ""
	m.localPermissionSaveIndex = 0
	m.supervisor.SetRunnerTitle(activeID, "")
	m.sessionState.SetSessionTitle("")
	m.sessionState.SetPreviousMessage(nil)
	m.clearBottomActivitiesForSession(activeID)

	// Update persisted tab to point to the new session.
	if m.tuiStore != nil {
		ctx := context.Background()
		oldPersistedID := m.persistedSessionID(activeID)
		if err := m.tuiStore.UpdateTabSessionID(ctx, oldPersistedID, newSess.ID); err != nil {
			slog.WarnContext(ctx, "Failed to update tab session ID after clear", "error", err)
		}
	}
	m.persistActiveTab(newSess.ID)

	m.reapplyKeyboardEnhancements()
	clearCmd := m.chatPage.AddLocalUserMessage("/clear")

	return m, tea.Sequence(
		m.chatPage.Init(),
		clearCmd,
		m.resizeAll(),
		m.editor.Focus(),
	)
}

// handleSpawnSession spawns a new session.
func (m *appModel) handleSpawnSession(workingDir string, background bool) (tea.Model, tea.Cmd) {
	// If no working dir specified, open the picker
	if workingDir == "" {
		return m.openWorkingDirPicker()
	}

	// Spawn the new session
	ctx := context.Background()
	sessionID, err := m.supervisor.SpawnSession(ctx, workingDir)
	if err != nil {
		return m, notification.ErrorCmd("Failed to spawn session: " + err.Error())
	}

	// Persist the new tab (for new tabs, persisted ID == runtime tab ID).
	if m.tuiStore != nil {
		if err := m.tuiStore.AddTab(ctx, sessionID, workingDir); err != nil {
			slog.WarnContext(ctx, "Failed to persist new tab", "error", err)
		}
	}

	if background {
		m.supervisor.SetBackground(sessionID, true)
		m.markWorkflowSession(sessionID)
		m.setWorkflowVisible(sessionID, true)

		var cmds []tea.Cmd
		if _, exists := m.chatPages[sessionID]; !exists {
			if runner := m.supervisor.GetRunner(sessionID); runner != nil {
				cp, _, ed := m.createSessionComponents(sessionID, runner.App, runner.App.Session())
				if cmd := cp.Init(); cmd != nil {
					cmds = append(cmds, cmd)
				}
				if cmd := ed.Init(); cmd != nil {
					cmds = append(cmds, cmd)
				}
			}
		}

		tabs, activeIdx := m.supervisor.GetTabs()
		if m.syncTabChrome(tabs, activeIdx) {
			cmds = append(cmds, m.resizeAll())
		}
		return m, tea.Batch(cmds...)
	}

	// Switch to the new session
	return m.handleSwitchTab(sessionID)
}

// openWorkingDirPicker opens the working directory picker dialog.
func (m *appModel) openWorkingDirPicker() (tea.Model, tea.Cmd) {
	var recentDirs, favoriteDirs []string
	var warningCmds []tea.Cmd
	if m.tuiStore != nil {
		var err error
		recentDirs, err = m.tuiStore.GetRecentDirs(context.Background(), 10)
		if err != nil {
			slog.Warn("Failed to load recent working directories", "error", err)
			warningCmds = append(warningCmds, notification.WarningCmd("Recent working dirs unavailable."))
		}
		favoriteDirs, err = m.tuiStore.GetFavoriteDirs(context.Background())
		if err != nil {
			slog.Warn("Failed to load favorite working directories", "error", err)
			warningCmds = append(warningCmds, notification.WarningCmd("Favorite working dirs unavailable."))
		}
	}

	// Use the active session's working directory so the picker reflects it
	// instead of the process CWD.
	var sessionWorkingDir string
	if runner := m.supervisor.GetRunner(m.supervisor.ActiveID()); runner != nil {
		sessionWorkingDir = runner.WorkingDir
	}

	openCmd := core.CmdHandler(dialog.OpenDialogMsg{
		Model: dialog.NewWorkingDirPickerDialog(recentDirs, favoriteDirs, m.tuiStore, sessionWorkingDir),
	})
	if len(warningCmds) == 0 {
		return m, openCmd
	}
	return m, tea.Batch(append(warningCmds, openCmd)...)
}

// stashedDialog holds a background dialog instance that was on screen when
// the user navigated away from a tab, paired with the runtime event that
// caused it to open. The event is used as an identity check on return: if
// the supervisor's pending event for the tab no longer matches, the agent
// has superseded the prompt and we discard the stash in favour of building
// a fresh dialog from the new event.
type stashedDialog struct {
	dialog dialog.Dialog
	event  tea.Msg
}

// handleSwitchTab switches to a different session.
// Existing chat pages and editors are preserved (not recreated) so that in-flight streaming
// content and draft text are retained when switching back to a tab.
func (m *appModel) handleSwitchTab(sessionID string) (tea.Model, tea.Cmd) {
	// If a background dialog is open on the
	// outgoing tab, capture both its originating event and the live dialog
	// instance before the supervisor flips activeID. We only commit the
	// re-stash after SwitchTo succeeds — otherwise a failed switch would
	// leave the supervisor with a stale pending event and the dialog still
	// on screen.
	//
	// Stashing the dialog instance (rather than rebuilding it from the event
	// on return) preserves any in-progress input the user typed. See issue #2770.
	var (
		backgroundEvent  tea.Msg
		backgroundDialog dialog.Dialog
		outgoingTabID    string
	)
	if m.dialogMgr.Open() && m.dialogMgr.TopIsBackground() {
		backgroundEvent = m.dialogMgr.TopBackgroundEvent()
		backgroundDialog = m.dialogMgr.TopDialog()
		outgoingTabID = m.supervisor.ActiveID()
	}

	runner, wasBackground := m.supervisor.SwitchTo(sessionID)
	if runner == nil {
		return m, notification.ErrorCmd("Session not found")
	}
	m.branchLabel = ""
	if wasBackground {
		m.setWorkflowVisible(sessionID, false)
	} else {
		m.mainSessionID = sessionID
	}

	// Now that the switch is committed, finalize the dialog hand-off.
	var closeBackgroundDialogCmd tea.Cmd
	if backgroundEvent != nil && outgoingTabID != "" && outgoingTabID != sessionID {
		m.supervisor.SetPendingEvent(outgoingTabID, backgroundEvent)
		if backgroundDialog != nil {
			m.stashedDialogs[outgoingTabID] = stashedDialog{
				dialog: backgroundDialog,
				event:  backgroundEvent,
			}
		}
		closeBackgroundDialogCmd = core.CmdHandler(dialog.CloseDialogMsg{})
	}

	// Blur current editor before switching
	m.editor.Blur()

	// If this tab has a pending session restore, load it through
	// replaceActiveSession — the same code path as the /sessions command.
	if oldSessionID, ok := m.pendingRestores[sessionID]; ok {
		delete(m.pendingRestores, sessionID)
		m.application = runner.App
		if store := runner.App.SessionStore(); store != nil {
			if sess, err := store.GetSession(context.Background(), oldSessionID); err == nil {
				m.persistActiveTab(sess.ID)
				model, cmd := m.replaceActiveSession(context.Background(), sess)

				if m.tuiStore != nil && sess.WorkingDir != "" {
					if err := m.tuiStore.UpdateTabWorkingDir(context.Background(), oldSessionID, sess.WorkingDir); err != nil {
						slog.Warn("Failed to update persisted working dir", "error", err)
					}
				}

				cmd = tea.Batch(cmd, m.applySidebarCollapsed(sessionID), closeBackgroundDialogCmd)
				return model, cmd
			}
		}
		// Fall through to normal tab switch if session couldn't be loaded.
	}

	// Get or create per-session components.
	_, pageExists := m.chatPages[sessionID]
	_, editorExists := m.editors[sessionID]

	if !pageExists || !editorExists {
		// Create all missing components at once.
		m.initSessionComponents(sessionID, runner.App, runner.App.Session())
		m.applySidebarCollapsed(sessionID)
	} else {
		// Reuse existing components — just update convenience pointers.
		m.application = runner.App
		m.sessionState = m.sessionStates[sessionID]
		m.chatPage = m.chatPages[sessionID]
		m.history = m.histories[sessionID]
		m.editor = m.editors[sessionID]
	}

	m.reapplyKeyboardEnhancements()
	var refreshCmd tea.Cmd
	refreshCmd = m.refreshCommandInputs()
	m.persistActiveTab(m.persistedSessionID(sessionID))

	// Sync editor working state and reset working spinner.
	m.editor.SetWorking(m.chatPage.IsWorking())
	m.workingSpinner.Stop()
	m.workingSpinner = spinner.New(spinner.ModeSpinnerOnly, styles.SpinnerDotsHighlightStyle)

	var cmds []tea.Cmd

	if !pageExists || !editorExists {
		if !pageExists {
			cmds = append(cmds, m.chatPage.Init())
		}
		if !editorExists {
			cmds = append(cmds, m.editor.Init())
		}
		cmds = append(cmds, m.editor.Focus(), m.resizeAll())
	} else {
		cmds = append(cmds, m.resizeAll(), m.chatPage.ScrollToBottom(), m.editor.Focus())
	}

	if m.chatPage.IsWorking() {
		cmds = append(cmds, m.workingSpinner.Init())
	}
	if pendingCmd := m.replayPendingEvent(sessionID); pendingCmd != nil {
		cmds = append(cmds, pendingCmd)
	}
	if closeBackgroundDialogCmd != nil {
		cmds = append(cmds, closeBackgroundDialogCmd)
	}
	if refreshCmd != nil {
		cmds = append(cmds, refreshCmd)
	}

	return m, tea.Batch(cmds...)
}

// applySidebarCollapsed applies and consumes the persisted sidebar collapsed state
// for the given tab ID. Returns a resize command if the state was applied, nil otherwise.
func (m *appModel) applySidebarCollapsed(sessionID string) tea.Cmd {
	collapsed, ok := m.pendingSidebarCollapsed[sessionID]
	if !ok {
		return nil
	}
	m.chatPage.SetSidebarSettings(chat.SidebarSettings{Collapsed: collapsed})
	delete(m.pendingSidebarCollapsed, sessionID)
	return m.resizeAll()
}

// replayPendingEvent checks if a session has a pending attention event (e.g. tool confirmation
// or max iterations) that was received while the tab was inactive.
// If found, it opens the appropriate dialog. The event was already processed by the chat page
// (updating the message list), but the dialog command was discarded for inactive sessions.
//
// If a stashed dialog instance is available for this session and its
// associated event still matches the pending one, the same instance is
// re-opened so any in-progress input survives the round trip (issue #2770).
// Otherwise the stash is discarded and a fresh dialog is built.
func (m *appModel) replayPendingEvent(sessionID string) tea.Cmd {
	pendingEvent := m.supervisor.ConsumePendingEvent(sessionID)
	if pendingEvent == nil {
		// No pending event: any stash is stale (e.g. the agent finished).
		delete(m.stashedDialogs, sessionID)
		return nil
	}

	sessionState, ok := m.sessionStates[sessionID]
	if !ok {
		delete(m.stashedDialogs, sessionID)
		return nil
	}

	// If we stashed the live dialog instance when leaving this tab and the
	// pending event hasn't changed, re-open the same instance so the user's
	// in-progress input is preserved.
	if stash, ok := m.stashedDialogs[sessionID]; ok {
		delete(m.stashedDialogs, sessionID)
		if stash.event == pendingEvent && stash.dialog != nil {
			return core.CmdHandler(dialog.OpenDialogMsg{
				Model:            stash.dialog,
				OriginatingEvent: pendingEvent,
			})
		}
	}

	switch ev := pendingEvent.(type) {
	case *runtime.ToolCallConfirmationEvent:
		return core.CmdHandler(dialog.OpenDialogMsg{
			Model:            dialog.NewToolConfirmationDialog(ev, sessionState),
			OriginatingEvent: ev,
		})

	case *runtime.MaxIterationsReachedEvent:
		return core.CmdHandler(dialog.OpenDialogMsg{
			Model:            dialog.NewMaxIterationsDialog(ev.MaxIterations, m.application),
			OriginatingEvent: ev,
		})

	}

	return nil
}

// handleReorderTab moves a tab from one position to another.
func (m *appModel) handleReorderTab(msg messages.ReorderTabMsg) (tea.Model, tea.Cmd) {
	m.supervisor.ReorderTab(msg.FromIdx, msg.ToIdx)

	if m.tuiStore != nil {
		tabs, _ := m.supervisor.GetTabs()
		ids := make([]string, len(tabs))
		for i, tab := range tabs {
			ids[i] = m.persistedSessionID(tab.SessionID)
		}
		if err := m.tuiStore.ReorderTab(context.Background(), ids); err != nil {
			slog.Warn("Failed to persist tab reorder", "error", err)
		}
	}

	return m, nil
}

// handleCloseTab closes a session tab.
func (m *appModel) handleCloseTab(sessionID string) (tea.Model, tea.Cmd) {
	wasActive := sessionID == m.supervisor.ActiveID()

	// Capture the working dir before closing so we can reuse it if this is the last tab.
	var closedWorkingDir string
	if runner := m.supervisor.GetRunner(sessionID); runner != nil {
		closedWorkingDir = runner.WorkingDir
	}

	// Compute persisted session-store ID *before* closing (runner goes away).
	persistedID := m.persistedSessionID(sessionID)

	nextActiveID := m.supervisor.CloseSession(sessionID)
	if sessionID == m.mainSessionID {
		m.mainSessionID = m.chooseMainSessionID(nextActiveID)
	}

	// Clean up per-session state
	delete(m.chatPages, sessionID)
	if ed, ok := m.editors[sessionID]; ok {
		ed.Cleanup()
		delete(m.editors, sessionID)
	}
	delete(m.sessionStates, sessionID)
	delete(m.histories, sessionID)
	delete(m.pendingRestores, sessionID)
	delete(m.pendingSidebarCollapsed, sessionID)
	delete(m.stashedDialogs, sessionID)
	m.clearBottomActivitiesForSession(sessionID)

	var cmds []tea.Cmd
	// Remove from persistent store using the persisted session-store ID.
	if m.tuiStore != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
		defer cancel()
		if err := m.tuiStore.RemoveTab(ctx, persistedID); err != nil {
			slog.ErrorContext(ctx, "Failed to remove tab from store", "error", err)
			cmds = append(cmds, notification.ErrorCmd(fmt.Sprintf("Failed to remove tab from tui state db: %v", err)))
		}
	}

	// If we closed all tabs, spawn a new one reusing the previous working dir.
	// We always provide a concrete dir to avoid showing the picker — pressing Esc
	// in the picker with zero tabs would leave the TUI in a broken state.
	if m.supervisor.Count() == 0 {
		workingDir := closedWorkingDir
		if workingDir == "" {
			workingDir, _ = os.Getwd()
		}
		if workingDir == "" {
			workingDir = "/"
		}
		return m.handleSpawnSession(workingDir, false)
	}

	// If the closed tab was active, switch to the next one
	if wasActive && nextActiveID != "" {
		return m.handleSwitchTab(nextActiveID)
	}
	cmds = append(cmds, m.resizeAllIfBottomSurfaceChanged())

	return m, tea.Batch(cmds...)
}
