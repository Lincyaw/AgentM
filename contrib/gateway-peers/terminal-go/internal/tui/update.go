package tui

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	tea "charm.land/bubbletea/v2"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/runtime"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/animation"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/completion"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/notification"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/spinner"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/dialog"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/internal/termfeatures"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/messages"
)

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
