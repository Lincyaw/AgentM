package tui

import (
	"os"
	"time"

	"charm.land/bubbles/v2/help"
	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"

	"github.com/AoyangSpace/agentm-terminal/internal/tui/commands"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/completion"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/editor"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/dialog"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/internal/editorname"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/messages"
)

// Help returns help information for the status bar.
func (m *appModel) Help() help.KeyMap {
	return core.NewSimpleHelp(m.Bindings())
}

// AllBindings returns ALL available key bindings for the help dialog (comprehensive list).
func (m *appModel) AllBindings() []key.Binding {
	sendBinding := key.NewBinding(
		key.WithKeys("enter"),
		key.WithHelp("Enter", "send"),
	)
	interruptBinding := key.NewBinding(
		key.WithKeys("esc"),
		key.WithHelp("Esc", "interrupt"),
	)
	shortcutsBinding := key.NewBinding(
		key.WithKeys("?"),
		key.WithHelp("?", "shortcuts"),
	)
	commandsBinding := key.NewBinding(
		key.WithKeys("/"),
		key.WithHelp("/", "commands"),
	)
	filesBinding := key.NewBinding(
		key.WithKeys("@"),
		key.WithHelp("@", "resources"),
	)
	agentsBinding := key.NewBinding(
		key.WithKeys("left"),
		key.WithHelp("←", "agents"),
	)
	quitBinding := key.NewBinding(
		key.WithKeys("ctrl+c"),
		key.WithHelp("Ctrl+c", "quit"),
	)

	tabBinding := key.NewBinding(
		key.WithKeys("tab"),
		key.WithHelp("Tab", "switch focus"),
	)

	bindings := []key.Binding{
		sendBinding,
		interruptBinding,
		shortcutsBinding,
		commandsBinding,
		filesBinding,
		agentsBinding,
		quitBinding,
		tabBinding,
	}
	bindings = append(bindings, m.tabBar.Bindings()...)

	// Additional global shortcuts
	bindings = append(bindings,
		key.NewBinding(
			key.WithKeys("ctrl+t"),
			key.WithHelp("Ctrl+t", m.ctrlTActionLabel()),
		),
		key.NewBinding(
			key.WithKeys("ctrl+k"),
			key.WithHelp("Ctrl+k", "commands"),
		),
		key.NewBinding(
			key.WithKeys("ctrl+y"),
			key.WithHelp("Ctrl+y", "toggle yolo mode"),
		),
		key.NewBinding(
			key.WithKeys("ctrl+o"),
			key.WithHelp("Ctrl+o", "detailed transcript"),
		),
		key.NewBinding(
			key.WithKeys("ctrl+e"),
			key.WithHelp("Ctrl+e", "verbose transcript"),
		),
		key.NewBinding(
			key.WithKeys("ctrl+s"),
			key.WithHelp("Ctrl+s", "stash prompt"),
		),
		key.NewBinding(
			key.WithKeys("alt+p", "meta+p", "ctrl+m"),
			key.WithHelp("Opt+p", "model picker"),
		),
		key.NewBinding(
			key.WithKeys("alt+t", "meta+t"),
			key.WithHelp("Opt+t", "thinking mode"),
		),
		key.NewBinding(
			key.WithKeys("ctrl+z"),
			key.WithHelp("Ctrl+z", "suspend"),
		),
		key.NewBinding(
			key.WithKeys("shift+tab", "btab"),
			key.WithHelp("Shift+Tab", "cycle thinking level"),
		),
	)

	if !m.hideSidebar {
		bindings = append(bindings, key.NewBinding(
			key.WithKeys("ctrl+b"),
			key.WithHelp("Ctrl+b", "toggle sidebar"),
		))
	}

	// Show newline help based on keyboard enhancement support
	if m.keyboardEnhancementsSupported {
		bindings = append(bindings, key.NewBinding(
			key.WithKeys("shift+enter"),
			key.WithHelp("Shift+Enter", "newline"),
		))
	} else {
		bindings = append(bindings, key.NewBinding(
			key.WithKeys("ctrl+j"),
			key.WithHelp("Ctrl+j", "newline"),
		))
	}

	if m.focusedPanel == PanelContent {
		bindings = append(bindings, m.chatPage.Bindings()...)
	} else {
		editorName := editorname.FromEnv(os.Getenv("VISUAL"), os.Getenv("EDITOR"))
		bindings = append(bindings,
			key.NewBinding(
				key.WithKeys("ctrl+g"),
				key.WithHelp("Ctrl+g", "edit in "+editorName),
			),
			key.NewBinding(
				key.WithKeys("ctrl+r"),
				key.WithHelp("Ctrl+r", "history search"),
			),
		)
	}
	return bindings
}

// Bindings returns the key bindings shown in the status bar (a curated subset).
// This filters AllBindings() to show only the most essential commands.
func (m *appModel) Bindings() []key.Binding {
	all := m.AllBindings()

	// Define which keys should appear in the status bar
	statusBarKeys := map[string]bool{
		"enter":       true, // send
		"esc":         true, // interrupt
		"?":           true, // shortcuts
		"/":           true, // commands
		"@":           true, // files
		"left":        true, // agents
		"ctrl+c":      true, // quit
		"ctrl+k":      true, // commands
		"ctrl+t":      true, // contextual: new tab, tasks, or activity
		"ctrl+o":      true, // detailed transcript
		"ctrl+e":      true, // verbose transcript
		"shift+enter": true, // newline
		"ctrl+j":      true, // newline fallback
		"ctrl+g":      true, // edit in external editor (editor context)
		"ctrl+r":      true, // history search (editor context)
		// Content panel bindings (↑↓, c, e, d) are always included
		"up":   true,
		"down": true,
		"c":    true,
		"e":    true,
		"d":    true,
	}

	// Filter to only include status bar keys
	var filtered []key.Binding
	seen := make(map[string]bool, len(statusBarKeys))
	for _, binding := range all {
		if len(binding.Keys()) > 0 {
			bindingKey := binding.Keys()[0]
			if statusBarKeys[bindingKey] && !seen[bindingKey] {
				filtered = append(filtered, binding)
				seen[bindingKey] = true
			}
		}
	}

	return filtered
}

// handleKeyPress handles all keyboard input with proper priority routing.
func (m *appModel) handleKeyPress(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	// Check if we should stop transcription on Enter or Escape
	if m.transcriber.IsRunning() {
		switch msg.String() {
		case "enter":
			model, cmd := m.handleStopSpeak()
			sendCmd := m.editor.SendContent()
			return model, tea.Batch(cmd, sendCmd)

		case "esc":
			return m.handleStopSpeak()
		}
	}

	// Ctrl+c is intercepted before normal routing:
	//   - With no dialog open: show an inline second-press confirmation.
	//   - With another dialog open: keep the explicit confirmation dialog so
	//     the original modal state is not discarded by accident.
	//   - With the exit confirmation already on top: forward the key.
	if msg.String() == "ctrl+c" {
		if m.dialogMgr.TopIsExitConfirmation() {
			return m.forwardDialog(msg)
		}
		if m.agentsModeOpen && m.editor.Value() != "" {
			now := time.Now()
			m.focusedPanel = PanelEditor
			m.editor.SetValue("")
			m.editorLines = m.desiredEditorLines()
			m.lastExitRequest = now
			m.lastExitClearedInput = now
			m.statusBar.InvalidateCache()
			return m, tea.Batch(
				core.CmdHandler(completion.CloseMsg{}),
				m.editor.Focus(),
				m.resizeAll(),
				invalidateStatusBarAfter(2*time.Second),
			)
		}
		if m.editor.Value() != "" {
			now := time.Now()
			m.focusedPanel = PanelEditor
			if m.editor.ShellMode() {
				m.editor.SetShellValue("")
			} else {
				m.editor.SetValue("")
			}
			m.editorLines = m.desiredEditorLines()
			m.lastExitRequest = now
			m.lastExitClearedInput = now
			m.statusBar.InvalidateCache()
			return m, tea.Batch(
				core.CmdHandler(completion.CloseMsg{}),
				m.closeInlineSurfaces(),
				m.editor.Focus(),
				m.resizeAll(),
				invalidateStatusBarAfter(2*time.Second),
			)
		}
		if m.dialogMgr.Open() {
			return m, core.CmdHandler(dialog.OpenDialogMsg{
				Model: dialog.NewExitConfirmationDialog(),
			})
		}
		now := time.Now()
		if now.Sub(m.lastExitRequest) <= 2*time.Second {
			m.cleanupAll()
			return m, tea.Sequence(leaveFooterLineBeforeQuit(), tea.Quit)
		}
		m.lastExitRequest = now
		m.statusBar.InvalidateCache()
		return m, invalidateStatusBarAfter(2 * time.Second)
	}
	if !m.lastExitRequest.IsZero() {
		m.lastExitRequest = time.Time{}
		m.lastExitClearedInput = time.Time{}
		m.statusBar.InvalidateCache()
	}

	// Dialog gets priority when open, EXCEPT for background dialogs, which
	// let tab-navigation keys keep working so
	// the user can switch to another conversation while the prompt waits.
	if m.dialogMgr.Open() {
		if m.dialogMgr.TopIsBackground() && !m.editor.IsHistorySearchActive() {
			m.tabBar.SetCloseTabEnabled(true)
			if cmd := m.tabBar.Update(msg); cmd != nil {
				return m, cmd
			}
		}
		return m.forwardDialog(msg)
	}

	if m.workflowTaskPickerOpen {
		switch msg.String() {
		case "up":
			m.moveWorkflowTaskSelection(-1)
			return m, nil
		case "down":
			m.moveWorkflowTaskSelection(1)
			return m, nil
		case "enter":
			return m.activateWorkflowTaskSelection()
		case "esc":
			m.closeWorkflowTaskPicker()
			return m, m.resizeAll()
		case "x":
			return m.stopWorkflowTaskSelection()
		}
	}

	if m.backgroundActivityDetail {
		switch msg.String() {
		case "left":
			m.backgroundActivityDetail = false
			m.backgroundActivityPrompt = true
			m.statusBar.InvalidateCache()
			return m, m.resizeAll()
		case "esc", "enter", " ":
			m.backgroundActivityDetail = false
			m.backgroundActivityPrompt = false
			m.statusBar.InvalidateCache()
			return m, m.resizeAll()
		case "x":
			if activity, ok := m.selectedBackgroundShellActivity(); ok && m.application != nil {
				m.application.CancelBackground(backgroundShellTaskID(activity))
			}
			m.backgroundActivityDetail = false
			m.backgroundActivityPrompt = false
			m.statusBar.InvalidateCache()
			return m, m.resizeAll()
		}
	}

	if m.backgroundActivityPrompt {
		switch msg.String() {
		case "enter":
			if _, ok := m.selectedBackgroundShellActivity(); !ok {
				m.backgroundActivityPrompt = false
				m.statusBar.InvalidateCache()
				return m, m.resizeAll()
			}
			m.backgroundActivityPrompt = false
			m.backgroundActivityDetail = true
			m.statusBar.InvalidateCache()
			return m, m.resizeAll()
		case "esc":
			m.backgroundActivityPrompt = false
			m.statusBar.InvalidateCache()
			return m, m.resizeAll()
		}
	}

	if m.agentsModeOpen {
		if m.agentsModeRenaming {
			return m.handleAgentsModeRenameKey(msg)
		}
		switch msg.String() {
		case "?":
			return m, m.toggleAgentsModeHelp()
		case "up":
			return m, m.moveAgentsModeSelection(-1)
		case "down":
			return m, m.moveAgentsModeSelection(1)
		case "ctrl+x":
			return m.handleAgentsModeDelete()
		case "ctrl+s":
			return m, m.toggleAgentsModeGrouped()
		case "ctrl+t":
			return m, m.toggleAgentsModePinned()
		case "ctrl+r":
			return m, m.startAgentsModeRename()
		case "right":
			return m, m.closeAgentsMode()
		case "esc":
			return m.handleAgentsModeEscape()
		case "enter":
			return m.handleAgentsModeEnter()
		case " ", "space":
			return m.handleAgentsModeSpace(msg)
		}
		if index := parseAltNumberKey(msg); index >= 0 {
			return m.handleAgentsModeOpenAtIndex(index)
		}
		return m.forwardEditor(msg)
	}

	if m.focusedPanel == PanelEditor && !m.editor.IsHistorySearchActive() && m.editor.Value() != "" {
		if msg.String() == "ctrl+t" {
			return m, nil
		}
	}

	if msg.String() == "ctrl+t" && m.hasBottomActivityRows() {
		return m, m.toggleBottomActivityRows()
	}

	if m.shortcutSheetOpen {
		switch msg.String() {
		case "?", "esc":
			return m, m.closeInlineSurfaces()
		case "/", "@":
			if cmd := m.closeInlineSurfaces(); cmd != nil {
				return m, tea.Batch(cmd, m.updateEditorCmd(msg))
			}
		}
	}

	if m.localPanelOpen {
		return m.handleLocalPanelKey(msg)
	}

	// Tab bar keys (Ctrl+t, Ctrl+p, Ctrl+n, Ctrl+w) are suppressed during
	// history search so that ctrl+n/ctrl+p cycle through matches instead.
	// Ctrl+w (close tab) is disabled when the editor is focused so that the
	// standard "delete word" shortcut works while typing.
	if !m.editor.IsHistorySearchActive() {
		m.tabBar.SetCloseTabEnabled(m.focusedPanel != PanelEditor)
		if cmd := m.tabBar.Update(msg); cmd != nil {
			return m, cmd
		}
	}

	// Completion popup gets priority when open
	if m.completions.Open() {
		if msg.String() == "enter" {
			parser := commands.NewParser(m.commandCategories()...)
			input := m.editor.Value()
			if cmd := parser.Parse(input); cmd != nil {
				m.editor.SetValue("")
				m.editorLines = m.desiredEditorLines()
				m.statusBar.InvalidateCache()
				return m, tea.Batch(
					core.CmdHandler(completion.CloseMsg{}),
					m.resizeAll(),
					cmd,
				)
			}
			if time.Since(m.lastEditorValueChangeAt) < 120*time.Millisecond {
				if cmd := parser.ParseUnknown(input); cmd != nil {
					m.editor.SetValue("")
					m.editorLines = m.desiredEditorLines()
					m.statusBar.InvalidateCache()
					return m, tea.Batch(
						core.CmdHandler(completion.CloseMsg{}),
						m.resizeAll(),
						cmd,
					)
				}
			}
			if m.completions.HasSelection() {
				return m.forwardCompletions(msg)
			}
			if cmd := parser.ParseUnknown(input); cmd != nil {
				m.editor.SetValue("")
				m.editorLines = m.desiredEditorLines()
				m.statusBar.InvalidateCache()
				return m, tea.Batch(
					core.CmdHandler(completion.CloseMsg{}),
					m.resizeAll(),
					cmd,
				)
			}
		}
		completionKey := core.IsNavigationKey(msg) || msg.String() == "tab"
		if completionKey && (msg.String() != "enter" || m.completions.HasSelection()) {
			return m.forwardCompletions(msg)
		}
		// For all other keys (typing), send to both completion (for filtering) and editor
		return m, tea.Batch(m.updateCompletionsCmd(msg), m.updateEditorCmd(msg))
	}

	if m.focusedPanel == PanelEditor && !m.editor.IsHistorySearchActive() {
		if msg.String() == "up" && m.editor.Value() == "" && m.queuedInputCount > 0 {
			return m.forwardChat(messages.PopQueuedInputMsg{})
		}
		switch msg.String() {
		case "ctrl+b", "ctrl+f", "ctrl+e", "ctrl+k":
			if m.editor.Value() != "" {
				return m.forwardEditor(msg)
			}
		case "ctrl+y":
			if m.editor.HasKillBuffer() {
				return m.forwardEditor(msg)
			}
		}
	}

	// Global keyboard shortcuts (active even during history search)
	switch {
	case msg.String() == "ctrl+h" && m.focusedPanel == PanelEditor && m.editor.Value() == "":
		m.idleIDEContextExtraHidden = true
		m.statusBar.InvalidateCache()
		return m, m.resizeAll()

	case key.Matches(msg, key.NewBinding(key.WithKeys("f1", "ctrl+?"))) &&
		m.focusedPanel == PanelEditor && m.editor.Value() == "":
		m.idleIDEContextExtraHidden = true
		m.idleFooterRightHidden = true
		m.statusBar.InvalidateCache()
		return m, m.resizeAll()

	case key.Matches(msg, key.NewBinding(key.WithKeys("ctrl+k"))) &&
		m.focusedPanel == PanelEditor && !m.editor.ShellMode() && m.editor.Value() != "":
		m.lastIdleFocusWarningReveal = time.Time{}
		m.statusBar.InvalidateCache()
		return m, m.resizeAll()

	case key.Matches(msg, key.NewBinding(key.WithKeys("ctrl+k"))) &&
		m.focusedPanel == PanelEditor && m.editor.Value() == "":
		m.shortcutSheetDismissed = true
		m.lastIdleFocusWarningReveal = time.Now()
		m.statusBar.InvalidateCache()
		return m, tea.Batch(m.resizeAll(), invalidateStatusBarAfter(idleFocusWarningRevealTTL))

	case msg.String() == "?" && m.focusedPanel == PanelEditor && m.editor.Value() == "":
		return m, m.toggleShortcutSheet()

	case msg.String() == "down" && m.focusedPanel == PanelEditor && m.editor.Value() == "" && m.hasWorkflowTasks():
		m.openWorkflowTaskPicker()
		return m, m.resizeAll()

	case msg.String() == "down" && m.focusedPanel == PanelEditor && m.editor.Value() == "" && m.backgroundShellCount() > 0:
		m.backgroundActivityPrompt = true
		m.backgroundActivityDetail = false
		m.shortcutSheetOpen = false
		m.workflowTaskPickerOpen = false
		m.statusBar.InvalidateCache()
		return m, m.resizeAll()

	case msg.String() == "left" && m.focusedPanel == PanelEditor && m.editor.Value() == "":
		return m, m.openAgentsMode()

	case msg.String() == "x" && m.focusedPanel == PanelEditor && m.editor.Value() == "" && m.activeIsWorkflowTask():
		if m.supervisor == nil {
			return m, nil
		}
		return m.handleCloseTab(m.supervisor.ActiveID())

	case key.Matches(msg, key.NewBinding(key.WithKeys("ctrl+z"))):
		return m, tea.Suspend

	case key.Matches(msg, key.NewBinding(key.WithKeys("ctrl+y"))):
		return m, core.CmdHandler(messages.ToggleYoloMsg{})

	case key.Matches(msg, key.NewBinding(key.WithKeys("ctrl+o"))):
		return m, m.toggleTranscriptDetailed()

	case key.Matches(msg, key.NewBinding(key.WithKeys("ctrl+e"))):
		return m, m.toggleTranscriptVerbose()

	case key.Matches(msg, key.NewBinding(key.WithKeys("ctrl+s"))):
		return m.handleStashPrompt()

	case key.Matches(msg, key.NewBinding(key.WithKeys("alt+p", "meta+p", "ctrl+m"))):
		return m.handleOpenModelPicker(false)

	case key.Matches(msg, key.NewBinding(key.WithKeys("alt+t", "meta+t"))):
		return m.handleOpenThinkingToggle()
	}

	// History search is a modal state — capture all remaining keys before normal routing
	if m.focusedPanel == PanelEditor && m.editor.IsHistorySearchActive() {
		if msg.String() == "enter" {
			cmd := m.updateEditorCmd(msg)
			sendCmd := m.editor.SendContent()
			m.editorLines = m.desiredEditorLines()
			m.statusBar.InvalidateCache()
			return m, tea.Batch(cmd, sendCmd, m.resizeAll())
		}
		return m.forwardEditor(msg)
	}

	switch {
	case key.Matches(msg, key.NewBinding(key.WithKeys("ctrl+g"))):
		return m.openExternalEditor()

	case key.Matches(msg, key.NewBinding(key.WithKeys("ctrl+r"))):
		if m.focusedPanel == PanelEditor && !m.editor.IsRecording() {
			model, cmd := m.editor.EnterHistorySearch()
			m.editor = model.(editor.Editor)
			return m, cmd
		}

	// Toggle sidebar (propagates to content view regardless of focus)
	case key.Matches(msg, key.NewBinding(key.WithKeys("ctrl+b"))):
		if m.hideSidebar {
			return m, nil
		}
		return m.forwardChat(msg)

	// Shift+Tab follows Claude Code's permission-mode footer cycle.
	case key.Matches(msg, key.NewBinding(key.WithKeys("shift+tab", "btab"))):
		return m.handleCyclePermissionMode()

	// Plain Tab should not steal editor focus. Claude Code keeps the composer
	// active, while completion popups handle Tab earlier in this function.
	case key.Matches(msg, key.NewBinding(key.WithKeys("tab"))):
		if cmd := m.editor.AcceptSuggestion(); cmd != nil {
			return m, cmd
		}
		if m.focusedPanel == PanelContent {
			m.focusedPanel = PanelEditor
			m.statusBar.InvalidateCache()
			m.chatPage.BlurMessages()
			return m, m.editor.Focus()
		}
		return m, nil

	// Esc: interrupt/cancel. Sending while the agent is busy is handled by
	// Enter via QueueIfBusy, so Esc never silently submits editor content.
	case key.Matches(msg, key.NewBinding(key.WithKeys("esc"))):
		if m.focusedPanel == PanelEditor && m.editor.ShellMode() && m.editor.Value() == "" {
			m.editor.SetValue("")
			m.statusBar.InvalidateCache()
			return m, nil
		}
		if cmd := m.closeInlineSurfaces(); cmd != nil {
			return m, cmd
		}
		if m.focusedPanel == PanelEditor && m.editor.Value() != "" && !m.chatPage.IsWorking() {
			now := time.Now()
			if now.Sub(m.lastEscClearRequest) <= 2*time.Second {
				if m.editor.ShellMode() {
					m.editor.SetShellValue("")
				} else {
					m.editor.SetValue("")
				}
				m.editorLines = m.desiredEditorLines()
				m.lastEscClearRequest = time.Time{}
				m.lastEscClearedInput = now
				return m, m.resizeAll()
			}
			m.lastEscClearRequest = now
			m.statusBar.InvalidateCache()
			return m, invalidateStatusBarAfter(2 * time.Second)
		}
		if m.focusedPanel == PanelEditor && m.editor.Value() != "" && m.chatPage.IsWorking() {
			m.lastEscClearRequest = time.Time{}
			return m.forwardChat(messages.CancelStreamPreserveInputMsg{})
		}
		m.lastEscClearRequest = time.Time{}
		return m.forwardChat(msg)

	default:
		// Handle ctrl+1 through ctrl+9 for quick agent switching
		if index := parseCtrlNumberKey(msg); index >= 0 {
			return m.handleSwitchToAgentByIndex(index)
		}
	}

	if m.transcriptDetailed {
		return m.forwardChat(msg)
	}

	// Focus-based routing
	switch m.focusedPanel {
	case PanelEditor:
		return m.forwardEditor(msg)
	case PanelContent:
		if shouldReturnToEditorForTextInput(msg) {
			m.focusedPanel = PanelEditor
			return m, tea.Batch(m.editor.Focus(), m.updateEditorCmd(msg), m.resizeAll())
		}
		return m.forwardChat(msg)
	}

	return m, nil
}

func shouldReturnToEditorForTextInput(msg tea.KeyPressMsg) bool {
	return msg.Key().Text != ""
}

func invalidateStatusBarAfter(delay time.Duration) tea.Cmd {
	return tea.Tick(delay, func(time.Time) tea.Msg {
		return messages.InvalidateStatusBarMsg{}
	})
}

func leaveFooterLineBeforeQuit() tea.Cmd {
	return func() tea.Msg {
		_, _ = os.Stdout.Write([]byte("\n"))
		return nil
	}
}

// parseCtrlNumberKey checks if msg is ctrl+1 through ctrl+9 and returns the index (0-8), or -1 if not matched
func parseCtrlNumberKey(msg tea.KeyPressMsg) int {
	s := msg.String()
	if len(s) == 6 && s[:5] == "ctrl+" && s[5] >= '1' && s[5] <= '9' {
		return int(s[5] - '1')
	}
	return -1
}

func parseAltNumberKey(msg tea.KeyPressMsg) int {
	s := msg.String()
	if len(s) == 5 && s[:4] == "alt+" && s[4] >= '1' && s[4] <= '9' {
		return int(s[4] - '1')
	}
	if len(s) == 6 && s[:5] == "meta+" && s[5] >= '1' && s[5] <= '9' {
		return int(s[5] - '1')
	}
	return -1
}

// switchFocus toggles between content and editor panels.
func (m *appModel) switchFocus() (tea.Model, tea.Cmd) {
	switch m.focusedPanel {
	case PanelEditor:
		// Check if editor has a suggestion to accept first
		if cmd := m.editor.AcceptSuggestion(); cmd != nil {
			return m, cmd
		}
		m.focusedPanel = PanelContent
		m.statusBar.InvalidateCache()
		m.editor.Blur()
		return m, m.chatPage.FocusMessages()
	case PanelContent:
		m.focusedPanel = PanelEditor
		m.statusBar.InvalidateCache()
		m.chatPage.BlurMessages()
		return m, m.editor.Focus()
	}
	return m, nil
}
