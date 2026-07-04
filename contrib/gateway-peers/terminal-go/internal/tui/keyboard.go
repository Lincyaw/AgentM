package tui

import (
	"os"
	"time"

	"charm.land/bubbles/v2/help"
	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"

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
			key.WithKeys("ctrl+h"),
			key.WithHelp("Ctrl+h", "help"),
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
			key.WithHelp("Ctrl+s", "cycle agent"),
		),
		key.NewBinding(
			key.WithKeys("ctrl+m"),
			key.WithHelp("Ctrl+m", "model picker"),
		),
		key.NewBinding(
			key.WithKeys("ctrl+z"),
			key.WithHelp("Ctrl+z", "suspend"),
		),
		key.NewBinding(
			key.WithKeys("shift+tab"),
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
		if m.dialogMgr.Open() {
			return m, core.CmdHandler(dialog.OpenDialogMsg{
				Model: dialog.NewExitConfirmationDialog(),
			})
		}
		now := time.Now()
		if now.Sub(m.lastExitRequest) <= 2*time.Second {
			m.cleanupAll()
			return m, tea.Quit
		}
		m.lastExitRequest = now
		m.statusBar.InvalidateCache()
		return m, nil
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
		if core.IsNavigationKey(msg) {
			return m.forwardCompletions(msg)
		}
		// For all other keys (typing), send to both completion (for filtering) and editor
		return m, tea.Batch(m.updateCompletionsCmd(msg), m.updateEditorCmd(msg))
	}

	// Global keyboard shortcuts (active even during history search)
	switch {
	case msg.String() == "?" && m.focusedPanel == PanelEditor && m.editor.Value() == "":
		return m, m.toggleShortcutSheet()

	case msg.String() == "down" && m.focusedPanel == PanelEditor && m.editor.Value() == "" && m.hasWorkflowTasks():
		m.openWorkflowTaskPicker()
		return m, m.resizeAll()

	case msg.String() == "left" && m.focusedPanel == PanelEditor && m.editor.Value() == "":
		return m.handleCycleAgent()

	case key.Matches(msg, key.NewBinding(key.WithKeys("ctrl+z"))):
		return m, tea.Suspend

	case key.Matches(msg, key.NewBinding(key.WithKeys("ctrl+k"))):
		categories := m.commandCategories()
		return m, core.CmdHandler(dialog.OpenDialogMsg{
			Model: dialog.NewCommandPaletteDialog(categories),
		})

	case key.Matches(msg, key.NewBinding(key.WithKeys("ctrl+y"))):
		return m, core.CmdHandler(messages.ToggleYoloMsg{})

	case key.Matches(msg, key.NewBinding(key.WithKeys("ctrl+o"))):
		return m, m.toggleTranscriptDetailed()

	case key.Matches(msg, key.NewBinding(key.WithKeys("ctrl+e"))):
		return m, m.toggleTranscriptVerbose()

	case key.Matches(msg, key.NewBinding(key.WithKeys("ctrl+s"))):
		return m.handleCycleAgent()

	case key.Matches(msg, key.NewBinding(key.WithKeys("ctrl+m"))):
		return m.handleOpenModelPicker()

	case key.Matches(msg, key.NewBinding(key.WithKeys("ctrl+h", "f1", "ctrl+?"))):
		return m, m.toggleShortcutSheet()
	}

	// History search is a modal state — capture all remaining keys before normal routing
	if m.focusedPanel == PanelEditor && m.editor.IsHistorySearchActive() {
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

	// Shift+Tab cycles the current model's thinking-effort level
	case key.Matches(msg, key.NewBinding(key.WithKeys("shift+tab"))):
		return m.handleCycleThinkingLevel()

	// Focus switching: Tab key toggles between content and editor
	case key.Matches(msg, key.NewBinding(key.WithKeys("tab"))):
		return m.switchFocus()

	// Esc: interrupt/cancel. Sending while the agent is busy is handled by
	// Enter via QueueIfBusy, so Esc never silently submits editor content.
	case key.Matches(msg, key.NewBinding(key.WithKeys("esc"))):
		if cmd := m.closeInlineSurfaces(); cmd != nil {
			return m, cmd
		}
		if m.focusedPanel == PanelEditor && m.editor.Value() != "" && !m.chatPage.IsWorking() {
			now := time.Now()
			if now.Sub(m.lastEscClearRequest) <= 2*time.Second {
				m.editor.SetValue("")
				m.lastEscClearRequest = time.Time{}
				return m, nil
			}
			m.lastEscClearRequest = now
			m.statusBar.InvalidateCache()
			return m, nil
		}
		m.lastEscClearRequest = time.Time{}
		return m.forwardChat(msg)

	default:
		// Handle ctrl+1 through ctrl+9 for quick agent switching
		if index := parseCtrlNumberKey(msg); index >= 0 {
			return m.handleSwitchToAgentByIndex(index)
		}
	}

	// Focus-based routing
	switch m.focusedPanel {
	case PanelEditor:
		return m.forwardEditor(msg)
	case PanelContent:
		return m.forwardChat(msg)
	}

	return m, nil
}

// parseCtrlNumberKey checks if msg is ctrl+1 through ctrl+9 and returns the index (0-8), or -1 if not matched
func parseCtrlNumberKey(msg tea.KeyPressMsg) int {
	s := msg.String()
	if len(s) == 6 && s[:5] == "ctrl+" && s[5] >= '1' && s[5] <= '9' {
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
