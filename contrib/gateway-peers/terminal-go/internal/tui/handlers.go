package tui

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	goruntime "runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/app"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/browser"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/chat"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/runtime"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/session"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/shellpath"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/skills"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/userconfig"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/clipboardutil"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/notification"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/tool/editfile"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/dialog"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/messages"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
)

// --- Session management ---

func (m *appModel) handleBranchFromEdit(msg messages.BranchFromEditMsg) (tea.Model, tea.Cmd) {
	store := m.application.SessionStore()
	if store == nil {
		return m, notification.ErrorCmd("No session store configured")
	}
	if msg.ParentSessionID == "" {
		return m, notification.ErrorCmd("No parent session for branch")
	}

	ctx := context.Background()

	parent, err := store.GetSession(ctx, msg.ParentSessionID)
	if err != nil {
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to load parent session: %v", err))
	}

	newSess, err := session.BranchSession(parent, msg.BranchAtPosition)
	if err != nil {
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to branch session: %v", err))
	}

	if err := store.AddSession(ctx, newSess); err != nil {
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to save branched session: %v", err))
	}

	if current := m.application.Session(); current != nil {
		newSess.HideToolResults = current.HideToolResults
		newSess.ToolsApproved = current.ToolsApproved
	}

	// Preserve sidebar settings across branch
	sidebarSettings := m.chatPage.GetSidebarSettings()

	activeID := m.supervisor.ActiveID()

	// Update tuistate so the tab points to the branched session on re-launch.
	if m.tuiStore != nil {
		oldPersistedID := m.persistedSessionID(activeID)
		if err := m.tuiStore.UpdateTabSessionID(ctx, oldPersistedID, newSess.ID); err != nil {
			slog.WarnContext(ctx, "Failed to update tab session ID after branch", "error", err)
		}
	}
	m.persistActiveTab(newSess.ID)

	// Replace the session in the app and rebuild all per-session components.
	m.application.ReplaceSession(ctx, newSess)
	m.initSessionComponents(activeID, m.application, newSess)
	m.dialogMgr = dialog.New()

	// Restore sidebar settings
	m.chatPage.SetSidebarSettings(sidebarSettings)

	m.reapplyKeyboardEnhancements()

	return m, tea.Sequence(
		m.chatPage.Init(),
		m.resizeAll(),
		m.editor.Focus(),
		core.CmdHandler(messages.SendMsg{
			Content:     msg.Content,
			Attachments: msg.Attachments,
		}),
	)
}

func (m *appModel) handleForkSession() (tea.Model, tea.Cmd) {
	currentSession := m.application.Session()
	if currentSession == nil {
		return m, notification.ErrorCmd("No active session to fork")
	}

	store := m.application.SessionStore()
	if store == nil {
		return m, notification.ErrorCmd("No session store configured")
	}

	spawner := m.supervisor.Spawner()
	if spawner == nil {
		return m, notification.ErrorCmd("Session spawning not available")
	}

	ctx := context.Background()

	// Fork the session and clone all messages.
	forkedSession, err := session.BranchSession(currentSession, len(currentSession.Messages))
	if err != nil {
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to fork session: %v", err))
	}

	if err := store.AddSession(ctx, forkedSession); err != nil {
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to save forked session: %v", err))
	}

	a, _, cleanup, err := spawner(ctx, forkedSession.WorkingDir)
	if err != nil {
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to create runtime for fork: %v", err))
	}

	a.ReplaceSession(ctx, forkedSession)
	m.supervisor.AddSession(ctx, a, forkedSession, forkedSession.WorkingDir, cleanup)

	if m.tuiStore != nil {
		if err := m.tuiStore.AddTab(ctx, forkedSession.ID, forkedSession.WorkingDir); err != nil {
			slog.WarnContext(ctx, "Failed to persist forked tab", "error", err)
		}
	}

	return m.handleSwitchTab(forkedSession.ID)
}

func (m *appModel) handleToggleSessionStar(sessionID string) (tea.Model, tea.Cmd) {
	store := m.application.SessionStore()
	if store == nil {
		return m, notification.ErrorCmd("No session store configured")
	}

	currentSess := m.application.Session()
	var starred bool
	if currentSess != nil && currentSess.ID == sessionID {
		previous := currentSess.Starred
		currentSess.Starred = !previous
		if err := store.UpdateSession(context.Background(), currentSess); err != nil {
			currentSess.Starred = previous
			return m, notification.ErrorCmd(fmt.Sprintf("Failed to save session: %v", err))
		}
		starred = currentSess.Starred
		m.chatPage.SetSessionStarred(currentSess.Starred)
	} else {
		sess, err := store.GetSession(context.Background(), sessionID)
		if err != nil {
			return m, notification.ErrorCmd(fmt.Sprintf("Failed to load session: %v", err))
		}
		starred = !sess.Starred
		if err := store.SetSessionStarred(context.Background(), sessionID, starred); err != nil {
			return m, notification.ErrorCmd(fmt.Sprintf("Failed to update session: %v", err))
		}
	}
	return m, core.CmdHandler(messages.SessionStarChangedMsg{SessionID: sessionID, Starred: starred})
}

func (m *appModel) handleSetSessionTitle(title string) (tea.Model, tea.Cmd) {
	if err := m.application.UpdateSessionTitle(context.Background(), title); err != nil {
		if errors.Is(err, app.ErrTitleGenerating) {
			return m, notification.WarningCmd("Title is being generated, please wait")
		}
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to set session title: %v", err))
	}
	return m, notification.SuccessCmd("Title set to: " + title)
}

func (m *appModel) handleRegenerateTitle() (tea.Model, tea.Cmd) {
	sess := m.application.Session()
	if sess == nil {
		return m, notification.ErrorCmd("No active session")
	}
	if len(sess.GetLastUserMessages(1)) == 0 {
		return m, notification.ErrorCmd("Cannot regenerate title: no user message in session")
	}
	if err := m.application.RegenerateSessionTitle(context.Background()); err != nil {
		if errors.Is(err, app.ErrTitleGenerating) {
			return m, notification.WarningCmd("Title is being generated, please wait")
		}
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to regenerate title: %v", err))
	}
	spinnerCmd := m.chatPage.SetTitleRegenerating(true)
	return m, tea.Batch(spinnerCmd, notification.SuccessCmd("Regenerating title..."))
}

func (m *appModel) handleDeleteSession(sessionID string) (tea.Model, tea.Cmd) {
	store := m.application.SessionStore()
	if store == nil {
		return m, notification.ErrorCmd("No session store configured")
	}

	if err := store.DeleteSession(context.Background(), sessionID); err != nil {
		return m, notification.ErrorCmd("Failed to delete session: " + err.Error())
	}

	return m, tea.Batch(
		notification.SuccessCmd("Session deleted."),
		core.CmdHandler(messages.SessionDeletedMsg{SessionID: sessionID}),
	)
}

// --- Export / Compact / Copy ---

func (m *appModel) handleExportSession(filename string) (tea.Model, tea.Cmd) {
	exportFile, err := m.application.ExportHTML(context.Background(), filename)
	if err != nil {
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to export session: %v", err))
	}
	return m, notification.SuccessCmd("Session exported to " + exportFile)
}

func (m *appModel) handleCompactSession(additionalPrompt string) (tea.Model, tea.Cmd) {
	if !m.application.HasController() && !sessionHasCompactableMessages(m.application.Session()) {
		command := "/compact"
		if strings.TrimSpace(additionalPrompt) != "" {
			command += " " + strings.TrimSpace(additionalPrompt)
		}
		return m, tea.Batch(
			m.chatPage.AddLocalUserMessage(command),
			m.chatPage.AddLocalNoticeMessage("Not enough messages to compact."),
			m.chatPage.ScrollToBottom(),
			m.resizeAll(),
		)
	}
	return m, m.chatPage.CompactSession(additionalPrompt)
}

func (m *appModel) handleBackgroundSession() (tea.Model, tea.Cmd) {
	cmds := []tea.Cmd{
		m.chatPage.AddLocalUserMessage("/background"),
		m.chatPage.ScrollToBottom(),
		m.resizeAll(),
	}
	if strings.TrimSpace(m.chatPage.ExportTranscript()) == "" {
		cmds = append(cmds, m.chatPage.AddLocalNoticeMessage("Nothing to background yet — send a message first."))
		return m, tea.Batch(cmds...)
	}
	m.cleanupAll()
	return m, tea.Sequence(tea.Batch(cmds...), tea.Quit)
}

func sessionHasCompactableMessages(sess *session.Session) bool {
	if sess == nil {
		return false
	}
	for _, message := range sess.GetAllMessages() {
		switch message.Message.Role {
		case chat.MessageRoleUser, chat.MessageRoleAssistant:
			if strings.TrimSpace(message.Message.Content) != "" {
				return true
			}
		}
	}
	return false
}

func (m *appModel) handleCopySessionToClipboard(argument string) (tea.Model, tea.Cmd) {
	command := "/copy"
	if strings.TrimSpace(argument) != "" {
		command += " " + strings.TrimSpace(argument)
	}
	index := copyResponseIndex(argument)
	sess := m.application.Session()
	response := m.chatPage.NthLatestAssistantMessage(index)
	if response == "" {
		response = nthLatestAssistantMessage(sess, index)
	}
	cmds := []tea.Cmd{
		m.chatPage.AddLocalUserMessage(command),
		m.chatPage.ScrollToBottom(),
		m.resizeAll(),
	}
	if response == "" {
		count := m.chatPage.AssistantMessageCount()
		if count == 0 {
			count = assistantMessageCount(sess)
		}
		notice := "No assistant message to copy"
		if count > 0 && index > count {
			notice = fmt.Sprintf("Only %d assistant messages available to copy", count)
		}
		cmds = append(cmds, m.chatPage.AddLocalNoticeMessage(notice))
		return m, tea.Batch(cmds...)
	}
	notice := copyResponseNotice(response)
	cmds = append(cmds,
		m.chatPage.AddLocalNoticeMessage(notice),
		copyToClipboard(response, ""),
	)
	return m, tea.Batch(cmds...)
}

func copyResponseNotice(response string) string {
	chars := len([]rune(response))
	lines := 1
	if response != "" {
		lines = strings.Count(response, "\n") + 1
	}
	notice := fmt.Sprintf("Copied to clipboard (%d characters, %d lines)", chars, lines)

	dir := copyResponseDir()
	filename := filepath.Join(dir, "response.md")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return notice + "\nCould not write response file: " + err.Error()
	}
	if err := os.WriteFile(filename, []byte(response), 0o600); err != nil {
		return notice + "\nCould not write response file: " + err.Error()
	}
	return notice + "\nAlso written to " + filename
}

func copyResponseDir() string {
	dirName := fmt.Sprintf("claude-%d", os.Getuid())
	if goruntime.GOOS == "windows" {
		return filepath.Join(os.TempDir(), dirName)
	}
	return filepath.Join("/tmp", dirName)
}

func copyResponseIndex(argument string) int {
	value := strings.TrimSpace(argument)
	if value == "" {
		return 1
	}
	index, err := strconv.Atoi(value)
	if err != nil || index < 1 {
		return 1
	}
	return index
}

func nthLatestAssistantMessage(sess *session.Session, index int) string {
	if sess == nil {
		return ""
	}
	seen := 0
	messages := sess.GetAllMessages()
	for _, message := range slices.Backward(messages) {
		if message.Message.Role != chat.MessageRoleAssistant {
			continue
		}
		content := strings.TrimSpace(message.Message.Content)
		if content == "" {
			continue
		}
		seen++
		if seen == index {
			return content
		}
	}
	return ""
}

func assistantMessageCount(sess *session.Session) int {
	if sess == nil {
		return 0
	}
	count := 0
	for _, message := range sess.GetAllMessages() {
		if message.Message.Role != chat.MessageRoleAssistant {
			continue
		}
		if strings.TrimSpace(message.Message.Content) != "" {
			count++
		}
	}
	return count
}

func (m *appModel) handleUndoSnapshot() (tea.Model, tea.Cmd) {
	if m.chatPage.IsWorking() {
		return m, notification.WarningCmd("Wait for the current response to finish before undoing")
	}
	result, err := m.application.UndoLastSnapshot(context.Background())
	if err != nil {
		if errors.Is(err, app.ErrNothingToUndo) {
			return m, notification.InfoCmd("No snapshot to undo")
		}
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to undo snapshot: %v", err))
	}

	text := fmt.Sprintf("Restored %d file%s from the last snapshot", result.RestoredFiles, plural(result.RestoredFiles))
	return m, notification.SuccessCmd(text)
}

func (m *appModel) handleShowSnapshotsDialog() (tea.Model, tea.Cmd) {
	snapshots := m.application.ListSnapshots()
	return m, core.CmdHandler(dialog.OpenDialogMsg{
		Model: dialog.NewSnapshotsDialog(snapshots),
	})
}

func (m *appModel) handleResetSnapshot(keep int) (tea.Model, tea.Cmd) {
	if m.chatPage.IsWorking() {
		return m, notification.WarningCmd("Wait for the current response to finish before resetting")
	}
	result, err := m.application.ResetSnapshot(context.Background(), keep)
	if err != nil {
		if errors.Is(err, app.ErrNothingToUndo) {
			return m, notification.InfoCmd("Nothing to reset")
		}
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to reset snapshot: %v", err))
	}

	target := "the original state"
	if keep > 0 {
		target = fmt.Sprintf("snapshot %d", keep)
	}
	text := fmt.Sprintf("Restored %d file%s to %s", result.RestoredFiles, plural(result.RestoredFiles), target)
	return m, notification.SuccessCmd(text)
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}

// copyToClipboard returns a sequenced command that copies text through both the
// OSC 52 escape sequence (for SSH/tmux compatibility) and the platform-native
// clipboard API, then reports the resulting status to the user.
func copyToClipboard(text, successMsg string) tea.Cmd {
	return clipboardutil.Copy(text, clipboardutil.WithSuccess(successMsg))
}

// --- Agent management ---

func (m *appModel) handleSwitchAgent(agentName string) (tea.Model, tea.Cmd) {
	if agentName == m.sessionState.CurrentAgentName() {
		return m, nil
	}

	if err := m.application.SwitchAgent(agentName); err != nil {
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to switch to agent '%s': %v", agentName, err))
	}
	m.sessionState.SetCurrentAgentName(agentName)
	return m, tea.Batch(
		m.updateChatCmd(messages.SessionToggleChangedMsg{}),
		notification.SuccessCmd(fmt.Sprintf("Switched to agent '%s'", agentName)),
	)
}

func (m *appModel) handleCycleAgent() (tea.Model, tea.Cmd) {
	availableAgents := m.sessionState.AvailableAgents()
	if len(availableAgents) <= 1 {
		return m, notification.InfoCmd("No other agents available")
	}
	currentIndex := -1
	for i, agent := range availableAgents {
		if agent.Name == m.sessionState.CurrentAgentName() {
			currentIndex = i
			break
		}
	}
	nextIndex := (currentIndex + 1) % len(availableAgents)
	return m.handleSwitchToAgentByIndex(nextIndex)
}

func (m *appModel) handleStashPrompt() (tea.Model, tea.Cmd) {
	content := m.editor.Value()
	if strings.TrimSpace(content) == "" {
		if m.stashedPrompt == "" {
			return m, nil
		}
		m.editor.SetValue(m.stashedPrompt)
		m.stashedPrompt = ""
		m.lastPromptStash = time.Time{}
		m.editorLines = m.desiredEditorLines()
		m.statusBar.InvalidateCache()
		return m, m.resizeAll()
	}

	m.stashedPrompt = content
	m.editor.SetValue("")
	m.editorLines = m.desiredEditorLines()
	m.lastPromptStash = time.Now()
	m.statusBar.InvalidateCache()
	return m, tea.Batch(m.resizeAll(), invalidateStatusBarAfter(promptStashStatusTTL))
}

func (m *appModel) showLocalSystemPanel(command, content string) (tea.Model, tea.Cmd) {
	m.syncWelcomeModelLine()
	m.localPanelOpen = true
	m.localPanelCommand = command
	m.localPanelDismissNotice = localPanelDismissNotice(command)
	switch command {
	case "/help":
		m.localHelpTab = helpTabGeneral
		m.localHelpOffset = 0
		content = m.localHelpContent()
	case "/export":
		m.resetLocalExportState()
		content = m.localExportContent()
	case "/permissions":
		m.resetLocalPermissionsState()
		content = m.localPermissionsContent()
	case "/skills":
		m.localSkillsDialog = dialog.NewSkillsDialog(m.application.CurrentAgentSkills())
		content = m.localSkillsContent()
	case "/config":
		m.localSettingsTab = settingsTabConfig
		m.localSettingsBodyFocused = true
		m.localConfigSelected = false
		m.localConfigSearch = ""
		content = m.localSettingsContent()
	case "/settings":
		m.localSettingsTab = settingsTabConfig
		m.localSettingsBodyFocused = true
		m.localConfigSelected = false
		m.localConfigSearch = ""
		content = m.localSettingsContent()
	case "/status":
		m.localSettingsTab = settingsTabStatus
		m.localSettingsBodyFocused = false
		content = m.localSettingsContent()
	case "/cost":
		m.localSettingsTab = settingsTabUsage
		m.localSettingsBodyFocused = false
		content = m.localSettingsContent()
	case "/usage":
		m.localSettingsTab = settingsTabUsage
		m.localSettingsBodyFocused = false
		content = m.localSettingsContent()
	}
	return m, tea.Batch(
		m.chatPage.AddLocalUserMessage(command),
		m.chatPage.AddLocalSystemMessage(content),
		m.chatPage.ScrollToBottom(),
		m.resizeAll(),
	)
}

func localPanelDismissNotice(command string) string {
	switch command {
	case "/help":
		return "Help dialog dismissed"
	case "/permissions":
		return "Permissions dialog dismissed"
	case "/export":
		return "Export cancelled"
	case "/skills":
		return "No changes"
	case "/config":
		return "Config dialog dismissed"
	default:
		return "Settings dialog dismissed"
	}
}

func (m *appModel) handleCloseLocalPanel() (tea.Model, tea.Cmd) {
	m.syncWelcomeModelLine()
	notice := m.localPanelDismissNotice
	if m.localSettingsTab == settingsTabConfig {
		notice = "Config dialog dismissed"
	}
	if (m.localPanelCommand == "/status" || m.localPanelCommand == "/cost" || m.localPanelCommand == "/usage") && m.localSettingsTab == settingsTabStats {
		notice = "Stats dialog dismissed"
	}
	if notice == "" {
		notice = "Settings dialog dismissed"
	}
	m.localPanelOpen = false
	m.localPanelCommand = ""
	m.localPanelDismissNotice = ""
	m.localSettingsTab = settingsTabStatus
	m.localSettingsBodyFocused = false
	m.localConfigSelected = false
	m.localConfigSearch = ""
	m.localHelpTab = helpTabGeneral
	m.localHelpOffset = 0
	m.resetLocalPermissionsState()
	m.localSkillsDialog = nil
	m.resetLocalExportState()
	m.statusBar.InvalidateCache()
	return m, tea.Batch(
		m.chatPage.RemoveLastSystemMessage(),
		m.chatPage.AddLocalNoticeMessage(notice),
		m.chatPage.ScrollToBottom(),
		m.resizeAll(),
	)
}

func (m *appModel) handleLocalPanelKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	switch m.localPanelCommand {
	case "/help":
		return m.handleLocalHelpKey(msg)
	case "/export":
		return m.handleLocalExportKey(msg)
	case "/skills":
		return m.handleLocalSkillsKey(msg)
	case "/permissions":
		switch m.localPermissionsMode {
		case permissionsModeRuleInput:
			return m.handleLocalPermissionsRuleInputKey(msg)
		case permissionsModeSaveLocation:
			return m.handleLocalPermissionsSaveLocationKey(msg)
		default:
			return m.handleLocalPermissionsListKey(msg)
		}
	case "/status", "/cost", "/usage", "/config", "/settings":
		return m.handleLocalSettingsKey(msg)
	}
	if msg.String() == "esc" {
		return m.handleCloseLocalPanel()
	}
	return m, nil
}

func (m *appModel) handleLocalHelpKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc":
		return m.handleCloseLocalPanel()
	case "left":
		if m.localHelpTab > helpTabGeneral {
			m.localHelpTab--
			m.localHelpOffset = 0
			return m, m.updateLocalSystemPanel(m.localHelpContent())
		}
	case "right":
		if m.localHelpTab < helpTabCustomCommands {
			m.localHelpTab++
			m.localHelpOffset = 0
			return m, m.updateLocalSystemPanel(m.localHelpContent())
		}
	case "up":
		if m.localHelpTab != helpTabGeneral && m.localHelpOffset > 0 {
			m.localHelpOffset--
			return m, m.updateLocalSystemPanel(m.localHelpContent())
		}
	case "down":
		if m.localHelpTab != helpTabGeneral {
			maxOffset := m.maxLocalHelpOffset()
			if m.localHelpOffset < maxOffset {
				m.localHelpOffset++
				return m, m.updateLocalSystemPanel(m.localHelpContent())
			}
		}
	case "pgup":
		if m.localHelpTab != helpTabGeneral && m.localHelpOffset > 0 {
			m.localHelpOffset = max(0, m.localHelpOffset-m.localHelpPageSize())
			return m, m.updateLocalSystemPanel(m.localHelpContent())
		}
	case "pgdown":
		if m.localHelpTab != helpTabGeneral {
			maxOffset := m.maxLocalHelpOffset()
			if m.localHelpOffset < maxOffset {
				m.localHelpOffset = min(maxOffset, m.localHelpOffset+m.localHelpPageSize())
				return m, m.updateLocalSystemPanel(m.localHelpContent())
			}
		}
	case "home":
		if m.localHelpTab != helpTabGeneral && m.localHelpOffset != 0 {
			m.localHelpOffset = 0
			return m, m.updateLocalSystemPanel(m.localHelpContent())
		}
	case "end":
		if m.localHelpTab != helpTabGeneral {
			maxOffset := m.maxLocalHelpOffset()
			if m.localHelpOffset != maxOffset {
				m.localHelpOffset = maxOffset
				return m, m.updateLocalSystemPanel(m.localHelpContent())
			}
		}
	}
	return m, nil
}

func (m *appModel) handleLocalExportKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	if m.localExportFilenameMode {
		return m.handleLocalExportFilenameKey(msg)
	}
	switch msg.String() {
	case "esc":
		return m.handleCloseLocalPanel()
	case "up", "k":
		if m.localExportIndex > 0 {
			m.localExportIndex--
			return m, m.updateLocalSystemPanel(m.localExportContent())
		}
	case "down", "j":
		if m.localExportIndex < 1 {
			m.localExportIndex++
			return m, m.updateLocalSystemPanel(m.localExportContent())
		}
	case "enter":
		transcript := m.localExportTranscript()
		if strings.TrimSpace(transcript) == "" {
			m.localPanelDismissNotice = "No conversation to export"
			return m.handleCloseLocalPanel()
		}
		switch m.localExportIndex {
		case 0:
			m.localPanelDismissNotice = "Conversation copied to clipboard"
			model, cmd := m.handleCloseLocalPanel()
			return model, tea.Batch(cmd, copyToClipboard(transcript, ""))
		default:
			m.localExportFilenameMode = true
			m.localExportFilename = defaultExportFilename(transcript, time.Now())
			return m, m.updateLocalSystemPanel(m.localExportContent())
		}
	}
	return m, nil
}

func (m *appModel) handleLocalExportFilenameKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc":
		m.localExportFilenameMode = false
		return m, m.updateLocalSystemPanel(m.localExportContent())
	case "enter":
		return m.handleLocalExportFilenameSubmit()
	case "backspace", "ctrl+h":
		if m.localExportFilename != "" {
			runes := []rune(m.localExportFilename)
			m.localExportFilename = string(runes[:len(runes)-1])
			return m, m.updateLocalSystemPanel(m.localExportContent())
		}
		return m, nil
	case "ctrl+u":
		m.localExportFilename = ""
		return m, m.updateLocalSystemPanel(m.localExportContent())
	}
	if text := msg.Key().Text; text != "" {
		m.localExportFilename += text
		return m, m.updateLocalSystemPanel(m.localExportContent())
	}
	return m, nil
}

func (m *appModel) handleLocalExportFilenameSubmit() (tea.Model, tea.Cmd) {
	transcript := m.localExportTranscript()
	if strings.TrimSpace(transcript) == "" {
		m.localPanelDismissNotice = "No conversation to export"
		return m.handleCloseLocalPanel()
	}
	filename := strings.TrimSpace(m.localExportFilename)
	if filename == "" {
		filename = defaultExportFilename(transcript, time.Now())
	}
	path := filename
	if !filepath.IsAbs(path) {
		cwd, err := os.Getwd()
		if err == nil && cwd != "" {
			path = filepath.Join(cwd, path)
		}
	}
	if err := os.WriteFile(path, []byte(transcript), 0o644); err != nil {
		m.localPanelDismissNotice = "Failed to export conversation"
		model, cmd := m.handleCloseLocalPanel()
		return model, tea.Batch(cmd, notification.ErrorCmd("Failed to export conversation: "+err.Error()))
	}
	m.localPanelDismissNotice = "Conversation exported to:\n" + path
	return m.handleCloseLocalPanel()
}

func (m *appModel) handleLocalSkillsKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	if m.localSkillsDialog == nil {
		m.localSkillsDialog = dialog.NewSkillsDialog(m.application.CurrentAgentSkills())
	}
	updated, cmd := m.localSkillsDialog.Update(msg)
	m.localSkillsDialog = updated.(dialog.Dialog)
	if cmd != nil && msg.String() == "esc" {
		return m.handleCloseLocalPanel()
	}
	return m, m.updateLocalSystemPanel(m.localSkillsContent())
}

func (m *appModel) handleLocalSettingsKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	key := msg.String()
	if m.localSettingsTab == settingsTabConfig {
		selectableConfigRows := len(m.localConfigRows()) > 0
		if !m.localSettingsBodyFocused {
			switch key {
			case "down":
				m.localSettingsBodyFocused = true
				m.localConfigSelected = false
				return m, m.updateLocalSystemPanel(m.localSettingsContent())
			}
		} else {
			switch key {
			case "up":
				m.localSettingsBodyFocused = false
				m.localConfigSelected = false
				return m, m.updateLocalSystemPanel(m.localSettingsContent())
			case "esc":
				if m.localConfigSelected {
					return m.handleCloseLocalPanel()
				}
				if m.localConfigSearch != "" {
					m.localConfigSearch = ""
					return m, m.updateLocalSystemPanel(m.localSettingsContent())
				}
				if selectableConfigRows {
					m.localConfigSelected = true
					return m, m.updateLocalSystemPanel(m.localSettingsContent())
				}
				return m.handleCloseLocalPanel()
			case "enter", "space", " ":
				if m.localConfigSelected {
					return m.handleLocalAutoCompactToggle()
				}
				if key == "enter" && selectableConfigRows {
					m.localConfigSelected = true
					return m, m.updateLocalSystemPanel(m.localSettingsContent())
				}
			case "down":
				if !m.localConfigSelected && selectableConfigRows {
					m.localConfigSelected = true
					return m, m.updateLocalSystemPanel(m.localSettingsContent())
				}
			case "/":
				if m.localConfigSelected {
					m.localConfigSelected = false
					return m, m.updateLocalSystemPanel(m.localSettingsContent())
				}
			case "backspace", "ctrl+h":
				if !m.localConfigSelected && m.localConfigSearch != "" {
					runes := []rune(m.localConfigSearch)
					m.localConfigSearch = string(runes[:len(runes)-1])
					return m, m.updateLocalSystemPanel(m.localSettingsContent())
				}
			case "ctrl+u":
				if !m.localConfigSelected && m.localConfigSearch != "" {
					m.localConfigSearch = ""
					return m, m.updateLocalSystemPanel(m.localSettingsContent())
				}
			}
			if !m.localConfigSelected {
				if text := msg.Key().Text; text != "" {
					m.localConfigSearch += text
					m.localConfigSelected = false
					return m, m.updateLocalSystemPanel(m.localSettingsContent())
				}
			}
		}
	}
	switch key {
	case "esc":
		return m.handleCloseLocalPanel()
	case "left":
		if m.localSettingsTab > settingsTabStatus {
			m.localSettingsTab--
			m.localSettingsBodyFocused = false
			m.localConfigSelected = false
			m.localConfigSearch = ""
			return m, m.updateLocalSystemPanel(m.localSettingsContent())
		}
	case "right":
		if m.localSettingsTab < settingsTabStats {
			m.localSettingsTab++
			m.localSettingsBodyFocused = false
			m.localConfigSelected = false
			m.localConfigSearch = ""
			return m, m.updateLocalSystemPanel(m.localSettingsContent())
		}
	case "tab":
		if m.localSettingsTab < settingsTabStats {
			m.localSettingsTab++
		} else {
			m.localSettingsTab = settingsTabStatus
		}
		m.localSettingsBodyFocused = false
		m.localConfigSelected = false
		m.localConfigSearch = ""
		return m, m.updateLocalSystemPanel(m.localSettingsContent())
	case "shift+tab", "backtab":
		if m.localSettingsTab > settingsTabStatus {
			m.localSettingsTab--
		} else {
			m.localSettingsTab = settingsTabStats
		}
		m.localSettingsBodyFocused = false
		m.localConfigSelected = false
		m.localConfigSearch = ""
		return m, m.updateLocalSystemPanel(m.localSettingsContent())
	}
	return m, nil
}

func (m *appModel) handleLocalAutoCompactToggle() (tea.Model, tea.Cmd) {
	m.autoCompactEnabled = !m.autoCompactEnabled
	if m.application != nil {
		m.application.SetAutoCompact(m.autoCompactEnabled)
	}
	return m, m.updateLocalSystemPanel(m.localSettingsContent())
}

func (m *appModel) handleLocalPermissionsListKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	key := msg.String()
	switch key {
	case "esc":
		return m.handleCloseLocalPanel()
	case "left":
		if m.localPermissionsTab > permissionsTabRecentlyDenied {
			m.localPermissionsTab--
			m.localPermissionsSelected = false
		}
	case "right":
		if m.localPermissionsTab < permissionsTabWorkspace {
			m.localPermissionsTab++
			m.localPermissionsSelected = false
		}
	case "down":
		m.localPermissionsSelected = true
	case "up":
		m.localPermissionsSelected = false
	case "enter":
		if m.localPermissionsSelected {
			m.localPermissionsMode = permissionsModeRuleInput
			m.localPermissionRuleInput = ""
			m.localPermissionRuleDraft = ""
			m.localPermissionSaveIndex = 0
			return m, m.updateLocalSystemPanel(m.localPermissionsAddRuleContent())
		}
	}
	return m, m.updateLocalSystemPanel(m.localPermissionsContent())
}

func (m *appModel) handleLocalPermissionsRuleInputKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc":
		m.localPermissionsMode = permissionsModeList
		m.localPermissionsTab = permissionsTabAllow
		m.localPermissionsSelected = false
		m.localPermissionRuleInput = ""
		return m, m.updateLocalSystemPanel(m.localPermissionsContent())
	case "enter":
		rule := strings.TrimSpace(m.localPermissionRuleInput)
		if rule == "" {
			return m, m.updateLocalSystemPanel(m.localPermissionsAddRuleContent())
		}
		m.localPermissionRuleDraft = rule
		m.localPermissionSaveIndex = 0
		m.localPermissionsMode = permissionsModeSaveLocation
		return m, m.updateLocalSystemPanel(m.localPermissionsSaveLocationContent())
	case "backspace", "ctrl+h":
		if m.localPermissionRuleInput != "" {
			runes := []rune(m.localPermissionRuleInput)
			m.localPermissionRuleInput = string(runes[:len(runes)-1])
			return m, m.updateLocalSystemPanel(m.localPermissionsAddRuleContent())
		}
		return m, nil
	case "ctrl+u":
		m.localPermissionRuleInput = ""
		return m, m.updateLocalSystemPanel(m.localPermissionsAddRuleContent())
	}
	if text := msg.Key().Text; text != "" {
		m.localPermissionRuleInput += text
		return m, m.updateLocalSystemPanel(m.localPermissionsAddRuleContent())
	}
	return m, nil
}

func (m *appModel) handleLocalPermissionsSaveLocationKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc":
		m.localPermissionsMode = permissionsModeList
		m.localPermissionsTab = permissionsTabAllow
		m.localPermissionsSelected = false
		m.localPermissionRuleDraft = ""
		m.localPermissionRuleInput = ""
		return m, m.updateLocalSystemPanel(m.localPermissionsContent())
	case "up":
		if m.localPermissionSaveIndex > 0 {
			m.localPermissionSaveIndex--
		}
		return m, m.updateLocalSystemPanel(m.localPermissionsSaveLocationContent())
	case "down":
		if m.localPermissionSaveIndex < 2 {
			m.localPermissionSaveIndex++
		}
		return m, m.updateLocalSystemPanel(m.localPermissionsSaveLocationContent())
	case "enter":
		rule := strings.TrimSpace(m.localPermissionRuleDraft)
		if rule != "" {
			kind := permissionRuleKind(m.localPermissionsTab)
			if err := m.savePermissionRule(kind, rule); err != nil {
				return m, notification.ErrorCmd("Failed to save permission rule: " + err.Error())
			}
			m.application.AddPermissionRule(kind, rule)
		}
		m.localPermissionsMode = permissionsModeList
		m.localPermissionsTab = permissionsTabAllow
		m.localPermissionsSelected = false
		m.localPermissionRuleInput = ""
		m.localPermissionRuleDraft = ""
		m.localPermissionSaveIndex = 0
		return m, m.updateLocalSystemPanel(m.localPermissionsContent())
	}
	return m, nil
}

func (m *appModel) savePermissionRule(kind, rule string) error {
	path, err := m.permissionSettingsPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	root := map[string]any{}
	if data, err := os.ReadFile(path); err == nil && len(strings.TrimSpace(string(data))) > 0 {
		if err := json.Unmarshal(data, &root); err != nil {
			return err
		}
	} else if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}

	permissions, _ := root["permissions"].(map[string]any)
	if permissions == nil {
		permissions = map[string]any{}
	}
	key := permissionSettingsKind(kind)
	values := permissionStringList(permissions[key])
	if !slices.Contains(values, rule) {
		values = append(values, rule)
	}
	permissions[key] = values
	root["permissions"] = permissions

	data, err := json.MarshalIndent(root, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func (m *appModel) permissionSettingsPath() (string, error) {
	switch m.localPermissionSaveIndex {
	case 1:
		return filepath.Join(m.currentWorkingDirectory(), ".claude", "settings.json"), nil
	case 2:
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		return filepath.Join(home, ".claude", "settings.json"), nil
	default:
		return filepath.Join(m.currentWorkingDirectory(), ".claude", "settings.local.json"), nil
	}
}

func permissionSettingsKind(kind string) string {
	switch strings.ToLower(strings.TrimSpace(kind)) {
	case "ask":
		return "ask"
	case "deny":
		return "deny"
	default:
		return "allow"
	}
}

func permissionStringList(value any) []string {
	switch typed := value.(type) {
	case []string:
		return slices.Clone(typed)
	case []any:
		values := make([]string, 0, len(typed))
		for _, item := range typed {
			if text, ok := item.(string); ok {
				values = append(values, text)
			}
		}
		return values
	default:
		return nil
	}
}

func (m *appModel) updateLocalSystemPanel(content string) tea.Cmd {
	return tea.Batch(
		m.chatPage.RemoveLastSystemMessage(),
		m.chatPage.AddLocalSystemMessage(content),
		m.chatPage.ScrollToBottom(),
		m.resizeAll(),
	)
}

func (m *appModel) handleShowHelp() (tea.Model, tea.Cmd) {
	return m.showLocalSystemPanel("/help", m.localHelpContent())
}

const (
	helpTabGeneral = iota
	helpTabCommands
	helpTabCustomCommands
)

func (m *appModel) localHelpContent() string {
	switch m.localHelpTab {
	case helpTabCommands:
		return m.localHelpCommandsContent(false)
	case helpTabCustomCommands:
		return m.localHelpCommandsContent(true)
	default:
		return localHelpGeneralContent(m.width)
	}
}

func localHelpGeneralContent(width int) string {
	introWidth := max(72, min(107, width-13))
	intro := wrapIndentedText(
		"Claude understands your codebase, makes edits with your permission, and executes commands — right from your terminal.",
		introWidth,
		"",
	)
	lines := []string{renderHelpTabs(helpTabGeneral), ""}
	lines = append(lines, intro...)
	lines = append(lines, "Shortcuts")
	lines = append(lines, renderHelpShortcutRows()...)
	lines = append(lines, "", "For more help: https://code.claude.com/docs/en/overview", "", "Esc to cancel")
	return strings.Join(lines, "\n")
}

func renderHelpShortcutRows() []string {
	rows := [][3]string{
		{"! for shell mode", "double tap esc to clear input", "ctrl + shift + _ to undo"},
		{"/ for commands", "shift + tab to auto-accept edits", "ctrl + z to suspend"},
		{"@ for file paths", "ctrl + o for verbose output", ""},
		{"", "ctrl + t to toggle tasks", "opt + p to switch model"},
		{"", "shift + ⏎ for newline", "ctrl + s to stash prompt"},
		{"", "", "ctrl + g to edit in $EDITOR"},
	}
	lines := make([]string, 0, len(rows))
	for _, row := range rows {
		lines = append(lines, padHelpShortcutCell(row[0], 26)+padHelpShortcutCell(row[1], 37)+row[2])
	}
	return lines
}

func padHelpShortcutCell(content string, width int) string {
	pad := max(0, width-lipgloss.Width(content))
	return content + strings.Repeat(" ", pad)
}

func (m *appModel) localHelpCommandsContent(custom bool) string {
	title := "Browse default commands"
	entries := m.helpCommandEntries(custom)
	active := helpTabCommands
	if custom {
		title = "Browse custom commands"
		active = helpTabCustomCommands
	}
	lines := []string{renderHelpTabs(active), "", title, ""}
	if len(entries) == 0 {
		lines = append(lines, "  No custom commands available")
	} else {
		visible, hasMoreBefore, hasMoreAfter := m.visibleHelpCommandEntries(entries)
		for i, entry := range visible {
			prefix := "  "
			switch {
			case hasMoreBefore && i == 0:
				prefix = "↑ "
			case hasMoreAfter && i == len(visible)-1:
				prefix = "↓ "
			}
			lines = append(lines, prefix+entry.command)
			lines = append(lines, "    "+truncateHelpDescription(entry.description))
		}
	}
	lines = append(lines, "", "", "", "", "For more help: https://code.claude.com/docs/en/overview", "", "Esc to cancel")
	return strings.Join(lines, "\n")
}

func (m *appModel) visibleHelpCommandEntries(entries []helpCommandEntry) ([]helpCommandEntry, bool, bool) {
	pageSize := m.localHelpPageSize()
	if pageSize >= len(entries) {
		m.localHelpOffset = 0
		return entries, false, false
	}
	maxOffset := max(0, len(entries)-pageSize)
	start := min(max(0, m.localHelpOffset), maxOffset)
	m.localHelpOffset = start
	end := min(len(entries), start+pageSize)
	return entries[start:end], start > 0, end < len(entries)
}

func (m *appModel) maxLocalHelpOffset() int {
	entries := m.helpCommandEntries(m.localHelpTab == helpTabCustomCommands)
	return max(0, len(entries)-m.localHelpPageSize())
}

func (m *appModel) localHelpPageSize() int {
	return max(6, min(15, (m.height-10)/2))
}

type helpCommandEntry struct {
	command     string
	description string
}

func (m *appModel) helpCommandEntries(custom bool) []helpCommandEntry {
	var entries []helpCommandEntry
	seen := make(map[string]bool)
	if custom {
		for _, category := range m.commandCategories() {
			if !isDefaultHelpCommandCategory(category.Name) {
				continue
			}
			for _, item := range category.Commands {
				if item.Hidden || item.SlashCommand == "" {
					continue
				}
				seen[item.SlashCommand] = true
			}
		}
	}
	for _, category := range m.commandCategories() {
		if custom == isDefaultHelpCommandCategory(category.Name) {
			continue
		}
		for _, item := range category.Commands {
			if item.Hidden || item.SlashCommand == "" {
				continue
			}
			if seen[item.SlashCommand] {
				continue
			}
			seen[item.SlashCommand] = true
			entries = append(entries, helpCommandEntry{
				command:     item.SlashCommand,
				description: item.Description,
			})
		}
	}
	return entries
}

func isDefaultHelpCommandCategory(name string) bool {
	switch name {
	case "Session", "Settings":
		return true
	default:
		return false
	}
}

func truncateHelpDescription(description string) string {
	const limit = 94
	if lipgloss.Width(description) <= limit {
		return description
	}
	return ansi.Truncate(description, limit, "…")
}

func renderHelpTabs(active int) string {
	tabs := []struct {
		index int
		label string
	}{
		{helpTabGeneral, "General"},
		{helpTabCommands, "Commands"},
		{helpTabCustomCommands, "Custom commands"},
	}
	parts := []string{helpTitleStyle().Render("Help")}
	previousActive := false
	for i, tab := range tabs {
		currentActive := tab.index == active
		separator := "   "
		if i == 0 {
			if currentActive {
				separator = " "
			} else {
				separator = "  "
			}
		} else if currentActive || previousActive {
			separator = "  "
		}
		label := tab.label
		if currentActive {
			label = helpActiveTabStyle().Render(" " + label + " ")
		}
		parts = append(parts, separator+label)
		previousActive = currentActive
	}
	return strings.Join(parts, "")
}

func helpTitleStyle() lipgloss.Style {
	return lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("110"))
}

func helpActiveTabStyle() lipgloss.Style {
	return lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("16")).
		Background(lipgloss.Color("110"))
}

func (m *appModel) handleShowStatus() (tea.Model, tea.Cmd) {
	return m.showLocalSystemPanel("/status", m.localSettingsContent())
}

const (
	settingsTabStatus = iota
	settingsTabConfig
	settingsTabUsage
	settingsTabStats
)

func (m *appModel) localSettingsContent() string {
	switch m.localSettingsTab {
	case settingsTabConfig:
		return m.localConfigContent()
	case settingsTabUsage:
		return m.localCostContent()
	case settingsTabStats:
		return m.localStatsContent()
	default:
		return m.localStatusContent()
	}
}

func (m *appModel) localStatusContent() string {
	sessionName := "/rename to add a name"
	sessionID := "(not started)"
	cwd := ""
	if sess := m.application.Session(); sess != nil {
		if title := strings.TrimSpace(sess.Title); title != "" && title != "agentm" {
			sessionName = sess.Title
		}
		if strings.TrimSpace(sess.ID) != "" {
			sessionID = sess.ID
		}
		cwd = strings.TrimSpace(sess.WorkingDir)
	}
	if cwd == "" {
		if wd, err := os.Getwd(); err == nil {
			cwd = wd
		}
	}
	if cwd == "" {
		cwd = "(unknown)"
	}

	authToken := "not configured"
	for _, key := range []string{"ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "AGENTM_API_KEY"} {
		if os.Getenv(key) != "" {
			authToken = key
			break
		}
	}
	baseURL := os.Getenv("ANTHROPIC_BASE_URL")
	if baseURL == "" {
		baseURL = os.Getenv("AGENTM_BASE_URL")
	}
	if baseURL == "" {
		baseURL = "(provider default)"
	}

	ide := "Not connected to Visual Studio Code extension"
	if file := ideContextFileFromEnv(); file != "" {
		ide = "Connected to Visual Studio Code extension (" + file + ")"
	}

	statusVersion := m.appVersion
	if statusVersion == "" || statusVersion == "dev" {
		statusVersion = "2.1.201"
	}
	modelStatus := m.localStatusModelLine()
	toolCount := m.advertisedGatewayToolCount()
	settingSources := localStatusSettingSources()

	return fmt.Sprintf(
		renderSettingsTabs(settingsTabStatus)+"\n\n"+
			"Version:             %s\n"+
			"Session name:        %s\n"+
			"Session ID:          %s\n"+
			"cwd:                 %s\n"+
			"Auth token:          %s\n"+
			"Anthropic base URL:  %s\n\n"+
			"Model:               %s\n"+
			"IDE:                 %s\n"+
			"Gateway tools:       %d advertised\n"+
			"Setting sources:     %s\n\n"+
			"Esc to cancel",
		statusVersion,
		sessionName,
		sessionID,
		cwd,
		authToken,
		baseURL,
		modelStatus,
		ide,
		toolCount,
		settingSources,
	)
}

func (m *appModel) localStatusModelLine() string {
	model := ""
	if m.application != nil {
		model = strings.TrimSpace(m.application.CurrentModel(context.Background()))
	}
	if model == "" {
		_, ref, _ := m.contextModelInfo()
		model = strings.TrimSpace(ref)
	}
	if model == "" {
		return "Default"
	}
	return "Default (" + modelDisplayNameForRef(model) + ")"
}

func (m *appModel) syncWelcomeModelLine() {
	if m.application == nil {
		return
	}
	model := strings.TrimSpace(m.application.CurrentModel(context.Background()))
	if line := welcomeModelLineForActiveModel(model, m.thinkingModeEnabled, m.thinkingLevel); line != "" {
		m.chatPage.SetWelcomeModelLine(line)
	}
}

func localStatusSettingSources() string {
	sources := make([]string, 0, 3)
	if agentMConfigExists() {
		sources = append(sources, "AgentM config")
	}
	if hasAgentMConfigEnv() {
		sources = append(sources, "Environment")
	}
	sources = append(sources, "Command line arguments")
	return strings.Join(sources, ", ")
}

func agentMConfigExists() bool {
	home := strings.TrimSpace(os.Getenv("AGENTM_HOME"))
	if home == "" {
		if userHome, err := os.UserHomeDir(); err == nil {
			home = filepath.Join(userHome, ".agentm")
		}
	}
	if home == "" {
		return false
	}
	if info, err := os.Stat(filepath.Join(home, "config.toml")); err == nil && !info.IsDir() {
		return true
	}
	return false
}

func hasAgentMConfigEnv() bool {
	keys := []string{
		"AGENTM_API_KEY",
		"AGENTM_BASE_URL",
		"AGENTM_MODEL",
		"AGENTM_DEFAULT_MODEL",
		"AGENTM_REASONING_EFFORT",
		"ANTHROPIC_AUTH_TOKEN",
		"ANTHROPIC_API_KEY",
		"ANTHROPIC_BASE_URL",
	}
	for _, key := range keys {
		if os.Getenv(key) != "" {
			return true
		}
	}
	return false
}

func welcomeModelLineForActiveModel(model string, thinkingEnabled bool, level string) string {
	model = strings.TrimSpace(model)
	if model == "" {
		return ""
	}
	model = modelDisplayNameForRef(model)
	if thinkingEnabled {
		level = normalizeThinkingLevel(level)
		if level == "off" {
			level = "high"
		}
		return model + " with " + level + " effort · API Usage Billing"
	}
	return model + " · API Usage Billing"
}

func (m *appModel) localConfigContent() string {
	searchWidth := max(32, m.width-6)
	rows := m.localConfigRows()

	lines := []string{
		renderSettingsTabs(settingsTabConfig),
		"",
		"",
		permissionsSearchBoxTop(searchWidth),
		settingsSearchBoxMiddle(searchWidth, m.localConfigSearch),
		permissionsSearchBoxBottom(searchWidth),
		"",
	}
	for i, row := range rows {
		prefix := "  "
		if m.localSettingsBodyFocused && m.localConfigSelected && i == 0 {
			prefix = "❯ "
		}
		lines = append(lines, fmt.Sprintf("%s%-42s %s", prefix, row.name, row.value))
	}
	if len(rows) == 0 {
		lines = append(lines, "  No settings found")
	}
	footer := "←/→/tab to switch · ↓ to return · Esc to close"
	if m.localSettingsBodyFocused {
		footer = "Type to filter · Enter/↓ to select · ↑ to tabs · Esc to clear"
	}
	if m.localSettingsBodyFocused && m.localConfigSelected {
		footer = "Enter/Space to change · / to search · Esc to close"
	}
	lines = append(lines, "", footer)
	return strings.Join(lines, "\n")
}

type localConfigRow struct {
	name  string
	value string
}

func (m *appModel) localConfigRows() []localConfigRow {
	rows := []localConfigRow{
		{name: "Auto-compact", value: boolSetting(m.autoCompactEnabled)},
	}
	query := strings.ToLower(strings.TrimSpace(m.localConfigSearch))
	if query == "" {
		return rows
	}
	filtered := make([]localConfigRow, 0, len(rows))
	for _, row := range rows {
		if strings.Contains(strings.ToLower(row.name), query) {
			filtered = append(filtered, row)
		}
	}
	return filtered
}

func (m *appModel) localStatsContent() string {
	var input, output, cacheRead, cacheWrite int64
	totalCost := 0.0
	wall := time.Duration(0)
	apiDuration := time.Duration(0)
	messageCount := 0
	usageRecords := 0
	sessionCount := 0
	modelTokens := make(map[string]int64)

	if sess := m.application.Session(); sess != nil {
		sessionCount = 1
		input, output = sess.Usage()
		apiDuration = sess.Duration().Round(time.Second)
		if !sess.CreatedAt.IsZero() {
			wall = time.Since(sess.CreatedAt).Round(time.Second)
			if wall < 0 {
				wall = 0
			}
		} else if !m.startedAt.IsZero() {
			wall = time.Since(m.startedAt).Round(time.Second)
		}
		messageCount = len(sess.GetAllMessages())
		for _, record := range sess.MessageUsageHistory {
			usageRecords++
			totalCost += record.Cost
			cacheRead += record.Usage.CachedInputTokens
			cacheWrite += record.Usage.CacheWriteTokens
			model := strings.TrimSpace(record.Model)
			if model == "" {
				model = "unknown"
			}
			modelTokens[model] += record.Usage.InputTokens + record.Usage.OutputTokens +
				record.Usage.CachedInputTokens + record.Usage.CacheWriteTokens
		}
	}
	totalTokens := input + output + cacheRead + cacheWrite
	favoriteModel := m.localStatsFavoriteModel(modelTokens)

	return fmt.Sprintf(
		renderSettingsTabs(settingsTabStats)+"\n\n"+
			"   Overview   Models\n\n"+
			"      Current session\n"+
			"      %s\n\n"+
			"  All time · Current session\n\n"+
			"  Favorite model: %-18s Total tokens: %s\n\n"+
			"  Sessions: %-23d Longest session: %s\n"+
			"  Messages: %-23d Usage records: %d\n"+
			"  Active days: %-20s Current streak: %s\n"+
			"  Total cost: $%-18.4f API duration: %s\n\n"+
			"  Esc to cancel",
		localStatsActivityBar(totalTokens),
		ansi.Truncate(favoriteModel, 18, "…"),
		formatLocalTokenCount(totalTokens),
		sessionCount,
		formatLocalDuration(wall),
		messageCount,
		usageRecords,
		localStatsActiveDays(sessionCount),
		localStatsCurrentStreak(sessionCount),
		totalCost,
		formatLocalDuration(apiDuration),
	)
}

func (m *appModel) localStatsFavoriteModel(modelTokens map[string]int64) string {
	bestModel := ""
	var bestTokens int64
	for model, tokens := range modelTokens {
		if bestModel == "" || tokens > bestTokens || (tokens == bestTokens && model < bestModel) {
			bestModel = model
			bestTokens = tokens
		}
	}
	if bestModel == "" && m.application != nil {
		bestModel = strings.TrimSpace(m.application.CurrentModel(context.Background()))
	}
	if bestModel == "" {
		return "None yet"
	}
	return modelDisplayNameForRef(bestModel)
}

func localStatsActivityBar(totalTokens int64) string {
	const width = 56
	if totalTokens <= 0 {
		return strings.Repeat("·", width)
	}
	filled := int(totalTokens/10_000) + 1
	if filled > width {
		filled = width
	}
	glyph := "░"
	switch {
	case totalTokens >= 1_000_000:
		glyph = "█"
	case totalTokens >= 100_000:
		glyph = "▓"
	case totalTokens >= 10_000:
		glyph = "▒"
	}
	return strings.Repeat("·", width-filled) + strings.Repeat(glyph, filled)
}

func localStatsActiveDays(sessionCount int) string {
	if sessionCount == 0 {
		return "0/0"
	}
	return "1/1"
}

func localStatsCurrentStreak(sessionCount int) string {
	if sessionCount == 0 {
		return "0 days"
	}
	return "1 day"
}

type contextUsageCategory struct {
	name   string
	tokens int64
	glyph  string
	color  string
}

type contextDetailRow struct {
	name   string
	tokens int64
}

type contextMemoryFile struct {
	path   string
	tokens int64
}

func wrapIndentedText(text string, width int, indent string) []string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return []string{indent}
	}
	var lines []string
	line := indent
	for _, word := range words {
		if strings.TrimSpace(line) == "" {
			line += word
			continue
		}
		next := line + " " + word
		if lipgloss.Width(next) > width && strings.TrimSpace(line) != "" {
			lines = append(lines, line)
			line = indent + word
			continue
		}
		line = next
	}
	lines = append(lines, line)
	return lines
}

func boolSetting(value bool) string {
	if value {
		return "true"
	}
	return "false"
}

func (m *appModel) localExportContent() string {
	if m.localExportFilenameMode {
		return m.localExportFilenameContent()
	}
	options := []struct {
		label string
		desc  string
	}{
		{"Copy to clipboard", "Copy the conversation to your system clipboard"},
		{"Save to file", "Save the conversation to a file in the current directory"},
	}
	lines := []string{
		"Export conversation",
		"Select export method",
		"",
	}
	for i, option := range options {
		prefix := "  "
		if i == m.localExportIndex {
			prefix = "❯ "
		}
		lines = append(lines, fmt.Sprintf("%s%d. %-18s %s", prefix, i+1, option.label, option.desc))
	}
	lines = append(lines, "", "Esc to cancel")
	return strings.Join(lines, "\n")
}

func (m *appModel) localExportFilenameContent() string {
	lines := []string{
		"Export conversation",
		"Select export method",
		"",
		"Enter filename:",
		"",
		"> " + m.localExportFilename,
		"",
		"Enter to save · Esc to go back",
	}
	return strings.Join(lines, "\n")
}

func (m *appModel) resetLocalExportState() {
	m.localExportIndex = 0
	m.localExportFilenameMode = false
	m.localExportFilename = ""
}

func (m *appModel) localExportTranscript() string {
	if transcript := m.chatPage.ExportTranscript(); strings.TrimSpace(transcript) != "" {
		return transcript
	}

	sess := m.application.Session()
	if sess == nil {
		return ""
	}
	var lines []string
	for _, message := range sess.GetAllMessages() {
		content := strings.TrimSpace(message.Message.Content)
		if content == "" {
			continue
		}
		switch message.Message.Role {
		case chat.MessageRoleUser:
			lines = append(lines, "User:\n"+content)
		case chat.MessageRoleAssistant:
			lines = append(lines, "Assistant:\n"+content)
		}
	}
	return strings.Join(lines, "\n\n")
}

func defaultExportFilename(transcript string, now time.Time) string {
	slug := exportFilenameSlug(transcript)
	if slug == "" {
		slug = "conversation"
	}
	return now.Format("2006-01-02-150405") + "-" + slug + ".txt"
}

func exportFilenameSlug(transcript string) string {
	for _, line := range strings.Split(transcript, "\n") {
		line = strings.TrimSpace(ansi.Strip(line))
		if !strings.HasPrefix(line, "❯") {
			continue
		}
		line = strings.TrimSpace(strings.TrimPrefix(line, "❯"))
		if line != "" && !strings.HasPrefix(line, "/") {
			return slugifyExportFilename(line)
		}
	}
	return ""
}

func slugifyExportFilename(s string) string {
	s = strings.ToLower(s)
	var b strings.Builder
	lastHyphen := false
	for _, r := range s {
		keep := (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9')
		if keep {
			b.WriteRune(r)
			lastHyphen = false
			if b.Len() >= 48 {
				break
			}
			continue
		}
		if (r == ' ' || r == '\t' || r == '\n' || r == '\r') && b.Len() > 0 && !lastHyphen {
			b.WriteByte('-')
			lastHyphen = true
		}
		if b.Len() >= 48 {
			break
		}
	}
	return strings.Trim(b.String(), "-")
}

func (m *appModel) localSkillsContent() string {
	if m.localSkillsDialog == nil {
		m.localSkillsDialog = dialog.NewSkillsDialog(m.application.CurrentAgentSkills())
	}
	_ = m.localSkillsDialog.SetSize(m.width, m.height)
	return normalizeLocalSkillsContent(m.localSkillsDialog.View())
}

func normalizeLocalSkillsContent(content string) string {
	lines := strings.Split(content, "\n")
	if len(lines) > 0 && strings.HasPrefix(strings.TrimSpace(ansi.Strip(lines[0])), "─") {
		lines = lines[1:]
	}
	for i, line := range lines {
		if strings.HasPrefix(line, "  ") {
			lines[i] = strings.TrimPrefix(line, "  ")
		}
	}
	return strings.Join(lines, "\n")
}

func projectClaudeMemoryFile() string {
	wd, err := os.Getwd()
	if err != nil || wd == "" {
		return ""
	}
	for dir := wd; dir != ""; dir = filepath.Dir(dir) {
		path := filepath.Join(dir, "CLAUDE.md")
		if info, err := os.Stat(path); err == nil && !info.IsDir() {
			return path
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
	}
	return ""
}

func (m *appModel) localContextContent() string {
	modelDisplay, modelRef, maxTokens := m.contextModelInfo()
	categories, memoryFiles := m.contextUsageCategories()
	totalTokens := int64(0)
	for _, category := range categories {
		totalTokens += category.tokens
	}
	if totalTokens > maxTokens {
		totalTokens = maxTokens
	}

	lines := []string{"Context Usage"}
	lines = append(lines, renderContextGrid(categories, totalTokens, maxTokens, modelDisplay, modelRef)...)

	tools, _ := m.application.CurrentAgentTools(context.Background())
	toolRows := make([]contextDetailRow, 0, len(tools))
	for _, tool := range tools {
		if strings.TrimSpace(tool.Name) == "" {
			continue
		}
		toolRows = append(toolRows, contextDetailRow{name: tool.Name, tokens: 0})
	}
	sort.Slice(toolRows, func(i, j int) bool { return toolRows[i].name < toolRows[j].name })
	appendContextRows(&lines, "Gateway tools", toolRows, 12, "No gateway tools advertised yet.")

	agentRows := m.contextAgentRows()
	appendContextRows(&lines, "Custom agents · ←", agentRows, 12, "No custom agents advertised yet.")

	lines = append(lines, "", "Memory files")
	if len(memoryFiles) == 0 {
		lines = append(lines, "└ No memory files found")
	} else {
		for i, file := range memoryFiles {
			lines = append(lines, contextTreeLine(i == len(memoryFiles)-1, displayContextPath(file.path), file.tokens))
		}
	}

	skillRows := contextSkillRows(m.application.CurrentAgentSkills())
	appendContextRows(&lines, "Skills · /skills", skillRows, 6, "No skills advertised yet.")

	return strings.Join(lines, "\n")
}

func (m *appModel) contextModelInfo() (display string, ref string, maxTokens int64) {
	display = "Opus 4.8 (1M context)"
	ref = "claude-opus-4-8[1m]"
	maxTokens = 1_000_000
	if m.application != nil {
		if active := strings.TrimSpace(m.application.CurrentModel(context.Background())); active != "" {
			display = active
			ref = active
		}
	}
	models := m.application.AvailableModels(context.Background())
	if len(models) > 0 {
		selected := models[0]
		for _, model := range models {
			if model.IsCurrent {
				selected = model
				break
			}
		}
		name := strings.TrimSpace(selected.Name)
		if name == "" {
			name = strings.TrimSpace(selected.Model)
		}
		if name != "" {
			display = name
		}
		modelRef := strings.TrimSpace(selected.Ref)
		if modelRef == "" {
			modelRef = strings.TrimSpace(selected.Model)
		}
		if modelRef != "" {
			ref = modelRef
		}
	}
	if !strings.Contains(strings.ToLower(display+" "+ref), "1m") {
		maxTokens = 200_000
	}
	return display, ref, maxTokens
}

func (m *appModel) contextUsageCategories() ([]contextUsageCategory, []contextMemoryFile) {
	tools, _ := m.application.CurrentAgentTools(context.Background())
	toolTokens := int64(len(tools)) * 220
	if len(tools) > 0 && toolTokens < 800 {
		toolTokens = 800
	}

	agentTokens := int64(0)
	for _, agent := range m.sessionState.AvailableAgents() {
		agentTokens += estimateContextTokens(agent.Name+" "+agent.Description+" "+agent.Model) + 40
	}

	memoryFiles := contextMemoryFiles()
	memoryTokens := int64(0)
	for _, file := range memoryFiles {
		memoryTokens += file.tokens
	}

	skillTokens := int64(0)
	for _, skill := range m.application.CurrentAgentSkills() {
		skillTokens += estimateContextTokens(skill.Name+" "+skill.Description+" "+skill.InlineContent) + 20
	}

	messageTokens := int64(1)
	if sess := m.application.Session(); sess != nil {
		input, output := sess.Usage()
		messageTokens += input + output
		for _, record := range sess.MessageUsageHistory {
			messageTokens += record.Usage.InputTokens + record.Usage.OutputTokens +
				record.Usage.CachedInputTokens + record.Usage.CacheWriteTokens
		}
	}

	categories := []contextUsageCategory{
		{name: "System prompt", tokens: 2_000, glyph: "⛁", color: "77"},
		{name: "System tools", tokens: toolTokens, glyph: "⛁", color: "111"},
		{name: "Custom agents", tokens: agentTokens, glyph: "⛁", color: "218"},
		{name: "Memory files", tokens: memoryTokens, glyph: "⛁", color: "179"},
		{name: "Skills", tokens: skillTokens, glyph: "⛁", color: "153"},
		{name: "Messages", tokens: messageTokens, glyph: "⛀", color: "150"},
	}
	return categories, memoryFiles
}

func renderContextGrid(categories []contextUsageCategory, totalTokens, maxTokens int64, modelDisplay, modelRef string) []string {
	const cols = 20
	const rows = 10
	const cells = cols * rows

	cellGlyphs := make([]string, 0, cells)
	for _, category := range categories {
		if category.tokens <= 0 {
			continue
		}
		count := int(category.tokens * cells / maxTokens)
		if count == 0 {
			count = 1
		}
		for i := 0; i < count && len(cellGlyphs) < cells; i++ {
			cellGlyphs = append(cellGlyphs, lipgloss.NewStyle().Foreground(lipgloss.Color(category.color)).Render(category.glyph))
		}
	}
	freeGlyph := lipgloss.NewStyle().Foreground(lipgloss.Color("240")).Render("⛶")
	for len(cellGlyphs) < cells {
		cellGlyphs = append(cellGlyphs, freeGlyph)
	}

	freeTokens := maxTokens - totalTokens
	if freeTokens < 0 {
		freeTokens = 0
	}
	categoryFor := func(name string) contextUsageCategory {
		for _, category := range categories {
			if category.name == name {
				return category
			}
		}
		return contextUsageCategory{name: name, glyph: "⛁", color: "244"}
	}
	categoryLine := func(name string) string {
		category := categoryFor(name)
		return fmt.Sprintf(
			"%s %s: %s tokens (%s)",
			colorContextGlyph(category.glyph, category.color),
			category.name,
			formatContextTokenCount(category.tokens),
			formatContextPercent(category.tokens, maxTokens),
		)
	}
	right := []string{
		modelDisplay,
		modelRef,
		fmt.Sprintf("%s/%s tokens (%s)", formatContextTokenCount(totalTokens), formatContextTokenCount(maxTokens), formatContextPercent(totalTokens, maxTokens)),
		"",
		"Estimated usage by category",
		categoryLine("System prompt"),
		categoryLine("System tools"),
		categoryLine("Custom agents"),
		categoryLine("Memory files"),
		categoryLine("Skills"),
	}

	lines := make([]string, 0, rows)
	for row := 0; row < rows; row++ {
		start := row * cols
		grid := strings.Join(cellGlyphs[start:start+cols], " ")
		if row < len(right) && right[row] != "" {
			lines = append(lines, grid+"   "+right[row])
		} else {
			lines = append(lines, grid)
		}
	}
	rightPad := strings.Repeat(" ", cols*2-1+3)
	lines = append(lines,
		rightPad+categoryLine("Messages"),
		rightPad+fmt.Sprintf("%s Free space: %s (%s)", freeGlyph, formatContextTokenCount(freeTokens), formatContextPercent(freeTokens, maxTokens)),
	)
	return lines
}

func colorContextGlyph(glyph, color string) string {
	return lipgloss.NewStyle().Foreground(lipgloss.Color(color)).Render(glyph)
}

func (m *appModel) contextAgentRows() []contextDetailRow {
	agents := m.sessionState.AvailableAgents()
	rows := make([]contextDetailRow, 0, len(agents))
	for _, agent := range agents {
		name := strings.TrimSpace(agent.Name)
		if name == "" {
			continue
		}
		tokens := estimateContextTokens(agent.Name+" "+agent.Description+" "+agent.Model) + 40
		rows = append(rows, contextDetailRow{name: name, tokens: tokens})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].name < rows[j].name })
	return rows
}

func contextSkillRows(skillList []skills.Skill) []contextDetailRow {
	rows := make([]contextDetailRow, 0, len(skillList))
	for _, skill := range skillList {
		name := strings.TrimSpace(skill.Name)
		if name == "" {
			continue
		}
		tokens := estimateContextTokens(skill.Name+" "+skill.Description+" "+skill.InlineContent) + 20
		rows = append(rows, contextDetailRow{name: name, tokens: tokens})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].name < rows[j].name })
	return rows
}

func (m *appModel) advertisedGatewayToolCount() int {
	tools, err := m.application.CurrentAgentTools(context.Background())
	if err != nil {
		return 0
	}
	count := 0
	for _, tool := range tools {
		if strings.TrimSpace(tool.Name) != "" {
			count++
		}
	}
	return count
}

func appendContextRows(lines *[]string, title string, rows []contextDetailRow, limit int, empty string) {
	*lines = append(*lines, "", title)
	if len(rows) == 0 {
		*lines = append(*lines, "└ "+empty)
		return
	}
	visible := rows
	if limit > 0 && len(rows) > limit {
		visible = rows[:limit]
	}
	for i, row := range visible {
		last := i == len(visible)-1 && len(visible) == len(rows)
		*lines = append(*lines, contextTreeLine(last, row.name, row.tokens))
	}
	if len(visible) < len(rows) {
		remaining := len(rows) - len(visible)
		*lines = append(*lines, fmt.Sprintf("└ %d more", remaining))
	}
}

func contextTreeLine(last bool, name string, tokens int64) string {
	prefix := "├ "
	if last {
		prefix = "└ "
	}
	return prefix + name + ": " + formatContextTokenEstimate(tokens)
}

func contextMemoryFiles() []contextMemoryFile {
	wd, err := os.Getwd()
	if err != nil || wd == "" {
		return nil
	}
	seen := map[string]bool{}
	var files []contextMemoryFile
	for dir := wd; dir != ""; dir = filepath.Dir(dir) {
		for _, name := range []string{"CLAUDE.md", "AGENTS.md"} {
			path := filepath.Join(dir, name)
			if seen[path] {
				continue
			}
			info, err := os.Stat(path)
			if err != nil || info.IsDir() {
				continue
			}
			seen[path] = true
			files = append(files, contextMemoryFile{path: path, tokens: estimateFileContextTokens(info.Size())})
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
	}
	sort.Slice(files, func(i, j int) bool { return files[i].path < files[j].path })
	return files
}

func estimateFileContextTokens(size int64) int64 {
	if size <= 0 {
		return 0
	}
	tokens := size / 4
	if tokens == 0 {
		return 1
	}
	return tokens
}

func estimateContextTokens(text string) int64 {
	text = strings.TrimSpace(text)
	if text == "" {
		return 0
	}
	tokens := int64(len([]rune(text)) / 4)
	if tokens == 0 {
		return 1
	}
	return tokens
}

func displayContextPath(path string) string {
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		if path == home {
			return "~"
		}
		if rel, ok := strings.CutPrefix(path, home+string(os.PathSeparator)); ok {
			return "~/" + rel
		}
	}
	return path
}

func formatContextTokenEstimate(tokens int64) string {
	if tokens < 20 {
		return "< 20 tokens"
	}
	return "~" + formatContextTokenCount(tokens) + " tokens"
}

func formatContextTokenCount(n int64) string {
	if n < 1000 {
		return fmt.Sprintf("%d", n)
	}
	if n < 1_000_000 {
		if n%1000 == 0 {
			return fmt.Sprintf("%dk", n/1000)
		}
		return fmt.Sprintf("%.1fk", float64(n)/1000)
	}
	if n%1_000_000 == 0 {
		return fmt.Sprintf("%dm", n/1_000_000)
	}
	return fmt.Sprintf("%.1fm", float64(n)/1_000_000)
}

func formatContextPercent(tokens, maxTokens int64) string {
	if maxTokens <= 0 {
		return "0%"
	}
	percent := float64(tokens) / float64(maxTokens) * 100
	if percent >= 1 {
		return fmt.Sprintf("%.0f%%", percent)
	}
	return fmt.Sprintf("%.1f%%", percent)
}

func (m *appModel) localCostContent() string {
	var input, output, cacheRead, cacheWrite int64
	totalCost := 0.0
	wall := time.Duration(0)
	modelRows := make([]localCostModelUsage, 0)
	modelIndex := make(map[string]int)

	if sess := m.application.Session(); sess != nil {
		input, output = sess.Usage()
		if !sess.CreatedAt.IsZero() {
			wall = time.Since(sess.CreatedAt).Round(time.Second)
			if wall < 0 {
				wall = 0
			}
		} else if !m.startedAt.IsZero() {
			wall = time.Since(m.startedAt).Round(time.Second)
		}
		for _, record := range sess.MessageUsageHistory {
			totalCost += record.Cost
			cacheRead += record.Usage.CachedInputTokens
			cacheWrite += record.Usage.CacheWriteTokens
			model := strings.TrimSpace(record.Model)
			if model == "" {
				model = "unknown"
			}
			if idx, ok := modelIndex[model]; ok {
				modelRows[idx].cost += record.Cost
				modelRows[idx].usage.InputTokens += record.Usage.InputTokens
				modelRows[idx].usage.OutputTokens += record.Usage.OutputTokens
				modelRows[idx].usage.CachedInputTokens += record.Usage.CachedInputTokens
				modelRows[idx].usage.CacheWriteTokens += record.Usage.CacheWriteTokens
			} else {
				modelIndex[model] = len(modelRows)
				modelRows = append(modelRows, localCostModelUsage{
					model: model,
					cost:  record.Cost,
					usage: record.Usage,
				})
			}
		}
	}

	usageBlock := settingsUsageLine(
		formatLocalTokenCount(input),
		formatLocalTokenCount(output),
		formatLocalTokenCount(cacheRead),
		formatLocalTokenCount(cacheWrite),
	)
	if len(modelRows) > 0 {
		usageBlock = settingsUsageByModelLines(modelRows)
	}

	return fmt.Sprintf(
		renderSettingsTabs(settingsTabUsage)+"\n\n"+
			"Session\n\n"+
			"Total cost:            $%.4f\n"+
			"Total duration (API):  0s\n"+
			"Total duration (wall): %s\n"+
			"Total code changes:    0 lines added, 0 lines removed\n"+
			"%s\n\n"+
			"Esc to cancel",
		totalCost,
		formatLocalDuration(wall),
		usageBlock,
	)
}

type localCostModelUsage struct {
	model string
	cost  float64
	usage chat.Usage
}

func settingsUsageByModelLines(rows []localCostModelUsage) string {
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].cost != rows[j].cost {
			return rows[i].cost > rows[j].cost
		}
		return rows[i].model < rows[j].model
	})

	part := lipgloss.NewStyle().Foreground(lipgloss.Color("246"))
	lines := []string{part.Render("Usage by model:")}
	for _, row := range rows {
		lines = append(lines, part.Render(fmt.Sprintf(
			"     %s:  %s input, %s output, %s cache read, %s cache write ($%.4f)",
			row.model,
			formatLocalTokenCount(row.usage.InputTokens),
			formatLocalTokenCount(row.usage.OutputTokens),
			formatLocalTokenCount(row.usage.CachedInputTokens),
			formatLocalTokenCount(row.usage.CacheWriteTokens),
			row.cost,
		)))
	}
	return strings.Join(lines, "\n")
}

func settingsUsageLine(input, output, cacheRead, cacheWrite string) string {
	part := lipgloss.NewStyle().Foreground(lipgloss.Color("246"))
	return part.Render("Usage:") + styledSpaces(part, 17) +
		part.Render(input) + " " + part.Render("input,") + " " +
		part.Render(output) + " " + part.Render("output,") + " " +
		part.Render(cacheRead) + " " + part.Render("cache") + " " + part.Render("read,") + " " +
		part.Render(cacheWrite) + " " + part.Render("cache") + " " + part.Render("write")
}

func styledSpaces(style lipgloss.Style, count int) string {
	var b strings.Builder
	for i := 0; i < count; i++ {
		b.WriteString(style.Render(" "))
	}
	return b.String()
}

func renderSettingsTabs(active int) string {
	tabs := []struct {
		index int
		label string
	}{
		{settingsTabStatus, "Status"},
		{settingsTabConfig, "Config"},
		{settingsTabUsage, "Usage"},
		{settingsTabStats, "Stats"},
	}
	parts := []string{permissionsTitleStyle().Render("Settings")}
	previousActive := false
	for i, tab := range tabs {
		currentActive := tab.index == active
		separator := "   "
		if i == 0 {
			if currentActive {
				separator = " "
			} else {
				separator = "  "
			}
		} else if currentActive || previousActive {
			separator = "  "
		}
		label := tab.label
		if currentActive {
			label = permissionsActiveTabStyle(false).Render(" " + label + " ")
		}
		parts = append(parts, separator+label)
		previousActive = currentActive
	}
	return strings.Join(parts, "")
}

const (
	permissionsModeList = iota
	permissionsModeRuleInput
	permissionsModeSaveLocation
)

const (
	permissionsTabRecentlyDenied = iota
	permissionsTabAllow
	permissionsTabAsk
	permissionsTabDeny
	permissionsTabWorkspace
)

func (m *appModel) resetLocalPermissionsState() {
	m.localPermissionsTab = permissionsTabAllow
	m.localPermissionsSelected = false
	m.localPermissionsMode = permissionsModeList
	m.localPermissionRuleInput = ""
	m.localPermissionRuleDraft = ""
	m.localPermissionSaveIndex = 0
}

func (m *appModel) localPermissionsContent() string {
	tab := m.localPermissionsTab
	if tab < permissionsTabRecentlyDenied || tab > permissionsTabWorkspace {
		tab = permissionsTabAllow
	}
	description, searchWidth := permissionsTabDescriptionAndSearchWidth(tab)
	addRulePrefix := "  "
	footer := "←/→ to switch · ↓ to select · Esc to cancel"
	if m.localPermissionsSelected {
		addRulePrefix = "❯ "
		footer = "↑/↓ to navigate · Enter to select · ←/→ to switch · Esc to cancel"
	}
	lines := []string{
		renderPermissionsTabs(tab, m.localPermissionsSelected),
		"",
		description,
		permissionsSearchBoxTop(searchWidth),
		permissionsSearchBoxMiddle(searchWidth),
		permissionsSearchBoxBottom(searchWidth),
		"",
		addRulePrefix + "1. Add a new rule…",
	}

	for i, pattern := range m.localPermissionRulesForTab(tab) {
		lines = append(lines, "  "+lipgloss.NewStyle().Foreground(lipgloss.Color("246")).Render(fmt.Sprintf("%d. ", i+2))+pattern)
	}

	if sess := m.application.Session(); sess != nil && sess.ToolsApproved {
		lines = append(lines, "", "Bypass permissions mode is active for this session.")
	}
	lines = append(lines, "", footer)
	return strings.Join(lines, "\n")
}

func (m *appModel) localPermissionsAddRuleContent() string {
	kind := permissionRuleKind(m.localPermissionsTab)
	boxWidth := m.localPermissionsRuleBoxWidth()
	return strings.Join([]string{
		permissionsTitleStyle().Render("Add " + kind + " permission rule"),
		"",
		"Permission rules are a tool name, optionally followed by a specifier in parentheses.",
		"e.g., " + lipgloss.NewStyle().Bold(true).Render("WebFetch") + " or " + lipgloss.NewStyle().Bold(true).Render("Bash(ls *)"),
		"",
		permissionsRuleBoxTop(boxWidth),
		permissionsRuleBoxMiddle(boxWidth, m.localPermissionRuleInput),
		permissionsRuleBoxBottom(boxWidth),
		"",
		"",
		"Enter to submit · Esc to cancel",
	}, "\n")
}

func (m *appModel) localPermissionsRuleBoxWidth() int {
	if m.width <= 0 {
		return 94
	}
	return max(32, m.width-6)
}

func (m *appModel) localPermissionsSaveLocationContent() string {
	rule := strings.TrimSpace(m.localPermissionRuleDraft)
	lines := []string{
		permissionsTitleStyle().Render("Add " + permissionRuleKind(m.localPermissionsTab) + " permission rule"),
		"",
		"  " + lipgloss.NewStyle().Bold(true).Render(rule),
		"  " + permissionRuleExplanation(rule),
		"",
		"",
		"Where should this rule be saved?",
	}
	options := []struct {
		label string
		desc  string
	}{
		{"Project settings (local)", "Saved in .claude/settings.local.json"},
		{"Project settings", "Checked in at .claude/settings.json"},
		{"User settings", "Saved in at ~/.claude/settings.json"},
	}
	for i, option := range options {
		lines = append(lines, renderPermissionSaveOption(i, option.label, option.desc, i == m.localPermissionSaveIndex))
	}
	lines = append(lines, "", "", "Enter to confirm · Esc to cancel")
	return strings.Join(lines, "\n")
}

func (m *appModel) localPermissionRulesForTab(tab int) []string {
	perms := m.application.PermissionsInfo()
	if perms == nil {
		return nil
	}
	switch tab {
	case permissionsTabAsk:
		return perms.Ask
	case permissionsTabDeny:
		return perms.Deny
	case permissionsTabAllow:
		return perms.Allow
	default:
		return nil
	}
}

func renderPermissionsTabs(active int, itemSelected bool) string {
	tabs := []struct {
		index int
		label string
	}{
		{permissionsTabRecentlyDenied, "Recently denied"},
		{permissionsTabAllow, "Allow"},
		{permissionsTabAsk, "Ask"},
		{permissionsTabDeny, "Deny"},
		{permissionsTabWorkspace, "Workspace"},
	}
	parts := []string{permissionsTitleStyle().Render("Permissions")}
	previousActive := false
	for i, tab := range tabs {
		currentActive := tab.index == active
		separator := "   "
		if i == 0 || currentActive || previousActive {
			separator = "  "
		}
		label := tab.label
		if currentActive {
			label = permissionsActiveTabStyle(itemSelected).Render(" " + label + " ")
		}
		parts = append(parts, separator+label)
		previousActive = currentActive
	}
	return strings.Join(parts, "")
}

func permissionsTitleStyle() lipgloss.Style {
	return lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("153"))
}

func permissionsActiveTabStyle(itemSelected bool) lipgloss.Style {
	if itemSelected {
		return lipgloss.NewStyle().Bold(true).Reverse(true)
	}
	return lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("16")).
		Background(lipgloss.Color("153"))
}

func permissionsTabDescriptionAndSearchWidth(tab int) (string, int) {
	switch tab {
	case permissionsTabRecentlyDenied:
		return "Denied tool requests from this session will appear here.", 58
	case permissionsTabAllow:
		return "Claude Code won't ask before using allowed tools.", 47
	case permissionsTabAsk:
		return "Claude Code will always ask for confirmation before using these tools.", 68
	case permissionsTabDeny:
		return "Claude Code will always reject requests to use denied tools.", 58
	case permissionsTabWorkspace:
		return "Workspace trust and project policy rules apply here.", 58
	default:
		return "Claude Code won't ask before using allowed tools.", 47
	}
}

func permissionsSearchBoxTop(width int) string {
	return lipgloss.NewStyle().Faint(true).Render("╭" + strings.Repeat("─", width) + "╮")
}

func permissionsSearchBoxMiddle(width int) string {
	label := " " + lipgloss.NewStyle().Foreground(lipgloss.Color("246")).Render("⌕ Search…") + " "
	padding := max(0, width-lipgloss.Width(label))
	border := lipgloss.NewStyle().Faint(true)
	return border.Render("│") + label + strings.Repeat(" ", padding) + border.Render("│")
}

func settingsSearchBoxMiddle(width int, query string) string {
	labelText := "⌕ Search settings…"
	style := lipgloss.NewStyle().Foreground(lipgloss.Color("246"))
	if query != "" {
		labelText = "⌕ " + query
		style = lipgloss.NewStyle()
	}
	labelText = ansi.Truncate(labelText, max(0, width-2), "…")
	label := " " + style.Render(labelText) + " "
	padding := max(0, width-lipgloss.Width(label))
	border := lipgloss.NewStyle().Faint(true)
	return border.Render("│") + label + strings.Repeat(" ", padding) + border.Render("│")
}

func permissionsSearchBoxBottom(width int) string {
	return lipgloss.NewStyle().Faint(true).Render("╰" + strings.Repeat("─", width) + "╯")
}

func permissionsRuleBoxTop(width int) string {
	return lipgloss.NewStyle().Faint(true).Render("╭" + strings.Repeat("─", width) + "╮")
}

func permissionsRuleBoxMiddle(width int, value string) string {
	display := value
	if display == "" {
		display = lipgloss.NewStyle().Faint(true).Render("Enter permission rule…")
	}
	cursor := ""
	if value != "" {
		cursor = lipgloss.NewStyle().Reverse(true).Render(" ")
	}
	content := " " + display + cursor
	padding := max(0, width-lipgloss.Width(content))
	border := lipgloss.NewStyle().Faint(true)
	return border.Render("│") + content + strings.Repeat(" ", padding) + border.Render("│")
}

func permissionsRuleBoxBottom(width int) string {
	return lipgloss.NewStyle().Faint(true).Render("╰" + strings.Repeat("─", width) + "╯")
}

func renderPermissionSaveOption(index int, label, desc string, selected bool) string {
	number := fmt.Sprintf("%d. ", index+1)
	labelGap := strings.Repeat(" ", max(2, 26-len([]rune(label))))
	numStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("246"))
	descStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("246"))
	if selected {
		accent := lipgloss.NewStyle().Foreground(lipgloss.Color("153"))
		return accent.Render("❯") + " " + numStyle.Render(number) + accent.Render(label) + labelGap + descStyle.Render(desc)
	}
	return "  " + numStyle.Render(number) + label + labelGap + descStyle.Render(desc)
}

func permissionRuleKind(tab int) string {
	switch tab {
	case permissionsTabAsk:
		return "ask"
	case permissionsTabDeny:
		return "deny"
	case permissionsTabWorkspace:
		return "workspace"
	default:
		return "allow"
	}
}

func permissionRuleExplanation(rule string) string {
	rule = strings.TrimSpace(rule)
	if command, ok := strings.CutPrefix(rule, "Bash("); ok && strings.HasSuffix(command, ")") {
		command = strings.TrimSuffix(command, ")")
		return "The Bash command " + command
	}
	if command, ok := strings.CutPrefix(rule, "bash:cmd="); ok {
		return "The Bash command " + command
	}
	if tool, _, ok := strings.Cut(rule, "("); ok {
		tool = strings.TrimSpace(tool)
		if tool != "" {
			return "Any use of the " + tool + " tool"
		}
	}
	if rule != "" {
		return "Any use of the " + rule + " tool"
	}
	return "Permission rule"
}

func formatLocalDuration(d time.Duration) string {
	if d < time.Second {
		return "0s"
	}
	total := int64(d.Round(time.Second) / time.Second)
	hours := total / 3600
	minutes := (total % 3600) / 60
	seconds := total % 60

	parts := make([]string, 0, 3)
	if hours > 0 {
		parts = append(parts, fmt.Sprintf("%dh", hours))
	}
	if minutes > 0 {
		parts = append(parts, fmt.Sprintf("%dm", minutes))
	}
	if seconds > 0 || len(parts) == 0 {
		parts = append(parts, fmt.Sprintf("%ds", seconds))
	}
	return strings.Join(parts, " ")
}

func formatLocalTokenCount(n int64) string {
	if n < 1000 {
		return fmt.Sprintf("%d", n)
	}
	return fmt.Sprintf("%.1fk", float64(n)/1000)
}

func (m *appModel) handleSwitchToAgentByIndex(index int) (tea.Model, tea.Cmd) {
	availableAgents := m.sessionState.AvailableAgents()
	if index >= 0 && index < len(availableAgents) {
		agentName := availableAgents[index].Name
		if agentName != m.sessionState.CurrentAgentName() {
			return m, core.CmdHandler(messages.SwitchAgentMsg{AgentName: agentName})
		}
	}
	return m, nil
}

// --- Toggles ---

func (m *appModel) handleToggleYolo() (tea.Model, tea.Cmd) {
	sess := m.application.Session()
	sess.ToolsApproved = !sess.ToolsApproved
	m.sessionState.SetYoloMode(sess.ToolsApproved)
	if sess.ToolsApproved {
		m.permissionMode = permissionFooterBypass
	} else {
		m.permissionMode = permissionFooterDefault
	}
	return m.forwardChat(messages.SessionToggleChangedMsg{})
}

func (m *appModel) handleCyclePermissionMode() (tea.Model, tea.Cmd) {
	m.permissionMode = (m.permissionMode + 1) % permissionFooterModeCount
	m.lastPermissionCycle = time.Now()
	toolsApproved := m.permissionMode == permissionFooterBypass
	if m.application != nil && m.application.Session() != nil {
		m.application.Session().ToolsApproved = toolsApproved
	}
	if m.sessionState != nil {
		m.sessionState.SetYoloMode(toolsApproved)
	}
	m.statusBar.InvalidateCache()
	model, cmd := m.forwardChat(messages.SessionToggleChangedMsg{})
	return model, tea.Batch(cmd, invalidateStatusBarAfter(permissionCycleStatusTTL), m.resizeAll())
}

// handleTogglePause toggles whether the runtime loop is paused at iteration
// boundaries. The pause kicks in once the in-flight LLM request and its tool
// calls finish; running /pause again resumes the loop.
func (m *appModel) handleTogglePause() (tea.Model, tea.Cmd) {
	paused, supported := m.application.TogglePause()
	switch {
	case !supported:
		return m, notification.InfoCmd("Pause is not supported with remote runtimes")
	case paused:
		return m, notification.InfoCmd("Runtime paused — /pause again to resume")
	default:
		return m, notification.SuccessCmd("Runtime resumed")
	}
}

func (m *appModel) handleToggleHideToolResults() (tea.Model, tea.Cmd) {
	return m.forwardChat(messages.ToggleHideToolResultsMsg{})
}

func (m *appModel) handleToggleSplitDiff() (tea.Model, tea.Cmd) {
	m.sessionState.ToggleSplitDiffView()
	enabled := m.sessionState.SplitDiffView()

	// Persist to global userconfig
	go persistSplitDiffView(enabled)

	return m, tea.Batch(
		m.updateChatCmd(editfile.ToggleDiffViewMsg{}),
		m.updateChatCmd(messages.SessionToggleChangedMsg{}),
	)
}

// persistSplitDiffView writes the current split-diff toggle to the user
// config without blocking the UI. Errors are logged but otherwise ignored
// because losing the persistence is non-fatal.
func persistSplitDiffView(enabled bool) {
	cfg, err := userconfig.Load()
	if err != nil {
		slog.Warn("Failed to load userconfig for split diff toggle", "error", err)
		return
	}
	if cfg.Settings == nil {
		cfg.Settings = &userconfig.Settings{}
	}
	cfg.Settings.SplitDiffView = &enabled
	if err := cfg.Save(); err != nil {
		slog.Warn("Failed to persist split diff setting to userconfig", "error", err)
	}
}

// --- Dialogs ---

func (m *appModel) handleShowCostDialog() (tea.Model, tea.Cmd) {
	return m.showLocalSystemPanel("/cost", m.localSettingsContent())
}

func (m *appModel) handleCycleSessionColor() (tea.Model, tea.Cmd) {
	colors := []string{"red", "green", "blue", "cyan", "purple", "orange", "pink", "yellow"}
	color := colors[m.sessionColorIndex%len(colors)]
	m.sessionColorIndex++
	if promptColor := sessionPromptColor(color); promptColor != "" {
		m.editor.SetPromptColor(promptColor)
	}
	return m, tea.Batch(
		m.chatPage.AddLocalUserMessage("/color"),
		m.chatPage.AddLocalNoticeMessage("Session color set to: "+color),
		m.chatPage.ScrollToBottom(),
		m.resizeAll(),
	)
}

func sessionPromptColor(color string) string {
	switch color {
	case "red":
		return "203"
	case "purple":
		return "141"
	case "green":
		return "114"
	case "blue":
		return "75"
	case "orange":
		return "215"
	case "pink":
		return "211"
	case "cyan":
		return "116"
	case "yellow":
		return "220"
	default:
		return ""
	}
}

func (m *appModel) handleShowConfigDialog() (tea.Model, tea.Cmd) {
	return m.showLocalSystemPanel("/config", m.localSettingsContent())
}

func (m *appModel) handleShowSettingsDialog() (tea.Model, tea.Cmd) {
	return m.showLocalSystemPanel("/settings", m.localSettingsContent())
}

func (m *appModel) handleShowContextDialog() (tea.Model, tea.Cmd) {
	return m, tea.Batch(
		m.chatPage.AddLocalUserMessage("/context"),
		m.chatPage.AddLocalNoticeMessage(m.localContextContent()),
		m.chatPage.ScrollToBottom(),
		m.resizeAll(),
	)
}

func (m *appModel) handleShowUsageDialog() (tea.Model, tea.Cmd) {
	return m.showLocalSystemPanel("/usage", m.localSettingsContent())
}

func (m *appModel) handleShowExportDialog() (tea.Model, tea.Cmd) {
	return m.showLocalSystemPanel("/export", m.localExportContent())
}

func (m *appModel) handleShowPermissionsDialog() (tea.Model, tea.Cmd) {
	return m.showLocalSystemPanel("/permissions", m.localPermissionsContent())
}

func (m *appModel) handleShowToolsDialog() (tea.Model, tea.Cmd) {
	agentTools, err := m.application.CurrentAgentTools(context.Background())
	if err != nil {
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to load tools: %v", err))
	}
	return m, core.CmdHandler(dialog.OpenDialogMsg{
		Model: dialog.NewToolsDialog(agentTools),
	})
}

func (m *appModel) handleShowSkillsDialog() (tea.Model, tea.Cmd) {
	return m.showLocalSystemPanel("/skills", m.localSkillsContent())
}

// --- Model picker ---

func (m *appModel) handleOpenModelPicker(showTranscript bool) (tea.Model, tea.Cmd) {
	// Switching itself is always supported by the gateway (switch_model). The
	// picker only needs the advertised model list from session_ready; when that
	// is empty we can still switch by name, so guide the user there instead of
	// claiming switching is unsupported.
	models := m.application.AvailableModels(context.Background())
	if len(models) == 0 {
		return m, notification.InfoCmd("No model list advertised — use /model <name> to switch")
	}
	if showTranscript {
		transcriptCmd := m.chatPage.AddLocalUserMessage("/model")
		transcriptHeight := m.chatPage.TranscriptHeight()
		pickerTop := modelPickerTopRowForTranscript(transcriptHeight, m.height)
		messageHeight := m.chatPage.TranscriptMessageHeight()
		m.chatPage.SetTranscriptTopContextLines(modelPickerTopContextLines(messageHeight, pickerTop))
		viewportHeight := m.chatPage.TranscriptViewportHeight()
		slack := modelPickerTranscriptSlack(transcriptHeight, viewportHeight, pickerTop)
		if slack > 0 {
			m.chatPage.AdjustBottomSlack(slack)
		}
		return m, tea.Sequence(
			transcriptCmd,
			tea.Batch(
				core.CmdHandler(dialog.OpenDialogMsg{
					Model: dialog.NewModelPickerDialogAtTop(models, pickerTop),
				}),
				m.chatPage.ScrollToBottom(),
				m.resizeAll(),
			),
			m.chatPage.ForceScrollToBottom(),
		)
	}
	m.chatPage.SetTranscriptTopContextLines(0)
	return m, tea.Batch(
		core.CmdHandler(dialog.OpenDialogMsg{
			Model: dialog.NewModelPickerDialog(models),
		}),
		m.chatPage.ScrollToBottom(),
		m.resizeAll(),
	)
}

const (
	modelPickerDefaultTranscriptTopRow = 5
	modelPickerMinimumRenderedHeight   = 18
)

func modelPickerTopRowForTranscript(transcriptHeight, screenHeight int) int {
	topRow := max(modelPickerDefaultTranscriptTopRow, transcriptHeight-1)
	maxTopRow := max(modelPickerDefaultTranscriptTopRow, screenHeight-modelPickerMinimumRenderedHeight)
	return max(0, min(topRow, maxTopRow))
}

func modelPickerTranscriptSlack(transcriptHeight, viewportHeight, pickerTop int) int {
	if transcriptHeight <= pickerTop+1 {
		return 0
	}
	return max(0, viewportHeight-pickerTop-1)
}

func modelPickerTopContextLines(messageHeight, pickerTop int) int {
	return max(0, pickerTop-messageHeight)
}

func (m *appModel) handleModelPickerCanceled(showTranscript bool) (tea.Model, tea.Cmd) {
	m.chatPage.SetTranscriptTopContextLines(0)
	m.chatPage.AdjustBottomSlack(-m.chatPage.TranscriptViewportHeight())
	m.statusBar.InvalidateCache()
	if !showTranscript {
		return m, tea.Batch(
			m.chatPage.ScrollToBottom(),
			m.resizeAll(),
		)
	}
	return m, tea.Batch(
		m.chatPage.AddLocalNoticeMessage(modelPickerCancelNotice(m.application.AvailableModels(context.Background()))),
		m.chatPage.ScrollToBottom(),
		m.resizeAll(),
	)
}

func modelPickerCancelNotice(models []runtime.ModelChoice) string {
	for _, model := range models {
		if model.IsCurrent {
			return "Kept model as " + modelPickerNoticeName(model)
		}
	}
	for _, model := range models {
		if model.IsDefault {
			return "Kept model as " + modelPickerNoticeName(model)
		}
	}
	return "Kept model as Opus 4.8 (1M context) (default)"
}

func modelPickerNoticeName(model runtime.ModelChoice) string {
	ref := strings.TrimSpace(model.Ref)
	if ref == "" {
		ref = strings.TrimSpace(model.Name)
	}
	name := modelDisplayNameForRef(ref)
	if normalizedModelRefString(ref) == "default" {
		return name + " (default)"
	}
	return name
}

func modelDisplayNameForRef(ref string) string {
	ref = strings.TrimSpace(ref)
	switch normalizedModelRefString(ref) {
	case "default":
		return "Opus 4.8 (1M context)"
	case "opus", "opus-1m", "claude-opus-4-8", "claude-opus-4-8-1m":
		return "Opus 4.8 (1M context)"
	case "sonnet-4-6", "claude-sonnet-4-6":
		return "Sonnet 4.6"
	case "sonnet", "claude-sonnet-5":
		return "Sonnet 5"
	case "sonnet-1m", "sonnet-5-1m", "claude-sonnet-5-1m":
		return "Sonnet 5 (1M context)"
	case "haiku", "claude-haiku-4-5":
		return "Haiku 4.5"
	default:
		return ref
	}
}

func normalizedModelNoticeRef(model runtime.ModelChoice) string {
	ref := model.Ref
	if ref == "" {
		ref = model.Name
	}
	return normalizedModelRefString(ref)
}

func normalizedModelRefString(ref string) string {
	ref = strings.ToLower(strings.TrimSpace(ref))
	ref = strings.TrimPrefix(ref, "anthropic/")
	ref = strings.NewReplacer("_", "-", "[", "-", "]", "", " ", "-").Replace(ref)
	for strings.Contains(ref, "--") {
		ref = strings.ReplaceAll(ref, "--", "-")
	}
	return strings.Trim(ref, "-")
}

func (m *appModel) handleOpenThinkingToggle() (tea.Model, tea.Cmd) {
	m.lastThinkingModeToggle = time.Time{}
	m.statusBar.InvalidateCache()
	return m, core.CmdHandler(dialog.OpenDialogMsg{
		Model: dialog.NewThinkingToggleDialog(m.thinkingModeEnabled),
	})
}

func (m *appModel) handleSetThinkingMode(enabled bool) (tea.Model, tea.Cmd) {
	m.thinkingModeEnabled = enabled
	if enabled {
		if strings.TrimSpace(m.thinkingLevel) == "" || m.thinkingLevel == "off" {
			m.thinkingLevel = "high"
		}
		m.application.SetThinkingLevel(m.thinkingLevel)
	} else {
		m.application.SetThinkingMode(false)
	}
	m.lastThinkingModeToggle = time.Now()
	m.syncWelcomeModelLine()
	m.statusBar.InvalidateCache()
	return m, tea.Batch(m.resizeAll(), invalidateStatusBarAfter(thinkingModeStatusTTL))
}

func (m *appModel) handleOpenEffortPicker(showTranscript bool) (tea.Model, tea.Cmd) {
	m.lastThinkingModeToggle = time.Time{}
	m.statusBar.InvalidateCache()
	if showTranscript {
		transcriptCmd := m.chatPage.AddLocalUserMessage("/effort")
		transcriptHeight := m.chatPage.TranscriptHeight()
		pickerTop := modelPickerTopRowForTranscript(transcriptHeight, m.height)
		messageHeight := m.chatPage.TranscriptMessageHeight()
		m.chatPage.SetTranscriptTopContextLines(modelPickerTopContextLines(messageHeight, pickerTop))
		viewportHeight := m.chatPage.TranscriptViewportHeight()
		slack := modelPickerTranscriptSlack(transcriptHeight, viewportHeight, pickerTop)
		if slack > 0 {
			m.chatPage.AdjustBottomSlack(slack)
		}
		return m, tea.Sequence(
			transcriptCmd,
			tea.Batch(
				core.CmdHandler(dialog.OpenDialogMsg{
					Model: dialog.NewEffortPickerDialogAtTop(m.thinkingLevel, pickerTop),
				}),
				m.chatPage.ScrollToBottom(),
				m.resizeAll(),
			),
			m.chatPage.ForceScrollToBottom(),
		)
	}
	return m, core.CmdHandler(dialog.OpenDialogMsg{
		Model: dialog.NewEffortPickerDialog(m.thinkingLevel),
	})
}

func (m *appModel) handleSetThinkingLevel(msg messages.SetThinkingLevelMsg) (tea.Model, tea.Cmd) {
	m.chatPage.SetTranscriptTopContextLines(0)
	m.chatPage.AdjustBottomSlack(-m.chatPage.TranscriptViewportHeight())
	level := msg.Level
	level = normalizeThinkingLevel(level)
	m.thinkingLevel = level
	m.thinkingModeEnabled = level != "off"
	m.application.SetThinkingLevel(level)
	m.lastThinkingModeToggle = time.Now()
	m.syncWelcomeModelLine()
	m.statusBar.InvalidateCache()
	cmds := []tea.Cmd{m.resizeAll(), invalidateStatusBarAfter(thinkingModeStatusTTL)}
	if msg.ShowTranscript {
		cmds = append(cmds,
			m.chatPage.AddLocalNoticeMessage("Set effort level to "+level),
			m.chatPage.ScrollToBottom(),
		)
	}
	return m, tea.Batch(cmds...)
}

func (m *appModel) handleEffortPickerCanceled(showTranscript bool) (tea.Model, tea.Cmd) {
	m.chatPage.SetTranscriptTopContextLines(0)
	m.chatPage.AdjustBottomSlack(-m.chatPage.TranscriptViewportHeight())
	m.statusBar.InvalidateCache()
	if !showTranscript {
		return m, tea.Batch(
			m.chatPage.ScrollToBottom(),
			m.resizeAll(),
		)
	}
	return m, tea.Batch(
		m.chatPage.AddLocalNoticeMessage("Cancelled"),
		m.chatPage.ScrollToBottom(),
		m.resizeAll(),
	)
}

func normalizeThinkingLevel(level string) string {
	switch strings.ToLower(strings.TrimSpace(level)) {
	case "off", "low", "medium", "high":
		return strings.ToLower(strings.TrimSpace(level))
	default:
		return "high"
	}
}

// handleCycleThinkingLevel advances the current agent's thinking-effort level
// (shift+tab). On success the new level is reflected in the sidebar via the
// re-emitted agent info; only failures surface a notification.
func (m *appModel) handleCycleThinkingLevel() (tea.Model, tea.Cmd) {
	if !m.application.SupportsModelSwitching() {
		return m, notification.InfoCmd("Thinking levels can't be changed with remote runtimes")
	}
	if _, err := m.application.CycleAgentThinkingLevel(context.Background()); err != nil {
		if errors.Is(err, runtime.ErrUnsupported) {
			return m, notification.InfoCmd("Current model does not support thinking levels")
		}
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to change thinking level: %v", err))
	}
	return m, nil
}

func (m *appModel) handleChangeModel(msg messages.ChangeModelMsg) (tea.Model, tea.Cmd) {
	m.chatPage.SetTranscriptTopContextLines(0)
	m.chatPage.AdjustBottomSlack(-m.chatPage.TranscriptViewportHeight())
	if err := m.application.SetCurrentAgentModel(context.Background(), msg.ModelRef); err != nil {
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to change model: %v", err))
	}
	if msg.WelcomeModelLine != "" {
		m.chatPage.SetWelcomeModelLine(msg.WelcomeModelLine)
	}
	if msg.TranscriptNotice != "" && msg.ShowTranscript {
		if msg.RevealFocusWarning {
			m.lastIdleFocusWarningReveal = time.Now()
		}
		m.statusBar.InvalidateCache()
		cmds := []tea.Cmd{
			m.chatPage.AddLocalNoticeMessage(msg.TranscriptNotice),
			m.chatPage.ScrollToBottom(),
			m.resizeAll(),
		}
		if msg.RevealFocusWarning {
			cmds = append(cmds, invalidateStatusBarAfter(idleFocusWarningRevealTTL))
		}
		return m, tea.Batch(cmds...)
	}
	if msg.TranscriptNotice != "" {
		m.modelSwitchStatus = modelPickerFooterNotice(msg.TranscriptNotice)
	} else if msg.ModelRef == "" {
		m.modelSwitchStatus = "Model reset to default"
	} else {
		m.modelSwitchStatus = modelSwitchStatusText(msg.ModelRef)
	}
	m.lastModelSwitch = time.Now()
	m.statusBar.InvalidateCache()
	return m, tea.Batch(m.resizeAll(), modelSwitchFooterRefreshAfter())
}

func modelPickerFooterNotice(notice string) string {
	notice = strings.TrimSpace(notice)
	if strings.HasPrefix(notice, "Set model to ") {
		return "Model set to " + strings.TrimPrefix(notice, "Set model to ")
	}
	if notice == "" {
		return "Model set for this session only"
	}
	return notice
}

func modelSwitchStatusText(modelRef string) string {
	ref := strings.ToLower(strings.TrimSpace(modelRef))
	switch ref {
	case "opus", "opus-1m", "claude-opus-4-8", "claude-opus-4-8-1m":
		return "Model set to opus[1m] (claude-opus-4-8[1m]) for this session only"
	case "sonnet-4-6", "claude-sonnet-4-6":
		return "Model set to sonnet (claude-sonnet-4-6) for this session only"
	case "sonnet", "claude-sonnet-5":
		return "Model set to sonnet (claude-sonnet-5) for this session only"
	case "sonnet-1m", "sonnet-5-1m", "claude-sonnet-5-1m":
		return "Model set to sonnet[1m] (claude-sonnet-5[1m]) for this session only"
	case "haiku", "claude-haiku-4-5":
		return "Model set to haiku (claude-haiku-4-5) for this session only"
	default:
		if ref == "" {
			ref = modelRef
		}
		return "Model set to " + ref + " for this session only"
	}
}

// --- Theme picker ---

func (m *appModel) handleOpenThemePicker() (tea.Model, tea.Cmd) {
	themeRefs, err := styles.ListThemeRefs()
	if err != nil {
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to list themes: %v", err))
	}
	currentTheme := styles.CurrentTheme()
	currentRef := styles.DefaultThemeRef
	if currentTheme != nil && currentTheme.Ref != "" {
		currentRef = currentTheme.Ref
	}
	available := make(map[string]bool, len(themeRefs))
	for _, ref := range themeRefs {
		available[ref] = true
	}
	return m, tea.Batch(
		m.chatPage.AddLocalUserMessage("/theme"),
		core.CmdHandler(dialog.OpenDialogMsg{
			Model: dialog.NewThemePickerDialog(claudeThemeChoices(available, currentRef), currentRef),
		}),
		m.chatPage.ScrollToBottom(),
		m.resizeAll(),
	)
}

func claudeThemeChoices(available map[string]bool, currentRef string) []dialog.ThemeChoice {
	specs := []struct {
		name       string
		ref        string
		syntaxName string
		current    bool
	}{
		{"Auto (match terminal)", "tokyo-night", "Monokai Extended (ctrl+t to disable)", currentRef == "tokyo-night"},
		{"Dark mode", styles.DefaultThemeRef, "Monokai Extended (ctrl+t to disable)", currentRef == styles.DefaultThemeRef},
		{"Light mode", "catppuccin-latte", "GitHub (ctrl+t to disable)", currentRef == "catppuccin-latte"},
		{"Dark mode (colorblind-friendly)", "gruvbox-dark", "Monokai Extended (ctrl+t to disable)", currentRef == "gruvbox-dark"},
		{"Light mode (colorblind-friendly)", "gruvbox-light", "GitHub (ctrl+t to disable)", currentRef == "gruvbox-light"},
		{"Dark mode (ANSI colors only)", "one-dark", "Monokai Extended (ctrl+t to disable)", currentRef == "one-dark"},
		{"Light mode (ANSI colors only)", "calm-roots", "GitHub (ctrl+t to disable)", currentRef == "calm-roots"},
		{"New custom theme…", "solarized-dark", "Custom theme", false},
	}
	choices := make([]dialog.ThemeChoice, 0, len(specs))
	for i, spec := range specs {
		ref := spec.ref
		if !available[ref] {
			ref = styles.DefaultThemeRef
		}
		choices = append(choices, dialog.ThemeChoice{
			Ref:        ref,
			Name:       spec.name,
			SyntaxName: spec.syntaxName,
			IsCurrent:  spec.current && ref == currentRef,
			IsBuiltin:  true,
			HasOrder:   true,
			Order:      i,
		})
	}
	return choices
}

func (m *appModel) handleChangeTheme(themeRef string) (tea.Model, tea.Cmd) {
	notice := "Theme set to " + claudeThemeNoticeName(themeRef)
	if styles.GetPersistedThemeRef() == themeRef {
		return m, tea.Batch(
			m.chatPage.AddLocalNoticeMessage(notice),
			m.chatPage.ScrollToBottom(),
			m.resizeAll(),
		)
	}
	theme, err := styles.LoadTheme(themeRef)
	if err != nil {
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to load theme: %v", err))
	}
	styles.ApplyTheme(theme)
	m.invalidateCachesForThemeChange()

	if err := styles.SaveThemeToUserConfig(themeRef); err != nil {
		slog.Warn("Failed to save theme to user config", "theme", themeRef, "error", err)
	}
	return m, tea.Batch(
		m.chatPage.AddLocalNoticeMessage(notice),
		m.chatPage.ScrollToBottom(),
		m.resizeAll(),
		core.CmdHandler(messages.ThemeChangedMsg{}),
	)
}

func claudeThemeNoticeName(themeRef string) string {
	switch themeRef {
	case "tokyo-night":
		return "auto"
	case styles.DefaultThemeRef, "one-dark":
		return "dark"
	case "catppuccin-latte", "calm-roots":
		return "light"
	case "gruvbox-dark":
		return "dark colorblind-friendly"
	case "gruvbox-light":
		return "light colorblind-friendly"
	default:
		return strings.TrimSpace(strings.TrimPrefix(themeRef, styles.UserThemePrefix))
	}
}

func (m *appModel) handleThemePreview(themeRef string) (tea.Model, tea.Cmd) {
	if current := styles.CurrentTheme(); current != nil && current.Ref == themeRef {
		return m, nil
	}
	theme, err := styles.LoadTheme(themeRef)
	if err != nil {
		return m, nil
	}
	styles.ApplyTheme(theme)
	return m.applyThemeChanged()
}

func (m *appModel) handleThemeCancelPreview(originalRef string) (tea.Model, tea.Cmd) {
	cmds := []tea.Cmd{
		m.chatPage.AddLocalNoticeMessage("Theme picker dismissed"),
		m.chatPage.ScrollToBottom(),
		m.resizeAll(),
	}
	if current := styles.CurrentTheme(); current == nil || current.Ref != originalRef {
		styles.ApplyThemeRef(originalRef)
		model, cmd := m.applyThemeChanged()
		if app, ok := model.(*appModel); ok {
			m = app
		} else {
			cmds = append(cmds, cmd)
			return model, tea.Batch(cmds...)
		}
		cmds = append(cmds, cmd)
	}
	return m, tea.Batch(cmds...)
}

func (m *appModel) invalidateCachesForThemeChange() {
	// markdown's style cache resets itself via styles.OnThemeChange.
	m.statusBar.InvalidateCache()
}

func (m *appModel) applyThemeChanged() (tea.Model, tea.Cmd) {
	m.invalidateCachesForThemeChange()
	return m, tea.Batch(
		m.updateDialogCmd(messages.ThemeChangedMsg{}),
		m.updateChatCmd(messages.ThemeChangedMsg{}),
	)
}

// handleThemeFileChanged hot-reloads a theme that was modified on disk.
func (m *appModel) handleThemeFileChanged(themeRef string) (tea.Model, tea.Cmd) {
	theme, err := styles.LoadTheme(themeRef)
	if err != nil {
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to hot-reload theme: %v", err))
	}
	styles.ApplyTheme(theme)
	return m, tea.Batch(
		notification.SuccessCmd("Theme hot-reloaded"),
		core.CmdHandler(messages.ThemeChangedMsg{}),
	)
}

// --- Miscellaneous ---

func (m *appModel) handleOpenURL(url string) (tea.Model, tea.Cmd) {
	if err := browser.Open(context.Background(), url); err != nil {
		slog.Warn("Failed to open URL", "url", url, "error", err)
		return m, notification.ErrorCmd("Failed to open URL in browser")
	}
	return m, nil
}

func (m *appModel) handleAgentCommand(command string) (tea.Model, tea.Cmd) {
	ctx := context.Background()

	// Inspect the command before resolving so we can detect /commands that
	// switch to a sub-agent. For those, we switch first and only then send
	// the resolved message — otherwise the message would be processed by
	// the previous agent.
	cmd, _, ok := m.application.LookupCommand(ctx, command)
	resolved := m.application.ResolveCommand(ctx, command)

	var cmds []tea.Cmd
	switchSucceeded := true
	if ok && cmd.Agent != "" && cmd.Agent != m.sessionState.CurrentAgentName() {
		// Attempt to switch agents. If the switch fails, handleSwitchAgent
		// returns an error notification command. We check if the agent actually
		// changed to determine success, rather than relying on the command type.
		prevAgent := m.sessionState.CurrentAgentName()
		switched, switchCmd := m.handleSwitchAgent(cmd.Agent)
		var ok bool
		if m, ok = switched.(*appModel); !ok {
			// This should never happen, but if it does, log and continue with the original model
			slog.WarnContext(ctx, "handleSwitchAgent returned unexpected type", "type", fmt.Sprintf("%T", switched))
			switchSucceeded = false
		} else {
			// Check if the agent actually changed to determine if the switch succeeded.
			// If it failed, we must not send the message to the wrong agent.
			switchSucceeded = m.sessionState.CurrentAgentName() != prevAgent
		}
		if switchCmd != nil {
			cmds = append(cmds, switchCmd)
		}
	}

	if resolved != "" && switchSucceeded {
		cmds = append(cmds, core.CmdHandler(messages.SendMsg{Content: resolved, BypassQueue: true}))
	}

	return m, tea.Batch(cmds...)
}

func (m *appModel) handleAttachFile(filePath string) (tea.Model, tea.Cmd) {
	if filePath != "" {
		if err := m.editor.AttachFile(filePath); err != nil {
			slog.Warn("failed to attach file", "path", filePath, "error", err)
			// Attachment failed — open the file picker with an error notification
			return m, tea.Batch(
				notification.ErrorCmd("Failed to attach "+filePath),
				core.CmdHandler(dialog.OpenDialogMsg{
					Model: dialog.NewFilePickerDialog(filePath),
				}),
			)
		}
		return m, notification.SuccessCmd("File attached: " + filePath)
	}

	// No path provided — open the file picker dialog
	return m, core.CmdHandler(dialog.OpenDialogMsg{
		Model: dialog.NewFilePickerDialog(filePath),
	})
}

// --- Speech-to-text ---

func (m *appModel) handleStartSpeak() (tea.Model, tea.Cmd) {
	if m.transcriber.IsRunning() {
		return m, nil
	}

	// Close any previous channel to unblock stale waitForTranscript goroutines.
	m.closeTranscriptCh()

	ch := make(chan string, 100)
	m.transcriptCh = ch
	err := m.transcriber.Start(context.Background(), func(delta string) {
		select {
		case ch <- delta:
		default:
		}
	})
	if err != nil {
		m.closeTranscriptCh()
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to start listening: %v", err))
	}

	return m, tea.Batch(
		notification.InfoCmd("🎤 Listening... (ENTER to send or ESC to cancel)"),
		m.editor.SetRecording(true),
		m.waitForTranscript(),
	)
}

func (m *appModel) handleStopSpeak() (tea.Model, tea.Cmd) {
	if !m.transcriber.IsRunning() {
		return m, nil
	}

	m.transcriber.Stop()
	m.closeTranscriptCh()

	return m, tea.Batch(m.editor.SetRecording(false), notification.SuccessCmd("Stopped listening"))
}

// waitForTranscript returns a command that blocks until the next transcript
// delta arrives and delivers it as a SpeakTranscriptMsg.
func (m *appModel) waitForTranscript() tea.Cmd {
	ch := m.transcriptCh
	return func() tea.Msg {
		delta, ok := <-ch
		if !ok {
			return nil
		}
		return messages.SpeakTranscriptMsg{Delta: delta}
	}
}

// closeTranscriptCh closes the transcript channel and sets it to nil,
// unblocking any goroutines waiting in waitForTranscript.
func (m *appModel) closeTranscriptCh() {
	if m.transcriptCh != nil {
		close(m.transcriptCh)
		m.transcriptCh = nil
	}
}

func (m *appModel) startShell() (tea.Model, tea.Cmd) {
	cmd := newInteractiveShellCmd("Type 'exit' to return to " + m.appName)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	// Run the shell in the active session's working directory so it matches
	// where the tools operate (e.g. the worktree created by --worktree),
	// rather than inheriting the process CWD.
	if runner := m.supervisor.GetRunner(m.supervisor.ActiveID()); runner != nil && runner.WorkingDir != "" {
		cmd.Dir = runner.WorkingDir
	}
	return m, tea.ExecProcess(cmd, nil)
}

// newInteractiveShellCmd returns a command that launches the user's preferred
// interactive shell. The command is owned by tea.ExecProcess, not by any
// request-scoped context, so exec.Command is intentional.
func newInteractiveShellCmd(exitMsg string) *exec.Cmd {
	if goruntime.GOOS != "windows" {
		shell := shellpath.DetectUnixShell()
		return execCmd(shell, "-i", "-c", `echo -e "\n`+exitMsg+`"; exec `+shell)
	}

	psArgs := []string{"-NoLogo", "-NoExit", "-Command", `Write-Host ""; Write-Host "` + exitMsg + `"`}
	if path, err := exec.LookPath("pwsh.exe"); err == nil {
		return execCmd(path, psArgs...)
	}
	if path, err := exec.LookPath("powershell.exe"); err == nil {
		return execCmd(path, psArgs...)
	}
	// Use absolute path to cmd.exe to prevent PATH hijacking (CWE-426).
	return execCmd(shellpath.WindowsCmdExe(), "/K", "echo. & echo "+exitMsg)
}

// execCmd is a thin wrapper around exec.Command used for interactive
// processes whose lifecycle is owned by tea.ExecProcess (not a context).
func execCmd(name string, args ...string) *exec.Cmd {
	return exec.Command(name, args...) //nolint:noctx // owned by tea.ExecProcess
}
