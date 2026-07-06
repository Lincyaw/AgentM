package tui

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/runtime"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/completion"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/internal/editorname"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/messages"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
)

const (
	workflowTaskPickerBaseFooter = "↑/↓ to select · Enter to view"
	bypassPermissionsFooter      = "⏵⏵ bypass permissions on (shift+tab to cycle)"
	workflowCompletionManageTTL  = 30 * time.Second
	promptStashStatusTTL         = 4 * time.Second
	modelSwitchStatusTTL         = 4 * time.Second
	thinkingModeStatusTTL        = 2 * time.Second
	idleFocusWarningRevealTTL    = 2 * time.Second
	tmuxFocusWarningTTL          = 30 * time.Second
	permissionCycleStatusTTL     = 6 * time.Second
)

var (
	tmuxFocusEventsOnce    sync.Once
	tmuxFocusEventsWarning string

	claudeFooterAccentStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("211"))
	claudeFooterAutoStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("220"))
	claudeFooterAcceptStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("147"))
	claudeFooterPlanStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("73"))
	claudeFooterTextStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("246"))
)

type permissionFooterMode int

const (
	permissionFooterBypass permissionFooterMode = iota
	permissionFooterAuto
	permissionFooterDefault
	permissionFooterAcceptEdits
	permissionFooterPlan
	permissionFooterModeCount
)

type startupFooterRefreshMsg struct{}

type modelSwitchFooterRefreshMsg struct{}

type workflowRow struct {
	sessionID      string
	title          string
	role           string
	isMain         bool
	active         bool
	running        bool
	needsAttention bool
	createdAt      time.Time
}

type bottomActivityKind int

const (
	bottomActivityNone bottomActivityKind = iota
	bottomActivityWorkflowOnly
	bottomActivityBackgroundOnly
	bottomActivityMixed
)

func (k bottomActivityKind) hasRows() bool {
	return k != bottomActivityNone
}

func (k bottomActivityKind) hasWorkflowTasks() bool {
	return k == bottomActivityWorkflowOnly || k == bottomActivityMixed
}

func (k bottomActivityKind) toggleTarget() string {
	switch k {
	case bottomActivityWorkflowOnly:
		return "tasks"
	case bottomActivityBackgroundOnly, bottomActivityMixed:
		return "activity rows"
	default:
		return ""
	}
}

func (k bottomActivityKind) ctrlTActionLabel() string {
	switch k {
	case bottomActivityWorkflowOnly:
		return "toggle tasks"
	case bottomActivityBackgroundOnly, bottomActivityMixed:
		return "toggle activity"
	default:
		return "new tab"
	}
}

func (m *appModel) bottomActivityKind() bottomActivityKind {
	hasWorkflow := len(m.workflowTaskTabs()) > 0
	hasBackground := len(m.backgroundActivities) > 0
	switch {
	case hasWorkflow && hasBackground:
		return bottomActivityMixed
	case hasWorkflow:
		return bottomActivityWorkflowOnly
	case hasBackground:
		return bottomActivityBackgroundOnly
	default:
		return bottomActivityNone
	}
}

func (m *appModel) hasWorkflowTasks() bool {
	return len(m.workflowTaskManageableTabs()) > 0
}

func (m *appModel) hasBottomActivityRows() bool {
	return m.bottomActivityKind().hasRows()
}

func (m *appModel) activeIsWorkflowTask() bool {
	if m.supervisor == nil {
		return false
	}
	activeID := m.supervisor.ActiveID()
	if activeID == "" || activeID == m.mainSessionID {
		return false
	}
	tab, ok := m.workflowTabInfo(activeID)
	return ok && tab.Background
}

func (m *appModel) workflowTaskTabs() []messages.TabInfo {
	tabs := m.allWorkflowTaskTabs()
	tasks := make([]messages.TabInfo, 0, len(tabs))
	for _, tab := range tabs {
		if m.workflowTaskVisible(tab) {
			tasks = append(tasks, tab)
		}
	}
	return tasks
}

func (m *appModel) allWorkflowTaskTabs() []messages.TabInfo {
	if m.supervisor == nil {
		return nil
	}
	tabs, _ := m.supervisor.GetTabs()
	tasks := make([]messages.TabInfo, 0, len(tabs))
	for _, tab := range tabs {
		if tab.SessionID == m.mainSessionID {
			continue
		}
		if tab.Background {
			tasks = append(tasks, tab)
		}
	}
	return tasks
}

func (m *appModel) workflowTaskManageableTabs() []messages.TabInfo {
	tabs := m.allWorkflowTaskTabs()
	tasks := make([]messages.TabInfo, 0, len(tabs))
	for _, tab := range tabs {
		if m.workflowTaskVisible(tab) || m.workflowTaskRecentlyCompleted(tab.SessionID) {
			tasks = append(tasks, tab)
		}
	}
	return tasks
}

func (m *appModel) workflowTabInfo(sessionID string) (messages.TabInfo, bool) {
	if m.supervisor == nil || sessionID == "" {
		return messages.TabInfo{}, false
	}
	tabs, _ := m.supervisor.GetTabs()
	for _, tab := range tabs {
		if tab.SessionID == sessionID {
			return tab, true
		}
	}
	return messages.TabInfo{}, false
}

func (m *appModel) markWorkflowSession(sessionID string) {
	if sessionID == "" {
		return
	}
	if m.workflowSessions == nil {
		m.workflowSessions = map[string]bool{}
	}
	m.workflowSessions[sessionID] = true
}

func (m *appModel) workflowSessionKnown(sessionID string) bool {
	if sessionID == "" {
		return false
	}
	if m.workflowSessions[sessionID] {
		return true
	}
	tab, ok := m.workflowTabInfo(sessionID)
	return ok && tab.Background
}

func (m *appModel) workflowTaskVisible(tab messages.TabInfo) bool {
	return tab.IsRunning || tab.NeedsAttention || m.workflowVisible[tab.SessionID]
}

func (m *appModel) workflowTaskRecentlyCompleted(sessionID string) bool {
	if sessionID == "" || len(m.workflowCompletedUntil) == 0 {
		return false
	}
	until, ok := m.workflowCompletedUntil[sessionID]
	if !ok {
		return false
	}
	if time.Now().After(until) {
		delete(m.workflowCompletedUntil, sessionID)
		return false
	}
	return true
}

func (m *appModel) workflowSessionIsBackground(sessionID string) bool {
	if sessionID == "" {
		return false
	}
	for _, tab := range m.allWorkflowTaskTabs() {
		if tab.SessionID == sessionID {
			return true
		}
	}
	return false
}

func (m *appModel) markWorkflowRecentlyCompleted(sessionID string) {
	if !m.workflowSessionIsBackground(sessionID) {
		return
	}
	m.markWorkflowSession(sessionID)
	if m.workflowCompletedUntil == nil {
		m.workflowCompletedUntil = map[string]time.Time{}
	}
	m.workflowCompletedUntil[sessionID] = time.Now().Add(workflowCompletionManageTTL)
}

func (m *appModel) clearWorkflowRecentlyCompleted(sessionID string) {
	if sessionID == "" || len(m.workflowCompletedUntil) == 0 {
		return
	}
	delete(m.workflowCompletedUntil, sessionID)
}

func (m *appModel) setWorkflowVisible(sessionID string, visible bool) {
	if sessionID == "" {
		return
	}
	if visible {
		if m.workflowVisible == nil {
			m.workflowVisible = map[string]bool{}
		}
		m.workflowVisible[sessionID] = true
		return
	}
	delete(m.workflowVisible, sessionID)
}

func (m *appModel) chooseMainSessionID(fallback string) string {
	if m.supervisor == nil {
		return fallback
	}
	tabs, _ := m.supervisor.GetTabs()
	for _, tab := range tabs {
		if !tab.Background {
			return tab.SessionID
		}
	}
	if len(tabs) > 0 {
		return tabs[0].SessionID
	}
	return fallback
}

func (m *appModel) workflowRows() []workflowRow {
	if m.supervisor == nil {
		return nil
	}
	activeID := m.supervisor.ActiveID()
	rows := []workflowRow{{
		sessionID: m.mainSessionID,
		title:     "main",
		isMain:    true,
		active:    activeID == m.mainSessionID,
	}}
	tabs := m.workflowTaskTabs()
	if m.workflowTaskPickerOpen {
		tabs = m.workflowTaskManageableTabs()
	}
	if m.activeIsWorkflowTask() {
		activeID := m.supervisor.ActiveID()
		found := false
		for _, tab := range tabs {
			if tab.SessionID == activeID {
				found = true
				break
			}
		}
		if !found {
			if tab, ok := m.workflowTabInfo(activeID); ok {
				tabs = append(tabs, tab)
			}
		}
	}
	for _, tab := range tabs {
		createdAt := time.Time{}
		if tab.CreatedAt > 0 {
			createdAt = time.Unix(tab.CreatedAt, 0)
		}
		rows = append(rows, workflowRow{
			sessionID:      tab.SessionID,
			title:          cmpNonEmpty(tab.Title, "Untitled task"),
			role:           m.workflowRole(tab.SessionID),
			active:         tab.SessionID == activeID,
			running:        tab.IsRunning,
			needsAttention: tab.NeedsAttention,
			createdAt:      createdAt,
		})
	}
	return rows
}

func (m *appModel) workflowRole(sessionID string) string {
	if state := m.sessionStates[sessionID]; state != nil {
		if name := strings.TrimSpace(state.CurrentAgentName()); name != "" {
			return name
		}
	}
	return "general-purpose"
}

func (m *appModel) openWorkflowTaskPicker() {
	if !m.hasWorkflowTasks() {
		return
	}
	m.workflowTaskPickerOpen = true
	m.backgroundActivityPrompt = false
	m.backgroundActivityDetail = false
	m.shortcutSheetOpen = false
	m.syncWorkflowTaskPickerIndex()
	m.statusBar.InvalidateCache()
}

func (m *appModel) closeWorkflowTaskPicker() {
	if !m.workflowTaskPickerOpen {
		return
	}
	m.workflowTaskPickerOpen = false
	m.statusBar.InvalidateCache()
}

func (m *appModel) syncWorkflowTaskPickerIndex() {
	rows := m.workflowRows()
	if len(rows) == 0 {
		m.workflowTaskPickerIndex = 0
		return
	}
	activeID := ""
	if m.supervisor != nil {
		activeID = m.supervisor.ActiveID()
	}
	for i, row := range rows {
		if row.sessionID == activeID {
			m.workflowTaskPickerIndex = i
			return
		}
	}
	_, idx, _ := m.selectedWorkflowTaskRow(rows)
	m.workflowTaskPickerIndex = idx
}

func (m *appModel) syncWorkflowTaskPickerState() {
	if !m.workflowTaskPickerOpen {
		return
	}
	if !m.hasWorkflowTasks() {
		m.closeWorkflowTaskPicker()
		return
	}
	m.syncWorkflowTaskPickerIndex()
}

func (m *appModel) moveWorkflowTaskSelection(delta int) {
	rows := m.workflowRows()
	_, idx, ok := m.selectedWorkflowTaskRow(rows)
	if !ok {
		return
	}
	m.workflowTaskPickerIndex = (idx + delta + len(rows)) % len(rows)
}

func (m *appModel) activateWorkflowTaskSelection() (tea.Model, tea.Cmd) {
	rows := m.workflowRows()
	row, _, ok := m.selectedWorkflowTaskRow(rows)
	if !ok {
		m.closeWorkflowTaskPicker()
		return m, nil
	}
	target := row.sessionID
	m.closeWorkflowTaskPicker()
	if target == "" {
		return m, nil
	}
	return m.handleSwitchTab(target)
}

func (m *appModel) stopWorkflowTaskSelection() (tea.Model, tea.Cmd) {
	rows := m.workflowRows()
	row, _, ok := m.selectedWorkflowTaskRow(rows)
	if !ok {
		return m, nil
	}
	if row.isMain || row.sessionID == "" {
		return m, nil
	}
	model, cmd := m.handleCloseTab(row.sessionID)
	m.syncWorkflowTaskPickerState()
	return model, cmd
}

func (m *appModel) workflowTaskPickerFooterText() string {
	rows := m.workflowRows()
	row, _, ok := m.selectedWorkflowTaskRow(rows)
	if !ok {
		return workflowTaskPickerBaseFooter
	}
	if row.isMain {
		return workflowTaskPickerBaseFooter
	}
	return "Enter to view · x to stop"
}

func (m *appModel) selectedWorkflowTaskRow(rows []workflowRow) (workflowRow, int, bool) {
	if len(rows) == 0 {
		return workflowRow{}, 0, false
	}
	idx := min(max(m.workflowTaskPickerIndex, 0), len(rows)-1)
	return rows[idx], idx, true
}

func (m *appModel) recordWorkflowTranscript(sessionID string, msg tea.Msg) {
	if sessionID == "" || sessionID == m.mainSessionID {
		return
	}
	m.trackWorkflowVisibility(sessionID, msg)
	switch ev := msg.(type) {
	case *runtime.AgentChoiceEvent:
		m.appendWorkflowTranscript(sessionID, ev.Content)
	case *runtime.AgentChoiceReasoningEvent:
		if m.transcriptDetailed {
			m.appendWorkflowTranscript(sessionID, ev.Content)
		}
	}
}

func (m *appModel) trackWorkflowVisibility(sessionID string, msg tea.Msg) {
	switch msg.(type) {
	case *runtime.StreamStartedEvent,
		*runtime.ToolCallConfirmationEvent,
		*runtime.MaxIterationsReachedEvent:
		m.clearWorkflowRecentlyCompleted(sessionID)
		m.setWorkflowVisible(sessionID, true)
	case *runtime.StreamStoppedEvent:
		m.setWorkflowVisible(sessionID, false)
		m.markWorkflowRecentlyCompleted(sessionID)
	}
}

func (m *appModel) appendWorkflowTranscript(sessionID, content string) {
	if strings.TrimSpace(content) == "" {
		return
	}
	current := m.workflowTranscripts[sessionID]
	if current != "" && !strings.HasSuffix(current, "\n") && isWorkflowCompletionNote(content) {
		current += "\n"
	}
	m.workflowTranscripts[sessionID] = current + content
}

func isWorkflowCompletionNote(content string) bool {
	return strings.HasPrefix(content, "✓ ") || strings.HasPrefix(content, "sub-agent ")
}

func (m *appModel) toggleBottomActivityRows() tea.Cmd {
	if !m.hasBottomActivityRows() {
		return nil
	}
	m.bottomActivityRowsHidden = !m.bottomActivityRowsHidden
	m.workflowTaskPickerOpen = false
	m.backgroundActivityPrompt = false
	m.backgroundActivityDetail = false
	m.statusBar.SetActivity(m.backgroundActivityText())
	m.statusBar.InvalidateCache()
	return m.resizeAll()
}

func (m *appModel) toggleShortcutSheet() tea.Cmd {
	wasOpen := m.shortcutSheetOpen
	m.shortcutSheetOpen = !m.shortcutSheetOpen
	if m.shortcutSheetOpen {
		m.shortcutSheetDismissed = false
		m.agentsModeOpen = false
		m.agentsModeHelp = false
		m.agentsModeReplyOpen = false
		m.agentsModeDeleteConfirmID = ""
		m.cancelAgentsModeRename()
		m.editor.SetPlaceholder("")
		m.workflowTaskPickerOpen = false
		m.backgroundActivityPrompt = false
		m.backgroundActivityDetail = false
	}
	if wasOpen && !m.shortcutSheetOpen {
		m.shortcutSheetDismissed = true
	}
	m.statusBar.InvalidateCache()
	return m.resizeAll()
}

func (m *appModel) closeInlineSurfaces() tea.Cmd {
	completionWasOpen := m.completions.Open()
	completionCmd := m.updateCompletionsCmd(completion.CloseMsg{})
	editorCompletionCmd := m.updateEditorCmd(completion.ClosedMsg{})
	changed := completionWasOpen || m.shortcutSheetOpen || m.agentsModeOpen || m.workflowTaskPickerOpen || m.backgroundActivityPrompt || m.backgroundActivityDetail
	if m.shortcutSheetOpen {
		m.shortcutSheetDismissed = true
	}
	m.shortcutSheetOpen = false
	m.agentsModeOpen = false
	m.agentsModeHelp = false
	m.agentsModeReplyOpen = false
	m.agentsModeDeleteConfirmID = ""
	m.cancelAgentsModeRename()
	m.editor.SetPlaceholder("")
	m.workflowTaskPickerOpen = false
	m.backgroundActivityPrompt = false
	m.backgroundActivityDetail = false
	if !changed {
		return tea.Batch(completionCmd, editorCompletionCmd)
	}
	m.statusBar.InvalidateCache()
	return tea.Batch(completionCmd, editorCompletionCmd, m.resizeAll())
}

func (m *appModel) toggleTranscriptDetailed() tea.Cmd {
	m.transcriptDetailed = !m.transcriptDetailed
	if !m.transcriptDetailed {
		m.transcriptVerbose = false
	}
	m.statusBar.InvalidateCache()
	return tea.Batch(
		m.resizeAll(),
		core.CmdHandler(messages.SetTranscriptDetailMsg{
			Detailed: m.transcriptDetailed,
			Verbose:  m.transcriptVerbose,
		}),
	)
}

func (m *appModel) toggleTranscriptVerbose() tea.Cmd {
	if !m.transcriptDetailed {
		return nil
	}
	m.transcriptVerbose = !m.transcriptVerbose
	m.statusBar.InvalidateCache()
	return tea.Batch(
		m.resizeAll(),
		core.CmdHandler(messages.SetTranscriptDetailMsg{
			Detailed: m.transcriptDetailed,
			Verbose:  m.transcriptVerbose,
		}),
	)
}

func (m *appModel) footerText() string {
	activityKind := m.bottomActivityKind()
	if time.Since(m.lastExitRequest) <= 2*time.Second {
		return claudeFooterTextStyle.Render("Press Ctrl-C again to exit")
	}
	switch {
	case m.shortcutSheetOpen:
		return ""
	case m.agentsModeOpen && m.agentsModeHelp:
		return "ctrl+r to rename          @ to mention            alt+1 to open    ? to close"
	case m.agentsModeOpen:
		return m.agentsModeFooterText()
	case m.transcriptDetailed && m.transcriptVerbose:
		return claudeFooterTextStyle.Render("Showing detailed transcript · ctrl+o to toggle · ctrl+e to collapse")
	case m.transcriptDetailed:
		return claudeFooterTextStyle.Render("Showing detailed transcript · ctrl+o to toggle · ctrl+e to show all")
	case m.workflowTaskPickerOpen:
		return m.workflowTaskPickerFooterText()
	case m.activeIsWorkflowTask():
		return "Enter to view · x to stop · ctrl+x ctrl+k to stop all agents"
	case m.backgroundActivityDetail:
		return "← to go back · Esc/Enter/Space to close · x to stop"
	case m.backgroundActivityPrompt:
		return joinBackgroundStatusParts(m.permissionModeFooterTextCompact(), m.backgroundShellText(), "Enter to view tasks")
	case m.editor.IsHistorySearchActive():
		searchText := m.editor.HistorySearchFooterText()
		spacing := "  "
		includeAgents := false
		if searchText == "search prompts:" {
			spacing = "   "
			includeAgents = true
		}
		if m.editor.ShellMode() {
			return searchText + spacing + claudeFooterAccentStyle.Render("! for shell mode")
		}
		return searchText + spacing + m.permissionModeFooterText(includeAgents)
	case m.chatPage.IsWorking():
		if m.focusedPanel == PanelEditor && !m.editor.ShellMode() && m.editor.Value() != "" {
			return m.permissionModeFooterText(false)
		}
		if shellText := m.backgroundShellText(); shellText != "" {
			return joinBackgroundStatusParts(m.permissionModeFooterTextCompact(), "esc to interrupt", shellText, "↓ to manage")
		}
		text := joinBackgroundStatusParts(m.permissionModeFooterText(false), "esc to interrupt", "← for agents")
		if m.hasWorkflowTasks() {
			text += " · ↓ to manage"
		}
		return text
	case activityKind.hasRows():
		if shellText := m.backgroundShellText(); shellText != "" {
			return joinBackgroundStatusParts(m.permissionModeFooterTextCompact(), shellText, "← for agents", "↓ to manage")
		}
		verb := "hide"
		if m.bottomActivityRowsHidden {
			verb = "show"
		}
		target := activityKind.toggleTarget()
		if activityKind.hasWorkflowTasks() {
			return "ctrl+t to " + verb + " " + target + " · ← for agents · ↓ to manage"
		}
		return "ctrl+t to " + verb + " " + target + " · ← for agents"
	case m.editor.PasteExpandHintVisible():
		return claudeFooterTextStyle.Render("paste again to expand")
	case m.editor.ShellMode():
		return claudeFooterAccentStyle.Render("! for shell mode")
	case m.editor.Value() != "":
		return m.permissionModeFooterText(false)
	default:
		return m.permissionModeFooterText(true)
	}
}

func (m *appModel) permissionModeFooterText(includeAgents bool) string {
	switch m.permissionMode {
	case permissionFooterAuto:
		return renderClaudePermissionFooter("⏵⏵ auto mode on", includeAgents, claudeFooterAutoStyle)
	case permissionFooterDefault:
		return renderClaudeFooterText("? for shortcuts", includeAgents)
	case permissionFooterAcceptEdits:
		return renderClaudePermissionFooter("⏵⏵ accept edits on", includeAgents, claudeFooterAcceptStyle)
	case permissionFooterPlan:
		return renderClaudePermissionFooter("⏸ plan mode on", includeAgents, claudeFooterPlanStyle)
	default:
		return renderClaudePermissionFooter("⏵⏵ bypass permissions on", includeAgents, claudeFooterAccentStyle)
	}
}

func (m *appModel) permissionModeFooterTextCompact() string {
	switch m.permissionMode {
	case permissionFooterAuto:
		return claudeFooterAutoStyle.Render("⏵⏵ auto mode on")
	case permissionFooterDefault:
		return claudeFooterTextStyle.Render("? for shortcuts")
	case permissionFooterAcceptEdits:
		return claudeFooterAcceptStyle.Render("⏵⏵ accept edits on")
	case permissionFooterPlan:
		return claudeFooterPlanStyle.Render("⏸ plan mode on")
	default:
		return claudeFooterAccentStyle.Render("⏵⏵ bypass permissions on")
	}
}

func renderClaudePermissionFooter(prefix string, includeAgents bool, prefixStyle lipgloss.Style) string {
	suffix := " (shift+tab to cycle)"
	if includeAgents {
		suffix += " · ← for agents"
	}
	return prefixStyle.Render(prefix) + claudeFooterTextStyle.Render(suffix)
}

func renderClaudeFooterText(text string, includeAgents bool) string {
	if includeAgents {
		text += " · ← for agents"
	}
	return claudeFooterTextStyle.Render(text)
}

func (m *appModel) agentsModeFooterText() string {
	if m.agentsModeRenaming {
		return "enter to save · esc to cancel"
	}
	if m.agentsModeReplyOpen {
		if m.editor.Value() != "" {
			return "enter to send · esc to close · ctrl+x to delete"
		}
		return "enter to open · space to close · ctrl+x to delete"
	}
	if m.agentsModeDeleteConfirmID != "" {
		return "ctrl+x to confirm"
	}
	permission := m.agentsModePermissionFooterText()
	if m.editor.Value() != "" {
		return renderAgentsModeFooter(permission, "enter to create · esc to clear")
	}
	if m.agentsModeSelectedID != "" && m.agentsModeSelectedID != m.activeSessionID() {
		return renderAgentsModeFooter(permission, "enter to open · space to reply · ctrl+x to delete · ? for shortcuts")
	}
	return renderAgentsModeFooter(permission, "enter to return · → to return · space to reply · ctrl+x to delete · ? for shortcuts")
}

func (m *appModel) agentsModePermissionFooterText() string {
	if m.permissionMode == permissionFooterBypass {
		return claudeFooterAccentStyle.Render("⏵⏵ bypass permissions")
	}
	return m.permissionModeFooterTextCompact()
}

func renderAgentsModeFooter(permission, suffix string) string {
	if suffix == "" {
		return agentsModeFooterModeStyle.Render(permission)
	}
	return agentsModeFooterModeStyle.Render(permission) + agentsModeRowTextStyle.Render(" · "+suffix)
}

func (m *appModel) footerActivityText() string {
	if m.agentsModeOpen {
		return ""
	}
	return m.backgroundActivityText()
}

func (m *appModel) ctrlTActionLabel() string {
	return m.bottomActivityKind().ctrlTActionLabel()
}

func (m *appModel) footerRightText() string {
	if m.shortcutSheetOpen {
		return ""
	}
	if m.agentsModeOpen || m.workflowTaskPickerOpen || m.backgroundActivityPrompt || m.backgroundActivityDetail {
		return ""
	}
	if m.editor.IsHistorySearchActive() {
		if m.editor.HistorySearchStartedInShellMode() {
			return m.effortFooterRightText()
		}
		return ""
	}
	if m.editor.HistoryNavigationHintVisible() {
		return claudeFooterTextStyle.Render("ctrl+r to search history")
	}
	if time.Since(m.lastEscClearRequest) <= 2*time.Second {
		return claudeFooterTextStyle.Render("Esc again to clear")
	}
	if m.editor.PasteExpandHintVisible() {
		if ideText := styledIDEContextStatusText(); ideText != "" {
			return ideText
		}
		return ""
	}
	if m.focusedPanel == PanelEditor && m.editor.ShowKillBufferHint() {
		return "Ctrl+Y to paste deleted text"
	}
	if m.focusedPanel == PanelEditor && !m.editor.ShellMode() && m.editor.Value() != "" {
		if strings.Contains(m.editor.Value(), "\n") {
			return "ctrl+g to edit in " + editorname.FromEnv(os.Getenv("VISUAL"), os.Getenv("EDITOR"))
		}
		if !m.chatPage.IsWorking() {
			return ""
		}
	}
	if m.chatPage.IsWorking() {
		return ""
	}
	if m.transcriptDetailed {
		return claudeFooterTextStyle.Render("verbose")
	}
	if m.idleFocusWarningRevealActive() &&
		!m.editor.ShellMode() && m.startupFocusWarningText() != "" {
		return ""
	}
	if m.promptStashStatusActive() {
		return m.stashedFooterRightText()
	}
	if m.modelSwitchStatusActive() {
		return ""
	}
	if m.permissionCycleStatusActive() {
		return ""
	}
	if m.editor.ShellMode() {
		return ""
	}
	if m.streamCancelFooterHidden && m.focusedPanel == PanelEditor && m.editor.Value() == "" {
		return ""
	}
	if !m.thinkingModeEnabled && m.editor.Value() == "" {
		if m.thinkingModeStatusActive() {
			return "Thinking off"
		}
		return ""
	}
	if m.thinkingModeEnabled && m.editor.Value() == "" && !m.lastThinkingModeToggle.IsZero() {
		if m.thinkingModeStatusActive() {
			return "Thinking on"
		}
	}
	if m.idleFooterRightHidden && m.focusedPanel == PanelEditor && !m.editor.ShellMode() && m.editor.Value() == "" {
		return ""
	}
	if m.shortcutSheetDismissed && m.focusedPanel == PanelEditor && !m.editor.ShellMode() && m.editor.Value() == "" {
		if ideText := styledIDEContextStatusText(); ideText != "" {
			return ideText
		}
		return m.effortFooterRightText()
	}
	if m.focusedPanel == PanelEditor && !m.editor.ShellMode() && m.editor.Value() == "" {
		return m.effortFooterRightText()
	}
	if ideText := styledIDEContextStatusText(); ideText != "" {
		return ideText
	}
	return m.effortFooterRightText()
}

func renderClaudeEffortFooterRight(level string) string {
	level = strings.TrimSpace(level)
	if level == "" {
		level = "high"
	}
	return claudeFooterTextStyle.Render("● " + level + " · /effort")
}

func (m *appModel) effortFooterRightText() string {
	if !m.thinkingModeEnabled {
		return ""
	}
	return renderClaudeEffortFooterRight(m.thinkingLevel)
}

func (m *appModel) stashedFooterRightText() string {
	return claudeFooterTextStyle.Render("› stashed")
}

func (m *appModel) footerExtraLines() []string {
	if m.agentsModeOpen {
		if m.agentsModeHelp {
			return []string{"  ctrl+s to switch views    ctrl+t to pin to top    esc to quit"}
		}
		return nil
	}
	if m.modelSwitchStatusActive() {
		return []string{rightAlignedFooterLine(m.width, styles.SecondaryStyle.Render(m.modelSwitchStatus))}
	}
	if m.escClearedInputActive() {
		return nil
	}
	if time.Since(m.lastExitRequest) <= 2*time.Second {
		if ideText := styledIDEContextStatusText(); ideText != "" {
			return []string{rightAlignedFooterLine(m.width, ideText)}
		}
		return nil
	}
	if warning := m.startupFocusWarningText(); warning != "" &&
		m.shouldShowFocusWarningLine() {
		return []string{warningFooterLine(m.width, warning)}
	}
	if m.completions.Open() ||
		m.shortcutSheetOpen ||
		m.workflowTaskPickerOpen ||
		m.backgroundActivityPrompt ||
		m.backgroundActivityDetail ||
		m.chatPage.IsWorking() ||
		!m.chatPage.IsTranscriptEmpty() ||
		m.editor.Value() != "" {
		return nil
	}
	if m.permissionCycleStatusActive() {
		return nil
	}
	return nil
}

func (m *appModel) shouldShowFocusWarningLine() bool {
	return !m.editor.ShellMode() &&
		!m.completions.Open() &&
		!m.shortcutSheetOpen &&
		!m.workflowTaskPickerOpen &&
		!m.backgroundActivityPrompt &&
		!m.backgroundActivityDetail &&
		!m.chatPage.IsWorking()
}

func (m *appModel) permissionCycleStatusActive() bool {
	return !m.lastPermissionCycle.IsZero() && time.Since(m.lastPermissionCycle) <= permissionCycleStatusTTL
}

func (m *appModel) thinkingModeStatusActive() bool {
	return !m.lastThinkingModeToggle.IsZero() && time.Since(m.lastThinkingModeToggle) <= thinkingModeStatusTTL
}

func (m *appModel) idleFocusWarningRevealActive() bool {
	return !m.lastIdleFocusWarningReveal.IsZero() && time.Since(m.lastIdleFocusWarningReveal) <= idleFocusWarningRevealTTL
}

func (m *appModel) exitRequestClearedInputActive() bool {
	return !m.lastExitClearedInput.IsZero() &&
		time.Since(m.lastExitRequest) <= 2*time.Second &&
		m.editor.Value() == ""
}

func (m *appModel) escClearedInputActive() bool {
	return !m.lastEscClearedInput.IsZero() && m.editor.Value() == ""
}

func (m *appModel) modelSwitchStatusActive() bool {
	return m.modelSwitchStatus != "" &&
		!m.lastModelSwitch.IsZero() &&
		time.Since(m.lastModelSwitch) <= modelSwitchStatusTTL
}

func rightAlignedFooterLine(width int, content string) string {
	padding := max(0, width-lipgloss.Width(content)-2)
	return strings.Repeat(" ", padding) + content
}

func warningFooterLine(width int, content string) string {
	text := ansi.Truncate(content, max(1, width-4), "…")
	return "  " + styles.SecondaryStyle.Render(text)
}

func (m *appModel) shouldShowStartupFocusWarning() bool {
	return !m.startedAt.IsZero() && time.Since(m.startedAt) <= tmuxFocusWarningTTL
}

func (m *appModel) startupFocusWarningText() string {
	if !m.shouldShowStartupFocusWarning() {
		return ""
	}
	return tmuxFocusEventsWarningText()
}

func tmuxFocusEventsWarningText() string {
	if os.Getenv("TMUX") == "" {
		return ""
	}
	tmuxFocusEventsOnce.Do(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
		defer cancel()
		out, err := exec.CommandContext(ctx, "tmux", "show-options", "-gv", "focus-events").Output()
		if err == nil && strings.TrimSpace(string(out)) == "on" {
			return
		}
		tmuxFocusEventsWarning = "tmux focus-events off · add 'set -g focus-events on' to ~/.tmux.conf and reattach for focus tracking"
	})
	return tmuxFocusEventsWarning
}

func startupFooterRefreshAfter() tea.Cmd {
	return tea.Tick(tmuxFocusWarningTTL+100*time.Millisecond, func(time.Time) tea.Msg {
		return startupFooterRefreshMsg{}
	})
}

func modelSwitchFooterRefreshAfter() tea.Cmd {
	return tea.Tick(modelSwitchStatusTTL+100*time.Millisecond, func(time.Time) tea.Msg {
		return modelSwitchFooterRefreshMsg{}
	})
}

func (m *appModel) bottomSurfaceHeight(width int) int {
	view := m.renderBottomSurface(width)
	if view == "" {
		return 0
	}
	return lipgloss.Height(view)
}

func (m *appModel) renderBottomSurface(width int) string {
	if m.completions.Open() {
		return m.completions.View()
	}
	if m.agentsModeOpen {
		return ""
	}
	var parts []string
	if m.shortcutSheetOpen {
		parts = append(parts, m.renderShortcutSheet(width))
	}
	if warningRows := m.renderTerminalWarnings(width); warningRows != "" {
		parts = append(parts, warningRows)
	}
	if m.backgroundActivityDetail {
		if details := m.renderBackgroundShellDetails(width); details != "" {
			parts = append(parts, details)
		}
	}
	if rows := m.renderBottomActivityRows(width); rows != "" {
		parts = append(parts, rows)
	}
	return strings.Join(parts, "\n")
}

func (m *appModel) promptStashStatusActive() bool {
	return !m.lastPromptStash.IsZero() && time.Since(m.lastPromptStash) <= promptStashStatusTTL
}

func (m *appModel) renderTerminalWarnings(width int) string {
	if len(m.terminalWarnings) == 0 || width <= 0 || m.shortcutSheetOpen {
		return ""
	}
	innerWidth := max(20, width-appPaddingHorizontal)
	lines := make([]string, 0, len(m.terminalWarnings))
	for _, warning := range m.terminalWarnings {
		line := " " + warning
		lines = append(lines, styles.SecondaryStyle.Render(ansi.Truncate(line, innerWidth, "…")))
	}
	return lipgloss.NewStyle().Padding(0, styles.AppPadding).Render(strings.Join(lines, "\n"))
}

func (m *appModel) renderShortcutSheet(width int) string {
	innerWidth := max(20, width-appPaddingHorizontal)
	rows := [][3]string{
		{"! for shell mode", "double tap esc to clear input", "ctrl + shift + _ to undo"},
		{"/ for commands", "shift + tab to auto-accept edits", "ctrl + z to suspend"},
		{"@ for file paths", "ctrl + o for verbose output", ""},
		{"", "ctrl + t to toggle tasks", "opt + p to switch model"},
		{"", "shift + ⏎ for newline", "ctrl + s to stash prompt"},
		{"", "", "ctrl + g to edit in $EDITOR"},
	}
	lines := make([]string, 0, len(rows))
	style := lipgloss.NewStyle().Foreground(lipgloss.Color("246"))
	for _, row := range rows {
		line := fmt.Sprintf(" %-22s  %-33s  %s", row[0], row[1], row[2])
		lines = append(lines, style.Render(ansi.Truncate(line, innerWidth, "")))
	}
	return lipgloss.NewStyle().Padding(0, styles.AppPadding).Render(strings.Join(lines, "\n"))
}

func (m *appModel) renderWorkflowDetail(width, height int) string {
	if m.supervisor == nil {
		return ""
	}
	sessionID := m.supervisor.ActiveID()
	text := strings.TrimSpace(m.workflowTranscripts[sessionID])
	if text == "" {
		return ""
	}
	innerWidth := max(20, width-appPaddingHorizontal)
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		lines[i] = ansi.Truncate(line, innerWidth, "…")
	}
	if height > 0 && len(lines) > height {
		lines = lines[len(lines)-height:]
	}
	return lipgloss.NewStyle().
		Width(width).
		Height(height).
		Padding(0, styles.AppPadding).
		Align(lipgloss.Left, lipgloss.Top).
		Render(styles.BaseStyle.Render(strings.Join(lines, "\n")))
}

func (m *appModel) renderBottomActivityRows(width int) string {
	activityKind := m.bottomActivityKind()
	activeWorkflow := m.activeIsWorkflowTask()
	if !activityKind.hasRows() && !m.workflowTaskPickerOpen && !activeWorkflow {
		return ""
	}
	if !m.workflowTaskPickerOpen && m.bottomActivityRowsHidden {
		return ""
	}
	if m.backgroundActivityPrompt || m.backgroundActivityDetail {
		return ""
	}
	if m.backgroundShellCount() > 0 && len(m.backgroundActivities) == m.backgroundShellCount() && !activeWorkflow && !m.workflowTaskPickerOpen {
		return ""
	}
	var workflowRows []workflowRow
	if m.workflowTaskPickerOpen || activityKind.hasWorkflowTasks() || activeWorkflow {
		workflowRows = m.workflowRows()
	}
	var activityRows []backgroundActivity
	hiddenActivityCount := 0
	if !m.bottomActivityRowsHidden {
		activityRows, hiddenActivityCount = m.backgroundActivityRows()
	}
	if len(workflowRows) == 0 && len(activityRows) == 0 {
		return ""
	}
	_, selected, ok := m.selectedWorkflowTaskRow(workflowRows)
	if !ok {
		selected = -1
	}
	innerWidth := max(20, width-appPaddingHorizontal)
	lines := make([]string, 0, len(workflowRows)+len(activityRows)+1)
	for i, row := range workflowRows {
		lines = append(lines, m.renderWorkflowRow(row, i == selected && m.workflowTaskPickerOpen, innerWidth))
	}
	if hiddenActivityCount > 0 {
		lines = append(lines, renderHiddenBackgroundActivitiesRow(hiddenActivityCount, innerWidth))
	}
	for _, row := range activityRows {
		lines = append(lines, m.renderBackgroundActivityRow(row, innerWidth))
	}
	return lipgloss.NewStyle().Padding(0, styles.AppPadding).Render(strings.Join(lines, "\n"))
}

func (m *appModel) renderWorkflowRow(row workflowRow, selected bool, width int) string {
	prefix := "  "
	if selected {
		prefix = "❯ "
	}
	icon := "◯"
	switch {
	case row.needsAttention:
		icon = "✻"
	case row.active:
		icon = "⏺"
	}

	var left string
	if row.isMain {
		left = prefix + icon + " main"
	} else {
		left = fmt.Sprintf("%s%s %-16s %s", prefix, icon, cmpNonEmpty(row.role, "agent"), row.title)
	}

	if !row.isMain {
		age := formatWorkflowAge(row.createdAt)
		if age != "" {
			gap := max(1, width-lipgloss.Width(left)-lipgloss.Width(age))
			left += strings.Repeat(" ", gap) + styles.MutedStyle.Render(age)
		}
	}

	if lipgloss.Width(left) > width {
		left = ansi.Truncate(left, width, "…")
	}
	if selected {
		return styles.HighlightWhiteStyle.Render(left)
	}
	if row.active {
		return styles.SecondaryStyle.Render(left)
	}
	return styles.MutedStyle.Render(left)
}

func formatWorkflowAge(createdAt time.Time) string {
	if createdAt.IsZero() {
		return ""
	}
	d := time.Since(createdAt)
	switch {
	case d < time.Minute:
		return fmt.Sprintf("%ds", max(0, int(d.Seconds())))
	case d < time.Hour:
		return fmt.Sprintf("%dm", int(d.Minutes()))
	default:
		return fmt.Sprintf("%dh", int(d.Hours()))
	}
}

func cmpNonEmpty(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}

func (m *appModel) clearBottomActivitiesForSession(sessionID string) {
	delete(m.workflowTranscripts, sessionID)
	delete(m.workflowVisible, sessionID)
	m.removeBackgroundActivitiesForSession(sessionID)
	m.statusBar.SetActivity(m.backgroundActivityText())
}
