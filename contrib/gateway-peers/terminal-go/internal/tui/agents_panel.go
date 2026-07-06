package tui

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/runtime"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/notification"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/messages"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
)

const (
	agentsModePlaceholder         = "describe a task for a new session"
	agentsModeReplyPlaceholder    = "reply"
	agentsModeReplyComposerHeight = 5
)

var (
	agentsModeRowTextStyle            = lipgloss.NewStyle().Foreground(lipgloss.Color("246"))
	agentsModeSelectedRowTextStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("231")).Background(lipgloss.Color("237"))
	agentsModeActiveIconStyle         = lipgloss.NewStyle().Foreground(lipgloss.Color("220"))
	agentsModeSelectedActiveIconStyle = agentsModeActiveIconStyle.Background(lipgloss.Color("237"))
	agentsModeSelectedCellStyle       = lipgloss.NewStyle().Background(lipgloss.Color("237"))
	agentsModeDeleteMetaStyle         = lipgloss.NewStyle().Foreground(lipgloss.Color("211"))
	agentsModeFooterModeStyle         = lipgloss.NewStyle().Foreground(lipgloss.Color("211"))
	agentsModeSectionTitleStyle       = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("246"))
)

type agentsModeRows struct {
	needsInput []messages.TabInfo
	working    []messages.TabInfo
	completed  []messages.TabInfo
}

func (m *appModel) openAgentsMode() tea.Cmd {
	if m.agentsModeOpen {
		return nil
	}
	m.agentsModeOpen = true
	m.agentsModeHelp = false
	m.agentsModeReplyOpen = false
	m.agentsModeDeleteConfirmID = ""
	m.cancelAgentsModeRename()
	m.shortcutSheetOpen = false
	m.workflowTaskPickerOpen = false
	m.backgroundActivityPrompt = false
	m.backgroundActivityDetail = false
	m.agentsModeSelectedID = m.activeSessionID()
	m.editor.SetPlaceholder(agentsModePlaceholder)
	m.statusBar.InvalidateCache()
	return m.resizeAll()
}

func (m *appModel) closeAgentsMode() tea.Cmd {
	if !m.agentsModeOpen {
		return nil
	}
	m.agentsModeOpen = false
	m.agentsModeHelp = false
	m.agentsModeReplyOpen = false
	m.agentsModeSelectedID = ""
	m.agentsModeDeleteConfirmID = ""
	m.cancelAgentsModeRename()
	m.editor.SetPlaceholder("")
	m.statusBar.InvalidateCache()
	return m.resizeAll()
}

func (m *appModel) toggleAgentsModeHelp() tea.Cmd {
	if !m.agentsModeOpen {
		return nil
	}
	m.agentsModeHelp = !m.agentsModeHelp
	m.agentsModeDeleteConfirmID = ""
	m.cancelAgentsModeRename()
	m.statusBar.InvalidateCache()
	return m.resizeAll()
}

func (m *appModel) toggleAgentsModeGrouped() tea.Cmd {
	if !m.agentsModeOpen {
		return nil
	}
	m.agentsModeGrouped = !m.agentsModeGrouped
	m.agentsModeHelp = false
	m.agentsModeDeleteConfirmID = ""
	m.cancelAgentsModeRename()
	m.statusBar.InvalidateCache()
	return m.resizeAll()
}

func (m *appModel) toggleAgentsModePinned() tea.Cmd {
	if !m.agentsModeOpen || m.agentsModeSelectedID == "" {
		return nil
	}
	if m.agentsModePinned == nil {
		m.agentsModePinned = map[string]bool{}
	}
	m.cancelAgentsModeRename()
	if m.agentsModePinned[m.agentsModeSelectedID] {
		delete(m.agentsModePinned, m.agentsModeSelectedID)
	} else {
		m.agentsModePinned[m.agentsModeSelectedID] = true
	}
	m.agentsModeHelp = false
	m.agentsModeDeleteConfirmID = ""
	m.statusBar.InvalidateCache()
	return m.resizeAll()
}

func (m *appModel) handleAgentsModeEnter() (tea.Model, tea.Cmd) {
	if m.agentsModeRenaming {
		return m, m.saveAgentsModeRename()
	}
	prompt := strings.TrimSpace(m.editor.Value())
	if m.agentsModeReplyOpen {
		if prompt != "" {
			return m.handleAgentsModeReplySend(prompt)
		}
		return m.handleAgentsModeOpenSelected()
	}
	if prompt != "" {
		return m.handleCreateAgentsModeSession(prompt)
	}
	return m.handleAgentsModeOpenSelected()
}

func (m *appModel) handleAgentsModeOpenSelected() (tea.Model, tea.Cmd) {
	if m.agentsModeSelectedID != "" && m.agentsModeSelectedID != m.activeSessionID() {
		targetID := m.agentsModeSelectedID
		cmd := m.closeAgentsMode()
		model, switchCmd := m.handleSwitchTab(targetID)
		return model, tea.Batch(cmd, switchCmd)
	}
	return m, m.closeAgentsMode()
}

func (m *appModel) handleAgentsModeOpenAtIndex(index int) (tea.Model, tea.Cmd) {
	tabs := m.agentsModeSelectableTabs()
	if index < 0 || index >= len(tabs) {
		return m, nil
	}
	m.agentsModeSelectedID = tabs[index].SessionID
	return m.handleAgentsModeOpenSelected()
}

func (m *appModel) handleAgentsModeEscape() (tea.Model, tea.Cmd) {
	if m.agentsModeRenaming {
		return m, m.cancelAgentsModeRenameCmd()
	}
	if m.agentsModeReplyOpen {
		m.editor.SetValue("")
		return m, m.closeAgentsModeReply()
	}
	if m.editor.Value() != "" {
		m.editor.SetValue("")
		m.editorLines = m.desiredEditorLines()
		m.statusBar.InvalidateCache()
		return m, m.resizeAll()
	}
	return m, m.closeAgentsMode()
}

func (m *appModel) handleAgentsModeSpace(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	if strings.TrimSpace(m.editor.Value()) != "" {
		return m.forwardEditor(msg)
	}
	m.editor.SetValue("")
	m.agentsModeDeleteConfirmID = ""
	m.cancelAgentsModeRename()
	if m.agentsModeReplyOpen {
		return m, m.closeAgentsModeReply()
	}
	return m, m.openAgentsModeReply()
}

func (m *appModel) moveAgentsModeSelection(delta int) tea.Cmd {
	tabs := m.agentsModeSelectableTabs()
	if len(tabs) == 0 {
		return nil
	}
	idx := 0
	for i, tab := range tabs {
		if tab.SessionID == m.agentsModeSelectedID {
			idx = i
			break
		}
	}
	idx = (idx + delta + len(tabs)) % len(tabs)
	m.agentsModeSelectedID = tabs[idx].SessionID
	m.agentsModeDeleteConfirmID = ""
	m.cancelAgentsModeRename()
	m.statusBar.InvalidateCache()
	return m.resizeAll()
}

func (m *appModel) startAgentsModeRename() tea.Cmd {
	if !m.agentsModeOpen || m.agentsModeSelectedID == "" {
		return nil
	}
	m.agentsModeRenaming = true
	m.agentsModeRenameTargetID = m.agentsModeSelectedID
	m.agentsModeRenameDraft = ""
	m.agentsModeHelp = false
	m.agentsModeReplyOpen = false
	m.agentsModeDeleteConfirmID = ""
	m.editor.SetValue("")
	m.editor.SetPlaceholder(agentsModePlaceholder)
	m.statusBar.InvalidateCache()
	return m.resizeAll()
}

func (m *appModel) handleAgentsModeRenameKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "enter":
		return m, m.saveAgentsModeRename()
	case "esc":
		return m, m.cancelAgentsModeRenameCmd()
	case "backspace", "ctrl+h":
		if m.agentsModeRenameDraft != "" {
			runes := []rune(m.agentsModeRenameDraft)
			m.agentsModeRenameDraft = string(runes[:len(runes)-1])
			m.statusBar.InvalidateCache()
			return m, m.resizeAll()
		}
		return m, nil
	}
	if text := msg.Key().Text; text != "" {
		m.agentsModeRenameDraft += text
		m.statusBar.InvalidateCache()
		return m, m.resizeAll()
	}
	return m, nil
}

func (m *appModel) saveAgentsModeRename() tea.Cmd {
	if m.agentsModeRenameTargetID != "" {
		title := strings.TrimSpace(m.agentsModeRenameDraft)
		if title != "" {
			m.setAgentsModeTitle(m.agentsModeRenameTargetID, title)
			if m.supervisor != nil {
				m.supervisor.SetRunnerTitle(m.agentsModeRenameTargetID, title)
			}
		}
	}
	m.cancelAgentsModeRename()
	m.statusBar.InvalidateCache()
	return m.resizeAll()
}

func (m *appModel) cancelAgentsModeRenameCmd() tea.Cmd {
	m.cancelAgentsModeRename()
	m.statusBar.InvalidateCache()
	return m.resizeAll()
}

func (m *appModel) cancelAgentsModeRename() {
	m.agentsModeRenaming = false
	m.agentsModeRenameTargetID = ""
	m.agentsModeRenameDraft = ""
}

func (m *appModel) handleAgentsModeDelete() (tea.Model, tea.Cmd) {
	targetID := m.agentsModeSelectedID
	if targetID == "" || m.supervisor == nil {
		return m, nil
	}
	if m.agentsModeDeleteConfirmID != targetID {
		m.cancelAgentsModeRename()
		m.agentsModeDeleteConfirmID = targetID
		m.setAgentsModePending(targetID, false)
		m.setWorkflowVisible(targetID, false)
		if runner := m.supervisor.GetRunner(targetID); runner != nil && runner.App != nil {
			runner.App.Interrupt()
		}
		m.statusBar.InvalidateCache()
		return m, m.resizeAll()
	}
	m.agentsModeDeleteConfirmID = ""
	delete(m.agentsModePinned, targetID)
	model, cmd := m.handleCloseTab(targetID)
	if mm, ok := model.(*appModel); ok {
		m = mm
	}
	m.syncAgentsModeSelection()
	m.statusBar.InvalidateCache()
	return m, tea.Batch(cmd, m.resizeAll())
}

func (m *appModel) openAgentsModeReply() tea.Cmd {
	if !m.agentsModeOpen {
		return nil
	}
	m.agentsModeReplyOpen = true
	m.agentsModeHelp = false
	m.agentsModeDeleteConfirmID = ""
	m.cancelAgentsModeRename()
	m.editor.SetValue("")
	m.editor.SetPlaceholder(agentsModeReplyPlaceholder)
	m.editorLines = minEditorLines
	m.statusBar.InvalidateCache()
	return m.resizeAll()
}

func (m *appModel) closeAgentsModeReply() tea.Cmd {
	if !m.agentsModeReplyOpen {
		return nil
	}
	m.agentsModeReplyOpen = false
	m.agentsModeDeleteConfirmID = ""
	m.cancelAgentsModeRename()
	m.editor.SetValue("")
	m.editor.SetPlaceholder(agentsModePlaceholder)
	m.editorLines = m.desiredEditorLines()
	m.statusBar.InvalidateCache()
	return m.resizeAll()
}

func (m *appModel) handleAgentsModeReplySend(prompt string) (tea.Model, tea.Cmd) {
	if m.supervisor == nil || m.agentsModeSelectedID == "" {
		return m, notification.ErrorCmd("Session not found")
	}
	runner := m.supervisor.GetRunner(m.agentsModeSelectedID)
	if runner == nil || runner.App == nil {
		return m, notification.ErrorCmd("Session not found")
	}
	m.setAgentsModePending(m.agentsModeSelectedID, true)
	m.agentsModeDeleteConfirmID = ""
	m.cancelAgentsModeRename()
	m.setWorkflowVisible(m.agentsModeSelectedID, true)
	go func() {
		ctx := context.Background()
		runner.App.RunCooperative(ctx, func() {}, runner.App.ResolveInput(ctx, prompt), nil)
	}()
	return m, m.closeAgentsModeReply()
}

func (m *appModel) handleCreateAgentsModeSession(prompt string) (tea.Model, tea.Cmd) {
	if m.supervisor == nil {
		return m, notification.ErrorCmd("Session spawning is not available")
	}
	workingDir := m.currentWorkingDirectory()
	if workingDir == "" {
		if wd, err := os.Getwd(); err == nil {
			workingDir = wd
		}
	}

	ctx := context.Background()
	sessionID, err := m.supervisor.SpawnSession(ctx, workingDir)
	if err != nil {
		return m, notification.ErrorCmd("Failed to spawn session: " + err.Error())
	}
	m.supervisor.SetBackground(sessionID, true)
	m.markWorkflowSession(sessionID)
	m.setWorkflowVisible(sessionID, true)
	m.setAgentsModeTitle(sessionID, prompt)
	m.setAgentsModePending(sessionID, true)
	m.agentsModeSelectedID = sessionID

	var cmds []tea.Cmd
	if runner := m.supervisor.GetRunner(sessionID); runner != nil && runner.App != nil {
		if _, exists := m.chatPages[sessionID]; !exists {
			cp, _, ed := m.createSessionComponents(sessionID, runner.App, runner.App.Session())
			if cmd := cp.Init(); cmd != nil {
				cmds = append(cmds, cmd)
			}
			if cmd := ed.Init(); cmd != nil {
				cmds = append(cmds, cmd)
			}
		}
		go runner.App.Run(context.Background(), func() {}, prompt, nil)
	}

	m.editor.SetValue("")
	m.editorLines = m.desiredEditorLines()
	m.statusBar.InvalidateCache()
	tabs, activeIdx := m.supervisor.GetTabs()
	if m.syncTabChrome(tabs, activeIdx) {
		cmds = append(cmds, m.resizeAll())
	} else {
		cmds = append(cmds, m.resizeAll())
	}
	return m, tea.Batch(cmds...)
}

func (m *appModel) setAgentsModeTitle(sessionID, title string) {
	if sessionID == "" || title == "" {
		return
	}
	if m.agentsModeTitles == nil {
		m.agentsModeTitles = map[string]string{}
	}
	m.agentsModeTitles[sessionID] = title
}

func (m *appModel) setAgentsModePending(sessionID string, pending bool) {
	if sessionID == "" {
		return
	}
	if pending {
		if m.agentsModePending == nil {
			m.agentsModePending = map[string]bool{}
		}
		m.agentsModePending[sessionID] = true
		return
	}
	delete(m.agentsModePending, sessionID)
}

func (m *appModel) handleAgentsModeRuntimeEvent(sessionID string, msg tea.Msg) {
	switch msg.(type) {
	case *runtime.StreamStartedEvent:
		m.setAgentsModePending(sessionID, false)
	case *runtime.StreamStoppedEvent, *runtime.ErrorEvent:
		m.setAgentsModePending(sessionID, false)
	}
}

func (m *appModel) renderAgentsModeView(width, height int) string {
	if width <= 0 || height <= 0 {
		return ""
	}

	tabs := m.agentsModeTabs()
	rows := m.splitAgentsModeRows(tabs)
	cwd := compactAgentsWorkingDirectory(m.currentWorkingDirectory())
	counts := fmt.Sprintf("%d awaiting input · %d working · %d completed", len(rows.needsInput), len(rows.working), len(rows.completed))
	if m.agentsModeGrouped {
		return m.renderGroupedAgentsModeView(width, height, tabs, cwd, counts)
	}

	lines := make([]string, 0, height)
	lines = append(lines, agentsModeHeaderLines(cwd, counts)...)
	lines = append(lines, m.renderAgentsModeSection("Needs input", rows.needsInput, width)...)
	lines = append(lines, m.renderAgentsModeSection("Working", rows.working, width)...)
	lines = append(lines, m.renderAgentsModeSection("Completed", rows.completed, width)...)

	for len(lines) < height {
		lines = append(lines, "")
	}
	if len(lines) > height {
		lines = lines[:height]
	}
	for i, line := range lines {
		lines[i] = ansi.Truncate(line, width, "")
	}
	return strings.Join(lines, "\n")
}

func (m *appModel) renderGroupedAgentsModeView(width, height int, tabs []messages.TabInfo, cwd, counts string) string {
	lines := make([]string, 0, height)
	lines = append(lines, agentsModeHeaderLines(cwd, counts)...)

	pinned, grouped := m.groupedAgentsModeTabs(tabs)
	if len(pinned) > 0 {
		lines = append(lines, agentsModeSectionTitleStyle.Render("Pinned"))
		for _, tab := range pinned {
			lines = append(lines, m.renderAgentsModeRow(tab, m.agentsModeTabCompleted(tab), width))
		}
		lines = append(lines, "")
	}

	for _, group := range grouped {
		lines = append(lines, agentsModeSectionTitleStyle.Render(group.dir))
		for _, tab := range group.tabs {
			lines = append(lines, m.renderAgentsModeRow(tab, m.agentsModeTabCompleted(tab), width))
		}
		lines = append(lines, "")
	}

	for len(lines) < height {
		lines = append(lines, "")
	}
	if len(lines) > height {
		lines = lines[:height]
	}
	for i, line := range lines {
		lines[i] = ansi.Truncate(line, width, "")
	}
	return strings.Join(lines, "\n")
}

func (m *appModel) agentsModeTabCompleted(tab messages.TabInfo) bool {
	return tab.Background && !tab.NeedsAttention && !tab.IsActive && !tab.IsRunning && !m.agentsModePending[tab.SessionID]
}

type agentsModeDirGroup struct {
	dir  string
	tabs []messages.TabInfo
}

func (m *appModel) groupedAgentsModeTabs(tabs []messages.TabInfo) ([]messages.TabInfo, []agentsModeDirGroup) {
	visible := m.agentsModeVisibleTabsFrom(tabs)
	pinned := make([]messages.TabInfo, 0)
	groups := make([]agentsModeDirGroup, 0)
	groupIndex := map[string]int{}
	for _, tab := range visible {
		if tab.SessionID != "" && m.agentsModePinned[tab.SessionID] {
			pinned = append(pinned, tab)
			continue
		}
		dir := compactAgentsWorkingDirectory(m.agentsModeWorkingDir(tab))
		if dir == "" {
			dir = "."
		}
		idx, ok := groupIndex[dir]
		if !ok {
			idx = len(groups)
			groupIndex[dir] = idx
			groups = append(groups, agentsModeDirGroup{dir: dir})
		}
		groups[idx].tabs = append(groups[idx].tabs, tab)
	}
	return pinned, groups
}

func (m *appModel) agentsModeTabs() []messages.TabInfo {
	if m.supervisor == nil {
		return []messages.TabInfo{m.currentAgentsModeTab()}
	}
	tabs, _ := m.supervisor.GetTabs()
	if len(tabs) == 0 {
		return []messages.TabInfo{m.currentAgentsModeTab()}
	}
	for i := range tabs {
		if title := m.agentsModeTitles[tabs[i].SessionID]; title != "" {
			tabs[i].Title = title
		}
	}
	return tabs
}

func (m *appModel) agentsModeVisibleTabs() []messages.TabInfo {
	return m.agentsModeVisibleTabsFrom(m.agentsModeTabs())
}

func (m *appModel) agentsModeSelectableTabs() []messages.TabInfo {
	return m.agentsModeVisibleTabs()
}

func (m *appModel) agentsModeVisibleTabsFrom(tabs []messages.TabInfo) []messages.TabInfo {
	rows := m.splitAgentsModeRows(tabs)
	visible := make([]messages.TabInfo, 0, len(rows.needsInput)+len(rows.working)+len(rows.completed))
	visible = append(visible, rows.needsInput...)
	visible = append(visible, rows.working...)
	visible = append(visible, rows.completed...)
	return visible
}

func (m *appModel) syncAgentsModeSelection() {
	tabs := m.agentsModeSelectableTabs()
	if len(tabs) == 0 {
		m.agentsModeSelectedID = ""
		return
	}
	for _, tab := range tabs {
		if tab.SessionID == m.agentsModeSelectedID {
			return
		}
	}
	m.agentsModeSelectedID = tabs[0].SessionID
}

func (m *appModel) currentAgentsModeTab() messages.TabInfo {
	createdAt := time.Now().Unix()
	workingDir := m.currentWorkingDirectory()
	if m.application != nil {
		if sess := m.application.Session(); sess != nil {
			if !sess.CreatedAt.IsZero() {
				createdAt = sess.CreatedAt.Unix()
			}
			if sess.WorkingDir != "" {
				workingDir = sess.WorkingDir
			}
		}
	}
	return messages.TabInfo{
		Title:      "current session",
		IsActive:   true,
		CreatedAt:  createdAt,
		WorkingDir: workingDir,
	}
}

func (m *appModel) agentsModeWorkingDir(tab messages.TabInfo) string {
	if tab.WorkingDir != "" {
		return tab.WorkingDir
	}
	if tab.IsActive {
		return m.currentWorkingDirectory()
	}
	if m.supervisor != nil && tab.SessionID != "" {
		if runner := m.supervisor.GetRunner(tab.SessionID); runner != nil {
			return runner.WorkingDir
		}
	}
	return m.currentWorkingDirectory()
}

func (m *appModel) splitAgentsModeRows(tabs []messages.TabInfo) agentsModeRows {
	rows := agentsModeRows{}
	for _, tab := range tabs {
		switch {
		case tab.IsRunning || m.agentsModePending[tab.SessionID]:
			rows.working = append(rows.working, tab)
		case tab.Background && !tab.NeedsAttention && !tab.IsActive:
			rows.completed = append(rows.completed, tab)
		default:
			rows.needsInput = append(rows.needsInput, tab)
		}
	}
	return rows
}

func agentsModeHeaderLines(cwd, counts string) []string {
	logo := lipgloss.NewStyle().Foreground(lipgloss.Color("174"))
	logoFill := logo.Background(lipgloss.Color("16"))
	muted := lipgloss.NewStyle().Foreground(lipgloss.Color("246"))
	bold := lipgloss.NewStyle().Bold(true)
	return []string{
		"",
		logo.Render(" ▐") + logoFill.Render("▛███▜") + logo.Render("▌") + "   " + bold.Render("Claude Code") + " " + muted.Render("v2.1.201"),
		logo.Render("▝▜") + logoFill.Render("█████") + logo.Render("▛▘") + "  " + muted.Render("Opus 4.8 (1M context) · "+cwd),
		logo.Render("  ▘▘ ▝▝  ") + "  " + muted.Render(counts),
		"",
	}
}

func (m *appModel) renderAgentsModeSection(title string, tabs []messages.TabInfo, width int) []string {
	if len(tabs) == 0 {
		return nil
	}
	lines := []string{agentsModeSectionTitleStyle.Render(title)}
	for _, tab := range tabs {
		lines = append(lines, m.renderAgentsModeRow(tab, title == "Completed", width))
	}
	lines = append(lines, "")
	return lines
}

func (m *appModel) renderAgentsModeRow(tab messages.TabInfo, completed bool, width int) string {
	confirmingDelete := tab.SessionID != "" && tab.SessionID == m.agentsModeDeleteConfirmID
	selected := tab.SessionID != "" && tab.SessionID == m.agentsModeSelectedID
	icon := "✻"
	if completed || confirmingDelete {
		icon = "∙"
	}

	label := tab.Title
	if tab.IsActive {
		label = "current session"
	}
	if m.agentsModeRenaming && tab.SessionID == m.agentsModeRenameTargetID {
		label = m.agentsModeRenameDraft
	} else if label == "" {
		label = "session"
	}
	label = agentsModePadRight(ansi.Truncate(label, 26, "…"), 27)

	meta := m.agentsModeRowMeta(tab, completed)

	iconStyle := agentsModeRowTextStyle
	if icon == "✻" {
		iconStyle = agentsModeActiveIconStyle
	}
	labelStyle := agentsModeRowTextStyle
	if selected {
		iconStyle = iconStyle.Background(lipgloss.Color("237"))
		labelStyle = agentsModeSelectedRowTextStyle
		if icon == "✻" {
			iconStyle = agentsModeSelectedActiveIconStyle
		}
	}

	metaStyle := agentsModeRowTextStyle
	if confirmingDelete {
		metaStyle = agentsModeDeleteMetaStyle
	}
	if selected {
		metaStyle = metaStyle.Background(lipgloss.Color("237"))
	}

	left := iconStyle.Render(icon) + labelStyle.Render(" "+label) + metaStyle.Render(meta)
	right := agentsModeAge(tab.CreatedAt)
	return agentsModeLineWithRight(left, right, width, selected)
}

func (m *appModel) agentsModeRowMeta(tab messages.TabInfo, completed bool) string {
	if tab.SessionID != "" && tab.SessionID == m.agentsModeDeleteConfirmID {
		return "stopped · ctrl+x again to delete"
	}
	if tab.IsActive {
		return filepath.Base(compactAgentsWorkingDirectory(m.currentWorkingDirectory())) + " · →"
	}
	if m.agentsModePending[tab.SessionID] {
		return cmpNonEmpty(m.agentsModeTitles[tab.SessionID], "working")
	}
	if tab.NeedsAttention {
		return "awaiting input · →"
	}
	if tab.IsRunning {
		return cmpNonEmpty(m.agentsModeTitles[tab.SessionID], "working")
	}
	if completed {
		return "stopped"
	}
	return "send a prompt to start"
}

func (m *appModel) selectedAgentsModeTab() (messages.TabInfo, bool) {
	if m.agentsModeSelectedID == "" {
		return messages.TabInfo{}, false
	}
	for _, tab := range m.agentsModeTabs() {
		if tab.SessionID == m.agentsModeSelectedID {
			return tab, true
		}
	}
	return messages.TabInfo{}, false
}

func (m *appModel) renderAgentsModeReplyComposer(width int) string {
	if width <= 0 {
		return ""
	}
	inner := max(1, width-2)
	border := strings.Repeat("─", inner)
	summary := m.agentsModeReplySummary()
	input := m.editor.Value()
	if input == "" {
		input = styles.SecondaryStyle.Render(agentsModeReplyPlaceholder)
	}
	return strings.Join([]string{
		"╭" + border + "╮",
		agentsModeBoxLine(" "+summary, width),
		agentsModeBoxLine("", width),
		agentsModeBoxLine(" ❯ "+input, width),
		"╰" + border + "╯",
	}, "\n")
}

func (m *appModel) agentsModeReplySummary() string {
	tab, ok := m.selectedAgentsModeTab()
	if !ok {
		return ""
	}
	completed := tab.Background && !tab.IsActive && !tab.IsRunning && !tab.NeedsAttention && !m.agentsModePending[tab.SessionID]
	meta := m.agentsModeRowMeta(tab, completed)
	age := agentsModeAge(tab.CreatedAt)
	if meta == "" {
		return age
	}
	return strings.TrimSpace(age + " " + meta)
}

func agentsModeBoxLine(content string, width int) string {
	inner := max(0, width-2)
	content = ansi.Truncate(content, inner, "")
	padding := max(0, inner-lipgloss.Width(content))
	return "│" + content + strings.Repeat(" ", padding) + "│"
}

func (m *appModel) activeSessionID() string {
	if m.supervisor == nil {
		return ""
	}
	return m.supervisor.ActiveID()
}

func agentsModeLineWithRight(left, right string, width int, selected bool) string {
	if width <= 0 {
		return ""
	}
	rightStyle := agentsModeRowTextStyle
	if selected {
		rightStyle = rightStyle.Background(lipgloss.Color("237"))
	}
	right = rightStyle.Render(right)
	leftWidth := lipgloss.Width(left)
	rightWidth := lipgloss.Width(right)
	if leftWidth+rightWidth+1 > width {
		left = ansi.Truncate(left, max(1, width-rightWidth-1), "…")
		leftWidth = lipgloss.Width(left)
	}
	padding := max(1, width-leftWidth-rightWidth)
	gap := strings.Repeat(" ", padding)
	if selected {
		gap = agentsModeSelectedCellStyle.Render(gap)
	}
	return left + gap + right
}

func agentsModePadRight(s string, width int) string {
	if lipgloss.Width(s) >= width {
		return s + " "
	}
	return s + strings.Repeat(" ", width-lipgloss.Width(s))
}

func agentsModeAge(createdAt int64) string {
	if createdAt <= 0 {
		return "0s"
	}
	elapsed := time.Since(time.Unix(createdAt, 0))
	if elapsed < 0 {
		elapsed = 0
	}
	switch {
	case elapsed < time.Minute:
		return fmt.Sprintf("%ds", int(elapsed.Seconds()))
	case elapsed < time.Hour:
		return fmt.Sprintf("%dm", int(elapsed.Minutes()))
	case elapsed < 24*time.Hour:
		return fmt.Sprintf("%dh", int(elapsed.Hours()))
	default:
		return fmt.Sprintf("%dd", int(elapsed.Hours()/24))
	}
}

func (m *appModel) currentWorkingDirectory() string {
	if m.application != nil {
		if sess := m.application.Session(); sess != nil && sess.WorkingDir != "" {
			return sess.WorkingDir
		}
	}
	if wd, err := os.Getwd(); err == nil {
		return wd
	}
	return ""
}

func compactAgentsWorkingDirectory(path string) string {
	if path == "" {
		return ""
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		if path == home {
			return "~"
		}
		if strings.HasPrefix(path, home+string(os.PathSeparator)) {
			return "~" + strings.TrimPrefix(path, home)
		}
	}
	return path
}
