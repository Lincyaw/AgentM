package tui

import (
	"fmt"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/runtime"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/messages"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
)

const (
	workflowTaskPickerBaseFooter = "↑/↓ to select · Enter to view"
)

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
	return m.bottomActivityKind().hasWorkflowTasks()
}

func (m *appModel) hasBottomActivityRows() bool {
	return m.bottomActivityKind().hasRows()
}

func (m *appModel) activeIsWorkflowTask() bool {
	if m.supervisor == nil {
		return false
	}
	activeID := m.supervisor.ActiveID()
	for _, tab := range m.workflowTaskTabs() {
		if tab.SessionID == activeID {
			return true
		}
	}
	return false
}

func (m *appModel) workflowTaskTabs() []messages.TabInfo {
	if m.supervisor == nil {
		return nil
	}
	tabs, _ := m.supervisor.GetTabs()
	tasks := make([]messages.TabInfo, 0, len(tabs))
	for _, tab := range tabs {
		if tab.SessionID == m.mainSessionID {
			continue
		}
		if tab.Background && m.workflowTaskVisible(tab) {
			tasks = append(tasks, tab)
		}
	}
	return tasks
}

func (m *appModel) workflowTaskVisible(tab messages.TabInfo) bool {
	return tab.IsRunning || tab.NeedsAttention || m.workflowVisible[tab.SessionID]
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
	for _, tab := range m.workflowTaskTabs() {
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
	return workflowTaskPickerBaseFooter + " · x to stop task"
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
		m.setWorkflowVisible(sessionID, true)
	case *runtime.StreamStoppedEvent:
		m.setWorkflowVisible(sessionID, false)
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
	m.statusBar.SetActivity(m.backgroundActivityText())
	m.statusBar.InvalidateCache()
	return m.resizeAll()
}

func (m *appModel) toggleShortcutSheet() tea.Cmd {
	m.shortcutSheetOpen = !m.shortcutSheetOpen
	if m.shortcutSheetOpen {
		m.workflowTaskPickerOpen = false
	}
	m.statusBar.InvalidateCache()
	return m.resizeAll()
}

func (m *appModel) closeInlineSurfaces() tea.Cmd {
	changed := m.shortcutSheetOpen || m.workflowTaskPickerOpen
	m.shortcutSheetOpen = false
	m.workflowTaskPickerOpen = false
	if !changed {
		return nil
	}
	m.statusBar.InvalidateCache()
	return m.resizeAll()
}

func (m *appModel) toggleTranscriptDetailed() tea.Cmd {
	m.transcriptDetailed = !m.transcriptDetailed
	if !m.transcriptDetailed {
		m.transcriptVerbose = false
	}
	m.statusBar.InvalidateCache()
	return core.CmdHandler(messages.SetTranscriptDetailMsg{
		Detailed: m.transcriptDetailed,
		Verbose:  m.transcriptVerbose,
	})
}

func (m *appModel) toggleTranscriptVerbose() tea.Cmd {
	if !m.transcriptDetailed {
		m.transcriptDetailed = true
	}
	m.transcriptVerbose = !m.transcriptVerbose
	m.statusBar.InvalidateCache()
	return core.CmdHandler(messages.SetTranscriptDetailMsg{
		Detailed: m.transcriptDetailed,
		Verbose:  m.transcriptVerbose,
	})
}

func (m *appModel) footerText() string {
	activityKind := m.bottomActivityKind()
	if time.Since(m.lastExitRequest) <= 2*time.Second {
		return "Press Ctrl-C again to exit"
	}
	switch {
	case m.transcriptDetailed && m.transcriptVerbose:
		return "Showing verbose transcript · ctrl+o to toggle · ctrl+e to collapse verbose"
	case m.transcriptDetailed:
		return "Showing detailed transcript · ctrl+o to toggle · ctrl+e to show all verbose"
	case m.workflowTaskPickerOpen:
		return m.workflowTaskPickerFooterText()
	case m.activeIsWorkflowTask():
		return "ctrl+n/p to switch tabs · ↓ to manage tasks"
	case activityKind.hasRows():
		verb := "hide"
		if m.bottomActivityRowsHidden {
			verb = "show"
		}
		target := activityKind.toggleTarget()
		if activityKind.hasWorkflowTasks() {
			return "ctrl+t to " + verb + " " + target + " · ← for agents · ↓ to manage"
		}
		return "ctrl+t to " + verb + " " + target + " · ← for agents"
	case m.sessionState != nil && m.sessionState.YoloMode():
		return "bypass permissions on (shift+tab to cycle) · ← for agents"
	default:
		return "? for shortcuts · ← for agents"
	}
}

func (m *appModel) ctrlTActionLabel() string {
	return m.bottomActivityKind().ctrlTActionLabel()
}

func (m *appModel) footerRightText() string {
	if time.Since(m.lastEscClearRequest) <= 2*time.Second {
		return "Esc again to clear"
	}
	return ""
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
		return lipgloss.NewStyle().Padding(0, styles.AppPadding).Render(m.completions.View())
	}
	var parts []string
	if m.shortcutSheetOpen {
		parts = append(parts, m.renderShortcutSheet(width))
	}
	if rows := m.renderBottomActivityRows(width); rows != "" {
		parts = append(parts, rows)
	}
	return strings.Join(parts, "\n")
}

func (m *appModel) renderShortcutSheet(width int) string {
	innerWidth := max(20, width-appPaddingHorizontal)
	rows := [][3]string{
		{"! for shell mode", "double tap esc to clear input", "ctrl + shift + _ to undo"},
		{"/ for commands", "shift + tab to auto-accept edits", "ctrl + z to suspend"},
		{"@ for file paths", "ctrl + o for verbose output", "ctrl + v to paste images"},
		{"/btw for side question", "ctrl + t to " + m.ctrlTActionLabel(), "opt + p to switch model"},
		{"", "shift + ⏎ for newline", "ctrl + s to stash prompt"},
		{"", "", "ctrl + g to edit in $EDITOR"},
		{"", "", "/keybindings to customize"},
	}
	lines := make([]string, 0, len(rows))
	for _, row := range rows {
		line := fmt.Sprintf("%-28s  %-36s  %s", row[0], row[1], row[2])
		lines = append(lines, styles.SecondaryStyle.Render(ansi.Truncate(line, innerWidth, "")))
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
	if !activityKind.hasRows() || (m.bottomActivityRowsHidden && !m.workflowTaskPickerOpen) {
		return ""
	}
	var workflowRows []workflowRow
	if activityKind.hasWorkflowTasks() {
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
