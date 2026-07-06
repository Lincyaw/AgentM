package tui

import (
	"strings"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/spinner"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
)

// layoutRegion represents a vertical region in the TUI layout.
type layoutRegion int

const (
	regionContent layoutRegion = iota
	regionResizeHandle
	regionTabBar
	regionEditor
	regionStatusBar
)

// hitTestRegion determines which layout region a Y coordinate falls in.
func (m *appModel) hitTestRegion(y int) layoutRegion {
	_, editorHeight := m.editor.GetSize()
	return hitTestFullRegion(y, m.contentHeight, m.tabBarHeight(), editorHeight+m.composerBottomBorderHeight())
}

// hitTestFullRegion is the pure layout calculation used in full mode where the
// screen is content | resize handle | [tab bar] | editor/composer border | status bar.
// It is exported as a free function (rather than a method) so that it can be
// unit-tested without constructing a full appModel.
func hitTestFullRegion(y, contentHeight, tabBarHeight, editorHeight int) layoutRegion {
	resizeHandleTop := contentHeight
	tabBarTop := resizeHandleTop + 1
	editorTop := tabBarTop + tabBarHeight

	switch {
	case y < resizeHandleTop:
		return regionContent
	case y < tabBarTop:
		return regionResizeHandle
	case y < editorTop:
		return regionTabBar
	default:
		if y < editorTop+editorHeight {
			return regionEditor
		}
		return regionStatusBar
	}
}

// editorTop returns the Y coordinate where the editor starts.
func (m *appModel) editorTop() int {
	return m.contentHeight + 1 + m.tabBarHeight()
}

// handleEditorResize adjusts editor height based on drag position.
func (m *appModel) handleEditorResize(y int) tea.Cmd {
	editorPadding := styles.EditorStyle.GetVerticalFrameSize()
	targetLines := m.height - y - 1 - editorPadding - m.tabBarHeight() - m.bottomSurfaceLayoutHeight
	newLines := max(minEditorLines, min(targetLines, m.maxEditorLines()))
	if newLines != m.editorLines {
		m.editorLines = newLines
		return m.resizeAll()
	}
	return nil
}

// renderResizeHandle renders the draggable separator between content and bottom panel.
func (m *appModel) renderResizeHandle(width int) string {
	if width <= 0 {
		return ""
	}
	if m.agentsModeReplyOpen || m.localPanelOpen {
		return ""
	}

	innerWidth := width

	centerStyle := m.composerBorderStyle()
	if m.isDragging {
		centerStyle = styles.ResizeHandleActiveStyle
	} else if m.isHoveringHandle {
		centerStyle = styles.ResizeHandleHoverStyle
	}

	centerPart := strings.Repeat("─", min(resizeHandleWidth, innerWidth))
	handle := centerStyle.Render(centerPart)

	fullLine := lipgloss.PlaceHorizontal(
		max(0, innerWidth), lipgloss.Center, handle,
		lipgloss.WithWhitespaceChars("─"),
		lipgloss.WithWhitespaceStyle(m.composerBorderStyle()),
	)

	if m.editor != nil {
		if label := m.editor.HistoryNavigationLabel(); label != "" {
			prefix := m.composerBorderStyle().Render("───")
			styledLabel := styles.SecondaryStyle.Render(" " + label + " ")
			suffixWidth := max(0, innerWidth-lipgloss.Width(prefix)-lipgloss.Width(styledLabel))
			return prefix + styledLabel + m.composerBorderStyle().Render(strings.Repeat("─", suffixWidth))
		}
	}

	if title := renderableComposerTitle(m.composerDividerTitle()); title != "" {
		label := styles.SecondaryStyle.Render(" " + title + " ──")
		labelWidth := lipgloss.Width(label)
		if labelWidth < innerWidth {
			fullLine = m.composerBorderStyle().Render(strings.Repeat("─", innerWidth-labelWidth)) + label
		}
	}

	return fullLine
}

func (m *appModel) composerDividerTitle() string {
	if m.branchLabel != "" {
		return m.branchLabel
	}
	return m.sessionState.SessionTitle()
}

func branchLabelFromNotice(content string) (string, bool) {
	const prefix = "Branched conversation \""
	if !strings.HasPrefix(content, prefix) {
		return "", false
	}
	rest := strings.TrimPrefix(content, prefix)
	end := strings.Index(rest, "\"")
	if end <= 0 {
		return "", false
	}
	return strings.TrimSpace(rest[:end]), true
}

func renderableComposerTitle(title string) string {
	title = strings.TrimSpace(title)
	switch strings.ToLower(title) {
	case "", "agentm", "agentm (mock)", "agentm terminal":
		return ""
	default:
		return title
	}
}

func (m *appModel) composerBottomBorderHeight() int {
	if m.agentsModeReplyOpen || m.localPanelOpen || m.transcriptDetailed {
		return 0
	}
	return 1
}

func (m *appModel) renderComposerBottomBorder(width int) string {
	if width <= 0 || m.composerBottomBorderHeight() == 0 {
		return ""
	}
	innerWidth := max(0, width)
	line := m.composerBorderStyle().Render(strings.Repeat("─", innerWidth))
	return line
}

func (m *appModel) composerBorderStyle() lipgloss.Style {
	if m.editor != nil && m.editor.ShellMode() {
		return lipgloss.NewStyle().Foreground(lipgloss.Color("211"))
	}
	return styles.ResizeHandleStyle
}

// View renders the model.
func (m *appModel) View() tea.View {
	m.syncWelcomeModelLine()
	windowTitle := m.windowTitle()

	if m.err != nil {
		return toFullscreenView(styles.ErrorStyle.Render(m.err.Error()), windowTitle, false)
	}

	if !m.ready {
		return toFullscreenView(
			styles.CenterStyle.
				Width(m.wWidth).
				Height(m.wHeight).
				Render(styles.MutedStyle.Render("Loading…")),
			windowTitle,
			false,
		)
	}

	contentView := m.chatPage.View()
	if m.agentsModeOpen {
		contentView = m.renderAgentsModeView(m.width, m.contentHeight)
	}
	if m.activeIsWorkflowTask() {
		if taskDetail := m.renderWorkflowDetail(m.width, m.contentHeight); taskDetail != "" {
			contentView = taskDetail
		}
	}

	resizeHandle := m.renderResizeHandle(m.width)

	tabBarView := ""
	if m.tabBarHeight() > 0 {
		tabBarView = m.tabBar.View()
	}

	editorView := ""
	if !m.localPanelOpen && !m.transcriptDetailed {
		editorView = m.editor.View()
		if m.agentsModeReplyOpen {
			editorView = m.renderAgentsModeReplyComposer(m.width)
		}
	}
	composerBottomBorderView := m.renderComposerBottomBorder(m.width)

	statusBarView := ""
	if !m.localPanelOpen {
		m.statusBar.SetActivity(m.footerActivityText())
		m.statusBar.SetModeLine(m.footerText())
		m.statusBar.SetModeLineRight(m.footerRightText())
	}
	if !m.localPanelOpen && m.statusBarHeight() > 0 {
		statusBarView = m.statusBar.View()
		if extraLines := m.footerExtraLines(); len(extraLines) > 0 {
			statusBarView = lipgloss.JoinVertical(lipgloss.Top, append([]string{statusBarView}, extraLines...)...)
		}
	}
	bottomSurfaceView := ""
	if !m.localPanelOpen {
		bottomSurfaceView = m.renderBottomSurface(m.width)
	}
	agentsCompletionSurface := ""
	if m.agentsModeOpen && m.completions.Open() {
		agentsCompletionSurface = bottomSurfaceView
		bottomSurfaceView = ""
	}

	viewParts := []string{
		contentView,
	}
	if agentsCompletionSurface != "" {
		viewParts = append(viewParts, agentsCompletionSurface)
	}
	if resizeHandle != "" {
		viewParts = append(viewParts, resizeHandle)
	}
	if tabBarView != "" {
		viewParts = append(viewParts, lipgloss.NewStyle().
			Padding(0, styles.AppPadding).
			Render(tabBarView))
	}
	if editorView != "" {
		viewParts = append(viewParts, editorView)
	}
	if composerBottomBorderView != "" {
		viewParts = append(viewParts, composerBottomBorderView)
	}
	if bottomSurfaceView != "" && m.backgroundActivityDetail {
		viewParts = append(viewParts, bottomSurfaceView)
	}
	if statusBarView != "" {
		viewParts = append(viewParts, statusBarView)
	}
	if bottomSurfaceView != "" && !m.backgroundActivityDetail {
		viewParts = append(viewParts, bottomSurfaceView)
	}
	baseView := lipgloss.JoinVertical(lipgloss.Top, viewParts...)

	hasOverlays := m.dialogMgr.Open() || m.notification.Open()

	if hasOverlays {
		baseLayer := lipgloss.NewLayer(baseView)
		var allLayers []*lipgloss.Layer
		allLayers = append(allLayers, baseLayer)

		if m.dialogMgr.Open() {
			dialogLayers := m.dialogMgr.GetLayers()
			allLayers = append(allLayers, dialogLayers...)
		}

		if m.notification.Open() {
			allLayers = append(allLayers, m.notification.GetLayer())
		}

		compositor := lipgloss.NewCompositor(allLayers...)
		return toFullscreenView(compositor.Render(), windowTitle, m.chatPage.IsWorking())
	}

	return toFullscreenView(baseView, windowTitle, m.chatPage.IsWorking())
}

// windowTitle returns the terminal window title for the current model state.
// When the agent is working, a rotating spinner character is prepended so that
// terminal multiplexers (tmux) can detect activity in the pane.
func (m *appModel) windowTitle() string {
	return formatWindowTitle(m.appName, m.sessionState.SessionTitle(), m.chatPage.IsWorking(), m.animFrame)
}

// formatWindowTitle assembles the terminal window title string from the
// individual inputs that contribute to it. Pure function extracted from the
// windowTitle method so that it can be unit-tested without constructing a full
// appModel.
func formatWindowTitle(appName, sessionTitle string, working bool, animFrame int) string {
	title := appName
	if sessionTitle != "" {
		title = sessionTitle + " - " + appName
	}
	if working {
		title = spinner.Frame(animFrame) + " " + title
	}
	return title
}

func toFullscreenView(content, windowTitle string, working bool) tea.View {
	view := tea.NewView(content)
	view.DisableBracketedPasteMode = false
	view.MouseMode = tea.MouseModeAllMotion
	view.BackgroundColor = styles.Background
	view.WindowTitle = windowTitle
	if working {
		view.ProgressBar = tea.NewProgressBar(tea.ProgressBarIndeterminate, 0)
	}
	return view
}
