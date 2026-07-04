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
	return hitTestFullRegion(y, m.contentHeight, m.tabBarHeight(), editorHeight)
}

// hitTestFullRegion is the pure layout calculation used in full mode where the
// screen is content | resize handle | [tab bar] | editor | status bar.
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
	minLines := minEditorLines
	maxLines := max(minLines, (m.height-6)/2)
	newLines := max(minLines, min(targetLines, maxLines))
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

	innerWidth := width - appPaddingHorizontal

	centerStyle := styles.ResizeHandleHoverStyle
	if m.isDragging {
		centerStyle = styles.ResizeHandleActiveStyle
	}

	centerPart := strings.Repeat("─", min(resizeHandleWidth, innerWidth))
	handle := centerStyle.Render(centerPart)

	fullLine := lipgloss.PlaceHorizontal(
		max(0, innerWidth), lipgloss.Center, handle,
		lipgloss.WithWhitespaceChars("─"),
		lipgloss.WithWhitespaceStyle(styles.ResizeHandleStyle),
	)

	var result string
	switch {
	case m.chatPage.IsWorking():
		workingText := "Working…"
		suffix := " " + m.workingSpinner.View() + " " + styles.SpinnerDotsHighlightStyle.Render(workingText)
		cancelKeyPart := styles.HighlightWhiteStyle.Render("Esc")
		suffix += " (" + cancelKeyPart + " to interrupt)"
		suffixWidth := lipgloss.Width(suffix)
		result = lipgloss.NewStyle().MaxWidth(innerWidth-suffixWidth).Render(fullLine) + suffix

	default:
		result = fullLine
	}

	return lipgloss.NewStyle().Padding(0, styles.AppPadding).Render(result)
}

// View renders the model.
func (m *appModel) View() tea.View {
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

	editorView := m.editor.View()

	m.statusBar.SetModeLine(m.footerText())
	m.statusBar.SetModeLineRight(m.footerRightText())
	statusBarView := ""
	if m.statusBarHeight() > 0 {
		statusBarView = m.statusBar.View()
	}
	bottomSurfaceView := m.renderBottomSurface(m.width)

	viewParts := []string{
		contentView,
		resizeHandle,
	}
	if tabBarView != "" {
		viewParts = append(viewParts, lipgloss.NewStyle().
			Padding(0, styles.AppPadding).
			Render(tabBarView))
	}
	viewParts = append(viewParts, editorView)
	if statusBarView != "" {
		viewParts = append(viewParts, statusBarView)
	}
	if bottomSurfaceView != "" {
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
	view.AltScreen = true
	view.MouseMode = tea.MouseModeAllMotion
	view.BackgroundColor = styles.Background
	view.WindowTitle = windowTitle
	if working {
		view.ProgressBar = tea.NewProgressBar(tea.ProgressBarIndeterminate, 0)
	}
	return view
}
