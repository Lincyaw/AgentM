package tui

import (
	tea "charm.land/bubbletea/v2"

	"github.com/AoyangSpace/agentm-terminal/internal/tui/messages"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
)

// handleMouseClick routes mouse clicks to the appropriate component based on Y coordinate.
func (m *appModel) handleMouseClick(msg tea.MouseClickMsg) (tea.Model, tea.Cmd) {
	if cmd := m.notification.HandleClick(msg.X, msg.Y); cmd != nil {
		return m, cmd
	}
	if id, text, ok := m.notification.CopyHit(msg.X, msg.Y); ok {
		return m, copyNotificationToClipboard(id, text)
	}

	if m.dialogMgr.Open() {
		// Background dialogs let tab-bar clicks pass through so the user can
		// keep navigating between tabs.
		if m.dialogMgr.TopIsBackground() && m.hitTestRegion(msg.Y) == regionTabBar {
			adjustedMsg := msg
			adjustedMsg.X = msg.X - styles.AppPadding
			adjustedMsg.Y = msg.Y - m.contentHeight - 1
			if cmd := m.tabBar.Update(adjustedMsg); cmd != nil {
				return m, cmd
			}
			return m, nil
		}
		return m.forwardDialog(msg)
	}

	region := m.hitTestRegion(msg.Y)

	switch region {
	case regionContent:
		return m.forwardChat(msg)

	case regionResizeHandle:
		if msg.Button == tea.MouseLeft {
			m.isDragging = true
		}
		return m, nil

	case regionTabBar:
		adjustedMsg := msg
		adjustedMsg.X = msg.X - styles.AppPadding
		adjustedMsg.Y = msg.Y - m.contentHeight - 1
		if cmd := m.tabBar.Update(adjustedMsg); cmd != nil {
			return m, cmd
		}
		return m, nil

	case regionEditor:
		if m.focusedPanel != PanelEditor {
			m.focusedPanel = PanelEditor
			m.statusBar.InvalidateCache()
			m.chatPage.BlurMessages()
		}
		adjustedMsg := msg
		adjustedMsg.X = msg.X - styles.AppPadding
		adjustedMsg.Y = msg.Y - m.editorTop()
		return m, tea.Batch(m.updateEditorCmd(adjustedMsg), m.editor.Focus())
	}

	return m, nil
}

// handleMouseMotion routes mouse motion events with adjusted coordinates.
func (m *appModel) handleMouseMotion(msg tea.MouseMotionMsg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd
	batchWith := func(cmd tea.Cmd) tea.Cmd {
		if cmd != nil {
			cmds = append(cmds, cmd)
		}
		return tea.Batch(cmds...)
	}

	updated, cmd := m.notification.HandleMouseMotion(msg.X, msg.Y)
	m.notification = updated
	if cmd != nil {
		cmds = append(cmds, cmd)
	}

	if m.isDragging {
		cmd := m.handleEditorResize(msg.Y)
		return m, batchWith(cmd)
	}

	if m.tabBar.IsDragging() {
		adjustedMsg := msg
		adjustedMsg.X = msg.X - styles.AppPadding
		if cmd := m.tabBar.Update(adjustedMsg); cmd != nil {
			return m, batchWith(cmd)
		}
		return m, tea.Batch(cmds...)
	}

	if m.dialogMgr.Open() {
		model, cmd := m.forwardDialog(msg)
		return model, batchWith(cmd)
	}

	region := m.hitTestRegion(msg.Y)
	m.isHoveringHandle = region == regionResizeHandle
	switch region {
	case regionContent:
		model, cmd := m.forwardChat(msg)
		return model, batchWith(cmd)
	case regionEditor:
		adjustedMsg := msg
		adjustedMsg.X = msg.X - styles.AppPadding
		adjustedMsg.Y = msg.Y - m.editorTop()
		model, cmd := m.forwardEditor(adjustedMsg)
		return model, batchWith(cmd)
	}

	return m, tea.Batch(cmds...)
}

// handleMouseRelease routes mouse release events with adjusted coordinates.
func (m *appModel) handleMouseRelease(msg tea.MouseReleaseMsg) (tea.Model, tea.Cmd) {
	if m.isDragging {
		m.isDragging = false
		return m, nil
	}

	if m.tabBar.IsDragging() {
		adjustedMsg := msg
		adjustedMsg.X = msg.X - styles.AppPadding
		if cmd := m.tabBar.Update(adjustedMsg); cmd != nil {
			return m, cmd
		}
		return m, nil
	}

	if m.dialogMgr.Open() {
		return m.forwardDialog(msg)
	}

	region := m.hitTestRegion(msg.Y)
	switch region {
	case regionContent:
		return m.forwardChat(msg)
	case regionEditor:
		adjustedMsg := msg
		adjustedMsg.X = msg.X - styles.AppPadding
		adjustedMsg.Y = msg.Y - m.editorTop()
		return m.forwardEditor(adjustedMsg)
	}

	return m, nil
}

// handleWheelCoalesced routes coalesced wheel events with adjusted coordinates.
func (m *appModel) handleWheelCoalesced(msg messages.WheelCoalescedMsg) (tea.Model, tea.Cmd) {
	if msg.Delta == 0 {
		return m, nil
	}

	if m.dialogMgr.Open() {
		return m.forwardDialog(msg)
	}

	region := m.hitTestRegion(msg.Y)
	switch region {
	case regionContent:
		return m.forwardChat(msg)
	case regionEditor:
		m.editor.ScrollByWheel(msg.Delta)
		return m, nil
	}

	return m, nil
}
