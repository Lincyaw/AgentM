package tui

import (
	"strings"
	"time"
	"unicode"

	tea "charm.land/bubbletea/v2"

	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/completion"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/editor"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/dialog"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/messages"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/page/chat"
)

// The Bubble Tea Update contract returns a tea.Model interface, which forces
// us to type-assert the result back to the concrete sub-model type before
// reassigning it. The helpers below centralise that boilerplate.
//
// Two flavours are provided:
//
//   - updateXCmd: forwards a message to a sub-model and returns the produced
//     tea.Cmd. Use this when several sub-models are updated within the same
//     handler so their commands can be batched.
//
//   - forwardX: same forwarding, but returns (m, cmd) so that handlers that
//     only update a single sub-model can simply `return m.forwardX(msg)`.
//     This avoids the gocritic evalOrder warning that fires on
//     `return m, m.updateX(msg)` because m is mutated by the call.

// updateChatCmd forwards a message to the chat page and returns its cmd.
func (m *appModel) updateChatCmd(msg tea.Msg) tea.Cmd {
	updated, cmd := m.chatPage.Update(msg)
	m.chatPage = updated.(chat.Page)
	activeID := ""
	if m.supervisor != nil {
		if activeID = m.supervisor.ActiveID(); activeID != "" {
			m.chatPages[activeID] = m.chatPage
		}
	}
	if send, ok := msg.(messages.SendMsg); ok && send.QueueIfBusy && m.chatPage.IsWorking() {
		return tea.Batch(cmd, m.editor.SetQueuedInputCount(1))
	}
	return cmd
}

// updateEditorCmd forwards a message to the editor and returns its cmd.
func (m *appModel) updateEditorCmd(msg tea.Msg) tea.Cmd {
	prevValue := m.editor.Value()
	prevShellMode := m.editor.ShellMode()
	prevHistorySearch := m.editor.IsHistorySearchActive()
	var focusCmd tea.Cmd
	if _, ok := msg.(tea.KeyPressMsg); ok && m.focusedPanel == PanelEditor && !prevHistorySearch && !m.editor.Focused() {
		// Long-lived terminals can leave the textarea blurred while parent focus
		// still points at the editor; restore focus before routing the key.
		focusCmd = m.editor.Focus()
	}
	updated, cmd := m.editor.Update(msg)
	m.editor = updated.(editor.Editor)
	cmds := []tea.Cmd{focusCmd, cmd}
	exitedHistorySearch := prevHistorySearch && !m.editor.IsHistorySearchActive()
	if m.editor.Value() != prevValue ||
		m.editor.ShellMode() != prevShellMode ||
		m.editor.IsHistorySearchActive() != prevHistorySearch {
		m.statusBar.InvalidateCache()
	}
	if m.editor.Value() != prevValue {
		m.lastEditorValueChangeAt = time.Now()
		m.lastIdleFocusWarningReveal = time.Time{}
		m.lastExitClearedInput = time.Time{}
		m.lastEscClearedInput = time.Time{}
		m.streamCancelFooterHidden = false
	}
	if m.editor.Value() != prevValue || exitedHistorySearch {
		if query, ok := editorCompletionQuery(m.editor.Value()); ok {
			cmds = append(cmds, core.CmdHandler(completion.QueryMsg{Query: query}))
			cmds = append(cmds, syncEditorCompletionQueryAfter(m.editor.Value(), query))
		}
	}
	if m.width > 0 && m.height > 0 {
		if desiredLines := m.desiredEditorLines(); desiredLines != m.editorLines {
			m.editorLines = desiredLines
			cmds = append(cmds, m.resizeAll())
		}
	}
	return tea.Batch(cmds...)
}

type editorCompletionQuerySyncMsg struct {
	value string
	query string
}

func syncEditorCompletionQueryAfter(value, query string) tea.Cmd {
	return tea.Tick(30*time.Millisecond, func(time.Time) tea.Msg {
		return editorCompletionQuerySyncMsg{value: value, query: query}
	})
}

func editorCompletionQuery(value string) (string, bool) {
	trimmed := strings.TrimRightFunc(value, unicode.IsSpace)
	if trimmed == "" {
		return "", false
	}
	idx := strings.LastIndexFunc(trimmed, unicode.IsSpace)
	word := trimmed
	if idx >= 0 {
		word = trimmed[idx+1:]
	}
	switch {
	case strings.HasPrefix(word, "@"):
		return strings.TrimPrefix(word, "@"), true
	case strings.HasPrefix(word, "/"):
		return strings.TrimPrefix(word, "/"), true
	default:
		return "", false
	}
}

// updateDialogCmd forwards a message to the dialog manager and returns its cmd.
func (m *appModel) updateDialogCmd(msg tea.Msg) tea.Cmd {
	updated, cmd := m.dialogMgr.Update(msg)
	m.dialogMgr = updated.(dialog.Manager)
	return cmd
}

// updateCompletionsCmd forwards a message to the completion manager and
// returns its cmd.
func (m *appModel) updateCompletionsCmd(msg tea.Msg) tea.Cmd {
	updated, cmd := m.completions.Update(msg)
	m.completions = updated.(completion.Manager)
	return cmd
}

// forwardChat is a convenience for handlers whose entire response is to
// forward the message to the chat page.
func (m *appModel) forwardChat(msg tea.Msg) (tea.Model, tea.Cmd) {
	cmd := m.updateChatCmd(msg)
	return m, cmd
}

// forwardEditor is a convenience for handlers whose entire response is to
// forward the message to the editor.
func (m *appModel) forwardEditor(msg tea.Msg) (tea.Model, tea.Cmd) {
	cmd := m.updateEditorCmd(msg)
	return m, cmd
}

// forwardDialog is a convenience for handlers whose entire response is to
// forward the message to the dialog manager.
func (m *appModel) forwardDialog(msg tea.Msg) (tea.Model, tea.Cmd) {
	cmd := m.updateDialogCmd(msg)
	return m, cmd
}

// forwardCompletions is a convenience for handlers whose entire response is
// to forward the message to the completion manager.
func (m *appModel) forwardCompletions(msg tea.Msg) (tea.Model, tea.Cmd) {
	cmd := m.updateCompletionsCmd(msg)
	return m, cmd
}
