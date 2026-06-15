package tui

import (
	tea "charm.land/bubbletea/v2"
	"github.com/atotto/clipboard"
)

// notificationCopiedMsg marks an existing notification as copied after its text
// has been written to the clipboard.
type notificationCopiedMsg struct {
	ID uint64
}

// copyNotificationToClipboard copies notification text using the same OSC 52 +
// best-effort platform clipboard pattern as the conversation copy handlers.
func copyNotificationToClipboard(id uint64, text string) tea.Cmd {
	return tea.Sequence(
		tea.SetClipboard(text),
		func() tea.Msg {
			_ = clipboard.WriteAll(text)
			return notificationCopiedMsg{ID: id}
		},
	)
}
