package clipboardutil

import (
	tea "charm.land/bubbletea/v2"
	"github.com/atotto/clipboard"

	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/notification"
)

type copyOptions struct {
	successMsg string
	completion tea.Msg
}

// Option configures a clipboard copy command.
type Option func(*copyOptions)

// WithSuccess returns a success notification after a successful native
// clipboard write. Failed writes are reported as warning/error notifications.
func WithSuccess(message string) Option {
	return func(o *copyOptions) {
		o.successMsg = message
	}
}

// WithCompletion returns msg after clipboard write steps. It is useful for UI
// state updates that depend on a best-effort copy having been attempted.
func WithCompletion(msg tea.Msg) Option {
	return func(o *copyOptions) {
		o.completion = msg
	}
}

// Copy returns a command that copies text using OSC 52 and the platform-native
// clipboard by default. If the native clipboard write fails, the command still
// leaves the OSC 52 attempt in place and reports the partial failure.
func Copy(text string, opts ...Option) tea.Cmd {
	return copy(text, true, opts...)
}

// CopyNative returns a command that copies text only through the platform-native
// clipboard. Use this for copy actions that should preserve the previous local
// clipboard behavior without emitting an OSC 52 sequence.
func CopyNative(text string, opts ...Option) tea.Cmd {
	return copy(text, false, opts...)
}

func copy(text string, useOSC52 bool, opts ...Option) tea.Cmd {
	var options copyOptions
	for _, opt := range opts {
		opt(&options)
	}

	cmds := make([]tea.Cmd, 0, 4)
	if useOSC52 {
		cmds = append(cmds, tea.SetClipboard(text))
	}
	cmds = append(cmds, func() tea.Msg {
		if err := clipboard.WriteAll(text); err != nil {
			if useOSC52 {
				return notification.ShowMsg{
					Text: "Copied via terminal clipboard; native clipboard failed: " + err.Error(),
					Type: notification.TypeWarning,
				}
			}
			return notification.ShowMsg{
				Text: "Failed to copy to clipboard: " + err.Error(),
				Type: notification.TypeError,
			}
		}
		if options.successMsg != "" {
			return notification.ShowMsg{Text: options.successMsg, Type: notification.TypeSuccess}
		}
		return nil
	})
	if options.completion != nil {
		cmds = append(cmds, func() tea.Msg { return options.completion })
	}
	return tea.Sequence(cmds...)
}
