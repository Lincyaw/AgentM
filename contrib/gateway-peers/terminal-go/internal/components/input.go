package components

import (
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/lipgloss"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// InputSubmitted is sent when the user presses Enter to submit text.
type InputSubmitted struct {
	Text string
}

// HistoryNav is sent when the user navigates input history.
type HistoryNav struct {
	Delta int // -1 = older, +1 = newer
}

// InputComplete is sent when Tab is pressed (for suggestion completion).
type InputComplete struct{}

// Input wraps a textarea with prompt prefix, history, and auto-grow.
type Input struct {
	textarea textarea.Model
	history  []string
	histIdx  int
	maxLines int
}

// NewInput creates a new Input component.
func NewInput() Input {
	ta := textarea.New()
	ta.Placeholder = "message..."
	ta.ShowLineNumbers = false
	ta.CharLimit = 0
	ta.SetHeight(1)
	ta.MaxHeight = 3
	ta.Prompt = ""
	ta.FocusedStyle.CursorLine = lipgloss.NewStyle()
	ta.BlurredStyle.CursorLine = lipgloss.NewStyle()
	ta.FocusedStyle.Base = lipgloss.NewStyle()
	ta.BlurredStyle.Base = lipgloss.NewStyle()
	ta.Focus()
	return Input{
		textarea: ta,
		histIdx:  -1,
		maxLines: 3,
	}
}

// Update handles messages for the input component.
func (i Input) Update(msg tea.Msg) (Input, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyEnter:
			text := strings.TrimSpace(i.textarea.Value())
			if text == "" {
				return i, nil
			}
			i.textarea.Reset()
			i.textarea.SetHeight(1)
			i.histIdx = -1
			return i, func() tea.Msg { return InputSubmitted{Text: text} }

		case tea.KeyCtrlJ:
			// Insert a newline
			i.textarea, _ = i.textarea.Update(tea.KeyMsg{Type: tea.KeyEnter})
			i.adjustHeight()
			return i, nil

		case tea.KeyTab:
			return i, func() tea.Msg { return InputComplete{} }

		case tea.KeyUp:
			if i.textarea.Line() == 0 {
				return i, func() tea.Msg { return HistoryNav{Delta: -1} }
			}

		case tea.KeyDown:
			if i.textarea.Line() >= i.textarea.LineCount()-1 {
				return i, func() tea.Msg { return HistoryNav{Delta: 1} }
			}
		}
	}

	var cmd tea.Cmd
	i.textarea, cmd = i.textarea.Update(msg)
	if cmd != nil {
		cmds = append(cmds, cmd)
	}
	i.adjustHeight()

	return i, tea.Batch(cmds...)
}

// adjustHeight sets the textarea height based on content line count.
func (i *Input) adjustHeight() {
	lines := i.textarea.LineCount()
	if lines < 1 {
		lines = 1
	}
	if lines > i.maxLines {
		lines = i.maxLines
	}
	i.textarea.SetHeight(lines)
}

// View renders the input with a prompt prefix.
func (i Input) View(width int, th *theme.Theme) string {
	prefix := th.InputPrompt.Render("> ")
	prefixWidth := lipgloss.Width(prefix)
	taWidth := width - prefixWidth
	if taWidth < 10 {
		taWidth = 10
	}
	i.textarea.SetWidth(taWidth)

	taView := i.textarea.View()
	// Join the prefix with the first line of the textarea
	lines := strings.Split(taView, "\n")
	if len(lines) == 0 {
		return prefix
	}
	lines[0] = prefix + lines[0]
	return strings.Join(lines, "\n")
}

// SetText replaces the textarea content.
func (i *Input) SetText(text string) {
	i.textarea.SetValue(text)
	i.adjustHeight()
}

// Text returns the current textarea content.
func (i *Input) Text() string {
	return i.textarea.Value()
}

// Focus sets focus on the textarea.
func (i *Input) Focus() tea.Cmd {
	return i.textarea.Focus()
}

// SetWidth updates the textarea width.
func (i *Input) SetWidth(w int) {
	prefixWidth := 2 // "> "
	taWidth := w - prefixWidth
	if taWidth < 10 {
		taWidth = 10
	}
	i.textarea.SetWidth(taWidth)
}

// PushHistory adds text to the input history.
func (i *Input) PushHistory(text string) {
	if text == "" {
		return
	}
	i.history = append(i.history, text)
	i.histIdx = -1
}

// HistoryPrev returns the previous history entry.
func (i *Input) HistoryPrev() string {
	if len(i.history) == 0 {
		return ""
	}
	if i.histIdx < 0 {
		i.histIdx = len(i.history) - 1
	} else if i.histIdx > 0 {
		i.histIdx--
	}
	return i.history[i.histIdx]
}

// HistoryNext returns the next history entry, or empty if at the end.
func (i *Input) HistoryNext() string {
	if len(i.history) == 0 || i.histIdx < 0 {
		return ""
	}
	if i.histIdx < len(i.history)-1 {
		i.histIdx++
		return i.history[i.histIdx]
	}
	i.histIdx = -1
	return ""
}

// Height returns the current visual height of the input.
func (i *Input) Height() int {
	return i.textarea.Height()
}
