package dialog

import (
	"strings"

	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/AoyangSpace/agentm-terminal/internal/tui/core"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core/layout"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/messages"
)

const (
	thinkingToggleTopRow = 5
	thinkingToggleIndent = 2
)

var (
	thinkingToggleAccentStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("153"))
	thinkingToggleTitleStyle   = thinkingToggleAccentStyle.Bold(true)
	thinkingToggleBodyStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("246"))
	thinkingToggleCheckedStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("114"))
)

type thinkingToggleDialog struct {
	BaseDialog

	enabled  bool
	selected int
	keyMap   pickerKeyMap
}

func NewThinkingToggleDialog(enabled bool) Dialog {
	selected := 1
	if enabled {
		selected = 0
	}
	return &thinkingToggleDialog{
		enabled:  enabled,
		selected: selected,
		keyMap:   defaultPickerKeyMap(),
	}
}

func (d *thinkingToggleDialog) Init() tea.Cmd { return nil }

func (d *thinkingToggleDialog) Update(msg tea.Msg) (layout.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		cmd := d.SetSize(msg.Width, msg.Height)
		return d, cmd

	case tea.MouseClickMsg:
		if msg.Button != tea.MouseLeft {
			return d, nil
		}
		row, _ := d.Position()
		switch msg.Y {
		case row + 4:
			d.selected = 0
			return d, nil
		case row + 5:
			d.selected = 1
			return d, nil
		}
		return d, nil

	case tea.KeyPressMsg:
		if cmd := HandleQuit(msg); cmd != nil {
			return d, cmd
		}
		switch {
		case key.Matches(msg, d.keyMap.Escape):
			return d, closeDialogCmd()
		case key.Matches(msg, d.keyMap.Up):
			if d.selected > 0 {
				d.selected--
			}
			return d, nil
		case key.Matches(msg, d.keyMap.Down):
			if d.selected < 1 {
				d.selected++
			}
			return d, nil
		case key.Matches(msg, d.keyMap.Enter):
			return d, tea.Sequence(
				closeDialogCmd(),
				core.CmdHandler(messages.SetThinkingModeMsg{Enabled: d.selected == 0}),
			)
		case msg.String() == "1":
			d.selected = 0
			return d, nil
		case msg.String() == "2":
			d.selected = 1
			return d, nil
		}
	}
	return d, nil
}

func (d *thinkingToggleDialog) Position() (row, col int) {
	if d.Height() <= thinkingToggleTopRow+4 {
		return 0, 0
	}
	return thinkingToggleTopRow, 0
}

func (d *thinkingToggleDialog) View() string {
	width := max(20, d.Width())
	contentWidth := max(1, width-(thinkingToggleIndent*2))

	lines := []string{
		thinkingToggleAccentStyle.Render(strings.Repeat("─", width)),
		thinkingToggleLine(thinkingToggleTitleStyle.Render("Toggle thinking mode"), contentWidth),
		thinkingToggleLine(thinkingToggleBodyStyle.Render("Enable or disable thinking for this session."), contentWidth),
		"",
		d.renderOption(0, "Enabled", d.enabled, "Claude will think before responding", contentWidth),
		d.renderOption(1, "Disabled", !d.enabled, "Claude will respond without extended thinking", contentWidth),
		"",
		thinkingToggleLine(thinkingToggleBodyStyle.Render("Enter to confirm · Esc to cancel"), contentWidth),
	}
	if d.Height() > 0 {
		row, _ := d.Position()
		for len(lines) < max(0, d.Height()-row) {
			lines = append(lines, strings.Repeat(" ", width))
		}
	}
	return strings.Join(lines, "\n")
}

func (d *thinkingToggleDialog) renderOption(index int, label string, checked bool, desc string, contentWidth int) string {
	cursor := "  "
	if d.selected == index {
		cursor = thinkingToggleAccentStyle.Render("❯") + " "
	}

	left := cursor + thinkingToggleLabel(index, label, checked, d.selected == index)
	labelWidth := 14
	if !d.enabled {
		labelWidth = 15
	}
	padding := strings.Repeat(" ", max(1, labelWidth-lipgloss.Width(labelForThinkingOption(index, label, checked))))
	line := left + thinkingToggleBodyStyle.Render(padding+desc)
	return thinkingToggleLine(line, contentWidth)
}

func thinkingToggleLabel(index int, label string, checked, selected bool) string {
	number := thinkingToggleBodyStyle.Render("1.")
	if index != 0 {
		number = thinkingToggleBodyStyle.Render("2.")
	}
	labelStyle := lipgloss.NewStyle()
	if selected {
		labelStyle = thinkingToggleAccentStyle
	}
	if checked {
		labelStyle = thinkingToggleCheckedStyle
	}
	text := number + " " + labelStyle.Render(label)
	if checked {
		text += " " + thinkingToggleCheckedStyle.Render("✔")
	}
	return text
}

func labelForThinkingOption(index int, label string, checked bool) string {
	text := ""
	if index == 0 {
		text = "1. " + label
	} else {
		text = "2. " + label
	}
	if checked {
		text += " ✔"
	}
	return text
}

func thinkingToggleLine(content string, contentWidth int) string {
	line := strings.Repeat(" ", thinkingToggleIndent) + content
	maxWidth := contentWidth + thinkingToggleIndent
	if lipgloss.Width(line) > maxWidth {
		return ansi.Truncate(line, maxWidth, "")
	}
	return line
}
