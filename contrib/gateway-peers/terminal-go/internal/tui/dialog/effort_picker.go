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
	effortPickerTopRow = 5
	effortPickerIndent = 2
)

var effortPickerLevels = []string{"low", "medium", "high"}

type effortPickerDialog struct {
	BaseDialog

	selected       int
	keyMap         pickerKeyMap
	topRow         int
	showTranscript bool
}

func NewEffortPickerDialog(level string) Dialog {
	return newEffortPickerDialog(level, effortPickerTopRow, false)
}

func NewEffortPickerDialogAtTop(level string, topRow int) Dialog {
	return newEffortPickerDialog(level, topRow, true)
}

func newEffortPickerDialog(level string, topRow int, showTranscript bool) Dialog {
	return &effortPickerDialog{
		selected:       effortPickerIndex(level),
		keyMap:         defaultPickerKeyMap(),
		topRow:         max(0, topRow),
		showTranscript: showTranscript,
	}
}

func effortPickerIndex(level string) int {
	level = strings.ToLower(strings.TrimSpace(level))
	for i, candidate := range effortPickerLevels {
		if candidate == level {
			return i
		}
	}
	return 2
}

func (d *effortPickerDialog) Init() tea.Cmd { return nil }

func (d *effortPickerDialog) Update(msg tea.Msg) (layout.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		cmd := d.SetSize(msg.Width, msg.Height)
		return d, cmd

	case tea.KeyPressMsg:
		if cmd := HandleQuit(msg); cmd != nil {
			return d, cmd
		}
		switch {
		case key.Matches(msg, d.keyMap.Escape):
			return d, tea.Sequence(closeDialogCmd(), core.CmdHandler(messages.EffortPickerCanceledMsg{
				ShowTranscript: d.showTranscript,
			}))
		case msg.String() == "left", key.Matches(msg, d.keyMap.Up):
			if d.selected > 0 {
				d.selected--
			}
			return d, nil
		case msg.String() == "right", key.Matches(msg, d.keyMap.Down):
			if d.selected < len(effortPickerLevels)-1 {
				d.selected++
			}
			return d, nil
		case key.Matches(msg, d.keyMap.Enter):
			return d, tea.Sequence(
				closeDialogCmd(),
				core.CmdHandler(messages.SetThinkingLevelMsg{
					Level:          effortPickerLevels[d.selected],
					ShowTranscript: d.showTranscript,
				}),
			)
		}
	}
	return d, nil
}

func (d *effortPickerDialog) Position() (row, col int) {
	if d.Height() <= d.topRow+4 {
		return 0, 0
	}
	return d.topRow, 0
}

func (d *effortPickerDialog) View() string {
	width := max(20, d.Width())
	contentWidth := max(1, width-(effortPickerIndent*2))
	slider := effortPickerSlider(d.selected)

	lines := []string{
		effortPickerAccentStyle().Render(strings.Repeat("─", width)),
		effortPickerLine(effortPickerTitleStyle().Render("Effort"), contentWidth),
		"",
		effortPickerLine(effortPickerBodyStyle().Render("Faster"+strings.Repeat(" ", 37)+"Smarter"), contentWidth),
		effortPickerLine(slider, contentWidth),
		effortPickerLine(effortPickerLabels(d.selected), contentWidth),
		"",
		effortPickerLine(effortPickerBodyStyle().Render("←/→ to adjust · Enter to confirm · Esc to cancel"), contentWidth),
	}
	if d.Height() > 0 {
		row, _ := d.Position()
		for len(lines) < max(0, d.Height()-row) {
			lines = append(lines, strings.Repeat(" ", width))
		}
	}
	return strings.Join(lines, "\n")
}

func effortPickerSlider(selected int) string {
	track := "────────────────────┆────────────────────┆────────────────────"
	markerPositions := []int{10, 31, 52}
	pos := markerPositions[max(0, min(selected, len(markerPositions)-1))]
	runes := []rune(track)
	if pos >= 0 && pos < len(runes) {
		runes[pos] = '▲'
	}
	return effortPickerAccentStyle().Render(string(runes))
}

func effortPickerLabels(selected int) string {
	labels := []string{"low", "medium", "high"}
	parts := make([]string, 0, len(labels))
	for i, label := range labels {
		style := effortPickerBodyStyle()
		if i == selected {
			style = effortPickerAccentStyle().Bold(true)
		}
		parts = append(parts, style.Render(label))
	}
	return strings.Join(parts, strings.Repeat(" ", 12))
}

func effortPickerLine(content string, contentWidth int) string {
	line := strings.Repeat(" ", effortPickerIndent) + content
	maxWidth := contentWidth + effortPickerIndent
	if lipgloss.Width(line) > maxWidth {
		return ansi.Truncate(line, maxWidth, "")
	}
	return line
}

func effortPickerAccentStyle() lipgloss.Style {
	return lipgloss.NewStyle().Foreground(lipgloss.Color("153"))
}

func effortPickerTitleStyle() lipgloss.Style {
	return effortPickerAccentStyle().Bold(true)
}

func effortPickerBodyStyle() lipgloss.Style {
	return lipgloss.NewStyle().Foreground(lipgloss.Color("246"))
}
