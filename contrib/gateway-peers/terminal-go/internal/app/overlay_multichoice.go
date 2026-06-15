package app

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// MultiChoiceOption represents a single selectable option in the dialog.
type MultiChoiceOption struct {
	Label       string // display text
	Description string // optional dim description below the label
	Key         string // unique key returned on selection
}

// MultiChoiceOverlay presents a list of options with keyboard navigation,
// digit shortcuts (1-9), and an optional free-text custom input.
type MultiChoiceOverlay struct {
	title       string
	options     []MultiChoiceOption
	selected    int  // currently highlighted option index (-1 = none when in custom mode)
	allowCustom bool // show a custom text input at the bottom
	customInput string
	inCustom    bool          // true when the user is typing custom text
	onSelect    func(key string) // callback; "" means cancelled
}

// NewMultiChoiceOverlay creates a multi-choice dialog overlay.
// onSelect is called with the chosen option's Key, the custom text, or ""
// on cancellation (Esc).
func NewMultiChoiceOverlay(
	title string,
	options []MultiChoiceOption,
	allowCustom bool,
	onSelect func(key string),
) *MultiChoiceOverlay {
	return &MultiChoiceOverlay{
		title:       title,
		options:     options,
		allowCustom: allowCustom,
		onSelect:    onSelect,
	}
}

func (m *MultiChoiceOverlay) Kind() OverlayKind { return OverlayMultiChoice }

func (m *MultiChoiceOverlay) Update(msg tea.Msg) (Overlay, tea.Cmd, bool) {
	km, ok := msg.(tea.KeyMsg)
	if !ok {
		return m, nil, false
	}

	key := km.String()

	// When typing custom text, most keys go to the input.
	if m.inCustom {
		switch key {
		case "esc":
			m.onSelect("")
			return m, nil, true
		case "enter":
			m.onSelect(m.customInput)
			return m, nil, true
		case "tab":
			// Switch back to option list.
			m.inCustom = false
			if m.selected < 0 && len(m.options) > 0 {
				m.selected = 0
			}
			return m, nil, false
		case "backspace":
			if len(m.customInput) > 0 {
				runes := []rune(m.customInput)
				m.customInput = string(runes[:len(runes)-1])
			}
			return m, nil, false
		default:
			// Accept single printable characters.
			if len([]rune(key)) == 1 {
				m.customInput += key
			}
			return m, nil, false
		}
	}

	// Option-list mode.
	switch key {
	case "esc":
		m.onSelect("")
		return m, nil, true

	case "enter":
		if m.selected >= 0 && m.selected < len(m.options) {
			m.onSelect(m.options[m.selected].Key)
			return m, nil, true
		}
		return m, nil, false

	case "up", "k":
		if m.selected > 0 {
			m.selected--
		}
		return m, nil, false

	case "down", "j":
		if m.selected < len(m.options)-1 {
			m.selected++
		}
		return m, nil, false

	case "tab":
		if m.allowCustom {
			m.inCustom = true
			m.selected = -1
			return m, nil, false
		}
		return m, nil, false

	default:
		// Digit shortcuts 1-9.
		if len(key) == 1 && key[0] >= '1' && key[0] <= '9' {
			idx := int(key[0] - '1')
			if idx < len(m.options) {
				m.selected = idx
				m.onSelect(m.options[idx].Key)
				return m, nil, true
			}
		}
		return m, nil, false
	}
}

func (m *MultiChoiceOverlay) View(width, height int, th *theme.Theme) string {
	var sb strings.Builder

	// Title
	sb.WriteString(th.OverlayTitle.Render(m.title))
	sb.WriteByte('\n')
	sb.WriteByte('\n')

	// Options
	for i, opt := range m.options {
		bullet := "[ ]"
		style := th.OverlayText
		if !m.inCustom && i == m.selected {
			bullet = "[●]" // filled circle
			style = th.OverlayActive
		}

		line := fmt.Sprintf(" %s %d. %s", bullet, i+1, opt.Label)
		sb.WriteString(style.Render(line))
		sb.WriteByte('\n')

		if opt.Description != "" {
			// Indent description under the label.
			desc := fmt.Sprintf("       %s", opt.Description)
			sb.WriteString(th.OverlayDim.Render(desc))
			sb.WriteByte('\n')
		}
	}

	// Custom input option
	if m.allowCustom {
		bullet := "[ ]"
		style := th.OverlayText
		if m.inCustom {
			bullet = "[●]"
			style = th.OverlayActive
		}

		if m.inCustom {
			cursor := "_"
			line := fmt.Sprintf(" %s Custom: %s%s", bullet, m.customInput, cursor)
			sb.WriteString(style.Render(line))
		} else {
			line := fmt.Sprintf(" %s Custom...", bullet)
			sb.WriteString(style.Render(line))
		}
		sb.WriteByte('\n')
	}

	// Hints
	sb.WriteByte('\n')
	hints := "  esc=cancel  enter=confirm  ↑/↓=navigate  1-9=select"
	if m.allowCustom {
		hints += "  tab=custom"
	}
	sb.WriteString(th.OverlayDim.Render(hints))

	content := th.OverlayBorder.Render(sb.String())
	return centerOverlay(content, width, height)
}
