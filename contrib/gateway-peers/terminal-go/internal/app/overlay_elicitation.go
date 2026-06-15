package app

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// ElicitationType identifies the kind of input an ElicitationOverlay collects.
type ElicitationType int

const (
	ElicitText        ElicitationType = iota // free-form text
	ElicitSelect                             // pick one option
	ElicitMultiSelect                        // pick one or more options
)

// ElicitationOption is a single selectable option.
type ElicitationOption struct {
	Label string
	Value string
}

// ElicitationOverlay presents a structured question from the agent or MCP.
type ElicitationOverlay struct {
	title     string
	message   string
	inputType ElicitationType
	options   []ElicitationOption
	selected  []bool // toggle state per option (MultiSelect)
	cursor    int    // current option cursor (Select/MultiSelect)
	textInput string // user text (Text mode)
	onResult  func(result string)
}

// NewElicitationOverlay creates an overlay for a structured question.
func NewElicitationOverlay(
	title, message string,
	inputType ElicitationType,
	options []ElicitationOption,
	onResult func(result string),
) *ElicitationOverlay {
	sel := make([]bool, len(options))
	return &ElicitationOverlay{
		title:     title,
		message:   message,
		inputType: inputType,
		options:   options,
		selected:  sel,
		onResult:  onResult,
	}
}

func (e *ElicitationOverlay) Kind() OverlayKind { return OverlayElicitation }

func (e *ElicitationOverlay) Update(msg tea.Msg) (Overlay, tea.Cmd, bool) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		key := msg.String()
		switch e.inputType {
		case ElicitText:
			return e.updateText(key)
		case ElicitSelect:
			return e.updateSelect(key)
		case ElicitMultiSelect:
			return e.updateMultiSelect(key)
		}
	}
	return e, nil, false
}

// --- Text mode ---

func (e *ElicitationOverlay) updateText(key string) (Overlay, tea.Cmd, bool) {
	switch key {
	case "esc":
		if e.onResult != nil {
			e.onResult("")
		}
		return e, nil, true

	case "enter":
		if e.onResult != nil {
			e.onResult(e.textInput)
		}
		return e, nil, true

	case "backspace":
		if len(e.textInput) > 0 {
			runes := []rune(e.textInput)
			e.textInput = string(runes[:len(runes)-1])
		}
		return e, nil, false

	default:
		if len(key) == 1 && key[0] >= 32 && key[0] < 127 {
			e.textInput += key
		} else if len([]rune(key)) == 1 {
			e.textInput += key
		}
		return e, nil, false
	}
}

// --- Select mode ---

func (e *ElicitationOverlay) updateSelect(key string) (Overlay, tea.Cmd, bool) {
	switch key {
	case "esc":
		if e.onResult != nil {
			e.onResult("")
		}
		return e, nil, true

	case "up", "k":
		if e.cursor > 0 {
			e.cursor--
		}
		return e, nil, false

	case "down", "j":
		if e.cursor < len(e.options)-1 {
			e.cursor++
		}
		return e, nil, false

	case "enter":
		if len(e.options) > 0 && e.onResult != nil {
			e.onResult(e.options[e.cursor].Value)
		}
		return e, nil, true
	}
	return e, nil, false
}

// --- MultiSelect mode ---

func (e *ElicitationOverlay) updateMultiSelect(key string) (Overlay, tea.Cmd, bool) {
	switch key {
	case "esc":
		if e.onResult != nil {
			e.onResult("")
		}
		return e, nil, true

	case "up", "k":
		if e.cursor > 0 {
			e.cursor--
		}
		return e, nil, false

	case "down", "j":
		if e.cursor < len(e.options)-1 {
			e.cursor++
		}
		return e, nil, false

	case " ":
		if e.cursor < len(e.selected) {
			e.selected[e.cursor] = !e.selected[e.cursor]
		}
		return e, nil, false

	case "enter":
		if e.onResult != nil {
			var vals []string
			for i, opt := range e.options {
				if i < len(e.selected) && e.selected[i] {
					vals = append(vals, opt.Value)
				}
			}
			e.onResult(strings.Join(vals, ","))
		}
		return e, nil, true
	}
	return e, nil, false
}

// --- View ---

func (e *ElicitationOverlay) View(width, height int, th *theme.Theme) string {
	var sb strings.Builder

	sb.WriteString(th.OverlayTitle.Render(e.title))
	sb.WriteByte('\n')

	if e.message != "" {
		sb.WriteByte('\n')
		sb.WriteString(th.OverlayText.Render(e.message))
		sb.WriteByte('\n')
	}

	sb.WriteByte('\n')

	switch e.inputType {
	case ElicitText:
		sb.WriteString(th.OverlayInput.Render(" > "))
		sb.WriteString(th.OverlayText.Render(e.textInput))
		sb.WriteString(th.OverlayDim.Render("_"))
		sb.WriteByte('\n')

	case ElicitSelect:
		for i, opt := range e.options {
			prefix := "  "
			style := th.OverlayText
			if i == e.cursor {
				prefix = "> "
				style = th.OverlayActive
			}
			sb.WriteString(style.Render(fmt.Sprintf("%s%s", prefix, opt.Label)))
			sb.WriteByte('\n')
		}

	case ElicitMultiSelect:
		for i, opt := range e.options {
			pointer := "  "
			if i == e.cursor {
				pointer = "> "
			}
			check := "[ ]"
			if i < len(e.selected) && e.selected[i] {
				check = "[x]"
			}
			style := th.OverlayText
			if i == e.cursor {
				style = th.OverlayActive
			}
			sb.WriteString(style.Render(fmt.Sprintf("%s%s %s", pointer, check, opt.Label)))
			sb.WriteByte('\n')
		}
	}

	sb.WriteByte('\n')
	switch e.inputType {
	case ElicitText:
		sb.WriteString(th.OverlayDim.Render("  enter=submit  esc=cancel"))
	case ElicitSelect:
		sb.WriteString(th.OverlayDim.Render("  up/down=navigate  enter=confirm  esc=cancel"))
	case ElicitMultiSelect:
		sb.WriteString(th.OverlayDim.Render("  up/down=navigate  space=toggle  enter=confirm  esc=cancel"))
	}

	content := th.OverlayBorder.Render(sb.String())
	return centerOverlay(content, width, height)
}
